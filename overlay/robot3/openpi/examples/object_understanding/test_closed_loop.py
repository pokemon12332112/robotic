import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading

from closed_loop import BENCHMARK_REVISION
from closed_loop import checked_scenes
from closed_loop import checkpoint_ready
from closed_loop_protocol import NOISE_SHAPE
from closed_loop_protocol import POLICY_TYPE
from closed_loop_protocol import LocalPolicy
from closed_loop_protocol import RecedingHorizonController
from closed_loop_protocol import observation
import numpy as np
from openpi_client import msgpack_numpy
import pytest
from serve_closed_loop import SeededPolicy
from websockets.exceptions import ConnectionClosed
from websockets.sync.server import serve


def images(value):
    return {name: {"rgb": np.full((8, 8, 3), value, np.uint8)} for name in ("top", "wrist", "left", "right")}


def test_replanning_uses_new_feedback_and_discards_unused_chunk():
    class Policy:
        def __init__(self):
            self.calls = []

        def infer(self, obs):
            self.calls.append(obs)
            return np.repeat((obs["observation/state"][0] + np.arange(16))[:, None], 9, axis=1)

    policy = Policy()
    controller = RecedingHorizonController(policy, 4)
    outputs = [controller.next_action(np.full(9, 10 * (i + 1)), images(i), "pick up the bar")[0] for i in range(5)]
    assert outputs == [10, 11, 12, 13, 50]
    assert len(policy.calls) == 2
    assert policy.calls[-1]["observation/image"][0, 0, 0] == 4
    assert set(policy.calls[-1]) == {"observation/state", "observation/image", "observation/wrist_image", "prompt"}


def test_server_rejects_labels_and_invalid_noise():
    class Policy:
        def infer(self, *args, **kwargs):
            raise AssertionError("Invalid request reached the model")

    policy = SeededPolicy(Policy())
    request = {**observation(np.zeros(9), images(0), "pick"), "_noise": np.zeros(NOISE_SHAPE)}
    with pytest.raises(ValueError, match="Unexpected policy input"):
        policy.infer({**request, "target_id": "object_0"})
    with pytest.raises(ValueError, match="Invalid sampling noise"):
        policy.infer({**request, "_noise": np.full(NOISE_SHAPE, np.nan)})


def test_original_test_scenes_are_complete_and_disjoint():
    root = Path(__file__).resolve().parents[2]
    manifest = checked_scenes(
        root / "outputs/object-understanding-closed-loop/scenes",
        root.parent / "dataset-inspection/episode-metadata.json",
    )
    assert len(manifest["episodes"]) == 148


def test_scene_train_label_is_rejected_even_with_matching_hash(tmp_path):
    payload = {"source_episode": "episode_0", "split": "train", "scenario_id": "s", "instruction": "pick"}
    data = json.dumps(payload).encode()
    (tmp_path / "episode.json").write_bytes(data)
    manifest = {
        "split": "eval",
        "benchmark_revision": BENCHMARK_REVISION,
        "episodes": [
            {
                "source_episode": "episode_0",
                "scene_file": "episode.json",
                "scenario_id": "s",
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "audit.json").write_text(
        json.dumps({"episode_0": {"split": "eval", "scenario_id": "s", "prompt": "pick"}})
    )
    with pytest.raises(ValueError, match="identity"):
        checked_scenes(tmp_path, tmp_path / "audit.json")


def test_partial_checkpoint_is_not_ready(tmp_path):
    assert not checkpoint_ready(tmp_path)
    (tmp_path / "params").mkdir()
    (tmp_path / "params/_METADATA").write_text("{}")
    (tmp_path / "_CHECKPOINT_METADATA").write_text("{}")
    assert not checkpoint_ready(tmp_path)


def test_websocket_round_trip_and_episode_noise_reset():
    received_noise = []

    class Model:
        def infer(self, obs, *, noise):
            received_noise.append(noise.copy())
            return {"actions": np.tile(obs["observation/state"], (16, 1))}

    policy = SeededPolicy(Model())

    def handler(socket):
        socket.send(
            msgpack_numpy.packb({"policy_type": POLICY_TYPE, "noise_shape": list(NOISE_SHAPE), "checkpoint": "/test"})
        )
        try:
            while True:
                result = policy.infer(msgpack_numpy.unpackb(socket.recv()))
                socket.send(msgpack_numpy.packb(result))
        except ConnectionClosed:
            pass

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = LocalPolicy(server.socket.getsockname()[1], expected_checkpoint="/test")
        try:
            obs = observation(np.arange(9), images(3), "pick up the bar")
            client.reset(42)
            result = client.infer(obs)
            client.infer(obs)
            client.reset(42)
            client.infer(obs)
            np.testing.assert_array_equal(result, np.tile(np.arange(9), (16, 1)))
            np.testing.assert_array_equal(received_noise[0], received_noise[2])
            assert not np.array_equal(received_noise[0], received_noise[1])
            # Exercise the actual separate simulator interpreter without loading Isaac Sim or the GPU.
            root = Path(__file__).resolve().parents[2]
            sim_python = root.parent.parent / "robot2/.venv-isaacsim/bin/python"
            if sim_python.is_file():
                env = dict(
                    os.environ,
                    PYTHONPATH=os.pathsep.join(
                        [
                            str(Path(__file__).parent),
                            str(root / "packages/openpi-client/src"),
                        ]
                    ),
                )
                completed = subprocess.run(
                    [
                        str(sim_python),
                        "-c",
                        "import sys,numpy as np; from closed_loop_protocol import LocalPolicy,observation; "
                        "p=LocalPolicy(int(sys.argv[1]),expected_checkpoint='/test'); "
                        "images={v:{'rgb':np.zeros((8,8,3),np.uint8)} for v in ['top','wrist']}; "
                        "a=p.infer(observation(np.arange(9),images,'pick')); "
                        "assert a.shape==(16,9); p.close()",
                        str(server.socket.getsockname()[1]),
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                assert completed.returncode == 0, completed.stderr
        finally:
            client.close()
            server.shutdown()
            thread.join(timeout=5)
