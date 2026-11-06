"""Small OpenPI client usable from the separate Isaac Sim Python environment."""

from collections import deque
import time

import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect

POLICY_TYPE = "pi05_object_understanding"
ACTION_HORIZON = 16
NOISE_SHAPE = (ACTION_HORIZON, 32)


def observation(joints, images, instruction):
    joints = np.asarray(joints, dtype=np.float32)
    if joints.shape != (9,) or not np.isfinite(joints).all():
        raise ValueError("Expected nine finite current joint positions")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A task instruction is required")
    selected = {}
    for view in ("top", "wrist"):
        rgb = np.asarray(images[view]["rgb"])
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"Expected uint8 HWC RGB from {view}")
        selected[view] = rgb
    # Explicit allowlist: labels, object poses, and oracle geometry never go to the policy.
    return {
        "observation/state": joints,
        "observation/image": selected["top"],
        "observation/wrist_image": selected["wrist"],
        "prompt": instruction,
    }


def action_chunk(result):
    chunk = np.asarray(result["actions"], dtype=np.float32)
    if chunk.shape != (ACTION_HORIZON, 9) or not np.isfinite(chunk).all():
        raise ValueError("Expected a finite (16, 9) absolute joint-target chunk")
    return chunk


class RecedingHorizonController:
    def __init__(self, policy, execution_horizon=4):
        if not 1 <= execution_horizon <= ACTION_HORIZON:
            raise ValueError("execution_horizon must be 1..16")
        self.policy = policy
        self.execution_horizon = execution_horizon
        self.pending = deque()
        self.inference_times = []

    def next_action(self, joints, images, instruction):
        if not self.pending:
            started = time.monotonic()
            chunk = self.policy.infer(observation(joints, images, instruction))
            chunk = action_chunk({"actions": chunk})
            self.inference_times.append(time.monotonic() - started)
            self.pending.extend(chunk[: self.execution_horizon].copy())
        return self.pending.popleft()


class LocalPolicy:
    def __init__(self, port, *, expected_checkpoint=None):
        self.socket = connect(f"ws://127.0.0.1:{port}", compression=None, max_size=8 * 1024**2, open_timeout=15)
        try:
            metadata = msgpack_numpy.unpackb(self.socket.recv(timeout=30))
            if metadata.get("policy_type") != POLICY_TYPE or metadata.get("noise_shape") != list(NOISE_SHAPE):
                raise ValueError("The connected server is not the expected pi0.5 policy")
            if expected_checkpoint is not None and metadata.get("checkpoint") != expected_checkpoint:
                raise ValueError("Policy server checkpoint differs from the evaluation configuration")
        except BaseException:
            self.socket.close()
            raise
        self.reset(0)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)

    def infer(self, obs):
        request = {**obs, "_noise": self.rng.standard_normal(NOISE_SHAPE).astype(np.float32)}
        self.socket.send(msgpack_numpy.packb(request))
        message = self.socket.recv(timeout=180)
        if isinstance(message, str):
            raise RuntimeError(f"Policy inference failed: {message}")
        return action_chunk(msgpack_numpy.unpackb(message))

    def close(self):
        self.socket.close()
