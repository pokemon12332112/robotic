"""Run/resume closed-loop testing, optionally waiting for an active training process."""

import argparse
from datetime import UTC
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
ROBOT2 = ROOT.parent.parent / "robot2"
BENCHMARK_REVISION = "31ffee567d014735f82741b03bebc9cf6c13778e"
ISAACLAB_REVISION = "44044b0963001962abe5701133427e37b003fffe"


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checked_scenes(scenes, audit_path):
    manifest = json.loads((scenes / "manifest.json").read_text())
    audit = json.loads(audit_path.read_text())
    expected = {name for name, meta in audit.items() if meta["split"] == "eval"}
    entries = manifest["episodes"]
    if manifest["split"] != "eval" or manifest["benchmark_revision"] != BENCHMARK_REVISION:
        raise ValueError("Unexpected benchmark or split in scene manifest")
    if {entry["source_episode"] for entry in entries} != expected or len(entries) != len(expected):
        raise ValueError("Scene manifest must match the complete original test split, without duplicates")
    for entry in entries:
        filename = entry["scene_file"]
        if Path(filename).name != filename:
            raise ValueError("Scene paths must stay inside the scenes directory")
        content = (scenes / filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Scene content changed: {filename}")
        payload = json.loads(content)
        metadata = audit[entry["source_episode"]]
        if (
            payload["split"] != "eval"
            or payload["source_episode"] != entry["source_episode"]
            or payload["scenario_id"] != metadata["scenario_id"]
            or payload["instruction"] != metadata["prompt"]
            or entry["scenario_id"] != metadata["scenario_id"]
        ):
            raise ValueError(f"Scene identity differs from held-out source: {filename}")
    return manifest


def checkpoint_ready(path):
    return all(
        (path / name).is_file()
        for name in (
            "_CHECKPOINT_METADATA",
            "params/_METADATA",
            "assets/local/object-understanding-train/norm_stats.json",
        )
    ) and not path.name.endswith(".orbax-checkpoint-tmp")


def checkpoint_identity(path):
    # Orbax checkpoints are immutable once committed; record their metadata and file inventory.
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*")):
        if file.is_file():
            stat = file.stat()
            digest.update(f"{file.relative_to(path)}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
            if file.name in {"_METADATA", "_CHECKPOINT_METADATA", "norm_stats.json"}:
                digest.update(file.read_bytes())
    return digest.hexdigest()


def evaluator_identity():
    digest = hashlib.sha256()
    directory = Path(__file__).parent
    for name in ("closed_loop.py", "closed_loop_episode.py", "closed_loop_protocol.py", "serve_closed_loop.py"):
        digest.update((directory / name).read_bytes())
    return digest.hexdigest()


def process_command(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return None


def wait_training(pid, checkpoint, update, *, timeout):
    command = process_command(pid)
    if command and (b"scripts/train.py" not in command or b"pi05_object_understanding_lora" not in command):
        raise ValueError("The specified PID is not the expected pi0.5 training process")
    started = time.monotonic()
    while command and process_command(pid) == command:
        if time.monotonic() - started > timeout:
            raise TimeoutError("Timed out waiting for training; evaluation has not run")
        update("waiting_for_training", training_pid=pid, checkpoint=str(checkpoint))
        time.sleep(10)
    if not checkpoint_ready(checkpoint):
        raise RuntimeError("Training exited without the expected completed checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/pi05_object_understanding_lora/object_understanding_v1/2999",
    )
    parser.add_argument("--scenes", type=Path, default=ROOT / "outputs/object-understanding-closed-loop/scenes")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/object-understanding-closed-loop/evaluation")
    parser.add_argument(
        "--benchmark", type=Path, default=ROBOT2 / "Isaac-GR00T/external_dependencies/spatial-understanding"
    )
    parser.add_argument("--sim-python", type=Path, default=ROBOT2 / ".venv-isaacsim/bin/python")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execution-horizon", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--wait-for-training-pid", type=int)
    parser.add_argument("--wait-timeout", type=int, default=86400)
    parser.add_argument("--episode-timeout", type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.execution_horizon <= 16 or (args.limit is not None and args.limit < 1):
        parser.error("execution-horizon must be 1..16; limit must be positive")
    if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").lower() not in {"yes", "y", "1"}:
        parser.error("The installed Isaac Sim runtime requires OMNI_KIT_ACCEPT_EULA=YES after license acceptance")
    if not args.sim_python.is_file():
        parser.error("Isaac Sim Python environment was not found")
    revision = subprocess.check_output(["git", "-C", str(args.benchmark), "rev-parse", "HEAD"], text=True).strip()
    if revision != BENCHMARK_REVISION:
        parser.error("Benchmark source revision differs from the scene exporter")
    isaaclab = args.benchmark.parent / "IsaacLab"
    lab_revision = subprocess.check_output(["git", "-C", str(isaaclab), "rev-parse", "HEAD"], text=True).strip()
    if lab_revision != ISAACLAB_REVISION:
        parser.error("Isaac Lab source revision differs from the verified simulator installation")
    manifest = checked_scenes(args.scenes, ROOT.parent / "dataset-inspection/episode-metadata.json")
    episodes = manifest["episodes"][: args.limit]
    checkpoint = args.checkpoint.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # Refuse concurrent jobs sharing a result directory.
    import fcntl

    lock = (args.output / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another evaluation is already using this output directory") from error
    status = {"requested_episodes": len(episodes), "checkpoint": str(checkpoint), "policy": "pi05_object_understanding"}
    results = []

    def update(stage, **kwargs):
        status.update(stage=stage, completed_episodes=len(results), updated_at=datetime.now(UTC).isoformat(), **kwargs)
        save(args.output / "status.json", status)

    server = None
    server_log = None
    try:
        update("preparing")
        package_versions = json.loads(
            subprocess.check_output(
                [
                    str(args.sim_python.absolute()),
                    "-c",
                    "import json,importlib.metadata as m; print(json.dumps({n:m.version(n) for n in "
                    "['isaacsim','isaaclab','torch','numpy','websockets','msgpack']}))",
                ],
                text=True,
            )
        )
        save(
            args.output / "runtime.json",
            {
                "simulator_packages": package_versions,
                "benchmark_revision": revision,
                "isaaclab_revision": lab_revision,
                "sim_python": str(args.sim_python.absolute()),
                "physics_device": "cpu",
                "rendering": "RTX GPU",
                "evaluator_identity": evaluator_identity(),
                "simulator_setup": "Reuses the existing robot2 Isaac Sim installation and benchmark reset implementation",
            },
        )
        if args.wait_for_training_pid:
            wait_training(args.wait_for_training_pid, checkpoint, update, timeout=args.wait_timeout)
        if not checkpoint_ready(checkpoint):
            raise FileNotFoundError(f"No completed trained checkpoint at {checkpoint}")
        configuration = {
            "checkpoint": str(checkpoint),
            "checkpoint_identity": checkpoint_identity(checkpoint),
            "execution_horizon": args.execution_horizon,
            "seed": args.seed,
            "benchmark_revision": revision,
            "isaaclab_revision": lab_revision,
            "evaluator_identity": evaluator_identity(),
            "simulator_packages": package_versions,
            "sim_python": str(args.sim_python.absolute()),
            "scene_manifest_sha256": hashlib.sha256((args.scenes / "manifest.json").read_bytes()).hexdigest(),
            "policy": "pi05_object_understanding",
            "policy_views": ["top", "wrist"],
        }
        config_path = args.output / "configuration.json"
        if config_path.exists() and json.loads(config_path.read_text()) != configuration:
            raise ValueError("Existing results use a different model, scenes, or evaluation settings")
        save(config_path, configuration)
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        # Leave GPU space for RTX rendering alongside the inference model.
        server_env = dict(env, XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.45")
        server_log = (args.output / "policy-server.log").open("a")
        update("starting_policy")
        server = subprocess.Popen(
            [
                sys.executable,
                "examples/object_understanding/serve_closed_loop.py",
                "--checkpoint",
                str(checkpoint),
                "--port",
                str(args.port),
            ],
            cwd=ROOT,
            env=server_env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 300
        while True:
            if server.poll() is not None:
                raise RuntimeError("Policy server exited; see policy-server.log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/healthz", timeout=2) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Policy server did not become ready")
            time.sleep(1)
        # Verify the server identity, including when an unrelated process owns the requested port.
        from closed_loop_protocol import LocalPolicy

        probe = LocalPolicy(args.port, expected_checkpoint=str(checkpoint))
        probe.close()
        sim_env = dict(env, PYTHONPATH=str(ROOT / "packages/openpi-client/src"), HEADLESS="1")
        sim_env.pop("DISPLAY", None)
        for entry in episodes:
            destination = args.output / entry["source_episode"]
            result_path = destination / "result.json"
            if not result_path.exists():
                destination.mkdir(exist_ok=True)
                update("running", current_episode=entry["source_episode"])
                with (destination / "simulator.log").open("w") as log:
                    process = subprocess.run(
                        [
                            str(args.sim_python.absolute()),
                            "examples/object_understanding/closed_loop_episode.py",
                            "--benchmark",
                            str(args.benchmark.resolve()),
                            "--scene",
                            str((args.scenes / entry["scene_file"]).resolve()),
                            "--output",
                            str(destination.resolve()),
                            "--checkpoint",
                            str(checkpoint),
                            "--port",
                            str(args.port),
                            "--execution-horizon",
                            str(args.execution_horizon),
                            "--seed",
                            str(args.seed + int(entry["source_episode"].split("_")[-1])),
                            "--device",
                            "cpu",
                        ],
                        cwd=ROOT,
                        env=sim_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=args.episode_timeout,
                        check=False,
                    )
                if process.returncode or not result_path.exists():
                    raise RuntimeError(
                        f"Simulator failed for {entry['source_episode']}; see {destination / 'simulator.log'}"
                    )
            result = json.loads(result_path.read_text())
            if (
                result["source_episode"] != entry["source_episode"]
                or result["scenario_id"] != entry["scenario_id"]
                or result["split"] != "eval"
                or result["checkpoint"] != str(checkpoint)
                or result["execution_horizon"] != args.execution_horizon
                or result["seed"] != args.seed + int(entry["source_episode"].split("_")[-1])
            ):
                raise ValueError("Cached result does not match this evaluation")
            results.append(result)
            save(
                args.output / "metrics.json",
                {
                    **configuration,
                    "evaluation": "closed_loop",
                    "split": "eval",
                    "completed_episodes": len(results),
                    "requested_episodes": len(episodes),
                    "success_rate": sum(r["success"] for r in results) / len(results),
                    "correct_selection_rate": sum(r["correct_selection"] for r in results) / len(results),
                    "episodes": results,
                },
            )
            update("running", current_episode=entry["source_episode"])
            print(
                f"Completed {len(results)}/{len(episodes)}; successes={sum(r['success'] for r in results)}", flush=True
            )
        update("complete", current_episode=None)
    except BaseException as error:
        update("error", error=str(error))
        raise
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        if server_log is not None:
            server_log.close()
        lock.close()


if __name__ == "__main__":
    main()
