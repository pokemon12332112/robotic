"""Convert the existing train/eval splits from HDF5 to local LeRobot v2.1 datasets.

Uses raw state_t and command_t, and aligns RGB with obs/camera_index rather than
assuming cameras and control have the same frame rate. Does not publish data.
"""

import argparse
import concurrent.futures
import json
import logging
from pathlib import Path
import re
import shutil
import tempfile
import time
import urllib.request

import datasets
import h5py
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

SOURCE_REPO = "phulocnguyen/object-understanding"
REVISION = "e3dbcd59b906768df0a2e7baa8b3124c6b0aaec1"
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]


def validate_episode(group, recording, metadata):
    """Reject schema mismatches and future-frame alignment before writing an episode."""
    if recording["robot"] != "franka" or recording["joint_names"] != JOINT_NAMES:
        raise ValueError("Unexpected robot or joint ordering")
    if not recording["action_schema"].startswith("joint_position_target;"):
        raise ValueError("Expected absolute joint-position targets")
    if not np.isclose(recording["control_dt_s"], 1 / 30):
        raise ValueError("Expected 30 Hz control")
    if recording["transition_schema"] != "obs=state_t; actions=command_t; states=state_t+1":
        raise ValueError("Unknown transition alignment")
    states = group["obs/joint_position"][:]
    actions = group["actions"][:]
    timestamps = group["timestamps"][:]
    count = len(actions)
    if count != metadata["num_transitions"] or states.shape != (count, 9) or actions.shape != (count, 9):
        raise ValueError("Invalid transition count or state/action shape")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Non-finite state/action values")
    if not np.allclose(timestamps, np.arange(count) / 30, atol=1e-6):
        raise ValueError("Unexpected action timestamps")
    for view in ("top", "wrist"):
        indices = group[f"obs/camera_index/{view}"][:]
        camera_times = group[f"sensors/{view}/timestamps"][:]
        rgb = group[f"sensors/{view}/rgb"]
        if rgb.shape != (len(camera_times), 256, 256, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"Unexpected {view} RGB format")
        if indices.shape != (count,) or np.any(indices < 0) or np.any(indices >= len(camera_times)):
            raise ValueError(f"Invalid {view} camera indices")
        if np.any(camera_times[indices] > timestamps + 1e-6):
            raise ValueError(f"Future {view} image leaks into an observation")
    return states.astype(np.float32), actions.astype(np.float32)


def download_episode(episode, destination):
    url = f"https://huggingface.co/datasets/{SOURCE_REPO}/resolve/{REVISION}/{episode}/dataset.hdf5"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as response, destination.open("wb") as target:
                shutil.copyfileobj(response, target)
            return
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def episode_files(selected, source_root, scratch):
    """Overlap network transfers while bounding temporary storage to four episodes."""

    def fetch(episode):
        local = source_root / episode / "dataset.hdf5" if source_root is not None else None
        if local is not None and local.is_file():
            return local, False
        local = Path(scratch) / f"{episode}.hdf5"
        download_episode(episode, local)
        return local, True

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending = {}
        for index in range(min(4, len(selected))):
            pending[index] = pool.submit(fetch, selected[index][0])
        for index, item in enumerate(selected):
            local, temporary = pending.pop(index).result()
            yield item, local
            if temporary:
                local.unlink()
            following = index + 4
            if following < len(selected):
                pending[following] = pool.submit(fetch, selected[following][0])


def convert(metadata_path, split, *, source_root=None, max_episodes=None, repo_id=None, resume=False):
    metadata = json.loads(metadata_path.read_text())
    if max_episodes is not None and max_episodes < 1:
        raise ValueError("max_episodes must be positive")
    selected = sorted((name, entry) for name, entry in metadata.items() if entry["split"] == split)
    if not selected:
        raise ValueError(f"No episodes in source split {split}")
    if max_episodes is not None:
        if repo_id is None:
            raise ValueError("Use a separate --repo-id for sample conversions")
        selected = selected[:max_episodes]
    repo_id = repo_id or f"local/object-understanding-{split}"
    if not re.fullmatch(r"[\w-]+/[\w-]+", repo_id):
        raise ValueError("repo-id must have the form namespace/name")
    output = HF_LEROBOT_HOME / repo_id
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or (staging.exists() and not resume):
        raise FileExistsError(f"Refusing to overwrite {output} or {staging}")
    if shutil.disk_usage(metadata_path.parent).free < 4 * 1024**3:
        raise OSError("At least 4 GiB free space is required before conversion")

    features = {
        name: {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]}
        for name in ("image", "wrist_image")
    }
    features.update({name: {"dtype": "float32", "shape": (9,), "names": JOINT_NAMES} for name in ("state", "actions")})
    provenance = {"source_repo": SOURCE_REPO, "revision": REVISION, "source_split": split, "episodes": []}
    if staging.exists():
        previous = json.loads((staging / "conversion.json").read_text())
        if any(previous[key] != provenance[key] for key in ("source_repo", "revision", "source_split")):
            raise ValueError("Cannot resume a different source/revision/split")
        completed = previous["episodes"]
        if [entry["source_episode"] for entry in completed] != [name for name, _ in selected[: len(completed)]]:
            raise ValueError("Cannot resume with a different episode selection")
        dataset = LeRobotDataset(repo_id, root=staging)
        if dataset.meta.total_episodes != len(completed):
            raise ValueError("Incomplete episode commit; inspect staging data before resuming")
        dataset.start_image_writer(num_threads=4)
        provenance = previous
    else:
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=staging,
            fps=30,
            robot_type="franka_panda",
            features=features,
            use_videos=False,
            image_writer_threads=4,
        )
    start = len(provenance["episodes"])
    try:
        with tempfile.TemporaryDirectory(prefix="object-understanding-", dir=output.parent) as scratch:
            for number, ((episode, entry), local) in enumerate(
                episode_files(selected[start:], source_root, scratch), start=start
            ):
                if not entry["success"] or not entry["finalized"] or entry["schema_version"] != 3:
                    raise ValueError(f"Episode {episode} is not a finalized successful schema-v3 recording")
                prompt = entry["prompt"]
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"Missing prompt for {episode}")
                with h5py.File(local, "r") as file:
                    groups = list(file["data"].values())
                    if len(groups) != 1:
                        raise ValueError(f"Expected one episode in {local}")
                    group = groups[0]
                    recording = json.loads(file.attrs["recording_json"])
                    states, actions = validate_episode(group, recording, entry)
                    images = {view: group[f"sensors/{view}/rgb"][:] for view in ("top", "wrist")}
                    indices = {view: group[f"obs/camera_index/{view}"][:] for view in images}
                    for index in range(len(actions)):
                        dataset.add_frame(
                            {
                                "image": images["top"][indices["top"][index]],
                                "wrist_image": images["wrist"][indices["wrist"][index]],
                                "state": states[index],
                                "actions": actions[index],
                                "task": prompt,
                            }
                        )
                    dataset.save_episode()
                provenance["episodes"].append({"source_episode": episode, "converted_episode": number, **entry})
                manifest_tmp = staging / "conversion.json.tmp"
                manifest_tmp.write_text(json.dumps(provenance, indent=2))
                manifest_tmp.replace(staging / "conversion.json")
                if (number + 1) % 10 == 0 or number + 1 == len(selected):
                    print(
                        f"{split}: {number + 1}/{len(selected)} episodes; {dataset.meta.total_frames} frames",
                        flush=True,
                    )
                if shutil.disk_usage(staging).free < 2 * 1024**3:
                    raise OSError("Stopping conversion: less than 2 GiB free")
    finally:
        dataset.stop_image_writer()
    staging.rename(output)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    datasets.disable_progress_bar()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", type=Path, required=True, help="Audited episode-metadata.json at the pinned revision"
    )
    parser.add_argument("--split", choices=["train", "eval"], required=True)
    parser.add_argument("--source-root", type=Path, help="Use existing local episode HDF5 files when present")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--repo-id", help="Override the local LeRobot ID; required for sample conversions")
    parser.add_argument("--resume", action="store_true", help="Resume an existing incomplete conversion")
    args = parser.parse_args()
    convert(
        args.metadata,
        args.split,
        source_root=args.source_root,
        max_episodes=args.max_episodes,
        repo_id=args.repo_id,
        resume=args.resume,
    )
