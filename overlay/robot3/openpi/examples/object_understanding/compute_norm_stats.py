"""Fit train-only normalization using numeric Parquet columns, without decoding RGB."""

import json

from lerobot.common.constants import HF_LEROBOT_HOME
import numpy as np
import pyarrow.parquet as pq

from openpi import transforms
from openpi.shared import normalize
from openpi.training import config


def action_chunks(states, actions, horizon):
    indices = np.minimum(np.arange(len(actions))[:, None] + np.arange(horizon), len(actions) - 1)
    chunks = actions[indices].copy()
    return transforms.DeltaActions(transforms.make_bool_mask(7, -2))({"state": states, "actions": chunks})["actions"]


def main():
    cfg = config.get_config("pi05_object_understanding_lora")
    root = HF_LEROBOT_HOME / cfg.data.repo_id
    provenance = json.loads((root / "conversion.json").read_text())
    episodes = provenance["episodes"]
    if provenance["source_split"] != "train" or any(e["split"] != "train" for e in episodes):
        raise ValueError("Normalization must use only the source training split")
    info = json.loads((root / "meta/info.json").read_text())
    if info["total_episodes"] != len(episodes):
        raise ValueError("Conversion is incomplete")
    files = sorted(root.glob("data/chunk-*/episode_*.parquet"))
    if len(files) != len(episodes):
        raise ValueError("Missing episode data")
    stats = {name: normalize.RunningStats() for name in ("state", "actions")}
    frames = 0
    for file in files:
        table = pq.read_table(file, columns=["state", "actions"])
        states = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        stats["state"].update(states.astype(np.float64))
        stats["actions"].update(action_chunks(states, actions, cfg.model.action_horizon).astype(np.float64))
        frames += len(states)
    if frames != info["total_frames"]:
        raise ValueError("Frame count mismatch")
    output = cfg.assets_dirs / cfg.data.repo_id
    normalize.save(output, {name: value.get_statistics() for name, value in stats.items()})
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "source_repo": provenance["source_repo"],
                "revision": provenance["revision"],
                "source_split": "train",
                "episodes": len(episodes),
                "frames": frames,
                "action_horizon": cfg.model.action_horizon,
                "arm_actions": "delta to current observed state",
                "finger_actions": "absolute metres",
            },
            indent=2,
        )
    )
    print(f"Fitted statistics on {len(episodes)} training episodes / {frames} frames: {output}")


if __name__ == "__main__":
    main()
