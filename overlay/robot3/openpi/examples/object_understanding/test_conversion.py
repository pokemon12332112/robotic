import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.policies.object_understanding_policy import data_transforms

spec = importlib.util.spec_from_file_location("convert_dataset", Path(__file__).with_name("convert_dataset.py"))
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def test_observation_alignment_and_future_frame_rejection(tmp_path):
    with h5py.File(tmp_path / "episode.hdf5", "w") as group:
        # The observed state must come from t, even if the post-action state differs.
        group["obs/joint_position"] = np.zeros((3, 9), np.float32)
        group["states/joint_position"] = np.ones((3, 9), np.float32)
        group["actions"] = np.full((3, 9), 2, np.float32)
        group["timestamps"] = np.arange(3) / 30
        for view in ("top", "wrist"):
            group[f"obs/camera_index/{view}"] = np.array([0, 0, 1])
            group[f"sensors/{view}/timestamps"] = np.array([0.0, 0.05])
            group[f"sensors/{view}/rgb"] = np.zeros((2, 256, 256, 3), np.uint8)
        recording = {
            "robot": "franka",
            "joint_names": converter.JOINT_NAMES,
            "action_schema": "joint_position_target; joint_names order; radians, finger metres",
            "control_dt_s": 1 / 30,
            "transition_schema": "obs=state_t; actions=command_t; states=state_t+1",
        }
        state, actions = converter.validate_episode(group, recording, {"num_transitions": 3})
        np.testing.assert_array_equal(state, 0)
        np.testing.assert_array_equal(actions, 2)
        group["obs/camera_index/top"][1] = 1
        with pytest.raises(ValueError, match="Future top image"):
            converter.validate_episode(group, recording, {"num_transitions": 3})


def test_action_round_trip_preserves_fingers_and_input():
    state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.02, 0.02], np.float32)
    actions = np.tile(state + 0.01, (16, 1))
    original = actions.copy()
    group = data_transforms(Pi0Config(pi05=True))
    result = transforms.compose(group.inputs)(
        {
            "observation/state": state,
            "observation/image": np.zeros((256, 256, 3), np.uint8),
            "observation/wrist_image": np.zeros((256, 256, 3), np.uint8),
            "actions": actions,
            "prompt": "pick up the object",
        }
    )
    np.testing.assert_allclose(result["actions"][:, :7], 0.01, atol=1e-7)
    np.testing.assert_array_equal(result["actions"][:, 7:], original[:, 7:])
    np.testing.assert_array_equal(actions, original)
    result["actions"] = np.pad(result["actions"], ((0, 0), (0, 23)))
    restored = transforms.compose(group.outputs)(result)
    np.testing.assert_allclose(restored["actions"], original, atol=1e-7)
    assert not result["image_mask"]["right_wrist_0_rgb"]


def test_normalization_chunks_repeat_episode_end_without_crossing_boundary():
    stats_spec = importlib.util.spec_from_file_location(
        "compute_norm_stats", Path(__file__).with_name("compute_norm_stats.py")
    )
    stats_module = importlib.util.module_from_spec(stats_spec)
    stats_spec.loader.exec_module(stats_module)
    states = np.zeros((3, 9), np.float32)
    states[:, :7] = np.array([10, 20, 30])[:, None]
    actions = states + 1
    actions[:, 7:] = np.array([0.04, 0.03, 0.02])[:, None]
    chunks = stats_module.action_chunks(states, actions, horizon=4)
    np.testing.assert_array_equal(chunks[:, :, 0], [[1, 11, 21, 21], [1, 11, 11, 11], [1, 1, 1, 1]])
    np.testing.assert_allclose(chunks[:, :, 7], [[0.04, 0.03, 0.02, 0.02], [0.03, 0.02, 0.02, 0.02], [0.02] * 4])
