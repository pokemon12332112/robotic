"""Run one original held-out scene in Isaac Lab against a local pi0.5 server.

Run with the separate Isaac Lab interpreter. No demonstration actions are loaded.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from closed_loop_protocol import LocalPolicy
from closed_loop_protocol import RecedingHorizonController
import numpy as np


def run(args, payload):
    from core.timing import capture_stride
    from data_collection.simulation import _configure_stage
    from environments.cameras import CameraRig
    from environments.tabletop import scene_cfg
    from isaaclab.app import launch_simulation
    import isaaclab.sim as sim_utils
    from isaaclab.utils import instantiate
    from isaaclab_physx.physics import PhysxCfg
    from tasks.common.success import check_success
    import torch

    config = SimpleNamespace(**payload["run_config"]["task_settings"])
    if payload["recording"]["robot"] != "franka":
        raise ValueError("Expected the recorded Franka robot")
    if not np.isclose(config.PHYSICS_DT * config.CONTROL_STEPS, 1 / 30):
        raise ValueError("Expected 30 Hz control")
    config.TABLE_COLOR = tuple(config.TABLE_COLOR)
    samples = payload["scene"]["randomization_samples"]
    if samples["camera"] or samples["light"] or "table" in samples:
        raise ValueError("This adapter requires the dataset's fixed cameras, lights and table")
    specs = payload["scene"]["objects"]
    if any(np.any(spec.get("com_offset_m", [0, 0, 0])) for spec in specs):
        raise ValueError("Nonzero recorded COM offsets require an additional reset adapter")
    initial = payload["initial_state"]
    definition = scene_cfg(specs, record_depth=False, settings=config)
    definition.light.spawn.intensity = config.DOME_INTENSITY
    definition.key_light.spawn.intensity = config.KEY_INTENSITY
    definition.table.spawn.visual_material.diffuse_color = tuple(config.TABLE_COLOR)
    definition.table.spawn.visual_material.roughness = config.MATERIAL_ROUGHNESS
    sim_cfg = sim_utils.SimulationCfg(dt=config.PHYSICS_DT, device="cpu", physics=PhysxCfg())
    args.device = "cpu"
    stride = capture_stride(config.PHYSICS_DT, config.CONTROL_STEPS, config.CAMERA_HZ)
    dt = config.PHYSICS_DT * config.CONTROL_STEPS
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    with launch_simulation(SimpleNamespace(sim=sim_cfg, scene=definition), args):
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = instantiate(definition)
        _configure_stage(config)
        sim.reset()
        robot = scene["robot"]
        if list(robot.joint_names) != payload["recording"]["joint_names"]:
            raise ValueError("Robot joint order differs from the trained action space")
        objects = [scene[spec["name"]] for spec in specs]
        hand_ids, _ = robot.find_bodies("panda_hand")

        def tensor(value):
            return torch.tensor([value], dtype=torch.float32, device=sim.device)

        robot.write_root_pose_to_sim_index(root_pose=tensor(initial["robot_root_pose_xyzw"]))
        robot.write_joint_position_to_sim_index(position=tensor(initial["joint_position"]))
        robot.write_joint_velocity_to_sim_index(velocity=tensor(initial["joint_velocity"]))
        robot.set_joint_position_target_index(target=tensor(initial["joint_position"]))
        for i, obj in enumerate(objects):
            obj.write_root_pose_to_sim_index(root_pose=tensor(initial["object_pose_xyzw"][i]))
            obj.write_root_velocity_to_sim_index(root_velocity=tensor(initial["object_velocity"][i]))
        scene.reset()
        scene.write_data_to_sim()
        if config.OBJECT_START_ASLEEP:
            from omni.physx import get_physx_simulation_interface
            import omni.usd
            from pxr import PhysicsSchemaTools
            from pxr import Sdf

            stage_id = omni.usd.get_context().get_stage_id()
            for spec in specs:
                path = Sdf.Path("/World/envs/env_0/" + spec["name"])
                get_physx_simulation_interface().put_to_sleep(stage_id, PhysicsSchemaTools.sdfPathToInt(path))
        cameras = CameraRig(scene, robot, sim, config, record_depth=False)
        for _ in range(4):
            sim.render()
        images = cameras.capture(0.0)
        joints = robot.data.joint_pos.torch[0].cpu().numpy().copy()
        centres = np.asarray(initial["object_pose_xyzw"])[:, :3]
        target = payload["scene"]["scenario"]["target_index"]
        peak_lifts = np.zeros(len(objects))
        max_movement = np.zeros(len(objects))
        initial_error = float(np.abs(joints - initial["joint_position"]).max())
        actual_poses = np.stack([obj.data.root_pose_w.torch[0].cpu().numpy() for obj in objects])
        object_reset_error = float(np.abs(actual_poses - initial["object_pose_xyzw"]).max())
        if initial_error > 1e-4:
            raise ValueError(f"Robot reset mismatch: {initial_error}")
        if object_reset_error > 1e-4:
            raise ValueError(f"Object reset mismatch: {object_reset_error}")
        np.savez_compressed(
            output / "initial_observation.npz",
            joints=joints,
            **{name: sample["rgb"] for name, sample in images.items()},
        )
        policy = LocalPolicy(args.port, expected_checkpoint=args.checkpoint)
        policy.reset(args.seed)
        controller = RecedingHorizonController(policy, args.execution_horizon)
        video = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "512x512",
                "-r",
                str(1 / dt),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                str(output / "rollout.mp4"),
            ],
            stdin=subprocess.PIPE,
        )
        actions, states = [], []
        inference_times = controller.inference_times
        success, hold, physics_tick, clipped = False, 0.0, 0, 0
        reason = "timeout"
        try:
            for tick in range(round(config.TIMEOUT_SECONDS / dt)):
                if not sim.is_running():
                    raise RuntimeError("Simulator stopped before completing the rollout")
                next_action = controller.next_action(joints, images, payload["instruction"])
                frame = np.concatenate(
                    [
                        np.concatenate([images["top"]["rgb"], images["left"]["rgb"]], axis=1),
                        np.concatenate([images["right"]["rgb"], images["wrist"]["rgb"]], axis=1),
                    ],
                    axis=0,
                )
                video.stdin.write(frame.astype(np.uint8).tobytes())
                desired = tensor(next_action.tolist())
                limits = robot.data.soft_joint_pos_limits.torch
                bounded = desired.clamp(limits[..., 0], limits[..., 1])
                clipped += int(bool(torch.any(bounded != desired)))
                states.append(joints.copy())
                actions.append(bounded[0].cpu().numpy().copy())
                robot.set_joint_position_target_index(target=bounded)
                for _ in range(config.CONTROL_STEPS):
                    scene.write_data_to_sim()
                    sim.step(render=False)
                    scene.update(config.PHYSICS_DT)
                    cameras.sync()
                    physics_tick += 1
                    if physics_tick % stride == 0:
                        images = cameras.capture(physics_tick * config.PHYSICS_DT)
                joints = robot.data.joint_pos.torch[0].cpu().numpy().copy()
                current = np.stack([obj.data.root_pose_w.torch[0, :3].cpu().numpy() for obj in objects])
                hand = robot.data.body_pose_w.torch[0, hand_ids[0], :3].cpu().numpy()
                if not all(np.isfinite(x).all() for x in (joints, current, hand)):
                    raise ValueError("Nonfinite simulation state")
                peak_lifts = np.maximum(peak_lifts, current[:, 2] - centres[:, 2])
                max_movement = np.maximum(max_movement, np.linalg.norm(current - centres, axis=1))
                condition = check_success(current, centres, target, hand, joints[7:], settings=config)
                hold = hold + dt if condition else 0.0
                success = hold >= config.SUCCESS_HOLD_SECONDS
                if tick % 30 == 0:
                    print(f"step={tick} target_lift={peak_lifts[target]:.4f}m", flush=True)
                if success:
                    reason = "success"
                    break
        finally:
            policy.close()
            video.stdin.close()
            if video.wait(timeout=60) != 0:
                raise RuntimeError("Video encoding failed")
        selected = int(np.argmax(peak_lifts)) if peak_lifts.max() > 0.02 else None
        result = {
            "source_episode": payload["source_episode"],
            "scenario_id": payload["scenario_id"],
            "split": "eval",
            "evaluation": "closed_loop",
            "policy": "pi05_object_understanding",
            "checkpoint": args.checkpoint,
            "success": bool(success),
            "reason": reason,
            "steps": len(actions),
            "execution_horizon": args.execution_horizon,
            "target_index": target,
            "selected_index": selected,
            "correct_selection": selected == target,
            "peak_object_lifts_m": peak_lifts.tolist(),
            "max_object_movement_m": max_movement.tolist(),
            "clipped_action_steps": clipped,
            "initial_joint_max_error": initial_error,
            "initial_object_pose_max_error": object_reset_error,
            "seed": args.seed,
            "wall_seconds": time.monotonic() - started,
            "inference_seconds": inference_times,
            "instruction": payload["instruction"],
        }
        np.savez_compressed(output / "trajectory.npz", states=np.asarray(states), actions=np.asarray(actions))
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=Path("external_dependencies/spatial-understanding"))
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--execution-horizon", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    from isaaclab.app import add_launcher_args

    add_launcher_args(parser)
    args = parser.parse_args()
    if not 1 <= args.execution_horizon <= 16:
        parser.error("execution-horizon must be between 1 and 16")
    payload = json.loads(args.scene.read_text())
    if payload["split"] != "eval":
        raise ValueError("Only original evaluation scenes are allowed")
    sys.path.insert(0, str(args.benchmark.resolve()))
    run(args, payload)


if __name__ == "__main__":
    main()
