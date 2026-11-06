"""Serve the fine-tuned policy with explicit per-episode sampling noise."""

import argparse
import logging
from pathlib import Path

from closed_loop_protocol import NOISE_SHAPE
from closed_loop_protocol import POLICY_TYPE
import numpy as np


class SeededPolicy:
    def __init__(self, policy):
        self.policy = policy

    def infer(self, observation):
        obs = dict(observation)
        noise = np.asarray(obs.pop("_noise"), dtype=np.float32)
        if noise.shape != NOISE_SHAPE or not np.isfinite(noise).all():
            raise ValueError("Invalid sampling noise")
        expected = {"observation/state", "observation/image", "observation/wrist_image", "prompt"}
        if set(obs) != expected:
            raise ValueError("Unexpected policy input fields")
        return self.policy.infer(obs, noise=noise)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8005)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    if not (checkpoint / "params/_METADATA").is_file():
        parser.error("Expected a completed OpenPI training checkpoint containing params/_METADATA")
    # Import the model only in the policy process, never in Isaac Sim.
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training.config import get_config

    config = get_config("pi05_object_understanding_lora")
    policy = policy_config.create_trained_policy(config, checkpoint)
    server = WebsocketPolicyServer(
        SeededPolicy(policy),
        host="127.0.0.1",
        port=args.port,
        metadata={
            "policy_type": POLICY_TYPE,
            "checkpoint": str(checkpoint),
            "noise_shape": list(NOISE_SHAPE),
            "action_dim": 9,
            "views": ["top", "wrist"],
        },
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
