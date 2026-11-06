"""Franka Panda inputs and nine joint-position outputs for object-understanding."""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model
from openpi.policies import libero_policy


@dataclasses.dataclass(frozen=True)
class ObjectUnderstandingInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # This shared adapter handles top/wrist images and masks the unused third camera.
        result = libero_policy.LiberoInputs(model.ModelType.PI05)(data)
        if result["state"].shape != (9,):
            raise ValueError("Expected seven Panda arm joints plus two finger positions")
        if "actions" in result:
            if result["actions"].shape[-1] != 9:
                raise ValueError("Expected nine joint-position action targets")
            # DeltaActions operates in place; do not mutate the caller's action array.
            result["actions"] = np.array(result["actions"], copy=True)
        return result


@dataclasses.dataclass(frozen=True)
class ObjectUnderstandingOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :9]}


def data_transforms(_model_config):
    # Model predicts relative arm joints and absolute fingers. Inference reverses this.
    mask = transforms.make_bool_mask(7, -2)
    return transforms.Group(
        inputs=[ObjectUnderstandingInputs(), transforms.DeltaActions(mask)],
        outputs=[transforms.AbsoluteActions(mask), ObjectUnderstandingOutputs()],
    )
