"""Validate split isolation and one real training batch; optionally perform one GPU update.

The optional update is discarded. No checkpoint or external experiment log is written.
"""

import argparse
import dataclasses
import functools
import importlib.util
import json
import os
from pathlib import Path

import jax
from lerobot.common.constants import HF_LEROBOT_HOME
import numpy as np

from openpi.training import config
from openpi.training import data_loader
from openpi.training import sharding


def main(*, gpu_step=False):
    cfg = dataclasses.replace(config.get_config("pi05_object_understanding_lora"), num_workers=0)
    manifests = {
        split: json.loads((HF_LEROBOT_HOME / f"local/object-understanding-{split}" / "conversion.json").read_text())
        for split in ("train", "eval")
    }
    for split, manifest in manifests.items():
        assert manifest["source_split"] == split
        assert all(e["split"] == split for e in manifest["episodes"])
    for key in ("source_episode", "scenario_id"):
        assert not (
            {e[key] for e in manifests["train"]["episodes"]} & {e[key] for e in manifests["eval"]["episodes"]}
        ), f"Split overlap: {key}"
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    batch = next(iter(data_loader.create_data_loader(cfg, sharding=data_sharding, num_batches=1)))
    observation, actions = batch
    assert actions.shape == (1, 16, 32)
    assert observation.tokenized_prompt.shape == (1, 200)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(batch))
    report = {
        "train_episodes": len(manifests["train"]["episodes"]),
        "test_episodes": len(manifests["eval"]["episodes"]),
        "actions_shape": list(actions.shape),
        "gpu_step": False,
    }
    print("Split isolation and normalized training batch: OK", flush=True)
    if gpu_step:
        if not any(device.platform == "gpu" for device in jax.devices()):
            raise RuntimeError("Run --gpu-step from a GPU-enabled terminal")
        jax.config.update("jax_compilation_cache_dir", os.environ["JAX_COMPILATION_CACHE_DIR"])
        spec = importlib.util.spec_from_file_location(
            "openpi_train_script", Path(__file__).resolve().parents[2] / "scripts/train.py"
        )
        train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train)
        train_rng, init_rng = jax.random.split(jax.random.key(cfg.seed))
        print("Loading base weights and initializing LoRA optimizer...", flush=True)
        state, state_sharding = train.init_train_state(cfg, init_rng, mesh, resume=False)
        step = jax.jit(
            functools.partial(train.train_step, cfg),
            in_shardings=(replicated, state_sharding, data_sharding),
            out_shardings=(state_sharding, replicated),
            donate_argnums=(1,),
        )
        print("Compiling and running one training step...", flush=True)
        with sharding.set_mesh(mesh):
            state, metrics = step(train_rng, state, batch)
        metrics = jax.device_get(metrics)
        assert int(state.step) == 1
        assert all(np.isfinite(value).all() for value in metrics.values())
        report.update(gpu_step=True, metrics={key: float(value) for key, value in metrics.items()})
        print("One GPU training step passed:", report["metrics"], flush=True)
    output = cfg.assets_dirs / "validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-step", action="store_true")
    main(**vars(parser.parse_args()))
