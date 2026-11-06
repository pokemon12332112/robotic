"""Check imports/source identity; optionally exercise the selected GPU runtime."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from bench import KIT, ENVS, source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=list(ENVS))
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    assert sys.version.split()[0] == ENVS[args.model]['python'], sys.version
    import torch
    if args.model == 'isaac':
        assert importlib.metadata.version('isaacsim') == '6.2.0.0'
        assert importlib.metadata.version('isaaclab') == '28.0.1'
    elif args.model == 'pi05':
        from openpi.training.config import get_config
        assert get_config('pi05_object_understanding_lora').batch_size == 1
    else:
        sys.path.insert(0, str(source(args.model) / 'examples/object_understanding'))
        __import__('model' if args.model == 'gam' else 'adapter')
    if args.model in {'oft', 'sf'}:
        from transformers.models.llama import modeling_llama
        expected = json.loads((KIT / 'locks/attention.json').read_text())[args.model]['transformers_llama_sha256']
        assert hashlib.sha256(Path(modeling_llama.__file__).read_bytes()).hexdigest() == expected
    report = {'model': args.model, 'python': sys.version.split()[0], 'torch': torch.__version__, 'imports': 'passed'}
    if args.gpu:
        assert torch.cuda.is_available(), 'CUDA is unavailable'
        assert torch.isfinite(torch.ones(32, device='cuda').square()).all()
        report['gpu'] = torch.cuda.get_device_name()
        if args.model == 'pi05':
            os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
            import jax
            import jax.numpy as jnp
            assert any(d.platform == 'gpu' for d in jax.devices())
            jax.block_until_ready(jnp.ones(32).sum())
        report['scope'] = 'GPU arithmetic only; run a one-scene evaluation to validate RTX rendering'
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
