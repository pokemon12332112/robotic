"""Validate imported weights and copy checkpoint metadata to this runtime's paths.

Only deserialize checkpoints you exported from your trusted workspace. Tensor,
optimizer, RNG and step values are retained; original files are backed up.
"""
import importlib
import json
from pathlib import Path
import sys

from bench import RUNTIME, WORKSPACE, sha, source


def main(model):
    receipt = RUNTIME / 'installation/import-manifest.json'
    manifest = json.loads(receipt.read_text())
    if model == 'pi05':
        print('pi0.5 Orbax checkpoints use relative tensor paths; no metadata rewrite needed. Use a new evaluation output.')
        return
    sys.path.insert(0, str(source(model) / 'examples/object_understanding'))
    module = importlib.import_module('model' if model == 'gam' else 'adapter')
    weights = [module.PRETRAINED]
    if model == 'gam':
        weights.append(module.BASE)
    if model == 'sf':
        weights.append(source(model) / 'ckpts/vggt/model.pt')
    for root in weights:
        paths = [root] if root.is_file() else sorted(p for p in root.iterdir() if p.suffix in {'.json', '.model', '.safetensors'})
        for path in paths:
            name = str(path.relative_to(WORKSPACE))
            expected = manifest['files'].get(name)
            if expected is None or sha(path) != expected['sha256']:
                raise ValueError(f'Imported pretrained asset not verified: {path}')
    import torch
    run = source(model) / 'outputs/object-understanding/train'
    records = []
    for path in sorted(run.glob('*.pt')):
        name = str(path.relative_to(WORKSPACE))
        expected = manifest['files'].get(name)
        if expected is None:
            raise ValueError(f'Checkpoint not part of the imported bundle: {path}')
        backup = path.with_suffix('.pt.before-relocation')
        if backup.exists():
            module.load_checkpoint(path)
            continue
        if sha(path) != expected['sha256']:
            raise ValueError(f'Checkpoint changed after import: {path}')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint.get('format') != module.FORMAT or checkpoint.get('step', 0) < 1:
            raise ValueError('Unexpected checkpoint format or untrained adapter')
        if checkpoint['code_identity'] != module.code_identity():
            raise ValueError('Training implementation changed; restore the exact source/environment, not the identity field')
        # Fingerprints describe the hash-verified assets above. Never change model/code identity.
        checkpoint['pretrained'] = module.file_identity(module.PRETRAINED)
        if model == 'gam':
            checkpoint['base'] = module.file_identity(module.BASE)
        temporary = path.with_suffix('.pt.relocation-tmp')
        torch.save(checkpoint, temporary)
        path.replace(backup)
        temporary.replace(path)
        module.load_checkpoint(path)
        records.append({'file': str(path), 'step': checkpoint['step'],
                        'original_sha256': expected['sha256'], 'relocated_sha256': sha(path),
                        'changed_fields': ['pretrained', *(['base'] if model == 'gam' else [])]})
    if not records and not list(run.glob('*.pt')):
        raise FileNotFoundError(f'No imported checkpoints in {run}')
    if records:
        (run / 'relocation.json').write_text(json.dumps(records, indent=2) + '\n')
    print(f'{model}: validated relocated checkpoints; use a new evaluation output directory.')


if __name__ == '__main__':
    main(sys.argv[1])
