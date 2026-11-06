"""Export/import large artifacts separately from Git, preserving bytes and nanosecond mtimes."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from bench import KIT, MODELS, RUNTIME, WORKSPACE, sha

MODEL_DIR = {'pi05': 'openpi', 'gam': 'Geometric-Action-Model', 'spatialvla': 'SpatialVLA',
             'oft': 'openvla-oft', 'sf': 'Spatial-Forcing/openvla-SF'}


def artifact_paths(models):
    paths = ['data/lerobot/local/object-understanding-train', 'data/lerobot/local/object-understanding-eval',
             'openpi/assets/pi05_object_understanding_lora']
    weights = json.loads((KIT / 'locks/weights.json').read_text())
    for model in models:
        root = MODEL_DIR[model]
        paths.append(root + ('/checkpoints/pi05_object_understanding_lora' if model == 'pi05'
                             else '/outputs/object-understanding/train'))
        if model == 'pi05':
            paths.append(weights[model][0]['destination'])
        elif model == 'gam':
            paths += [root + '/checkpoints/track4world_da3.pth',
                      root + '/.cache/huggingface/download/checkpoints/track4world_da3.pth.metadata',
                      root + '/checkpoints_hf/3da-libero-gam/pretrained',
                      root + '/checkpoints_hf/3da-libero-gam/.cache/huggingface/download/pretrained',
                      '.cache/huggingface/hub/models--google-t5--t5-base/refs',
                      '.cache/huggingface/hub/models--google-t5--t5-base/snapshots']
        else:
            paths += [s['destination'] for s in weights[model]]
    return paths


def safe_relative(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Unsafe bundle member: {name}')
    return path


def export(workspace, destination, models, dry_run=False):
    workspace, destination = workspace.resolve(), destination.resolve()
    if destination == workspace or workspace in destination.parents:
        raise ValueError('Export outside the source workspace to prevent recursive copying')
    selected = [workspace / name for name in artifact_paths(models)]
    missing = [str(p.relative_to(workspace)) for p in selected if not p.exists()]
    if dry_run:
        print(json.dumps({'source': str(workspace), 'destination': str(destination),
                          'include': [str(p.relative_to(workspace)) for p in selected if p.exists()],
                          'missing': missing}, indent=2))
        return
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Choose an empty export destination')
    destination.mkdir(parents=True, exist_ok=True)
    entries, seen, hardlinks = {}, set(), {}
    for root in selected:
        if not root.exists():
            continue
        files = [root] if root.is_file() else sorted(root.rglob('*'))
        for original in files:
            if (not original.is_file() or original.name == 'relocation.json'
                or original.name.endswith(('.lock', '.tmp', '.incomplete', '.before-relocation', '.relocation-tmp'))):
                continue
            name = str(original.relative_to(workspace))
            if name in seen:
                continue
            seen.add(name)
            target = destination / 'robot3' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            before = original.stat()
            inode = (before.st_dev, before.st_ino)
            if inode in hardlinks:
                os.link(hardlinks[inode], target)
            else:
                shutil.copy2(original, target)  # Dereference HF cache links; never copy environments.
                hardlinks[inode] = target
            fingerprint = sha(target)
            after = original.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise RuntimeError(f'Source changed during export; stop training and export to a new folder: {original}')
            entries[name] = {'sha256': fingerprint, 'bytes': target.stat().st_size,
                             'mtime_ns': before.st_mtime_ns}
    manifest = {'format': 'robot_benchmark_artifacts_v1', 'models': models,
                'source_manifest_sha256': sha(KIT / 'locks/sources.json'),
                'missing': missing, 'files': entries}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Exported {len(entries)} files. Transfer this folder with rsync -aH (outside Git).')
    if missing:
        print('Missing optional/current run paths: ' + ', '.join(missing))


def import_bundle(bundle, workspace=WORKSPACE):
    bundle, workspace = bundle.resolve(), workspace.resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest['format'] != 'robot_benchmark_artifacts_v1':
        raise ValueError('Unknown artifact format')
    if manifest['source_manifest_sha256'] != sha(KIT / 'locks/sources.json'):
        raise ValueError('Bundle belongs to different pinned source revisions')
    # Verify every member before writing. Do not bypass existing differing artifacts.
    for name, entry in manifest['files'].items():
        path = safe_relative(name)
        original = bundle / 'robot3' / path
        target = workspace / path
        if not original.resolve().is_relative_to(bundle):
            raise ValueError(f'Bundle symlink escapes its root: {name}')
        if not target.resolve().is_relative_to(workspace):
            raise ValueError(f'Destination symlink escapes runtime: {name}')
        if original.stat().st_size != entry['bytes'] or sha(original) != entry['sha256']:
            raise ValueError(f'Bundle checksum mismatch: {name}')
        if target.exists() and sha(target) != entry['sha256']:
            raise FileExistsError(f'Different local artifact exists: {target}; use a fresh runtime')
    hardlinks = {}
    for name, entry in manifest['files'].items():
        original, target = bundle / 'robot3' / name, workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        key = (entry['sha256'], entry['mtime_ns'])
        if not target.exists():
            if key in hardlinks:
                os.link(hardlinks[key], target)
            else:
                shutil.copy2(original, target)
        os.utime(target, ns=(entry['mtime_ns'], entry['mtime_ns']))
        hardlinks[key] = target
    receipt = workspace.parent / 'installation/import-manifest.json'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Import verified. Run ./bench adopt MODEL inside the environment/container used for training.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('export')
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('import')
    p.add_argument('bundle', type=Path)
    args = parser.parse_args()
    if args.command == 'export':
        export(args.workspace, args.destination, args.models, args.dry_run)
    else:
        import_bundle(args.bundle)


if __name__ == '__main__':
    main()
