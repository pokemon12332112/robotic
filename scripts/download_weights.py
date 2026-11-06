"""Download pinned original initialization weights; never include these in Git."""
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.parse
import urllib.request

from bench import KIT, WORKSPACE, source


def crc32c(path):
    import google_crc32c
    digest = google_crc32c.Checksum()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode()


def main(model):
    specs = json.loads((KIT / 'locks/weights.json').read_text())[model]
    if model == 'pi05':
        destination = WORKSPACE / specs[0]['destination']
        for item in json.loads((KIT / 'locks/pi05-gcs.json').read_text()):
            relative = item['name'].removeprefix('checkpoints/pi05_base/')
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == int(item['size']) and crc32c(target) == item['crc32c']:
                continue
            url = ('https://storage.googleapis.com/download/storage/v1/b/openpi-assets/o/' +
                   urllib.parse.quote(item['name'], safe='') + '?alt=media&generation=' + item['generation'])
            temporary = target.with_name(target.name + '.download')
            print('Downloading pi05_base/' + relative, flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as output:
                while block := response.read(8 * 1024**2):
                    output.write(block)
            if temporary.stat().st_size != int(item['size']) or crc32c(temporary) != item['crc32c']:
                raise ValueError(f'GCS size/checksum mismatch: {target}')
            temporary.replace(target)
        return
    from huggingface_hub import snapshot_download
    # GAM runs offline to ensure T5 stays pinned. Downloads explicitly go online.
    import huggingface_hub.constants
    huggingface_hub.constants.HF_HUB_OFFLINE = False
    os.environ.pop('HF_HUB_OFFLINE', None)
    for spec in specs:
        kwargs = dict(repo_id=spec['repo'], revision=spec['revision'],
                      repo_type=spec.get('repo_type', 'model'),
                      allow_patterns=spec.get('allow_patterns'),
                      cache_dir=str(WORKSPACE / '.cache/huggingface/hub'))
        if not spec.get('cache_only'):
            kwargs['local_dir'] = str(WORKSPACE / spec['destination'])
        print(f"Downloading {spec['repo']} at {spec['revision']}", flush=True)
        snapshot_download(**kwargs)
        if spec.get('cache_only'):
            # The original conditioner loads by model ID; pin its local main ref
            # to the explicit revision, then use HF_HUB_OFFLINE during execution.
            ref = WORKSPACE / '.cache/huggingface/hub' / ('models--' + spec['repo'].replace('/', '--')) / 'refs/main'
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_text(spec['revision'])
    if model == 'sf':
        alias = source(model) / 'ckpts/model.pt'
        if not alias.exists():
            alias.symlink_to('vggt/model.pt')


if __name__ == '__main__':
    main(sys.argv[1])
