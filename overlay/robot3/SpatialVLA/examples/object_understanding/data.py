"""Read the audited, causally aligned local conversion; never resplit test data."""
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

WORKSPACE = Path(__file__).resolve().parents[3]
DATA_ROOT = WORKSPACE / 'data/lerobot/local'
REVISION = 'e3dbcd59b906768df0a2e7baa8b3124c6b0aaec1'
HORIZON = 8


def image_tensor(rgb):
    image = Image.fromarray(rgb) if isinstance(rgb, np.ndarray) else rgb
    image = image.convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255


def audit_datasets(root=DATA_ROOT):
    audit_path = WORKSPACE / 'dataset-inspection/episode-metadata.json'
    audit = json.loads(audit_path.read_text())
    manifests = {}
    for split, count in [('train', 584), ('eval', 148)]:
        path = Path(root) / f'object-understanding-{split}' / 'conversion.json'
        manifest = json.loads(path.read_text())
        assert manifest['source_repo'] == 'phulocnguyen/object-understanding'
        assert manifest['revision'] == REVISION and manifest['source_split'] == split
        expected = {k for k, v in audit.items() if v['split'] == split}
        entries = manifest['episodes']
        assert len(entries) == count == len(expected)
        assert {e['source_episode'] for e in entries} == expected
        assert {e['converted_episode'] for e in entries} == set(range(count))
        for entry in entries:
            original = audit[entry['source_episode']]
            for field in ['scenario_id', 'prompt', 'num_transitions', 'split']:
                assert entry[field] == original[field], (entry['source_episode'], field)
        manifests[split] = manifest
    assert not ({e['scenario_id'] for e in manifests['train']['episodes']} &
                {e['scenario_id'] for e in manifests['eval']['episodes']})
    digest = hashlib.sha256(audit_path.read_bytes())
    for split in ['train', 'eval']:
        digest.update(json.dumps(manifests[split], sort_keys=True).encode())
    return manifests, digest.hexdigest()


def parquet_path(root, index):
    return Path(root) / 'data' / f'chunk-{index // 1000:03d}' / f'episode_{index:06d}.parquet'


def compute_stats(root=DATA_ROOT):
    manifests, identity = audit_datasets(root)
    training = Path(root) / 'object-understanding-train'
    arrays = {'state': [], 'actions': []}
    digest = hashlib.sha256()
    for entry in manifests['train']['episodes']:
        path = parquet_path(training, entry['converted_episode'])
        # Include file contents in provenance, not only manifest metadata.
        digest.update(path.read_bytes())
        table = pq.read_table(path, columns=list(arrays))
        for key in arrays:
            values = np.asarray(table[key].to_pylist(), dtype=np.float32)
            if values.shape != (entry['num_transitions'], 9) or not np.isfinite(values).all():
                raise ValueError(f'Invalid {key} in {path}')
            arrays[key].append(values)
    stats = {'dataset_identity': identity, 'train_content_sha256': digest.hexdigest(),
             'source_revision': REVISION, 'split': 'train', 'episodes': 584, 'horizon': HORIZON,
             'action_semantics': 'absolute_joint_position_target', 'views': ['top', 'wrist']}
    for key, pieces in arrays.items():
        values = np.concatenate(pieces)
        stats[key] = {'mean': values.mean(axis=0).tolist(),
                      'std': np.maximum(values.std(axis=0), 1e-3).tolist()}
    return stats


class RobotDataset(Dataset):
    def __init__(self, stats, root=DATA_ROOT):
        manifests, identity = audit_datasets(root)
        if identity != stats['dataset_identity'] or stats['split'] != 'train':
            raise ValueError('Normalization provenance does not match the training split')
        self.root = Path(root) / 'object-understanding-train'
        self.entries = sorted(manifests['train']['episodes'], key=lambda e: e['converted_episode'])
        self.ends = np.cumsum([e['num_transitions'] for e in self.entries])
        self.stats = stats

    def __len__(self):
        return int(self.ends[-1])

    @lru_cache(maxsize=2)
    def episode(self, index):
        return pq.read_table(parquet_path(self.root, index)).to_pydict()

    def __getitem__(self, index):
        episode = int(np.searchsorted(self.ends, index, side='right'))
        t = int(index - (self.ends[episode - 1] if episode else 0))
        entry = self.entries[episode]
        rows = self.episode(entry['converted_episode'])
        length = len(rows['actions'])

        def images(frame):
            return torch.stack([image_tensor(Image.open(io.BytesIO(rows[key][frame]['bytes'])))
                                for key in ('image', 'wrist_image')])

        indices = np.minimum(np.arange(t, t + HORIZON), length - 1)
        state = np.asarray(rows['state'][t], dtype=np.float32)
        actions = np.asarray(rows['actions'], dtype=np.float32)[indices]
        return {'images': images(t),
                'state': torch.tensor((state - np.array(self.stats['state']['mean'])) /
                                      np.array(self.stats['state']['std']), dtype=torch.float32),
                'actions': torch.tensor((actions - np.array(self.stats['actions']['mean'])) /
                                        np.array(self.stats['actions']['std']), dtype=torch.float32),
                'action_mask': torch.tensor(np.arange(t, t + HORIZON) < length),
                'prompt': entry['prompt'], 'source_episode': entry['source_episode']}
