"""Check split isolation, temporal boundaries, joint-space semantics and feedback."""
import threading

import numpy as np
import pytest
import torch
from websockets.sync.server import serve
from openpi_client import msgpack_numpy

from data import HORIZON, RobotDataset, audit_datasets, compute_stats
from model import LoRALinear
from protocol import LocalPolicy, POLICY_TYPE, RecedingHorizonController
from serve import validate_observation


@pytest.fixture(scope='module')
def dataset():
    return RobotDataset(compute_stats())


def test_original_split_and_training_only_statistics(dataset):
    manifests, identity = audit_datasets()
    assert len(dataset) == 31996
    assert dataset.stats['dataset_identity'] == identity
    assert dataset.stats['split'] == 'train'
    assert not ({e['source_episode'] for e in manifests['train']['episodes']} &
                {e['source_episode'] for e in manifests['eval']['episodes']})
    # Recompute a training column independently, without touching eval Parquet.
    actions = np.concatenate([np.asarray(dataset.episode(e['converted_episode'])['actions'], dtype=np.float32)
                              for e in dataset.entries])
    np.testing.assert_allclose(dataset.stats['actions']['mean'], actions.mean(0), rtol=1e-5)


def test_episode_boundary_mask_and_joint_action_roundtrip(dataset):
    sample = dataset[int(dataset.ends[0]) - 1]
    assert sample['action_mask'].tolist() == [True] + [False] * (HORIZON - 1)
    assert not sample['future_valid']
    stats = dataset.stats['actions']
    restored = sample['actions'].numpy() * np.asarray(stats['std']) + np.asarray(stats['mean'])
    expected = dataset.episode(0)['actions'][-1]
    np.testing.assert_allclose(restored, np.tile(expected, (HORIZON, 1)), atol=1e-6)
    sample = dataset[0]
    rows = dataset.episode(0)
    restored_state = sample['state'].numpy() * np.asarray(dataset.stats['state']['std']) + np.asarray(dataset.stats['state']['mean'])
    np.testing.assert_allclose(restored_state, rows['state'][0], atol=1e-6)


def test_lora_starts_as_base_and_has_trainable_update():
    base = torch.nn.Linear(4, 3)
    layer = LoRALinear(base, rank=2)
    inputs = torch.randn(2, 4)
    torch.testing.assert_close(layer(inputs), base(inputs))
    layer(inputs).square().mean().backward()
    assert layer.b.grad.abs().sum() > 0
    assert base.weight.grad is None


def test_replans_on_new_observation_discards_tail():
    observations = []

    class Policy:
        def infer(self, obs):
            observations.append(obs)
            return np.full((HORIZON, 9), obs['state'][0], dtype=np.float32)

    controller = RecedingHorizonController(Policy(), execution_horizon=4)
    images = {view: {'rgb': np.zeros((256, 256, 3), dtype=np.uint8)} for view in ['top', 'wrist']}
    outputs = [controller.next_action(np.full(9, t), images, 'lift the cylinder')[0] for t in range(5)]
    assert outputs == [0, 0, 0, 0, 4]
    assert len(observations) == 2
    assert set(observations[0]) == {'state', 'top', 'wrist', 'prompt'}
    validate_observation(observations[0])
    with pytest.raises(ValueError):
        validate_observation({**observations[0], 'target_index': 1})


def test_real_websocket_protocol():
    expected = np.arange(HORIZON * 9, dtype=np.float32).reshape(HORIZON, 9)

    def handler(socket):
        socket.send(msgpack_numpy.packb({'policy_type': POLICY_TYPE, 'horizon': HORIZON, 'checkpoint': '/test'}))
        for message in socket:
            validate_observation(msgpack_numpy.unpackb(message))
            socket.send(msgpack_numpy.packb({'actions': expected}))

    with serve(handler, '127.0.0.1', 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        policy = LocalPolicy(server.socket.getsockname()[1], expected_checkpoint='/test')
        try:
            obs = {'state': np.zeros(9, dtype=np.float32), 'prompt': 'lift the cylinder',
                   'top': np.zeros((256, 256, 3), dtype=np.uint8),
                   'wrist': np.zeros((256, 256, 3), dtype=np.uint8)}
            np.testing.assert_array_equal(policy.infer(obs), expected)
        finally:
            policy.close()
            server.shutdown()
            thread.join(timeout=5)
