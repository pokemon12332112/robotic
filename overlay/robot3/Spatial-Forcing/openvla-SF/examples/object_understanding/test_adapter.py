"""Check split isolation, temporal boundaries, joint-space semantics and feedback."""
import threading

import numpy as np
import pytest
import torch
from websockets.sync.server import serve
from openpi_client import msgpack_numpy

from data import HORIZON, RobotDataset, audit_datasets, compute_stats
from adapter import PRETRAINED, prepare_tokens
from data import image_tensor, image_processor
from transformers import AutoTokenizer, LlamaConfig, LlamaModel
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
    assert 'future_images' not in sample
    assert sample['images'].shape == (2, 6, 224, 224)
    stats = dataset.stats['actions']
    restored = sample['actions'].numpy() * np.asarray(stats['std']) + np.asarray(stats['mean'])
    expected = dataset.episode(0)['actions'][-1]
    np.testing.assert_allclose(restored, np.tile(expected, (HORIZON, 1)), atol=1e-6)
    sample = dataset[0]
    rows = dataset.episode(0)
    restored_state = sample['state'].numpy() * np.asarray(dataset.stats['state']['std']) + np.asarray(dataset.stats['state']['mean'])
    np.testing.assert_allclose(restored_state, rows['state'][0], atol=1e-6)


def test_prompt_queries_padding_and_alignment():
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, local_files_only=True)
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = False
    ids, attention, queries, lengths = prepare_tokens(
        tokenizer, ['lift the cube', 'lift the smaller cylinder to the left of the large green cube'], 'cpu')
    assert queries.sum(1).tolist() == [72, 72]
    assert lengths[0] < lengths[1]
    assert attention[0, -1] == 0
    for i, length in enumerate(lengths):
        assert ids[i, length - 1] == 29871
        assert ids[i, length + 72] == 2
        assert torch.all(ids[i][queries[i]] == 1)
        assert attention[i].sum() == length + 73


def test_custom_attention_is_bidirectional_and_padding_is_masked():
    torch.manual_seed(2)
    config = LlamaConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4,
                         attention_dropout=0, use_cache=False)
    config._attn_implementation = 'sdpa'
    model = LlamaModel(config).eval()
    mask = torch.tensor([[1, 1, 1, 0]])
    with torch.no_grad():
        original = model(torch.tensor([[1, 2, 3, 0]]), attention_mask=mask).last_hidden_state
        future_changed = model(torch.tensor([[1, 2, 4, 0]]), attention_mask=mask).last_hidden_state
        padding_changed = model(torch.tensor([[1, 2, 3, 5]]), attention_mask=mask).last_hidden_state
    assert not torch.allclose(original[:, 0], future_changed[:, 0])
    torch.testing.assert_close(original[:, :3], padding_changed[:, :3])


def test_two_backbone_image_preprocessing():
    from PIL import Image
    rgb = np.random.default_rng(42).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    actual = image_tensor(rgb)
    expected = image_processor()(Image.fromarray(rgb), return_tensors='pt')['pixel_values'][0]
    assert actual.shape == (6, 224, 224)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual[:3], actual[3:])


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


def test_scene_audit_runs_in_compatible_environment_and_rejects_wrong_split(tmp_path):
    import json
    import subprocess
    from data import WORKSPACE
    from evaluate import checked_scenes
    scenes = WORKSPACE / 'openpi/outputs/object-understanding-closed-loop/scenes'
    report = checked_scenes(scenes)
    assert len(report['manifest']['episodes']) == 148
    assert report['manifest']['split'] == 'eval'
    invalid = dict(report['manifest'], split='train')
    (tmp_path / 'manifest.json').write_text(json.dumps(invalid))
    with pytest.raises(subprocess.CalledProcessError):
        checked_scenes(tmp_path)


def test_teacher_preprocessing_preserves_view_order_and_current_rgb():
    from vggt.utils.load_fn import preprocess_normed_images
    from prismatic.util.pooling_utils import custom_pooling
    views = [image_tensor(np.full((256, 256, 3), value, dtype=np.uint8)) for value in [51, 204]]
    pixels = preprocess_normed_images(torch.cat(views)[None], image_processor(), 2)
    assert pixels.shape == (1, 2, 3, 518, 518)
    torch.testing.assert_close(pixels[0, 0], torch.full_like(pixels[0, 0], 0.2))
    torch.testing.assert_close(pixels[0, 1], torch.full_like(pixels[0, 1], 0.8))
    features = torch.stack([torch.ones(1369, 32), torch.full((1369, 32), 2.0)])[None]
    reference = torch.empty(1, 512, 1)
    pooled = custom_pooling(features, (37, 37), (518, 518), reference, 'bilinear', False)
    assert pooled.shape == (1, 512, 32)
    torch.testing.assert_close(pooled[:, :256], torch.ones(1, 256, 32))
    torch.testing.assert_close(pooled[:, 256:], torch.full((1, 256, 32), 2.0))
    assert not torch.allclose(pooled, custom_pooling(features, (37, 37), (518, 518),
                                                   reference, 'bilinear', True))


def test_spatial_alignment_depends_on_teacher_and_backpropagates():
    from prismatic.models.projectors import AlignProjector
    torch.manual_seed(5)
    projector = AlignProjector(32, 16, 'cosine', use_vlm_norm=True)
    policy_features = torch.randn(1, 16, 32, requires_grad=True)
    teacher_features = torch.randn(1, 16, 32)
    loss = projector(policy_features, teacher_features)
    other = projector(policy_features, -teacher_features)
    assert not torch.allclose(loss, other)
    loss.backward()
    assert torch.isfinite(policy_features.grad).all() and policy_features.grad.norm() > 0
    assert projector.fc1.weight.grad.norm() > 0
    assert teacher_features.grad is None
