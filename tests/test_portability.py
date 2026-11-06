import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

KIT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT / 'scripts'))
import bench
import transfer


class PackagingTests(unittest.TestCase):
    def test_all_sources_are_public_urls_and_immutable(self):
        for entry in bench.SOURCES.values():
            self.assertTrue(entry['url'].startswith('https://github.com/'))
            self.assertRegex(entry['commit'], r'^[a-f0-9]{40}$')
            self.assertNotIn('..', Path(entry['path']).parts)

    def test_locks_have_no_workstation_paths(self):
        for path in (KIT / 'locks').glob('*'):
            self.assertNotIn('/home/', path.read_text(), path)
            self.assertNotIn('file:///', path.read_text(), path)
        for model in bench.ENVS:
            self.assertIn('torch==', (KIT / 'locks' / f'{model}.txt').read_text())

    def test_scene_hashes_and_original_split(self):
        base = KIT / 'overlay/robot3'
        metadata = json.loads((base / 'dataset-inspection/episode-metadata.json').read_text())
        scenes = base / 'openpi/outputs/object-understanding-closed-loop/scenes'
        manifest = json.loads((scenes / 'manifest.json').read_text())
        self.assertEqual(len(manifest['episodes']), 148)
        self.assertEqual(sum(m['split'] == 'train' for m in metadata.values()), 584)
        self.assertEqual({e['source_episode'] for e in manifest['episodes']},
                         {name for name, m in metadata.items() if m['split'] == 'eval'})
        for entry in manifest['episodes']:
            self.assertEqual(bench.sha(scenes / entry['scene_file']), entry['sha256'])

    def test_overlay_manifest_protects_training_code_identity(self):
        manifest = json.loads((KIT / 'locks/overlay-sha256.json').read_text())
        actual = {str(p.relative_to(KIT)): bench.sha(p) for p in (KIT / 'overlay').rglob('*') if p.is_file()}
        self.assertEqual(actual, manifest)

    def test_launch_arguments_survive_paths_and_forwarding(self):
        for model in bench.MODELS:
            for command in ['train', 'eval']:
                out = subprocess.check_output([str(KIT / 'bench'), command, model, '--dry-run',
                                               '--output', '/tmp/path with spaces'], text=True,
                                              env=dict(os.environ, BENCH_RUNTIME='/tmp/runtime with spaces'))
                payload = json.loads(out)
                self.assertEqual(payload['argv'][-2:], ['--output', '/tmp/path with spaces'])
                self.assertTrue(payload['cwd'].startswith('/tmp/runtime with spaces/robot3/'))
                self.assertIn('/robot2/.venv-isaacsim/', payload['simulator_python'])

    def test_shell_environment_selects_correct_model(self):
        for model in bench.ENVS:
            env = bench.environment(model)
            self.assertEqual(env['PATH'].split(os.pathsep)[0], str(bench.venv(model) / 'bin'))
            self.assertIn(str(bench.WORKSPACE / 'openpi/packages/openpi-client/src'), env['PYTHONPATH'])
        self.assertEqual(bench.environment('gam')['HF_HUB_OFFLINE'], '1')

    def test_scheduler_dry_run_and_duplicate_gpu_rejection(self):
        path = KIT / 'scripts/parallel_runs.py'
        report = json.loads(subprocess.check_output([sys.executable, str(path), 'train', '--dry-run'], text=True))
        self.assertEqual(len(report['queue']), 5)
        self.assertEqual(len(report['gpu_slots']), 4)
        result = subprocess.run([sys.executable, str(path), 'train', '--dry-run', '--gpus', '0,0'], capture_output=True)
        self.assertNotEqual(result.returncode, 0)


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.bundle = self.root / 'bundle'
        self.target = self.root / 'runtime/robot3'
        self.name = 'Spatial-Forcing/openvla-SF/outputs/object-understanding/train/latest.pt'
        path = self.source / self.name
        path.parent.mkdir(parents=True)
        path.write_bytes(b'trusted-test-payload')
        os.utime(path, ns=(1234567890123456789, 1234567890123456789))
        os.link(path, path.with_name('step_000001.pt'))
        path.with_name('.lock').write_text('skip me')
        self.secret = self.source / '.cache/huggingface/token'
        self.secret.parent.mkdir(parents=True)
        self.secret.write_text('not-a-real-token-do-not-copy')
        with contextlib.redirect_stdout(io.StringIO()):
            transfer.export(self.source, self.bundle, ['sf'])

    def tearDown(self):
        self.temp.cleanup()

    def test_roundtrip_checksums_nanoseconds_hardlinks_and_allowlist(self):
        with contextlib.redirect_stdout(io.StringIO()):
            transfer.import_bundle(self.bundle, self.target)
        actual = self.target / self.name
        self.assertEqual(actual.read_bytes(), b'trusted-test-payload')
        self.assertEqual(actual.stat().st_mtime_ns, 1234567890123456789)
        self.assertEqual(actual.stat().st_ino, actual.with_name('step_000001.pt').stat().st_ino)
        self.assertFalse((self.bundle / 'robot3/.cache/huggingface/token').exists())
        self.assertFalse(actual.with_name('.lock').exists())

    def test_corruption_prevents_any_import(self):
        (self.bundle / 'robot3' / self.name).write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            transfer.import_bundle(self.bundle, self.target)
        self.assertFalse((self.target / self.name).exists())

    def test_different_existing_file_not_overwritten(self):
        target = self.target / self.name
        target.parent.mkdir(parents=True)
        target.write_bytes(b'existing-run')
        with self.assertRaises(FileExistsError):
            transfer.import_bundle(self.bundle, self.target)
        self.assertEqual(target.read_bytes(), b'existing-run')

    def test_traversal_and_symlink_escape_rejected(self):
        with self.assertRaises(ValueError):
            transfer.safe_relative('../outside')
        with self.assertRaises(ValueError):
            transfer.safe_relative('/outside')
        path = self.bundle / 'robot3' / self.name
        path.unlink()
        path.symlink_to(self.source / self.name)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            transfer.import_bundle(self.bundle, self.target)

    def test_export_cannot_recurse_inside_source(self):
        with self.assertRaises(ValueError):
            transfer.export(self.source, self.source / 'bad', ['sf'], dry_run=True)


if __name__ == '__main__':
    unittest.main()
