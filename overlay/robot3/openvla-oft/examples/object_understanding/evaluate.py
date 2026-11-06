"""Run/resume the original 148 held-out Isaac scenes with a trained OpenVLA-OFT adapter."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from data import WORKSPACE, audit_datasets
from adapter import ROOT, load_checkpoint
from protocol import LocalPolicy, POLICY_TYPE
from train import save_json

HERE = Path(__file__).resolve().parent
ROBOT2 = WORKSPACE.parent / 'robot2'


def checked_scenes(scenes):
    # The shared auditor uses Python 3.11+ syntax/APIs; OFT is pinned to Python 3.10.
    script = '''import importlib.util,json,sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('scene_audit', sys.argv[1])
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
manifest = audit.checked_scenes(Path(sys.argv[2]), Path(sys.argv[3]))
print(json.dumps({'manifest': manifest, 'benchmark_revision': audit.BENCHMARK_REVISION,
                  'isaaclab_revision': audit.ISAACLAB_REVISION}))
'''
    return json.loads(subprocess.check_output([
        str(WORKSPACE / 'openpi/.venv/bin/python'), '-c', script,
        str(WORKSPACE / 'openpi/examples/object_understanding/closed_loop.py'),
        str(scenes.resolve()), str(WORKSPACE / 'dataset-inspection/episode-metadata.json'),
    ], text=True))


def source_identity():
    digest = hashlib.sha256()
    for name in ['evaluate.py', 'episode.py', 'protocol.py', 'serve.py', 'adapter.py', 'data.py']:
        digest.update((HERE / name).read_bytes())
    digest.update((WORKSPACE / 'openpi/examples/object_understanding/closed_loop.py').read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'outputs/object-understanding/train/latest.pt')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/object-understanding/evaluation')
    parser.add_argument('--scenes', type=Path, default=WORKSPACE / 'openpi/outputs/object-understanding-closed-loop/scenes')
    parser.add_argument('--sim-python', type=Path, default=ROBOT2 / '.venv-isaacsim/bin/python')
    parser.add_argument('--benchmark', type=Path, default=ROBOT2 / 'Isaac-GR00T/external_dependencies/spatial-understanding')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--execution-horizon', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--port', type=int, default=8008)
    parser.add_argument('--episode-timeout', type=int, default=1800)
    args = parser.parse_args()
    if not 1 <= args.execution_horizon <= 8 or (args.limit is not None and args.limit < 1):
        parser.error('Execution horizon must be 1..8 and limit must be positive')
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    _, identity = audit_datasets()
    if checkpoint['stats']['dataset_identity'] != identity:
        raise ValueError('Checkpoint training dataset does not match the audited source')
    # Reuse the established, read-only audit/reset validation from pi0.5.
    audit = checked_scenes(args.scenes)
    manifest = audit['manifest']
    for path, expected in [(args.benchmark, audit['benchmark_revision']),
                           (args.benchmark.parent / 'IsaacLab', audit['isaaclab_revision'])]:
        revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != expected:
            raise ValueError(f'Unexpected simulator source revision: {path}')
    episodes = manifest['episodes'][:args.limit]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Copy the committed adapter into this run: training may replace latest.pt.
        fingerprint = checkpoint['_loaded_sha256']
        runtime = json.loads(subprocess.check_output([str(args.sim_python), '-c',
            "import json,importlib.metadata as m;print(json.dumps({k:m.version(k) for k in ['isaacsim','isaaclab','torch','numpy']}))"], text=True))
        config = {'policy': POLICY_TYPE, 'checkpoint_source': str(checkpoint_path),
                  'checkpoint_sha256': fingerprint, 'training_step': checkpoint['step'],
                  'execution_horizon': args.execution_horizon, 'seed': args.seed,
                  'dataset_identity': identity, 'evaluator_identity': source_identity(),
                  'scene_manifest_sha256': hashlib.sha256((args.scenes / 'manifest.json').read_bytes()).hexdigest(),
                  'runtime': runtime, 'benchmark_revision': audit['benchmark_revision'],
                  'isaaclab_revision': audit['isaaclab_revision']}
        config_path = output / 'configuration.json'
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError('Existing output uses different weights or settings; choose a new --output')
        save_json(config_path, config)
        snapshot = output / 'adapter.pt'
        if not snapshot.exists():
            # Use the already loaded object to avoid a latest.pt replacement race.
            import torch
            temporary = output / 'adapter.pt.tmp'
            checkpoint['_source_sha256'] = fingerprint
            torch.save(checkpoint, temporary)
            temporary.replace(snapshot)
        else:
            if load_checkpoint(snapshot).get('_source_sha256') != fingerprint:
                raise ValueError('Cached adapter snapshot differs from this evaluation')
        checkpoint_name = str(snapshot.resolve())
        status = {'stage': 'starting_policy', 'completed_episodes': 0,
                  'requested_episodes': len(episodes), 'policy': POLICY_TYPE}
        save_json(output / 'status.json', status)
        env = dict(os.environ, PYTHONUNBUFFERED='1')
        server = None
        results = []
        try:
            with (output / 'policy-server.log').open('a') as log:
                server = subprocess.Popen([sys.executable, str(HERE / 'serve.py'), '--checkpoint',
                                           checkpoint_name, '--port', str(args.port)], env=env,
                                          stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                deadline = time.monotonic() + 600
                while True:
                    if server.poll() is not None:
                        raise RuntimeError('OpenVLA-OFT server exited; see policy-server.log')
                    try:
                        probe = LocalPolicy(args.port, expected_checkpoint=checkpoint_name)
                        probe.close()
                        break
                    except (OSError, TimeoutError):
                        if time.monotonic() > deadline:
                            raise TimeoutError('OpenVLA-OFT server did not become ready')
                        time.sleep(1)
                sim_env = dict(env, PYTHONPATH=str(WORKSPACE / 'openpi/packages/openpi-client/src'), HEADLESS='1')
                # Do not leak OpenVLA-OFT's Python/graphics runtime into the Isaac environment.
                for key in ['DISPLAY', 'LD_LIBRARY_PATH', 'PYOPENGL_PLATFORM', 'MUJOCO_GL', 'EGL_PLATFORM']:
                    sim_env.pop(key, None)
                for entry in episodes:
                    destination = output / entry['source_episode']
                    destination.mkdir(exist_ok=True)
                    result_path = destination / 'result.json'
                    seed = args.seed + int(entry['source_episode'].split('_')[-1])
                    status.update(stage='running', current_episode=entry['source_episode'])
                    save_json(output / 'status.json', status)
                    if not result_path.exists():
                        with (destination / 'simulator.log').open('w') as sim_log:
                            subprocess.run([str(args.sim_python), str(HERE / 'episode.py'),
                                '--benchmark', str(args.benchmark.resolve()), '--scene',
                                str((args.scenes / entry['scene_file']).resolve()), '--output', str(destination),
                                '--checkpoint', checkpoint_name, '--port', str(args.port),
                                '--execution-horizon', str(args.execution_horizon), '--seed', str(seed),
                                '--device', 'cpu'], env=sim_env, stdout=sim_log, stderr=subprocess.STDOUT,
                                check=True, timeout=args.episode_timeout, cwd=ROOT)
                    result = json.loads(result_path.read_text())
                    if (result['source_episode'] != entry['source_episode'] or result['split'] != 'eval'
                        or result['scenario_id'] != entry['scenario_id'] or result['checkpoint'] != checkpoint_name
                        or result['execution_horizon'] != args.execution_horizon or result['seed'] != seed):
                        raise ValueError('Cached episode result does not match this run')
                    results.append(result)
                    save_json(output / 'metrics.json', {**config, 'split': 'eval', 'evaluation': 'closed_loop',
                        'completed_episodes': len(results), 'requested_episodes': len(episodes),
                        'success_rate': sum(r['success'] for r in results) / len(results),
                        'correct_selection_rate': sum(r['correct_selection'] for r in results) / len(results),
                        'episodes': results})
                    status['completed_episodes'] = len(results)
                    save_json(output / 'status.json', status)
                    print(f'Completed {len(results)}/{len(episodes)} OpenVLA-OFT episodes', flush=True)
                status.update(stage='complete', current_episode=None)
        except BaseException as error:
            status.update(stage='error', error=str(error))
            raise
        finally:
            save_json(output / 'status.json', status)
            if server is not None:
                server.terminate()
                try:
                    server.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == '__main__':
    main()
