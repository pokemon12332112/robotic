"""Portable benchmark launcher. Uses only stdlib until entering a model environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys

KIT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get('BENCH_RUNTIME', KIT / 'runtime')).expanduser().resolve()
WORKSPACE = RUNTIME / 'robot3'
SOURCES = json.loads((KIT / 'locks/sources.json').read_text())
ENVS = json.loads((KIT / 'locks/environments.json').read_text())
MODELS = ('pi05', 'gam', 'spatialvla', 'oft', 'sf')


def run(args, **kwargs):
    print('+ ' + shlex.join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def source(model):
    return RUNTIME / ENVS[model]['source']


def venv(model):
    if model == 'isaac':
        return RUNTIME / 'robot2/.venv-isaacsim'
    if model == 'sf':
        return source(model).parent / '.venv'
    return source(model) / '.venv'


def environment(model):
    env = dict(os.environ)
    for name in ['PYTHONHOME', 'VIRTUAL_ENV', 'PYTHONPATH', 'LD_LIBRARY_PATH',
                 'PYOPENGL_PLATFORM', 'MUJOCO_GL', 'EGL_PLATFORM']:
        env.pop(name, None)
    root = source(model)
    paths = [str(root), str(WORKSPACE / 'openpi/packages/openpi-client/src')]
    if model == 'pi05':
        paths.insert(0, str(root / 'src'))
    if model == 'gam':
        paths = [str(root / 'src'), str(root / 'Depth-Anything-3/src'), *paths]
        env.update(DA3_ROOT=str(root), DA3_CODE_ROOT=str(root),
                   DA3_BASE_CKPT=str(root / 'checkpoints/track4world_da3.pth'))
        # Use the pinned cached snapshot; never silently refresh T5's main branch.
        env['HF_HUB_OFFLINE'] = '1'
    if model in {'oft', 'sf'}:
        paths.append(str(root / 'LIBERO'))
        env['LIBERO_CONFIG_PATH'] = str(root / '.local/libero_config')
    cache = WORKSPACE / '.cache'
    env.update(PYTHONPATH=os.pathsep.join(paths), VIRTUAL_ENV=str(venv(model)),
               PATH=str(venv(model) / 'bin') + os.pathsep + env.get('PATH', ''),
               UV_CACHE_DIR=str(RUNTIME / '.uv-cache'), UV_PYTHON_INSTALL_DIR=str(RUNTIME / '.uv-python'),
               HF_HOME=str(cache / 'huggingface'), HF_LEROBOT_HOME=str(WORKSPACE / 'data/lerobot'),
               OPENPI_DATA_HOME=str(cache / 'openpi'), XDG_CACHE_HOME=str(cache / model),
               MPLCONFIGDIR=str(cache / model / 'matplotlib'), NUMBA_CACHE_DIR=str(cache / model / 'numba'),
               TORCH_EXTENSIONS_DIR=str(cache / model / 'torch_extensions'),
               JAX_COMPILATION_CACHE_DIR=str(cache / 'jax'),
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', TF_CPP_MIN_LOG_LEVEL='2',
               GIT_LFS_SKIP_SMUDGE='1', UV_LINK_MODE='copy',
               XLA_PYTHON_CLIENT_MEM_FRACTION=env.get('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9'))
    return env


def selected(model, simulator=True):
    return [*MODELS, *(['isaac'] if simulator else [])] if model == 'all' else [model]


def sources(local_cache=None):
    """Clone immutable revisions and apply the reviewed source overlay without clobbering edits."""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    git_env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
    for key, spec in SOURCES.items():
        target = RUNTIME / spec['path']
        if not (target / '.git').exists():
            if target.exists() and any(target.iterdir()):
                raise RuntimeError(f'Refusing nonempty non-git source directory: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            if local_cache:
                origin = Path(local_cache) / spec['path']
                run(['git', 'clone', '--no-local', '--no-checkout', origin, target], env=git_env)
                run(['git', '-C', target, 'remote', 'set-url', 'origin', spec['url']])
            else:
                run(['git', 'init', target])
                run(['git', '-C', target, 'remote', 'add', 'origin', spec['url']])
                run(['git', '-C', target, 'fetch', '--depth', '1', 'origin', spec['commit']], env=git_env)
            run(['git', '-C', target, 'checkout', '--detach', spec['commit']], env=git_env)
        revision = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != spec['commit']:
            raise RuntimeError(f'{target}: revision changed; use a new BENCH_RUNTIME')
        if spec.get('initialize_submodules'):
            run(['git', '-C', target, 'submodule', 'update', '--init', '--recursive'], env=git_env)
        if 'patch' in spec:
            patch = KIT / spec['patch']
            applied = subprocess.run(['git', '-C', str(target), 'apply', '--reverse', '--check', str(patch)],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            if not applied:
                run(['git', '-C', target, 'apply', '--check', patch])
                run(['git', '-C', target, 'apply', patch])
    for original in sorted((KIT / 'overlay').rglob('*')):
        if not original.is_file():
            continue
        target = RUNTIME / original.relative_to(KIT / 'overlay')
        if target.exists() and sha(target) != sha(original):
            raise RuntimeError(f'Local overlay edit detected, refusing overwrite: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(original, target)
    print('Pinned sources and benchmark overlay ready.')


def install_env(model):
    if not source(model).is_dir():
        raise RuntimeError('Run ./bench sources first')
    if platform.machine() != 'x86_64' or sys.platform != 'linux':
        raise RuntimeError('These verified dependency locks target Linux x86_64; use the Docker setup')
    if model == 'isaac' and os.environ.get('ACCEPT_EULA', '').upper() != 'Y':
        raise RuntimeError('Read the NVIDIA license, then set ACCEPT_EULA=Y to install/run Isaac')
    env = environment(model)
    spec = ENVS[model]
    run(['uv', 'python', 'install', spec['python']], env=env)
    if not (venv(model) / 'bin/python').exists():
        run(['uv', 'venv', '--python', spec['python'], venv(model)], env=env)
    lines = (KIT / 'locks' / f'{model}.txt').read_text().replace('${RUNTIME}', str(RUNTIME)).splitlines()
    editable = [line[3:] for line in lines if line.startswith('-e ')]
    ordinary = [line for line in lines if not line.startswith('-e ')]
    generated = RUNTIME / 'installation'
    generated.mkdir(exist_ok=True)
    resolved = generated / f'{model}.txt'
    resolved.write_text('\n'.join(line for line in ordinary if not line.startswith('deepspeed==')) + '\n')
    command = ['uv', 'pip', 'install', '--python', venv(model) / 'bin/python', '--no-deps',
               '--index-strategy', 'unsafe-best-match', '--prerelease', 'allow',
               '--extra-index-url', 'https://download.pytorch.org/whl/' + spec['cuda_index']]
    if model == 'isaac':
        command += ['--extra-index-url', 'https://pypi.nvidia.com']
    # Some frozen sdists import torch during their build; install the complete
    # torch runtime and build tools first, before building those sdists.
    bootstrap = [line for line in ordinary if line.split('==')[0] in {'torch', 'torchvision', 'torchaudio',
                 'setuptools', 'wheel', 'packaging', 'ninja', 'numpy', 'typing-extensions', 'filelock',
                 'sympy', 'networkx', 'jinja2', 'markupsafe', 'fsspec', 'triton'} or line.startswith('nvidia-')]
    run([*command, *bootstrap], env=env)
    env['DS_BUILD_OPS'] = '0'
    run([*command, '-r', resolved], env=env)
    for requirement in ordinary:
        if requirement.startswith('deepspeed=='):
            run([*command, '--no-build-isolation', requirement], env=env)
    for path in editable:
        run([*command, '-e', path], env=env)
    if model == 'pi05':
        site = subprocess.check_output([str(venv(model) / 'bin/python'), '-c',
                                      'import sysconfig;print(sysconfig.get_paths()["purelib"])'], text=True).strip()
        for original in (source(model) / 'src/openpi/models_pytorch/transformers_replace').rglob('*.py'):
            target = Path(site) / 'transformers' / original.relative_to(source(model) / 'src/openpi/models_pytorch/transformers_replace')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix('.replacement')
            shutil.copy2(original, temporary)
            temporary.replace(target)
    if model in {'oft', 'sf'}:
        config = source(model) / '.local/libero_config/config.yaml'
        config.parent.mkdir(parents=True, exist_ok=True)
        libero = source(model) / 'LIBERO/libero/libero'
        config.write_text('\n'.join(f'{k}: {v}' for k, v in {
            'benchmark_root': libero, 'bddl_files': libero / 'bddl_files', 'init_states': libero / 'init_files',
            'datasets': WORKSPACE / 'data/libero-unused', 'assets': libero / 'assets'}.items()) + '\n')
    # Record diagnostics, including known upstream metadata conflicts documented below.
    check = subprocess.run(['uv', 'pip', 'check', '--python', str(venv(model) / 'bin/python')],
                           env=env, text=True, capture_output=True)
    (generated / f'{model}-pip-check.txt').write_text(check.stdout + check.stderr)
    if check.returncode and model not in {'gam', 'isaac'}:
        raise RuntimeError(f'Dependency check failed; see {generated}/{model}-pip-check.txt')
    run([venv(model) / 'bin/python', KIT / 'scripts/check_environment.py', model], env=env)
    (generated / f'{model}-ready.json').write_text(json.dumps({
        'lock_sha256': sha(KIT / 'locks' / f'{model}.txt'), 'python': spec['python']}, indent=2) + '\n')


def python(model, args, *, replace=False):
    executable = venv(model) / 'bin/python'
    if not executable.exists():
        raise RuntimeError(f'Missing environment: ./bench env {model}')
    if replace:
        os.chdir(source(model))
        os.execve(str(executable), [str(executable), *map(str, args)], environment(model))
    return run([executable, *args], cwd=source(model), env=environment(model))


def data(extra):
    for split in ['train', 'eval']:
        destination = WORKSPACE / f'data/lerobot/local/object-understanding-{split}'
        if not (destination / 'conversion.json').is_file():
            python('pi05', ['examples/object_understanding/convert_dataset.py', '--metadata',
                           WORKSPACE / 'dataset-inspection/episode-metadata.json', '--split', split, *extra])
    python('pi05', ['examples/object_understanding/compute_norm_stats.py'])


def launch(action, model, extra, dry_run=False):
    if model not in MODELS:
        raise ValueError('Choose a policy model')
    if action == 'train':
        args = (['scripts/train.py', 'pi05_object_understanding_lora', '--exp-name', 'object_understanding_v1',
                 '--log-interval', '1', '--save-interval', '1000', '--keep-period', '1']
                if model == 'pi05' else ['examples/object_understanding/train.py'])
    else:
        if not dry_run and os.environ.get('ACCEPT_EULA', '').upper() != 'Y':
            raise RuntimeError('Set ACCEPT_EULA=Y after accepting NVIDIA Isaac Sim terms')
        os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
        args = ['examples/object_understanding/' + ('closed_loop.py' if model == 'pi05' else 'evaluate.py')]
    args += extra
    if dry_run:
        print(json.dumps({'cwd': str(source(model)), 'argv': [str(venv(model) / 'bin/python'), *args],
                          'data': str(WORKSPACE / 'data/lerobot/local'),
                          'simulator_python': str(venv('isaac') / 'bin/python')}, indent=2))
        return
    if action == 'train':
        # Keep a complete terminal loss history for pi0.5 as well as adapters' metrics.jsonl.
        from datetime import datetime, timezone
        folder = source(model) / 'outputs/terminal-logs'
        folder.mkdir(parents=True, exist_ok=True)
        logfile = folder / f'train-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{os.getpid()}.log'
        print(f'Full terminal log: {logfile}', flush=True)
        with logfile.open('w') as handle:
            child = subprocess.Popen([str(venv(model) / 'bin/python'), *args], cwd=source(model),
                                     env=environment(model), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in child.stdout:
                    print(line, end='', flush=True)
                    handle.write(line)
                    handle.flush()
                if child.wait():
                    raise RuntimeError(f'Training failed; see {logfile}')
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait()
    else:
        python(model, args, replace=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('sources', help='Fetch pinned upstream code and apply benchmark adapters')
    p.add_argument('--local-cache', type=Path, help='Optional parent containing robot3/ and robot2/ git checkouts')
    for name in ['env', 'weights', 'setup', 'shell', 'check', 'adopt', 'test']:
        p = commands.add_parser(name)
        choices = [*MODELS, 'isaac', 'all'] if name in {'env', 'setup', 'check'} else list(MODELS) + (['all'] if name in {'weights', 'adopt', 'test'} else ['isaac'])
        p.add_argument('model', choices=choices)
    p = commands.add_parser('data', help='Convert original train/eval splits and compute pi0.5 normalization')
    p.add_argument('--resume', action='store_true')
    for name in ['train', 'eval']:
        p = commands.add_parser(name, add_help=False)
        p.add_argument('model', choices=MODELS)
        p.add_argument('--dry-run', action='store_true')
    commands.add_parser('doctor', help='Show system/environment readiness; no model loading')
    commands.add_parser('transfer', add_help=False, help='Export/import checksum-verified large artifacts')
    args, extra = parser.parse_known_args()
    if extra and args.command not in {'train', 'eval', 'transfer'}:
        parser.error('Unexpected arguments: ' + ' '.join(extra))
    if extra[:1] == ['--']:
        extra = extra[1:]
    if args.command == 'sources':
        sources(args.local_cache)
    elif args.command == 'transfer':
        run([sys.executable, KIT / 'scripts/transfer.py', *extra])
    elif args.command == 'adopt':
        for model in selected(args.model, simulator=False):
            python(model, [KIT / 'scripts/adopt_checkpoint.py', model])
    elif args.command == 'test':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        for model in selected(args.model, simulator=False):
            filename = 'test_conversion.py' if model == 'pi05' else 'test_adapter.py'
            python(model, ['-m', 'pytest', 'examples/object_understanding/' + filename, '-q'])
    elif args.command in {'env', 'setup'}:
        if args.command == 'setup':
            sources()
        for model in selected(args.model):
            install_env(model)
            if args.command == 'setup' and model != 'isaac':
                python(model, [KIT / 'scripts/download_weights.py', model])
    elif args.command == 'weights':
        for model in selected(args.model, simulator=False):
            python(model, [KIT / 'scripts/download_weights.py', model])
    elif args.command == 'data':
        data(['--resume'] if args.resume else [])
    elif args.command in {'train', 'eval'}:
        launch(args.command, args.model, extra, args.dry_run)
    elif args.command == 'shell':
        if not (venv(args.model) / 'bin/python').exists():
            raise RuntimeError('Install the environment first')
        os.chdir(source(args.model))
        os.execve('/bin/bash', ['bash', '--noprofile', '--norc'], environment(args.model))
    elif args.command == 'check':
        for model in selected(args.model):
            python(model, [KIT / 'scripts/check_environment.py', model, '--gpu'])
    elif args.command == 'doctor':
        print(json.dumps({'platform': platform.platform(), 'runtime': str(RUNTIME),
              'disk_free_gib': shutil.disk_usage(KIT).free // 1024**3,
              'environments': {m: (venv(m) / 'bin/python').exists() for m in ENVS}}, indent=2))
        for cmd in [['nvidia-smi'], ['uv', '--version'], ['ffmpeg', '-version']]:
            try:
                run(cmd)
            except (OSError, subprocess.CalledProcessError) as error:
                print(str(error), file=sys.stderr)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))
