# Server setup

## Supported target

Linux **x86_64**, Docker Engine with Compose, NVIDIA Container Toolkit, and an
NVIDIA driver compatible with the selected GPU and CUDA stacks. The original
policy smoke checks ran on RTX 4090 / 24 GiB. The tested locks include CUDA 12.1,
12.4, 12.6 and CUDA 13.0 for the simulator. An old host driver can therefore
work for a policy while failing Isaac. Consult the version-specific NVIDIA
requirements instead of downgrading the locked simulator to make it install.

Plan disk for code, six environments, pretrained weights, converted data and
saved checkpoints. **Several hundred GB free** is a practical planning allowance
for all five models; it is not a measured hard minimum. OFT/SF alone can retain
roughly 18/24 GiB of optimizer checkpoints at their default 10k budgets. More
host RAM is needed when running several large-model processes simultaneously.

The exact server GPU model is not yet specified. These locks preserve older
PyTorch 2.2/2.5 and FlashAttention builds; new architectures such as Blackwell
may need different builds and a fresh compatibility audit. Do not silently
upgrade a single package and label the result an exact reproduction.

## Docker workflow

1. Install Docker using the [official Linux instructions](https://docs.docker.com/engine/install/ubuntu/).
2. Install/configure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
3. Clone the kit, copy `.env.example` to `.env`, and set your GPU index. Accept
   the NVIDIA license explicitly with `ACCEPT_EULA=Y` before installing Isaac.
4. Run the commands below. Installation and dataset conversion are sequential;
   do them once before scheduling GPU experiments.

```bash
bash docker/run.sh build
bash docker/run.sh doctor
bash docker/run.sh sources
bash docker/run.sh env all
bash docker/run.sh weights all
bash docker/run.sh data
bash docker/run.sh check all
bash docker/run.sh test all
```

`setup all` combines sources, environments and weights. `data --resume` continues
an interrupted conversion. Completed conversions are not overwritten. Models
never silently substitute a different dataset split or pretrained revision.

If you have already exported the original dataset and checkpoints, install
sources/environments, then import that bundle **instead of regenerating data**;
see TRANSFER.md. Checkpoint resume validates the original Parquet bytes.

Each container mounts this repository at `/workspace`, keeps environments and
caches under `/workspace/runtime/`, uses a host-matching UID/GID, and exposes
one selected GPU. `shell MODEL` starts Bash with the selected Python and model
directory. No host networking or published policy ports are required: simulator
and policy server run as separate processes inside the same container.

The Docker image provides system libraries and uv; Python environments live on
the persistent mount. Rebuilding the base does not erase trained checkpoints.
The container user's home is persisted in `runtime/container-home/` so Isaac's
home-directory caches survive container restarts as well.
After a verified installation, retain the Docker image by digest or `docker save`
if you need to reproduce the exact system-package build. Apt mirrors may update
even though the Ubuntu and uv image digests and Python/source versions are pinned.

For a compute-only server, install policy environments with `env MODEL`, including
`env pi05` for conversion/auditing, and skip `env isaac`. The existing evaluators
launch Isaac locally; remotely serving a policy from a second machine is not
implemented by this kit. Transfer the checkpoint bundle to the RTX machine and
run both policy inference and Isaac there.

## First GPU smoke checks

Use separate output directories so short tests cannot be confused with training:

```bash
bash docker/run.sh train sf --max-steps 1 --save-every 1 \
  --check-alignment-gradients --output outputs/smoke/train
bash docker/run.sh train sf --resume --max-steps 2 --save-every 1 \
  --output outputs/smoke/train
bash docker/run.sh eval sf --limit 1 \
  --checkpoint outputs/smoke/train/latest.pt --output outputs/smoke/evaluation
```

For `gam`, `spatialvla`, and `oft`, use the same commands but omit
`--check-alignment-gradients`. For π0.5, use its audited one-step diagnostic:

```bash
bash docker/run.sh shell pi05
python examples/object_understanding/validate_training.py --gpu-step
```

That π0.5 diagnostic discards its update. Train a separate short experiment if
you want an evaluation checkpoint. A two-step policy is not expected to establish
task performance. Inspect the metrics and video before launching all 148 scenes.

## Native Linux alternative

Install the system packages listed in `docker/Dockerfile` using your administrator's
normal process. Install uv **0.11.29**, then use `./bench` in place of
`bash docker/run.sh` for every command. Export `ACCEPT_EULA=Y` in your shell;
native commands do not read Compose's `.env`. Use `CUDA_VISIBLE_DEVICES=2` to
select a single native GPU.

`BENCH_RUNTIME=/absolute/path` selects a separate persistent runtime. Set it
consistently before every native command. Paths in generated environment locks
are rendered relative to that root; no workstation username is required.

## Known installation metadata conflicts

If an older checkout fails while installing GAM's `deepspeed==0.18.8` with
`MissingCUDAException: CUDA_HOME does not exist`, run `git pull --ff-only`, then
repeat `bash docker/run.sh env all`. Existing environments and downloaded
packages are reused; rebuilding the Docker image is unnecessary.

DeepSpeed's setup probes the CUDA compiler even when `DS_BUILD_OPS=0` if it sees
a GPU. The installer now hides CUDA and selects CPU only for that package's
build subprocess. GAM training and checks retain normal GPU visibility.
This benchmark adapter uses PyTorch directly, not DeepSpeed. Optional upstream
DeepSpeed GPU/distributed workflows (including a standalone GPU-side DeepSpeed
import) can still require a matching CUDA compiler; they are outside this kit's
supported single-GPU training path.

Locks reproduce the installed working environments, including upstream metadata
conflicts. Installations use explicit complete package lists with `--no-deps`;
they do not ask pip to resolve a different environment. GAM's Robosuite declares
`opencv-python`, while the working headless installation supplies
`opencv-python-headless`. Isaac's observed installation reports six conflicts
in NumPy/Numba/cmeel, fsspec, urllib3 and optional onnxruntime-gpu requirements.
Isaac Lab itself overrides several Isaac Sim constraints in its pinned source.
These known conflicts are recorded, not described as a clean dependency check.

Inspect `runtime/installation/*-pip-check.txt`. `check MODEL` exercises imports
and GPU arithmetic, but a real one-scene rollout is required to validate the
renderer and simulator. Do not replace pinned Isaac Lab source with a different
release; its scene/reset API is coupled to the evaluator.

## Reference documentation

- [NVIDIA Isaac requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html): RTX-capable GPU and compatible driver; A100/H100 lack supported RT cores.
- [Isaac Python installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_python.html): Python 3.12 and GLIBC 2.35+ for the pinned installation family.
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html): host GPU access from Docker.

First-time Isaac rendering can download robot/USD assets from NVIDIA. Allow
the asset endpoints documented by NVIDIA, or prepare a separately audited local
asset cache. Source and Python pins alone do not provide a fully offline simulator.
