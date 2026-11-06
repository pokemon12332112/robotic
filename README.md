# Object-understanding robot benchmark

A portable workspace for **OpenPI π0.5, SpatialVLA, OpenVLA-OFT, Geometric Action
Model (GAM), and Spatial-Forcing** on `phulocnguyen/object-understanding`.

Start with [fresh server setup](docs/SERVER.md) or [transfer existing training](docs/TRANSFER.md).
This folder is the Git repository to publish. It contains launchers, source
patches, benchmark adapters, exact dependency versions, dataset metadata, and
the 148 recorded evaluation scene definitions. Large artifacts live in ignored
`runtime/`; do not commit model weights, Python environments, datasets, or videos.

## Quick start on a Linux/NVIDIA server

Install Docker Engine, Docker Compose, a compatible NVIDIA driver, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Use Linux x86_64. The Docker base uses Ubuntu 24.04. No host Python/CUDA installation
is needed beyond the driver; each policy and Isaac get their own Python environment.

```bash
git clone https://github.com/pokemon12332112/robotic.git robot-benchmark-kit
cd robot-benchmark-kit
cp .env.example .env
# Edit .env: set ACCEPT_EULA=Y after reading/accepting NVIDIA's Isaac Sim license.
bash docker/run.sh build
bash docker/run.sh setup all
bash docker/run.sh data
bash docker/run.sh check all
bash docker/run.sh test all
```

`setup all` downloads pinned upstream code and pretrained weights and installs
six separate environments. `data` downloads and converts the original training
and evaluation episodes and computes train-only π0.5 normalization. These steps
can take substantial disk space and download time. They do **not** train models.
For a training-only server without RTX rendering, install individual policy
environments and omit `env isaac`; see the server guide.

## Train and evaluate

All commands below run from this folder. Complete training before evaluating.
The five adapters keep the original **584 training episodes / 31,996 frames**
and **148 held-out evaluation scenes**. No test episode is used for training
or normalization.

| Model ID | Train | Closed-loop test | Default total steps | Batch |
| --- | --- | --- | ---: | ---: |
| `pi05` | `bash docker/run.sh train pi05` | `bash docker/run.sh eval pi05` | 3,000 | 1 |
| `spatialvla` | `bash docker/run.sh train spatialvla` | `bash docker/run.sh eval spatialvla` | 3,000 | 1 |
| `oft` | `bash docker/run.sh train oft` | `bash docker/run.sh eval oft` | 10,000 | 1 |
| `gam` | `bash docker/run.sh train gam` | `bash docker/run.sh eval gam` | 3,000 | 1 |
| `sf` | `bash docker/run.sh train sf` | `bash docker/run.sh eval sf` | 10,000 | 1 |

These are the existing starting budgets, **not equivalent training budgets or
validated convergence points**. Change them explicitly when comparing models.
Every model uses current RGB, joint state, and language to predict nine absolute
Panda joint targets. These are custom joint-space adaptations, not exact
reproductions of the papers' native robot/action settings.

```bash
# Change budget and resume (max-steps is the TOTAL target, not additional steps):
bash docker/run.sh train sf --max-steps 3000
bash docker/run.sh train sf --resume --max-steps 6000

# pi0.5 uses upstream argument names and a stable experiment name:
bash docker/run.sh train pi05 --resume --num-train-steps 6000
# Its checkpoints are zero-indexed: after 6000 steps, evaluate checkpoint 5999.
bash docker/run.sh eval pi05 \
  --checkpoint checkpoints/pi05_object_understanding_lora/object_understanding_v1/5999 \
  --output outputs/eval-step-5999

# One held-out scene with video:
bash docker/run.sh eval oft --limit 1 --output outputs/pilot

# Inspect every model's native options:
bash docker/run.sh train sf --help
bash docker/run.sh eval sf --help
```

Relative model arguments such as `--output` and `--checkpoint` resolve from the
model's source directory, **not this repository root**. Use absolute container
paths beginning `/workspace/runtime/...` if needed. π0.5's default evaluation
checkpoint is `object_understanding_v1/2999`; pass another checkpoint explicitly
after changing the training budget or experiment name.

## Use 4–8 GPUs

Each adapter currently uses **one GPU per job**. Select a GPU for one container:

```bash
GPU_ID=2 bash docker/run.sh train sf
```

Run distinct models concurrently, queueing remaining models when slots are full:

```bash
python3 scripts/parallel_runs.py train --gpus 0,1,2,3 --dry-run
python3 scripts/parallel_runs.py train --gpus 0,1,2,3
# After all models have checkpoints, on RTX-capable GPUs:
python3 scripts/parallel_runs.py eval --gpus 0,1,2,3
```

Eight GPUs can be listed as `--gpus 0,1,2,3,4,5,6,7`; only five slots are used for
five model jobs. This launcher does not implement distributed training of one
model across GPUs. Different models use separate output directories and Docker
network namespaces. Scheduler logs are in `runtime/scheduler-logs/`.

Isaac camera rendering requires an RTX-capable GPU; NVIDIA lists A100/H100 GPUs
without RT cores as unsupported for Isaac. A compute-only server can train,
then transfer checkpoints to an RTX machine for testing.
[NVIDIA requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)

## Access an environment

```bash
bash docker/run.sh shell sf
# Inside: current directory is Spatial-Forcing/openvla-SF; python selects its venv.
python --version
python examples/object_understanding/train.py --help
exit
```

Use `pi05`, `spatialvla`, `oft`, `gam`, or `isaac` similarly. Outside Docker,
`./bench shell MODEL` provides the same interface after native installation.
Do not move/reuse `.venv` directories between machines or between host and
container paths; reinstall from the locks at the final execution path.

## Where results go

| ID | Model directory under `runtime/robot3/` |
| --- | --- |
| `pi05` | `openpi/` |
| `spatialvla` | `SpatialVLA/` |
| `oft` | `openvla-oft/` |
| `gam` | `Geometric-Action-Model/` |
| `sf` | `Spatial-Forcing/openvla-SF/` |

For the four non-π0.5 adapters, training results are in
`MODEL/outputs/object-understanding/train/`: `latest.pt`, `metrics.jsonl`,
`configuration.json`, `normalization.json`, and `status.json`. OFT/SF also retain
numbered checkpoints every 250 steps. Each completed optimizer step logs its
loss; SF logs action and spatial alignment components, and GAM logs action and
future-feature components. Resumes append; interrupted unsaved steps can repeat.

π0.5 saves Orbax checkpoints under
`openpi/checkpoints/pi05_object_understanding_lora/object_understanding_v1/`.
The launcher logs its loss every step to `openpi/outputs/terminal-logs/` and saves
every 1,000 steps plus the final step. All models retain terminal logs.

Evaluation writes `metrics.json`, status/configuration, policy/simulator logs,
per-episode trajectories and `episode_XXXX/rollout.mp4`. Non-π0.5 default directory:
`MODEL/outputs/object-understanding/evaluation/`; π0.5:
`openpi/outputs/object-understanding-closed-loop/evaluation/`.
Use a new `--output` when changing weights/settings or moving machines.

## Benchmark and reproducibility

See [method details](docs/METHODS.md), [reproducibility limits and checks](docs/VALIDATION.md),
and [checkpoint transfer](docs/TRANSFER.md). The runtime retains a small `robot2/`
simulator sibling folder because the audited evaluators expect that layout; it
does not depend on any files outside this kit's runtime.

The source metadata's `accepted_demonstration=false` flag is retained. This
packaging does not independently certify the demonstrations. Recorded scene
files contain original public dataset collection paths as provenance; those
paths are not opened at runtime.

## Publish this folder

```bash
git init -b main
git add .
git status --short
git commit -m "Add reproducible object-understanding benchmark runners"
git remote add origin https://github.com/pokemon12332112/robotic.git
git push -u origin main
```

Review the staged list before pushing. `runtime/`, `.env`, datasets, weights,
videos, and artifact bundles are ignored. There is no requirement for Git LFS
in this launcher repository. Upstream repositories retain their own licenses;
see [third-party notices](THIRD_PARTY.md).
