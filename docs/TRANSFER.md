# Transfer existing training without putting weights in Git

Stop the training jobs being exported, or wait for a committed checkpoint and
ensure no writer is changing the selected training directories. Export copies
only an explicit allowlist: converted train/eval data, normalization, original
initialization weights, T5 cache, and each model's training checkpoints/history.
It excludes Python environments, tokens, arbitrary caches, and evaluation videos.
Missing optional run paths are reported rather than invented.

## On the current workstation

Run this from `robot-benchmark-kit/`. The source workspace is its parent here;
the bundle destination must be outside the original workspace:

```bash
./bench transfer export --workspace .. --destination /PATH/robot-artifacts --dry-run
./bench transfer export --workspace .. --destination /PATH/robot-artifacts
# Or export just one model plus shared data:
./bench transfer export --workspace .. --destination /PATH/sf-artifacts --models sf
```

Use an empty destination. The exporter preserves nanosecond timestamps and
hardlinks, dereferences HF weight symlinks into self-contained files, and writes
a SHA-256 manifest. It verifies that source file size/time/inode did not change
during copying. It does not delete or rewrite your existing workspace.

```bash
rsync -aH --info=progress2 /PATH/robot-artifacts/ USER@SERVER:/PATH/robot-benchmark-kit/artifacts/import/
```

Keep the bundle outside Git. The `artifacts/` folder is ignored. Exporting all
weights can require tens of GB; the dry-run lists exactly which paths will be
included. Hashing and transfer take time. Keep an extra copy of the original
checkpoints until server validation succeeds.

## On the destination server

```bash
cd /PATH/robot-benchmark-kit
cp .env.example .env
# Set ACCEPT_EULA=Y after reading the NVIDIA license.
bash docker/run.sh build
bash docker/run.sh sources
bash docker/run.sh env all
bash docker/run.sh transfer import /workspace/artifacts/import
bash docker/run.sh adopt all
bash docker/run.sh check all
```

Do not run `data` to regenerate an imported dataset. Resume checks intentionally
require its original Parquet content hash. Import verifies **every** checksum
before writing, refuses different existing files, restores exact timestamps,
and records `runtime/installation/import-manifest.json`.

`adopt MODEL` runs using that model's actual Python/source environment. It
re-hashes the imported pretrained assets, checks the checkpoint's original
training-code identity, and updates only relocation metadata (`pretrained`,
and GAM's `base`). GAM includes absolute paths in these fields. No model tensor,
optimizer, RNG, training step, normalization, or training-code hash is changed.
Original files are retained as `*.pt.before-relocation`; an audit trail is saved
as `relocation.json`. Run adoption in Docker if you will train in Docker; host
and container paths differ. Never bypass an implementation mismatch by editing
`code_identity`. Only load trusted checkpoints from your own exported workspace.

π0.5 Orbax directories are copied in full, including optimizer/train-state and
normalization assets; there is no pickle metadata relocation step for them.
Evaluation should always use a **new result directory** after migration: saved
evaluation fingerprints include paths/runtime information. Existing evaluation
videos/results can be archived separately, but should not be mixed with a new run.

```bash
# Continue your imported run to a larger TOTAL step target:
bash docker/run.sh train sf --resume --max-steps 13000
bash docker/run.sh train pi05 --resume --num-train-steps 9000

# Test imported weights first, with one scene and a new output:
bash docker/run.sh eval sf --limit 1 --output outputs/server-validation
bash docker/run.sh eval pi05 \
  --checkpoint checkpoints/pi05_object_understanding_lora/object_understanding_v1/5999 \
  --limit 1 --output outputs/server-validation
```

Select the actual saved π0.5 step, and keep the original batch size/rank/learning
rate/seed/alignment weight when resuming other models. Fresh experiments may
choose new settings and output directories. If you exported only a subset of
models, use `adopt sf` etc. instead of `adopt all`.

Moving a native runtime after its environments are installed also requires
recreating those environments. Do not `rsync .venv`; virtualenv launchers and
editable installs embed absolute paths.
