# Validation record

Validated on 2026-11-06 on the existing Linux x86_64 / RTX 4090 workstation.
The clean source tree was reconstructed at a different path, and environments
were **installed anew inside Docker** rather than copying the original virtualenvs.
Existing package downloads were reused through uv's verified package cache.

| Check | Result |
| --- | --- |
| Docker image build, pinned Ubuntu/uv image digests | Passed |
| Container GPU access and host-matching user | Passed |
| Ten source checkouts, pinned commits and patches | Passed |
| Re-running source setup without overwriting changes | Passed |
| Fresh π0.5, GAM, SpatialVLA, OFT, SF, Isaac environments | All six installed; imports passed |
| GPU arithmetic in all six environments, including JAX | Passed; no model weights loaded |
| Portable launcher/transfer/provenance tests | 12 passed |
| π0.5 conversion/action tests in Docker | 3 passed |
| GAM adapter tests in Docker | 5 passed |
| SpatialVLA adapter tests in Docker | 7 passed |
| OpenVLA-OFT adapter tests in Docker | 8 passed |
| Spatial-Forcing adapter tests in Docker | 10 passed |
| Held-out scene manifest | All 148 original entries and SHA-256 hashes verified |
| Real existing GAM checkpoint relocation | Passed at step 3000; all 142 adapter tensors and all optimizer/RNG/non-provenance fields unchanged |
| π0.5 initialization | All 29 cached files match pinned GCS CRC32C checksums; a generation-specific download and reuse check passed |
| Custom OFT/SF attention implementation | Installed source hashes match original environments |
| Git-sized payload | Approximately 4.2 MiB; no weights, datasets, environments, credentials or videos included |

The adapter tests used the original converted data and pretrained assets mounted
read-only. They exercised the relocated code and freshly installed dependencies.
The real GAM checkpoint migration used a copy; original assets/checkpoints were
mounted read-only. Small transfer tests separately covered corruption, symlink
escapes, collisions, hardlinks, nanosecond timestamps and the export allowlist.

## What this does not claim

No new full training run or full benchmark was launched by packaging. The
existing Spatial-Forcing evaluation was already using the GPU. A new Isaac
rendering rollout inside this Docker image was **not** performed, and no 4–8 GPU
server was available for execution testing. The multi-GPU scheduler was checked
in dry-run mode; it launches the separately verified single-model commands.

All five original adapters already had workstation training/closed-loop smoke
checks before packaging. Those results are distinct from fresh container
renderer validation. Run the one-scene test in SERVER.md on the target server
before a full evaluation. Driver, GPU architecture, rendering assets and numeric
kernels can change behavior even with pinned source and Python dependencies.

The simulator lock intentionally reproduces its six observed package-metadata
conflicts; GAM retains the known headless OpenCV metadata mismatch. They are
explained in SERVER.md. Successful imports/GPU arithmetic are not proof that
all optional upstream features work.

## Repeat the checks

```bash
# No network, model downloads, or GPU needed:
python3 -m unittest discover -s tests -v
bash -n bench docker/run.sh
docker compose config --quiet

# After setup and data/weights preparation:
bash docker/run.sh check all
bash docker/run.sh test all
```

Machine-readable evidence is in `validation.json` at the repository root.
