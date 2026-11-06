# What is reproduced

Source revisions are pinned in `locks/sources.json`; the unmodified upstream
repositories are fetched on setup, then `patches/` and `overlay/` are applied.
The overlay preserves the existing training implementation bytes so saved
checkpoint code identities remain meaningful. Environment locks preserve the
installed package versions and VCS revisions; dependency isolation matters.

| Model | Adaptation |
| --- | --- |
| π0.5 | JAX LoRA, 16-step chunk, arm deltas during training plus absolute finger targets; inference returns absolute nine-joint targets. |
| SpatialVLA | Pretrained visual-language/depth encoding with language LoRA and continuous nine-joint head; not its native discrete spatial action-token objective. |
| OpenVLA-OFT | Official bidirectional Llama fork, rank-8 Q/V LoRA, two frozen visual streams, proprio token, 72 parallel action placeholders and narrowed residual L1 head. |
| GAM | Frozen geometric backbone and T5, predictor LoRA, joint-space head, future geometric-feature supervision. Future images are targets during training, never inference observations. |
| Spatial-Forcing | OFT joint adapter plus official frozen VGGT teacher and layer-24 visual-token alignment: action L1 + 0.5 × cosine alignment loss. Teacher sees only current views and is not loaded during inference. |

The source dataset is pinned to revision
`e3dbcd59b906768df0a2e7baa8b3124c6b0aaec1`. Audited public metadata and held-out
scene definitions are included, but the images/actions themselves are downloaded
or transferred. Every adapter checks the original split and training-only
normalization. Its input is the current top/wrist RGB, current nine joints, and
prompt. Object identities, evaluator object poses and target indices do not enter
the policy. Episode-end action padding is masked where the adapter implements it;
π0.5 uses its audited upstream chunk/padding conventions.

Closed-loop testing recreates the same 148 held-out scenes in Isaac. The policy
predicts a chunk, the simulator executes four actions, and the policy replans
from fresh observations. Evaluation ends at task success or eight simulated
seconds (240 control steps at 30 Hz, physics at 120 Hz). The same camera geometry,
timing, reset states and benchmark commit are retained.

Success uses the benchmark's saved settings and checker: target lift at least
5 cm, target within 16 cm of the hand, summed finger positions below 7.5 cm,
all distractor centers within 4 cm of initial positions, sustained for 0.5 s.
The finger condition is a position proxy, not force/contact sensing. See the
pinned `spatial-understanding/tasks/common/success.py` for exact inequalities.

Changing server, GPU, CUDA libraries, renderer, or floating-point kernels can
change numerical results and simulated trajectories. Reproducible setup means
pinned code/data/settings with auditable runs; it does not promise bit-identical
GPU training or identical success rates on every machine.
