# Anchored Transport Likelihood (`atl`)

**Proper self-supervised alignment for event cameras.** Every self-supervised event-motion loss in use today (contrast
maximisation and its descendants) scores events *after warping them into a reference frame*. We show that this makes
the loss **improper**: its optimum is not the true motion. That is why each method needs patches (random reference
times, nearest-neighbour averaging, regulariser weights, or ground-truth "anchoring").

We replace it with the **anchored transport score**: each *raw* event is scored under the density predicted by all
other events transported to its own timestamp. The evaluation point cannot move, so the deformation lever disappears.
With smoothed, dequantised anchors and an exposure correction, the score is proper at every kernel width, and it needs
no reference time and no regulariser weight.

**Target paper:** MotionPriorCMax (Hamann et al., ECCV 2024), the self-supervised state of the art for dense,
continuous-time event motion.

| Benchmark | Metric | MotionPriorCMax |
|---|---|---|
| DSEC-Flow test | EPE / AE / %Out | 3.20 / 8.53 / 15.21 |
| EVIMO2 300 ms | TEPE / TAE / %Out | 6.14 / 16.98 / 0.254 |
| Secondary: Taming CMax (DSEC) | EPE | 2.33 |
| Secondary: MultiCM (DSEC, model-based) | EPE | 3.47 |

> **Status (Nov 2026).** Theory, code, a failure-mode suite and a small real-data comparison are done. **The network
> training that decides the benchmark comparison has not been run** (no GPU in the development container). See
> [`docs/GPU_RUNBOOK.md`](docs/GPU_RUNBOOK.md).

## Evidence so far

All synthetic numbers use exact ground truth. Full tables and commands: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

**Dense continuous-time trajectories (the target's own representation).** Fitted from the ground truth, then compared
on *held-out* events in the scene interior (≥20 px from the border). Values are the preference for the drifted field
over the truth; positive means the loss prefers the wrong field.

| Objective | Translation | Zoom | Affine |
|---|---|---|---|
| CMax variance + TV | +43% | — | — |
| Gradient-magnitude focus + TV | +18% | — | — |
| MotionPriorCMax (official loss) | +0.85% | +2.2% | +1.05% |
| **Anchored score (ours)** | **+0.04% (noise)** | **+0.09%** | **+0.05%** |
| Interior drift, MPCM → ours | 2.9 → 1.2 px | 7.6 → 3.2 px | 5.7 → 2.9 px |

**Collapse** (translation data, translation + zoom model):
* CMax prefers zoom 41–102× over the truth;
* multi-reference CMax 12–36×;
* Shiba-regularised CMax, EMin, ST-PPP, Zhu and Taming timestamp losses also prefer wrong motion;
* ours: proper along the whole path.

**Unsupervised metrics.** FWL rates a collapsed field 90× better than the truth; RSAT also prefers it. Our held-out
score ranks the truth first.

**Real DSEC** (thun_00_a, 3 ground-truth windows, per-window fitting, no learning):

| Objective | EPE | AE | 3PE |
|---|---|---|---|
| Ours | **2.30** | **15.1** | **21.3%** |
| MPCM's gradient focus | 3.23 | 17.3 | 30.7% |
| Multi-reference CMax | 4.76 | 30.3 | 57.4% |
| CMax variance (collapses) | 52.9 | 101.9 | 98.7% |

**Known weaknesses** are listed openly in [`docs/ATTACK_SURFACE.md`](docs/ATTACK_SURFACE.md). The main ones:
* the upside is bounded: only ~56% of DSEC ground-truth pixels have events;
* per-window convergence from zero is slower than MPCM's;
* a border band drifts more than the interior.

## Install

```bash
git clone <this repo> && cd anchored-transport-likelihood
pip install -e ".[dev]"          # torch >= 2.1; the C++/CUDA extension JIT-compiles on first use
pytest -q                        # 20 tests: kernels vs references (values + gradients), objectives, I/O
```
Optional: `pip install pykeops` (faster KNN for the MotionPriorCMax baseline on GPU).

## Quick start

```bash
python scripts/experiments/scan_collapse.py exact          # one-screen collapse comparison of all losses
python scripts/experiments/run_suite.py --mode exact       # failure suite F1-F4 -> outputs/suite_exact.md
python scripts/experiments/diag_dense_drift.py --objective mpcm --scene zoom    # dense held-out bias test
python scripts/train.py --config configs/dsec_mpcm.yaml    # baseline: MotionPriorCMax loss
python scripts/train.py --config configs/dsec_ats.yaml     # ours: only the loss differs
python bench/bench_ops.py --device cuda --events 1000000   # kernel / loss timings
```

## Repository map

```
atl/                 the Python package (see atl/README.md)
  core/              EventPacket: packed, time-sorted event batches
  ops/               differentiable ops; C++/CUDA fast paths + pure-torch references
  motion/            parametric / dense / continuous-time pixel-trajectory motion models
  objectives/        baselines (CMax family, MotionPriorCMax) and ours (anchored scores); metrics
  data/              synthetic DVS generator, DSEC, EVIMO2
  models/            U-Net (matched to MotionPriorCMax) and the EVIMO2 RAFT-spline adapter
  estimators/        model-based (per-window) optimisation
  train/             self-supervised trainer with a pluggable loss
  eval/              EPE / AE / NPE and trajectory metrics
  suite/             failure-mode suite F1-F4
csrc/                C++ (OpenMP) and CUDA kernels (see csrc/README.md, docs/KERNELS.md)
configs/             training configs (baseline vs ours)
scripts/             train.py, tools/ (checkpoint conversion, DSEC submission), experiments/
tests/               unit tests
bench/               timing harness
results/             committed evidence (tables, logs) behind every number above
docs/                theory, architecture, file-by-file code guide, kernels, experiments, plans
```

## Documentation

| Document | Contents |
|---|---|
| [`docs/THEORY.md`](docs/THEORY.md) | Definitions, theorems, proofs and corrections |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Layers, data flow, conventions, how to extend |
| [`docs/CODE_GUIDE.md`](docs/CODE_GUIDE.md) | Every file explained |
| [`docs/KERNELS.md`](docs/KERNELS.md) | C++/CUDA design, layouts, complexity, build and verification |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Every experiment: command, result, interpretation |
| [`docs/GPU_RUNBOOK.md`](docs/GPU_RUNBOOK.md) | Step-by-step for the decisive GPU runs |
| [`docs/ATTACK_PLAN.md`](docs/ATTACK_PLAN.md) | How we beat the target |
| [`docs/ATTACK_SURFACE.md`](docs/ATTACK_SURFACE.md) | Weaknesses, reviewer attacks, fixes |
| [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) | How we got here, including the ideas that failed |

Third-party code: see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
