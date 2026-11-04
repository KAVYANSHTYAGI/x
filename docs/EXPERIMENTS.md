# Experiments

Every number in the README is reproduced by a command below. Committed outputs are in `results/`. All runs so far
are CPU and synthetic, except E5 (real DSEC, per-window, no learning).

## E1 — Collapse scan
`python scripts/experiments/scan_collapse.py exact` (or `linear`).

* **Setup:** translation data, translation + zoom model.
* **Result:** each objective's loss along the zoom path, relative to the truth (negative = prefers wrong motion).
* **Findings:**
  * CMax goes to −41× at h = 8 and −102× at h = 16; multi-reference CMax to −12× / −36×.
  * EMin, ST-PPP, Zhu and Taming are all improper.
  * MotionPriorCMax's focus term and ours are proper along this path.
  * Our exact and sliced versions agree.

## E2 — Failure suite F1–F4
`python scripts/experiments/run_suite.py --mode exact` → `results/suite/suite_exact.md`.

* **F1:** collapse (as E1), plus optimisation from the truth.
* **F2, reference bias: inconclusive.** All biases are 2–3% (Taming +150% at h* = 0.5). Ours was +7% at h* = 0.1.
* **F3, λ sweep: inconclusive.** Local optimisation from the truth did not trigger collapse; R_div at λ = 100 gives a
  −10% zoom bias. Ours: −6% (unexplained; suspected stochastic jitter). **[rerun deterministically]**
* **F4:** FWL ranks the collapsed field first (361.9 vs 4.03), and RSAT does too; the held-out anchored score ranks
  the truth first.

## E3 — Dense trajectories F6 (in-sample, superseded)
`python scripts/experiments/exp_dense_trajectories.py` → `results/dense/exp_dense_v1_insample.json`,
`results/logs/`.

The first version compared in-sample losses, which is invalid (every fit beats the truth in-sample). Kept for the
record.

## E4 — Dense held-out drift, interior vs border (main synthetic result)
`python scripts/experiments/diag_dense_drift.py --objective {mpcm,mpcm_no_tv,gradmag,cmax_var,ats_exact} --scene {translation,zoom,affine} [--anchor_margin 6]`
→ `results/dense/diag_*.txt`.

* **Setup:**
  1. Free per-tile degree-1 trajectories, initialised at the ground truth.
  2. 150 Adam steps under each objective.
  3. Compare fitted vs ground truth on **held-out events** (fresh DVS realisation), restricted to margins
     ≥ 0 / 10 / 20 / 30 px from the border.
* **Results:** see the README table. Every baseline prefers the drifted field in the interior. Ours is at noise
  level once smoothing and the exposure correction are applied.
* **The fixes this experiment forced** (`docs/THEORY.md` §5b):
  1. In-sample → held-out evaluation.
  2. Smoothed anchors (finite-kernel bias): necessary in theory (1-D test) but not sufficient here.
  3. Exposure correction (frame-entry effect); interior preference fell to +0.04%.
  4. Anchor margin of 6 px (border band).

## E5 — Real DSEC, per-window, no learning
`python scripts/experiments/exp_dsec_modelbased.py --windows 5,20,36` → `results/dsec/`.

* **Setup:**
  * thun_00_a, ground-truth windows 12 / 42 / 74, 150k events each;
  * coarse-to-fine bilinear velocity grid (3×4 → 24×32), 60 Adam steps per level;
  * same for every objective; evaluated on all valid ground-truth pixels.

| Objective | EPE | AE | 3PE |
|---|---|---|---|
| zero flow | 8.99 | 77.5 | 79.6% |
| CMax variance | 52.9 | 101.9 | 98.7% |
| CMax multi-reference | 4.76 | 30.3 | 57.4% |
| MotionPriorCMax's focus (7 fixed references, no KNN / TV) | 3.23 | 17.3 | 30.7% |
| **ours** | **2.30** | **15.1** | **21.3%** |

Ours is best on 2 of 3 windows. Three windows is thin; extend to all 41 windows and add the full MotionPriorCMax loss.

## E6 — Timings
`python bench/bench_ops.py`: see `docs/KERNELS.md`.

## Pending (GPU): see `GPU_RUNBOOK.md`
* DSEC training: baseline reproduction, ours, ablations.
* EVIMO2.
* Metric validity across checkpoints.
