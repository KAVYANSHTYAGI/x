# Attack plan: beating MotionPriorCMax with a proper alignment objective

## 0. The story in one paragraph

Self-supervised event motion estimation runs on one idea: contrast maximisation (CMax). CMax warps events to a
reference time and maximises the sharpness of the resulting image.

The state of the art for dense, continuous-time motion is **MotionPriorCMax** (Hamann et al., ECCV'24). It needs
four patches to work, and its own ablations show that removing any of them costs 1.3–4 px of EPE on DSEC:
* a random reference time per step;
* averaging each trajectory over its 32 nearest neighbours;
* a Charbonnier TV regulariser with a tuned weight;
* masking of events warped out of the frame.

**Our claim:** these patches exist because sharpness-of-warped-events objectives are *improper*: their optimum is not
the true motion. We show this across the whole family (CMax, multi-reference, gradient magnitude, Shiba's
regularisers, EMin, ST-PPP, Zhu'19 and Taming timestamp losses, and the full MotionPriorCMax loss) on a common
failure-mode suite.

**Our objective:** the *anchored transport score* scores raw events, not warped ones. It has:
* smoothed and dequantised anchors (proper at every kernel width);
* an exposure correction for field-of-view entry and exit.

It needs no reference time and no regulariser weight to be proper.

## 1. Failure-mode suite (synthetic, exact ground truth) — `atl/suite`, `scripts/*`

| Task | What it tests | Baselines that fail | Ours |
|---|---|---|---|
| F1 collapse (translation data, translation + zoom model) | optimum on a contraction path | CMax var (−41× / −102× at h = 8 / 32), multi-ref (−12× / −36×), Shiba R_D / R_A at λ=1, EMin, ST-PPP, Zhu TS, Taming TS | proper along the whole path |
| F2 reference bias (true zoom) | first-order bias of single-reference CMax, ∝ 1/T | CMax t_ref=0 (bias ∝ 1/T, matches Thm 4) | unbiased |
| F3 λ dilemma | one regulariser weight for all data | λ_min varies 4.6×; λ safe for long windows erases true divergence (−65..−100%) in short ones | no λ |
| F4 metric fooling | unsupervised metrics rank a collapsed field | FWL, RSAT | held-out score ranks the truth first |
| **F6 dense trajectories (the target's representation)** | **held-out bias of the fitted field, interior vs border** | **MPCM full: interior preference +0.6% → +1.2%, growing away from the border (bias, not boundary)** | **no interior preference (≤ noise)** |

F6 details (translation scene, 96×128, 50k events, 150 Adam steps from ground truth; held-out = new DVS realisation):

| Held-out events | MPCM (official loss) | MPCM no TV | Ours: plain | + smoothed / dequantised | + exposure correction |
|---|---|---|---|---|---|
| all | +0.93% | +1.00% | +0.30% | — | +0.17% |
| interior ≥10 px | +0.64% | +0.37% | −0.11% | — | +0.04% |
| interior ≥20 px | +0.85% | +0.56% | ≈0 | — | +0.04% |
| interior ≥30 px | +1.16% | +0.72% | ≈0 | — | +0.05% (noise ≈ 0.08%) |
| drift EPE interior / border | 2.7 / 3.0 px | 4.5 / 5.9 px | 1.0 / 2.9 px | | 1.0 / 2.9 px (0.9 / 2.4 with a 6 px anchor margin) |

What went wrong along the way, kept on record:
1. The first F6 version compared losses *in-sample*. That is invalid, since every fitted estimator beats the truth
   in-sample. It was replaced by held-out evaluation.
2. The plain anchored score has a finite-kernel "deconvolution" bias. It is fixed exactly by smoothing anchors with
   the same kernel (proper at every h). The 1-D test is in tests.
3. The remaining drift was a field-of-view-entry effect. Exposure correction removes the interior part; the border
   band is handled by a raw-position anchor margin (motion-independent, so it stays proper).
4. The target's own L1 gradient-magnitude focus does **not** collapse on parametric zoom paths. Its failure is the
   dense-field bias above, not collapse.

## 2. Real data without learning — `scripts/experiments/exp_dsec_modelbased.py`

Per-window coarse-to-fine dense flow on DSEC ground-truth windows (thun_00_a), same model and optimiser for each
objective, scored against DSEC ground truth on valid pixels. Results: `results/dsec/dsec_modelbased.log`.

Setup: 3 windows (12, 42, 74), 150k events each; grids 3×4 → 24×32; 60 Adam steps per level. Totals are
pixel-weighted:

| Objective | EPE | AE | 3PE | per-window EPE |
|---|---|---|---|---|
| zero flow | 8.99 | 77.5 | 79.6% | 10.75 / 8.56 / 7.99 |
| CMax variance (t_ref=0) | 52.9 (collapses) | 101.9 | 98.7% | 64.0 / 82.1 / 22.0 |
| CMax multi-reference | 4.76 | 30.3 | 57.4% | 4.97 / 5.88 / 3.74 |
| gradient-magnitude focus (MPCM's term, 7 fixed refs) | 3.23 | 17.3 | 30.7% | **2.15** / 6.15 / 1.76 |
| **anchored score (ours)** | **2.30** | **15.1** | **21.3%** | 2.32 / **3.41** / **1.42** |

Caveat: this is one sequence and three windows. The focus row is MPCM's sharpness term without its KNN
look-up table and TV, so it is not the full loss.

**Suite F1–F4** (`results/suite/suite_exact.md`):
* **F1** reproduces the collapse table.
* **F4:** FWL ranks the collapsed field first (361.9 vs truth 4.03) and RSAT does too (0.049 vs 0.712). The held-out
  anchored score ranks the truth first.
* **F2 and F3 did not separate the methods cleanly in this configuration.**
  * All biases are 2–3%, except Taming (+150% at h* = 0.5).
  * Our sliced score shows +7% at h* = 0.1 and **−6% in F3's zoom optimisation**.
  * Suspected cause: the stochastic anchor jitter was not fixed between evaluations and steps in the suite, plus
    field-of-view exit under expansion.
  * **[todo]** rerun F2/F3 with deterministic seeds and the exact kernel before claiming anything there.

## 3. The decisive experiment (GPU)

Everything identical except the loss:
* data: 24 DSEC training sequences, windows at every second image timestamp, 100 ms;
* network: U-Net with 15 voxel bins, degree-1 polynomial trajectories on 4×4 tiles;
* optimisation: Adam 1e-4, total batch 14, 100 epochs.

**Step 0 — reproduce the baseline through our pipeline.**
```
python scripts/tools/convert_mpcm_checkpoint.py <official DSEC ckpt> runs/mpcm_official/best.pt
python scripts/tools/dsec_submit.py --ckpt runs/mpcm_official/best.pt ...      # must reproduce EPE 3.20 on the server
```
**Step 1 — retrain the baseline with our implementation of their loss** (`configs/dsec_mpcm.yaml`).
Gate: within ~0.1 EPE of 3.20 on the validation sequences.

**Step 2 — ours** (`configs/dsec_ats.yaml`): anchored exact score, h annealed 4 → 1 px over the first 30%,
60k anchors per sample, 6 px anchor margin, exposure correction.

**Step 3 — ablations: remove the target's patches under our loss** (each should be unnecessary).
* TV weight 0;
* one fixed "reference" (not applicable: there is none);
* no KNN smoothing (ours uses no look-up table).

Then ablate our own parts:
* no anchor smoothing;
* no dequantisation;
* no exposure correction;
* bandwidth annealing vs fixed h;
* exact vs sliced kernel.

**Step 4 — EVIMO2 continuous time (300 ms, TEPE).** Uses their RAFT-spline / BFlow backbone with Bezier trajectories,
wrapped with our loss (`atl/data/evimo2.py` loader + `evaluate_tracks`). Target: TEPE 6.14.

**Secondary target:** Taming CMax (DSEC EPE 2.33).

**Reporting:**
* DSEC test-server numbers (EPE / AE / 1PE / 3PE);
* per-sequence breakdown;
* forward-motion sequences separately: theory predicts the largest gain where motion has divergence.

## 4. Risks and what would change the plan

* **Basin of attraction.** In per-window optimisation from zero, MPCM's KNN-32 + TV converges faster (4.8 px vs
  8.7–9.2 px after 150 steps). In learning, the network shares strength across the dataset, but if training is slow:
  * use wider annealing (h 8 → 1);
  * use a short-lag curriculum (τ small → 1);
  * or warm-start from an MPCM-trained network and fine-tune with our loss. Fine-tuning still demonstrates that the
    proper loss removes the bias.
* **Compute.** The exact pair kernel at 1M events per sample and τ = 1 needs about N·S plan entries (~16M), roughly
  0.6 GB per sample. If it is too heavy: subsample sources to about 300k (unbiased for the KDE), use τ = 0.5, or
  switch to the sliced kernel.
* **Real-sensor effects** (flicker, noise bursts, refractory period) affect every brightness-constancy method. The
  ε floor acts as an outlier class.
* **Venue.**
  * ICCV (deadline ≈ March 2027) fits the vision results.
  * ICML (≈ late Jan 2027) needs the theory up front: proper scoring for self-supervised alignment, the
    improperness theorems, and the finite-kernel fix.
  * Decide after Step 2's validation numbers.
