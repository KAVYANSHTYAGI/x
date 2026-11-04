# The issue, our solution, its weaknesses, attack surface, and every fix we can think of

Status labels: **[measured]** = we have a number; **[theory]** = argued or proved on paper; **[open]** = unknown.

---

## 0. TL;DR

* **The issue is real and measurable.** Every sharpness-of-warped-events loss, the target's included, prefers wrong motion
  fields on held-out data [measured, synthetic]. The patches the field uses cannot fix that in principle [theory].
* **Our fix works where it acts.** Interior bias is at noise level and drift is 2–2.4× smaller [measured, synthetic].
  On 3 real DSEC windows it gives the lowest EPE among the objectives tested [measured].
* **The biggest unknown is not the theory; it is the *upside*.**
  * On DSEC only ~56% of ground-truth pixels have any event in the window [measured, thun_00_a window 10]. The rest
    are filled by the network's spatial prior, where no alignment loss acts directly.
  * If most of MotionPriorCMax's 3.20 EPE lives in event-free pixels, a proper loss helps less than hoped.
  * **This must be measured first** (Exp 1 below).
* **The second unknown is optimisation.** Our score converged more slowly from zero than the target's KNN+TV loss in
  per-window fitting [measured]. There are principled fixes (§5).
* **The main reviewer attack will be "known tools"**: proper scoring rules, leave-one-out likelihood, spread
  divergence, anchored GMM registration. The defence is the *impossibility results* plus a general *principle*
  instantiated by several divergences (§5A), not one loss.

---

## 1. The issue, stated precisely

**Objects.**
* Data: events (x_i, t_i).
* Model: a transport Φ_θ (motion).
* Self-supervised objective: F(θ, data).
* F is **proper** if its population optimum is the true θ.

**Reference-frame objectives.** CMax, multi-reference CMax, gradient magnitude, EMin, ST-PPP, the timestamp losses
and MotionPriorCMax all have the form F = Ψ(events moved to t_ref). They score in a frame that θ itself deforms.
That deformation is a free lever which cannot be told apart from alignment:
* for superlinear Ψ (variance, entropy, Poisson) the lever is **contraction**;
* for L1-gradient Ψ the lever is **scatter / decorrelation**.

**Consequences, each observed.**
1. Collapse on parametric models with divergence. CMax prefers zoom 41–102× over the truth [measured].
2. Local bias of dense fields on held-out data, interior ≥20 px:
   * MotionPriorCMax: +0.85% (translation), +2.2% (zoom), +1.05% (affine);
   * gradient focus: +18%;
   * CMax variance: +43% [measured].
3. Unsupervised metrics are fooled. FWL rates the collapsed field 90× better than the truth; RSAT also prefers it
   [measured].
4. Patch dependence. MotionPriorCMax's own ablations: 1 reference time → EPE 7.26, 3 references → 4.46, KNN = 1 → 4.58,
   full method 3.20 [paper]. STSC (CVPR'26) needs ground truth "to anchor the scale" [paper].

**What the issue is NOT.** It is not the only reason self-supervised event flow (DSEC EPE 2.3–3.2) trails supervised
flow (≈0.7). Other sources:
* event-free pixels, which only the prior can fill;
* brightness-constancy violations: occlusion, flicker, specularities;
* network capacity;
* nonlinear motion over 100 ms.

A proper loss addresses the alignment bias only. The paper must quantify that share (Exp 1).

---

## 2. How we solve it (mechanism) — and why each part exists

Score each **raw** event by the density predicted at its own position and time by all *other* events transported to
its timestamp:

    S(θ) = mean_i log( ε + Σ_{j≠i} w(t_i - t_j) K_h(x̃_i - Φ_θ(x_j; t_j -> t_i)) / (Σ_j w(t_i - t_j) · f_vis,i) )

| Component | Why it is there | Evidence |
|---|---|---|
| Raw anchor (evaluation point never moves) | removes the deformation lever | collapse scans, dense tests [measured] |
| Leave-one-out (j ≠ i) | otherwise each event predicts itself | standard; required for consistency |
| Anchor smoothing x̃_i = x_i + h z | without it, finite h rewards local contraction ("deconvolution"); with it the population score is −KL(p⊛K_h ‖ q⊛K_h), proper at every h | 1-D test [measured]. This is a spread-KL: cite Zhang et al., ICML'20 |
| Dequantisation U[−½, ½]² | integer pixel coordinates are not a transport-invariant density | [theory] |
| Exposure correction f_vis | content entering or leaving the frame has no sources at other times | interior bias 0.85% → 0.04%. *Note: the 0.85% here is MotionPriorCMax's value from §1, not our score before the correction.* |
| Raw-position anchor margin (6 px) | residual border band; motion-independent selection, so it stays proper | border EPE 4.7 → 3.8 px [measured] |
| Kernel-width annealing (4 → 1 px) | basin of attraction | partial [measured] |
| Exact pair kernel (C++/CUDA) | exact LOO and exact Gaussian; no slice approximations | tests vs brute force [measured] |

---

## 3. Weaknesses (where it lacks)

Severity: H = could sink the paper, M = costs a table or a reviewer point, L = cosmetic.

| # | Weakness | Sev | Evidence | Fixes (detail in §5) |
|---|---|---|---|---|
| W1 | **Upside unknown.** 44% of evaluated DSEC pixels have no events; bias reduction may move total EPE little | H | 56% event coverage [measured] | Exp 1 decomposition; combine with a better spatial prior; report event-pixel EPE |
| W2 | **Basin / convergence.** From zero, 8.7–9.2 px vs MPCM 4.8 px after 150 per-window steps | H | [measured] | lag curriculum; OT/MMD warm-up (5A2–3); EM/Gauss–Newton steps; warm start from an MPCM checkpoint; coarse-to-fine |
| W3 | **Non-stationary firing.** Events fire ∝ \|∇L·v\|; when motion direction or speed changes in the window, different edges fire at different times, so q ≠ p at the truth at finite h (O(1) term; only the h → 0 support term is guaranteed) | M–H | [theory] | short lag window τ; rate-aware weighting (5A4); polarity- or direction-split scores; full point-process likelihood |
| W4 | **Occlusion / disocclusion inside the image.** Same mechanism as frame entry, but the exposure correction only knows the frame | M | [theory] | mutual (past/future) anchoring (5A1); robust ε mixture per anchor; layered motion |
| W5 | **Border band** still drifts more than the interior | M | 3.8 vs 1.25 px [measured] | margin + network prior; unbalanced OT (5A2); one-sided anchoring at borders |
| W6 | **Hot pixels and flicker.** Static event sources favour zero motion locally (shared by all CMax methods) | M | [theory] | hot-pixel mask from rate statistics; flicker detection (periodic timestamps); learned ε |
| W7 | **Compute.** O(N·k) pairs and a plan of N·S entries; ~16M entries / ~0.6 GB per 1M-event sample at τ = 1 | M | [estimate] | anchor/source subsampling (unbiased KDE); τ = 0.5; fused shared-memory kernel; plan reuse; fp16 distances |
| W8 | **Gradient variance** from the stochastic anchor jitter | M | F2/F3 noise suspected [unverified] | several jitter draws per anchor (neighbour lists reused); common random numbers; quasi-random jitter |
| W9 | **"You swapped λ for h, τ, ε, margin, annealing."** | M | — | show these do not affect properness (proved for h; τ and w arbitrary); sensitivity sweeps; pick h by held-out score (LOO-CV) |
| W10 | **F2/F3 inconclusive**; our score was −6% on F3 zoom | M | [measured, unexplained] | rerun with fixed seeds and the exact kernel; finer grid; more steps |
| W11 | **Sliced kernel approximations** (slice width, finite-difference log-det) | L | sliced worse than exact in F6 [measured] | use exact; sliced only for nonlinear priors (Bezier on EVIMO2) with narrow slices |
| W12 | **DSEC rectification breaks the integer lattice** that dequantisation assumes | L–M | [theory] | dequantise in raw sensor coordinates, then rectify through the map |
| W13 | **Synthetic realism.** Our DVS simulator lacks refractory period, leak, shot noise and bandwidth limits | M | — | re-run the suite with v2e / ESIM noise models; rely on real-data tests |
| W14 | **Real-data evidence is thin**: 1 sequence, 3 windows; the gradient-focus baseline lacks KNN+TV | M | [measured] | Exp 2 on all GT windows; the full MPCM loss in the comparison |
| W15 | **Statistical efficiency vs CMax** on well-posed classes is unknown | L–M | [open] | asymptotic variance derivation; rotation benchmark (ECD) |
| W16 | **Theory gaps**: L1-gradient "scatter" theorem; transported LOO-KDE consistency lemma; non-stationary firing bias bound | M (ICML) | [open] | write proofs (§6) |

---

## 4. Attack surface (what a reviewer will say) and the defence each needs

| # | Attack | Defence | Evidence still needed |
|---|---|---|---|
| R1 | "Collapse is known (Shiba 2022) and already handled by regularisers / multi-reference." | Impossibility: no reference distribution (κ > 4) and no data-independent λ fixes it. Even the patched state of the art is biased in the interior. | Have synthetic; need real-data local-bias test (Exp 3) |
| R2 | "Networks never output collapsed fields, so improperness is academic." | The harm is *local* bias near the truth, not global collapse. Measured on dense fields with the official loss. | Exp 3 on real DSEC; Exp 1 to show the bias matters for EPE |
| R3 | "The fix is a known tool (LOO likelihood / proper scoring / spread divergence / CPD)." | Concede the tools. Novelty: (a) the diagnosis of a whole widely used family, with impossibility theorems for its patches; (b) a general principle (raw-vs-transported divergences) with several proper instantiations; (c) the metric; (d) state-of-the-art results | 5A2–3 instantiations; second domain |
| R4 | "Smoothing anchors adds noise; why not h → 0?" | At finite h the plain score is provably biased (measured); smoothing makes it exact at every h. Variance is reduced by multiple draws. | Variance-vs-draws plot |
| R5 | "Your theory assumes a transported static density; event firing depends on motion." | Edge-regime theorem: support alignment dominates as log(1/h). Remaining finite-h bias bound plus rate-aware variant. | W3 bound; rate-aware ablation |
| R6 | "Gains come from tuning, not properness." | Same network, data and schedule; equal tuning budget; sensitivity curves for both. Remove the target's patches: theirs degrades, ours does not need them. | Ablation table on DSEC validation |
| R7 | "Not better on DSEC." | Main risk; see §7 hedges | Exp 4 |
| R8 | "Too slow." | Report ms/step vs MPCM's KNN look-up (theirs is O(n²) KNN per bin without KeOps). Subsampling with error bars. | GPU benchmarks |
| R9 | "Event cameras are niche for ICML." | Second domain: groupwise registration / congealing, LiDAR deskew | Exp 6 |
| R10 | "Your metric is just your loss evaluated on held-out data: circular." | Validate against ground truth: rank correlation of the metric with true EPE across many checkpoints and methods, vs FWL / RSAT | Exp 5 |
| R11 | "Exposure correction uses the current motion: circular or exploitable." | Detached; motion-independent alternatives (margin); show results with and without | Ablation |
| R12 | "Synthetic simulator favours your model." | v2e / ESIM noise; real-data tests | W13 |
| R13 | "Does properness survive model misspecification (linear prior over 100 ms, nonlinear real motion)?" | Proper score → KL-projection onto the model class; improper score → projection plus extra bias. Show on EVIMO2 (Bezier) and a degree-1 vs degree-2 ablation. | Ablation |
| R14 | "Hot pixels / flicker / night sequences." | Shared by all brightness-constancy methods; filter plus learned ε | Per-sequence results |

---

## 5. Solutions of every type

### 5A. Objective-level (the "principle": compare raw data at time t with data transported to t)
1. **Mutual (past/future) anchoring** for occlusion and borders (W4, W5).
   * Compute separate predictive densities from past-only and future-only sources.
   * Score each anchor by their mixture, with the weight given by visible exposure.
   * Content that appears (disocclusion) is predicted by the future, content that disappears by the past.
   * Proper if the mixture weights are the exposure fractions [to prove].
2. **Unbalanced optimal transport** between the raw slice at t and the transported slices (Sinkhorn divergence,
   debiased).
   * Proper (zero iff equal), with a *wide basin*: OT gradients do not vanish far from the truth.
   * Creation and destruction of mass handles frame entry and occlusion natively.
   * Multiscale Sinkhorn (geomloss / KeOps) handles ~1M points per second on a GPU.
   * Use as a warm-up for W2, or as a standalone variant.
3. **MMD / energy distance** between raw and transported slices.
   * Proper with characteristic kernels; multi-scale kernels give a wide basin; unbiased U-statistics; random
     features are cheap.
   * Energy distance is scale-free (no h at all).
4. **Rate-aware anchored likelihood** (W3).
   * Weight each source by the predicted firing rate ratio |∇L·v|(t_i) / |∇L·v|(t_j).
   * ∇L is estimated from the transported event image; the motion predicts the rate change.
   * This is the full spatio-temporal point-process model: map-free intensity = transported events × a physics rate
     term. It reconnects to the first (Poisson) idea.
5. **Polarity- or direction-split scores** (W3): separate densities per polarity; or the minimum over split vs pooled.
6. **Robust per-anchor mixture** (W6): log(ε_i + ρ_i) with ε fitted by EM (noise / outlier class), plus an optional
   hot-pixel mask.
7. **Temporal kernel w as a design freedom.** Any positive w keeps properness, so short-lag-heavy w helps the basin
   and long-lag w gives precision (curriculum, below).

### 5B. Optimisation-level (W2)
1. Kernel-width annealing (have) → extend to 8 → 1 px; pick the final h by held-out score.
2. **Lag curriculum.** Start w on |Δt| < δ (pairs barely move, so the basin is wide) and widen to the full window.
   Proper at every stage.
3. OT / MMD warm-up (5A2–3), then switch to the anchored likelihood for statistical sharpness.
4. EM / Gauss–Newton with soft correspondences (Thm 8) for model-based estimation; second-order steps escape plateaus.
5. Warm-start from an MPCM-trained network, then fine-tune with our loss. Even if from-scratch training lags, this
   isolates "proper loss removes bias" (an honest secondary result).
6. Coarse-to-fine spatial pyramid (event coordinates downsampled ×2^l).

### 5C. Model-level
1. Network spatial prior for event-free pixels (W1). Our loss does not act there, so use a standard edge-aware
   smoothness, or let the U-Net prior do it. Report event-pixel EPE separately.
2. Degree-2 / Bezier trajectories on DSEC (nonlinear motion in 100 ms; R13).
3. Two-layer motion or occlusion-aware visibility (W4) — heavier; only if 5A1 is insufficient.

### 5D. Data-level
1. Dequantise in raw coordinates, then rectify (W12).
2. Hot-pixel and flicker filtering (W6); report with and without.
3. Event subsampling with variance control (W7): anchors ~60k, sources ~300k per sample.

### 5E. Systems-level (W7)
1. Fused CUDA kernel with shared-memory cell tiles (neighbour cells staged once per block), fp16 distances, fp32
   accumulation.
2. Plan reuse across forward and backward (have) and across jitter draws (W8).
3. Restrict τ (0.5) and use sliced splats for very dense windows.

### 5F. Evaluation-level
1. Error decomposition: event pixels vs event-free pixels; static vs moving; border vs interior; per sequence
   (forward motion vs turning).
2. Local-bias-at-truth test on real data (Exp 3).
3. Metric validity study (Exp 5).

### 5G. Positioning-level
1. ICML: principle + impossibility + several proper instantiations + metric + two domains + state-of-the-art event
   results.
2. ICCV: event results first, theory compact.
3. Hedge: analysis paper (TMLR / ICML) if the state-of-the-art margin is small (§7).

---

## 6. Theory to write (closes W16, R5)
1. **Scatter theorem for L1-gradient sharpness.** Its sup is attained by decorrelated configurations reachable by dense
   fields. Plus a local bias lemma: a non-zero first variation at the truth for edge scenes.
2. **Transported LOO-KDE consistency** (Thm 5a lemma) and the edge-regime separation lemma (Thm 5b).
3. **Non-stationary firing.** A bound on the finite-h bias in terms of the rate variation along trajectories, and
   properness of the rate-aware variant.
4. **Mutual-anchoring properness** with exposure weights.
5. **Impossibility for regularisers**, stated with explicit constants (λ_min ∝ κ; bias ∝ λ).

---

## 7. De-risking experiment queue (ordered by information per unit cost)

| # | Experiment | What it decides | Cost / where |
|---|---|---|---|
| **1** | **EPE decomposition of the official MotionPriorCMax checkpoint** on DSEC validation (zurich_city_05_b, 11_c): event vs event-free pixels, border vs interior, per sequence | The size of the prize (W1). If event-pixel EPE is a small share of 3.20, re-plan (5C1 matters more) | CPU feasible here with their checkpoint plus about 2 validation sequences (several GB) |
| **2** | Real-DSEC per-window comparison on **all 41 GT windows of thun_00_a**, with the full MPCM loss (KNN + TV) as a row | Hardens the current 3-window result (W14) | CPU here, a few hours |
| **3** | **Local bias at the truth on real data.** Set the motion to the DSEC ground-truth flow, take the gradient of each objective with half the events, step along it, and evaluate the step on the other half. A proper loss should not improve on held-out events. | Real-data proof of the issue (R1, R2) | CPU here, cheap |
| 4 | DSEC training: baseline reproduction, then ours (and the warm-start fine-tune) | The headline | GPU |
| 5 | Metric validity: Spearman correlation of held-out score / FWL / RSAT with ground-truth EPE over ~30 checkpoints | R10, plus a standalone contribution | GPU (checkpoints from Exp 4) |
| 6 | Second domain: groupwise affine registration (congealing) where entropy objectives shrink | R9 | CPU / GPU, small |
| 7 | OT / MMD instantiations (5A2–3) and the lag curriculum (5B2) on the dense synthetic test from zero | W2 | CPU here |
| 8 | F2/F3 rerun, deterministic, exact kernel | W10 | CPU here |

### Hedges if Exp 4 does not beat 3.20
* Fine-tuning gain from the MPCM checkpoint (proper loss removes bias).
* State of the art in the model-based track vs MultiCM (3.47).
* Win on event-pixel EPE and on divergent-motion sequences.
* A theory + metric analysis paper.
* Combine ours with their patches if they are complementary. That is still a win, framed as "proper core + priors".
