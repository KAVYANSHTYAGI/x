# Continuous-time event estimation — an algorithmic direction, iterated

> **Superseded for the mathematics by `PROPER_ALIGNMENT_MATH.md`.** That file contains numbered theorems, proofs,
> numerical checks and corrections. P1/P2/P3/P4 here correspond to its Thm 5 / Thm 7 / Prop 6 / Thm 8. It also adds
> the negative results proving the gap (Thms 1–4): improperness, the multi-reference bound, the regulariser dilemma
> and the reference bias.

Status: thinking document. Two cheap 1-D/2-D toy checks were run to test claims (scripts in `ct_toys/`);
no real data, no training. Everything marked **[verify]** must be checked against the full papers before it is claimed.

---

## 0. The arena (what the provided papers do)

| Work | Turns events into... | then estimates with |
|---|---|---|
| GP-CT on factor graphs (Holmes, Lilge, Guo, Dellaert, Barfoot, arXiv 2605.09073; GTSAM blog) | — (generic Gaussian measurements at known times) | WNOA GP prior, O(1) interpolation = one elimination step |
| Magnus GP prior on Lie groups (Barfoot, Le Gentil, Lilge, 2601.03360) | — | global LTV GP on Lie groups |
| AsynEVO 2402.16398, AsynEIO 2411.12175 (+2412.03136) | feature tracklets | sparse GP regression, sliding-window factor graph |
| AERO-VIS 2605.07885 | learned keypoints | asynchronous stereo event-inertial SLAM |
| DAPEVO 2609.30947 | patch correlations | learned recurrent update + bundle adjustment |
| Normal flow 2407.12239; minimal solvers 2606.09218; line pose 2606.09139 | normal flow / flow / event–line pairs | linear and polynomial minimal solvers |
| CMax, EMin, CMax-SLAM, EROAM | sharpness / entropy of warped-event image | windowed or spline optimization |
| ST-PPP (Gu et al., ICCV'21) | per-pixel Poisson, Gamma-marginalized rates | rotation, constant velocity, 30k-event packets |

Common thread: almost all of them *derive an intermediate* (tracks, flow, IWE contrast) and estimate on that.
The only intermediate-free family, CMax/EMin, has a known pathology, **event collapse** (Shiba et al. 2022). Collapse
forces short windows, low-DoF warps and hand-tuned regularizers, which is exactly what keeps it out of continuous-time estimation.

---

## 1. Iteration log

### v1 — "Poisson point-process factors on a GP factor graph" → rejected as an ML contribution
Claims were: (A1) interval-wise factorization keeps the information matrix block-tridiagonal; (A2) Fisher-scoring/IRLS
factor; (A3) MLE consistency means no collapse; (A4) time-rescaling KS tests drive knot insertion.

Why it fails the novelty bar:
* A1/A2 are the standard point-process state-space machinery: block-tridiagonal Newton for point-process
  observations of a Markov latent is Paninski et al., *A new look at state-space models for neural data* (J. Comput.
  Neurosci. 2010) and Koyama & Paninski; point-process adaptive filters are Eden et al. 2004; IRLS for Poisson GLMs is textbook.
* A4: time-rescaling goodness-of-fit is Brown et al. 2002. Using it to place knots is new-ish but small.
* A3 is only as good as the intensity model, which needs a photometric map, and ST-PPP (closest prior) still collapses under zoom.
Verdict: correct engineering, but an application of known tools. Fine for a robotics venue, thin for ICML.

### v2 — "Exact level-crossing (first-passage) likelihood with silence/censoring constraints" → rejected
Idea: events are threshold crossings with a per-pixel reset, not Poisson. Event density under jittered thresholds
C_k ~ N(C, s²) is N(ΔL(t_i); C, s²)·|∂_tΔL(t_i)|, and silent pixels give the survival factor 1 − Φ((max|ΔL| − C)/s).
Hoped-for headline: "rate models throw away a (C/s)² factor of Fisher information per event."

Killed by two findings:
1. **Toy check** (`ct_toys/fisher_toy.py`, one pixel, constant drift). The rate estimator applied to level-crossing data
   is nearly as good, because crossings are almost periodic, so counts barely fluctuate:

   | s/C | events in window | RMSE ratio, rate / level-crossing |
   |---|---|---|
   | 0.05 | 3–10 | 1.8–2.5× |
   | 0.05 | 200 | 1.2× |
   | 0.10–0.30 | any | 1.0–1.7× |
   | 1.0 | small | level-crossing is worse |

   The (C/s)² argument compared a *model's* implied Fisher information, not estimator efficiency under the true
   process. A 1–2× gain is not a paper.
2. **Prior art.** EDFilter (CVPR 2026, arXiv 2504.07503) already models events as first-passage probability flux of a drifted
   diffusion, with a hazard likelihood (used for denoising). Hashimoto–Serizawa–Kishida (arXiv 2603.05011) already do
   receding-horizon MLE of continuous-time (Neural-ODE) dynamics and the threshold under a history-dependent point process
   with a compensator. Bardow et al. (CVPR'16) used "no event ⇒ |ΔL|<C" constraints. Gallego et al. (TPAMI'18) used the ΔL≈C residual.

### v3 — **Proper alignment scores: every event is its own reference** → current direction
Found by asking *why* CMax, EMin and ST-PPP collapse while being "likelihood-like". Two toy checks support the answer (§3).

---

## 2. v3 — the idea

### 2.1 Problem (specific)
Sharpness-type alignment objectives (CMax variance, EMin entropy, ST-PPP packet likelihood) evaluate the score at
*warped* event positions. The parameters move the evaluation points, so the objective can be raised by warps that
contract space: event collapse. Consequences in the literature:
* hand-tuned regularizers (Shiba 2022: R_D = mean div f, R_A = mean |det J|, a weight λ and trimming thresholds α);
* multi-reference tricks (Shiba ECCV'22);
* restriction to "well-posed" low-DoF warps (rotation, 2-DoF flow);
* short packets with constant velocity;
* an arbitrary reference time t_ref.

None of these is compatible with long-horizon continuous-time trajectories or high-DoF / dense warps.
Gu et al. state explicitly that the Jacobian correction for volume-changing warps is "left for future work".

### 2.2 Objective
Events e_i = (x_i, t_i, p_i). A motion θ (finite-dim warp, dense flow, or a continuous-time trajectory) induces flow
maps Φ^θ_{s→t} on the sensor plane, with Φ_{t→t} = id and Φ_{s→t}∘Φ_{r→s} = Φ_{r→t}. Define

    S(θ) = (1/N) Σ_i log[ β/|Ω| + (1−β) ρ̂^θ_{−i}(x_i) ],
    ρ̂^θ_{−i}(x) = Σ_{j≠i, p_j=p_i} w(t_i−t_j) K_H(x − Φ^θ_{t_j→t_i}(x_j)) / Σ_{j≠i} w(t_i−t_j)

In words: the leave-one-out predictive log-density of each **raw** event, under the other events transported to *its own
timestamp*. Properties:
* The evaluation point x_i is a measured quantity that no θ can move.
* The kernel K_H lives in sensor coordinates, where pixels and noise live.
* There is no t_ref.
* β is an explicit noise-event component, which covers the joint noise and motion setting of Shiba ICCV'25.

### 2.3 Theory (to be written as propositions)

**P1. Properness (Gibbs).** Suppose events are generated by transporting a scene density with the true motion θ*, so that
x | t ~ p_t. As N→∞ and H→0 at the usual KDE rates,

    S(θ) → −E_t[ H(p_t) + KL(p_t ‖ q^θ_t) ],   q^θ_t = ∫ w(t−s) (Φ^θ_{s→t})_# p_s ds / ∫ w.

At θ*, (Φ*_{s→t})_# p_s = p_t, so KL = 0 and θ* is a global maximizer. Equality holds only for motions whose transports
preserve the scene measure, i.e. genuine geometric symmetries such as the aperture problem along a straight edge.
These are real non-identifiabilities, not artifacts. S is a cross-entropy (a log score, which is strictly proper).
CMax maximizes ‖ρ̂_warped‖² ≈ exp(−Rényi-2 entropy) of the *warped* set and EMin minimizes its Shannon entropy.
Entropy of a moved set is not a proper score, and Dirac-like configurations maximize it.

**P2. Where the Jacobian lives (unifies the collapse regularizers).** Take y_j = Φ_{t_j→0}(x_j) and
A_i = DΦ_{t_i→0}(x_i). For small H,

    log ρ̂_{−i}(x_i) ≈ log ρ̃_{−i}(y_i; A_i H A_iᵀ) + log|det A_i|.

The first term is EMin's per-event log-density in the warped frame (with event-adapted bandwidth). The second is the exact
change-of-variables term with weight 1. Jacobi's formula d/dt log det DΦ = (div f)∘Φ gives log|det A_i| = −∫ div f.
Hence Shiba's two regularizers are first-order surrogates of this single exact term:
* R_A ≈ log det linearized at 1 (log d ≈ d − 1);
* R_D is its time derivative.

The weight λ and the thresholds α disappear.

**P3. Boundedness.** For every θ, S(θ) ≤ log(β/|Ω| + K_H(0)), because the kernel is normalized in fixed coordinates.
CMax is unbounded along collapse paths. Boundedness alone is not enough (bounded EMin still collapses, §3); P1 is the operative statement.

**P4. Algorithm and complexity.** S is a log-sum, so EM / MM applies. The E-step gives responsibilities r_ij ∝ w K_H(·),
and the M-step solves

    min_θ Σ_ij r_ij ‖x_i − Φ^θ_{t_j→t_i}(x_j)‖²_{H⁻¹} + ‖θ‖²_GP.

This is a weighted nonlinear least squares, so monotone ascent with line search holds.
* With a GP trajectory (WNOA/WNOJ on SO(3)/SE(3)), each pair term touches the ≤4 knots bounding t_i and t_j
  (two-time interpolation factors, a generalization of GTSAM's `WnoaInterpFactor` to two query times).
* With temporal support |t_i−t_j| ≤ τ, the Gauss–Newton matrix is block-banded with bandwidth b = ⌈τ/Δ⌉+1.
* Cost: pair assembly O(N·k̄) with a spatial hash, plus banded Cholesky O(K b² d³).

The method is windowless, tracker-free and reference-free. Laplace covariance comes from the final WLS. H is chosen by
maximizing S itself (LOO cross-validation), and deterministic annealing of H widens the basin.

Relation to Zhu–Atanasov–Daniilidis (ICRA'17, EM data association for event flow): for pure translation their pairwise
residual (x_i − x_j) − v(t_i − t_j) is exactly a transport j→i evaluated at the raw x_i. So their method is the
translation / constant-flow special case. Translation is volume-preserving, which explains why it never collapsed.
**[verify the exact form in the paper]**

### 2.4 What is and isn't new
Not new:
* GP-CT factor graphs (Barfoot / Dellaert);
* EM soft association (Zhu'17, CPD);
* point-process alignment (Gu'21);
* the existence of collapse and its regularizers (Shiba'22);
* LOO-KDE likelihood (statistics).

New (pending **[verify]** of a targeted search for "event alignment cross-entropy / log-det / reference-free"):
1. The diagnosis: CMax, EMin and ST-PPP are entropy scores of moved points (improper); collapse is impropriety.
2. The proper, reference-free score and its decomposition (P2), deriving Shiba's regularizers exactly via Jacobi's formula.
3. The continuous-time banded EM-GN algorithm, which makes collapse-prone models usable over long horizons.
4. Hyperparameter removal: λ, α, t_ref and window length are gone; H is selected by LOO.

---

## 3. Evidence so far (toy, `ct_toys/collapse_toy.py`)
Setup: synthetic edges translating at v = (60, 0) px/s for 0.5 s, 128² sensor, about 4k events. Model: translation + zoom
rate h, with truth h = 0. This is Shiba's collapse setting.

| h | CMax (t_ref=0), ratio to truth | CMax multi-ref, ratio to truth | EMin, Δ from truth | EMin + exact log det, Δ | **S (ours)**, Δ |
|---|---|---|---|---|---|
| −4 | 0.04 | 0.68 | −3.69 | −1.67 | −1.60 |
| 1 | 0.82 | 0.57 | −0.30 | −0.81 | −0.72 |
| 4 | 1.69 | 0.68 | +0.14 | −1.88 | −1.62 |
| 8 | 7.65 | 2.72 | +1.20 | −2.84 | −2.16 |
| 16 | **20.4** | **7.49** | **+2.36** | −5.72 | −2.78 |

* CMax collapses, and multi-reference only delays collapse.
* EMin collapses.
* Adding the exact log-det term (weight 1) to EMin removes collapse; this is P2 numerically.
* S peaks at the truth and decreases monotonically in both directions.
* At h = 0, S and CMax both peak at v_x = 60 with similar curvature, so no loss of precision is visible.

One toy is not evidence for real data, but it is consistent with P1–P2 and costs nothing to reproduce.

---

## 4. Plan
* **Stage 0 (theory, paper-and-pencil).** Prove P1 with KDE rates, P2 with explicit remainder, P3; characterize
  identifiable symmetry sets for rotation, Sim(2), homography and affine warps.
* **Stage 1 (finite-dim warps, real data).** ECD (DAVIS240C shapes / poster / boxes / dynamic_rotation): angular-velocity RMS vs
  CMax, EMin, ST-PPP, and CMax + Shiba regularizers.
  Collapse-prone models (Sim(2), homography, 4-DoF) on the MVSEC / DSEC sequences from Shiba's protocol:
  no regularizer, no tuning.
* **Stage 2 (continuous time).** WNOA GP on SO(3), EM-GN with two-time factors, on ECD rotational sequences and the
  CMax-SLAM datasets vs CMax-SLAM and EROAM. Report ATE, consistency (NEES from Laplace) and compute vs τ.
* **Stage 3 (ML-facing).** Use S as the self-supervised loss for event optical flow (DSEC-Flow, MVSEC) in a public CMax-trained
  pipeline (Shiba "Secrets" code, Paredes-Vallés "Taming CMax"), replacing the multi-reference loss and regularizers.
* **Stretch.** 6-DoF with stereo depth (ESIM synthetic → TUM-VIE / VECtor). Generality to other sharpness-based
  alignment such as LiDAR scan deskewing via map entropy, which would make an ICML framing as a general principle stronger.

GPU kernels (reuse evkit infra: decoders, EVB store, deterministic block reductions):
* (a) hash of transported events in map space (cube map on S² for rotation);
* (b) fused pair kernel: transport, residual, kernel weight, responsibilities, and J^T W J scattered into ≤4-knot blocks;
* (c) banded block Cholesky.
Not yet written.

## 5. Kill criteria
* A prior paper already uses a raw-anchored cross-entropy or exact log-det for event alignment → reposition as the
  continuous-time algorithm only.
* On ECD rotation (no collapse risk), S is worse than CMax by >5% RMS → the contribution narrows to collapse-prone models.
* As a flow-learning loss, S does not beat multi-reference CMax on DSEC → drop Stage 3 from the paper.
* Compute: if pair assembly is more than about 20× CMax cost at equal accuracy, the windowless claim needs subsampling theory.

## 6. Risks / open questions
* Bandwidth: anisotropic, edge-aligned kernels (normal-flow aware, cf. 2407.12239) may be needed for precision on edges.
* Depth is a nuisance for 6-DoF transport. Stage it; do not promise monocular 6-DoF.
* Temporal kernel w and τ trade bandwidth (sparsity) against information. Is τ selectable by LOO too?
* Polarity: restricting to same-polarity pairs is physically motivated, but reverses under motion reversal. Check.

## Sources
* GP-CT: https://gtsam.org/2026/05/20/gp-ct-in-gtsam.html · arXiv 2605.09073 · 2601.03360 · SLAM Handbook repo
* Event SLAM/VO: 2411.12175 · 2402.16398 · 2412.03136 · 2605.07885 · 2609.30947 · 2306.01188
* Geometric: 2606.09139 · 2606.09218 · 2407.12239
* Alignment: Gu et al. ICCV'21 (2106.06887) · Shiba et al. Sensors'22 (2207.04007) · Shiba ECCV'22 (2207.10022) · Shiba ICCV'25 (2504.04029)
* Prior art that killed v2: EDFilter (2504.07503) · Hashimoto et al. (2603.05011) · Bardow CVPR'16 · Gallego TPAMI'18 (1607.03468)
* Prior art that weakened v1: Paninski et al. 2010 (J. Comput. Neurosci. 29:107) · Eden et al. 2004 · Brown et al. 2002
