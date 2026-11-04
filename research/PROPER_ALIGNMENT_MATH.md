# Proper alignment scores for event cameras — mathematics, gap proof, and plan

Phase: **initial mathematics + proving the gap + planning**. No method is built here.

Every quantitative prediction below was checked on synthetic toys:
* `ct_toys/collapse_toy.py`
* `ct_toys/theory_checks.py` (blocks A–E)
* `ct_toys/dilemma_scaling.py`

Toys show internal consistency of the maths, not real-data performance.

Notation for statement status:
* **[proved]**: full argument below.
* **[sketch]**: argument complete in outline, technical conditions still to be written.
* **[derived+checked]**: asymptotic or perturbative derivation, matched numerically.
* **[conjecture]**: not yet proved.

---

## 0. The gap in one paragraph

Every self-supervised event-alignment objective in use from 2018 to 2024 scores the sharpness of events *moved into a
reference frame*:
* CMax variance (Gallego CVPR'18);
* gradient-magnitude sharpness (Hamann ECCV'24);
* entropy minimisation, EMin (Nunes & Demiris ECCV'20);
* per-pixel Poisson / negative-binomial likelihood, ST-PPP (Gu ICCV'21).

We prove (Thm 1) that any such objective is **improper**: its population optimum is not the true motion for any warp
class that contains a contraction path. Such classes include zoom, Sim(2), affine, homography, forward translation and
dense flow. The degenerate optimum ("event collapse") is *data-independent*.

We then prove that each patch in the literature fails to restore properness:
* multi-reference and random-reference losses (Thm 2, universal bound κ > 4);
* zero-sum gauge constraints (Remark 2.1);
* divergence / deformation regularisers (Thm 3, no data-independent weight).

Even where collapse is avoided locally, single-reference CMax is first-order **biased** on divergent motion, by
Δh = γσ²(t̄ − t_ref)/(σ_t²ρ²) (Thm 4).

We give an **anchored** score that is proper (Thm 5): each raw event is scored under the other events transported to its
own timestamp. It contains the exact change-of-variables term that the regularisers approximate (Thm 7). It admits a
monotone EM / Gauss–Newton algorithm with block-banded structure on a continuous-time GP trajectory (Thm 8).

The objectives in use and their collapse fixes are listed in §1.

---

## 1. Literature evidence (verified from the papers' text; see Sources)

| Work | Objective | Evaluated at | Collapse handling | Likelihood / Jacobian? |
|---|---|---|---|---|
| CMax, Gallego CVPR'18 | IWE variance at t_ref | warped events | restrict warp class | no |
| EMin, Nunes & Demiris ECCV'20 | entropy of warped events | warped events | — (Gu'21 reports collapse) | no |
| ST-PPP, Gu et al. ICCV'21 | per-pixel NB marginal of aligned counts | warped events | padding; "Jacobian correction… future work" | likelihood without Jacobian |
| Shiba et al., Sensors'22 | CMax − λR, R_D = mean div f, R_A = mean \|det J\|, trimmed at α | warped events | regulariser, weight λ, thresholds α | "Bayesian" framing only; no change of variables |
| Shiba et al., AIS'23 | time derivative of the Jacobian (rate of area change) | warped events | regulariser | no |
| Shiba et al., ECCV'22 ("Secrets") | multi-reference focus loss | warped events | multiple t_ref | no |
| Paredes-Vallés et al., ICCV'23 ("Taming") | avg-timestamp IWE, multi-ref, multi-timescale | warped events | iterative warping, masking; states FWL/RSAT *metrics* are fooled by collapse | no |
| Hamann et al., ECCV'24 (MotionPriorCMax) | 1/∫‖∇I‖ at random t_ref + λ·TV(displacement), λ = 3e-3 | warped events | regulariser + 32-trajectory soft assignment (N_traj = 1 collapses) | no |
| Zhu et al., ICRA'17 | EM soft association, pairwise | **raw** (for translation) | none needed (translation) | special case of §4, **[verify]** exact form |
| Groupwise registration (congealing, CMTK) | stack entropy | common frame | zero-sum transform constraints | — (see Remark 2.1) |

The continuous-time arena (GP factor graphs; AsynEVO/AsynEIO; CMax-SLAM; EROAM) either uses tracker-derived
intermediates or restricts CMax to rotations, the "well-posed" class. **No proper, map-free, tracker-free objective
exists for continuous-time trajectories in collapse-prone classes**, which includes any motion with a forward component.

---

## 2. Setting and definitions

**D1 (data).** Events are (x_i, t_i) ∈ Ω × [0, T], i = 1..N, with Ω ⊂ ℝ² the bounded sensor domain. Model: event times
have density π on [0, T], and given t, x has spatial density p_t on Ω. The pairs are independent across events
(binomial / Poisson process conditional on N). Polarity is ignored; see O6.

**D2 (motion).** A motion θ ∈ Θ induces a two-time flow Φ^θ_{s→t}: the C¹ diffeomorphisms (onto their images) generated
by a velocity field f^θ(x, t), with Φ_{t→t} = id and Φ_{s→t} = Φ_{r→t} ∘ Φ_{s→r}.
Truth is θ*: there is a firing measure m_t on reference coordinates with p_t ∝ (Φ*_{0→t})_# m_t restricted to Ω.
* *Stationary* means m_t ≡ m.
* For a real event camera m_t(y) ∝ |∇M(y) · v_t(y)|, which is generally *not* stationary.

**D3 (proper).** An objective F_N(θ) with population limit F(θ) = lim F_N / N is *proper* on (Θ, 𝒟) if F(θ*) ≥ F(θ)
for all θ ∈ Θ and all (θ*, m) ∈ 𝒟. It is *strictly proper* modulo symmetries if equality forces the transports of θ to
agree with those of θ* on the support. It is *asymptotically proper* if this holds in the vanishing-kernel limit.
(This is Fisher consistency. "Proper" is by analogy with scoring rules.)

**D4 (reference-frame objective).** F(θ) = Ψ(Y_θ) with Y_θ = {Φ^θ_{t_j→t_ref}(x_j)}_j, or an average over t_ref ∼ ν.
Ψ is a *concentration functional* if it is translation-invariant, continuous at the coincident configuration, and
Ψ(Y) ≤ Ψ_max(N) with equality iff all points coincide (or, in a weak version, including coincidence).

**D5 (contraction path).** A curve θ(s) in Θ, a centre c ∈ Ω and a set of times A with π(A) = m > 0 such that
sup_{x∈Ω} |Φ^{θ(s)}_{t→t_ref}(x) − c| → 0 for all t ∈ A as s → ∞.

Classification [proved by explicit paths]:

| Warp class | Contraction path? |
|---|---|
| 2-DoF translation, SO(3) rotation (bounded perspective distortion), planar SE(2) | **no** (well-posed) |
| zoom / 1-DoF divergence, Sim(2), 4-DoF in-plane, affine, homography, 6-DoF with forward translation and a depth model, unregularised dense flow / per-pixel trajectories | **yes** |

**D6 (anchored score).** With temporal weight w ≥ 0 supported on [−τ, τ], kernel K_H normalised in sensor coordinates,
and floor ε ≥ 0:

    S_N(θ) = (1/N) Σ_i log( ε + ρ̂^θ_{−i}(x_i) ),
    ρ̂^θ_{−i}(x) = Σ_{j≠i} w(t_i − t_j) K_H(x − Φ^θ_{t_j→t_i}(x_j)) / Σ_{j≠i} w(t_i − t_j).

The evaluation point x_i is a raw measurement, so no θ moves it. There is no t_ref.

---

## 3. Negative results: the gap, proved

### Lemma 1 (the published functionals are concentration functionals) [proved]
* (a) **IWE energy.** ∫ I_σ² with I_σ = Σ_j N(·; y_j, σ²):
  ∫ I² = Σ_{j,k} (4πσ²)^{-1} exp(−|y_j − y_k|²/4σ²) ≤ N²/(4πσ²), with equality iff all y_j coincide.
  The IWE variance equals ∫I²/|Ω| − (N/|Ω|)² whenever mass stays in Ω.
* (b) **Negative KDE entropy (EMin).** −Ĥ = (1/N) Σ_i log((1/N) Σ_k K(y_i − y_k)) ≤ log K(0), with equality iff coincident.
* (c) **Profile per-pixel Poisson.** The log-likelihood is Σ_x k_x log(k_x/T) − k_x with Σ_x k_x = N. Σ k log k is
  Schur-convex, so the maximum is all counts in one bin. For the ST-PPP negative-binomial marginal with r < 1,
  concentration is also favoured; the authors report collapse **[verify analytically]**.
* (d) **Gradient-magnitude sharpness (MotionPriorCMax).** By the triangle inequality,
  ∫‖∇ Σ_j K_j‖ ≤ Σ_j ∫‖∇K_j‖ = N ∫‖∇K‖, with equality at coincidence (weak version: also at full scatter).
  Aligned *edges* are strictly below the bound, because kernel gradients cancel along the edge tangent.

### Theorem 1 (improperness of reference-frame objectives) [proved]
Let F = Ψ(Y_θ) with Ψ a concentration functional, and let Θ contain a contraction path with m = 1. Then for every dataset
whose true aligned configuration Y_θ* is not fully coincident:

    lim_{s→∞} F(θ(s)) = Ψ_max(N) > F(θ*).

The optimum is attained by a configuration that carries no information about the data.

*Proof.* Along the path every y_j → c, so continuity at coincidence gives F → Ψ_max. Strictness follows from the
equality condition in D4. ∎

For partial contraction (fraction m, the rest masked or kept), (a) gives lim ∫I² ≥ (mN)²/(4πσ²).

**Corollary 1.1 (headroom of edge scenes)** [derived+checked]. Take events spread uniformly along edges of total length
L (px), with IWE blur σ and normal position noise σ_n. Let σ_tot² = σ² + σ_n² + 1/12, the last term being pixel
quantisation. Then

    F(θ*) = ∫I² ≈ (N/L)² · L · ∫φ_{σ_tot}² = N² / (2√π σ_tot L),
    κ := Ψ_max / F(θ*) = L σ_tot / (2√π σ²).

Collapse also removes the noise blur σ_n, so **noisier data are more improper**. CMax is improper whenever κm² > 1.
Natural scenes have L ~ 10³–10⁴ px, so κ ~ 10²–10³.

Check (block D; L ≈ 168 px, σ_n = 0.5):

| blur σ (px) | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|
| κ predicted | 144 | 54.6 | 24.6 | 12.0 |
| G(collapse)/G* observed (h=400, finite path) | 133.5 | 48.6 | 22.7 | 10.6 |

**The collapse margin grows as 1/σ²·σ_tot**, so sharper kernels make CMax *more* improper.

### Theorem 2 (no reference distribution restores properness) [proved for the zoom paths]
Let F_ν = E_{t_ref∼ν} F_{t_ref} with F = ∫I², and mask events warped out of Ω. The zoom family
Φ_{s→t}(x) = c + (x − c) e^{h(t−s)} has two contraction paths:
* h → +∞ contracts events with t > t_ref, so m⁺(t_ref) = P(t > t_ref);
* h → −∞ contracts events with t < t_ref, so m⁻(t_ref) = P(t < t_ref).

Then lim_{h→±∞} F_ν / F_ν(θ*) ≥ κ E_ν[(m^±)²]. Since m⁺ + m⁻ = 1,

    E[(m⁺)²] + E[(m⁻)²] ≥ ½ E[(m⁺ + m⁻)²] = ½,  so max_± E_ν[(m^±)²] ≥ ¼.

**For every reference distribution ν, F_ν is improper whenever κ > 4.**
* Uniform ν (MotionPriorCMax): E[m²] = 1/3.
* Three references {0, T/2, T} (Shiba ECCV'22): 5/12.

Check (block A, κ = 53.6):

| scheme | predicted lim / G* | observed at h = 400 |
|---|---|---|
| single t_ref = 0 | 53.6 | 48.6 |
| 3 references | 22.4 | 20.0 |
| uniform (21 references) | 18.7 | 16.3 |

The remaining gap is the events within ~1/h of t_ref that have not finished contracting.

**Remark 2.1 (zero-sum / mean-transform constraints do not help).** Groupwise registration fixes drift by constraining
the mean transform to the identity. With t_ref = t̄, the zoom path has mean log-scale −h(t̄ − t_ref) = 0 for every h, so
it satisfies the constraint, yet half the events still collapse (m = ½, limit ≥ κ/4). The degeneracy is time-dependent
contraction, not a global gauge.

### Theorem 3 (regulariser dilemma) [(a),(b) proved; (c) checked numerically]
Let J_λ = F − λR, with R ≥ 0 vanishing on area-preserving warps.
* **(a)** If R ≤ R̄ along a contraction path (e.g. trimmed R_A), J_λ is improper for every λ < (κm² − 1) F(θ*) / R̄.
  The threshold is data-dependent: it scales with κ, i.e. with L, σ_n and 1/σ.
* **(b)** If the true motion is not area-preserving and ∇R(θ*) ≠ 0, the maximiser moves:
  θ_λ − θ* = λ [−∇²F(θ*)]^{-1}(∇F(θ*)/λ − ∇R(θ*)) + O(λ²). So R introduces an O(λ) bias even when ∇F(θ*) = 0.
* **(c) Checked**, block E plus `dilemma_scaling.py`. Setup: R = 2|h| (symmetric divergence penalty), mid-window
  reference (so Thm 4's bias is ≈ 0), true zoom h* = 1.

  | condition | λ_min (no collapse) | bias at λ_min | bias at the largest λ_min of all rows |
  |---|---|---|---|
  | T = 0.25, σ = 0.5 / 1 / 2 / 4 | 0.053 / 0.033 / 0.019 / 0.012 | 0 to −0.10 | 0 / −0.05 / −0.15 / −0.60 |
  | T = 0.05, any σ | ≈ 0 (no collapse within \|h\| ≤ 60) | ≈ −0.05 | **−0.65 to −1.00** |

  λ/4 collapses in every T = 0.25 scene tested.

  **Conclusion: no data-independent λ is both collapse-free and unbiased.** The λ that is safe for long windows and sharp
  kernels erases the true divergence in short windows (−100% bias).

  *Correction to my earlier claim:* I predicted the bias at λ_min would be about κ times Thm 4's bias. That is
  **refuted**: the bias at a well-tuned λ_min is small. The failure is the *data-dependence of λ_min*.

### Theorem 4 (first-order reference bias of single-reference CMax) [derived+checked]
Setup: true zoom h*, events uniform along edges, IWE energy F, perturbation δ = h* − h.
1. Warped points are ŷ_j = c + (y_j − c) e^{δ(t_j − t_ref)}.
2. Split e^{δ(t_j − t_ref)} = e^{δ(t̄ − t_ref)} · e^{δ(t_j − t̄)}:
   * a global scaling s = e^{δ(t̄ − t_ref)}; F ∝ s^{−γ} with γ = 1 for long edges and γ = 2 for points;
   * a relative part. A pair on one edge separates along the normal by δ(t_j − t_k)ρ_n, where ρ_n = (y − c)·n,
     so the pairwise kernel variance becomes 2σ_tot² + 2δ²σ_t²ρ_n².
3. Hence log F ≈ const − γδ(t̄ − t_ref) − ½ E log(1 + δ²σ_t²ρ_n²/σ_tot²).
4. Stationarity gives

       ĥ − h* = γ σ_tot² (t̄ − t_ref) / (σ_t² ρ²),   ρ² = E[((y − c)·n)²].

5. For t_ref = 0 and uniform t: **Δh = 6γσ_tot² / (T ρ²)**. This is zero at first order iff t_ref = t̄.
   The anchored score S has no reference, so no such term.

(An earlier version had 12 instead of 6: I had used σ_t² where the pairwise time-difference variance is 2σ_t².)

Check (local maximiser near the truth, h* = 1, 5 scenes × 3 window lengths):

| scene | ρ² | observed / predicted (γ = 1) at T = 0.02 / 0.05 / 0.10 |
|---|---|---|
| 0 | 251 | 1.86 / 1.79 / 1.81 |
| 1 | 279 | 0.89 / 0.81 / 0.83 |
| 2 | 306 | 1.43 / 1.46 / 1.38 |
| 3 | 380 | 1.61 / 1.54 / 1.27 |
| 4 | 248 | 1.35 / 1.32 / 1.33 |

* Scaling ∝ 1/T is confirmed: each scene's ratio is constant across T.
* The constant is scene-dependent, inside or close to γ ∈ [1, 2].
* Mid-window reference: |bias| ≤ 0.1 at T ≥ 0.05.
* S (global search): bias −0.019 / −0.009 / −0.003, within ±1 sd.
* For longer windows the *global* CMax(t_ref = 0) optimum is the collapse branch (G/G* rises without bound past h ≈ 8
  at T = 0.25), so a global search mixes Thm 1 and Thm 4.

*Practical reading.* CMax over-estimates divergence, i.e. under-estimates time-to-contact. The bias is largest for
short windows (low latency) and features near the focus of expansion (small ρ), which is exactly where TTC matters.
Example: σ_tot ≈ 1.15 px, ρ = 15 px, T = 20 ms gives Δh ≈ 1.6γ s⁻¹.

---

## 4. Positive results: the anchored score

### Theorem 5a (properness, smooth stationary regime) [sketch]
Assume:
* stationarity; p_t bounded with compact support and finite entropy; flows C¹ with bounded derivatives;
* Gaussian K_H with H = h²I, h → 0 and Nh² → ∞; w bounded and positive on [−τ, τ];
* floor ε_N → 0 slowly.

Then S_N(θ) → S(θ) in probability for each θ, where

    S(θ) = ∫ π(t) ∫ p_t log q^θ_t = −E_t[ H(p_t) + KL(p_t ‖ q^θ_t) ],
    q^θ_t = ∫ w(t − s) π(s) (Φ^θ_{s→t})_# p_s ds / ∫ w(t − s) π(s) ds.

At θ*, (Φ*_{s→t})_# p_s = p_t, so q* = p and S(θ*) = max. Equality holds iff q^θ_t = p_t for π-almost every t.

*Proof sketch.*
1. Given t_i, the transported points {Φ_{t_j→t_i}(x_j)}_{j≠i} are independent draws from the w-tilted mixture q^θ_{t_i}
   and are independent of x_i.
2. The leave-one-out KDE converges to q^θ_{t_i}. Uniform integrability of log(ε_N + ·) follows from the conditions used
   for Kullback–Leibler consistency of likelihood cross-validation (Chow–Geman–Wu 1983; Hall 1987).
3. Gibbs' inequality finishes the argument.

To write fully: the transported-sample version of the LOO-KDE consistency lemma (O7).

*Caveat (fixed floor).* For fixed ε > 0, argmax_q E_p log(ε + q) is a sharpened (p − c)₊, not p, so a fixed floor is
slightly improper. Either take ε_N → 0 or fit ε as a parameter.

### Theorem 5b (edge regime; non-stationary firing and noise allowed) [sketch]
Assume:
* p_t = (1 − β) p_t^sig + β u_Ω, where p_t^sig is supported on C² curves Γ_t = Φ*_{0→t}(Γ_0);
* the line density on Γ_t is bounded in [a, b], and may vary arbitrarily in t (non-stationary firing);
* for θ, define the misaligned set M_θ = {(x, t), x ∈ Γ_t : for w-almost every s ≠ t, Φ^θ_{s→t}(Γ_s) crosses Γ_t at x
  transversally with relative normal speed ≥ v₀ > 0}.

Then, as h → 0,

    S_h(θ*) − S_h(θ) ≥ (1 − β) P(M_θ) · log(1/h) − C(a, b, β, v₀, τ, w).

*Argument.*
* **At θ*:** every same-scene point transported from any s lands on Γ_{t_i}, so ρ̂ ≈ (line density) / (√(2π) h)
  and log ρ̂ = log(1/h) + O(1).
* **On M_θ:** transported curves pass within h of x_i only for s in a set of w-measure O(h/v₀), so ρ̂ = O(1).
* Non-stationary weights and noise enter only the O(1) constant.

Consequences:
* In the edge regime S is asymptotically strictly proper modulo symmetries (P(M_θ) = 0 iff the transported curves
  coincide with Γ_t, e.g. sliding along straight edges: the aperture problem, shared by every events-only method).
* The margin grows like log(1/h), while CMax's collapse margin grows like 1/σ² (Cor 1.1).
* **Interpretation:** S ≈ const − E_i log(smear width at event i, measured in the sensor frame at t_i). CMax measures
  sharpness in the t_ref frame, which the warp can shrink. S measures blur where the event actually is.

Check (block D, translation truth, zoom alternative h = 8), margin S(θ*) − S(θ) in nats:

| kernel width h (px) | 4 | 2 | 1 | 0.5 |
|---|---|---|---|---|
| margin | 1.49 | 2.09 | 2.64 | 3.18 |

That is ≈ +0.55 nats per halving, i.e. ≈ 0.8 log 2, matching P(M_θ) ≈ 0.8.

### Proposition 6 (boundedness; what "anchoring" buys) [proved]
* S_N(θ) ≤ log(ε + K_H(0)) for all θ, because the kernel is normalised in fixed coordinates.
* Boundedness is **not** sufficient: EMin with leave-one-out and a fixed aligned-frame bandwidth is bounded and still
  collapses (collapse toy: +2.36 nats at h = 16).
* The operative property is *anchoring*: S compares transported points against untransported observations, and a
  contraction cannot be applied at an event's own time (Φ_{t→t} = id).
* Note: reference-frame objectives also depend on θ only through transports. So the issue is **not** gauge
  invariance (an earlier note said it was; corrected). It is *where* the score is evaluated.

### Theorem 7 (exact Jacobian decomposition; the regularisers are surrogates) [derived+checked]
Fix any t_ref and set y_j = Φ_{t_j→t_ref}(x_j), A_i = DΦ_{t_i→t_ref}(x_i). Taylor expansion of Φ_{t_ref→t_i} around y_i,
plus the Gaussian identity K_H(A⁻¹δ) = |det A| · N(δ; 0, A H Aᵀ), gives

    log ρ̂_{−i}(x_i) = log ρ̃_{−i}(y_i; A_i H A_iᵀ) + log |det A_i| + O(h ‖D²Φ‖).

So S = (negative entropy of the warped events with event-adapted bandwidth) + **exact change-of-variables term, weight 1**.

Jacobi's formula, d/du log det DΦ_{t→u}(x) = (∇·f)(Φ_{t→u}(x), u), gives

    log|det A_i| = ∫_{t_i}^{t_ref} (∇·f)(Φ_{t_i→u}(x_i), u) du.

Hence both of Shiba's regularisers are first-order surrogates of this one term:
* R_A = mean det A_i matches log d ≈ d − 1;
* R_D (Sensors'22) and the AIS'23 time-derivative regulariser use div f at the event, which is the integrand at
  u = t_i, i.e. first order in (t_ref − t_i).

The exact version has weight 1, a log, integration along the trajectory, and no trimming.

Check (collapse toy, at h = 16): EMin +2.36 (collapses); EMin + exact log-det −5.72; S −2.78. Both of the latter peak at the truth.

**Algorithmic corollary:** implementing S needs **no Jacobian determinant at all**. The term appears automatically
because residuals are measured in sensor coordinates at the anchor's time.

### Theorem 8 (algorithm: MM / EM on a GP trajectory; banded structure) [proved / standard]
Write π_ij(θ) = w_ij K_H(x_i − Φ^θ_{t_j→t_i}(x_j)), and let the prior be log p(θ) (WNOA/WNOJ GP on SO(3)/SE(3), Markov
knots t_k).
1. **MM.** Jensen: log Σ_j π_ij ≥ Σ_j r_ij log(π_ij / r_ij), with r_ij ∝ π_ij(θ^k), tight at θ^k.
   The M-step is

       min_θ ½ Σ_ij r_ij ‖x_i − Φ^θ_{t_j→t_i}(x_j)‖²_{H⁻¹} − log p(θ),

   a weighted nonlinear least squares. One Gauss–Newton step with backtracking on the MAP objective is a generalised EM,
   so the MAP objective is monotone non-decreasing. Limit points are stationary under standard regularity (Wu 1983).
   The floor ε and noise events act as an outlier class.
2. **Locality.** For the SDE-based GP, the interpolated state at τ depends only on the two bracketing knots (Markov
   property). So a pair term touches ≤ 4 knots: {k(t_i), k(t_i) + 1, k(t_j), k(t_j) + 1}.
3. **Bandedness.** With |t_i − t_j| ≤ τ, the Gauss–Newton matrix is block-banded with half-bandwidth
   b = max #knots in any window of length τ, plus 1 (= ⌈τ/Δ⌉ + 1 for uniform spacing Δ).
   Cost per iteration is O(N k̄) pair assembly (fixed-radius hashing; k̄ = mean neighbours) plus O(K b² d³) banded Cholesky.
4. **Uncertainty.** The Laplace covariance comes from the final weighted least squares (sandwich form for misspecification).

### Corollaries for learning and evaluation [sketch]
* **Fisher consistency for self-supervised flow.** If a network predicts flows per window, the population risk
  E_D[−S(f_φ(D); D)] is minimised by the true flow (given realisability). CMax-type risks are minimised by collapse
  fields whenever those are realisable (Thm 1). Current pipelines therefore rely on inductive bias and regularisers.
* **A proper unsupervised metric.** Held-out S (or leave-one-out S) scores flow quality without ground truth and cannot
  be fooled by collapse. FWL and RSAT are ratios of reference-frame sharpness, so Thm 1 applies to them; Taming CMax
  reports them being fooled.

---

## 5. What is *not* claimed; known relatives

* **Anchored registration is a known principle elsewhere.** CPD and EM-ICP evaluate a moving GMM at fixed data points.
  Zhu'17 is the pairwise translation case. Groupwise registration uses constraints (Remark 2.1). The contribution here
  is the *theory for event alignment* (Thms 1–4, 7), the continuous-time algorithm (Thm 8), and evidence.
  It is not "inventing anchoring".
* **Brightness constancy, the aperture problem and flicker** are shared limitations with CMax.
* **Field-of-view entry.** Content entering the frame is not predicted by earlier events. This gives an O(1) bias at
  finite h, analogous to CMax's padding issue.
* **Timestamp-image losses** (Zhu'19, Taming's avg-timestamp IWE) are not concentration functionals. Their degeneracies
  differ (e.g. warping late events out of frame) and are **not analysed yet** (O9). Do not claim Thm 1 for them.
* **Efficiency.** On well-posed classes (rotation, translation), S has no proven accuracy advantage over CMax (O1).
  In the toy, the v-profiles have similar curvature.

---

## 5b. Corrections forced by the code experiments (atl/, Nov 2026)

**C1. The L1 gradient-magnitude focus does not collapse along contraction paths.**
This is MotionPriorCMax's focus term, 1/∫‖∇I‖ (Lemma 1(d), Thm 1).
* ∫‖∇I‖ is homogeneous of degree 1 in event mass. Concentrating mass gains nothing, and with partial contraction
  plus border masking it loses mass.
* In the collapse scan (`atl/scripts/scan_collapse.py`, exact and linear warps) it is the only baseline that is
  proper along the zoom path.
* Its degeneracy is elsewhere. The bound N∫‖∇K‖ is also attained by *scattered* configurations (no gradient
  cancellation), and that is reachable only by dense fields.
* Measured (F6, dense pixel trajectories with the official loss, KNN-32 and TV 0.003, from ground truth): the field
  drifts 2.9 px and is preferred over ground truth **on held-out events** by 0.28%. Without TV the preference is 1.1%
  (5.2 px drift). The random-reference gradient focus alone shows 8.3%.
* Thm 1 must be stated for variance / entropy / Poisson-type functionals (strictly superlinear in mass). The L1-grad
  case needs its own "scatter" theorem **[todo]**.

**C2. Finite-kernel (deconvolution) bias of the plain anchored score; fixed exactly.**
* At kernel width h the population score is E_p log(q ⊛ K_h). Gibbs makes the optimum q ⊛ K_h = p, not q = p.
* A dense field can approach that optimum by locally contracting transported events around edges, which undoes
  the kernel blur. Parametric warps cannot.
* Measured (F6): the plain exact score prefers a 2.1 px-drifted field on held-out events by 0.22%; the sliced
  version by 0.57%.
* **Fix (Prop. 5c, proved):** smooth the anchors with the same kernel, x̃_i = x_i + h·z_i with z_i ~ N(0, I), so
  the anchors are distributed as p ⊛ K_h. Then

      S_h → −H(p ⊛ K_h) − KL(p ⊛ K_h ‖ q ⊛ K_h),

  maximised iff q ⊛ K_h = p ⊛ K_h, i.e. q = p, because Gaussian convolution is injective (its Fourier transform is
  nowhere zero). The score is **proper at every h**, not only as h → 0.
* Pixel-quantised coordinates are handled by uniform dequantisation (U[−½, ½]², standard in density modelling).
* 1-D check: `tests/test_objectives.py::test_smoothed_score_has_no_deconvolution_bias`. The plain score peaks at
  contraction scale < 1; the smoothed score peaks at 1.
* Dense F6 rerun: see `atl/outputs/exp_dense_v3_fix.json`.

**C3. In-sample comparisons are not evidence of bias.** Any fitted estimator beats the truth in-sample. Bias claims
must use held-out events (a fresh DVS realisation of the same scene and motion). Every F6 number above is held-out.

## 6. Open mathematical questions (this phase)

| ID | Question | Why it matters |
|---|---|---|
| O1 | Asymptotic variance (sandwich) of S vs CMax on well-posed classes; 1-D edge model with clutter | Do we lose efficiency where CMax works? |
| O2 | Finite-N, finite-h bias of argmax S; is E S_N maximised at θ* for finite N? | Short windows, few events |
| O3 | Characterise the symmetry set {θ : P(M_θ) = 0} for rotation, Sim(2), homography, dense flow | Identifiability statements |
| O4 | Temporal support τ: information from lag-Δ pairs ∝ Δ²/σ_tot² vs GP interpolation error and bandwidth b ∝ τ/Δ; optimal τ | Continuous-time design rule |
| O5 | 6-DoF transport needs depth. Per-event free depth is a Neyman–Scott nuisance (each event can slide along its epipolar line) → finite-sample overfitting. Is S profiled over a smooth depth field proper? Marginalised under a GP depth prior? | The forward-motion case, where CMax fails worst |
| O6 | Polarity: pooled vs same-polarity pairs under motion reversal (non-stationary m_t) | Real data |
| O7 | Write the transported-sample LOO-KDE consistency lemma (Thm 5a) and the transversality lemma (Thm 5b) rigorously | Full proofs |
| O8 | Basin of attraction of S vs CMax; annealing in h | Practicality |
| O9 | Properness analysis of timestamp-image losses (Zhu'19, Taming) and of ST-PPP's NB marginal | Completes the negative side |
| O10 | Regulariser dilemma as a theorem: lower bound on λ_min(κ, T) and on the bias at fixed λ | Upgrade Thm 3(c) from numerics to a proof |

---

## 7. Plan for this phase (maths and gap only; no system building)

1. **Write full proofs:** Lemma 1(c) for the NB marginal; Thm 5a lemma (O7); Thm 5b transversality; Thm 7 with
   explicit remainder.
2. **Close the negative side:**
   * O9 (timestamp losses);
   * O10 (dilemma bound);
   * a classification table of the exact published loss of each paper in §1 against Thms 1–3.
3. **Literature verification still owed** (full text, not abstracts):
   * Zhu'17 exact objective; Gu'21 NB details; Nunes EMin;
   * Stoffregen FWL / RSAT definitions; Shiba AIS'23 exact regulariser;
   * Hamann'24 assignment step;
   * a targeted search for "leave-one-out" or "raw-anchored" event alignment and for groupwise registration with
     leave-one-out templates.
4. **Theory checks to add** (cheap, synthetic):
   * Thm 1/2 on homography and dense-flow classes;
   * Thm 4 with point features (γ = 2);
   * O1 by simulation;
   * O5: per-event-depth overfitting demonstration.
5. **Decision gate** (before any engineering):
   * proceed if O7/Thm 5a and O5 resolve favourably and the §1 table survives full-text reading;
   * otherwise narrow to the rotation + dense-flow paper (no 6-DoF claims).
6. **Then** (later phase) the staged real-data plan in `CT_DIRECTION.md` §4.

## Sources
* Gallego et al., CVPR'18 (arXiv 1804.01306)
* Nunes & Demiris, ECCV'20
* Gu, Learned-Miller, Sheldon, Gallego, Bideau, ICCV'21 (arXiv 2106.06887)
* Shiba, Aoki, Gallego, Sensors'22 (arXiv 2207.04007); ECCV'22 (arXiv 2207.10022); AIS'23 (doi 10.1002/aisy.202200251)
* Paredes-Vallés et al., ICCV'23 (arXiv 2303.05214)
* Hamann et al., ECCV'24 (arXiv 2407.10802)
* Zhu, Atanasov, Daniilidis, ICRA'17
* Chow, Geman, Wu, Ann. Stat. 1983; Hall, Ann. Stat. 1987 (likelihood cross-validation)
* Wu, Ann. Stat. 1983 (EM convergence)
* Myronenko & Song, TPAMI 2010 (CPD)
* Holmes et al. 2605.09073; Barfoot et al. 2601.03360 (GP-CT)
