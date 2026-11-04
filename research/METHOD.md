# Native Generative Pretraining for Event Streams

**A point-process likelihood over raw event-camera data, with an event-driven continuous-time backbone
and fused CUDA kernels.**

Working document, 2026-10-24. Everything in this file is implemented in `evpp/` (model, PyTorch
reference) and `evkit/` (C++/CUDA kernels, data path) unless it is marked **[planned]**. Section 12
separates what has been verified from what has not. **No result on real data exists yet: this is an
unvalidated research hypothesis, not a validated method.** Passing implementation tests shows the code
computes the stated equations; it is not evidence that the hypotheses are true.

*Revision 2 (2026-10-24) incorporates an external review: sharper hypotheses (§3.2), narrower novelty
claim (§3.4), explicit causal readout (§5.5), clock identifiability and gauge fixing (§5.3.1), corrected
dilation-gap convention (§6.4), per-shift robustness scope (§3.5), evidence table, stricter decision rule
and a staged plan (§10).*

---

## Contents

1. [The problem](#1-the-problem)
2. [Prior work: what exists, how it fails](#2-prior-work-what-exists-how-it-fails)
3. [What we propose and why it is different](#3-what-we-propose-and-why-it-is-different)
4. [Pipeline](#4-pipeline)
5. [Mathematical formulation](#5-mathematical-formulation)
6. [Outputs and label-free diagnostics](#6-outputs-and-label-free-diagnostics)
7. [Training and evaluation data](#7-training-and-evaluation-data)
8. [Systems: C++/CUDA implementation and the maths behind each optimisation](#8-systems-ccuda-implementation-and-the-maths-behind-each-optimisation)
9. [Complexity and memory](#9-complexity-and-memory)
10. [Experimental protocol, baselines, ablations](#10-experimental-protocol-baselines-ablations)
11. [Risks and how the method could fail](#11-risks-and-how-the-method-could-fail)
12. [Status: verified vs not](#12-status-verified-vs-not)
13. [References](#13-references)

---

## 1. The problem

An event camera does not produce images. Each pixel independently emits an event
$e = (t, x, y, p)$ when its log-brightness changes by a contrast threshold $C$, with microsecond
timestamps and polarity $p \in \{\mathrm{ON}, \mathrm{OFF}\}$. A recording is therefore a **marked
spatio-temporal point process** with $10^5$–$10^8$ events per second.

**Event vision has no self-supervised objective native to this data.** Every pretraining method we
found (Section 2.1) first turns events into something an image model can consume:

* fixed-window histograms or voxel grids,
* pseudo-grayscale video,
* or features distilled from an image foundation model.

Each of these choices throws information away:

* sub-window timing,
* the arrival process itself (the rate),
* the ordering of events,
* and anything an RGB teacher cannot see.

They also tie the model to a window length, a sensor resolution and the teacher.

**The question this project asks:** if we instead learn the probability distribution of the raw
event stream directly, using a point-process likelihood ("when and where does the next event fire?"),
do we get representations that are

1. better for downstream tasks,
2. more robust to real acquisition shifts (camera motion, lighting, sensor),
3. calibrated in a way we can check without labels,
4. and usable for things frame-based encoders cannot do: decide as soon as enough evidence arrives, score
   anomalies or noise per event, generate or forecast events, and infer sensor parameters?

---

## 2. Prior work: what exists, how it fails

Evidence tags:

* **[R]** we extracted the paper's own text (first ~100k characters; appendices may be missed).
* **[A]** abstract only.
* **[S]** search summaries only.

"(stated)" means the authors say it; "(inferred)" is a reviewer-style reading. Verify any claim you
build on.

### 2.1 Event-camera pretraining

| Method | What it does | How it fails / limitation | What we do instead |
|---|---|---|---|
| **MEM** (Klenk et al., WACV'24) **[S]** | Masked event modelling with a dVAE tokenizer on frame-like event images | Frame tokens; timing below the frame window is gone | Tokens are individual events with exact gaps |
| **ECDP** (2023) **[S]** | Contrastive event–image pretraining | Needs paired RGB; event images | No RGB needed |
| **TESPEC** (ICCV'25) **[R]** | MAE-style masking of 50 ms histograms with a recurrent Swin; target = pseudo-grayscale video | Reconstructs *frames*, so sub-50 ms timing is discarded. Expensive (135 h vs 45 h for ECDP, stated). Driving data only (stated). Small gains, no seeds (inferred) | The target is the event stream itself: an exact likelihood, no window |
| **GEP** (CVPR'26 Findings) **[R]** | Align an event encoder to frozen DINOv2, then a causal transformer predicts future latent embeddings of event/image tokens | **Not a likelihood**: MSE on teacher latents. Events are accumulated over a fixed Δt and normalised by the 99th percentile, so the rate is removed by construction. Bounded by the image teacher. Possible pretrain/test overlap (N-ImageNet, DSEC) (inferred) | No teacher; the arrival process is modelled, not normalised away |
| **Scaling Dense Event-Stream Pretraining** (CVPR 2026) **[A]** | Structure-aware distillation from vision foundation models | Same teacher dependence; abstract gives no scaling curves | Native objective that can be scaled without a teacher |
| **Physics-inspired pretraining** (2025) **[S]** | Masked modelling on temporal intensity-difference maps, then event–image contrast | Uses the event-generation idea only as a *data transform*, still frame-based | Uses the event-generation model as the *intensity* of the likelihood (variant V4) |

**Common failure:** all of them predict summaries (frames, counts, teacher features), not events. None
gives a probability of the data, so none can be checked for calibration, score individual events, or
generate streams.

### 2.2 Event-native backbones and tokenizers

| Method | What it does | Limitation | Relation to us |
|---|---|---|---|
| **EVA** (ICLR'26) **[R]** | One token per event, RWKV-6 linear attention, patch-wise; self-supervised "next representation" (event count / time surface) | Targets are summaries, not a likelihood. Uses inter-event deltas, not absolute time; no rate-shift study. Latency 14.7 ms per 8,192 events (inferred higher than prior A2S). Small detection gain over RVT-B (47.7 vs 47.2) | Closest backbone; our contribution is the **objective**, and our scan runs in O(1) per event with exact decay |
| **Event-SSM** (Schöne et al.) **[R]** | S5 SSM driven by inter-event gaps; associative scan | Test-set model selection (stated); 32k-event training slices vs full-length eval (stated); "no explicit event-generating mechanism" (stated) | We add exactly that mechanism: a generative head |
| **SSMs for event cameras** (Zubić et al., CVPR'24) **[R]** | S4/S5 with learnable timescales; change inference frequency without retraining | Aliasing at high frequency; bandlimit chosen empirically (stated); trained at one rate (inferred) | Our timescale is a learned per-stream **clock** inferred from the data, not a global knob |
| **Event2Vec** (ICML'26) **[R]** | Spatial MLP embedding of (x,y,p) plus temporal embedding of timestamp differences; fixed event budget | Trained end-to-end (no word2vec-style objective in the main text); a fixed event count discards rate; 3 classification sets | Evidence that event tokenisation is in scope at ICML; no generative objective |
| **Spiking Patches** (IROS'26) **[A]**, **Neural Events** (2026) **[A]** | Asynchronous tokenizers | Abstracts only; no likelihood | Compatible with our future micro-window tokens |
| **Graph / point methods** (AEGNN, EventMamba, ...) **[S]** | Spatio-temporal graphs or point clouds | Trail dense recurrent models on detection (Gen1: AsyNet 14.5 vs RED 40 mAP in one comparison) | We do not use graphs |

### 2.3 Neural temporal point processes (the ML side)

| Method | What it does | Limitation for event cameras | What we take / change |
|---|---|---|---|
| **Neural STPP** (Chen et al., ICLR'21) **[R]** | Continuous normalising flows for space–time intensity | 4–1,741 events per sequence; ODE solver restarted per event (stated); quadratic attention (inferred) | Same likelihood principle at 10³–10⁴× the sequence length via a linear scan |
| **Auto-STPP** (NeurIPS'23) **[S]** | Exact compensator for decomposable 3-D intensities | Restricted functional form | Our log-time quadrature handles a general intensity |
| **Mamba Hawkes** (2024) **[R]** | Mamba with Δ = inter-event gap; Hawkes-style intensity | Designed for Hawkes processes only (stated); discrete marks; average length up to ~2k | Generalised: spatial marks, continuous coordinates |
| **S2P2** (NeurIPS'25) **[R]** | Latent linear Hawkes layers: diagonal continuous-time SSM with impulses; MC compensator; parallel scan | **No space, no non-categorical marks, no high rates, no simultaneous events (all stated)** | Closest template. We add space, polarity, sub-cell marks, dequantised ties, a clock, deterministic quadrature and CUDA kernels |
| **Add-Thin** (NeurIPS'23), **EdiTPP** (ICLR'26), **EventFlow** (AISTATS'26) **[R/S]** | Non-autoregressive diffusion / edit / flow generation of event times | Times only, no marks or space; sampling cost | Possible future generative head; the add/thin noise process matches threshold/noise physics |
| **FIM-PP** (ICLR'26) **[R]** | Transformer pretrained on 14.4 M synthetic events from ~72 K Hawkes/Poisson processes; infers intensity in context | "Prior is not universal" (stated); bounded by marks and length seen in training (stated); no space | Template for in-context **sensor identification** (V4 + sensor encoder) |
| **FEP** (2026) **[R]** | Pretrained on 696 real event-log datasets | Types + timestamps only (stated); no vision, no high-rate streams | Shows likelihood pretraining transfers zero-shot, but never on sensor data |

**Common failure:** sequence lengths three to five orders of magnitude below event cameras, and no
spatial coordinates.

### 2.4 Neural-spike foundation models (closest structural analogue)

| Method | What it does | Limitation | Lesson |
|---|---|---|---|
| **POYO** (2023) **[R]** | One token per spike, unit embeddings, rotary time, Perceiver latents | Supervised decoding, not generative (stated); needs a self-supervised objective to scale (stated) | Per-event tokens scale |
| **Neuroformer** (ICLR'24) **[S]**, **NDT3** **[S]** | Autoregressive generative pretraining of spikes | NDT3: scaling autoregressive transformers "seems unlikely to resolve" **sensor variability** (stated) | The same wall exists for event cameras; we attack it with an explicit clock and sensor parameters |
| **Neural Pile + tokenizer evaluation** (2025/26) **[A]** | 476 B-token spike corpus; tokenizer benchmark | Reconstruction quality and downstream utility diverge | We must report both likelihood and downstream accuracy |

### 2.5 Physics, estimation and time-warp theory

| Work | Relevance | Limitation |
|---|---|---|
| Event-generation model ([Gallego et al. survey](https://rpg.ifi.uzh.ch/docs/EventVisionSurvey.pdf)) | Ideal pixel: event when $\lvert\Delta \log I\rvert = C$, so rate $\propto \lvert\partial_t \log I\rvert / C$ | Real sensors add threshold noise, refractory period, light-dependent shot noise |
| Complementary filter, asynchronous Kalman filter | Continuous-time intensity state updated per event | Linear, hand-designed, no learning or downstream use |
| Neural-ODE threshold identification (2026) | Online estimation of dynamics and threshold | System identification only |
| Ev-NeRF | Event-generation model as a rendering loss | Gaussian accumulation loss, not a point-process likelihood |
| Tallec & Ollivier (ICLR'18) | Gated RNNs are quasi-invariant to time warps; leaky ones only to uniform rescaling | Theory for 1-D sequences, never applied to events |
| N-ImageNet DiST, Ev-TTA, motion-equivariant TNT | Speed / illumination robustness | Frame-level representations, no generative model |

---

## 3. What we propose and why it is different

### 3.1 Thesis

Pretrain an event encoder by **maximising the exact likelihood of the raw event stream** under a marked
spatio-temporal point process whose history is summarised by an **event-driven continuous-time
state-space model with a learned clock**.

### 3.2 Hypotheses (each one falsifiable)

The central bet is that modelling the complete event-generating process gives more useful representations
than reconstructing masked events or predicting counts. **This is an empirical claim, not a consequence of
the architecture.** Predicting events well and learning transferable features are different achievements.

| | Claim | Evidence required |
|---|---|---|
| **H1 (representation)** | Event-level likelihood pretraining learns better representations than masked-event and next-count objectives at equal backbone, data and compute | Controlled frozen-probe and fine-tune comparisons, 3+ seeds, confidence intervals |
| **H2 (robustness, per shift)** | The advantage is larger under real acquisition shifts | Clean accuracy and the degradation on **each** shift type reported separately: camera speed/trajectory, brightness, sensor |
| **H3 (clock)** | The learned clock causally contributes to robustness to tempo changes, and only to those | Clock off / on / on + warp loss at fixed everything else; response of $\log\tau$ to controlled dilations; stability across seeds |
| **H4 (diagnostics)** | Calibration and temporal-gain diagnostics carry information beyond accuracy, e.g. they predict which shifts hurt | Correlation between label-free diagnostics and downstream degradation across shifts and models |

### 3.3 Contributions, if the hypotheses hold

1. A likelihood-based (generative) pretraining objective on raw event-camera streams at sensor scale (no prior instance found; priority not yet verified, §3.4).
2. A point-process head that scales to $10^6$+ events per window: impulse-driven diagonal SSM, an
   arrival-time readout, deterministic log-time quadrature, and dequantised ties.
3. Label-free diagnostics for event models: time-rescaling calibration, temporal information gain, and a
   dilation-equivariance gap.
4. A systems contribution: fused CUDA scan and readout kernels with analytic backward passes, plus a
   decoder pipeline for EVT2/EVT3/DAT at hundreds of Mevents/s.

### 3.4 What is and is not new

Already published, so **not** claimed as ours:

* SSM-based event processing with learnable timescales and frequency robustness: Zubić et al., CVPR 2024.
* Self-supervised event pretraining that targets long-range temporal information: TESPEC, ICCV 2025.
* Masked event modelling: MEM, WACV 2024.
* Large-scale event pretraining with transfer: GEP (CVPR 2026 Findings) and *Scaling Dense Event-Stream
  Pretraining* (CVPR 2026).

What is potentially distinctive is the **combination**, none of which we found published:

1. A normalised, event-by-event likelihood objective over time **and** space and polarity, at
   sensor-scale sequence lengths.
2. Explicit continuous-time state evolution to the arrival time, used to predict marks.
3. Controlled tests of whether the model actually uses temporal structure (temporal gain, dilation gap).
4. A systematic link between likelihood calibration and downstream robustness.

**Priority is not established.** The references here are a starting point; an exhaustive search
(`READING_PLAN.md`, final section) must precede any novelty claim in a submission.

| Axis | Prior event pretraining | Prior TPP models | This work |
|---|---|---|---|
| Training signal | Frames, counts, teacher features | Likelihood | **Likelihood** |
| Unit of data | Window / frame | Event | **Event** |
| Space | Yes (pixels) | No | **Yes (patch, sub-cell, polarity)** |
| Sequence length | Window-bounded | $10^2$–$10^3$ | **$10^3$–$10^6$+** |
| Needs RGB / teacher | Usually | No | **No** |
| Calibration checkable | No | Yes | **Yes** |
| Generates events | No | Yes (times only) | **Yes (time + space + polarity)** |

### 3.5 Scope of the robustness claim

The mechanisms address different shifts, and none of them guarantees semantic invariance:

| Shift | What changes in the stream | What in the model could help | What it cannot do |
|---|---|---|---|
| Speed (same path, faster) | Gaps scale roughly uniformly | Latent clock (§5.3), warp loss | — |
| Camera trajectory | *Which* structures fire, and when | Likelihood must explain arrival patterns | The clock does not undo a different path |
| Brightness | Number and distribution of events, noise rate | V4 noise / threshold terms, likelihood of noise | The clock does nothing |
| Sensor | Resolution, threshold, noise, refractoriness | Normalised quantiser, V4 sensor parameters | No guarantee without training on that sensor class |

Results must be reported per row. Improvements are not attributed to temporal modelling unless the clock
ablation (H3) shows it.

---

## 4. Pipeline

```
 ┌──────────────── data path (C++, evkit) ─────────────────┐
 raw .raw (EVT2/EVT3) / .dat / .h5
   │  parallel decoders: chunk summaries → state prefix → in-place decode → compaction
   ▼
 time-sorted events (t, x, y, p)  ──►  EVB store (mmap, 1 ms time index)   or decode on the fly
   │  random windows of L consecutive events
   ▼
 tokenizer (CPU or GPU, bit-identical): patch k, sub-cell c, polarity p, (u, v), dequantised Δt
 └──────────────────────────────────────────────────────────┘
   ▼
 ┌──────────────── model (PyTorch + evkit CUDA ops) ───────┐
 token embedding x_i
   ▼
 latent clock τ_i  →  effective gaps Δt̃_i = τ_{i-1} Δt_i
   ▼
 L × event-driven SSM layers (fused CUDA scan, fwd + bwd)
   ▼
 s_i (per-event state) + decaying readout ŷ_i(t) (fused CUDA readout)
   ▼
 heads: time (when) → patch → polarity → sub-cell (where / what)
   ▼
 loss: −log p(e_i | history), nats per event   (+ optional warp-consistency loss)
 └──────────────────────────────────────────────────────────┘
   ▼
 outputs: encoder s_i · bits/event · calibration (KS) · temporal gain · samples ·
          anytime class posterior · sensor parameters θ̂
```

---

## 5. Mathematical formulation

### 5.1 Data and tokenisation

A window is a time-ordered sequence $e_1,\dots,e_n$ with $e_i = (t_i, x_i, y_i, p_i)$, $t_i$ in µs,
$p_i\in\{0,1\}$, on a $W\times H$ sensor.

Coordinates are normalised so one model can read any sensor. The quantiser uses a $G\times G$ patch grid
with $Q\times Q$ sub-cells per patch; $K=G^2$ patches, $S=Q^2$ sub-cells:

$$u_i=\frac{x_i+\tfrac12}{W},\qquad v_i=\frac{y_i+\tfrac12}{H},\qquad
\mathrm{cu}=\lfloor u_i GQ\rfloor,\ \ \mathrm{cv}=\lfloor v_i GQ\rfloor$$

$$k_i=\left\lfloor\tfrac{\mathrm{cv}}{Q}\right\rfloor G+\left\lfloor\tfrac{\mathrm{cu}}{Q}\right\rfloor,
\qquad c_i=(\mathrm{cv}\bmod Q)\,Q+(\mathrm{cu}\bmod Q)$$

**Dequantised gaps.** Timestamps are integers (µs), and many events share a timestamp. Adding uniform
noise inside the quantum makes the time density proper:

$$\Delta t_i=\big((t_i-t_{i-1})+U_i\big)\cdot 10^{-6}\ \mathrm{s},\qquad U_i\sim\mathcal U(0,1)$$

The continuous NLL is an upper bound on the discrete one:
$-\log P_{\text{disc}} \le \text{NLL}_{\text{time}} - \log q$ with $q = 1\,\mu s$.

### 5.2 Token embedding

$$\mathbf x_i=E_{\text{patch}}[k_i]+E_{\text{sub}}[c_i]+E_{\text{pol}}[p_i]+W_{\text{pos}}\,\mathrm{FF}(u_i,v_i)+\mathrm{MLP}\big(\phi(\Delta t_i)\big)$$

$$\mathrm{FF}(u,v)=\big[\sin 2^j\pi u,\ \cos 2^j\pi u,\ \sin 2^j\pi v,\ \cos 2^j\pi v\big]_{j<F},\qquad
\phi(\Delta t)=\Big[\tfrac{\log\Delta t+9}{4},\ \tanh\tfrac{\log\Delta t+9}{4}\Big]$$

### 5.3 Latent clock

The clock is a causal, strictly positive tempo. It is a count-based EMA (independent of the time axis),
and its head is zero-initialised so $\tau=1$ at the start of training:

$$\mathbf c_i=\beta\,\mathbf c_{i-1}+(1-\beta)\tanh(W_c\mathbf x_i),\qquad
\log\tau_i=\mathrm{clip}\big(\mathbf w^\top\mathbf c_i,-4,4\big)$$

$$\widetilde{\Delta t}_i=\tau_{i-1}\,\Delta t_i$$

The clock after event $i-1$ scales the gap *before* event $i$, so nothing about event $i$ leaks into its
own prediction. This is tested.

Motivation (Tallec & Ollivier): a time warp $t\mapsto w(t)$ can be absorbed by a multiplicative factor on
the elapsed time, so a learned $\tau$ lets fixed dynamics run faster or slower per stream. For a stream
recorded $f$ times faster, gaps become $\Delta t_i/f$; a clock with $\tau'=f\tau$ leaves the effective gaps
unchanged: $(\Delta t_i/f)(f\tau_{i-1})=\tau_{i-1}\Delta t_i$.

#### 5.3.1 Identifiability of the clock

Being *able* to learn a clock does not mean the model *will* learn a meaningful one. The likelihood may be
optimised by other parameters, leaving $\tau$ unused or absorbing unrelated changes.

There is also an exact symmetry. Rates and clock only appear as the product
$r_{dn}\,\tau_{i-1}\,\Delta t_i$, so

$$(r_{dn},\ \tau_i)\ \longmapsto\ (c\,r_{dn},\ \tau_i/c),\qquad c>0$$

leaves every state, output and likelihood unchanged. Initialisation ($\tau=1$) and clipping pick one
solution but do not make $\tau$ interpretable.

* **Gauge fixing (implemented, `clock_gauge`).** Add
  $\gamma_\tau\big(\overline{\log\tau}\big)^2$ with $\overline{\log\tau}$ the batch mean, which pins the
  mean log-tempo to 0. $\tau$ then means *tempo relative to the training average*, never an absolute
  physical speed.
* **What we may claim.** Only what H3's experiments show: the response of $\log\tau$ to controlled
  dilations, its stability across seeds, and the accuracy effect of turning it off. No claim of physical
  tempo estimation without those.

### 5.4 Event-driven SSM layer

Per layer $\ell$, channel $d\in[D]$, state $n\in[N]$, with rates $r_{dn}=\exp(\log r_{dn})>0$:

$$[\mathbf v_i;\mathbf g_i]=W_{\text{in}}\mathbf x^{(\ell-1)}_i$$

$$h_{i,dn}=a_{i,dn}\,h_{i-1,dn}+B_{dn}\,v_{i,d},\qquad a_{i,dn}=\exp\!\big(-r_{dn}\,\widetilde{\Delta t}_i\big)$$

$$y_{i,d}=\sum_n C_{dn}h_{i,dn}+D_d\,v_{i,d}$$

$$\mathbf x^{(\ell)}_i=\mathrm{LN}\Big(\mathbf x^{(\ell-1)}_i+W_{\text{out}}\big(\mathrm{GELU}(\mathbf y_i)\odot\sigma(\mathbf g_i)\big)\Big),
\qquad \text{then a residual FFN with LN}$$

* **Exact between events, for the assumed dynamics.** In the impulse form the event adds $B v_i$ and the
  state then decays exactly, so there is no discretisation error and no window. The model is still a
  learned approximation: it is not a physical simulator of the camera.
* **Initialisation.** Rates start log-spaced between $1/t_{\max}$ and $1/t_{\min}$ (default 1 s to
  0.1 ms).
* **Notation.** The last layer's output is $\mathbf s_i$.

### 5.5 Continuous-time readout

After event $i$, the last layer's state keeps decaying. For its first $d_{\text{int}}$ channels:

$$\hat y_{i,d}(t)=\sum_n C_{dn}\,e^{-r_{dn}\tau_i (t-t_i)}\,h_{i,dn},\qquad t>t_i$$

This gives the model a representation at **every instant**, not only at events. It is what the
time-dependent heads use.

**Causal form used to predict event $i$.** The heads use the *previous* event's state and the *previous*
clock, evolved over the observed gap:

$$\hat y_{i-1,d}(t_i)=\sum_n C_{dn}\,e^{-r_{dn}\,\tau_{i-1}\,\Delta t_i}\,h_{i-1,dn}
=\sum_n C_{dn}\,e^{-r_{dn}\widetilde{\Delta t}_i}\,h_{i-1,dn}$$

using the same effective-time convention as the state transition. Nothing derived from event $i$'s own
token enters its prediction except its gap $\Delta t_i$, which is the quantity the time head scores first.
In code: `H_prev` = $h_{i-1}$, `dt_eff` = $\tau_{i-1}\Delta t_i$ (`evpp/models/evpp.py`). This is checked by
`test_context_for_event_i_ignores_event_i_own_marks`: changing event $j$'s patch, sub-cell, polarity and
coordinates leaves $s_{j-1}$, $h_{j-1}$, $\tau_{j-1}$ and $\widetilde{\Delta t}_j$ unchanged, for all variants.

### 5.6 Likelihood factorisation (the objective)

$$p(e_1,\dots,e_n)=\prod_{i=1}^n\underbrace{p(\Delta t_i\mid\mathcal H_{i-1})}_{\text{when}}\;
\underbrace{p(k_i\mid\mathbf z_i)}_{\text{where (coarse)}}\;
\underbrace{p(p_i\mid k_i,\mathbf z_i)}_{\text{polarity}}\;
\underbrace{p(c_i\mid k_i,p_i,\mathbf z_i)}_{\text{where (fine)}}$$

The context combines the history state with the state **evolved to the arrival time**:

$$\mathbf z_i=\mathrm{MLP}\big[\mathbf s_{i-1},\ \hat{\mathbf y}_{i-1}(t_i),\ \phi(\Delta t_i)\big]$$

The mark heads, each a categorical:

$$p(k\mid\mathbf z)=\mathrm{softmax}(W_k\mathbf z),\qquad
p(p\mid k,\mathbf z)=\mathrm{softmax}\big(W_p(\mathbf z+E^k_k)\big),\qquad
p(c\mid k,p,\mathbf z)=\mathrm{softmax}\big(W_c(\mathbf z+E^k_k+E^p_p)\big)$$

**Reference measures.** Time is measured in seconds (Lebesgue); marks use the counting measure on
(patch, polarity, sub-cell). This makes all variants and the Poisson baseline directly comparable in
nats/event.

### 5.7 Time heads: three variants

**V2, intensity-free (main).** A mixture of log-normals:

$$p(\Delta t\mid\mathbf s)=\sum_{m=1}^M w_m\,\frac{1}{\Delta t\,\sigma_m\sqrt{2\pi}}
\exp\!\Big(-\frac{(\log\Delta t-\mu_m)^2}{2\sigma_m^2}\Big),\qquad (w,\mu,\log\sigma)=\mathrm{MLP}(\mathbf s_{i-1})$$

It gives a closed-form density, closed-form CDF (used for calibration) and direct sampling.

**V3, continuous intensity.** A ground intensity read from the decaying state, plus its compensator:

$$\lambda^*(t)=\exp\big(g(\hat{\mathbf y}_{i-1}(t),\mathbf s_{i-1})\big),\qquad
-\log p(\Delta t_i)=-\log\lambda^*(t_i)+\int_{t_{i-1}}^{t_i}\lambda^*(t)\,dt$$

The compensator uses **log-time Gauss–Legendre quadrature**. A gap can span six decades; log-spaced
nodes resolve both the post-event transient and the long tail. It is deterministic and
differentiable. With $\ell=\min(10^{-6}\,\mathrm s,\Delta t)$ and nodes/weights $(x_s,w_s)$ on $[0,1]$:

$$\int_0^{\Delta t}\!\lambda(\tau)d\tau\approx\ell\,\lambda(\ell/2)+\big(\log\Delta t-\log\ell\big)\sum_s w_s\,\tau_s\,\lambda(\tau_s),
\qquad \tau_s=\exp\!\big(\log\ell+(\log\Delta t-\log\ell)x_s\big)$$

The change of variables $\tau=e^{v}$, $d\tau=\tau\,dv$ is where the $\tau_s$ factor comes from. The
rule is exact to 0.2% for exponential intensities over $10^{-7}$–$0.4$ s (tested).

**V4, physics-structured.** The event-generation model sets the intensities, with sensor parameters
$\theta=(\ln C,\ln r_n)$:

$$\lambda_{k,\mathrm{ON}}(t)=\frac{\mathrm{softplus}(+\sigma_k(t))}{C}+\frac{r_n}{2K},\qquad
\lambda_{k,\mathrm{OFF}}(t)=\frac{\mathrm{softplus}(-\sigma_k(t))}{C}+\frac{r_n}{2K}$$

$$\sigma_k(t)=\big[\mathrm{MLP}(\hat{\mathbf y}_{i-1}(t))\big]_k\approx\partial_t\log I\ \text{in patch }k,\qquad
\Lambda(t)=\sum_{k,p}\lambda_{k,p}(t)$$

$$-\log p(\Delta t_i,k_i,p_i)=-\log\lambda_{k_i,p_i}(t_i)+\int_{t_{i-1}}^{t_i}\Lambda(t)\,dt$$

$\theta$ can be learned globally, given (oracle), or **inferred in context** by a sensor encoder that
pools backbone states over the first $n_{\text{ctx}}$ events of the window. Those events are excluded
from the loss so they cannot leak. On simulated sensors the encoder is supervised by
$\lVert\hat\theta-\theta\rVert^2$.

*Identifiability caveat:* without that supervision, $C$ is only defined up to the scale of $\sigma_k$.

### 5.8 Training objective

$$\mathcal L_{\text{NLL}}=-\frac1n\sum_{i=1}^n\log p(e_i\mid\mathcal H_{i-1})\quad[\text{nats/event}]$$

Optional self-supervised warp consistency. Feed the same window with all gaps divided by $f$; the clock
must speed up by $f$ and pooled content must not move:

$$\mathcal L_{\text{warp}}=\frac1n\sum_i\big(\log\tau'_i-\log\tau_i-\log f\big)^2+\lVert\bar{\mathbf s}'-\mathrm{sg}(\bar{\mathbf s})\rVert^2,
\qquad f\sim\mathrm{LogUniform}(\tfrac12,2)$$

$$\mathcal L=\mathcal L_{\text{NLL}}+\beta\,\mathcal L_{\text{warp}}\ \ (+\ \gamma\,\lVert\hat\theta-\theta\rVert^2\ \text{for V4 on simulated data})$$

### 5.9 Baselines on the same backbone

These exist so that H1 is a clean comparison.

* **Masked event model.** Hide 50% of event identities (keep their gaps). A forward and a time-reversed
  backbone pass give bidirectional context, and the model predicts $(k,p,c)$ of hidden events with
  cross-entropy. *Fairness:* bidirectional context is its native inference mode, so its probe features use
  both passes (an advantage over our causal model on offline tasks). We also report a causal-only probe
  for every model, so streaming comparisons are like for like.
* **Next-count prediction (EVA-style).** At each event, predict the per-patch counts in the next
  $h=5$ ms with a Poisson NLL:
  $$\sum_k\big(\hat\mu_k-m_k\log\hat\mu_k\big)$$
* **Patch-Poisson (V1).** A fixed-bin count model reported as a point-process density, so it is
  comparable in nats/event:
  $$\sum_c(\lambda_c-n_c\log\lambda_c)+n\log(wS)$$
  This tests whether event-level timing matters.
* **Homogeneous Poisson.** Per-window MLE $\hat\Lambda=n/T$ with uniform marks:
  $$-\log\hat\Lambda+1+\log(2KS)\ \text{nats/event}$$
  This is the "no structure" reference.

---

## 6. Outputs and label-free diagnostics

### 6.1 Outputs

1. **Encoder.** Per-event states $\mathbf s_i$, plus pooled window features
   $[\bar{\mathbf s},\ \overline{\log\tau},\ \overline{\log(1/\Delta t)}]$.
2. **Bits/event.** $\mathcal L_{\text{NLL}}/\ln 2$, reported against the Poisson baseline.
3. **Samples.** Generation by ancestral sampling: $\Delta t\to$ evolve the state $\to k\to p\to c$.
4. **Anytime classification.** With a class embedding $\mathbf e_c$ added to the tokens and heads:
   $$\log P(c\mid e_{1:i})=\log\pi_c+\sum_{j\le i}\log p(e_j\mid\mathcal H_{j-1},c)-\log\sum_{c'}(\cdot)$$
   Stop at the first $i$ where $\max_c P(c\mid e_{1:i})\ge 1-\alpha$. Latency is then measured in events
   or seconds.
5. **Sensor parameters** $\hat\theta$ (V4).
6. **Per-event log-likelihoods.** These give anomaly and noise scores for free.

### 6.2 Time-rescaling calibration

By the time-rescaling theorem, if $\lambda^*$ is the true conditional intensity then the compensator
increments are i.i.d. unit exponentials:

$$\xi_i=\int_{t_{i-1}}^{t_i}\lambda^*(t)\,dt\ \sim\ \mathrm{Exp}(1)$$

Equivalently, for V2 the PIT values $u_i=F(\Delta t_i\mid\mathbf s_{i-1})\sim\mathcal U(0,1)$.

We report the Kolmogorov–Smirnov statistic
$D=\sup_x\lvert\hat F(x)-F(x)\rvert$, with p-value
$2\sum_{k\ge1}(-1)^{k-1}e^{-2k^2\lambda^2}$ where $\lambda=(\sqrt n+0.12+0.11/\sqrt n)D$, plus QQ
quantiles. It is computed under each real shift and needs **no labels**.

### 6.3 Temporal information gain

$$G_{\text{perturb}}=\frac{\mathrm{NLL}(\text{perturbed history})-\mathrm{NLL}(\text{true history})}{\ln2}\ \text{bits/event},
\qquad\text{perturb}\in\{\text{reverse},\text{shuffle order},\text{shuffle polarity}\}$$

$G\approx 0$ means the data or model does not use time. This turns "benchmarks barely test time" into a
number per dataset.

### 6.4 Dilation-equivariance gap

**Convention.** $\mathcal D_f$ divides every gap by $f$ (the stream runs $f$ times faster):
$\Delta t_i\mapsto\Delta t_i/f$, marks unchanged. The density of the transformed gaps picks up a Jacobian
$f$ per event, so an exactly speed-equivariant model satisfies

$$\log p(\mathcal D_f x)=\log p(x)+n\log f
\quad\Longleftrightarrow\quad
\frac{\mathrm{NLL}_{\text{tot}}(\mathcal D_f x)-\mathrm{NLL}_{\text{tot}}(x)}{n}=-\log f .$$

The per-event residual is therefore

$$\boxed{\ \delta_f=\frac{\mathrm{NLL}_{\text{tot}}(\mathcal D_f x)-\mathrm{NLL}_{\text{tot}}(x)}{n}+\log f\ }$$

and $\delta_f=0$ for an equivariant model. (Revision 1 of this document wrote the NLL difference without
the $1/n$ and without defining $\mathcal D_f$; the code always used per-event means and this convention.)

**Tests** (`tests/test_core.py`):

* The per-window MLE Poisson baseline is exactly equivariant: $\delta_f=0$.
* A *fixed*-rate Poisson model with rate $\lambda$ has
  $\delta_f=\lambda\,\overline{\Delta t}\,(1/f-1)+\log f$, which the implementation reproduces
  analytically. This test also caught a bug: the diagnostic crashed on non-neural baselines.

---

## 7. Training and evaluation data

Details and links are in [`DATASETS.md`](DATASETS.md).

* **Pretraining (no labels), about 73 h of real recordings across three sensor generations:**

  | Dataset | Hours | Sensor |
  |---|---:|---|
  | 1Mpx automotive | 14–15 | Gen4, 1280×720 |
  | eTraM | ~10 | EVK4 HD, static traffic camera |
  | M3ED | ~3.4 | stereo 1280×720; car, legged robot, drone |
  | EventKitchen | 5.5 | stereo Gen4 |
  | Gen1 | 39 | ATIS, 304×240 |
  | DSEC | ~1 | Gen3.1, 640×480 |

  Gen1 and DSEC can serve as held-out sensors. The model is trained on random windows of $L$
  consecutive events: 4k–16k at first; longer streams are run with carried state.
* **Evaluation (labels):**
  * N-ImageNet: ~1.3 M training recordings, 1000 classes, **plus 9 validation sets with real
    camera-trajectory (5) and brightness (4) changes**.
  * HARDVS: 107,646 recordings, 300 activities.
  * DailyDVS-200: >22 k recordings.
  * Optionally EvDET200K or Gen1/1Mpx detection.
* DVS128 Gesture and N-Caltech101 are kept only as a sanity table.

---

## 8. Systems: C++/CUDA implementation and the maths behind each optimisation

### 8.1 Event-driven scan: fused CUDA kernel (`evkit/cuda/ssm_scan.cu`)

**Problem.** The PyTorch reference materialises $a,\,Bv$ and $h$ as $(B,L,D,N)$ tensors per layer and
keeps them for autograd. At $B{=}8$, $L{=}4096$, $D{=}256$, $N{=}16$ that is $1.3\times10^8$ floats
(537 MB) **per tensor per layer**.

**Thread mapping.** One thread owns one recurrence $(b,d,n)$. A block has 256 threads, which covers
$DT=256/N$ channels × $N$ states of one batch row.

* The $N$ states of a channel are $N$ adjacent lanes, so the output sum is a butterfly reduction in
  registers:
  $$y_{i,d}=\sum_n C_{dn}h_{i,dn}\quad\Longrightarrow\quad\texttt{val += \_\_shfl\_xor\_sync(mask, val, o)},\ o=N/2,\dots,1$$
* There are $\log_2 N$ shuffles per event, with no shared memory and no atomics.

**Time tiling.** Events are processed in tiles of $K_c=32$. Each tile of $v$ (and later $y$, $g_y$,
$g_v$) is staged through shared memory with coalesced loads and stores; the inner loop over the tile
runs entirely in registers. Inputs read once use `ld.global.nc.L1::no_allocate`, so they don't evict
reused data from L1.

**Inline PTX for the decay.** We need $a=e^{-r\widetilde{\Delta t}}=2^{-r\log_2 e\cdot\widetilde{\Delta t}}$.
The exponent is precomputed per thread as $\nu=-r\log_2 e$, and the decay is one SFU instruction:

```
ex2.approx.ftz.f32  a, (ν · Δt̃)          ; max rel. error ≈ 2^-22, argument always ≤ 0
```

`fast_exp=false` switches to `exp2f` for validation.

**Selective state saving.** Per-event states are written only for the first $n_{\text{save}}=d_{\text{int}}$
channels of the last layer (16 of 256 by default) — the only states the heads read. Writes are fully
coalesced because $(d,n)$ is contiguous over the thread index.

**Two schedules.** Let $C=\lceil L/K_c\rceil$ chunks, and let chunk $c$ cover events $[s_c,e_c)$.

* **Sequential** (used when $B\cdot\lceil D/DT\rceil\ge 2\times$ the number of SMs). Each block walks all
  chunks in order, carrying $h$ in a register. One pass over the data; the state at each chunk start is
  written as a checkpoint for the backward pass.
* **Chunk-parallel** (small batch, very long stream), in three phases:
  1. *Local scan* from zero state. In parallel over chunks, for each chunk $c$:
     $$S_c=\sum_{i\in c}\Big(\prod_{j\in c,\,j>i}a_j\Big)B v_i,\qquad \Sigma_c=\sum_{i\in c}\widetilde{\Delta t}_i$$
  2. *Carry*, sequential over chunks only, $O(BDN\cdot C)$. It uses the fact that a product of
     exponentials is an exponential of a sum:
     $$\prod_{i\in c}a_i=e^{-r\Sigma_c},\qquad h^{\text{start}}_{c+1}=e^{-r\Sigma_c}\,h^{\text{start}}_c+S_c,\qquad h^{\text{start}}_0=h_0$$
  3. *Finalise*: rescan each chunk from $h^{\text{start}}_c$ to produce $y$ and the saved states.

**Backward pass (analytic, hand-written).** Let $q_{i,dn}=C_{dn}\,\bar y_{i,d}+\bar H_{i,dn}$, where the
bars are upstream gradients. The adjoint runs in reverse:

$$\lambda_i=q_i+a_{i+1}\lambda_{i+1},\qquad \lambda_{L-1}\ \text{also receives}\ \bar h_{\text{last}}\ (a_L:=1)$$

$$\bar B_{dn}=\sum_i\lambda_{i}v_{i,d},\qquad \bar C_{dn}=\sum_i\bar y_{i,d}h_{i,dn},\qquad
\bar D_d=\sum_i\bar y_{i,d}v_{i,d},\qquad \bar v_{i,d}=\sum_n\lambda_{i,dn}B_{dn}+D_d\bar y_{i,d}$$

$$\bar a_i=\lambda_i h_{i-1}\ \Rightarrow\ \bar r_{dn}=-\sum_i\widetilde{\Delta t}_i\,a_i\,\bar a_i,\qquad
\overline{\Delta t}_i=-\sum_{d,n}r_{dn}\,a_i\,\bar a_i,\qquad \bar h_0=a_0\lambda_0$$

* **Recomputation instead of storage.** Each chunk recomputes $h$ from its checkpoint into shared memory
  ($K_c\times256$ floats = 32 KB), then runs the adjoint in reverse. Storage is
  $O(BCDN)=O(BLDN/32)$ instead of $O(BLDN)$.
* **Chunk-parallel backward** mirrors the forward. Phase 1 computes local adjoints $\mu_{s_c}$ with zero
  right carry. Phase 2 carries right to left using the shifted sum
  $\Sigma'_c=\sum_{j=s_c+1}^{e_c}\widetilde{\Delta t}_j$ (with $\widetilde{\Delta t}_L:=0$):
  $$\lambda_{s_c}=\mu_{s_c}+e^{-r\Sigma'_c}\lambda_{e_c}$$
  Phase 3 recomputes and accumulates gradients.
* **Deterministic reductions.** $\bar B,\bar C,\bar r,\bar D$ accumulate in registers per thread and are
  written as fixed-order partial sums over (batch, chunk-group), then reduced by a column-sum kernel.
  $\overline{\Delta t}$ (a sum over all $d,n$) is a full-warp shuffle reduction, then a fixed-order sum
  over warps and channel tiles. There are no float atomics, so gradients are bit-reproducible.
* **Race found and fixed.** In the backward kernel, `dt_s[len]` (the first gap of the next chunk) was
  written by two threads when the last chunk was short. It now has exactly one writer per slot.

**`ptxas -v` on sm_80** (clang 18 + ptxas 12.9):

| Kernel | Registers | Shared memory | Spills |
|---|---:|---:|---:|
| Forward | 31–38 | 2–8 KB | 0 |
| Chunk-local / carry | 25–40 | ≤ 4 KB | 0 |
| Backward | 60–80 | 37–46 KB | 0 |

The backward kernel fits about three to four 256-thread blocks per A100 SM. $N\in\{8,16,32\}$ are
instantiated as templates; $N\ge 8$ keeps the static shared memory under 48 KB.

### 8.2 Fused decayed readout (`evkit/cuda/readout.cu`)

Forward, for item $(b,l)$, query $s$ and channel $d$:

$$Y_{s,d}=\sum_m C_{dm}\,e^{-r_{dm}\tau_s}H_{dm}$$

Gradients:

$$\bar H_{dm}=\sum_s\bar Y_{s,d}C_{dm}e_{s,dm},\qquad
\bar\tau_s=-\sum_{d,m}\bar Y_{s,d}C_{dm}H_{dm}r_{dm}e_{s,dm},$$

$$\bar r_{dm}=-\sum_s\bar Y_{s,d}C_{dm}H_{dm}\tau_se_{s,dm},\qquad
\bar C_{dm}=\sum_s\bar Y_{s,d}H_{dm}e_{s,dm}$$

* **Memory.** The PyTorch broadcast materialises $(B,L,S,n,N)$; the kernel materialises only
  $(B,L,S,n)$, an **$N$-fold saving** in four fused operations.
* **Layout.** A block is $n\cdot N$ threads (one per $(d,m)$), grid-stride over items. $\tau$ and
  $\bar Y$ are staged in shared memory, the sum over $m$ is a group shuffle, and the sum over all
  $(d,m)$ for $\bar\tau$ is a warp shuffle plus shared memory. $\bar r,\bar C$ are deterministic
  per-block partials.
* Registers: 30–32, no spills.

### 8.3 Tokeniser with a host/device-identical RNG (`evkit/include/evkit/tokenize.hpp`)

Dequantisation noise uses **Philox4x32-10** (counter-based). Ten rounds of

$$(c_0,c_1,c_2,c_3)\leftarrow\big(\mathrm{hi}(M_1c_2)\oplus c_1\oplus k_0,\ \mathrm{lo}(M_1c_2),\ \mathrm{hi}(M_0c_0)\oplus c_3\oplus k_1,\ \mathrm{lo}(M_0c_0)\big),
\qquad k\leftarrow k+(W_0,W_1)$$

with $M_0=\texttt{0xD2511F53}$, $M_1=\texttt{0xCD9E8D57}$, $W_0=\texttt{0x9E3779B9}$,
$W_1=\texttt{0xBB67AE85}$, and $U=(\text{out}_0\gg 8)\cdot2^{-24}$.

* The counter is the **global event index**, so tokens are identical regardless of chunking, thread
  count or device. CPU and GPU are bit-identical by construction; this is tested on CPU and the CUDA
  test is written.
* Verified against the published Random123 known-answer vectors.

### 8.4 Parallel exact decoding of Prophesee streams (`evkit/src/events.cpp`)

EVT3 is stateful: time-high/low registers, the current row $y$, a vector base $x$ and polarity, and a
24-bit time base that wraps every $2^{24}\,\mu s\approx16.8$ s. Naively splitting the file across
threads corrupts events near the chunk boundaries.

**The decoder as a state monoid.** For each chunk, a fast scan computes a *summary* $\Phi_k$ that maps
any entry state to the exit state:

$$\Phi_k=\big(\text{has}_y,\,y;\ \text{has}_{tl},\,t_{\text{low}};\ \text{has}_{\text{base}},\,x_{\text{after}},\,p\ \text{or}\ \Delta x;\ \text{has}_{th},\,th_{\text{first}},\,th_{\text{last}},\,\#\text{wraps};\ \text{bound}_k\big)$$

Composition is associative. A sequential prefix over only $T$ summaries gives each chunk its exact
entry state:

$$E_{k+1}=\Phi_k(E_k),\qquad \text{epoch}'=\text{epoch}+\mathbb 1[\text{has}_{th}\wedge th_{\text{first}}<th_{\text{prev}}]+\#\text{wraps}$$

Then all chunks decode in parallel from their entry states.

* **Allocation-free.** The summary also counts the exact upper bound of events per chunk. Outputs are
  allocated once, each thread writes its slice through raw pointers, and a compaction pass closes gaps
  (non-zero only for events before the stream's first TIME_HIGH or ADDR_Y).
* **Zero-copy.** Words are read straight from the mmap through `aligned(1)` typedefs (payloads start at
  arbitrary byte offsets after the text header). An allocator with no-op value-initialisation makes
  output pages be first-touched in parallel by the writer threads, not by a single-threaded memset.

**Measured here (4 shared CPU cores):**

| Path | Before | After |
|---|---:|---:|
| EVT3, 1 thread | 9.8 Mev/s | ~70 Mev/s |
| EVT3, 4 threads | 18 Mev/s | ~210 Mev/s |
| EVT2, 4 threads | 64 Mev/s | ~280 Mev/s |

The output is exactly equal to sequential decoding for 1–16 threads (tested). EVT2 parallelises the
same way with a TIME_HIGH prefix; DAT is embarrassingly parallel.

### 8.5 EVB training store

```
[128-B header][t:int64×n][x:uint16×n][y:uint16×n][p:uint8×n][pad][index:uint64×n_idx]
```

$\text{index}[k]=\min\{i: t_i\ge t_0+k\cdot1\,\mathrm{ms}\}$, so a time window is found with two index
loads plus a binary search inside one millisecond: effectively O(1) on multi-hour recordings. The file
is mmap-able and zero-copy into torch. Short-recording datasets are *packed* into shards with a
sample index (sample → event range, label).

### 8.6 PyTorch integration

`EvPPConfig(impl="evkit")` swaps the SSM scan in every layer and the readout at the quadrature nodes
for the fused ops, via `torch.autograd.Function` wrappers. Dispatch is by tensor device. The latent
clock and head MLPs stay in PyTorch: they are tiny (clock dimension 8) or matmul-bound.

---

## 9. Complexity and memory

Per layer, for $B$ windows of $L$ events with $D$ channels and $N$ states:

| | PyTorch reference | evkit |
|---|---|---|
| Scan FLOPs | $O(BLDN)$ | $O(BLDN)$ (the same `ex2` + FMA per element) |
| Scan activation memory kept for backward | $O(BLDN)$ (decay, drive, states) | $O(BLDN/32)$ checkpoints + $O(BL\,d_{\text{int}}N)$ on the last layer only |
| Example ($B{=}8,L{=}4096,D{=}256,N{=}16$) | ≈ 537 MB per tensor per layer | ≈ 17 MB checkpoints + 33 MB saved states |
| Compensator readout ($S$ nodes) | $O(BLS\,d_{\text{int}}N)$ memory | $O(BLS\,d_{\text{int}})$ memory |
| Sequential depth | $O(\log L)$ (Hillis–Steele) | $O(L/K_c)$ chunks, or $O(L)$ per block with full GPU occupancy |
| Streaming inference | $O(DN)$ state per stream | $O(DN)$ state per stream, constant per event |

Likelihood heads cost $O(B L (K + 2 + S))$ for the categorical heads, plus $M$ mixture components for
the time head. At $K=256$, $S=16$ this is small next to the backbone.

---

## 10. Experimental protocol, baselines, ablations

### 10.1 What each experiment can and cannot establish

| Experiment | Can establish | Cannot establish alone |
|---|---|---|
| Lower validation NLL | Better event prediction under the chosen likelihood | Better semantic representations |
| Better frozen linear probe | More linearly accessible downstream information | Superiority across all tasks |
| Better fine-tuned accuracy | Better transfer under the stated budget | That the clock caused the gain |
| Better accuracy under speed shifts | Speed robustness | Robustness to brightness, trajectory or sensor |
| Better calibration | More reliable probabilities under the tested conditions | Better classification |
| Higher temporal gain | Greater use of the perturbed temporal structure | That the structure is meaningful motion rather than noise |
| Kernel / backend equivalence | Two implementations agree in the tested scope | Anything about usefulness |

### 10.2 Required comparisons

* **Same backbone, data and compute:** V2 against masked-event modelling, next-count prediction and
  patch-Poisson. V3 and V4 are deferred (§10.5).
* **Clock ablation:** off / on / on + warp loss / on + gauge, with nothing else changed.
* **Timestamp interventions:** reverse order, shuffle order with gaps kept, shuffle gaps with order kept,
  controlled dilations $\mathcal D_f$.
* **Robustness breakdown:** clean accuracy and the drop on each shift type of §3.5, reported separately.
* **Reliability:** at least 3 seeds, mean ± 95% CI, checkpoint selection on a validation split carved from
  training data, never on test.
* **External baselines:** published GEP, EVA, TESPEC, MEM and Scaling-Dense numbers where protocols match;
  rows marked as non-controlled otherwise.
* **Causal vs bidirectional:** a causal-only probe for every model, in addition to each model's native
  probe (§5.9).

**Downstream tasks:**

1. Frozen linear probe and fine-tune on N-ImageNet (or mini), HARDVS and DailyDVS-200.
2. Shift sets: the 9 N-ImageNet robustness sets, plus held-out sensors (Gen1, DSEC).
3. Label-free metrics: bits/event vs Poisson, KS calibration per shift, temporal gain $G$, dilation gap
   $\delta_f$.
4. Anytime: accuracy vs events consumed, with the stopping threshold calibrated on held-out data.

### 10.3 Ablations

| Axis | Levels |
|---|---|
| Objective | V2 / masked / next-count / patch-Poisson (V3, V4 in stage 4) |
| Clock | off / on / on + warp loss / on + gauge |
| Arrival-time readout $\hat y_{i-1}(t_i)$ in the mark context | on / off |
| Dequantisation | on / off (ties collapsed) |
| Noise weighting | uniform / down-weighted sub-cell NLL |
| Tokenisation | per event / micro-window tokens **[planned]** |
| Scale | model width × window length × data hours |

### 10.4 Decision rule (fixed before running)

Define $\Delta_{\text{shift}}$ as V2 minus the best same-budget baseline on mean shifted-set accuracy, and
$\Delta_{\text{clean}}$ likewise on clean accuracy, each with a 95% CI over 3 seeds.

* **Main representation paper** if $\Delta_{\text{shift}}\ge 1.0$ point with CI excluding 0, and
  $\Delta_{\text{clean}}\ge -0.5$ point.
* **Analysis paper** (calibration and temporal diagnostics) if that fails but H4 holds: the diagnostics
  rank shifts by downstream degradation with Spearman $\rho\ge 0.6$ across models.
* **Stop or rethink** if neither holds.

The thresholds are a proposal to agree on before the first run; what matters is that they are fixed in
advance.

### 10.5 Staged plan

1. **Verify the mathematics and implementation.**
   * Done here: dilation convention; clock and readout indices traced through the forward pass;
     own-token leakage test; likelihood on simulated streams with known intensities; causality; state
     carry; normalisation; gradients.
   * Still open: CUDA kernels on a GPU (`build/test_cuda`, CUDA section of `tests/python`).
2. **Smallest decisive real-data experiment.** Pretrain V2 and the three same-backbone baselines on one
   unlabelled corpus (e.g. 1Mpx + eTraM). Probe on N-ImageNet mini clean and shifted, and on controlled
   dilations. Record NLL, calibration and temporal gain alongside accuracy.
3. **Test the causal explanation.** Clock ablations, diagnostic–degradation correlation, an inspection of
   whether temporal gain reflects motion or noise, repeated over seeds.
4. **Publication decision** (§10.4). Only then expand to V3, V4, larger corpora and the full systems stack.

---

## 11. Risks and how the method could fail

1. **Likelihood ≠ utility.** Spike-tokenizer work found reconstruction and downstream utility diverge.
   The model may spend capacity on noise (hot pixels, background activity). *Mitigation:* report both;
   use V4's explicit noise term; down-weight sub-cell NLL.
2. **Cost at sensor scale.** $10^6$ events/s per window means very long sequences. *Mitigation:*
   micro-window tokens **[planned]**, chunked state carry, the fused kernels.
3. **Strong incumbents.** EVA and GEP may win on clean benchmarks. *Mitigation:* the claim rests on
   shift robustness, calibration and capabilities, not only clean accuracy.
4. **"Where" heads at 1280×720.** $K=256$ patches is coarse. *Mitigation:* hierarchical heads
   **[planned]**; the normalised quantiser already supports any $G,Q$.
5. **Calibration tests over-reject.** KS on $10^5$ events rejects any imperfect model. *Mitigation:*
   report statistics and QQ curves, compare models relative to each other.
6. **Clock is weak or unidentified.** In toy tests the tempo loss only partially converges, and the
   $(r,\tau)$ scale symmetry (§5.3.1) means its values carry no absolute meaning. It may add nothing on real
   data; it is an ablation, not a pillar.
7. **Robustness over-claimed.** The clock only addresses timing (§3.5). Brightness, trajectory and sensor
   shifts need their own evidence.
8. **V4 identifiability.** $C$ is not identified without supervision (Section 5.7). *Mitigation:*
   simulator supervision; validation on paired brightness variants.
9. **Novelty.** Our searches found no raw-event likelihood pretraining, but they were limited.
   *Mitigation:* a dedicated arXiv check before submission (listed in `READING_PLAN.md`).

---

## 12. Status: verified vs not

| Item | Status |
|---|---|
| Model variants V1–V4, baselines, clock (+ gauge option), diagnostics, probe, anytime posterior | Implemented in `evpp/`; 31 tests pass: causality of every variant, **no own-token leakage into event i's context**, chunked state == full pass, normalisation, quadrature exactness, KS on known intensities, **analytic dilation gap** |
| C++ CPU kernels, decoders, EVB, tokeniser | 5/5 C++ test suites pass (finite-difference gradients, Random123 known-answer vectors, exact round trips including EVT3 wraps, parallel == sequential) |
| evkit backend inside the model | Same loss and every parameter gradient as the PyTorch backend for V2, V3 and V4 (24 Python tests pass) |
| CUDA kernels | Compile to PTX and SASS for sm_80/sm_90 with no spills; **not run: no GPU in this sandbox**. `test_cuda` and `bench_cuda` are ready to run |
| Training on real data | **Not done.** Only toy-simulator smoke runs, which are not evidence |
| H1–H4 | **Untested.** The review's mathematical items are resolved in code and text; the hypotheses still need the stage-2 experiment |

---

## 13. References

**Event-camera pretraining and backbones**

* Klenk et al., *Masked Event Modeling*, WACV 2024, [arXiv:2212.10368](https://arxiv.org/abs/2212.10368)
* *Event Camera Data Pre-training* (ECDP), [arXiv:2301.01928](https://arxiv.org/abs/2301.01928)
* Mohammadi et al., *TESPEC*, ICCV 2025, [arXiv:2508.00913](https://arxiv.org/abs/2508.00913)
* Cao et al., *Generative Event Pretraining with Foundation Model Alignment* (GEP), CVPR 2026 Findings, [arXiv:2603.23032](https://arxiv.org/abs/2603.23032)
* *Scaling Dense Event-Stream Pretraining from Visual Foundation Models*, CVPR 2026, [arXiv:2603.03969](https://arxiv.org/abs/2603.03969)
* Hao et al., *Maximizing Asynchronicity in Event-based Neural Networks* (EVA), ICLR 2026, [arXiv:2505.11165](https://arxiv.org/abs/2505.11165)
* Schöne et al., *Scalable Event-by-event Processing … with Deep State-Space Models* (Event-SSM), [arXiv:2404.18508](https://arxiv.org/abs/2404.18508)
* Zubić et al., *State Space Models for Event Cameras*, CVPR 2024, [arXiv:2402.15584](https://arxiv.org/abs/2402.15584)
* Fang & Panda, *Event2Vec*, ICML 2026, [arXiv:2504.15371](https://arxiv.org/abs/2504.15371)
* *Spiking Patches*, [arXiv:2510.26614](https://arxiv.org/abs/2510.26614)
* *Neural Events*, [arXiv:2606.19835](https://arxiv.org/abs/2606.19835)

**Neural temporal point processes**

* Chen, Amos & Nickel, *Neural Spatio-Temporal Point Processes*, ICLR 2021, [arXiv:2011.04583](https://arxiv.org/abs/2011.04583)
* Zhou & Yu, *Automatic Integration for Spatiotemporal Neural Point Processes*, NeurIPS 2023, [arXiv:2310.06179](https://arxiv.org/abs/2310.06179)
* *Mamba Hawkes Process*, [arXiv:2407.05302](https://arxiv.org/abs/2407.05302)
* Chang et al., *Deep Continuous-Time State-Space Models for Marked Event Sequences* (S2P2), NeurIPS 2025, [arXiv:2412.19634](https://arxiv.org/abs/2412.19634)
* Lüdke et al., *Add and Thin*, NeurIPS 2023, [arXiv:2311.01139](https://arxiv.org/abs/2311.01139)
* *Edit-based Flow Matching for TPPs* (EdiTPP), ICLR 2026, [arXiv:2510.06050](https://arxiv.org/abs/2510.06050)
* Kerrigan et al., *EventFlow*, AISTATS 2026, [arXiv:2410.07430](https://arxiv.org/abs/2410.07430)
* Shchur et al., *Intensity-Free Learning of Temporal Point Processes*, ICLR 2020, [arXiv:1909.12127](https://arxiv.org/abs/1909.12127)
* Berghaus et al., *In-Context Learning of TPPs with Foundation Inference Models* (FIM-PP), ICLR 2026, [arXiv:2509.24762](https://arxiv.org/abs/2509.24762)
* *Foundation Models for Event Prediction Across Heterogeneous Event Logs* (FEP), [arXiv:2610.17503](https://arxiv.org/abs/2610.17503)

**Neural-spike foundation models**

* Azabou et al., *POYO*, [arXiv:2310.16046](https://arxiv.org/abs/2310.16046)
* Antoniades et al., *Neuroformer*, ICLR 2024, [arXiv:2311.00136](https://arxiv.org/abs/2311.00136)
* *NDT3: A Generalist Intracortical Motor Decoder*, [bioRxiv 2025.02.02.634313](https://www.biorxiv.org/content/10.1101/2025.02.02.634313)
* *Universal translator for neural dynamics*, [arXiv:2407.14668](https://arxiv.org/abs/2407.14668)
* *The Neural Pile*, [NeurIPS 2025](https://neurips.cc/virtual/2025/132685)
* *Evaluating Neural Data Tokenizers*, [NeurIPS 2026](https://neurips.cc/virtual/2026/oral/139126)

**Physics, theory, robustness**

* Gallego et al., *Event-based Vision: A Survey*, [PDF](https://rpg.ifi.uzh.ch/docs/EventVisionSurvey.pdf)
* Scheerlinck et al., *Continuous-time Intensity Estimation Using Event Cameras*, [arXiv:1811.00386](https://arxiv.org/abs/1811.00386)
* Wang et al., *An Asynchronous Kalman Filter for Hybrid Event Cameras*, [arXiv:2012.05590](https://arxiv.org/abs/2012.05590)
* *Receding-Horizon Maximum-Likelihood Estimation of Neural-ODE Dynamics and Thresholds from Event Cameras*, [arXiv:2603.05011](https://arxiv.org/abs/2603.05011)
* Hwang et al., *Ev-NeRF*, [arXiv:2206.12455](https://arxiv.org/abs/2206.12455)
* Tallec & Ollivier, *Can Recurrent Neural Networks Warp Time?*, ICLR 2018, [arXiv:1804.11188](https://arxiv.org/abs/1804.11188)
* Kim et al., *N-ImageNet*, ICCV 2021, [arXiv:2112.01041](https://arxiv.org/abs/2112.01041); *Ev-TTA*, CVPR 2022, [arXiv:2203.12247](https://arxiv.org/abs/2203.12247)
* Zhu et al., *Motion Equivariant Networks for Event Cameras* (TNT), [arXiv:1902.06820](https://arxiv.org/abs/1902.06820)
* Salmon et al., *Parallel Random Numbers: As Easy as 1, 2, 3* (Philox), SC 2011

**Formats**

* Prophesee EVT 2.0 / 3.0 specifications: [docs.prophesee.ai](https://docs.prophesee.ai/stable/data/encoding_formats/index.html)
