# Method sketches: raw-event point-process pretraining (EvPP)

Companion to `../event_pp_atlas.md`. Everything here is implemented in `evpp/` (pure PyTorch, no PyG).
Status legend: **[tested]** a unit test checks it; **[runs]** executes end-to-end on the toy simulator
but nothing about its quality is established; **[design]** described, not implemented.

## 0. What is and is not verified

* 27 tests pass (CPU, ~16 s): scan equality, quadrature exactness, density normalisation, **causality
  of every variant**, chunked-state == full-pass, clock identity at init, dilation-equivariance of the
  Poisson baseline, KS diagnostics on a process with known intensity, and "loss goes down" smoke
  training for every variant.
* All scripts execute. **Numbers they print come from a 32x32 toy simulator, a few hundred steps and
  tiny models. They are pipeline checks, not evidence.** Do not quote them.
* Not implemented / not run: real datasets, GPU-scale training, any comparison against EVA/GEP/TESPEC.

## 1. Notation and the likelihood

A window is a time-ordered event sequence e_1..e_n, e_i = (t_i, k_i, p_i, c_i): time, patch k in
{1..K}, polarity p in {0,1}, sub-cell c in {1..S} (coordinates are normalised, so sensors of
different resolution share one quantiser; defaults K=64, S=16). dt_i = t_i - t_{i-1} is
**dequantised** ((round(gap/q)+U(0,1))*q) so ties (dt=0) have a proper density.

Point-process likelihood of the sorted sequence (time reference measure = seconds, marks = counting
measure on (k,p,c)):

    log p(e_1..e_n) = sum_i [ log lambda_g(t_i | H_i)  +  log p(k_i,p_i,c_i | t_i, H_i) ]  -  int lambda_g(t) dt
                       \_______ time ______/            \________ marks ________/            \_ compensator _/

Every variant below instantiates this; they differ in how time and marks are parametrised. The tail
after the last event is not scored (windows end at an event). Report **nats/event** and compare to
the homogeneous-Poisson reference (`HomogeneousPoisson`).

## 2. Shared backbone: event-driven continuous-time SSM  [tested]

Per layer, per channel d and state n (rates r_dn>0, log-spaced timescales tmin..tmax):

    h_i = exp(-r * tau_{i-1} * dt_i) * h_{i-1} + B * v_i        (impulse at each event, exact decay between)
    y_i = C . h_i + skip * v_i

* Parallel training via a Hillis-Steele scan (`scan_assoc`), O(1)-state streaming inference; chunks can be
  fed with carried state (tested to match the full pass).
* **Latent clock** `tau`: a causal, strictly positive tempo (count-EMA of a tanh feature, zero-init so
  tau=1 at start). tau after event i-1 scales the gap before event i, so a speed change is absorbed by
  tau rather than by relearning dynamics. History-only by construction (tested via causality).
* **Continuous-time readout**: only the last layer's linear readout `C . (exp(-r dt) h)` continues to
  evolve between events; this is what intensity heads evaluate at any t in a gap.
* The head for event i sees s_{i-1}, the state read out *at the arrival time* t_i, and features of dt_i.

## 3. The variants

### V2 `objective="ar"` - intensity-free autoregressive  [tested + runs]
Time: mixture of log-normals over dt_i given s_{i-1} (closed-form density, CDF -> PIT, sampling).
Marks: patch -> polarity -> sub-cell categoricals conditioned on z_i.
* **+** no integral, cheapest, exact likelihood, can sample (`EvPP.sample`).
* **-** the time law is not tied to the decaying state, so the "intensity" is implicit; a mixture of
  log-normals can be poor for strongly bursty/lagged dynamics.

### V3 `objective="intensity"` - ground intensity + quadrature compensator  [tested + runs]
Lambda(t) = exp(g([yhat(t); s])), yhat(t) the decaying readout. Compensator per gap by Gauss-Legendre
in **log time** (exact for exponential intensities across 1e-7..0.4 s - tested), deterministic and
differentiable. Gives xi_i = int over gap, the time-rescaling statistic.
* **+** a genuine continuous-time intensity (Hawkes-like), natural for the time-rescaling diagnostic.
* **-** n_quad evaluations per gap (memory ~ B*L*n_quad*d_int*N), softplus/exp link needs care at
  extreme rates; ties give zero-width gaps (handled by dequantisation).

### V4 `objective="physics"` - physics-structured intensities  [tested + runs]
Per-patch rates  lambda_{k,ON/OFF}(t) = softplus(+-s_k(t))/C + r_n/(2K)  with s_k = d(log I)/dt read from
the state, theta = (ln C, ln r_n) the **sensor parameters**. Ground intensity = sum of rates; marks
(patch, polarity) = rate ratios; sub-cell from the shared head.
`sensor_mode`: `learned` (global default), `oracle` (given), `infer` (SensorEncoder pools the backbone states of the first
`sensor_ctx` events; those events are excluded from the loss; supervised by an aux MSE on simulated theta).
* **+** the only variant where the fitted parameters mean something physical; enables sensor ID and
  zero-shot adaptation to a new threshold.
* **-** **identifiability**: without supervision ln C is only defined up to the scale of s_k (the network can
  absorb it). Anchoring options: simulator supervision, known lux, paired recordings (N-ImageNet variants).
  On the toy simulator the inferred theta is only weakly correlated with the truth
  (corr ~0.2 for ln C, ~0.4 for ln r_n after 700 steps): treat the module as a starting point.

### V1 `PatchPoisson` - fixed-bin count model  [tested + runs]
Poisson counts per (patch, polarity) per 1 ms bin from the same SSM. Reported as a point-process density
(`sum(lam - n log lam) + n log(w S)`) so it is comparable with V2-V4.
* **+** robust, tie-free, simplest; the ablation "does event-level timing matter?".
* **-** nothing finer than the bin; dense (B x bins x 2K) activations.

### Non-likelihood baselines on the same backbone  [tested + runs]
`MaskedEventModel` (bidirectional, hide 50% of event identities) and `NextCountPrediction` (EVA-style
per-patch count in the next 5 ms). These make the core claim testable: **likelihood vs reconstruction vs
summary targets at equal backbone, data and compute.**

## 4. Diagnostics that need no labels  [tested]

| Diagnostic | Code | What it tells you |
|---|---|---|
| time-rescaling KS | `time_rescaling_report` | xi_i ~ Exp(1) (or PIT ~ U(0,1)) iff the temporal law is calibrated; run it **under rate shifts** |
| dilation gap | `dilation_gap` | (nll(dilate(x,f)) - nll(x)) + ln f = 0 iff the likelihood is exactly speed-equivariant; the Poisson baseline passes (tested) |
| temporal information gain | `temporal_gain` | bits/event lost under reverse / shuffle / polarity-shuffle; ~0 means the data does not need time |
| warp consistency | `EvPP.warp_consistency` | self-supervised tempo loss: dilate by f -> log tau must rise by ln f, pooled content must not move |

Interpretation guardrail: KS on thousands of events rejects almost any imperfect model. Report the
statistic and QQ quantiles, not just p.

## 5. Downstream readouts

* `eval/probe.py`: frozen features (pooled states + mean log tau + mean log rate) -> linear probe.
* `eval/sprt.py` (R5): class-conditioned model; running posterior over classes; stop when max posterior >= 1-alpha.
  The toy run stops after a median of ~19 events and stopping time shrinks with event rate, but the
  achieved accuracy (0.84) is below the 0.95 target: generative posteriors are over-confident, so
  calibrate the threshold on held-out data before claiming latency gains.

## 6. Ablation matrix for the paper (all runnable via `scripts/compare_objectives.py` once real streams are plugged in)

| Axis | Levels |
|---|---|
| Objective (R1) | ar / intensity / physics / poisson / mem / nextcount |
| Clock (R4) | off / on / on + warp-consistency |
| Rate augmentation | none / dilation only / dilation + thinning + densify + noise |
| Tokenisation | 1 event per token (here) vs patch tokens (design: group events in 50-500 us micro-windows) |
| Sensor conditioning (R2/R3) | none / learned / inferred / oracle |
| Scale | d_model 48 -> 256, window 128 -> 4096 events |

Evaluation: bits/event vs homogeneous Poisson; linear-probe and fine-tuned accuracy; accuracy under shifts
along **four physical axes** (speed, thinning, densify, noise); KS calibration under the same shifts;
temporal gain. Report 3+ seeds and select checkpoints on a held-out validation split (not test).

## 7. Known weak points to resolve before real experiments

1. **Cost.** Per-event tokens at 1e5-1e6 events/s need chunking and likely patch/micro-window tokens;
   quadrature memory grows with n_quad. Profile on a GPU first.
2. **Where-prediction.** Patch categorical over K=64 is fine for 32x32 toys; for 1280x720 use hierarchical
   or coarse-to-fine heads (the quantiser already supports G, Q; only the head would change).
3. **Ties/bursts.** Dequantisation makes the density proper but spends probability mass on within-quantum
   ordering; a set-valued micro-window likelihood is cleaner (design).
4. **Time-rescaling interpretation.** The statistic covers only the temporal ground process; marks need a
   separate calibration check (ECE / PIT on the categorical).
5. **Clock capacity.** In the toy test the tempo loss falls by about 60% in 300 steps (tempo objective alone); check whether the
   clock is actually useful on real data or if the SSM timescales already cover the range.
6. **Reference measures.** Cell counting measure in space and seconds in time; fix this convention when comparing
   to any published bits/event.
7. **Real-sensor behaviour** (refractory period, hot pixels, polarity asymmetry, per-pixel thresholds) is absent
   from the simulator and from V4 beyond (C, r_n).

## 8. Suggested first real-data run (for when GPU time exists)

1. DVS Gesture + N-Caltech101 + Gen1 windows through the same `Stream` interface (write 3 loaders).
2. Pretrain `ar` and `intensity` (d_model 128, L=1024, bf16), plus `mem` and `nextcount` at equal FLOPs.
3. Linear probe at native rate and under the four shifts; compute KS + temporal gain on the pretrained models.
4. Decision rule: continue only if (a) likelihood pretraining >= masked/summary pretraining on the shifted
   probes, or (b) the temporal-gain/calibration analysis yields a clear, reproducible finding on its own.
