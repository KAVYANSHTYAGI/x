# Event-stream point-process foundation models: literature atlas, weaknesses, and where to build

Compiled 2026-10-23. Scope: "Option 1" = a rate-native, continuous-time model of raw event streams
trained with a point-process likelihood (next-event / intensity), used as a pretrained backbone.

## How much to trust this

Evidence tags on every paper:

* **[R]** I fetched the paper's text (HTML/PDF) and a summariser extracted method, results and limitations.
  The summariser only saw the first ~100k characters, so appendix limitations can be missed.
* **[A]** abstract page only. Claims about method details beyond the abstract are NOT verified.
* **[S]** appears only in search-result summaries.

"(stated)" = the authors say it. "(inferred)" = a reviewer-style concern from the extracted text, not
the authors' words. Verify anything you build a claim on by reading the paper yourself. Section 7 lists
what to read first. I did not run any of these methods.

---

## 1. The map in one paragraph

Three communities each hold one third of Option 1 and barely cite each other:

1. **Event vision** has tokenizers, linear-recurrent backbones and pretraining, but its objectives are
   masked reconstruction or feature alignment on **pseudo-frames**, never a likelihood over raw events.
2. **Temporal point-process (TPP) ML** has the likelihood, the theory and now foundation models, but only
   on **low-dimensional marked sequences** (tens to thousands of events, discrete marks, no space).
3. **Neural-spike foundation models** (POYO, NDT3, Neuroformer, Neural Pile) tokenise **individual
   spikes** at scale, and say outright that *sensor variability* is not solved by scaling.

Nobody I found combines raw-event tokens + continuous-time likelihood + spatial coordinates + pretraining
across sensors. That intersection is the opening.

---

## 2. Cluster A: event-vision pretraining

| Paper | What it does | Weaknesses relevant to us |
|---|---|---|
| **[GEP](https://arxiv.org/abs/2603.23032)** (CVPR'26 Findings) **[R]** | Stage 1 aligns an event encoder to frozen DINOv2; stage 2 trains a causal transformer on interleaved event/image tokens, predicting future latent embeddings. Event-1.8M pairs. N-ImageNet top-1 65.1 (ViT-S), 75.2 (ViT-B); N-Caltech101 93.1 / 96.5. | **Not a likelihood**: MSE on latents (stated by design). Events are accumulated into fixed-Δt pseudo-frames, so **rate is hidden by normalisation** (99th-percentile). Depends on a frozen image teacher, so cannot learn what images lack (microsecond timing, rate). Pretraining data overlaps downstream sets N-ImageNet and DSEC (inferred). No variance / seeds (inferred). No limitations section. |
| **[Scaling Dense Event-Stream Pretraining](https://arxiv.org/abs/2603.03969)** **[A]** | Distils vision foundation models into event representations with a structure-aware loss. | Abstract gives no sizes, metrics or scaling curves. Same image-teacher dependence as GEP (inferred). Do not cite as a "scaling law" paper without reading it. |
| **[TESPEC](https://arxiv.org/abs/2508.00913)** (ICCV'25) **[R]** | MAE-style masking on 50 ms histogram segments with a recurrent Swin; target is a pseudo-grayscale video. 1Mpx pretraining (~15 h). Gen1/1Mpx mAP 51.6/50.6. | Reconstructs *frames*, so timing finer than 50 ms is discarded. Pretraining is expensive (135 h vs 45 h for ECDP, stated). 1Mpx is driving only (stated). Noted that with pretraining, feedforward models can beat recurrent ones, which is a warning for any recurrent design. Gains over ECDDP ~0.6 mAP, no seeds (inferred). |
| [MEM](https://arxiv.org/abs/2212.10368) **[S]** | Masked event modelling with a dVAE tokenizer. | Frame-like tokens; N-Caltech/N-Cars/N-ImageNet only. |
| [ECDP](https://arxiv.org/abs/2301.01928) **[S]** | Contrastive event–image pretraining. | Needs paired RGB; still frames. |
| [Physics-inspired pretraining](https://arxiv.org/abs/2508.05507) **[S]** | Masked modelling on temporal-intensity-difference maps, then event–image contrast. | Two-stage, image-paired. Closest in spirit to using the event-generation model, but as a *data transform*, not a likelihood. |
| [TGVFM](https://arxiv.org/abs/2511.06238) **[S]** | Adapts image VFMs to events with temporal context fusion. | Dense-task oriented. |

**Cluster take-away:** every strong event pretrainer either needs an image teacher or reconstructs
frames. None models *when and where the next event arrives*. The rate is normalised away or never studied.

---

## 3. Cluster B: event-native backbones and tokenizers

| Paper | What it does | Weaknesses |
|---|---|---|
| **[EVA](https://arxiv.org/abs/2505.11165)** (ICLR'26) **[R]** | Per-event tokens (polarity,row,col + sinusoidal time), RWKV-6 linear attention, patch-wise encoding; self-supervised next-**representation** prediction (event count, time surface). Gesture 96.9 (file voting), N-Caltech101 86.3, Gen1 47.5/47.7 mAP. | Targets are *count/time-surface* summaries, not a likelihood. Uses inter-event time differences, **not absolute time**, to cope with long sequences (stated) but no rate-shift study. Latency 14.7 ms / 8,192 events, higher than earlier A2S (inferred). Detection gain over RVT-B small (47.7 vs 47.2). Ablations only on Gesture. N-Cars needed Gen1 pretraining. Appendix limitations not read. |
| **[Event-SSM](https://arxiv.org/abs/2404.18508)** **[R]** | S5 SSM driven by inter-event gaps Δk with a learnable δ; associative scan; event pooling. SHD 95.9, SSC 88.4, DVS Gesture 97.7. | **Test-set model selection** (stated). 32,768-event training slices vs full-sample eval (stated). "No explicit event-generating mechanism" (stated). Does **not** test rate or time-scaling robustness despite using Δk. Below prior best on DVS Gesture. Simulated on GPU, no latency/energy on event hardware (inferred). |
| **[Zubić et al., SSMs for event cameras](https://arxiv.org/abs/2402.15584)** (CVPR'24) **[R]** | S4/S4D/S5 in a ViT with learnable timescale; rescale at inference to change frequency. 3.76 mAP average loss up to 200 Hz vs ~21–25 for RVT/GET. | Aliasing is a real limit, bandlimit α=0.5 chosen empirically (stated). Single training rate (20 Hz) (inferred). Reported RVT degradation inconsistent across tables (inferred). Gains over best methods <1 mAP. Handles *frequency of windows*, not event-rate shifts of the scene. |
| **[Event2Vec](https://arxiv.org/abs/2504.15371)** (ICML'26) **[R]** | Event = spatial MLP embedding of (x,y,p) + temporal embedding from timestamp *differences*; fixed event budget by random sampling or K-means; Transformer. Gesture 97.6, ASL-DVS 99.9, DVS-Lip 75.9. | No "word2vec-style" objective in the main text; trained end-to-end (so name overpromises). Fixed event count per stream **discards rate** by construction (inferred). Only 3 classification sets. DVS-Lip gain relies on clustering + pretraining; random-sampling result (70.6) trails prior work. Notable as ICML venue evidence: event tokenisation is acceptable at ICML. |
| [Spiking Patches](https://arxiv.org/abs/2510.26614) **[A]** | Asynchronous tokenizer; up to 3.4× faster than voxel tokens, 10.4× than frames, matched accuracy. | Abstract only; rate behaviour unknown. |
| [Neural Events](https://arxiv.org/abs/2606.19835) **[A]** | Re-tokenises streams into learned discrete "neural events"; halves event rate with similar accuracy. | Abstract only; objective, datasets unknown. |
| [EventMamba](https://arxiv.org/abs/2405.06116) **[S]** | Point-based Mamba for event classification/regression. | Your own port showed FiLM+curriculum helps (67.4→74.1); the backbone itself has no rate mechanism. |
| [STREAM](https://arxiv.org/abs/2411.12603) **[S]** | SSM for sparse geometric data. | Generic; not event-specific. |

**Cluster take-away:** the architectural ingredients for Option 1 exist (per-event tokens, RWKV/SSM,
inter-event Δt). What is missing is (a) a likelihood objective and (b) any analysis of rate.

---

## 4. Cluster C: neural TPP modelling

| Paper | What it does | Weaknesses relevant to us |
|---|---|---|
| **[Neural STPP](https://arxiv.org/abs/2011.04583)** (ICLR'21) **[R]** | Jump and Attentive CNFs for continuous-space-time intensity; seismology, COVID, fMRI. | Sequences of **4–1,741 events** (stated by datasets). Jump CNF restarts the ODE solver after every event, cost grows with count (stated). Needed regularisation for stability (stated). Only 3 runs; attention is quadratic (inferred). Cannot touch 10⁵–10⁷ events. |
| **[Auto-STPP](https://arxiv.org/abs/2310.06179)** (NeurIPS'23) **[S]** | Automatic integration (ProdNet) for the 3D compensator. | Exact integral for a restricted decomposable form; scaling beyond 3D / huge counts unclear. Still the best answer to "how do I get the compensator integral cheaply". |
| **[Mamba Hawkes](https://arxiv.org/abs/2407.05302)** **[R]** | Mamba with Δ = inter-event gap; intensity = sum of softplus terms. Sequences up to avg 2,074. | "Specifically designed for the Hawkes process" (stated). No seeds; some baselines missing; MHP-E combines two architectures so gains may be capacity (inferred). Discrete marks only. |
| **[S2P2](https://arxiv.org/abs/2412.19634)** (NeurIPS'25) **[R]** | Latent linear Hawkes layers: diagonal continuous-time SSM driven by event impulses; intensity read off the continuous state; MC compensator; parallel scan. Avg +33% likelihood on 8 datasets. | **Space, non-categorical marks, high rates all unsupported (stated / left to future work)**. Cannot model simultaneous events (stated) which event cameras produce constantly. Needs numerical integration for expected next-event time (stated). Five seeds, one GPU. **Best architectural template for us.** |
| **[Add-Thin](https://arxiv.org/abs/2311.01139)** (NeurIPS'23) **[R]** | Diffusion over whole sequences: add Poisson points, thin existing; 100 denoising steps. | No marks (stated), no space; sampling cost (stated). Small datasets (Taxi 182 seqs). Notably *thinning/adding Poisson points* is exactly the event-camera threshold/noise axis (a useful hook). |
| **[EdiTPP](https://arxiv.org/abs/2510.06050)** (ICLR'26) **[R]** | Edit flow (insert/delete/substitute) for TPPs. Mean lengths 15–1,129. | Times only on [0,T], no marks/space (inferred); hand-set substitution distance; many gaps within std. |
| [EventFlow](https://arxiv.org/abs/2410.07430) (AISTATS'26) **[S]** | Non-autoregressive flow matching over joint event times; 20–53% lower forecast error. | Times only. |
| [Intensity-free TPP](https://arxiv.org/abs/1909.12127) **[S]** | Mixture/flow over inter-event times; closed-form sampling. | The standard alternative to intensity-based when the compensator is costly; relevant for per-pixel time marks. |
| [EasyTPP](https://arxiv.org/abs/2307.08097), [TPP survey](https://arxiv.org/abs/2501.14291), [Neural STPP review](https://arxiv.org/abs/2502.09341) **[S]** | Benchmark and surveys. | Use for baselines and positioning. The surveys note non-autoregressive models lack explicit likelihood and are costly to sample. |

**Cluster take-away:** the TPP literature has solved *how to put a likelihood on events* but at scales
3–5 orders of magnitude below event cameras, and without space. Practical bridge: S2P2-style
continuous-time state + Auto-STPP/MC compensator + patch-level factorisation.

---

## 5. Cluster D: foundation / in-context TPP

| Paper | What it does | Weaknesses |
|---|---|---|
| **[FIM-PP](https://arxiv.org/abs/2509.24762)** (ICLR'26) **[R]** | Pretrain a transformer on **14.4M synthetic events from ~72K Hawkes/Poisson processes**; infers conditional intensity from a context set in-context; up to 22 marks. | Prior is "not universal"; capped by max marks and sequence length (stated). Mark accuracy collapses on Taxi zero-shot (0.47 vs 0.91). Real-data transfer shown mainly by **qualitative** plots on Retweet (inferred). Baselines copied from prior work. No space (stated). **Template for in-context sensor identification.** |
| **[FEP](https://arxiv.org/abs/2610.17503)** **[R]** | Pretrained on **696 real event-log datasets**, tested on 50 held-out; quantile time normalisation, permutation-robust mark embeddings. | Event types + timestamps only (stated). Gains over best local model small, hard-tier accuracy trails (0.35 vs 0.38). Test-set separation self-verified (inferred). No space, no vision, **no high-rate streams** (inferred; length-100 efficiency tests). |
| [On Foundation Models for TPPs](https://arxiv.org/abs/2510.12640) **[S]** | Position/study on TPP foundation models. | Read for framing. |
| [STEP](https://arxiv.org/abs/2410.10648) **[S]** | Autoregressive tabular-event predictor; one token per event, attributes decoded autoregressively. | Useful design precedent for decoding (pixel, polarity, time) autoregressively per event. |

**Cluster take-away:** "foundation model for events" already means *log-likelihood pretraining* in TPP
land, and it works zero-shot on small sequences. Nobody has taken it to spatial, high-rate sensors.

---

## 6. Cluster E: neural-spike foundation models (closest structural analogue)

| Paper | What it does | Weaknesses |
|---|---|---|
| **[POYO](https://arxiv.org/abs/2310.16046)** **[R]** | One token per spike (unit embedding + time via RoPE); Perceiver cross-attention to 512 latents; handles any number of units; 100+ h, 158–178 sessions. | **Supervised decoding, not generative** (stated). No runtime/memory measurements; "scaling laws" from three depths; tokenisation never ablated against binning; inconsistent session/unit counts across text (inferred). Primates only. Needs a self-supervised objective to scale (stated). |
| **[POYO+](https://torch-brain.readthedocs.io/en/stable/generated/api/autosummary/torch_brain.models.POYOPlus.html)** **[S]** | Multi-task, cell-type-aware decoding. | Docs-level evidence only. |
| **[Neuroformer](https://arxiv.org/abs/2311.00136)** (ICLR'24) **[S]** | Autoregressive spatiotemporal generation of spikes; multimodal; few-shot behaviour decoding; recovers simulated connectivity. | Demonstrated on small circuits / mouse visual cortex scale (inferred); not stress-tested for rate drift. |
| **[NDT3](https://www.biorxiv.org/content/10.1101/2025.02.02.634313)** **[S]** | Autoregressive transformer, up to 2000 h, 45M / 350M params. | Authors say scaling autoregressive transformers "seems unlikely to resolve" **sensor variability and output stereotypy** (stated). *This is exactly our rate/sensor issue.* |
| **[IBL "universal translator"](https://arxiv.org/abs/2407.14668)** **[A+S]** | Masked reconstruction across time/neuron/region; single-spike resolution. | Full text exceeded the fetch size limit; unread. |
| [Neural Pile](https://neurips.cc/virtual/2025/132685), [tokenizer-evaluation framework](https://neurips.cc/virtual/2026/oral/139126) **[A]** | 476B-token spike corpus + 8B baseline; framework showing **reconstruction quality and downstream utility diverge** for tokenizers. | Abstract only. The divergence result is directly useful for justifying a likelihood-vs-downstream study on events. |

**Cluster take-away:** the neuro community has *already* moved from supervised decoding (POYO) to
generative pretraining (Neuroformer, NDT3, Neural Pile) and hit the wall we expect: session-to-session
rate and sensor drift. That is both evidence the idea works and the open problem.

---

## 7. Cluster F: physics, estimation, theory

| Paper | Relevance | Weakness / caveat |
|---|---|---|
| Event generation model (e.g. [Gallego et al.](https://rpg.ifi.uzh.ch/docs/EventVisionSurvey.pdf)) **[S]** | Ideal sensor: an event fires when log-brightness changes by threshold C, so expected intensity ∝ \|∂ₜ log I\| / C. A *physical* intensity function. | Real sensors deviate: threshold noise, refractory period, shot noise (Poisson) with light-dependent rate (non-monotonic in low light). |
| [Scheerlinck complementary filter](https://arxiv.org/abs/1811.00386) **[S]** | Continuous-time asynchronous intensity state from events (~10 ops/event). | Linear filter; no learning, no uncertainty. |
| [Asynchronous Kalman filter](https://arxiv.org/abs/2012.05590) **[S]** | EKF with explicit event/frame noise model, exact discretisation run only at events. | Image reconstruction only; Gaussian noise assumed. |
| [Neural-ODE + threshold ID](https://arxiv.org/abs/2603.05011) **[S]** | Estimates dynamics and contrast threshold online. | System identification, not recognition; per-pixel scalars. |
| [Ev-NeRF](https://arxiv.org/abs/2206.12455) **[S]** | Event generation model as a loss on rendered frames. | Gaussian-ish accumulation loss, **not a Poisson/point-process likelihood** (inferred). |
| [Tallec & Ollivier](https://arxiv.org/abs/1804.11188) **[S]** | Gated RNNs are quasi-invariant to general time warps; leaky RNNs only to uniform rescaling. | Theory for 1-D sequences; not tested on events. Motivates why a learned clock matters. |
| [Zhu et al., motion-equivariant TNT](https://arxiv.org/abs/1902.06820), [DiST/N-ImageNet](https://arxiv.org/abs/2112.01041), [Ev-TTA](https://arxiv.org/abs/2203.12247) **[S]** | Prior speed-invariance and test-time adaptation for events. | Frame-level representations; no generative model. |
| Simulators: [ADV2E](https://arxiv.org/abs/2411.12250), [ControlEvents](https://arxiv.org/abs/2509.22864), [E-Motion](https://arxiv.org/abs/2410.08649) **[S]** | Where scalable *unlabelled* training events can come from. | Noise realism remains open (ADV2E). Synthetic-to-real gaps documented (fall-detection 98.7→81.4). |

---

## 8. Our own starting point (CE-GNN) and what it taught us

* Rate-FiLM curriculum on DVS Gesture reproduces in a pure-torch port (baseline 67–69, curriculum 81–85).
* Pilot, shipped checkpoints only: **replacing the rate by a constant returns the curriculum model to
  baseline level (66.8 vs 67.8); a wrong rate drops it to 54.5**. Under a 2× speed-up it scores 82.9 with the
  true new rate but 47.9 with the stale one, below the 51.8 baseline. Thinning, densifying and noise
  behave the same way. Partial grid (about 88% of tasks); noise ×3 incomplete; single model family.
* Reading: on this backbone, the rate scalar works as an *oracle calibration* of neighbourhood density.
  A model that infers rate and tempo itself removes the oracle. That is the motivation for Option 1.
* The PointNet++/graph backbone is on the losing side of the 2026 representation debate (async graph
  methods trail dense/recurrent on Gen1), so new work should move to RWKV/SSM tokens.

---

## 9. Cross-cutting weaknesses (what nearly every paper above shares)

1. **No raw-event likelihood at event-camera scale.** Largest TPP sequences are ~10³; event streams are
   10⁵–10⁷.
2. **Rate is normalised away** (percentile scaling, fixed event budgets, window accumulation) or ignored.
   Event-SSM and EVA use Δt but never study rate shift.
3. **Teacher or frame dependence.** The strongest event pretrainers lean on image models or pseudo-frames.
4. **Single seed, test-set selection, copied baselines** recur (Event-SSM, FIM-PP, Mamba Hawkes, EdiTPP).
5. **No cross-sensor evaluation.** NDT3 says sensor variability isn't fixed by scale; event vision has not
   even tested it for pretraining.
6. **No simultaneous-event handling.** Event cameras emit ties and bursts. S2P2 explicitly cannot.
7. **Benchmarks barely test time** (N-Caltech101 uses identical camera motion for every class), so
   gains from temporal modelling are hard to attribute.

---

## 10. Hard technical problems Option 1 must solve (the real research content)

| Problem | Why hard | Candidate answers |
|---|---|---|
| **Compensator ∫λ over (x,y,t)** | Space is 10⁴–10⁶ pixels × millions of µs | Patch-factorised Poisson counts; MC with importance sampling near events (S2P2); structured separable intensity (Auto-STPP ProdNet); low-rank in space |
| **Predicting *where*** | Categorical over pixels is huge | Hierarchical / coarse-to-fine tokens; mixture-of-Gaussians over (x,y) conditioned on patch; autoregressive (patch → within-patch) like STEP |
| **Ties and bursts** | Same-µs events; no strict ordering | Break ties by pixel order; treat simultaneous events as a set within a micro-window; dequantise timestamps |
| **Compute** | Quadratic attention or ODE-per-event | Linear recurrence (RWKV/S5/S2P2 scan); event pooling; patch tokens |
| **Sensor/rate drift** | Learned dynamics have fixed timescales | Learn a **latent clock** (Δ rescaled by an inferred tempo); condition on sensor embedding à la POYO unit embeddings |
| **Evaluation** | Likelihood ≠ downstream utility (see tokenizer-eval result) | Report bits/event, calibration (time-rescaling test), linear-probe/fine-tune, and rate-shift transfer |

---

## 11. Candidate research directions

Ordered by how much I would bet on them. "Novel vs" = what it adds over the closest papers above.

### R1. Raw-event next-event likelihood pretraining (core)
* **Claim:** a continuous-time state-space model with per-event tokens, trained by NLL over
  (Δt, patch, within-patch position, polarity), gives better linear-probe and fine-tuned accuracy and
  better rate-shift transfer than masked/feature-aligned pretraining at the same compute.
* **Novel vs:** GEP (latent MSE on pseudo-frames), TESPEC (frame reconstruction), EVA (summary targets),
  S2P2/FIM-PP (no space, short sequences).
* **Key experiments:** compare objectives with identical backbone and data; scale unlabelled data.
* **Risks:** likelihood may not help downstream (tokenizer-eval warns of divergence); compute; 1–2
  strong baselines (EVA, GEP) can already beat you on benchmarks.

### R2. Physics-structured intensity with learnable sensor parameters
* **Claim:** parametrise λ via the event-generation model (latent log-brightness dynamics, threshold C,
  noise rate, refractory) so the *fitted parameters are the sensor state*. Pretrain on simulated sensors
  with randomised parameters (FIM-PP style), then infer them in context from a few milliseconds.
* **Novel vs:** Neural-ODE threshold ID (not amortised, not recognition), FIM-PP (no physics, no space),
  Ev-NeRF (Gaussian loss).
* **Risks:** simulator realism (ADV2E/ControlEvents show it is unsolved); real-shift evaluation needs
  N-ImageNet variants.

### R3. One model across sensors: "pixel/unit embeddings" for events, spikes and photon streams
* **Claim:** treat coordinates as continuous (RoPE on x,y,t) plus a sensor embedding so one pretrained
  model handles DVS128, ATIS, Gen4, spike cameras and even neural spikes.
* **Novel vs:** POYO/NDT3 (neural only; NDT3 flags sensor variability as unsolved), event vision
  (single-sensor pretraining).
* **Risks:** very large scope; best treated as an extension or a second paper.

### R4. Time-rescaling as a free self-supervised diagnostic and regulariser
* **Claim:** by the time-rescaling theorem, a well-calibrated intensity maps event times to a unit-rate
  Poisson process. Use the KS residual as (a) a label-free calibration metric under rate shift and
  (b) an auxiliary loss. Report it per sensor and per rate.
* **Novel vs:** unused in event vision to my knowledge; ties directly to the CE-GNN rate story.
* **Risks:** low risk, modest standalone novelty. Best as a component of R1.

### R5. Generative classification with sequential stopping
* **Claim:** class-conditional intensity models give per-event log-likelihood ratios; a sequential test
  gives an optimal stopping rule, so latency scales with information arrival, not with a window.
* **Novel vs:** early-recognition works (heuristic), EVA/Event-SSM (fixed-length).
* **Risks:** generative classifiers usually trail discriminative ones; frame as few-shot / anytime.

### R6. Temporal-information gain as a benchmark metric
* **Claim:** bits/event gained by conditioning on true vs shuffled/reversed history quantifies how much a
  dataset *needs* time. Predicts which benchmarks can test temporal models.
* **Novel vs:** Vicente-Sola-style frame-sufficiency arguments (qualitative); your own reverse/shuffle
  tests (accuracy-based).
* **Risks:** a metric paper alone is a hard sell; superb companion analysis for R1.

### R7. Scaling-law study for event streams
* **Claim:** loss vs (data hours, parameters, tokens/s) with simulated + real unlabelled corpora.
* **Novel vs:** none found with fitted curves (Scaling Dense unread).
* **Risks:** compute heavy; needs many unlabelled hours (Gen1 ~39 h, 1Mpx ~14 h are labelled sets;
  rest must be simulated).

### R8. Tokenizer evaluation for events (à la the 2026 spike-tokenizer paper)
* **Claim:** compare Spiking Patches, Event2Vec, EVA, Neural Events and fixed-window tokens on
  reconstruction, utility, cross-instance generalisation and compression.
* **Risks:** benchmark paper; depends on others' code. Good workshop-scale spin-off.

---

## 12. Recommendation

**Thesis for an ICML 2027 submission:**
*A rate-native, continuous-time point-process pretraining objective for raw event streams, with a
latent clock, time-rescaling calibration and a rate/sensor-shift evaluation.* = **R1 + R4 + R6**, with
**R2** as the stretch contribution and **R3/R5/R7/R8** as follow-ups.

Why this combination: R1 supplies the method, R4 supplies a theory-backed diagnostic and the link to
your rate story, R6 turns the "benchmarks barely test time" gap into evidence your objective matters.

**First things to read** (highest value per hour): S2P2, FIM-PP, EVA appendix H, POYO (self-supervised
discussion), NDT3 limitations, the IBL universal-translator paper, GEP stage 2, the spike-tokenizer
evaluation paper, Auto-STPP.

**First things to settle before coding:** likelihood factorisation over (patch, position, polarity),
tie-breaking, the compensator estimator, and which downstream sets/shifts to report. Keep a
simulated-pretraining path (v2e/ESIM) in the plan, because real unlabelled hours are scarce.

## 13. What I could not verify

* Appendix limitations for EVA, Event-SSM, Zubić, GEP, POYO (text beyond ~100k characters not read).
* Spiking Patches, Neural Events, Scaling Dense, tokenizer-eval, Neural Pile: abstract only.
* NDT3 and Neuroformer: search summaries only.
* Whether any 2026 paper already trains a raw-event point-process likelihood. My searches found none, but
  search coverage is limited, so run a final check on arXiv (cs.CV / stat.ML) before committing.
