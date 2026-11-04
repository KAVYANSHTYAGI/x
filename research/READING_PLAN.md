# Reading plan for Option 1 (raw-event point-process pretraining)

Ordered by "decision this paper informs". Time estimates are for a careful read of the parts that matter,
not the whole paper. Tick the question boxes as you answer them; the answers become design decisions in
`evpp/` or related-work text.

## Tier 1 - read before writing any real-data code (about 12 h)

| # | Paper | Hours | Read for | Questions to answer |
|---|---|---:|---|---|
| 1 | [S2P2](https://arxiv.org/abs/2412.19634) (NeurIPS'25) | 2 | Latent linear Hawkes layer, MC compensator, parallel scan, ablations | - [ ] How do they compute the likelihood integral and with how many samples? - [ ] Do input-dependent decays help and why? - [ ] What breaks on simultaneous events? (our `evpp` dequantises; is that enough?) |
| 2 | [FIM-PP](https://arxiv.org/abs/2509.24762) (ICLR'26) | 2 | Synthetic prior design, in-context inference | - [ ] How were the 72K processes sampled? Could we sample *sensor* parameters the same way (R2)? - [ ] What fails zero-shot (Taxi marks) and what fixes it? |
| 3 | [EVA](https://arxiv.org/abs/2505.11165) (ICLR'26), **esp. Appendix H** | 2 | Per-event tokens + RWKV, why inter-event deltas not absolute time, their stated limitations | - [ ] Which SSL target mattered (count vs time surface vs next)? - [ ] Their latency/throughput breakdown: what would per-event likelihood cost? |
| 4 | [POYO](https://arxiv.org/abs/2310.16046) + [NDT3](https://www.biorxiv.org/content/10.1101/2025.02.02.634313) | 3 | Per-spike tokens, unit embeddings, RoPE time; NDT3's statement that scaling does not fix sensor variability | - [ ] Which tokenisation choices were ablated (none in POYO)? - [ ] What exactly did NDT3 find about sensor variability and output stereotypy, and is our rate issue the same? |
| 5 | [GEP](https://arxiv.org/abs/2603.23032) stage 2 | 1.5 | Autoregressive pretraining on event+image tokens: what is predicted, how time bins are used | - [ ] Exactly where does our objective differ (likelihood on raw events vs latent MSE on pseudo-frames)? - [ ] Their Event-1.8M composition: leakage with downstream sets? |
| 6 | [Auto-STPP](https://arxiv.org/abs/2310.06179) | 1.5 | Exact 3-D compensator via ProdNet | - [ ] Could a separable intensity replace our quadrature for the spatial part? |

## Tier 2 - positioning and baselines (about 10 h)

| # | Paper | Hours | Read for |
|---|---|---:|---|
| 7 | [IBL "universal translator"](https://arxiv.org/abs/2407.14668) | 2 | Masked multi-task pretraining on single spikes; how rate drift across sessions is handled |
| 8 | [Neural Pile + tokenizer evaluation](https://neurips.cc/virtual/2026/oral/139126) | 1 | Their four desiderata for tokenisers and the finding that reconstruction and utility diverge; reuse the metric set |
| 9 | [Event-SSM](https://arxiv.org/abs/2404.18508) | 1.5 | Their Δt discretisation vs our impulse-decay form; test-set selection caveat |
| 10 | [Zubić et al.](https://arxiv.org/abs/2402.15584) | 1.5 | Learnable timescales + aliasing, frequency generalisation protocol |
| 11 | [TESPEC](https://arxiv.org/abs/2508.00913) | 1.5 | Masked-reconstruction baseline, how to match its compute |
| 12 | [Mamba Hawkes](https://arxiv.org/abs/2407.05302), [Neuroformer](https://arxiv.org/abs/2311.00136) | 1.5 | Selective-SSM intensity; autoregressive spike generation |
| 13 | [Tallec & Ollivier](https://arxiv.org/abs/1804.11188) | 1 | The time-warp invariance argument behind the latent clock |

## Tier 3 - skim when the corresponding module becomes relevant

* [Neural STPP](https://arxiv.org/abs/2011.04583), [Add-Thin](https://arxiv.org/abs/2311.01139), [EdiTPP](https://arxiv.org/abs/2510.06050), [EventFlow](https://arxiv.org/abs/2410.07430): non-autoregressive generative TPPs. Relevant if we add a generation / forecasting task.
* [Neural-ODE threshold ID](https://arxiv.org/abs/2603.05011), [Ev-NeRF](https://arxiv.org/abs/2206.12455), [asynchronous Kalman filter](https://arxiv.org/abs/2012.05590): physics side for V4.
* [N-ImageNet](https://arxiv.org/abs/2112.01041), [Ev-TTA](https://arxiv.org/abs/2203.12247): the real paired-shift benchmark and the incumbent adaptation method.
* [Event2Vec](https://arxiv.org/abs/2504.15371) (ICML'26): shows the venue accepts event tokenisation; check what they claim about throughput.
* [Spiking Patches](https://arxiv.org/abs/2510.26614), [Neural Events](https://arxiv.org/abs/2606.19835), [Scaling Dense Event-Stream Pretraining](https://arxiv.org/abs/2603.03969): only abstracts were read for the atlas; confirm their numbers before citing.

## Final novelty check (do last, 1 h)

Search arXiv for: "event camera" with each of "temporal point process", "log-likelihood pretraining",
"next-event prediction", "intensity function"; also cs.NE and q-bio.NC for spike-camera or neural-spike crossovers.
If something close appears, the atlas section 11 (R1) must be re-scoped before any experiments.
