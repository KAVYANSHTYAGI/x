# Code guide: every file explained

Read [`ARCHITECTURE.md`](ARCHITECTURE.md) first for the conventions (x/y order, normalised time, packed events).

---

## `atl/core`

### `events.py` — `EventPacket`
The single container passed between all modules.
* **Fields:**
  * `xy` (N, 2): x, y in pixels;
  * `t` (N): time normalised to [0, 1];
  * `p` (N): polarity ±1;
  * `b` (N): batch index;
  * `batch_size`, `height`, `width`;
  * `duration`: seconds per window.
* **Invariant:** events are sorted by (b, t). `from_arrays` sorts a single sample; `collate` concatenates samples in
  order.
* **Helpers:**
  * `select(mask)`: subset of events;
  * `split(frac)`: random disjoint split, used for held-out evaluation;
  * `subsample(k)`;
  * `inside(xy, margin)`: image-box test. A negative margin shrinks the box (used for the anchor border margin).

---

## `atl/ops` — differentiable kernels

### `_ext.py`
JIT-compiles `csrc/` with `torch.utils.cpp_extension.load` on first use:
* CPU always; CUDA if available;
* cache directory `~/.cache/atl_ext`;
* falls back to the reference ops if the build fails or `ATL_FORCE_REF=1`.

`ext_supports(t)` says whether the compiled module handles a tensor's device.

### `reference.py`
Pure-torch definitions of every kernel. **They are the specification**: tests check C++/CUDA values and gradients
against them.
* `splat_bilinear`, `splat_gauss` (truncated at ceil(3σ), normalised by 2πσ²), `gather_bilinear` (zero padding),
  `voxel_grid`.
* `ats_pair_sums`: brute force O(M·N). For each anchor i it computes
  `S_i = Σ_{j≠i, |t_i−t_j|≤τ, |r|≤3h} exp(−|r|²/2h²)`, where `r = x̃_i − x_j − v_j (t_i − t_j)`.

### `functional.py`
Autograd wrappers that dispatch to the extension or the reference.
* `splat`, `gather`, `blur` (separable Gaussian), `voxel_grid` (+ mean/std normalisation over non-zero voxels, as in
  MotionPriorCMax).
* **`build_pair_plan`**: the host-side spatial hash for the exact anchored kernel.
  1. Partition time into S slices.
  2. For each source j, create one entry per slice s with |t_j − c_s| ≤ τ + Δ/2, holding the position transported
     to the slice centre, `y_j = x_j + v_j (c_s − t_j)`.
  3. Bucket the entries into square cells of size `3h + vmax·Δ/2` over the anchors' bounding box. That size
     guarantees every source within 3h of an anchor (after the ≤ Δ/2 residual move) sits in the anchor's 3×3 cell
     neighbourhood.
  4. Sort by key (batch, slice, cell) and build CSR offsets with `bincount` + `cumsum`.

  Everything here uses torch ops, so the same code runs on CPU and GPU.
* `ats_pair_sums(..., anchor_xy=None)`: the exact sums with a gradient w.r.t. the per-event velocities `v` only (the
  data are fixed). `anchor_xy` lets the anchored score pass smoothed/dequantised anchor positions.
* `window_counts`: the number of other events within ±τ of each anchor, via `searchsorted` on a (b, t) key. This
  relies on the sort invariant.

---

## `csrc` — see [`KERNELS.md`](KERNELS.md)

| File | Contents |
|---|---|
| `atl_ops.h` | declarations and tensor layouts |
| `atl_cpu.cpp` | OpenMP loops (the readable specification) + pybind11 bindings with CPU/CUDA dispatch |
| `atl_cuda.cu` | the same loops, one thread per event or anchor |

---

## `atl/motion` — how events are transported

### `base.py`
* `Motion` interface (`transport`, `velocity`, `flow_field`, `transport_logdet`, `regulariser_fields`,
  `linear_at_event`).
* `transport_logdet` computes log|det ∂transport/∂x| by central differences. The sliced anchored score uses it for
  the small within-slice step (exact change of variables).

### `parametric.py`
Autonomous velocity fields v(x; θ), batched as θ of shape (B, k).
* **Models:**
  * `Translation`, `Zoom`, `TranslationZoom` (the collapse test), `Sim2`, `Affine`;
  * `PlanarFlow` (8-parameter linearised homography);
  * `Rotation3D` (rotational interaction matrix, intrinsics K);
  * `DenseLinearFlow` (bilinear velocity grid with `upsample()` for coarse-to-fine).
* **`mode="linear"`** is the classic CMax warp x′ = x + v(x)Δt. It is linear at the event, so the exact pair kernel
  applies.
* **`mode="exact"`** is the true flow. Affine fields use the augmented matrix exponential
  `exp([[A, b], [0, 0]] Δt)` (robust for large |AΔt|); other fields use RK4. This is the setting of the theory: it has
  the group property, and collapse paths are exponential.
* `sample_grid`: differentiable bilinear field lookup with border clamping, shared with the trajectories.

### `trajectories.py` — MotionPriorCMax's representation
* **Model:** one trajectory per 4×4 tile starting at the tile centre at t = 0:
  `q(p, t) = p + Σ_k c_k(p) g_k(t)`, with basis `g = t^k` (polynomial) or Bernstein `B_{k,K}` with control point 0
  pinned (Bezier).
* **Coefficients:** channel order follows MotionPriorCMax ([y…, x…]); `_coeff_at` interpolates bilinearly between
  tile centres.
* **Transport of an event (x, t):** solve `p + d(p, t) = x` by fixed-point iteration (`origin`), then evaluate at the
  target time. Exact for smooth fields; O(N) per target time.
* `velocity`: only for the degree-1 polynomial (then `linear_at_event = True`).
* `tile_trajectories`: the tile positions used by MotionPriorCMax's KNN look-up table.
* `flow_field`: tile flows, bilinearly upsampled to full resolution (evaluation).

---

## `atl/objectives` — what "aligned" means

### `base.py`
* `Objective`.
* `IWEConfig` and `make_iwe`: the image of warped events, with:
  * bilinear vote + optional blur, or a Gaussian splat;
  * polarity split;
  * border masking;
  * MotionPriorCMax's (1 − |t − t_ref|) weights.
* `sobel`.
* `ref_times`: `"random"` → one U(0, 1) draw per call; a float → fixed; a list → multi-reference.

### `cmax.py` — reference-frame baselines
| Class | Paper | Notes |
|---|---|---|
| `Sharpness(metric, t_ref, form)` | CMax (Gallego'18), multi-reference (Shiba'22), MotionPriorCMax's focus term | `metric` ∈ variance / mean_square / grad_l1 / grad_l2; `form` "neg" (model-based) or "inv" (1/G, learning) |
| `EMin` | Nunes & Demiris'20 | KDE entropy of the warped events by IWE look-up, leave-one-out |
| `STPPP` | Gu et al.'21 | per-pixel negative binomial of aligned counts (r = 0.1, q = 0.39) |
| `Timestamp(taming)` | Zhu'19 / Paredes-Vallés'23 | average-timestamp images; the Taming version is multi-reference and normalised by active pixels |
| `Regularized(base, reg, weight)` | Shiba'22 R_D / R_A; MotionPriorCMax TV | R_D uses div v·(t_ref − t), the first-order log-det; R_A uses the finite-difference det; both penalise only beyond a dead zone (trimmed versions) |

Helpers:
* `charbonnier_tv`: MotionPriorCMax's smoothness.
* `_self_term`: an event's own contribution to a bilinear look-up of its Gaussian splat (used for leave-one-out).

### `mpcm.py` — the target loss, ported faithfully
Port of `src/losses/focus.py` with the official DSEC configuration (listed in the module docstring). The pipeline:
1. Tile trajectories at t_ref and at the 15 bin centres.
2. For each 4×4 look-up-table cell and bin, take the mean displacement-to-t_ref of its 32 nearest trajectories.
   KNN runs on pykeops if installed, otherwise chunked `cdist` + `topk`. Indices carry no gradient, as in the
   original.
3. Displace each event by its (bin, cell) entry.
4. Build the polarity-split IWE with σ = 1 blur, (1 − |t − t_ref|) weights and the border mask.
5. Loss = 1 / mean(|Sobel_x| + |Sobel_y|) + 0.003 · Charbonnier TV of the look-up table.

### `ats.py` — ours
* `visible_fraction`: the exposure correction. It is the fraction of the anchor's source event mass recorded while
  its content was inside the frame, checked at 16 slices and computed without gradient.
* **`AnchoredExact`:**
  * dequantise the events (U[−½, ½]²);
  * smooth the anchors (x̃ = x + h z), which makes the score proper at every h;
  * build the pair sums via the C++/CUDA kernel with per-event velocities;
  * normalise by the window counts × exposure;
  * score = `log(ε + S / (W · 2πh²))`.

  Options: `n_anchors` (subsampling), `anchor_margin` (raw-position border margin), and `set_bandwidth` for
  annealing.
* **`AnchoredSliced`:** any motion model. For each time slice:
  1. Transport all sources to the slice centre and splat them with a Gaussian.
  2. Move the anchors by the small step t_i → c_s, adding the exact log-det.
  3. Look the density up bilinearly.
  4. Subtract the anchor's own splat (`_cross_term` handles smoothed anchors).
* **Metrics:**
  * `heldout_score`: proper, ground-truth-free;
  * `flow_warp_loss` (FWL) and `rsat`: the baselines' metrics, shown to be fooled by collapse.

### `__init__.py`
`build(name, **kw)` registry, used by configs, the trainer and the suite.

---

## `atl/data`

| File | Contents |
|---|---|
| `synthetic.py` | Analytic `Texture`; `simulate(kind, θ, ...)` renders L(x, t) = L0(Φ_{t→0}(x)) and runs a per-pixel DVS threshold model (threshold mismatch, interpolated crossing times, optional noise). Returns events, the true motion, θ and the dense ground-truth flow. `event_seed` gives a fresh realisation of the same scene (held-out tests). |
| `synthetic_dataset.py` | Random translation / zoom / affine samples in DSEC item format (CPU smoke tests) |
| `dsec.py` | `EventSlicer` (fast `ms_to_idx` slicing of `events.h5`, blosc via hdf5plugin); `DSECSequence` (official or MotionPriorCMax layout; windows `gt` / `test` / `images` / `sliding`; rectification); `DSECFlowDataset`; `collate`; `read_flow_png` / `write_flow_png` (benchmark format); MotionPriorCMax's TRAIN / VAL / TEST splits |
| `evimo2.py` | `EVIMO2Scene` (MotionPriorCMax layout: npy events, `dataset_multiflow_10steps_vis.h5` tracks at 50 ms steps, object masks, 0.4 s context); `evaluate_tracks` (TEPE / TAE / %Out on object pixels) |

---

## `atl/models`

| File | Contents |
|---|---|
| `unet.py` | The 5-level U-Net matched to MotionPriorCMax's DSEC network (64…1024 channels, transposed-conv upsampling, BN, Kaiming init) |
| `bflow_adapter.py` | Converts the official RAFT-spline Bezier parameters into `PixelTrajectories`, so the EVIMO2 backbone can be trained with any loss. `check_against_official` must pass once (channel order) before use. |

---

## `atl/estimators/fit.py`
`fit(objective, motion_fn, params, events, steps, lr, optimizer, callback)`: Adam or L-BFGS on motion parameters.
`motion_fn` rebuilds the motion from the parameters (needed for `PixelTrajectories`). The callback supports annealing.

## `atl/train/trainer.py`
* `TrainConfig`: YAML-loadable. The loss is a dict for `objectives.build`; `anneal: [h0, h1, fraction]`.
* `Trainer`:
  * DDP-ready via torchrun;
  * per step: voxel grid → U-Net → `PixelTrajectories` → loss (optionally on a subsample, `max_events`);
  * gradient clipping;
  * JSONL log;
  * validation on GT windows (pixel-weighted EPE / AE / NPE);
  * `best.pt` / `last.pt` checkpoints.

## `atl/eval/flow_metrics.py`
* `flow_errors`: EPE, AE (angle of (u, v, 1)), and 1/2/3PE. "%Out" = 3PE.
* `trajectory_errors`.
* `Accumulator`: pixel-weighted global averages, as the benchmark computes them.

## `atl/suite/tasks.py`
* F1 collapse (path preference + optimisation from the truth).
* F2 reference bias.
* F3 λ dilemma.
* F4 metric ranking.

`objectives()` lists the deterministic versions compared.

---

## `scripts`

| Path | Purpose |
|---|---|
| `train.py` | Training entry point (`--config`, `--set key=value` overrides) |
| `tools/convert_mpcm_checkpoint.py` | Official checkpoint → our U-Net (step zero: reproduce their numbers) |
| `tools/dsec_submit.py` | DSEC test inference → benchmark PNG folder |
| `experiments/scan_collapse.py` | One-screen collapse table for all losses |
| `experiments/run_suite.py` | F1–F4 → Markdown + JSON |
| `experiments/exp_dense_trajectories.py` | F6: dense trajectories from GT / zero, in-sample vs held-out |
| `experiments/diag_dense_drift.py` | Interior vs border held-out bias of a fitted dense field |
| `experiments/exp_dsec_modelbased.py` | Real DSEC per-window coarse-to-fine comparison |
| `experiments/run_all_evidence.sh` | Runs every CPU experiment sequentially |

## `tests`

| File | Contents |
|---|---|
| `test_ops.py` | Every kernel vs its reference: values and gradients, CPU (+ CUDA if available), several τ / slice counts |
| `test_objectives.py` | Exact-flow group property; trajectory inversion; sliced ≈ exact; properness along the zoom path; CMax collapse; exact-kernel gradient vs reference; MotionPriorCMax loss backprop; metric ranking; DSEC PNG round trip; finite-kernel bias fix (1-D) |

## Other directories

| Path | Contents |
|---|---|
| `bench/bench_ops.py` | Per-kernel and per-loss timings (forward + backward) |
| `results/` | Committed outputs behind every reported number (`results/README.md`) |
| `research/` | Earlier notes and prototypes (see `docs/RESEARCH_LOG.md`) |
