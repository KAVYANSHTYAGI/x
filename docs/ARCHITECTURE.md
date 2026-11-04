# Architecture

## 1. Layers

Each layer depends only on the layers below it.

```
 scripts/                      train.py · tools/ · experiments/           (entry points, no library logic)
   │
 atl/train  atl/estimators  atl/suite  atl/eval                           (procedures: learn / fit / test / score)
   │
 atl/objectives                                                           (losses: what "aligned" means)
   │        ╲
 atl/motion  atl/models                                                   (motion parametrisations; networks)
   │
 atl/ops  ──►  csrc/ (C++/OpenMP + CUDA)                                  (differentiable kernels)
   │
 atl/core  (EventPacket)        atl/data (loaders, simulator) ──► atl/core
```

Design rules:
* **Objectives never know where motion comes from.** An objective receives `(events, motion)` and only calls the
  `Motion` interface. The same loss therefore works for a 1-parameter zoom, a dense grid, or a network's trajectory
  field.
* **Motion never knows how it will be scored.** A motion model answers one question: "where is the scene point seen
  by event (x, t) at time t′?"
* **Kernels are stateless and layout-documented.** Python builds plans (sorting, hashing) with torch ops, so they
  work on CPU and GPU. C++/CUDA does only the hot loops, and each has a pure-torch reference used in tests.
* **Every comparison is controlled.** In training only `loss.name` differs between baseline and ours
  (`configs/dsec_mpcm.yaml` vs `configs/dsec_ats.yaml`).

## 2. Data flow

### Learning (DSEC / EVIMO2)
```
DSECFlowDataset[i] ─► dict(x, y, t, p, flow, valid)        atl/data/dsec.py  (rectified, t ∈ [0,1] per 100 ms window)
      │ collate
      ▼
EventPacket (packed, sorted by (b, t))                      atl/core/events.py
      │ voxel_grid (15 bins, trilinear, mean/std)           atl/ops  (C++/CUDA)
      ▼
UNet(15 → 2K) ─► coefficient grid (B, 2K, H, W)             atl/models/unet.py
      │
      ▼
PixelTrajectories (one trajectory per 4×4 tile)             atl/motion/trajectories.py
      │                                    ┌─ MotionPriorCMaxLoss (KNN look-up table, focus, TV)   atl/objectives/mpcm.py
      ├──────── objective(events, motion) ─┤
      │                                    └─ AnchoredExact / AnchoredSliced (ours)              atl/objectives/ats.py
      ▼
loss.backward() ─► Adam                                     atl/train/trainer.py
validation: motion.flow_field ─► EPE / AE / NPE             atl/eval/flow_metrics.py
test:       motion.flow_field ─► 16-bit PNG                 scripts/tools/dsec_submit.py
```

### Model-based (per window, no network)
```
EventPacket + Motion(theta)   (Translation / Zoom / Affine / DenseLinearFlow / PixelTrajectories(grid))
      └─► atl.estimators.fit(objective, motion_fn, [theta], events) ─► theta*  ─► compare with ground truth
```
Used by the failure suite (`atl/suite`), the dense-drift diagnostics and the DSEC model-based comparison.

## 3. Conventions (read before touching code)

| Quantity | Convention |
|---|---|
| Pixel coordinates | `xy[:, 0] = x` (column), `xy[:, 1] = y` (row); pixel centres at integers |
| Time | normalised to [0, 1] inside each window; `EventPacket.duration` holds seconds |
| Velocity | pixels per window (normalised time unit) |
| Polarity | float ±1 |
| Batching | packed arrays + `b` batch index, **sorted by (b, t)** (`window_counts` relies on it) |
| Flow tensors | (B, 2, H, W) with channel 0 = x, channel 1 = y |
| Trajectory coefficients | MotionPriorCMax order: channels [y₁…y_K, x₁…x_K]; tile centres at `tile//2 :: tile` |
| DSEC flow PNG | R = x·128 + 2¹⁵, G = y·128 + 2¹⁵, B = valid (OpenCV loads BGR) |
| Losses | always **minimised**; anchored scores return `loss = −score` |

## 4. Core interfaces

```python
class Motion(nn.Module):                                # atl/motion/base.py
    linear_at_event: bool                               # True -> x' = x + v (t' - t) exactly (exact pair kernel)
    def transport(xy, t, t_dst, b) -> xy'               # required
    def velocity(xy, t, b) -> v                         # if linear_at_event
    def flow_field(B, H, W, t0=0, t1=1) -> (B,2,H,W)    # prediction / evaluation
    def transport_logdet(xy, t, t_dst, b)               # transport + log|det J| (finite differences by default)
    def regulariser_fields(B, H, W)                     # field for smoothness regularisers (or None)

class Objective:                                        # atl/objectives/base.py
    def __call__(events, motion) -> {"loss": tensor, ...diagnostics}
    reference_frame: bool                               # True for the CMax family, False for anchored scores

atl.objectives.build(name, **kw)                        # registry: cmax_var, cmax_multiref, cmax_gradmag_rand,
                                                        # shiba_rd, shiba_ra, emin, stppp, zhu_ts, taming_ts,
                                                        # mpcm, ats_exact, ats_sliced
```

## 5. How to extend

**A new objective.**
1. Subclass `Objective` in `atl/objectives/` and return `{"loss": ...}`.
2. Register it in `atl/objectives/__init__.py::build`.
3. Add it to `atl/suite/tasks.py::objectives()`. Run `scripts/experiments/scan_collapse.py` and
   `diag_dense_drift.py --objective <name>` to see how it behaves on the failure tasks.

**A new motion model.**
* Parametric autonomous field: subclass `VectorFieldMotion` and implement `field(xy, b)`, plus `affine_parts(b)` if
  the field is affine (enables the exact matrix-exponential flow); register it in `MODELS`.
* Anything else: subclass `Motion` and implement `transport`.

**A new dataset.**
1. Return dict items with `x, y, t (∈[0,1]), p (±1), name, seq, file_index, duration` (+ `flow`, `valid` if
   labelled).
2. Reuse `atl.data.dsec.collate(items, H=..., W=...)`.
3. Add a branch to `atl.train.trainer.build_datasets`.

**A new kernel.**
1. Add the CPU loop to `csrc/atl_cpu.cpp` and the mirrored CUDA kernel to `csrc/atl_cuda.cu`; declare both in
   `csrc/atl_ops.h`; bind in `PYBIND11_MODULE`.
2. Add a pure-torch reference in `atl/ops/reference.py` and an autograd wrapper in `atl/ops/functional.py`.
3. Add a value + gradient equality test in `tests/test_ops.py`.

## 6. Build and runtime switches

| Variable | Effect |
|---|---|
| `ATL_FORCE_REF=1` | use the pure-torch reference ops (no compilation) |
| `ATL_NO_CUDA=1` | build / use the CPU extension even if CUDA is available |
| `ATL_BUILD_DIR=<dir>` | JIT build cache (default `~/.cache/atl_ext/{cpu,cuda}`) |

## 7. Randomness and determinism

* Stochastic losses take a `seed` (anchor subsampling, anchor smoothing, slice selection). The MotionPriorCMax loss
  draws `t_ref` from torch's global RNG, as the official code does.
* **Comparisons between objectives must use deterministic evaluators** (fixed reference-time grids, all anchors,
  averaged jitter seeds). See `scripts/experiments/exp_dense_trajectories.py::evaluator`.
* **Bias claims must use held-out events** (a fresh simulation with `event_seed`), never in-sample loss values.
