# C++/CUDA kernels

Source: `csrc/atl_ops.h` (declarations and layouts), `csrc/atl_cpu.cpp` (OpenMP + bindings), `csrc/atl_cuda.cu`
(CUDA).

## Layouts
All tensors are contiguous float32 except indices (int64).

| Tensor | Shape | Meaning |
|---|---|---|
| `pos` | (N, 2) | x, y |
| `w` | (N) | per-event weight |
| `bidx` | (N) | batch index |
| `img` | (B, H, W) | image |

## Kernels

| Kernel | Maths | Parallelism | Backward |
|---|---|---|---|
| `splat_bilinear` | img[b, y, x] += w · bilinear weights of pos | one thread per event; atomic adds | gather of grad_img → d pos, d w (no atomics) |
| `splat_gauss` | img += w · N(p; pos, σ²), \|p − round(pos)\| ≤ R = ⌈3σ⌉, normalised by 2πσ² | one thread per event; (2R+1)² atomics | gather (no atomics) |
| `gather_bilinear` | val = bilinear sample, zero outside | one thread per event | d img by atomics, d pos by bilinear derivatives |
| `ats_pair_fwd` | S_i = Σ_{j≠i, \|Δt\|≤τ, \|r\|≤3h} exp(−\|r\|²/2h²), r = x̃_i − (y_j + v_j(t_i − c_s)) | one thread per anchor; loops over a 3×3 cell neighbourhood in its slice's hash grid | — |
| `ats_pair_bwd` | ∂S_i/∂v_j = k·r·(t_i − t_j)/h² | one thread per anchor; atomics into grad_v[j] | — |
| `voxel_grid` | trilinear polarity voting in (x, y, t) | one thread per event; atomics | none |

## The anchored pair plan (built in Python)

`atl/ops/functional.py::build_pair_plan` turns the all-pairs sum into local work.
1. Time is cut into S slices of width Δ, with centres c_s.
2. Every source j gets one entry per slice whose centre is within τ + Δ/2 of t_j, holding
   `y_j = x_j + v_j (c_s − t_j)`, its position at the slice centre.
3. For an anchor i in slice s, the exact residual is `x̃_i − (y_j + v_j (t_i − c_s))`. The correction term is at
   most vmax·Δ/2, so with cell size `3h + vmax·Δ/2` every contributing source lies in the anchor's 3×3 cell
   neighbourhood.
4. Entries are sorted by key `((b·S + s)·GH + cy)·GW + cx`, and CSR offsets come from `bincount` + `cumsum`.

The C++ and CUDA kernels only walk those lists.

**Cost:**
* plan: O(N · (2τ/Δ + 1)) entries;
* pairs: O(M · k̄), with k̄ = mean neighbours (hundreds at h = 1 px on aligned edges).

**Memory:** about 28 bytes per plan entry, i.e. ~0.45 GB for N = 1M, S = 16, τ = 1.
* Reduce with τ < 1, fewer slices (larger cells), or source subsampling.

## Verification status
* CPU kernels: values and gradients equal to the torch references (`tests/test_ops.py`, 9 tests).
* CUDA kernels: compile for sm_80 and sm_90 with clang (device and host). 22–48 registers, **0 spill bytes**.
  **Never executed in development (no GPU)**: run `pytest -q` on a GPU machine; the tests parametrise over `cuda`
  automatically.

Compile-only check without a GPU (as used in development):
```
clang++ -x cuda --cuda-device-only --cuda-gpu-arch=sm_80 --cuda-path=<cuda root> -std=c++20 -O3 \
  -DATL_WITH_CUDA $(python -c "import torch.utils.cpp_extension as c;print(' '.join('-I'+p for p in c.include_paths()))") \
  -I<python include> -Icsrc -S -o atl.ptx csrc/atl_cuda.cu
ptxas -arch=sm_80 -O3 -v atl.ptx -o /dev/null      # registers / spills
```
A CPU-only torch wheel lacks `c10/cuda/impl/cuda_cmake_macros.h`. For this check only, supply a one-line shim:
`#define C10_CUDA_BUILD_SHARED_LIBS`.

## Measured timings (CPU, 4 threads, 100k events, 240×320; `bench/bench_ops.py`)

| Op | fwd + bwd |
|---|---|
| bilinear splat | 4.4 ms |
| Gaussian splat | 45 ms |
| gather | 5.2 ms |
| pair kernel, 20k anchors | 251 ms |
| **full loss: MotionPriorCMax** | 1253 ms (KNN dominates) |
| **full loss: ours** | 556 ms |

GPU numbers: run `python bench/bench_ops.py --device cuda --events 1000000`.

## Possible optimisations (not yet needed)
* Shared-memory staging of a block's neighbour cells.
* fp16 distances with fp32 accumulation.
* Reusing the plan across jitter draws.
* Warp-aggregated atomics in the backward pass.
