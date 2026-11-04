# evkit: C++/CUDA kernels and data path for event point-process models

High-performance backend for `../evpp` (the raw-event likelihood models): fused CUDA kernels with
hand-written backward passes for the two operations that dominate the model, plus a multithreaded C++
data path from raw sensor files to tokens.

## What is in here

| Component | Files | What it does |
|---|---|---|
| Event-driven SSM scan | `cuda/ssm_scan.cu`, `src/ssm_cpu.cpp` | `h_i = exp(-r dt_i) h_{i-1} + B v_i`, `y_i = C·h_i + skip·v_i`, forward + backward, never materialises the (B,L,D,N) state |
| Decayed readout | `cuda/readout.cu`, `src/ssm_cpu.cpp` | `Y[s] = Σ_m C exp(-r τ_s) H` at the compensator quadrature nodes, fused (N-fold less memory than the PyTorch broadcast) |
| Tokeniser | `include/evkit/tokenize.hpp`, `cuda/tokenize.cu` | raw (t, x, y, p) → (patch, sub-cell, polarity, u, v, dequantised dt); **bit-identical on CPU and GPU** via a shared Philox4x32-10 RNG |
| Raw decoders | `src/events.cpp` | Prophesee EVT 2.0 / EVT 3.0 `.raw`, Gen1/1Mpx `.dat`; chunk-parallel, exactly equal to a sequential decode |
| EVB store | `src/evb.cpp` | mmap-able struct-of-arrays event store with a 1 ms time index (O(1) window seeks) |
| PyTorch binding | `torch/evkit_torch.cpp`, `python/evkit/ops.py` | autograd ops; dispatch by tensor device; `EvPPConfig(impl="evkit")` switches the model over |
| Tools | `tools/evkit_convert.cpp`, `tools/convert_events.py` | raw/dat → EVB; HDF5 (DSEC, M3ED) and packed N-ImageNet-style shards |

## Kernel design (scan)

* **Mapping.** One thread per (batch, channel, state). The N states of a channel are N adjacent lanes,
  so `Σ_n C·h` is a `__shfl_xor` butterfly in registers. Blocks are 256 threads = 256/N channels.
* **Memory.** Time is tiled through shared memory in 32-event chunks (coalesced loads and stores of v, y,
  gy, gv). Streamed inputs use `ld.global.nc.L1::no_allocate` so L1 stays free for reused tiles. Per-event
  states are written only for the `n_save` channels the heads need (16 of 256 by default).
* **Inline PTX.** The decay `exp(-r·dt)` uses `ex2.approx.ftz.f32` (one SFU instruction; argument is
  always ≤ 0). `fast_exp=False` switches to `exp2f` for validation.
* **Two schedules.**
  * *Sequential* (enough blocks): each block walks the whole stream carrying h in a register, one pass,
    writing checkpoints on the fly.
  * *Chunk-parallel* (small batch, long streams): local scans, then a carry pass
    (`h_{c+1} = exp(-r Σdt_c) h_c + S_c`), then a finalise pass.

  `mode=0` picks automatically from the SM count.
* **Backward.** Each chunk is recomputed from its checkpoint into shared memory, then the adjoint
  `λ_i = C·gy_i + gH_i + a_{i+1} λ_{i+1}` runs in reverse. Parameter gradients accumulate in registers and
  are written as fixed-order partial sums: deterministic, no float atomics.
* **Register and shared-memory report** (`ptxas -v`, sm_80):

  | Kernels | Registers | Shared memory | Spills |
  |---|---:|---:|---:|
  | Forward | 31–38 | 2–8 KB | none |
  | Backward | 60–80 | 37–46 KB | none |
  | Readout | 30–32 | — | none |

  The backward pass fits about three to four 256-thread blocks per A100 SM.

## Status: what is verified, and where

| Check | Where | Result |
|---|---|---|
| C++ unit tests (`ctest`): Philox known-answer vectors; EVT2/EVT3/DAT round trip incl. two 24-bit time-base wraps and vector words; parallel decode == sequential for 1–16 threads; EVB time-range queries; tokeniser chunk/thread invariance; scan + readout gradients vs double-precision finite differences | this sandbox (CPU) | 5/5 pass |
| Python tests: scan values and all 7 gradients vs a float64 PyTorch reference at L = 1, 31, 32, 33, 100; readout; tokeniser; EVB; **full EvPP model, evkit backend == PyTorch backend (loss and every parameter gradient) for ar / intensity / physics** | this sandbox (CPU) | 24 pass, 7 CUDA-only skipped |
| CUDA compile: all kernels for sm_80 and sm_90 → PTX → SASS (clang 18 + ptxas 12.9), host objects compile | this sandbox | compiles, no spills |
| CUDA kernels vs CPU reference (`build/test_cuda`, `tests/python` CUDA section), GPU timings (`build/bench_cuda`) | **needs your GPU** | not run: there is no GPU here |

Measured on this sandbox (4 shared CPU cores; numbers will scale with cores and memory bandwidth):

| Path | Mevents/s |
|---|---:|
| EVT3 decode, 1 thread | ~70 |
| EVT3 decode, 4 threads | ~210 |
| EVT2 decode, 4 threads | ~280 |
| Tokenise, 4 threads | ~150–200 |

Before the allocation-free rewrite, EVT3 decoding ran at 9.8 Mev/s on one thread and 18 Mev/s on four.

## Build and run

```bash
# C++ library, tools, tests
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release [-DEVKIT_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="80;90"]
cmake --build build -j && (cd build && ctest --output-on-failure)
./build/test_cuda && ./build/bench_cuda          # on a GPU machine

# PyTorch extension (CUDA kernels included automatically when a GPU + CUDA_HOME are present)
pip install -e .                                  # or rely on the JIT fallback in python/evkit/_loader.py
python -m pytest -q tests/python

# use from the model
EvPPConfig(objective="intensity", impl="evkit")
```

## Known limits / next optimisation steps (in priority order, after GPU profiling)

1. Profile on the target GPU first (`nsys` / `ncu`). Everything below is a guess until then.
2. bf16 inputs/outputs for v / y / gy (accumulate in fp32), halving DRAM traffic of the memory-bound scan.
3. Fuse `in_proj` + gating into the scan prologue/epilogue (removes two (B,L,D) round trips per layer).
4. EVT3 seek index, so pretraining can read random windows straight from compressed `.raw` (≈1.7–2 B/event)
   instead of EVB (13 B/event). This matters for the TB-scale datasets.
5. d_state is restricted to 8 / 16 / 32 on CUDA; `(n_int·N) % 32 == 0` and ≤ 64 query times for the readout.
6. The LatentClock scan and the head MLPs still run in PyTorch. They are small (clock_dim = 8) or matmul-bound.
