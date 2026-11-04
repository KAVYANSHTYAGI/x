// evkit: high-performance kernels and I/O for event-stream point-process models.
#pragma once
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__CUDA__)
#define EVKIT_HD __host__ __device__ __forceinline__
#define EVKIT_DEVICE __device__ __forceinline__
#else
#define EVKIT_HD inline
#define EVKIT_DEVICE inline
#endif

#define EVKIT_RESTRICT __restrict__

namespace evkit {

constexpr float kLog2e = 1.4426950408889634f;

// Shapes of the event-driven SSM scan (all tensors row-major, float32 unless noted):
//   v      (B, L, D)      drive values (input projection of the tokens)
//   dt     (B, L)         effective elapsed time before each event (clock-scaled), seconds
//   rate   (D, N)         decay rates r > 0 (1/s)
//   Bm, Cm (D, N)         input / output matrices of the diagonal system
//   skip   (D)            feed-through
//   h0     (B, D, N)      initial state (nullable -> zeros)
// Recurrence (impulse at each event, exact decay in between):
//   h_i = exp(-rate * dt_i) * h_{i-1} + Bm * v_i ;   y_i = sum_n Cm * h_i + skip * v_i
// Outputs: y (B, L, D), H_save (B, L, n_save, N) states of the first n_save channels, h_last (B, D, N).
struct ScanShape {
  int64_t B, L, D, N;
  int64_t n_save;  // number of leading channels whose per-event states are saved (0 = none)
};

// Readout of decayed states at arbitrary query times (the continuous-time part of the heads):
//   Y[b,l,s,d] = sum_n Cm[d,n] * exp(-rate[d,n] * tau[b,l,s]) * H[b,l,d,n]
//   H (B, L, n, N), tau (B, L, S), rate/Cm (n, N)  ->  Y (B, L, S, n)
struct ReadoutShape {
  int64_t B, L, S, n, N;
};

}  // namespace evkit
