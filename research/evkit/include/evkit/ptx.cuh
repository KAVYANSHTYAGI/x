// Device helpers, including the few places where inline PTX is worth it.
#pragma once
#include <cuda_runtime.h>

#include "common.hpp"

namespace evkit::gpu {

// 2^x with the SFU's approximate instruction (max rel. error ~2^-22, flushes denormals).  This is the
// single most frequent instruction of the scan (one per (event, channel, state)), and exp2f / __expf
// add range-reduction instructions we do not need because the argument is always <= 0.
__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// exp(-r * t) given neg_r_log2e = -r * log2(e).  FAST=false uses exp2f (accurate) for validation.
template <bool FAST>
__device__ __forceinline__ float exp_neg(float neg_r_log2e, float t) {
  const float x = neg_r_log2e * t;
  if constexpr (FAST) return ex2_approx(x);
  else return exp2f(x);
}

// Read-only, streamed-once global load that does not allocate in L1 (keeps L1 for the reused tiles).
__device__ __forceinline__ float ld_stream(const float* p) {
  float v;
  asm("ld.global.nc.L1::no_allocate.f32 %0, [%1];" : "=f"(v) : "l"(p));
  return v;
}

// Butterfly sum over aligned groups of G lanes (G a power of two <= 32); every lane gets the group sum.
template <int G>
__device__ __forceinline__ float group_sum(float v) {
#pragma unroll
  for (int o = G / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

}  // namespace evkit::gpu
