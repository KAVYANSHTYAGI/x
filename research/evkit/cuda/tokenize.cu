// GPU tokenisation: same function as the CPU path (tokenize_one is __host__ __device__), so outputs are
// bit-identical for identical (seed, index_offset).
#include <stdexcept>
#include <string>

#include "evkit/cuda_ops.hpp"
#include "evkit/ptx.cuh"
#include "evkit/tokenize.hpp"

namespace evkit::gpu {
namespace {

__global__ void tokenize_kernel(const int64_t* __restrict__ t, const uint16_t* __restrict__ x,
                                const uint16_t* __restrict__ y, const uint8_t* __restrict__ p, int64_t n,
                                TokenizeParams prm, int32_t* __restrict__ patch, int32_t* __restrict__ sub,
                                int32_t* __restrict__ pol, float* __restrict__ u, float* __restrict__ v,
                                float* __restrict__ dt) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t prev = i ? t[i - 1] : prm.t_prev;
    tokenize_one(prm, static_cast<uint64_t>(i), t[i], prev, x[i], y[i], p[i], patch[i], sub[i], pol[i], u[i], v[i], dt[i]);
  }
}

}  // namespace

void tokenize(const int64_t* t_us, const uint16_t* x, const uint16_t* y, const uint8_t* p, int64_t n,
              const TokenizeParams& prm, int32_t* patch, int32_t* sub, int32_t* pol, float* u, float* v, float* dt,
              void* stream) {
  if (n == 0) return;
  const int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>((n + threads - 1) / threads, 65535));
  tokenize_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(t_us, x, y, p, n, prm, patch, sub, pol, u, v, dt);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw std::runtime_error(std::string("evkit CUDA tokenize: ") + cudaGetErrorString(e));
}

}  // namespace evkit::gpu
