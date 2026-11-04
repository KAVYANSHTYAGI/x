// Decayed-state readout at query times (used for the compensator quadrature and arrival-time readout):
//   Y[it, s, d] = sum_m Cm[d,m] exp(-rate[d,m] tau[it, s]) H[it, d, m]          it = flattened (b, l)
// Fusing the exponential with the reduction avoids materialising the (B, L, S, n, N) tensor that the
// PyTorch broadcast version builds - an N-fold memory saving and one kernel instead of four.
// Block = n*N threads (one per (d, m)); grid-stride over items; outputs staged through shared memory.
#include <stdexcept>
#include <string>

#include "evkit/cuda_ops.hpp"
#include "evkit/ptx.cuh"

namespace evkit::gpu {
namespace {

#define EVKIT_CUDA_CHECK(expr)                                                                   \
  do {                                                                                           \
    cudaError_t _e = (expr);                                                                     \
    if (_e != cudaSuccess) throw std::runtime_error(std::string("evkit CUDA: ") + cudaGetErrorString(_e)); \
  } while (0)

constexpr int kMaxS = 64;
constexpr int kGrid = 1024;

template <int N, bool FAST>
__global__ void readout_fwd_kernel(ReadoutShape s, const float* __restrict__ H, const float* __restrict__ tau,
                                   const float* __restrict__ rate, const float* __restrict__ Cm, float* __restrict__ Y) {
  __shared__ float tau_s[kMaxS];
  __shared__ float y_s[kMaxS * 32];
  const int tid = threadIdx.x, d = tid / N, m = tid % N;
  const int nN = static_cast<int>(s.n) * N, S = static_cast<int>(s.S), n = static_cast<int>(s.n);
  const float nrl = -rate[tid] * kLog2e, c = Cm[tid];
  for (int64_t it = blockIdx.x; it < s.B * s.L; it += gridDim.x) {
    const float h = H[it * nN + tid];
    for (int k = tid; k < S; k += blockDim.x) tau_s[k] = tau[it * S + k];
    __syncthreads();
    const float ch = c * h;
    for (int k = 0; k < S; ++k) {
      const float val = group_sum<N>(ch * exp_neg<FAST>(nrl, tau_s[k]));
      if (m == 0) y_s[k * n + d] = val;
    }
    __syncthreads();
    for (int j = tid; j < S * n; j += blockDim.x) Y[it * S * n + j] = y_s[j];
    __syncthreads();
  }
}

template <int N, bool FAST>
__global__ void readout_bwd_kernel(ReadoutShape s, const float* __restrict__ H, const float* __restrict__ tau,
                                   const float* __restrict__ rate, const float* __restrict__ Cm,
                                   const float* __restrict__ gY, float* __restrict__ gH, float* __restrict__ gtau,
                                   float* __restrict__ part_r, float* __restrict__ part_c) {
  __shared__ float tau_s[kMaxS];
  __shared__ float gy_s[kMaxS * 32];
  __shared__ float red[kMaxS * 32];                       // [S][warps]  (<= 32 warps)
  const int tid = threadIdx.x, d = tid / N, lane = tid & 31, warp = tid >> 5;
  const int warps = blockDim.x >> 5;
  const int nN = static_cast<int>(s.n) * N, S = static_cast<int>(s.S), n = static_cast<int>(s.n);
  const float r = rate[tid], nrl = -r * kLog2e, c = Cm[tid];
  float acc_r = 0.f, acc_c = 0.f;
  for (int64_t it = blockIdx.x; it < s.B * s.L; it += gridDim.x) {
    const float h = H[it * nN + tid];
    for (int k = tid; k < S; k += blockDim.x) tau_s[k] = tau[it * S + k];
    for (int j = tid; j < S * n; j += blockDim.x) gy_s[j] = gY[it * S * n + j];
    __syncthreads();
    float gh = 0.f;
    for (int k = 0; k < S; ++k) {
      const float e = exp_neg<FAST>(nrl, tau_s[k]);
      const float g = gy_s[k * n + d];
      gh = fmaf(g * c, e, gh);
      const float ghe = g * h * e;
      acc_c += ghe;
      acc_r = fmaf(-tau_s[k] * c, ghe, acc_r);
      const float gt = group_sum<32>(-r * c * ghe);
      if (lane == 0) red[k * warps + warp] = gt;
    }
    gH[it * nN + tid] = gh;
    __syncthreads();
    for (int k = tid; k < S; k += blockDim.x) {
      float acc = 0.f;
      for (int w = 0; w < warps; ++w) acc += red[k * warps + w];
      gtau[it * S + k] = acc;
    }
    __syncthreads();
  }
  part_r[int64_t(blockIdx.x) * nN + tid] = acc_r;
  part_c[int64_t(blockIdx.x) * nN + tid] = acc_c;
}

__global__ void reduce_rows_kernel2(const float* __restrict__ in, int64_t rows, int64_t cols, float* __restrict__ out) {
  for (int64_t j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; j < cols; j += int64_t(gridDim.x) * blockDim.x) {
    float acc = 0.f;
    for (int64_t r = 0; r < rows; ++r) acc += in[r * cols + j];
    out[j] = acc;
  }
}

void validate(const ReadoutShape& s) {
  if (s.N != 8 && s.N != 16 && s.N != 32) throw std::runtime_error("evkit: readout N must be 8, 16 or 32");
  if ((s.n * s.N) % 32 != 0 || s.n * s.N > 1024) throw std::runtime_error("evkit: readout needs n*N % 32 == 0 and <= 1024");
  if (s.S > kMaxS) throw std::runtime_error("evkit: readout supports at most 64 query times per item");
  if (s.S * s.n > kMaxS * 32) throw std::runtime_error("evkit: readout S*n too large");
}

int grid_for(const ReadoutShape& s) { return static_cast<int>(std::min<int64_t>(s.B * s.L, kGrid)); }

}  // namespace

size_t readout_bwd_workspace_floats(const ReadoutShape& s) { return 2 * static_cast<size_t>(kGrid * s.n * s.N); }

void readout_fwd(const ReadoutShape& s, const float* H, const float* tau, const float* rate, const float* Cm,
                 float* Y, bool fast, void* stream) {
  validate(s);
  auto st = static_cast<cudaStream_t>(stream);
  const int threads = static_cast<int>(s.n * s.N), grid = grid_for(s);
  if (grid == 0) return;
#define EVKIT_RF(NN) (fast ? readout_fwd_kernel<NN, true><<<grid, threads, 0, st>>>(s, H, tau, rate, Cm, Y) \
                           : readout_fwd_kernel<NN, false><<<grid, threads, 0, st>>>(s, H, tau, rate, Cm, Y))
  switch (s.N) { case 8: EVKIT_RF(8); break; case 16: EVKIT_RF(16); break; default: EVKIT_RF(32); }
#undef EVKIT_RF
  EVKIT_CUDA_CHECK(cudaGetLastError());
}

void readout_bwd(const ReadoutShape& s, const float* H, const float* tau, const float* rate, const float* Cm,
                 const float* gY, float* gH, float* gtau, float* grate, float* gC, float* ws, bool fast, void* stream) {
  validate(s);
  auto st = static_cast<cudaStream_t>(stream);
  const int threads = static_cast<int>(s.n * s.N), grid = grid_for(s);
  if (grid == 0) return;
  float* part_r = ws;
  float* part_c = ws + int64_t(kGrid) * s.n * s.N;
#define EVKIT_RB(NN) (fast ? readout_bwd_kernel<NN, true><<<grid, threads, 0, st>>>(s, H, tau, rate, Cm, gY, gH, gtau, part_r, part_c) \
                           : readout_bwd_kernel<NN, false><<<grid, threads, 0, st>>>(s, H, tau, rate, Cm, gY, gH, gtau, part_r, part_c))
  switch (s.N) { case 8: EVKIT_RB(8); break; case 16: EVKIT_RB(16); break; default: EVKIT_RB(32); }
#undef EVKIT_RB
  const int64_t cols = s.n * s.N;
  reduce_rows_kernel2<<<static_cast<int>((cols + 255) / 256), 256, 0, st>>>(part_r, grid, cols, grate);
  reduce_rows_kernel2<<<static_cast<int>((cols + 255) / 256), 256, 0, st>>>(part_c, grid, cols, gC);
  EVKIT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace evkit::gpu
