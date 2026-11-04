// Event-driven diagonal SSM scan: forward and backward CUDA kernels.
//
//   h_i = exp(-r dt_i) h_{i-1} + Bm v_i ,  y_i = sum_n Cm h_i + skip v_i         (per batch b, channel d, state n)
//
// Thread mapping: one thread per (b, d, n); a block owns DT = 256/N channels x N states of one batch row,
// so the N states of a channel are N adjacent lanes and the reduction over n is a shuffle butterfly.
// Time is processed in tiles of kChunk events staged through shared memory (coalesced global traffic).
//
// Scheduling (chosen per call, see `pick_mode`):
//   kSequential: grid (D/DT, B, 1); every block walks all chunks carrying h in a register.  One pass over
//                v; checkpoints (state before each chunk) are written on the fly for the backward.
//   kChunked:    grid (D/DT, B, Z); three phases - chunk-local scans from zero state, a carry pass over
//                chunks (h_c+1 = exp(-r sum dt_c) h_c + S_c), and a finalize pass from the checkpoints.
//                Used when B * D/DT blocks cannot fill the GPU (small batch, long streams).
// Backward: per chunk, recompute h inside the chunk from its checkpoint into shared memory, then run the
// adjoint  lam_i = Cm gy_i + gH_i + a_{i+1} lam_{i+1}  in reverse.  Parameter gradients are accumulated
// in registers and written as fixed-order partial sums (deterministic, no float atomics).
#include <stdexcept>
#include <string>

#include "evkit/cuda_ops.hpp"
#include "evkit/ptx.cuh"

namespace evkit::gpu {
namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

__host__ __device__ __forceinline__ int64_t imin64(int64_t x, int64_t y) { return x < y ? x : y; }

#define EVKIT_CUDA_CHECK(expr)                                                                   \
  do {                                                                                           \
    cudaError_t _e = (expr);                                                                     \
    if (_e != cudaSuccess) throw std::runtime_error(std::string("evkit CUDA: ") + cudaGetErrorString(_e)); \
  } while (0)

struct Fwd {
  int64_t B, L, D, ns, C;
  const float *v, *dt, *rate, *Bm, *Cm, *skip, *h0;
  float *y, *H, *h_last, *ckpt, *S_local, *dtsum;
};

struct Bwd {
  int64_t B, L, D, ns, C, Z, n_dtiles;
  const float *v, *dt, *rate, *Bm, *Cm, *skip, *h0, *ckpt, *gy, *gH, *g_hlast;
  float *gv, *gh0;
  float *mu_local, *dts, *adj, *part_B, *part_C, *part_r, *part_skip, *part_dt;
};

template <int DT>
__device__ __forceinline__ void load_tile(float (*dst)[DT], const float* __restrict__ src, int64_t b, int64_t L,
                                          int64_t D, int64_t i0, int len, int64_t d0) {
  for (int idx = threadIdx.x; idx < kChunk * DT; idx += kThreads) {
    const int t = idx / DT, dd = idx % DT;
    const int64_t d = d0 + dd;
    dst[t][dd] = (t < len && d < D) ? ld_stream(src + (b * L + i0 + t) * D + d) : 0.f;
  }
}

template <int DT>
__device__ __forceinline__ void store_tile(float* __restrict__ dst, const float (*src)[DT], int64_t b, int64_t L,
                                           int64_t D, int64_t i0, int len, int64_t d0) {
  for (int idx = threadIdx.x; idx < kChunk * DT; idx += kThreads) {
    const int t = idx / DT, dd = idx % DT;
    const int64_t d = d0 + dd;
    if (t < len && d < D) dst[(b * L + i0 + t) * D + d] = src[t][dd];
  }
}

// ------------------------------------------------------------------------------------------- forward
template <int N, bool FAST, bool SEQ>
__global__ void __launch_bounds__(kThreads) scan_fwd_kernel(Fwd a) {
  constexpr int DT = kThreads / N;
  __shared__ float v_s[kChunk][DT];
  __shared__ float y_s[kChunk][DT];
  __shared__ float dt_s[kChunk];
  const int tid = threadIdx.x, dl = tid / N, n = tid % N;
  const int64_t b = blockIdx.y, d0 = int64_t(blockIdx.x) * DT, d = d0 + dl;
  const bool valid = d < a.D;
  const int64_t p = valid ? d * N + n : 0;
  const float nrl = valid ? -a.rate[p] * kLog2e : 0.f;
  const float bm = valid ? a.Bm[p] : 0.f, cm = valid ? a.Cm[p] : 0.f;
  const float sk = valid ? a.skip[d] : 0.f;
  const bool save = valid && d < a.ns;

  float h = 0.f;
  if (SEQ) h = (a.h0 && valid) ? a.h0[(b * a.D + d) * N + n] : 0.f;
  const int64_t c0 = SEQ ? 0 : blockIdx.z, cs = SEQ ? 1 : gridDim.z;
  for (int64_t c = c0; c < a.C; c += cs) {
    const int64_t i0 = c * kChunk;
    const int len = static_cast<int>(imin64(kChunk, a.L - i0));
    const int64_t sidx = ((b * a.C + c) * a.D + d) * N + n;
    if (SEQ) { if (valid) a.ckpt[sidx] = h; }
    else h = valid ? a.ckpt[sidx] : 0.f;
    load_tile<DT>(v_s, a.v, b, a.L, a.D, i0, len, d0);
    if (tid < kChunk) dt_s[tid] = tid < len ? a.dt[b * a.L + i0 + tid] : 0.f;
    __syncthreads();
    for (int t = 0; t < len; ++t) {
      const float vt = v_s[t][dl];
      h = fmaf(exp_neg<FAST>(nrl, dt_s[t]), h, bm * vt);
      const float yv = group_sum<N>(cm * h);
      if (n == 0) y_s[t][dl] = fmaf(sk, vt, yv);
      if (save) a.H[((b * a.L + i0 + t) * a.ns + d) * N + n] = h;   // coalesced: (d, n) contiguous over tid
    }
    __syncthreads();
    store_tile<DT>(a.y, y_s, b, a.L, a.D, i0, len, d0);
    __syncthreads();
  }
  if (SEQ && valid && a.h_last) a.h_last[(b * a.D + d) * N + n] = h;
}

template <int N, bool FAST>
__global__ void __launch_bounds__(kThreads) scan_local_kernel(Fwd a) {
  constexpr int DT = kThreads / N;
  __shared__ float v_s[kChunk][DT];
  __shared__ float dt_s[kChunk];
  const int tid = threadIdx.x, dl = tid / N, n = tid % N;
  const int64_t b = blockIdx.y, d0 = int64_t(blockIdx.x) * DT, d = d0 + dl;
  const bool valid = d < a.D;
  const int64_t p = valid ? d * N + n : 0;
  const float nrl = valid ? -a.rate[p] * kLog2e : 0.f;
  const float bm = valid ? a.Bm[p] : 0.f;
  for (int64_t c = blockIdx.z; c < a.C; c += gridDim.z) {
    const int64_t i0 = c * kChunk;
    const int len = static_cast<int>(imin64(kChunk, a.L - i0));
    load_tile<DT>(v_s, a.v, b, a.L, a.D, i0, len, d0);
    if (tid < kChunk) dt_s[tid] = tid < len ? a.dt[b * a.L + i0 + tid] : 0.f;
    __syncthreads();
    float h = 0.f;
    for (int t = 0; t < len; ++t) h = fmaf(exp_neg<FAST>(nrl, dt_s[t]), h, bm * v_s[t][dl]);
    if (valid) a.S_local[((b * a.C + c) * a.D + d) * N + n] = h;
    if (blockIdx.x == 0 && tid == 0) {
      float ds = 0.f;
      for (int t = 0; t < len; ++t) ds += dt_s[t];
      a.dtsum[b * a.C + c] = ds;
    }
    __syncthreads();
  }
}

template <bool FAST>
__global__ void scan_carry_kernel(Fwd a, int64_t N) {
  const int64_t DN = a.D * N, total = a.B * DN;
  for (int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += int64_t(gridDim.x) * blockDim.x) {
    const int64_t b = idx / DN, dn = idx % DN;
    const float nrl = -a.rate[dn] * kLog2e;
    float h = a.h0 ? a.h0[idx] : 0.f;
    for (int64_t c = 0; c < a.C; ++c) {
      const int64_t sidx = (b * a.C + c) * DN + dn;
      a.ckpt[sidx] = h;
      h = fmaf(exp_neg<FAST>(nrl, a.dtsum[b * a.C + c]), h, a.S_local[sidx]);
    }
    if (a.h_last) a.h_last[idx] = h;
  }
}

// ------------------------------------------------------------------------------------------- backward
template <int N, bool FAST>
__global__ void __launch_bounds__(kThreads) scan_bwd_local_kernel(Bwd a) {
  constexpr int DT = kThreads / N;
  __shared__ float gy_s[kChunk][DT];
  __shared__ float dt_s[kChunk + 1];
  const int tid = threadIdx.x, dl = tid / N, n = tid % N;
  const int64_t b = blockIdx.y, d0 = int64_t(blockIdx.x) * DT, d = d0 + dl;
  const bool valid = d < a.D, save = valid && d < a.ns;
  const int64_t p = valid ? d * N + n : 0;
  const float nrl = valid ? -a.rate[p] * kLog2e : 0.f;
  const float cm = valid ? a.Cm[p] : 0.f;
  for (int64_t c = blockIdx.z; c < a.C; c += gridDim.z) {
    const int64_t i0 = c * kChunk;
    const int len = static_cast<int>(imin64(kChunk, a.L - i0));
    const int64_t e = i0 + len;
    load_tile<DT>(gy_s, a.gy, b, a.L, a.D, i0, len, d0);
    // dt_s[0..len) = this chunk's gaps, dt_s[len] = first gap of the next chunk (0 past the end).
    // One writer per slot (threads 0..len), so no race when len < kChunk.
    if (tid <= len) dt_s[tid] = tid < len ? a.dt[b * a.L + i0 + tid] : (e < a.L ? a.dt[b * a.L + e] : 0.f);
    __syncthreads();
    float mu = 0.f, a_next = 1.f;
    for (int t = len - 1; t >= 0; --t) {
      float q = cm * gy_s[t][dl];
      if (a.gH && save) q += a.gH[((b * a.L + i0 + t) * a.ns + d) * N + n];
      mu = fmaf(a_next, mu, q);
      a_next = exp_neg<FAST>(nrl, dt_s[t]);
    }
    if (valid) a.mu_local[((b * a.C + c) * a.D + d) * N + n] = mu;
    if (blockIdx.x == 0 && tid == 0) {          // sum of dt over (s_c, e_c]  (dt_L := 0)
      float ds = dt_s[len];
      for (int t = 1; t < len; ++t) ds += dt_s[t];
      a.dts[b * a.C + c] = ds;
    }
    __syncthreads();
  }
}

template <bool FAST>
__global__ void scan_bwd_carry_kernel(Bwd a, int64_t N) {
  const int64_t DN = a.D * N, total = a.B * DN;
  for (int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += int64_t(gridDim.x) * blockDim.x) {
    const int64_t b = idx / DN, dn = idx % DN;
    const float nrl = -a.rate[dn] * kLog2e;
    float lam = a.g_hlast ? a.g_hlast[idx] : 0.f;
    for (int64_t c = a.C - 1; c >= 0; --c) {
      const int64_t sidx = (b * a.C + c) * DN + dn;
      a.adj[sidx] = lam;                                   // lam at the first index of the next chunk
      lam = fmaf(exp_neg<FAST>(nrl, a.dts[b * a.C + c]), lam, a.mu_local[sidx]);
    }
    if (a.gh0) a.gh0[idx] = exp_neg<FAST>(nrl, a.dt[b * a.L]) * lam;   // gh0 = a_0 lam_0
  }
}

template <int N, bool FAST, bool SEQ>
__global__ void __launch_bounds__(kThreads) scan_bwd_kernel(Bwd a) {
  constexpr int DT = kThreads / N;
  __shared__ float v_s[kChunk][DT];
  __shared__ float gy_s[kChunk][DT];
  __shared__ float gv_s[kChunk][DT];
  __shared__ float h_s[kChunk][kThreads];               // recomputed states of the chunk (own column only)
  __shared__ float gdt_w[kChunk][kWarps];
  __shared__ float dt_s[kChunk + 1];
  const int tid = threadIdx.x, dl = tid / N, n = tid % N, lane = tid & 31, warp = tid >> 5;
  const int64_t b = blockIdx.y, d0 = int64_t(blockIdx.x) * DT, d = d0 + dl;
  const bool valid = d < a.D, save = valid && d < a.ns;
  const int64_t p = valid ? d * N + n : 0;
  const float r = valid ? a.rate[p] : 0.f;
  const float nrl = -r * kLog2e;
  const float bm = valid ? a.Bm[p] : 0.f, cm = valid ? a.Cm[p] : 0.f;
  const float sk = valid ? a.skip[d] : 0.f;
  float acc_B = 0.f, acc_C = 0.f, acc_r = 0.f, acc_sk = 0.f;
  float lam = 0.f;
  if (SEQ) lam = (a.g_hlast && valid) ? a.g_hlast[(b * a.D + d) * N + n] : 0.f;
  const int64_t zi = SEQ ? 0 : blockIdx.z;

  for (int64_t k = SEQ ? 0 : blockIdx.z; k < a.C; k += SEQ ? 1 : gridDim.z) {
    const int64_t c = a.C - 1 - k;                       // reverse chunk order
    const int64_t i0 = c * kChunk;
    const int len = static_cast<int>(imin64(kChunk, a.L - i0));
    const int64_t e = i0 + len;
    const int64_t sidx = ((b * a.C + c) * a.D + d) * N + n;
    load_tile<DT>(v_s, a.v, b, a.L, a.D, i0, len, d0);
    load_tile<DT>(gy_s, a.gy, b, a.L, a.D, i0, len, d0);
    // dt_s[0..len) = this chunk's gaps, dt_s[len] = first gap of the next chunk (0 past the end).
    // One writer per slot (threads 0..len), so no race when len < kChunk.
    if (tid <= len) dt_s[tid] = tid < len ? a.dt[b * a.L + i0 + tid] : (e < a.L ? a.dt[b * a.L + e] : 0.f);
    const float h_start = valid ? a.ckpt[sidx] : 0.f;
    if (!SEQ) lam = valid ? a.adj[sidx] : 0.f;
    __syncthreads();

    float h = h_start;
    for (int t = 0; t < len; ++t) {
      h = fmaf(exp_neg<FAST>(nrl, dt_s[t]), h, bm * v_s[t][dl]);
      h_s[t][tid] = h;
    }
    float a_next = (e < a.L) ? exp_neg<FAST>(nrl, dt_s[len]) : 1.f;
    for (int t = len - 1; t >= 0; --t) {
      const float g = gy_s[t][dl], vt = v_s[t][dl], dtt = dt_s[t];
      float q = cm * g;
      if (a.gH && save) q += a.gH[((b * a.L + i0 + t) * a.ns + d) * N + n];
      lam = fmaf(a_next, lam, q);
      const float at = exp_neg<FAST>(nrl, dtt);
      const float hprev = t > 0 ? h_s[t - 1][tid] : h_start;
      const float ga = lam * hprev * at;                 // dL/da * a  (shared by d/dr and d/ddt)
      acc_B = fmaf(lam, vt, acc_B);
      acc_C = fmaf(g, h_s[t][tid], acc_C);
      acc_r = fmaf(-dtt, ga, acc_r);
      const float gvv = group_sum<N>(lam * bm);
      if (n == 0) {
        gv_s[t][dl] = fmaf(sk, g, gvv);
        acc_sk = fmaf(g, vt, acc_sk);
      }
      const float gdt_c = group_sum<32>(-r * ga);      // sum over this warp's (channel, state) pairs
      if (lane == 0) gdt_w[t][warp] = gdt_c;
      a_next = at;
    }
    __syncthreads();
    if (tid < len) {
      float s = 0.f;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) s += gdt_w[tid][w];
      a.part_dt[(int64_t(blockIdx.x) * a.B + b) * a.L + i0 + tid] = s;
    }
    store_tile<DT>(a.gv, gv_s, b, a.L, a.D, i0, len, d0);
    __syncthreads();
    if (SEQ && c == 0 && a.gh0 && valid) a.gh0[(b * a.D + d) * N + n] = a_next * lam;   // a_0 lam_0
  }
  if (valid) {
    const int64_t pp = ((b * a.Z + zi) * a.D + d) * N + n;
    a.part_B[pp] = acc_B;
    a.part_C[pp] = acc_C;
    a.part_r[pp] = acc_r;
    if (n == 0) a.part_skip[(b * a.Z + zi) * a.D + d] = acc_sk;
  }
}

__global__ void reduce_rows_kernel(const float* __restrict__ in, int64_t rows, int64_t cols, float* __restrict__ out) {
  for (int64_t j = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; j < cols; j += int64_t(gridDim.x) * blockDim.x) {
    float s = 0.f;
    for (int64_t r = 0; r < rows; ++r) s += in[r * cols + j];
    out[j] = s;
  }
}

// ------------------------------------------------------------------------------------------- launch
int device_sms() {
  int dev = 0, sms = 0;
  EVKIT_CUDA_CHECK(cudaGetDevice(&dev));
  EVKIT_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  return sms;
}

int dt_for(int64_t N) { return kThreads / static_cast<int>(N); }

ScanMode pick_mode(const ScanShape& s, ScanMode m) {
  if (m != ScanMode::kAuto) return m;
  const int64_t blocks = s.B * ((s.D + dt_for(s.N) - 1) / dt_for(s.N));
  return (blocks >= 2 * device_sms() || num_chunks(s.L) == 1) ? ScanMode::kSequential : ScanMode::kChunked;
}

int64_t z_for(const ScanShape& s, ScanMode m) {
  return m == ScanMode::kSequential ? 1 : std::min<int64_t>(num_chunks(s.L), kMaxZ);
}

void check_N(int64_t N) {
  if (N != 8 && N != 16 && N != 32) throw std::runtime_error("evkit: d_state must be 8, 16 or 32 on CUDA");
}

void reduce_rows(const float* in, int64_t rows, int64_t cols, float* out, cudaStream_t st) {
  if (!out || cols == 0) return;
  const int threads = 256;
  const int blocks = static_cast<int>(std::min<int64_t>((cols + threads - 1) / threads, 4096));
  reduce_rows_kernel<<<blocks, threads, 0, st>>>(in, rows, cols, out);
}

template <int N, bool FAST>
void fwd_impl(const ScanShape& s, Fwd& a, ScanMode m, cudaStream_t st) {
  const dim3 block(kThreads);
  const int64_t gx = (s.D + kThreads / N - 1) / (kThreads / N);
  if (m == ScanMode::kSequential) {
    scan_fwd_kernel<N, FAST, true><<<dim3(gx, s.B, 1), block, 0, st>>>(a);
  } else {
    const int64_t Z = z_for(s, m);
    scan_local_kernel<N, FAST><<<dim3(gx, s.B, Z), block, 0, st>>>(a);
    const int64_t total = s.B * s.D * s.N;
    scan_carry_kernel<FAST><<<static_cast<int>(std::min<int64_t>((total + 255) / 256, 65535)), 256, 0, st>>>(a, s.N);
    scan_fwd_kernel<N, FAST, false><<<dim3(gx, s.B, Z), block, 0, st>>>(a);
  }
  EVKIT_CUDA_CHECK(cudaGetLastError());
}

template <int N, bool FAST>
void bwd_impl(const ScanShape& s, Bwd& a, ScanMode m, cudaStream_t st) {
  const dim3 block(kThreads);
  const int64_t gx = a.n_dtiles;
  if (m == ScanMode::kSequential) {
    scan_bwd_kernel<N, FAST, true><<<dim3(gx, s.B, 1), block, 0, st>>>(a);
  } else {
    scan_bwd_local_kernel<N, FAST><<<dim3(gx, s.B, a.Z), block, 0, st>>>(a);
    const int64_t total = s.B * s.D * s.N;
    scan_bwd_carry_kernel<FAST><<<static_cast<int>(std::min<int64_t>((total + 255) / 256, 65535)), 256, 0, st>>>(a, s.N);
    scan_bwd_kernel<N, FAST, false><<<dim3(gx, s.B, a.Z), block, 0, st>>>(a);
  }
  EVKIT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

size_t scan_fwd_workspace_floats(const ScanShape& s) {
  const int64_t C = num_chunks(s.L);
  return static_cast<size_t>(s.B * C * s.D * s.N + s.B * C);
}

size_t scan_bwd_workspace_floats(const ScanShape& s) {
  const int64_t C = num_chunks(s.L), Z = kMaxZ;
  const int64_t ndt = (s.D + kThreads / s.N - 1) / (kThreads / s.N);
  return static_cast<size_t>(2 * s.B * C * s.D * s.N + s.B * C       // mu_local, adj, dts
                             + 3 * s.B * Z * s.D * s.N + s.B * Z * s.D // param partials
                             + ndt * s.B * s.L);                       // gdt partials
}

void scan_fwd(const ScanShape& s, const float* v, const float* dt, const float* rate, const float* Bm,
              const float* Cm, const float* skip, const float* h0, float* y, float* H_save, float* h_last,
              float* ckpt, float* ws, bool fast, ScanMode mode, void* stream) {
  check_N(s.N);
  auto st = static_cast<cudaStream_t>(stream);
  const int64_t C = num_chunks(s.L);
  Fwd a{s.B, s.L, s.D, s.n_save, C, v, dt, rate, Bm, Cm, skip, h0, y, H_save, h_last, ckpt,
        ws, ws + s.B * C * s.D * s.N};
  const ScanMode m = pick_mode(s, mode);
#define EVKIT_FWD(NN) fast ? fwd_impl<NN, true>(s, a, m, st) : fwd_impl<NN, false>(s, a, m, st)
  switch (s.N) { case 8: EVKIT_FWD(8); break; case 16: EVKIT_FWD(16); break; default: EVKIT_FWD(32); }
#undef EVKIT_FWD
}

void scan_bwd(const ScanShape& s, const float* v, const float* dt, const float* rate, const float* Bm,
              const float* Cm, const float* skip, const float* h0, const float* ckpt, const float* gy,
              const float* gH, const float* g_hlast, float* gv, float* gdt, float* grate, float* gB, float* gC,
              float* gskip, float* gh0, float* ws, bool fast, ScanMode mode, void* stream) {
  check_N(s.N);
  auto st = static_cast<cudaStream_t>(stream);
  const ScanMode m = pick_mode(s, mode);
  const int64_t C = num_chunks(s.L), Z = z_for(s, m);
  const int64_t ndt = (s.D + kThreads / s.N - 1) / (kThreads / s.N);
  const int64_t big = s.B * C * s.D * s.N, part = s.B * kMaxZ * s.D * s.N;
  float* mu_local = ws;
  float* adj = mu_local + big;
  float* dts = adj + big;
  float* part_B = dts + s.B * C;
  float* part_C = part_B + part;
  float* part_r = part_C + part;
  float* part_skip = part_r + part;
  float* part_dt = part_skip + s.B * kMaxZ * s.D;
  Bwd a{s.B, s.L, s.D, s.n_save, C, Z, ndt, v, dt, rate, Bm, Cm, skip, h0, ckpt, gy, gH, g_hlast,
        gv, gh0, mu_local, dts, adj, part_B, part_C, part_r, part_skip, part_dt};
#define EVKIT_BWD(NN) fast ? bwd_impl<NN, true>(s, a, m, st) : bwd_impl<NN, false>(s, a, m, st)
  switch (s.N) { case 8: EVKIT_BWD(8); break; case 16: EVKIT_BWD(16); break; default: EVKIT_BWD(32); }
#undef EVKIT_BWD
  const int64_t rows = s.B * Z;
  reduce_rows(part_B, rows, s.D * s.N, gB, st);
  reduce_rows(part_C, rows, s.D * s.N, gC, st);
  reduce_rows(part_r, rows, s.D * s.N, grate, st);
  reduce_rows(part_skip, rows, s.D, gskip, st);
  reduce_rows(part_dt, ndt, s.B * s.L, gdt, st);
  EVKIT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace evkit::gpu
