// CUDA implementations of the atl ops.  One thread per event (splat / gather / voxel) or per anchor (pairs);
// scatters use float atomics.  Loop bodies mirror atl_cpu.cpp exactly, which is the tested specification.
#include "atl_ops.h"

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace atl {
namespace cuda {
namespace {

constexpr int kThreads = 256;
inline int blocks_for(int64_t n) { return (int)((n + kThreads - 1) / kThreads); }

__device__ __forceinline__ float fast_exp(float x) { return __expf(x); }

template <typename F>
__device__ __forceinline__ void bilinear_corners(float x, float y, int64_t H, int64_t W, F&& f) {
  const float fx0 = floorf(x), fy0 = floorf(y);
  const int64_t x0 = (int64_t)fx0, y0 = (int64_t)fy0;
  const float ax = x - fx0, ay = y - fy0;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int64_t yy = y0 + j;
    if (yy < 0 || yy >= H) continue;
    const float wy = j ? ay : 1.f - ay, dwy = j ? 1.f : -1.f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int64_t xx = x0 + i;
      if (xx < 0 || xx >= W) continue;
      const float wx = i ? ax : 1.f - ax, dwx = i ? 1.f : -1.f;
      f(yy, xx, wx * wy, dwx * wy, wx * dwy);
    }
  }
}

// ---------------------------------------------------------------------------------------------- splat / gather
__global__ void k_splat_bilinear_fwd(const float* __restrict__ P, const float* __restrict__ Wt,
                                     const int64_t* __restrict__ Bi, float* I, int64_t N, int64_t H, int64_t W) {
  const int64_t n = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float wn = Wt[n];
  if (wn == 0.f) return;
  float* base = I + Bi[n] * H * W;
  bilinear_corners(P[2 * n], P[2 * n + 1], H, W,
                   [&](int64_t yy, int64_t xx, float c, float, float) { atomicAdd(base + yy * W + xx, wn * c); });
}

__global__ void k_splat_bilinear_bwd(const float* __restrict__ G, const float* __restrict__ P,
                                     const float* __restrict__ Wt, const int64_t* __restrict__ Bi, float* GP,
                                     float* GWp, int64_t N, int64_t H, int64_t W) {
  const int64_t n = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float* base = G + Bi[n] * H * W;
  float sx = 0.f, sy = 0.f, sw = 0.f;
  bilinear_corners(P[2 * n], P[2 * n + 1], H, W, [&](int64_t yy, int64_t xx, float c, float dcx, float dcy) {
    const float g = __ldg(base + yy * W + xx);
    sw += g * c; sx += g * dcx; sy += g * dcy;
  });
  GWp[n] = sw; GP[2 * n] = Wt[n] * sx; GP[2 * n + 1] = Wt[n] * sy;
}

__global__ void k_splat_gauss_fwd(const float* __restrict__ P, const float* __restrict__ Wt,
                                  const int64_t* __restrict__ Bi, float* I, int64_t N, int64_t H, int64_t W,
                                  float inv2s2, float norm, int R) {
  const int64_t n = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float wn = Wt[n] * norm;
  if (wn == 0.f) return;
  const float x = P[2 * n], y = P[2 * n + 1];
  const int64_t cx = (int64_t)lroundf(x), cy = (int64_t)lroundf(y);
  float* base = I + Bi[n] * H * W;
  for (int64_t yy = max(cy - R, (int64_t)0); yy <= min(cy + R, H - 1); ++yy) {
    const float dy = (float)yy - y;
    for (int64_t xx = max(cx - R, (int64_t)0); xx <= min(cx + R, W - 1); ++xx) {
      const float dx = (float)xx - x;
      atomicAdd(base + yy * W + xx, wn * fast_exp(-(dx * dx + dy * dy) * inv2s2));
    }
  }
}

__global__ void k_splat_gauss_bwd(const float* __restrict__ G, const float* __restrict__ P,
                                  const float* __restrict__ Wt, const int64_t* __restrict__ Bi, float* GP, float* GWp,
                                  int64_t N, int64_t H, int64_t W, float s2, float inv2s2, float norm, int R) {
  const int64_t n = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float x = P[2 * n], y = P[2 * n + 1];
  const int64_t cx = (int64_t)lroundf(x), cy = (int64_t)lroundf(y);
  const float* base = G + Bi[n] * H * W;
  float sw = 0.f, sx = 0.f, sy = 0.f;
  for (int64_t yy = max(cy - R, (int64_t)0); yy <= min(cy + R, H - 1); ++yy) {
    const float dy = (float)yy - y;
    for (int64_t xx = max(cx - R, (int64_t)0); xx <= min(cx + R, W - 1); ++xx) {
      const float dx = (float)xx - x;
      const float gk = norm * fast_exp(-(dx * dx + dy * dy) * inv2s2) * __ldg(base + yy * W + xx);
      sw += gk; sx += gk * dx / s2; sy += gk * dy / s2;
    }
  }
  GWp[n] = sw; GP[2 * n] = Wt[n] * sx; GP[2 * n + 1] = Wt[n] * sy;
}

__global__ void k_gather_fwd(const float* __restrict__ I, const float* __restrict__ P,
                             const int64_t* __restrict__ Bi, float* O, int64_t N, int64_t H, int64_t W) {
  const int64_t n = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float* base = I + Bi[n] * H * W;
  float s = 0.f;
  bilinear_corners(P[2 * n], P[2 * n + 1], H, W,
                   [&](int64_t yy, int64_t xx, float c, float, float) { s += c * __ldg(base + yy * W + xx); });
  O[n] = s;
}

__global__ void k_gather_bwd(const float* __restrict__ GV, const float* __restrict__ I, const float* __restrict__ P,
                             const int64_t* __restrict__ Bi, float* GI, float* GP, int64_t N, int64_t H, int64_t W) {
  const int64_t n = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float g = GV[n];
  const float* base = I + Bi[n] * H * W;
  float* gbase = GI + Bi[n] * H * W;
  float sx = 0.f, sy = 0.f;
  bilinear_corners(P[2 * n], P[2 * n + 1], H, W, [&](int64_t yy, int64_t xx, float c, float dcx, float dcy) {
    const float v = __ldg(base + yy * W + xx);
    sx += dcx * v; sy += dcy * v;
    if (g != 0.f) atomicAdd(gbase + yy * W + xx, g * c);
  });
  GP[2 * n] = g * sx; GP[2 * n + 1] = g * sy;
}

// ---------------------------------------------------------------------------------------------- anchored pairs
struct PairArgs {
  const float *SY, *SV, *ST, *AX, *AT, *SC;
  const int64_t *SID, *CS, *AC, *AID;
  int64_t S, GH, GW;
  float h, tau, cut;
};

template <typename Visit>
__device__ __forceinline__ void for_each_pair(int64_t i, const PairArgs& a, Visit&& visit) {
  const float xi = a.AX[2 * i], yi = a.AX[2 * i + 1], ti = a.AT[i];
  const int64_t g = a.AC[3 * i], cx = a.AC[3 * i + 1], cy = a.AC[3 * i + 2], id = a.AID[i];
  const float cs = a.SC[g % a.S];
  const float r2max = a.cut * a.cut * a.h * a.h, inv2h2 = 1.f / (2.f * a.h * a.h);
  for (int64_t yy = max(cy - 1, (int64_t)0); yy <= min(cy + 1, a.GH - 1); ++yy) {
    for (int64_t xx = max(cx - 1, (int64_t)0); xx <= min(cx + 1, a.GW - 1); ++xx) {
      const int64_t cell = (g * a.GH + yy) * a.GW + xx;
      const int64_t k1 = __ldg(a.CS + cell + 1);
      for (int64_t k = __ldg(a.CS + cell); k < k1; ++k) {
        if (__ldg(a.SID + k) == id) continue;
        const float dt = ti - __ldg(a.ST + k);
        if (fabsf(dt) > a.tau) continue;
        const float rx = xi - (__ldg(a.SY + 2 * k) + __ldg(a.SV + 2 * k) * (ti - cs));
        const float ry = yi - (__ldg(a.SY + 2 * k + 1) + __ldg(a.SV + 2 * k + 1) * (ti - cs));
        const float d2 = rx * rx + ry * ry;
        if (d2 > r2max) continue;
        visit(k, rx, ry, dt, fast_exp(-d2 * inv2h2));
      }
    }
  }
}

__global__ void k_ats_fwd(PairArgs a, float* O, int64_t M) {
  const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= M) return;
  float acc = 0.f;
  for_each_pair(i, a, [&](int64_t, float, float, float, float k) { acc += k; });
  O[i] = acc;
}

__global__ void k_ats_bwd(PairArgs a, const float* __restrict__ G, float* GV, int64_t M, float invh2) {
  const int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i >= M) return;
  const float gi = G[i];
  if (gi == 0.f) return;
  for_each_pair(i, a, [&](int64_t k, float rx, float ry, float dt, float kern) {
    const float f = gi * kern * dt * invh2;
    const int64_t j = __ldg(a.SID + k);
    atomicAdd(GV + 2 * j, f * rx);
    atomicAdd(GV + 2 * j + 1, f * ry);
  });
}

// ---------------------------------------------------------------------------------------------- voxel grid
__global__ void k_voxel(const float* __restrict__ P, const float* __restrict__ T, const float* __restrict__ Pol,
                        const int64_t* __restrict__ Bi, float* V, int64_t N, int64_t C, int64_t H, int64_t W) {
  const int64_t n = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float x = P[2 * n], y = P[2 * n + 1], tn = T[n] * (float)(C - 1);
  const int64_t x0 = (int64_t)floorf(x), y0 = (int64_t)floorf(y), t0 = (int64_t)floorf(tn);
  for (int dt = 0; dt < 2; ++dt) {
    const int64_t tt = t0 + dt;
    if (tt < 0 || tt >= C) continue;
    const float wt = 1.f - fabsf((float)tt - tn);
    for (int dy = 0; dy < 2; ++dy) {
      const int64_t yy = y0 + dy;
      if (yy < 0 || yy >= H) continue;
      const float wy = 1.f - fabsf((float)yy - y);
      for (int dx = 0; dx < 2; ++dx) {
        const int64_t xx = x0 + dx;
        if (xx < 0 || xx >= W) continue;
        atomicAdd(V + ((Bi[n] * C + tt) * H + yy) * W + xx, Pol[n] * wt * wy * (1.f - fabsf((float)xx - x)));
      }
    }
  }
}

inline PairArgs make_args(torch::Tensor sy, torch::Tensor sv, torch::Tensor st, torch::Tensor sid, torch::Tensor cs,
                          torch::Tensor ax, torch::Tensor at, torch::Tensor ac, torch::Tensor aid, torch::Tensor sc,
                          int64_t S, int64_t GH, int64_t GW, double h, double tau, double cut) {
  PairArgs a;
  a.SY = sy.data_ptr<float>(); a.SV = sv.data_ptr<float>(); a.ST = st.data_ptr<float>();
  a.AX = ax.data_ptr<float>(); a.AT = at.data_ptr<float>(); a.SC = sc.data_ptr<float>();
  a.SID = sid.data_ptr<int64_t>(); a.CS = cs.data_ptr<int64_t>(); a.AC = ac.data_ptr<int64_t>();
  a.AID = aid.data_ptr<int64_t>(); a.S = S; a.GH = GH; a.GW = GW;
  a.h = (float)h; a.tau = (float)tau; a.cut = (float)cut;
  return a;
}

}  // namespace

#define ATL_STREAM c10::cuda::getCurrentCUDAStream().stream()
#define ATL_GUARD(t) const c10::cuda::OptionalCUDAGuard guard_(t.device())

torch::Tensor splat_bilinear_fwd(torch::Tensor pos, torch::Tensor w, torch::Tensor b, int64_t B, int64_t H, int64_t W) {
  ATL_GUARD(pos);
  auto img = torch::zeros({B, H, W}, pos.options());
  const int64_t N = pos.size(0);
  if (N) k_splat_bilinear_fwd<<<blocks_for(N), kThreads, 0, ATL_STREAM>>>(pos.data_ptr<float>(), w.data_ptr<float>(),
                                                                         b.data_ptr<int64_t>(), img.data_ptr<float>(), N, H, W);
  return img;
}

std::vector<torch::Tensor> splat_bilinear_bwd(torch::Tensor g, torch::Tensor pos, torch::Tensor w, torch::Tensor b) {
  ATL_GUARD(pos);
  auto gp = torch::zeros_like(pos); auto gw = torch::zeros_like(w);
  const int64_t N = pos.size(0);
  if (N) k_splat_bilinear_bwd<<<blocks_for(N), kThreads, 0, ATL_STREAM>>>(
      g.data_ptr<float>(), pos.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<int64_t>(), gp.data_ptr<float>(),
      gw.data_ptr<float>(), N, g.size(1), g.size(2));
  return {gp, gw};
}

torch::Tensor splat_gauss_fwd(torch::Tensor pos, torch::Tensor w, torch::Tensor b, int64_t B, int64_t H, int64_t W,
                              double sigma, int64_t R) {
  ATL_GUARD(pos);
  auto img = torch::zeros({B, H, W}, pos.options());
  const int64_t N = pos.size(0);
  const float s2 = (float)(sigma * sigma);
  if (N) k_splat_gauss_fwd<<<blocks_for(N), kThreads, 0, ATL_STREAM>>>(
      pos.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<int64_t>(), img.data_ptr<float>(), N, H, W,
      1.f / (2.f * s2), 1.f / (2.f * (float)M_PI * s2), (int)R);
  return img;
}

std::vector<torch::Tensor> splat_gauss_bwd(torch::Tensor g, torch::Tensor pos, torch::Tensor w, torch::Tensor b,
                                           double sigma, int64_t R) {
  ATL_GUARD(pos);
  auto gp = torch::zeros_like(pos); auto gw = torch::zeros_like(w);
  const int64_t N = pos.size(0);
  const float s2 = (float)(sigma * sigma);
  if (N) k_splat_gauss_bwd<<<blocks_for(N), kThreads, 0, ATL_STREAM>>>(
      g.data_ptr<float>(), pos.data_ptr<float>(), w.data_ptr<float>(), b.data_ptr<int64_t>(), gp.data_ptr<float>(),
      gw.data_ptr<float>(), N, g.size(1), g.size(2), s2, 1.f / (2.f * s2), 1.f / (2.f * (float)M_PI * s2), (int)R);
  return {gp, gw};
}

torch::Tensor gather_bilinear_fwd(torch::Tensor img, torch::Tensor pos, torch::Tensor b) {
  ATL_GUARD(pos);
  const int64_t N = pos.size(0);
  auto out = torch::zeros({N}, pos.options());
  if (N) k_gather_fwd<<<blocks_for(N), kThreads, 0, ATL_STREAM>>>(img.data_ptr<float>(), pos.data_ptr<float>(),
                                                                 b.data_ptr<int64_t>(), out.data_ptr<float>(), N,
                                                                 img.size(1), img.size(2));
  return out;
}

std::vector<torch::Tensor> gather_bilinear_bwd(torch::Tensor g, torch::Tensor img, torch::Tensor pos, torch::Tensor b) {
  ATL_GUARD(pos);
  auto gi = torch::zeros_like(img); auto gp = torch::zeros_like(pos);
  const int64_t N = pos.size(0);
  if (N) k_gather_bwd<<<blocks_for(N), kThreads, 0, ATL_STREAM>>>(g.data_ptr<float>(), img.data_ptr<float>(),
                                                                 pos.data_ptr<float>(), b.data_ptr<int64_t>(),
                                                                 gi.data_ptr<float>(), gp.data_ptr<float>(), N,
                                                                 img.size(1), img.size(2));
  return {gi, gp};
}

torch::Tensor ats_pair_fwd(torch::Tensor sy, torch::Tensor sv, torch::Tensor st, torch::Tensor sid, torch::Tensor cs,
                           torch::Tensor ax, torch::Tensor at, torch::Tensor ac, torch::Tensor aid, torch::Tensor sc,
                           int64_t S, int64_t GH, int64_t GW, double h, double tau, double cut) {
  ATL_GUARD(ax);
  const int64_t M = ax.size(0);
  auto out = torch::zeros({M}, ax.options());
  if (M) k_ats_fwd<<<blocks_for(M), kThreads, 0, ATL_STREAM>>>(
      make_args(sy, sv, st, sid, cs, ax, at, ac, aid, sc, S, GH, GW, h, tau, cut), out.data_ptr<float>(), M);
  return out;
}

torch::Tensor ats_pair_bwd(torch::Tensor g, torch::Tensor sy, torch::Tensor sv, torch::Tensor st, torch::Tensor sid,
                           torch::Tensor cs, torch::Tensor ax, torch::Tensor at, torch::Tensor ac, torch::Tensor aid,
                           torch::Tensor sc, int64_t S, int64_t GH, int64_t GW, double h, double tau, double cut,
                           int64_t N) {
  ATL_GUARD(ax);
  const int64_t M = ax.size(0);
  auto gv = torch::zeros({N, 2}, ax.options());
  if (M) k_ats_bwd<<<blocks_for(M), kThreads, 0, ATL_STREAM>>>(
      make_args(sy, sv, st, sid, cs, ax, at, ac, aid, sc, S, GH, GW, h, tau, cut), g.data_ptr<float>(),
      gv.data_ptr<float>(), M, (float)(1.0 / (h * h)));
  return gv;
}

torch::Tensor voxel_grid(torch::Tensor pos, torch::Tensor t, torch::Tensor p, torch::Tensor b, int64_t B, int64_t C,
                         int64_t H, int64_t W) {
  ATL_GUARD(pos);
  auto vox = torch::zeros({B, C, H, W}, pos.options());
  const int64_t N = pos.size(0);
  if (N) k_voxel<<<blocks_for(N), kThreads, 0, ATL_STREAM>>>(pos.data_ptr<float>(), t.data_ptr<float>(),
                                                            p.data_ptr<float>(), b.data_ptr<int64_t>(),
                                                            vox.data_ptr<float>(), N, C, H, W);
  return vox;
}

}  // namespace cuda
}  // namespace atl
