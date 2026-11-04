// CPU (OpenMP) implementations of the atl ops + Python bindings.  The CUDA versions in atl_cuda.cu mirror these
// loops one-to-one (one thread per event / anchor), so this file is also the readable specification.
#include "atl_ops.h"

#include <cmath>
#include <omp.h>

namespace atl {
namespace {

inline void check_cpu(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

template <typename F>
inline void bilinear_corners(float x, float y, int64_t H, int64_t W, F&& f) {
  // f(index_y, index_x, weight, dweight_dx, dweight_dy) for the (up to) 4 in-image corners
  const float fx0 = std::floor(x), fy0 = std::floor(y);
  const int64_t x0 = (int64_t)fx0, y0 = (int64_t)fy0;
  const float ax = x - fx0, ay = y - fy0;
  const float wx[2] = {1.f - ax, ax}, wy[2] = {1.f - ay, ay};
  const float dwx[2] = {-1.f, 1.f}, dwy[2] = {-1.f, 1.f};
  for (int j = 0; j < 2; ++j) {
    const int64_t yy = y0 + j;
    if (yy < 0 || yy >= H) continue;
    for (int i = 0; i < 2; ++i) {
      const int64_t xx = x0 + i;
      if (xx < 0 || xx >= W) continue;
      f(yy, xx, wx[i] * wy[j], dwx[i] * wy[j], wx[i] * dwy[j]);
    }
  }
}

}  // namespace

// ------------------------------------------------------------------------------------------------ bilinear splat
torch::Tensor splat_bilinear_fwd(torch::Tensor pos, torch::Tensor w, torch::Tensor bidx, int64_t B, int64_t H,
                                 int64_t W) {
  check_cpu(pos, "pos"); check_cpu(w, "w"); check_cpu(bidx, "bidx");
  auto img = torch::zeros({B, H, W}, pos.options());
  const int64_t N = pos.size(0);
  const float* P = pos.data_ptr<float>(); const float* Wt = w.data_ptr<float>();
  const int64_t* Bi = bidx.data_ptr<int64_t>(); float* I = img.data_ptr<float>();
#pragma omp parallel for schedule(static)
  for (int64_t n = 0; n < N; ++n) {
    const float wn = Wt[n];
    if (wn == 0.f) continue;
    float* base = I + Bi[n] * H * W;
    bilinear_corners(P[2 * n], P[2 * n + 1], H, W, [&](int64_t yy, int64_t xx, float c, float, float) {
#pragma omp atomic
      base[yy * W + xx] += wn * c;
    });
  }
  return img;
}

std::vector<torch::Tensor> splat_bilinear_bwd(torch::Tensor gimg, torch::Tensor pos, torch::Tensor w,
                                              torch::Tensor bidx) {
  check_cpu(gimg, "gimg"); check_cpu(pos, "pos");
  const int64_t N = pos.size(0), H = gimg.size(1), W = gimg.size(2);
  auto gpos = torch::zeros_like(pos); auto gw = torch::zeros_like(w);
  const float* G = gimg.data_ptr<float>(); const float* P = pos.data_ptr<float>(); const float* Wt = w.data_ptr<float>();
  const int64_t* Bi = bidx.data_ptr<int64_t>(); float* GP = gpos.data_ptr<float>(); float* GW = gw.data_ptr<float>();
#pragma omp parallel for schedule(static)
  for (int64_t n = 0; n < N; ++n) {
    const float* base = G + Bi[n] * H * W;
    float sx = 0.f, sy = 0.f, sw = 0.f;
    bilinear_corners(P[2 * n], P[2 * n + 1], H, W, [&](int64_t yy, int64_t xx, float c, float dcx, float dcy) {
      const float g = base[yy * W + xx];
      sw += g * c; sx += g * dcx; sy += g * dcy;
    });
    GW[n] = sw; GP[2 * n] = Wt[n] * sx; GP[2 * n + 1] = Wt[n] * sy;
  }
  return {gpos, gw};
}

// ------------------------------------------------------------------------------------------------ gaussian splat
torch::Tensor splat_gauss_fwd(torch::Tensor pos, torch::Tensor w, torch::Tensor bidx, int64_t B, int64_t H, int64_t W,
                              double sigma, int64_t R) {
  check_cpu(pos, "pos"); check_cpu(w, "w"); check_cpu(bidx, "bidx");
  auto img = torch::zeros({B, H, W}, pos.options());
  const int64_t N = pos.size(0);
  const float inv2s2 = 1.f / (2.f * (float)(sigma * sigma)), norm = 1.f / (2.f * (float)M_PI * (float)(sigma * sigma));
  const float* P = pos.data_ptr<float>(); const float* Wt = w.data_ptr<float>();
  const int64_t* Bi = bidx.data_ptr<int64_t>(); float* I = img.data_ptr<float>();
#pragma omp parallel for schedule(static)
  for (int64_t n = 0; n < N; ++n) {
    const float wn = Wt[n] * norm;
    if (wn == 0.f) continue;
    const float x = P[2 * n], y = P[2 * n + 1];
    const int64_t cx = (int64_t)std::lround(x), cy = (int64_t)std::lround(y);
    float* base = I + Bi[n] * H * W;
    for (int64_t yy = std::max<int64_t>(cy - R, 0); yy <= std::min<int64_t>(cy + R, H - 1); ++yy) {
      const float dy = (float)yy - y;
      for (int64_t xx = std::max<int64_t>(cx - R, 0); xx <= std::min<int64_t>(cx + R, W - 1); ++xx) {
        const float dx = (float)xx - x;
        const float v = wn * std::exp(-(dx * dx + dy * dy) * inv2s2);
#pragma omp atomic
        base[yy * W + xx] += v;
      }
    }
  }
  return img;
}

std::vector<torch::Tensor> splat_gauss_bwd(torch::Tensor gimg, torch::Tensor pos, torch::Tensor w, torch::Tensor bidx,
                                           double sigma, int64_t R) {
  const int64_t N = pos.size(0), H = gimg.size(1), W = gimg.size(2);
  auto gpos = torch::zeros_like(pos); auto gw = torch::zeros_like(w);
  const float s2 = (float)(sigma * sigma), inv2s2 = 1.f / (2.f * s2), norm = 1.f / (2.f * (float)M_PI * s2);
  const float* G = gimg.data_ptr<float>(); const float* P = pos.data_ptr<float>(); const float* Wt = w.data_ptr<float>();
  const int64_t* Bi = bidx.data_ptr<int64_t>(); float* GP = gpos.data_ptr<float>(); float* GWp = gw.data_ptr<float>();
#pragma omp parallel for schedule(static)
  for (int64_t n = 0; n < N; ++n) {
    const float x = P[2 * n], y = P[2 * n + 1];
    const int64_t cx = (int64_t)std::lround(x), cy = (int64_t)std::lround(y);
    const float* base = G + Bi[n] * H * W;
    float sw = 0.f, sx = 0.f, sy = 0.f;
    for (int64_t yy = std::max<int64_t>(cy - R, 0); yy <= std::min<int64_t>(cy + R, H - 1); ++yy) {
      const float dy = (float)yy - y;
      for (int64_t xx = std::max<int64_t>(cx - R, 0); xx <= std::min<int64_t>(cx + R, W - 1); ++xx) {
        const float dx = (float)xx - x;
        const float gk = norm * std::exp(-(dx * dx + dy * dy) * inv2s2) * base[yy * W + xx];
        sw += gk; sx += gk * dx / s2; sy += gk * dy / s2;
      }
    }
    GWp[n] = sw; GP[2 * n] = Wt[n] * sx; GP[2 * n + 1] = Wt[n] * sy;
  }
  return {gpos, gw};
}

// ------------------------------------------------------------------------------------------------ bilinear gather
torch::Tensor gather_bilinear_fwd(torch::Tensor img, torch::Tensor pos, torch::Tensor bidx) {
  check_cpu(img, "img"); check_cpu(pos, "pos"); check_cpu(bidx, "bidx");
  const int64_t N = pos.size(0), H = img.size(1), W = img.size(2);
  auto out = torch::zeros({N}, pos.options());
  const float* I = img.data_ptr<float>(); const float* P = pos.data_ptr<float>();
  const int64_t* Bi = bidx.data_ptr<int64_t>(); float* O = out.data_ptr<float>();
#pragma omp parallel for schedule(static)
  for (int64_t n = 0; n < N; ++n) {
    const float* base = I + Bi[n] * H * W;
    float s = 0.f;
    bilinear_corners(P[2 * n], P[2 * n + 1], H, W,
                     [&](int64_t yy, int64_t xx, float c, float, float) { s += c * base[yy * W + xx]; });
    O[n] = s;
  }
  return out;
}

std::vector<torch::Tensor> gather_bilinear_bwd(torch::Tensor gval, torch::Tensor img, torch::Tensor pos,
                                               torch::Tensor bidx) {
  const int64_t N = pos.size(0), H = img.size(1), W = img.size(2);
  auto gimg = torch::zeros_like(img); auto gpos = torch::zeros_like(pos);
  const float* I = img.data_ptr<float>(); const float* P = pos.data_ptr<float>(); const float* GV = gval.data_ptr<float>();
  const int64_t* Bi = bidx.data_ptr<int64_t>(); float* GI = gimg.data_ptr<float>(); float* GP = gpos.data_ptr<float>();
#pragma omp parallel for schedule(static)
  for (int64_t n = 0; n < N; ++n) {
    const float g = GV[n];
    const float* base = I + Bi[n] * H * W;
    float* gbase = GI + Bi[n] * H * W;
    float sx = 0.f, sy = 0.f;
    bilinear_corners(P[2 * n], P[2 * n + 1], H, W, [&](int64_t yy, int64_t xx, float c, float dcx, float dcy) {
      const float v = base[yy * W + xx];
      sx += dcx * v; sy += dcy * v;
#pragma omp atomic
      gbase[yy * W + xx] += g * c;
    });
    GP[2 * n] = g * sx; GP[2 * n + 1] = g * sy;
  }
  return {gimg, gpos};
}

// ------------------------------------------------------------------------------------------------ anchored pairs
namespace {
template <typename Visit>
inline void for_each_pair(int64_t i, const float* AX, const float* AT, const int64_t* AC, const int64_t* AID,
                          const float* SY, const float* SV, const float* ST, const int64_t* SID, const int64_t* CS,
                          const float* SC, int64_t S, int64_t GH, int64_t GW, float h, float tau, float cut,
                          Visit&& visit) {
  const float xi = AX[2 * i], yi = AX[2 * i + 1], ti = AT[i];
  const int64_t g = AC[3 * i], cx = AC[3 * i + 1], cy = AC[3 * i + 2], id = AID[i];
  const float cs = SC[g % S];
  const float r2max = cut * cut * h * h, inv2h2 = 1.f / (2.f * h * h);
  for (int64_t yy = std::max<int64_t>(cy - 1, 0); yy <= std::min<int64_t>(cy + 1, GH - 1); ++yy) {
    for (int64_t xx = std::max<int64_t>(cx - 1, 0); xx <= std::min<int64_t>(cx + 1, GW - 1); ++xx) {
      const int64_t cell = (g * GH + yy) * GW + xx;
      for (int64_t k = CS[cell]; k < CS[cell + 1]; ++k) {
        if (SID[k] == id) continue;
        const float dt = ti - ST[k];
        if (std::fabs(dt) > tau) continue;
        const float rx = xi - (SY[2 * k] + SV[2 * k] * (ti - cs));
        const float ry = yi - (SY[2 * k + 1] + SV[2 * k + 1] * (ti - cs));
        const float d2 = rx * rx + ry * ry;
        if (d2 > r2max) continue;
        visit(k, rx, ry, dt, std::exp(-d2 * inv2h2));
      }
    }
  }
}
}  // namespace

torch::Tensor ats_pair_fwd(torch::Tensor src_y, torch::Tensor src_v, torch::Tensor src_t, torch::Tensor src_id,
                           torch::Tensor cell_start, torch::Tensor anc_x, torch::Tensor anc_t, torch::Tensor anc_cell,
                           torch::Tensor anc_id, torch::Tensor slice_c, int64_t S, int64_t GH, int64_t GW, double h,
                           double tau, double cut) {
  const int64_t M = anc_x.size(0);
  auto out = torch::zeros({M}, anc_x.options());
  const float* SY = src_y.data_ptr<float>(); const float* SV = src_v.data_ptr<float>(); const float* ST = src_t.data_ptr<float>();
  const int64_t* SID = src_id.data_ptr<int64_t>(); const int64_t* CS = cell_start.data_ptr<int64_t>();
  const float* AX = anc_x.data_ptr<float>(); const float* AT = anc_t.data_ptr<float>();
  const int64_t* AC = anc_cell.data_ptr<int64_t>(); const int64_t* AID = anc_id.data_ptr<int64_t>();
  const float* SC = slice_c.data_ptr<float>(); float* O = out.data_ptr<float>();
#pragma omp parallel for schedule(dynamic, 64)
  for (int64_t i = 0; i < M; ++i) {
    double acc = 0.0;
    for_each_pair(i, AX, AT, AC, AID, SY, SV, ST, SID, CS, SC, S, GH, GW, (float)h, (float)tau, (float)cut,
                  [&](int64_t, float, float, float, float k) { acc += k; });
    O[i] = (float)acc;
  }
  return out;
}

torch::Tensor ats_pair_bwd(torch::Tensor gS, torch::Tensor src_y, torch::Tensor src_v, torch::Tensor src_t,
                           torch::Tensor src_id, torch::Tensor cell_start, torch::Tensor anc_x, torch::Tensor anc_t,
                           torch::Tensor anc_cell, torch::Tensor anc_id, torch::Tensor slice_c, int64_t S, int64_t GH,
                           int64_t GW, double h, double tau, double cut, int64_t N) {
  const int64_t M = anc_x.size(0);
  auto gv = torch::zeros({N, 2}, anc_x.options());
  const float* G = gS.data_ptr<float>();
  const float* SY = src_y.data_ptr<float>(); const float* SV = src_v.data_ptr<float>(); const float* ST = src_t.data_ptr<float>();
  const int64_t* SID = src_id.data_ptr<int64_t>(); const int64_t* CS = cell_start.data_ptr<int64_t>();
  const float* AX = anc_x.data_ptr<float>(); const float* AT = anc_t.data_ptr<float>();
  const int64_t* AC = anc_cell.data_ptr<int64_t>(); const int64_t* AID = anc_id.data_ptr<int64_t>();
  const float* SC = slice_c.data_ptr<float>(); float* GV = gv.data_ptr<float>();
  const float invh2 = 1.f / (float)(h * h);
#pragma omp parallel for schedule(dynamic, 64)
  for (int64_t i = 0; i < M; ++i) {
    const float gi = G[i];
    if (gi == 0.f) continue;
    for_each_pair(i, AX, AT, AC, AID, SY, SV, ST, SID, CS, SC, S, GH, GW, (float)h, (float)tau, (float)cut,
                  [&](int64_t k, float rx, float ry, float dt, float kern) {
                    // dS/dv_j = k * r * (t_i - t_j) / h^2
                    const float f = gi * kern * dt * invh2;
                    const int64_t j = SID[k];
#pragma omp atomic
                    GV[2 * j] += f * rx;
#pragma omp atomic
                    GV[2 * j + 1] += f * ry;
                  });
  }
  return gv;
}

// ------------------------------------------------------------------------------------------------ voxel grid
torch::Tensor voxel_grid(torch::Tensor pos, torch::Tensor t, torch::Tensor p, torch::Tensor bidx, int64_t B, int64_t C,
                         int64_t H, int64_t W) {
  check_cpu(pos, "pos"); check_cpu(t, "t"); check_cpu(p, "p"); check_cpu(bidx, "bidx");
  auto vox = torch::zeros({B, C, H, W}, pos.options());
  const int64_t N = pos.size(0);
  const float* P = pos.data_ptr<float>(); const float* T = t.data_ptr<float>(); const float* Pol = p.data_ptr<float>();
  const int64_t* Bi = bidx.data_ptr<int64_t>(); float* V = vox.data_ptr<float>();
#pragma omp parallel for schedule(static)
  for (int64_t n = 0; n < N; ++n) {
    const float x = P[2 * n], y = P[2 * n + 1], tn = T[n] * (float)(C - 1);
    const int64_t x0 = (int64_t)std::floor(x), y0 = (int64_t)std::floor(y), t0 = (int64_t)std::floor(tn);
    for (int dt = 0; dt < 2; ++dt) {
      const int64_t tt = t0 + dt;
      if (tt < 0 || tt >= C) continue;
      const float wt = 1.f - std::fabs((float)tt - tn);
      for (int dy = 0; dy < 2; ++dy) {
        const int64_t yy = y0 + dy;
        if (yy < 0 || yy >= H) continue;
        const float wy = 1.f - std::fabs((float)yy - y);
        for (int dx = 0; dx < 2; ++dx) {
          const int64_t xx = x0 + dx;
          if (xx < 0 || xx >= W) continue;
          const float wx = 1.f - std::fabs((float)xx - x);
#pragma omp atomic
          V[((Bi[n] * C + tt) * H + yy) * W + xx] += Pol[n] * wt * wy * wx;
        }
      }
    }
  }
  return vox;
}

}  // namespace atl

// ------------------------------------------------------------------------------------------------ dispatch + bindings
namespace {
#ifdef ATL_WITH_CUDA
#define ATL_DISPATCH(first, fn, ...) ((first).is_cuda() ? atl::cuda::fn(__VA_ARGS__) : atl::fn(__VA_ARGS__))
#else
#define ATL_DISPATCH(first, fn, ...) atl::fn(__VA_ARGS__)
#endif
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("splat_bilinear_fwd", [](torch::Tensor pos, torch::Tensor w, torch::Tensor b, int64_t B, int64_t H, int64_t W) {
    return ATL_DISPATCH(pos, splat_bilinear_fwd, pos, w, b, B, H, W); });
  m.def("splat_bilinear_bwd", [](torch::Tensor g, torch::Tensor pos, torch::Tensor w, torch::Tensor b) {
    return ATL_DISPATCH(pos, splat_bilinear_bwd, g, pos, w, b); });
  m.def("splat_gauss_fwd", [](torch::Tensor pos, torch::Tensor w, torch::Tensor b, int64_t B, int64_t H, int64_t W,
                              double s, int64_t R) { return ATL_DISPATCH(pos, splat_gauss_fwd, pos, w, b, B, H, W, s, R); });
  m.def("splat_gauss_bwd", [](torch::Tensor g, torch::Tensor pos, torch::Tensor w, torch::Tensor b, double s, int64_t R) {
    return ATL_DISPATCH(pos, splat_gauss_bwd, g, pos, w, b, s, R); });
  m.def("gather_bilinear_fwd", [](torch::Tensor img, torch::Tensor pos, torch::Tensor b) {
    return ATL_DISPATCH(pos, gather_bilinear_fwd, img, pos, b); });
  m.def("gather_bilinear_bwd", [](torch::Tensor g, torch::Tensor img, torch::Tensor pos, torch::Tensor b) {
    return ATL_DISPATCH(pos, gather_bilinear_bwd, g, img, pos, b); });
  m.def("ats_pair_fwd", [](torch::Tensor sy, torch::Tensor sv, torch::Tensor st, torch::Tensor sid, torch::Tensor cs,
                           torch::Tensor ax, torch::Tensor at, torch::Tensor ac, torch::Tensor aid, torch::Tensor sc,
                           int64_t S, int64_t GH, int64_t GW, double h, double tau, double cut) {
    return ATL_DISPATCH(ax, ats_pair_fwd, sy, sv, st, sid, cs, ax, at, ac, aid, sc, S, GH, GW, h, tau, cut); });
  m.def("ats_pair_bwd", [](torch::Tensor g, torch::Tensor sy, torch::Tensor sv, torch::Tensor st, torch::Tensor sid,
                           torch::Tensor cs, torch::Tensor ax, torch::Tensor at, torch::Tensor ac, torch::Tensor aid,
                           torch::Tensor sc, int64_t S, int64_t GH, int64_t GW, double h, double tau, double cut,
                           int64_t N) {
    return ATL_DISPATCH(ax, ats_pair_bwd, g, sy, sv, st, sid, cs, ax, at, ac, aid, sc, S, GH, GW, h, tau, cut, N); });
  m.def("voxel_grid", [](torch::Tensor pos, torch::Tensor t, torch::Tensor p, torch::Tensor b, int64_t B, int64_t C,
                         int64_t H, int64_t W) { return ATL_DISPATCH(pos, voxel_grid, pos, t, p, b, B, C, H, W); });
  m.attr("with_cuda") =
#ifdef ATL_WITH_CUDA
      true;
#else
      false;
#endif
}
