#include "evkit/ssm_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

namespace evkit::cpu {

namespace {

int threads_for(int requested, int64_t work_items) {
  int T = requested > 0 ? requested : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(T, work_items)));
}

template <class F>
void parallel_for(int64_t n, int T, F&& f) {   // f(begin, end, worker_id)
  if (T <= 1) { f(int64_t(0), n, 0); return; }
  std::vector<std::thread> th;
  for (int k = 0; k < T; ++k) th.emplace_back([&, k] { f(n * k / T, n * (k + 1) / T, k); });
  for (auto& h : th) h.join();
}

}  // namespace

// ---------------------------------------------------------------------------------- scan forward
void scan_fwd(const ScanShape& s, const float* v, const float* dt, const float* rate, const float* Bm,
              const float* Cm, const float* skip, const float* h0, float* y, float* H_save, float* h_last,
              int n_threads) {
  const int64_t B = s.B, L = s.L, D = s.D, N = s.N, ns = s.n_save;
  const int T = threads_for(n_threads, B * D);
  parallel_for(B * D, T, [&](int64_t lo, int64_t hi, int) {
    std::vector<float> h(N);
    for (int64_t bd = lo; bd < hi; ++bd) {
      const int64_t b = bd / D, d = bd % D;
      for (int64_t n = 0; n < N; ++n) h[n] = h0 ? h0[(b * D + d) * N + n] : 0.f;
      const float* r = rate + d * N;
      const float* bm = Bm + d * N;
      const float* cm = Cm + d * N;
      for (int64_t i = 0; i < L; ++i) {
        const float vi = v[(b * L + i) * D + d];
        const float di = dt[b * L + i];
        float acc = skip[d] * vi;
        for (int64_t n = 0; n < N; ++n) {
          h[n] = std::exp(-r[n] * di) * h[n] + bm[n] * vi;
          acc += cm[n] * h[n];
        }
        y[(b * L + i) * D + d] = acc;
        if (H_save && d < ns) std::memcpy(H_save + ((b * L + i) * ns + d) * N, h.data(), N * sizeof(float));
      }
      if (h_last) std::memcpy(h_last + (b * D + d) * N, h.data(), N * sizeof(float));
    }
  });
}

// ---------------------------------------------------------------------------------- scan backward
// Adjoint:  lam_i = Cm * gy_i + gH_i + a_{i+1} lam_{i+1},  lam_{L-1} also receives g_hlast (a_L := 1).
//   gB += lam_i v_i ; gv_i = sum_n lam_i Bm + skip gy_i ; gC += gy_i h_i ; gskip += gy_i v_i
//   with a_i = exp(-r dt_i):  g_a = lam_i h_{i-1};  grate += -dt_i a_i g_a ;  gdt_i += sum -r a_i g_a
//   gh0 = a_0 lam_0
void scan_bwd(const ScanShape& s, const float* v, const float* dt, const float* rate, const float* Bm,
              const float* Cm, const float* skip, const float* h0, const float* gy, const float* gH,
              const float* g_hlast, float* gv, float* gdt, float* grate, float* gB, float* gC, float* gskip,
              float* gh0, int n_threads) {
  const int64_t B = s.B, L = s.L, D = s.D, N = s.N, ns = s.n_save;
  // parallel over channels d; each worker owns its d's parameter grads and a private gdt buffer
  const int T = threads_for(n_threads, D);
  std::vector<std::vector<float>> gdt_part(T, std::vector<float>(B * L, 0.f));
  std::fill(grate, grate + D * N, 0.f);
  std::fill(gB, gB + D * N, 0.f);
  std::fill(gC, gC + D * N, 0.f);
  std::fill(gskip, gskip + D, 0.f);
  parallel_for(D, T, [&](int64_t lo, int64_t hi, int w) {
    std::vector<float> hs((L + 1) * N);  // hs[(i+1)*N + n] = h_i, hs[0..N) = h_{-1} = h0
    std::vector<float> lam(N), a_next_v(N);
    float* gdt_w = gdt_part[w].data();
    for (int64_t d = lo; d < hi; ++d) {
      const float* r = rate + d * N;
      const float* bm = Bm + d * N;
      const float* cm = Cm + d * N;
      for (int64_t b = 0; b < B; ++b) {
        for (int64_t n = 0; n < N; ++n) hs[n] = h0 ? h0[(b * D + d) * N + n] : 0.f;
        for (int64_t i = 0; i < L; ++i) {
          const float vi = v[(b * L + i) * D + d], di = dt[b * L + i];
          for (int64_t n = 0; n < N; ++n)
            hs[(i + 1) * N + n] = std::exp(-r[n] * di) * hs[i * N + n] + bm[n] * vi;
        }
        for (int64_t n = 0; n < N; ++n) {
          lam[n] = g_hlast ? g_hlast[(b * D + d) * N + n] : 0.f;
          a_next_v[n] = 1.f;                     // a_L := 1 (g_hlast enters lam_{L-1} unscaled)
        }
        for (int64_t i = L - 1; i >= 0; --i) {
          const float vi = v[(b * L + i) * D + d], di = dt[b * L + i];
          const float g = gy[(b * L + i) * D + d];
          float gvi = skip[d] * g;
          float gdti = 0.f;
          for (int64_t n = 0; n < N; ++n) {
            float q = cm[n] * g;
            if (gH && d < ns) q += gH[((b * L + i) * ns + d) * N + n];
            lam[n] = q + a_next_v[n] * lam[n];
            const float a = std::exp(-r[n] * di);
            const float ga = lam[n] * hs[i * N + n];
            gB[d * N + n] += lam[n] * vi;
            gC[d * N + n] += g * hs[(i + 1) * N + n];
            grate[d * N + n] += -di * a * ga;
            gdti += -r[n] * a * ga;
            gvi += lam[n] * bm[n];
            a_next_v[n] = a;
          }
          gskip[d] += g * vi;
          gv[(b * L + i) * D + d] = gvi;
          gdt_w[b * L + i] += gdti;
        }
        if (gh0)
          for (int64_t n = 0; n < N; ++n) gh0[(b * D + d) * N + n] = a_next_v[n] * lam[n];
      }
    }
  });
  for (int64_t j = 0; j < B * L; ++j) {
    float acc = 0.f;
    for (int w = 0; w < T; ++w) acc += gdt_part[w][j];
    gdt[j] = acc;
  }
}

// ---------------------------------------------------------------------------------- readout
void readout_fwd(const ReadoutShape& s, const float* H, const float* tau, const float* rate, const float* Cm,
                 float* Y, int n_threads) {
  const int64_t items = s.B * s.L, S = s.S, n = s.n, N = s.N;
  parallel_for(items, threads_for(n_threads, items), [&](int64_t lo, int64_t hi, int) {
    for (int64_t it = lo; it < hi; ++it)
      for (int64_t k = 0; k < S; ++k) {
        const float t = tau[it * S + k];
        for (int64_t d = 0; d < n; ++d) {
          float acc = 0.f;
          for (int64_t m = 0; m < N; ++m)
            acc += Cm[d * N + m] * std::exp(-rate[d * N + m] * t) * H[(it * n + d) * N + m];
          Y[(it * S + k) * n + d] = acc;
        }
      }
  });
}

void readout_bwd(const ReadoutShape& s, const float* H, const float* tau, const float* rate, const float* Cm,
                 const float* gY, float* gH, float* gtau, float* grate, float* gC, int n_threads) {
  const int64_t items = s.B * s.L, S = s.S, n = s.n, N = s.N;
  const int T = threads_for(n_threads, items);
  std::vector<std::vector<float>> gr(T, std::vector<float>(n * N, 0.f)), gc(T, std::vector<float>(n * N, 0.f));
  parallel_for(items, T, [&](int64_t lo, int64_t hi, int w) {
    for (int64_t it = lo; it < hi; ++it) {
      for (int64_t j = 0; j < n * N; ++j) gH[it * n * N + j] = 0.f;
      for (int64_t k = 0; k < S; ++k) {
        const float t = tau[it * S + k];
        float gt = 0.f;
        for (int64_t d = 0; d < n; ++d) {
          const float g = gY[(it * S + k) * n + d];
          for (int64_t m = 0; m < N; ++m) {
            const float r = rate[d * N + m], c = Cm[d * N + m], h = H[(it * n + d) * N + m];
            const float e = std::exp(-r * t);
            gH[(it * n + d) * N + m] += g * c * e;
            gt += g * c * h * e * (-r);
            gr[w][d * N + m] += g * c * h * e * (-t);
            gc[w][d * N + m] += g * h * e;
          }
        }
        gtau[it * S + k] = gt;
      }
    }
  });
  for (int64_t j = 0; j < n * N; ++j) {
    float a = 0.f, c = 0.f;
    for (int w = 0; w < T; ++w) { a += gr[w][j]; c += gc[w][j]; }
    grate[j] = a;
    gC[j] = c;
  }
}

}  // namespace evkit::cpu
