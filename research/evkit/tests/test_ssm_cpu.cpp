// CPU reference ops: forward vs a double-precision naive implementation, backward vs double-precision
// central finite differences of a random scalar loss.
#include <cstdio>

#include "check.hpp"
#include "evkit/ssm_cpu.hpp"

namespace {

struct ScanProblem {
  evkit::ScanShape s;
  std::vector<float> v, dt, rate, Bm, Cm, skip, h0, wy, wH, wh;
};

ScanProblem make(int64_t B, int64_t L, int64_t D, int64_t N, int64_t ns, uint32_t seed) {
  std::mt19937 g(seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(0.f, 1.f);
  ScanProblem p;
  p.s = {B, L, D, N, ns};
  auto fill = [&](std::vector<float>& x, size_t n, float sc) { x.resize(n); for (auto& e : x) e = sc * nd(g); };
  fill(p.v, B * L * D, 1.f);
  p.dt.resize(B * L);
  for (auto& e : p.dt) e = std::exp(std::log(1e-5f) + ud(g) * std::log(1e4f));   // 1e-5 .. 0.1 s
  p.rate.resize(D * N);
  for (auto& e : p.rate) e = std::exp(std::log(1.f) + ud(g) * std::log(1e4f));    // 1 .. 1e4 /s
  fill(p.Bm, D * N, 0.5f); fill(p.Cm, D * N, 0.5f); fill(p.skip, D, 1.f); fill(p.h0, B * D * N, 0.3f);
  fill(p.wy, B * L * D, 1.f); fill(p.wH, B * L * ns * N, 1.f); fill(p.wh, B * D * N, 1.f);
  return p;
}

// double-precision loss = <wy, y> + <wH, H_save> + <wh, h_last>
double loss_d(const ScanProblem& p, const std::vector<double>& v, const std::vector<double>& dt,
              const std::vector<double>& rate, const std::vector<double>& Bm, const std::vector<double>& Cm,
              const std::vector<double>& skip, const std::vector<double>& h0) {
  const auto& s = p.s;
  double L = 0;
  for (int64_t b = 0; b < s.B; ++b)
    for (int64_t d = 0; d < s.D; ++d) {
      std::vector<double> h(s.N);
      for (int64_t n = 0; n < s.N; ++n) h[n] = h0[(b * s.D + d) * s.N + n];
      for (int64_t i = 0; i < s.L; ++i) {
        const double vi = v[(b * s.L + i) * s.D + d];
        double y = skip[d] * vi;
        for (int64_t n = 0; n < s.N; ++n) {
          h[n] = std::exp(-rate[d * s.N + n] * dt[b * s.L + i]) * h[n] + Bm[d * s.N + n] * vi;
          y += Cm[d * s.N + n] * h[n];
          if (d < s.n_save) L += p.wH[((b * s.L + i) * s.n_save + d) * s.N + n] * h[n];
        }
        L += p.wy[(b * s.L + i) * s.D + d] * y;
      }
      for (int64_t n = 0; n < s.N; ++n) L += p.wh[(b * s.D + d) * s.N + n] * h[n];
    }
  return L;
}

std::vector<double> to_d(const std::vector<float>& x) { return std::vector<double>(x.begin(), x.end()); }

}  // namespace

int main() {
  const ScanProblem p = make(2, 37, 5, 4, 3, 1);
  const auto& s = p.s;
  // ------------------------------------------------ forward
  std::vector<float> y(s.B * s.L * s.D), H(s.B * s.L * s.n_save * s.N), hl(s.B * s.D * s.N);
  evkit::cpu::scan_fwd(s, p.v.data(), p.dt.data(), p.rate.data(), p.Bm.data(), p.Cm.data(), p.skip.data(),
                       p.h0.data(), y.data(), H.data(), hl.data(), 3);
  double Lf = 0;
  for (size_t i = 0; i < y.size(); ++i) Lf += p.wy[i] * y[i];
  for (size_t i = 0; i < H.size(); ++i) Lf += p.wH[i] * H[i];
  for (size_t i = 0; i < hl.size(); ++i) Lf += p.wh[i] * hl[i];
  const auto v = to_d(p.v), dt = to_d(p.dt), rate = to_d(p.rate), Bm = to_d(p.Bm), Cm = to_d(p.Cm),
             skip = to_d(p.skip), h0 = to_d(p.h0);
  CHECK_CLOSE(Lf, loss_d(p, v, dt, rate, Bm, Cm, skip, h0), 1e-5, 1e-4);

  // ------------------------------------------------ backward vs finite differences
  std::vector<float> gv(v.size()), gdt(dt.size()), gr(rate.size()), gB(Bm.size()), gC(Cm.size()), gs(skip.size()),
      gh0(h0.size());
  evkit::cpu::scan_bwd(s, p.v.data(), p.dt.data(), p.rate.data(), p.Bm.data(), p.Cm.data(), p.skip.data(),
                       p.h0.data(), p.wy.data(), p.wH.data(), p.wh.data(), gv.data(), gdt.data(), gr.data(),
                       gB.data(), gC.data(), gs.data(), gh0.data(), 3);
  auto fd = [&](int which, size_t idx) {
    std::vector<double> a[7] = {v, dt, rate, Bm, Cm, skip, h0};
    const double eps = 1e-6 * std::max(1.0, std::fabs(a[which][idx]));
    a[which][idx] += eps;
    const double lp = loss_d(p, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    a[which][idx] -= 2 * eps;
    const double lm = loss_d(p, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    return (lp - lm) / (2 * eps);
  };
  const std::vector<float>* grads[7] = {&gv, &gdt, &gr, &gB, &gC, &gs, &gh0};
  const char* names[7] = {"v", "dt", "rate", "Bm", "Cm", "skip", "h0"};
  for (int w = 0; w < 7; ++w) {
    const size_t n = grads[w]->size();
    for (size_t k = 0; k < std::min<size_t>(n, 25); ++k) {
      const size_t idx = (k * 7919) % n;
      const double num = fd(w, idx), ana = (*grads[w])[idx];
      if (!(std::fabs(num - ana) <= 2e-3 * std::max(1.0, std::fabs(num)))) {
        std::fprintf(stderr, "grad %s[%zu]: analytic %.6g numeric %.6g\n", names[w], idx, ana, num);
        return 1;
      }
    }
  }

  // ------------------------------------------------ readout fwd/bwd
  {
    const evkit::ReadoutShape r{2, 7, 5, 3, 4};
    std::mt19937 g(9);
    std::normal_distribution<float> nd;
    std::uniform_real_distribution<float> ud;
    std::vector<float> Hh(r.B * r.L * r.n * r.N), tau(r.B * r.L * r.S), rt(r.n * r.N), C(r.n * r.N),
        wY(r.B * r.L * r.S * r.n);
    for (auto& e : Hh) e = nd(g);
    for (auto& e : tau) e = std::exp(std::log(1e-6f) + ud(g) * std::log(1e5f));
    for (auto& e : rt) e = std::exp(ud(g) * std::log(1e4f));
    for (auto& e : C) e = nd(g);
    for (auto& e : wY) e = nd(g);
    std::vector<float> Y(wY.size());
    evkit::cpu::readout_fwd(r, Hh.data(), tau.data(), rt.data(), C.data(), Y.data(), 2);
    auto loss = [&](std::vector<double> H_, std::vector<double> t_, std::vector<double> r_, std::vector<double> c_) {
      double L = 0;
      for (int64_t it = 0; it < r.B * r.L; ++it)
        for (int64_t k = 0; k < r.S; ++k)
          for (int64_t d = 0; d < r.n; ++d) {
            double acc = 0;
            for (int64_t m = 0; m < r.N; ++m)
              acc += c_[d * r.N + m] * std::exp(-r_[d * r.N + m] * t_[it * r.S + k]) * H_[(it * r.n + d) * r.N + m];
            L += wY[(it * r.S + k) * r.n + d] * acc;
          }
      return L;
    };
    double Lf2 = 0;
    for (size_t i = 0; i < Y.size(); ++i) Lf2 += wY[i] * Y[i];
    CHECK_CLOSE(Lf2, loss(to_d(Hh), to_d(tau), to_d(rt), to_d(C)), 1e-5, 1e-4);
    std::vector<float> gH(Hh.size()), gt(tau.size()), grr(rt.size()), gc(C.size());
    evkit::cpu::readout_bwd(r, Hh.data(), tau.data(), rt.data(), C.data(), wY.data(), gH.data(), gt.data(), grr.data(),
                            gc.data(), 2);
    std::vector<double> base[4] = {to_d(Hh), to_d(tau), to_d(rt), to_d(C)};
    const std::vector<float>* gg[4] = {&gH, &gt, &grr, &gc};
    for (int w = 0; w < 4; ++w)
      for (size_t k = 0; k < std::min<size_t>(gg[w]->size(), 20); ++k) {
        const size_t idx = (k * 104729) % gg[w]->size();
        std::vector<double> a[4] = {base[0], base[1], base[2], base[3]};   // real copy (a C-array would decay)
        const double eps = 1e-6 * std::max(1.0, std::fabs(a[w][idx]));
        a[w][idx] += eps;
        const double lp = loss(a[0], a[1], a[2], a[3]);
        a[w][idx] -= 2 * eps;
        const double lm = loss(a[0], a[1], a[2], a[3]);
        const double num = (lp - lm) / (2 * eps), ana = (*gg[w])[idx];
        if (!(std::fabs(num - ana) <= 2e-3 * std::max(1.0, std::fabs(num)))) {
          std::fprintf(stderr, "readout grad %d[%zu]: analytic %.6g numeric %.6g\n", w, idx, ana, num);
          return 1;
        }
      }
  }
  std::printf("test_ssm_cpu OK\n");
  return 0;
}
