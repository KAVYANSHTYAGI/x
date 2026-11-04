// GPU kernels vs the CPU reference.  Run on a CUDA machine:  ./build/test_cuda
// Checks every (N, scheduling mode, fast/precise exp) combination, forward and backward, plus readout and
// bit-identical tokenisation.  Exits non-zero on the first mismatch.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "evkit/cuda_ops.hpp"
#include "evkit/ssm_cpu.hpp"
#include "evkit/tokenize.hpp"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); std::exit(2); } } while (0)

template <class T>
struct Dev {
  T* p = nullptr;
  size_t n = 0;
  explicit Dev(size_t n_) : n(n_) { if (n) CK(cudaMalloc(&p, n * sizeof(T))); }
  Dev(const std::vector<T>& h) : Dev(h.size()) { if (n) CK(cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice)); }
  ~Dev() { if (p) cudaFree(p); }
  std::vector<T> get() const { std::vector<T> h(n); if (n) CK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return h; }
};

static double max_rel(const std::vector<float>& a, const std::vector<float>& b, double atol) {
  double m = 0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]) / (atol + std::fabs(b[i])));
  return m;
}

static int failures = 0;
static void expect(const char* what, double err, double tol) {
  const bool ok = err <= tol;
  std::printf("  %-34s max rel err %.3e  %s\n", what, err, ok ? "ok" : "FAIL");
  failures += !ok;
}

int main() {
  std::mt19937 g(0);
  std::normal_distribution<float> nd;
  std::uniform_real_distribution<float> ud;
  for (int N : {8, 16, 32}) {
    const evkit::ScanShape s{3, 517, 70, N, 12};
    const int64_t BLD = s.B * s.L * s.D, DN = s.D * s.N;
    std::vector<float> v(BLD), dt(s.B * s.L), rate(DN), Bm(DN), Cm(DN), skip(s.D), h0(s.B * DN), gy(BLD),
        gH(s.B * s.L * s.n_save * s.N), ghl(s.B * DN);
    for (auto& e : v) e = nd(g);
    for (auto& e : dt) e = std::exp(std::log(1e-6f) + ud(g) * std::log(5e4f));
    for (auto& e : rate) e = std::exp(ud(g) * std::log(2e4f));
    for (auto* x : {&Bm, &Cm, &skip, &h0, &gy, &gH, &ghl}) for (auto& e : *x) e = 0.5f * nd(g);
    std::vector<float> y(BLD), H(gH.size()), hl(ghl.size());
    evkit::cpu::scan_fwd(s, v.data(), dt.data(), rate.data(), Bm.data(), Cm.data(), skip.data(), h0.data(), y.data(),
                         H.data(), hl.data());
    std::vector<float> gv(BLD), gdt(dt.size()), gr(DN), gB(DN), gC(DN), gs(s.D), gh0(h0.size());
    evkit::cpu::scan_bwd(s, v.data(), dt.data(), rate.data(), Bm.data(), Cm.data(), skip.data(), h0.data(), gy.data(),
                         gH.data(), ghl.data(), gv.data(), gdt.data(), gr.data(), gB.data(), gC.data(), gs.data(), gh0.data());
    Dev<float> dv(v), ddt(dt), drate(rate), dBm(Bm), dCm(Cm), dskip(skip), dh0(h0), dgy(gy), dgH(gH), dghl(ghl);
    for (int mode : {1, 2})
      for (bool fast : {false, true}) {
        std::printf("N=%d mode=%s exp=%s\n", N, mode == 1 ? "sequential" : "chunked", fast ? "ex2.approx" : "exp2f");
        Dev<float> oy(BLD), oH(H.size()), ohl(hl.size()), ckpt(s.B * evkit::gpu::num_chunks(s.L) * DN),
            ws(evkit::gpu::scan_fwd_workspace_floats(s));
        evkit::gpu::scan_fwd(s, dv.p, ddt.p, drate.p, dBm.p, dCm.p, dskip.p, dh0.p, oy.p, oH.p, ohl.p, ckpt.p, ws.p, fast,
                             static_cast<evkit::gpu::ScanMode>(mode), nullptr);
        CK(cudaDeviceSynchronize());
        const double tol = fast ? 5e-3 : 5e-5;
        expect("fwd y", max_rel(oy.get(), y, 1e-3), tol);
        expect("fwd H_save", max_rel(oH.get(), H, 1e-3), tol);
        expect("fwd h_last", max_rel(ohl.get(), hl, 1e-3), tol);
        Dev<float> ogv(BLD), ogdt(dt.size()), ogr(DN), ogB(DN), ogC(DN), ogs(s.D), ogh0(h0.size()),
            wsb(evkit::gpu::scan_bwd_workspace_floats(s));
        evkit::gpu::scan_bwd(s, dv.p, ddt.p, drate.p, dBm.p, dCm.p, dskip.p, dh0.p, ckpt.p, dgy.p, dgH.p, dghl.p, ogv.p,
                             ogdt.p, ogr.p, ogB.p, ogC.p, ogs.p, ogh0.p, wsb.p, fast,
                             static_cast<evkit::gpu::ScanMode>(mode), nullptr);
        CK(cudaDeviceSynchronize());
        const double tb = fast ? 1e-2 : 2e-4;
        expect("bwd gv", max_rel(ogv.get(), gv, 1e-2), tb);
        expect("bwd gdt", max_rel(ogdt.get(), gdt, 1e-2), tb);
        expect("bwd grate", max_rel(ogr.get(), gr, 1e-2), tb);
        expect("bwd gB", max_rel(ogB.get(), gB, 1e-2), tb);
        expect("bwd gC", max_rel(ogC.get(), gC, 1e-2), tb);
        expect("bwd gskip", max_rel(ogs.get(), gs, 1e-2), tb);
        expect("bwd gh0", max_rel(ogh0.get(), gh0, 1e-2), tb);
      }
    // readout
    const evkit::ReadoutShape r{3, 129, 9, 16, N};
    std::vector<float> RH(r.B * r.L * r.n * r.N), tau(r.B * r.L * r.S), rr(r.n * r.N), rc(r.n * r.N), gY(r.B * r.L * r.S * r.n);
    for (auto& e : RH) e = nd(g);
    for (auto& e : tau) e = std::exp(std::log(1e-7f) + ud(g) * std::log(3e5f));
    for (auto& e : rr) e = std::exp(ud(g) * std::log(1e4f));
    for (auto& e : rc) e = nd(g);
    for (auto& e : gY) e = nd(g);
    std::vector<float> Y(gY.size()), gH2(RH.size()), gt(tau.size()), grr(rr.size()), grc(rc.size());
    evkit::cpu::readout_fwd(r, RH.data(), tau.data(), rr.data(), rc.data(), Y.data());
    evkit::cpu::readout_bwd(r, RH.data(), tau.data(), rr.data(), rc.data(), gY.data(), gH2.data(), gt.data(), grr.data(), grc.data());
    Dev<float> dH(RH), dtau(tau), dr(rr), dc(rc), dgY(gY), dY(Y.size()), dgH2(RH.size()), dgt(tau.size()), dgr(rr.size()),
        dgc(rc.size()), rws(evkit::gpu::readout_bwd_workspace_floats(r));
    evkit::gpu::readout_fwd(r, dH.p, dtau.p, dr.p, dc.p, dY.p, false, nullptr);
    evkit::gpu::readout_bwd(r, dH.p, dtau.p, dr.p, dc.p, dgY.p, dgH2.p, dgt.p, dgr.p, dgc.p, rws.p, false, nullptr);
    CK(cudaDeviceSynchronize());
    std::printf("readout N=%d\n", N);
    expect("readout Y", max_rel(dY.get(), Y, 1e-3), 5e-5);
    expect("readout gH", max_rel(dgH2.get(), gH2, 1e-2), 2e-4);
    expect("readout gtau", max_rel(dgt.get(), gt, 1e-2), 2e-4);
    expect("readout grate", max_rel(dgr.get(), grr, 1e-2), 2e-4);
    expect("readout gC", max_rel(dgc.get(), grc, 1e-2), 2e-4);
  }
  // tokenisation: bit-identical
  {
    const size_t n = 1 << 20;
    std::vector<int64_t> t(n);
    std::vector<uint16_t> x(n), y(n);
    std::vector<uint8_t> p(n);
    int64_t acc = 0;
    for (size_t i = 0; i < n; ++i) { acc += g() % 50; t[i] = acc; x[i] = g() % 1280; y[i] = g() % 720; p[i] = g() & 1; }
    const evkit::TokenizeParams prm{1280, 720, 16, 4, 77, 0, -1};
    std::vector<int32_t> a(n), b(n), c(n);
    std::vector<float> u(n), vv(n), d(n);
    evkit::tokenize_cpu(t.data(), x.data(), y.data(), p.data(), n, prm, a.data(), b.data(), c.data(), u.data(), vv.data(), d.data());
    Dev<int64_t> dt_(t); Dev<uint16_t> dx(x), dy(y); Dev<uint8_t> dp(p);
    Dev<int32_t> da(n), db(n), dc2(n); Dev<float> du(n), dv2(n), dd(n);
    evkit::gpu::tokenize(dt_.p, dx.p, dy.p, dp.p, n, prm, da.p, db.p, dc2.p, du.p, dv2.p, dd.p, nullptr);
    CK(cudaDeviceSynchronize());
    const bool same = da.get() == a && db.get() == b && dd.get() == d && du.get() == u;
    std::printf("tokenize bit-identical: %s\n", same ? "ok" : "FAIL");
    failures += !same;
  }
  std::printf(failures ? "\n%d FAILURES\n" : "\nALL OK\n", failures);
  return failures ? 1 : 0;
}
