// Kernel benchmark (run on a GPU):  ./build/bench_cuda
// Reports time, events/s and achieved DRAM bandwidth for the scan (both schedules) and the readout, at
// shapes typical for pretraining (many windows) and for long-stream inference (one huge stream).
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "evkit/cuda_ops.hpp"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("CUDA error %s\n", cudaGetErrorString(e)); return 2; } } while (0)

template <class F>
static float time_ms(F&& f, int iters = 20) {
  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  f();                                  // warm-up
  cudaEventRecord(a);
  for (int i = 0; i < iters; ++i) f();
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a); cudaEventDestroy(b);
  return ms / iters;
}

static float* alloc(size_t n, float val = 0.01f) {
  float* p;
  cudaMalloc(&p, n * sizeof(float));
  std::vector<float> h(n, val);
  cudaMemcpy(p, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  return p;
}

int main() {
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s, %d SMs\n\n", prop.name, prop.multiProcessorCount);
  struct Case { const char* name; evkit::ScanShape s; };
  const Case cases[] = {
      {"pretrain  B=32 L=4096  D=256 N=16", {32, 4096, 256, 16, 16}},
      {"pretrain  B=8  L=16384 D=512 N=16", {8, 16384, 512, 16, 16}},
      {"stream    B=1  L=1M    D=256 N=16", {1, 1 << 20, 256, 16, 16}},
      {"stream    B=1  L=1M    D=128 N=32", {1, 1 << 20, 128, 32, 16}},
  };
  std::printf("%-38s %-10s %9s %9s %12s %10s\n", "scan", "schedule", "fwd ms", "bwd ms", "Mevents/s", "fwd GB/s");
  for (const auto& c : cases) {
    const auto& s = c.s;
    const size_t BLD = s.B * s.L * s.D, DN = s.D * s.N, C = evkit::gpu::num_chunks(s.L);
    float *v = alloc(BLD), *dt = alloc(s.B * s.L, 1e-4f), *rate = alloc(DN, 100.f), *Bm = alloc(DN), *Cm = alloc(DN),
          *skip = alloc(s.D), *y = alloc(BLD), *H = alloc(s.B * s.L * s.n_save * s.N), *hl = alloc(s.B * DN),
          *ck = alloc(s.B * C * DN), *ws = alloc(evkit::gpu::scan_fwd_workspace_floats(s)), *gy = alloc(BLD),
          *gv = alloc(BLD), *gdt = alloc(s.B * s.L), *g1 = alloc(DN), *g2 = alloc(DN), *g3 = alloc(DN), *g4 = alloc(s.D),
          *wsb = alloc(evkit::gpu::scan_bwd_workspace_floats(s));
    for (int mode : {1, 2}) {
      const auto m = static_cast<evkit::gpu::ScanMode>(mode);
      const float f = time_ms([&] { evkit::gpu::scan_fwd(s, v, dt, rate, Bm, Cm, skip, nullptr, y, H, hl, ck, ws, true, m, nullptr); });
      const float b = time_ms([&] {
        evkit::gpu::scan_bwd(s, v, dt, rate, Bm, Cm, skip, nullptr, ck, gy, nullptr, nullptr, gv, gdt, g1, g2, g3, g4,
                             nullptr, wsb, true, m, nullptr);
      });
      const double bytes = 4.0 * (2.0 * BLD + s.B * s.L + s.B * s.L * s.n_save * s.N + s.B * C * DN);
      std::printf("%-38s %-10s %9.3f %9.3f %12.1f %10.1f\n", c.name, mode == 1 ? "seq" : "chunked", f, b,
                  s.B * s.L / ((f + b) * 1e-3) * 1e-6, bytes / (f * 1e-3) * 1e-9);
    }
    for (float* p : {v, dt, rate, Bm, Cm, skip, y, H, hl, ck, ws, gy, gv, gdt, g1, g2, g3, g4, wsb}) cudaFree(p);
  }
  {
    const evkit::ReadoutShape r{32, 4096, 9, 16, 16};
    float *H = alloc(r.B * r.L * r.n * r.N), *tau = alloc(r.B * r.L * r.S, 1e-4f), *rate = alloc(r.n * r.N, 100.f),
          *Cm = alloc(r.n * r.N), *Y = alloc(r.B * r.L * r.S * r.n);
    const float f = time_ms([&] { evkit::gpu::readout_fwd(r, H, tau, rate, Cm, Y, true, nullptr); });
    std::printf("\nreadout B=32 L=4096 S=9 n=16 N=16: %.3f ms fwd (%.1f GB/s)\n", f,
                4.0 * (r.B * r.L * (r.n * r.N + r.S + r.S * r.n)) / (f * 1e-3) * 1e-9);
  }
  return 0;
}
