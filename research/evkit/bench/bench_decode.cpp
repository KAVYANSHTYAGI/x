// Decoder / tokenizer throughput on synthetic EVT2 / EVT3 files (bursty, 1280x720).
//   bench_decode [n_events=20000000]
#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>

#include "../tests/check.hpp"
#include "evkit/events.hpp"
#include "evkit/tokenize.hpp"

template <class F>
static double secs(F&& f) {
  const auto a = std::chrono::steady_clock::now();
  f();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
}

int main(int argc, char** argv) {
  const size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 20000000ull;
  const auto ev = random_events(n, 1280, 720, n / 10.0, 1);   // ~10 Mev/s scene
  {
    auto w = evkit::encode_evt3(ev, true);
    std::ofstream f("/tmp/evkit_bench.raw", std::ios::binary);
    f << "% format EVT3;height=720;width=1280\n% end\n";
    f.write(reinterpret_cast<const char*>(w.data()), w.size() * 2);
    std::printf("EVT3: %.2f bytes/event\n", 2.0 * w.size() / n);
  }
  {
    auto w = evkit::encode_evt2(ev);
    std::ofstream f("/tmp/evkit_bench2.raw", std::ios::binary);
    f << "% format EVT2;height=720;width=1280\n% end\n";
    f.write(reinterpret_cast<const char*>(w.data()), w.size() * 4);
  }
  const int hw = std::max(1u, std::thread::hardware_concurrency());
  for (const char* path : {"/tmp/evkit_bench.raw", "/tmp/evkit_bench2.raw"})
    for (int th : {1, hw}) {
      evkit::EventBuffer out;
      const double s = secs([&] { evkit::decode_file(path, out, th); });
      std::printf("decode %-22s threads=%-2d %8.1f Mev/s\n", path, th, out.size() / s * 1e-6);
    }
  std::vector<int32_t> a(n), b(n), c(n);
  std::vector<float> u(n), v(n), dt(n);
  evkit::TokenizeParams prm{1280, 720, 16, 4, 7, 0, -1};
  for (int th : {1, hw}) {
    const double s = secs([&] {
      evkit::tokenize_cpu(ev.t.data(), ev.x.data(), ev.y.data(), ev.p.data(), n, prm, a.data(), b.data(), c.data(),
                          u.data(), v.data(), dt.data(), th);
    });
    std::printf("tokenize threads=%-2d %8.1f Mev/s\n", th, n / s * 1e-6);
  }
  return 0;
}
