#include "evkit/tokenize.hpp"

#include <algorithm>
#include <thread>
#include <vector>

namespace evkit {

void tokenize_cpu(const int64_t* t_us, const uint16_t* x, const uint16_t* y, const uint8_t* p, size_t n,
                  const TokenizeParams& prm, int32_t* patch, int32_t* sub, int32_t* pol, float* u, float* v,
                  float* dt, int n_threads) {
  int T = n_threads > 0 ? n_threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  T = std::max(1, std::min<int>(T, static_cast<int>(n / 262144) + 1));
  auto work = [&](size_t s, size_t e) {
    for (size_t i = s; i < e; ++i) {
      const int64_t prev = i ? t_us[i - 1] : prm.t_prev;
      tokenize_one(prm, i, t_us[i], prev, x[i], y[i], p[i], patch[i], sub[i], pol[i], u[i], v[i], dt[i]);
    }
  };
  if (T == 1) { work(0, n); return; }
  std::vector<std::thread> th;
  for (int k = 0; k < T; ++k) th.emplace_back(work, n * k / T, n * (k + 1) / T);
  for (auto& h : th) h.join();
}

}  // namespace evkit
