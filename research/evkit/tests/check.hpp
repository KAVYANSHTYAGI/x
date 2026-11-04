// Minimal test helpers (no external framework).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "evkit/events.hpp"

#define CHECK(cond)                                                                 \
  do {                                                                              \
    if (!(cond)) {                                                                  \
      std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
      std::exit(1);                                                                 \
    }                                                                               \
  } while (0)

#define CHECK_CLOSE(a, b, rtol, atol)                                                                     \
  do {                                                                                                    \
    const double _a = (a), _b = (b);                                                                      \
    if (!(std::fabs(_a - _b) <= (atol) + (rtol) * std::fabs(_b))) {                                       \
      std::fprintf(stderr, "CHECK_CLOSE failed %s:%d: %s=%.9g vs %s=%.9g\n", __FILE__, __LINE__, #a, _a, #b, _b); \
      std::exit(1);                                                                                       \
    }                                                                                                     \
  } while (0)

// Random time-sorted events with bursts of identical timestamps on one row (exercises EVT3 vectors).
inline evkit::EventBuffer random_events(size_t n, int W, int H, double duration_us, uint32_t seed) {
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<double> ut(0.0, duration_us);
  std::uniform_int_distribution<int> ux(0, W - 1), uy(0, H - 1), up(0, 1), burst(1, 6);
  std::vector<int64_t> times;
  while (times.size() < n) times.push_back(static_cast<int64_t>(ut(g)));
  std::sort(times.begin(), times.end());
  evkit::EventBuffer ev;
  size_t i = 0;
  while (ev.size() < n) {
    const int64_t t = times[i++ % times.size()];
    const int y = uy(g), p = up(g);
    int x = ux(g);
    const int k = burst(g);
    for (int j = 0; j < k && ev.size() < n; ++j) {      // same t, y, p; increasing x
      if (x >= W) break;
      ev.push(static_cast<uint16_t>(x), static_cast<uint16_t>(y), static_cast<uint8_t>(p), t);
      x += 1 + static_cast<int>(g() % 3);
    }
  }
  // keep global time order
  std::vector<size_t> idx(ev.size());
  for (size_t k = 0; k < idx.size(); ++k) idx[k] = k;
  std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return ev.t[a] < ev.t[b]; });
  evkit::EventBuffer out;
  for (size_t k : idx) out.push(ev.x[k], ev.y[k], ev.p[k], ev.t[k]);
  return out;
}

inline bool same_events(const evkit::EventBuffer& a, const evkit::EventBuffer& b) {
  return a.t == b.t && a.x == b.x && a.y == b.y && a.p == b.p;
}
