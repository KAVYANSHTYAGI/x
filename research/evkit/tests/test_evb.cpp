#include <algorithm>
#include <cstdio>

#include "check.hpp"
#include "evkit/evb.hpp"

int main() {
  const auto ev = random_events(300000, 640, 480, 25e6, 11);
  evkit::write_evb("/tmp/evkit_test.evb", ev, 640, 480, "IMX636", 1000);
  evkit::EvbReader r("/tmp/evkit_test.evb");
  CHECK(r.size() == ev.size());
  CHECK(r.header().width == 640 && r.header().height == 480);
  for (size_t i = 0; i < ev.size(); i += 997) {
    CHECK(r.t()[i] == ev.t[i] && r.x()[i] == ev.x[i] && r.y()[i] == ev.y[i] && r.p()[i] == ev.p[i]);
  }
  std::mt19937_64 g(3);
  for (int k = 0; k < 2000; ++k) {
    const int64_t a = static_cast<int64_t>(g() % 26000000) - 500000;
    const int64_t b = a + static_cast<int64_t>(g() % 3000000);
    auto [lo, hi] = r.time_range(a, b);
    const size_t lo_ref = std::lower_bound(ev.t.begin(), ev.t.end(), a) - ev.t.begin();
    const size_t hi_ref = std::lower_bound(ev.t.begin(), ev.t.end(), b) - ev.t.begin();
    CHECK(lo == lo_ref && hi == hi_ref);
  }
  std::printf("test_evb OK\n");
  return 0;
}
