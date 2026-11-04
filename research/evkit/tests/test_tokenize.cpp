// Tokenisation: thread-count invariance and chunking invariance (index_offset + t_prev).
#include <cstdio>
#include <cstring>

#include "check.hpp"
#include "evkit/tokenize.hpp"

struct Tok {
  std::vector<int32_t> patch, sub, pol;
  std::vector<float> u, v, dt;
  explicit Tok(size_t n) : patch(n), sub(n), pol(n), u(n), v(n), dt(n) {}
};

static void run(const evkit::EventBuffer& ev, size_t s, size_t e, evkit::TokenizeParams prm, Tok& o, int th) {
  evkit::tokenize_cpu(ev.t.data() + s, ev.x.data() + s, ev.y.data() + s, ev.p.data() + s, e - s, prm,
                      o.patch.data() + s, o.sub.data() + s, o.pol.data() + s, o.u.data() + s, o.v.data() + s,
                      o.dt.data() + s, th);
}

int main() {
  const auto ev = random_events(500000, 1280, 720, 5e6, 5);
  const size_t n = ev.size();
  evkit::TokenizeParams prm{1280, 720, 16, 4, 1234, 0, -1};
  Tok a(n), b(n), c(n);
  run(ev, 0, n, prm, a, 1);
  run(ev, 0, n, prm, b, 8);
  const size_t cut = 123457;
  run(ev, 0, cut, prm, c, 3);
  evkit::TokenizeParams p2 = prm;
  p2.index_offset = cut;
  p2.t_prev = ev.t[cut - 1];
  run(ev, cut, n, p2, c, 3);
  for (const Tok* o : {&b, &c}) {
    CHECK(std::memcmp(a.patch.data(), o->patch.data(), n * 4) == 0);
    CHECK(std::memcmp(a.sub.data(), o->sub.data(), n * 4) == 0);
    CHECK(std::memcmp(a.dt.data(), o->dt.data(), n * 4) == 0);
    CHECK(std::memcmp(a.u.data(), o->u.data(), n * 4) == 0);
  }
  // semantics
  for (size_t i = 1; i < n; i += 1013) {
    const double gap = static_cast<double>(ev.t[i] - ev.t[i - 1]);
    CHECK(a.dt[i] >= gap * 1e-6 * (1 - 1e-6) && a.dt[i] < (gap + 1.0) * 1e-6 * (1 + 1e-6));
    CHECK(a.patch[i] >= 0 && a.patch[i] < 256 && a.sub[i] >= 0 && a.sub[i] < 16);
    const int cu = static_cast<int>((ev.x[i] + 0.5f) / 1280.f * 64), cv = static_cast<int>((ev.y[i] + 0.5f) / 720.f * 64);
    CHECK(a.patch[i] == (cv / 4) * 16 + cu / 4 && a.sub[i] == (cv % 4) * 4 + cu % 4);
  }
  CHECK(a.dt[0] > 0 && a.dt[0] < 1e-6f);
  std::printf("test_tokenize OK\n");
  return 0;
}
