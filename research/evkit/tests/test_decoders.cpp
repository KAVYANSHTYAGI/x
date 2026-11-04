// Round-trip and parallel-exactness tests for EVT2 / EVT3 / DAT.
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>

#include "check.hpp"
#include "evkit/events.hpp"

static std::string tmp(const std::string& name) { return std::string("/tmp/evkit_test_") + name; }

template <class W>
static void write_raw(const std::string& path, const std::string& header, const std::vector<W>& words) {
  std::ofstream f(path, std::ios::binary);
  f << header;
  f.write(reinterpret_cast<const char*>(words.data()), words.size() * sizeof(W));
}

int main() {
  // 40 s span -> two wrap-arounds of the EVT3 24-bit time base (16.78 s)
  const auto ev = random_events(200000, 1280, 720, 40e6, 7);
  CHECK(ev.is_time_sorted());

  // ---------------- EVT2
  write_raw(tmp("a.raw"), "% camera_integrator_name Prophesee\n% format EVT2;height=720;width=1280\n% end\n", evkit::encode_evt2(ev));
  for (int th : {1, 3, 8}) {
    evkit::EventBuffer out;
    auto h = evkit::decode_file(tmp("a.raw"), out, th);
    CHECK(h.format == evkit::RawFormat::kEvt2 && h.width == 1280 && h.height == 720);
    CHECK(same_events(out, ev));
  }
  // ---------------- EVT3 (with vectors) - exact round trip, any thread count
  auto w3 = evkit::encode_evt3(ev, true);
  write_raw(tmp("b.raw"), "% format EVT3;height=720;width=1280\n% end\n", w3);
  for (int th : {1, 2, 5, 16}) {
    evkit::EventBuffer out;
    auto h = evkit::decode_file(tmp("b.raw"), out, th);
    CHECK(h.format == evkit::RawFormat::kEvt3);
    if (!same_events(out, ev)) {
      std::fprintf(stderr, "EVT3 mismatch with %d threads: got %zu events, want %zu\n", th, out.size(), ev.size());
      for (size_t i = 0; i < std::min(out.size(), ev.size()); ++i)
        if (out.t[i] != ev.t[i] || out.x[i] != ev.x[i] || out.y[i] != ev.y[i] || out.p[i] != ev.p[i]) {
          std::fprintf(stderr, "  first diff at %zu: got (%lld,%d,%d,%d) want (%lld,%d,%d,%d)\n", i, (long long)out.t[i],
                       out.x[i], out.y[i], out.p[i], (long long)ev.t[i], ev.x[i], ev.y[i], ev.p[i]);
          break;
        }
      return 1;
    }
  }
  // vectors must actually have been used (otherwise the vector path is untested)
  size_t n_vec = 0;
  for (auto w : w3) n_vec += (w >> 12) == 0x4;
  CHECK(n_vec > 1000);
  // EVT3 without vectors
  write_raw(tmp("c.raw"), "% format EVT3;height=720;width=1280\n% end\n", evkit::encode_evt3(ev, false));
  {
    evkit::EventBuffer out;
    evkit::decode_file(tmp("c.raw"), out, 4);
    CHECK(same_events(out, ev));
  }
  // ---------------- streaming decoder fed in odd-sized pieces == whole-buffer decode
  {
    evkit::Evt3Decoder d;
    evkit::EventBuffer out;
    size_t pos = 0, step = 1;
    while (pos < w3.size()) {
      const size_t k = std::min(step, w3.size() - pos);
      d.decode(w3.data() + pos, k, out);
      pos += k;
      step = step * 3 % 977 + 1;
    }
    CHECK(same_events(out, ev));
  }
  // ---------------- DAT
  {
    auto bytes = evkit::encode_dat(ev);
    std::ofstream f(tmp("d.dat"), std::ios::binary);
    f.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    f.close();
    evkit::EventBuffer out;
    auto h = evkit::decode_file(tmp("d.dat"), out, 4);
    CHECK(h.format == evkit::RawFormat::kDat);
    CHECK(same_events(out, ev));
  }
  // ---------------- sort_by_time on a shuffled copy restores order (stable)
  {
    evkit::EventBuffer sh = ev;
    std::reverse(sh.t.begin(), sh.t.end()); std::reverse(sh.x.begin(), sh.x.end());
    std::reverse(sh.y.begin(), sh.y.end()); std::reverse(sh.p.begin(), sh.p.end());
    CHECK(sh.sort_by_time(4));
    CHECK(sh.is_time_sorted());
  }
  std::printf("test_decoders OK (%zu events, %zu EVT3 words, %zu VECT_12)\n", ev.size(), w3.size(), n_vec);
  return 0;
}
