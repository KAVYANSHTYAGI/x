// evkit-convert: raw sensor file (.raw EVT2/EVT3, .dat Gen1/1Mpx) -> EVB training store.
//   evkit-convert input.raw output.evb [--threads N] [--sensor TAG] [--width W] [--height H] [--format evt2|evt3|dat]
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#include "evkit/evb.hpp"
#include "evkit/events.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s input.{raw,dat} output.evb [--threads N] [--sensor TAG] [--width W] [--height H] [--format F]\n", argv[0]);
    return 2;
  }
  std::string in = argv[1], out = argv[2], sensor;
  int threads = 0, W = -1, H = -1;
  evkit::RawFormat hint = evkit::RawFormat::kUnknown;
  for (int i = 3; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--threads")) threads = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--sensor")) sensor = argv[i + 1];
    else if (!std::strcmp(argv[i], "--width")) W = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--height")) H = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--format")) {
      std::string f = argv[i + 1];
      hint = f == "evt2" ? evkit::RawFormat::kEvt2 : f == "evt3" ? evkit::RawFormat::kEvt3 : evkit::RawFormat::kDat;
    }
  }
  evkit::EventBuffer ev;
  const auto t0 = std::chrono::steady_clock::now();
  auto hdr = evkit::decode_file(in, ev, threads, hint);
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (W < 0) W = hdr.width;
  if (H < 0) H = hdr.height;
  if (W <= 0 || H <= 0) {
    W = 0; H = 0;
    for (size_t i = 0; i < ev.size(); ++i) { W = std::max<int>(W, ev.x[i] + 1); H = std::max<int>(H, ev.y[i] + 1); }
    std::fprintf(stderr, "warning: sensor size not in header, using bounding box %dx%d (pass --width/--height)\n", W, H);
  }
  const bool sorted = ev.sort_by_time(threads);
  evkit::write_evb(out, std::move(ev), W, H, sensor);
  evkit::EvbReader r(out);
  const double dur = (r.header().t1 - r.header().t0) * 1e-6;
  std::printf("%s: %llu events, %.2f s, %.2f Mev/s mean rate, %dx%d, decode %.1f Mev/s%s\n", out.c_str(),
              (unsigned long long)r.size(), dur, dur > 0 ? r.size() / dur * 1e-6 : 0.0, W, H,
              r.size() / sec * 1e-6, sorted ? " (re-sorted)" : "");
  return 0;
}
