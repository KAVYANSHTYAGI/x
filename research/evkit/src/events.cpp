// Raw event decoding.  Exactness contract (tested): decode_file(..., n_threads = k) produces the same
// events in the same order for every k, identical to a single sequential pass of the streaming decoder.
#include "evkit/events.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace evkit {

// ------------------------------------------------------------------------------------------ buffer
void EventBuffer::append(const EventBuffer& o) {
  t.insert(t.end(), o.t.begin(), o.t.end());
  x.insert(x.end(), o.x.begin(), o.x.end());
  y.insert(y.end(), o.y.begin(), o.y.end());
  p.insert(p.end(), o.p.begin(), o.p.end());
}

bool EventBuffer::is_time_sorted() const {
  for (size_t i = 1; i < t.size(); ++i)
    if (t[i] < t[i - 1]) return false;
  return true;
}

bool EventBuffer::sort_by_time(int n_threads) {
  if (is_time_sorted()) return false;
  const size_t n = t.size();
  std::vector<uint32_t> idx(n);
  std::iota(idx.begin(), idx.end(), 0u);
  auto less = [&](uint32_t a, uint32_t b) { return t[a] < t[b]; };
  int T = n_threads > 0 ? n_threads : std::max(1u, std::thread::hardware_concurrency());
  T = std::max(1, std::min<int>(T, static_cast<int>(n / 65536) + 1));
  std::vector<size_t> cut(T + 1);
  for (int k = 0; k <= T; ++k) cut[k] = n * k / T;
  {
    std::vector<std::thread> th;
    for (int k = 0; k < T; ++k)
      th.emplace_back([&, k] { std::stable_sort(idx.begin() + cut[k], idx.begin() + cut[k + 1], less); });
    for (auto& h : th) h.join();
  }
  for (int width = 1; width < T; width *= 2)          // pairwise stable merges
    for (int k = 0; k + width < T; k += 2 * width)
      std::inplace_merge(idx.begin() + cut[k], idx.begin() + cut[k + width],
                         idx.begin() + cut[std::min(T, k + 2 * width)], less);
  auto permute = [&](auto& v) {
    auto tmp = v;
    for (size_t i = 0; i < n; ++i) v[i] = tmp[idx[i]];
  };
  permute(t); permute(x); permute(y); permute(p);
  return true;
}

// ------------------------------------------------------------------------------------------ header
RawHeader parse_header(const uint8_t* data, size_t n, RawFormat hint) {
  RawHeader h;
  size_t pos = 0;
  while (pos < n && data[pos] == '%') {
    size_t e = pos;
    while (e < n && data[e] != '\n') ++e;
    std::string line(reinterpret_cast<const char*>(data + pos), e - pos);
    h.text += line + "\n";
    pos = (e < n) ? e + 1 : e;
    auto lower = line;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower.rfind("% end", 0) == 0) break;
    auto grab_int = [&](const std::string& key) -> int {
      auto k = lower.find(key);
      if (k == std::string::npos) return -1;
      return std::atoi(lower.c_str() + k + key.size());
    };
    if (lower.find("evt 3") != std::string::npos || lower.find("evt3") != std::string::npos) h.format = RawFormat::kEvt3;
    if (lower.find("evt 2") != std::string::npos || lower.find("evt2") != std::string::npos) h.format = RawFormat::kEvt2;
    if (int w = grab_int("width="); w > 0) h.width = w;
    if (int w = grab_int("width "); w > 0) h.width = w;
    if (int g = grab_int("height="); g > 0) h.height = g;
    if (int g = grab_int("height "); g > 0) h.height = g;
  }
  h.data_offset = pos;
  if (h.format == RawFormat::kUnknown) h.format = hint;
  if (h.format == RawFormat::kDat) h.data_offset = std::min(n, pos + 2);  // skip (event type, event size)
  return h;
}

// ------------------------------------------------------------------------------------------ EVT 2.0
size_t evt2_event_bound(const u32_unaligned* w, size_t n) {
  size_t k = 0;
  for (size_t i = 0; i < n; ++i) k += (w[i] >> 28) <= 1u;
  return k;
}

size_t Evt2Decoder::decode_into(const u32_unaligned* w, size_t n, EventSpan o) {
  size_t k = 0;
  uint64_t hi = time_high_ << 6;
  bool ready = have_high_;
  for (size_t i = 0; i < n; ++i) {
    const uint32_t v = w[i];
    const uint32_t type = v >> 28;
    if (type <= 1u) {
      if (!ready) continue;
      o.t[k] = static_cast<int64_t>(hi | ((v >> 22) & 0x3Fu));
      o.x[k] = static_cast<uint16_t>((v >> 11) & 0x7FFu);
      o.y[k] = static_cast<uint16_t>(v & 0x7FFu);
      o.p[k] = static_cast<uint8_t>(type);
      ++k;
    } else if (type == 0x8u) {
      time_high_ = v & 0x0FFFFFFFu;
      hi = time_high_ << 6;
      have_high_ = ready = true;
    }
  }
  return k;
}

void Evt2Decoder::decode(const u32_unaligned* w, size_t n, EventBuffer& out) {
  const size_t at = out.size();
  resize(out, at + evt2_event_bound(w, n));
  resize(out, at + decode_into(w, n, span_of(out, at)));
}

// ------------------------------------------------------------------------------------------ EVT 3.0
size_t evt3_event_bound(const u16_unaligned* w, size_t n) {
  size_t k = 0;
  for (size_t i = 0; i < n; ++i) {
    const uint16_t v = w[i];
    switch (v >> 12) {
      case 0x2: ++k; break;
      case 0x4: k += __builtin_popcount(v & 0xFFFu); break;
      case 0x5: k += __builtin_popcount(v & 0xFFu); break;
      default: break;
    }
  }
  return k;
}

// Hot loop: no per-event capacity checks, timestamp recomputed only when TIME_LOW / TIME_HIGH change.
size_t Evt3Decoder::decode_into(const u16_unaligned* w, size_t n, EventSpan o) {
  size_t k = 0;
  int64_t ts = now();
  bool ready = have_high_ && have_y_;
  for (size_t i = 0; i < n; ++i) {
    const uint16_t v = w[i];
    switch (v >> 12) {
      case 0x0:
        y_ = v & 0x7FFu;
        have_y_ = true;
        ready = have_high_;
        break;
      case 0x2:
        if (ready) {
          o.t[k] = ts; o.x[k] = v & 0x7FFu; o.y[k] = y_; o.p[k] = static_cast<uint8_t>((v >> 11) & 1u);
          ++k;
        }
        break;
      case 0x3: base_x_ = v & 0x7FFu; pol_ = static_cast<uint8_t>((v >> 11) & 1u); break;
      case 0x4:
      case 0x5: {
        const bool v12 = (v >> 12) == 0x4;
        uint32_t valid = v & (v12 ? 0xFFFu : 0xFFu);
        if (ready) {
          while (valid) {
            const int b = __builtin_ctz(valid);
            o.t[k] = ts; o.x[k] = static_cast<uint16_t>(base_x_ + b); o.y[k] = y_; o.p[k] = pol_;
            ++k;
            valid &= valid - 1;
          }
        }
        base_x_ = static_cast<uint16_t>(base_x_ + (v12 ? 12 : 8));
        break;
      }
      case 0x6:
        time_low_ = v & 0xFFFu;
        ts = now();
        break;
      case 0x8: {
        const uint32_t h = v & 0xFFFu;
        if (have_high_ && h < last_high_) ++epoch_;      // 24-bit time base wrapped (every ~16.78 s)
        time_high_ = h;
        last_high_ = h;
        have_high_ = true;
        ready = have_y_;
        ts = now();
        break;
      }
      default: break;                                     // EXT_TRIGGER, OTHERS, CONTINUED_*: ignored
    }
  }
  return k;
}

void Evt3Decoder::decode(const u16_unaligned* w, size_t n, EventBuffer& out) {
  const size_t at = out.size();
  resize(out, at + evt3_event_bound(w, n));
  resize(out, at + decode_into(w, n, span_of(out, at)));
}

// ------------------------------------------------------------------------------------------ DAT
void decode_dat(const uint8_t* payload, size_t n_bytes, EventBuffer& out) {
  const size_t n = n_bytes / 8, at = out.size();
  resize(out, at + n);
  for (size_t i = 0; i < n; ++i) {
    uint32_t ts, data;
    std::memcpy(&ts, payload + 8 * i, 4);
    std::memcpy(&data, payload + 8 * i + 4, 4);
    out.t[at + i] = static_cast<int64_t>(ts);
    out.x[at + i] = static_cast<uint16_t>(data & 0x3FFFu);
    out.y[at + i] = static_cast<uint16_t>((data >> 14) & 0x3FFFu);
    out.p[at + i] = static_cast<uint8_t>((data >> 28) & 0x1u);
  }
}

// ------------------------------------------------------------------------------------------ parallel
namespace {

struct MappedFile {
  const uint8_t* data = nullptr;
  size_t size = 0;
  int fd = -1;
  explicit MappedFile(const std::string& path) {
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("evkit: cannot open " + path);
    struct stat st;
    ::fstat(fd, &st);
    size = static_cast<size_t>(st.st_size);
    if (size) {
      void* m = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
      if (m == MAP_FAILED) throw std::runtime_error("evkit: mmap failed for " + path);
      ::madvise(m, size, MADV_SEQUENTIAL);
      data = static_cast<const uint8_t*>(m);
    }
  }
  ~MappedFile() {
    if (data) ::munmap(const_cast<uint8_t*>(data), size);
    if (fd >= 0) ::close(fd);
  }
};


// EVT3 per-chunk summary = how a chunk transforms the decoder state, independent of the entry state.
struct Evt3Summary {
  bool has_y = false, has_tl = false, has_base = false, has_th = false;
  uint16_t y = 0, base_after = 0; uint8_t pol = 0; uint16_t base_delta = 0;
  uint32_t tl = 0, first_th = 0, last_th = 0; int64_t wraps = 0;
  size_t bound = 0;                 // max #events the chunk can emit
};

Evt3Summary summarize_evt3(const u16_unaligned* w, size_t n) {
  Evt3Summary s;
  uint32_t prev_th = 0;
  for (size_t i = 0; i < n; ++i) {
    const uint16_t v = w[i];
    switch (v >> 12) {
      case 0x0: s.has_y = true; s.y = v & 0x7FFu; break;
      case 0x2: ++s.bound; break;
      case 0x3: s.has_base = true; s.base_after = v & 0x7FFu; s.pol = (v >> 11) & 1u; break;
      case 0x4: if (s.has_base) s.base_after += 12; else s.base_delta += 12; s.bound += __builtin_popcount(v & 0xFFFu); break;
      case 0x5: if (s.has_base) s.base_after += 8; else s.base_delta += 8; s.bound += __builtin_popcount(v & 0xFFu); break;
      case 0x6: s.has_tl = true; s.tl = v & 0xFFFu; break;
      case 0x8: {
        const uint32_t h = v & 0xFFFu;
        if (!s.has_th) { s.has_th = true; s.first_th = h; }
        else if (h < prev_th) ++s.wraps;
        prev_th = h; s.last_th = h;
        break;
      }
      default: break;
    }
  }
  return s;
}

struct Evt3State {
  bool have_y = false, have_th = false;
  uint16_t y = 0, base = 0; uint8_t pol = 0;
  uint32_t tl = 0, th = 0; int64_t epoch = 0;
};

Evt3State apply(const Evt3State& e, const Evt3Summary& s) {
  Evt3State o = e;
  if (s.has_y) { o.y = s.y; o.have_y = true; }
  if (s.has_tl) o.tl = s.tl;
  if (s.has_base) { o.base = s.base_after; o.pol = s.pol; } else { o.base = static_cast<uint16_t>(e.base + s.base_delta); }
  if (s.has_th) {
    o.epoch = e.epoch + ((e.have_th && s.first_th < e.th) ? 1 : 0) + s.wraps;
    o.th = s.last_th; o.have_th = true;
  }
  return o;
}

int resolve_threads(int n_threads, size_t work) {
  int T = n_threads > 0 ? n_threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  return std::max(1, std::min<int>(T, static_cast<int>(work / (1u << 16)) + 1));
}

}  // namespace

// Friend-free state injection: rebuild a decoder whose state equals `st` by replaying a tiny synthetic
// prefix.  (Keeps the public decoder API minimal and guarantees the same code path as sequential.)
static Evt3Decoder make_evt3_decoder(const Evt3State& st) {
  Evt3Decoder d;
  std::vector<uint16_t> pre;
  if (st.have_th) {
    d.set_epoch(st.epoch, st.th);
    pre.push_back(static_cast<uint16_t>(0x8000u | (st.th & 0xFFFu)));   // re-assert TIME_HIGH (no wrap: equal)
  }
  pre.push_back(static_cast<uint16_t>(0x6000u | (st.tl & 0xFFFu)));
  pre.push_back(static_cast<uint16_t>(0x3000u | (static_cast<uint16_t>(st.pol) << 11) | (st.base & 0x7FFu)));
  EventBuffer sink;
  d.decode(pre.data(), pre.size(), sink);
  if (st.have_y) {
    const uint16_t wy = static_cast<uint16_t>(st.y & 0x7FFu);
    d.decode(&wy, 1, sink);
  }
  return d;
}

// Decode chunk k into out[offset[k] ...]; afterwards close the (rare) gaps left by chunks that emitted fewer
// events than their bound (events before the stream's first TIME_HIGH / ADDR_Y).
static void compact(EventBuffer& out, const std::vector<size_t>& offset, const std::vector<size_t>& count) {
  size_t w = 0;
  for (size_t k = 0; k < count.size(); ++k) {
    if (w != offset[k] && count[k]) {
      std::memmove(out.t.data() + w, out.t.data() + offset[k], count[k] * sizeof(int64_t));
      std::memmove(out.x.data() + w, out.x.data() + offset[k], count[k] * sizeof(uint16_t));
      std::memmove(out.y.data() + w, out.y.data() + offset[k], count[k] * sizeof(uint16_t));
      std::memmove(out.p.data() + w, out.p.data() + offset[k], count[k]);
    }
    w += count[k];
  }
  resize(out, w);
}

template <class F>
static void run_parallel(int T, F&& f) {
  if (T == 1) { f(0); return; }
  std::vector<std::thread> th;
  for (int k = 0; k < T; ++k) th.emplace_back(f, k);
  for (auto& t : th) t.join();
}

RawHeader decode_file(const std::string& path, EventBuffer& out, int n_threads, RawFormat hint) {
  MappedFile f(path);
  if (hint == RawFormat::kUnknown) {
    auto dot = path.rfind('.');
    if (dot != std::string::npos && path.substr(dot) == ".dat") hint = RawFormat::kDat;
  }
  RawHeader h = parse_header(f.data, f.size, hint);
  const uint8_t* payload = f.data + h.data_offset;
  const size_t nbytes = f.size - h.data_offset;
  out.clear();

  if (h.format == RawFormat::kDat) {
    const size_t n = nbytes / 8;
    const int T = resolve_threads(n_threads, n);
    resize(out, n);
    run_parallel(T, [&](int k) {
      const size_t s = n * k / T, e = n * (k + 1) / T;
      for (size_t i = s; i < e; ++i) {
        uint32_t ts, data;
        std::memcpy(&ts, payload + 8 * i, 4);
        std::memcpy(&data, payload + 8 * i + 4, 4);
        out.t[i] = static_cast<int64_t>(ts);
        out.x[i] = static_cast<uint16_t>(data & 0x3FFFu);
        out.y[i] = static_cast<uint16_t>((data >> 14) & 0x3FFFu);
        out.p[i] = static_cast<uint8_t>((data >> 28) & 0x1u);
      }
    });
    return h;
  }
  if (h.format == RawFormat::kEvt2) {
    const size_t n = nbytes / 4;
    const u32_unaligned* w = reinterpret_cast<const u32_unaligned*>(payload);   // zero-copy from the mmap
    const int T = resolve_threads(n_threads, n);
    std::vector<size_t> cut(T + 1), bound(T), offset(T), count(T);
    for (int k = 0; k <= T; ++k) cut[k] = n * k / T;
    std::vector<int64_t> last_th(T, -1);
    run_parallel(T, [&](int k) {           // pass 1: event bound + last TIME_HIGH per chunk
      size_t b = 0;
      int64_t lt = -1;
      for (size_t i = cut[k]; i < cut[k + 1]; ++i) {
        const uint32_t ty = w[i] >> 28;
        b += ty <= 1u;
        if (ty == 0x8u) lt = w[i] & 0x0FFFFFFFu;
      }
      bound[k] = b;
      last_th[k] = lt;
    });
    std::vector<int64_t> entry_th(T);
    int64_t entry = -1;
    size_t total = 0;
    for (int k = 0; k < T; ++k) {
      entry_th[k] = entry;
      if (last_th[k] >= 0) entry = last_th[k];
      offset[k] = total;
      total += bound[k];
    }
    resize(out, total);
    run_parallel(T, [&](int k) {           // pass 2: decode in place
      Evt2Decoder d;
      if (entry_th[k] >= 0) {
        const uint32_t seed = 0x80000000u | static_cast<uint32_t>(entry_th[k]);
        d.decode_into(&seed, 1, span_of(out, 0));        // a TIME_HIGH word emits nothing
      }
      count[k] = d.decode_into(w + cut[k], cut[k + 1] - cut[k], span_of(out, offset[k]));
    });
    compact(out, offset, count);
    return h;
  }
  if (h.format == RawFormat::kEvt3) {
    const size_t n = nbytes / 2;
    const u16_unaligned* w = reinterpret_cast<const u16_unaligned*>(payload);   // zero-copy from the mmap
    const int T = resolve_threads(n_threads, n);
    std::vector<size_t> cut(T + 1), offset(T), count(T);
    for (int k = 0; k <= T; ++k) cut[k] = n * k / T;
    std::vector<Evt3Summary> sum(T);
    run_parallel(T, [&](int k) { sum[k] = summarize_evt3(w + cut[k], cut[k + 1] - cut[k]); });
    std::vector<Evt3State> entry(T);
    Evt3State s;
    size_t total = 0;
    for (int k = 0; k < T; ++k) {
      entry[k] = s;
      s = apply(s, sum[k]);
      offset[k] = total;
      total += sum[k].bound;
    }
    resize(out, total);
    run_parallel(T, [&](int k) {
      Evt3Decoder d = (k == 0) ? Evt3Decoder() : make_evt3_decoder(entry[k]);
      count[k] = d.decode_into(w + cut[k], cut[k + 1] - cut[k], span_of(out, offset[k]));
    });
    compact(out, offset, count);
    return h;
  }
  throw std::runtime_error("evkit: unknown raw format for " + path + " (pass a format hint)");
}

// ------------------------------------------------------------------------------------------ encoders
std::vector<uint32_t> encode_evt2(const EventBuffer& ev) {
  std::vector<uint32_t> w;
  int64_t cur_high = -1;
  for (size_t i = 0; i < ev.size(); ++i) {
    const int64_t hi = ev.t[i] >> 6;
    if (hi != cur_high) { w.push_back(0x80000000u | static_cast<uint32_t>(hi & 0x0FFFFFFF)); cur_high = hi; }
    w.push_back((static_cast<uint32_t>(ev.p[i] & 1u) << 28) | (static_cast<uint32_t>(ev.t[i] & 0x3F) << 22) |
                (static_cast<uint32_t>(ev.x[i] & 0x7FF) << 11) | static_cast<uint32_t>(ev.y[i] & 0x7FF));
  }
  return w;
}

std::vector<uint16_t> encode_evt3(const EventBuffer& ev, bool use_vectors) {
  std::vector<uint16_t> w;
  int64_t cur_high = -1, cur_low = -1;
  int cur_y = -1;
  size_t i = 0;
  const size_t n = ev.size();
  while (i < n) {
    const int64_t t = ev.t[i];
    const int64_t hi = t >> 12, lo = t & 0xFFF;
    if (hi != cur_high) { w.push_back(static_cast<uint16_t>(0x8000u | (hi & 0xFFF))); cur_high = hi; cur_low = -1; }
    if (lo != cur_low) { w.push_back(static_cast<uint16_t>(0x6000u | lo)); cur_low = lo; }
    if (ev.y[i] != cur_y) { w.push_back(static_cast<uint16_t>(ev.y[i] & 0x7FF)); cur_y = ev.y[i]; }
    // group a run of events with the same (t, y, p) and x in [x0, x0 + 12) -> VECT_BASE_X + VECT_12
    size_t j = i + 1;
    uint32_t mask = 1u;
    if (use_vectors) {
      while (j < n && ev.t[j] == t && ev.y[j] == ev.y[i] && ev.p[j] == ev.p[i] &&
             ev.x[j] > ev.x[j - 1] && ev.x[j] < ev.x[i] + 12) {
        mask |= 1u << (ev.x[j] - ev.x[i]);
        ++j;
      }
    }
    if (j - i >= 2) {
      w.push_back(static_cast<uint16_t>(0x3000u | (static_cast<uint16_t>(ev.p[i] & 1u) << 11) | (ev.x[i] & 0x7FF)));
      w.push_back(static_cast<uint16_t>(0x4000u | mask));
    } else {
      j = i + 1;
      w.push_back(static_cast<uint16_t>(0x2000u | (static_cast<uint16_t>(ev.p[i] & 1u) << 11) | (ev.x[i] & 0x7FF)));
    }
    i = j;
  }
  return w;
}

std::vector<uint8_t> encode_dat(const EventBuffer& ev) {
  const std::string hdr = "% Data file containing CD events.\n% Version 2\n";
  std::vector<uint8_t> b(hdr.begin(), hdr.end());
  b.push_back(0x0C);  // event type (CD)
  b.push_back(8);     // event size
  for (size_t i = 0; i < ev.size(); ++i) {
    const uint32_t ts = static_cast<uint32_t>(ev.t[i]);
    const uint32_t data = (ev.x[i] & 0x3FFFu) | ((ev.y[i] & 0x3FFFu) << 14) | ((ev.p[i] & 1u) << 28);
    const uint8_t* a = reinterpret_cast<const uint8_t*>(&ts);
    const uint8_t* d = reinterpret_cast<const uint8_t*>(&data);
    b.insert(b.end(), a, a + 4);
    b.insert(b.end(), d, d + 4);
  }
  return b;
}

}  // namespace evkit
