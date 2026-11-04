// Event containers and raw-sensor decoders (Prophesee EVT 2.0 / EVT 3.0 .raw, Gen1/1Mpx .dat).
//
// Bit layouts follow docs.prophesee.ai/stable/data/encoding_formats/{evt2,evt3}.html:
//   EVT2 (32-bit words, type = bits 31..28): CD_OFF 0x0 / CD_ON 0x1: ts_lo6 = 27..22, x = 21..11, y = 10..0
//        EVT_TIME_HIGH 0x8: ts bits 33..6 in 27..0.   t = (time_high << 6) | ts_lo6
//   EVT3 (16-bit words, type = bits 15..12): ADDR_Y 0x0 (y = 10..0), ADDR_X 0x2 (pol = 11, x = 10..0),
//        VECT_BASE_X 0x3 (pol = 11, x = 10..0), VECT_12 0x4 (valid 11..0, base += 12),
//        VECT_8 0x5 (valid 7..0, base += 8), TIME_LOW 0x6 (t 11..0), TIME_HIGH 0x8 (t 23..12),
//        EXT_TRIGGER 0xA, OTHERS 0xE, CONTINUED_4 0x7, CONTINUED_12 0xF.  24-bit time base wraps every 2^24 us.
//   DAT  ('%' header lines, then 2 bytes: event type, event size; events = u32 ts, u32 data with
//        x = data & 0x3FFF, y = (data >> 14) & 0x3FFF, p = (data >> 28) & 0x1).
#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common.hpp"

namespace evkit {

// Allocator whose value-initialisation is a no-op: resize() does not zero-fill, so a multi-GB output is
// first touched (page-faulted) in parallel by the threads that write it instead of by one memset.
template <class T>
struct UninitAllocator : std::allocator<T> {
  template <class U> struct rebind { using other = UninitAllocator<U>; };
  UninitAllocator() = default;
  template <class U> UninitAllocator(const UninitAllocator<U>&) noexcept {}
  template <class U> void construct(U* p) noexcept { ::new (static_cast<void*>(p)) U; }
  template <class U, class... A> void construct(U* p, A&&... a) { ::new (static_cast<void*>(p)) U(std::forward<A>(a)...); }
};
template <class T> using uvector = std::vector<T, UninitAllocator<T>>;

// Unaligned-safe word types: raw payloads start right after a text header, at any byte offset.
typedef uint16_t __attribute__((aligned(1), may_alias)) u16_unaligned;
typedef uint32_t __attribute__((aligned(1), may_alias)) u32_unaligned;

struct EventBuffer {
  uvector<int64_t> t;   // microseconds
  uvector<uint16_t> x, y;
  uvector<uint8_t> p;

  size_t size() const { return t.size(); }
  void reserve(size_t n) { t.reserve(n); x.reserve(n); y.reserve(n); p.reserve(n); }
  void clear() { t.clear(); x.clear(); y.clear(); p.clear(); }
  void push(uint16_t xx, uint16_t yy, uint8_t pp, int64_t tt) {
    x.push_back(xx); y.push_back(yy); p.push_back(pp); t.push_back(tt);
  }
  void append(const EventBuffer& o);
  bool is_time_sorted() const;
  // Stable sort by timestamp (only touches data if it is not already sorted). Returns true if it sorted.
  bool sort_by_time(int n_threads = 0);
};

// Raw output pointers for the allocation-free decode path (caller guarantees capacity >= event bound).
struct EventSpan {
  int64_t* t;
  uint16_t* x;
  uint16_t* y;
  uint8_t* p;
  EventSpan offset(size_t k) const { return {t + k, x + k, y + k, p + k}; }
};

inline EventSpan span_of(EventBuffer& b, size_t at = 0) { return {b.t.data() + at, b.x.data() + at, b.y.data() + at, b.p.data() + at}; }
inline void resize(EventBuffer& b, size_t n) { b.t.resize(n); b.x.resize(n); b.y.resize(n); b.p.resize(n); }

// Upper bounds on the number of CD events a word range can emit (exact unless events precede the first
// TIME_HIGH / ADDR_Y).  Used to allocate outputs once.
size_t evt2_event_bound(const u32_unaligned* w, size_t n);
size_t evt3_event_bound(const u16_unaligned* w, size_t n);

enum class RawFormat { kUnknown, kEvt2, kEvt3, kDat };

struct RawHeader {
  RawFormat format = RawFormat::kUnknown;
  int width = -1, height = -1;
  size_t data_offset = 0;     // first byte of the event payload
  std::string text;           // raw header text (all '%' lines)
};

// Parses the '%'-prefixed text header of .raw / .dat files.  `hint` resolves files whose header does
// not name the format (DAT files carry no 'format' key; their payload starts with 2 bytes type/size).
RawHeader parse_header(const uint8_t* data, size_t n, RawFormat hint = RawFormat::kUnknown);

// Streaming decoders keep their state across calls so a file can be fed in arbitrary chunks.
class Evt2Decoder {
 public:
  void decode(const u32_unaligned* w, size_t n, EventBuffer& out);   // appends
  size_t decode_into(const u32_unaligned* w, size_t n, EventSpan out);   // returns #events written
  void reset() { have_high_ = false; time_high_ = 0; }
 private:
  bool have_high_ = false;
  uint64_t time_high_ = 0;
};

class Evt3Decoder {
 public:
  void decode(const u16_unaligned* w, size_t n, EventBuffer& out);   // appends
  size_t decode_into(const u16_unaligned* w, size_t n, EventSpan out);   // returns #events written
  void reset() { *this = Evt3Decoder(); }
  // used by the parallel decoder to seed the wrap counter of a chunk
  void set_epoch(int64_t epoch, uint32_t last_high) { epoch_ = epoch; last_high_ = last_high; have_high_ = true; }
  int64_t epoch() const { return epoch_; }
 private:
  int64_t epoch_ = 0;          // number of 2^24 us wrap-arounds seen
  uint32_t last_high_ = 0;     // previous TIME_HIGH value (12 bits)
  uint32_t time_high_ = 0, time_low_ = 0;
  bool have_high_ = false, have_y_ = false;
  uint16_t y_ = 0, base_x_ = 0;
  uint8_t pol_ = 0;
  int64_t now() const { return (epoch_ << 24) | (static_cast<int64_t>(time_high_) << 12) | time_low_; }
};

void decode_dat(const uint8_t* payload, size_t n_bytes, EventBuffer& out);

// Decode a whole file (mmap).  Multi-threaded for EVT2/EVT3 (chunk-parallel with state warm-up and a
// sequential TIME_HIGH prefix pass for EVT3 wrap-around).  n_threads = 0 -> hardware concurrency.
RawHeader decode_file(const std::string& path, EventBuffer& out, int n_threads = 0,
                      RawFormat hint = RawFormat::kUnknown);

// Encoders (used by tests and for writing synthetic data in the native formats).
std::vector<uint32_t> encode_evt2(const EventBuffer& ev);
std::vector<uint16_t> encode_evt3(const EventBuffer& ev, bool use_vectors = true);
std::vector<uint8_t> encode_dat(const EventBuffer& ev);

}  // namespace evkit
