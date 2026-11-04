// EVB: evkit's training-time event store.
//
//   [Header 128 B][t: int64 x n][x: uint16 x n][y: uint16 x n][p: uint8 x n][pad to 8][index: uint64 x n_index]
//
// * Struct-of-arrays, mmap-able, zero-copy into numpy/torch.
// * index[k] = first event with t >= t0 + k * index_dt_us, so a random time window is two loads + a
//   short binary search: O(1) seeks across multi-hour recordings (the access pattern of pretraining).
// * Events are guaranteed time-sorted on write.
#pragma once
#include <memory>
#include <string>

#include "events.hpp"

namespace evkit {

#pragma pack(push, 1)
struct EvbHeader {
  char magic[8];          // "EVKITEVB"
  uint32_t version;       // 1
  uint16_t width, height;
  uint64_t n_events;
  int64_t t0, t1;         // first / last timestamp (us)
  uint32_t index_dt_us;
  uint32_t flags;
  uint64_t n_index;
  uint64_t off_t, off_x, off_y, off_p, off_index;
  char sensor[32];        // free-form sensor tag, e.g. "IMX636", "ATIS-Gen1"
};
#pragma pack(pop)
static_assert(sizeof(EvbHeader) <= 128, "EVB header must fit in 128 bytes");

// Writes `ev` (sorted on the fly if needed).
void write_evb(const std::string& path, EventBuffer ev, int width, int height,
               const std::string& sensor = "", uint32_t index_dt_us = 1000);

class EvbReader {
 public:
  explicit EvbReader(const std::string& path);
  ~EvbReader();
  EvbReader(const EvbReader&) = delete;
  EvbReader& operator=(const EvbReader&) = delete;

  const EvbHeader& header() const { return *hdr_; }
  size_t size() const { return hdr_->n_events; }
  const int64_t* t() const { return t_; }
  const uint16_t* x() const { return x_; }
  const uint16_t* y() const { return y_; }
  const uint8_t* p() const { return p_; }

  // [begin, end) event range with t in [t_start, t_end)
  std::pair<size_t, size_t> time_range(int64_t t_start, int64_t t_end) const;
  size_t lower_bound(int64_t t) const;

 private:
  const uint8_t* base_ = nullptr;
  size_t bytes_ = 0;
  int fd_ = -1;
  const EvbHeader* hdr_ = nullptr;
  const int64_t* t_ = nullptr;
  const uint16_t *x_ = nullptr, *y_ = nullptr;
  const uint8_t* p_ = nullptr;
  const uint64_t* index_ = nullptr;
};

}  // namespace evkit
