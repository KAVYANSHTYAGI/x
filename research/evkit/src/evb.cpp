#include "evkit/evb.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace evkit {

namespace {
size_t align8(size_t v) { return (v + 7) & ~size_t(7); }
}

void write_evb(const std::string& path, EventBuffer ev, int width, int height, const std::string& sensor,
               uint32_t index_dt_us) {
  ev.sort_by_time();
  const uint64_t n = ev.size();
  EvbHeader h{};
  std::memcpy(h.magic, "EVKITEVB", 8);
  h.version = 1;
  h.width = static_cast<uint16_t>(width);
  h.height = static_cast<uint16_t>(height);
  h.n_events = n;
  h.t0 = n ? ev.t.front() : 0;
  h.t1 = n ? ev.t.back() : 0;
  h.index_dt_us = index_dt_us;
  std::strncpy(h.sensor, sensor.c_str(), sizeof(h.sensor) - 1);
  h.n_index = n ? static_cast<uint64_t>((h.t1 - h.t0) / index_dt_us) + 2 : 0;
  h.off_t = 128;
  h.off_x = h.off_t + 8 * n;
  h.off_y = h.off_x + 2 * n;
  h.off_p = h.off_y + 2 * n;
  h.off_index = align8(h.off_p + n);

  std::vector<uint64_t> index(h.n_index);
  size_t j = 0;
  for (uint64_t k = 0; k < h.n_index; ++k) {
    const int64_t tk = h.t0 + static_cast<int64_t>(k) * index_dt_us;
    while (j < n && ev.t[j] < tk) ++j;
    index[k] = j;
  }

  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("evkit: cannot write " + path);
  std::vector<uint8_t> hdr(128, 0);
  std::memcpy(hdr.data(), &h, sizeof(h));
  std::fwrite(hdr.data(), 1, 128, f);
  std::fwrite(ev.t.data(), 8, n, f);
  std::fwrite(ev.x.data(), 2, n, f);
  std::fwrite(ev.y.data(), 2, n, f);
  std::fwrite(ev.p.data(), 1, n, f);
  const size_t pad = h.off_index - (h.off_p + n);
  const uint8_t zeros[8] = {0};
  std::fwrite(zeros, 1, pad, f);
  std::fwrite(index.data(), 8, index.size(), f);
  std::fclose(f);
}

EvbReader::EvbReader(const std::string& path) {
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) throw std::runtime_error("evkit: cannot open " + path);
  struct stat st;
  ::fstat(fd_, &st);
  bytes_ = static_cast<size_t>(st.st_size);
  void* m = ::mmap(nullptr, bytes_, PROT_READ, MAP_SHARED, fd_, 0);
  if (m == MAP_FAILED) throw std::runtime_error("evkit: mmap failed for " + path);
  base_ = static_cast<const uint8_t*>(m);
  hdr_ = reinterpret_cast<const EvbHeader*>(base_);
  if (std::memcmp(hdr_->magic, "EVKITEVB", 8) != 0 || hdr_->version != 1)
    throw std::runtime_error("evkit: not an EVB v1 file: " + path);
  t_ = reinterpret_cast<const int64_t*>(base_ + hdr_->off_t);
  x_ = reinterpret_cast<const uint16_t*>(base_ + hdr_->off_x);
  y_ = reinterpret_cast<const uint16_t*>(base_ + hdr_->off_y);
  p_ = base_ + hdr_->off_p;
  index_ = reinterpret_cast<const uint64_t*>(base_ + hdr_->off_index);
}

EvbReader::~EvbReader() {
  if (base_) ::munmap(const_cast<uint8_t*>(base_), bytes_);
  if (fd_ >= 0) ::close(fd_);
}

size_t EvbReader::lower_bound(int64_t t) const {
  const uint64_t n = hdr_->n_events;
  if (n == 0 || t <= hdr_->t0) return 0;
  if (t > hdr_->t1) return n;
  const uint64_t k = static_cast<uint64_t>((t - hdr_->t0) / hdr_->index_dt_us);
  const size_t lo = index_[k];
  const size_t hi = (k + 1 < hdr_->n_index) ? index_[k + 1] : n;
  return static_cast<size_t>(std::lower_bound(t_ + lo, t_ + hi, t) - t_);
}

std::pair<size_t, size_t> EvbReader::time_range(int64_t t_start, int64_t t_end) const {
  return {lower_bound(t_start), lower_bound(t_end)};
}

}  // namespace evkit
