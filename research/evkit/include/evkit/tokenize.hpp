// Event tokenisation: raw (t_us, x, y, p) -> model tokens, identical on CPU and GPU.
//
//   u = (x + 0.5) / W,  v = (y + 0.5) / H                       normalised coordinates (sensor-agnostic)
//   fine cell (cu, cv) = floor(u * G * Q), floor(v * G * Q)
//   patch = (cv / Q) * G + cu / Q,   sub = (cv % Q) * Q + cu % Q
//   dt    = ((t_i - t_{i-1}) + U_i) * 1e-6 s,   U_i = Philox(seed, global_index_i)  (dequantisation)
// The first event of a stream uses t_{-1} = t_prev (pass t_prev < 0 for "stream starts here": dt = U*1e-6).
#pragma once
#include "common.hpp"
#include "philox.hpp"

namespace evkit {

struct TokenizeParams {
  int width, height, grid, sub;
  uint64_t seed;
  uint64_t index_offset;   // global index of event 0 (keeps noise identical under chunking)
  int64_t t_prev;          // timestamp of the event before event 0, or < 0
};

EVKIT_HD void tokenize_one(const TokenizeParams& prm, uint64_t i, int64_t t_i, int64_t t_im1, uint16_t x,
                           uint16_t y, uint8_t p, int32_t& patch, int32_t& sub, int32_t& pol, float& u, float& v,
                           float& dt) {
  const int GQ = prm.grid * prm.sub;
  u = (static_cast<float>(x) + 0.5f) / static_cast<float>(prm.width);
  v = (static_cast<float>(y) + 0.5f) / static_cast<float>(prm.height);
  int cu = static_cast<int>(u * GQ), cv = static_cast<int>(v * GQ);
  cu = cu < 0 ? 0 : (cu >= GQ ? GQ - 1 : cu);
  cv = cv < 0 ? 0 : (cv >= GQ ? GQ - 1 : cv);
  patch = (cv / prm.sub) * prm.grid + cu / prm.sub;
  sub = (cv % prm.sub) * prm.sub + cu % prm.sub;
  pol = p ? 1 : 0;
  const float noise = Philox4x32::uniform(prm.seed, prm.index_offset + i);
  const int64_t gap = (t_im1 >= 0) ? (t_i - t_im1) : 0;
  dt = (static_cast<float>(gap) + noise) * 1e-6f;
}

// CPU, multi-threaded (n_threads = 0 -> hardware concurrency).
void tokenize_cpu(const int64_t* t_us, const uint16_t* x, const uint16_t* y, const uint8_t* p, size_t n,
                  const TokenizeParams& prm, int32_t* patch, int32_t* sub, int32_t* pol, float* u, float* v,
                  float* dt, int n_threads = 0);

}  // namespace evkit
