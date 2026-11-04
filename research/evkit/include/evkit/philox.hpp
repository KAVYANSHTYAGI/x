// Philox4x32-10 counter-based RNG (Salmon et al., SC'11), host+device.
// Same code on CPU and GPU => dequantisation noise is bit-identical across backends and independent
// of how the stream is chunked or which thread processes which event (counter = global event index).
#pragma once
#include "common.hpp"

namespace evkit {

struct Philox4x32 {
  static constexpr uint32_t M0 = 0xD2511F53u, M1 = 0xCD9E8D57u;
  static constexpr uint32_t W0 = 0x9E3779B9u, W1 = 0xBB67AE85u;

  EVKIT_HD static void mulhilo(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo) {
    const uint64_t p = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    hi = static_cast<uint32_t>(p >> 32);
    lo = static_cast<uint32_t>(p);
  }

  // ctr[4], key[2] -> out[4]
  EVKIT_HD static void generate(const uint32_t ctr_in[4], const uint32_t key_in[2], uint32_t out[4]) {
    uint32_t c0 = ctr_in[0], c1 = ctr_in[1], c2 = ctr_in[2], c3 = ctr_in[3];
    uint32_t k0 = key_in[0], k1 = key_in[1];
#pragma unroll
    for (int r = 0; r < 10; ++r) {
      uint32_t hi0, lo0, hi1, lo1;
      mulhilo(M0, c0, hi0, lo0);
      mulhilo(M1, c2, hi1, lo1);
      const uint32_t n0 = hi1 ^ c1 ^ k0;
      const uint32_t n1 = lo1;
      const uint32_t n2 = hi0 ^ c3 ^ k1;
      const uint32_t n3 = lo0;
      c0 = n0; c1 = n1; c2 = n2; c3 = n3;
      k0 += W0; k1 += W1;
    }
    out[0] = c0; out[1] = c1; out[2] = c2; out[3] = c3;
  }

  // Uniform in [0, 1) with 24-bit resolution, one value per (seed, counter).
  EVKIT_HD static float uniform(uint64_t seed, uint64_t counter, uint32_t stream = 0) {
    const uint32_t ctr[4] = {static_cast<uint32_t>(counter), static_cast<uint32_t>(counter >> 32), stream, 0u};
    const uint32_t key[2] = {static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
    uint32_t out[4];
    generate(ctr, key, out);
    return static_cast<float>(out[0] >> 8) * (1.0f / 16777216.0f);
  }
};

}  // namespace evkit
