// Known-answer tests for Philox4x32-10 (Random123 kat_vectors) + basic distribution sanity.
#include <cstdio>

#include "check.hpp"
#include "evkit/philox.hpp"

int main() {
  using evkit::Philox4x32;
  struct Kat { uint32_t ctr[4], key[2], out[4]; };
  const Kat kats[] = {
      {{0, 0, 0, 0}, {0, 0}, {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}},
      {{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu}, {0xffffffffu, 0xffffffffu},
       {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}},
      {{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u}, {0xa4093822u, 0x299f31d0u},
       {0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}},
  };
  for (const auto& k : kats) {
    uint32_t out[4];
    Philox4x32::generate(k.ctr, k.key, out);
    for (int i = 0; i < 4; ++i) {
      if (out[i] != k.out[i]) {
        std::fprintf(stderr, "philox KAT mismatch word %d: got %08x want %08x\n", i, out[i], k.out[i]);
        return 1;
      }
    }
  }
  double s = 0, s2 = 0;
  const int n = 1 << 20;
  for (int i = 0; i < n; ++i) {
    const double u = Philox4x32::uniform(42, i);
    CHECK(u >= 0.0 && u < 1.0);
    s += u; s2 += u * u;
  }
  CHECK_CLOSE(s / n, 0.5, 0, 2e-3);
  CHECK_CLOSE(s2 / n - (s / n) * (s / n), 1.0 / 12.0, 0, 2e-3);
  std::printf("test_philox OK\n");
  return 0;
}
