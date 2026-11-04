// Host API of the CUDA kernels (raw device pointers; no torch dependency).
// Every function launches on `stream` (a cudaStream_t passed as void*) and throws std::runtime_error on
// a launch error.  All tensors are contiguous float32 unless stated otherwise.
#pragma once
#include "common.hpp"
#include "tokenize.hpp"

namespace evkit::gpu {

constexpr int kChunk = 32;          // events per time tile / checkpoint interval
constexpr int kMaxZ = 8;            // max chunk-parallel blocks along time in the chunked scan

inline int64_t num_chunks(int64_t L) { return (L + kChunk - 1) / kChunk; }

enum class ScanMode { kAuto = 0, kSequential = 1, kChunked = 2 };

// ckpt (B, C, D, N): state before each chunk; written by scan_fwd, read by scan_bwd (C = num_chunks(L)).
size_t scan_fwd_workspace_floats(const ScanShape& s);
size_t scan_bwd_workspace_floats(const ScanShape& s);

void scan_fwd(const ScanShape& s, const float* v, const float* dt, const float* rate, const float* Bm,
              const float* Cm, const float* skip, const float* h0, float* y, float* H_save, float* h_last,
              float* ckpt, float* workspace, bool fast_exp, ScanMode mode, void* stream);

void scan_bwd(const ScanShape& s, const float* v, const float* dt, const float* rate, const float* Bm,
              const float* Cm, const float* skip, const float* h0, const float* ckpt, const float* gy,
              const float* gH, const float* g_hlast, float* gv, float* gdt, float* grate, float* gB, float* gC,
              float* gskip, float* gh0, float* workspace, bool fast_exp, ScanMode mode, void* stream);

size_t readout_bwd_workspace_floats(const ReadoutShape& s);
void readout_fwd(const ReadoutShape& s, const float* H, const float* tau, const float* rate, const float* Cm,
                 float* Y, bool fast_exp, void* stream);
void readout_bwd(const ReadoutShape& s, const float* H, const float* tau, const float* rate, const float* Cm,
                 const float* gY, float* gH, float* gtau, float* grate, float* gC, float* workspace, bool fast_exp,
                 void* stream);

void tokenize(const int64_t* t_us, const uint16_t* x, const uint16_t* y, const uint8_t* p, int64_t n,
              const TokenizeParams& prm, int32_t* patch, int32_t* sub, int32_t* pol, float* u, float* v, float* dt,
              void* stream);

}  // namespace evkit::gpu
