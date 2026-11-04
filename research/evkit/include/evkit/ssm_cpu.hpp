// CPU reference (and CPU backend) for the two hot ops.  Multi-threaded, exact expf, deterministic.
#pragma once
#include "common.hpp"

namespace evkit::cpu {

// Forward of the event-driven diagonal SSM scan.  h0 may be null.  H_save may be null if n_save == 0.
void scan_fwd(const ScanShape& s, const float* v, const float* dt, const float* rate, const float* Bm,
              const float* Cm, const float* skip, const float* h0, float* y, float* H_save, float* h_last,
              int n_threads = 0);

// Backward.  gH (B,L,n_save,N) and g_hlast (B,D,N) may be null (treated as zero); gh0 may be null.
// Every output gradient is overwritten (not accumulated).
void scan_bwd(const ScanShape& s, const float* v, const float* dt, const float* rate, const float* Bm,
              const float* Cm, const float* skip, const float* h0, const float* gy, const float* gH,
              const float* g_hlast, float* gv, float* gdt, float* grate, float* gB, float* gC, float* gskip,
              float* gh0, int n_threads = 0);

void readout_fwd(const ReadoutShape& s, const float* H, const float* tau, const float* rate, const float* Cm,
                 float* Y, int n_threads = 0);

void readout_bwd(const ReadoutShape& s, const float* H, const float* tau, const float* rate, const float* Cm,
                 const float* gY, float* gH, float* gtau, float* grate, float* gC, int n_threads = 0);

}  // namespace evkit::cpu
