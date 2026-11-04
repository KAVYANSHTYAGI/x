// Shared declarations for the atl low-level ops (CPU: atl_cpu.cpp, CUDA: atl_cuda.cu).
//
// Layout conventions (all tensors contiguous):
//   pos  (N,2) float32  [x, y] pixel coordinates, pixel centres at integers
//   w    (N)   float32  per-event weight
//   bidx (N)   int64    batch index in [0, B)
//   img  (B,H,W) float32
//
// Kernels
//   splat_bilinear   : img[b, y, x] += w * bilinear(pos)               (fwd, bwd -> d pos, d w)
//   splat_gauss      : img[b, y, x] += w * N(x,y; pos, s^2), |d|<=R     (fwd, bwd -> d pos, d w)
//   gather_bilinear  : val = bilinear sample of img at pos (zero pad)  (fwd, bwd -> d img, d pos)
//   ats_pair         : anchored leave-one-out Gaussian sums with per-event velocities (fwd, bwd -> d v)
//   voxel_grid       : trilinear (x, y, t) polarity voting (no grad)
#pragma once
#include <torch/extension.h>

namespace atl {

torch::Tensor splat_bilinear_fwd(torch::Tensor pos, torch::Tensor w, torch::Tensor bidx, int64_t B, int64_t H, int64_t W);
std::vector<torch::Tensor> splat_bilinear_bwd(torch::Tensor gimg, torch::Tensor pos, torch::Tensor w, torch::Tensor bidx);

torch::Tensor splat_gauss_fwd(torch::Tensor pos, torch::Tensor w, torch::Tensor bidx, int64_t B, int64_t H, int64_t W,
                              double sigma, int64_t radius);
std::vector<torch::Tensor> splat_gauss_bwd(torch::Tensor gimg, torch::Tensor pos, torch::Tensor w, torch::Tensor bidx,
                                           double sigma, int64_t radius);

torch::Tensor gather_bilinear_fwd(torch::Tensor img, torch::Tensor pos, torch::Tensor bidx);
std::vector<torch::Tensor> gather_bilinear_bwd(torch::Tensor gval, torch::Tensor img, torch::Tensor pos, torch::Tensor bidx);

// Anchored pair sums.  Sources are pre-sorted by (batch, slice, cell) on the Python side:
//   src_y   (Ns,2) source position transported to the slice centre  y_j = x_j + v_j (c_s - t_j)
//   src_v   (Ns,2) source velocity, src_t (Ns) source time, src_id (Ns) original event index (for LOO + grads)
//   cell_start (B*S*GH*GW + 1) CSR offsets into the sorted source arrays
//   anchors: anc_x (M,2) raw position, anc_t (M), anc_cell (M,3) = (grid index g = b*S + s, cx, cy), anc_id (M)
//   slice_c (S) slice centre times.
// Returns S_i = sum_{j != i, |t_i - t_j| <= tau, |r_ij| <= cut*h} exp(-|r_ij|^2 / 2h^2),
//   r_ij = x_i - (y_j + v_j (t_i - c_s)) = x_i - x_j - v_j (t_i - t_j).
torch::Tensor ats_pair_fwd(torch::Tensor src_y, torch::Tensor src_v, torch::Tensor src_t, torch::Tensor src_id,
                           torch::Tensor cell_start, torch::Tensor anc_x, torch::Tensor anc_t, torch::Tensor anc_cell,
                           torch::Tensor anc_id, torch::Tensor slice_c, int64_t S, int64_t GH, int64_t GW,
                           double h, double tau, double cut);
// Gradient of sum_i gS_i * S_i w.r.t. per-event velocities v (N,2), N = number of events.
torch::Tensor ats_pair_bwd(torch::Tensor gS, torch::Tensor src_y, torch::Tensor src_v, torch::Tensor src_t,
                           torch::Tensor src_id, torch::Tensor cell_start, torch::Tensor anc_x, torch::Tensor anc_t,
                           torch::Tensor anc_cell, torch::Tensor anc_id, torch::Tensor slice_c, int64_t S, int64_t GH,
                           int64_t GW, double h, double tau, double cut, int64_t N);

torch::Tensor voxel_grid(torch::Tensor pos, torch::Tensor t, torch::Tensor p, torch::Tensor bidx, int64_t B,
                         int64_t C, int64_t H, int64_t W);

#ifdef ATL_WITH_CUDA
namespace cuda {
torch::Tensor splat_bilinear_fwd(torch::Tensor pos, torch::Tensor w, torch::Tensor bidx, int64_t B, int64_t H, int64_t W);
std::vector<torch::Tensor> splat_bilinear_bwd(torch::Tensor gimg, torch::Tensor pos, torch::Tensor w, torch::Tensor bidx);
torch::Tensor splat_gauss_fwd(torch::Tensor pos, torch::Tensor w, torch::Tensor bidx, int64_t B, int64_t H, int64_t W,
                              double sigma, int64_t radius);
std::vector<torch::Tensor> splat_gauss_bwd(torch::Tensor gimg, torch::Tensor pos, torch::Tensor w, torch::Tensor bidx,
                                           double sigma, int64_t radius);
torch::Tensor gather_bilinear_fwd(torch::Tensor img, torch::Tensor pos, torch::Tensor bidx);
std::vector<torch::Tensor> gather_bilinear_bwd(torch::Tensor gval, torch::Tensor img, torch::Tensor pos, torch::Tensor bidx);
torch::Tensor ats_pair_fwd(torch::Tensor src_y, torch::Tensor src_v, torch::Tensor src_t, torch::Tensor src_id,
                           torch::Tensor cell_start, torch::Tensor anc_x, torch::Tensor anc_t, torch::Tensor anc_cell,
                           torch::Tensor anc_id, torch::Tensor slice_c, int64_t S, int64_t GH, int64_t GW,
                           double h, double tau, double cut);
torch::Tensor ats_pair_bwd(torch::Tensor gS, torch::Tensor src_y, torch::Tensor src_v, torch::Tensor src_t,
                           torch::Tensor src_id, torch::Tensor cell_start, torch::Tensor anc_x, torch::Tensor anc_t,
                           torch::Tensor anc_cell, torch::Tensor anc_id, torch::Tensor slice_c, int64_t S, int64_t GH,
                           int64_t GW, double h, double tau, double cut, int64_t N);
torch::Tensor voxel_grid(torch::Tensor pos, torch::Tensor t, torch::Tensor p, torch::Tensor bidx, int64_t B,
                         int64_t C, int64_t H, int64_t W);
}  // namespace cuda
#endif

}  // namespace atl
