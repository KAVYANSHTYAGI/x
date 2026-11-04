"""Dense continuous-time pixel trajectories (MotionPriorCMax, Hamann et al. ECCV'24).

A network predicts a coefficient grid C (B, 2K, H, W).  One trajectory per ``tile`` x ``tile`` patch starts at the
patch centre p (MotionPriorCMax: offset tile//2) at t = 0:

    q(p, t) = p + sum_k c_k(p) (g_k(t) - g_k(0))

with basis g = polynomial (t^k, k = 1..K) or Bezier (Bernstein polynomials of degree K with the first control point
pinned to the start).  Coordinates here are (x, y); MotionPriorCMax stores (y, x) -- the faithful loss handles that.

Transport of an event (x, t) to t' needs the trajectory through (x, t):
* ``assign="invert"`` : solve p + d(p, t) = x by fixed-point iteration on the bilinearly interpolated coefficient
                         field, then x' = p + d(p, t').  Exact for smooth fields, O(N) per target time.
* the KNN look-up table of MotionPriorCMax is implemented inside its loss (atl.objectives.mpcm).
"""
from __future__ import annotations

import math

import torch

from .base import Motion, as_time
from .parametric import sample_grid


def basis(t: torch.Tensor, K: int, kind: str = "polynomial") -> torch.Tensor:
    """(..., K) basis values g_k(t) - g_k(0)."""
    t = t[..., None]
    if kind == "polynomial":
        k = torch.arange(1, K + 1, device=t.device, dtype=t.dtype)
        return t ** k
    if kind == "bezier":
        # degree K Bernstein basis, control point 0 fixed at the start => use B_{k,K}(t), k = 1..K
        k = torch.arange(1, K + 1, device=t.device, dtype=t.dtype)
        logc = (math.lgamma(K + 1) - torch.lgamma(k + 1) - torch.lgamma(K - k + 1))
        return torch.exp(logc) * t ** k * (1 - t) ** (K - k)
    raise ValueError(kind)


class PixelTrajectories(Motion):
    """One continuous-time trajectory per tile, coefficients predicted by a network (see module docstring)."""
    def __init__(self, coeff_grid: torch.Tensor, K: int, basis_kind: str = "polynomial", tile: int = 4,
                 assign: str = "invert", n_fixed_point: int = 6):
        super().__init__()
        self.coeff_grid = coeff_grid                       # (B, 2K, H, W): channels [y_1..y_K, x_1..x_K] (MPCM order)
        self.K, self.basis_kind, self.tile, self.assign, self.n_fp = K, basis_kind, tile, assign, n_fixed_point
        B, C, H, W = coeff_grid.shape
        assert C == 2 * K
        self.B, self.H, self.W = B, H, W
        s = tile // 2
        self.tile_c = coeff_grid[:, :, s::tile, s::tile]  # (B, 2K, Hq, Wq) coefficients at tile centres
        self.Hq, self.Wq = self.tile_c.shape[-2:]
        self.offset = float(s)
        self.linear_at_event = (K == 1 and basis_kind == "polynomial")

    # coefficients at continuous positions (bilinear between tile centres)
    def _coeff_at(self, p: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        g = (p - self.offset) / self.tile
        c = sample_grid(self.tile_c, g, b)                # (N, 2K)
        cy, cx = c[:, :self.K], c[:, self.K:]
        return torch.stack([cx, cy], -1)                  # (N, K, 2) in (x, y)

    def displacement(self, p, b, t):
        """Displacement q(p, t) - p of the trajectories starting at ``p``, evaluated at times ``t``."""
        g = basis(t, self.K, self.basis_kind)             # (N, K)
        return (self._coeff_at(p, b) * g[..., None]).sum(1)

    def origin(self, xy, t, b):
        """Trajectory start p with q(p, t) = xy (fixed point p = xy - d(p, t))."""
        p = xy
        for _ in range(self.n_fp):
            p = xy - self.displacement(p, b, t)
        return p

    def transport(self, xy, t, t_dst, b):
        p = self.origin(xy, t, b)
        return p + self.displacement(p, b, as_time(t_dst, t))

    def velocity(self, xy, t, b):
        if not self.linear_at_event:
            raise NotImplementedError("per-event velocity only for the degree-1 polynomial prior")
        p = self.origin(xy, t, b)
        return self._coeff_at(p, b)[:, 0]

    # tile trajectories (B, n_t, Hq*Wq, 2) in (x, y), as used by MotionPriorCMax's look-up table
    def tile_trajectories(self, times: torch.Tensor) -> torch.Tensor:
        """Positions (B, n_t, n_tiles, 2) of every tile trajectory at the given times."""
        B, Hq, Wq, K = self.B, self.Hq, self.Wq, self.K
        dev = self.tile_c.device
        ys = torch.arange(Hq, device=dev, dtype=torch.float32) * self.tile + self.offset
        xs = torch.arange(Wq, device=dev, dtype=torch.float32) * self.tile + self.offset
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        p = torch.stack([gx, gy], -1).reshape(1, 1, -1, 2)
        c = self.tile_c.reshape(B, 2, K, Hq * Wq)                          # [y-coeffs; x-coeffs]
        cxy = torch.stack([c[:, 1], c[:, 0]], -1)                          # (B, K, n, 2)
        g = basis(times, K, self.basis_kind)                               # (n_t, K)
        d = torch.einsum("tk,bknc->btnc", g, cxy)
        return p + d

    def flow_field(self, B, H, W, t0=0.0, t1=1.0):
        """Dense flow from t0 to t1, tiles replicated then bilinearly upsampled (MotionPriorCMax evaluation)."""
        times = torch.tensor([t0, t1], device=self.tile_c.device, dtype=torch.float32)
        tr = self.tile_trajectories(times)                                 # (B, 2, n, 2)
        f = (tr[:, 1] - tr[:, 0]).reshape(self.B, self.Hq, self.Wq, 2).permute(0, 3, 1, 2)
        return torch.nn.functional.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)

    def regulariser_fields(self, B, H, W):
        """Tile coefficient grid (B, 2K, Hq, Wq) for smoothness regularisers."""
        return self.tile_c
