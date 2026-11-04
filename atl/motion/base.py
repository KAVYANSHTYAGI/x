"""Motion model interface.

A motion model answers one question: *where is the scene point seen by event (x, t) at time t'?*

    transport(xy, t, t_dst, b) -> xy'           (all events; t_dst scalar or per-event)

Models whose transport is linear in time with a per-event velocity (the classic contrast-maximisation warp
x' = x + v(x) (t' - t), and pixel trajectories with a degree-1 prior) also expose ``velocity(xy, t, b)`` and set
``linear_at_event = True``; that enables the exact anchored pair kernel.

Times are normalised to [0, 1] within the window; velocities are in pixels per window.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class Motion(nn.Module):
    linear_at_event: bool = False

    def transport(self, xy: torch.Tensor, t: torch.Tensor, t_dst, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def velocity(self, xy: torch.Tensor, t: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} has no per-event velocity")

    # dense displacement of the pixel grid from t0 to t1 (prediction / evaluation / regularisers)
    def flow_field(self, B: int, H: int, W: int, t0: float = 0.0, t1: float = 1.0) -> torch.Tensor:
        dev = next(self.parameters()).device if any(True for _ in self.parameters()) else "cpu"
        ys, xs = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                                torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
        xy = torch.stack([xs, ys], -1).reshape(1, -1, 2).expand(B, -1, -1).reshape(-1, 2)
        b = torch.arange(B, device=dev).repeat_interleave(H * W)
        t = torch.full((xy.shape[0],), float(t0), device=dev)
        out = self.transport(xy, t, float(t1), b) - xy
        return out.view(B, H, W, 2).permute(0, 3, 1, 2)

    def transport_logdet(self, xy, t, t_dst, b, delta: float = 0.5):
        """Transport plus log|det d transport / d xy| by central finite differences (generic fallback)."""
        y = self.transport(xy, t, t_dst, b)
        ex = xy.new_tensor([delta, 0.0]); ey = xy.new_tensor([0.0, delta])
        jx = (self.transport(xy + ex, t, t_dst, b) - self.transport(xy - ex, t, t_dst, b)) / (2 * delta)
        jy = (self.transport(xy + ey, t, t_dst, b) - self.transport(xy - ey, t, t_dst, b)) / (2 * delta)
        det = jx[:, 0] * jy[:, 1] - jx[:, 1] * jy[:, 0]
        return y, torch.log(det.abs().clamp(min=1e-12))

    def regulariser_fields(self, B: int, H: int, W: int) -> Optional[torch.Tensor]:
        """Velocity / displacement field used by smoothness regularisers (None if not applicable)."""
        return None


def as_time(t_dst, like: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(t_dst):
        return t_dst.to(like.dtype).expand_as(like) if t_dst.dim() == 0 else t_dst.to(like.dtype)
    return torch.full_like(like, float(t_dst))
