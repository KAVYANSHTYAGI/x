"""Optical-flow metrics as used by the DSEC benchmark / MotionPriorCMax.

EPE: mean end-point error (px over the window); AE: mean angular error (deg) of (u, v, 1) vectors;
NPE: % of pixels with EPE > N px (DSEC reports 1/2/3PE; "%Out" in MotionPriorCMax = 3PE).
Trajectory metrics (EVIMO2, MotionPriorCMax Tab. 2): TEPE / TAE / %Out averaged over the evaluation timestamps.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch


def flow_errors(pred: torch.Tensor, gt: torch.Tensor, valid: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """pred, gt: (B, 2, H, W); valid: (B, H, W) bool (GT validity, optionally AND event mask)."""
    if valid is None:
        valid = torch.isfinite(gt).all(1) & (gt.abs().sum(1) > 0)
    d = (pred - gt).norm(dim=1)
    v = valid
    n = v.sum().clamp(min=1)
    u1, v1 = pred[:, 0], pred[:, 1]
    u2, v2 = gt[:, 0], gt[:, 1]
    cos = (u1 * u2 + v1 * v2 + 1) / (torch.sqrt(u1 ** 2 + v1 ** 2 + 1) * torch.sqrt(u2 ** 2 + v2 ** 2 + 1))
    ae = torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
    out = {"EPE": float(d[v].sum() / n), "AE": float(ae[v].sum() / n), "n": int(n)}
    for k in (1, 2, 3):
        out[f"{k}PE"] = float(100.0 * (d[v] > k).float().sum() / n)
    return out


def trajectory_errors(pred_tracks: torch.Tensor, gt_tracks: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    """pred/gt: (B, T, 2, H, W) displacements at T evaluation times; valid (B, H, W)."""
    res = [flow_errors(pred_tracks[:, k], gt_tracks[:, k], valid) for k in range(pred_tracks.shape[1])]
    return {"TEPE": sum(r["EPE"] for r in res) / len(res), "TAE": sum(r["AE"] for r in res) / len(res),
            "%Out": sum(r["3PE"] for r in res) / len(res) / 100.0}


class Accumulator:
    """Pixel-weighted accumulation over a dataset (benchmark-style global averages)."""

    def __init__(self):
        self.s = {}
        self.n = 0

    def add(self, errs: Dict[str, float]):
        n = errs["n"]
        for k, v in errs.items():
            if k != "n":
                self.s[k] = self.s.get(k, 0.0) + v * n
        self.n += n

    def result(self) -> Dict[str, float]:
        return {k: v / max(self.n, 1) for k, v in self.s.items()}
