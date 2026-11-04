"""Objective interface + shared image-of-warped-events (IWE) machinery.

Every objective is a callable ``obj(events, motion) -> dict`` with key ``loss`` (to MINIMISE) plus diagnostics.
Reference-frame objectives (the CMax family) build an IWE at t_ref from transported events; anchored objectives
(ours) evaluate densities at raw events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from ..core.events import EventPacket
from ..ops import blur, splat


class Objective:
    name = "objective"
    reference_frame = True        # evaluates at warped positions (improper on contraction classes; see theory)

    def __call__(self, ev: EventPacket, motion) -> dict:
        raise NotImplementedError

    def __repr__(self):
        return f"{type(self).__name__}({self.name})"


@dataclass
class IWEConfig:
    kernel: str = "bilinear"            # "bilinear" vote (+ optional Gaussian blur) or "gauss" splat
    sigma: float = 1.0                  # blur / kernel width (px)
    polarity: str = "pooled"            # "pooled" | "split" (one IWE per polarity, MotionPriorCMax)
    mask_border: bool = True            # drop events warped outside the image
    scale_by_dt: bool = False           # weight (1 - |t - t_ref|) (MotionPriorCMax)


def make_iwe(ev: EventPacket, xy_ref: torch.Tensor, t_ref, cfg: IWEConfig,
             extra_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """IWE of events transported to ``t_ref``.  Returns (B, H, W) or (2B, H, W) for split polarity."""
    w = torch.ones_like(ev.t)
    if extra_weight is not None:
        w = w * extra_weight
    if cfg.scale_by_dt:
        tr = t_ref if torch.is_tensor(t_ref) else torch.tensor(float(t_ref), device=ev.device)
        w = w * (1 - (ev.t - tr).abs().clamp(0, 1)).detach()
    if cfg.mask_border:
        w = w * ev.inside(xy_ref).to(w.dtype).detach()
    B = ev.batch_size
    b = ev.b
    nb = B
    if cfg.polarity == "split":
        b = ev.b * 2 + (ev.p < 0).long()
        nb = 2 * B
    if cfg.kernel == "gauss":
        return splat(xy_ref, w, b, nb, ev.height, ev.width, kernel="gauss", sigma=cfg.sigma)
    img = splat(xy_ref, w, b, nb, ev.height, ev.width, kernel="bilinear")
    return blur(img, cfg.sigma) if cfg.sigma > 0 else img


def sobel(img: torch.Tensor):
    kx = img.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = img.new_tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3)
    x = img[:, None]
    return (torch.nn.functional.conv2d(x, kx, padding=1)[:, 0], torch.nn.functional.conv2d(x, ky, padding=1)[:, 0])


def ref_times(spec, device) -> Sequence:
    """'random' -> one uniform t_ref per call (MotionPriorCMax); float -> fixed; list -> multi-reference."""
    if spec == "random":
        return [torch.rand((), device=device)]
    if isinstance(spec, (int, float)):
        return [float(spec)]
    return [float(s) for s in spec]
