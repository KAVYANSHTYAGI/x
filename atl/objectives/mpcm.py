"""Faithful re-implementation of the MotionPriorCMax loss (Hamann et al., ECCV'24; official code
github.com/tub-rip/MotionPriorCMax, src/losses/focus.py, MIT licence), ported to packed events / (x, y) order.

DSEC configuration of the official repo (config/exe/flow_training/dsec.yaml):
    num_tref=1 (random t_ref ~ U(0,1)), num_bins=15, num_knn=32, smooth_weight=0.003, lut_superpixel_size=4,
    focus_loss_norm=l1 (gradient magnitude), dist_norm=l2, scale_iwe_by_dt=True, mask_image_border=True,
    polarity_aware_batching=True, interpolation_scheme=mean, smooth_type=on_flow_to_tref.

Pipeline: tile trajectories at (t_ref, bin mids) -> for every look-up-table cell and bin, mean displacement-to-t_ref
of its K nearest trajectories -> events displaced by the LUT entry of (bin, cell) -> polarity-split bilinear IWE,
sigma=1 blur, weights (1-|t-t_ref|), border mask -> focus = 1 / mean(|Sobel_x| + |Sobel_y|) -> + lambda * Charbonnier
smoothness of the LUT displacement field.
"""
from __future__ import annotations

import torch

from ..core.events import EventPacket
from ..motion.trajectories import PixelTrajectories
from .base import IWEConfig, Objective, make_iwe, sobel
from .cmax import charbonnier_tv

try:  # same KNN backend as the official code when available
    from pykeops.torch import LazyTensor  # type: ignore
    _HAS_KEOPS = True
except Exception:  # pragma: no cover
    _HAS_KEOPS = False


def knn_indices(queries: torch.Tensor, points: torch.Tensor, k: int, chunk: int = 4096) -> torch.Tensor:
    """queries (Q, 2), points (B, T, n, 2) -> indices (B, T, Q, k) of the k nearest points (L2)."""
    B, T, n, _ = points.shape
    if _HAS_KEOPS and points.is_cuda:
        x_i = LazyTensor(queries[None, None, :, None, :].contiguous())
        y_j = LazyTensor(points[:, :, None, :, :].contiguous())
        return ((x_i - y_j) ** 2).sum(-1).argKmin(k, dim=3)
    out = []
    for s in range(0, queries.shape[0], chunk):
        q = queries[s:s + chunk]
        d = torch.cdist(q[None, None].expand(B, T, -1, -1).reshape(B * T, -1, 2), points.reshape(B * T, n, 2))
        out.append(torch.topk(d, k, dim=-1, largest=False).indices.view(B, T, -1, k))
    return torch.cat(out, 2)


class MotionPriorCMaxLoss(Objective):
    name = "mpcm_focus"

    def __init__(self, num_bins: int = 15, num_knn: int = 32, smooth_weight: float = 0.003, lut: int = 4,
                 t_ref="random", scale_by_dt: bool = True, polarity_split: bool = True, sigma: float = 1.0,
                 focus_norm: str = "l1"):
        self.num_bins, self.num_knn, self.smooth_weight, self.lut = num_bins, num_knn, smooth_weight, lut
        self.t_ref, self.scale_by_dt, self.polarity_split, self.sigma = t_ref, scale_by_dt, polarity_split, sigma
        self.focus_norm = focus_norm

    def lut_flow(self, motion: PixelTrajectories, H: int, W: int, t_ref: torch.Tensor) -> torch.Tensor:
        dev = motion.tile_c.device
        edges = torch.linspace(0, 1, self.num_bins + 1, device=dev)
        t_mid = 0.5 * (edges[:-1] + edges[1:])
        times = torch.cat([t_ref.view(1), t_mid])
        tr = motion.tile_trajectories(times)                       # (B, 1 + nb, n, 2)
        at_ref, at_mid = tr[:, :1], tr[:, 1:]
        off = self.lut / 2.0 - 0.5
        ys = torch.arange(0, H, self.lut, device=dev, dtype=torch.float32) + off
        xs = torch.arange(0, W, self.lut, device=dev, dtype=torch.float32) + off
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx, gy], -1).reshape(-1, 2)            # (Q, 2)
        k = min(self.num_knn, at_mid.shape[2])
        idx = knn_indices(grid, at_mid.detach(), k)                # (B, nb, Q, k) -- indices carry no gradient
        flow_to_ref = at_ref - at_mid                              # (B, nb, n, 2)
        B, nb, n, _ = flow_to_ref.shape
        Q = grid.shape[0]
        flat = flow_to_ref.reshape(B * nb, n, 2)
        ii = idx.reshape(B * nb, Q * k)
        g = torch.gather(flat, 1, ii[..., None].expand(-1, -1, 2)).view(B, nb, Q, k, 2)
        lut = g.mean(3)                                            # (B, nb, Q, 2)
        return lut.view(B, nb, len(ys), len(xs), 2), edges

    def __call__(self, ev: EventPacket, motion: PixelTrajectories):
        H, W = ev.height, ev.width
        t_ref = torch.rand((), device=ev.device) if self.t_ref == "random" else torch.tensor(float(self.t_ref),
                                                                                             device=ev.device)
        lut, edges = self.lut_flow(motion, H, W, t_ref)
        nb = self.num_bins
        bins = (torch.searchsorted(edges, ev.t.contiguous()) - 1).clamp(0, nb - 1)
        iy = torch.div(ev.xy[:, 1], self.lut, rounding_mode="floor").long().clamp(0, lut.shape[2] - 1)
        ix = torch.div(ev.xy[:, 0], self.lut, rounding_mode="floor").long().clamp(0, lut.shape[3] - 1)
        disp = lut[ev.b, bins, iy, ix]
        warped = ev.xy + disp
        cfg = IWEConfig(kernel="bilinear", sigma=self.sigma, polarity="split" if self.polarity_split else "pooled",
                        mask_border=True, scale_by_dt=self.scale_by_dt)
        iwe = make_iwe(ev, warped, t_ref, cfg)
        gx, gy = sobel(iwe)
        G = (gx.abs() + gy.abs()).mean() if self.focus_norm == "l1" else (gx ** 2 + gy ** 2).mean()
        focus = 1.0 / (G + 1e-12)
        B = ev.batch_size
        field = lut.permute(0, 1, 4, 2, 3).reshape(B * nb, 2, lut.shape[2], lut.shape[3])
        smooth = self.smooth_weight * charbonnier_tv(field) if self.smooth_weight > 0 else focus.new_zeros(())
        return {"loss": focus + smooth, "focus": focus.detach(), "smooth": smooth.detach(), "sharpness": G.detach()}
