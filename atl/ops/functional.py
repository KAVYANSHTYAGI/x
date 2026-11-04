"""Differentiable ops with C++/CUDA fast paths.

* :func:`splat`        events -> image (bilinear vote or truncated Gaussian), grads w.r.t. positions and weights
* :func:`gather`       bilinear sample of an image at event positions, grads w.r.t. image and positions
* :func:`ats_pair_sums` exact anchored leave-one-out Gaussian sums for per-event-velocity motion, grad w.r.t. v
* :func:`voxel_grid`   trilinear polarity voting (network input), no grad
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import reference as ref
from ._ext import ext_supports, get_ext


def _f32(t):
    return t.contiguous().float()


# ------------------------------------------------------------------------------------------- splat
class _SplatBilinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, w, b, B, H, W):
        pos, w, b = _f32(pos), _f32(w), b.contiguous().long()
        ctx.save_for_backward(pos, w, b)
        return get_ext().splat_bilinear_fwd(pos, w, b, B, H, W)

    @staticmethod
    def backward(ctx, g):
        pos, w, b = ctx.saved_tensors
        gp, gw = get_ext().splat_bilinear_bwd(g.contiguous().float(), pos, w, b)
        return gp, gw, None, None, None, None


class _SplatGauss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos, w, b, B, H, W, sigma, radius):
        pos, w, b = _f32(pos), _f32(w), b.contiguous().long()
        ctx.save_for_backward(pos, w, b)
        ctx.sigma, ctx.radius = sigma, radius
        return get_ext().splat_gauss_fwd(pos, w, b, B, H, W, sigma, radius)

    @staticmethod
    def backward(ctx, g):
        pos, w, b = ctx.saved_tensors
        gp, gw = get_ext().splat_gauss_bwd(g.contiguous().float(), pos, w, b, ctx.sigma, ctx.radius)
        return gp, gw, None, None, None, None, None, None


def splat(pos: torch.Tensor, w: torch.Tensor, b: torch.Tensor, B: int, H: int, W: int,
          kernel: str = "bilinear", sigma: float = 1.0, radius: int | None = None) -> torch.Tensor:
    """Accumulate weighted events into a (B, H, W) image."""
    if kernel == "bilinear":
        if ext_supports(pos):
            return _SplatBilinear.apply(pos, w, b, B, H, W)
        return ref.splat_bilinear(pos, w, b, B, H, W)
    if kernel == "gauss":
        radius = int(math.ceil(3 * sigma)) if radius is None else radius
        if ext_supports(pos):
            return _SplatGauss.apply(pos, w, b, B, H, W, float(sigma), int(radius))
        return ref.splat_gauss(pos, w, b, B, H, W, sigma, radius)
    raise ValueError(kernel)


# ------------------------------------------------------------------------------------------- gather
class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, img, pos, b):
        img, pos, b = _f32(img), _f32(pos), b.contiguous().long()
        ctx.save_for_backward(img, pos, b)
        return get_ext().gather_bilinear_fwd(img, pos, b)

    @staticmethod
    def backward(ctx, g):
        img, pos, b = ctx.saved_tensors
        gi, gp = get_ext().gather_bilinear_bwd(g.contiguous().float(), img, pos, b)
        return gi, gp, None


def gather(img: torch.Tensor, pos: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bilinear sample of a (B, H, W) image at event positions (zero outside); differentiable in both."""
    if ext_supports(pos):
        return _Gather.apply(img, pos, b)
    return ref.gather_bilinear(img, pos, b)


# ------------------------------------------------------------------------------------------- blur
def gaussian_kernel1d(sigma: float, device, dtype=torch.float32):
    """Normalised 1-D Gaussian kernel of radius ceil(3 sigma)."""
    r = max(int(math.ceil(3 * sigma)), 1)
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-x * x / (2 * sigma * sigma))
    return k / k.sum()


def blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a (B, H, W) or (B, C, H, W) image (zero padding)."""
    if sigma <= 0:
        return img
    squeeze = img.dim() == 3
    x = img[:, None] if squeeze else img
    C = x.shape[1]
    k = gaussian_kernel1d(sigma, img.device, img.dtype)
    r = (k.numel() - 1) // 2
    x = torch.nn.functional.conv2d(x, k.view(1, 1, 1, -1).expand(C, 1, 1, -1), padding=(0, r), groups=C)
    x = torch.nn.functional.conv2d(x, k.view(1, 1, -1, 1).expand(C, 1, -1, 1), padding=(r, 0), groups=C)
    return x[:, 0] if squeeze else x


# ------------------------------------------------------------------------------------------- anchored pairs
@dataclass
class PairPlan:
    """Host-side spatial hash for :func:`ats_pair_sums` (sources sorted by (batch, slice, cell))."""
    src_y: torch.Tensor
    src_v: torch.Tensor
    src_t: torch.Tensor
    src_id: torch.Tensor
    cell_start: torch.Tensor
    anc_x: torch.Tensor
    anc_t: torch.Tensor
    anc_cell: torch.Tensor
    anc_id: torch.Tensor
    slice_c: torch.Tensor
    S: int
    GH: int
    GW: int


def build_pair_plan(x, t, v, b, B, H, W, anchors, h, tau, cut=3.0, n_slices=16, anchor_xy=None) -> PairPlan:
    """Host-side spatial hash of (source, slice) pairs sorted by (batch, slice, cell) for the pair kernel."""
    dev = x.device
    S = int(n_slices)
    delta = 1.0 / S
    with torch.no_grad():
        vd = v.detach().float()
        vmax = float(vd.norm(dim=1).max()) if vd.numel() else 0.0
        cell = max(cut * h + vmax * delta / 2 + 1e-3, 1.0)
        # grid covers the anchors' bounding box plus one cell (sources further away cannot contribute)
        xa_all = (x[anchors] if anchor_xy is None else anchor_xy).float()
        if xa_all.numel():
            ox, oy = float(xa_all[:, 0].min()) - cell, float(xa_all[:, 1].min()) - cell
            ex, ey = float(xa_all[:, 0].max()) + cell, float(xa_all[:, 1].max()) + cell
        else:
            ox, oy, ex, ey = -cell, -cell, W + cell, H + cell
        GW = int(math.ceil((ex - ox) / cell)) + 1
        GH = int(math.ceil((ey - oy) / cell)) + 1
        # (source, slice) pairs with |t_j - c_s| <= tau + delta/2
        s_lo = torch.ceil((t - tau - delta / 2) / delta - 0.5).clamp(0, S - 1).long()
        s_hi = torch.floor((t + tau + delta / 2) / delta - 0.5).clamp(0, S - 1).long()
        cnt = (s_hi - s_lo + 1).clamp(min=0)
        rep = torch.repeat_interleave(torch.arange(t.numel(), device=dev), cnt)
        start = torch.cumsum(cnt, 0) - cnt
        s = s_lo[rep] + (torch.arange(rep.numel(), device=dev) - start[rep])
        cs_all = (torch.arange(S, device=dev, dtype=torch.float32) + 0.5) * delta
        y = x[rep].float() + vd[rep] * (cs_all[s] - t[rep])[:, None]
        cx = torch.floor((y[:, 0] - ox) / cell).long()
        cy = torch.floor((y[:, 1] - oy) / cell).long()
        keep = (cx >= 0) & (cx < GW) & (cy >= 0) & (cy < GH)
        rep, s, y, cx, cy = rep[keep], s[keep], y[keep], cx[keep], cy[keep]
        key = ((b[rep] * S + s) * GH + cy) * GW + cx
        order = torch.argsort(key)
        key, rep, y = key[order], rep[order], y[order]
        ncell = B * S * GH * GW
        counts = torch.bincount(key, minlength=ncell)
        cell_start = torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)]).long()
        # anchors
        ta = t[anchors]
        sa = torch.clamp(torch.floor(ta / delta), 0, S - 1).long()
        xa = (x[anchors] if anchor_xy is None else anchor_xy).float()
        acx = torch.floor((xa[:, 0] - ox) / cell).long().clamp(0, GW - 1)
        acy = torch.floor((xa[:, 1] - oy) / cell).long().clamp(0, GH - 1)
        anc_cell = torch.stack([b[anchors] * S + sa, acx, acy], 1).contiguous()
    return PairPlan(y.contiguous(), vd[rep].contiguous(), t[rep].float().contiguous(), rep.contiguous(),
                    cell_start, xa.contiguous(), ta.float().contiguous(), anc_cell, anchors.long().contiguous(),
                    cs_all.contiguous(), S, GH, GW)


class _ATSPair(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, plan: PairPlan, h, tau, cut):
        e = get_ext()
        ctx.plan, ctx.args, ctx.n = plan, (float(h), float(tau), float(cut)), v.shape[0]
        p = plan
        return e.ats_pair_fwd(p.src_y, p.src_v, p.src_t, p.src_id, p.cell_start, p.anc_x, p.anc_t, p.anc_cell,
                              p.anc_id, p.slice_c, p.S, p.GH, p.GW, *ctx.args)

    @staticmethod
    def backward(ctx, g):
        p = ctx.plan
        gv = get_ext().ats_pair_bwd(g.contiguous().float(), p.src_y, p.src_v, p.src_t, p.src_id, p.cell_start,
                                    p.anc_x, p.anc_t, p.anc_cell, p.anc_id, p.slice_c, p.S, p.GH, p.GW,
                                    *ctx.args, ctx.n)
        return gv, None, None, None, None


def ats_pair_sums(x, t, v, b, B, H, W, anchors, h, tau, cut=3.0, n_slices=16, force_ref=False, anchor_xy=None):
    """S_i = sum_{j != i} 1[|t_i - t_j| <= tau] exp(-|x_i - x_j - v_j (t_i - t_j)|^2 / 2h^2) for anchors i.

    ``x`` are RAW event positions (never moved), ``v`` per-event velocities in px per unit (normalised) time.
    Gradient flows to ``v`` only (data are fixed)."""
    if force_ref or not ext_supports(x):
        return ref.ats_pair_sums(x.float(), t.float(), v.float(), b, anchors, h, tau, cut, anchor_xy=anchor_xy)
    plan = build_pair_plan(x, t, v, b, B, H, W, anchors, h, tau, cut, n_slices, anchor_xy=anchor_xy)
    return _ATSPair.apply(v.float(), plan, h, tau, cut)


def window_counts(t: torch.Tensor, b: torch.Tensor, idx: torch.Tensor, tau: float) -> torch.Tensor:
    """#{j != i, same batch, |t_j - t_i| <= tau} for i in idx (events sorted by (b, t))."""
    key = b.double() * 4.0 + t.double()
    ks, _ = torch.sort(key)
    ki = key[idx]
    lo = torch.searchsorted(ks, ki - tau, right=False)
    hi = torch.searchsorted(ks, ki + tau, right=True)
    return (hi - lo - 1).clamp(min=0).to(t.dtype)


# ------------------------------------------------------------------------------------------- voxel grid
def voxel_grid(pos, t, p, b, B, C, H, W, normalize: str | None = "mean_std") -> torch.Tensor:
    """Trilinear polarity voxel grid (B, C, H, W), optionally mean/std normalised over non-zero voxels."""
    with torch.no_grad():
        if ext_supports(pos):
            vox = get_ext().voxel_grid(_f32(pos), _f32(t), _f32(p), b.contiguous().long(), B, C, H, W)
        else:
            vox = ref.voxel_grid(pos.float(), t.float(), p.float(), b, B, C, H, W)
        if normalize == "mean_std":
            for k in range(B):
                v = vox[k]
                nz = v != 0
                if nz.any():
                    m, s = v[nz].mean(), v[nz].std()
                    v[nz] = (v[nz] - m) / s if s > 0 else v[nz] - m
        elif normalize == "max":
            mx = vox.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-9)
            vox = vox / mx
    return vox
