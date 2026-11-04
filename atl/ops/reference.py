"""Pure-torch reference implementations (differentiable through autograd).  They define the semantics of the
C++/CUDA kernels and are used by the tests and as a fallback."""
from __future__ import annotations

import math

import torch


def _corners(pos: torch.Tensor):
    x, y = pos[:, 0], pos[:, 1]
    x0, y0 = torch.floor(x), torch.floor(y)
    ax, ay = x - x0, y - y0
    out = []
    for dy, wy in ((0, 1 - ay), (1, ay)):
        for dx, wx in ((0, 1 - ax), (1, ax)):
            out.append(((x0 + dx).long(), (y0 + dy).long(), wx * wy))
    return out


def splat_bilinear(pos, w, b, B, H, W):
    img = pos.new_zeros(B * H * W)
    for xx, yy, c in _corners(pos):
        m = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
        idx = (b[m] * H + yy[m]) * W + xx[m]
        img = img.index_add(0, idx, (w * c)[m])
    return img.view(B, H, W)


def splat_gauss(pos, w, b, B, H, W, sigma, radius):
    img = pos.new_zeros(B * H * W)
    cx, cy = torch.round(pos[:, 0]).long(), torch.round(pos[:, 1]).long()
    norm = 1.0 / (2 * math.pi * sigma * sigma)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            xx, yy = cx + dx, cy + dy
            m = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            d2 = (xx.to(pos.dtype) - pos[:, 0]) ** 2 + (yy.to(pos.dtype) - pos[:, 1]) ** 2
            val = w * norm * torch.exp(-d2 / (2 * sigma * sigma))
            img = img.index_add(0, ((b[m] * H + yy[m]) * W + xx[m]), val[m])
    return img.view(B, H, W)


def gather_bilinear(img, pos, b):
    B, H, W = img.shape
    flat = img.reshape(-1)
    out = pos.new_zeros(pos.shape[0])
    for xx, yy, c in _corners(pos):
        m = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
        idx = ((b * H + yy.clamp(0, H - 1)) * W + xx.clamp(0, W - 1))
        out = out + torch.where(m, c * flat[idx], torch.zeros_like(c))
    return out


def ats_pair_sums(x, t, v, b, anchors, h, tau, cut=3.0, chunk=2048, anchor_xy=None):
    """Brute force S_i = sum_{j != i, same batch, |t_i-t_j|<=tau, |r|<=cut h} exp(-|r|^2/2h^2),
    r = x_i - x_j - v_j (t_i - t_j).  O(M N); differentiable w.r.t. v."""
    outs = []
    for s in range(0, anchors.numel(), chunk):
        a = anchors[s:s + chunk]
        xa = x[a] if anchor_xy is None else anchor_xy[s:s + chunk]
        dt = t[a][:, None] - t[None, :]                                      # (m, N)
        r = xa[:, None, :] - x[None] - v[None] * dt[..., None]
        d2 = (r * r).sum(-1)
        ok = (b[a][:, None] == b[None]) & (dt.abs() <= tau) & (d2 <= (cut * h) ** 2)
        ok = ok & (a[:, None] != torch.arange(t.numel(), device=t.device)[None])
        outs.append(torch.where(ok, torch.exp(-d2 / (2 * h * h)), torch.zeros_like(d2)).sum(1))
    return torch.cat(outs) if outs else x.new_zeros(0)


def voxel_grid(pos, t, p, b, B, C, H, W):
    vox = pos.new_zeros(B * C * H * W)
    tn = t * (C - 1)
    t0 = torch.floor(tn)
    for dt in (0, 1):
        tt = (t0 + dt).long()
        wt = 1 - (tt.to(pos.dtype) - tn).abs()
        for xx, yy, c in _corners(pos):
            m = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H) & (tt >= 0) & (tt < C)
            idx = (((b * C + tt) * H + yy) * W + xx)[m]
            vox = vox.index_add(0, idx, (p * wt * c)[m])
    return vox.view(B, C, H, W)
