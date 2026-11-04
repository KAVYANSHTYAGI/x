"""Anchored Transport Score (ours) and unsupervised evaluation metrics.

    S(theta) = mean_i log( eps + rho_{-i}(x_i) ),
    rho_{-i}(x) = (1 / W_i) sum_{j != i, |t_i - t_j| <= tau} K_h( x - Phi_{t_j -> t_i}(x_j) )

Each RAW event x_i is scored under the density predicted by all other events transported to *its own* timestamp.
The evaluation point never moves, so contraction cannot raise the score (no event collapse); there is no reference
time and no regulariser weight.  See PROPER_ALIGNMENT_MATH.md (Thm 5, 7).

Two implementations:
* :class:`AnchoredExact`  -- exact Gaussian kernel and exact leave-one-out for motion that is linear at the event
  (x_j + v_j (t_i - t_j)); C++/CUDA pair kernel with spatial hashing (atl.ops.ats_pair_sums).
* :class:`AnchoredSliced` -- any motion model.  Anchors are grouped in time slices; for slice s all sources are
  transported to the slice centre c_s and splatted with a Gaussian kernel, anchors are moved by the small step
  t_i -> c_s and the change of variables log|det| of that step is added exactly (finite differences).  Equal to the
  exact score up to O(slice width) in the kernel shape.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from ..core.events import EventPacket
from ..ops import ats_pair_sums, gather, splat, window_counts
from .base import Objective
from .cmax import _self_term


@torch.no_grad()
def visible_fraction(motion, xa, ta, ba, ev: EventPacket, tau: float, n_check: int = 16) -> torch.Tensor:
    """Fraction of source events (within +-tau of the anchor) recorded while the anchor's scene content was inside
    the field of view: content entering / leaving the frame has no sources at the other times, so the predictive
    density must be normalised by the visible source mass (exposure correction).  Checked at n_check time slices,
    weighted by the per-batch event counts in each slice."""
    edges = torch.linspace(0, 1, n_check + 1, device=ev.device)
    centres = 0.5 * (edges[:-1] + edges[1:])
    sl = torch.clamp((ev.t * n_check).long(), 0, n_check - 1)
    counts = torch.zeros(ev.batch_size, n_check, device=ev.device)
    counts.index_put_((ev.b, sl), torch.ones_like(ev.t), accumulate=True)
    vis = torch.zeros_like(ta); tot = torch.zeros_like(ta)
    for k in range(n_check):
        c = float(centres[k])
        near = ((ta - c).abs() <= tau).to(ta.dtype)
        y = motion.transport(xa, ta, c, ba)
        inside = ((y[:, 0] >= 0) & (y[:, 0] <= ev.width - 1) & (y[:, 1] >= 0) & (y[:, 1] <= ev.height - 1)).to(ta.dtype)
        w = counts[ba, k] * near
        vis += w * inside; tot += w
    return (vis / tot.clamp(min=1)).clamp(min=1e-3)


def _pick_anchors(ev: EventPacket, n_anchors: Optional[int], gen: Optional[torch.Generator]) -> torch.Tensor:
    if n_anchors is None or n_anchors >= ev.n:
        return torch.arange(ev.n, device=ev.device)
    idx = torch.randperm(ev.n, generator=gen, device="cpu")[:n_anchors].to(ev.device)
    return torch.sort(idx).values


class AnchoredExact(Objective):
    name = "ats_exact"
    reference_frame = False

    def __init__(self, h: float = 1.0, tau: float = 1.0, eps_frac: float = 1e-2, n_anchors: Optional[int] = 20000,
                 n_slices: int = 16, cut: float = 3.0, seed: Optional[int] = None, smooth_anchors: bool = True,
                 dequantize: bool = True, visibility: bool = True, anchor_margin: float = 0.0):
        self.visibility, self.anchor_margin = visibility, anchor_margin
        self.h, self.tau, self.eps_frac, self.n_anchors, self.n_slices, self.cut = h, tau, eps_frac, n_anchors, n_slices, cut
        # smooth_anchors: anchors ~ p * K_h, so the population score is -KL(p*K_h || q*K_h): proper at ANY h.
        # dequantize: uniform [-1/2, 1/2]^2 jitter of integer pixel coordinates (sources and anchors).
        self.smooth_anchors, self.dequantize = smooth_anchors, dequantize
        self.gen = torch.Generator().manual_seed(seed) if seed is not None else None

    def log_scores(self, ev: EventPacket, motion, anchors: Optional[torch.Tensor] = None):
        if not getattr(motion, "linear_at_event", False):
            raise ValueError("AnchoredExact needs a motion that is linear at the event; use AnchoredSliced")
        if anchors is None:
            anchors = _pick_anchors(ev, self.n_anchors, self.gen)
            if self.anchor_margin > 0:        # selection on RAW positions: independent of the motion, stays proper
                anchors = anchors[ev.inside(ev.xy[anchors], margin=-self.anchor_margin)]
        xs = ev.xy
        if self.dequantize:
            xs = xs + (torch.rand(xs.shape, generator=self.gen).to(xs) - 0.5)
        xa = xs[anchors]
        if self.smooth_anchors:
            xa = xa + self.h * torch.randn(xa.shape, generator=self.gen).to(xa)
        v = motion.velocity(xs, ev.t, ev.b)
        S = ats_pair_sums(xs, ev.t, v, ev.b, ev.batch_size, ev.height, ev.width, anchors, self.h, self.tau,
                          self.cut, self.n_slices, anchor_xy=xa)
        Wn = window_counts(ev.t, ev.b, anchors, self.tau).clamp(min=1)
        if self.visibility:
            Wn = Wn * visible_fraction(motion, xa.detach(), ev.t[anchors], ev.b[anchors], ev, self.tau)
        eps = self.eps_frac / (ev.height * ev.width)
        return torch.log(eps + S / (Wn * 2 * math.pi * self.h ** 2)), anchors

    def set_bandwidth(self, h: float):
        self.h = float(h)

    def __call__(self, ev, motion):
        ls, _ = self.log_scores(ev, motion)
        s = ls.mean()
        return {"loss": -s, "score": s.detach()}


class AnchoredSliced(Objective):
    name = "ats_sliced"
    reference_frame = False

    def __init__(self, n_slices: int = 10, slices_per_step: Optional[int] = None, sigma: float = 1.0,
                 tau: float = 1.0, eps_frac: float = 1e-2, n_anchors_per_slice: Optional[int] = None,
                 logdet: bool = True, seed: Optional[int] = None, smooth_anchors: bool = True,
                 dequantize: bool = True, visibility: bool = True):
        self.smooth_anchors, self.dequantize, self.visibility = smooth_anchors, dequantize, visibility
        self.S, self.sps, self.sigma, self.tau, self.eps_frac = n_slices, slices_per_step, sigma, tau, eps_frac
        self.nap, self.logdet = n_anchors_per_slice, logdet
        self.gen = torch.Generator().manual_seed(seed) if seed is not None else None

    def log_scores(self, ev: EventPacket, motion, anchor_mask: Optional[torch.Tensor] = None,
                   source_mask: Optional[torch.Tensor] = None):
        S, dev = self.S, ev.device
        delta = 1.0 / S
        sl = torch.clamp(torch.floor(ev.t / delta), 0, S - 1).long()
        slices = list(range(S))
        if self.sps is not None and self.sps < S:
            slices = torch.randperm(S, generator=self.gen)[: self.sps].tolist()
        eps = self.eps_frac / (ev.height * ev.width)
        xs = ev.xy
        if self.dequantize:
            xs = xs + (torch.rand(xs.shape, generator=self.gen).to(xs) - 0.5)
        xsm = xs + self.sigma * torch.randn(xs.shape, generator=self.gen).to(xs) if self.smooth_anchors else xs
        out, idx_all = [], []
        for s in slices:
            c = (s + 0.5) * delta
            src = (ev.t - c).abs() <= self.tau + delta / 2
            if source_mask is not None:
                src = src & source_mask
            anc = sl == s
            if anchor_mask is not None:
                anc = anc & anchor_mask
            ai = torch.nonzero(anc).squeeze(1)
            if ai.numel() == 0:
                continue
            if self.nap is not None and ai.numel() > self.nap:
                ai = ai[torch.randperm(ai.numel(), generator=self.gen)[: self.nap].to(dev)]
            si = torch.nonzero(src).squeeze(1)
            y = motion.transport(xs[si], ev.t[si], c, ev.b[si])
            img = splat(y, torch.ones_like(ev.t[si]), ev.b[si], ev.batch_size, ev.height, ev.width,
                        kernel="gauss", sigma=self.sigma)
            if self.logdet:
                a, ld = motion.transport_logdet(xsm[ai], ev.t[ai], c, ev.b[ai])
            else:
                a, ld = motion.transport(xsm[ai], ev.t[ai], c, ev.b[ai]), 0.0
            val = gather(img, a, ev.b[ai])
            # leave-one-out: remove the anchor's own splat (its source copy sits at transport(xs_i))
            if self.smooth_anchors:
                y_own = motion.transport(xs[ai], ev.t[ai], c, ev.b[ai])
                own = src[ai].to(val.dtype) * _cross_term(a, y_own, self.sigma, ev.height, ev.width)
            else:
                own = src[ai].to(val.dtype) * _self_term(a, self.sigma, ev.height, ev.width)
            nsrc = torch.bincount(ev.b[si], minlength=ev.batch_size).to(val.dtype)[ev.b[ai]] - src[ai].to(val.dtype)
            if self.visibility:
                nsrc = nsrc * visible_fraction(motion, xsm[ai].detach(), ev.t[ai], ev.b[ai], ev, self.tau)
            rho = (val - own).clamp(min=0) / nsrc.clamp(min=1)
            out.append(torch.log(eps + rho) + ld)
            idx_all.append(ai)
        if not out:
            z = ev.t.new_zeros(0)
            return z, z.long()
        return torch.cat(out), torch.cat(idx_all)

    def set_bandwidth(self, h: float):
        self.sigma = float(h)

    def __call__(self, ev, motion):
        ls, _ = self.log_scores(ev, motion)
        s = ls.mean() if ls.numel() else ev.t.new_zeros(())
        return {"loss": -s, "score": s.detach()}


def _cross_term(a: torch.Tensor, y: torch.Tensor, sigma: float, H: int, W: int) -> torch.Tensor:
    """Bilinear look-up at ``a`` of the Gaussian splat centred at ``y`` (truncated at radius ceil(3 sigma))."""
    R = int(math.ceil(3 * sigma))
    cy0, cx0 = torch.round(y[:, 1]), torch.round(y[:, 0])
    x0, y0 = torch.floor(a[:, 0]), torch.floor(a[:, 1])
    ax, ay = a[:, 0] - x0, a[:, 1] - y0
    s = torch.zeros_like(ax)
    norm = 1.0 / (2 * math.pi * sigma * sigma)
    for dy, wy in ((0, 1 - ay), (1, ay)):
        for dx, wx in ((0, 1 - ax), (1, ax)):
            px, py = x0 + dx, y0 + dy
            ok = ((px >= 0) & (px < W) & (py >= 0) & (py < H) & ((px - cx0).abs() <= R) & ((py - cy0).abs() <= R))
            d2 = (px - y[:, 0]) ** 2 + (py - y[:, 1]) ** 2
            s = s + wx * wy * ok.to(a.dtype) * norm * torch.exp(-d2 / (2 * sigma * sigma))
    return s


# --------------------------------------------------------------------------------------------- metrics
@torch.no_grad()
def heldout_score(ev: EventPacket, motion, frac: float = 0.2, seed: int = 0, sigma: float = 1.0,
                  n_slices: int = 10) -> float:
    """Proper unsupervised metric: anchored log score of held-out events under the other events.  Higher is
    better; cannot be raised by collapse (unlike FWL / RSAT)."""
    g = torch.Generator().manual_seed(seed)
    held = (torch.rand(ev.n, generator=g) < frac).to(ev.device)
    obj = AnchoredSliced(n_slices=n_slices, sigma=sigma)
    ls, _ = obj.log_scores(ev, motion, anchor_mask=held, source_mask=~held)
    return float(ls.mean())


@torch.no_grad()
def flow_warp_loss(ev: EventPacket, motion, t_ref: float = 0.0, sigma: float = 1.0) -> float:
    """FWL (Stoffregen et al. ECCV'20): var(IWE with motion) / var(IWE without motion).  Higher = sharper."""
    from .base import IWEConfig, make_iwe
    cfg = IWEConfig(kernel="bilinear", sigma=sigma)
    y = motion.transport(ev.xy, ev.t, t_ref, ev.b)
    a = make_iwe(ev, y, t_ref, cfg).var(dim=(-2, -1))
    z = make_iwe(ev, ev.xy, t_ref, cfg).var(dim=(-2, -1))
    return float((a / z.clamp(min=1e-12)).mean())


@torch.no_grad()
def rsat(ev: EventPacket, motion, n_ref: int = 5) -> float:
    """RSAT (Paredes-Valles et al. ICCV'23): timestamp loss with motion / without motion.  Lower = better."""
    from .cmax import Timestamp

    class _Zero:
        def transport(self, xy, t, t_dst, b):
            return xy
    ts = Timestamp(taming=True, n_ref=n_ref)
    return float(ts(ev, motion)["loss"] / ts(ev, _Zero())["loss"].clamp(min=1e-12))
