"""Reference-frame baselines: contrast maximisation and its descendants.

* ``Sharpness(metric="variance")``          Gallego et al. CVPR'18 (CMax), single t_ref
* ``Sharpness(..., t_ref=[...])``            multi-reference focus loss, Shiba et al. ECCV'22 ("Secrets")
* ``Sharpness(metric="grad_l1", t_ref="random")``  MotionPriorCMax's focus term (Hamann et al. ECCV'24)
* ``EMin``                                  entropy minimisation (Nunes & Demiris ECCV'20), KDE form via IWE look-up
* ``STPPP``                                 per-pixel negative-binomial likelihood of aligned counts (Gu et al. ICCV'21)
* ``Timestamp``                             average-timestamp loss (Zhu et al. CVPR'19); ``Timestamp(taming=True)``
                                            uses the multi-reference, active-pixel normalisation of Paredes-Valles et
                                            al. ICCV'23
* ``Regularized``                           base + lambda * R  (Shiba et al. Sensors'22 R_D / R_A, Charbonnier TV)
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from ..core.events import EventPacket
from ..ops import gather
from .base import IWEConfig, Objective, make_iwe, ref_times, sobel


def _metric(iwe: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "variance":
        return iwe.var(dim=(-2, -1)).mean()
    if metric == "mean_square":
        return (iwe ** 2).mean()
    if metric in ("grad_l1", "grad_l2"):
        gx, gy = sobel(iwe)
        return (gx.abs() + gy.abs()).mean() if metric == "grad_l1" else (gx ** 2 + gy ** 2).mean()
    raise ValueError(metric)


class Sharpness(Objective):
    """Sharpness of the IWE at one or several reference times (variance, mean square, gradient L1/L2)."""
    def __init__(self, metric: str = "variance", t_ref=0.0, iwe: Optional[IWEConfig] = None, form: str = "neg",
                 name: Optional[str] = None):
        self.metric, self.t_ref, self.form = metric, t_ref, form
        self.iwe = iwe or IWEConfig()
        self.name = name or f"sharp[{metric},tref={t_ref}]"

    def sharpness(self, ev: EventPacket, motion):
        """Mean sharpness over the reference times."""
        vals = []
        for tr in ref_times(self.t_ref, ev.device):
            xy = motion.transport(ev.xy, ev.t, tr, ev.b)
            vals.append(_metric(make_iwe(ev, xy, tr, self.iwe), self.metric))
        return torch.stack(vals).mean()

    def __call__(self, ev, motion):
        g = self.sharpness(ev, motion)
        loss = -g if self.form == "neg" else 1.0 / (g + 1e-12)
        return {"loss": loss, "sharpness": g.detach()}


class EMin(Objective):
    """Shannon entropy of the warped events, KDE evaluated by IWE look-up at the *warped* events (leave-one-out).
    Minimising entropy = maximising mean log density at moved points (improper: collapse maximises it)."""
    name = "emin"

    def __init__(self, t_ref=0.0, sigma: float = 1.0, eps_frac: float = 1e-2):
        self.t_ref, self.sigma, self.eps_frac = t_ref, sigma, eps_frac

    def __call__(self, ev, motion):
        out = []
        for tr in ref_times(self.t_ref, ev.device):
            y = motion.transport(ev.xy, ev.t, tr, ev.b)
            iwe = make_iwe(ev, y, tr, IWEConfig(kernel="gauss", sigma=self.sigma, mask_border=False))
            dens = gather(iwe, y, ev.b) - _self_term(y, self.sigma, ev.height, ev.width)
            nb = torch.bincount(ev.b, minlength=ev.batch_size).clamp(min=2).to(dens.dtype)[ev.b] - 1
            eps = self.eps_frac / (ev.height * ev.width)
            out.append(torch.log(eps + dens.clamp(min=0) / nb).mean())
        s = torch.stack(out).mean()
        return {"loss": -s, "score": s.detach()}


class STPPP(Objective):
    """Gu et al. ICCV'21: aligned per-pixel counts ~ Poisson(lambda_x), lambda_x ~ Gamma -> negative binomial."""
    name = "stppp"

    def __init__(self, t_ref=0.0, r: float = 0.1, q: float = 0.39, sigma: float = 1.0):
        self.t_ref, self.r, self.q, self.sigma = t_ref, r, q, sigma

    def __call__(self, ev, motion):
        out = []
        for tr in ref_times(self.t_ref, ev.device):
            y = motion.transport(ev.xy, ev.t, tr, ev.b)
            k = make_iwe(ev, y, tr, IWEConfig(kernel="bilinear", sigma=self.sigma, polarity="split"))
            r, q = self.r, self.q
            ll = (torch.lgamma(k + r) - torch.lgamma(k + 1) - math.lgamma(r) + r * math.log(1 - q) + k * math.log(q))
            out.append(ll.sum() / max(ev.n, 1))
        s = torch.stack(out).mean()
        return {"loss": -s, "loglik": s.detach()}


class Timestamp(Objective):
    """Average-timestamp image loss.  Zhu'19: warp to t_ref in {1 (fwd), 0 (bwd)}, timestamps |t - t_ref|,
    loss = sum_x T+(x)^2 + T-(x)^2.  taming=True: multi-reference over a uniform grid and normalisation by the
    number of active pixels (Paredes-Valles et al. ICCV'23)."""

    def __init__(self, taming: bool = False, n_ref: int = 5, sigma: float = 0.0):
        self.taming, self.n_ref, self.sigma = taming, n_ref, sigma
        self.name = "taming_ts" if taming else "zhu_ts"

    def __call__(self, ev, motion):
        refs = torch.linspace(0, 1, self.n_ref).tolist() if self.taming else [0.0, 1.0]
        tot = []
        for tr in refs:
            y = motion.transport(ev.xy, ev.t, tr, ev.b)
            cfg = IWEConfig(kernel="bilinear", sigma=self.sigma, polarity="split")
            cnt = make_iwe(ev, y, tr, cfg)
            ts = make_iwe(ev, y, tr, cfg, extra_weight=(ev.t - tr).abs())
            T = ts / (cnt + 1e-9)
            if self.taming:
                act = (cnt > 1e-3).float().sum(dim=(-2, -1)).clamp(min=1)
                tot.append(((T ** 2).sum(dim=(-2, -1)) / act).mean())
            else:
                tot.append((T ** 2).sum(dim=(-2, -1)).mean() / (ev.height * ev.width))
        loss = torch.stack(tot).mean()
        return {"loss": loss}


def _self_term(y: torch.Tensor, sigma: float, H: int, W: int) -> torch.Tensor:
    """Contribution of an event's own Gaussian splat to the bilinear look-up at its own position."""
    x0, y0 = torch.floor(y[:, 0]), torch.floor(y[:, 1])
    ax, ay = y[:, 0] - x0, y[:, 1] - y0
    s = torch.zeros_like(ax)
    norm = 1.0 / (2 * math.pi * sigma * sigma)
    for dy, wy in ((0, 1 - ay), (1, ay)):
        for dx, wx in ((0, 1 - ax), (1, ax)):
            cx, cy = x0 + dx, y0 + dy
            inside = ((cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)).to(y.dtype)
            d2 = (cx - y[:, 0]) ** 2 + (cy - y[:, 1]) ** 2
            s = s + wx * wy * inside * norm * torch.exp(-d2 / (2 * sigma * sigma))
    return s


# ------------------------------------------------------------------------------------------- regularisers
def charbonnier_tv(field: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """MotionPriorCMax smoothness: Charbonnier of Sobel gradients of a (B, C, H, W) field."""
    B, C, H, W = field.shape
    gx, gy = sobel(field.reshape(B * C, H, W))
    return 0.5 * (torch.sqrt(gx ** 2 + eps ** 2).mean() + torch.sqrt(gy ** 2 + eps ** 2).mean())


def _warp_logdet_first_order(ev, motion, tr):
    """First-order log-det of the warp to t_ref at each event: div v (x) * (t_ref - t)  (Shiba R_D surrogate)."""
    div = motion.divergence(ev.xy, ev.b)
    tr = tr if torch.is_tensor(tr) else torch.tensor(float(tr), device=ev.device)
    return div * (tr - ev.t)


class Regularized(Objective):
    """base + weight * R.  reg in {"div" (Shiba R_D), "det" (Shiba R_A), "tv" (MotionPriorCMax Charbonnier)}.
    R_D / R_A penalise only contraction beyond a dead zone (margin), as in the trimmed versions of Shiba et al."""

    def __init__(self, base: Objective, reg: str, weight: float, margin: Optional[float] = None, t_ref=None):
        self.base, self.reg, self.weight = base, reg, weight
        self.t_ref = getattr(base, "t_ref", 0.0) if t_ref is None else t_ref
        self.margin = margin if margin is not None else {"div": -0.2, "det": 0.8, "tv": 0.0}[reg]
        self.name = f"{base.name}+{weight:g}*R_{reg}"

    def penalty(self, ev, motion):
        """Value of the regulariser R for the current motion."""
        if self.reg == "tv":
            f = motion.regulariser_fields(ev.batch_size, ev.height, ev.width)
            return charbonnier_tv(f) if f is not None else ev.t.new_zeros(())
        vals = []
        for tr in ref_times(self.t_ref, ev.device):
            if self.reg == "div":
                ld = _warp_logdet_first_order(ev, motion, tr)
                vals.append(torch.relu(self.margin - ld).mean())
            else:
                _, ld = motion.transport_logdet(ev.xy, ev.t, tr, ev.b)
                vals.append(torch.relu(self.margin - ld.exp()).mean())
        return torch.stack(vals).mean()

    def __call__(self, ev, motion):
        out = self.base(ev, motion)
        r = self.penalty(ev, motion)
        out = dict(out)
        out["loss"] = out["loss"] + self.weight * r
        out["reg"] = r.detach()
        return out
