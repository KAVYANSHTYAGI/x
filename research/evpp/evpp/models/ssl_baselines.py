"""Non-likelihood pretraining objectives on the SAME continuous-time backbone.

These exist so that the central question of the project - "does a point-process likelihood beat
masked / summary-target pretraining at equal backbone, data and compute?" - can be asked cleanly.

MaskedEventModel   MEM/TESPEC-style: hide 50% of event identities (keep their gaps), a forward and a
                   time-reversed pass of the backbone give bidirectional context, predict (patch,
                   polarity, sub-cell) of the hidden events.
NextCountPrediction  EVA-style summary target: at every event predict the per-patch event COUNT in the
                   next `horizon` seconds (Poisson NLL on counts).  Uses the same causal backbone.
Both expose `loss/nll`-style forward returning {"loss", "parts"} and `features(batch)` for probing.
Assumes fixed-length windows (no padding) for the reversed pass - true for the window sampler.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import EvPPConfig
from .evpp import TokenEmbed
from .heads import dt_features
from .ssm import CTSSMStack


class MaskedEventModel(nn.Module):
    def __init__(self, cfg: EvPPConfig, mask_ratio: float = 0.5):
        super().__init__()
        D = cfg.d_model
        self.cfg, self.ratio = cfg, mask_ratio
        self.embed = TokenEmbed(cfg)
        self.mask_tok = nn.Parameter(torch.zeros(D))
        self.fwd, self.bwd = CTSSMStack(cfg), CTSSMStack(cfg)
        self.patch = nn.Linear(2 * D, cfg.n_patches)
        self.pol = nn.Linear(2 * D, 2)
        self.sub = nn.Linear(2 * D, cfg.n_sub)

    def _hidden(self, batch, hide):
        x = self.embed(batch)
        dtonly = self.embed.dt(dt_features(batch["dt"]))
        x = torch.where(hide[..., None], dtonly + self.mask_tok, x)       # hidden events keep only their gap
        dt = batch["dt"]
        dt_next = torch.cat([dt[:, 1:], torch.full_like(dt[:, :1], 1e-3)], 1)
        sf = self.fwd(x, dt)["s"]
        sb = self.bwd(x.flip(1), dt_next.flip(1))["s"].flip(1)
        return torch.cat([sf, sb], -1)

    def nll(self, batch, state=None):
        m = batch["mask"]
        hide = (torch.rand_like(batch["dt"]) < self.ratio) & m
        h = self._hidden(batch, hide)
        ce = lambda lg, y: F.cross_entropy(lg[hide], y[hide])
        parts = {"patch": ce(self.patch(h), batch["patch"]), "pol": ce(self.pol(h), batch["pol"]),
                 "sub": ce(self.sub(h), batch["sub"])}
        return {"loss": sum(parts.values()), "parts": parts}

    forward = nll

    def features(self, batch):
        h = self._hidden(batch, torch.zeros_like(batch["mask"]))
        m = batch["mask"][..., None].float()
        return (h * m).sum(1) / m.sum(1).clamp_min(1.0)


class NextCountPrediction(nn.Module):
    def __init__(self, cfg: EvPPConfig, horizon: float = 5e-3):
        super().__init__()
        self.cfg, self.h = cfg, horizon
        self.embed = TokenEmbed(cfg)
        self.stack = CTSSMStack(cfg)
        self.out = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.n_patches))
        with torch.no_grad():
            self.out[-1].bias.fill_(math.log(0.5))

    def nll(self, batch, state=None):
        s = self.stack(self.embed(batch), batch["dt"])["s"]                  # state after event i
        t = torch.cumsum(batch["dt"], 1)                                      # (B,L) seconds from window start
        B, L = t.shape
        onehot = F.one_hot(batch["patch"], self.cfg.n_patches).float() * batch["mask"][..., None]
        cum = torch.cat([torch.zeros(B, 1, self.cfg.n_patches), onehot.cumsum(1)], 1)       # (B,L+1,K)
        end = torch.searchsorted(t.contiguous(), (t + self.h).contiguous(), right=True)     # events <= t_i + h
        target = cum.gather(1, end[..., None].expand(-1, -1, self.cfg.n_patches)) \
            - cum.gather(1, (torch.arange(L)[None, :] + 1).expand(B, -1)[..., None].expand(-1, -1, self.cfg.n_patches))
        valid = (t + self.h <= t[:, -1:]) & batch["mask"]                     # target window fully observed
        lam = F.softplus(self.out(s)) + 1e-4
        nll = (lam - target * lam.log()).sum(-1)
        loss = (nll * valid).sum() / valid.sum().clamp_min(1)
        return {"loss": loss, "parts": {"count": loss}}

    forward = nll

    def features(self, batch):
        s = self.stack(self.embed(batch), batch["dt"])["s"]
        m = batch["mask"][..., None].float()
        return (s * m).sum(1) / m.sum(1).clamp_min(1.0)
