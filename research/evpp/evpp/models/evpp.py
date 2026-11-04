"""EvPP: event point-process model (variants "ar", "intensity", "physics") and PatchPoisson (V1).

All variants return per-event negative log-likelihoods in NATS with these reference measures:
    time   : Lebesgue measure in seconds (dequantised timestamps)
    marks  : counting measure on (patch, polarity, sub-cell)
so numbers are comparable across variants and against the HomogeneousPoisson baseline.
Convert to a discrete timestamp likelihood with  nll_time - ln(dt_quantum).
The tail after the last event of a window is NOT scored (windows end at an event).
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import EvPPConfig
from .heads import (LogNormalMix, IntensityTime, PhysicsIntensity, SpaceHead,
                    gauss_legendre_01, integrate_gap)
from .ssm import CTSSMStack


class TokenEmbed(nn.Module):
    def __init__(self, cfg: EvPPConfig):
        super().__init__()
        D, nf = cfg.d_model, cfg.n_freq
        self.patch = nn.Embedding(cfg.n_patches, D)
        self.sub = nn.Embedding(cfg.n_sub, D)
        self.pol = nn.Embedding(2, D)
        self.register_buffer("freqs", (2.0 ** torch.arange(nf)) * math.pi)
        self.pos = nn.Linear(4 * nf, D)            # Fourier features of normalised (u, v): sensor-agnostic
        self.dt = nn.Sequential(nn.Linear(2, D), nn.GELU(), nn.Linear(D, D))
        self.cond = nn.Linear(cfg.cond_dim, D) if cfg.cond_dim > 0 else None

    def forward(self, batch, cond=None):
        from .heads import dt_features
        au = batch["u"][..., None] * self.freqs
        av = batch["v"][..., None] * self.freqs
        ff = torch.cat([au.sin(), au.cos(), av.sin(), av.cos()], -1)
        x = self.patch(batch["patch"]) + self.sub(batch["sub"]) + self.pol(batch["pol"]) \
            + self.pos(ff) + self.dt(dt_features(batch["dt"]))
        if self.cond is not None and cond is not None:
            x = x + self.cond(cond)[:, None]
        return x


class SensorEncoder(nn.Module):
    """Pooled (mean+max) encoder of the backbone states over a short burn-in context -> theta_hat = (ln C, ln r_n).

    Pools CAUSAL backbone states (not raw token embeddings) so it can see pairwise structure such as
    same-pixel re-firing, which is where the contrast threshold shows up."""

    def __init__(self, cfg: EvPPConfig):
        super().__init__()
        D = cfg.d_model
        self.phi = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.rho = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, 2))
        with torch.no_grad():
            self.rho[-1].weight.mul_(0.1)
            self.rho[-1].bias.copy_(torch.tensor([math.log(0.3), math.log(50.0)]))

    def forward(self, u_ctx, mask):
        h = self.phi(u_ctx)
        m = mask[..., None].float()
        mean = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        mx = h.masked_fill(~mask[..., None], -1e4).max(1).values
        return self.rho(torch.cat([mean, mx], -1))


class EvPP(nn.Module):
    def __init__(self, cfg: EvPPConfig):
        super().__init__()
        assert cfg.objective in ("ar", "intensity", "physics")
        self.cfg = cfg
        D = cfg.d_model
        self.embed = TokenEmbed(cfg)
        self.stack = CTSSMStack(cfg)
        self.s_init = nn.Parameter(torch.zeros(D))
        self.space = SpaceHead(D, cfg.d_int, cfg.n_patches, cfg.n_sub)
        self.cond_head = nn.Linear(cfg.cond_dim, D) if cfg.cond_dim > 0 else None
        if cfg.objective == "ar":
            self.time = LogNormalMix(D, cfg.n_mix, -cfg.init_log_rate)
        elif cfg.objective == "intensity":
            self.time = IntensityTime(D, cfg.d_int, cfg.init_log_rate)
        else:
            self.phys = PhysicsIntensity(D, cfg.d_int, cfg.n_patches)
            self.sensor = SensorEncoder(cfg) if cfg.sensor_mode == "infer" else None
        nodes, weights = gauss_legendre_01(cfg.n_quad)
        self.register_buffer("nodes", nodes)
        self.register_buffer("weights", weights)

    # ------------------------------------------------------------------ encoding
    def encode(self, batch: Dict[str, torch.Tensor], state: Optional[dict] = None):
        cond = batch.get("cond")
        u = self.embed(batch, cond)
        out = self.stack(u, batch["dt"], state)
        n = self.cfg.d_int
        s, h_last = out["s"], out["H_save"]
        B = s.shape[0]
        s0 = (state["s"] if state is not None else self.s_init.expand(B, -1))
        H0 = (state["H"] if state is not None else torch.zeros_like(h_last[:, 0]))
        s_prev = torch.cat([s0[:, None], s[:, :-1]], 1)
        H_prev = torch.cat([H0[:, None], h_last[:, :-1]], 1)
        if self.cond_head is not None and cond is not None:
            s_prev = s_prev + self.cond_head(cond)[:, None]
        new_state = dict(out["state"])
        new_state["s"], new_state["H"] = s[:, -1], h_last[:, -1]
        return {**out, "u": u, "s_prev": s_prev, "H_prev": H_prev, "new_state": new_state}

    def _yhat_at(self, H, tau_eff):
        """Last-layer linear readout of state H evolved by tau_eff.  H: (...,n,N), tau_eff: broadcastable (...)."""
        if self.cfg.impl == "evkit" and H.dim() == 4 and tau_eff.dim() == 2:
            return self._evkit_readout(H, tau_eff)
        return self.stack.layers[-1].readout(H, tau_eff, self.cfg.d_int)

    def _yhat_nodes(self, H, tau_eff):
        """Readout of H (B,L,n,N) at S query times per event, tau_eff (B,L,S) -> (B,L,S,n)."""
        if self.cfg.impl == "evkit":
            return self._evkit_readout(H, tau_eff)
        return self.stack.layers[-1].readout(H[:, :, None], tau_eff, self.cfg.d_int)

    def _evkit_readout(self, H, tau_eff):
        from evkit import readout
        layer, n = self.stack.layers[-1], self.cfg.d_int
        return readout(H, tau_eff, layer.rate()[:n], layer.C[:n])

    def _theta(self, batch, enc):
        mode = self.cfg.sensor_mode
        B = enc["s"].shape[0]
        if mode == "oracle":
            return batch["theta"]
        if mode == "infer":
            n = self.cfg.sensor_ctx
            return self.sensor(enc["s"][:, :n], batch["mask"][:, :n])      # causal backbone states see bursts / repeats
        return self.phys.default_theta[None].expand(B, -1)

    # ------------------------------------------------------------------ likelihood
    def nll(self, batch: Dict[str, torch.Tensor], state: Optional[dict] = None, theta_weight: float = 1.0):
        cfg = self.cfg
        enc = self.encode(batch, state)
        dt, mask = batch["dt"], batch["mask"].clone()
        if cfg.objective == "physics" and cfg.sensor_mode == "infer":
            mask[:, :cfg.sensor_ctx] = False                 # burn-in events are not scored (they define theta_hat)
        s_prev, H_prev, dt_eff = enc["s_prev"], enc["H_prev"], enc["dt_eff"]
        scale = enc["log_tau_prev"].exp()
        yhat_end = self._yhat_at(H_prev, dt_eff)                                  # (B,L,n) state evolved to arrival
        z = self.space.context(s_prev, yhat_end, dt)
        comps, extra = {}, {}

        if cfg.objective == "ar":
            comps["time"] = self.time.nll(s_prev, dt)
        else:
            def readout_nodes(tau_eff):                                           # (B,L,S) -> (B,L,S,n)
                return self._yhat_nodes(H_prev, tau_eff)

            if cfg.objective == "intensity":
                def Lam_fn(tau_eff):
                    y = readout_nodes(tau_eff)
                    sp = s_prev[:, :, None].expand(-1, -1, y.shape[2], -1)
                    return self.time.log_rate(y, sp).exp()
                integral = integrate_gap(Lam_fn, dt, scale, self.nodes, self.weights, cfg.quad_tmin)
                log_Lam_end = self.time.log_rate(yhat_end, s_prev)
                comps["time"] = -log_Lam_end + integral
                extra["xi"] = integral
            else:                                                                  # physics
                theta = self._theta(batch, enc)
                def Lam_fn(tau_eff):
                    return self.phys.total(readout_nodes(tau_eff), theta)
                integral = integrate_gap(Lam_fn, dt, scale, self.nodes, self.weights, cfg.quad_tmin)
                rates = self.phys.rates(yhat_end, theta)                           # (B,L,K,2)
                Lam_end = rates.sum((-1, -2))
                lam_mark = rates.gather(2, batch["patch"][..., None, None].expand(-1, -1, 1, 2))[:, :, 0]
                lam_mark = lam_mark.gather(-1, batch["pol"][..., None])[..., 0]
                comps["time"] = -Lam_end.log() + integral
                comps["mark"] = -(lam_mark.log() - Lam_end.log())                  # patch & polarity from physics
                extra["xi"] = integral
                extra["theta"] = theta

        if cfg.objective != "physics":
            comps["patch"] = -self.space.patch_logp(z, batch["patch"])
            comps["pol"] = -self.space.pol_logp(z, batch["patch"], batch["pol"])
        comps["sub"] = -self.space.sub_logp(z, batch["patch"], batch["pol"], batch["sub"])
        if cfg.objective == "ar":
            extra["pit"] = self.time.cdf(s_prev, dt)

        m = mask.float()
        n_ev = m.sum().clamp_min(1.0)
        parts = {k: (v * m).sum() / n_ev for k, v in comps.items()}
        loss = sum(parts.values())
        if cfg.use_clock and cfg.clock_gauge > 0:
            # gauge fixing: r and tau only enter as r * tau * dt, so (r, tau) -> (c r, tau / c) is a symmetry.
            # Pinning the mean log-tempo to 0 makes tau interpretable as a *relative* tempo.
            mean_lt = (enc["log_tau"] * batch["mask"]).sum() / batch["mask"].sum().clamp_min(1)
            parts["clock_gauge"] = cfg.clock_gauge * mean_lt ** 2
            loss = loss + parts["clock_gauge"]
        if cfg.objective == "physics" and cfg.sensor_mode == "infer" and "theta" in batch:
            parts["theta"] = F.mse_loss(extra["theta"], batch["theta"])
            loss = loss + theta_weight * parts["theta"]
        return {"loss": loss, "parts": parts, "per_event": comps, "mask": mask, "extra": extra,
                "state": enc["new_state"], "log_tau": enc["log_tau"]}

    forward = nll

    # ------------------------------------------------------------------ features / consistency
    def features(self, batch):
        enc = self.encode(batch)
        m = batch["mask"][..., None].float()
        n = m.sum(1).clamp_min(1.0)
        mean_s = (enc["s"] * m).sum(1) / n
        mean_lt = (enc["log_tau"][..., None] * m).sum(1) / n
        mean_lr = (-torch.log(batch["dt"].clamp_min(1e-9))[..., None] * m).sum(1) / n
        return torch.cat([mean_s, mean_lt, mean_lr], -1)

    def warp_consistency(self, batch, f: float):
        """Self-supervised tempo loss:  dilate the stream by f  ->  (i) log tau must shift by +ln f
        (the clock must speed up so dt_eff is unchanged), (ii) pooled content features must not change."""
        b2 = dict(batch)
        b2["dt"] = batch["dt"] / f
        e1, e2 = self.encode(batch), self.encode(b2)
        m = batch["mask"].float()
        n = m.sum().clamp_min(1.0)
        tempo = ((((e2["log_tau"] - e1["log_tau"]) - math.log(f)) ** 2) * m).sum() / n
        mm = m[..., None]
        f1 = (e1["s"] * mm).sum(1) / mm.sum(1).clamp_min(1.0)
        f2 = (e2["s"] * mm).sum(1) / mm.sum(1).clamp_min(1.0)
        content = F.mse_loss(f2, f1.detach())
        return {"tempo": tempo, "content": content}

    # ------------------------------------------------------------------ generation (ar only)
    @torch.no_grad()
    def sample(self, n_events: int, B: int = 1, generator: Optional[torch.Generator] = None, quant=None):
        assert self.cfg.objective == "ar", "sampling is implemented for the intensity-free variant"
        cfg = self.cfg
        dev = self.s_init.device
        state = None
        s_prev = self.s_init.expand(B, -1)
        H_prev = torch.zeros(B, cfg.d_int, cfg.d_state, device=dev)
        log_tau = torch.zeros(B, device=dev)
        out = {k: [] for k in ("dt", "patch", "sub", "pol")}
        for _ in range(n_events):
            dt = self.time.sample(s_prev, generator).clamp_min(1e-9)
            dt_eff = dt * log_tau.exp()
            yhat = self._yhat_at(H_prev, dt_eff)
            z = self.space.context(s_prev, yhat, dt)
            patch, pol, sub = self.space.sample(z, generator)
            if quant is not None:
                uu, vv = quant.decode(patch.cpu().numpy(), sub.cpu().numpy())
                u, v = torch.tensor(uu, dtype=torch.float32, device=dev), torch.tensor(vv, dtype=torch.float32, device=dev)
            else:
                u = v = torch.zeros(B, device=dev)
            step = {"patch": patch[:, None], "sub": sub[:, None], "pol": pol[:, None], "u": u[:, None],
                    "v": v[:, None], "dt": dt[:, None], "mask": torch.ones(B, 1, dtype=torch.bool, device=dev)}
            enc = self.encode(step, state)
            state = enc["new_state"]
            s_prev, H_prev, log_tau = state["s"], state["H"], state["log_tau"]
            for k, val in (("dt", dt), ("patch", patch), ("sub", sub), ("pol", pol)):
                out[k].append(val)
        return {k: torch.stack(v, 1) for k, v in out.items()}


# =============================================================================================
class PatchPoisson(nn.Module):
    """V1: fixed-bin Poisson count model over (patch, polarity) cells.  Robust to ties and cheap,
    but timing finer than `bin_width` is not modelled (time is uniform inside a bin).
    NLL is reported as a point-process DENSITY so it is comparable with the other variants:
        nll = sum_c (lam_c - n_c ln lam_c)  +  n ln(bin_width * S)
    (Poisson count likelihood without the ln n_c! terms; uniform time inside the bin, uniform sub-cell)."""

    def __init__(self, cfg: EvPPConfig, bin_width: float = 1e-3):
        super().__init__()
        self.cfg, self.w = cfg, bin_width
        K, D = cfg.n_patches, cfg.d_model
        self.inp = nn.Linear(2 * K, D)
        self.stack = CTSSMStack(cfg)
        self.s_init = nn.Parameter(torch.zeros(D))
        self.out = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 2 * K))
        with torch.no_grad():
            self.out[-1].bias.fill_(math.log(0.2))

    def counts(self, batch):
        t = torch.cumsum(batch["dt"] * batch["mask"], 1)
        T = t.max(1).values
        n_complete = torch.floor(T / self.w).long()                  # only complete bins are scored
        nb = int(n_complete.max().item()) + 1
        b = torch.floor(t / self.w).long()
        cell = batch["patch"] * 2 + batch["pol"]
        K2 = 2 * self.cfg.n_patches
        valid = batch["mask"] & (b < n_complete[:, None])
        counts = torch.zeros(t.shape[0], nb, K2, device=t.device)
        idx = (b.clamp(max=nb - 1) * K2 + cell)
        counts.view(t.shape[0], -1).scatter_add_(1, idx, valid.float())
        bin_valid = torch.arange(nb, device=t.device)[None] < n_complete[:, None]
        return counts, bin_valid, valid

    def nll(self, batch, state=None):
        counts, bin_valid, ev_valid = self.counts(batch)
        B, nb, K2 = counts.shape
        x = self.inp(torch.log1p(counts))
        dt = torch.full((B, nb), self.w, device=counts.device)
        out = self.stack(x, dt, state)
        s_prev = torch.cat([self.s_init.expand(B, -1)[:, None], out["s"][:, :-1]], 1)
        lam = F.softplus(self.out(s_prev)) + 1e-6                          # expected counts per bin
        bv = bin_valid[..., None].float()
        nll_counts = ((lam - counts * lam.log()) * bv).sum((1, 2))
        n = ev_valid.float().sum(1)
        total = nll_counts + n * (math.log(self.w) + math.log(self.cfg.n_sub))
        per_event = total.sum() / n.sum().clamp_min(1.0)
        return {"loss": per_event, "parts": {"total": per_event}, "state": out["state"]}

    forward = nll

    def features(self, batch):
        counts, bin_valid, _ = self.counts(batch)
        B, nb, _ = counts.shape
        out = self.stack(self.inp(torch.log1p(counts)), torch.full((B, nb), self.w, device=counts.device))
        m = bin_valid[..., None].float()
        mean_s = (out["s"] * m).sum(1) / m.sum(1).clamp_min(1.0)
        rate = counts.sum(-1, keepdim=True) / self.w
        mean_lr = (torch.log(rate + 1.0) * m).sum(1) / m.sum(1).clamp_min(1.0)
        return torch.cat([mean_s, torch.zeros_like(mean_lr), mean_lr], -1)


class HomogeneousPoisson:
    """Per-window MLE homogeneous Poisson + uniform marks.  Exactly dilation-equivariant (used in tests
    and as the 'no-structure' reference for bits/event)."""

    def __init__(self, cfg: EvPPConfig):
        self.cfg = cfg

    def nll(self, batch):
        m = batch["mask"].float()
        n = m.sum(1)
        T = (batch["dt"] * m).sum(1)
        lam = n / T
        per_stream = -n * lam.log() + n + n * math.log(2 * self.cfg.n_patches * self.cfg.n_sub)
        return per_stream.sum() / n.sum()
