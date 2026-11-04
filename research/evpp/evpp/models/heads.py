"""Likelihood heads.

Event i is predicted from the history state (events < i) and the teacher-forced gap dt_i:

    p(e_i | H) = p(dt_i | H)  *  p(patch_i | z_i)  *  p(pol_i | patch_i, z_i)  *  p(sub_i | patch_i, pol_i, z_i)

with z_i = MLP[ s_{i-1},  yhat_{i-1}(t_i),  features(log dt_i) ]  where yhat_{i-1}(t_i) is the linear
readout of the last layer's state *evolved to the arrival time* - the continuous-time part.

Time heads (swap them to change the objective):
    LogNormalMix     intensity-free; exact density of dt_i, no integral needed                 ("ar")
    IntensityTime    ground intensity Lambda(t) from the decaying state; compensator by
                     Gauss-Legendre quadrature in LOG time (stable for heavy-tailed gaps)      ("intensity")
    PhysicsIntensity per-patch intensities  lambda_{k,p}(t) = softplus(+-s_k(t))/C + r_n/(2K)  ("physics")
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2 * math.pi)


def dt_features(dt):
    lg = torch.log(dt.clamp_min(1e-9))
    return torch.stack([(lg + 9.0) / 4.0, torch.tanh((lg + 9.0) / 4.0)], dim=-1)   # (...,2)


# ---------------------------------------------------------------------------------------------
class LogNormalMix(nn.Module):
    """p(dt) = sum_k w_k LogNormal(dt; mu_k, sigma_k).   NLL is w.r.t. Lebesgue measure on dt (seconds)."""

    def __init__(self, d_model, n_mix, init_log_dt):
        super().__init__()
        self.n_mix = n_mix
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 3 * n_mix))
        with torch.no_grad():
            last = self.net[-1]
            last.weight.mul_(0.1)
            last.bias.zero_()
            last.bias[n_mix:2 * n_mix] = init_log_dt + torch.linspace(-2.0, 2.0, n_mix)   # spread means
            last.bias[2 * n_mix:] = 0.0                                                    # sigma ~ 1

    def params(self, s):
        logw, mu, log_sig = self.net(s).chunk(3, dim=-1)
        return F.log_softmax(logw, -1), mu, log_sig.clamp(-4.0, 2.0)

    def nll(self, s, dt):
        logw, mu, log_sig = self.params(s)
        x = torch.log(dt.clamp_min(1e-12))[..., None]
        lp = -0.5 * ((x - mu) / log_sig.exp()) ** 2 - log_sig - 0.5 * LOG2PI
        return -(torch.logsumexp(logw + lp, -1) - x[..., 0])       # density over dt = density over log dt / dt

    def cdf(self, s, dt):
        """F(dt) under the model; PIT values  u_i = F(dt_i) ~ U(0,1) iff the time model is calibrated."""
        logw, mu, log_sig = self.params(s)
        x = torch.log(dt.clamp_min(1e-12))[..., None]
        z = (x - mu) / log_sig.exp()
        return (logw.exp() * 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))).sum(-1)

    @torch.no_grad()
    def sample(self, s, generator=None):
        logw, mu, log_sig = self.params(s)
        k = torch.multinomial(logw.exp(), 1, generator=generator)
        m, sg = mu.gather(-1, k), log_sig.gather(-1, k).exp()
        return torch.exp(m + sg * torch.randn(m.shape, generator=generator))[..., 0]


# ---------------------------------------------------------------------------------------------
def gauss_legendre_01(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return torch.tensor((x + 1.0) / 2.0, dtype=torch.float32), torch.tensor(w / 2.0, dtype=torch.float32)


def integrate_gap(Lam_fn, dt, scale, nodes, weights, tmin):
    """int_0^dt Lambda(tau) dtau for every gap, in PHYSICAL time.

    Lam_fn(tau_eff) maps (B,L,S) effective times since the previous event to (B,L,S) intensities
    (tau_eff = scale * tau; `scale` is the latent-clock speed in force during the gap).
    Split at lo = min(tmin, dt): [0, lo] by the midpoint rule, [lo, dt] by Gauss-Legendre in log tau
    (a gap can span 6 decades; log-spaced nodes keep the quadrature accurate for both the sharp
    post-event transient and the long tail).  Deterministic and differentiable.
    """
    lo = dt.clamp(max=tmin)
    seg1 = lo * Lam_fn((scale * lo * 0.5)[..., None])[..., 0]
    span = torch.log(dt) - torch.log(lo)                                     # >= 0, 0 when dt <= tmin
    tau = torch.exp(torch.log(lo)[..., None] + span[..., None] * nodes)      # (B,L,S)
    Lam = Lam_fn(scale[..., None] * tau)
    return seg1 + span * (Lam * tau * weights).sum(-1)


class IntensityTime(nn.Module):
    """Ground intensity  Lambda(t) = exp( g([yhat(t); s]) )  where yhat(t) decays between events."""

    def __init__(self, d_model, d_int, init_log_rate):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_int + d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1)
            self.net[-1].bias.fill_(init_log_rate)

    def log_rate(self, yhat, s):
        return self.net(torch.cat([yhat, s], -1))[..., 0].clamp(-10.0, 25.0)


# ---------------------------------------------------------------------------------------------
class SpaceHead(nn.Module):
    """Marks:  patch -> polarity -> sub-cell, each a categorical conditioned on z and the previous marks."""

    def __init__(self, d_model, d_int, n_patches, n_sub, cond_dim=0):
        super().__init__()
        self.K, self.S = n_patches, n_sub
        self.ctx = nn.Sequential(nn.Linear(d_model + d_int + 2, d_model), nn.GELU())
        self.patch = nn.Linear(d_model, n_patches)
        self.pol = nn.Linear(d_model, 2)
        self.sub = nn.Linear(d_model, n_sub)
        self.patch_emb = nn.Embedding(n_patches, d_model)
        self.pol_emb = nn.Embedding(2, d_model)

    def context(self, s_prev, yhat, dt):
        return self.ctx(torch.cat([s_prev, yhat, dt_features(dt)], -1))

    def patch_logp(self, z, patch):
        return F.log_softmax(self.patch(z), -1).gather(-1, patch[..., None])[..., 0]

    def pol_logp(self, z, patch, pol):
        zp = z + self.patch_emb(patch)
        return F.log_softmax(self.pol(zp), -1).gather(-1, pol[..., None])[..., 0]

    def sub_logp(self, z, patch, pol, sub):
        zpp = z + self.patch_emb(patch) + self.pol_emb(pol)
        return F.log_softmax(self.sub(zpp), -1).gather(-1, sub[..., None])[..., 0]

    @torch.no_grad()
    def log_prob_all(self, z):
        """(K,2,S) tensor of log p(patch, pol, sub | z) for ONE context vector z (D,) - used to test normalisation."""
        K, S = self.K, self.S
        lp_patch = F.log_softmax(self.patch(z), -1)                                   # (K,)
        zp = z[None] + self.patch_emb.weight                                          # (K,D)
        lp_pol = F.log_softmax(self.pol(zp), -1)                                      # (K,2)
        zpp = zp[:, None] + self.pol_emb.weight[None]                                 # (K,2,D)
        lp_sub = F.log_softmax(self.sub(zpp), -1)                                     # (K,2,S)
        return lp_patch[:, None, None] + lp_pol[:, :, None] + lp_sub

    @torch.no_grad()
    def sample(self, z, generator=None):
        patch = torch.multinomial(F.softmax(self.patch(z), -1), 1, generator=generator)[..., 0]
        zp = z + self.patch_emb(patch)
        pol = torch.multinomial(F.softmax(self.pol(zp), -1), 1, generator=generator)[..., 0]
        zpp = zp + self.pol_emb(pol)
        sub = torch.multinomial(F.softmax(self.sub(zpp), -1), 1, generator=generator)[..., 0]
        return patch, pol, sub


# ---------------------------------------------------------------------------------------------
class PhysicsIntensity(nn.Module):
    """Intensities from the event-generation model.

    An ideal pixel fires when log-brightness changes by C, so the expected rate of ON / OFF events in
    patch k is  softplus(+-s_k) / C  with s_k = d(log I)/dt aggregated over the patch (learned from
    the readout), plus a background rate  r_n / (2K)  per (patch, polarity).

    theta = (ln C, ln r_n) are SENSOR parameters.  Caveat (documented in the notes): without
    supervision on simulated sensors, C is only identified up to the scale of s_k.
    """

    def __init__(self, d_model, d_int, n_patches):
        super().__init__()
        self.K = n_patches
        self.to_s = nn.Sequential(nn.Linear(d_int, d_model), nn.GELU(), nn.Linear(d_model, n_patches))
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.default_theta = nn.Parameter(torch.tensor([math.log(0.3), math.log(50.0)]))   # learned fallback

    def rates(self, yhat, theta):
        """yhat (...,d_int); theta (B,2) broadcast over the middle dims.  Returns (...,K,2) [OFF, ON] rates (Hz)."""
        s = self.to_s(yhat) * self.log_scale.exp()
        shape = [theta.shape[0]] + [1] * (s.dim() - 2) + [1]
        C = theta[:, 0].exp().view(shape)                              # (B,1,...,1)
        rn = theta[:, 1].exp().view(shape)
        off, on = F.softplus(-s) / C, F.softplus(s) / C
        bg = rn / (2.0 * self.K)
        return torch.stack([off + bg, on + bg], dim=-1)

    def total(self, yhat, theta):
        return self.rates(yhat, theta).sum((-1, -2))
