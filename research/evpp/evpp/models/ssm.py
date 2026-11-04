"""Event-driven continuous-time state-space backbone.

Each layer is a bank of diagonal linear systems driven by *impulses at event times*:

    h(t_i^+) = exp(-rate * tau_i * dt_i) * h(t_{i-1}^+) + B * v_i        (rate > 0, per channel/state)
    y_i      = C . h(t_i^+) + skip * v_i

so between events the state decays exactly (no discretisation error, no window), and a time-scale
change of the stream is equivalent to changing dt_i.  `tau_i` is the LATENT CLOCK: a causal,
learned, strictly positive tempo (tau after event i-1 scales the gap before event i), so the same
learned dynamics can run "faster" or "slower" per stream (cf. Tallec & Ollivier, gated time-warping).

Causality contract (tested): everything used to predict event i depends only on events < i
(plus dt_i, which is the teacher-forced gap that the head conditions on).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scan import scan


class CTSSMLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int, tmin: float, tmax: float):
        super().__init__()
        D, N = d_model, d_state
        # log-spaced timescales between tmin and tmax (rates are 1/timescale)
        rates = torch.exp(torch.linspace(math.log(1.0 / tmax), math.log(1.0 / tmin), N))
        self.log_rate = nn.Parameter(rates.log()[None, :].repeat(D, 1) + 0.1 * torch.randn(D, N))
        self.B = nn.Parameter(torch.randn(D, N) / math.sqrt(N))
        self.C = nn.Parameter(torch.randn(D, N) / math.sqrt(N))
        self.skip = nn.Parameter(torch.ones(D))
        self.in_proj = nn.Linear(D, 2 * D)
        self.out_proj = nn.Linear(D, D)
        self.norm1 = nn.LayerNorm(D)
        self.ff = nn.Sequential(nn.Linear(D, 2 * D), nn.GELU(), nn.Linear(2 * D, D))
        self.norm2 = nn.LayerNorm(D)

    def rate(self):
        return self.log_rate.exp()                                    # (D, N)

    def forward(self, u, dt_eff, h0=None, impl="assoc", n_save=0):
        """u: (B,L,D); dt_eff: (B,L) elapsed (clock-scaled) time before each event; h0: (B,D,N).

        Returns (x, H_save, h_last): H_save (B,L,n_save,N) = per-event states of the first n_save channels
        (None if n_save == 0), h_last (B,D,N).  impl="evkit" runs the fused C++/CUDA kernel, which never
        materialises the (B,L,D,N) state tensor.
        """
        v, g = self.in_proj(u).chunk(2, dim=-1)
        if impl == "evkit":
            from evkit import decay_scan
            y, H_save, h_last = decay_scan(v, dt_eff, self.rate(), self.B, self.C, self.skip, h0, n_save)
            H_save = H_save if n_save else None
        else:
            decay = torch.exp(-self.rate()[None, None] * dt_eff[..., None, None])   # (B,L,D,N) in (0,1]
            drive = self.B[None, None] * v[..., None]
            h = scan(decay, drive, h0, impl)                                         # (B,L,D,N)
            y = (h * self.C[None, None]).sum(-1) + self.skip * v
            H_save, h_last = (h[:, :, :n_save] if n_save else None), h[:, -1]
        y = self.out_proj(F.gelu(y) * torch.sigmoid(g))
        x = self.norm1(u + y)
        x = self.norm2(x + self.ff(x))
        return x, H_save, h_last

    def readout(self, h, dt_eff, n: int):
        """Linear readout of the first `n` channels of a (decayed) state:  C . (exp(-rate*dt) * h).

        h: (..., n, N); dt_eff broadcastable to h.shape[:-2].  Returns (..., n).  This is the only
        quantity that continues to evolve between events, which is what makes the intensity a
        function of continuous time.
        """
        r = self.rate()[:n]
        return (h * torch.exp(-r * dt_eff[..., None, None]) * self.C[:n]).sum(-1)


class LatentClock(nn.Module):
    """Causal tempo estimator.  tau_i = exp(c_i) > 0 is the clock speed *after* event i.

    c_i = w . EMA_i(tanh(W u_j)) with a count-based EMA (so it does not depend on the time axis),
    and w zero-initialised -> tau == 1 at init (the model starts as a fixed-clock SSM).
    """

    def __init__(self, d_model: int, clock_dim: int, beta: float):
        super().__init__()
        self.inp = nn.Linear(d_model, clock_dim)
        self.out = nn.Linear(clock_dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.beta = beta

    def forward(self, u, state=None, impl="assoc"):
        a = torch.full_like(u[..., :1], self.beta).expand(-1, -1, self.inp.out_features)
        b = (1.0 - self.beta) * torch.tanh(self.inp(u))
        e = scan(a, b, state, impl)                                  # (B,L,clock_dim)
        log_tau = self.out(e)[..., 0].clamp(-4.0, 4.0)
        return log_tau, e[:, -1]


class CTSSMStack(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [CTSSMLayer(cfg.d_model, cfg.d_state, cfg.tmin, cfg.tmax) for _ in range(cfg.n_layers)])
        self.clock = LatentClock(cfg.d_model, cfg.clock_dim, cfg.clock_beta) if cfg.use_clock else None

    def init_state(self, B, device):
        c = self.cfg
        return {
            "h": [torch.zeros(B, c.d_model, c.d_state, device=device) for _ in self.layers],
            "clock": torch.zeros(B, c.clock_dim, device=device),
            "log_tau": torch.zeros(B, device=device),
        }

    def forward(self, u, dt, state=None, impl="assoc"):
        """u: (B,L,D) token embeddings, dt: (B,L).  Returns dict with last-layer features and states."""
        B = u.shape[0]
        st = state if state is not None else self.init_state(B, u.device)
        if self.clock is not None:
            log_tau, clock_state = self.clock(u, st["clock"], impl)
        else:
            log_tau, clock_state = torch.zeros_like(dt), st["clock"]
        # clock after event i-1 scales the gap before event i  (history only -> no leakage)
        log_tau_prev = torch.cat([st["log_tau"][:, None], log_tau[:, :-1]], dim=1)
        dt_eff = dt * log_tau_prev.exp()
        x, new_h = u, []
        layer_impl = "evkit" if getattr(self.cfg, "impl", "torch") == "evkit" else impl
        n_last = getattr(self.cfg, "d_int", 0)
        for k, (layer, h0) in enumerate(zip(self.layers, st["h"])):
            n_save = n_last if k == len(self.layers) - 1 else 0
            x, H_save, h_last = layer(x, dt_eff, h0, layer_impl, n_save)
            new_h.append(h_last)
        new_state = {"h": new_h, "clock": clock_state, "log_tau": log_tau[:, -1]}
        return {"s": x, "H_save": H_save, "log_tau": log_tau, "log_tau_prev": log_tau_prev,
                "dt_eff": dt_eff, "state": new_state}
