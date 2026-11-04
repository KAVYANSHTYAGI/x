"""Label-free diagnostics for an event likelihood model.

R4  time-rescaling calibration
    If the model's temporal law is right, the compensator increments between consecutive events,
        xi_i = int_{t_{i-1}}^{t_i} Lambda(t) dt,
    are i.i.d. Exp(1) ("time-rescaling theorem"); equivalently for the intensity-free head the PIT
    values F(dt_i) are i.i.d. U(0,1).  KS tests on these give a calibration check that needs no labels
    and can be computed under any rate shift.

R6  temporal information gain
    bits/event lost when the model is fed a stream whose time structure is destroyed (reverse,
    shuffle) relative to the real one.  A dataset/model pair where this is ~0 does not need time.

    dilation gap
    If log p were exactly equivariant to speed changes, log p(dilate(x,f)) = log p(x) + n ln f.
    The per-event residual measures how much a model's likelihood depends on the absolute time scale.
"""
import math
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch

from ..data import transforms as T
from ..data.stream import Quantizer, collate, tokenize, windows


# --------------------------------------------------------------------------- KS tests
def _ks_pvalue(d: float, n: int) -> float:
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    if lam < 1e-3:
        return 1.0
    return float(min(1.0, max(0.0, 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101)))))


def ks_uniform(u: np.ndarray):
    u = np.sort(np.asarray(u, dtype=np.float64))
    n = len(u)
    cdf = (np.arange(1, n + 1)) / n
    d = max(np.max(cdf - u), np.max(u - (cdf - 1.0 / n)))
    return float(d), _ks_pvalue(d, n)


def ks_exp1(xi: np.ndarray):
    return ks_uniform(1.0 - np.exp(-np.asarray(xi, dtype=np.float64)))


# --------------------------------------------------------------------------- calibration
@torch.no_grad()
def time_rescaling_report(model, batches: Sequence[Dict[str, torch.Tensor]]):
    """Collect xi (intensity / physics) or PIT (ar) over batches; return KS stats and QQ points."""
    model.eval()
    vals = []
    kind = None
    for b in batches:
        o = model(b)
        m = o["mask"].cpu().numpy().astype(bool)
        if "xi" in o["extra"]:
            kind, v = "xi~Exp(1)", o["extra"]["xi"].cpu().numpy()
        elif "pit" in o["extra"]:
            kind, v = "pit~U(0,1)", o["extra"]["pit"].cpu().numpy()
        else:
            raise ValueError("model exposes no time-rescaling statistic")
        vals.append(v[m])
    v = np.concatenate(vals)
    d, p = ks_exp1(v) if kind.startswith("xi") else ks_uniform(v)
    q = np.linspace(0.05, 0.95, 10)
    return {"kind": kind, "n": len(v), "ks_stat": d, "ks_p": p,
            "mean": float(v.mean()),                       # 1.0 for xi, 0.5 for PIT when calibrated
            "quantiles": np.quantile(v, q).tolist()}


# --------------------------------------------------------------------------- dilation equivariance
@torch.no_grad()
def dilation_gap(model, batch: Dict[str, torch.Tensor], f: float):
    """Per-event residual  (nll(dilate(x,f)) - nll(x)) + ln f   in nats.  0 <=> exactly equivariant.

    Convention: dilate(x, f) divides every gap by f (the stream runs f times faster).  The density of the
    dilated gaps is f^n p(x), so per event  NLL(dilate(x,f))/n = NLL(x)/n - ln f  for an equivariant model.
    `nll` here is the per-event mean.  The dilation is applied to the (dequantised) gaps, so no
    re-quantisation noise enters.  Works for nn.Modules and for plain baselines with an .nll(batch).
    """
    if hasattr(model, "eval"):
        model.eval()
    b2 = dict(batch)
    b2["dt"] = batch["dt"] / f
    def loss(b):
        r = model.nll(b)
        return float(r["loss"]) if isinstance(r, dict) else float(r)
    return (loss(b2) - loss(batch)) + math.log(f)


# --------------------------------------------------------------------------- temporal information gain
PERTURBATIONS: Dict[str, Callable] = {
    "reverse": lambda s, rng: T.time_reverse(s),
    "shuffle": lambda s, rng: T.time_shuffle(s, rng),
    "polshuffle": lambda s, rng: T.polarity_shuffle(s, rng),
}


@torch.no_grad()
def temporal_gain(model, streams: List, quant: Quantizer, L: int = 128, quantum: float = 1e-6,
                  seed: int = 0, perturbations=("reverse", "shuffle", "polshuffle")):
    """Mean NLL (nats/event) on real vs perturbed streams.  'gain' = perturbed - real (>0: time structure used)."""
    model.eval()
    out = {}
    def score(strs, tag):
        rng = np.random.default_rng(seed)                 # same window positions & dequantisation noise
        items = []
        for s in strs:
            tok = tokenize(s, quant, quantum, rng)
            items.append(windows(tok, L, rng, 1)[0])
        return float(model(collate(items))["loss"])
    real = score(streams, "real")
    out["real_nll"] = real
    for name in perturbations:
        rng = np.random.default_rng(seed + 1)
        pert = [PERTURBATIONS[name](s, rng) for s in streams]
        out[f"{name}_nll"] = score(pert, name)
        out[f"{name}_gain_bits"] = (out[f"{name}_nll"] - real) / math.log(2)
    return out
