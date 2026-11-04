"""Minimal pretraining loop on simulated streams (CPU-friendly).  Used by scripts/ and the smoke tests."""
import math
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import EvPPConfig
from .data.sim import sample_params, simulate
from .data.stream import Quantizer, Stream, collate, tokenize, windows


def build_streams(n: int, seed: int = 0, randomize: bool = True, n_classes: int = 4, **sim_over) -> List[Stream]:
    rng = np.random.default_rng(seed)
    return [simulate(i % n_classes, sample_params(rng, randomize, **sim_over), rng) for i in range(n)]


class WindowSampler:
    """Pre-tokenises streams once (fixed dequantisation noise) and serves random windows."""

    def __init__(self, streams: List[Stream], cfg: EvPPConfig, L: int = 128, seed: int = 0, with_theta=False):
        self.rng = np.random.default_rng(seed)
        q = Quantizer(cfg.grid, cfg.sub)
        self.tokens = []
        for s in streams:
            t = tokenize(s, q, cfg.dt_quantum, self.rng)
            if with_theta:
                t["theta"] = s.meta["theta"]
            t["label"] = s.label
            self.tokens.append(t)
        self.L = L

    def batch(self, B: int):
        idx = self.rng.integers(0, len(self.tokens), B)
        items = []
        for i in idx:
            tok = self.tokens[i]
            w = windows({k: v for k, v in tok.items() if isinstance(v, np.ndarray)}, self.L, self.rng, 1)[0]
            if "theta" in tok:
                w["theta"] = tok["theta"]
            w["label"] = tok["label"]
            items.append(w)
        return collate(items)


def train(model, sampler: WindowSampler, steps: int, B: int = 8, lr: float = 2e-3, warp_weight: float = 0.0,
          dilate_aug: float = 0.0, class_cond: Optional[int] = None, theta_weight: float = 1.0,
          nll_weight: float = 1.0,
          log_every: int = 50, seed: int = 0, verbose: bool = True) -> List[Dict[str, float]]:
    """Returns the training history (list of dicts of per-part NLL, nats/event)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    hist, t0 = [], time.time()
    for step in range(steps):
        b = sampler.batch(B)
        if class_cond:
            b["cond"] = torch.nn.functional.one_hot(b["label"], class_cond).float()
        if dilate_aug > 0 and rng.random() < dilate_aug:
            f = float(math.exp(rng.uniform(math.log(0.5), math.log(2.0))))
            b["dt"] = b["dt"] / f                                   # speed augmentation on the gaps
        model.train()
        out = model(b, theta_weight=theta_weight) if hasattr(model, "cfg") and model.cfg.objective == "physics" else model(b)
        loss = nll_weight * out["loss"]
        if warp_weight > 0:
            f = float(math.exp(rng.uniform(math.log(0.5), math.log(2.0))))
            wc = model.warp_consistency(b, f)
            loss = loss + warp_weight * (wc["tempo"] + wc["content"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        rec = {k: float(v.detach()) for k, v in out["parts"].items()}
        rec["loss"] = float(out["loss"].detach())
        hist.append(rec)
        if verbose and (step % log_every == 0 or step == steps - 1):
            parts = " ".join(f"{k}={v:6.3f}" for k, v in rec.items())
            print(f"[{step:4d}/{steps}] {parts}  ({time.time() - t0:.0f}s)", flush=True)
    return hist
