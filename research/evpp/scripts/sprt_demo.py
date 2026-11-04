"""R5: class-conditional likelihood model + sequential stopping on simulated streams.

    python scripts/sprt_demo.py --steps 400
Prints accuracy and the number of events needed at confidence 1-alpha, split by event rate tercile
(the claim to test on real data: stopping time scales with the information arrival rate).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from evpp import EvPPConfig, EvPP
from evpp.data.stream import Quantizer, collate, tokenize, windows
from evpp.eval.sprt import anytime_log_posterior, stopping_index
from evpp.training import WindowSampler, build_streams, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--L", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    C = 4
    cfg = EvPPConfig(objective="ar", cond_dim=C, d_model=48, d_state=8, n_layers=2, d_int=8)
    quant = Quantizer(cfg.grid, cfg.sub)
    tr, te = build_streams(64, a.seed), build_streams(32, a.seed + 1)
    model = EvPP(cfg)
    train(model, WindowSampler(tr, cfg, L=a.L, seed=a.seed), steps=a.steps, class_cond=C, log_every=max(1, a.steps // 4))

    rng = np.random.default_rng(a.seed + 9)
    items = []
    for s in te:
        w = windows(tokenize(s, quant, cfg.dt_quantum, rng), a.L, rng, 1)[0]
        w["label"] = s.label
        items.append(w)
    b = collate(items)
    lp = anytime_log_posterior(model, b, C)
    idx, pred = stopping_index(lp, b["mask"], a.alpha)
    acc = (pred == b["label"]).float().mean().item()
    t = torch.cumsum(b["dt"], 1)
    t_stop = t[torch.arange(len(idx)), idx]
    rate = (b["mask"].sum(1) / t[:, -1]).numpy()
    order = np.argsort(rate)
    print(f"\naccuracy at stop = {acc:.3f}   median events = {idx.float().median():.0f}   median time = {t_stop.median() * 1e3:.2f} ms")
    for name, sl in zip(("low-rate", "mid-rate", "high-rate"), np.array_split(order, 3)):
        print(f"  {name:9s} events={idx[sl].float().mean():6.1f}  time={t_stop[sl].mean() * 1e3:6.2f} ms")


if __name__ == "__main__":
    main()
