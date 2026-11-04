"""Protocol skeleton for the central comparison (R1): same backbone, same unlabelled data, different
pretraining objective -> frozen-feature linear probe, tested at the native rate and under rate shifts
along DIFFERENT physical axes.

    python scripts/compare_objectives.py --objectives ar intensity physics poisson mem nextcount \
        --steps 400 --n_pretrain 120 --n_probe_train 48 --n_probe_test 48

This runs on the built-in simulator so it is a *pipeline check*, not evidence: swap
`evpp.training.build_streams` for a loader of real streams (DVS Gesture, N-Caltech101, Gen1 ...)
that returns `evpp.data.stream.Stream` objects to get the real experiment.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from evpp import EvPPConfig, EvPP, PatchPoisson, HomogeneousPoisson
from evpp.data import transforms as T
from evpp.data.stream import Quantizer, collate, tokenize, windows
from evpp.eval.probe import extract, fit_linear_probe
from evpp.models.ssl_baselines import MaskedEventModel, NextCountPrediction
from evpp.training import WindowSampler, build_streams, train

SHIFTS = {
    "native": lambda s, rng: s,
    "speed x2": lambda s, rng: T.dilate(s, 2.0),
    "speed x0.5": lambda s, rng: T.dilate(s, 0.5),
    "thin x0.5": lambda s, rng: T.thin(s, 0.5, rng),
    "densify x2": lambda s, rng: T.densify(s, 2.0, rng),
    "noise +100%": lambda s, rng: T.add_noise(s, 1.0, rng),
}


def make_model(name, cfg):
    if name in ("ar", "intensity", "physics"):
        c = EvPPConfig(**{**cfg.to_dict(), "objective": name})
        return EvPP(c), c
    if name == "poisson":
        return PatchPoisson(cfg, bin_width=1e-3), cfg
    if name == "mem":
        return MaskedEventModel(cfg), cfg
    if name == "nextcount":
        return NextCountPrediction(cfg), cfg
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objectives", nargs="+", default=["ar", "intensity", "physics", "poisson", "mem", "nextcount"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--n_pretrain", type=int, default=64)
    ap.add_argument("--n_probe_train", type=int, default=32)
    ap.add_argument("--n_probe_test", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = EvPPConfig(d_model=a.d_model, d_state=8, n_layers=2, d_int=8, n_quad=6)
    quant = Quantizer(cfg.grid, cfg.sub)
    pre = build_streams(a.n_pretrain, seed=a.seed)
    tr = build_streams(a.n_probe_train, seed=a.seed + 1)
    te = build_streams(a.n_probe_test, seed=a.seed + 2)
    rng = np.random.default_rng(a.seed + 3)
    te_shift = {k: [f(s, rng) for s in te] for k, f in SHIFTS.items()}
    results = {}
    for name in a.objectives:
        print(f"\n=== {name} ===", flush=True)
        model, c = make_model(name, cfg)
        sampler = WindowSampler(pre, c, L=a.L, seed=a.seed, with_theta=(name == "physics" and c.sensor_mode == "oracle"))
        hist = train(model, sampler, steps=a.steps, B=a.B, log_every=max(1, a.steps // 4), seed=a.seed)
        Xtr, ytr, _ = extract(model, tr, quant, L=a.L, n_win=4, seed=a.seed)
        row = {"final_loss": float(np.mean([h["loss"] for h in hist[-10:]]))}
        for k, strs in te_shift.items():
            Xte, yte, _ = extract(model, strs, quant, L=a.L, n_win=4, seed=a.seed + 7)
            row[k] = fit_linear_probe(Xtr, ytr, Xte, yte, n_classes=4)
        results[name] = row
        print({k: round(v, 3) for k, v in row.items()}, flush=True)

    # reference: structure-free likelihood
    b = collate([windows(tokenize(s, quant, cfg.dt_quantum, rng), a.L, rng, 1)[0] for s in te])
    results["_homogeneous_poisson_nll"] = float(HomogeneousPoisson(cfg).nll(b))
    print("\n" + "objective".ljust(12) + "".join(k.rjust(13) for k in SHIFTS))
    for name in a.objectives:
        print(name.ljust(12) + "".join(f"{results[name][k]:13.3f}" for k in SHIFTS))
    if a.out:
        json.dump(results, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
