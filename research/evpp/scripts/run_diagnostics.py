"""Train one likelihood variant on simulated streams, then run the label-free diagnostics:

  * bits/event vs the structure-free homogeneous Poisson baseline
  * time-rescaling calibration (KS on compensator increments / PIT) at native rate and under shifts
  * dilation-equivariance gap
  * temporal information gain (reverse / shuffle / polarity-shuffle controls)

    python scripts/run_diagnostics.py --objective intensity --steps 300 [--warp 1.0] [--dilate_aug 0.5]
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from evpp import EvPPConfig, EvPP, HomogeneousPoisson
from evpp.data import transforms as T
from evpp.data.stream import Quantizer, collate, tokenize, windows
from evpp.eval.diagnostics import dilation_gap, temporal_gain, time_rescaling_report
from evpp.training import WindowSampler, build_streams, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="intensity", choices=["ar", "intensity", "physics"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warp", type=float, default=0.0, help="weight of the tempo/content warp-consistency loss")
    ap.add_argument("--dilate_aug", type=float, default=0.0)
    ap.add_argument("--n_train", type=int, default=64)
    ap.add_argument("--n_test", type=int, default=24)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = EvPPConfig(objective=a.objective, d_model=48, d_state=8, n_layers=2, d_int=8, n_quad=6)
    quant = Quantizer(cfg.grid, cfg.sub)
    tr, te = build_streams(a.n_train, a.seed), build_streams(a.n_test, a.seed + 1)
    model = EvPP(cfg)
    train(model, WindowSampler(tr, cfg, L=a.L, seed=a.seed), steps=a.steps, warp_weight=a.warp,
          dilate_aug=a.dilate_aug, log_every=max(1, a.steps // 5), seed=a.seed)

    rng = np.random.default_rng(a.seed + 5)
    def batches(strs, n=3):
        return [collate([windows(tokenize(s, quant, cfg.dt_quantum, rng), a.L, rng, 1)[0] for s in strs[i::n]])
                for i in range(n)]

    base = HomogeneousPoisson(cfg)
    b0 = batches(te)
    nll = float(np.mean([model(b)["loss"].item() for b in b0]))
    nll0 = float(np.mean([float(base.nll(b)) for b in b0]))
    print(f"\nnats/event: model {nll:.3f} | homogeneous Poisson {nll0:.3f} | gain {(nll0 - nll) / math.log(2):.2f} bits/event")

    print("\ntime-rescaling calibration (KS p-value; small = miscalibrated)")
    shifts = {"native": lambda s: s, "speed x2": lambda s: T.dilate(s, 2.0), "speed x0.5": lambda s: T.dilate(s, 0.5),
              "thin x0.5": lambda s: T.thin(s, 0.5, rng), "densify x2": lambda s: T.densify(s, 2.0, rng)}
    for k, f in shifts.items():
        r = time_rescaling_report(model, batches([f(s) for s in te]))
        print(f"  {k:12s} {r['kind']:11s} mean={r['mean']:.3f}  KS={r['ks_stat']:.3f}  p={r['ks_p']:.3g}  (n={r['n']})")

    print("\ndilation-equivariance gap (nats/event, 0 = exactly equivariant)")
    for f in (0.5, 2.0, 4.0):
        print(f"  f={f:<4}  {np.mean([dilation_gap(model, b, f) for b in b0]):+.3f}")

    print("\ntemporal information gain")
    tg = temporal_gain(model, te, quant, L=a.L)
    for k, v in tg.items():
        print(f"  {k:22s} {v:+.3f}")


if __name__ == "__main__":
    main()
