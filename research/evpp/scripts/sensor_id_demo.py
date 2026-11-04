"""R2: in-context sensor identification with the physics-structured intensity.

Pretrain on simulated sensors with randomised (contrast threshold C, noise rate), supervise the
SensorEncoder's theta_hat = (ln C, ln r_n) with the known simulator values (aux loss), then read the
sensor parameters of unseen streams from their first `sensor_ctx` events - no gradient steps.

    python scripts/sensor_id_demo.py --steps 600

IMPORTANT caveat built into the design: without the supervised aux loss, ln C is only identified up to
the scale of the learned brightness-rate s_k(t); real-data use needs anchoring (known lux, frames,
or simulator pretraining followed by checks on paired recordings such as N-ImageNet variants).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from evpp import EvPPConfig, EvPP
from evpp.data.stream import Quantizer, collate, tokenize, windows
from evpp.training import WindowSampler, build_streams, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = EvPPConfig(objective="physics", sensor_mode="infer", sensor_ctx=48, d_model=48, d_state=8,
                     n_layers=2, d_int=8, n_quad=6)
    quant = Quantizer(cfg.grid, cfg.sub)
    tr, te = build_streams(96, a.seed), build_streams(48, a.seed + 1)
    model = EvPP(cfg)
    train(model, WindowSampler(tr, cfg, L=160, seed=a.seed, with_theta=True), steps=a.steps, theta_weight=5.0,
          log_every=max(1, a.steps // 5))

    rng = np.random.default_rng(a.seed + 4)
    items = []
    for s in te:
        w = windows(tokenize(s, quant, cfg.dt_quantum, rng), 160, rng, 1)[0]
        w["theta"] = s.meta["theta"]
        items.append(w)
    b = collate(items)
    model.eval()
    with torch.no_grad():
        theta_hat = model(b)["extra"]["theta"].numpy()
    truth = b["theta"].numpy()
    for j, name in enumerate(("ln C", "ln r_noise_total")):
        r = np.corrcoef(theta_hat[:, j], truth[:, j])[0, 1]
        mae = np.abs(theta_hat[:, j] - truth[:, j]).mean()
        base = np.abs(truth[:, j] - truth[:, j].mean()).mean()
        print(f"{name:18s} corr={r:+.3f}  MAE={mae:.3f}  (predict-the-mean MAE={base:.3f})")


if __name__ == "__main__":
    main()
