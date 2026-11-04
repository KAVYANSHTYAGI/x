import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from evpp import EvPPConfig
from evpp.data.sim import simulate, sample_params
from evpp.data.stream import Quantizer, tokenize, collate, windows

torch.manual_seed(0)


def tiny_cfg(**kw):
    base = dict(grid=4, sub=2, d_model=16, d_state=4, n_layers=2, d_int=4, n_mix=3, n_quad=6,
                clock_dim=4, tmin=1e-4, tmax=0.5)
    base.update(kw)
    return EvPPConfig(**base)


@pytest.fixture(scope="session")
def streams():
    rng = np.random.default_rng(0)
    return [simulate(c % 4, sample_params(rng, True), rng) for c in range(8)]


def make_batch(streams, cfg, L=48, n=4, seed=0, with_theta=False):
    rng = np.random.default_rng(seed)
    q = Quantizer(cfg.grid, cfg.sub)
    items = []
    for i in range(n):
        s = streams[i % len(streams)]
        tok = tokenize(s, q, cfg.dt_quantum, rng)
        w = windows(tok, L, rng, 1)[0]
        if with_theta:
            w["theta"] = s.meta["theta"]
        w["label"] = s.label
        items.append(w)
    return collate(items)
