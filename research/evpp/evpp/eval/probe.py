"""Frozen-backbone evaluation: features -> linear probe.  The point of pretraining is that this works."""
from typing import List

import numpy as np
import torch
import torch.nn as nn

from ..data.stream import Quantizer, collate, tokenize, windows


@torch.no_grad()
def extract(model, streams: List, quant: Quantizer, L: int = 128, n_win: int = 4, quantum: float = 1e-6, seed: int = 0):
    """Features of n_win random windows per stream; returns (X, y, stream_id)."""
    model.eval()
    rng = np.random.default_rng(seed)
    X, Y, I = [], [], []
    for i, s in enumerate(streams):
        tok = tokenize(s, quant, quantum, rng)
        items = windows(tok, L, rng, n_win)
        X.append(model.features(collate(items)).cpu())
        Y += [s.label] * len(items)
        I += [i] * len(items)
    return torch.cat(X), torch.tensor(Y), torch.tensor(I)


def fit_linear_probe(Xtr, ytr, Xte, yte, n_classes, steps=300, lr=0.05, wd=1e-3, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    f = lambda X: (X - mu) / sd
    clf = nn.Linear(Xtr.shape[1], n_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.cross_entropy(clf(f(Xtr)), ytr).backward()
        opt.step()
    with torch.no_grad():
        return (clf(f(Xte)).argmax(1) == yte).float().mean().item()
