import numpy as np
import pytest
import torch

from conftest import tiny_cfg
from evpp import EvPP, PatchPoisson, HomogeneousPoisson
from evpp.training import build_streams, WindowSampler, train
from evpp.eval.diagnostics import ks_exp1, ks_uniform, time_rescaling_report, dilation_gap, temporal_gain
from evpp.data.stream import Quantizer


@pytest.fixture(scope="module")
def train_streams():
    return build_streams(12, seed=1)


@pytest.mark.parametrize("obj,mode", [("ar", "learned"), ("intensity", "learned"),
                                      ("physics", "oracle"), ("physics", "infer")])
def test_training_reduces_loss(train_streams, obj, mode):
    cfg = tiny_cfg(objective=obj, sensor_mode=mode, sensor_ctx=8)
    sampler = WindowSampler(train_streams, cfg, L=48, seed=0, with_theta=(mode != "learned"))
    model = EvPP(cfg)
    hist = train(model, sampler, steps=60, B=6, lr=3e-3, verbose=False)
    first = np.mean([h["loss"] for h in hist[:5]])
    last = np.mean([h["loss"] for h in hist[-5:]])
    assert np.isfinite(last) and last < first - 0.3, (first, last)


def test_warp_consistency_trains_tempo(train_streams):
    cfg = tiny_cfg(objective="ar")
    sampler = WindowSampler(train_streams, cfg, L=48, seed=0)
    model = EvPP(cfg)
    b = sampler.batch(6)
    before = model.warp_consistency(b, 2.0)["tempo"].item()
    # tempo objective in isolation (the clock's zero-initialised head needs a few hundred steps to bootstrap)
    train(model, sampler, steps=300, B=6, lr=1e-2, warp_weight=1.0, nll_weight=0.0, verbose=False)
    after = model.warp_consistency(b, 2.0)["tempo"].item()
    assert after < 0.5 * before, (before, after)   # smoke test: the signal flows and is learnable


def test_ks_detects_wrong_intensity():
    """Validates the diagnostic itself with a process whose intensity is known exactly."""
    rng = np.random.default_rng(0)
    lam = lambda t: 50.0 * (1.5 + np.sin(2 * np.pi * 3 * t))          # inhomogeneous Poisson
    Lam = lambda t: 50.0 * (1.5 * t - (np.cos(2 * np.pi * 3 * t) - 1) / (2 * np.pi * 3))
    T, lmax = 200.0, 125.0
    cand = np.cumsum(rng.exponential(1 / lmax, int(T * lmax * 1.2)))
    cand = cand[cand < T]
    ev = cand[rng.random(len(cand)) < lam(cand) / lmax]               # thinning -> exact sample
    xi_true = np.diff(Lam(ev))
    assert ks_exp1(xi_true)[1] > 0.01
    assert ks_exp1(2.0 * xi_true)[1] < 1e-6                           # a model with 2x the intensity is flagged
    assert ks_uniform(rng.random(5000))[1] > 0.01
    assert ks_uniform(rng.random(5000) ** 2)[1] < 1e-6


def test_diagnostics_run_end_to_end(train_streams):
    cfg = tiny_cfg(objective="intensity")
    sampler = WindowSampler(train_streams, cfg, L=48, seed=0)
    model = EvPP(cfg)
    train(model, sampler, steps=30, B=6, verbose=False)
    batches = [sampler.batch(6) for _ in range(3)]
    rep = time_rescaling_report(model, batches)
    assert rep["n"] > 0 and 0.0 <= rep["ks_p"] <= 1.0
    g = dilation_gap(model, batches[0], 2.0)
    assert np.isfinite(g)
    tg = temporal_gain(model, train_streams[:6], Quantizer(cfg.grid, cfg.sub), L=48)
    assert "reverse_gain_bits" in tg and np.isfinite(tg["shuffle_gain_bits"])
    # AR variant exposes PIT instead
    ar = EvPP(tiny_cfg(objective="ar"))
    assert time_rescaling_report(ar, batches)["kind"].startswith("pit")


def test_ssl_baselines_train_and_probe_shapes(train_streams):
    from evpp.models.ssl_baselines import MaskedEventModel, NextCountPrediction
    cfg = tiny_cfg()
    sampler = WindowSampler(train_streams, cfg, L=48, seed=0)
    for M in (MaskedEventModel, NextCountPrediction, lambda c: PatchPoisson(c, 2e-4)):
        model = M(cfg)
        hist = train(model, sampler, steps=40, B=6, lr=3e-3, verbose=False)
        assert np.isfinite(hist[-1]["loss"]) and np.mean([h["loss"] for h in hist[-5:]]) < np.mean([h["loss"] for h in hist[:5]])
        f = model.features(sampler.batch(6))
        assert f.shape[0] == 6 and torch.isfinite(f).all()
