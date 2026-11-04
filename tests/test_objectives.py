import math

import numpy as np
import pytest
import torch

from atl.core.events import EventPacket
from atl.data.synthetic import simulate
from atl.motion import PixelTrajectories, Translation, TranslationZoom, Zoom
from atl.objectives import (AnchoredExact, AnchoredSliced, MotionPriorCMaxLoss, Sharpness, flow_warp_loss,
                            heldout_score)


@pytest.fixture(scope="module")
def trans_sample():
    return simulate("translation", [12.0, -4.0], H=48, W=64, n_steps=80, seed=2, max_events=6000, n_shapes=15)


def test_exact_group_property():
    m = TranslationZoom(1, 50, 60, mode="exact", init=torch.tensor([[3.0, -2.0, 1.3]]))
    x = torch.tensor([[5.0, 7.0], [40.0, 30.0]]); b = torch.zeros(2, dtype=torch.long); t = torch.tensor([0.1, 0.8])
    y = m.transport(x, t, 0.5, b)
    back = m.transport(y, torch.full((2,), 0.5), t, b)
    assert torch.allclose(back, x, atol=1e-4)


def test_trajectory_inversion_recovers_origin():
    torch.manual_seed(0)
    g = torch.randn(1, 2, 40, 48) * 3
    g = torch.nn.functional.avg_pool2d(g, 9, 1, 4)              # smooth field
    m = PixelTrajectories(g, K=1)
    p = torch.rand(200, 2) * torch.tensor([40.0, 30.0]) + 4
    b = torch.zeros(200, dtype=torch.long)
    t = torch.rand(200)
    x = p + m.displacement(p, b, t)
    assert torch.allclose(m.origin(x, t, b), p, atol=1e-3)


def test_sliced_matches_exact_for_linear_motion(trans_sample):
    ev = trans_sample.events
    m = Translation(1, ev.height, ev.width, mode="linear", init=torch.tensor([[12.0, -4.0]]))
    ex = AnchoredExact(h=1.0, n_anchors=None, smooth_anchors=False, dequantize=False)(ev, m)["score"]
    sl = AnchoredSliced(n_slices=40, sigma=1.0, smooth_anchors=False, dequantize=False)(ev, m)["score"]
    assert abs(float(ex) - float(sl)) < 0.15, (float(ex), float(sl))


@pytest.mark.parametrize("obj_name", ["exact", "sliced"])
def test_anchored_score_is_maximised_at_truth_along_zoom_path(trans_sample, obj_name):
    ev = trans_sample.events
    mode = "linear" if obj_name == "exact" else "exact"
    vals = []
    for h in (-4.0, -1.0, 0.0, 1.0, 4.0):
        obj = AnchoredExact(h=1.0, n_anchors=None, seed=0) if obj_name == "exact" else AnchoredSliced(n_slices=10, seed=0)
        m = TranslationZoom(1, ev.height, ev.width, mode=mode, init=torch.tensor([[12.0, -4.0, h]]))
        vals.append(float(obj(ev, m)["score"]))
    assert max(range(5), key=lambda i: vals[i]) == 2, vals


def test_cmax_prefers_collapse_on_exact_zoom_path(trans_sample):
    ev = trans_sample.events
    obj = Sharpness("variance", t_ref=0.0)
    mk = lambda h: TranslationZoom(1, ev.height, ev.width, mode="exact", init=torch.tensor([[12.0, -4.0, h]]))
    assert float(obj(ev, mk(16.0))["loss"]) < float(obj(ev, mk(0.0))["loss"])


def test_anchored_exact_gradient_matches_reference(trans_sample):
    ev = trans_sample.events.subsample(1500)
    th = torch.tensor([[11.0, -3.5]])
    m1 = Translation(1, ev.height, ev.width, mode="linear", init=th)
    m2 = Translation(1, ev.height, ev.width, mode="linear", init=th)
    anchors = torch.arange(0, ev.n, 2)
    o = AnchoredExact(h=1.2, n_anchors=None, smooth_anchors=False, dequantize=False)
    l1, _ = o.log_scores(ev, m1, anchors)
    l1.mean().backward()
    from atl.ops import reference as R
    from atl.ops import window_counts
    v = m2.velocity(ev.xy, ev.t, ev.b)
    S = R.ats_pair_sums(ev.xy, ev.t, v, ev.b, anchors, 1.2, 1.0, 3.0)
    Wn = window_counts(ev.t, ev.b, anchors, 1.0).clamp(min=1)
    l2 = torch.log(o.eps_frac / (ev.height * ev.width) + S / (Wn * 2 * math.pi * 1.2 ** 2))
    l2.mean().backward()
    assert torch.allclose(l1, l2, atol=1e-4)
    assert torch.allclose(m1.theta.grad, m2.theta.grad, rtol=1e-3, atol=1e-5)


def test_mpcm_loss_runs_and_backprops(trans_sample):
    ev = trans_sample.events
    g = torch.zeros(1, 2, ev.height, ev.width, requires_grad=True)
    out = MotionPriorCMaxLoss(num_knn=8)(ev, PixelTrajectories(g, K=1))
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and g.grad is not None and g.grad.abs().sum() > 0


def test_metrics_rank(trans_sample):
    ev = trans_sample.events
    good = TranslationZoom(1, ev.height, ev.width, mode="exact", init=torch.tensor([[12.0, -4.0, 0.0]]))
    bad = TranslationZoom(1, ev.height, ev.width, mode="exact", init=torch.tensor([[0.0, 0.0, 0.0]]))
    assert heldout_score(ev, good) > heldout_score(ev, bad)
    assert flow_warp_loss(ev, good) > flow_warp_loss(ev, bad)


def test_dsec_png_roundtrip(tmp_path):
    from atl.data.dsec import read_flow_png, write_flow_png
    f = (np.random.default_rng(0).normal(size=(2, 20, 30)) * 30).astype(np.float32)
    f = np.round(f * 128) / 128
    v = np.random.default_rng(1).random((20, 30)) > 0.3
    write_flow_png(tmp_path / "f.png", f, v)
    f2, v2 = read_flow_png(tmp_path / "f.png")
    assert np.allclose(f2[:, v], f[:, v], atol=1e-6) and (v2 == v).all()


def test_smoothed_score_has_no_deconvolution_bias():
    """1-D check of the finite-kernel fix: anchors ~ N(0, w^2) on a line, sources = anchors scaled by s (a local
    contraction).  The plain anchored score prefers s < 1 (q * K_h matched to p); the smoothed score prefers s = 1."""
    torch.manual_seed(0)
    w, h, n = 1.0, 1.0, 4000
    a = torch.randn(n) * w
    src = torch.randn(n) * w
    def score(scale, smooth):
        g = torch.Generator().manual_seed(1)
        xa = a + (h * torch.randn(n, generator=g) if smooth else 0.0)
        d = xa[:, None] - scale * src[None]
        return torch.log(torch.exp(-d ** 2 / (2 * h * h)).mean(1)).mean().item()
    scales = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3]
    plain = [score(s, False) for s in scales]
    smooth = [score(s, True) for s in scales]
    assert scales[max(range(6), key=lambda i: plain[i])] < 1.0        # biased towards contraction
    assert scales[max(range(6), key=lambda i: smooth[i])] == 1.0      # proper
