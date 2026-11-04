"""Failure-mode suite: every objective, same data, same motion classes, same optimiser.

F1 collapse          data = translation; model = translation + zoom (contains a contraction path).
                     (a) min over the zoom path of [loss(h) - loss(truth)] (< 0: objective prefers a wrong motion)
                     (b) optimise all 3 parameters from the truth; report |h| and translation error.
F2 reference bias    data = true zoom h*; model = zoom.  1-D argmax near the truth; relative bias (h_hat - h*)/h*.
F3 lambda dilemma    Shiba-regularised CMax over a lambda sweep on a translation scene (collapse risk) and a zoom
                     scene (bias risk); no single lambda is good on both.
F4 metric fooling    candidate motions (truth, collapsed, zero, perturbed) ranked by FWL, RSAT and held-out score.
Objectives with randomness (random t_ref, anchor subsampling) are evaluated deterministically (fixed reference grid,
all anchors) in the comparisons; they are optimised with their training-time randomness.
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch

from ..data.synthetic import simulate
from ..estimators.fit import fit
from ..motion import TranslationZoom, Zoom
from ..objectives import (AnchoredExact, AnchoredSliced, EMin, Regularized, STPPP, Sharpness, Timestamp,
                          flow_warp_loss, heldout_score, rsat)
from ..objectives.base import IWEConfig

GRID = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]


def objectives(mode: str = "exact", deterministic: bool = True) -> Dict[str, object]:
    """The objectives compared by the suite (deterministic versions for evaluation)."""
    rand = GRID if deterministic else "random"
    objs = {
        "CMax variance (t_ref=0)": Sharpness("variance", t_ref=0.0),
        "CMax multi-ref {0,.5,1}": Sharpness("variance", t_ref=[0.0, 0.5, 1.0]),
        "MPCM focus (|grad| L1, rand t_ref)": Sharpness("grad_l1", t_ref=rand, form="inv",
                                                        iwe=IWEConfig(polarity="split", scale_by_dt=True)),
        "Shiba R_D (lambda=1)": Regularized(Sharpness("variance", t_ref=0.0), "div", 1.0),
        "Shiba R_A (lambda=1)": Regularized(Sharpness("variance", t_ref=0.0), "det", 1.0),
        "EMin": EMin(t_ref=0.0),
        "ST-PPP": STPPP(t_ref=0.0),
        "Zhu timestamp": Timestamp(taming=False),
        "Taming timestamp": Timestamp(taming=True),
        "ATS sliced (ours)": AnchoredSliced(n_slices=10),
    }
    if mode == "linear":
        objs["ATS exact (ours)"] = AnchoredExact(h=1.0, n_anchors=None)
    return objs


def _loss(obj, ev, m):
    with torch.no_grad():
        return float(obj(ev, m)["loss"])


def f1_collapse(mode="exact", seed=3, steps=80, lr=0.05) -> List[dict]:
    """F1: preference along a contraction path + optimisation from the truth."""
    s = simulate("translation", [25.0, 5.0], H=96, W=128, seed=seed, max_events=40000)
    ev = s.events
    rows = []
    hs = [-16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0]
    for name, obj in objectives(mode).items():
        mk = lambda h: TranslationZoom(1, ev.height, ev.width, mode=mode, init=torch.tensor([[25.0, 5.0, h]]))
        l0 = _loss(obj, ev, mk(0.0))
        rel = [(_loss(obj, ev, mk(h)) - l0) / max(abs(l0), 1e-12) for h in hs]
        k = min(range(len(hs)), key=lambda i: rel[i])
        m = mk(0.0)
        fit(obj, lambda: m, [m.theta], ev, steps=steps, lr=lr)
        th = m.theta.detach()[0]
        rows.append({"objective": name, "min_rel_loss_on_path": rel[k], "argmin_h": hs[k] if rel[k] < 0 else 0.0,
                     "opt_from_truth_h": float(th[2]), "opt_from_truth_trans_err": float((th[:2] - th.new_tensor([25., 5.])).norm()),
                     "proper_on_path": rel[k] >= 0})
    return rows


def f2_reference_bias(h_true=(0.1, 0.25, 0.5), seed=7, mode="exact") -> List[dict]:
    """F2: relative bias of the 1-D optimum under a true zoom, for several zoom rates."""
    rows = []
    data = {h: simulate("zoom", [h], H=96, W=128, seed=seed, max_events=40000).events for h in h_true}
    for name, obj in objectives(mode).items():
        row = {"objective": name}
        for h, ev in data.items():
            hs = torch.linspace(h * 0.0, h * 2.5, 26).tolist()
            vals = [_loss(obj, ev, Zoom(1, ev.height, ev.width, mode=mode, init=torch.tensor([[x]]))) for x in hs]
            k = min(range(len(hs)), key=lambda i: vals[i])
            if 0 < k < len(hs) - 1:                                  # parabolic refinement
                y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
                den = y0 - 2 * y1 + y2
                off = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
                hk = hs[k] + off * (hs[1] - hs[0])
            else:
                hk = hs[k]
            row[f"rel_bias@h*={h}"] = (hk - h) / h
        rows.append(row)
    return rows


def f3_lambda(seed=5, mode="exact", lambdas=(0.0, 1e-2, 1e-1, 1.0, 10.0, 100.0), steps=80, lr=0.05) -> List[dict]:
    """F3: Shiba-regularised CMax over a lambda sweep (translation and zoom scenes) vs ours."""
    rows = []
    ev_t = simulate("translation", [25.0, 5.0], H=96, W=128, seed=seed, max_events=40000).events
    ev_z = simulate("zoom", [0.4], H=96, W=128, seed=seed, max_events=40000).events
    scale = None
    for reg in ("div", "det"):
        for lam in lambdas:
            obj = Regularized(Sharpness("variance", t_ref=0.0), reg, lam)
            # translation scene: optimise translation+zoom from truth; report spurious |h|
            m = TranslationZoom(1, 96, 128, mode=mode, init=torch.tensor([[25.0, 5.0, 0.0]]))
            fit(obj, lambda: m, [m.theta], ev_t, steps=steps, lr=lr)
            h_spur = float(m.theta.detach()[0, 2])
            # zoom scene: optimise zoom from truth; report relative bias
            mz = Zoom(1, 96, 128, mode=mode, init=torch.tensor([[0.4]]))
            fit(obj, lambda: mz, [mz.theta], ev_z, steps=steps, lr=lr * 0.2)
            bias = (float(mz.theta.detach()[0, 0]) - 0.4) / 0.4
            rows.append({"regulariser": f"R_{reg}", "lambda": lam, "translation_scene_spurious_h": h_spur,
                         "zoom_scene_rel_bias": bias})
    for name in ("ATS sliced (ours)",):
        obj = AnchoredSliced(n_slices=10)
        m = TranslationZoom(1, 96, 128, mode=mode, init=torch.tensor([[25.0, 5.0, 0.0]]))
        fit(obj, lambda: m, [m.theta], ev_t, steps=steps, lr=lr)
        mz = Zoom(1, 96, 128, mode=mode, init=torch.tensor([[0.4]]))
        fit(obj, lambda: mz, [mz.theta], ev_z, steps=steps, lr=lr * 0.2)
        rows.append({"regulariser": name, "lambda": "none", "translation_scene_spurious_h": float(m.theta.detach()[0, 2]),
                     "zoom_scene_rel_bias": (float(mz.theta.detach()[0, 0]) - 0.4) / 0.4})
    return rows


def f4_metrics(seed=9, mode="exact") -> List[dict]:
    """F4: ranking of candidate motions by FWL, RSAT and the held-out anchored score."""
    s = simulate("translation", [25.0, 5.0], H=96, W=128, seed=seed, max_events=40000)
    ev = s.events
    cands = {"truth": [25.0, 5.0, 0.0], "collapsed (h=+8)": [25.0, 5.0, 8.0], "collapsed (h=-8)": [25.0, 5.0, -8.0],
             "zero motion": [0.0, 0.0, 0.0], "truth + 4px": [29.0, 5.0, 0.0]}
    rows = []
    for k, th in cands.items():
        m = TranslationZoom(1, 96, 128, mode=mode, init=torch.tensor([th]))
        f = m.flow_field(1, 96, 128)[0].detach()
        epe = float((f - s.flow_gt).norm(dim=0).mean())
        rows.append({"candidate": k, "EPE": epe, "FWL (higher=better)": flow_warp_loss(ev, m),
                     "RSAT (lower=better)": rsat(ev, m), "held-out ATS (higher=better)": heldout_score(ev, m)})
    return rows
