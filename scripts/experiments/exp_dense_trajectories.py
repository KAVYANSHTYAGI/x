"""F6: dense per-pixel trajectories (MotionPriorCMax's representation) optimised directly under each objective.

A free coefficient grid (degree-1 polynomial prior, one trajectory per 4x4 tile) is optimised per window from
(a) the ground truth and (b) zero.  A proper objective keeps (a) at the truth and moves (b) towards it.
Reported: end-point error (px over the window) of the tile flow on pixels with events, and the change of the
objective relative to its value at the ground truth (negative = objective prefers a wrong field).
"""
import argparse
import json
import time

import torch

from atl.data import simulate
from atl.estimators.fit import fit
from atl.motion import PixelTrajectories
from atl.objectives import AnchoredExact, AnchoredSliced, MotionPriorCMaxLoss, Regularized, Sharpness
from atl.objectives.base import IWEConfig

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=120)
ap.add_argument("--lr", type=float, default=0.3)
ap.add_argument("--scenes", default="translation,zoom")
ap.add_argument("--objectives", default="mpcm,mpcm_no_tv,mpcm_knn1,gradmag_rand,cmax_var,cmax_multiref,ats_exact,ats_sliced")
ap.add_argument("--H", type=int, default=96)
ap.add_argument("--W", type=int, default=128)
ap.add_argument("--out", default="outputs/exp_dense_trajectories.json")
ap.add_argument("--tv", type=float, default=0.003, help="identical Charbonnier TV prior for every non-MPCM objective")
ap.add_argument("--anneal", default="8,1", help="bandwidth annealing (start,end) for the anchored scores")
args = ap.parse_args()

SCENES = {"translation": ("translation", [20.0, 6.0]), "zoom": ("zoom", [0.35]),
          "affine": ("affine", [8.0, -4.0, 0.25, 0.08, -0.06, 0.15])}


def make_objective(name):
    """Training-time version of an objective (with its own randomness)."""
    return {
        "mpcm": lambda: MotionPriorCMaxLoss(),
        "mpcm_no_tv": lambda: MotionPriorCMaxLoss(smooth_weight=0.0),
        "mpcm_knn1": lambda: MotionPriorCMaxLoss(num_knn=1),
        "gradmag_rand": lambda: Sharpness("grad_l1", t_ref="random", form="inv",
                                          iwe=IWEConfig(polarity="split", scale_by_dt=True)),
        "cmax_var": lambda: Sharpness("variance", t_ref=0.0, form="inv"),
        "cmax_multiref": lambda: Sharpness("variance", t_ref=[0.0, 0.5, 1.0], form="inv"),
        "ats_exact": lambda: AnchoredExact(h=1.0, n_anchors=6000, seed=0),
        "ats_exact_plain": lambda: AnchoredExact(h=1.0, n_anchors=6000, seed=0, smooth_anchors=False, dequantize=False),
        "ats_sliced": lambda: AnchoredSliced(n_slices=8, slices_per_step=3, n_anchors_per_slice=2000, seed=0),
        "ats_sliced_plain": lambda: AnchoredSliced(n_slices=8, slices_per_step=3, n_anchors_per_slice=2000, seed=0,
                                                   smooth_anchors=False, dequantize=False),
    }[name]()


def with_tv(name, obj):
    """Add the identical Charbonnier TV prior to every non-MotionPriorCMax objective."""
    if name.startswith("mpcm") or args.tv <= 0:
        return obj
    return Regularized(obj, "tv", args.tv)


def evaluator(name):
    """Deterministic version of the objective (fixed anchors / fixed reference times) for loss comparisons."""
    if name.startswith("mpcm"):
        base = make_objective(name)
        def f(ev, m):
            vals = []
            for tr in torch.linspace(0.05, 0.95, 7).tolist():
                base.t_ref = tr
                vals.append(float(base(ev, m)["loss"]))
            return sum(vals) / len(vals)
        return f
    if name == "gradmag_rand":
        base = Sharpness("grad_l1", t_ref=torch.linspace(0.05, 0.95, 7).tolist(), form="inv",
                         iwe=IWEConfig(polarity="split", scale_by_dt=True))
        return lambda ev, m: float(with_tv(name, base)(ev, m)["loss"])
    if name.startswith("ats_exact"):
        kw = dict(smooth_anchors=False, dequantize=False) if name == "ats_exact_plain" else {}
        return lambda ev, m: sum(float(with_tv(name, AnchoredExact(h=1.0, n_anchors=None, seed=k, **kw))(ev, m)["loss"])
                                 for k in range(4)) / 4
    if name.startswith("ats_sliced"):
        kw = dict(smooth_anchors=False, dequantize=False) if name == "ats_sliced_plain" else {}
        return lambda ev, m: sum(float(with_tv(name, AnchoredSliced(n_slices=8, seed=k, **kw))(ev, m)["loss"])
                                 for k in range(4)) / 4
    obj = with_tv(name, make_objective(name))
    return lambda ev, m: float(obj(ev, m)["loss"])


def coeff_from_flow(flow, H, W):
    """GT flow (2, H, W) -> degree-1 coefficient grid in MotionPriorCMax channel order (y, x)."""
    g = torch.zeros(1, 2, H, W)
    g[0, 0], g[0, 1] = flow[1], flow[0]              # MotionPriorCMax channel order (y, x)
    return g


results = {}
for scene in args.scenes.split(","):
    kind, th = SCENES[scene]
    s = simulate(kind, th, H=args.H, W=args.W, seed=11, max_events=50000)
    ev = s.events
    # fresh realisation of the same scene + motion (different thresholds / noise / subsample) for held-out tests
    ev_ho = simulate(kind, th, H=args.H, W=args.W, seed=11, max_events=50000, event_seed=777).events
    mask = torch.zeros(args.H, args.W, dtype=torch.bool)
    mask[ev.xy[:, 1].long().clamp(0, args.H - 1), ev.xy[:, 0].long().clamp(0, args.W - 1)] = True
    print(f"\n=== scene {scene}  events={ev.n}  mean|flow|={float(s.flow_gt.norm(dim=0)[mask].mean()):.1f}px")

    def epe(motion):
        """EPE of the motion's tile flow on pixels with events."""
        f = motion.flow_field(1, args.H, args.W)[0].detach()
        return float((f - s.flow_gt).norm(dim=0)[mask].mean())

    for name in args.objectives.split(","):
        for init in ("gt", "zero"):
            torch.manual_seed(0)
            grid = coeff_from_flow(s.flow_gt, args.H, args.W) if init == "gt" else torch.zeros(1, 2, args.H, args.W)
            grid = grid.clone().requires_grad_(True)
            core = make_objective(name)
            obj = with_tv(name, core)
            mk = lambda: PixelTrajectories(grid, K=1)
            ev_f = evaluator(name)
            with torch.no_grad():
                l_gt = ev_f(ev, PixelTrajectories(coeff_from_flow(s.flow_gt, args.H, args.W), K=1))
            e0 = epe(mk())
            t0 = time.time()
            cb = None
            if hasattr(core, "set_bandwidth") and init == "zero":
                h0, h1 = [float(v) for v in args.anneal.split(",")]
                n_ann = int(0.6 * args.steps)
                cb = lambda it, m, h: core.set_bandwidth(h0 + (h1 - h0) * min(1.0, (it + 1) / n_ann))
                core.set_bandwidth(h0)
            hist = fit(obj, mk, [grid], ev, steps=args.steps, lr=args.lr, callback=cb)
            with torch.no_grad():
                l_end = ev_f(ev, mk())
                gt_m = PixelTrajectories(coeff_from_flow(s.flow_gt, args.H, args.W), K=1)
                ho_gt, ho_end = ev_f(ev_ho, gt_m), ev_f(ev_ho, mk())
            e1 = epe(mk())
            rel = (l_end - l_gt) / max(abs(l_gt), 1e-12)
            rel_ho = (ho_end - ho_gt) / max(abs(ho_gt), 1e-12)
            results[f"{scene}/{name}/{init}"] = {"epe_start": e0, "epe_end": e1, "loss_gt": l_gt, "loss_end": l_end,
                                                 "rel_loss_vs_gt_insample": rel, "rel_loss_vs_gt_heldout": rel_ho,
                                                 "sec": time.time() - t0}
            flag = "  <-- prefers fitted field on HELD-OUT events (bias)" if rel_ho < -1e-3 else ""
            print(f"  {name:14s} init={init:4s}  EPE {e0:6.2f} -> {e1:6.2f} px   in-sample {rel:+.4f}"
                  f"   held-out {rel_ho:+.4f}  ({time.time() - t0:.0f}s){flag}", flush=True)

import os
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(results, open(args.out, "w"), indent=1)
print("saved", args.out)
