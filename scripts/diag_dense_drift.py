"""Where does dense-field drift come from?  Fit pixel trajectories from GT under an objective, then compare GT vs
fitted on HELD-OUT events restricted to the interior (margin m px), where field-of-view entry/exit cannot matter.
    python scripts/diag_dense_drift.py --objective ats_exact|mpcm|mpcm_no_tv|gradmag|cmax_var"""
import argparse

import torch

from atl.data import simulate
from atl.estimators.fit import fit
from atl.motion import PixelTrajectories
from atl.objectives import AnchoredExact, MotionPriorCMaxLoss, Regularized, Sharpness
from atl.objectives.base import IWEConfig

ap = argparse.ArgumentParser()
ap.add_argument("--objective", default="ats_exact")
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--scene", default="translation")
ap.add_argument("--anchor_margin", type=float, default=0.0)
a = ap.parse_args()
H, W = 96, 128
SC = {"translation": ("translation", [20.0, 6.0]), "zoom": ("zoom", [0.35]), "affine": ("affine", [8.0, -4.0, 0.25, 0.08, -0.06, 0.15])}
kind, th = SC[a.scene]
s = simulate(kind, th, H=H, W=W, seed=11, max_events=50000)
ho = simulate(kind, th, H=H, W=W, seed=11, max_events=50000, event_seed=777).events
ev = s.events
gt = torch.zeros(1, 2, H, W); gt[0, 0], gt[0, 1] = s.flow_gt[1], s.flow_gt[0]
grid = gt.clone().requires_grad_(True)
GRID = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
train = {"ats_exact": lambda: Regularized(AnchoredExact(h=1.0, n_anchors=6000, seed=0, anchor_margin=a.anchor_margin), "tv", 0.003),
         "mpcm": lambda: MotionPriorCMaxLoss(), "mpcm_no_tv": lambda: MotionPriorCMaxLoss(smooth_weight=0.0),
         "gradmag": lambda: Regularized(Sharpness("grad_l1", t_ref="random", form="inv",
                                                  iwe=IWEConfig(polarity="split", scale_by_dt=True)), "tv", 0.003),
         "cmax_var": lambda: Regularized(Sharpness("variance", t_ref=0.0, form="inv"), "tv", 0.003)}[a.objective]()
fit(train, lambda: PixelTrajectories(grid, K=1), [grid], ev, steps=a.steps, lr=0.3)
fitted = PixelTrajectories(grid.detach(), K=1); truth = PixelTrajectories(gt, K=1)
err = (fitted.flow_field(1, H, W)[0] - s.flow_gt).norm(dim=0)


def score(evs, m):
    """Higher = better, deterministic.  Data term only (no regulariser), on the given event subset."""
    if a.objective == "ats_exact":
        return sum(float(AnchoredExact(h=1.0, n_anchors=None, seed=k)(evs, m)["score"]) for k in range(6)) / 6
    if a.objective.startswith("mpcm"):
        o = MotionPriorCMaxLoss(smooth_weight=0.0)
        vals = []
        for tr in GRID:
            o.t_ref = tr
            vals.append(-float(o(evs, m)["loss"]))
        return sum(vals) / len(vals)
    if a.objective == "gradmag":
        return -float(Sharpness("grad_l1", t_ref=GRID, form="inv", iwe=IWEConfig(polarity="split", scale_by_dt=True))(evs, m)["loss"])
    return -float(Sharpness("variance", t_ref=0.0, form="inv")(evs, m)["loss"])


print(f"objective={a.objective} scene={a.scene}")
for m in (10, 20, 30):
    inner = torch.zeros(H, W, dtype=torch.bool); inner[m:H - m, m:W - m] = True
    print(f"  EPE interior(margin {m}) {float(err[inner].mean()):.2f}   border {float(err[~inner].mean()):.2f}")
for margin in (0, 10, 20, 30):
    x, y = ho.xy[:, 0], ho.xy[:, 1]
    sub = ho.select((x >= margin) & (x <= W - 1 - margin) & (y >= margin) & (y <= H - 1 - margin))
    st, sf = score(sub, truth), score(sub, fitted)
    print(f"  held-out events with margin>={margin:2d}px (n={sub.n}):  score(fitted)-score(GT) = {sf - st:+.5f}"
          f"  (relative {(sf - st) / max(abs(st), 1e-12):+.4f})")
