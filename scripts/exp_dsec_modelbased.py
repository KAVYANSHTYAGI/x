"""Real data, no learning: per-window dense flow fitted under each objective on DSEC ground-truth windows.

Same motion model for every objective (coarse-to-fine bilinear velocity grid, classic linear warp, as in MultiCM),
same optimiser and steps; evaluation against DSEC ground truth on all valid pixels (benchmark protocol).
"""
import argparse
import json
import os
import time

import torch

from atl.core.events import EventPacket
from atl.data.dsec import DSECSequence
from atl.estimators.fit import fit
from atl.eval.flow_metrics import Accumulator, flow_errors
from atl.motion import DenseLinearFlow
from atl.objectives import AnchoredExact, Regularized, Sharpness
from atl.objectives.base import IWEConfig

ap = argparse.ArgumentParser()
ap.add_argument("--seq", default="/home/user/data/dsec/train/thun_00_a")
ap.add_argument("--windows", default="5,12,20,28,36")
ap.add_argument("--max_events", type=int, default=150000)
ap.add_argument("--levels", default="3x4,6x8,12x16,24x32")
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--lr", type=float, default=1.0)
ap.add_argument("--objectives", default="zero,cmax_var,cmax_multiref,gradmag_multiref,ats_exact")
ap.add_argument("--tv", type=float, default=0.0)
ap.add_argument("--out", default="outputs/exp_dsec_modelbased.json")
args = ap.parse_args()

GRID = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]


def make(name):
    o = {"cmax_var": lambda: Sharpness("variance", t_ref=0.0),
         "cmax_multiref": lambda: Sharpness("variance", t_ref=[0.0, 0.5, 1.0]),
         "gradmag_multiref": lambda: Sharpness("grad_l1", t_ref=GRID, form="inv",
                                               iwe=IWEConfig(polarity="split", scale_by_dt=True)),
         "ats_exact": lambda: AnchoredExact(h=1.0, n_anchors=20000, seed=0)}[name]()
    return Regularized(o, "tv", args.tv) if args.tv > 0 else o


seq = DSECSequence(args.seq, windows="gt")
levels = [tuple(int(v) for v in l.split("x")) for l in args.levels.split(",")]
acc = {n: Accumulator() for n in args.objectives.split(",")}
rows = []
for wi in [int(w) for w in args.windows.split(",")]:
    d = seq.sample(wi, max_events=args.max_events, seed=wi)
    ev = EventPacket.from_arrays(d["x"], d["y"], d["t"], d["p"], 480, 640, normalize_time=False)
    gt = torch.from_numpy(d["flow"])[None]; valid = torch.from_numpy(d["valid"])[None]
    print(f"\nwindow {d['name']}  events(used)={ev.n}  valid px={int(valid.sum())}  mean|gt|={float(gt.norm(dim=1)[valid].mean()):.2f}")
    for name in args.objectives.split(","):
        t0 = time.time()
        m = DenseLinearFlow(1, 480, 640, grid=levels[0], mode="linear")
        if name != "zero":
            obj = make(name)
            core = obj.base if isinstance(obj, Regularized) else obj
            n_lv = len(levels)
            for li, lv in enumerate(levels):
                if li > 0:
                    m.upsample(lv[0] // m.grid.shape[-2])
                if hasattr(core, "set_bandwidth"):              # coarse-to-fine kernel: 8 px -> 1 px
                    core.set_bandwidth(max(1.0, 8.0 / (2 ** li)))
                fit(obj, lambda: m, [m.grid], ev, steps=args.steps, lr=args.lr / (2 ** li))
        with torch.no_grad():
            pred = m.flow_field(1, 480, 640)
        e = flow_errors(pred, gt, valid)
        acc[name].add(e)
        rows.append({"window": d["name"], "objective": name, **e, "sec": time.time() - t0})
        print(f"  {name:18s} EPE {e['EPE']:6.3f}  AE {e['AE']:6.2f}  3PE {e['3PE']:6.2f}%   ({time.time() - t0:.0f}s)", flush=True)

print("\n=== totals (pixel-weighted) ===")
summary = {}
for n, a in acc.items():
    r = a.result(); summary[n] = r
    print(f"  {n:18s} EPE {r['EPE']:6.3f}  AE {r['AE']:6.2f}  1PE {r['1PE']:6.2f}%  3PE {r['3PE']:6.2f}%")
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump({"rows": rows, "summary": summary, "args": vars(args)}, open(args.out, "w"), indent=1)
