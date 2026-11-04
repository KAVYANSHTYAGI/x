"""Quick cross-baseline collapse scan: data = pure translation; model = translation + zoom (contains a contraction
path).  Every objective is evaluated along the zoom path h with the translation fixed at its true value."""
import sys

import torch

from atl.data import simulate
from atl.motion import TranslationZoom
from atl.objectives import build

mode = sys.argv[1] if len(sys.argv) > 1 else "exact"
torch.manual_seed(0)
s = simulate("translation", [25.0, 5.0], H=120, W=160, seed=3, max_events=60000)
ev = s.events
names = ["cmax_var", "cmax_multiref", "cmax_gradmag_rand", "shiba_rd", "shiba_ra", "emin", "stppp", "zhu_ts",
         "taming_ts", "ats_sliced"] + (["ats_exact"] if mode == "linear" else [])
hs = [-8.0, -2.0, -0.5, 0.0, 0.5, 2.0, 8.0, 32.0]
print(f"mode={mode}  events={ev.n}  (loss relative to truth h=0; NEGATIVE = objective prefers the wrong motion)")
print(f"{'objective':20s}" + "".join(f"{h:>8g}" for h in hs))
for name in names:
    obj = build(name, seed=0, n_anchors=8000)
    row = []
    for h in hs:
        m = TranslationZoom(1, ev.height, ev.width, mode=mode, init=torch.tensor([[25.0, 5.0, h]]))
        torch.manual_seed(1)
        with torch.no_grad():
            vals = [float(obj(ev, m)["loss"]) for _ in range(3 if "rand" in name else 1)]
        row.append(sum(vals) / len(vals))
    base = row[hs.index(0.0)]
    scale = abs(base) if abs(base) > 1e-12 else 1.0
    print(f"{name:20s}" + "".join(f"{(r - base) / scale:+8.3f}" for r in row))
