"""Timing of the low-level ops and of every loss on one realistic window.

    python bench/bench_ops.py --device cuda --events 1000000      # DSEC-like (1M events, 480x640)
    python bench/bench_ops.py --device cpu  --events 100000

Reports milliseconds per forward+backward for: bilinear / Gaussian splat, bilinear gather, the exact anchored pair
kernel, and the full losses (MotionPriorCMax vs ours) on a degree-1 pixel-trajectory field.  Use it to size
``n_anchors`` / ``max_events`` / ``tau`` in configs/dsec_ats.yaml before training.
"""
import argparse
import time

import torch

from atl.core.events import EventPacket
from atl.motion import PixelTrajectories
from atl.objectives import AnchoredExact, MotionPriorCMaxLoss
from atl.ops import ats_pair_sums, gather, splat

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--events", type=int, default=200000)
ap.add_argument("--H", type=int, default=480)
ap.add_argument("--W", type=int, default=640)
ap.add_argument("--anchors", type=int, default=60000)
ap.add_argument("--repeat", type=int, default=5)
a = ap.parse_args()
dev = torch.device(a.device)


def timeit(fn, n=a.repeat):
    fn()                                                       # warm-up (JIT build / caches)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / n


g = torch.Generator().manual_seed(0)
N, H, W = a.events, a.H, a.W
xy = torch.stack([torch.rand(N, generator=g) * (W - 1), torch.rand(N, generator=g) * (H - 1)], 1).to(dev)
t = torch.sort(torch.rand(N, generator=g)).values.to(dev)
p = torch.where(torch.rand(N, generator=g) > 0.5, 1.0, -1.0).to(dev)
b = torch.zeros(N, dtype=torch.long, device=dev)
ev = EventPacket(xy, t, p, b, 1, H, W)
w = torch.ones(N, device=dev)
pos = xy.clone().requires_grad_(True)
print(f"device={dev}  events={N}  image={H}x{W}")
print(f"  splat bilinear  fwd+bwd  {timeit(lambda: splat(pos, w, b, 1, H, W).sum().backward()):9.2f} ms")
print(f"  splat gauss s=1 fwd+bwd  {timeit(lambda: splat(pos, w, b, 1, H, W, kernel='gauss').sum().backward()):9.2f} ms")
img = torch.rand(1, H, W, device=dev, requires_grad=True)
print(f"  gather          fwd+bwd  {timeit(lambda: gather(img, pos, b).sum().backward()):9.2f} ms")
v = (torch.randn(N, 2, generator=g) * 5).to(dev).requires_grad_(True)
anc = torch.randperm(N, generator=g)[: a.anchors].sort().values.to(dev)
print(f"  ats pair ({a.anchors} anchors) fwd+bwd {timeit(lambda: ats_pair_sums(xy, t, v, b, 1, H, W, anc, 1.0, 1.0).sum().backward()):9.2f} ms")
grid = (torch.randn(1, 2, H, W, generator=g) * 3).to(dev).requires_grad_(True)
mk = lambda: PixelTrajectories(grid, K=1)
mp = MotionPriorCMaxLoss()
ours = AnchoredExact(n_anchors=a.anchors)
print(f"  loss MotionPriorCMax     fwd+bwd  {timeit(lambda: mp(ev, mk())['loss'].backward(), n=max(1, a.repeat // 2)):9.2f} ms")
print(f"  loss anchored (ours)     fwd+bwd  {timeit(lambda: ours(ev, mk())['loss'].backward(), n=max(1, a.repeat // 2)):9.2f} ms")
