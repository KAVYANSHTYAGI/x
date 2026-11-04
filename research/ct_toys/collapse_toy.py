# Shiba-style collapse test.  Data: edges translating with v=(60,0) px/s for 0.5 s, 128x128 sensor.
# Model warp (aligned point y, time t):  x(t) = c + (y-c) e^{h t} + v t     (h = zoom rate; truth h=0)
#   CMax  : variance of the image of events warped to t_ref = 0 (Gaussian blur, pixel grid)  -- standard
#   CMax-multi: mean variance over t_ref in {0, T/2, T}   (Shiba ECCV'22 "multi-reference")
#   LOO   : (1/N) sum_i log( eps + mean_{j!=i} K_bw( W_{t_j->t_i}(x_j) - x_i ) )   -- each event is its own reference;
#           the evaluation point x_i is the RAW event, which no warp can move.
import numpy as np
import torch
def gaussian_filter(img, s):
    r = np.arange(-4, 5); g = np.exp(-0.5 * r**2 / s**2); g /= g.sum()
    img = np.apply_along_axis(lambda a: np.convolve(a, g, "same"), 0, img)
    return np.apply_along_axis(lambda a: np.convolve(a, g, "same"), 1, img)
rng = np.random.default_rng(1)
S, T, v_true, c = 128, 0.5, np.array([60.0, 0.0]), np.array([64.0, 64.0])
segs = []
for _ in range(12):
    p = rng.uniform(20, 90, 2); d = rng.normal(size=2); d /= np.linalg.norm(d)
    segs.append((p, d, rng.uniform(8, 25)))
N = 4000
k = rng.integers(0, len(segs), N); s = rng.uniform(0, 1, N); t = np.sort(rng.uniform(0, T, N))
y = np.stack([segs[i][0] + segs[i][1] * segs[i][2] * u for i, u in zip(k, s)])
x = y + v_true * t[:, None] + rng.normal(0, 0.5, (N, 2))
keep = (x >= 0).all(1) & (x < S).all(1); x, t = x[keep], t[keep]; N = len(t)

def to_aligned(x, t, v, h):  return c + (x - v * t[:, None] - c) * np.exp(-h * t)[:, None]
def from_aligned(y, t, v, h): return c + (y - c) * np.exp(h * t)[:, None] + v * t[:, None]

def cmax(v, h, tref):
    w = from_aligned(to_aligned(x, t, v, h), np.full(N, tref), v, h)
    img, _, _ = np.histogram2d(w[:, 0], w[:, 1], bins=S, range=[[0, S], [0, S]])
    return gaussian_filter(img, 1.0).var()

anchors = rng.choice(N, 1200, replace=False)
def loo(v, h, bw=1.5, eps=1e-6):
    y = to_aligned(x, t, v, h); tot = 0.0
    for i in anchors:
        xh = from_aligned(y, np.full(N, t[i]), v, h)          # every event transported to t_i
        d2 = ((xh - x[i]) ** 2).sum(1); d2[i] = np.inf          # leave one out
        tot += np.log(eps + np.exp(-0.5 * d2 / bw**2).sum() / ((N - 1) * 2 * np.pi * bw**2))
    return tot / len(anchors)

print(f"N={N} events.  Columns normalised to their value at the truth (h=0).")
print(f"{'h':>6} | {'CMax(tref=0)':>12} {'CMax-multi':>10} {'LOO':>9}")
base = (cmax(v_true, 0, 0), np.mean([cmax(v_true, 0, r) for r in (0, T/2, T)]), loo(v_true, 0))
for h in (-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8, 16):
    a = cmax(v_true, h, 0); b = np.mean([cmax(v_true, h, r) for r in (0, T/2, T)]); l = loo(v_true, h)
    print(f"{h:6.1f} | {a/base[0]:12.3f} {b/base[1]:10.3f} {l-base[2]:+9.3f}")
print("\nSanity: profile over v_x at h=0")
for vx in (40, 50, 55, 60, 65, 70, 80):
    vv = np.array([vx, 0.0])
    print(f"vx={vx:3d} | CMax {cmax(vv,0,0)/base[0]:.3f}   LOO {loo(vv,0)-base[2]:+.3f}")

# ---------------- ablation: where does the protection come from?
#   EMin-like : mean_i log KDE_{-i}(y_i) in the ALIGNED frame (t_ref=0), fixed bandwidth  (entropy of warped events)
#   EMin+logdet: same + mean_i log|det dy/dx|(x_i) = -2 h t_i        (exact change-of-variables term, weight 1)
#   Ours (LOO, raw-anchored) from above
def emin(v, h, bw=1.5, logdet=False):
    yy = to_aligned(x, t, v, h); tot = 0.0
    for i in anchors:
        d2 = ((yy - yy[i]) ** 2).sum(1); d2[i] = np.inf
        tot += np.log(1e-6 + np.exp(-0.5 * d2 / bw**2).sum() / ((N - 1) * 2 * np.pi * bw**2))
        if logdet: tot += -2 * h * t[i]
    return tot / len(anchors)
print("\nAblation (difference from value at truth h=0; >0 means the objective prefers the wrong motion)")
b1, b2, b3 = emin(v_true, 0), emin(v_true, 0, logdet=True), loo(v_true, 0)
print(f"{'h':>6} | {'EMin':>8} {'EMin+logdet':>11} {'Ours':>8}")
for h in (-4, -1, 0, 1, 4, 8, 16):
    print(f"{h:6.1f} | {emin(v_true,h)-b1:+8.3f} {emin(v_true,h,logdet=True)-b2:+11.3f} {loo(v_true,h)-b3:+8.3f}")
