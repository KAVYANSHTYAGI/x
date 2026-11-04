# Thm 3 refined: does the collapse-free regulariser weight lam_min depend on the data (headroom kappa, window T)?
# If lam_min varies by >> the factor-4 safety margin seen in theory_checks.py (E), no fixed lam is safe for all data,
# and choosing lam for the hardest case costs bias on easy cases.  Objective J = G/G* - lam*2|h|, t_ref = T/2.
import numpy as np, theory_checks as tc
def blurw(img, w):
    r = np.arange(-int(4*w)-1, int(4*w)+2); g = np.exp(-0.5*r**2/w**2); g /= g.sum()
    img = np.apply_along_axis(lambda a: np.convolve(a, g, "same"), 0, img)
    return np.apply_along_axis(lambda a: np.convolve(a, g, "same"), 1, img)
def Gw(x, t, h, tref, w):
    y = tc.warp(x, t, h, tref)
    img, _, _ = np.histogram2d(y[:, 0], y[:, 1], bins=tc.S_, range=[[0, tc.S_], [0, tc.S_]])
    return (blurw(img, w) ** 2).sum()
hs = np.concatenate([np.linspace(-60, -3, 39), np.linspace(-3, 5, 161), np.linspace(5, 60, 36)])
rows = []
for T in (0.05, 0.25):
    for w in (0.5, 1.0, 2.0, 4.0):
        rng = np.random.default_rng(300); segs = tc.scene(rng)
        x, t, _, _ = tc.events(rng, segs, 6000, T, 1.0)
        g = np.array([Gw(x, t, h, T/2, w) for h in hs]); g /= Gw(x, t, 1.0, T/2, w); R = 2*np.abs(hs)
        loc = np.abs(hs - 1.0) < 1.5
        lam = 1e-5
        while not loc[np.argmax(g - lam*R)]: lam *= 1.1
        rows.append((T, w, g.max(), lam, g, R))
lam_max = max(r[3] for r in rows)
print(" T     width  max G/G* (headroom)  lam_min   bias@lam_min  bias@lam=max_lam_min(all rows)")
for T, w, gm, lam, g, R in rows:
    b1 = hs[np.argmax(g - lam*R)] - 1.0; b2 = hs[np.argmax(g - lam_max*R)] - 1.0
    print(f" {T:4.2f}  {w:4.1f}   {gm:8.1f}            {lam:8.4f}    {b1:+.3f}        {b2:+.3f}")
