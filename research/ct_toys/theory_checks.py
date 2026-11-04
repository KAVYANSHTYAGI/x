# Numerical checks of two derived predictions (see PROPER_ALIGNMENT_MATH.md, Thm 2 and Thm 4).
#  (A) Multi-reference CMax limit along the contraction path:  lim G_nu / G* = kappa * E_nu[m^2],
#      kappa = G_max / G* (sharpness headroom), m(t_ref) = fraction of events that contract onto the centre.
#  (B) Single-reference CMax first-order bias for a true zoom h*:  h_hat - h* ~= 2 s_tot^2 (tbar - t_ref) / (var_t * rho2)
#      = 12 s_tot^2 / (T rho2) for t_ref = 0, uniform t.  ~0 for t_ref = T/2.  Proper score S: ~0.
import sys, numpy as np
S_, c = 128, np.array([64.0, 64.0])
def blur(img, s=1.0):
    r = np.arange(-4, 5); g = np.exp(-0.5 * r**2 / s**2); g /= g.sum()
    img = np.apply_along_axis(lambda a: np.convolve(a, g, "same"), 0, img)
    return np.apply_along_axis(lambda a: np.convolve(a, g, "same"), 1, img)
def scene(rng, nseg=12):
    segs = []
    for _ in range(nseg):
        p = rng.uniform(34, 94, 2); d = rng.normal(size=2); d /= np.linalg.norm(d)
        segs.append((p, d, rng.uniform(8, 20)))
    return segs
def events(rng, segs, N, T, h_true, sn=0.5):
    k = rng.integers(0, len(segs), N); s = rng.uniform(0, 1, N); t = np.sort(rng.uniform(0, T, N))
    y = np.stack([segs[i][0] + segs[i][1] * segs[i][2] * u for i, u in zip(k, s)])
    n = np.stack([np.array([-segs[i][1][1], segs[i][1][0]]) for i in k])        # edge normals
    x = c + (y - c) * np.exp(h_true * t)[:, None] + rng.normal(0, sn, (N, 2))
    m = (x >= 0).all(1) & (x < S_).all(1)
    return x[m], t[m], y[m], n[m]
def warp(x, t, h, tref):                          # transport event (x,t) to time tref under zoom rate h
    return c + (x - c) * np.exp(h * (tref - t))[:, None]
def G(x, t, h, tref):                            # int I^2 of the blurred IWE (events outside the frame are masked)
    w = warp(x, t, h, tref)
    img, _, _ = np.histogram2d(w[:, 0], w[:, 1], bins=S_, range=[[0, S_], [0, S_]])
    return (blur(img) ** 2).sum()
def Sscore(x, t, h, anchors, bw=1.0, eps=1e-6):
    tot = 0.0
    for i in anchors:
        xh = warp(x, t, h, t[i]); d2 = ((xh - x[i]) ** 2).sum(1); d2[i] = np.inf
        tot += np.log(eps + np.exp(-0.5 * d2 / bw**2).sum() / ((len(t) - 1) * 2 * np.pi * bw**2))
    return tot / len(anchors)
def argmax_1d(f, lo, hi, n=41, iters=3):
    for _ in range(iters):
        hs = np.linspace(lo, hi, n); v = np.array([f(h) for h in hs]); k = int(np.argmax(v))
        lo, hi = hs[max(k - 2, 0)], hs[min(k + 2, n - 1)]
    return hs[k]

def main(which):
  if "A" in which:
      print("(A) multi-reference limit, translation-free data (truth h=0), T=0.5")
      rng = np.random.default_rng(0); segs = scene(rng); T = 0.5
      x, t, _, _ = events(rng, segs, 6000, T, 0.0)
      one = np.zeros((S_, S_)); one[64, 64] = len(t); Gmax = (blur(one) ** 2).sum()
      for refs, name in (([0.0], "t_ref=0"), ([0, T/2, T], "3 refs {0,T/2,T}"), (list(np.linspace(0, T, 21)), "uniform (21 refs)")):
          Gs = np.mean([G(x, t, 0.0, r) for r in refs]); Gl = np.mean([G(x, t, 400.0, r) for r in refs])
          m2 = np.mean([np.mean(t >= r) ** 2 for r in refs])
          print(f"  {name:18s} kappa={Gmax/Gs:6.1f}  predicted lim/G*={Gmax/Gs*m2:6.2f}  observed G(h=400)/G*={Gl/Gs:6.2f}")
  if "B" in which:
      # NOTE: GLOBAL search on [-3, 6]; for longer windows the global argmax of CMax(t_ref=0) falls into the collapse
      # branch, so this mixes Thm 1 and Thm 4.  'pred' here uses the original (factor-2 wrong) constant 12; the
      # corrected local prediction is 6*gamma*s_tot2/(T rho2), gamma in [1,2] -- checked in block C.
      print("\n(B) bias of GLOBAL argmax under a TRUE zoom h*=1.0 /s (mean of 6 scenes)")
      s_tot2 = 1.0 + 0.25 + 1/12                     # IWE blur^2 + event noise^2 + pixel quantisation
      for T in (0.05, 0.1, 0.25):
          r0, rm, rs, pred = [], [], [], []
          for seed in range(6):
              rng = np.random.default_rng(100 + seed); segs = scene(rng)
              x, t, y, n = events(rng, segs, 6000, T, 1.0)
              rho2 = np.mean(((y - c) * n).sum(1) ** 2)
              pred.append(12 * s_tot2 / (T * rho2))
              r0.append(argmax_1d(lambda h: G(x, t, h, 0.0), -3, 6) - 1.0)
              rm.append(argmax_1d(lambda h: G(x, t, h, T / 2), -3, 6) - 1.0)
              anchors = np.random.default_rng(seed).choice(len(t), 500, replace=False)
              rs.append(argmax_1d(lambda h: Sscore(x, t, h, anchors), -3, 6, n=25) - 1.0)
          print(f"  T={T:5.2f}s  predicted CMax(t_ref=0) bias {np.mean(pred):+.3f} | observed: CMax t_ref=0 {np.mean(r0):+.3f}"
                f"  CMax t_ref=T/2 {np.mean(rm):+.3f}  S {np.mean(rs):+.3f} (sd {np.std(rs):.3f})")
  if "C" in which:
      print("\n(C) LOCAL bias of CMax(t_ref=0) near the truth vs Thm 4:  dh = 6*gamma*s_tot^2/(T*rho2), gamma in [1,2]")
      s_tot2 = 1.0 + 0.25 + 1/12
      print("   T    scene  rho2  pred(gamma=1)  observed  observed/pred   t_ref=T/2")
      for T in (0.02, 0.05, 0.1):
          for seed in range(5):
              rng = np.random.default_rng(200 + seed); segs = scene(rng)
              x, t, y, n = events(rng, segs, 12000, T, 1.0)
              rho2 = np.mean(((y - c) * n).sum(1) ** 2); pred = 6 * s_tot2 / (T * rho2)
              hs = np.linspace(0.0, 2.0 + 6 * pred, 81)
              def loc(tref):
                  g = np.array([G(x, t, h, tref) for h in hs]); k = int(np.clip(np.argmax(g), 3, len(hs) - 4))
                  a, b, _ = np.polyfit(hs[k-3:k+4], g[k-3:k+4], 2); return -b / (2 * a) - 1.0
              o0, om = loc(0.0), loc(T / 2)
              print(f"  {T:5.2f}   {seed}   {rho2:5.0f}    {pred:+6.3f}      {o0:+6.3f}      {o0/pred:5.2f}       {om:+6.3f}")
  if "D" in which:
      print("\n(D) margins vs kernel width (truth h=0, contraction path h=400 for CMax, h=8 for S)")
      rng = np.random.default_rng(0); segs = scene(rng); T = 0.5
      x, t, _, _ = events(rng, segs, 6000, T, 0.0)
      anchors = np.random.default_rng(1).choice(len(t), 800, replace=False)
      print("  width | CMax collapse ratio G(h=400)/G*  (Thm 1: ~ 1/width) | S(truth)-S(h=8) nats (Thm 5b: ~ log 1/width)")
      for wdt in (0.5, 1.0, 2.0, 4.0):
          def Gw(h):
              w = warp(x, t, h, 0.0)
              img, _, _ = np.histogram2d(w[:, 0], w[:, 1], bins=S_, range=[[0, S_], [0, S_]])
              r = np.arange(-int(4*wdt)-1, int(4*wdt)+2); g = np.exp(-0.5 * r**2 / wdt**2); g /= g.sum()
              img = np.apply_along_axis(lambda a: np.convolve(a, g, "same"), 0, img)
              img = np.apply_along_axis(lambda a: np.convolve(a, g, "same"), 1, img)
              return (img ** 2).sum()
          print(f"  {wdt:4.1f}  | {Gw(400.0)/Gw(0.0):8.1f}"
                f"                                       | {Sscore(x, t, 0.0, anchors, bw=wdt) - Sscore(x, t, 8.0, anchors, bw=wdt):+.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ABCD")

def dilemma(T=0.25, h_true=1.0, seeds=4):
    """Thm 3: J = G/G* - lam * R,  R = |div f| = 2|h| (symmetric divergence penalty, mid-window reference so the
    intrinsic reference bias of Thm 4 is ~0).  Find the smallest lam whose global argmax over h in [-40,40] is the
    local (non-collapsed) maximum, and report the bias that lam induces on the true divergent motion."""
    print(f"\n(E) regulariser dilemma, true zoom h*={h_true}, T={T}, t_ref=T/2")
    print("  scene | lam_min (no collapse) | bias at lam_min | bias at 0.25*lam_min (collapses?)")
    for seed in range(seeds):
        rng = np.random.default_rng(300 + seed); segs = scene(rng)
        x, t, _, _ = events(rng, segs, 6000, T, h_true)
        hs = np.concatenate([np.linspace(-40, -3, 38), np.linspace(-3, 5, 161), np.linspace(5, 40, 36)])
        g = np.array([G(x, t, h, T / 2) for h in hs]); g /= G(x, t, h_true, T / 2); R = 2 * np.abs(hs)
        loc_mask = np.abs(hs - h_true) < 1.5
        def solve(lam):
            J = g - lam * R; k = int(np.argmax(J)); return hs[k], bool(loc_mask[k])
        lam = 1e-4
        while not solve(lam)[1]: lam *= 1.25
        b = solve(lam)[0] - h_true; b4, ok4 = solve(lam / 4)
        print(f"   {seed}    |      {lam:8.4f}          |    {b:+.3f}      |   {b4 - h_true:+.3f} ({'local' if ok4 else 'COLLAPSE'})")

if __name__ == "__main__" and "E" in (sys.argv[1] if len(sys.argv) > 1 else ""):
    dilemma()
