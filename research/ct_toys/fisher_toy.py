# 1-D pixel: dL/dt = a (a = |grad L| * speed). Events at level crossings with per-event threshold C_k ~ N(C, s^2),
# reference resets at each event.  Estimate a from the events in a window of length T.
#   Poisson / rate estimator (what CMax/ST-PPP/rate models effectively use): a_hat = C * N / T
#   Level-crossing MLE using inter-event gaps g_k = C_k / a:   maximise sum_k [ log a - (a g_k - C)^2 / (2 s^2) ]
#   CRLB per gap under LC (from C_k = a g_k ~ N(C,s^2), reparam): I(a) = (C/s)^2 / a^2 + O(1/a^2)
import numpy as np
rng = np.random.default_rng(0)
C, a = 0.25, 1.0
def lc_mle(g, s):
    # d/da: n/a - sum g (a g - C)/s^2 = 0  ->  (sum g^2) a^2 - C (sum g) a - n s^2 = 0
    A, B, n = (g*g).sum(), (g).sum(), len(g)
    return (C*B + np.sqrt((C*B)**2 + 4*A*n*s*s)) / (2*A)
print(f"{'s/C':>5} {'E[N]':>5} | {'rel.RMSE rate':>13} {'rel.RMSE LC':>11} {'LC CRLB':>8} | ratio")
for sr in (0.05, 0.1, 0.3, 1.0):
    s = sr * C
    for EN in (3, 5, 10, 50, 200):
        T = EN * C / a
        er_p, er_l = [], []
        for _ in range(4000):
            phase = rng.uniform(0, 1)                 # unknown reference level at window start
            thr = np.clip(rng.normal(C, s, 4*EN+20), 1e-3, None)
            t = np.cumsum(thr / a); t = t - phase * t[0]
            t = t[(t >= 0) & (t < T)]
            if len(t) < 3: continue
            er_p.append(C * len(t) / T - a)
            er_l.append(lc_mle(np.diff(t), s) - a)
        rp, rl = np.sqrt(np.mean(np.square(er_p))), np.sqrt(np.mean(np.square(er_l)))
        crlb = a * sr / np.sqrt(EN - 1)
        print(f"{sr:5.2f} {EN:5d} | {rp:13.4f} {rl:11.4f} {crlb:8.4f} | {rp/rl:5.1f}x")
