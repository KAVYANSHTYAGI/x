"""evkit ops vs pure-PyTorch references (CPU here; CUDA section runs automatically when a GPU is present).

    python -m pytest -q evkit/tests/python
"""
import os
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "evpp"))

import evkit  # noqa: E402

CUDA = torch.cuda.is_available() and evkit.C.with_cuda
DEVICES = ["cpu"] + (["cuda"] if CUDA else [])


# --------------------------------------------------------------------------- references (float64)
def ref_scan(v, dt, rate, Bm, Cm, skip, h0, n_save):
    B, L, D = v.shape
    N = rate.shape[1]
    h = torch.zeros(B, D, N, dtype=v.dtype, device=v.device) if h0 is None else h0
    ys, Hs = [], []
    for i in range(L):
        h = torch.exp(-rate * dt[:, i, None, None]) * h + Bm * v[:, i, :, None]
        ys.append((h * Cm).sum(-1) + skip * v[:, i])
        Hs.append(h[:, :n_save])
    return torch.stack(ys, 1), torch.stack(Hs, 1), h


def ref_readout(H, tau, rate, Cm):
    return (Cm * torch.exp(-rate * tau[..., None, None]) * H[:, :, None]).sum(-1)


def problem(B=2, L=45, D=12, N=8, n_save=4, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    v = r(B, L, D)
    dt = torch.exp(torch.empty(B, L).uniform_(np.log(1e-6), np.log(0.05), generator=g))
    rate = torch.exp(torch.empty(D, N).uniform_(0.0, np.log(2e4), generator=g))
    Bm, Cm, skip, h0 = 0.5 * r(D, N), 0.5 * r(D, N), r(D), 0.3 * r(B, D, N)
    return [t.to(device) for t in (v, dt, rate, Bm, Cm, skip, h0)], n_save


def grads(fn, inputs, weights):
    xs = [x.detach().clone().requires_grad_(True) for x in inputs]
    outs = fn(*xs)
    loss = sum((o * w).sum() for o, w in zip(outs, weights))
    return [o.detach() for o in outs], torch.autograd.grad(loss, xs, allow_unused=True)


# --------------------------------------------------------------------------- scan
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("L", [1, 31, 32, 33, 100])
def test_decay_scan_matches_reference(device, mode, L):
    (v, dt, rate, Bm, Cm, skip, h0), ns = problem(L=L, N=8, device=device)
    g = torch.Generator().manual_seed(1)
    ws = [torch.randn(2, L, 12, generator=g).to(device), torch.randn(2, L, ns, 8, generator=g).to(device),
          torch.randn(2, 12, 8, generator=g).to(device)]
    inputs = [v, dt, rate, Bm, Cm, skip, h0]
    out_e, g_e = grads(lambda *a: evkit.decay_scan(*a, n_save=ns, fast_exp=False, mode=mode), inputs, ws)
    out_r, g_r = grads(lambda *a: ref_scan(*a, ns), [x.double() for x in inputs], [w.double() for w in ws])
    for a, b in zip(out_e, out_r):
        torch.testing.assert_close(a.double(), b, rtol=2e-4, atol=2e-5)
    names = ["v", "dt", "rate", "Bm", "Cm", "skip", "h0"]
    for n, a, b in zip(names, g_e, g_r):
        torch.testing.assert_close(a.double(), b, rtol=2e-3, atol=2e-4, msg=lambda m, n=n: f"grad {n}: {m}")


def test_decay_scan_without_h0_and_partial_grads():
    (v, dt, rate, Bm, Cm, skip, _), ns = problem()
    v.requires_grad_(True)
    y, H, hl = evkit.decay_scan(v, dt, rate, Bm, Cm, skip, None, n_save=ns)
    (y.sum()).backward()                                   # H and h_last unused -> None grads in backward
    yr, _, _ = ref_scan(v.detach().double(), dt.double(), rate.double(), Bm.double(), Cm.double(), skip.double(), None, ns)
    torch.testing.assert_close(y.detach().double(), yr, rtol=2e-4, atol=2e-5)
    assert torch.isfinite(v.grad).all()


# --------------------------------------------------------------------------- readout
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("S", [1, 7, 9])
def test_readout_matches_reference(device, S):
    g = torch.Generator().manual_seed(3)
    B, L, n, N = 2, 11, 4, 8
    H = torch.randn(B, L, n, N, generator=g)
    tau = torch.exp(torch.empty(B, L, S).uniform_(np.log(1e-7), np.log(0.3), generator=g))
    rate = torch.exp(torch.empty(n, N).uniform_(0, np.log(1e4), generator=g))
    Cm = torch.randn(n, N, generator=g)
    w = torch.randn(B, L, S, n, generator=g)
    inputs = [t.to(device) for t in (H, tau, rate, Cm)]
    out_e, g_e = grads(lambda *a: [evkit.readout(*a, fast_exp=False)], inputs, [w.to(device)])
    out_r, g_r = grads(lambda *a: [ref_readout(*a)], [x.double() for x in inputs], [w.double().to(device)])
    torch.testing.assert_close(out_e[0].double(), out_r[0], rtol=2e-4, atol=2e-5)
    for a, b in zip(g_e, g_r):
        torch.testing.assert_close(a.double(), b, rtol=2e-3, atol=2e-4)


# --------------------------------------------------------------------------- tokenize / I/O
def test_tokenize_chunk_invariant_and_correct():
    g = torch.Generator().manual_seed(0)
    n = 10000
    t = torch.sort(torch.randint(0, 5_000_000, (n,), generator=g)).values
    x = torch.randint(0, 1280, (n,), generator=g).short()
    y = torch.randint(0, 720, (n,), generator=g).short()
    p = torch.randint(0, 2, (n,), generator=g).to(torch.uint8)
    full = evkit.tokenize(t, x, y, p, 1280, 720, 16, 4, seed=9)
    k = 3777
    a = evkit.tokenize(t[:k], x[:k], y[:k], p[:k], 1280, 720, 16, 4, seed=9)
    b = evkit.tokenize(t[k:], x[k:], y[k:], p[k:], 1280, 720, 16, 4, seed=9, index_offset=k, t_prev=int(t[k - 1]))
    for f, i, j in zip(full, a, b):
        assert torch.equal(f, torch.cat([i, j]))
    patch, sub, pol, u, v, dt = full
    gap = (t[1:] - t[:-1]).double() * 1e-6
    assert ((dt[1:].double() >= gap - 1e-9) & (dt[1:].double() < gap + 1e-6 + 1e-9)).all()
    assert patch.max() < 256 and sub.max() < 16


def test_evb_roundtrip(tmp_path):
    n = 50000
    g = torch.Generator().manual_seed(2)
    t = torch.sort(torch.randint(0, 3_000_000, (n,), generator=g)).values
    x, y = torch.randint(0, 640, (n,), generator=g).short(), torch.randint(0, 480, (n,), generator=g).short()
    p = torch.randint(0, 2, (n,), generator=g).to(torch.uint8)
    path = str(tmp_path / "s.evb")
    evkit.write_evb(path, t, x, y, p, 640, 480, "test")
    r = evkit.EvbReader(path)
    assert len(r) == n and r.info()["sensor"] == "test"
    w = r.window(1_000_000, 1_500_000)
    m = (t >= 1_000_000) & (t < 1_500_000)
    assert torch.equal(w["t"], t[m]) and torch.equal(w["x"], x[m]) and torch.equal(w["p"], p[m])


# --------------------------------------------------------------------------- full model: evkit == torch backend
@pytest.mark.parametrize("objective", ["ar", "intensity", "physics"])
@pytest.mark.parametrize("device", DEVICES)
def test_evpp_backends_agree(objective, device):
    from evpp import EvPPConfig, EvPP
    from evpp.training import build_streams, WindowSampler
    base = dict(grid=4, sub=2, d_model=16, d_state=8, n_layers=2, d_int=4, n_mix=3, n_quad=6, clock_dim=4,
                objective=objective)
    torch.manual_seed(0)
    m_t = EvPP(EvPPConfig(**base, impl="torch")).to(device)
    m_e = EvPP(EvPPConfig(**base, impl="evkit")).to(device)
    m_e.load_state_dict(m_t.state_dict())
    streams = build_streams(4, seed=3)
    b = WindowSampler(streams, m_t.cfg, L=64, seed=0).batch(3)
    b = {k: v.to(device) for k, v in b.items()}
    o_t, o_e = m_t(b), m_e(b)
    torch.testing.assert_close(o_e["loss"], o_t["loss"], rtol=1e-4, atol=1e-4)
    o_t["loss"].backward()
    o_e["loss"].backward()
    for (n, p_t), (_, p_e) in zip(m_t.named_parameters(), m_e.named_parameters()):
        if p_t.grad is None:
            assert p_e.grad is None or p_e.grad.abs().max() == 0, n
            continue
        torch.testing.assert_close(p_e.grad, p_t.grad, rtol=2e-3, atol=2e-4, msg=lambda m, n=n: f"{n}: {m}")


# --------------------------------------------------------------------------- CUDA-only: GPU == CPU kernels
@pytest.mark.skipif(not CUDA, reason="needs a CUDA build of evkit and a GPU")
@pytest.mark.parametrize("N", [8, 16, 32])
@pytest.mark.parametrize("fast", [False, True])
def test_cuda_scan_equals_cpu(N, fast):
    (v, dt, rate, Bm, Cm, skip, h0), ns = problem(B=3, L=257, D=40, N=N, n_save=8)
    cpu = evkit.C.scan_fwd(v, dt, rate, Bm, Cm, skip, h0, ns, False, 0)
    for mode in (1, 2):
        gpu = evkit.C.scan_fwd(*[t.cuda() for t in (v, dt, rate, Bm, Cm, skip, h0)], ns, fast, mode)
        tol = 2e-3 if fast else 2e-5
        for a, b in zip(cpu[:3], gpu[:3]):
            torch.testing.assert_close(b.cpu(), a, rtol=tol, atol=tol)


@pytest.mark.skipif(not CUDA, reason="needs a CUDA build of evkit and a GPU")
def test_cuda_tokenize_bit_identical():
    n = 100000
    g = torch.Generator().manual_seed(0)
    t = torch.sort(torch.randint(0, 9_000_000, (n,), generator=g)).values
    x, y = torch.randint(0, 1280, (n,), generator=g).short(), torch.randint(0, 720, (n,), generator=g).short()
    p = torch.randint(0, 2, (n,), generator=g).to(torch.uint8)
    a = evkit.tokenize(t, x, y, p, 1280, 720, 16, 4, seed=5)
    b = evkit.tokenize(t.cuda(), x.cuda(), y.cuda(), p.cuda(), 1280, 720, 16, 4, seed=5)
    for i, j in zip(a, b):
        assert torch.equal(i, j.cpu())
