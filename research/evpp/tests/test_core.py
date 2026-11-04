import math
import numpy as np
import pytest
import torch

from conftest import tiny_cfg, make_batch
from evpp import EvPP, PatchPoisson, HomogeneousPoisson
from evpp.models.scan import scan_ref, scan_assoc
from evpp.models.heads import gauss_legendre_01, integrate_gap


# ------------------------------------------------------------------ scan
@pytest.mark.parametrize("L", [1, 2, 7, 33, 64])
def test_scan_assoc_matches_reference(L):
    torch.manual_seed(L)
    a = torch.rand(3, L, 4, 5).clamp(0.05, 1.0)
    b = torch.randn(3, L, 4, 5)
    h0 = torch.randn(3, 4, 5)
    for h in (None, h0):
        assert torch.allclose(scan_ref(a, b, h), scan_assoc(a, b, h), atol=1e-5, rtol=1e-4)


def test_scan_gradients_match():
    a = torch.rand(2, 17, 3).clamp(0.1, 1.0).requires_grad_()
    b = torch.randn(2, 17, 3, requires_grad=True)
    g1 = torch.autograd.grad(scan_ref(a, b).pow(2).sum(), (a, b))
    g2 = torch.autograd.grad(scan_assoc(a, b).pow(2).sum(), (a, b))
    for x, y in zip(g1, g2):
        assert torch.allclose(x, y, atol=1e-4, rtol=1e-3)


# ------------------------------------------------------------------ quadrature
def test_gap_quadrature_exact_for_exponential():
    """int_0^dt  r exp(-r tau) dtau = 1 - exp(-r dt), with a scale (clock) factor, across 6 decades of dt."""
    nodes, w = gauss_legendre_01(12)
    r, scale = 800.0, torch.tensor([[1.0, 2.0, 0.5, 1.0, 1.0]])
    dt = torch.tensor([[1e-7, 1e-5, 1e-4, 3e-3, 0.4]])
    Lam = lambda tau: r * torch.exp(-r * tau)
    got = integrate_gap(Lam, dt, scale, nodes, w, 1e-6)
    want = (1 - torch.exp(-r * scale * dt)) / scale            # physical-time integral of Lam(scale*tau)
    assert torch.allclose(got, want, rtol=2e-3, atol=1e-9), (got, want)


# ------------------------------------------------------------------ normalisation
def test_space_head_is_normalised():
    cfg = tiny_cfg()
    m = EvPP(cfg)
    z = torch.randn(cfg.d_model)
    lp = m.space.log_prob_all(z)
    assert lp.shape == (cfg.n_patches, 2, cfg.n_sub)
    assert abs(lp.exp().sum().item() - 1.0) < 1e-5


def test_lognormal_mixture_integrates_to_one():
    cfg = tiny_cfg(objective="ar")
    m = EvPP(cfg)
    s = torch.randn(1, cfg.d_model)
    x = torch.linspace(-30, 8, 200001)                                   # integrate over log dt
    dt = x.exp()[None]
    p_dt = (-m.time.nll(s.expand(dt.shape[1], -1)[None], dt)).exp()[0]  # density in dt
    total = torch.trapz(p_dt * dt[0], x)                                 # d dt = dt d(log dt)
    assert abs(total.item() - 1.0) < 5e-3, total


# ------------------------------------------------------------------ causality
@pytest.mark.parametrize("obj,mode", [("ar", "learned"), ("intensity", "learned"),
                                      ("physics", "learned"), ("physics", "infer")])
def test_per_event_nll_is_causal(streams, obj, mode):
    cfg = tiny_cfg(objective=obj, sensor_mode=mode, sensor_ctx=8)
    m = EvPP(cfg).eval()
    b = make_batch(streams, cfg, L=40, n=3, with_theta=(mode != "learned"))
    j = 25
    b2 = {k: v.clone() for k, v in b.items()}
    for k in ("patch", "sub", "pol"):
        b2[k][:, j:] = (b2[k][:, j:] + 1) % (2 if k == "pol" else 4)
    b2["dt"][:, j:] = b2["dt"][:, j:] * 3.7                              # change the future arbitrarily
    with torch.no_grad():
        o1, o2 = m(b), m(b2)
    for name in o1["per_event"]:
        a, c = o1["per_event"][name][:, :j], o2["per_event"][name][:, :j]
        assert torch.allclose(a, c, atol=1e-5), f"{obj}/{mode}: '{name}' leaks the future"


# ------------------------------------------------------------------ streaming
@pytest.mark.parametrize("use_clock", [True, False])
def test_chunked_state_equals_full_pass(streams, use_clock):
    cfg = tiny_cfg(objective="ar", use_clock=use_clock)
    m = EvPP(cfg).eval()
    b = make_batch(streams, cfg, L=48, n=2)
    cut = 20
    two_d = {k: v for k, v in b.items() if v.dim() == 2}
    first = {k: v[:, :cut] for k, v in two_d.items()}
    second = {k: v[:, cut:] for k, v in two_d.items()}
    with torch.no_grad():
        full = m(b)
        o1 = m(first)
        o2 = m(second, state=o1["state"])
    for name in full["per_event"]:
        cat = torch.cat([o1["per_event"][name], o2["per_event"][name]], 1)
        assert torch.allclose(cat, full["per_event"][name], atol=1e-4, rtol=1e-4), name


# ------------------------------------------------------------------ misc
def test_clock_is_identity_at_init(streams):
    cfg = tiny_cfg()
    m = EvPP(cfg)
    b = make_batch(streams, cfg)
    assert torch.allclose(m.encode(b)["log_tau"], torch.zeros_like(b["dt"]))


def test_homogeneous_poisson_is_dilation_equivariant(streams):
    cfg = tiny_cfg()
    b = make_batch(streams, cfg, L=64, n=4)
    base = HomogeneousPoisson(cfg)
    f = 3.0
    b2 = dict(b); b2["dt"] = b["dt"] / f
    gap = (base.nll(b2) - base.nll(b)).item() + math.log(f)              # (nll' - nll) per event + ln f
    assert abs(gap) < 1e-4


def test_generation_shapes():
    cfg = tiny_cfg(objective="ar")
    m = EvPP(cfg).eval()
    out = m.sample(12, B=2)
    assert out["dt"].shape == (2, 12) and (out["dt"] > 0).all()
    assert out["patch"].max() < cfg.n_patches


def test_patch_poisson_runs_and_is_finite(streams):
    cfg = tiny_cfg()
    pp = PatchPoisson(cfg, bin_width=2e-4)
    b = make_batch(streams, cfg, L=64, n=3)
    o = pp(b)
    assert torch.isfinite(o["loss"])
    o["loss"].backward()


# ------------------------------------------------------------------ review follow-ups
@pytest.mark.parametrize("obj", ["ar", "intensity", "physics"])
def test_context_for_event_i_ignores_event_i_own_marks(streams, obj):
    """The context used to predict event j (s_{j-1}, h_{j-1}, tau_{j-1}, and dt_eff_j = tau_{j-1} dt_j) must not
    depend on event j's own patch / sub-cell / polarity.  Only its gap dt_j is teacher-forced."""
    cfg = tiny_cfg(objective=obj)
    m = EvPP(cfg).eval()
    b = make_batch(streams, cfg, L=40, n=3)
    j = 17
    b2 = {k: v.clone() for k, v in b.items()}
    b2["patch"][:, j] = (b2["patch"][:, j] + 3) % cfg.n_patches
    b2["sub"][:, j] = (b2["sub"][:, j] + 1) % cfg.n_sub
    b2["pol"][:, j] = 1 - b2["pol"][:, j]
    b2["u"][:, j] = 1.0 - b2["u"][:, j]
    with torch.no_grad():
        e1, e2 = m.encode(b), m.encode(b2)
    for key in ("s_prev", "H_prev", "log_tau_prev", "dt_eff"):
        assert torch.allclose(e1[key][:, :j + 1], e2[key][:, :j + 1], atol=1e-6), f"{obj}: {key} at j sees event j"
    assert not torch.allclose(e1["s_prev"][:, j + 1], e2["s_prev"][:, j + 1])   # ...but j+1 does see it


def test_dilation_gap_matches_analytic_fixed_rate_poisson(streams):
    """For a FIXED-rate Poisson model (not refit), nll/event = -ln(lam) + lam*mean(dt) + ln(2KS).
    Dividing gaps by f gives gap = lam*mean(dt)*(1/f - 1) + ln f  (non-zero), and the MLE refit gives 0."""
    from evpp.eval.diagnostics import dilation_gap
    cfg = tiny_cfg()
    b = make_batch(streams, cfg, L=64, n=4)
    lam, f = 2.5e4, 3.0
    marks = math.log(2 * cfg.n_patches * cfg.n_sub)

    class FixedPoisson:
        def nll(self, batch):
            m = batch["mask"].float()
            return (-math.log(lam) + lam * batch["dt"] * m).sum() / m.sum() + marks

    mean_dt = float((b["dt"] * b["mask"]).sum() / b["mask"].sum())
    want = lam * mean_dt * (1.0 / f - 1.0) + math.log(f)
    assert abs(dilation_gap(FixedPoisson(), b, f) - want) < 1e-5
    assert abs(dilation_gap(HomogeneousPoisson(cfg), b, f)) < 1e-4
