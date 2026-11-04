"""C++ (and, where available, CUDA) ops agree with the pure-torch references, values and gradients."""
import math

import pytest
import torch

from atl.ops import functional as F
from atl.ops import reference as R
from atl.ops._ext import get_ext

EXT = get_ext()
needs_ext = pytest.mark.skipif(EXT is None, reason="extension not built")
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def rand_events(n=500, B=2, H=20, W=24, seed=0, dev="cpu"):
    g = torch.Generator().manual_seed(seed)
    pos = torch.stack([torch.rand(n, generator=g) * (W + 4) - 2, torch.rand(n, generator=g) * (H + 4) - 2], 1)
    w = torch.rand(n, generator=g) + 0.1
    b = torch.randint(0, B, (n,), generator=g)
    t = torch.rand(n, generator=g)
    order = torch.argsort(b.double() * 4 + t.double())
    return pos[order].to(dev), w[order].to(dev), b[order].to(dev), t[order].to(dev)


@needs_ext
@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("kernel", ["bilinear", "gauss"])
def test_splat_matches_reference(dev, kernel):
    pos, w, b, _ = rand_events(dev=dev)
    B, H, W = 2, 20, 24
    pos.requires_grad_(True); w.requires_grad_(True)
    out = F.splat(pos, w, b, B, H, W, kernel=kernel, sigma=1.3)
    pos2, w2 = pos.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True)
    if kernel == "bilinear":
        ref = R.splat_bilinear(pos2, w2, b, B, H, W)
    else:
        ref = R.splat_gauss(pos2, w2, b, B, H, W, 1.3, int(math.ceil(3 * 1.3)))
    assert torch.allclose(out, ref, atol=1e-4)
    g = torch.randn_like(out)
    (out * g).sum().backward(); (ref * g).sum().backward()
    assert torch.allclose(pos.grad, pos2.grad, atol=1e-3)
    assert torch.allclose(w.grad, w2.grad, atol=1e-4)


@needs_ext
@pytest.mark.parametrize("dev", DEVS)
def test_gather_matches_reference(dev):
    pos, _, b, _ = rand_events(dev=dev)
    img = torch.randn(2, 20, 24, device=dev, requires_grad=True)
    pos.requires_grad_(True)
    out = F.gather(img, pos, b)
    img2, pos2 = img.detach().clone().requires_grad_(True), pos.detach().clone().requires_grad_(True)
    ref = R.gather_bilinear(img2, pos2, b)
    assert torch.allclose(out, ref, atol=1e-5)
    g = torch.randn_like(out)
    (out * g).sum().backward(); (ref * g).sum().backward()
    assert torch.allclose(img.grad, img2.grad, atol=1e-4)
    assert torch.allclose(pos.grad, pos2.grad, atol=1e-4)


@needs_ext
@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("tau,S", [(1.0, 8), (0.2, 16), (0.05, 4)])
def test_ats_pair_matches_bruteforce(dev, tau, S):
    pos, _, b, t = rand_events(n=600, dev=dev, seed=3)
    g = torch.Generator().manual_seed(1)
    v = (torch.randn(pos.shape[0], 2, generator=g) * 6).to(dev).requires_grad_(True)
    anchors = torch.arange(0, pos.shape[0], 3, device=dev)
    h = 1.5
    s_ext = F.ats_pair_sums(pos, t, v, b, 2, 20, 24, anchors, h, tau, cut=3.0, n_slices=S)
    v2 = v.detach().clone().requires_grad_(True)
    s_ref = R.ats_pair_sums(pos, t, v2, b, anchors, h, tau, 3.0)
    assert torch.allclose(s_ext, s_ref, atol=1e-4, rtol=1e-4), (s_ext - s_ref).abs().max()
    gS = torch.randn_like(s_ext)
    (s_ext * gS).sum().backward(); (s_ref * gS).sum().backward()
    assert torch.allclose(v.grad, v2.grad, atol=1e-4, rtol=1e-3), (v.grad - v2.grad).abs().max()


@needs_ext
@pytest.mark.parametrize("dev", DEVS)
def test_voxel_matches_reference(dev):
    pos, _, b, t = rand_events(dev=dev)
    p = torch.where(torch.rand(pos.shape[0], device=dev) > 0.5, 1.0, -1.0)
    a = F.voxel_grid(pos, t, p, b, 2, 5, 20, 24, normalize=None)
    r = R.voxel_grid(pos, t, p, b, 2, 5, 20, 24)
    assert torch.allclose(a, r, atol=1e-5)


def test_window_counts():
    t = torch.tensor([0.0, 0.1, 0.2, 0.9, 0.0, 0.5])
    b = torch.tensor([0, 0, 0, 0, 1, 1])
    c = F.window_counts(t, b, torch.arange(6), 0.15)
    assert c.tolist() == [1, 2, 1, 0, 0, 0]


def test_gauss_splat_mass_is_normalised():
    pos = torch.tensor([[10.3, 7.6]])
    img = F.splat(pos, torch.ones(1), torch.zeros(1, dtype=torch.long), 1, 20, 24, kernel="gauss", sigma=1.0)
    assert abs(img.sum().item() - 1.0) < 3e-3   # 3-sigma truncation
