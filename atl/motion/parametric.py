"""Parametric and dense *autonomous* velocity fields v(x; theta) (time-invariant within a window).

Two transports are provided for every model:

* ``mode="linear"``  x' = x + v(x) (t' - t)       -- the classic contrast-maximisation warp (CMax, MultiCM, ...)
* ``mode="exact"``   integrate dx/ds = v(x) from t to t' (closed-form matrix exponential for affine fields,
                     RK4 otherwise).  This is a genuine flow (group property), the setting of the theory.

Parameters are batched: ``theta`` has shape (B, k); event k uses theta[b_k].
Velocities are in pixels per (normalised) window.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from .base import Motion, as_time


class VectorFieldMotion(Motion):
    """Autonomous velocity field v(x; theta) with ``mode='linear'`` (CMax warp) or ``'exact'`` (true flow)."""
    n_params: int = 0

    @property
    def linear_at_event(self) -> bool:
        """True only for the classic linear warp x' = x + v(x) dt (enables the exact anchored pair kernel)."""
        return self.mode == "linear"

    def __init__(self, batch_size: int, height: int, width: int, mode: str = "linear", rk4_steps: int = 8,
                 init: Optional[torch.Tensor] = None):
        """True for the classic linear warp only (enables the exact anchored pair kernel)."""
        super().__init__()
        self.H, self.W, self.mode, self.rk4_steps = height, width, mode, rk4_steps
        self.c = torch.tensor([(width - 1) / 2.0, (height - 1) / 2.0])
        theta = torch.zeros(batch_size, self.n_params) if init is None else init.clone().float()
        self.theta = nn.Parameter(theta)

    # -- field ------------------------------------------------------------------------------------------
    def field(self, xy: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def velocity(self, xy, t, b):
        return self.field(xy, b)

    def affine_parts(self, b):
        """(A (N,2,2), t (N,2)) if v(x) = A (x - c) + t, else None (enables closed-form exact transport)."""
        return None

    # -- transport --------------------------------------------------------------------------------------
    def transport(self, xy, t, t_dst, b):
        dt = as_time(t_dst, t) - t
        if self.mode == "linear":
            return xy + self.field(xy, b) * dt[:, None]
        parts = self.affine_parts(b)
        if parts is not None:
            A, tr = parts
            c = self.c.to(xy)
            # exact flow of v = A(x - c) + t via the augmented exponential exp([[A, t], [0, 0]] dt)
            n = A.shape[0]
            Z = torch.zeros(n, 3, 3, device=xy.device, dtype=xy.dtype)
            Z[:, :2, :2] = A
            Z[:, :2, 2] = tr
            E = torch.linalg.matrix_exp(Z * dt[:, None, None])
            return c + torch.bmm(E[:, :2, :2], (xy - c)[..., None])[..., 0] + E[:, :2, 2]
        # generic RK4
        n = self.rk4_steps
        h = dt / n
        x = xy
        for _ in range(n):
            k1 = self.field(x, b)
            k2 = self.field(x + 0.5 * h[:, None] * k1, b)
            k3 = self.field(x + 0.5 * h[:, None] * k2, b)
            k4 = self.field(x + h[:, None] * k3, b)
            x = x + h[:, None] * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return x

    def divergence(self, xy, b, delta: float = 0.5):
        """div v at ``xy`` by central differences (used by Shiba's R_D surrogate)."""
        ex = xy.new_tensor([delta, 0.0]); ey = xy.new_tensor([0.0, delta])
        return ((self.field(xy + ex, b)[:, 0] - self.field(xy - ex, b)[:, 0]) +
                (self.field(xy + ey, b)[:, 1] - self.field(xy - ey, b)[:, 1])) / (2 * delta)


class Translation(VectorFieldMotion):
    """v = (tx, ty): constant velocity."""
    n_params = 2

    def field(self, xy, b):
        return self.theta[b]

    def affine_parts(self, b):
        z = self.theta.new_zeros(b.numel(), 2, 2)
        return z, self.theta[b]


class Zoom(VectorFieldMotion):
    """v = h (x - c): 1-DoF divergence (Shiba et al.'s canonical collapse example)."""
    n_params = 1

    def field(self, xy, b):
        return self.theta[b, :1] * (xy - self.c.to(xy))

    def affine_parts(self, b):
        h = self.theta[b, 0]
        A = torch.zeros(b.numel(), 2, 2, device=h.device, dtype=h.dtype)
        A[:, 0, 0] = h; A[:, 1, 1] = h
        return A, torch.zeros(b.numel(), 2, device=h.device, dtype=h.dtype)


class TranslationZoom(VectorFieldMotion):
    """v = (tx, ty) + h (x - c): translation that may also contract (theory toy)."""
    n_params = 3

    def field(self, xy, b):
        th = self.theta[b]
        return th[:, :2] + th[:, 2:3] * (xy - self.c.to(xy))

    def affine_parts(self, b):
        th = self.theta[b]
        A = torch.zeros(b.numel(), 2, 2, device=th.device, dtype=th.dtype)
        A[:, 0, 0] = th[:, 2]; A[:, 1, 1] = th[:, 2]
        return A, th[:, :2]


class Sim2(VectorFieldMotion):
    """v = t + s (x - c) + w J (x - c): translation, divergence, in-plane rotation."""
    n_params = 4

    def _A(self, th):
        A = torch.zeros(th.shape[0], 2, 2, device=th.device, dtype=th.dtype)
        A[:, 0, 0] = th[:, 2]; A[:, 1, 1] = th[:, 2]; A[:, 0, 1] = -th[:, 3]; A[:, 1, 0] = th[:, 3]
        return A

    def field(self, xy, b):
        th = self.theta[b]
        return th[:, :2] + torch.bmm(self._A(th), (xy - self.c.to(xy))[..., None])[..., 0]

    def affine_parts(self, b):
        th = self.theta[b]
        return self._A(th), th[:, :2]


class Affine(VectorFieldMotion):
    """v = A (x - c) + t, theta = (tx, ty, a11, a12, a21, a22)."""
    n_params = 6

    def field(self, xy, b):
        A, tr = self.affine_parts(b)
        return tr + torch.bmm(A, (xy - self.c.to(xy))[..., None])[..., 0]

    def affine_parts(self, b):
        th = self.theta[b]
        return th[:, 2:6].view(-1, 2, 2), th[:, :2]


class PlanarFlow(VectorFieldMotion):
    """8-parameter quadratic motion field of a moving plane (linearised homography), coordinates normalised by
    L = max(H, W) / 2 for conditioning.  Contains zoom, so it admits contraction paths."""
    n_params = 8

    def field(self, xy, b):
        L = max(self.H, self.W) / 2.0
        a = self.theta[b]
        u = (xy - self.c.to(xy)) / L
        x, y = u[:, 0], u[:, 1]
        vx = a[:, 0] + a[:, 1] * x + a[:, 2] * y + a[:, 6] * x * x + a[:, 7] * x * y
        vy = a[:, 3] + a[:, 4] * x + a[:, 5] * y + a[:, 6] * x * y + a[:, 7] * y * y
        return torch.stack([vx, vy], 1) * L


class Rotation3D(VectorFieldMotion):
    """Pure camera rotation with angular velocity omega (rad per window) and intrinsics (fx, fy, cx, cy).
    Motion field from the rotational interaction matrix; ``exact`` uses RK4 on the field (equivalent to the
    rotation homography up to integration error)."""
    n_params = 3

    def __init__(self, batch_size, height, width, K=None, **kw):
        super().__init__(batch_size, height, width, **kw)
        f = max(height, width)
        K = K if K is not None else (f, f, (width - 1) / 2.0, (height - 1) / 2.0)
        self.register_buffer("K", torch.tensor(K, dtype=torch.float32))

    def field(self, xy, b):
        fx, fy, cx, cy = self.K
        w = self.theta[b]
        u = (xy[:, 0] - cx) / fx
        v = (xy[:, 1] - cy) / fy
        du = u * v * w[:, 0] - (1 + u * u) * w[:, 1] + v * w[:, 2]
        dv = (1 + v * v) * w[:, 0] - u * v * w[:, 1] - u * w[:, 2]
        return torch.stack([du * fx, dv * fy], 1)


class DenseLinearFlow(VectorFieldMotion):
    """Dense velocity field on a (Hg, Wg) grid spanning the image, bilinearly interpolated (MultiCM-style; the
    grid can be refined coarse-to-fine with :meth:`upsample`)."""

    def __init__(self, batch_size, height, width, grid=(1, 1), mode="linear", init=None, rk4_steps=4):
        Motion.__init__(self)
        self.H, self.W, self.mode, self.rk4_steps = height, width, mode, rk4_steps
        self.c = torch.tensor([(width - 1) / 2.0, (height - 1) / 2.0])
        g = torch.zeros(batch_size, 2, *grid) if init is None else init.clone().float()
        self.grid = nn.Parameter(g)

    @property
    def theta(self):
        """Alias so that dense grids expose the same ``theta`` attribute as parametric models."""
        return self.grid

    def field(self, xy, b):
        Hg, Wg = self.grid.shape[-2:]
        gx = xy[:, 0] / max(self.W - 1, 1) * (Wg - 1)
        gy = xy[:, 1] / max(self.H - 1, 1) * (Hg - 1)
        return sample_grid(self.grid, torch.stack([gx, gy], 1), b)

    def affine_parts(self, b):
        return None

    def upsample(self, factor: int = 2):
        """Refine the velocity grid by ``factor`` (coarse-to-fine optimisation)."""
        with torch.no_grad():
            Hg, Wg = self.grid.shape[-2:]
            new = torch.nn.functional.interpolate(self.grid, size=(Hg * factor, Wg * factor), mode="bilinear",
                                                  align_corners=True)
        self.grid = nn.Parameter(new)
        return self

    def regulariser_fields(self, B, H, W):
        """The velocity grid itself (for smoothness regularisers)."""
        return self.grid


def sample_grid(field: torch.Tensor, gpos: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Bilinear sample of a (B, C, Hg, Wg) field at grid coordinates gpos (N, 2) = (gx, gy), border clamp.
    Returns (N, C).  Differentiable w.r.t. field and positions."""
    B, C, Hg, Wg = field.shape
    gx = gpos[:, 0].clamp(0, Wg - 1)
    gy = gpos[:, 1].clamp(0, Hg - 1)
    x0 = torch.floor(gx).clamp(max=max(Wg - 2, 0)); y0 = torch.floor(gy).clamp(max=max(Hg - 2, 0))
    ax, ay = gx - x0, gy - y0
    x0, y0 = x0.long(), y0.long()
    x1, y1 = (x0 + 1).clamp(max=Wg - 1), (y0 + 1).clamp(max=Hg - 1)
    f = field.permute(0, 2, 3, 1).reshape(-1, C)
    base = b * (Hg * Wg)
    def at(yy, xx):
        """Gather the field at integer grid indices (yy, xx) for every event."""
        return f[base + yy * Wg + xx]
    return (at(y0, x0) * ((1 - ax) * (1 - ay))[:, None] + at(y0, x1) * (ax * (1 - ay))[:, None] +
            at(y1, x0) * ((1 - ax) * ay)[:, None] + at(y1, x1) * (ax * ay)[:, None])


MODELS = {"translation": Translation, "zoom": Zoom, "translation_zoom": TranslationZoom, "sim2": Sim2,
          "affine": Affine, "planar": PlanarFlow, "rotation": Rotation3D, "dense": DenseLinearFlow}
