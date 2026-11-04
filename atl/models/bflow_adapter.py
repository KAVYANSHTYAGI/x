"""Adapter for MotionPriorCMax's EVIMO2 backbone (RAFT-spline / BFlow, src/models/raft_spline in the official repo).

Their ``BezierCurves`` hold (B, 2 (n_ctrl - 1), h, w) parameters: control points 1..K of a degree-K Bezier curve
whose control point 0 is pinned at the pixel (zero displacement), i.e. displacement(t) = sum_k B_{k,K}(t) P_k --
exactly ``atl.motion.PixelTrajectories(basis_kind="bezier", K)``.  This module converts the parameters so the same
backbone can be trained with any atl loss on EVIMO2.

VERIFY ON FIRST USE: the official per-point channel order is assumed to be (x, y) interleaved per control point.
``check_against_official`` compares displacements with their curve implementation and must pass before training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from ..motion import PixelTrajectories


def bezier_params_to_coeff_grid(params: torch.Tensor, out_hw=None) -> torch.Tensor:
    """(B, 2K, h, w) official Bezier params [(x1,y1),(x2,y2),...] -> atl coefficient grid (B, 2K, H, W) [y1..yK, x1..xK].
    Optional bilinear upsampling to the event resolution (displacements are in pixels of the OUTPUT grid: if the
    official parameters live on a downsampled grid, scale them as their code does before calling this)."""
    B, C, h, w = params.shape
    K = C // 2
    p = params.view(B, K, 2, h, w)
    grid = torch.cat([p[:, :, 1], p[:, :, 0]], 1)
    if out_hw is not None and (h, w) != tuple(out_hw):
        grid = F.interpolate(grid, size=out_hw, mode="bilinear", align_corners=False)
    return grid


def trajectories_from_bflow(params: torch.Tensor, out_hw, tile: int = 4) -> PixelTrajectories:
    """Wrap official Bezier parameters as :class:`PixelTrajectories` (any atl loss can then train them)."""
    K = params.shape[1] // 2
    return PixelTrajectories(bezier_params_to_coeff_grid(params, out_hw), K=K, basis_kind="bezier", tile=tile)


def load_official_raft_spline(repo: str = "/home/user/third_party/MotionPriorCMax"):
    """Import the official RAFTSpline class (needs their requirements, e.g. numba)."""
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src.models.raft_spline.raft import RAFTSpline  # type: ignore
    return RAFTSpline


@torch.no_grad()
def check_against_official(params: torch.Tensor, repo: str = "/home/user/third_party/MotionPriorCMax", atol=1e-4):
    """Assert that our conversion reproduces the official curve displacements (run once before training)."""
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from src.models.raft_spline.curves.bezier import BezierCurves  # type: ignore
    curves = BezierCurves(params)
    ts = [0.25, 0.5, 1.0]
    traj = trajectories_from_bflow(params, params.shape[-2:], tile=1)
    for t in ts:
        theirs = curves.get_flow_from_reference(t)                 # (B, 2, h, w) assumed (x, y)
        ours = traj.flow_field(params.shape[0], *params.shape[-2:], 0.0, t)
        if not torch.allclose(theirs.float(), ours, atol=atol):
            raise AssertionError(f"channel-order / basis mismatch at t={t}: max diff {(theirs - ours).abs().max()}")
    return True
