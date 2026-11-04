"""Packed event container used by every module.

Events of a whole batch are stored *packed* (one flat array per field plus a batch index), which is what the
splat / gather / pairwise kernels consume directly and avoids padding millions of events.

Conventions
-----------
* ``xy[:, 0]`` is the column (x), ``xy[:, 1]`` the row (y), both in (rectified) pixel units, pixel centres at
  integers.
* ``t`` is normalised time in [0, 1] inside the window of the sample; ``duration`` (seconds) converts back.
* ``p`` is polarity in {-1, +1} (float).
* ``b`` is the batch index (int64), events are **sorted by (b, t)**; ``offsets[k]:offsets[k+1]`` is sample k.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch


@dataclass
class EventPacket:
    xy: torch.Tensor                    # (N, 2) float32
    t: torch.Tensor                     # (N,)   float32 in [0, 1]
    p: torch.Tensor                     # (N,)   float32 in {-1, +1}
    b: torch.Tensor                     # (N,)   int64
    batch_size: int
    height: int
    width: int
    duration: Optional[torch.Tensor] = None   # (B,) seconds per window
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ construction
    @staticmethod
    def from_arrays(x, y, t, p, height: int, width: int, duration: float | None = None,
                    normalize_time: bool = True, device="cpu") -> "EventPacket":
        """Single sample from numpy / torch arrays (t in any unit; normalised to [0, 1] if requested)."""
        def tt(a, dt):
            return torch.as_tensor(np.asarray(a), dtype=dt, device=device) if not torch.is_tensor(a) else a.to(device, dt)
        x, y, t, p = tt(x, torch.float32), tt(y, torch.float32), tt(t, torch.float64), tt(p, torch.float32)
        if normalize_time and t.numel() > 0:
            t0, t1 = t.min(), t.max()
            t = (t - t0) / torch.clamp(t1 - t0, min=1e-12)
        t = t.to(torch.float32)
        order = torch.argsort(t, stable=True)
        p = torch.where(p > 0, 1.0, -1.0).to(torch.float32)
        n = t.numel()
        dur = None if duration is None else torch.tensor([float(duration)], device=device)
        return EventPacket(torch.stack([x, y], 1)[order], t[order], p[order],
                           torch.zeros(n, dtype=torch.long, device=device), 1, height, width, dur)

    @staticmethod
    def collate(samples: Sequence["EventPacket"]) -> "EventPacket":
        assert len(samples) > 0
        h, w = samples[0].height, samples[0].width
        xs, ts, ps, bs, durs = [], [], [], [], []
        for k, s in enumerate(samples):
            assert s.batch_size == 1 and s.height == h and s.width == w
            xs.append(s.xy); ts.append(s.t); ps.append(s.p)
            bs.append(torch.full_like(s.b, k))
            durs.append(s.duration if s.duration is not None else torch.ones(1, device=s.t.device))
        return EventPacket(torch.cat(xs), torch.cat(ts), torch.cat(ps), torch.cat(bs), len(samples), h, w,
                           torch.cat(durs))

    # ------------------------------------------------------------------ utilities
    @property
    def n(self) -> int:
        return int(self.t.numel())

    @property
    def device(self):
        return self.t.device

    @property
    def offsets(self) -> torch.Tensor:
        counts = torch.bincount(self.b, minlength=self.batch_size)
        return torch.cat([counts.new_zeros(1), torch.cumsum(counts, 0)])

    def to(self, device) -> "EventPacket":
        d = None if self.duration is None else self.duration.to(device)
        return EventPacket(self.xy.to(device), self.t.to(device), self.p.to(device), self.b.to(device),
                           self.batch_size, self.height, self.width, d, dict(self.meta))

    def select(self, mask_or_index: torch.Tensor) -> "EventPacket":
        """Subset of events (keeps batch structure; order preserved)."""
        m = mask_or_index
        return EventPacket(self.xy[m], self.t[m], self.p[m], self.b[m], self.batch_size, self.height,
                           self.width, self.duration, dict(self.meta))

    def split(self, frac: float, generator: Optional[torch.Generator] = None):
        """Random disjoint split (e.g. held-out events for the unsupervised metric)."""
        u = torch.rand(self.n, generator=generator, device="cpu").to(self.device)
        return self.select(u < frac), self.select(u >= frac)

    def subsample(self, k: int, generator: Optional[torch.Generator] = None) -> "EventPacket":
        if self.n <= k:
            return self
        idx = torch.randperm(self.n, generator=generator, device="cpu")[:k].to(self.device)
        idx, _ = torch.sort(idx)
        return self.select(idx)

    def inside(self, xy: Optional[torch.Tensor] = None, margin: float = 0.0) -> torch.Tensor:
        xy = self.xy if xy is None else xy
        return ((xy[:, 0] >= -margin) & (xy[:, 0] <= self.width - 1 + margin) &
                (xy[:, 1] >= -margin) & (xy[:, 1] <= self.height - 1 + margin))

    def __repr__(self):
        return (f"EventPacket(n={self.n}, B={self.batch_size}, HxW={self.height}x{self.width}, "
                f"device={self.device})")
