"""Model-based estimation: optimise motion parameters of one (batch of) window(s) under any objective."""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import torch


def fit(objective, motion_fn: Callable[[], object], params: List[torch.Tensor], ev, steps: int = 200,
        lr: float = 0.05, optimizer: str = "adam", callback: Optional[Callable[[int, object, dict], None]] = None,
        log_every: int = 0, grad_clip: Optional[float] = None) -> Dict[str, list]:
    """``motion_fn()`` rebuilds the motion from ``params`` (so derived models such as PixelTrajectories work).
    Returns the history of the loss (and whatever ``callback`` records)."""
    if optimizer == "adam":
        opt = torch.optim.Adam(params, lr=lr)
    elif optimizer == "lbfgs":
        opt = torch.optim.LBFGS(params, lr=lr, max_iter=20, line_search_fn="strong_wolfe")
    else:
        raise ValueError(optimizer)
    hist = {"loss": [], "time": []}
    t0 = time.time()
    for it in range(steps):
        def closure():
            """One objective evaluation + backward (LBFGS needs a closure; Adam uses it once)."""
            opt.zero_grad()
            out = objective(ev, motion_fn())
            out["loss"].backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            closure.out = out
            return out["loss"]
        opt.step(closure)
        out = closure.out
        hist["loss"].append(float(out["loss"].detach()))
        hist["time"].append(time.time() - t0)
        if callback is not None:
            callback(it, motion_fn(), hist)
        if log_every and (it % log_every == 0 or it == steps - 1):
            print(f"  it {it:4d} loss {hist['loss'][-1]:+.5f}")
    return hist
