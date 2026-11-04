"""Autograd wrappers.  Device dispatch happens in C++ (CPU kernels or CUDA kernels by tensor device)."""
from typing import Optional

import torch

from ._loader import C


class _DecayScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, dt, rate, Bm, Cm, skip, h0, n_save, fast_exp, mode):
        args = [t.contiguous().float() for t in (v, dt, rate, Bm, Cm, skip)]
        h0c = h0.contiguous().float() if h0 is not None else None
        y, H, h_last, ckpt = C.scan_fwd(*args, h0c, int(n_save), bool(fast_exp), int(mode))
        ctx.save_for_backward(*args, h0c if h0c is not None else torch.empty(0, device=v.device), ckpt)
        ctx.has_h0, ctx.n_save, ctx.fast, ctx.mode = h0 is not None, int(n_save), bool(fast_exp), int(mode)
        return y, H, h_last

    @staticmethod
    def backward(ctx, gy, gH, ghl):
        v, dt, rate, Bm, Cm, skip, h0, ckpt = ctx.saved_tensors
        gy = torch.zeros_like(v) if gy is None else gy.contiguous().float()
        gH = None if (gH is None or ctx.n_save == 0) else gH.contiguous().float()
        ghl = None if ghl is None else ghl.contiguous().float()
        gv, gdt, grate, gB, gC, gskip, gh0 = C.scan_bwd(v, dt, rate, Bm, Cm, skip, h0 if ctx.has_h0 else None, ckpt,
                                                       gy, gH, ghl, ctx.n_save, ctx.fast, ctx.mode)
        return gv, gdt, grate, gB, gC, gskip, (gh0 if ctx.has_h0 else None), None, None, None


def decay_scan(v, dt, rate, Bm, Cm, skip, h0: Optional[torch.Tensor] = None, n_save: int = 0,
               fast_exp: bool = True, mode: int = 0):
    """Event-driven diagonal SSM scan.  v (B,L,D), dt (B,L), rate/Bm/Cm (D,N), skip (D), h0 (B,D,N) or None.
    Returns y (B,L,D) = sum_n Cm h + skip v, H_save (B,L,n_save,N), h_last (B,D,N).
    mode: 0 auto, 1 sequential, 2 chunk-parallel (CUDA only; ignored on CPU)."""
    return _DecayScan.apply(v, dt, rate, Bm, Cm, skip, h0, n_save, fast_exp, mode)


class _Readout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, tau, rate, Cm, fast_exp):
        H, tau, rate, Cm = [t.contiguous().float() for t in (H, tau, rate, Cm)]
        ctx.save_for_backward(H, tau, rate, Cm)
        ctx.fast = bool(fast_exp)
        return C.readout_fwd(H, tau, rate, Cm, ctx.fast)

    @staticmethod
    def backward(ctx, gY):
        H, tau, rate, Cm = ctx.saved_tensors
        gH, gtau, grate, gC = C.readout_bwd(H, tau, rate, Cm, gY.contiguous().float(), ctx.fast)
        return gH, gtau, grate, gC, None


def readout(H, tau, rate, Cm, fast_exp: bool = True):
    """Y[b,l,s,d] = sum_m Cm[d,m] exp(-rate[d,m] tau[b,l,s]) H[b,l,d,m].
    H (B,L,n,N); tau (B,L,S) -> (B,L,S,n)  or  tau (B,L) -> (B,L,n)."""
    if tau.dim() == 2:
        return _Readout.apply(H, tau[..., None], rate, Cm, fast_exp)[:, :, 0]
    return _Readout.apply(H, tau, rate, Cm, fast_exp)


def tokenize(t_us, x, y, p, width, height, grid, sub, seed=0, index_offset=0, t_prev=-1):
    """Raw events -> (patch, sub, pol, u, v, dt).  Bit-identical on CPU and CUDA for the same seed."""
    return C.tokenize(t_us.long(), x.short(), y.short(), p.to(torch.uint8), int(width), int(height), int(grid),
                      int(sub), int(seed), int(index_offset), int(t_prev))


def decode_file(path: str, n_threads: int = 0, format: str = ""):
    """Decode .raw (EVT2/EVT3) or .dat into time-sorted tensors {t, x, y, p, width, height, header}."""
    return C.decode_file(path, int(n_threads), format)


def write_evb(path, t, x, y, p, width, height, sensor=""):
    C.write_evb(path, t.long(), x.short(), y.short(), p.to(torch.uint8), int(width), int(height), sensor)


EvbReader = C.EvbReader
