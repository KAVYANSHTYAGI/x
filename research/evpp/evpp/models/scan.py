"""First-order linear recurrences  h_i = a_i * h_{i-1} + b_i  (elementwise), two implementations.

`scan_ref`   sequential reference (O(L) python loop) - used for tests and as the streaming kernel.
`scan_assoc` Hillis-Steele inclusive scan: O(L log L) work but log-depth, no python loop over L.

Both accept an initial state h0 (B, ...) so that long streams can be processed in chunks.
Inputs a, b: (B, L, ...) with 0 < a <= 1 (decays), so products never overflow.
"""
import torch


def scan_ref(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor = None) -> torch.Tensor:
    B, L = a.shape[:2]
    h = torch.zeros_like(b[:, 0]) if h0 is None else h0
    out = []
    for i in range(L):
        h = a[:, i] * h + b[:, i]
        out.append(h)
    return torch.stack(out, dim=1)


def scan_assoc(a: torch.Tensor, b: torch.Tensor, h0: torch.Tensor = None) -> torch.Tensor:
    L = a.shape[1]
    A, Bv = a, b
    d = 1
    while d < L:
        A_prev = torch.cat([torch.ones_like(A[:, :d]), A[:, :-d]], dim=1)
        B_prev = torch.cat([torch.zeros_like(Bv[:, :d]), Bv[:, :-d]], dim=1)
        Bv = A * B_prev + Bv            # uses the *old* A on purpose
        A = A * A_prev
        d *= 2
    if h0 is not None:
        Bv = Bv + A * h0.unsqueeze(1)   # A_i is now prod_{r<=i} a_r
    return Bv


def scan(a, b, h0=None, impl: str = "assoc"):
    return (scan_assoc if impl == "assoc" else scan_ref)(a, b, h0)
