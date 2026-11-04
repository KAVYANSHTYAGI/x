"""Generative classification with sequential stopping (R5).

Train ONE EvPP with a class-conditioning vector (cfg.cond_dim = n_classes, one-hot).  Then for a
stream x_{1:n} and every class c the model gives per-event log-likelihoods ll_c(i), so the running
log posterior is
        log P(c | x_{1:i}) = log pi_c + sum_{j<=i} ll_c(j) - logsumexp_c' (...)
and a sequential probability ratio rule stops at the first i where max_c P(c | x_{1:i}) >= 1 - alpha.
Latency is then measured in EVENTS (or seconds via the cumulative gaps) and scales with the rate at
which evidence arrives - exactly the "rate as information arrival" reading.
"""
from typing import Dict

import torch


@torch.no_grad()
def anytime_log_posterior(model, batch: Dict[str, torch.Tensor], n_classes: int, log_prior=None):
    """Returns (B, C, L) running log-posterior.  Padded positions repeat the last valid value."""
    model.eval()
    B, L = batch["dt"].shape
    lls = []
    for c in range(n_classes):
        b = dict(batch)
        b["cond"] = torch.nn.functional.one_hot(torch.full((B,), c), n_classes).float().to(batch["dt"].device)
        o = model(b)
        ll = -sum(o["per_event"].values())                            # (B,L) per-event log-likelihood
        lls.append((ll * batch["mask"]).cumsum(1))
    cum = torch.stack(lls, 1)                                         # (B,C,L)
    if log_prior is not None:
        cum = cum + log_prior[None, :, None]
    return cum - torch.logsumexp(cum, dim=1, keepdim=True)


def stopping_index(log_post: torch.Tensor, mask: torch.Tensor, alpha: float = 0.05):
    """First position where the max posterior >= 1-alpha, else the last valid position.  Returns (idx, pred)."""
    conf = log_post.max(1).values.exp()                               # (B,L)
    hit = (conf >= 1 - alpha) & mask
    L = conf.shape[1]
    last = mask.long().sum(1) - 1
    first = torch.where(hit.any(1), hit.float().argmax(1), last)
    pred = log_post[torch.arange(len(first)), :, first].argmax(1)
    return first, pred
