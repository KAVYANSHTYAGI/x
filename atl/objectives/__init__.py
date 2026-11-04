"""Alignment objectives: reference-frame baselines (CMax family, MotionPriorCMax) and anchored scores (ours).

``build(name, **kw)`` is the registry used by configs, the trainer and the failure suite."""
from .ats import AnchoredExact, AnchoredSliced, flow_warp_loss, heldout_score, rsat
from .base import IWEConfig, Objective, make_iwe
from .cmax import EMin, Regularized, Sharpness, STPPP, Timestamp, charbonnier_tv
from .mpcm import MotionPriorCMaxLoss


def build(name: str, **kw) -> Objective:
    """Registry used by configs, the failure suite and the trainer."""
    reg = {
        # ---- baselines (reference-frame)
        "cmax_var": lambda: Sharpness("variance", t_ref=kw.get("t_ref", 0.0)),
        "cmax_multiref": lambda: Sharpness("variance", t_ref=kw.get("t_refs", [0.0, 0.5, 1.0]), name="cmax_multiref"),
        "cmax_gradmag_rand": lambda: Sharpness("grad_l1", t_ref="random", name="gradmag_randref",
                                               iwe=IWEConfig(polarity="split", scale_by_dt=True)),
        "shiba_rd": lambda: Regularized(Sharpness("variance", t_ref=kw.get("t_ref", 0.0)), "div", kw.get("weight", 1.0)),
        "shiba_ra": lambda: Regularized(Sharpness("variance", t_ref=kw.get("t_ref", 0.0)), "det", kw.get("weight", 1.0)),
        "emin": lambda: EMin(t_ref=kw.get("t_ref", 0.0)),
        "stppp": lambda: STPPP(t_ref=kw.get("t_ref", 0.0)),
        "zhu_ts": lambda: Timestamp(taming=False),
        "taming_ts": lambda: Timestamp(taming=True),
        "mpcm": lambda: MotionPriorCMaxLoss(**{k: v for k, v in kw.items() if k in (
            "num_bins", "num_knn", "smooth_weight", "lut", "t_ref", "scale_by_dt", "polarity_split", "sigma")}),
        # ---- ours (anchored)
        "ats_exact": lambda: AnchoredExact(**{k: v for k, v in kw.items() if k in (
            "h", "tau", "eps_frac", "n_anchors", "n_slices", "seed", "smooth_anchors", "dequantize", "visibility",
            "anchor_margin")}),
        "ats_sliced": lambda: AnchoredSliced(**{k: v for k, v in kw.items() if k in (
            "n_slices", "slices_per_step", "sigma", "tau", "eps_frac", "n_anchors_per_slice", "seed")}),
    }
    if name not in reg:
        raise KeyError(f"unknown objective {name}; available: {sorted(reg)}")
    return reg[name]()


__all__ = ["Objective", "IWEConfig", "make_iwe", "Sharpness", "EMin", "STPPP", "Timestamp", "Regularized",
           "charbonnier_tv", "MotionPriorCMaxLoss", "AnchoredExact", "AnchoredSliced", "heldout_score",
           "flow_warp_loss", "rsat", "build"]
