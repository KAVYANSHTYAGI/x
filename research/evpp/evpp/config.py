"""Configuration for the event point-process models (EvPP).

One dataclass drives every variant; `objective` picks the likelihood:

    "ar"        autoregressive per-event model, time ~ mixture of log-normals (intensity-free)
    "intensity" same backbone, time via a continuous-state ground intensity + quadrature compensator
    "physics"   ground intensity AND marks derived from per-patch brightness-change rates / threshold
    (the patch-Poisson count model, V1, is a separate class: see models/evpp.py::PatchPoisson)
"""
from dataclasses import dataclass, asdict


@dataclass
class EvPPConfig:
    # --- event quantiser (sensor-agnostic: works on normalised coordinates) -------------------
    grid: int = 8            # G: patches per side            -> K = G*G patches
    sub: int = 4             # Q: sub-cells per patch side    -> S = Q*Q cells per patch
    n_freq: int = 4          # Fourier frequencies for normalised (u, v) coordinates
    # --- backbone ---------------------------------------------------------------------------
    d_model: int = 64
    d_state: int = 16
    n_layers: int = 3
    d_int: int = 16          # channels of the last layer exposed to the continuous-time readout
    tmin: float = 1e-4       # fastest SSM timescale (seconds)
    tmax: float = 1.0        # slowest SSM timescale (seconds)
    use_clock: bool = True   # learned latent clock (tempo) rescaling the elapsed time
    clock_dim: int = 8
    clock_beta: float = 0.95 # count-based EMA decay inside the clock
    clock_gauge: float = 0.0 # weight of (mean log tau)^2: fixes the r <-> tau scale symmetry (recommend 1e-2)
    cond_dim: int = 0        # >0: extra conditioning vector (class, sensor embedding, ...)
    impl: str = "torch"      # "torch" (pure PyTorch reference) | "evkit" (fused C++/CUDA kernels)
    # --- objective --------------------------------------------------------------------------
    objective: str = "ar"
    n_mix: int = 8           # log-normal mixture components ("ar")
    n_quad: int = 8          # Gauss-Legendre nodes in log-time for the compensator
    quad_tmin: float = 1e-6  # seconds; below this the gap integral uses a single midpoint
    dt_quantum: float = 1e-6 # timestamp resolution of the sensor (dequantisation width)
    init_log_rate: float = 9.0   # ln(Hz) used to initialise the time head biases
    # --- sensor parameters (objective == "physics") -----------------------------------------
    sensor_mode: str = "learned"  # "learned" | "oracle" (batch["theta"]) | "infer" (SensorEncoder)
    sensor_ctx: int = 32          # burn-in events used by the SensorEncoder (excluded from the loss)

    @property
    def n_patches(self) -> int:
        return self.grid * self.grid

    @property
    def n_sub(self) -> int:
        return self.sub * self.sub

    def to_dict(self):
        return asdict(self)
