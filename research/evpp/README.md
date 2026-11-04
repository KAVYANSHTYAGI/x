# evpp - sketch code for raw-event point-process pretraining

Pure PyTorch (>=2.1), no PyG. ~1.9k lines including tests. See `METHOD_SKETCHES.md` for the math,
what is verified, and the weak points; `../event_pp_atlas.md` for the literature.

```
evpp/
  config.py             EvPPConfig (one dataclass drives every variant)
  data/stream.py        Stream, Quantizer (normalised coords -> patch + sub-cell), tokenize, windows, collate
  data/transforms.py    dilate / thin / densify / add_noise / reverse / shuffle / polshuffle
  data/sim.py           ESIM-style simulator with explicit (speed, threshold C, noise) knobs, 4 motion classes
  models/scan.py        linear-recurrence scans (reference + log-depth)
  models/ssm.py         event-driven continuous-time SSM, latent clock, stack
  models/heads.py       LogNormalMix, IntensityTime, PhysicsIntensity, SpaceHead, log-time quadrature
  models/evpp.py        EvPP (ar | intensity | physics), SensorEncoder, PatchPoisson, HomogeneousPoisson
  models/ssl_baselines.py  MaskedEventModel, NextCountPrediction (same backbone)
  eval/diagnostics.py   time-rescaling KS, dilation gap, temporal information gain
  eval/probe.py         frozen-feature linear probe
  eval/sprt.py          class-conditional anytime posterior + stopping rule
  training.py           minimal training loop, WindowSampler
scripts/                compare_objectives.py  run_diagnostics.py  sprt_demo.py  sensor_id_demo.py
tests/                  27 tests
```

```bash
pip install -r requirements.txt
python -m pytest -q tests                                   # ~16 s on CPU
python scripts/run_diagnostics.py --objective intensity --steps 300
python scripts/compare_objectives.py --steps 300            # R1 protocol skeleton (toy data)
python scripts/sprt_demo.py                                 # R5
python scripts/sensor_id_demo.py                            # R2
```

Minimal use:

```python
from evpp import EvPPConfig, EvPP
from evpp.training import build_streams, WindowSampler, train

cfg = EvPPConfig(objective="intensity")          # "ar" | "intensity" | "physics"
model = EvPP(cfg)
streams = build_streams(64)                      # replace with your real Stream objects
train(model, WindowSampler(streams, cfg, L=128), steps=300)
```

To use real data, produce `evpp.data.stream.Stream(x, y, t_seconds, polarity, W, H, label)` objects;
everything downstream (quantiser, windows, diagnostics, probes) works unchanged.

Fast backend: `EvPPConfig(..., impl="evkit")` runs the SSM scan and the compensator readout on the fused
C++/CUDA kernels in `../evkit` (same loss and gradients as the PyTorch path, tested).

**The toy simulator results are not evidence.** Everything printed by the scripts is a pipeline check.
