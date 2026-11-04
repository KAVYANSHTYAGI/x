# Configs

| File | Purpose |
|---|---|
| `dsec_mpcm.yaml` | baseline: MotionPriorCMax loss with the official DSEC settings |
| `dsec_ats.yaml` | ours: anchored exact score; everything else identical |
| `smoke_synth.yaml` | 3-step CPU smoke test on synthetic data |

Override any key from the command line: `python scripts/train.py --config X --set loss.n_anchors=20000`.
