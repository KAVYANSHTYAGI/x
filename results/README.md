# Results (committed evidence)

| Path | Contents | Command |
|---|---|---|
| `suite/` | F1–F4 tables | `scripts/experiments/run_suite.py` |
| `dense/diag_<scene>_<objective>.txt` | held-out interior / border bias of fitted dense fields (main synthetic result) | `diag_dense_drift.py` |
| `dense/exp_dense_*` | F6 runs: v1 in-sample (superseded), v3 with the smoothing fix | |
| `dsec/` | real-DSEC per-window comparison | `exp_dsec_modelbased.py` |
| `logs/` | raw logs of the F6 runs | |

Interpretation: [`../docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).
