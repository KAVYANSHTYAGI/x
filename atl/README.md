# `atl` package

Layers, bottom to top:

| Package | Role |
|---|---|
| `core` | event batches |
| `ops` | kernels |
| `motion` | how events move |
| `objectives` | the losses |
| `models` | the networks |
| `estimators` / `train` / `suite` / `eval` | procedures |
| `data` | loaders |

Every file is explained in [`../docs/CODE_GUIDE.md`](../docs/CODE_GUIDE.md); conventions are in
[`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md).
