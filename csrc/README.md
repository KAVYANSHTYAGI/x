# C++ / CUDA kernels

| File | Contents |
|---|---|
| `atl_ops.h` | declarations and layouts |
| `atl_cpu.cpp` | OpenMP loops (the specification) + pybind11 bindings |
| `atl_cuda.cu` | the same loops, one thread per event or anchor |

Built automatically on first use by `atl/ops/_ext.py`. Details: [`../docs/KERNELS.md`](../docs/KERNELS.md).
