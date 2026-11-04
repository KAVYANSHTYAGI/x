"""Import the compiled extension; fall back to a JIT build (torch.utils.cpp_extension.load)."""
import glob
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _jit_build():
    import torch
    from torch.utils.cpp_extension import load, CUDA_HOME
    srcs = [os.path.join(_ROOT, "torch", "evkit_torch.cpp")] + sorted(glob.glob(os.path.join(_ROOT, "src", "*.cpp")))
    with_cuda = torch.cuda.is_available() and CUDA_HOME is not None and os.environ.get("EVKIT_NO_CUDA") != "1"
    kw = dict(extra_include_paths=[os.path.join(_ROOT, "include")],
              extra_cflags=["-O3", "-march=native"], verbose=os.environ.get("EVKIT_VERBOSE") == "1")
    if with_cuda:
        srcs += sorted(glob.glob(os.path.join(_ROOT, "cuda", "*.cu")))
        kw["extra_cflags"].append("-DEVKIT_WITH_CUDA")
        kw["extra_cuda_cflags"] = ["-O3", "--use_fast_math", "-lineinfo", "-DEVKIT_WITH_CUDA"]
    return load(name="evkit_C" + ("_cuda" if with_cuda else "_cpu"), sources=srcs, **kw)


try:
    from . import _C as C  # built by setup.py
except ImportError:
    C = _jit_build()
