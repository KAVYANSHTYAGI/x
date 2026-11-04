"""JIT build + load of the C++/CUDA extension (csrc/).  Falls back to the pure-torch reference ops if the build is
unavailable or ``ATL_FORCE_REF=1``.  Build artefacts go to ``$ATL_BUILD_DIR`` (default ~/.cache/atl_ext)."""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import torch

_EXT = None
_TRIED = False
CSRC = Path(__file__).resolve().parents[2] / "csrc"


def get_ext():
    """Return the compiled module or None."""
    global _EXT, _TRIED
    if _TRIED:
        return _EXT
    _TRIED = True
    if os.environ.get("ATL_FORCE_REF", "0") == "1":
        return None
    try:
        from torch.utils.cpp_extension import load
        with_cuda = torch.cuda.is_available() and os.environ.get("ATL_NO_CUDA", "0") != "1"
        sources = [str(CSRC / "atl_cpu.cpp")]
        cflags = ["-O3", "-fopenmp"]
        if with_cuda:
            sources.append(str(CSRC / "atl_cuda.cu"))
            cflags.append("-DATL_WITH_CUDA")
        build_dir = Path(os.environ.get("ATL_BUILD_DIR", Path.home() / ".cache" / "atl_ext"))
        build_dir = build_dir / ("cuda" if with_cuda else "cpu")
        build_dir.mkdir(parents=True, exist_ok=True)
        _EXT = load(name="atl_ops_cuda" if with_cuda else "atl_ops_cpu", sources=sources,
                    extra_cflags=cflags, extra_ldflags=["-fopenmp"],
                    extra_cuda_cflags=["-O3", "--use_fast_math", "-DATL_WITH_CUDA", "-lineinfo"],
                    extra_include_paths=[str(CSRC)], build_directory=str(build_dir), verbose=False)
    except Exception as e:  # pragma: no cover - environment dependent
        warnings.warn(f"atl: C++/CUDA extension unavailable, using torch reference ops ({e})")
        _EXT = None
    return _EXT


def ext_supports(t: torch.Tensor) -> bool:
    ext = get_ext()
    if ext is None:
        return False
    return (not t.is_cuda) or bool(ext.with_cuda)
