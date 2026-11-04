"""pip install -e .     (CUDA kernels are built automatically when torch sees a GPU and CUDA_HOME is set;
force with EVKIT_FORCE_CUDA=1, disable with EVKIT_NO_CUDA=1; arch list via TORCH_CUDA_ARCH_LIST="8.0;9.0")."""
import glob
import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CppExtension, CUDAExtension

ROOT = os.path.dirname(os.path.abspath(__file__))
srcs = ["torch/evkit_torch.cpp"] + sorted(glob.glob("src/*.cpp"))
with_cuda = (os.environ.get("EVKIT_NO_CUDA") != "1" and CUDA_HOME is not None and
             (torch.cuda.is_available() or os.environ.get("EVKIT_FORCE_CUDA") == "1"))
common = dict(include_dirs=[os.path.join(ROOT, "include")])
if with_cuda:
    ext = CUDAExtension("evkit._C", srcs + sorted(glob.glob("cuda/*.cu")), define_macros=[("EVKIT_WITH_CUDA", None)],
                        extra_compile_args={"cxx": ["-O3", "-march=native"],
                                            "nvcc": ["-O3", "--use_fast_math", "-lineinfo", "-Xptxas=-v"]}, **common)
else:
    ext = CppExtension("evkit._C", srcs, extra_compile_args=["-O3", "-march=native"], **common)

setup(name="evkit", version="0.1.0", packages=["evkit"], package_dir={"evkit": "python/evkit"},
      ext_modules=[ext], cmdclass={"build_ext": BuildExtension})
