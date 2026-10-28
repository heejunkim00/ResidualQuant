"""Build the native binding; supported kernels and selection live in C++."""

import os
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault(
    "MAX_JOBS",
    str(
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count() or 1
    ),
)

root = Path(__file__).resolve().parent
include_dirs = [str(root / "csrc")]
include_dirs.extend(
    str(path)
    for path in sorted(
        (Path(torch.__file__).parent.parent / "nvidia").glob("*/include")
    )
    if path.parent.name in ("cublas", "cusparse", "cusolver")
)

setup(
    ext_modules=[
        CUDAExtension(
            "residual_quantization._cuda",
            sources=[
                "csrc/dispatch/binding.cpp",
                "csrc/attention/api.cpp",
                "csrc/quantize/api.cpp",
                *[
                    str(path.relative_to(root))
                    for path in sorted((root / "csrc" / "dispatch").glob("decode_*.cu"))
                ],
            ],
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "--extra-device-vectorization",
                    "--restrict",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
