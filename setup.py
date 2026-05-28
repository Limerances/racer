import os
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


def build_extensions():
    if os.environ.get("RACER_DISABLE_CUDA_EXT") == "1":
        return [], {}

    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    except Exception:
        return [], {}

    cuda_home = CUDA_HOME or os.environ.get("CUDA_HOME")
    if not cuda_home:
        return [], {}

    nvcc = Path(cuda_home) / "bin" / "nvcc"
    if not nvcc.exists():
        return [], {}

    ext = CUDAExtension(
        name="racer._C",
        sources=[
            str(ROOT / "racer" / "csrc" / "binding.cpp"),
            str(ROOT / "racer" / "csrc" / "racer_cuda.cu"),
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["-O3", "--use_fast_math"],
        },
    )
    return [ext], {"build_ext": BuildExtension}


ext_modules, cmdclass = build_extensions()

setup(
    name="racer",
    version="0.1.0",
    packages=find_packages(),
    package_data={"racer": ["csrc/*.cpp", "csrc/*.cu"]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
