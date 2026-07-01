"""CUDA extension loading and ABI probing for RACER codec kernels."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_LOAD_ERRORS: list[str] = []


def _record_load_error(source: str, exc: Exception) -> None:
    _LOAD_ERRORS.append(f"{source}: {type(exc).__name__}: {exc}")


@lru_cache(maxsize=1)
def extension():
    root = Path(__file__).resolve().parents[1]
    if os.environ.get("RACER_JIT_COMPILE") == "1":
        try:
            from torch.utils.cpp_extension import load

            return load(
                name="racer_jit_C",
                sources=[str(root / "csrc" / "binding.cpp"), str(root / "csrc" / "racer_cuda.cu")],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                extra_cflags=["-O3"],
                verbose=False,
            )
        except Exception as exc:
            _record_load_error("jit", exc)
            pass

    try:
        from racer import _C  # type: ignore

        return _C
    except Exception as exc:
        _record_load_error("package", exc)
        return None


def extension_available() -> bool:
    return extension() is not None


def extension_function(name: str):
    ext = extension()
    if ext is None or not hasattr(ext, name):
        details = ""
        if _LOAD_ERRORS:
            details = "; load errors: " + " | ".join(_LOAD_ERRORS)
        raise RuntimeError(
            f"RACER CUDA extension function {name!r} is required; rebuild the package "
            "or set RACER_JIT_COMPILE=1 in a CUDA build environment"
            f"{details}"
        )
    return getattr(ext, name)


def optional_extension_function(*names: str):
    ext = extension()
    if ext is None:
        return None
    for name in names:
        if hasattr(ext, name):
            return getattr(ext, name)
    return None
