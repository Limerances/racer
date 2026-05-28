"""ctypes bridge to an installed Jerasure/GF-Complete build.

The wrapper is optional: tests skip when the shared library is unavailable.
When present, it is used as an external reference for Cauchy matrices and
GF(2^8) region multiplication. RACER never serializes Python objects through
Jerasure; it passes contiguous uint8 memory regions.
"""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch


DEFAULT_JERASURE_LIB = Path("/workspace/Jerasure/src/.libs/libJerasure.so.2.0.0")


class JerasureUnavailable(RuntimeError):
    """Raised when Jerasure cannot be loaded."""


def _lib_path() -> Path:
    return Path(os.environ.get("RACER_JERASURE_LIB", str(DEFAULT_JERASURE_LIB)))


@lru_cache(maxsize=1)
def _lib() -> ctypes.CDLL:
    path = _lib_path()
    if not path.exists():
        raise JerasureUnavailable(f"Jerasure library not found at {path}")
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:  # pragma: no cover - depends on host linker state.
        raise JerasureUnavailable(f"failed to load Jerasure library at {path}: {exc}") from exc

    lib.galois_single_multiply.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.galois_single_multiply.restype = ctypes.c_int
    lib.galois_single_divide.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.galois_single_divide.restype = ctypes.c_int
    lib.galois_inverse.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.galois_inverse.restype = ctypes.c_int
    lib.galois_w08_region_multiply.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.galois_w08_region_multiply.restype = None
    lib.galois_region_xor.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.galois_region_xor.restype = None
    lib.cauchy_original_coding_matrix.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.cauchy_original_coding_matrix.restype = ctypes.POINTER(ctypes.c_int)
    lib.jerasure_matrix_encode.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_int,
    ]
    lib.jerasure_matrix_encode.restype = None
    return lib


@lru_cache(maxsize=1)
def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(None)
    libc.free.argtypes = [ctypes.c_void_p]
    libc.free.restype = None
    return libc


def available() -> bool:
    try:
        _lib()
    except JerasureUnavailable:
        return False
    return True


def gf_mul(a: int, b: int, w: int = 8) -> int:
    if w != 8:
        raise NotImplementedError("RACER Jerasure bridge currently supports w=8")
    return int(_lib().galois_single_multiply(int(a), int(b), int(w))) & 0xFF


def gf_div(a: int, b: int, w: int = 8) -> int:
    if w != 8:
        raise NotImplementedError("RACER Jerasure bridge currently supports w=8")
    return int(_lib().galois_single_divide(int(a), int(b), int(w))) & 0xFF


def gf_inv(a: int, w: int = 8) -> int:
    if w != 8:
        raise NotImplementedError("RACER Jerasure bridge currently supports w=8")
    return int(_lib().galois_inverse(int(a), int(w))) & 0xFF


def cauchy_original_coding_matrix(k: int, m: int, w: int = 8) -> list[list[int]]:
    if w != 8:
        raise NotImplementedError("RACER Jerasure bridge currently supports w=8")
    ptr = _lib().cauchy_original_coding_matrix(int(k), int(m), int(w))
    if not ptr:
        raise RuntimeError("Jerasure returned a null Cauchy matrix")
    try:
        return [
            [int(ptr[i * k + j]) & 0xFF for j in range(k)]
            for i in range(m)
        ]
    finally:
        _libc().free(ctypes.cast(ptr, ctypes.c_void_p))


def _as_cpu_uint8_flat(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dtype != torch.uint8:
        raise TypeError(f"{name} must be torch.uint8")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    return tensor.contiguous().view(-1)


def _pad_to_word(tensor: torch.Tensor) -> tuple[torch.Tensor, int]:
    flat = _as_cpu_uint8_flat(tensor, "tensor")
    nbytes = int(flat.numel())
    padded = ((nbytes + 7) // 8) * 8
    if padded == nbytes:
        return flat.clone(), nbytes
    out = torch.zeros(padded, dtype=torch.uint8)
    out[:nbytes].copy_(flat)
    return out, nbytes


def _char_ptr(tensor: torch.Tensor) -> ctypes.c_char_p:
    return ctypes.cast(ctypes.c_void_p(int(tensor.data_ptr())), ctypes.c_char_p)


def region_multiply(src: torch.Tensor, coeff: int, *, add_to: torch.Tensor | None = None) -> torch.Tensor:
    """Multiply a CPU uint8 region by a GF(2^8) coefficient using Jerasure.

    If `add_to` is supplied, Jerasure XORs the product into it and the same
    tensor is returned. The exposed result length always matches the original
    source length; internal padding only satisfies Jerasure's longword
    alignment requirement.
    """

    coeff = int(coeff) & 0xFF
    src_padded, original_nbytes = _pad_to_word(src)
    if add_to is None:
        dst_padded = torch.zeros_like(src_padded)
        add = 0
    else:
        dst_flat = _as_cpu_uint8_flat(add_to, "add_to")
        if int(dst_flat.numel()) != original_nbytes:
            raise ValueError("add_to must have the same original numel as src")
        dst_padded, _ = _pad_to_word(dst_flat)
        add = 1

    if coeff == 0:
        if add_to is None:
            return dst_padded[:original_nbytes].clone()
        return add_to
    if coeff == 1:
        if add_to is None:
            return src_padded[:original_nbytes].clone()
        dst_flat = _as_cpu_uint8_flat(add_to, "add_to")
        dst_flat.bitwise_xor_(src_padded[:original_nbytes])
        return add_to

    _lib().galois_w08_region_multiply(
        _char_ptr(src_padded),
        coeff,
        int(src_padded.numel()),
        _char_ptr(dst_padded),
        add,
    )
    result = dst_padded[:original_nbytes].clone()
    if add_to is not None:
        _as_cpu_uint8_flat(add_to, "add_to").copy_(result)
        return add_to
    return result


def matrix_encode(data_chunks: Sequence[torch.Tensor], coding_matrix: Sequence[Sequence[int]]) -> list[torch.Tensor]:
    """Encode parity chunks with Jerasure's `jerasure_matrix_encode`.

    `coding_matrix` is the m x k parity matrix C, not the full systematic E.
    """

    if not data_chunks:
        raise ValueError("data_chunks must be non-empty")
    k = len(data_chunks)
    m = len(coding_matrix)
    if m == 0:
        raise ValueError("coding_matrix must have at least one row")
    if any(len(row) != k for row in coding_matrix):
        raise ValueError("coding_matrix must have shape m x k")

    padded_inputs = [_pad_to_word(chunk)[0] for chunk in data_chunks]
    size = max(int(chunk.numel()) for chunk in padded_inputs)
    for idx, chunk in enumerate(padded_inputs):
        if int(chunk.numel()) != size:
            out = torch.zeros(size, dtype=torch.uint8)
            out[: chunk.numel()].copy_(chunk)
            padded_inputs[idx] = out

    outputs = [torch.zeros(size, dtype=torch.uint8) for _ in range(m)]
    matrix_values = [int(value) & 0xFF for row in coding_matrix for value in row]
    matrix = (ctypes.c_int * len(matrix_values))(*matrix_values)
    data_ptrs = (ctypes.c_char_p * k)(*[_char_ptr(chunk) for chunk in padded_inputs])
    coding_ptrs = (ctypes.c_char_p * m)(*[_char_ptr(chunk) for chunk in outputs])
    _lib().jerasure_matrix_encode(k, m, 8, matrix, data_ptrs, coding_ptrs, size)

    original = int(_as_cpu_uint8_flat(data_chunks[0], "data_chunks[0]").numel())
    return [out[:original].clone() for out in outputs]
