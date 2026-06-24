"""Checkpoint Storage Daemon client/server and daemon-owned backends.

The CSD owns checkpoint chunk lifetime. Training processes connect as
producers/consumers and may restart without invalidating committed chunks held
by the daemon process.

Daemon-owned storage backends implemented here:
- native_pinned: daemon-owned cudaHostAlloc host memory with CUDA IPC copy;
- EGM: real daemon-owned EGM runtime integration when an external EGM runtime
  is provided.

The old fd/mmap compatibility transport is intentionally not selectable from
the public API or daemon CLI. Socket byte and fd/mmap copy requests raise hard
errors instead of silently falling back.

Checkpoint durability across training-process restart must stay with the
daemon-owned pinned-memory or EGM backend.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import Future, ThreadPoolExecutor
import ctypes
import hashlib
import json
from dataclasses import dataclass
import mmap
import multiprocessing as mp
from multiprocessing.connection import AuthenticationError, Client, Listener
from multiprocessing.reduction import recv_handle, send_handle
import os
from pathlib import Path
import threading
import time
import uuid
from typing import Any
import zlib

import torch

from .csd_manifest import CsdManifestStore

try:
    import xxhash as _xxhash
except ImportError:  # pragma: no cover - optional performance dependency
    _xxhash = None


_CUDART: Any | None = None
_LIBC: Any | None = None
_CUDA_MEMCPY_HOST_TO_DEVICE = 1
_CUDA_MEMCPY_DEVICE_TO_HOST = 2
_CUDA_MEMCPY_DEVICE_TO_DEVICE = 3
_CUDA_MEMCPY_DEFAULT = 4
_CUDA_EVENT_DISABLE_TIMING = 2
_CUDA_EVENT_INTERPROCESS = 4
_CUDA_STREAM_NON_BLOCKING = 1
_CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1
_CUDA_HOST_ALLOC_DEFAULT = 0
_CUDA_ERROR_NOT_READY = 34
_CUDA_ERROR_NOT_PERMITTED = 600


def _visible_cuda_device_count() -> int:
    try:
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0


class _CudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


class _CudaIpcEventHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


def _load_cudart() -> Any | None:
    global _CUDART
    if _CUDART is not None:
        return _CUDART
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            cudart = ctypes.CDLL(name)
            cudart.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            cudart.cudaHostRegister.restype = ctypes.c_int
            cudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
            cudart.cudaHostUnregister.restype = ctypes.c_int
            cudart.cudaMemcpy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            cudart.cudaMemcpy.restype = ctypes.c_int
            cudart.cudaMemcpyAsync.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            cudart.cudaMemcpyAsync.restype = ctypes.c_int
            cudart.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
            cudart.cudaHostAlloc.restype = ctypes.c_int
            cudart.cudaFreeHost.argtypes = [ctypes.c_void_p]
            cudart.cudaFreeHost.restype = ctypes.c_int
            cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            cudart.cudaMalloc.restype = ctypes.c_int
            cudart.cudaFree.argtypes = [ctypes.c_void_p]
            cudart.cudaFree.restype = ctypes.c_int
            cudart.cudaSetDevice.argtypes = [ctypes.c_int]
            cudart.cudaSetDevice.restype = ctypes.c_int
            cudart.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(_CudaIpcMemHandle), ctypes.c_void_p]
            cudart.cudaIpcGetMemHandle.restype = ctypes.c_int
            cudart.cudaIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _CudaIpcMemHandle, ctypes.c_uint]
            cudart.cudaIpcOpenMemHandle.restype = ctypes.c_int
            cudart.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
            cudart.cudaIpcCloseMemHandle.restype = ctypes.c_int
            cudart.cudaEventCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
            cudart.cudaEventCreateWithFlags.restype = ctypes.c_int
            cudart.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            cudart.cudaEventRecord.restype = ctypes.c_int
            cudart.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
            cudart.cudaEventSynchronize.restype = ctypes.c_int
            cudart.cudaEventQuery.argtypes = [ctypes.c_void_p]
            cudart.cudaEventQuery.restype = ctypes.c_int
            cudart.cudaEventDestroy.argtypes = [ctypes.c_void_p]
            cudart.cudaEventDestroy.restype = ctypes.c_int
            cudart.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
            cudart.cudaEventElapsedTime.restype = ctypes.c_int
            cudart.cudaIpcGetEventHandle.argtypes = [ctypes.POINTER(_CudaIpcEventHandle), ctypes.c_void_p]
            cudart.cudaIpcGetEventHandle.restype = ctypes.c_int
            cudart.cudaIpcOpenEventHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _CudaIpcEventHandle]
            cudart.cudaIpcOpenEventHandle.restype = ctypes.c_int
            cudart.cudaStreamCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
            cudart.cudaStreamCreateWithFlags.restype = ctypes.c_int
            cudart.cudaStreamWaitEvent.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
            cudart.cudaStreamWaitEvent.restype = ctypes.c_int
            cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
            cudart.cudaStreamSynchronize.restype = ctypes.c_int
            cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
            cudart.cudaStreamDestroy.restype = ctypes.c_int
            _CUDART = cudart
            return cudart
        except OSError:
            continue
    return None


def _load_libc() -> Any | None:
    global _LIBC
    if _LIBC is not None:
        return _LIBC
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.mlock.restype = ctypes.c_int
        libc.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.munlock.restype = ctypes.c_int
        _LIBC = libc
        return libc
    except OSError:
        return None


def _cuda_host_register_buffer(buffer: Any, nbytes: int) -> int | None:
    if int(nbytes) <= 0 or not torch.cuda.is_available():
        return None
    cudart = _load_cudart()
    if cudart is None:
        return None
    ptr = _buffer_pointer(buffer)
    if ptr is None:
        return None
    err = int(cudart.cudaHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(int(nbytes)), 0))
    if err != 0:
        return None
    return int(ptr)


def _cuda_host_unregister_pointer(ptr: int | None) -> None:
    if ptr is None:
        return
    cudart = _load_cudart()
    if cudart is None:
        return
    try:
        cudart.cudaHostUnregister(ctypes.c_void_p(int(ptr)))
    except Exception:
        pass


def _mlock_buffer(buffer: Any, nbytes: int) -> int | None:
    if int(nbytes) <= 0:
        return None
    libc = _load_libc()
    if libc is None:
        return None
    ptr = _buffer_pointer(buffer)
    if ptr is None:
        return None
    err = int(libc.mlock(ctypes.c_void_p(ptr), ctypes.c_size_t(int(nbytes))))
    if err != 0:
        return None
    return int(ptr)


def _munlock_pointer(ptr: int | None, nbytes: int) -> None:
    if ptr is None or int(nbytes) <= 0:
        return
    libc = _load_libc()
    if libc is None:
        return
    try:
        libc.munlock(ctypes.c_void_p(int(ptr)), ctypes.c_size_t(int(nbytes)))
    except Exception:
        pass


def _buffer_pointer(buffer: Any) -> int | None:
    try:
        return int(ctypes.addressof(ctypes.c_char.from_buffer(buffer)))
    except (TypeError, ValueError):
        return None


def _cuda_memcpy(dst_ptr: int, src_ptr: int, nbytes: int, kind: int) -> None:
    cudart = _load_cudart()
    if cudart is None:
        raise RuntimeError("CUDA runtime is required for native CSD CUDA copy")
    err = int(
        cudart.cudaMemcpy(
            ctypes.c_void_p(int(dst_ptr)),
            ctypes.c_void_p(int(src_ptr)),
            ctypes.c_size_t(int(nbytes)),
            ctypes.c_int(int(kind)),
        )
    )
    if err != 0:
        raise RuntimeError(f"cudaMemcpy failed during native CSD transfer with error code {err}")


def _cuda_check(err: int, message: str) -> None:
    if int(err) != 0:
        raise RuntimeError(f"{message}: CUDA error code {int(err)}")


def _cuda_set_device(device: int) -> None:
    cudart = _load_cudart()
    if cudart is None:
        raise RuntimeError("CUDA runtime is required for CSD CUDA IPC")
    _cuda_check(cudart.cudaSetDevice(ctypes.c_int(int(device))), "cudaSetDevice failed")


def _cuda_warm_device(device: int) -> None:
    cudart = _load_cudart()
    if cudart is None:
        raise RuntimeError("CUDA runtime is required for CSD CUDA IPC")
    _cuda_set_device(int(device))
    if not hasattr(cudart, "cudaFree"):
        return
    _cuda_check(cudart.cudaFree(ctypes.c_void_p(0)), f"cudaFree(0) warmup failed on device {device}")


def _current_stream_ptr(device: torch.device | int | None = None) -> int:
    if device is None:
        stream = torch.cuda.current_stream()
    else:
        stream = torch.cuda.current_stream(device=torch.device("cuda", int(device)))
    return int(stream.cuda_stream)


def _cuda_event_elapsed_ms(start_event: int, end_event: int) -> float:
    cudart = _load_cudart()
    if cudart is None:
        raise RuntimeError("CUDA runtime is required for event timing")
    value = ctypes.c_float()
    _cuda_check(
        cudart.cudaEventElapsedTime(
            ctypes.byref(value),
            ctypes.c_void_p(int(start_event)),
            ctypes.c_void_p(int(end_event)),
        ),
        "cudaEventElapsedTime failed",
    )
    return float(value.value)


@dataclass
class _StagingRegion:
    slab: "_StagingSlab"
    offset: int
    nbytes: int


@dataclass
class _StagingSlab:
    slab_id: str
    ptr: int
    nbytes: int
    mem_handle: bytes
    free_ranges: list[tuple[int, int]]


class _CudaStagingPool:
    def __init__(self, device: int, *, slab_bytes: int | None = None) -> None:
        cudart = _load_cudart()
        if cudart is None:
            raise RuntimeError("CUDA runtime is required for CSD staging")
        self.device = int(device)
        self.slab_bytes = int(slab_bytes or int(os.environ.get("RACER_CSD_STAGING_SLAB_BYTES", str(256 * 1024 * 1024))))
        self._slabs: list[_StagingSlab] = []
        self._lock = threading.RLock()
        _cuda_set_device(self.device)
        stream = ctypes.c_void_p()
        _cuda_check(
            cudart.cudaStreamCreateWithFlags(ctypes.byref(stream), ctypes.c_uint(_CUDA_STREAM_NON_BLOCKING)),
            "cudaStreamCreateWithFlags staging pool failed",
        )
        self.stream = int(stream.value)

    @staticmethod
    def _align(value: int, alignment: int = 256) -> int:
        return (int(value) + int(alignment) - 1) // int(alignment) * int(alignment)

    def _new_slab(self, nbytes: int) -> _StagingSlab:
        cudart = _load_cudart()
        assert cudart is not None
        size = max(self.slab_bytes, self._align(int(nbytes)))
        ptr = ctypes.c_void_p()
        _cuda_set_device(self.device)
        _cuda_check(cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size)), "cudaMalloc persistent staging slab failed")
        slab = _StagingSlab(
            slab_id=f"staging_{self.device}_{len(self._slabs):06d}",
            ptr=int(ptr.value),
            nbytes=int(size),
            mem_handle=_ipc_handle_bytes_from_ptr(int(ptr.value)),
            free_ranges=[(0, int(size))],
        )
        self._slabs.append(slab)
        return slab

    def allocate(self, nbytes: int) -> _StagingRegion:
        nbytes = self._align(int(nbytes))
        with self._lock:
            for slab in self._slabs:
                for index, (offset, length) in enumerate(list(slab.free_ranges)):
                    aligned = self._align(offset)
                    padding = aligned - offset
                    if length - padding >= nbytes:
                        before = [] if padding <= 0 else [(offset, padding)]
                        after_start = aligned + nbytes
                        after_len = (offset + length) - after_start
                        after = [] if after_len <= 0 else [(after_start, after_len)]
                        slab.free_ranges[index : index + 1] = before + after
                        return _StagingRegion(slab=slab, offset=aligned, nbytes=nbytes)
            slab = self._new_slab(nbytes)
            slab.free_ranges = [(nbytes, slab.nbytes - nbytes)] if slab.nbytes > nbytes else []
            return _StagingRegion(slab=slab, offset=0, nbytes=nbytes)

    def release(self, region: _StagingRegion) -> None:
        with self._lock:
            ranges = list(region.slab.free_ranges)
            ranges.append((int(region.offset), int(region.nbytes)))
            ranges.sort()
            merged: list[tuple[int, int]] = []
            for offset, length in ranges:
                if not merged or merged[-1][0] + merged[-1][1] < offset:
                    merged.append((offset, length))
                else:
                    prev_offset, prev_len = merged[-1]
                    merged[-1] = (prev_offset, max(prev_offset + prev_len, offset + length) - prev_offset)
            region.slab.free_ranges = merged

    def __del__(self) -> None:
        cudart = _load_cudart()
        if cudart is None:
            return
        for slab in getattr(self, "_slabs", []):
            try:
                cudart.cudaFree(ctypes.c_void_p(int(slab.ptr)))
            except Exception:
                pass
        try:
            cudart.cudaStreamDestroy(ctypes.c_void_p(int(getattr(self, "stream", 0))))
        except Exception:
            pass


_STAGING_POOLS: dict[int, _CudaStagingPool] = {}
_STAGING_POOLS_LOCK = threading.RLock()


def _get_staging_pool(device: int) -> _CudaStagingPool:
    with _STAGING_POOLS_LOCK:
        pool = _STAGING_POOLS.get(int(device))
        if pool is None:
            pool = _CudaStagingPool(int(device))
            _STAGING_POOLS[int(device)] = pool
        return pool


def prewarm_cuda_ipc_staging(
    *,
    device: int | None = None,
    slab_bytes: int | None = None,
    count: int = 1,
) -> dict[str, Any]:
    """Preallocate reusable CUDA IPC staging slabs for CSD transfers.

    This allocates daemon-client owned CUDA staging buffers in the training
    process and immediately returns them to the staging pool. The final
    checkpoint bytes still live in the CSD daemon backend; these slabs only keep
    the first PUT from paying cudaMalloc and cudaIpcGetMemHandle costs.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to prewarm CSD CUDA IPC staging")
    cuda_device = int(torch.cuda.current_device() if device is None else device)
    nbytes = int(slab_bytes or int(os.environ.get("RACER_CSD_STAGING_SLAB_BYTES", str(256 * 1024 * 1024))))
    slab_count = max(0, int(count))
    pool = _get_staging_pool(cuda_device)
    regions: list[_StagingRegion] = []
    start = time.perf_counter()
    try:
        for _ in range(slab_count):
            regions.append(pool.allocate(nbytes))
    finally:
        for region in reversed(regions):
            pool.release(region)
    return {
        "device": cuda_device,
        "slab_bytes": nbytes,
        "count": slab_count,
        "slabs_in_pool": len(pool._slabs),
        "elapsed_ms": (time.perf_counter() - start) * 1000.0,
    }


@dataclass
class CudaIpcView:
    device: int
    nbytes: int
    base_offset: int
    dtype: str
    shape: list[int]
    mem_handle: bytes
    event_handle: bytes
    requires_staging: bool = False
    staging_id: str | None = None
    _keepalive: Any | None = None
    _event_ptr: int | None = None
    _staging_ptr: int | None = None
    _stream_ptr: int | None = None
    _source_event_ptr: int | None = None
    _copy_back_tensor: torch.Tensor | None = None
    _staging_pool: _CudaStagingPool | None = None
    _staging_region: _StagingRegion | None = None
    profile: dict[str, Any] | None = None

    def as_request(self) -> dict[str, Any]:
        return {
            "device": int(self.device),
            "nbytes": int(self.nbytes),
            "base_offset": int(self.base_offset),
            "dtype": str(self.dtype),
            "shape": list(self.shape),
            "mem_handle": bytes(self.mem_handle),
            "event_handle": bytes(self.event_handle),
            "requires_staging": bool(self.requires_staging),
            "staging_id": self.staging_id,
            "producer_pid": os.getpid(),
            "allocation_id": hashlib.sha1(bytes(self.mem_handle)).hexdigest(),
            "staging_kind": "persistent_slab" if self._staging_region is not None else ("none" if not self.requires_staging else "ephemeral"),
            "profile": dict(self.profile or {}),
        }

    def release(self) -> None:
        cudart = _load_cudart()
        if cudart is not None and self._event_ptr is not None:
            try:
                cudart.cudaEventDestroy(ctypes.c_void_p(int(self._event_ptr)))
            except Exception:
                pass
        if self._staging_pool is not None and self._staging_region is not None:
            self._staging_pool.release(self._staging_region)
        elif cudart is not None and self._staging_ptr is not None:
            try:
                cudart.cudaFree(ctypes.c_void_p(int(self._staging_ptr)))
            except Exception:
                pass
        if cudart is not None and self._source_event_ptr is not None:
            try:
                cudart.cudaEventDestroy(ctypes.c_void_p(int(self._source_event_ptr)))
            except Exception:
                pass
        if cudart is not None and self._stream_ptr is not None:
            try:
                cudart.cudaStreamDestroy(ctypes.c_void_p(int(self._stream_ptr)))
            except Exception:
                pass
        self._event_ptr = None
        self._staging_ptr = None
        self._stream_ptr = None
        self._source_event_ptr = None
        self._copy_back_tensor = None
        self._staging_pool = None
        self._staging_region = None
        self._keepalive = None

    def materialize_after_read(self) -> None:
        if self._copy_back_tensor is None or self._staging_ptr is None:
            return
        tensor = self._copy_back_tensor.detach().contiguous().view(-1)
        _cuda_set_device(int(tensor.device.index or 0))
        _cuda_memcpy(
            int(tensor.data_ptr()),
            int(self._staging_ptr),
            int(self.nbytes),
            _CUDA_MEMCPY_DEVICE_TO_DEVICE,
        )


def _ipc_handle_bytes_from_ptr(ptr: int) -> bytes:
    cudart = _load_cudart()
    if cudart is None:
        raise RuntimeError("CUDA runtime is required for CSD CUDA IPC")
    handle = _CudaIpcMemHandle()
    _cuda_check(
        cudart.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(int(ptr))),
        "cudaIpcGetMemHandle failed",
    )
    return ctypes.string_at(ctypes.byref(handle), 64)


def _record_interprocess_event(device: int, stream_ptr: int) -> tuple[bytes, int]:
    cudart = _load_cudart()
    if cudart is None:
        raise RuntimeError("CUDA runtime is required for CSD CUDA IPC")
    _cuda_set_device(device)
    event = ctypes.c_void_p()
    flags = _CUDA_EVENT_DISABLE_TIMING | _CUDA_EVENT_INTERPROCESS
    _cuda_check(cudart.cudaEventCreateWithFlags(ctypes.byref(event), ctypes.c_uint(flags)), "cudaEventCreateWithFlags failed")
    _cuda_check(cudart.cudaEventRecord(event, ctypes.c_void_p(int(stream_ptr))), "cudaEventRecord failed")
    handle = _CudaIpcEventHandle()
    try:
        _cuda_check(cudart.cudaIpcGetEventHandle(ctypes.byref(handle), event), "cudaIpcGetEventHandle failed")
    except Exception:
        cudart.cudaEventDestroy(event)
        raise
    return ctypes.string_at(ctypes.byref(handle), 64), int(event.value)


def export_cuda_ipc_view(
    tensor: torch.Tensor,
    stream: Any | None = None,
    *,
    staging_role: str = "source",
) -> CudaIpcView:
    """Export a CUDA uint8 tensor view for CSD native pinned transfer.

    The returned object owns any staging allocation/event needed to keep the IPC
    source valid until the daemon acknowledges the operation.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("export_cuda_ipc_view expects a torch.Tensor")
    if tensor.device.type != "cuda":
        raise ValueError("export_cuda_ipc_view requires a CUDA tensor")
    if tensor.dtype != torch.uint8:
        raise TypeError("CSD CUDA IPC views currently require torch.uint8 tensors")
    if not tensor.is_contiguous():
        raise ValueError("CSD CUDA IPC views require contiguous tensors")
    cudart = _load_cudart()
    if cudart is None:
        raise RuntimeError("CUDA runtime is required for CSD CUDA IPC")
    device = int(tensor.device.index or 0)
    _cuda_set_device(device)
    flat = tensor.detach().view(-1)
    nbytes = int(flat.numel())
    stream_ptr = int(stream.cuda_stream) if stream is not None else _current_stream_ptr(device)
    storage_ptr = int(flat.untyped_storage().data_ptr())
    data_ptr = int(flat.data_ptr())
    base_offset = data_ptr - storage_ptr
    role = str(staging_role).lower()
    if role not in {"source", "target"}:
        raise ValueError("staging_role must be 'source' or 'target'")
    global_direct = os.environ.get("RACER_CSD_DIRECT_TENSOR_IPC", "0") == "1"
    if role == "target":
        direct_ipc_enabled = global_direct or os.environ.get("RACER_CSD_DIRECT_READ_IPC", "0") == "1"
    else:
        direct_ipc_enabled = global_direct or os.environ.get("RACER_CSD_DIRECT_WRITE_IPC", "0") == "1"
    profile: dict[str, Any] = {
        "direct_ipc_enabled": bool(direct_ipc_enabled),
        "direct_ipc_role": role,
        "direct_ipc_error": "",
        "staging_kind": "none",
        "staging_alloc_us": 0.0,
        "staging_copy_enqueue_us": 0.0,
    }

    if direct_ipc_enabled:
        try:
            mem_handle = _ipc_handle_bytes_from_ptr(storage_ptr)
            event_handle, event_ptr = _record_interprocess_event(device, stream_ptr)
            return CudaIpcView(
                device=device,
                nbytes=nbytes,
                base_offset=base_offset,
                dtype=str(tensor.dtype),
                shape=list(tensor.shape),
                mem_handle=mem_handle,
                event_handle=event_handle,
                requires_staging=False,
                _keepalive=tensor,
                _event_ptr=event_ptr,
                profile=profile,
            )
        except Exception as exc:
            profile["direct_ipc_error"] = str(exc)
    else:
        profile["direct_ipc_error"] = "disabled"

    pool = _get_staging_pool(device)
    alloc_start = time.perf_counter()
    region = pool.allocate(nbytes)
    profile["staging_alloc_us"] = (time.perf_counter() - alloc_start) * 1_000_000.0
    profile["staging_kind"] = "persistent_slab"
    staging_ptr = int(region.slab.ptr + region.offset)
    if role == "target":
        try:
            mem_handle = region.slab.mem_handle
            event_handle, event_ptr = _record_interprocess_event(device, stream_ptr)
        except Exception:
            pool.release(region)
            raise
        return CudaIpcView(
            device=device,
            nbytes=nbytes,
            base_offset=int(region.offset),
            dtype=str(tensor.dtype),
            shape=list(tensor.shape),
            mem_handle=mem_handle,
            event_handle=event_handle,
            requires_staging=True,
            staging_id=region.slab.slab_id,
            _keepalive=tensor,
            _event_ptr=event_ptr,
            _staging_ptr=staging_ptr,
            _copy_back_tensor=tensor,
            _staging_pool=pool,
            _staging_region=region,
            profile=profile,
        )

    source_event = ctypes.c_void_p()
    try:
        _cuda_check(
            cudart.cudaEventCreateWithFlags(ctypes.byref(source_event), ctypes.c_uint(_CUDA_EVENT_DISABLE_TIMING)),
            "cudaEventCreateWithFlags source staging event failed",
        )
        _cuda_check(
            cudart.cudaEventRecord(source_event, ctypes.c_void_p(int(stream_ptr))),
            "cudaEventRecord source staging event failed",
        )
        enqueue_start = time.perf_counter()
        _cuda_check(
            cudart.cudaStreamWaitEvent(ctypes.c_void_p(pool.stream), source_event, ctypes.c_uint(0)),
            "cudaStreamWaitEvent staging source failed",
        )
        _cuda_check(
            cudart.cudaMemcpyAsync(
                ctypes.c_void_p(staging_ptr),
                ctypes.c_void_p(data_ptr),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(_CUDA_MEMCPY_DEVICE_TO_DEVICE),
                ctypes.c_void_p(pool.stream),
            ),
            "cudaMemcpyAsync tensor->staging failed",
        )
        mem_handle = region.slab.mem_handle
        event_handle, event_ptr = _record_interprocess_event(device, int(pool.stream))
        profile["staging_copy_enqueue_us"] = (time.perf_counter() - enqueue_start) * 1_000_000.0
    except Exception:
        pool.release(region)
        if source_event.value:
            cudart.cudaEventDestroy(source_event)
        raise
    return CudaIpcView(
        device=device,
        nbytes=nbytes,
        base_offset=int(region.offset),
        dtype=str(tensor.dtype),
        shape=list(tensor.shape),
        mem_handle=mem_handle,
        event_handle=event_handle,
        requires_staging=True,
        staging_id=region.slab.slab_id,
        _keepalive=tensor,
        _event_ptr=event_ptr,
        _staging_ptr=staging_ptr,
        _source_event_ptr=int(source_event.value),
        _staging_pool=pool,
        _staging_region=region,
        profile=profile,
    )


def _authkey(value: bytes | str | None) -> bytes:
    if value is None:
        return b"racer-csd"
    if isinstance(value, bytes):
        return value
    return str(value).encode("utf-8")


def _safe_tag(tag: str) -> str:
    return base64.urlsafe_b64encode(str(tag).encode("utf-8")).decode("ascii").rstrip("=")


def _sample64_checksum_view(view: memoryview, *, sample_count: int = 4096) -> str:
    if view.format != "B":
        view = view.cast("B")
    nbytes = int(view.nbytes)
    if nbytes == 0:
        return "sample64-v1:0:0:0000000000000000:00:00"
    first = int(view[0]) & 0xFF
    last = int(view[nbytes - 1]) & 0xFF
    samples = min(int(sample_count), nbytes)
    if samples <= 1:
        total = first
    else:
        total = 0
        for index in range(samples):
            offset = index * (nbytes - 1) // (samples - 1)
            total = (total + (int(view[offset]) & 0xFF)) & 0xFFFFFFFFFFFFFFFF
    return f"sample64-v1:{nbytes}:{samples}:{total:016x}:{first:02x}:{last:02x}"


@dataclass
class BackendChunk:
    tensor: torch.Tensor
    metadata: dict[str, Any]
    nbytes: int
    capacity_nbytes: int | None = None
    slot_id: str | None = None
    fd: int | None = None
    mapping: mmap.mmap | None = None
    registered_ptr: int | None = None
    locked_ptr: int | None = None
    host_ptr: int | None = None
    segment_id: str | None = None
    offset: int = 0


class StorageBackend:
    """Daemon-side storage backend protocol."""

    name = "base"

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "restart_aware": False,
            "daemon_owned": False,
            "cuda_native_pinned": False,
            "uses_memfd_mmap": False,
            "supports_cuda_ipc": False,
            "supports_async_copy": False,
        }

    def allocate(self, tag: str, chunk_id: str, nbytes: int, metadata: dict[str, Any] | None = None) -> torch.Tensor:
        raise NotImplementedError

    def write_from_bytes(
        self,
        tag: str,
        chunk_id: str,
        data: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.name} does not support socket/CPU byte writes; use CUDA IPC native transport"
        )

    def write_from_cuda(
        self,
        tag: str,
        chunk_id: str,
        tensor: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.name} does not support copy-style CUDA writes; use write_from_cuda_ipc"
        )

    def read_to_bytes(self, tag: str, chunk_id: str) -> bytes:
        raise NotImplementedError

    def read_to_cuda(self, tag: str, chunk_id: str, device: torch.device | str) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.name} does not support socket/CPU byte reads; use read_to_cuda_ipc"
        )

    def prepare_fd_region(
        self,
        tag: str,
        chunk_id: str,
        nbytes: int,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int]:
        raise NotImplementedError(f"{self.name} does not support fd transport")

    def export_fd_region(self, tag: str, chunk_id: str) -> tuple[dict[str, Any], int]:
        raise NotImplementedError(f"{self.name} does not support fd transport")

    def metadata(self, tag: str, chunk_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def list_chunks(self, tag: str) -> list[str]:
        raise NotImplementedError

    def free(self, tag: str, chunk_id: str) -> None:
        raise NotImplementedError

    def checksum(self, tag: str, chunk_id: str, checksum_type: str = "sha256") -> str:
        data = self.read_to_bytes(tag, chunk_id)
        normalized = str(checksum_type).lower().replace("-", "_")
        if normalized in {"sample64", "sample64_v1", "fast", "sampled"}:
            return _sample64_checksum_view(memoryview(data))
        if normalized in {"xxh64", "xxhash64", "xxh64_v1"}:
            if _xxhash is None:
                raise RuntimeError("checksum_type=xxh64 requires the optional xxhash package")
            return _xxhash.xxh64(data).hexdigest()
        if normalized in {"crc32", "crc32_v1"}:
            return f"{zlib.crc32(data) & 0xFFFFFFFF:08x}"
        return hashlib.sha256(data).hexdigest()

    def free_tag(self, tag: str) -> None:
        for chunk_id in self.list_chunks(tag):
            self.free(tag, chunk_id)


class FdMmapHostBackend(StorageBackend):
    """Disabled historical fd/mmap host backend."""

    name = "fd_mmap_host"

    def __init__(
        self,
        *,
        pin_memory: bool | None = None,
        lock_fd_memory: bool = True,
        cuda_register_fd: bool = False,
        fd_pool_slot_size: int = 0,
        fd_pool_slot_count: int = 0,
        fd_pool_max_free: int = 64,
    ) -> None:
        raise RuntimeError(
            "FdMmapHostBackend is disabled. RACER CSD storage must be "
            "daemon-owned native_pinned or daemon-owned EGM."
        )
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        self.lock_fd_memory = bool(lock_fd_memory)
        self.cuda_register_fd = bool(cuda_register_fd)
        self.fd_pool_max_free = int(fd_pool_max_free)
        self._chunks: dict[str, dict[str, BackendChunk]] = {}
        self._free_fd_chunks: list[BackendChunk] = []
        self._next_fd_slot_id = 0
        self._lock = threading.RLock()
        slot_size = int(fd_pool_slot_size)
        slot_count = int(fd_pool_slot_count)
        if slot_size > 0 and slot_count > 0:
            for index in range(slot_count):
                self._free_fd_chunks.append(
                    self._create_fd_chunk(
                        tag="pool",
                        chunk_id=f"slot_{index:06d}",
                        capacity_nbytes=slot_size,
                    )
                )

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "restart_aware": True,
            "daemon_owned": True,
            "cuda_native_pinned": False,
            "uses_cudaHostAlloc": False,
            "uses_memfd_mmap": True,
            "supports_cuda_ipc": False,
            "supports_async_copy": False,
        }

    def _allocate_host(self, nbytes: int) -> torch.Tensor:
        if self.pin_memory:
            try:
                return torch.empty(int(nbytes), dtype=torch.uint8, device="cpu", pin_memory=True)
            except RuntimeError:
                self.pin_memory = False
        return torch.empty(int(nbytes), dtype=torch.uint8, device="cpu")

    def _release_chunk(self, chunk: BackendChunk) -> None:
        _cuda_host_unregister_pointer(chunk.registered_ptr)
        _munlock_pointer(chunk.locked_ptr, chunk.capacity_nbytes or chunk.nbytes)
        chunk.tensor = torch.empty(0, dtype=torch.uint8)
        if chunk.mapping is not None:
            try:
                chunk.mapping.close()
            except BufferError:
                pass
        if chunk.fd is not None:
            try:
                os.close(int(chunk.fd))
            except OSError:
                pass

    def _create_fd_chunk(self, *, tag: str, chunk_id: str, capacity_nbytes: int) -> BackendChunk:
        if not hasattr(os, "memfd_create"):
            raise RuntimeError("pinned fd transport requires os.memfd_create")
        fd = os.memfd_create(
            f"racer-csd-{_safe_tag(tag)}-{_safe_tag(chunk_id)}",
            flags=getattr(os, "MFD_CLOEXEC", 0),
        )
        os.ftruncate(fd, int(capacity_nbytes))
        mapping = mmap.mmap(fd, int(capacity_nbytes), access=mmap.ACCESS_WRITE)
        tensor = torch.frombuffer(mapping, dtype=torch.uint8, count=int(capacity_nbytes))
        locked_ptr = _mlock_buffer(mapping, int(capacity_nbytes)) if self.lock_fd_memory else None
        registered_ptr = (
            _cuda_host_register_buffer(mapping, int(capacity_nbytes)) if self.cuda_register_fd else None
        )
        with self._lock:
            slot_id = f"fdslot_{self._next_fd_slot_id:08d}"
            self._next_fd_slot_id += 1
        return BackendChunk(
            tensor=tensor,
            metadata={},
            nbytes=0,
            capacity_nbytes=int(capacity_nbytes),
            slot_id=slot_id,
            fd=fd,
            mapping=mapping,
            registered_ptr=registered_ptr,
            locked_ptr=locked_ptr,
        )

    def _fd_record_metadata(
        self,
        metadata: dict[str, Any] | None,
        *,
        chunk: BackendChunk,
        nbytes: int,
    ) -> dict[str, Any]:
        record_metadata = dict(metadata or {})
        record_metadata.update(
            {
                "storage_backend": self.name,
                "storage_transport": "fd_mmap_host",
                "stored_device": "cpu",
                "is_pinned_host": chunk.locked_ptr is not None or chunk.registered_ptr is not None,
                "cuda_native_pinned": False,
                "uses_memfd_mmap": True,
                "daemon_host_locked": chunk.locked_ptr is not None,
                "daemon_cuda_registered": chunk.registered_ptr is not None,
                "daemon_owned": True,
                "nbytes": int(nbytes),
                "capacity_nbytes": int(chunk.capacity_nbytes or nbytes),
                "fd_slot_id": chunk.slot_id,
            }
        )
        return record_metadata

    def _take_fd_chunk(self, nbytes: int) -> BackendChunk | None:
        with self._lock:
            for index, chunk in enumerate(self._free_fd_chunks):
                if int(chunk.capacity_nbytes or 0) >= int(nbytes):
                    return self._free_fd_chunks.pop(index)
        return None

    def _return_fd_chunk_to_pool(self, chunk: BackendChunk) -> bool:
        if chunk.fd is None or chunk.mapping is None or int(chunk.capacity_nbytes or 0) <= 0:
            return False
        with self._lock:
            if self.fd_pool_max_free >= 0 and len(self._free_fd_chunks) >= self.fd_pool_max_free:
                return False
            chunk.metadata = {}
            chunk.nbytes = 0
            self._free_fd_chunks.append(chunk)
            return True

    def allocate(self, tag: str, chunk_id: str, nbytes: int, metadata: dict[str, Any] | None = None) -> torch.Tensor:
        record_metadata = dict(metadata or {})
        record_metadata.update(
            {
                "storage_backend": self.name,
                "stored_device": "cpu",
                "is_pinned_host": bool(self.pin_memory),
                "cuda_native_pinned": False,
                "uses_memfd_mmap": False,
                "daemon_owned": True,
                "nbytes": int(nbytes),
            }
        )
        tensor = self._allocate_host(int(nbytes))
        with self._lock:
            old = self._chunks.setdefault(str(tag), {}).pop(str(chunk_id), None)
            if old is not None:
                self._release_chunk(old)
            self._chunks.setdefault(str(tag), {})[str(chunk_id)] = BackendChunk(
                tensor=tensor,
                metadata=record_metadata,
                nbytes=int(nbytes),
                capacity_nbytes=int(nbytes),
            )
        return tensor

    def prepare_fd_region(
        self,
        tag: str,
        chunk_id: str,
        nbytes: int,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int]:
        chunk = self._take_fd_chunk(int(nbytes))
        if chunk is None:
            chunk = self._create_fd_chunk(
                tag=str(tag),
                chunk_id=str(chunk_id),
                capacity_nbytes=int(nbytes),
            )
        chunk.nbytes = int(nbytes)
        record_metadata = self._fd_record_metadata(metadata, chunk=chunk, nbytes=int(nbytes))
        chunk.metadata = record_metadata
        with self._lock:
            old = self._chunks.setdefault(str(tag), {}).pop(str(chunk_id), None)
            if old is not None:
                if not self._return_fd_chunk_to_pool(old):
                    self._release_chunk(old)
            self._chunks.setdefault(str(tag), {})[str(chunk_id)] = chunk
        return dict(record_metadata), int(chunk.fd)

    def export_fd_region(self, tag: str, chunk_id: str) -> tuple[dict[str, Any], int]:
        try:
            with self._lock:
                chunk = self._chunks[str(tag)][str(chunk_id)]
        except KeyError as exc:
            raise KeyError(f"CSD pinned chunk {chunk_id!r} for tag {tag!r} is not resident") from exc
        if chunk.fd is None:
            raise RuntimeError(f"CSD fd_mmap_host chunk {chunk_id!r} is not backed by fd transport")
        return dict(chunk.metadata), int(chunk.fd)

    def read_to_bytes(self, tag: str, chunk_id: str) -> bytes:
        try:
            with self._lock:
                chunk = self._chunks[str(tag)][str(chunk_id)]
        except KeyError as exc:
            raise KeyError(f"CSD pinned chunk {chunk_id!r} for tag {tag!r} is not resident") from exc
        return chunk.tensor.detach().narrow(0, 0, int(chunk.nbytes)).contiguous().numpy().tobytes()

    def metadata(self, tag: str, chunk_id: str) -> dict[str, Any]:
        try:
            with self._lock:
                return dict(self._chunks[str(tag)][str(chunk_id)].metadata)
        except KeyError as exc:
            raise KeyError(f"CSD pinned chunk metadata {chunk_id!r} for tag {tag!r} is not resident") from exc

    def list_chunks(self, tag: str) -> list[str]:
        with self._lock:
            return sorted(self._chunks.get(str(tag), {}))

    def free(self, tag: str, chunk_id: str) -> None:
        with self._lock:
            chunks = self._chunks.get(str(tag))
            if chunks is None:
                return
            chunk = chunks.pop(str(chunk_id), None)
            if chunk is not None:
                if not self._return_fd_chunk_to_pool(chunk):
                    self._release_chunk(chunk)
            if not chunks:
                self._chunks.pop(str(tag), None)

@dataclass
class _NativeSegment:
    segment_id: str
    ptr: int
    nbytes: int
    offset: int = 0


@dataclass
class _NativeFreeBlock:
    segment_id: str
    offset: int
    nbytes: int


@dataclass
class _NativeAllocation:
    segment: _NativeSegment
    offset: int
    nbytes: int
    source: str
    allocate_ms: float


@dataclass
class _NativeCopyOp:
    op_id: str
    device: int
    stream: int
    wait_start_event: int
    copy_start_event: int
    complete_event: int
    remote_ptr: int | None
    remote_event: int | None
    cache_key: str | None = None
    stream_owned: bool = True
    profile: dict[str, Any] | None = None
    done: bool = False
    error: str | None = None
    resources_released: bool = False


class NativePinnedMemoryBackend(StorageBackend):
    """Daemon-owned cudaHostAlloc backend.

    Data survives training-process restart while the CSD daemon remains alive.
    If the daemon itself exits, cudaHostAlloc bytes are lost; the sqlite
    manifest can be recovered but chunk bytes cannot.
    """

    name = "native_pinned"

    def __init__(
        self,
        *,
        total_bytes: int = 0,
        segment_bytes: int = 256 * 1024 * 1024,
        device: int = 0,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("NativePinnedMemoryBackend requires CUDA access in the CSD daemon process")
        if _load_cudart() is None:
            raise RuntimeError("NativePinnedMemoryBackend requires CUDA runtime")
        self.total_bytes = int(total_bytes)
        self.segment_bytes = int(segment_bytes)
        self.device = int(device)
        self._segments: list[_NativeSegment] = []
        self._segment_by_id: dict[str, _NativeSegment] = {}
        self._free_blocks: list[_NativeFreeBlock] = []
        self._chunks: dict[str, dict[str, BackendChunk]] = {}
        self._ops: dict[str, _NativeCopyOp] = {}
        self._ipc_mem_cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._ipc_lock = threading.RLock()
        self._stream_lock = threading.RLock()
        self._copy_stream_pool_size = max(1, int(os.environ.get("RACER_CSD_COPY_STREAMS_PER_DEVICE", "4")))
        self._copy_streams: dict[int, list[int]] = {}
        self._copy_stream_next: dict[int, int] = {}
        self._stop_poller = threading.Event()
        _cuda_set_device(self.device)
        if self.total_bytes > 0:
            self.create_pool(self.total_bytes, self.segment_bytes)
        if os.environ.get("RACER_CSD_PREWARM_CUDA_CONTEXTS", "1") == "1":
            cudart = _load_cudart()
            for cuda_device in range(_visible_cuda_device_count()):
                _cuda_warm_device(cuda_device)
                if cudart is not None and hasattr(cudart, "cudaStreamCreateWithFlags"):
                    self._get_copy_stream(cuda_device)
        self._poller: threading.Thread | None = None
        if os.environ.get("RACER_CSD_ENABLE_POLLER", "0") == "1":
            self._poller = threading.Thread(target=self._poll_loop, name="racer-csd-native-poller", daemon=True)
            self._poller.start()

    def capabilities(self) -> dict[str, Any]:
        stats = self.pool_stats()
        return {
            "backend": self.name,
            "restart_aware": True,
            "daemon_owned": True,
            "cuda_native_pinned": True,
            "uses_cudaHostAlloc": True,
            "uses_memfd_mmap": False,
            "supports_cuda_ipc": True,
            "supports_async_copy": True,
            "pool_total_bytes": stats["pool_total_bytes"],
            "pool_free_bytes": stats["pool_free_bytes"],
            "pool_segment_count": stats["pool_segment_count"],
            "segment_bytes": int(self.segment_bytes),
            "copy_streams_per_device": int(self._copy_stream_pool_size),
        }

    def create_pool(self, total_bytes: int, segment_bytes: int) -> None:
        remaining = int(total_bytes)
        segment_size = max(1, int(segment_bytes))
        while remaining > 0:
            self._new_segment(min(segment_size, remaining))
            remaining -= min(segment_size, remaining)

    def _new_segment(self, nbytes: int) -> _NativeSegment:
        cudart = _load_cudart()
        if cudart is None:
            raise RuntimeError("CUDA runtime is required for cudaHostAlloc")
        ptr = ctypes.c_void_p()
        _cuda_check(
            cudart.cudaHostAlloc(ctypes.byref(ptr), ctypes.c_size_t(int(nbytes)), ctypes.c_uint(_CUDA_HOST_ALLOC_DEFAULT)),
            "cudaHostAlloc failed for CSD native pinned segment",
        )
        segment = _NativeSegment(segment_id=f"seg_{len(self._segments):08d}", ptr=int(ptr.value), nbytes=int(nbytes))
        self._segments.append(segment)
        self._segment_by_id[segment.segment_id] = segment
        return segment

    def pool_stats(self) -> dict[str, int]:
        with self._lock:
            free_bump_bytes = 0
            for segment in self._segments:
                free_bump_bytes += max(0, int(segment.nbytes) - int(segment.offset))
            free_list_bytes = sum(int(block.nbytes) for block in self._free_blocks)
            return {
                "pool_total_bytes": sum(int(segment.nbytes) for segment in self._segments),
                "pool_free_bytes": int(free_bump_bytes + free_list_bytes),
                "pool_free_list_bytes": int(free_list_bytes),
                "pool_segment_count": len(self._segments),
            }

    @staticmethod
    def _align(value: int, alignment: int = 256) -> int:
        value = int(value)
        alignment = max(1, int(alignment))
        return (value + alignment - 1) // alignment * alignment

    def _allocate_location(self, nbytes: int, *, alignment: int = 256) -> _NativeAllocation:
        nbytes = int(nbytes)
        allocate_start = time.perf_counter()
        with self._lock:
            for index, block in enumerate(list(self._free_blocks)):
                aligned_offset = self._align(block.offset, alignment)
                block_end = int(block.offset) + int(block.nbytes)
                if aligned_offset + nbytes > block_end:
                    continue
                before_nbytes = aligned_offset - int(block.offset)
                after_offset = aligned_offset + nbytes
                after_nbytes = block_end - after_offset
                self._free_blocks.pop(index)
                if before_nbytes > 0:
                    self._free_blocks.append(_NativeFreeBlock(block.segment_id, int(block.offset), int(before_nbytes)))
                if after_nbytes > 0:
                    self._free_blocks.append(_NativeFreeBlock(block.segment_id, int(after_offset), int(after_nbytes)))
                self._coalesce_free_blocks_locked()
                segment = self._segment_by_id[block.segment_id]
                return _NativeAllocation(
                    segment=segment,
                    offset=int(aligned_offset),
                    nbytes=nbytes,
                    source="free_list",
                    allocate_ms=(time.perf_counter() - allocate_start) * 1000.0,
                )
            for segment in self._segments:
                offset = self._align(segment.offset, alignment)
                if offset + nbytes <= segment.nbytes:
                    segment.offset = offset + nbytes
                    return _NativeAllocation(
                        segment=segment,
                        offset=int(offset),
                        nbytes=nbytes,
                        source="pool_bump",
                        allocate_ms=(time.perf_counter() - allocate_start) * 1000.0,
                    )
            size = max(self.segment_bytes, self._align(nbytes, alignment))
            segment = self._new_segment(size)
            segment.offset = nbytes
            return _NativeAllocation(
                segment=segment,
                offset=0,
                nbytes=nbytes,
                source="dynamic_cudaHostAlloc",
                allocate_ms=(time.perf_counter() - allocate_start) * 1000.0,
            )

    def _coalesce_free_blocks_locked(self) -> None:
        if not self._free_blocks:
            return
        blocks = sorted(self._free_blocks, key=lambda item: (item.segment_id, int(item.offset)))
        merged: list[_NativeFreeBlock] = []
        for block in blocks:
            if int(block.nbytes) <= 0:
                continue
            if not merged or merged[-1].segment_id != block.segment_id:
                merged.append(_NativeFreeBlock(block.segment_id, int(block.offset), int(block.nbytes)))
                continue
            prev = merged[-1]
            prev_end = int(prev.offset) + int(prev.nbytes)
            if int(block.offset) <= prev_end:
                block_end = int(block.offset) + int(block.nbytes)
                prev.nbytes = max(prev_end, block_end) - int(prev.offset)
            else:
                merged.append(_NativeFreeBlock(block.segment_id, int(block.offset), int(block.nbytes)))
        self._free_blocks = merged

    def _free_location_locked(self, segment_id: str | None, offset: int | None, nbytes: int | None) -> None:
        if not segment_id or segment_id not in self._segment_by_id:
            return
        offset = 0 if offset is None else int(offset)
        nbytes = 0 if nbytes is None else int(nbytes)
        if nbytes <= 0:
            return
        segment = self._segment_by_id[segment_id]
        if offset < 0 or offset + nbytes > int(segment.nbytes):
            return
        self._free_blocks.append(_NativeFreeBlock(str(segment_id), int(offset), int(nbytes)))
        self._coalesce_free_blocks_locked()

    def allocate(self, tag: str, chunk_id: str, nbytes: int, metadata: dict[str, Any] | None = None) -> torch.Tensor:
        # Compatibility path for small CPU tests. Native async paths use
        # allocate_location/write_from_cuda_ipc and do not expose the host
        # pointer to training processes.
        allocation = self._allocate_location(int(nbytes))
        segment = allocation.segment
        offset = allocation.offset
        location = {
            "backend": self.name,
            "segment_id": segment.segment_id,
            "offset": int(offset),
            "nbytes": int(nbytes),
        }
        record_metadata = dict(metadata or {})
        record_metadata.update(
            {
                "storage_backend": self.name,
                "storage_transport": "native_pinned",
                "stored_device": "cpu",
                "daemon_storage_device": "cpu",
                "daemon_owned": True,
                "cuda_native_pinned": True,
                "uses_cudaHostAlloc": True,
                "nbytes": int(nbytes),
                "location": location,
                "segment_id": segment.segment_id,
                "offset": int(offset),
                "allocation_source": allocation.source,
                "allocation_ms": allocation.allocate_ms,
            }
        )
        chunk = BackendChunk(
            tensor=torch.empty(0, dtype=torch.uint8),
            metadata=record_metadata,
            nbytes=int(nbytes),
            capacity_nbytes=int(nbytes),
            host_ptr=int(segment.ptr + offset),
            segment_id=segment.segment_id,
            offset=int(offset),
        )
        with self._lock:
            self._chunks.setdefault(str(tag), {})[str(chunk_id)] = chunk
        return torch.empty(0, dtype=torch.uint8)

    def write_from_bytes(
        self,
        tag: str,
        chunk_id: str,
        data: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("NativePinnedMemoryBackend forbids CPU byte writes; use write_from_cuda_ipc")

    @staticmethod
    def _ipc_cache_key(view: dict[str, Any]) -> str:
        digest = hashlib.sha1(bytes(view["mem_handle"])).hexdigest()
        return f"{int(view.get('producer_pid', -1))}:{int(view['device'])}:{digest}"

    def _open_ipc_mem(self, view: dict[str, Any]) -> tuple[int, int | None, str, float, float, bool, float]:
        cudart = _load_cudart()
        if cudart is None:
            raise RuntimeError("CUDA runtime is required for CUDA IPC")
        set_device_start = time.perf_counter()
        _cuda_set_device(int(view["device"]))
        set_device_ms = (time.perf_counter() - set_device_start) * 1000.0
        cacheable = bool(view.get("requires_staging")) and bool(view.get("staging_id"))
        cache_key = self._ipc_cache_key(view) if cacheable else ""
        open_start = time.perf_counter()
        opened_new = False
        remote_ptr_value: int
        if cacheable:
            with self._ipc_lock:
                cached = self._ipc_mem_cache.get(cache_key)
                if cached is not None:
                    cached["refcount"] = int(cached.get("refcount", 0)) + 1
                    remote_ptr_value = int(cached["ptr"])
                else:
                    remote_ptr = ctypes.c_void_p()
                    mem_handle = _CudaIpcMemHandle()
                    raw_mem_handle = bytes(view["mem_handle"])
                    if len(raw_mem_handle) != 64:
                        raise RuntimeError(f"CUDA IPC mem handle must be 64 bytes, got {len(raw_mem_handle)}")
                    ctypes.memmove(ctypes.byref(mem_handle), raw_mem_handle, 64)
                    _cuda_check(
                        cudart.cudaIpcOpenMemHandle(
                            ctypes.byref(remote_ptr),
                            mem_handle,
                            ctypes.c_uint(_CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),
                        ),
                        "cudaIpcOpenMemHandle failed",
                    )
                    remote_ptr_value = int(remote_ptr.value)
                    self._ipc_mem_cache[cache_key] = {
                        "ptr": remote_ptr_value,
                        "refcount": 1,
                        "device": int(view["device"]),
                    }
                    opened_new = True
        else:
            remote_ptr = ctypes.c_void_p()
            mem_handle = _CudaIpcMemHandle()
            raw_mem_handle = bytes(view["mem_handle"])
            if len(raw_mem_handle) != 64:
                raise RuntimeError(f"CUDA IPC mem handle must be 64 bytes, got {len(raw_mem_handle)}")
            ctypes.memmove(ctypes.byref(mem_handle), raw_mem_handle, 64)
            _cuda_check(
                cudart.cudaIpcOpenMemHandle(
                    ctypes.byref(remote_ptr),
                    mem_handle,
                    ctypes.c_uint(_CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),
                ),
                "cudaIpcOpenMemHandle failed",
            )
            remote_ptr_value = int(remote_ptr.value)
            opened_new = True
        ipc_open_us = (time.perf_counter() - open_start) * 1_000_000.0
        event = ctypes.c_void_p()
        event_handle = _CudaIpcEventHandle()
        raw_event_handle = bytes(view["event_handle"])
        if len(raw_event_handle) != 64:
            raise RuntimeError(f"CUDA IPC event handle must be 64 bytes, got {len(raw_event_handle)}")
        ctypes.memmove(ctypes.byref(event_handle), raw_event_handle, 64)
        event_open_start = time.perf_counter()
        _cuda_check(
            cudart.cudaIpcOpenEventHandle(ctypes.byref(event), event_handle),
            "cudaIpcOpenEventHandle failed",
        )
        event_open_us = (time.perf_counter() - event_open_start) * 1_000_000.0
        return remote_ptr_value, int(event.value), (cache_key if cacheable else ""), ipc_open_us, event_open_us, opened_new, set_device_ms

    def _get_copy_stream(self, device: int) -> int:
        cudart = _load_cudart()
        assert cudart is not None
        device = int(device)
        _cuda_set_device(int(device))
        with self._stream_lock:
            streams = self._copy_streams.get(device)
            if streams is None:
                streams = []
                for _ in range(self._copy_stream_pool_size):
                    stream = ctypes.c_void_p()
                    _cuda_check(
                        cudart.cudaStreamCreateWithFlags(
                            ctypes.byref(stream),
                            ctypes.c_uint(_CUDA_STREAM_NON_BLOCKING),
                        ),
                        "cudaStreamCreateWithFlags failed",
                    )
                    streams.append(int(stream.value))
                self._copy_streams[device] = streams
                self._copy_stream_next[device] = 0
            index = int(self._copy_stream_next.get(device, 0)) % len(streams)
            self._copy_stream_next[device] = index + 1
            return int(streams[index])

    def _new_stream_and_events(self, device: int) -> tuple[int, int, int, int, bool]:
        cudart = _load_cudart()
        if cudart is None:
            raise RuntimeError("CUDA runtime is required for async copy")
        stream = self._get_copy_stream(int(device))
        event_flags = 0 if os.environ.get("RACER_CSD_CUDA_EVENT_TIMING", "0") == "1" else _CUDA_EVENT_DISABLE_TIMING
        wait_start = ctypes.c_void_p()
        copy_start = ctypes.c_void_p()
        complete = ctypes.c_void_p()
        _cuda_check(
            cudart.cudaEventCreateWithFlags(ctypes.byref(wait_start), ctypes.c_uint(event_flags)),
            "cudaEventCreateWithFlags failed",
        )
        _cuda_check(
            cudart.cudaEventCreateWithFlags(ctypes.byref(copy_start), ctypes.c_uint(event_flags)),
            "cudaEventCreateWithFlags failed",
        )
        _cuda_check(
            cudart.cudaEventCreateWithFlags(ctypes.byref(complete), ctypes.c_uint(event_flags)),
            "cudaEventCreateWithFlags failed",
        )
        return stream, int(wait_start.value), int(copy_start.value), int(complete.value), False

    def write_from_cuda_ipc(
        self,
        tag: str,
        chunk_id: str,
        view: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        nbytes = int(view["nbytes"])
        allocation = self._allocate_location(nbytes)
        segment = allocation.segment
        offset = allocation.offset
        location = {"backend": self.name, "segment_id": segment.segment_id, "offset": int(offset), "nbytes": nbytes}
        record_metadata = dict(metadata or {})
        record_metadata.update(
            {
                "storage_backend": self.name,
                "storage_transport": "native_pinned",
                "stored_device": "cpu",
                "daemon_storage_device": "cpu",
                "daemon_owned": True,
                "cuda_native_pinned": True,
                "uses_cudaHostAlloc": True,
                "supports_cuda_ipc": True,
                "nbytes": nbytes,
                "valid_nbytes": int(record_metadata.get("valid_nbytes", nbytes)),
                "location": location,
                "segment_id": segment.segment_id,
                "offset": int(offset),
                "allocation_source": allocation.source,
                "allocation_ms": allocation.allocate_ms,
            }
        )
        chunk = BackendChunk(
            tensor=torch.empty(0, dtype=torch.uint8),
            metadata=record_metadata,
            nbytes=nbytes,
            capacity_nbytes=nbytes,
            host_ptr=int(segment.ptr + offset),
            segment_id=segment.segment_id,
            offset=int(offset),
        )
        remote_ptr, remote_event, cache_key, ipc_open_us, event_open_us, opened_new, set_device_ms = self._open_ipc_mem(view)
        event_create_start = time.perf_counter()
        stream, wait_start, copy_start, complete, stream_owned = self._new_stream_and_events(int(view["device"]))
        event_create_ms = (time.perf_counter() - event_create_start) * 1000.0
        cudart = _load_cudart()
        assert cudart is not None
        enqueue_api_start = time.perf_counter()
        _cuda_check(
            cudart.cudaEventRecord(ctypes.c_void_p(wait_start), ctypes.c_void_p(stream)),
            "cudaEventRecord wait_start failed",
        )
        _cuda_check(
            cudart.cudaStreamWaitEvent(ctypes.c_void_p(stream), ctypes.c_void_p(remote_event), ctypes.c_uint(0)),
            "cudaStreamWaitEvent failed for source IPC event",
        )
        _cuda_check(
            cudart.cudaEventRecord(ctypes.c_void_p(copy_start), ctypes.c_void_p(stream)),
            "cudaEventRecord copy_start failed",
        )
        src_ptr = int(remote_ptr) + int(view.get("base_offset", 0))
        copy_wall_start = time.perf_counter()
        _cuda_check(
            cudart.cudaMemcpyAsync(
                ctypes.c_void_p(int(chunk.host_ptr)),
                ctypes.c_void_p(src_ptr),
                ctypes.c_size_t(nbytes),
                ctypes.c_int(_CUDA_MEMCPY_DEFAULT),
                ctypes.c_void_p(stream),
            ),
            "cudaMemcpyAsync D2H to native pinned failed",
        )
        _cuda_check(cudart.cudaEventRecord(ctypes.c_void_p(complete), ctypes.c_void_p(stream)), "cudaEventRecord failed")
        enqueue_api_ms = (time.perf_counter() - enqueue_api_start) * 1000.0
        enqueue_wall_ms = (time.perf_counter() - copy_wall_start) * 1000.0
        op_id = str(uuid.uuid4())
        with self._lock:
            self._chunks.setdefault(str(tag), {})[str(chunk_id)] = chunk
            self._ops[op_id] = _NativeCopyOp(
                op_id=op_id,
                device=int(view["device"]),
                stream=stream,
                wait_start_event=wait_start,
                copy_start_event=copy_start,
                complete_event=complete,
                remote_ptr=remote_ptr,
                remote_event=remote_event,
                cache_key=cache_key,
                stream_owned=stream_owned,
                profile={
                    "daemon_allocate_ms": allocation.allocate_ms,
                    "daemon_allocate_source": allocation.source,
                    "daemon_ipc_open_us": ipc_open_us,
                    "daemon_ipc_event_open_us": event_open_us,
                    "daemon_ipc_opened_new": opened_new,
                    "daemon_set_device_ms": set_device_ms,
                    "daemon_event_create_ms": event_create_ms,
                    "daemon_enqueue_api_ms": enqueue_api_ms,
                    "daemon_memcpy_enqueue_wall_ms": enqueue_wall_ms,
                    "daemon_event_wait_ms": 0.0,
                    "daemon_memcpy_ms_cuda_event": 0.0,
                    "daemon_memcpy_ms_wall": 0.0,
                },
            )
        return op_id, dict(record_metadata)

    def read_to_cuda_ipc(self, tag: str, chunk_id: str, view: dict[str, Any]) -> str:
        with self._lock:
            chunk = self._chunks[str(tag)][str(chunk_id)]
        remote_ptr, remote_event, cache_key, ipc_open_us, event_open_us, opened_new, set_device_ms = self._open_ipc_mem(view)
        stream, wait_start, copy_start, complete, stream_owned = self._new_stream_and_events(int(view["device"]))
        cudart = _load_cudart()
        assert cudart is not None
        _cuda_check(
            cudart.cudaEventRecord(ctypes.c_void_p(wait_start), ctypes.c_void_p(stream)),
            "cudaEventRecord wait_start failed",
        )
        _cuda_check(
            cudart.cudaStreamWaitEvent(ctypes.c_void_p(stream), ctypes.c_void_p(remote_event), ctypes.c_uint(0)),
            "cudaStreamWaitEvent failed for destination IPC event",
        )
        _cuda_check(
            cudart.cudaEventRecord(ctypes.c_void_p(copy_start), ctypes.c_void_p(stream)),
            "cudaEventRecord copy_start failed",
        )
        dst_ptr = int(remote_ptr) + int(view.get("base_offset", 0))
        copy_wall_start = time.perf_counter()
        _cuda_check(
            cudart.cudaMemcpyAsync(
                ctypes.c_void_p(dst_ptr),
                ctypes.c_void_p(int(chunk.host_ptr)),
                ctypes.c_size_t(int(chunk.nbytes)),
                ctypes.c_int(_CUDA_MEMCPY_DEFAULT),
                ctypes.c_void_p(stream),
            ),
            "cudaMemcpyAsync H2D from native pinned failed",
        )
        _cuda_check(cudart.cudaEventRecord(ctypes.c_void_p(complete), ctypes.c_void_p(stream)), "cudaEventRecord failed")
        enqueue_wall_ms = (time.perf_counter() - copy_wall_start) * 1000.0
        op_id = str(uuid.uuid4())
        with self._lock:
            self._ops[op_id] = _NativeCopyOp(
                op_id=op_id,
                device=int(view["device"]),
                stream=stream,
                wait_start_event=wait_start,
                copy_start_event=copy_start,
                complete_event=complete,
                remote_ptr=remote_ptr,
                remote_event=remote_event,
                cache_key=cache_key,
                stream_owned=stream_owned,
                profile={
                    "daemon_ipc_open_us": ipc_open_us,
                    "daemon_ipc_event_open_us": event_open_us,
                    "daemon_ipc_opened_new": opened_new,
                    "daemon_set_device_ms": set_device_ms,
                    "daemon_memcpy_enqueue_wall_ms": enqueue_wall_ms,
                    "daemon_event_wait_ms": 0.0,
                    "daemon_memcpy_ms_cuda_event": 0.0,
                    "daemon_memcpy_ms_wall": 0.0,
                },
            )
        return op_id

    def _complete_op_locked(self, op: _NativeCopyOp) -> None:
        if op.resources_released:
            op.done = True
            return
        cudart = _load_cudart()
        assert cudart is not None
        try:
            profile = dict(op.profile or {})
            try:
                profile["daemon_event_wait_ms"] = _cuda_event_elapsed_ms(op.wait_start_event, op.copy_start_event)
                profile["daemon_memcpy_ms_cuda_event"] = _cuda_event_elapsed_ms(op.copy_start_event, op.complete_event)
            except Exception as exc:
                profile["daemon_event_timing_error"] = str(exc)
            op.profile = profile
            op.done = True
        except BaseException as exc:
            op.error = str(exc)
            raise
        finally:
            if op.remote_event is not None:
                try:
                    cudart.cudaEventDestroy(ctypes.c_void_p(int(op.remote_event)))
                except Exception:
                    pass
            for event in (op.wait_start_event, op.copy_start_event, op.complete_event):
                try:
                    cudart.cudaEventDestroy(ctypes.c_void_p(int(event)))
                except Exception:
                    pass
            if not op.cache_key and op.remote_ptr is not None:
                try:
                    cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(int(op.remote_ptr)))
                except Exception:
                    pass
            if op.stream_owned:
                try:
                    cudart.cudaStreamDestroy(ctypes.c_void_p(int(op.stream)))
                except Exception:
                    pass
            op.resources_released = True

    def wait(self, op_id: str) -> None:
        cudart = _load_cudart()
        assert cudart is not None
        with self._lock:
            op = self._ops[str(op_id)]
            if op.error:
                raise RuntimeError(op.error)
            if op.done:
                if op.error:
                    raise RuntimeError(op.error)
                return
        _cuda_set_device(int(op.device))
        wait_start = time.perf_counter()
        try:
            _cuda_check(cudart.cudaEventSynchronize(ctypes.c_void_p(op.complete_event)), "native pinned copy wait failed")
            wait_wall_ms = (time.perf_counter() - wait_start) * 1000.0
            profile = dict(op.profile or {})
            profile["daemon_memcpy_ms_wall"] = wait_wall_ms
            op.profile = profile
            with self._lock:
                if not op.done:
                    self._complete_op_locked(op)
        except BaseException as exc:
            with self._lock:
                op.error = str(exc)
            raise

    def poll(self, op_id: str) -> str:
        with self._lock:
            op = self._ops[str(op_id)]
        if op.done:
            return "FAILED" if op.error else "DONE"
        cudart = _load_cudart()
        assert cudart is not None
        _cuda_set_device(int(op.device))
        err = int(cudart.cudaEventQuery(ctypes.c_void_p(int(op.complete_event))))
        if err == 0:
            with self._lock:
                self._complete_op_locked(op)
            return "DONE"
        if err == _CUDA_ERROR_NOT_PERMITTED:
            with self._lock:
                profile = dict(op.profile or {})
                profile["daemon_event_query_deferred_error_code"] = err
                op.profile = profile
            return "RUNNING"
        if err != _CUDA_ERROR_NOT_READY:
            message = f"cudaEventQuery failed for native pinned op {op_id}: CUDA error code {err}"
            with self._lock:
                op.error = message
            return "FAILED"
        return "RUNNING"

    def profile(self, op_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._ops[str(op_id)].profile or {})

    def _poll_loop(self) -> None:
        while not self._stop_poller.is_set():
            with self._lock:
                op_ids = [op_id for op_id, op in self._ops.items() if not op.done]
            for op_id in op_ids:
                try:
                    self.poll(op_id)
                except Exception:
                    pass
            self._stop_poller.wait(0.0005)

    def checksum(self, tag: str, chunk_id: str, checksum_type: str = "sha256") -> str:
        with self._lock:
            chunk = self._chunks[str(tag)][str(chunk_id)]
        view = memoryview((ctypes.c_ubyte * int(chunk.nbytes)).from_address(int(chunk.host_ptr)))
        normalized = str(checksum_type).lower().replace("-", "_")
        if normalized in {"sample64", "sample64_v1", "fast", "sampled"}:
            return _sample64_checksum_view(view)
        if normalized in {"xxh64", "xxhash64", "xxh64_v1"}:
            if _xxhash is None:
                raise RuntimeError("checksum_type=xxh64 requires the optional xxhash package")
            return _xxhash.xxh64(view).hexdigest()
        if normalized in {"crc32", "crc32_v1"}:
            return f"{zlib.crc32(view) & 0xFFFFFFFF:08x}"
        return hashlib.sha256(view).hexdigest()

    def read_to_bytes(self, tag: str, chunk_id: str) -> bytes:
        with self._lock:
            chunk = self._chunks[str(tag)][str(chunk_id)]
        return ctypes.string_at(ctypes.c_void_p(int(chunk.host_ptr)), int(chunk.nbytes))

    def metadata(self, tag: str, chunk_id: str) -> dict[str, Any]:
        try:
            with self._lock:
                return dict(self._chunks[str(tag)][str(chunk_id)].metadata)
        except KeyError as exc:
            raise KeyError(f"CSD native_pinned chunk metadata {chunk_id!r} for tag {tag!r} is not resident") from exc

    def list_chunks(self, tag: str) -> list[str]:
        with self._lock:
            return sorted(self._chunks.get(str(tag), {}))

    def free(self, tag: str, chunk_id: str) -> None:
        with self._lock:
            chunks = self._chunks.get(str(tag))
            if chunks is None:
                return
            chunk = chunks.pop(str(chunk_id), None)
            if chunk is not None:
                self._free_location_locked(chunk.segment_id, chunk.offset, chunk.capacity_nbytes)
            if not chunks:
                self._chunks.pop(str(tag), None)

    def free_tag(self, tag: str) -> None:
        with self._lock:
            chunks = self._chunks.pop(str(tag), None)
            if not chunks:
                return
            for chunk in chunks.values():
                self._free_location_locked(chunk.segment_id, chunk.offset, chunk.capacity_nbytes)

    def __del__(self) -> None:
        try:
            self._stop_poller.set()
        except Exception:
            pass
        try:
            poller = getattr(self, "_poller", None)
            if poller is not None:
                poller.join(timeout=1.0)
        except Exception:
            pass
        cudart = _load_cudart()
        if cudart is None:
            return
        for cached in getattr(self, "_ipc_mem_cache", {}).values():
            try:
                cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(int(cached["ptr"])))
            except Exception:
                pass
        for streams in getattr(self, "_copy_streams", {}).values():
            for stream in streams:
                try:
                    cudart.cudaStreamDestroy(ctypes.c_void_p(int(stream)))
                except Exception:
                    pass
        for segment in getattr(self, "_segments", []):
            try:
                cudart.cudaFreeHost(ctypes.c_void_p(int(segment.ptr)))
            except Exception:
                pass


class EgmBackend(StorageBackend):
    """Daemon-owned EGM allocation backend.

    EGM is only valid here when a real daemon-side runtime can allocate and
    export/import EGM handles. A torch.cuda.MemPool by itself is not enough,
    because it does not give the CSD a restart-aware native handle transport.
    CPU byte and fd/mmap fallbacks are intentionally forbidden for this backend.
    """

    name = "egm"
    page_size = 2 * 1024 * 1024
    _REQUIRED_RUNTIME_METHODS = ("write_from_cuda_ipc", "read_to_cuda_ipc")

    def __init__(
        self,
        *,
        runtime: Any | None = None,
        mem_pool: Any | None = None,
        pool_id: str | None = None,
        owner_node: str | None = None,
        owner_tray: str | None = None,
        home_device: int | None = None,
        numa_id: int | None = None,
        accessing_devices: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        if runtime is None:
            if mem_pool is not None:
                raise NotImplementedError(
                    "EgmBackend requires a native daemon-owned EGM runtime with handle transport; "
                    "torch.cuda.MemPool alone would force a copy-style fallback, which is forbidden"
                )
            raise NotImplementedError(
                "EgmBackend requires a real daemon-owned EGM runtime with native handle transport; "
                "host-memory fallback is forbidden"
            )
        missing = [name for name in self._REQUIRED_RUNTIME_METHODS if not hasattr(runtime, name)]
        if missing:
            raise TypeError(f"EGM runtime is missing required methods: {', '.join(missing)}")
        self.runtime = runtime
        self.mem_pool = mem_pool
        self.pool_id = pool_id
        self.owner_node = owner_node
        self.owner_tray = owner_tray
        self.home_device = 0 if home_device is None else int(home_device)
        self.numa_id = None if numa_id is None else int(numa_id)
        self.accessing_devices = tuple(int(device) for device in accessing_devices or ())
        self._metadata: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.RLock()

    def capabilities(self) -> dict[str, Any]:
        runtime_caps_fn = getattr(self.runtime, "capabilities", None)
        runtime_caps = dict(runtime_caps_fn()) if callable(runtime_caps_fn) else {}
        return {
            "backend": self.name,
            "restart_aware": True,
            "daemon_owned": True,
            "cuda_native_pinned": False,
            "egm_native": True,
            "uses_cudaHostAlloc": False,
            "uses_memfd_mmap": False,
            "supports_cuda_ipc": True,
            "supports_async_copy": True,
            "supports_egm_native_transport": True,
            "supports_zero_copy_region": bool(runtime_caps.get("supports_zero_copy_region", True)),
            "pool_id": self.pool_id,
            "owner_node": self.owner_node,
            "owner_tray": self.owner_tray,
            "home_device": int(self.home_device),
            "numa_id": self.numa_id,
            "page_size": int(self.page_size),
            **runtime_caps,
        }

    def allocate(self, tag: str, chunk_id: str, nbytes: int, metadata: dict[str, Any] | None = None) -> torch.Tensor:
        raise NotImplementedError(
            "EgmBackend does not expose tensor allocation or CPU byte storage. "
            "Use write_from_cuda_ipc/read_to_cuda_ipc with a native EGM runtime handle."
        )

    def _normalize_metadata(
        self,
        tag: str,
        chunk_id: str,
        nbytes: int,
        metadata: dict[str, Any] | None,
        runtime_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record_metadata = dict(metadata or {})
        record_metadata.update(dict(runtime_metadata or {}))
        location = dict(record_metadata.get("location") or {})
        allocation_id = (
            record_metadata.get("allocation_id")
            or record_metadata.get("egm_allocation_id")
            or location.get("allocation_id")
            or location.get("egm_allocation_id")
        )
        if allocation_id is None:
            allocation_id = f"{_safe_tag(tag)}:{_safe_tag(chunk_id)}"
        offset = int(record_metadata.get("offset", location.get("offset", 0)) or 0)
        access_handle = record_metadata.get("access_handle", location.get("access_handle"))
        location.update(
            {
                "backend": self.name,
                "pool_id": record_metadata.get("pool_id", self.pool_id),
                "allocation_id": allocation_id,
                "offset": offset,
                "nbytes": int(nbytes),
                "owner_node": record_metadata.get("owner_node", self.owner_node),
                "owner_tray": record_metadata.get("owner_tray", self.owner_tray),
                "access_handle": access_handle,
            }
        )
        record_metadata.update(
            {
                "storage_backend": self.name,
                "storage_transport": "egm_native",
                "stored_device": f"cuda:{self.home_device}",
                "daemon_owned": True,
                "egm_native": True,
                "egm_pool_id": location.get("pool_id"),
                "egm_allocation_id": allocation_id,
                "egm_offset": offset,
                "egm_owner_node": location.get("owner_node"),
                "egm_owner_tray": location.get("owner_tray"),
                "egm_access_handle": access_handle,
                "egm_home_device": int(self.home_device),
                "egm_numa_id": self.numa_id,
                "egm_accessing_devices": list(self.accessing_devices),
                "egm_page_size": int(self.page_size),
                "nbytes": int(nbytes),
                "offset": offset,
                "location": location,
                "committed": bool(record_metadata.get("committed", False)),
                "checksum": record_metadata.get("checksum", ""),
            }
        )
        with self._lock:
            self._metadata.setdefault(str(tag), {})[str(chunk_id)] = dict(record_metadata)
        return record_metadata

    def write_from_bytes(
        self,
        tag: str,
        chunk_id: str,
        data: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("EgmBackend forbids CPU/socket byte put; use native EGM handle transport")

    def write_from_cuda(
        self,
        tag: str,
        chunk_id: str,
        tensor: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("EgmBackend forbids copy-style CUDA put; use write_from_cuda_ipc")

    def read_to_bytes(self, tag: str, chunk_id: str) -> bytes:
        raise NotImplementedError("EgmBackend forbids CPU/socket byte get; use native EGM handle transport")

    def read_to_cuda(self, tag: str, chunk_id: str, device: torch.device | str) -> torch.Tensor:
        raise NotImplementedError("EgmBackend forbids copy-style CUDA get; use read_to_cuda_ipc")

    def export_fd_region(self, tag: str, chunk_id: str) -> tuple[dict[str, Any], int]:
        raise NotImplementedError("EgmBackend does not support fd/mmap fallback transport")

    def prepare_fd_region(
        self,
        tag: str,
        chunk_id: str,
        nbytes: int,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int]:
        raise NotImplementedError("EgmBackend does not support fd/mmap fallback transport")

    def write_from_cuda_ipc(
        self,
        tag: str,
        chunk_id: str,
        view: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        result = self.runtime.write_from_cuda_ipc(str(tag), str(chunk_id), dict(view), dict(metadata or {}))
        if isinstance(result, tuple):
            op_id, runtime_metadata = result
        elif isinstance(result, dict):
            op_id = result.get("op_id", str(uuid.uuid4()))
            runtime_metadata = result
        else:
            op_id = str(result)
            runtime_metadata = {}
        nbytes = int(runtime_metadata.get("nbytes", dict(view).get("nbytes", dict(metadata or {}).get("nbytes", 0))) or 0)
        record_metadata = self._normalize_metadata(str(tag), str(chunk_id), nbytes, metadata, runtime_metadata)
        return str(op_id), record_metadata

    def read_to_cuda_ipc(self, tag: str, chunk_id: str, view: dict[str, Any]) -> str:
        result = self.runtime.read_to_cuda_ipc(str(tag), str(chunk_id), dict(view))
        if isinstance(result, dict):
            return str(result.get("op_id", str(uuid.uuid4())))
        return str(result)

    def wait(self, op_id: str) -> None:
        wait_fn = getattr(self.runtime, "wait", None)
        if callable(wait_fn):
            wait_fn(str(op_id))

    def poll(self, op_id: str) -> str:
        poll_fn = getattr(self.runtime, "poll", None)
        if callable(poll_fn):
            return str(poll_fn(str(op_id)))
        return "DONE"

    def profile(self, op_id: str) -> dict[str, Any]:
        profile_fn = getattr(self.runtime, "profile", None)
        if callable(profile_fn):
            return dict(profile_fn(str(op_id)))
        return {}

    def metadata(self, tag: str, chunk_id: str) -> dict[str, Any]:
        try:
            with self._lock:
                local = dict(self._metadata[str(tag)][str(chunk_id)])
        except KeyError as exc:
            raise KeyError(f"CSD EGM chunk metadata {chunk_id!r} for tag {tag!r} is not resident") from exc
        runtime_metadata_fn = getattr(self.runtime, "metadata", None)
        if callable(runtime_metadata_fn):
            try:
                local.update(dict(runtime_metadata_fn(str(tag), str(chunk_id))))
            except KeyError:
                pass
        return self._normalize_metadata(str(tag), str(chunk_id), int(local.get("nbytes", 0) or 0), local)

    def list_chunks(self, tag: str) -> list[str]:
        runtime_list_fn = getattr(self.runtime, "list_chunks", None)
        if callable(runtime_list_fn):
            return sorted(str(item) for item in runtime_list_fn(str(tag)))
        with self._lock:
            return sorted(self._metadata.get(str(tag), {}))

    def free(self, tag: str, chunk_id: str) -> None:
        free_fn = getattr(self.runtime, "free", None)
        if callable(free_fn):
            free_fn(str(tag), str(chunk_id))
        with self._lock:
            chunks = self._metadata.get(str(tag))
            if chunks is None:
                return
            chunks.pop(str(chunk_id), None)
            if not chunks:
                self._metadata.pop(str(tag), None)


class CheckpointStorageDaemon:
    def __init__(self, backend: StorageBackend, *, metadata_dir: str | Path | None = None) -> None:
        self.backend = backend
        self.metadata_dir = None if metadata_dir is None else Path(metadata_dir)
        self.manifest_store: CsdManifestStore | None = None
        self._entries: dict[str, dict[str, Any]] = {}
        self._async_ops: dict[str, dict[str, Any]] = {}
        self._pending_seals: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.RLock()
        self._put_executor = ThreadPoolExecutor(
            max_workers=max(1, int(os.environ.get("RACER_CSD_PUT_WORKERS", "8"))),
            thread_name_prefix="racer-csd-put",
        )
        self._prewarm_put_workers()
        if self.metadata_dir is not None:
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
            self.manifest_store = CsdManifestStore(self.metadata_dir / "csd_manifest.sqlite")
            self._load_metadata_index()

    def close(self) -> None:
        self._put_executor.shutdown(wait=True, cancel_futures=False)

    def _prewarm_put_workers(self) -> None:
        if os.environ.get("RACER_CSD_PREWARM_PUT_WORKERS", "1") != "1":
            return
        device_count = _visible_cuda_device_count()
        if device_count <= 0:
            return
        worker_count = max(1, int(os.environ.get("RACER_CSD_PUT_WORKERS", "8")))
        start = threading.Event()

        def _warm_worker() -> None:
            start.wait()
            for cuda_device in range(device_count):
                _cuda_warm_device(cuda_device)

        futures = [self._put_executor.submit(_warm_worker) for _ in range(worker_count)]
        start.set()
        for future in futures:
            future.result()

    def _metadata_path(self, tag: str) -> Path:
        if self.metadata_dir is None:
            raise RuntimeError("metadata_dir is not configured")
        return self.metadata_dir / f"{_safe_tag(tag)}.pt"

    def _persist(self, tag: str) -> None:
        if self.metadata_dir is None:
            return
        with self._lock:
            entry = dict(self._entries[str(tag)])
        entry["persisted_at"] = time.time()
        path = self._metadata_path(str(tag))
        tmp = path.with_suffix(".tmp")
        torch.save(entry, tmp)
        tmp.replace(path)

    def _load_metadata_index(self) -> None:
        assert self.metadata_dir is not None
        for path in self.metadata_dir.glob("*.pt"):
            try:
                entry = torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                continue
            if not isinstance(entry, dict) or "tag" not in entry:
                continue
            tag = str(entry["tag"])
            entry.setdefault("chunks", {})
            entry["data_resident"] = False
            with self._lock:
                self._entries[tag] = entry

    def begin(
        self,
        tag: str,
        manifest_base: dict[str, Any] | None = None,
        expected_chunks: int | None = None,
    ) -> dict[str, Any]:
        actual = str(tag)
        manifest = dict(manifest_base or {})
        if self.manifest_store is not None:
            self.manifest_store.begin_checkpoint(
                actual,
                manifest,
                expected_chunks=expected_chunks,
                backend=self.backend.name,
            )
        with self._lock:
            self._entries[actual] = {
                "tag": actual,
                "manifest": manifest,
                "chunks": {},
                "committed": False,
                "data_resident": True,
                "storage_backend": self.backend.name,
                "backend_capabilities": self.backend.capabilities(),
            }
        self._persist(actual)
        return {"tag": actual, "committed": False, "backend": self.backend.name, "capabilities": self.backend.capabilities()}

    def put_chunk(self, tag: str, chunk_id: str, data: bytes, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        raise RuntimeError("CSD socket byte put is disabled; use put_cuda_tensor/CUDA IPC")

    def prepare_fd_chunk(
        self,
        tag: str,
        chunk_id: str,
        nbytes: int,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], int]:
        raise RuntimeError("CSD fd/mmap put transport is disabled; use put_cuda_tensor/CUDA IPC")

    def seal_fd_chunk(self, tag: str, chunk_id: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        raise RuntimeError("CSD fd/mmap seal is disabled; use put_cuda_tensor/CUDA IPC")

    def put_manifest(self, tag: str, manifest: dict[str, Any]) -> None:
        actual = str(tag)
        with self._lock:
            if actual not in self._entries:
                self._entries[actual] = {
                    "tag": actual,
                    "manifest": {},
                    "chunks": {},
                    "committed": False,
                    "data_resident": True,
                    "storage_backend": self.backend.name,
                }
            entry = self._entries[actual]
            entry["manifest"] = dict(manifest)
            self._merge_chunk_metadata(entry)
        if self.manifest_store is not None:
            self.manifest_store.update_manifest(actual, dict(manifest))
        self._persist(actual)

    def _merge_chunk_metadata(self, entry: dict[str, Any]) -> None:
        manifest = dict(entry.get("manifest") or {})
        chunk_metadata = dict(entry.get("chunks") or {})
        merged_chunks = []
        for chunk in manifest.get("chunks", []):
            item = dict(chunk)
            item.update(chunk_metadata.get(str(item.get("chunk_id")), {}))
            merged_chunks.append(item)
        if merged_chunks:
            manifest["chunks"] = merged_chunks
            manifest["chunk_owner"] = {chunk["chunk_id"]: chunk["owner_rank"] for chunk in merged_chunks}
            manifest["checksum"] = hashlib.sha256(
                "".join(str(chunk.get("checksum", "")) for chunk in merged_chunks).encode()
            ).hexdigest()
        manifest["committed"] = bool(entry.get("committed", False))
        manifest["data_resident"] = bool(entry.get("data_resident", False))
        manifest["storage_backend"] = self.backend.name
        manifest["backend_capabilities"] = self.backend.capabilities()
        manifest["daemon_owned"] = True
        entry["manifest"] = manifest

    def _flush_pending_seals(self, tag: str) -> float:
        if self.manifest_store is None:
            return 0.0
        actual = str(tag)
        with self._lock:
            seals = list(self._pending_seals.pop(actual, []))
        if not seals:
            return 0.0
        start = time.perf_counter()
        for seal in seals:
            self.manifest_store.seal_chunk(
                actual,
                seal["chunk_id"],
                location=seal["location"],
                checksum_type=seal["checksum_type"],
                checksum=seal["checksum"],
                nbytes=seal["nbytes"],
                valid_nbytes=seal["valid_nbytes"],
            )
        return (time.perf_counter() - start) * 1000.0

    def commit(self, tag: str) -> None:
        actual = str(tag)
        try:
            with self._lock:
                entry = self._entries[actual]
        except KeyError as exc:
            raise KeyError(f"unknown CSD checkpoint tag {actual!r}") from exc
        flush_ms = self._flush_pending_seals(actual)
        if self.manifest_store is not None:
            self.manifest_store.commit_checkpoint(actual)
        with self._lock:
            entry["committed"] = True
            entry["data_resident"] = True
            self._merge_chunk_metadata(entry)
            entry.setdefault("profile", {})["sqlite_batch_flush_ms"] = flush_ms
        self._persist(actual)

    def get_manifest(self, tag: str) -> dict[str, Any]:
        actual = str(tag)
        with self._lock:
            entry = self._entries.get(actual)
            if entry is not None:
                if not bool(entry.get("committed", False)):
                    raise KeyError(f"CSD checkpoint {actual!r} is not committed")
                self._merge_chunk_metadata(entry)
                manifest = dict(entry["manifest"])
            elif self.manifest_store is None:
                raise KeyError(f"unknown CSD checkpoint manifest for tag {actual!r}")
            else:
                manifest = {}
        if self.manifest_store is not None:
            manifest = self.manifest_store.manifest_for_tag(actual)
            manifest["data_resident"] = bool(self.backend.list_chunks(actual))
            manifest["daemon_owned"] = True
            manifest["backend_capabilities"] = self.backend.capabilities()
        return manifest

    def get_chunk(self, tag: str, chunk_id: str) -> tuple[bytes, dict[str, Any]]:
        raise RuntimeError("CSD socket byte get is disabled; use read_into_cuda_tensor/CUDA IPC")

    def get_chunk_fd(self, tag: str, chunk_id: str) -> tuple[dict[str, Any], int]:
        raise RuntimeError("CSD fd/mmap get transport is disabled; use read_into_cuda_tensor/CUDA IPC")

    def get_metadata(self, tag: str, chunk_id: str) -> dict[str, Any]:
        self.get_manifest(str(tag))
        return self.backend.metadata(str(tag), str(chunk_id))

    def put_cuda_ipc(
        self,
        tag: str,
        chunk_id: str,
        view: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        put_total_start = time.perf_counter()
        if not hasattr(self.backend, "write_from_cuda_ipc"):
            raise RuntimeError(f"CSD backend {self.backend.name!r} does not support CUDA IPC put")
        actual = str(tag)
        chunk = str(chunk_id)
        sqlite_ms = 0.0
        if self.manifest_store is not None:
            sqlite_start = time.perf_counter()
            self.manifest_store.reserve_chunk(actual, chunk, metadata or {}, backend=self.backend.name)
            op_id = str(uuid.uuid4())
            self.manifest_store.begin_operation(op_id, actual, chunk, "PUT")
            self.manifest_store.mark_operation_running(op_id)
            self.manifest_store.mark_chunk_copying(actual, chunk, op_id)
            sqlite_ms = (time.perf_counter() - sqlite_start) * 1000.0
        else:
            op_id = str(uuid.uuid4())

        def _run_backend_write() -> tuple[str, dict[str, Any], dict[str, Any]]:
            backend_write_start = time.perf_counter()
            backend_op_id, record_metadata = self.backend.write_from_cuda_ipc(actual, chunk, view, metadata)  # type: ignore[attr-defined]
            backend_write_ms = (time.perf_counter() - backend_write_start) * 1000.0
            return (
                backend_op_id,
                dict(record_metadata),
                {"daemon_put_backend_write_ms": backend_write_ms},
            )

        submit_start = time.perf_counter()
        backend_future = self._put_executor.submit(_run_backend_write)
        submit_ms = (time.perf_counter() - submit_start) * 1000.0
        put_dispatch_ms = (time.perf_counter() - put_total_start) * 1000.0
        put_profile = {
            "sqlite_ms": sqlite_ms,
            "export_profile": dict(view.get("profile") or {}),
            "daemon_put_submit_ms": submit_ms,
            "daemon_put_dispatch_ms": put_dispatch_ms,
            "daemon_put_total_ms": put_dispatch_ms,
        }
        with self._lock:
            self._async_ops[op_id] = {
                "backend_op_id": None,
                "backend_future": backend_future,
                "op_type": "PUT",
                "tag": actual,
                "chunk_id": chunk,
                "metadata": dict(metadata or {}),
                "profile": put_profile,
            }
            entry = self._entries.setdefault(
                actual,
                {
                    "tag": actual,
                    "manifest": {},
                    "chunks": {},
                    "committed": False,
                    "data_resident": True,
                    "storage_backend": self.backend.name,
                    "backend_capabilities": self.backend.capabilities(),
                },
            )
            entry.setdefault("chunks", {}).setdefault(chunk, dict(metadata or {}))
        return {
            "op_id": op_id,
            "backend_op_id": None,
            "metadata": dict(metadata or {}),
            "profile": put_profile,
        }

    def _resolve_backend_future(self, op_id: str, op: dict[str, Any]) -> dict[str, Any]:
        future = op.get("backend_future")
        if not isinstance(future, Future):
            return op
        backend_op_id, record_metadata, future_profile = future.result()
        with self._lock:
            current = self._async_ops[str(op_id)]
            current["backend_op_id"] = backend_op_id
            current["metadata"] = dict(record_metadata)
            current.pop("backend_future", None)
            profile = dict(current.get("profile") or {})
            profile.update(dict(future_profile or {}))
            profile["daemon_put_total_ms"] = float(profile.get("daemon_put_dispatch_ms", 0.0)) + float(
                profile.get("daemon_put_backend_write_ms", 0.0)
            )
            current["profile"] = profile
            self._entries.setdefault(
                current["tag"],
                {
                    "tag": current["tag"],
                    "manifest": {},
                    "chunks": {},
                    "committed": False,
                    "data_resident": True,
                    "storage_backend": self.backend.name,
                    "backend_capabilities": self.backend.capabilities(),
                },
            ).setdefault("chunks", {})[current["chunk_id"]] = dict(record_metadata)
            return dict(current)

    def read_to_cuda_ipc(self, tag: str, chunk_id: str, view: dict[str, Any]) -> dict[str, Any]:
        self.get_manifest(str(tag))
        if not hasattr(self.backend, "read_to_cuda_ipc"):
            raise RuntimeError(f"CSD backend {self.backend.name!r} does not support CUDA IPC read")
        actual = str(tag)
        chunk = str(chunk_id)
        op_id = str(uuid.uuid4())
        if self.manifest_store is not None:
            self.manifest_store.begin_operation(op_id, actual, chunk, "GET")
            self.manifest_store.mark_operation_running(op_id)
        backend_op_id = self.backend.read_to_cuda_ipc(actual, chunk, view)  # type: ignore[attr-defined]
        with self._lock:
            self._async_ops[op_id] = {
                "backend_op_id": backend_op_id,
                "op_type": "GET",
                "tag": actual,
                "chunk_id": chunk,
                "metadata": self.backend.metadata(actual, chunk),
            }
        return {"op_id": op_id, "backend_op_id": backend_op_id}

    def wait(self, op_id: str) -> dict[str, Any]:
        op_id = str(op_id)
        with self._lock:
            op = dict(self._async_ops[op_id])
        try:
            op = self._resolve_backend_future(op_id, op)
            self.backend.wait(op["backend_op_id"])  # type: ignore[attr-defined]
            profile = dict(op.get("profile") or {})
            if hasattr(self.backend, "profile"):
                profile.update(self.backend.profile(op["backend_op_id"]))  # type: ignore[attr-defined]
            if op["op_type"] == "PUT":
                metadata = dict(self.backend.metadata(op["tag"], op["chunk_id"]))
                checksum_type = str(metadata.get("checksum_type", metadata.get("requested_checksum_type", "sha256"))).lower()
                checksum_ms = 0.0
                if checksum_type in {"none", "off", "disabled"}:
                    checksum = ""
                    metadata.update({"checksum_type": "none", "checksum": ""})
                elif checksum_type in {"sample64", "sample64_v1", "fast", "sampled"}:
                    checksum_start = time.perf_counter()
                    checksum = self.backend.checksum(op["tag"], op["chunk_id"], checksum_type="sample64")
                    checksum_ms = (time.perf_counter() - checksum_start) * 1000.0
                    metadata.update({"checksum_type": "sample64", "checksum": checksum})
                elif checksum_type in {"xxh64", "xxhash64", "xxh64_v1"}:
                    checksum_start = time.perf_counter()
                    checksum = self.backend.checksum(op["tag"], op["chunk_id"], checksum_type="xxh64")
                    checksum_ms = (time.perf_counter() - checksum_start) * 1000.0
                    metadata.update({"checksum_type": "xxh64", "checksum": checksum})
                elif checksum_type in {"crc32", "crc32_v1"}:
                    checksum_start = time.perf_counter()
                    checksum = self.backend.checksum(op["tag"], op["chunk_id"], checksum_type="crc32")
                    checksum_ms = (time.perf_counter() - checksum_start) * 1000.0
                    metadata.update({"checksum_type": "crc32", "checksum": checksum})
                else:
                    checksum_start = time.perf_counter()
                    checksum = self.backend.checksum(op["tag"], op["chunk_id"], checksum_type="sha256")
                    checksum_ms = (time.perf_counter() - checksum_start) * 1000.0
                    metadata.update({"checksum_type": "sha256", "checksum": checksum})
                profile["checksum_ms"] = checksum_ms
                with self._lock:
                    self._entries[op["tag"]].setdefault("chunks", {})[op["chunk_id"]] = metadata
                if self.manifest_store is not None:
                    seal = {
                        "chunk_id": op["chunk_id"],
                        "location": dict(metadata.get("location") or {}),
                        "checksum_type": str(metadata.get("checksum_type", "sha256")),
                        "checksum": checksum,
                        "nbytes": int(metadata.get("nbytes", 0) or 0),
                        "valid_nbytes": int(metadata.get("valid_nbytes", metadata.get("nbytes", 0)) or 0),
                    }
                    if str(metadata.get("manifest_update_mode", "")).lower() == "batch":
                        with self._lock:
                            self._pending_seals.setdefault(op["tag"], []).append(seal)
                    else:
                        sqlite_start = time.perf_counter()
                        self.manifest_store.seal_chunk(op["tag"], **seal)
                        profile["sqlite_ms"] = float(profile.get("sqlite_ms", 0.0)) + (time.perf_counter() - sqlite_start) * 1000.0
            if self.manifest_store is not None:
                sqlite_start = time.perf_counter()
                self.manifest_store.mark_operation_done(op_id)
                profile["sqlite_ms"] = float(profile.get("sqlite_ms", 0.0)) + (time.perf_counter() - sqlite_start) * 1000.0
            result = {"op_id": op_id, "state": "DONE", "profile": profile}
            if os.environ.get("RACER_CSD_PROFILE_LOG", "0") == "1":
                print(
                    "[CSD_PROFILE] "
                    + json.dumps(
                        {
                            "op_id": op_id,
                            "op_type": op.get("op_type"),
                            "tag": op.get("tag"),
                            "chunk_id": op.get("chunk_id"),
                            "profile": profile,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            return result
        except BaseException as exc:
            if self.manifest_store is not None:
                self.manifest_store.mark_operation_failed(op_id, str(exc))
            return {"op_id": op_id, "state": "FAILED", "error": str(exc)}

    def poll(self, op_id: str) -> dict[str, Any]:
        op_id = str(op_id)
        with self._lock:
            op = dict(self._async_ops[op_id])
        try:
            state = self.backend.poll(op["backend_op_id"])  # type: ignore[attr-defined]
        except AttributeError:
            state = "RUNNING"
        if state == "DONE":
            return self.wait(op_id)
        return {"op_id": op_id, "state": state}

    def list_chunks(self, tag: str) -> list[str]:
        return self.backend.list_chunks(str(tag))

    def capabilities(self) -> dict[str, Any]:
        return dict(self.backend.capabilities())

    def list_tags(self) -> list[str]:
        if self.manifest_store is not None:
            return self.manifest_store.list_tags(committed_only=True)
        with self._lock:
            return sorted(tag for tag, entry in self._entries.items() if bool(entry.get("committed", False)))

    def delete(self, tag: str) -> None:
        actual = str(tag)
        self.backend.free_tag(actual)
        with self._lock:
            self._entries.pop(actual, None)
        if self.manifest_store is not None:
            self.manifest_store.delete_checkpoint(actual)
        if self.metadata_dir is not None:
            path = self._metadata_path(actual)
            if path.exists():
                path.unlink()


def _backend_from_name(name: str, options: dict[str, Any] | None = None) -> StorageBackend:
    opts = dict(options or {})
    normalized = str(name).lower().replace("-", "_")
    if normalized in {"native_pinned", "cuda_pinned", "daemon_native_pinned"}:
        return NativePinnedMemoryBackend(**opts)
    if normalized in {"egm", "daemon_egm"}:
        return EgmBackend(**opts)
    raise ValueError(
        f"unsupported CSD backend {name!r}; only native_pinned and real EGM native transport are enabled"
    )


def _serve(address, authkey: bytes, backend_name: str, backend_options: dict[str, Any], metadata_dir: str | None, ready_conn=None) -> None:
    daemon = CheckpointStorageDaemon(
        _backend_from_name(backend_name, backend_options),
        metadata_dir=metadata_dir,
    )
    caps = daemon.capabilities()
    visible_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(
        "[CSD] "
        f"backend={daemon.backend.name} "
        f"restart_aware={str(caps.get('restart_aware', False)).lower()} "
        f"cuda_native_pinned={str(caps.get('cuda_native_pinned', False)).lower()} "
        f"capabilities={json.dumps(caps, sort_keys=True)} "
        f"cuda_devices_visible={visible_devices}",
        flush=True,
    )
    if isinstance(address, str):
        Path(address).parent.mkdir(parents=True, exist_ok=True)
        try:
            Path(address).unlink()
        except FileNotFoundError:
            pass
    listener = Listener(address, authkey=authkey)
    shutdown_event = threading.Event()
    workers: set[threading.Thread] = set()
    if ready_conn is not None:
        ready_conn.send(listener.address)
        ready_conn.close()

    def handle_connection(conn) -> None:
        nonlocal listener
        try:
            while not shutdown_event.is_set():
                try:
                    request = conn.recv()
                except EOFError:
                    break
                op = request.get("op")
                try:
                    if op == "begin":
                        result = daemon.begin(
                            request["tag"],
                            request.get("manifest_base"),
                            expected_chunks=request.get("expected_chunks"),
                        )
                    elif op == "prepare_fd_chunk":
                        result, fd = daemon.prepare_fd_chunk(
                            request["tag"],
                            request["chunk_id"],
                            int(request["nbytes"]),
                            request.get("metadata"),
                        )
                        conn.send({"ok": True, "result": result})
                        send_handle(conn, fd, int(request["pid"]))
                        continue
                    elif op == "seal_fd_chunk":
                        result = daemon.seal_fd_chunk(
                            request["tag"],
                            request["chunk_id"],
                            request.get("metadata"),
                        )
                    elif op == "put_chunk":
                        result = daemon.put_chunk(
                            request["tag"],
                            request["chunk_id"],
                            request["data"],
                            request.get("metadata"),
                        )
                    elif op == "put_manifest":
                        daemon.put_manifest(request["tag"], request["manifest"])
                        result = None
                    elif op == "commit":
                        daemon.commit(request["tag"])
                        result = None
                    elif op == "get_manifest":
                        result = daemon.get_manifest(request["tag"])
                    elif op == "get_chunk":
                        data, metadata = daemon.get_chunk(request["tag"], request["chunk_id"])
                        result = {"data": data, "metadata": metadata}
                    elif op == "get_chunk_fd":
                        result, fd = daemon.get_chunk_fd(request["tag"], request["chunk_id"])
                        conn.send({"ok": True, "result": result})
                        send_handle(conn, fd, int(request["pid"]))
                        continue
                    elif op == "get_metadata":
                        result = daemon.get_metadata(request["tag"], request["chunk_id"])
                    elif op == "put_cuda_ipc":
                        result = daemon.put_cuda_ipc(
                            request["tag"],
                            request["chunk_id"],
                            request["view"],
                            request.get("metadata"),
                        )
                    elif op == "read_to_cuda_ipc":
                        result = daemon.read_to_cuda_ipc(
                            request["tag"],
                            request["chunk_id"],
                            request["view"],
                        )
                    elif op == "wait":
                        result = daemon.wait(request["op_id"])
                    elif op == "poll":
                        result = daemon.poll(request["op_id"])
                    elif op == "capabilities":
                        result = daemon.capabilities()
                    elif op == "list_chunks":
                        result = daemon.list_chunks(request["tag"])
                    elif op == "list_tags":
                        result = daemon.list_tags()
                    elif op == "delete":
                        daemon.delete(request["tag"])
                        result = None
                    elif op == "shutdown":
                        shutdown_event.set()
                        result = None
                    else:
                        raise ValueError(f"unknown CSD op {op!r}")
                    conn.send({"ok": True, "result": result})
                except BaseException as exc:
                    conn.send({"ok": False, "error_type": exc.__class__.__name__, "message": str(exc)})
                if shutdown_event.is_set():
                    break
        finally:
            conn.close()
            if shutdown_event.is_set():
                try:
                    listener.close()
                except Exception:
                    pass

    try:
        while not shutdown_event.is_set():
            try:
                conn = listener.accept()
            except (AuthenticationError, EOFError):
                continue
            except OSError:
                if shutdown_event.is_set():
                    break
                raise
            worker = threading.Thread(target=handle_connection, args=(conn,), daemon=True)
            workers.add(worker)
            worker.start()
            workers = {thread for thread in workers if thread.is_alive()}
    finally:
        listener.close()
        daemon.close()
        for worker in list(workers):
            worker.join(timeout=5.0)
        if isinstance(address, str):
            try:
                Path(address).unlink()
            except FileNotFoundError:
                pass


class CheckpointStorageDaemonClient:
    """Client implementing RACER chunk-storage methods over CSD IPC."""

    fd_threshold_bytes = 1

    def __init__(
        self,
        address: Any,
        *,
        authkey: bytes | str | None = None,
        cuda_register_fd_mappings: bool = False,
    ) -> None:
        self.address = address
        self.authkey = _authkey(authkey)
        self.cuda_register_fd_mappings = bool(cuda_register_fd_mappings)
        self._fd_mapping_cache: dict[str, dict[str, Any]] = {}
        self._pending_cuda_views: dict[str, CudaIpcView] = {}
        self._conn = None
        self._conn_pid = os.getpid()
        self._conn_lock = threading.RLock()

    def _close_cached_conn_locked(self) -> None:
        conn = self._conn
        self._conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def _cached_conn_locked(self):
        pid = os.getpid()
        if self._conn is not None and self._conn_pid == pid:
            return self._conn
        self._close_cached_conn_locked()
        self._conn = Client(self.address, authkey=self.authkey)
        self._conn_pid = pid
        return self._conn

    def _request(self, payload: dict[str, Any]) -> Any:
        with self._conn_lock:
            last_error: BaseException | None = None
            for attempt in range(2):
                conn = self._cached_conn_locked()
                try:
                    conn.send(payload)
                    response = conn.recv()
                    break
                except (EOFError, OSError, BrokenPipeError) as exc:
                    last_error = exc
                    self._close_cached_conn_locked()
                    if attempt == 0:
                        continue
                    raise RuntimeError(f"CSD request failed after reconnect: {exc}") from exc
            else:
                raise RuntimeError(f"CSD request failed: {last_error}")
        if not response.get("ok", False):
            message = response.get("message", "unknown CSD error")
            error_type = response.get("error_type", "RuntimeError")
            if error_type == "KeyError":
                raise KeyError(message)
            raise RuntimeError(message)
        return response.get("result")

    def close(self) -> None:
        with self._conn_lock:
            self._close_cached_conn_locked()

    def _request_with_fd(self, payload: dict[str, Any]) -> tuple[Any, int]:
        if not isinstance(self.address, str):
            raise RuntimeError("CSD fd transport requires a Unix-domain socket address")
        conn = Client(self.address, authkey=self.authkey)
        try:
            request = dict(payload)
            request["pid"] = os.getpid()
            conn.send(request)
            response = conn.recv()
            if not response.get("ok", False):
                message = response.get("message", "unknown CSD error")
                error_type = response.get("error_type", "RuntimeError")
                if error_type == "KeyError":
                    raise KeyError(message)
                raise RuntimeError(message)
            fd = recv_handle(conn)
            return response.get("result"), int(fd)
        finally:
            conn.close()

    def _map_fd_region(
        self,
        fd: int,
        nbytes: int,
        metadata: dict[str, Any] | None,
        *,
        register_for_cuda: bool,
    ) -> tuple[mmap.mmap, int | None, bool]:
        meta = dict(metadata or {})
        slot_id = meta.get("fd_slot_id")
        capacity = int(meta.get("capacity_nbytes") or nbytes)
        if slot_id is not None:
            cached = self._fd_mapping_cache.get(str(slot_id))
            if cached is not None and int(cached["capacity_nbytes"]) >= int(nbytes):
                os.close(int(fd))
                if (
                    register_for_cuda
                    and self.cuda_register_fd_mappings
                    and cached.get("registered_ptr") is None
                ):
                    cached["registered_ptr"] = _cuda_host_register_buffer(
                        cached["mapping"],
                        int(cached["capacity_nbytes"]),
                    )
                return cached["mapping"], cached.get("registered_ptr"), True
        mapping = mmap.mmap(int(fd), capacity, access=mmap.ACCESS_WRITE)
        os.close(int(fd))
        registered_ptr = (
            _cuda_host_register_buffer(mapping, capacity)
            if register_for_cuda and self.cuda_register_fd_mappings
            else None
        )
        if slot_id is not None:
            self._fd_mapping_cache[str(slot_id)] = {
                "mapping": mapping,
                "registered_ptr": registered_ptr,
                "capacity_nbytes": capacity,
            }
            return mapping, registered_ptr, True
        return mapping, registered_ptr, False

    def _copy_tensor_to_fd(
        self,
        fd: int,
        nbytes: int,
        source: torch.Tensor,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        mapping, registered_ptr, cached = self._map_fd_region(
            fd,
            int(nbytes),
            metadata,
            register_for_cuda=source.device.type == "cuda",
        )
        try:
            if source.device.type == "cuda":
                host_ptr = _buffer_pointer(mapping)
                if host_ptr is None:
                    raise RuntimeError("could not resolve native CSD fd mapping pointer")
                _cuda_memcpy(host_ptr, int(source.data_ptr()), int(nbytes), _CUDA_MEMCPY_DEVICE_TO_HOST)
            else:
                target = torch.frombuffer(mapping, dtype=torch.uint8, count=int(nbytes))
                try:
                    target.copy_(source, non_blocking=False)
                finally:
                    del target
        finally:
            if not cached:
                _cuda_host_unregister_pointer(registered_ptr)
                mapping.close()

    def _tensor_from_fd(
        self,
        fd: int,
        nbytes: int,
        target: torch.device,
        metadata: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        if int(nbytes) <= 0:
            os.close(int(fd))
            return torch.empty(0, dtype=torch.uint8, device=target)
        mapping, registered_ptr, cached = self._map_fd_region(
            fd,
            int(nbytes),
            metadata,
            register_for_cuda=target.type == "cuda",
        )
        try:
            if target.type == "cpu":
                source = torch.frombuffer(mapping, dtype=torch.uint8, count=int(nbytes))
                try:
                    result = source.clone()
                finally:
                    del source
                return result
            if registered_ptr is None:
                registered_ptr = (
                    _cuda_host_register_buffer(mapping, int(nbytes))
                    if self.cuda_register_fd_mappings
                    else None
                )
                slot_id = dict(metadata or {}).get("fd_slot_id")
                if cached and registered_ptr is not None and slot_id is not None:
                    self._fd_mapping_cache[str(slot_id)]["registered_ptr"] = registered_ptr
            host_ptr = _buffer_pointer(mapping)
            if host_ptr is None:
                raise RuntimeError("could not resolve native CSD fd mapping pointer")
            out = torch.empty(int(nbytes), dtype=torch.uint8, device=target)
            _cuda_memcpy(int(out.data_ptr()), host_ptr, int(nbytes), _CUDA_MEMCPY_HOST_TO_DEVICE)
            return out
        finally:
            if not cached:
                _cuda_host_unregister_pointer(registered_ptr)
                mapping.close()

    def begin(
        self,
        tag: str,
        manifest_base: dict[str, Any] | None = None,
        expected_chunks: int | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "op": "begin",
            "tag": str(tag),
            "manifest_base": dict(manifest_base or {}),
        }
        if expected_chunks is not None:
            payload["expected_chunks"] = int(expected_chunks)
        return self._request(payload)

    def put_chunk(self, tag: str, chunk_id: str, tensor_or_handle: torch.Tensor, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        raise RuntimeError("CheckpointStorageDaemonClient.put_chunk is disabled; use put_cuda_tensor")

    def put(self, tag: str, chunk_id: str, tensor: torch.Tensor, metadata: dict[str, Any] | None = None) -> None:
        raise RuntimeError("CheckpointStorageDaemonClient.put is disabled; use put_cuda_tensor")

    def put_cuda_tensor(
        self,
        tag: str,
        chunk_id: str,
        tensor: torch.Tensor,
        metadata: dict[str, Any] | None = None,
        stream: Any | None = None,
    ) -> str:
        total_start = time.perf_counter()
        contiguous_start = time.perf_counter()
        flat = tensor.detach().contiguous().view(-1)
        contiguous_ms = (time.perf_counter() - contiguous_start) * 1000.0
        export_start = time.perf_counter()
        view = export_cuda_ipc_view(flat, stream=stream)
        export_ms = (time.perf_counter() - export_start) * 1000.0
        try:
            request_start = time.perf_counter()
            result = self._request(
                {
                    "op": "put_cuda_ipc",
                    "tag": str(tag),
                    "chunk_id": str(chunk_id),
                    "metadata": dict(metadata or {}),
                    "view": view.as_request(),
                }
            )
            request_ms = (time.perf_counter() - request_start) * 1000.0
        except BaseException:
            view.release()
            raise
        op_id = str(result["op_id"])
        profile = dict(view.profile or {})
        for key, value in dict(result.get("profile") or {}).items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                profile[key] = value
        profile.update(
            {
                "client_put_contiguous_ms": contiguous_ms,
                "client_put_export_ms": export_ms,
                "client_put_rpc_ms": request_ms,
                "client_put_total_ms": (time.perf_counter() - total_start) * 1000.0,
            }
        )
        view.profile = profile
        self._pending_cuda_views[op_id] = view
        return op_id

    def put_manifest(self, tag: str, manifest: dict[str, Any]) -> None:
        self._request({"op": "put_manifest", "tag": str(tag), "manifest": dict(manifest)})

    def commit(self, tag: str) -> None:
        self._request({"op": "commit", "tag": str(tag)})

    def get_manifest(self, tag: str) -> dict[str, Any]:
        return dict(self._request({"op": "get_manifest", "tag": str(tag)}))

    def get_chunk(self, tag: str, chunk_id: str, device: torch.device | str = "cpu") -> torch.Tensor:
        raise RuntimeError("CheckpointStorageDaemonClient.get_chunk is disabled; use read_into_cuda_tensor")

    def read_into_cuda_tensor(
        self,
        tag: str,
        chunk_id: str,
        dst_tensor: torch.Tensor,
        stream: Any | None = None,
    ) -> str:
        if dst_tensor.device.type != "cuda":
            raise ValueError("read_into_cuda_tensor requires a CUDA destination tensor")
        if dst_tensor.dtype is not torch.uint8:
            raise TypeError("read_into_cuda_tensor only supports torch.uint8 destination tensors")
        if not dst_tensor.is_contiguous():
            raise ValueError("read_into_cuda_tensor requires a contiguous destination tensor")
        view = export_cuda_ipc_view(dst_tensor.detach().view(-1), stream=stream, staging_role="target")
        try:
            result = self._request(
                {
                    "op": "read_to_cuda_ipc",
                    "tag": str(tag),
                    "chunk_id": str(chunk_id),
                    "view": view.as_request(),
                }
            )
        except BaseException:
            view.release()
            raise
        op_id = str(result["op_id"])
        self._pending_cuda_views[op_id] = view
        return op_id

    def wait(self, op_id: str) -> dict[str, Any]:
        op = str(op_id)
        view = self._pending_cuda_views.pop(op, None)
        try:
            result = dict(self._request({"op": "wait", "op_id": op}))
            if view is not None:
                profile = dict(result.get("profile") or {})
                for key, value in dict(view.profile or {}).items():
                    if isinstance(value, (int, float, str, bool)) or value is None:
                        profile[key] = value
                result["profile"] = profile
            if result.get("state") != "FAILED" and view is not None:
                view.materialize_after_read()
        finally:
            if view is not None:
                view.release()
        if result.get("state") == "FAILED":
            raise RuntimeError(str(result.get("error", "CSD async operation failed")))
        return result

    def poll(self, op_id: str) -> dict[str, Any]:
        return dict(self._request({"op": "poll", "op_id": str(op_id)}))

    def capabilities(self) -> dict[str, Any]:
        return dict(self._request({"op": "capabilities"}))

    def get(self, tag: str, chunk_id: str, device: torch.device | str = "cpu") -> torch.Tensor:
        raise RuntimeError("CheckpointStorageDaemonClient.get is disabled; use read_into_cuda_tensor")

    def get_metadata(self, tag: str, chunk_id: str) -> dict[str, Any]:
        return dict(self._request({"op": "get_metadata", "tag": str(tag), "chunk_id": str(chunk_id)}))

    def list_chunks(self, tag: str) -> list[str]:
        return list(self._request({"op": "list_chunks", "tag": str(tag)}))

    def list_tags(self) -> list[str]:
        return list(self._request({"op": "list_tags"}))

    def delete(self, tag: str) -> None:
        self._request({"op": "delete", "tag": str(tag)})

    def shutdown(self) -> None:
        self._request({"op": "shutdown"})


@dataclass
class StartedCheckpointStorageDaemon:
    process: mp.Process
    client: CheckpointStorageDaemonClient
    address: Any
    authkey: bytes

    def shutdown(self) -> None:
        try:
            self.client.shutdown()
        finally:
            self.process.join(timeout=5.0)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=5.0)


def start_checkpoint_storage_daemon(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    socket_path: str | Path | None = None,
    authkey: bytes | str | None = None,
    backend: str = "native_pinned",
    backend_options: dict[str, Any] | None = None,
    metadata_dir: str | Path | None = None,
    ready_timeout_s: float = 30.0,
) -> StartedCheckpointStorageDaemon:
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    key = _authkey(authkey)
    proc = ctx.Process(
        target=_serve,
        kwargs={
            "address": str(socket_path) if socket_path is not None else (host, int(port)),
            "authkey": key,
            "backend_name": backend,
            "backend_options": dict(backend_options or {}),
            "metadata_dir": None if metadata_dir is None else str(metadata_dir),
            "ready_conn": child_conn,
        },
        daemon=False,
    )
    proc.start()
    child_conn.close()
    if not parent_conn.poll(float(ready_timeout_s)):
        proc.join(timeout=0.1)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)
        raise RuntimeError(f"CSD daemon did not become ready within {ready_timeout_s:.1f}s")
    try:
        address = parent_conn.recv()
    except EOFError as exc:
        proc.join(timeout=1.0)
        raise RuntimeError(f"CSD daemon exited before reporting ready; exitcode={proc.exitcode}") from exc
    parent_conn.close()
    return StartedCheckpointStorageDaemon(
        process=proc,
        client=CheckpointStorageDaemonClient(address, authkey=key),
        address=address,
        authkey=key,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a RACER checkpoint storage daemon")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7007)
    parser.add_argument("--socket-path", default=None)
    parser.add_argument("--authkey", default="racer-csd")
    parser.add_argument(
        "--backend",
        choices=["native_pinned", "egm"],
        default="native_pinned",
    )
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--native-pinned-total-bytes", type=int, default=0)
    parser.add_argument("--native-pinned-segment-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--native-pinned-device", type=int, default=0)
    args = parser.parse_args()
    backend_options: dict[str, Any] = {}
    if args.backend == "native_pinned":
        backend_options = {
            "total_bytes": int(args.native_pinned_total_bytes),
            "segment_bytes": int(args.native_pinned_segment_bytes),
            "device": int(args.native_pinned_device),
        }
    _serve(
        str(args.socket_path) if args.socket_path else (args.host, int(args.port)),
        _authkey(args.authkey),
        args.backend,
        backend_options,
        args.metadata_dir,
    )


if __name__ == "__main__":
    main()
