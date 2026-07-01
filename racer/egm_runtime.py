"""CUDA Runtime EGM adapter for the Checkpoint Storage Daemon.

This module provides a daemon-side EGM runtime for ``racer.csd.EgmBackend``.
It stores chunks in a CUDA Host NUMA memory pool and uses CUDA IPC plus
stream-ordered copies for put/get.  The implementation intentionally does not
provide socket/CPU byte transport.
"""

from __future__ import annotations

import ctypes
import os
from typing import Any

from .csd import (
    NativePinnedMemoryBackend,
    _NativeSegment,
    _cuda_check,
    _load_cudart,
    _visible_cuda_device_count,
)


_CUDA_MEM_LOCATION_TYPE_HOST_NUMA = 3
_CUDA_MEM_ACCESS_FLAGS_PROT_READ_WRITE = 3
_CUDA_MEM_ALLOCATION_TYPE_PINNED = 1
_CUDA_MEM_HANDLE_TYPE_NONE = 0
_CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134
_CU_DEVICE_ATTRIBUTE_HOST_NUMA_MEMORY_POOLS_SUPPORTED = 142


class _CudaMemLocation(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("id", ctypes.c_int),
    ]


class _CudaMemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("location", _CudaMemLocation),
        ("flags", ctypes.c_int),
    ]


class _CudaMemPoolProps(ctypes.Structure):
    _fields_ = [
        ("allocType", ctypes.c_int),
        ("handleTypes", ctypes.c_int),
        ("location", _CudaMemLocation),
        ("win32SecurityAttributes", ctypes.c_void_p),
        ("maxSize", ctypes.c_size_t),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 54),
    ]


def _parse_csv_ints(value: str | None) -> list[int]:
    if value in (None, ""):
        return []
    out: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(str(value))


def _configure_cudart_mempool_api() -> Any:
    cudart = _load_cudart()
    if cudart is None:
        raise RuntimeError("CUDA runtime is required for EGM mempool runtime")
    required = [
        "cudaMemPoolCreate",
        "cudaMemPoolSetAccess",
        "cudaMallocFromPoolAsync",
        "cudaFreeAsync",
        "cudaMemPoolDestroy",
    ]
    missing = [name for name in required if not hasattr(cudart, name)]
    if missing:
        raise RuntimeError(f"CUDA runtime is missing EGM mempool APIs: {', '.join(missing)}")
    cudart.cudaMemPoolCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(_CudaMemPoolProps)]
    cudart.cudaMemPoolCreate.restype = ctypes.c_int
    cudart.cudaMemPoolSetAccess.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_CudaMemAccessDesc),
        ctypes.c_size_t,
    ]
    cudart.cudaMemPoolSetAccess.restype = ctypes.c_int
    cudart.cudaMallocFromPoolAsync.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    cudart.cudaMallocFromPoolAsync.restype = ctypes.c_int
    cudart.cudaFreeAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cudart.cudaFreeAsync.restype = ctypes.c_int
    cudart.cudaMemPoolDestroy.argtypes = [ctypes.c_void_p]
    cudart.cudaMemPoolDestroy.restype = ctypes.c_int
    return cudart


def _load_cuda_driver() -> Any | None:
    for name in ("libcuda.so", "libcuda.so.1"):
        try:
            cuda = ctypes.CDLL(name)
            cuda.cuInit.argtypes = [ctypes.c_uint]
            cuda.cuInit.restype = ctypes.c_int
            cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            cuda.cuDeviceGet.restype = ctypes.c_int
            cuda.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
            cuda.cuDeviceGetAttribute.restype = ctypes.c_int
            return cuda
        except OSError:
            continue
    return None


def _detect_host_numa_id(device: int) -> int:
    cuda = _load_cuda_driver()
    if cuda is None:
        raise RuntimeError("CSD_EGM_NUMA_ID is required because libcuda.so could not be loaded")
    err = int(cuda.cuInit(0))
    if err != 0:
        raise RuntimeError(f"CSD_EGM_NUMA_ID is required because cuInit failed with error code {err}")
    cu_device = ctypes.c_int()
    err = int(cuda.cuDeviceGet(ctypes.byref(cu_device), ctypes.c_int(int(device))))
    if err != 0:
        raise RuntimeError(f"CSD_EGM_NUMA_ID is required because cuDeviceGet({device}) failed with error code {err}")
    supported = ctypes.c_int()
    err = int(
        cuda.cuDeviceGetAttribute(
            ctypes.byref(supported),
            ctypes.c_int(_CU_DEVICE_ATTRIBUTE_HOST_NUMA_MEMORY_POOLS_SUPPORTED),
            cu_device,
        )
    )
    if err == 0 and int(supported.value) == 0:
        raise RuntimeError(f"CUDA device {device} does not report HOST_NUMA memory pool support")
    numa = ctypes.c_int()
    err = int(
        cuda.cuDeviceGetAttribute(
            ctypes.byref(numa),
            ctypes.c_int(_CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID),
            cu_device,
        )
    )
    if err != 0 or int(numa.value) < 0:
        raise RuntimeError(f"CSD_EGM_NUMA_ID is required because HOST_NUMA_ID query failed for device {device}")
    return int(numa.value)


class CudaMempoolEgmRuntime(NativePinnedMemoryBackend):
    """EGM runtime backed by CUDA Host NUMA stream-ordered memory pools."""

    name = "cuda_mempool_egm_runtime"

    def __init__(
        self,
        *,
        numa_id: int | None = None,
        home_device: int = 0,
        accessing_devices: list[int] | tuple[int, ...] | None = None,
        total_bytes: int = 0,
        segment_bytes: int = 1024 * 1024 * 1024,
        max_pool_bytes: int = 0,
    ) -> None:
        self.numa_id = None if numa_id is None else int(numa_id)
        self.home_device = int(home_device)
        self.accessing_devices = tuple(int(item) for item in (accessing_devices or ()))
        self.max_pool_bytes = int(max_pool_bytes or total_bytes or 0)
        self._egm_pool: int | None = None
        self._egm_alloc_stream: int | None = None
        self._egm_destroyed = False
        self._create_egm_pool()
        super().__init__(total_bytes=0, segment_bytes=int(segment_bytes), device=int(home_device))
        self.total_bytes = int(total_bytes)
        self.segment_bytes = int(segment_bytes)
        if self.total_bytes > 0:
            self.create_pool(self.total_bytes, self.segment_bytes)

    def _create_egm_pool(self) -> None:
        cudart = _configure_cudart_mempool_api()
        props = _CudaMemPoolProps()
        props.allocType = _CUDA_MEM_ALLOCATION_TYPE_PINNED
        props.handleTypes = _CUDA_MEM_HANDLE_TYPE_NONE
        if self.numa_id is None:
            self.numa_id = _detect_host_numa_id(int(self.home_device))
        props.location = _CudaMemLocation(_CUDA_MEM_LOCATION_TYPE_HOST_NUMA, int(self.numa_id))
        props.win32SecurityAttributes = None
        props.maxSize = ctypes.c_size_t(max(0, int(self.max_pool_bytes))).value
        props.usage = 0
        pool = ctypes.c_void_p()
        _cuda_check(cudart.cudaMemPoolCreate(ctypes.byref(pool), ctypes.byref(props)), "cudaMemPoolCreate failed")
        self._egm_pool = int(pool.value)

    def _set_access_if_needed(self) -> None:
        if not self.accessing_devices:
            return
        cudart = _configure_cudart_mempool_api()
        descs = (_CudaMemAccessDesc * len(self.accessing_devices))()
        for index, device in enumerate(self.accessing_devices):
            descs[index].location = _CudaMemLocation(1, int(device))
            descs[index].flags = _CUDA_MEM_ACCESS_FLAGS_PROT_READ_WRITE
        _cuda_check(
            cudart.cudaMemPoolSetAccess(
                ctypes.c_void_p(int(self._egm_pool or 0)),
                descs,
                ctypes.c_size_t(len(self.accessing_devices)),
            ),
            "cudaMemPoolSetAccess failed for EGM pool",
        )

    def _new_segment(self, nbytes: int) -> _NativeSegment:
        cudart = _configure_cudart_mempool_api()
        self._set_access_if_needed()
        stream = self._get_copy_stream(int(self.home_device))
        ptr = ctypes.c_void_p()
        _cuda_check(
            cudart.cudaMallocFromPoolAsync(
                ctypes.byref(ptr),
                ctypes.c_size_t(int(nbytes)),
                ctypes.c_void_p(int(self._egm_pool or 0)),
                ctypes.c_void_p(int(stream)),
            ),
            "cudaMallocFromPoolAsync failed for EGM segment",
        )
        _cuda_check(cudart.cudaStreamSynchronize(ctypes.c_void_p(int(stream))), "EGM segment allocation sync failed")
        segment = _NativeSegment(segment_id=f"egm_seg_{len(self._segments):08d}", ptr=int(ptr.value), nbytes=int(nbytes))
        self._segments.append(segment)
        self._segment_by_id[segment.segment_id] = segment
        return segment

    def capabilities(self) -> dict[str, Any]:
        stats = self.pool_stats()
        return {
            "egm_runtime": "cuda_mempool_host_numa",
            "supports_egm_native_transport": True,
            "supports_cuda_ipc": True,
            "supports_async_copy": True,
            "uses_cudaHostAlloc": False,
            "uses_cuda_mempool": True,
            "cuda_mempool_location": "host_numa",
            "numa_id": self.numa_id,
            "home_device": int(self.home_device),
            "accessing_devices": list(self.accessing_devices),
            "pool_total_bytes": stats["pool_total_bytes"],
            "pool_free_bytes": stats["pool_free_bytes"],
            "pool_segment_count": stats["pool_segment_count"],
            "segment_bytes": int(self.segment_bytes),
        }

    def _clean_metadata(self, tag: str, chunk_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(metadata)
        cleaned.pop("cuda_native_pinned", None)
        cleaned.pop("uses_cudaHostAlloc", None)
        cleaned["storage_transport"] = "egm_cuda_mempool"
        cleaned["stored_device"] = f"host_numa:{self.numa_id}"
        cleaned["daemon_storage_device"] = cleaned["stored_device"]
        cleaned["daemon_owned"] = True
        cleaned["egm_native"] = True
        cleaned["uses_cuda_mempool"] = True
        cleaned["egm_runtime"] = "cuda_mempool_host_numa"
        cleaned["egm_home_device"] = int(self.home_device)
        cleaned["egm_numa_id"] = self.numa_id
        cleaned["egm_accessing_devices"] = list(self.accessing_devices)
        location = dict(cleaned.get("location") or {})
        location.update(
            {
                "backend": "egm",
                "runtime": "cuda_mempool_host_numa",
                "allocation_id": f"{tag}:{chunk_id}",
                "nbytes": int(cleaned.get("nbytes", 0) or 0),
                "numa_id": self.numa_id,
                "home_device": int(self.home_device),
            }
        )
        cleaned["location"] = location
        return cleaned

    def write_from_cuda_ipc(
        self,
        tag: str,
        chunk_id: str,
        view: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        op_id, runtime_metadata = super().write_from_cuda_ipc(tag, chunk_id, view, metadata)
        cleaned = self._clean_metadata(str(tag), str(chunk_id), runtime_metadata)
        with self._lock:
            self._chunks[str(tag)][str(chunk_id)].metadata = dict(cleaned)
        return op_id, cleaned

    def read_to_cuda_ipc(self, tag: str, chunk_id: str, view: dict[str, Any]) -> str:
        return super().read_to_cuda_ipc(tag, chunk_id, view)

    def metadata(self, tag: str, chunk_id: str) -> dict[str, Any]:
        return self._clean_metadata(str(tag), str(chunk_id), super().metadata(tag, chunk_id))

    def checksum(self, tag: str, chunk_id: str, checksum_type: str = "sha256") -> str:
        return super().checksum(tag, chunk_id, checksum_type=checksum_type)

    def __del__(self) -> None:
        if getattr(self, "_egm_destroyed", False):
            return
        self._egm_destroyed = True
        cudart = _load_cudart()
        if cudart is None:
            return
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
        stream = None
        try:
            stream = self._get_copy_stream(int(getattr(self, "home_device", 0)))
        except Exception:
            stream = None
        if stream is not None and hasattr(cudart, "cudaFreeAsync"):
            for segment in getattr(self, "_segments", []):
                try:
                    cudart.cudaFreeAsync(ctypes.c_void_p(int(segment.ptr)), ctypes.c_void_p(int(stream)))
                except Exception:
                    pass
            try:
                cudart.cudaStreamSynchronize(ctypes.c_void_p(int(stream)))
            except Exception:
                pass
        for cached in getattr(self, "_ipc_mem_cache", {}).values():
            try:
                cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(int(cached["ptr"])))
            except Exception:
                pass
        for streams in getattr(self, "_copy_streams", {}).values():
            for copy_stream in streams:
                try:
                    cudart.cudaStreamDestroy(ctypes.c_void_p(int(copy_stream)))
                except Exception:
                    pass
        pool = getattr(self, "_egm_pool", None)
        if pool is not None and hasattr(cudart, "cudaMemPoolDestroy"):
            try:
                cudart.cudaMemPoolDestroy(ctypes.c_void_p(int(pool)))
            except Exception:
                pass


def create_runtime(
    *,
    numa_id: int | None = None,
    home_device: int | None = None,
    accessing_devices: list[int] | tuple[int, ...] | str | None = None,
    total_bytes: int | None = None,
    segment_bytes: int | None = None,
    max_pool_bytes: int | None = None,
) -> CudaMempoolEgmRuntime:
    """Create the default RACER EGM runtime from explicit args or env vars."""

    if numa_id is None:
        numa_id = _env_int("CSD_EGM_NUMA_ID", None)
    if home_device is None:
        home_device = _env_int("CSD_EGM_HOME_DEVICE", 0)
    if accessing_devices is None:
        env_devices = os.environ.get("CSD_EGM_ACCESSING_DEVICES")
        accessing = _parse_csv_ints(env_devices)
    elif isinstance(accessing_devices, str):
        accessing = _parse_csv_ints(accessing_devices)
    else:
        accessing = [int(item) for item in accessing_devices]
    if not accessing:
        count = _visible_cuda_device_count()
        accessing = list(range(count)) if count > 0 else [int(home_device or 0)]
    if total_bytes is None:
        total_bytes = _env_int("CSD_EGM_TOTAL_BYTES", _env_int("CSD_EGM_POOL_BYTES", 0))
    if segment_bytes is None:
        segment_bytes = _env_int("CSD_EGM_SEGMENT_BYTES", 1024 * 1024 * 1024)
    if max_pool_bytes is None:
        max_pool_bytes = _env_int("CSD_EGM_MAX_POOL_BYTES", int(total_bytes or 0))
    return CudaMempoolEgmRuntime(
        numa_id=numa_id,
        home_device=int(home_device or 0),
        accessing_devices=accessing,
        total_bytes=int(total_bytes or 0),
        segment_bytes=int(segment_bytes or 1024 * 1024 * 1024),
        max_pool_bytes=int(max_pool_bytes or 0),
    )
