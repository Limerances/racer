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
    """EGM runtime backed by topology-aware CUDA Host NUMA memory pools."""

    name = "cuda_mempool_egm_runtime"
    dynamic_allocation_source = "dynamic_cuda_mempool"

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
        self.home_device = int(home_device)
        requested_devices = tuple(dict.fromkeys(int(item) for item in (accessing_devices or ())))
        if not requested_devices:
            count = _visible_cuda_device_count()
            requested_devices = tuple(range(count)) if count > 0 else (int(self.home_device),)
        self.accessing_devices = requested_devices
        self._home_numa_override = None if numa_id is None else int(numa_id)
        self._device_numa_ids: dict[int, int] = {}
        for device in self.accessing_devices:
            if self._home_numa_override is not None and int(device) == int(self.home_device):
                detected_numa = int(self._home_numa_override)
            else:
                detected_numa = _detect_host_numa_id(int(device))
            self._device_numa_ids[int(device)] = int(detected_numa)
        self.numa_id = (
            int(self._home_numa_override)
            if self._home_numa_override is not None
            else int(
                self._device_numa_ids.get(
                    int(self.home_device),
                    self._device_numa_ids[int(self.accessing_devices[0])],
                )
            )
        )
        self.max_pool_bytes = int(max_pool_bytes or total_bytes or 0)
        self._device_max_pool_bytes = self._split_bytes_evenly(
            self.max_pool_bytes,
            self.accessing_devices,
        )
        self._egm_pools: dict[int, int] = {}
        self._egm_destroyed = False
        self._create_egm_pools()
        primary_device = (
            int(self.home_device)
            if int(self.home_device) in self._egm_pools
            else int(self.accessing_devices[0])
        )
        self._egm_pool: int | None = int(self._egm_pools[primary_device])
        super().__init__(total_bytes=0, segment_bytes=int(segment_bytes), device=int(primary_device))
        self._next_segment_id = 0
        self.total_bytes = int(total_bytes)
        self.segment_bytes = int(segment_bytes)
        self._device_preallocated_bytes = {int(device): 0 for device in self.accessing_devices}
        if self.total_bytes > 0:
            self.create_pool(self.total_bytes, self.segment_bytes)

    @staticmethod
    def _split_bytes_evenly(total_bytes: int, devices: tuple[int, ...]) -> dict[int, int]:
        total = max(0, int(total_bytes))
        if not devices:
            return {}
        base, remainder = divmod(total, len(devices))
        return {
            int(device): int(base + (1 if index < remainder else 0))
            for index, device in enumerate(devices)
        }

    def _pool_label(self, device: int) -> str:
        device = int(device)
        return f"egm_device_{device}_numa_{self._device_numa_ids[device]}"

    def _create_egm_pools(self) -> None:
        cudart = _configure_cudart_mempool_api()
        for device in self.accessing_devices:
            device = int(device)
            props = _CudaMemPoolProps()
            props.allocType = _CUDA_MEM_ALLOCATION_TYPE_PINNED
            props.handleTypes = _CUDA_MEM_HANDLE_TYPE_NONE
            props.location = _CudaMemLocation(
                _CUDA_MEM_LOCATION_TYPE_HOST_NUMA,
                int(self._device_numa_ids[device]),
            )
            props.win32SecurityAttributes = None
            props.maxSize = ctypes.c_size_t(
                max(0, int(self._device_max_pool_bytes.get(device, 0)))
            ).value
            props.usage = 0
            pool = ctypes.c_void_p()
            _cuda_check(
                cudart.cudaMemPoolCreate(ctypes.byref(pool), ctypes.byref(props)),
                f"cudaMemPoolCreate failed for EGM device {device}",
            )
            self._egm_pools[device] = int(pool.value)
            self._set_pool_access(int(pool.value))

    def _set_pool_access(self, pool: int) -> None:
        if not self.accessing_devices:
            return
        cudart = _configure_cudart_mempool_api()
        descs = (_CudaMemAccessDesc * len(self.accessing_devices))()
        for index, device in enumerate(self.accessing_devices):
            descs[index].location = _CudaMemLocation(1, int(device))
            descs[index].flags = _CUDA_MEM_ACCESS_FLAGS_PROT_READ_WRITE
        _cuda_check(
            cudart.cudaMemPoolSetAccess(
                ctypes.c_void_p(int(pool)),
                descs,
                ctypes.c_size_t(len(self.accessing_devices)),
            ),
            "cudaMemPoolSetAccess failed for topology-aware EGM pool",
        )

    def _resolve_allocation_device(self, allocation_device: int | None) -> int:
        if allocation_device is None:
            candidate = (
                int(self.home_device)
                if int(self.home_device) in self._egm_pools
                else int(self.accessing_devices[0])
            )
        else:
            candidate = int(allocation_device)
        if candidate not in self._egm_pools:
            raise RuntimeError(
                f"CUDA device {candidate} is not configured for topology-aware EGM; "
                f"configured devices={list(self.accessing_devices)}"
            )
        return candidate

    def create_pool(self, total_bytes: int, segment_bytes: int) -> None:
        segment_size = max(1, int(segment_bytes))
        shares = self._split_bytes_evenly(int(total_bytes), self.accessing_devices)
        for device in self.accessing_devices:
            device = int(device)
            remaining = int(shares.get(device, 0))
            while remaining > 0:
                allocation_bytes = min(segment_size, remaining)
                self._new_segment(allocation_bytes, allocation_device=device)
                self._device_preallocated_bytes[device] = (
                    int(self._device_preallocated_bytes.get(device, 0)) + int(allocation_bytes)
                )
                remaining -= allocation_bytes

    def _segment_matches_allocation(
        self,
        segment: _NativeSegment,
        allocation_device: int | None,
    ) -> bool:
        requested_device = self._resolve_allocation_device(allocation_device)
        return segment.allocation_device is not None and int(segment.allocation_device) == requested_device

    def _allocate_location(
        self,
        nbytes: int,
        *,
        alignment: int = 256,
        allocation_device: int | None = None,
    ):
        return super()._allocate_location(
            int(nbytes),
            alignment=int(alignment),
            allocation_device=self._resolve_allocation_device(allocation_device),
        )

    def _new_segment(
        self,
        nbytes: int,
        *,
        allocation_device: int | None = None,
    ) -> _NativeSegment:
        device = self._resolve_allocation_device(allocation_device)
        cudart = _configure_cudart_mempool_api()
        stream = self._get_copy_stream(device)
        ptr = ctypes.c_void_p()
        _cuda_check(
            cudart.cudaMallocFromPoolAsync(
                ctypes.byref(ptr),
                ctypes.c_size_t(int(nbytes)),
                ctypes.c_void_p(int(self._egm_pools[device])),
                ctypes.c_void_p(int(stream)),
            ),
            f"cudaMallocFromPoolAsync failed for EGM device {device}",
        )
        _cuda_check(
            cudart.cudaStreamSynchronize(ctypes.c_void_p(int(stream))),
            f"EGM segment allocation sync failed for device {device}",
        )
        numa_id = int(self._device_numa_ids[device])
        segment = _NativeSegment(
            segment_id=f"egm_dev{device}_numa{numa_id}_seg_{self._next_segment_id:08d}",
            ptr=int(ptr.value),
            nbytes=int(nbytes),
            allocation_device=device,
            numa_id=numa_id,
        )
        self._next_segment_id += 1
        self._segments.append(segment)
        self._segment_by_id[segment.segment_id] = segment
        return segment

    def _device_pool_stats(self) -> dict[int, dict[str, int | str]]:
        with self._lock:
            free_list_by_segment: dict[str, int] = {}
            for block in self._free_blocks:
                free_list_by_segment[block.segment_id] = (
                    int(free_list_by_segment.get(block.segment_id, 0)) + int(block.nbytes)
                )
            result: dict[int, dict[str, int | str]] = {}
            for device in self.accessing_devices:
                device = int(device)
                segments = [
                    segment
                    for segment in self._segments
                    if segment.allocation_device is not None
                    and int(segment.allocation_device) == device
                ]
                result[device] = {
                    "pool_id": self._pool_label(device),
                    "numa_id": int(self._device_numa_ids[device]),
                    "max_pool_bytes": int(self._device_max_pool_bytes.get(device, 0)),
                    "preallocated_bytes": int(self._device_preallocated_bytes.get(device, 0)),
                    "pool_total_bytes": sum(int(segment.nbytes) for segment in segments),
                    "pool_free_bytes": sum(
                        max(0, int(segment.nbytes) - int(segment.offset))
                        + int(free_list_by_segment.get(segment.segment_id, 0))
                        for segment in segments
                    ),
                    "pool_segment_count": len(segments),
                }
            return result

    def trim_free_segments(self) -> dict[str, int]:
        """Return fully unused EGM segments to CUDA and trim pool backing pages."""

        cudart = _configure_cudart_mempool_api()
        with self._lock:
            closed_ipc_handles = 0
            for cached in list(self._ipc_mem_cache.values()):
                ptr = int(cached.get("ptr", 0))
                if ptr:
                    _cuda_check(
                        cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr)),
                        "cudaIpcCloseMemHandle failed during EGM trim",
                    )
                    closed_ipc_handles += 1
            self._ipc_mem_cache.clear()
            free_bytes_by_segment: dict[str, int] = {}
            for block in self._free_blocks:
                free_bytes_by_segment[str(block.segment_id)] = (
                    int(free_bytes_by_segment.get(str(block.segment_id), 0)) + int(block.nbytes)
                )
            releasable = [
                segment
                for segment in self._segments
                if int(free_bytes_by_segment.get(str(segment.segment_id), 0))
                + max(0, int(segment.nbytes) - int(segment.offset))
                >= int(segment.nbytes)
            ]
            if not releasable:
                return {"released_segments": 0, "released_bytes": 0, "closed_ipc_handles": closed_ipc_handles}
            by_device: dict[int, list[_NativeSegment]] = {}
            for segment in releasable:
                device = self._resolve_allocation_device(segment.allocation_device)
                by_device.setdefault(int(device), []).append(segment)
            for device, segments in by_device.items():
                stream = self._get_copy_stream(int(device))
                for segment in segments:
                    _cuda_check(
                        cudart.cudaFreeAsync(
                            ctypes.c_void_p(int(segment.ptr)),
                            ctypes.c_void_p(int(stream)),
                        ),
                        f"cudaFreeAsync EGM trim failed for device {device}",
                    )
                _cuda_check(
                    cudart.cudaStreamSynchronize(ctypes.c_void_p(int(stream))),
                    f"EGM trim sync failed for device {device}",
                )
                if hasattr(cudart, "cudaMemPoolTrimTo"):
                    _cuda_check(
                        cudart.cudaMemPoolTrimTo(
                            ctypes.c_void_p(int(self._egm_pools[int(device)])),
                            ctypes.c_size_t(0),
                        ),
                        f"cudaMemPoolTrimTo failed for EGM device {device}",
                    )
            released_ids = {str(segment.segment_id) for segment in releasable}
            self._segments = [
                segment for segment in self._segments if str(segment.segment_id) not in released_ids
            ]
            for segment_id in released_ids:
                self._segment_by_id.pop(segment_id, None)
            self._free_blocks = [
                block for block in self._free_blocks if str(block.segment_id) not in released_ids
            ]
            reset_pools = 0
            if not self._segments:
                for device, pool in list(self._egm_pools.items()):
                    _cuda_check(
                        cudart.cudaMemPoolDestroy(ctypes.c_void_p(int(pool))),
                        f"cudaMemPoolDestroy failed during EGM trim for device {device}",
                    )
                self._egm_pools = {}
                self._create_egm_pools()
                primary_device = (
                    int(self.home_device)
                    if int(self.home_device) in self._egm_pools
                    else int(self.accessing_devices[0])
                )
                self._egm_pool = int(self._egm_pools[primary_device])
                reset_pools = len(self._egm_pools)
            released_bytes = sum(int(segment.nbytes) for segment in releasable)
            for segment in releasable:
                device = self._resolve_allocation_device(segment.allocation_device)
                self._device_preallocated_bytes[int(device)] = max(
                    0,
                    int(self._device_preallocated_bytes.get(int(device), 0)) - int(segment.nbytes),
                )
            replenish_total = (
                self.total_bytes
                if os.environ.get("RACER_EGM_REPLENISH_AFTER_TRIM", "1").lower()
                in {"1", "true", "yes", "on"}
                else 0
            )
            target_by_device = self._split_bytes_evenly(replenish_total, self.accessing_devices)
            current_by_device = {int(device): 0 for device in self.accessing_devices}
            for segment in self._segments:
                device = self._resolve_allocation_device(segment.allocation_device)
                current_by_device[int(device)] += int(segment.nbytes)
            replenished_segments = 0
            replenished_bytes = 0
            for device in self.accessing_devices:
                missing = max(
                    0,
                    int(target_by_device.get(int(device), 0)) - int(current_by_device.get(int(device), 0)),
                )
                while missing > 0:
                    allocation_bytes = min(int(self.segment_bytes), int(missing))
                    self._new_segment(allocation_bytes, allocation_device=int(device))
                    self._device_preallocated_bytes[int(device)] = (
                        int(self._device_preallocated_bytes.get(int(device), 0)) + allocation_bytes
                    )
                    replenished_segments += 1
                    replenished_bytes += allocation_bytes
                    missing -= allocation_bytes
            return {
                "released_segments": len(releasable),
                "released_bytes": int(released_bytes),
                "replenished_segments": int(replenished_segments),
                "replenished_bytes": int(replenished_bytes),
                "closed_ipc_handles": int(closed_ipc_handles),
            }
    def capabilities(self) -> dict[str, Any]:
        stats = self.pool_stats()
        per_device = self._device_pool_stats()
        per_numa: dict[int, dict[str, Any]] = {}
        for device, device_stats in per_device.items():
            numa = int(device_stats["numa_id"])
            entry = per_numa.setdefault(
                numa,
                {
                    "devices": [],
                    "max_pool_bytes": 0,
                    "preallocated_bytes": 0,
                    "pool_total_bytes": 0,
                    "pool_free_bytes": 0,
                    "pool_segment_count": 0,
                },
            )
            entry["devices"].append(int(device))
            for key in (
                "max_pool_bytes",
                "preallocated_bytes",
                "pool_total_bytes",
                "pool_free_bytes",
                "pool_segment_count",
            ):
                entry[key] = int(entry[key]) + int(device_stats[key])
        return {
            "egm_runtime": "cuda_mempool_host_numa",
            "egm_topology": "per_device_host_numa",
            "topology_aware": True,
            "supports_egm_native_transport": True,
            "supports_cuda_ipc": True,
            "supports_async_copy": True,
            "supports_zero_copy_region": False,
            "uses_cudaHostAlloc": False,
            "uses_cuda_mempool": True,
            "cuda_mempool_location": "host_numa",
            "numa_id": self.numa_id,
            "home_device": int(self.home_device),
            "accessing_devices": list(self.accessing_devices),
            "device_numa_map": {
                str(device): int(numa_id)
                for device, numa_id in sorted(self._device_numa_ids.items())
            },
            "pool_count": len(self._egm_pools),
            "numa_pool_count": len(set(self._device_numa_ids.values())),
            "device_pool_map": {
                str(device): {
                    "pool_id": self._pool_label(device),
                    "numa_id": int(self._device_numa_ids[device]),
                }
                for device in self.accessing_devices
            },
            "device_pool_stats": {
                str(device): dict(device_stats)
                for device, device_stats in per_device.items()
            },
            "numa_pool_stats": {
                str(numa): dict(numa_stats)
                for numa, numa_stats in sorted(per_numa.items())
            },
            "preallocated_total_bytes": sum(
                int(value) for value in self._device_preallocated_bytes.values()
            ),
            "max_pool_bytes": int(self.max_pool_bytes),
            "pool_total_bytes": stats["pool_total_bytes"],
            "pool_free_bytes": stats["pool_free_bytes"],
            "pool_segment_count": stats["pool_segment_count"],
            "segment_bytes": int(self.segment_bytes),
        }

    def _placement_from_metadata(
        self,
        metadata: dict[str, Any],
    ) -> tuple[int, int, str]:
        location = dict(metadata.get("location") or {})
        segment_id = metadata.get("segment_id", location.get("segment_id"))
        segment = self._segment_by_id.get(str(segment_id)) if segment_id is not None else None
        if segment is not None and segment.allocation_device is not None:
            device = int(segment.allocation_device)
            numa_id = int(
                segment.numa_id
                if segment.numa_id is not None
                else self._device_numa_ids[device]
            )
        else:
            requested_device = metadata.get(
                "allocation_device",
                metadata.get(
                    "source_device",
                    location.get(
                        "allocation_device",
                        location.get("source_device", self.home_device),
                    ),
                ),
            )
            device = self._resolve_allocation_device(int(requested_device))
            numa_id = int(self._device_numa_ids[device])
        return device, numa_id, self._pool_label(device)

    def _clean_metadata(self, tag: str, chunk_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(metadata)
        cleaned.pop("cuda_native_pinned", None)
        cleaned.pop("uses_cudaHostAlloc", None)
        device, numa_id, pool_id = self._placement_from_metadata(cleaned)
        cleaned["storage_transport"] = "egm_cuda_mempool"
        cleaned["stored_device"] = f"cuda:{device}"
        cleaned["daemon_storage_device"] = f"host_numa:{numa_id}"
        cleaned["daemon_owned"] = True
        cleaned["egm_native"] = True
        cleaned["uses_cuda_mempool"] = True
        cleaned["egm_runtime"] = "cuda_mempool_host_numa"
        cleaned["egm_topology"] = "per_device_host_numa"
        cleaned["pool_id"] = pool_id
        cleaned["egm_pool_id"] = pool_id
        cleaned["source_device"] = device
        cleaned["allocation_device"] = device
        cleaned["allocation_numa_id"] = numa_id
        cleaned["egm_home_device"] = device
        cleaned["egm_numa_id"] = numa_id
        cleaned["egm_accessing_devices"] = list(self.accessing_devices)
        location = dict(cleaned.get("location") or {})
        location.update(
            {
                "backend": "egm",
                "runtime": "cuda_mempool_host_numa",
                "topology": "per_device_host_numa",
                "pool_id": pool_id,
                "allocation_id": f"{tag}:{chunk_id}",
                "nbytes": int(cleaned.get("nbytes", 0) or 0),
                "source_device": device,
                "allocation_device": device,
                "home_device": device,
                "numa_id": numa_id,
            }
        )
        cleaned["location"] = location
        return cleaned

    def _prepare_record_metadata(
        self,
        tag: str,
        chunk_id: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return self._clean_metadata(str(tag), str(chunk_id), metadata)

    def write_from_cuda_ipc(
        self,
        tag: str,
        chunk_id: str,
        view: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        return super().write_from_cuda_ipc(tag, chunk_id, view, metadata)

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
        segments_by_device: dict[int, list[_NativeSegment]] = {}
        for segment in getattr(self, "_segments", []):
            device = (
                int(segment.allocation_device)
                if segment.allocation_device is not None
                else int(getattr(self, "home_device", 0))
            )
            segments_by_device.setdefault(device, []).append(segment)
        if hasattr(cudart, "cudaFreeAsync"):
            for device, segments in segments_by_device.items():
                try:
                    stream = self._get_copy_stream(int(device))
                except Exception:
                    continue
                for segment in segments:
                    try:
                        cudart.cudaFreeAsync(
                            ctypes.c_void_p(int(segment.ptr)),
                            ctypes.c_void_p(int(stream)),
                        )
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
        if hasattr(cudart, "cudaMemPoolDestroy"):
            for pool in getattr(self, "_egm_pools", {}).values():
                try:
                    cudart.cudaMemPoolDestroy(ctypes.c_void_p(int(pool)))
                except Exception:
                    pass
        self._egm_pools = {}


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
