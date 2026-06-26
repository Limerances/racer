"""CUDA chunk storage for RACER code rows and manifests.

Storage records distinguish formal train-rank row ownership from spare-rank
temporary compute placement. Spare ranks can compute/cache data, but they do not
own erasure-code rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class StoredReductionGroup:
    index: int
    rows: list[torch.Tensor]
    data_ranks: tuple[int | None, ...]
    reduction_group_bytes: int
    shapes: dict[int, tuple[int, ...]]
    numels: dict[int, int]


@dataclass
class StoredCheckpoint:
    tag: str
    reduction_groups: list[StoredReductionGroup]
    matrix: list[list[int]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def devices(self) -> set[torch.device]:
        out: set[torch.device] = set()
        for group in self.reduction_groups:
            for row in group.rows:
                out.add(row.device)
        return out


@dataclass
class ChunkRecord:
    tensor: torch.Tensor
    metadata: dict[str, Any]


class InProcessCudaStorage:
    """Disabled historical current-process tensor storage."""

    def __init__(self) -> None:
        raise RuntimeError(
            "InProcessCudaStorage is disabled. RACER checkpoint storage must be "
            "daemon-owned native_pinned or daemon-owned EGM."
        )
        self._chunks: dict[str, dict[str, ChunkRecord]] = {}
        self._manifests: dict[str, dict[str, Any]] = {}

    def _store_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype != torch.uint8:
            raise TypeError("RACER storage only supports torch.uint8 chunks")
        if tensor.device.type != "cuda":
            raise ValueError("RACER CUDA storage requires CUDA tensors")
        return tensor.detach().contiguous()

    def _load_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clone()

    def put(self, tag: str, chunk_id: str, tensor: torch.Tensor, metadata: dict[str, Any] | None = None) -> None:
        self._chunks.setdefault(str(tag), {})[str(chunk_id)] = ChunkRecord(
            tensor=self._store_tensor(tensor),
            metadata=dict(metadata or {}),
        )

    def get(self, tag: str, chunk_id: str) -> torch.Tensor:
        try:
            return self._load_tensor(self._chunks[str(tag)][str(chunk_id)].tensor)
        except KeyError as exc:
            raise KeyError(f"unknown RACER chunk {chunk_id!r} for tag {tag!r}") from exc

    def get_metadata(self, tag: str, chunk_id: str) -> dict[str, Any]:
        try:
            return dict(self._chunks[str(tag)][str(chunk_id)].metadata)
        except KeyError as exc:
            raise KeyError(f"unknown RACER chunk metadata {chunk_id!r} for tag {tag!r}") from exc

    def list_chunks(self, tag: str) -> list[str]:
        return sorted(self._chunks.get(str(tag), {}))

    def delete(self, tag: str) -> None:
        self._chunks.pop(str(tag), None)
        self._manifests.pop(str(tag), None)

    def put_manifest(self, tag: str, manifest: dict[str, Any]) -> None:
        self._manifests[str(tag)] = dict(manifest)

    def get_manifest(self, tag: str) -> dict[str, Any]:
        try:
            return dict(self._manifests[str(tag)])
        except KeyError as exc:
            raise KeyError(f"unknown RACER manifest for tag {tag!r}") from exc


class CpuPinnedStorage(InProcessCudaStorage):
    """Disabled historical current-process CPU pinned storage."""

    def __init__(self, *, pin_memory: bool | None = None) -> None:
        super().__init__()
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)

    def _allocate_host(self, numel: int) -> torch.Tensor:
        if self.pin_memory:
            try:
                return torch.empty(int(numel), dtype=torch.uint8, device="cpu", pin_memory=True)
            except RuntimeError:
                # Some CI/container environments expose a CUDA build without a
                # working pinned allocator. Keep the storage usable as a CPU
                # backend, but make the pinning status observable in metadata.
                self.pin_memory = False
        return torch.empty(int(numel), dtype=torch.uint8, device="cpu")

    def _store_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype != torch.uint8:
            raise TypeError("RACER storage only supports torch.uint8 chunks")
        flat = tensor.detach().contiguous().view(-1)
        host = self._allocate_host(int(flat.numel()))
        host.copy_(flat, non_blocking=flat.device.type == "cuda")
        if flat.device.type == "cuda":
            index = flat.device.index if flat.device.index is not None else torch.cuda.current_device()
            torch.cuda.synchronize(index)
        return host

    def put(self, tag: str, chunk_id: str, tensor: torch.Tensor, metadata: dict[str, Any] | None = None) -> None:
        record_metadata = dict(metadata or {})
        record_metadata["is_pinned_host"] = bool(self.pin_memory)
        record_metadata["stored_device"] = "cpu"
        self._chunks.setdefault(str(tag), {})[str(chunk_id)] = ChunkRecord(
            tensor=self._store_tensor(tensor),
            metadata=record_metadata,
        )


class EgmStorage(InProcessCudaStorage):
    """Disabled historical in-process EGM-like storage."""

    page_size = 2 * 1024 * 1024

    def __init__(
        self,
        *,
        mem_pool: Any | None = None,
        home_device: int | None = None,
        numa_id: int | None = None,
        accessing_devices: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.mem_pool = mem_pool
        self.home_device = None if home_device is None else int(home_device)
        self.numa_id = None if numa_id is None else int(numa_id)
        self.accessing_devices = tuple(int(device) for device in accessing_devices or ())

    def is_available(self) -> bool:
        return (
            self.mem_pool is not None
            and torch.cuda.is_available()
            and hasattr(torch.cuda, "use_mem_pool")
        )

    def _target_device(self) -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("RACER EGM storage requires CUDA")
        index = self.home_device if self.home_device is not None else torch.cuda.current_device()
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(
                f"RACER EGM storage home_device={index} is outside visible CUDA devices"
            )
        return torch.device("cuda", index)

    def _store_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype != torch.uint8:
            raise TypeError("RACER storage only supports torch.uint8 chunks")
        if self.mem_pool is None:
            raise RuntimeError(
                "RACER EGM storage requires storage_options={'mem_pool': torch.cuda.MemPool(...)} "
                "backed by a native EGM allocator. Expected allocator shape: "
                "cuMemCreate/CUDA mempool with pinned Host NUMA location and access enabled "
                "for the RACER train/spare GPUs."
            )
        if not hasattr(torch.cuda, "use_mem_pool"):
            raise RuntimeError("this PyTorch build does not expose torch.cuda.use_mem_pool")

        flat = tensor.detach().contiguous().view(-1)
        device = self._target_device()
        with torch.cuda.device(device):
            with torch.cuda.use_mem_pool(self.mem_pool):
                stored = torch.empty(int(flat.numel()), dtype=torch.uint8, device=device)
            stored.copy_(flat, non_blocking=True)
            torch.cuda.synchronize(device)
        return stored

    def put(self, tag: str, chunk_id: str, tensor: torch.Tensor, metadata: dict[str, Any] | None = None) -> None:
        record_metadata = dict(metadata or {})
        record_metadata.update(
            {
                "storage_backend": "egm",
                "stored_device": f"cuda:{self.home_device}" if self.home_device is not None else "cuda",
                "egm_home_device": self.home_device,
                "egm_numa_id": self.numa_id,
                "egm_accessing_devices": list(self.accessing_devices),
                "egm_page_size": self.page_size,
                "egm_mem_pool_configured": self.mem_pool is not None,
            }
        )
        self._chunks.setdefault(str(tag), {})[str(chunk_id)] = ChunkRecord(
            tensor=self._store_tensor(tensor),
            metadata=record_metadata,
        )


class InProcessStorage(InProcessCudaStorage):
    """Disabled historical checkpoint-level in-process index."""

    def __init__(self) -> None:
        super().__init__()
        self._items: dict[str, StoredCheckpoint] = {}
        self._latest_tag: str | None = None

    def put(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        if len(args) == 1 and isinstance(args[0], StoredCheckpoint):
            checkpoint = args[0]
            self._items[checkpoint.tag] = checkpoint
            self._latest_tag = checkpoint.tag
            return
        return super().put(*args, **kwargs)

    def get(self, tag: str | None = None, chunk_id: str | None = None):  # type: ignore[override]
        if chunk_id is not None:
            return super().get(str(tag), chunk_id)
        actual = tag if tag is not None else self._latest_tag
        if actual is None:
            raise KeyError("no checkpoint has been stored")
        try:
            return self._items[actual]
        except KeyError as exc:
            raise KeyError(f"unknown RACER checkpoint tag: {actual}") from exc

    def tags(self) -> list[str]:
        return list(self._items.keys())

    def delete(self, tag: str) -> None:
        actual = str(tag)
        super().delete(actual)
        self._items.pop(actual, None)
        if self._latest_tag == actual:
            self._latest_tag = next(reversed(self._items), None) if self._items else None
