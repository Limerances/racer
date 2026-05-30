"""CUDA storage backends for RACER code chunks and manifests.

Storage records distinguish formal train-rank chunk ownership from spare-rank
temporary compute/cache placement. Spare ranks can cache data, but they do not
own erasure-code chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class StoredStripe:
    index: int
    rows: list[torch.Tensor]
    data_ranks: tuple[int | None, ...]
    stripe_bytes: int
    shapes: dict[int, tuple[int, ...]]
    numels: dict[int, int]


@dataclass
class StoredCheckpoint:
    tag: str
    stripes: list[StoredStripe]
    matrix: list[list[int]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def devices(self) -> set[torch.device]:
        out: set[torch.device] = set()
        for stripe in self.stripes:
            for row in stripe.rows:
                out.add(row.device)
        return out


@dataclass
class ChunkRecord:
    tensor: torch.Tensor
    metadata: dict[str, Any]


class StorageBackend:
    """Interface for chunk and manifest storage backends."""

    backend_name = "base"

    def put(self, tag: str, chunk_id: str, tensor: torch.Tensor, metadata: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    def get(self, tag: str, chunk_id: str) -> torch.Tensor:
        raise NotImplementedError

    def get_metadata(self, tag: str, chunk_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def list_chunks(self, tag: str) -> list[str]:
        raise NotImplementedError

    def delete(self, tag: str) -> None:
        raise NotImplementedError

    def put_manifest(self, tag: str, manifest: dict[str, Any]) -> None:
        raise NotImplementedError

    def get_manifest(self, tag: str) -> dict[str, Any]:
        raise NotImplementedError


class InProcessCudaStorage(StorageBackend):
    """Current-process tensor storage, preserving CUDA chunk placement."""

    backend_name = "in_process_cuda"

    def __init__(self) -> None:
        self._chunks: dict[str, dict[str, ChunkRecord]] = {}
        self._manifests: dict[str, dict[str, Any]] = {}

    def _store_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dtype != torch.uint8:
            raise TypeError("RACER storage only supports torch.uint8 chunks")
        if tensor.device.type != "cuda":
            raise ValueError("RACER CUDA storage requires CUDA tensors")
        return tensor.detach().contiguous().clone()

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


class EgmStorage(StorageBackend):
    backend_name = "egm"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("TODO: connect RACER storage to NVL72 / GB200 EGM")


class InProcessStorage(InProcessCudaStorage):
    """Compatibility wrapper for older RacerContext checkpoint-level access."""

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


def create_storage_backend(name: str, **kwargs: Any) -> StorageBackend:
    if name == "in_process_cuda":
        return InProcessCudaStorage()
    if name == "egm":
        return EgmStorage(**kwargs)
    raise ValueError(f"unsupported CUDA storage backend: {name}")
