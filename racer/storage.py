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
    """Current-process tensor storage, preserving CUDA chunk placement."""

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


class InProcessStorage(InProcessCudaStorage):
    """Compatibility wrapper for checkpoint-level access inside RacerContext."""

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
