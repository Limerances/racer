"""Storage backends for RACER code chunks and manifests.

Storage records distinguish formal train-rank chunk ownership from spare-rank
temporary compute/cache placement. Spare ranks can cache data, but they do not
own erasure-code chunks in phase 1.

TODO: implement EGM-backed storage for NVL72 / GB200 restart recovery.
TODO: add persistent recovery validation that can rebuild state after process
restart using file or EGM manifests.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
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


class _DictStorage(StorageBackend):
    def __init__(self) -> None:
        self._chunks: dict[str, dict[str, ChunkRecord]] = {}
        self._manifests: dict[str, dict[str, Any]] = {}

    def _store_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().contiguous().clone()

    def _load_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clone()

    def put(self, tag: str, chunk_id: str, tensor: torch.Tensor, metadata: dict[str, Any] | None = None) -> None:
        if tensor.dtype != torch.uint8:
            raise TypeError("RACER storage only supports torch.uint8 chunks")
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


class InProcessCudaStorage(_DictStorage):
    """Current-process tensor storage, preserving CUDA chunk placement."""

    backend_name = "in_process_cuda"


class CpuPinnedStorage(_DictStorage):
    """Pinned host-memory storage that models ECCHECK-style CPU checkpoint placement."""

    backend_name = "cpu_pinned"

    def _store_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        host = tensor.detach().contiguous()
        if host.device.type == "cpu":
            if host.is_pinned():
                return host
            try:
                return host.pin_memory()
            except RuntimeError:
                return host.clone()
        try:
            host_cpu = torch.empty(int(host.numel()), dtype=torch.uint8, pin_memory=True)
        except RuntimeError:
            host_cpu = torch.empty(int(host.numel()), dtype=torch.uint8)
        host_cpu.copy_(host.view(-1), non_blocking=host_cpu.is_pinned())
        torch.cuda.synchronize(host.device)
        return host_cpu.view(host.shape)

    def _load_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.clone()


class FileMMapStorage(StorageBackend):
    """File-backed uint8 chunk storage for restart/persistence tests."""

    backend_name = "file_mmap"

    def __init__(self, root_dir: str | Path | None = None) -> None:
        if root_dir is None:
            base = Path("/dev/shm") if Path("/dev/shm").exists() else Path(tempfile.gettempdir())
            root_dir = base / "racer_mmap_storage"
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _tag_dir(self, tag: str) -> Path:
        return self.root_dir / str(tag)

    def _chunk_path(self, tag: str, chunk_id: str) -> Path:
        return self._tag_dir(tag) / f"{chunk_id}.bin"

    def _metadata_path(self, tag: str, chunk_id: str) -> Path:
        return self._tag_dir(tag) / f"{chunk_id}.json"

    def _manifest_path(self, tag: str) -> Path:
        return self._tag_dir(tag) / "manifest.json"

    def put(self, tag: str, chunk_id: str, tensor: torch.Tensor, metadata: dict[str, Any] | None = None) -> None:
        if tensor.dtype != torch.uint8:
            raise TypeError("RACER storage only supports torch.uint8 chunks")
        tag_dir = self._tag_dir(tag)
        tag_dir.mkdir(parents=True, exist_ok=True)
        host = tensor.detach().contiguous().cpu().view(-1)
        path = self._chunk_path(tag, chunk_id)
        path.write_bytes(host.numpy().tobytes())
        meta = dict(metadata or {})
        meta.setdefault("numel", int(host.numel()))
        meta.setdefault("shape", list(tensor.shape))
        meta.setdefault("dtype", "uint8")
        self._metadata_path(tag, chunk_id).write_text(json.dumps(meta, sort_keys=True))

    def get(self, tag: str, chunk_id: str) -> torch.Tensor:
        meta = self.get_metadata(tag, chunk_id)
        path = self._chunk_path(tag, chunk_id)
        if not path.exists():
            raise KeyError(f"unknown RACER chunk {chunk_id!r} for tag {tag!r}")
        data = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint8).clone()
        shape = tuple(int(v) for v in meta.get("shape", [int(data.numel())]))
        return data.view(shape)

    def get_metadata(self, tag: str, chunk_id: str) -> dict[str, Any]:
        path = self._metadata_path(tag, chunk_id)
        if not path.exists():
            raise KeyError(f"unknown RACER chunk metadata {chunk_id!r} for tag {tag!r}")
        return json.loads(path.read_text())

    def list_chunks(self, tag: str) -> list[str]:
        tag_dir = self._tag_dir(tag)
        if not tag_dir.exists():
            return []
        return sorted(path.stem for path in tag_dir.glob("*.bin"))

    def delete(self, tag: str) -> None:
        tag_dir = self._tag_dir(tag)
        if not tag_dir.exists():
            return
        for path in tag_dir.iterdir():
            path.unlink()
        tag_dir.rmdir()

    def put_manifest(self, tag: str, manifest: dict[str, Any]) -> None:
        tag_dir = self._tag_dir(tag)
        tag_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path(tag).write_text(json.dumps(manifest, sort_keys=True))

    def get_manifest(self, tag: str) -> dict[str, Any]:
        path = self._manifest_path(tag)
        if not path.exists():
            raise KeyError(f"unknown RACER manifest for tag {tag!r}")
        return json.loads(path.read_text())


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
    if name == "in_process_cpu":
        return CpuPinnedStorage()
    if name == "cpu_pinned":
        return CpuPinnedStorage()
    if name == "file_mmap":
        return FileMMapStorage(**kwargs)
    if name == "egm":
        return EgmStorage(**kwargs)
    raise ValueError(f"unsupported storage backend: {name}")
