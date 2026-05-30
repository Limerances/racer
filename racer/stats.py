"""Stats and handles for RACER single-process API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StoreStats:
    metadata_ms: float
    flatten_ms: float
    gpu_encode_ms: float
    storage_write_ms: float
    storage_read_ms: float
    gpu_decode_ms: float
    unflatten_ms: float
    total_ms: float
    bytes_total: int
    effective_gbps: float
    k: int
    m: int
    n: int
    tag: str


@dataclass(frozen=True)
class LoadStats:
    metadata_ms: float
    flatten_ms: float
    gpu_encode_ms: float
    storage_write_ms: float
    storage_read_ms: float
    gpu_decode_ms: float
    unflatten_ms: float
    total_ms: float
    bytes_total: int
    effective_gbps: float
    k: int
    m: int
    n: int
    tag: str


@dataclass(frozen=True)
class StoreHandle:
    stats: StoreStats

    def wait(self) -> StoreStats:
        return self.stats


@dataclass(frozen=True)
class LoadResult:
    obj: Any
    stats: LoadStats
