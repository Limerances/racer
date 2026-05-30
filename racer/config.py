"""Validated RACER configuration for the spare-GPU CUDA route.

The central invariant is `k + m == len(train_ranks)`. Spare ranks are
compute-only resources and are never counted in the erasure-code matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RacerConfig:
    k: int
    m: int
    train_ranks: tuple[int, ...]
    spare_ranks: tuple[int, ...]
    backend: str = "cuda"
    storage_backend: str = "in_process_cuda"
    w: int = 8
    buffer_size: int = 64 * 1024 * 1024
    optimize_cauchy: bool = False
    process_group: Any = None
    async_op: bool = True
    routing_strategy: str = "spare_compute"
    include_spares_in_train: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_ranks", tuple(int(r) for r in self.train_ranks))
        object.__setattr__(self, "spare_ranks", tuple(int(r) for r in self.spare_ranks))

        if self.include_spares_in_train:
            raise NotImplementedError("include_spares_in_train=True is intentionally not implemented")

        if self.k + self.m != len(self.train_ranks):
            raise ValueError(
                "RACER invariant failed: k + m must equal len(train_ranks); "
                "spare_ranks are not part of the erasure-code E matrix"
            )
        if self.k <= 0 or self.m <= 0:
            raise ValueError("k and m must be positive")
        if self.w != 8:
            raise NotImplementedError("RACER implements GF(2^8) only")
        if len(set(self.train_ranks)) != len(self.train_ranks):
            raise ValueError("train_ranks must be unique")
        if len(set(self.spare_ranks)) != len(self.spare_ranks):
            raise ValueError("spare_ranks must be unique")
        overlap = set(self.train_ranks) & set(self.spare_ranks)
        if overlap:
            raise ValueError(f"spare_ranks must not overlap train_ranks: {sorted(overlap)}")
        if not self.spare_ranks:
            raise ValueError("spare_ranks must contain at least one spare GPU")
        if self.backend != "cuda":
            raise ValueError("backend must be 'cuda'; the CPU data path has been removed")
        if self.storage_backend not in {"in_process_cuda", "egm"}:
            raise ValueError("storage_backend must be 'in_process_cuda' or 'egm'")
        if self.routing_strategy != "spare_compute":
            raise ValueError("routing_strategy must be 'spare_compute'; CPU/local routes have been removed")
