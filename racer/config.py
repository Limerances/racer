"""Validated RACER configuration for the spare-GPU CUDA route.

The central invariant is `k + m == len(train_ranks)`. Spare ranks are
compute-only CUDA resources and are never counted in the erasure-code matrix.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RacerConfig:
    k: int
    m: int
    train_ranks: tuple[int, ...]
    spare_ranks: tuple[int, ...]
    buffer_size: int = 64 * 1024 * 1024
    optimize_cauchy: bool = False

    @property
    def w(self) -> int:
        return 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "train_ranks", tuple(int(r) for r in self.train_ranks))
        object.__setattr__(self, "spare_ranks", tuple(int(r) for r in self.spare_ranks))

        if self.k + self.m != len(self.train_ranks):
            raise ValueError(
                "RACER invariant failed: k + m must equal len(train_ranks); "
                "spare_ranks are not part of the erasure-code E matrix"
            )
        if self.k <= 0 or self.m <= 0:
            raise ValueError("k and m must be positive")
        if len(set(self.train_ranks)) != len(self.train_ranks):
            raise ValueError("train_ranks must be unique")
        if len(set(self.spare_ranks)) != len(self.spare_ranks):
            raise ValueError("spare_ranks must be unique")
        overlap = set(self.train_ranks) & set(self.spare_ranks)
        if overlap:
            raise ValueError(f"spare_ranks must not overlap train_ranks: {sorted(overlap)}")
        if not self.spare_ranks:
            raise ValueError("spare_ranks must contain at least one spare GPU")
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
