"""RACER checkpoint data containers.

Storage ownership lives in the checkpoint storage daemon. This module only keeps
the in-process value objects used by store/load and repair code.
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
