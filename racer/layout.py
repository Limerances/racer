"""Rank and virtual-zero layout helpers for RACER.

`ElasticLayout` maps `len(train_ranks)` checkpoint packets into k data groups
and pads only with logical virtual-zero slots when W % k != 0. Spare ranks are
kept outside data groups and reduction groups.

TODO: add tray-level coding-unit layouts once rank-level coding is validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class Stripe:
    index: int
    data_ranks: tuple[int | None, ...]

    def ranks(self) -> tuple[int, ...]:
        return tuple(rank for rank in self.data_ranks if rank is not None)


@dataclass(frozen=True)
class RacerLayout:
    train_ranks: tuple[int, ...]
    k: int
    stripes: tuple[Stripe, ...]

    @classmethod
    def build(cls, train_ranks: list[int] | tuple[int, ...], k: int) -> "RacerLayout":
        if k <= 0:
            raise ValueError("k must be positive")
        ranks = tuple(int(r) for r in train_ranks)
        stripes = []
        for start in range(0, len(ranks), k):
            chunk = list(ranks[start : start + k])
            while len(chunk) < k:
                chunk.append(None)
            stripes.append(Stripe(index=len(stripes), data_ranks=tuple(chunk)))
        return cls(train_ranks=ranks, k=k, stripes=tuple(stripes))

    def locate_rank(self, rank: int) -> tuple[int, int]:
        target = int(rank)
        for stripe in self.stripes:
            for col, candidate in enumerate(stripe.data_ranks):
                if candidate == target:
                    return stripe.index, col
        raise KeyError(f"rank {rank} is not in train_ranks")

    def stripe_for_rank(self, rank: int) -> Stripe:
        stripe_idx, _ = self.locate_rank(rank)
        return self.stripes[stripe_idx]


@dataclass(frozen=True)
class ElasticSlot:
    slot_id: int
    data_group_id: int
    relative_index: int
    train_rank: int | None
    is_virtual_zero: bool
    valid_nbytes: int | None = None


@dataclass(frozen=True)
class ElasticLayout:
    train_ranks: tuple[int, ...]
    spare_ranks: tuple[int, ...]
    k: int
    m: int
    pad_mode: str
    q: int
    virtual_W: int
    num_virtual_zero: int
    data_groups: tuple[tuple[ElasticSlot, ...], ...]
    reduction_groups: tuple[tuple[ElasticSlot, ...], ...]

    @classmethod
    def build(
        cls,
        train_ranks: list[int] | tuple[int, ...],
        spare_ranks: list[int] | tuple[int, ...],
        k: int,
        m: int,
        pad_mode: str = "virtual_zero",
    ) -> "ElasticLayout":
        if pad_mode != "virtual_zero":
            raise ValueError("phase-1 ElasticLayout only supports pad_mode='virtual_zero'")
        if k <= 0 or m <= 0:
            raise ValueError("k and m must be positive")

        train = tuple(int(r) for r in train_ranks)
        spare = tuple(int(r) for r in spare_ranks)
        if len(set(train)) != len(train):
            raise ValueError("train_ranks must be unique")
        if len(set(spare)) != len(spare):
            raise ValueError("spare_ranks must be unique")
        overlap = set(train) & set(spare)
        if overlap:
            raise ValueError(f"spare_ranks must not overlap train_ranks: {sorted(overlap)}")
        if k + m != len(train):
            raise ValueError("ElasticLayout requires k + m == len(train_ranks)")

        W = len(train)
        q = ceil(W / k)
        virtual_W = q * k
        num_virtual_zero = virtual_W - W

        mutable_groups: list[list[ElasticSlot]] = [[] for _ in range(k)]
        mutable_reductions: list[list[ElasticSlot]] = [[] for _ in range(q)]
        for slot_id in range(virtual_W):
            relative_index = slot_id // k
            data_group_id = slot_id % k
            is_virtual = slot_id >= W
            slot = ElasticSlot(
                slot_id=slot_id,
                data_group_id=data_group_id,
                relative_index=relative_index,
                train_rank=None if is_virtual else train[slot_id],
                is_virtual_zero=is_virtual,
                valid_nbytes=0 if is_virtual else None,
            )
            mutable_groups[data_group_id].append(slot)
            mutable_reductions[relative_index].append(slot)

        return cls(
            train_ranks=train,
            spare_ranks=spare,
            k=k,
            m=m,
            pad_mode=pad_mode,
            q=q,
            virtual_W=virtual_W,
            num_virtual_zero=num_virtual_zero,
            data_groups=tuple(tuple(group) for group in mutable_groups),
            reduction_groups=tuple(tuple(group) for group in mutable_reductions),
        )

    def real_slots(self) -> tuple[ElasticSlot, ...]:
        return tuple(slot for group in self.data_groups for slot in group if not slot.is_virtual_zero)

    def virtual_zero_slots(self) -> tuple[ElasticSlot, ...]:
        return tuple(slot for group in self.data_groups for slot in group if slot.is_virtual_zero)

    def locate_rank(self, rank: int) -> ElasticSlot:
        target = int(rank)
        for slot in self.real_slots():
            if slot.train_rank == target:
                return slot
        raise KeyError(f"rank {rank} is not in train_ranks")
