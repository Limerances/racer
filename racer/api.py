"""Public RACER API.

The user-facing surface is `racer.init`, `racer.store`, and `racer.load`.
RACER accepts either raw `Dict[int, torch.uint8 Tensor]` packets or
`Dict[int, state_dict]` checkpoint payloads keyed by train rank.
"""

from __future__ import annotations

from .config import RacerConfig
from .context import RacerContext, StoreHandle

_DEFAULT_CONTEXT: RacerContext | None = None


def init(
    k: int,
    m: int,
    train_ranks: list[int],
    spare_ranks: list[int],
    buffer_size: int = 64 * 1024 * 1024,
    optimize_cauchy: bool = False,
) -> RacerContext:
    global _DEFAULT_CONTEXT
    config = RacerConfig(
        k=k,
        m=m,
        train_ranks=tuple(train_ranks),
        spare_ranks=tuple(spare_ranks),
        buffer_size=buffer_size,
        optimize_cauchy=optimize_cauchy,
    )
    _DEFAULT_CONTEXT = RacerContext(config)
    return _DEFAULT_CONTEXT


def get_context() -> RacerContext:
    if _DEFAULT_CONTEXT is None:
        raise RuntimeError("racer.init(...) must be called before using the default context")
    return _DEFAULT_CONTEXT


def store(
    obj,
    tag: str | None = None,
    context: RacerContext | None = None,
    async_op: bool = False,
) -> StoreHandle:
    ctx = context if context is not None else get_context()
    return ctx.store(obj, tag=tag, async_op=async_op)


def load(
    tag: str | None = None,
    failed_train_ranks: list[int] | None = None,
    survivor_rows: list[int] | None = None,
    requested_train_ranks: list[int] | None = None,
    context: RacerContext | None = None,
):
    ctx = context if context is not None else get_context()
    return ctx.load(
        tag=tag,
        failed_train_ranks=failed_train_ranks,
        survivor_rows=survivor_rows,
        requested_train_ranks=requested_train_ranks,
    )
