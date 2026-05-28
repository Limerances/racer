"""Small in-process benchmark helpers for RACER contexts."""

from __future__ import annotations

import time

import torch

from .context import RacerContext


def bench_store(
    context: RacerContext,
    obj: dict[int, torch.Tensor],
    iters: int = 10,
    sync: bool = True,
) -> dict[str, float]:
    if iters <= 0:
        raise ValueError("iters must be positive")
    start = time.perf_counter()
    bytes_per_iter = sum(t.numel() for t in obj.values())
    for i in range(iters):
        handle = context.store(obj, tag=f"bench_{i}", async_op=not sync)
        if sync:
            handle.wait()
    if context.config.backend == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    gib = (bytes_per_iter * iters) / (1024**3)
    return {"seconds": elapsed, "input_gib": gib, "input_gib_per_s": gib / elapsed}
