"""Shared benchmark helpers for CUDA-only RACER examples."""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from racer import cauchy, codec_cuda
from racer.layout import ElasticLayout
from racer.routing import make_planner


CSV_COLUMNS = [
    "timestamp",
    "W_train",
    "num_spares",
    "k",
    "m",
    "q",
    "virtual_slots",
    "size_bytes",
    "encode_ms",
    "p2p_ms",
    "xor_ms",
    "store_wall_ms",
    "decode_ms",
    "load_wall_ms",
    "bytes_sent",
    "num_messages",
    "compute_bytes_on_train_ranks",
    "compute_bytes_on_accelerators",
    "skipped_virtual_zero_bytes",
    "correct",
]


INVALID_CONFIG_MESSAGE = (
    "Invalid RACER configuration: k + m must equal len(train_ranks). "
    "spare_ranks are compute-only and are not part of the erasure-code matrix."
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def result_path(prefix: str) -> Path:
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rank = os.environ.get("RANK")
    suffix = f"_rank{rank}" if rank is not None else ""
    return out_dir / f"racer_bench_{prefix}_{stamp}{suffix}.csv"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_COLUMNS})


def parse_rank_list(value: str | None) -> list[int]:
    if value is None or value == "":
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def parse_size(value: str) -> int:
    text = value.strip().upper()
    scale = 1
    if text.endswith("K"):
        scale = 1024
        text = text[:-1]
    elif text.endswith("M"):
        scale = 1024**2
        text = text[:-1]
    elif text.endswith("G"):
        scale = 1024**3
        text = text[:-1]
    return int(float(text) * scale)


def parse_sizes(value: str) -> list[int]:
    return [parse_size(part) for part in value.split(",") if part.strip()]


def validate_config(k: int, m: int, train_ranks: list[int]) -> None:
    if k + m != len(train_ranks):
        raise SystemExit(INVALID_CONFIG_MESSAGE)


def maybe_init_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank


def barrier_if_distributed() -> None:
    if "RANK" not in os.environ:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


def destroy_distributed() -> None:
    if "RANK" not in os.environ:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def sync_cuda(device: torch.device | str | None = None) -> None:
    if device is None:
        torch.cuda.synchronize()
    else:
        torch.cuda.synchronize(device)


def time_cuda(fn, device: torch.device) -> tuple[float, object]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end)), result


def make_cuda_chunks(k: int, size_bytes: int, device: torch.device) -> list[torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(0)
    return [torch.randint(0, 256, (size_bytes,), dtype=torch.uint8, device=device) for _ in range(k)]


def benchmark_codec(
    *,
    k: int,
    m: int,
    size_bytes: int,
    train_ranks: list[int] | None = None,
    spare_ranks: list[int] | None = None,
) -> dict:
    train_ranks = train_ranks if train_ranks is not None else list(range(k + m))
    spare_ranks = spare_ranks if spare_ranks is not None else [k + m]
    validate_config(k, m, train_ranks)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    layout = ElasticLayout.build(train_ranks, spare_ranks, k, m)
    E = cauchy.generate_systematic_matrix(k, m)
    failed_rows = list(range(m))
    survivor_rows = [row for row in range(k + m) if row not in failed_rows]

    device = torch.device("cuda:0")
    chunks = make_cuda_chunks(k, size_bytes, device)
    encode_ms, code = time_cuda(lambda: codec_cuda.apply_matrix_cuda(chunks, E), device)
    decode_ms, decoded = time_cuda(
        lambda: codec_cuda.decode_blocks([code[row] for row in survivor_rows], survivor_rows, E),
        device,
    )
    correct = all(torch.equal(decoded[i], chunks[i]) for i in range(k))

    return make_result_row(
        k=k,
        m=m,
        train_ranks=train_ranks,
        spare_ranks=spare_ranks,
        layout=layout,
        size_bytes=size_bytes,
        encode_ms=encode_ms,
        p2p_ms=0.0,
        xor_ms=0.0,
        store_wall_ms=encode_ms,
        decode_ms=decode_ms,
        load_wall_ms=decode_ms,
        correct=correct,
    )


def make_result_row(
    *,
    k: int,
    m: int,
    train_ranks: list[int],
    spare_ranks: list[int],
    layout: ElasticLayout,
    size_bytes: int,
    encode_ms: float,
    p2p_ms: float,
    xor_ms: float,
    store_wall_ms: float,
    decode_ms: float,
    load_wall_ms: float,
    correct: bool,
    cost=None,
) -> dict:
    return {
        "timestamp": utc_timestamp(),
        "W_train": len(train_ranks),
        "num_spares": len(spare_ranks),
        "k": k,
        "m": m,
        "q": layout.q,
        "virtual_slots": layout.num_virtual_zero,
        "size_bytes": size_bytes,
        "encode_ms": f"{encode_ms:.3f}",
        "p2p_ms": f"{p2p_ms:.3f}",
        "xor_ms": f"{xor_ms:.3f}",
        "store_wall_ms": f"{store_wall_ms:.3f}",
        "decode_ms": f"{decode_ms:.3f}",
        "load_wall_ms": f"{load_wall_ms:.3f}",
        "bytes_sent": 0 if cost is None else cost.total_bytes_sent,
        "num_messages": 0 if cost is None else cost.num_messages,
        "compute_bytes_on_train_ranks": 0 if cost is None else cost.compute_bytes_on_train_ranks,
        "compute_bytes_on_accelerators": 0 if cost is None else cost.compute_bytes_on_accelerators,
        "skipped_virtual_zero_bytes": 0 if cost is None else cost.skipped_virtual_zero_bytes,
        "correct": bool(correct),
    }


def routing_cost(config, layout: ElasticLayout, E: list[list[int]], size_bytes: int):
    return make_planner(config).plan(layout, E, size_bytes).cost
