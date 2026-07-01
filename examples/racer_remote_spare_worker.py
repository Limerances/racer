#!/usr/bin/env python3
"""Run a remote RACER spare worker outside the Megatron torchrun world."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def _expand_int_list(value: str) -> list[int]:
    out: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            step = 1 if end >= start else -1
            out.extend(range(start, end + step, step))
        else:
            out.append(int(item))
    return out


def _storage_options(args: argparse.Namespace) -> dict[str, Any]:
    if args.racer_csd_socket_path:
        return {
            "address": str(args.racer_csd_socket_path),
            "authkey": args.racer_csd_authkey,
            "cuda_register_fd_mappings": False,
        }
    return {
        "address": (str(args.racer_csd_host), int(args.racer_csd_port)),
        "authkey": args.racer_csd_authkey,
        "cuda_register_fd_mappings": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--racer-root", default="/workspace/racer")
    parser.add_argument("--megatron-root", default="/workspace/Megatron-LM-FT")
    parser.add_argument("--store-host", required=True)
    parser.add_argument("--runtime-port", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--spare-rank", type=int, required=True)
    parser.add_argument("--spare-cuda-device", type=int, default=0)
    parser.add_argument("--racer-k", type=int, required=True)
    parser.add_argument("--racer-m", type=int, required=True)
    parser.add_argument("--racer-train-ranks", required=True)
    parser.add_argument("--racer-spare-ranks", required=True)
    parser.add_argument("--racer-buffer-size", type=int, default=1024 * 1024 * 1024)
    parser.add_argument("--racer-storage-backend", choices=["csd_native_pinned", "csd_egm"], required=True)
    parser.add_argument("--racer-csd-host", default="127.0.0.1")
    parser.add_argument("--racer-csd-port", type=int, default=7007)
    parser.add_argument("--racer-csd-socket-path", default=None)
    parser.add_argument("--racer-csd-authkey", default="racer-csd")
    parser.add_argument("--racer-csd-local-ranks", default=None)
    parser.add_argument("--racer-csd-local-coordinator-rank", default=None)
    parser.add_argument("--racer-optimize-cauchy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for root in [args.racer_root, args.megatron_root]:
        path = str(Path(root).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)

    local_ranks = args.racer_csd_local_ranks or str(int(args.spare_rank))
    local_coordinator = args.racer_csd_local_coordinator_rank or str(int(args.spare_rank))
    os.environ.setdefault("RACER_CSD_PER_NODE", "1")
    os.environ["RACER_CSD_LOCAL_RANKS"] = local_ranks
    os.environ["RACER_CSD_LOCAL_COORDINATOR_RANK"] = local_coordinator

    from megatron.training.racer_checkpointing import _spare_worker_main  # type: ignore

    train_ranks = _expand_int_list(args.racer_train_ranks)
    spare_ranks = _expand_int_list(args.racer_spare_ranks)
    config_data = {
        "k": int(args.racer_k),
        "m": int(args.racer_m),
        "train_ranks": train_ranks,
        "spare_ranks": spare_ranks,
        "buffer_size": int(args.racer_buffer_size),
        "optimize_cauchy": bool(args.racer_optimize_cauchy),
    }
    _spare_worker_main(
        port=int(args.runtime_port),
        store_host=str(args.store_host),
        world_size=int(args.world_size),
        spare_rank=int(args.spare_rank),
        spare_cuda_device=int(args.spare_cuda_device),
        config_data=config_data,
        racer_path=str(args.racer_root),
        storage_backend=str(args.racer_storage_backend),
        storage_options=_storage_options(args),
    )


if __name__ == "__main__":
    main()
