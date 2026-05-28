#!/usr/bin/env python3
"""Benchmark RACER with GPT2/Megatron-style multi-tensor checkpoints."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import racer
from bench_utils import parse_bool, parse_rank_list, parse_size, result_path
from racer import codec_cuda
from racer.gpt2_synthetic import (
    available_profiles,
    estimate_rank_nbytes,
    estimate_total_nbytes,
    make_rank_states,
    profile_config,
    state_dict_byte_equal,
    states_nbytes,
    tensor_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=available_profiles(), default="gpt2-124m")
    parser.add_argument("--tp", type=int, default=4, help="tensor parallel size used by the synthetic checkpoint")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--include-optimizer", type=parse_bool, default=True)
    parser.add_argument("--include-master-weights", type=parse_bool, default=True)
    parser.add_argument("--backend", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--storage-backend", default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--train-ranks", default="0,1,2,3")
    parser.add_argument("--spare-ranks", default="4")
    parser.add_argument("--failed-rank", type=int, default=0)
    parser.add_argument("--load-ranks", choices=["failed", "all"], default="failed")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--tag", default="gpt2_checkpoint_bench")
    parser.add_argument("--chunk-size", default="1G", help="RACER chunk size for chunked spare-GPU encode/decode")
    parser.add_argument("--fill", action="store_true", help="fill tensors deterministically instead of leaving them uninitialized")
    parser.add_argument("--verify", action="store_true", help="byte-compare the recovered failed-rank state")
    parser.add_argument("--max-tensors", type=int, default=None, help="debug/smoke-test cap on tensors per rank")
    parser.add_argument("--csv", action="store_true", help="write one CSV row per measured iteration")
    parser.add_argument("--estimate-only", action="store_true", help="print the synthetic checkpoint size and exit")
    return parser.parse_args()


def gbps(nbytes: int, ms: float) -> float:
    if ms <= 0.0:
        return 0.0
    return (float(nbytes) / 1e9) / (ms / 1000.0)


def mean(rows: list[dict], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows) if rows else 0.0


def sync_cuda_devices(ranks: list[int]) -> None:
    if not torch.cuda.is_available():
        return
    for rank in ranks:
        torch.cuda.synchronize(rank)


def write_rows(rows: list[dict]) -> Path:
    path = result_path("gpt2_checkpoint")
    columns = list(rows[0]) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    args = parse_args()
    train_ranks = parse_rank_list(args.train_ranks)
    spare_ranks = parse_rank_list(args.spare_ranks)
    if args.k + args.m != len(train_ranks):
        raise SystemExit("invalid RACER config: k + m must equal len(train_ranks); spare ranks are extra")
    if args.failed_rank not in train_ranks:
        raise SystemExit("--failed-rank must be one of --train-ranks")
    if args.backend == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available")
        needed = max(train_ranks + spare_ranks) if spare_ranks else max(train_ranks)
        if torch.cuda.device_count() <= needed:
            raise SystemExit(f"need CUDA device rank {needed}, visible device_count={torch.cuda.device_count()}")

    storage_backend = args.storage_backend
    if storage_backend is None:
        storage_backend = "in_process_cuda" if args.backend == "cuda" else "in_process_cpu"

    config = profile_config(
        args.profile,
        tensor_parallel=args.tp,
        dtype=args.dtype,
        include_optimizer=args.include_optimizer,
        include_master_weights=args.include_master_weights,
    )
    estimated_rank_bytes = estimate_rank_nbytes(config)
    estimated_total_bytes = estimate_total_nbytes(config, train_ranks=len(train_ranks))
    per_rank_tensors = tensor_count(config)

    print(f"profile={args.profile} tp={config.tensor_parallel} dtype={config.dtype}")
    print(f"train_ranks={train_ranks} spare_ranks={spare_ranks} k={args.k} m={args.m}")
    print(f"tensors_per_rank={per_rank_tensors}")
    print(f"estimated_bytes_per_rank={estimated_rank_bytes} ({estimated_rank_bytes / 1024**3:.3f} GiB)")
    print(f"estimated_total_train_bytes={estimated_total_bytes} ({estimated_total_bytes / 1024**3:.3f} GiB)")
    chunk_size = parse_size(args.chunk_size)
    print(f"chunk_size_bytes={chunk_size}")
    print(f"cuda_extension_available={codec_cuda.extension_available() if args.backend == 'cuda' else False}")
    if args.estimate_only:
        return

    states = make_rank_states(
        train_ranks,
        config,
        backend=args.backend,
        fill=args.fill or args.verify,
        max_tensors=args.max_tensors,
    )
    actual_total_bytes = states_nbytes(states)
    print(f"actual_materialized_train_bytes={actual_total_bytes} ({actual_total_bytes / 1024**3:.3f} GiB)")

    ctx = racer.init(
        k=args.k,
        m=args.m,
        train_ranks=train_ranks,
        spare_ranks=spare_ranks,
        backend=args.backend,
        storage_backend=storage_backend,
        buffer_size=chunk_size,
        async_op=False,
        routing_strategy="spare_compute",
    )

    measured_rows: list[dict] = []
    recovered = None
    total_runs = args.warmup + args.iters
    for run in range(total_runs):
        measured = run >= args.warmup
        sync_cuda_devices(train_ranks + spare_ranks)
        handle = racer.store(states, tag=args.tag, context=ctx, async_op=False)
        handle.wait()
        sync_cuda_devices(train_ranks + spare_ranks)
        requested_train_ranks = train_ranks if args.load_ranks == "all" else None
        recovered = racer.load(
            tag=args.tag,
            failed_train_ranks=[args.failed_rank],
            requested_train_ranks=requested_train_ranks,
            context=ctx,
        )
        sync_cuda_devices(train_ranks + spare_ranks)

        correct = ""
        if args.verify:
            ranks_to_check = train_ranks if args.load_ranks == "all" else [args.failed_rank]
            correct = all(state_dict_byte_equal(states[rank], recovered[rank]) for rank in ranks_to_check)
            if not correct:
                raise RuntimeError("recovered state does not match original bytes")

        store_profile = dict(ctx.last_store_profile)
        load_profile = dict(ctx.last_load_profile)
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "iteration": run - args.warmup,
            "profile": args.profile,
            "backend": args.backend,
            "storage_backend": storage_backend,
            "k": args.k,
            "m": args.m,
            "train_ranks": ",".join(map(str, train_ranks)),
            "spare_ranks": ",".join(map(str, spare_ranks)),
            "failed_rank": args.failed_rank,
            "load_ranks": args.load_ranks,
            "tensors_per_rank": len(states[args.failed_rank]),
            "actual_total_train_bytes": actual_total_bytes,
            "store_flatten_ms": store_profile.get("flatten_ms", 0.0),
            "store_stripe_pack_ms": store_profile.get("stripe_pack_ms", 0.0),
            "store_ec_encode_ms": store_profile.get("ec_encode_ms", 0.0),
            "store_storage_device_copy_ms": store_profile.get("storage_device_copy_ms", 0.0),
            "store_data_direct_save_ms": store_profile.get("data_direct_save_ms", 0.0),
            "store_parity_chunk_save_ms": store_profile.get("parity_chunk_save_ms", 0.0),
            "store_spare_buffer_alloc_ms": store_profile.get("spare_buffer_alloc_ms", 0.0),
            "store_manifest_build_ms": store_profile.get("manifest_build_ms", 0.0),
            "store_chunk_storage_write_ms": store_profile.get("chunk_storage_write_ms", 0.0),
            "store_data_chunk_write_ms": store_profile.get("data_chunk_write_ms", 0.0),
            "store_parity_chunk_write_ms": store_profile.get("parity_chunk_write_ms", 0.0),
            "store_checkpoint_index_ms": store_profile.get("checkpoint_index_ms", 0.0),
            "store_encode_ms": store_profile.get("encode_ms", 0.0),
            "store_total_ms": store_profile.get("total_ms", 0.0),
            "store_effective_gbps": gbps(actual_total_bytes, float(store_profile.get("total_ms", 0.0))),
            "store_data_row_bytes": store_profile.get("data_row_bytes", 0),
            "store_parity_row_bytes": store_profile.get("parity_row_bytes", 0),
            "store_storage_device_copy_gbps": gbps(
                int(store_profile.get("data_row_bytes", 0)) + int(store_profile.get("parity_row_bytes", 0)),
                float(store_profile.get("storage_device_copy_ms", 0.0)),
            ),
            "store_data_direct_save_gbps": gbps(
                int(store_profile.get("data_row_bytes", 0)),
                float(store_profile.get("data_direct_save_ms", 0.0)),
            ),
            "store_parity_chunk_save_gbps": gbps(
                int(store_profile.get("parity_row_bytes", 0)),
                float(store_profile.get("parity_chunk_save_ms", 0.0)),
            ),
            "store_chunk_storage_write_gbps": gbps(
                int(store_profile.get("data_chunk_bytes", 0)) + int(store_profile.get("parity_chunk_bytes", 0)),
                float(store_profile.get("chunk_storage_write_ms", 0.0)),
            ),
            "load_storage_read_ms": load_profile.get("storage_read_ms", 0.0),
            "load_checkpoint_sync_ms": load_profile.get("checkpoint_sync_ms", 0.0),
            "load_metadata_ms": load_profile.get("metadata_ms", 0.0),
            "load_survivor_to_compute_ms": load_profile.get("survivor_to_compute_ms", 0.0),
            "load_decode_matrix_ms": load_profile.get("decode_matrix_ms", 0.0),
            "load_ec_decode_ms": load_profile.get("ec_decode_ms", 0.0),
            "load_raw_payload_to_output_device_ms": load_profile.get("raw_payload_to_output_device_ms", 0.0),
            "load_unflatten_ms": load_profile.get("unflatten_ms", 0.0),
            "load_final_sync_ms": load_profile.get("final_sync_ms", 0.0),
            "load_decode_ms": load_profile.get("decode_ms", 0.0),
            "load_total_ms": load_profile.get("total_ms", 0.0),
            "load_effective_gbps": gbps(int(load_profile.get("bytes_total", 0)), float(load_profile.get("total_ms", 0.0))),
            "load_bytes_total": load_profile.get("bytes_total", 0),
            "verified": correct,
        }
        if measured:
            measured_rows.append(row)
        phase = "measure" if measured else "warmup"
        print(
            f"{phase} iter={run - args.warmup}: "
            f"store_total_ms={float(row['store_total_ms']):.3f} "
            f"load_total_ms={float(row['load_total_ms']):.3f} "
            f"store_GBps={float(row['store_effective_gbps']):.3f} "
            f"load_GBps={float(row['load_effective_gbps']):.3f} "
            f"verified={row['verified']}"
        )
        del recovered
        recovered = None

    if measured_rows:
        print("summary_mean:")
        print(f"  store_flatten_ms={mean(measured_rows, 'store_flatten_ms'):.3f}")
        print(f"  store_stripe_pack_ms={mean(measured_rows, 'store_stripe_pack_ms'):.3f}")
        print(f"  store_ec_encode_ms={mean(measured_rows, 'store_ec_encode_ms'):.3f}")
        print(f"  store_storage_device_copy_ms={mean(measured_rows, 'store_storage_device_copy_ms'):.3f}")
        print(f"  store_data_direct_save_ms={mean(measured_rows, 'store_data_direct_save_ms'):.3f}")
        print(f"  store_parity_chunk_save_ms={mean(measured_rows, 'store_parity_chunk_save_ms'):.3f}")
        print(f"  store_spare_buffer_alloc_ms={mean(measured_rows, 'store_spare_buffer_alloc_ms'):.3f}")
        print(f"  store_storage_device_copy_gbps={mean(measured_rows, 'store_storage_device_copy_gbps'):.3f}")
        print(f"  store_data_direct_save_gbps={mean(measured_rows, 'store_data_direct_save_gbps'):.3f}")
        print(f"  store_parity_chunk_save_gbps={mean(measured_rows, 'store_parity_chunk_save_gbps'):.3f}")
        print(f"  store_manifest_build_ms={mean(measured_rows, 'store_manifest_build_ms'):.3f}")
        print(f"  store_chunk_storage_write_ms={mean(measured_rows, 'store_chunk_storage_write_ms'):.3f}")
        print(f"  store_chunk_storage_write_gbps={mean(measured_rows, 'store_chunk_storage_write_gbps'):.3f}")
        print(f"  store_total_ms={mean(measured_rows, 'store_total_ms'):.3f}")
        print(f"  load_storage_read_ms={mean(measured_rows, 'load_storage_read_ms'):.3f}")
        print(f"  load_checkpoint_sync_ms={mean(measured_rows, 'load_checkpoint_sync_ms'):.3f}")
        print(f"  load_survivor_to_compute_ms={mean(measured_rows, 'load_survivor_to_compute_ms'):.3f}")
        print(f"  load_decode_matrix_ms={mean(measured_rows, 'load_decode_matrix_ms'):.3f}")
        print(f"  load_ec_decode_ms={mean(measured_rows, 'load_ec_decode_ms'):.3f}")
        print(f"  load_raw_payload_to_output_device_ms={mean(measured_rows, 'load_raw_payload_to_output_device_ms'):.3f}")
        print(f"  load_unflatten_ms={mean(measured_rows, 'load_unflatten_ms'):.3f}")
        print(f"  load_final_sync_ms={mean(measured_rows, 'load_final_sync_ms'):.3f}")
        print(f"  load_total_ms={mean(measured_rows, 'load_total_ms'):.3f}")
        print(f"  store_effective_gbps={mean(measured_rows, 'store_effective_gbps'):.3f}")
        print(f"  load_effective_gbps={mean(measured_rows, 'load_effective_gbps'):.3f}")

    if args.csv and measured_rows:
        path = write_rows(measured_rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
