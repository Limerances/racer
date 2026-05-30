#!/usr/bin/env python3
"""Jerasure C CPU baseline for GPT2/Megatron-style RACER checkpoints.

This is not RACER's spare-GPU runtime and is not a fallback. It is an explicit
comparison route: CUDA source tensors are flattened/offloaded to host memory,
Jerasure performs GF region multiply on CPU buffers, and parity is produced by
XOR reduction of those contributions.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from bench_utils import parse_bool, parse_rank_list
from racer import cauchy, gf256, jerasure, routing
from racer.config import RacerConfig
from racer.gpt2_synthetic import (
    available_profiles,
    estimate_rank_nbytes,
    estimate_total_nbytes,
    make_rank_states,
    profile_config,
    states_nbytes,
    tensor_count,
)
from racer.layout import RacerLayout
from racer.state_dict_codec import flatten_state_dict, unflatten_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=available_profiles(), required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--include-optimizer", type=parse_bool, default=True)
    parser.add_argument("--include-master-weights", type=parse_bool, default=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--train-ranks", default="0,1,2,3")
    parser.add_argument("--spare-ranks", default="4")
    parser.add_argument("--failed-rank", type=int, default=0)
    parser.add_argument("--load-ranks", choices=["failed", "all"], default="all")
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--fill", action="store_true")
    parser.add_argument("--max-tensors", type=int, default=None, help="debug/smoke-test cap on tensors per rank")
    return parser.parse_args()


def gbps(nbytes: int, ms: float) -> float:
    if ms <= 0.0:
        return 0.0
    return (float(nbytes) / 1e9) / (ms / 1000.0)


def mean(rows: list[dict], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows) if rows else 0.0


def sync_devices(ranks: list[int]) -> None:
    if not torch.cuda.is_available():
        return
    for rank in ranks:
        torch.cuda.synchronize(rank)


def pad_block(payload: torch.Tensor, size: int) -> torch.Tensor:
    flat = payload.contiguous().view(-1)
    if int(flat.numel()) == size:
        return flat
    out = torch.zeros(size, dtype=torch.uint8)
    out[: flat.numel()].copy_(flat)
    return out


def build_code_rows(
    layout: RacerLayout,
    payloads: dict[int, torch.Tensor],
    parity_matrix: list[list[int]],
) -> tuple[list[list[torch.Tensor]], dict[str, float | int]]:
    code_by_stripe: list[list[torch.Tensor]] = []
    local_mul_ms = 0.0
    reduction_xor_ms = 0.0
    data_save_ms = 0.0
    parity_save_ms = 0.0
    modeled_p2p_bytes = 0
    modeled_p2p_messages = 0

    for stripe in layout.stripes:
        real_ranks = [rank for rank in stripe.data_ranks if rank is not None]
        stripe_bytes = max((int(payloads[rank].numel()) for rank in real_ranks), default=1)
        data_rows: list[torch.Tensor] = []
        for rank in stripe.data_ranks:
            start = time.perf_counter()
            if rank is None:
                row = torch.zeros(stripe_bytes, dtype=torch.uint8)
            else:
                row = pad_block(payloads[rank], stripe_bytes).clone()
            data_save_ms += (time.perf_counter() - start) * 1000.0
            data_rows.append(row)

        parity_rows: list[torch.Tensor] = []
        for parity_id in range(len(parity_matrix)):
            parity = torch.zeros(stripe_bytes, dtype=torch.uint8)
            for col, rank in enumerate(stripe.data_ranks):
                if rank is None:
                    continue
                coeff = int(parity_matrix[parity_id][col]) & 0xFF
                block = data_rows[col]
                start = time.perf_counter()
                contribution = jerasure.region_multiply(block, coeff)
                local_mul_ms += (time.perf_counter() - start) * 1000.0
                modeled_p2p_bytes += int(contribution.numel())
                modeled_p2p_messages += 1

                start = time.perf_counter()
                parity.bitwise_xor_(contribution)
                reduction_xor_ms += (time.perf_counter() - start) * 1000.0
                del contribution
            start = time.perf_counter()
            parity_rows.append(parity.clone())
            parity_save_ms += (time.perf_counter() - start) * 1000.0
        code_by_stripe.append(data_rows + parity_rows)

    return code_by_stripe, {
        "data_save_ms": data_save_ms,
        "local_jerasure_mul_ms": local_mul_ms,
        "modeled_reduction_xor_ms": reduction_xor_ms,
        "parity_save_ms": parity_save_ms,
        "modeled_p2p_bytes": modeled_p2p_bytes,
        "modeled_p2p_messages": modeled_p2p_messages,
    }


def decode_needed(
    *,
    layout: RacerLayout,
    code_by_stripe: list[list[torch.Tensor]],
    E: list[list[int]],
    config: RacerConfig,
    rank_metadata: dict[int, object],
    failed_rank: int,
    requested: list[int],
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[str, float | int]]:
    failed_rows = {layout.locate_rank(failed_rank)[1]}
    by_stripe: dict[int, list[tuple[int, int]]] = {}
    for rank in requested:
        stripe_idx, col = layout.locate_rank(rank)
        by_stripe.setdefault(stripe_idx, []).append((rank, col))

    storage_read_ms = 0.0
    decode_matrix_ms = 0.0
    local_jerasure_decode_ms = 0.0
    reduction_xor_ms = 0.0
    raw_payload_to_device_ms = 0.0
    unflatten_ms = 0.0
    results: dict[int, dict[str, torch.Tensor]] = {}

    for stripe_idx, rank_cols in by_stripe.items():
        rows = code_by_stripe[stripe_idx]
        decoded_cache: list[torch.Tensor] | None = None
        for rank, col in rank_cols:
            if rank != failed_rank:
                start = time.perf_counter()
                payload = rows[col].clone()
                storage_read_ms += (time.perf_counter() - start) * 1000.0
            else:
                if decoded_cache is None:
                    survivor_rows = [row for row in range(len(E)) if row not in failed_rows][: config.k]
                    start = time.perf_counter()
                    selected = gf256.select_rows(E, survivor_rows)
                    inverse = gf256.invert_matrix(selected)
                    decode_matrix_ms += (time.perf_counter() - start) * 1000.0
                    survivor_blocks = [rows[row] for row in survivor_rows]
                    decoded_cache = []
                    for out_col in range(config.k):
                        out = torch.zeros_like(survivor_blocks[0])
                        for src_idx, survivor in enumerate(survivor_blocks):
                            coeff = int(inverse[out_col][src_idx]) & 0xFF
                            start = time.perf_counter()
                            contribution = jerasure.region_multiply(survivor, coeff)
                            local_jerasure_decode_ms += (time.perf_counter() - start) * 1000.0
                            start = time.perf_counter()
                            out.bitwise_xor_(contribution)
                            reduction_xor_ms += (time.perf_counter() - start) * 1000.0
                            del contribution
                        decoded_cache.append(out)
                payload = decoded_cache[col]

            failed = rank == failed_rank
            target = routing.output_device_for_rank(config, rank, failed)
            start = time.perf_counter()
            start_unflatten = time.perf_counter()
            result = unflatten_state_dict(rank_metadata[rank], payload, target_device=target)
            unflatten_ms += (time.perf_counter() - start_unflatten) * 1000.0
            if target.type == "cuda":
                torch.cuda.synchronize(target)
            raw_payload_to_device_ms += (time.perf_counter() - start) * 1000.0
            results[rank] = result

    return results, {
        "storage_read_ms": storage_read_ms,
        "decode_matrix_ms": decode_matrix_ms,
        "local_jerasure_decode_ms": local_jerasure_decode_ms,
        "modeled_reduction_xor_ms": reduction_xor_ms,
        "raw_payload_to_device_ms": raw_payload_to_device_ms,
        "unflatten_ms": unflatten_ms,
        "bytes_total": sum(
            int(sum(t.numel() * t.element_size() for t in state.values())) for state in results.values()
        ),
    }


def run_once(args: argparse.Namespace, states: dict[int, dict[str, torch.Tensor]], config_obj, train_ranks, spare_ranks) -> dict:
    layout = RacerLayout.build(train_ranks, args.k)
    E = cauchy.generate_systematic_matrix(args.k, args.m)
    C = cauchy.generate_cauchy_matrix(args.k, args.m)
    runtime_config = RacerConfig(
        k=args.k,
        m=args.m,
        train_ranks=tuple(train_ranks),
        spare_ranks=tuple(spare_ranks),
        backend="cuda",
        storage_backend="in_process_cuda",
    )

    total_store_start = time.perf_counter()
    flatten_start = time.perf_counter()
    flat_by_rank = {
        rank: flatten_state_dict(rank, states[rank], target_device="cpu")
        for rank in train_ranks
    }
    flatten_ms = (time.perf_counter() - flatten_start) * 1000.0
    payloads = {rank: flat.payload for rank, flat in flat_by_rank.items()}
    rank_metadata = {rank: flat.metadata for rank, flat in flat_by_rank.items()}

    code_by_stripe, store_profile = build_code_rows(layout, payloads, C)
    store_total_ms = (time.perf_counter() - total_store_start) * 1000.0

    requested = [args.failed_rank] if args.load_ranks == "failed" else list(train_ranks)
    total_load_start = time.perf_counter()
    _, load_profile = decode_needed(
        layout=layout,
        code_by_stripe=code_by_stripe,
        E=E,
        config=runtime_config,
        rank_metadata=rank_metadata,
        failed_rank=args.failed_rank,
        requested=requested,
    )
    load_total_ms = (time.perf_counter() - total_load_start) * 1000.0

    bytes_total = sum(int(payload.numel()) for payload in payloads.values())
    return {
        "store_flatten_offload_ms": flatten_ms,
        **store_profile,
        "store_total_ms": store_total_ms,
        "store_effective_gbps": gbps(bytes_total, store_total_ms),
        **{f"load_{key}": value for key, value in load_profile.items()},
        "load_total_ms": load_total_ms,
        "load_effective_gbps": gbps(int(load_profile["bytes_total"]), load_total_ms),
    }


def main() -> None:
    args = parse_args()
    if not jerasure.available():
        raise SystemExit("Jerasure C library is unavailable; cannot run this baseline")
    train_ranks = parse_rank_list(args.train_ranks)
    spare_ranks = parse_rank_list(args.spare_ranks)
    if args.k + args.m != len(train_ranks):
        raise SystemExit("k + m must equal len(train_ranks)")
    if args.failed_rank not in train_ranks:
        raise SystemExit("--failed-rank must be in --train-ranks")
    needed = max(train_ranks + spare_ranks)
    if not torch.cuda.is_available() or torch.cuda.device_count() <= needed:
        raise SystemExit("not enough CUDA devices for Jerasure baseline source tensors")

    config_obj = profile_config(
        args.profile,
        tensor_parallel=args.tp,
        dtype=args.dtype,
        include_optimizer=args.include_optimizer,
        include_master_weights=args.include_master_weights,
    )
    print(f"profile={args.profile} tp={args.tp} dtype={args.dtype} route=jerasure_cpu_baseline", flush=True)
    print(f"train_ranks={train_ranks} spare_ranks={spare_ranks} k={args.k} m={args.m}", flush=True)
    print(f"tensors_per_rank={tensor_count(config_obj)}", flush=True)
    rank_bytes = estimate_rank_nbytes(config_obj)
    total_bytes = estimate_total_nbytes(config_obj, train_ranks=len(train_ranks))
    print(f"estimated_bytes_per_rank={rank_bytes} ({rank_bytes / 1024**3:.3f} GiB)", flush=True)
    print(f"estimated_total_train_bytes={total_bytes} ({total_bytes / 1024**3:.3f} GiB)", flush=True)
    if args.estimate_only:
        return

    states = make_rank_states(train_ranks, config_obj, backend="cuda", fill=args.fill, max_tensors=args.max_tensors)
    sync_devices(train_ranks + spare_ranks)
    actual_bytes = states_nbytes(states)
    print(f"actual_materialized_train_bytes={actual_bytes} ({actual_bytes / 1024**3:.3f} GiB)", flush=True)

    rows: list[dict] = []
    total_runs = args.warmup + args.iters
    for run in range(total_runs):
        measured = run >= args.warmup
        sync_devices(train_ranks + spare_ranks)
        row = run_once(args, states, config_obj, train_ranks, spare_ranks)
        sync_devices(train_ranks + spare_ranks)
        phase = "measure" if measured else "warmup"
        print(
            f"{phase} iter={run - args.warmup}: "
            f"store_total_ms={row['store_total_ms']:.3f} "
            f"load_total_ms={row['load_total_ms']:.3f} "
            f"store_GBps={row['store_effective_gbps']:.3f} "
            f"load_GBps={row['load_effective_gbps']:.3f}",
            flush=True,
        )
        if measured:
            rows.append(row)

    if rows:
        print("summary_mean:", flush=True)
        keys = [
            "store_flatten_offload_ms",
            "data_save_ms",
            "local_jerasure_mul_ms",
            "modeled_reduction_xor_ms",
            "parity_save_ms",
            "store_total_ms",
            "load_storage_read_ms",
            "load_decode_matrix_ms",
            "load_local_jerasure_decode_ms",
            "load_modeled_reduction_xor_ms",
            "load_unflatten_ms",
            "load_raw_payload_to_device_ms",
            "load_total_ms",
            "store_effective_gbps",
            "load_effective_gbps",
        ]
        for key in keys:
            print(f"  {key}={mean(rows, key):.3f}", flush=True)
        print(f"  modeled_p2p_bytes={int(mean(rows, 'modeled_p2p_bytes'))}", flush=True)
        print(f"  modeled_p2p_messages={int(mean(rows, 'modeled_p2p_messages'))}", flush=True)


if __name__ == "__main__":
    main()
