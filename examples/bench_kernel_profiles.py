#!/usr/bin/env python3
"""Kernel-only RACER GF benchmark using GPT profile rank payload sizes.

This benchmark intentionally avoids checkpoint flattening, P2P, storage, and
Jerasure. It allocates k CUDA byte buffers whose length equals the selected
profile's per-rank checkpoint payload, then times RACER's CUDA GF matrix kernel
for the parity rows used by store().
"""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from racer import cauchy, codec_cuda, gf256
from racer.gpt2_synthetic import available_profiles, estimate_rank_nbytes, profile_config


def parse_profiles(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def gbps(nbytes: int, ms: float) -> float:
    if ms <= 0.0:
        return 0.0
    return (float(nbytes) / 1e9) / (ms / 1000.0)


def expected_value(coeff_row: torch.Tensor, fill_values: list[int], table: torch.Tensor) -> int:
    acc = 0
    for coeff, value in zip(coeff_row.tolist(), fill_values):
        c = int(coeff) & 0xFF
        if c:
            acc ^= int(table[c][int(value) & 0xFF].item())
    return acc & 0xFF


def run_profile(args: argparse.Namespace, profile: str) -> dict:
    device = torch.device(args.device)
    config = profile_config(
        profile,
        tensor_parallel=args.tp,
        dtype=args.dtype,
        include_optimizer=args.include_optimizer,
        include_master_weights=args.include_master_weights,
    )
    rank_bytes = int(estimate_rank_nbytes(config))
    if args.max_rank_bytes is not None:
        rank_bytes = min(rank_bytes, int(args.max_rank_bytes))

    E = cauchy.generate_systematic_matrix(args.k, args.m)
    coeff_rows = E[args.k : args.k + args.m]
    coeff = gf256.coefficients_to_tensor(coeff_rows, device)
    table = gf256.torch_mul_table(device)
    fill_values = [int(args.fill_base + args.fill_step * col) & 0xFF for col in range(args.k)]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    inputs = [torch.empty(rank_bytes, dtype=torch.uint8, device=device) for _ in range(args.k)]
    outputs = [torch.empty(rank_bytes, dtype=torch.uint8, device=device) for _ in range(args.m)]
    for value, tensor in zip(fill_values, inputs):
        tensor.fill_(value)
    torch.cuda.synchronize(device)

    for _ in range(args.warmup):
        codec_cuda.apply_matrix_cuda(inputs, coeff_rows, outputs=outputs)
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(args.iters):
        codec_cuda.apply_matrix_cuda(inputs, coeff_rows, outputs=outputs)
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - start) * 1000.0 / float(args.iters)

    verified = True
    if args.verify:
        sample_positions = sorted({0, min(4096, rank_bytes - 1), rank_bytes // 2, rank_bytes - 1})
        for row, output in enumerate(outputs):
            expected = expected_value(coeff[row], fill_values, table)
            actual = [int(output[pos].item()) for pos in sample_positions]
            if any(value != expected for value in actual):
                verified = False
                raise RuntimeError(
                    f"{profile} output row {row} failed sample verification: expected={expected} actual={actual}"
                )

    input_bytes = int(args.k * rank_bytes)
    output_bytes = int(args.m * rank_bytes)
    touched_bytes = input_bytes + output_bytes
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": profile,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "k": args.k,
        "m": args.m,
        "rank_bytes": rank_bytes,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "touched_bytes": touched_bytes,
        "wall_ms": f"{wall_ms:.3f}",
        "input_GBps": f"{gbps(input_bytes, wall_ms):.3f}",
        "touched_GBps": f"{gbps(touched_bytes, wall_ms):.3f}",
        "peak_GiB": f"{torch.cuda.max_memory_allocated(device) / 1024**3:.3f}",
        "verified": verified,
        "vector_table_kernel": codec_cuda._optional_extension_function("apply_matrix_cuda_vector_table") is not None,
    }

    del inputs, outputs
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", default="paper-gpt2-1.6b,paper-gpt2-5.3b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--include-optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-master-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--fill-base", type=int, default=7)
    parser.add_argument("--fill-step", type=int, default=2)
    parser.add_argument("--max-rank-bytes", type=int, default=None)
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    if args.iters <= 0:
        raise SystemExit("--iters must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    valid = set(available_profiles())
    profiles = parse_profiles(args.profiles)
    unknown = [profile for profile in profiles if profile not in valid]
    if unknown:
        raise SystemExit(f"unknown profiles: {unknown}; choices={sorted(valid)}")
    if codec_cuda._optional_extension_function("apply_matrix_cuda_vector_table") is None:
        print("warning: rebuilt vector-table kernel is not available; benchmark will use fallback wrapper path", file=sys.stderr)

    rows = []
    for profile in profiles:
        row = run_profile(args, profile)
        rows.append(row)
        print(row, flush=True)

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
