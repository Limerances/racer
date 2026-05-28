"""Single-process RACER codec benchmark."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from bench_utils import benchmark_codec, parse_sizes, result_path, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default=None, help="Comma-separated sizes, e.g. 64M,256M")
    parser.add_argument("--bytes", type=int, default=None, help="Backward-compatible single size in bytes")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--cpu-baseline", action="store_true")
    parser.add_argument("--routing-strategy", default="local")
    args = parser.parse_args()

    if args.sizes:
        sizes = parse_sizes(args.sizes)
    elif args.bytes is not None:
        sizes = [args.bytes]
    else:
        sizes = [64 * 1024 * 1024]

    rows = []
    for size_bytes in sizes:
        row = benchmark_codec(
            k=args.k,
            m=args.m,
            size_bytes=size_bytes,
            backend=args.backend,
            cpu_baseline=args.cpu_baseline,
            routing_strategy=args.routing_strategy,
        )
        rows.append(row)
        print(row)

    path = result_path("local_codec")
    write_csv(path, rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
