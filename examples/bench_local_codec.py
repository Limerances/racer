"""Single-process CUDA RACER codec benchmark."""

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
    parser.add_argument("--train-ranks", default=None, help="Optional comma-separated train ranks")
    parser.add_argument("--spare-ranks", default=None, help="Optional comma-separated spare ranks")
    parser.add_argument("--routing-strategy", default="spare_compute", choices=["spare_compute"])
    args = parser.parse_args()

    if args.sizes:
        sizes = parse_sizes(args.sizes)
    elif args.bytes is not None:
        sizes = [args.bytes]
    else:
        sizes = [64 * 1024 * 1024]

    train_ranks = None if args.train_ranks is None else [int(v) for v in args.train_ranks.split(",") if v]
    spare_ranks = None if args.spare_ranks is None else [int(v) for v in args.spare_ranks.split(",") if v]
    rows = []
    for size_bytes in sizes:
        row = benchmark_codec(
            k=args.k,
            m=args.m,
            size_bytes=size_bytes,
            routing_strategy=args.routing_strategy,
            train_ranks=train_ranks,
            spare_ranks=spare_ranks,
        )
        rows.append(row)
        print(row)

    path = result_path("local_codec")
    write_csv(path, rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
