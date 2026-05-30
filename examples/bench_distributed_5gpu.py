"""5-GPU RACER benchmark with real torchrun P2P execution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import time

import torch

import racer
from racer.config import RacerConfig
from racer.distributed import distributed_load, distributed_store
from bench_utils import (
    barrier_if_distributed,
    destroy_distributed,
    make_result_row,
    maybe_init_distributed,
    parse_rank_list,
    parse_sizes,
    result_path,
    routing_cost,
    validate_config,
    write_csv,
)


def _make_obj(train_ranks: list[int], size_bytes: int) -> dict[int, torch.Tensor]:
    torch.manual_seed(0)
    obj = {}
    for rank in train_ranks:
        obj[rank] = torch.randint(0, 256, (size_bytes,), dtype=torch.uint8, device=f"cuda:{rank}")
    return obj


def _make_local_packet(rank: int, local_rank: int, train_ranks: list[int], size_bytes: int) -> torch.Tensor | None:
    if rank not in train_ranks:
        return None
    torch.manual_seed(rank)
    return torch.randint(0, 256, (size_bytes,), dtype=torch.uint8, device=f"cuda:{local_rank}")


def _sync_all() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _max_wall_ms(begin: float) -> float:
    elapsed_device = torch.device("cuda", torch.cuda.current_device())
    elapsed = torch.tensor([(time.perf_counter() - begin) * 1000.0], dtype=torch.float64, device=elapsed_device)
    if "RANK" in __import__("os").environ:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ranks", required=True)
    parser.add_argument("--spare-ranks", default="")
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--sizes", type=str, default="64M")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--test-nondivisible", action="store_true")
    parser.add_argument("--failures", type=str, default=None)
    args = parser.parse_args()

    train_ranks = parse_rank_list(args.train_ranks)
    spare_ranks = parse_rank_list(args.spare_ranks)
    validate_config(args.k, args.m, train_ranks)

    rank, world, local_rank = maybe_init_distributed()
    rows = []
    try:
        if not torch.cuda.is_available():
            raise SystemExit("bench_distributed_5gpu.py requires CUDA")
        if max(train_ranks + spare_ranks) >= torch.cuda.device_count():
            raise SystemExit(
                f"visible CUDA device_count={torch.cuda.device_count()} is too small for ranks {train_ranks + spare_ranks}"
            )

        if args.test_nondivisible and len(train_ranks) % args.k == 0:
            raise SystemExit("--test-nondivisible was requested, but len(train_ranks) % k == 0")

        if world > 1:
            if world <= max(train_ranks + spare_ranks):
                raise SystemExit(
                    f"torchrun world_size={world} is too small for ranks {train_ranks + spare_ranks}"
                )
            config = RacerConfig(
                k=args.k,
                m=args.m,
                train_ranks=tuple(train_ranks),
                spare_ranks=tuple(spare_ranks),
            )
            for size_bytes in parse_sizes(args.sizes):
                local_packet = _make_local_packet(rank, local_rank, train_ranks, size_bytes)
                _sync_all()
                barrier_if_distributed()
                begin = time.perf_counter()
                state = distributed_store(
                    config=config,
                    local_packet=local_packet,
                    tag=f"bench_{size_bytes}",
                )
                store_wall_ms = _max_wall_ms(begin)

                failures = parse_rank_list(args.failures) if args.failures else [train_ranks[0]]
                if len(failures) > args.m:
                    raise SystemExit(f"requested {len(failures)} failures, but m={args.m}")
                begin = time.perf_counter()
                load_result = distributed_load(state=state, failed_train_ranks=failures if args.verify else [])
                load_wall_ms = _max_wall_ms(begin)

                local_correct = True
                if args.verify and rank in failures:
                    assert local_packet is not None
                    local_correct = torch.equal(load_result.recovered[rank], local_packet)
                correct_device = torch.device("cuda", torch.cuda.current_device())
                correct_tensor = torch.tensor([1 if local_correct else 0], dtype=torch.int32, device=correct_device)
                import torch.distributed as dist

                dist.all_reduce(correct_tensor, op=dist.ReduceOp.MIN)
                correct = bool(int(correct_tensor.item()))

                if rank == 0:
                    row = make_result_row(
                        k=args.k,
                        m=args.m,
                        train_ranks=train_ranks,
                        spare_ranks=spare_ranks,
                        layout=state.layout,
                        size_bytes=size_bytes,
                        encode_ms=store_wall_ms,
                        p2p_ms=0.0,
                        xor_ms=0.0,
                        store_wall_ms=store_wall_ms,
                        decode_ms=load_wall_ms,
                        load_wall_ms=load_wall_ms,
                        correct=correct,
                        cost=state.routing_cost,
                    )
                    rows.append(row)
                    print(row)

            if rank == 0:
                path = result_path("distributed_5gpu")
                write_csv(path, rows)
                print(f"wrote {path}")
            barrier_if_distributed()
            return

        ctx = racer.init(
            k=args.k,
            m=args.m,
            train_ranks=train_ranks,
            spare_ranks=spare_ranks,
        )
        failures = parse_rank_list(args.failures) if args.failures else [train_ranks[0]]
        if len(failures) > args.m:
            raise SystemExit(f"requested {len(failures)} failures, but m={args.m}")

        for size_bytes in parse_sizes(args.sizes):
            obj = _make_obj(train_ranks, size_bytes)
            _sync_all()
            begin = time.perf_counter()
            handle = racer.store(obj, tag=f"bench_{size_bytes}", context=ctx)
            handle.wait()
            _sync_all()
            store_wall_ms = (time.perf_counter() - begin) * 1000.0

            begin = time.perf_counter()
            recovered = racer.load(
                tag=f"bench_{size_bytes}",
                failed_train_ranks=failures if args.verify else None,
                context=ctx,
            )
            _sync_all()
            load_wall_ms = (time.perf_counter() - begin) * 1000.0

            correct = True
            if args.verify:
                correct = all(torch.equal(recovered[rank].to(obj[rank].device), obj[rank]) for rank in failures)

            cost = routing_cost(ctx.config, ctx.elastic_layout, ctx.matrix, size_bytes)
            row = make_result_row(
                k=args.k,
                m=args.m,
                train_ranks=train_ranks,
                spare_ranks=spare_ranks,
                layout=ctx.elastic_layout,
                size_bytes=size_bytes,
                encode_ms=store_wall_ms,
                p2p_ms=0.0,
                xor_ms=0.0,
                store_wall_ms=store_wall_ms,
                decode_ms=load_wall_ms,
                load_wall_ms=load_wall_ms,
                correct=correct,
                cost=cost,
            )
            rows.append(row)
            print(row)

        path = result_path("distributed_5gpu")
        write_csv(path, rows)
        print(f"wrote {path}")
        barrier_if_distributed()
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
