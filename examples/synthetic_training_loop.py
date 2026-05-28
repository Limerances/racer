"""Synthetic training loop for estimating checkpoint cadence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import math
import statistics
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
    parse_bool,
    parse_rank_list,
    parse_size,
    percentile,
    result_path,
    routing_cost,
    validate_config,
    write_csv,
)


def _make_obj(train_ranks: list[int], packet_size: int) -> dict[int, torch.Tensor]:
    torch.manual_seed(0)
    return {
        rank: torch.empty(packet_size, dtype=torch.uint8, device=f"cuda:{rank}")
        for rank in train_ranks
    }


def _make_local_packet(rank: int, local_rank: int, train_ranks: list[int], packet_size: int) -> torch.Tensor | None:
    if rank not in train_ranks:
        return None
    torch.manual_seed(rank)
    return torch.empty(packet_size, dtype=torch.uint8, device=f"cuda:{local_rank}")


def _fill_local_packet(packet: torch.Tensor | None, rank: int, step: int) -> None:
    if packet is not None:
        packet.fill_((step + rank) % 256)


def _fill_obj(obj: dict[int, torch.Tensor], step: int) -> None:
    for rank, tensor in obj.items():
        tensor.fill_((step + rank) % 256)


def _sync_all() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _make_matmul_state(device: torch.device, matmul_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    a = torch.randn((matmul_size, matmul_size), device=device)
    b = torch.randn((matmul_size, matmul_size), device=device)
    return a, b


def _training_step(mode: str, iter_ms: float, matmul_state) -> None:
    if mode == "sleep":
        time.sleep(iter_ms / 1000.0)
        return
    if matmul_state is None:
        time.sleep(iter_ms / 1000.0)
        return
    a, b = matmul_state
    _ = a @ b
    _sync_all()


def _measure_baseline(mode: str, iter_ms: float, matmul_state, samples: int = 5) -> float:
    times = []
    for _ in range(samples):
        begin = time.perf_counter()
        _training_step(mode, iter_ms, matmul_state)
        times.append((time.perf_counter() - begin) * 1000.0)
    return statistics.mean(times) if times else iter_ms


def _max_distributed_ms(begin: float) -> float:
    elapsed_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    elapsed = torch.tensor([(time.perf_counter() - begin) * 1000.0], dtype=torch.float64, device=elapsed_device)
    if "RANK" in __import__("os").environ:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return float(elapsed.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ranks", default="0,1,2,3")
    parser.add_argument("--spare-ranks", default="4")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--packet-size", default="256M")
    parser.add_argument("--iter-ms", type=float, default=500.0)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--async-store", default="true")
    parser.add_argument("--routing-strategy", default="spare_compute")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--mode", choices=["sleep", "matmul"], default="sleep")
    parser.add_argument("--matmul-size", type=int, default=2048)
    args = parser.parse_args()

    train_ranks = parse_rank_list(args.train_ranks)
    spare_ranks = parse_rank_list(args.spare_ranks)
    validate_config(args.k, args.m, train_ranks)
    async_store = parse_bool(args.async_store)
    packet_size = parse_size(args.packet_size)

    rank, world, local_rank = maybe_init_distributed(backend="nccl")
    rows = []
    try:
        if not torch.cuda.is_available():
            raise SystemExit("synthetic_training_loop.py currently requires CUDA")
        if max(train_ranks + spare_ranks) >= torch.cuda.device_count():
            raise SystemExit(
                f"visible CUDA device_count={torch.cuda.device_count()} is too small for ranks {train_ranks + spare_ranks}"
            )

        if world > 1:
            config = RacerConfig(
                k=args.k,
                m=args.m,
                train_ranks=tuple(train_ranks),
                spare_ranks=tuple(spare_ranks),
                backend="cuda",
                storage_backend="in_process_cuda",
                async_op=False,
                routing_strategy=args.routing_strategy,
            )
            local_packet = _make_local_packet(rank, local_rank, train_ranks, packet_size)
            _fill_local_packet(local_packet, rank, 0)
            matmul_state = None
            if args.mode == "matmul" and rank in train_ranks:
                matmul_state = _make_matmul_state(torch.device(f"cuda:{local_rank}"), args.matmul_size)
            baseline_iter_ms = _measure_baseline(args.mode, args.iter_ms, matmul_state)

            iter_times = []
            store_wall_times = []
            last_state = None
            last_step = -1
            for step in range(args.iters):
                begin_iter = time.perf_counter()
                _training_step(args.mode, args.iter_ms, matmul_state)
                if step % args.save_interval == 0:
                    _fill_local_packet(local_packet, rank, step)
                    barrier_if_distributed()
                    begin_store = time.perf_counter()
                    last_state = distributed_store(
                        config=config,
                        local_packet=local_packet,
                        tag=f"synthetic_{step:06d}",
                        use_jerasure=True,
                    )
                    store_wall_times.append(_max_distributed_ms(begin_store))
                    last_step = step
                iter_times.append((time.perf_counter() - begin_iter) * 1000.0)

            correct = True
            load_wall_ms = 0.0
            if args.verify and last_state is not None:
                failed = [train_ranks[0]]
                begin_load = time.perf_counter()
                recovered = distributed_load(state=last_state, failed_train_ranks=failed)
                load_wall_ms = _max_distributed_ms(begin_load)
                if rank in failed:
                    assert local_packet is not None
                    correct = torch.equal(recovered.recovered[rank].cpu(), local_packet.cpu())
                import torch.distributed as dist

                correct_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
                correct_tensor = torch.tensor([1 if correct else 0], dtype=torch.int32, device=correct_device)
                dist.all_reduce(correct_tensor, op=dist.ReduceOp.MIN)
                correct = bool(int(correct_tensor.item()))

            avg_store_wall_ms = statistics.mean(store_wall_times) if store_wall_times else 0.0
            max_store_wall_ms = max(store_wall_times) if store_wall_times else 0.0
            avg_iter_ms = statistics.mean(iter_times) if iter_times else 0.0
            recommended = 1
            if async_store and baseline_iter_ms > 0 and avg_store_wall_ms > 0:
                recommended = max(1, math.ceil(avg_store_wall_ms / baseline_iter_ms))
            feasible = recommended <= 1

            if rank == 0:
                assert last_state is not None
                row = make_result_row(
                    k=args.k,
                    m=args.m,
                    train_ranks=train_ranks,
                    spare_ranks=spare_ranks,
                    layout=last_state.layout,
                    routing_strategy=args.routing_strategy,
                    size_bytes=packet_size,
                    encode_ms=avg_store_wall_ms,
                    p2p_ms=0.0,
                    xor_ms=0.0,
                    store_wall_ms=avg_store_wall_ms,
                    decode_ms=load_wall_ms,
                    load_wall_ms=load_wall_ms,
                    correct=correct,
                    cost=last_state.routing_cost,
                )
                rows.append(row)
                path = result_path("synthetic_loop")
                write_csv(path, rows)
                summary = {
                    "baseline_iter_ms": round(baseline_iter_ms, 3),
                    "avg_iter_ms_with_racer": round(avg_iter_ms, 3),
                    "p50_iter_ms": round(percentile(iter_times, 50), 3),
                    "p90_iter_ms": round(percentile(iter_times, 90), 3),
                    "p99_iter_ms": round(percentile(iter_times, 99), 3),
                    "avg_store_wall_ms": round(avg_store_wall_ms, 3),
                    "max_store_wall_ms": round(max_store_wall_ms, 3),
                    "async_queue_depth": 0,
                    "recommended_min_save_interval": recommended,
                    "whether_iter_level_checkpoint_is_feasible": feasible,
                    "correct": correct,
                    "last_step": last_step,
                    "csv": str(path),
                }
                print(summary)
            barrier_if_distributed()
            return

        ctx = racer.init(
            k=args.k,
            m=args.m,
            train_ranks=train_ranks,
            spare_ranks=spare_ranks,
            backend="cuda",
            storage_backend="in_process_cuda",
            async_op=async_store,
            routing_strategy=args.routing_strategy,
        )
        obj = _make_obj(train_ranks, packet_size)
        _fill_obj(obj, 0)

        matmul_state = None
        if args.mode == "matmul":
            matmul_state = _make_matmul_state(torch.device(f"cuda:{train_ranks[0]}"), args.matmul_size)
        baseline_iter_ms = _measure_baseline(args.mode, args.iter_ms, matmul_state)

        pending = []
        max_queue_depth = 0
        iter_times = []
        store_wall_times = []
        last_tag = None

        for step in range(args.iters):
            begin_iter = time.perf_counter()
            _training_step(args.mode, args.iter_ms, matmul_state)

            if step % args.save_interval == 0:
                _fill_obj(obj, step)
                tag = f"synthetic_{step:06d}"
                begin_store = time.perf_counter()
                handle = racer.store(obj, tag=tag, context=ctx, async_op=async_store)
                if not async_store:
                    handle.wait()
                store_wall_ms = (time.perf_counter() - begin_store) * 1000.0
                store_wall_times.append(store_wall_ms)
                pending.append(handle)
                last_tag = tag

            if async_store:
                still_pending = []
                for handle in pending:
                    if not handle.done():
                        still_pending.append(handle)
                pending = still_pending
            max_queue_depth = max(max_queue_depth, len(pending))
            iter_times.append((time.perf_counter() - begin_iter) * 1000.0)

        for handle in pending:
            handle.wait()

        correct = True
        load_wall_ms = 0.0
        if args.verify and last_tag is not None:
            failed = [train_ranks[0]]
            begin_load = time.perf_counter()
            recovered = racer.load(tag=last_tag, failed_train_ranks=failed, context=ctx)
            _sync_all()
            load_wall_ms = (time.perf_counter() - begin_load) * 1000.0
            correct = all(torch.equal(recovered[rank].cpu(), obj[rank].cpu()) for rank in failed)

        avg_store_wall_ms = statistics.mean(store_wall_times) if store_wall_times else 0.0
        max_store_wall_ms = max(store_wall_times) if store_wall_times else 0.0
        avg_iter_ms = statistics.mean(iter_times) if iter_times else 0.0
        recommended = 1
        if async_store and baseline_iter_ms > 0 and avg_store_wall_ms > 0:
            recommended = max(1, math.ceil(avg_store_wall_ms / baseline_iter_ms))
        feasible = recommended <= 1

        cost = routing_cost(ctx.config, ctx.elastic_layout, ctx.matrix, packet_size)
        row = make_result_row(
            k=args.k,
            m=args.m,
            train_ranks=train_ranks,
            spare_ranks=spare_ranks,
            layout=ctx.elastic_layout,
            routing_strategy=args.routing_strategy,
            size_bytes=packet_size,
            encode_ms=avg_store_wall_ms,
            p2p_ms=0.0,
            xor_ms=0.0,
            store_wall_ms=avg_store_wall_ms,
            decode_ms=load_wall_ms,
            load_wall_ms=load_wall_ms,
            correct=correct,
            cost=cost,
        )
        rows.append(row)
        path = result_path("synthetic_loop")
        write_csv(path, rows)

        summary = {
            "baseline_iter_ms": round(baseline_iter_ms, 3),
            "avg_iter_ms_with_racer": round(avg_iter_ms, 3),
            "p50_iter_ms": round(percentile(iter_times, 50), 3),
            "p90_iter_ms": round(percentile(iter_times, 90), 3),
            "p99_iter_ms": round(percentile(iter_times, 99), 3),
            "avg_store_wall_ms": round(avg_store_wall_ms, 3),
            "max_store_wall_ms": round(max_store_wall_ms, 3),
            "async_queue_depth": max_queue_depth,
            "recommended_min_save_interval": recommended,
            "whether_iter_level_checkpoint_is_feasible": feasible,
            "correct": correct,
            "csv": str(path),
        }
        print(summary)
        barrier_if_distributed()
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
