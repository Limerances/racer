#!/usr/bin/env python3
"""Small torchrun rendezvous smoke test for local multi-node simulation."""

from __future__ import annotations

import os
import socket

import torch
import torch.distributed as dist


def main() -> None:
    backend = os.environ.get("BACKEND", "gloo")
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("BACKEND=nccl requires CUDA")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world = dist.get_world_size()
    value = torch.tensor([rank + 1], dtype=torch.int64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    dist.barrier()

    expected = world * (world + 1) // 2
    if int(value.item()) != expected:
        raise RuntimeError(f"all_reduce mismatch: got {int(value.item())}, expected {expected}")

    print(
        "OK "
        f"host={socket.gethostname()} "
        f"rank={rank} "
        f"world={world} "
        f"local_rank={os.environ.get('LOCAL_RANK')} "
        f"group_rank={os.environ.get('GROUP_RANK')} "
        f"backend={backend} "
        f"sum={int(value.item())} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
