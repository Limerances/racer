"""Small validation and CUDA synchronization helpers for RACER."""

from __future__ import annotations

from typing import Mapping

import torch


def uint8_view_no_serialize(tensor: torch.Tensor) -> torch.Tensor:
    """Return a flat uint8 view of one contiguous tensor without serialization.

    This helper is for synthetic checkpoint packets that already live as tensor
    storage. It does not pickle, `torch.save`, or otherwise serialize a Python
    state_dict. Non-contiguous tensors are rejected because making them
    contiguous would introduce an implicit packing copy.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("expected a torch.Tensor")
    if not tensor.is_contiguous():
        raise ValueError("uint8_view_no_serialize requires a contiguous tensor")
    if tensor.dtype == torch.uint8:
        return tensor.view(-1)
    return tensor.view(torch.uint8).view(-1)


def require_uint8_tensor_map(obj: object) -> Mapping[int, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError("phase 1 racer.store expects obj to be Dict[int, torch.Tensor]")
    for key, value in obj.items():
        if not isinstance(key, int):
            raise TypeError("checkpoint dict keys must be train rank integers")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"checkpoint value for rank {key} is not a torch.Tensor")
        if value.dtype != torch.uint8:
            raise TypeError(f"checkpoint value for rank {key} must have dtype torch.uint8")
    return obj


def normalize_rank_set(values: list[int] | tuple[int, ...] | None) -> set[int]:
    if values is None:
        return set()
    return {int(v) for v in values}


def cuda_device(rank: int) -> torch.device:
    return torch.device("cuda", int(rank))


def synchronize_devices(devices: set[torch.device]) -> None:
    for device in devices:
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.cuda.current_device()
            with torch.cuda.device(index):
                torch.cuda.synchronize(index)
