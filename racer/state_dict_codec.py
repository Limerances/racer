"""Flatten and unflatten supported single-process state_dict payloads.

This module is not a CPU Reed-Solomon codec. It only serializes tensor metadata
and exposes byte payload tensors so the EC matrix work can happen on GPU.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


SUPPORTED_DTYPE_NAMES = {
    "torch.float32",
    "torch.float16",
    "torch.bfloat16",
    "torch.int64",
    "torch.uint8",
}


@dataclass(frozen=True)
class TensorMetadata:
    key: str
    dtype: str
    shape: tuple[int, ...]
    device: str
    requires_grad: bool
    was_contiguous: bool
    offset: int
    nbytes: int


@dataclass(frozen=True)
class RankStateMetadata:
    source_train_rank: int
    tensors: list[TensorMetadata]
    payload_nbytes: int


@dataclass(frozen=True)
class FlattenedRankState:
    source_train_rank: int
    metadata: RankStateMetadata
    payload: Any


def flatten_state_dict(
    source_train_rank: int,
    state_dict: Mapping[str, Any],
    *,
    target_device: Any | None = None,
) -> FlattenedRankState:
    """Flatten one train rank state_dict into one ECCheck data packet payload."""

    torch = _torch()
    if not isinstance(state_dict, Mapping):
        raise TypeError("each rank payload must be a dict[str, torch.Tensor].")

    pieces = []
    tensor_meta: list[TensorMetadata] = []
    offset = 0
    for key, tensor in state_dict.items():
        if not isinstance(key, str):
            raise TypeError("state_dict tensor keys must be strings.")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict[{key!r}] must be a torch.Tensor.")
        dtype_name = str(tensor.dtype)
        if dtype_name not in SUPPORTED_DTYPE_NAMES:
            raise TypeError(f"Unsupported tensor dtype for {key!r}: {tensor.dtype}.")

        original_device = str(tensor.device)
        was_contiguous = tensor.is_contiguous()
        contiguous = tensor.detach().contiguous()
        byte_view = contiguous.view(torch.uint8).reshape(-1)
        if target_device is not None:
            byte_view = byte_view.to(target_device, non_blocking=True)
        else:
            byte_view = byte_view.clone()
        byte_view = byte_view.contiguous()
        nbytes = int(byte_view.numel())
        pieces.append(byte_view)
        tensor_meta.append(
            TensorMetadata(
                key=key,
                dtype=dtype_name,
                shape=tuple(int(dim) for dim in tensor.shape),
                device=original_device,
                requires_grad=bool(tensor.requires_grad),
                was_contiguous=was_contiguous,
                offset=offset,
                nbytes=nbytes,
            )
        )
        offset += nbytes

    if pieces:
        payload = torch.cat(pieces).contiguous()
    else:
        device = target_device if target_device is not None else "cpu"
        payload = torch.empty(0, dtype=torch.uint8, device=device)
    metadata = RankStateMetadata(
        source_train_rank=source_train_rank,
        tensors=tensor_meta,
        payload_nbytes=int(payload.numel()),
    )
    return FlattenedRankState(
        source_train_rank=source_train_rank,
        metadata=metadata,
        payload=payload,
    )


def unflatten_state_dict(
    metadata: RankStateMetadata,
    payload: Any,
    *,
    target_device: Any | None = None,
) -> dict[str, Any]:
    """Rebuild one train rank state_dict from a byte payload tensor."""

    torch = _torch()
    if not isinstance(payload, torch.Tensor):
        raise TypeError("payload must be a torch.Tensor.")
    if payload.dtype is not torch.uint8:
        raise TypeError("payload must have dtype torch.uint8.")
    if payload.numel() < metadata.payload_nbytes:
        raise ValueError(
            f"payload has {payload.numel()} bytes, expected at least {metadata.payload_nbytes}."
        )

    result: dict[str, Any] = {}
    for tensor_meta in metadata.tensors:
        dtype = _dtype_from_name(tensor_meta.dtype, torch)
        byte_slice = payload[tensor_meta.offset : tensor_meta.offset + tensor_meta.nbytes]
        device = target_device if target_device is not None else tensor_meta.device
        byte_slice = byte_slice.clone().to(device, non_blocking=True).contiguous()
        tensor = byte_slice.view(dtype).reshape(tensor_meta.shape).clone()
        if tensor_meta.requires_grad and tensor.is_floating_point():
            tensor.requires_grad_(True)
        result[tensor_meta.key] = tensor
    return result


def _dtype_from_name(name: str, torch: Any) -> Any:
    mapping = {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.int64": torch.int64,
        "torch.uint8": torch.uint8,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise TypeError(f"Unsupported tensor dtype metadata: {name}.") from exc


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for RACER state_dict payloads.") from exc
    return torch
