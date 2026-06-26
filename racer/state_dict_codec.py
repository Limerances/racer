"""Flatten and unflatten supported single-process state_dict payloads.

This module only serializes tensor metadata and exposes byte payload tensors so
the EC matrix work can happen on GPU.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import copy
from typing import Any


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
class NonTensorMetadata:
    key: str
    value: Any


@dataclass(frozen=True)
class RankStateMetadata:
    source_train_rank: int
    tensors: list[TensorMetadata]
    payload_nbytes: int
    non_tensors: list[NonTensorMetadata] = field(default_factory=list)


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
    non_tensor_meta: list[NonTensorMetadata] = []
    for key, tensor in state_dict.items():
        if not isinstance(key, str):
            raise TypeError("state_dict keys must be strings.")
        if not isinstance(tensor, torch.Tensor):
            non_tensor_meta.append(NonTensorMetadata(key=key, value=copy.deepcopy(tensor)))
            continue
        if tensor.layout != torch.strided:
            raise TypeError(f"state_dict[{key!r}] must be a dense strided tensor.")
        dtype_name = str(tensor.dtype)

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
        payload_device = target_device if target_device is not None else "cpu"
        payload = torch.empty(0, dtype=torch.uint8, device=payload_device)
    metadata = RankStateMetadata(
        source_train_rank=source_train_rank,
        tensors=tensor_meta,
        payload_nbytes=int(payload.numel()),
        non_tensors=non_tensor_meta,
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

    result: dict[str, Any] = {
        item.key: copy.deepcopy(item.value)
        for item in getattr(metadata, "non_tensors", [])
    }
    for tensor_meta in metadata.tensors:
        dtype = _dtype_from_name(tensor_meta.dtype, torch)
        device = target_device if target_device is not None else tensor_meta.device
        byte_slice = _tensor_byte_slice(
            payload,
            offset=tensor_meta.offset,
            nbytes=tensor_meta.nbytes,
            dtype=dtype,
            target_device=device,
            torch=torch,
        )
        tensor = byte_slice.view(dtype).reshape(tensor_meta.shape)
        if tensor_meta.requires_grad and tensor.is_floating_point():
            if not tensor.is_leaf:
                tensor = tensor.clone()
            tensor.requires_grad_(True)
        result[tensor_meta.key] = tensor
    return result


def _tensor_byte_slice(
    payload: Any,
    *,
    offset: int,
    nbytes: int,
    dtype: Any,
    target_device: Any,
    torch: Any,
) -> Any:
    byte_slice = payload.narrow(0, int(offset), int(nbytes))
    target = torch.device(target_device)
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())
    if byte_slice.device != target:
        return byte_slice.to(target, non_blocking=True).contiguous()

    element_size = torch.empty((), dtype=dtype).element_size()
    if element_size > 1 and int(byte_slice.storage_offset()) % int(element_size) != 0:
        return byte_slice.clone().contiguous()
    return byte_slice


def _dtype_from_name(name: str, torch: Any) -> Any:
    if not name.startswith("torch."):
        raise TypeError(f"Unsupported tensor dtype metadata: {name}.")
    attr = name.split(".", 1)[1]
    dtype = getattr(torch, attr, None)
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"Unsupported tensor dtype metadata: {name}.")
    return dtype


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for RACER state_dict payloads.") from exc
    return torch
