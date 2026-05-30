"""Synthetic GPT2/Megatron-style checkpoint states for RACER benchmarks.

This module intentionally does not import Megatron. It models checkpoint tensor
shapes and optimizer payload size closely enough for RACER encode/decode
performance tests: every train rank owns many model tensors plus optional Adam
optimizer tensors under tensor parallelism.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from typing import Iterable

import torch


@dataclass(frozen=True)
class GPT2CheckpointConfig:
    profile: str = "gpt2-124m"
    num_layers: int = 12
    hidden_size: int = 768
    num_attention_heads: int = 12
    ffn_hidden_size: int = 3072
    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    tensor_parallel: int = 4
    dtype: str = "bf16"
    include_optimizer: bool = True
    include_master_weights: bool = True
    include_rng_state: bool = True


@dataclass(frozen=True)
class TensorSpec:
    key: str
    shape: tuple[int, ...]
    dtype: str
    category: str

    @property
    def numel(self) -> int:
        out = 1
        for dim in self.shape:
            out *= int(dim)
        return out

    @property
    def nbytes(self) -> int:
        return self.numel * dtype_nbytes(self.dtype)


_PROFILES = {
    "gpt2-124m": GPT2CheckpointConfig(
        profile="gpt2-124m",
        num_layers=12,
        hidden_size=768,
        num_attention_heads=12,
        ffn_hidden_size=3072,
        vocab_size=50257,
        max_position_embeddings=1024,
        tensor_parallel=4,
        dtype="bf16",
    ),
    "gpt2-medium-355m": GPT2CheckpointConfig(
        profile="gpt2-medium-355m",
        num_layers=24,
        hidden_size=1024,
        num_attention_heads=16,
        ffn_hidden_size=4096,
        vocab_size=50257,
        max_position_embeddings=1024,
        tensor_parallel=4,
        dtype="bf16",
    ),
    "paper-gpt2-1.6b": GPT2CheckpointConfig(
        profile="paper-gpt2-1.6b",
        num_layers=48,
        hidden_size=1600,
        num_attention_heads=32,
        ffn_hidden_size=6400,
        vocab_size=50257,
        max_position_embeddings=1024,
        tensor_parallel=4,
        dtype="bf16",
    ),
    "paper-gpt2-5.3b": GPT2CheckpointConfig(
        profile="paper-gpt2-5.3b",
        num_layers=64,
        hidden_size=2560,
        num_attention_heads=40,
        ffn_hidden_size=10240,
        vocab_size=50257,
        max_position_embeddings=1024,
        tensor_parallel=4,
        dtype="bf16",
    ),
    "paper-gpt2-20b": GPT2CheckpointConfig(
        profile="paper-gpt2-20b",
        num_layers=64,
        hidden_size=5120,
        num_attention_heads=40,
        ffn_hidden_size=20480,
        vocab_size=50257,
        max_position_embeddings=1024,
        tensor_parallel=4,
        dtype="bf16",
    ),
    # Mirrors the small TP4 GPT functional-test shape observed in
    # Megatron-LM-FT: 12 layers, hidden 512, 8 heads, TP=4, bf16.
    "megatron-test-tp4": GPT2CheckpointConfig(
        profile="megatron-test-tp4",
        num_layers=12,
        hidden_size=512,
        num_attention_heads=8,
        ffn_hidden_size=2048,
        vocab_size=50257,
        max_position_embeddings=1024,
        tensor_parallel=4,
        dtype="bf16",
    ),
}


def profile_config(
    profile: str = "gpt2-124m",
    *,
    tensor_parallel: int | None = None,
    dtype: str | None = None,
    include_optimizer: bool | None = None,
    include_master_weights: bool | None = None,
) -> GPT2CheckpointConfig:
    try:
        config = _PROFILES[profile]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROFILES))
        raise ValueError(f"unknown GPT2 checkpoint profile {profile!r}; choices: {choices}") from exc
    updates = {}
    if tensor_parallel is not None:
        updates["tensor_parallel"] = int(tensor_parallel)
    if dtype is not None:
        updates["dtype"] = dtype
    if include_optimizer is not None:
        updates["include_optimizer"] = bool(include_optimizer)
    if include_master_weights is not None:
        updates["include_master_weights"] = bool(include_master_weights)
    return replace(config, **updates)


def available_profiles() -> list[str]:
    return sorted(_PROFILES)


def iter_tensor_specs(config: GPT2CheckpointConfig) -> Iterable[TensorSpec]:
    tp = int(config.tensor_parallel)
    if tp <= 0:
        raise ValueError("tensor_parallel must be positive")
    hidden = int(config.hidden_size)
    ffn = int(config.ffn_hidden_size)
    if hidden % tp != 0:
        raise ValueError("hidden_size must be divisible by tensor_parallel for TP-sharded projections")
    if ffn % tp != 0:
        raise ValueError("ffn_hidden_size must be divisible by tensor_parallel")
    if (3 * hidden) % tp != 0:
        raise ValueError("3 * hidden_size must be divisible by tensor_parallel")

    model_dtype = normalize_dtype_name(config.dtype)
    vocab_per_rank = ceil(int(config.vocab_size) / tp)
    hidden_per_rank = hidden // tp
    qkv_per_rank = (3 * hidden) // tp
    ffn_per_rank = ffn // tp

    model_specs: list[TensorSpec] = [
        TensorSpec("model.embedding.word_embeddings.weight", (vocab_per_rank, hidden), model_dtype, "model"),
        TensorSpec("model.embedding.position_embeddings.weight", (int(config.max_position_embeddings), hidden), model_dtype, "model"),
        TensorSpec("model.final_layernorm.weight", (hidden,), model_dtype, "model"),
        TensorSpec("model.final_layernorm.bias", (hidden,), model_dtype, "model"),
    ]

    for layer in range(int(config.num_layers)):
        prefix = f"model.decoder.layers.{layer}"
        model_specs.extend(
            [
                TensorSpec(f"{prefix}.input_layernorm.weight", (hidden,), model_dtype, "model"),
                TensorSpec(f"{prefix}.input_layernorm.bias", (hidden,), model_dtype, "model"),
                TensorSpec(
                    f"{prefix}.self_attention.query_key_value.weight",
                    (qkv_per_rank, hidden),
                    model_dtype,
                    "model",
                ),
                TensorSpec(f"{prefix}.self_attention.query_key_value.bias", (qkv_per_rank,), model_dtype, "model"),
                TensorSpec(
                    f"{prefix}.self_attention.dense.weight",
                    (hidden, hidden_per_rank),
                    model_dtype,
                    "model",
                ),
                TensorSpec(f"{prefix}.self_attention.dense.bias", (hidden,), model_dtype, "model"),
                TensorSpec(f"{prefix}.post_attention_layernorm.weight", (hidden,), model_dtype, "model"),
                TensorSpec(f"{prefix}.post_attention_layernorm.bias", (hidden,), model_dtype, "model"),
                TensorSpec(
                    f"{prefix}.mlp.dense_h_to_4h.weight",
                    (ffn_per_rank, hidden),
                    model_dtype,
                    "model",
                ),
                TensorSpec(f"{prefix}.mlp.dense_h_to_4h.bias", (ffn_per_rank,), model_dtype, "model"),
                TensorSpec(
                    f"{prefix}.mlp.dense_4h_to_h.weight",
                    (hidden, ffn_per_rank),
                    model_dtype,
                    "model",
                ),
                TensorSpec(f"{prefix}.mlp.dense_4h_to_h.bias", (hidden,), model_dtype, "model"),
            ]
        )

    for spec in model_specs:
        yield spec
        if config.include_optimizer:
            if config.include_master_weights and spec.dtype != "fp32":
                yield TensorSpec(f"optimizer.fp32_master.{spec.key}", spec.shape, "fp32", "optimizer")
            yield TensorSpec(f"optimizer.state.{spec.key}.exp_avg", spec.shape, "fp32", "optimizer")
            yield TensorSpec(f"optimizer.state.{spec.key}.exp_avg_sq", spec.shape, "fp32", "optimizer")

    if config.include_rng_state:
        yield TensorSpec("rng.cuda_rng_state", (4096,), "uint8", "metadata")
        yield TensorSpec("metadata.consumed_train_samples", (1,), "int64", "metadata")
        yield TensorSpec("metadata.iteration", (1,), "int64", "metadata")


def estimate_rank_nbytes(config: GPT2CheckpointConfig) -> int:
    return sum(spec.nbytes for spec in iter_tensor_specs(config))


def estimate_total_nbytes(config: GPT2CheckpointConfig, train_ranks: int | None = None) -> int:
    ranks = int(train_ranks if train_ranks is not None else config.tensor_parallel)
    return estimate_rank_nbytes(config) * ranks


def tensor_count(config: GPT2CheckpointConfig) -> int:
    return sum(1 for _ in iter_tensor_specs(config))


def make_rank_state(
    rank: int,
    config: GPT2CheckpointConfig,
    *,
    device: torch.device | str = "cuda:0",
    fill: bool = False,
    max_tensors: int | None = None,
) -> dict[str, torch.Tensor]:
    dev = torch.device(device)
    state: dict[str, torch.Tensor] = {}
    for index, spec in enumerate(iter_tensor_specs(config)):
        if max_tensors is not None and index >= int(max_tensors):
            break
        tensor = torch.empty(spec.shape, dtype=torch_dtype(spec.dtype), device=dev)
        if fill:
            _fill_tensor(tensor, rank=rank, index=index)
        state[spec.key] = tensor
    return state


def make_rank_states(
    ranks: Iterable[int],
    config: GPT2CheckpointConfig,
    *,
    fill: bool = False,
    max_tensors: int | None = None,
) -> dict[int, dict[str, torch.Tensor]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to materialize RACER benchmark states")
    states: dict[int, dict[str, torch.Tensor]] = {}
    for rank in ranks:
        states[int(rank)] = make_rank_state(
            int(rank),
            config,
            device=torch.device("cuda", int(rank)),
            fill=fill,
            max_tensors=max_tensors,
        )
    return states


def state_nbytes(state: dict[str, torch.Tensor]) -> int:
    return sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in state.values())


def states_nbytes(states: dict[int, dict[str, torch.Tensor]]) -> int:
    return sum(state_nbytes(state) for state in states.values())


def state_dict_byte_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    if set(left) != set(right):
        return False
    touched: set[torch.device] = set()
    for key in left:
        a = left[key].detach().contiguous().view(torch.uint8)
        b = right[key].detach().contiguous().view(torch.uint8)
        if b.device != a.device:
            b = b.to(a.device, non_blocking=True)
        if a.device.type == "cuda":
            touched.add(a.device)
        if not torch.equal(a, b):
            return False
    for device in touched:
        torch.cuda.synchronize(device)
    return True


def normalize_dtype_name(name: str) -> str:
    lowered = str(name).lower()
    aliases = {
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "fp32": "fp32",
        "float32": "fp32",
        "uint8": "uint8",
        "int64": "int64",
    }
    try:
        return aliases[lowered]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype name: {name}") from exc


def torch_dtype(name: str) -> torch.dtype:
    normalized = normalize_dtype_name(name)
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "uint8": torch.uint8,
        "int64": torch.int64,
    }
    return mapping[normalized]


def dtype_nbytes(name: str) -> int:
    return torch.empty((), dtype=torch_dtype(name)).element_size()


def _fill_tensor(tensor: torch.Tensor, *, rank: int, index: int) -> None:
    value = (int(rank) * 17 + int(index) * 13) & 0xFF
    if tensor.dtype == torch.uint8:
        tensor.fill_(value)
    elif tensor.dtype == torch.int64:
        tensor.fill_(value)
    else:
        tensor.fill_(float(value % 31) / 31.0)
