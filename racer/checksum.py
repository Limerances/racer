"""Checksum helpers for stored RACER chunks and manifests."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import time
from typing import Any
import zlib

import torch

from .storage import StoredCheckpoint

try:
    import xxhash as _xxhash
except ImportError:  # pragma: no cover - optional performance dependency
    _xxhash = None


def tensor_checksum(
    tensor: torch.Tensor,
    *,
    buffer_size: int,
    sync_device: Callable[[torch.device], None],
    mode: str = "fast",
) -> str:
    flat = tensor.detach().contiguous().view(-1)
    numel = int(flat.numel())
    if flat.device.type == "cuda":
        sync_device(flat.device)
    normalized_mode = str(mode).lower().replace("-", "_")
    if normalized_mode in {"sha256", "strict", "strict_sha256"}:
        return sha256_tensor_checksum(flat, buffer_size=buffer_size)
    return sampled_tensor_checksum(flat)


def sampled_tensor_checksum(flat: torch.Tensor, *, sample_count: int = 4096) -> str:
    numel = int(flat.numel())
    if numel == 0:
        return "sample64-v1:0:0:0000000000000000:00:00"

    first = int(flat[0].item()) & 0xFF
    last = int(flat[-1].item()) & 0xFF
    full_sum_limit = 16 * 1024 * 1024
    if numel <= full_sum_limit:
        total = int(torch.sum(flat, dtype=torch.int64).item()) & 0xFFFFFFFFFFFFFFFF
        return f"sum64-v1:{numel}:{total:016x}:{first:02x}:{last:02x}"

    samples = min(int(sample_count), numel)
    if samples == 1:
        indices = torch.zeros(1, dtype=torch.long, device=flat.device)
    else:
        indices = torch.arange(samples, dtype=torch.long, device=flat.device)
        indices = indices * (numel - 1) // (samples - 1)
    sample = flat.index_select(0, indices)
    total = int(torch.sum(sample, dtype=torch.int64).item()) & 0xFFFFFFFFFFFFFFFF
    return f"sample64-v1:{numel}:{samples}:{total:016x}:{first:02x}:{last:02x}"


def sha256_tensor_checksum(flat: torch.Tensor, *, buffer_size: int) -> str:
    numel = int(flat.numel())
    digest = hashlib.sha256()
    if numel:
        chunk_size = max(1, int(buffer_size))
        for offset in range(0, numel, chunk_size):
            end = min(offset + chunk_size, numel)
            length = int(end - offset)
            part = flat.narrow(0, offset, length).detach().cpu().contiguous()
            digest.update(memoryview(part.numpy()))
    return f"sha256-v1:{numel}:{digest.hexdigest()}"


def manifest_checksum(chunks: list[dict[str, Any]]) -> str:
    return hashlib.sha256("".join(str(chunk.get("checksum", "")) for chunk in chunks).encode()).hexdigest()


def _sha256_hex_tensor(tensor: torch.Tensor) -> str:
    flat = tensor.detach().contiguous().view(-1)
    digest = hashlib.sha256()
    if int(flat.numel()):
        cpu = flat.detach().cpu().contiguous()
        digest.update(memoryview(cpu.numpy()))
    return digest.hexdigest()


def _crc32_hex_tensor(tensor: torch.Tensor) -> str:
    flat = tensor.detach().contiguous().view(-1)
    if int(flat.numel()) == 0:
        return "00000000"
    cpu = flat.detach().cpu().contiguous()
    return f"{zlib.crc32(memoryview(cpu.numpy())) & 0xFFFFFFFF:08x}"


def _xxh64_hex_tensor(tensor: torch.Tensor) -> str:
    if _xxhash is None:
        raise RuntimeError("checksum_type=xxh64 requires the optional xxhash package")
    flat = tensor.detach().contiguous().view(-1)
    if int(flat.numel()) == 0:
        return _xxhash.xxh64(b"").hexdigest()
    cpu = flat.detach().cpu().contiguous()
    return _xxhash.xxh64(memoryview(cpu.numpy())).hexdigest()


def verify_checkpoint_checksums(
    checkpoint: StoredCheckpoint,
    *,
    checksum_fn: Callable[[torch.Tensor], str],
    unavailable_rows: set[int] | None = None,
) -> dict[str, Any]:
    manifest = checkpoint.metadata
    chunks = manifest.get("chunks")
    if not chunks:
        return {"checksum_verify_ms": 0.0, "checksum_verified_chunks": 0}
    unavailable = set(unavailable_rows or set())
    start = time.perf_counter()
    verified = 0
    actual_checksums: list[str] = []
    for chunk in chunks:
        row = int(chunk["row"])
        if row in unavailable:
            actual_checksums.append(str(chunk.get("checksum", "")))
            continue
        group_index = int(chunk["reduction_group_index"])
        expected = str(chunk.get("checksum", ""))
        try:
            tensor = checkpoint.reduction_groups[group_index].rows[row]
        except IndexError as exc:
            raise RuntimeError(
                f"checkpoint {checkpoint.tag!r} is missing chunk row {row} in reduction group {group_index}"
            ) from exc
        checksum_type = str(chunk.get("checksum_type", "")).lower().replace("-", "_")
        if expected.startswith(("sum64-v1:", "sample64-v1:")):
            actual = sampled_tensor_checksum(tensor.detach().contiguous().view(-1))
        elif expected.startswith("sha256-v1:"):
            actual = sha256_tensor_checksum(tensor.detach().contiguous().view(-1), buffer_size=16 * 1024 * 1024)
        elif checksum_type in {"sha256", "sha256_v1"}:
            actual = _sha256_hex_tensor(tensor)
        elif checksum_type in {"sample64", "sample64_v1", "fast", "sampled"}:
            actual = sampled_tensor_checksum(tensor.detach().contiguous().view(-1))
        elif checksum_type in {"xxh64", "xxhash64", "xxh64_v1"}:
            actual = _xxh64_hex_tensor(tensor)
        elif checksum_type in {"crc32", "crc32_v1"}:
            actual = _crc32_hex_tensor(tensor)
        else:
            actual = checksum_fn(tensor)
        if expected and actual != expected:
            raise RuntimeError(
                f"RACER checksum mismatch for {chunk.get('chunk_id')}: expected {expected}, got {actual}"
            )
        actual_checksums.append(actual)
        verified += 1
    expected_manifest = manifest.get("checksum")
    if expected_manifest:
        actual_manifest = hashlib.sha256("".join(actual_checksums).encode()).hexdigest()
        if actual_manifest != expected_manifest:
            raise RuntimeError(
                f"RACER manifest checksum mismatch for {checkpoint.tag!r}: "
                f"expected {expected_manifest}, got {actual_manifest}"
            )
    return {
        "checksum_verify_ms": (time.perf_counter() - start) * 1000.0,
        "checksum_verified_chunks": verified,
    }
