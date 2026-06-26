"""RACER runtime context implementing store/load over synthetic packets.

The context owns the immutable RACER configuration, the train-rank-only E
matrix and elastic data/reduction group metadata. Phase 1 keeps
the API local to one process while preserving the intended distributed
semantics in manifests and routing plans.

TODO: add a Megatron adapter for sharded state_dict checkpoints.
TODO: move async storage and recovery work into a long-lived racerd daemon.
"""

from __future__ import annotations

from collections.abc import Mapping
import gc
import os
import threading
import time
from typing import Any, Sequence

import torch

from . import cauchy, checksum, codec_cuda, gf256, manifest, routing
from .config import RacerConfig
from .handles import RepairHandle, StoreHandle
from .layout import ElasticLayout, ElasticSlot
from .state_dict_codec import (
    RankStateMetadata,
    flatten_state_dict,
    unflatten_state_dict,
)
from .storage import StoredCheckpoint, StoredReductionGroup
from .utils import normalize_rank_set, require_uint8_tensor_map, synchronize_devices


class RacerContext:
    def __init__(self, config: RacerConfig, chunk_storage: Any) -> None:
        self.config = config
        self.matrix = cauchy.systematic_matrix(
            config.k,
            config.m,
            config.w,
            optimize=config.optimize_cauchy,
        )
        self.elastic_layout = ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m)
        if chunk_storage is None:
            raise ValueError(
                "RacerContext requires an explicit daemon-owned chunk_storage; "
                "in-process checkpoint storage is disabled."
            )
        self.chunk_storage = chunk_storage
        if not self._chunk_storage_is_restart_aware_daemon():
            raise ValueError(
                "RacerContext only accepts restart-aware daemon-owned checkpoint storage. "
                "In-process CUDA, CPU pinned, fd/mmap, and other fallback storage paths are disabled."
            )
        self._local_checkpoint_index_enabled = False
        self.storage = None
        self.train_rank_to_row = {rank: row for row, rank in enumerate(config.train_ranks)}
        self.last_routing_plan: routing.RoutingPlan | None = None
        self.last_store_profile: dict[str, Any] = {}
        self.last_load_profile: dict[str, Any] = {}
        self._counter = 0

        if not torch.cuda.is_available():
            raise RuntimeError("RACER requires CUDA")
        max_rank = max(config.train_ranks + config.spare_ranks)
        if torch.cuda.device_count() <= max_rank:
            raise RuntimeError(
                f"CUDA device_count={torch.cuda.device_count()} is insufficient for rank {max_rank}"
            )

    @property
    def n_train(self) -> int:
        return len(self.config.train_ranks)

    def _next_tag(self) -> str:
        self._counter += 1
        return f"racer_{self._counter:06d}"

    def _chunk_storage_capabilities(self) -> dict[str, Any]:
        capabilities_fn = getattr(self.chunk_storage, "capabilities", None)
        if capabilities_fn is None:
            return {}
        try:
            return dict(capabilities_fn())
        except Exception:
            return {}

    def _chunk_storage_is_restart_aware_daemon(self) -> bool:
        caps = self._chunk_storage_capabilities()
        return bool(caps.get("restart_aware", False)) and bool(caps.get("daemon_owned", False))

    def _chunk_storage_supports_cuda_ipc(self) -> bool:
        if not hasattr(self.chunk_storage, "put_cuda_tensor") or not hasattr(self.chunk_storage, "wait"):
            return False
        caps = self._chunk_storage_capabilities()
        return bool(caps.get("supports_cuda_ipc", False)) and bool(caps.get("supports_async_copy", False))

    def _validate_obj(self, obj: object) -> Mapping[int, torch.Tensor]:
        tensor_map = require_uint8_tensor_map(obj)
        keys = set(tensor_map.keys())
        train = set(self.config.train_ranks)
        spare = set(self.config.spare_ranks)
        missing = train - keys
        extra = keys - train
        spare_present = keys & spare
        if missing:
            raise ValueError(f"checkpoint obj is missing train ranks: {sorted(missing)}")
        if spare_present:
            raise ValueError(f"spare ranks must not appear in checkpoint obj: {sorted(spare_present)}")
        if extra:
            raise ValueError(f"checkpoint obj contains ranks outside train_ranks: {sorted(extra)}")
        for rank, tensor in tensor_map.items():
            if tensor.device.type != "cuda":
                raise ValueError(f"rank {rank} tensor must be a CUDA tensor")
        return tensor_map


    def _validate_rank_keys(self, obj: Mapping[int, Any]) -> None:
        keys = {int(rank) for rank in obj.keys()}
        train = set(self.config.train_ranks)
        spare = set(self.config.spare_ranks)
        missing = train - keys
        extra = keys - train
        spare_present = keys & spare
        if missing:
            raise ValueError(f"checkpoint obj is missing train ranks: {sorted(missing)}")
        if spare_present:
            raise ValueError(f"spare ranks must not appear in checkpoint obj: {sorted(spare_present)}")
        if extra:
            raise ValueError(f"checkpoint obj contains ranks outside train_ranks: {sorted(extra)}")

    def _state_dict_target_device(self, rank: int) -> torch.device:
        return torch.device("cuda", int(rank))

    def _prepare_obj(
        self,
        obj: object,
    ) -> tuple[Mapping[int, torch.Tensor], str, dict[int, RankStateMetadata]]:
        if not isinstance(obj, Mapping):
            raise TypeError("checkpoint obj must be Mapping[int, torch.Tensor | Mapping[str, torch.Tensor]]")

        values = list(obj.values())
        if all(isinstance(value, torch.Tensor) for value in values):
            return self._validate_obj(obj), "raw_tensor", {}

        if all(isinstance(value, Mapping) for value in values):
            self._validate_rank_keys(obj)
            packets: dict[int, torch.Tensor] = {}
            metadata: dict[int, RankStateMetadata] = {}
            for rank in self.config.train_ranks:
                flat = flatten_state_dict(
                    int(rank),
                    obj[int(rank)],  # type: ignore[arg-type]
                    target_device=self._state_dict_target_device(int(rank)),
                )
                packets[int(rank)] = flat.payload
                metadata[int(rank)] = flat.metadata
            return self._validate_obj(packets), "state_dict", metadata

        raise TypeError(
            "checkpoint obj values must be all torch.Tensor payloads or all state_dict mappings"
        )

    def _metadata_to_manifest(self, metadata: RankStateMetadata) -> dict[str, Any]:
        return manifest.state_metadata_to_manifest(metadata)

    def _metadata_from_manifest(self, data: Mapping[str, Any]) -> RankStateMetadata:
        return manifest.state_metadata_from_manifest(dict(data))

    def _rank_state_metadata(self, checkpoint: StoredCheckpoint, rank: int) -> RankStateMetadata:
        by_rank = checkpoint.metadata.get("rank_state_metadata", {})
        try:
            data = by_rank[str(int(rank))]
        except KeyError as exc:
            raise KeyError(f"checkpoint {checkpoint.tag!r} has no state_dict metadata for rank {rank}") from exc
        if isinstance(data, RankStateMetadata):
            return data
        return self._metadata_from_manifest(data)

    def _collect_devices(self, values: Sequence[Any]) -> set[torch.device]:
        devices: set[torch.device] = set()
        for value in values:
            if isinstance(value, torch.Tensor):
                devices.add(value.device)
            elif isinstance(value, Mapping):
                for nested in value.values():
                    if isinstance(nested, torch.Tensor):
                        devices.add(nested.device)
        return devices

    def _sync_compute_device(self, device: torch.device) -> None:
        # Cross-GPU copies feeding spare-side kernels need an explicit phase-1
        # ordering point. Later routing work can replace this with CUDA events.
        if device.type == "cuda":
            index = device.index if device.index is not None else torch.cuda.current_device()
            with torch.cuda.device(index):
                torch.cuda.synchronize(index)

    def _sync_tensor_device(self, tensor: torch.Tensor) -> None:
        if tensor.device.type == "cuda":
            self._sync_compute_device(tensor.device)

    def _empty_storage_row(self, numel: int, device: torch.device) -> torch.Tensor:
        if device.type != "cuda":
            raise ValueError("RACER storage rows must be allocated on CUDA devices")
        return torch.empty(int(numel), dtype=torch.uint8, device=device)

    def _host_buffer_owner_rank(self, data_ranks: tuple[int | None, ...], row: int) -> int | None:
        if row < self.config.k:
            rank = data_ranks[row]
            return None if rank is None else int(rank)
        if self.config.spare_ranks:
            parity_id = (int(row) - self.config.k) % len(self.config.spare_ranks)
            return int(self.config.spare_ranks[parity_id])
        return int(self.config.train_ranks[int(row)])

    def _parity_compute_device(self, parity_id: int) -> torch.device:
        rank = self.config.spare_ranks[int(parity_id) % len(self.config.spare_ranks)]
        return torch.device("cuda", int(rank))

    def _codec_apply_matrix(
        self,
        blocks: Sequence[torch.Tensor],
        matrix: Sequence[Sequence[int]],
        outputs: Sequence[torch.Tensor] | None = None,
    ) -> list[torch.Tensor]:
        return codec_cuda.apply_matrix_cuda(blocks, matrix, outputs=outputs)

    def _codec_encode(self, blocks: Sequence[torch.Tensor], matrix: Sequence[Sequence[int]]) -> list[torch.Tensor]:
        return codec_cuda.encode_blocks(blocks, matrix)

    def _codec_decode(
        self,
        blocks: Sequence[torch.Tensor],
        survivor_rows: Sequence[int],
        matrix: Sequence[Sequence[int]],
    ) -> list[torch.Tensor]:
        return codec_cuda.decode_blocks(blocks, survivor_rows, matrix)

    def _checksum(self, tensor: torch.Tensor) -> str:
        return checksum.tensor_checksum(
            tensor,
            buffer_size=int(self.config.buffer_size),
            sync_device=self._sync_compute_device,
        )

    def _verify_checkpoint_checksums(
        self,
        checkpoint: StoredCheckpoint,
        unavailable_rows: set[int] | None = None,
    ) -> dict[str, Any]:
        return checksum.verify_checkpoint_checksums(
            checkpoint,
            checksum_fn=self._checksum,
            unavailable_rows=unavailable_rows,
        )

    def _chunk_id(self, reduction_group_index: int, row: int) -> str:
        return manifest.chunk_id(reduction_group_index, row)

    def _build_manifest(
        self,
        tag: str,
        stored_reduction_groups: list[StoredReductionGroup],
        plan: routing.RoutingPlan,
    ) -> dict:
        return manifest.build_manifest(
            tag=tag,
            config=self.config,
            matrix=[row[:] for row in self.matrix],
            elastic_layout=self.elastic_layout,
            stored_reduction_groups=stored_reduction_groups,
            plan=plan,
            host_buffer_owner_rank=self._host_buffer_owner_rank,
            checksum_fn=self._checksum,
        )

    def _write_chunks_and_manifest(
        self,
        tag: str,
        stored_reduction_groups: list[StoredReductionGroup],
        manifest_data: dict,
    ) -> dict[str, Any]:
        return manifest.write_chunks_and_manifest(
            chunk_storage=self.chunk_storage,
            tag=tag,
            stored_reduction_groups=stored_reduction_groups,
            manifest=manifest_data,
        )

    def _checkpoint_from_chunk_storage(self, tag: str | None) -> StoredCheckpoint:
        if tag is None:
            raise KeyError("tag is required when loading from chunk storage")
        return manifest.checkpoint_from_chunk_storage(tag=tag, chunk_storage=self.chunk_storage)

    def _checkpoint_for_load(self, tag: str | None) -> StoredCheckpoint:
        if not self._local_checkpoint_index_enabled:
            return self._checkpoint_from_chunk_storage(tag)
        try:
            if self.storage is None:
                return self._checkpoint_from_chunk_storage(tag)
            return self.storage.get(tag)
        except KeyError:
            return self._checkpoint_from_chunk_storage(tag)

    def _drop_existing_checkpoint(self, tag: str) -> tuple[bool, float]:
        start = time.perf_counter()
        actual = str(tag)
        existed = (
            (
                self._local_checkpoint_index_enabled
                and self.storage is not None
                and actual in self.storage.tags()
            )
            or bool(self.chunk_storage.list_chunks(actual))
        )
        if not existed:
            return False, 0.0
        if self._local_checkpoint_index_enabled and self.storage is not None:
            self.storage.delete(actual)
        self.chunk_storage.delete(actual)
        if os.environ.get("RACER_AGGRESSIVE_CUDA_CLEANUP", "0") == "1":
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return True, (time.perf_counter() - start) * 1000.0

    def _chunk_ranges(self, total: int) -> list[tuple[int, int]]:
        chunk_size = max(1, int(self.config.buffer_size))
        return [(start, min(start + chunk_size, total)) for start in range(0, int(total), chunk_size)]

    def _store_reduction_group_chunked_parity(
        self,
        obj: Mapping[int, torch.Tensor],
        reduction_group_index: int,
        reduction_group: tuple[ElasticSlot, ...],
        compute_device: torch.device,
    ) -> tuple[StoredReductionGroup, dict[str, Any], set[torch.device]]:
        data_ranks = tuple(slot.train_rank for slot in reduction_group)
        shapes: dict[int, tuple[int, ...]] = {}
        numels: dict[int, int] = {}
        reduction_group_bytes = 0
        flats: dict[int, torch.Tensor] = {}
        for rank in data_ranks:
            if rank is None:
                continue
            flat = obj[rank].contiguous().view(-1)
            flats[int(rank)] = flat
            shapes[int(rank)] = tuple(obj[rank].shape)
            numels[int(rank)] = int(flat.numel())
            reduction_group_bytes = max(reduction_group_bytes, int(flat.numel()))
        if reduction_group_bytes == 0:
            reduction_group_bytes = 1

        parity_devices: dict[torch.device, list[int]] = {}
        for parity_id in range(self.config.m):
            parity_devices.setdefault(self._parity_compute_device(parity_id), []).append(parity_id)

        touched_devices = set(parity_devices) or {compute_device}
        stored_rows: list[torch.Tensor] = []
        data_direct_save_ms = 0.0
        parity_chunk_save_ms = 0.0
        spare_buffer_alloc_ms = 0.0
        reduction_group_pack_ms = 0.0
        ec_encode_ms = 0.0
        data_row_bytes = 0
        parity_row_bytes = 0
        data_row_count = 0
        parity_row_count = 0

        for col, rank in enumerate(data_ranks):
            dst = routing.storage_device_for_row(self.config, col)
            touched_devices.add(dst)
            start = time.perf_counter()
            if rank is None:
                row = torch.empty(0, dtype=torch.uint8, device=dst)
            else:
                flat = flats[int(rank)]
                row = self._empty_storage_row(reduction_group_bytes, dst)
                copy_len = int(flat.numel())
                if copy_len:
                    row.narrow(0, 0, copy_len).copy_(flat, non_blocking=True)
                    self._sync_tensor_device(row)
                if copy_len < reduction_group_bytes:
                    row.narrow(0, copy_len, reduction_group_bytes - copy_len).zero_()
            self._sync_tensor_device(row)
            data_direct_save_ms += (time.perf_counter() - start) * 1000.0
            data_row_bytes += int(row.numel())
            data_row_count += 1
            stored_rows.append(row)

        parity_rows: list[torch.Tensor] = []
        for parity_id in range(self.config.m):
            row_index = self.config.k + parity_id
            dst = routing.storage_device_for_row(self.config, row_index)
            touched_devices.add(dst)
            parity = self._empty_storage_row(reduction_group_bytes, dst)
            parity_rows.append(parity)
            parity_row_bytes += int(parity.numel())
            parity_row_count += 1

        spare_alloc_start = time.perf_counter()
        staging_len = min(int(self.config.buffer_size), int(reduction_group_bytes))
        staging: dict[torch.device, dict[str, Any]] = {}
        for device, parity_ids in parity_devices.items():
            inputs = [torch.empty(staging_len, dtype=torch.uint8, device=device) for _ in range(self.config.k)]
            outputs = [torch.empty(staging_len, dtype=torch.uint8, device=device) for _ in parity_ids]
            staging[device] = {"parity_ids": parity_ids, "inputs": inputs, "outputs": outputs}
        synchronize_devices(set(staging))
        spare_buffer_alloc_ms += (time.perf_counter() - spare_alloc_start) * 1000.0

        parity_matrix = [row[:] for row in self.matrix[self.config.k :]]
        for offset, end in self._chunk_ranges(reduction_group_bytes):
            chunk_len = int(end - offset)
            copied_devices: set[torch.device] = set()
            pack_start = time.perf_counter()
            for device, buffers in staging.items():
                input_slices = [buf.narrow(0, 0, chunk_len) for buf in buffers["inputs"]]
                for col, rank in enumerate(data_ranks):
                    block = input_slices[col]
                    if rank is None:
                        block.zero_()
                        continue
                    flat = flats[int(rank)]
                    if offset < int(flat.numel()):
                        copy_len = min(chunk_len, int(flat.numel()) - offset)
                        block.narrow(0, 0, copy_len).copy_(
                            flat.narrow(0, offset, copy_len),
                            non_blocking=True,
                        )
                        if copy_len < chunk_len:
                            block.narrow(0, copy_len, chunk_len - copy_len).zero_()
                    else:
                        block.zero_()
                copied_devices.add(device)
            synchronize_devices(copied_devices)
            reduction_group_pack_ms += (time.perf_counter() - pack_start) * 1000.0

            ec_start = time.perf_counter()
            parity_chunks_by_id: dict[int, torch.Tensor] = {}
            for device, buffers in staging.items():
                parity_ids = buffers["parity_ids"]
                input_slices = [buf.narrow(0, 0, chunk_len) for buf in buffers["inputs"]]
                output_slices = [buf.narrow(0, 0, chunk_len) for buf in buffers["outputs"]]
                coeff_rows = [parity_matrix[parity_id] for parity_id in parity_ids]
                encoded = self._codec_apply_matrix(input_slices, coeff_rows, outputs=output_slices)
                for parity_id, chunk in zip(parity_ids, encoded):
                    parity_chunks_by_id[int(parity_id)] = chunk
                self._sync_compute_device(device)
            ec_encode_ms += (time.perf_counter() - ec_start) * 1000.0

            save_start = time.perf_counter()
            for parity_id in range(self.config.m):
                chunk = parity_chunks_by_id[parity_id]
                dst_slice = parity_rows[parity_id].narrow(0, offset, chunk_len)
                dst_slice.copy_(chunk[:chunk_len], non_blocking=True)
                self._sync_tensor_device(dst_slice)
            parity_chunk_save_ms += (time.perf_counter() - save_start) * 1000.0

        stored_rows.extend(parity_rows)
        return (
            StoredReductionGroup(
                index=reduction_group_index,
                rows=stored_rows,
                data_ranks=data_ranks,
                reduction_group_bytes=reduction_group_bytes,
                shapes=shapes,
                numels=numels,
            ),
            {
                "reduction_group_pack_ms": reduction_group_pack_ms,
                "ec_encode_ms": ec_encode_ms,
                "storage_device_copy_ms": data_direct_save_ms + parity_chunk_save_ms,
                "data_direct_save_ms": data_direct_save_ms,
                "parity_chunk_save_ms": parity_chunk_save_ms,
                "spare_buffer_alloc_ms": spare_buffer_alloc_ms,
                "data_row_bytes": data_row_bytes,
                "parity_row_bytes": parity_row_bytes,
                "data_row_count": data_row_count,
                "parity_row_count": parity_row_count,
            },
            touched_devices,
        )

    def store(
        self,
        obj: object,
        tag: str | None = None,
        async_op: bool = False,
    ) -> StoreHandle:
        actual_tag = tag if tag is not None else self._next_tag()
        if async_op:
            handle = StoreHandle(actual_tag, self, set(), True)

            def run_store() -> None:
                try:
                    completed = self.store(obj, tag=actual_tag, async_op=False)
                    handle.devices = set(completed.devices)
                    handle.stats = dict(completed.stats or {})
                except BaseException as exc:
                    handle._error = exc

            worker = threading.Thread(
                target=run_store,
                name=f"racer-store-{actual_tag}",
                daemon=True,
            )
            handle._worker = worker
            worker.start()
            return handle

        total_start = time.perf_counter()
        prepare_start = time.perf_counter()
        tensor_map, payload_kind, rank_state_metadata = self._prepare_obj(obj)
        flatten_ms = (time.perf_counter() - prepare_start) * 1000.0
        actual_async = bool(async_op)
        dropped_existing, overwrite_cleanup_ms = self._drop_existing_checkpoint(actual_tag)
        compute_device = routing.compute_device(self.config)

        stored_reduction_groups: list[StoredReductionGroup] = []
        touched_devices = {compute_device}
        max_reduction_group_bytes = 0
        reduction_group_pack_ms = 0.0
        ec_encode_ms = 0.0
        storage_device_copy_ms = 0.0
        storage_device_copy_sync_ms = 0.0
        data_direct_save_ms = 0.0
        parity_chunk_save_ms = 0.0
        spare_buffer_alloc_ms = 0.0
        data_row_bytes = 0
        parity_row_bytes = 0
        data_row_count = 0
        parity_row_count = 0

        for reduction_group_index, reduction_group in enumerate(self.elastic_layout.reduction_groups):
            stored_reduction_group, reduction_group_profile, reduction_group_devices = self._store_reduction_group_chunked_parity(
                tensor_map,
                reduction_group_index,
                reduction_group,
                compute_device,
            )
            touched_devices |= reduction_group_devices
            max_reduction_group_bytes = max(max_reduction_group_bytes, int(stored_reduction_group.reduction_group_bytes))
            reduction_group_pack_ms += float(reduction_group_profile["reduction_group_pack_ms"])
            ec_encode_ms += float(reduction_group_profile["ec_encode_ms"])
            storage_device_copy_ms += float(reduction_group_profile["storage_device_copy_ms"])
            data_direct_save_ms += float(reduction_group_profile.get("data_direct_save_ms", 0.0))
            parity_chunk_save_ms += float(reduction_group_profile.get("parity_chunk_save_ms", 0.0))
            spare_buffer_alloc_ms += float(reduction_group_profile.get("spare_buffer_alloc_ms", 0.0))
            data_row_bytes += int(reduction_group_profile["data_row_bytes"])
            parity_row_bytes += int(reduction_group_profile["parity_row_bytes"])
            data_row_count += int(reduction_group_profile["data_row_count"])
            parity_row_count += int(reduction_group_profile["parity_row_count"])
            stored_reduction_groups.append(stored_reduction_group)

        encode_ms = reduction_group_pack_ms + ec_encode_ms + storage_device_copy_ms
        checkpoint = StoredCheckpoint(
            tag=actual_tag,
            reduction_groups=stored_reduction_groups,
            matrix=[row[:] for row in self.matrix],
            metadata={},
        )
        metadata_start = time.perf_counter()
        plan = routing.make_planner(self.config).plan(self.elastic_layout, self.matrix, max_reduction_group_bytes)
        self.last_routing_plan = plan
        manifest = self._build_manifest(actual_tag, stored_reduction_groups, plan)
        manifest["payload_kind"] = payload_kind
        manifest["rank_state_metadata"] = {
            str(rank): self._metadata_to_manifest(metadata)
            for rank, metadata in rank_state_metadata.items()
        }
        checkpoint.metadata = manifest
        metadata_ms = (time.perf_counter() - metadata_start) * 1000.0
        storage_write_start = time.perf_counter()
        chunk_write_profile = self._write_chunks_and_manifest(actual_tag, stored_reduction_groups, manifest)
        storage_write_ms = (time.perf_counter() - storage_write_start) * 1000.0
        checkpoint_index_start = time.perf_counter()
        if self._local_checkpoint_index_enabled and self.storage is not None:
            self.storage.put(checkpoint)
        checkpoint_index_ms = (time.perf_counter() - checkpoint_index_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        bytes_total = sum(int(tensor.numel()) for tensor in tensor_map.values())
        self.last_store_profile = {
            "tag": actual_tag,
            "payload_kind": payload_kind,
            "dropped_existing_checkpoint": dropped_existing,
            "overwrite_cleanup_ms": overwrite_cleanup_ms,
            "flatten_ms": flatten_ms,
            "reduction_group_pack_ms": reduction_group_pack_ms,
            "ec_encode_ms": ec_encode_ms,
            "storage_device_copy_ms": storage_device_copy_ms,
            "storage_device_copy_sync_ms": storage_device_copy_sync_ms,
            "data_direct_save_ms": data_direct_save_ms,
            "parity_chunk_save_ms": parity_chunk_save_ms,
            "spare_buffer_alloc_ms": spare_buffer_alloc_ms,
            "encode_ms": encode_ms,
            "manifest_build_ms": metadata_ms,
            "metadata_ms": metadata_ms,
            "chunk_storage_write_ms": storage_write_ms,
            "storage_write_ms": storage_write_ms,
            "checkpoint_index_ms": checkpoint_index_ms,
            "total_ms": total_ms,
            "bytes_total": bytes_total,
            "data_row_bytes": data_row_bytes,
            "parity_row_bytes": parity_row_bytes,
            "data_row_count": data_row_count,
            "parity_row_count": parity_row_count,
            **chunk_write_profile,
        }
        handle = StoreHandle(
            actual_tag,
            self,
            touched_devices | (checkpoint.devices() if self._local_checkpoint_index_enabled else set()),
            actual_async,
            stats=dict(self.last_store_profile),
        )
        if not actual_async:
            handle.wait()
        return handle

    def _failed_owner_rows(self, failed_ranks: set[int]) -> set[int]:
        return {
            row
            for row, owner in enumerate(self.config.train_ranks)
            if int(owner) in failed_ranks
        }

    def _validate_replacement_mapping(
        self,
        replacement_mapping: Mapping[int, int] | None,
        failed_ranks: set[int],
    ) -> dict[int, int]:
        mapping = {int(k): int(v) for k, v in dict(replacement_mapping or {}).items()}
        unknown = set(mapping) - failed_ranks
        if unknown:
            raise ValueError(f"replacement_mapping contains non-failed train ranks: {sorted(unknown)}")
        if torch.cuda.is_available():
            for failed_rank, replacement in mapping.items():
                if replacement < 0 or replacement >= torch.cuda.device_count():
                    raise ValueError(
                        f"replacement rank/device {replacement} for failed rank {failed_rank} "
                        f"is outside visible CUDA devices"
                    )
        return mapping

    def _output_device_for_rank(
        self,
        rank: int,
        failed: bool,
        replacement_mapping: Mapping[int, int],
    ) -> torch.device:
        if failed:
            replacement = replacement_mapping.get(int(rank), int(self.config.spare_ranks[0]))
            return torch.device("cuda", int(replacement))
        return torch.device("cuda", int(rank))

    def _normalize_survivor_rows(
        self,
        survivor_rows: list[int] | None,
        failed_rows: set[int],
    ) -> list[int]:
        if survivor_rows is None:
            rows = [row for row in range(self.n_train) if row not in failed_rows]
        else:
            rows = []
            for value in survivor_rows:
                v = int(value)
                if 0 <= v < self.n_train:
                    rows.append(v)
                elif v in self.train_rank_to_row:
                    rows.append(self.train_rank_to_row[v])
                else:
                    raise ValueError(f"survivor row/rank {value} is not valid")
            bad = set(rows) & failed_rows
            if bad:
                raise ValueError(f"survivor_rows includes failed rows: {sorted(bad)}")
        if len(rows) < self.config.k:
            raise RuntimeError(
                f"not enough survivor rows to decode: have {len(rows)}, need k={self.config.k}"
            )
        return rows

    def _decode_reduction_group(
        self,
        checkpoint: StoredCheckpoint,
        reduction_group: StoredReductionGroup,
        survivor_rows: list[int],
        compute_device: torch.device,
        needed_cols: Sequence[int],
    ) -> tuple[dict[int, torch.Tensor], dict[str, float]]:
        chosen = survivor_rows[: self.config.k]
        cols = sorted({int(col) for col in needed_cols})
        if not cols:
            return {}, {"survivor_to_compute_ms": 0.0, "decode_matrix_ms": 0.0, "ec_decode_ms": 0.0}

        matrix_start = time.perf_counter()
        selected = gf256.select_rows(checkpoint.matrix, chosen)
        inverse = gf256.invert_matrix(selected)
        coeff_rows = [inverse[col] for col in cols]
        decode_matrix_ms = (time.perf_counter() - matrix_start) * 1000.0

        reduction_group_bytes = int(reduction_group.reduction_group_bytes)
        decoded_by_col = {
            col: torch.empty(reduction_group_bytes, dtype=torch.uint8, device=compute_device)
            for col in cols
        }
        survivor_to_compute_ms = 0.0
        ec_decode_ms = 0.0

        for offset, end in self._chunk_ranges(reduction_group_bytes):
            chunk_len = int(end - offset)
            survivor_copy_start = time.perf_counter()
            blocks = []
            for row in chosen:
                if row < self.config.k and reduction_group.data_ranks[row] is None:
                    block = torch.zeros(chunk_len, dtype=torch.uint8, device=compute_device)
                else:
                    block = reduction_group.rows[row].contiguous().view(-1).narrow(0, offset, chunk_len).to(
                        compute_device,
                        non_blocking=True,
                    )
                blocks.append(block)
            self._sync_compute_device(compute_device)
            survivor_to_compute_ms += (time.perf_counter() - survivor_copy_start) * 1000.0

            ec_start = time.perf_counter()
            decoded_chunks = self._codec_encode(blocks, coeff_rows)
            self._sync_compute_device(compute_device)
            ec_decode_ms += (time.perf_counter() - ec_start) * 1000.0

            for col, chunk in zip(cols, decoded_chunks):
                decoded_by_col[col].narrow(0, offset, chunk_len).copy_(chunk[:chunk_len], non_blocking=True)
            del blocks, decoded_chunks

        self._sync_compute_device(compute_device)
        return decoded_by_col, {
            "survivor_to_compute_ms": survivor_to_compute_ms,
            "decode_matrix_ms": decode_matrix_ms,
            "ec_decode_ms": ec_decode_ms,
        }

    def load(
        self,
        tag: str | None = None,
        failed_train_ranks: list[int] | None = None,
        survivor_rows: list[int] | None = None,
        requested_train_ranks: list[int] | None = None,
        replacement_mapping: Mapping[int, int] | None = None,
    ) -> dict[int, Any]:
        total_start = time.perf_counter()
        storage_read_start = time.perf_counter()
        checkpoint = self._checkpoint_for_load(tag)
        storage_read_ms = (time.perf_counter() - storage_read_start) * 1000.0

        checkpoint_sync_start = time.perf_counter()
        synchronize_devices(checkpoint.devices())
        checkpoint_sync_ms = (time.perf_counter() - checkpoint_sync_start) * 1000.0

        metadata_start = time.perf_counter()
        failed_ranks = normalize_rank_set(failed_train_ranks)
        unknown_failed = failed_ranks - set(self.config.train_ranks)
        if unknown_failed:
            raise ValueError(f"failed_train_ranks contains non-train ranks: {sorted(unknown_failed)}")
        failed_owner_rows = self._failed_owner_rows(failed_ranks)
        replacement = self._validate_replacement_mapping(replacement_mapping, failed_ranks)
        checksum_profile = self._verify_checkpoint_checksums(checkpoint, failed_owner_rows)

        if requested_train_ranks is not None:
            requested = [int(rank) for rank in requested_train_ranks]
        elif failed_train_ranks is not None:
            requested = [int(rank) for rank in failed_train_ranks]
        else:
            requested = list(self.config.train_ranks)
        unknown_requested = set(requested) - set(self.config.train_ranks)
        if unknown_requested:
            raise ValueError(f"requested_train_ranks contains non-train ranks: {sorted(unknown_requested)}")
        metadata_ms = (time.perf_counter() - metadata_start) * 1000.0

        compute_device = routing.compute_device(self.config)
        raw_results: dict[int, torch.Tensor] = {}

        ranks_by_reduction_group: dict[int, list[tuple[int, int]]] = {}
        for rank in requested:
            slot = self.elastic_layout.locate_rank(int(rank))
            ranks_by_reduction_group.setdefault(slot.relative_index, []).append((int(rank), slot.data_group_id))

        survivor_to_compute_ms = 0.0
        decode_matrix_ms = 0.0
        ec_decode_ms = 0.0
        raw_payload_to_output_device_ms = 0.0
        raw_payload_to_output_device_sync_ms = 0.0

        decode_start = time.perf_counter()
        for reduction_group_idx, rank_cols in ranks_by_reduction_group.items():
            reduction_group = checkpoint.reduction_groups[reduction_group_idx]
            survivors = self._normalize_survivor_rows(survivor_rows, failed_owner_rows)
            decode_cols = sorted(
                {
                    col
                    for _, col in rank_cols
                    if not (survivor_rows is None and col not in failed_owner_rows)
                }
            )
            decoded_by_col: dict[int, torch.Tensor] = {}
            if decode_cols:
                decoded_by_col, decode_profile = self._decode_reduction_group(
                    checkpoint,
                    reduction_group,
                    survivors,
                    compute_device,
                    decode_cols,
                )
                survivor_to_compute_ms += decode_profile["survivor_to_compute_ms"]
                decode_matrix_ms += decode_profile["decode_matrix_ms"]
                ec_decode_ms += decode_profile["ec_decode_ms"]

            for rank, col in rank_cols:
                direct_allowed = survivor_rows is None and col not in failed_owner_rows
                if direct_allowed:
                    source = reduction_group.rows[col].contiguous().view(-1)
                    source_is_direct = True
                else:
                    source = decoded_by_col[col].contiguous().view(-1)
                    source_is_direct = False

                copy_start = time.perf_counter()
                numel = reduction_group.numels[rank]
                shape = reduction_group.shapes[rank]
                failed = rank in failed_ranks
                dst = self._output_device_for_rank(rank, failed, replacement)
                if source.device.type == "cuda":
                    self._sync_compute_device(source.device)
                flat = source[:numel]
                if flat.device == dst:
                    if source_is_direct:
                        # Preserve load isolation when returning a directly stored row on the same device.
                        recovered = flat.clone().view(shape)
                    else:
                        recovered = flat.view(shape)
                else:
                    recovered = flat.to(dst, non_blocking=False).view(shape)
                    if recovered.device.type == "cuda":
                        sync_start = time.perf_counter()
                        self._sync_compute_device(recovered.device)
                        raw_payload_to_output_device_sync_ms += (time.perf_counter() - sync_start) * 1000.0
                raw_payload_to_output_device_ms += (time.perf_counter() - copy_start) * 1000.0
                raw_results[rank] = recovered

        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        unflatten_start = time.perf_counter()
        if checkpoint.metadata.get("payload_kind", "raw_tensor") == "state_dict":
            results: dict[int, Any] = {
                rank: unflatten_state_dict(
                    self._rank_state_metadata(checkpoint, rank),
                    payload,
                    target_device=payload.device,
                )
                for rank, payload in raw_results.items()
            }
        else:
            results = raw_results
        unflatten_ms = (time.perf_counter() - unflatten_start) * 1000.0

        final_sync_start = time.perf_counter()
        synchronize_devices(self._collect_devices(list(results.values())))
        final_sync_ms = (time.perf_counter() - final_sync_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        self.last_load_profile = {
            "tag": checkpoint.tag,
            "payload_kind": checkpoint.metadata.get("payload_kind", "raw_tensor"),
            "storage_read_ms": storage_read_ms,
            "checkpoint_sync_ms": checkpoint_sync_ms,
            "metadata_ms": metadata_ms,
            **checksum_profile,
            "survivor_to_compute_ms": survivor_to_compute_ms,
            "decode_matrix_ms": decode_matrix_ms,
            "ec_decode_ms": ec_decode_ms,
            "raw_payload_to_output_device_ms": raw_payload_to_output_device_ms,
            "raw_payload_to_output_device_sync_ms": raw_payload_to_output_device_sync_ms,
            "decode_ms": decode_ms,
            "unflatten_ms": unflatten_ms,
            "final_sync_ms": final_sync_ms,
            "total_ms": total_ms,
            "bytes_total": sum(int(tensor.numel()) for tensor in raw_results.values()),
            "requested_train_ranks": requested,
            "failed_train_ranks": sorted(failed_ranks),
            "failed_owner_rows": sorted(failed_owner_rows),
            "replacement_mapping": dict(replacement),
        }
        return results

    def repair(
        self,
        tag: str | None = None,
        failed_train_ranks: list[int] | None = None,
        replacement_mapping: Mapping[int, int] | None = None,
        async_op: bool = False,
    ) -> RepairHandle:
        if async_op:
            handle = RepairHandle(
                tag="" if tag is None else str(tag),
                repaired_rows={},
                replacement_mapping={},
                stats={},
                async_op=True,
            )

            def run_repair() -> None:
                try:
                    completed = self.repair(
                        tag=tag,
                        failed_train_ranks=failed_train_ranks,
                        replacement_mapping=replacement_mapping,
                        async_op=False,
                    )
                    handle.tag = completed.tag
                    handle.repaired_rows = dict(completed.repaired_rows)
                    handle.replacement_mapping = dict(completed.replacement_mapping)
                    handle.stats = dict(completed.stats)
                except BaseException as exc:
                    handle._error = exc

            worker = threading.Thread(
                target=run_repair,
                name=f"racer-repair-{handle.tag or 'latest'}",
                daemon=True,
            )
            handle._worker = worker
            worker.start()
            return handle

        total_start = time.perf_counter()
        checkpoint = self._checkpoint_for_load(tag)

        failed_ranks = normalize_rank_set(failed_train_ranks)
        unknown_failed = failed_ranks - set(self.config.train_ranks)
        if unknown_failed:
            raise ValueError(f"failed_train_ranks contains non-train ranks: {sorted(unknown_failed)}")
        if not failed_ranks:
            return RepairHandle(
                tag=checkpoint.tag,
                repaired_rows={},
                replacement_mapping={},
                stats={"repair_ms": 0.0, "repaired_row_count": 0},
            )
        failed_owner_rows = self._failed_owner_rows(failed_ranks)
        replacement = self._validate_replacement_mapping(replacement_mapping, failed_ranks)
        checksum_profile = self._verify_checkpoint_checksums(checkpoint, failed_owner_rows)
        compute_device = routing.compute_device(self.config)

        manifest = checkpoint.metadata
        chunk_by_key = {
            (int(chunk["reduction_group_index"]), int(chunk["row"])): chunk
            for chunk in manifest.get("chunks", [])
        }
        repaired_rows: dict[str, int] = {}
        decode_matrix_ms = 0.0
        repair_decode_ms = 0.0
        repair_encode_ms = 0.0
        repair_copy_ms = 0.0

        for reduction_group in checkpoint.reduction_groups:
            survivors = self._normalize_survivor_rows(None, failed_owner_rows)
            missing_data_cols = [
                row
                for row in sorted(failed_owner_rows)
                if row < self.config.k and reduction_group.data_ranks[row] is not None
            ]
            decoded_by_col: dict[int, torch.Tensor] = {}
            if missing_data_cols:
                decoded_by_col, decode_profile = self._decode_reduction_group(
                    checkpoint,
                    reduction_group,
                    survivors,
                    compute_device,
                    missing_data_cols,
                )
                decode_matrix_ms += float(decode_profile["decode_matrix_ms"])
                repair_decode_ms += float(decode_profile["ec_decode_ms"]) + float(
                    decode_profile["survivor_to_compute_ms"]
                )

            data_cache: dict[int, torch.Tensor] = {}

            def data_block(col: int) -> torch.Tensor:
                col = int(col)
                if col in data_cache:
                    return data_cache[col]
                group_bytes = int(reduction_group.reduction_group_bytes)
                if reduction_group.data_ranks[col] is None:
                    block = torch.zeros(group_bytes, dtype=torch.uint8, device=compute_device)
                elif col in decoded_by_col:
                    block = decoded_by_col[col].contiguous().view(-1)
                    if block.device != compute_device:
                        block = block.to(compute_device, non_blocking=True)
                else:
                    block = reduction_group.rows[col].contiguous().view(-1).to(
                        compute_device,
                        non_blocking=True,
                    )
                data_cache[col] = block
                return block

            for row in sorted(failed_owner_rows):
                logical_owner = int(self.config.train_ranks[row])
                replacement_rank = int(replacement.get(logical_owner, self.config.spare_ranks[0]))
                dst = torch.device("cuda", replacement_rank)
                start = time.perf_counter()
                if row < self.config.k:
                    if reduction_group.data_ranks[row] is None:
                        repaired = torch.empty(0, dtype=torch.uint8, device=dst)
                    else:
                        repaired = data_block(row).to(dst, non_blocking=True).contiguous()
                else:
                    encode_start = time.perf_counter()
                    inputs = [data_block(col) for col in range(self.config.k)]
                    repaired = self._codec_apply_matrix(inputs, [checkpoint.matrix[row]])[0]
                    self._sync_compute_device(compute_device)
                    repair_encode_ms += (time.perf_counter() - encode_start) * 1000.0
                    repaired = repaired.to(dst, non_blocking=True).contiguous()
                self._sync_compute_device(dst)
                repair_copy_ms += (time.perf_counter() - start) * 1000.0
                reduction_group.rows[row] = repaired

                chunk_id = self._chunk_id(reduction_group.index, row)
                chunk = chunk_by_key.get((reduction_group.index, row))
                if chunk is not None:
                    chunk["logical_owner_rank"] = logical_owner
                    chunk["owner_rank"] = replacement_rank
                    chunk["repaired_from_failed_rank"] = logical_owner
                    chunk["stored_device"] = str(repaired.device)
                    chunk["num_bytes"] = int(repaired.numel())
                    chunk["checksum"] = self._checksum(repaired)
                    if self._chunk_storage_supports_cuda_ipc() and repaired.device.type == "cuda":
                        op_id = self.chunk_storage.put_cuda_tensor(checkpoint.tag, chunk_id, repaired, chunk)
                        self.chunk_storage.wait(op_id)
                    else:
                        raise RuntimeError(
                            "repair requires daemon storage with CUDA IPC async write support; "
                            "CPU/socket fallback put is disabled"
                        )
                repaired_rows[chunk_id] = replacement_rank

        chunks = manifest.get("chunks", [])
        if chunks:
            manifest["chunk_owner"] = {chunk["chunk_id"]: chunk["owner_rank"] for chunk in chunks}
            manifest["checksum"] = checksum.manifest_checksum(chunks)
            self.chunk_storage.put_manifest(checkpoint.tag, manifest)
        checkpoint.metadata = manifest
        if self._local_checkpoint_index_enabled and self.storage is not None:
            self.storage.put(checkpoint)

        stats = {
            **checksum_profile,
            "repair_ms": (time.perf_counter() - total_start) * 1000.0,
            "decode_matrix_ms": decode_matrix_ms,
            "repair_decode_ms": repair_decode_ms,
            "repair_encode_ms": repair_encode_ms,
            "repair_copy_ms": repair_copy_ms,
            "repaired_row_count": len(repaired_rows),
            "failed_train_ranks": sorted(failed_ranks),
            "failed_owner_rows": sorted(failed_owner_rows),
        }
        return RepairHandle(
            tag=checkpoint.tag,
            repaired_rows=repaired_rows,
            replacement_mapping=dict(replacement),
            stats=stats,
        )
