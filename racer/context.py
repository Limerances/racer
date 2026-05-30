"""RACER runtime context implementing store/load over synthetic packets.

The context owns the immutable RACER configuration, the train-rank-only E
matrix, elastic layout metadata, and configured storage backend. Phase 1 keeps
the API local to one process while preserving the intended distributed
semantics in manifests and routing plans.

TODO: add a Megatron adapter for sharded state_dict checkpoints.
TODO: move async storage and recovery work into a long-lived racerd daemon.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import time
from typing import Any, Sequence

import torch

from . import cauchy, codec_cuda, gf256, routing
from .config import RacerConfig
from .layout import ElasticLayout, RacerLayout, Stripe
from .state_dict_codec import RankStateMetadata, TensorMetadata, flatten_state_dict, unflatten_state_dict
from .storage import InProcessStorage, StoredCheckpoint, StoredStripe, create_storage_backend
from .utils import normalize_rank_set, require_uint8_tensor_map, synchronize_devices


@dataclass
class StoreHandle:
    tag: str
    context: "RacerContext"
    devices: set[torch.device]
    async_op: bool
    stats: dict[str, Any] | None = None

    def wait(self) -> "StoreHandle":
        synchronize_devices(self.devices)
        return self

    def done(self) -> bool:
        if not self.async_op:
            return True
        for device in self.devices:
            if device.type == "cuda":
                with torch.cuda.device(device):
                    if not torch.cuda.current_stream(device).query():
                        return False
        return True


class RacerContext:
    def __init__(self, config: RacerConfig) -> None:
        self.config = config
        self.matrix = cauchy.systematic_matrix(
            config.k,
            config.m,
            config.w,
            optimize=config.optimize_cauchy,
        )
        self.layout = RacerLayout.build(config.train_ranks, config.k)
        self.elastic_layout = ElasticLayout.build(config.train_ranks, config.spare_ranks, config.k, config.m)
        self.storage = InProcessStorage()
        self.chunk_storage = create_storage_backend(config.storage_backend)
        self.train_rank_to_row = {rank: row for row, rank in enumerate(config.train_ranks)}
        self.last_routing_plan: routing.RoutingPlan | None = None
        self.last_store_profile: dict[str, Any] = {}
        self.last_load_profile: dict[str, Any] = {}
        self._counter = 0

        if config.backend == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("backend='cuda' requested but torch.cuda is unavailable")
            max_rank = max(config.train_ranks + config.spare_ranks) if config.spare_ranks else max(config.train_ranks)
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
        return {
            "source_train_rank": int(metadata.source_train_rank),
            "payload_nbytes": int(metadata.payload_nbytes),
            "tensors": [
                {
                    "key": item.key,
                    "dtype": item.dtype,
                    "shape": list(item.shape),
                    "device": item.device,
                    "requires_grad": bool(item.requires_grad),
                    "was_contiguous": bool(item.was_contiguous),
                    "offset": int(item.offset),
                    "nbytes": int(item.nbytes),
                }
                for item in metadata.tensors
            ],
        }

    def _metadata_from_manifest(self, data: Mapping[str, Any]) -> RankStateMetadata:
        tensors = [
            TensorMetadata(
                key=str(item["key"]),
                dtype=str(item["dtype"]),
                shape=tuple(int(dim) for dim in item["shape"]),
                device=str(item["device"]),
                requires_grad=bool(item["requires_grad"]),
                was_contiguous=bool(item["was_contiguous"]),
                offset=int(item["offset"]),
                nbytes=int(item["nbytes"]),
            )
            for item in data.get("tensors", [])
        ]
        return RankStateMetadata(
            source_train_rank=int(data["source_train_rank"]),
            tensors=tensors,
            payload_nbytes=int(data["payload_nbytes"]),
        )

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

    def _host_buffer_owner_rank(self, stripe: Stripe, row: int) -> int | None:
        if row < self.config.k:
            rank = stripe.data_ranks[row]
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
        flat = tensor.detach().contiguous().view(-1)
        numel = int(flat.numel())
        if numel == 0:
            return "sample64-v1:0:0:0000000000000000:00:00"
        if flat.device.type == "cuda":
            self._sync_compute_device(flat.device)
        first = int(flat[0].item()) & 0xFF
        last = int(flat[-1].item()) & 0xFF
        full_sum_limit = 16 * 1024 * 1024
        if numel <= full_sum_limit:
            total = int(torch.sum(flat, dtype=torch.int64).item()) & 0xFFFFFFFFFFFFFFFF
            return f"sum64-v1:{numel}:{total:016x}:{first:02x}:{last:02x}"

        sample_count = 4096
        if sample_count == 1:
            indices = torch.zeros(1, dtype=torch.long, device=flat.device)
        else:
            indices = torch.arange(sample_count, dtype=torch.long, device=flat.device)
            indices = indices * (numel - 1) // (sample_count - 1)
        sample = flat.index_select(0, indices)
        total = int(torch.sum(sample, dtype=torch.int64).item()) & 0xFFFFFFFFFFFFFFFF
        return f"sample64-v1:{numel}:{sample_count}:{total:016x}:{first:02x}:{last:02x}"

    def _chunk_id(self, stripe_index: int, row: int) -> str:
        return f"stripe_{stripe_index:06d}_row_{row:03d}"

    def _build_manifest(
        self,
        tag: str,
        stored_stripes: list[StoredStripe],
        plan: routing.RoutingPlan,
    ) -> dict:
        chunks = []
        virtual_slots = []
        for group in self.elastic_layout.reduction_groups:
            for slot in group:
                if slot.is_virtual_zero:
                    virtual_slots.append(
                        {
                            "slot_id": slot.slot_id,
                            "data_group_id": slot.data_group_id,
                            "relative_index": slot.relative_index,
                            "train_rank": None,
                            "is_virtual_zero": True,
                            "valid_nbytes": slot.valid_nbytes,
                        }
                    )

        for stripe in stored_stripes:
            group = self.elastic_layout.reduction_groups[stripe.index]
            slot_entries = [
                {
                    "slot_id": slot.slot_id,
                    "data_group_id": slot.data_group_id,
                    "relative_index": slot.relative_index,
                    "train_rank": slot.train_rank,
                    "is_virtual_zero": slot.is_virtual_zero,
                    "valid_nbytes": stripe.numels.get(slot.train_rank, 0) if slot.train_rank is not None else 0,
                    "shape": list(stripe.shapes.get(slot.train_rank, ())) if slot.train_rank is not None else [],
                }
                for slot in group
            ]
            for row, tensor in enumerate(stripe.rows):
                owner = int(self.config.train_ranks[row])
                chunk_id = self._chunk_id(stripe.index, row)
                host_owner = self._host_buffer_owner_rank(stripe, row)
                chunk = {
                    "chunk_id": chunk_id,
                    "stripe_index": stripe.index,
                    "row": row,
                    "chunk_role": "data" if row < self.config.k else "parity",
                    "parity_id": None if row < self.config.k else row - self.config.k,
                    "owner_rank": owner,
                    "host_buffer_owner_rank": host_owner,
                    "host_buffer_role": "train_local" if row < self.config.k else "spare_local",
                    "is_spare_owned": False,
                    "stored_device": str(tensor.device),
                    "is_pinned_host": False,
                    "num_bytes": int(tensor.numel()),
                    "checksum": self._checksum(tensor),
                    "slots": slot_entries,
                }
                chunks.append(chunk)

        return {
            "tag": tag,
            "version": 1,
            "k": self.config.k,
            "m": self.config.m,
            "n_train": self.n_train,
            "E": [row[:] for row in self.matrix],
            "train_ranks": list(self.config.train_ranks),
            "spare_ranks": list(self.config.spare_ranks),
            "storage_backend": self.config.storage_backend,
            "routing_strategy": self.config.routing_strategy,
            "routing_plan": asdict(plan),
            "routing_cost": asdict(plan.cost),
            "elastic_layout": {
                "pad_mode": self.elastic_layout.pad_mode,
                "q": self.elastic_layout.q,
                "virtual_W": self.elastic_layout.virtual_W,
                "num_virtual_zero": self.elastic_layout.num_virtual_zero,
            },
            "virtual_slots": virtual_slots,
            "chunks": chunks,
            "chunk_owner": {chunk["chunk_id"]: chunk["owner_rank"] for chunk in chunks},
            "checksum": hashlib.sha256("".join(chunk["checksum"] for chunk in chunks).encode()).hexdigest(),
        }

    def _write_chunks_and_manifest(self, tag: str, stored_stripes: list[StoredStripe], manifest: dict) -> dict[str, Any]:
        by_id = {chunk["chunk_id"]: chunk for chunk in manifest["chunks"]}
        metrics: dict[str, Any] = {
            "data_chunk_write_ms": 0.0,
            "parity_chunk_write_ms": 0.0,
            "manifest_write_ms": 0.0,
            "data_chunk_bytes": 0,
            "parity_chunk_bytes": 0,
            "data_chunk_count": 0,
            "parity_chunk_count": 0,
        }
        for stripe in stored_stripes:
            for row, tensor in enumerate(stripe.rows):
                chunk_id = self._chunk_id(stripe.index, row)
                role = "data" if row < self.config.k else "parity"
                start = time.perf_counter()
                self.chunk_storage.put(tag, chunk_id, tensor, by_id[chunk_id])
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                metrics[f"{role}_chunk_write_ms"] += elapsed_ms
                metrics[f"{role}_chunk_bytes"] += int(tensor.numel())
                metrics[f"{role}_chunk_count"] += 1
        start = time.perf_counter()
        self.chunk_storage.put_manifest(tag, manifest)
        metrics["manifest_write_ms"] = (time.perf_counter() - start) * 1000.0
        return metrics

    def _checkpoint_from_chunk_storage(self, tag: str | None) -> StoredCheckpoint:
        if tag is None:
            raise KeyError("tag is required when loading from chunk storage")
        manifest = self.chunk_storage.get_manifest(tag)
        stripes: list[StoredStripe] = []
        chunks_by_stripe: dict[int, list[dict]] = {}
        for chunk in manifest["chunks"]:
            chunks_by_stripe.setdefault(int(chunk["stripe_index"]), []).append(chunk)
        for stripe_index in sorted(chunks_by_stripe):
            chunk_entries = sorted(chunks_by_stripe[stripe_index], key=lambda item: int(item["row"]))
            rows = [self.chunk_storage.get(tag, entry["chunk_id"]) for entry in chunk_entries]
            slots = chunk_entries[0]["slots"]
            data_ranks = tuple(slot["train_rank"] for slot in slots)
            shapes: dict[int, tuple[int, ...]] = {}
            numels: dict[int, int] = {}
            for slot in slots:
                rank = slot["train_rank"]
                if rank is not None:
                    shape = tuple(int(v) for v in slot.get("shape", []))
                    shapes[int(rank)] = shape if shape else (int(slot["valid_nbytes"]),)
                    numels[int(rank)] = int(slot["valid_nbytes"])
            stripes.append(
                StoredStripe(
                    index=stripe_index,
                    rows=rows,
                    data_ranks=data_ranks,
                    stripe_bytes=max((int(row.numel()) for row in rows), default=0),
                    shapes=shapes,
                    numels=numels,
                )
            )
        return StoredCheckpoint(tag=tag, stripes=stripes, matrix=manifest["E"], metadata=manifest)

    def _stripe_data_blocks(
        self,
        obj: Mapping[int, torch.Tensor],
        stripe: Stripe,
        compute_device: torch.device,
    ) -> tuple[list[torch.Tensor], int, dict[int, tuple[int, ...]], dict[int, int]]:
        shapes: dict[int, tuple[int, ...]] = {}
        numels: dict[int, int] = {}
        stripe_bytes = 0
        for rank in stripe.data_ranks:
            if rank is None:
                continue
            tensor = obj[rank]
            shapes[rank] = tuple(tensor.shape)
            numels[rank] = tensor.numel()
            stripe_bytes = max(stripe_bytes, tensor.numel())
        if stripe_bytes == 0:
            stripe_bytes = 1

        blocks: list[torch.Tensor] = []
        for rank in stripe.data_ranks:
            block = torch.zeros(stripe_bytes, dtype=torch.uint8, device=compute_device)
            if rank is not None:
                flat = obj[rank].contiguous().view(-1).to(compute_device, non_blocking=True)
                block[: flat.numel()].copy_(flat, non_blocking=True)
            blocks.append(block)
        return blocks, stripe_bytes, shapes, numels


    def _chunk_ranges(self, total: int) -> list[tuple[int, int]]:
        chunk_size = max(1, int(self.config.buffer_size))
        return [(start, min(start + chunk_size, total)) for start in range(0, int(total), chunk_size)]

    def _store_stripe_chunked_parity(
        self,
        obj: Mapping[int, torch.Tensor],
        stripe: Stripe,
        compute_device: torch.device,
    ) -> tuple[StoredStripe, dict[str, Any], set[torch.device]]:
        shapes: dict[int, tuple[int, ...]] = {}
        numels: dict[int, int] = {}
        stripe_bytes = 0
        flats: dict[int, torch.Tensor] = {}
        for rank in stripe.data_ranks:
            if rank is None:
                continue
            flat = obj[rank].contiguous().view(-1)
            flats[int(rank)] = flat
            shapes[int(rank)] = tuple(obj[rank].shape)
            numels[int(rank)] = int(flat.numel())
            stripe_bytes = max(stripe_bytes, int(flat.numel()))
        if stripe_bytes == 0:
            stripe_bytes = 1

        parity_devices: dict[torch.device, list[int]] = {}
        for parity_id in range(self.config.m):
            parity_devices.setdefault(self._parity_compute_device(parity_id), []).append(parity_id)

        touched_devices = set(parity_devices) or {compute_device}
        stored_rows: list[torch.Tensor] = []
        data_direct_save_ms = 0.0
        parity_chunk_save_ms = 0.0
        spare_buffer_alloc_ms = 0.0
        stripe_pack_ms = 0.0
        ec_encode_ms = 0.0
        data_row_bytes = 0
        parity_row_bytes = 0
        data_row_count = 0
        parity_row_count = 0

        for col, rank in enumerate(stripe.data_ranks):
            dst = routing.storage_device_for_row(self.config, col)
            touched_devices.add(dst)
            start = time.perf_counter()
            if rank is None:
                row = torch.empty(0, dtype=torch.uint8, device=dst)
            else:
                flat = flats[int(rank)]
                row = self._empty_storage_row(stripe_bytes, dst)
                copy_len = int(flat.numel())
                if copy_len:
                    row.narrow(0, 0, copy_len).copy_(flat, non_blocking=True)
                    self._sync_tensor_device(row)
                if copy_len < stripe_bytes:
                    row.narrow(0, copy_len, stripe_bytes - copy_len).zero_()
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
            parity = self._empty_storage_row(stripe_bytes, dst)
            parity_rows.append(parity)
            parity_row_bytes += int(parity.numel())
            parity_row_count += 1

        spare_alloc_start = time.perf_counter()
        staging_len = min(int(self.config.buffer_size), int(stripe_bytes))
        staging: dict[torch.device, dict[str, Any]] = {}
        for device, parity_ids in parity_devices.items():
            inputs = [torch.empty(staging_len, dtype=torch.uint8, device=device) for _ in range(self.config.k)]
            outputs = [torch.empty(staging_len, dtype=torch.uint8, device=device) for _ in parity_ids]
            staging[device] = {"parity_ids": parity_ids, "inputs": inputs, "outputs": outputs}
        synchronize_devices(set(staging))
        spare_buffer_alloc_ms += (time.perf_counter() - spare_alloc_start) * 1000.0

        parity_matrix = [row[:] for row in self.matrix[self.config.k :]]
        for offset, end in self._chunk_ranges(stripe_bytes):
            chunk_len = int(end - offset)
            copied_devices: set[torch.device] = set()
            pack_start = time.perf_counter()
            for device, buffers in staging.items():
                input_slices = [buf.narrow(0, 0, chunk_len) for buf in buffers["inputs"]]
                for col, rank in enumerate(stripe.data_ranks):
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
            stripe_pack_ms += (time.perf_counter() - pack_start) * 1000.0

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
            StoredStripe(
                index=stripe.index,
                rows=stored_rows,
                data_ranks=stripe.data_ranks,
                stripe_bytes=stripe_bytes,
                shapes=shapes,
                numels=numels,
            ),
            {
                "stripe_pack_ms": stripe_pack_ms,
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
        async_op: bool | None = None,
    ) -> StoreHandle:
        total_start = time.perf_counter()
        prepare_start = time.perf_counter()
        tensor_map, payload_kind, rank_state_metadata = self._prepare_obj(obj)
        flatten_ms = (time.perf_counter() - prepare_start) * 1000.0
        actual_tag = tag if tag is not None else self._next_tag()
        actual_async = self.config.async_op if async_op is None else bool(async_op)
        compute_device = routing.compute_device(self.config)

        stored_stripes: list[StoredStripe] = []
        touched_devices = {compute_device}
        max_stripe_bytes = 0
        stripe_pack_ms = 0.0
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

        for stripe in self.layout.stripes:
            stored_stripe, stripe_profile, stripe_devices = self._store_stripe_chunked_parity(
                tensor_map,
                stripe,
                compute_device,
            )
            touched_devices |= stripe_devices
            max_stripe_bytes = max(max_stripe_bytes, int(stored_stripe.stripe_bytes))
            stripe_pack_ms += float(stripe_profile["stripe_pack_ms"])
            ec_encode_ms += float(stripe_profile["ec_encode_ms"])
            storage_device_copy_ms += float(stripe_profile["storage_device_copy_ms"])
            data_direct_save_ms += float(stripe_profile.get("data_direct_save_ms", 0.0))
            parity_chunk_save_ms += float(stripe_profile.get("parity_chunk_save_ms", 0.0))
            spare_buffer_alloc_ms += float(stripe_profile.get("spare_buffer_alloc_ms", 0.0))
            data_row_bytes += int(stripe_profile["data_row_bytes"])
            parity_row_bytes += int(stripe_profile["parity_row_bytes"])
            data_row_count += int(stripe_profile["data_row_count"])
            parity_row_count += int(stripe_profile["parity_row_count"])
            stored_stripes.append(stored_stripe)

        encode_ms = stripe_pack_ms + ec_encode_ms + storage_device_copy_ms
        checkpoint = StoredCheckpoint(
            tag=actual_tag,
            stripes=stored_stripes,
            matrix=[row[:] for row in self.matrix],
            metadata={},
        )
        metadata_start = time.perf_counter()
        plan = routing.make_planner(self.config).plan(self.elastic_layout, self.matrix, max_stripe_bytes)
        self.last_routing_plan = plan
        manifest = self._build_manifest(actual_tag, stored_stripes, plan)
        manifest["payload_kind"] = payload_kind
        manifest["rank_state_metadata"] = {
            str(rank): self._metadata_to_manifest(metadata)
            for rank, metadata in rank_state_metadata.items()
        }
        checkpoint.metadata = manifest
        metadata_ms = (time.perf_counter() - metadata_start) * 1000.0
        storage_write_start = time.perf_counter()
        chunk_write_profile = self._write_chunks_and_manifest(actual_tag, stored_stripes, manifest)
        storage_write_ms = (time.perf_counter() - storage_write_start) * 1000.0
        checkpoint_index_start = time.perf_counter()
        self.storage.put(checkpoint)
        checkpoint_index_ms = (time.perf_counter() - checkpoint_index_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0
        bytes_total = sum(int(tensor.numel()) for tensor in tensor_map.values())
        self.last_store_profile = {
            "tag": actual_tag,
            "payload_kind": payload_kind,
            "flatten_ms": flatten_ms,
            "stripe_pack_ms": stripe_pack_ms,
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
            touched_devices | checkpoint.devices(),
            actual_async,
            stats=dict(self.last_store_profile),
        )
        if not actual_async:
            handle.wait()
        return handle

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

    def _decode_stripe(
        self,
        checkpoint: StoredCheckpoint,
        stripe: StoredStripe,
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

        stripe_bytes = int(stripe.stripe_bytes)
        decoded_by_col = {
            col: torch.empty(stripe_bytes, dtype=torch.uint8, device=compute_device)
            for col in cols
        }
        survivor_to_compute_ms = 0.0
        ec_decode_ms = 0.0

        for offset, end in self._chunk_ranges(stripe_bytes):
            chunk_len = int(end - offset)
            survivor_copy_start = time.perf_counter()
            blocks = []
            for row in chosen:
                if row < self.config.k and stripe.data_ranks[row] is None:
                    block = torch.zeros(chunk_len, dtype=torch.uint8, device=compute_device)
                else:
                    block = stripe.rows[row].contiguous().view(-1).narrow(0, offset, chunk_len).to(
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
    ) -> dict[int, Any]:
        total_start = time.perf_counter()
        storage_read_start = time.perf_counter()
        try:
            checkpoint = self.storage.get(tag)
        except KeyError:
            checkpoint = self._checkpoint_from_chunk_storage(tag)
        storage_read_ms = (time.perf_counter() - storage_read_start) * 1000.0

        checkpoint_sync_start = time.perf_counter()
        synchronize_devices(checkpoint.devices())
        checkpoint_sync_ms = (time.perf_counter() - checkpoint_sync_start) * 1000.0

        metadata_start = time.perf_counter()
        failed_ranks = normalize_rank_set(failed_train_ranks)
        unknown_failed = failed_ranks - set(self.config.train_ranks)
        if unknown_failed:
            raise ValueError(f"failed_train_ranks contains non-train ranks: {sorted(unknown_failed)}")
        failed_cols_by_stripe: dict[int, set[int]] = {}
        for rank in failed_ranks:
            stripe_idx, col = self.layout.locate_rank(int(rank))
            failed_cols_by_stripe.setdefault(stripe_idx, set()).add(col)

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

        ranks_by_stripe: dict[int, list[tuple[int, int]]] = {}
        for rank in requested:
            stripe_idx, col = self.layout.locate_rank(int(rank))
            ranks_by_stripe.setdefault(stripe_idx, []).append((int(rank), col))

        survivor_to_compute_ms = 0.0
        decode_matrix_ms = 0.0
        ec_decode_ms = 0.0
        raw_payload_to_output_device_ms = 0.0
        raw_payload_to_output_device_sync_ms = 0.0

        decode_start = time.perf_counter()
        for stripe_idx, rank_cols in ranks_by_stripe.items():
            stripe = checkpoint.stripes[stripe_idx]
            stripe_failed_cols = failed_cols_by_stripe.get(stripe_idx, set())
            survivors = self._normalize_survivor_rows(survivor_rows, stripe_failed_cols)
            decode_cols = sorted(
                {
                    col
                    for _, col in rank_cols
                    if not (survivor_rows is None and col not in stripe_failed_cols)
                }
            )
            decoded_by_col: dict[int, torch.Tensor] = {}
            if decode_cols:
                decoded_by_col, decode_profile = self._decode_stripe(
                    checkpoint,
                    stripe,
                    survivors,
                    compute_device,
                    decode_cols,
                )
                survivor_to_compute_ms += decode_profile["survivor_to_compute_ms"]
                decode_matrix_ms += decode_profile["decode_matrix_ms"]
                ec_decode_ms += decode_profile["ec_decode_ms"]

            for rank, col in rank_cols:
                direct_allowed = survivor_rows is None and col not in stripe_failed_cols
                if direct_allowed:
                    source = stripe.rows[col].contiguous().view(-1)
                    source_is_direct = True
                else:
                    source = decoded_by_col[col].contiguous().view(-1)
                    source_is_direct = False

                copy_start = time.perf_counter()
                numel = stripe.numels[rank]
                shape = stripe.shapes[rank]
                failed = rank in failed_ranks
                dst = routing.output_device_for_rank(self.config, rank, failed)
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
        }
        return results
