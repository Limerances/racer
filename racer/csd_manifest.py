"""SQLite manifest store for the RACER Checkpoint Storage Daemon."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any


CHECKPOINT_STATES = {"PREPARING", "WRITING", "COMMITTING", "COMMITTED", "ABORTED", "DELETING"}
CHUNK_STATES = {"RESERVED", "COPYING", "SEALED", "FAILED"}
OP_STATES = {"QUEUED", "RUNNING", "DONE", "FAILED"}


def now_ns() -> int:
    return int(time.time_ns())


def _json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _loads(raw: str | bytes | None, default: Any) -> Any:
    if raw in (None, ""):
        return default
    return json.loads(raw)


@dataclass(frozen=True)
class ChunkRecord:
    tag: str
    chunk_id: str
    row_id: int | None
    logical_owner_rank: int | None
    writer_rank: int | None
    backend: str
    location: dict[str, Any]
    offset: int
    nbytes: int
    valid_nbytes: int
    checksum_type: str
    checksum: str
    state: str


class CsdManifestStore:
    """Durable CSD manifest state machine backed by sqlite3.

    This records indexes and operation state. It does not make daemon-owned
    cudaHostAlloc bytes durable across daemon process death.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    tag TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    created_at_ns INTEGER NOT NULL,
                    committed_at_ns INTEGER,
                    backend TEXT NOT NULL,
                    k INTEGER,
                    m INTEGER,
                    train_ranks_json TEXT,
                    spare_ranks_json TEXT,
                    e_matrix_json_or_bytes BLOB,
                    layout_json TEXT,
                    tensor_specs_json TEXT,
                    manifest_json TEXT,
                    expected_chunks INTEGER NOT NULL DEFAULT 0,
                    total_valid_bytes INTEGER NOT NULL DEFAULT 0,
                    error TEXT
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    tag TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    row_id INTEGER,
                    logical_owner_rank INTEGER,
                    writer_rank INTEGER,
                    backend TEXT NOT NULL,
                    location_json TEXT,
                    offset INTEGER NOT NULL DEFAULT 0,
                    nbytes INTEGER NOT NULL DEFAULT 0,
                    valid_nbytes INTEGER NOT NULL DEFAULT 0,
                    checksum_type TEXT,
                    checksum TEXT,
                    state TEXT NOT NULL,
                    sealed_at_ns INTEGER,
                    PRIMARY KEY(tag, chunk_id)
                );
                CREATE TABLE IF NOT EXISTS operations (
                    op_id TEXT PRIMARY KEY,
                    tag TEXT NOT NULL,
                    chunk_id TEXT,
                    op_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    error TEXT,
                    start_ns INTEGER,
                    end_ns INTEGER
                );
                """
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def begin_checkpoint(
        self,
        tag: str,
        manifest_base: dict[str, Any],
        *,
        expected_chunks: int | None,
        backend: str,
    ) -> None:
        tag = str(tag)
        manifest = dict(manifest_base or {})
        chunks = list(manifest.get("chunks", []))
        expected = len(chunks) if expected_chunks is None else int(expected_chunks)
        train_ranks = manifest.get("train_ranks", [])
        spare_ranks = manifest.get("spare_ranks", [])
        layout = manifest.get("elastic_layout", {})
        with self._lock:
            row = self._conn.execute("SELECT state FROM checkpoints WHERE tag=?", (tag,)).fetchone()
            if row is not None and str(row[0]) == "COMMITTED":
                raise RuntimeError(f"CSD checkpoint {tag!r} is already committed")
            with self._conn:
                self._conn.execute("DELETE FROM chunks WHERE tag=?", (tag,))
                self._conn.execute("DELETE FROM operations WHERE tag=?", (tag,))
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoints(
                        tag, state, created_at_ns, committed_at_ns, backend, k, m,
                        train_ranks_json, spare_ranks_json, e_matrix_json_or_bytes,
                        layout_json, tensor_specs_json, manifest_json, expected_chunks,
                        total_valid_bytes, error
                    ) VALUES (?, 'PREPARING', ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)
                    """,
                    (
                        tag,
                        now_ns(),
                        str(backend),
                        int(manifest.get("k", 0) or 0),
                        int(manifest.get("m", 0) or 0),
                        _json(train_ranks),
                        _json(spare_ranks),
                        _json(manifest.get("E", [])),
                        _json(layout),
                        _json(manifest.get("tensor_specs", manifest.get("tensors", []))),
                        _json(manifest),
                        expected,
                    ),
                )
                self._conn.execute("UPDATE checkpoints SET state='WRITING' WHERE tag=?", (tag,))

    def update_manifest(self, tag: str, manifest: dict[str, Any]) -> None:
        tag = str(tag)
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE checkpoints SET manifest_json=?, k=?, m=?, train_ranks_json=?, spare_ranks_json=?, "
                    "e_matrix_json_or_bytes=?, layout_json=? WHERE tag=?",
                    (
                        _json(manifest),
                        int(manifest.get("k", 0) or 0),
                        int(manifest.get("m", 0) or 0),
                        _json(manifest.get("train_ranks", [])),
                        _json(manifest.get("spare_ranks", [])),
                        _json(manifest.get("E", [])),
                        _json(manifest.get("elastic_layout", {})),
                        tag,
                    ),
                )
                for raw_chunk in manifest.get("chunks", []):
                    if not isinstance(raw_chunk, dict) or "chunk_id" not in raw_chunk:
                        continue
                    chunk = dict(raw_chunk)
                    chunk_id = str(chunk["chunk_id"])
                    current = self._conn.execute(
                        """
                        SELECT row_id, logical_owner_rank, writer_rank, backend, location_json,
                               offset, nbytes, valid_nbytes, checksum_type, checksum
                        FROM chunks WHERE tag=? AND chunk_id=?
                        """,
                        (tag, chunk_id),
                    ).fetchone()
                    if current is None:
                        continue
                    location = dict(_loads(current[4], {}))
                    if isinstance(chunk.get("location"), dict):
                        location = dict(chunk["location"])
                    nbytes = int(chunk.get("nbytes", chunk.get("num_bytes", current[6])) or 0)
                    valid_nbytes = int(chunk.get("valid_nbytes", nbytes if current[7] is None else current[7]) or 0)
                    sealed_checksum_type = str(current[8] or "")
                    sealed_checksum = str(current[9] or "")
                    checksum_type = sealed_checksum_type or str(chunk.get("checksum_type", ""))
                    checksum = sealed_checksum or str(chunk.get("checksum", ""))
                    self._conn.execute(
                        """
                        UPDATE chunks
                        SET row_id=?, logical_owner_rank=?, writer_rank=?, backend=?,
                            location_json=?, offset=?, nbytes=?, valid_nbytes=?,
                            checksum_type=?, checksum=?
                        WHERE tag=? AND chunk_id=?
                        """,
                        (
                            chunk.get("row", chunk.get("row_id", current[0])),
                            chunk.get("owner_rank", chunk.get("logical_owner_rank", current[1])),
                            chunk.get("writer_rank", current[2]),
                            str(chunk.get("backend", current[3])),
                            _json(location),
                            int(location.get("offset", current[5] or 0) or 0),
                            nbytes,
                            valid_nbytes,
                            checksum_type,
                            checksum,
                            tag,
                            chunk_id,
                        ),
                    )

    def commit_metadata_checkpoint(
        self,
        tag: str,
        manifest: dict[str, Any],
        *,
        backend: str,
    ) -> bool:
        """Atomically publish a metadata-only checkpoint.

        Returns ``True`` when this call created the committed record and
        ``False`` for an idempotent retry of the same committed manifest.
        Metadata checkpoints deliberately use a separate fast path: there are
        no chunk rows to seal, and the manifest must become visible in one
        SQLite transaction rather than through the normal begin/update/commit
        sequence.
        """

        actual = str(tag)
        value = dict(manifest or {})
        if list(value.get("chunks", [])):
            raise ValueError("metadata-only CSD checkpoints cannot contain chunks")
        encoded = _json(value)
        timestamp = now_ns()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT state, manifest_json, expected_chunks, created_at_ns "
                    "FROM checkpoints WHERE tag=?",
                    (actual,),
                ).fetchone()
                if row is not None:
                    state = str(row[0])
                    previous = str(row[1] or "{}")
                    expected = int(row[2] or 0)
                    if expected != 0 or previous != encoded:
                        raise RuntimeError(
                            f"CSD metadata checkpoint {actual!r} already exists with different content"
                        )
                    if state == "COMMITTED":
                        self._conn.execute("COMMIT")
                        return False
                    if state not in {"PREPARING", "WRITING", "COMMITTING"}:
                        raise RuntimeError(
                            f"CSD metadata checkpoint {actual!r} cannot commit from state={state}"
                        )
                    chunk_count = int(
                        self._conn.execute(
                            "SELECT COUNT(*) FROM chunks WHERE tag=?", (actual,)
                        ).fetchone()[0]
                    )
                    if chunk_count:
                        raise RuntimeError(
                            f"CSD metadata checkpoint {actual!r} unexpectedly owns {chunk_count} chunk(s)"
                        )
                    created_at = int(row[3])
                else:
                    created_at = timestamp

                self._conn.execute("DELETE FROM chunks WHERE tag=?", (actual,))
                self._conn.execute("DELETE FROM operations WHERE tag=?", (actual,))
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoints(
                        tag, state, created_at_ns, committed_at_ns, backend, k, m,
                        train_ranks_json, spare_ranks_json, e_matrix_json_or_bytes,
                        layout_json, tensor_specs_json, manifest_json, expected_chunks,
                        total_valid_bytes, error
                    ) VALUES (?, 'COMMITTED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, NULL)
                    """,
                    (
                        actual,
                        created_at,
                        timestamp,
                        str(backend),
                        int(value.get("k", 0) or 0),
                        int(value.get("m", 0) or 0),
                        _json(value.get("train_ranks", [])),
                        _json(value.get("spare_ranks", [])),
                        _json(value.get("E", [])),
                        _json(value.get("elastic_layout", {})),
                        _json(value.get("tensor_specs", value.get("tensors", []))),
                        encoded,
                    ),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise
        return True

    def reserve_chunk(self, tag: str, chunk_id: str, metadata: dict[str, Any], *, backend: str) -> None:
        tag = str(tag)
        chunk_id = str(chunk_id)
        meta = dict(metadata or {})
        with self._lock:
            state_row = self._conn.execute("SELECT state FROM checkpoints WHERE tag=?", (tag,)).fetchone()
            if state_row is None:
                raise KeyError(f"unknown CSD checkpoint tag {tag!r}")
            if str(state_row[0]) != "WRITING":
                raise RuntimeError(f"CSD checkpoint {tag!r} is not writable; state={state_row[0]}")
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks(
                        tag, chunk_id, row_id, logical_owner_rank, writer_rank, backend,
                        location_json, offset, nbytes, valid_nbytes, checksum_type, checksum, state, sealed_at_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, NULL, 0, ?, ?, NULL, NULL, 'RESERVED', NULL)
                    """,
                    (
                        tag,
                        chunk_id,
                        meta.get("row", meta.get("row_id")),
                        meta.get("owner_rank", meta.get("logical_owner_rank")),
                        meta.get("writer_rank", meta.get("owner_rank")),
                        str(backend),
                        int(meta.get("nbytes", meta.get("num_bytes", 0)) or 0),
                        int(meta.get("valid_nbytes", meta.get("nbytes", meta.get("num_bytes", 0))) or 0),
                    ),
                )

    def begin_committed_chunk_update(
        self,
        tag: str,
        chunk_id: str,
        metadata: dict[str, Any],
        *,
        backend: str,
    ) -> None:
        tag = str(tag)
        chunk_id = str(chunk_id)
        with self._lock:
            state_row = self._conn.execute("SELECT state FROM checkpoints WHERE tag=?", (tag,)).fetchone()
            if state_row is None:
                raise KeyError(f"unknown CSD checkpoint tag {tag!r}")
            if str(state_row[0]) != "COMMITTED":
                raise RuntimeError(f"CSD checkpoint {tag!r} is not committed; state={state_row[0]}")
            current = self._conn.execute(
                "SELECT 1 FROM chunks WHERE tag=? AND chunk_id=?",
                (tag, chunk_id),
            ).fetchone()
            if current is None:
                raise KeyError(f"unknown CSD chunk {chunk_id!r} for committed checkpoint {tag!r}")

    def mark_chunk_copying(self, tag: str, chunk_id: str, op_id: str) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE chunks SET state='COPYING' WHERE tag=? AND chunk_id=? AND state IN ('RESERVED', 'COPYING')",
                    (str(tag), str(chunk_id)),
                )
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO operations(op_id, tag, chunk_id, op_type, state, error, start_ns, end_ns)
                    VALUES (?, ?, ?, 'PUT', 'RUNNING', NULL, ?, NULL)
                    """,
                    (str(op_id), str(tag), str(chunk_id), now_ns()),
                )

    def seal_chunk(
        self,
        tag: str,
        chunk_id: str,
        *,
        location: dict[str, Any],
        checksum_type: str,
        checksum: str,
        nbytes: int,
        valid_nbytes: int,
    ) -> None:
        loc = dict(location or {})
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE chunks
                    SET state='SEALED', location_json=?, offset=?, nbytes=?, valid_nbytes=?,
                        checksum_type=?, checksum=?, sealed_at_ns=?
                    WHERE tag=? AND chunk_id=? AND state IN ('RESERVED', 'COPYING', 'SEALED')
                    """,
                    (
                        _json(loc),
                        int(loc.get("offset", 0) or 0),
                        int(nbytes),
                        int(valid_nbytes),
                        str(checksum_type),
                        str(checksum),
                        now_ns(),
                        str(tag),
                        str(chunk_id),
                    ),
                )

    def begin_operation(self, op_id: str, tag: str, chunk_id: str | None, op_type: str) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO operations(op_id, tag, chunk_id, op_type, state, error, start_ns, end_ns)
                    VALUES (?, ?, ?, ?, 'QUEUED', NULL, NULL, NULL)
                    """,
                    (str(op_id), str(tag), None if chunk_id is None else str(chunk_id), str(op_type)),
                )

    def mark_operation_running(self, op_id: str) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE operations SET state='RUNNING', start_ns=COALESCE(start_ns, ?) WHERE op_id=?",
                    (now_ns(), str(op_id)),
                )

    def mark_operation_done(self, op_id: str) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE operations SET state='DONE', end_ns=? WHERE op_id=?",
                    (now_ns(), str(op_id)),
                )

    def mark_operation_failed(self, op_id: str, error: str) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE operations SET state='FAILED', error=?, end_ns=? WHERE op_id=?",
                    (str(error), now_ns(), str(op_id)),
                )

    def commit_checkpoint(self, tag: str) -> None:
        tag = str(tag)
        with self._lock:
            row = self._conn.execute(
                "SELECT state, expected_chunks FROM checkpoints WHERE tag=?", (tag,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown CSD checkpoint tag {tag!r}")
            state, expected = str(row[0]), int(row[1])
            if state == "COMMITTED":
                return
            if state != "WRITING":
                raise RuntimeError(f"CSD checkpoint {tag!r} cannot commit from state={state}")
            sealed = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE tag=? AND state='SEALED'", (tag,)
                ).fetchone()[0]
            )
            if sealed != expected:
                raise RuntimeError(
                    f"CSD checkpoint {tag!r} cannot commit: sealed_chunks={sealed}, expected_chunks={expected}"
                )
            total_valid = int(
                self._conn.execute(
                    "SELECT COALESCE(SUM(valid_nbytes), 0) FROM chunks WHERE tag=? AND state='SEALED'", (tag,)
                ).fetchone()[0]
            )
            with self._conn:
                self._conn.execute("UPDATE checkpoints SET state='COMMITTING' WHERE tag=?", (tag,))
                self._conn.execute(
                    "UPDATE checkpoints SET state='COMMITTED', committed_at_ns=?, total_valid_bytes=? WHERE tag=?",
                    (now_ns(), total_valid, tag),
                )

    def get_checkpoint(self, tag: str) -> dict[str, Any]:
        tag = str(tag)
        with self._lock:
            row = self._conn.execute("SELECT * FROM checkpoints WHERE tag=?", (tag,)).fetchone()
            if row is None:
                raise KeyError(f"unknown CSD checkpoint manifest for tag {tag!r}")
            columns = [item[1] for item in self._conn.execute("PRAGMA table_info(checkpoints)")]
            data = dict(zip(columns, row))
            if str(data.get("state")) != "COMMITTED":
                raise KeyError(f"CSD checkpoint {tag!r} is not committed")
            return data

    def list_chunks(self, tag: str) -> list[ChunkRecord]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT tag, chunk_id, row_id, logical_owner_rank, writer_rank, backend, location_json,
                       offset, nbytes, valid_nbytes, checksum_type, checksum, state
                FROM chunks WHERE tag=? ORDER BY chunk_id
                """,
                (str(tag),),
            ).fetchall()
        return [
            ChunkRecord(
                tag=str(row[0]),
                chunk_id=str(row[1]),
                row_id=None if row[2] is None else int(row[2]),
                logical_owner_rank=None if row[3] is None else int(row[3]),
                writer_rank=None if row[4] is None else int(row[4]),
                backend=str(row[5]),
                location=dict(_loads(row[6], {})),
                offset=int(row[7]),
                nbytes=int(row[8]),
                valid_nbytes=int(row[9]),
                checksum_type=str(row[10] or ""),
                checksum=str(row[11] or ""),
                state=str(row[12]),
            )
            for row in rows
        ]

    def manifest_for_tag(self, tag: str) -> dict[str, Any]:
        checkpoint = self.get_checkpoint(tag)
        manifest = dict(_loads(checkpoint.get("manifest_json"), {}))
        chunks_by_id = {record.chunk_id: record for record in self.list_chunks(tag)}
        merged_chunks: list[dict[str, Any]] = []
        for chunk in manifest.get("chunks", []):
            item = dict(chunk)
            record = chunks_by_id.get(str(item.get("chunk_id")))
            if record is not None:
                item.update(
                    {
                        "backend": record.backend,
                        "location": record.location,
                        "offset": record.offset,
                        "nbytes": record.nbytes,
                        "valid_nbytes": record.valid_nbytes,
                        "checksum_type": record.checksum_type,
                        "checksum": record.checksum,
                        "chunk_state": record.state,
                    }
                )
            merged_chunks.append(item)
        if not merged_chunks and chunks_by_id:
            for record in chunks_by_id.values():
                merged_chunks.append(
                    {
                        "chunk_id": record.chunk_id,
                        "row": record.row_id,
                        "owner_rank": record.logical_owner_rank,
                        "backend": record.backend,
                        "location": record.location,
                        "offset": record.offset,
                        "nbytes": record.nbytes,
                        "valid_nbytes": record.valid_nbytes,
                        "checksum_type": record.checksum_type,
                        "checksum": record.checksum,
                        "chunk_state": record.state,
                    }
                )
        manifest["chunks"] = merged_chunks
        if merged_chunks:
            manifest["checksum"] = hashlib.sha256(
                "".join(str(chunk.get("checksum", "")) for chunk in merged_chunks).encode()
            ).hexdigest()
        manifest["committed"] = True
        manifest["checkpoint_state"] = "COMMITTED"
        manifest["storage_backend"] = checkpoint.get("backend")
        manifest["expected_chunks"] = int(checkpoint.get("expected_chunks") or 0)
        manifest["total_valid_bytes"] = int(checkpoint.get("total_valid_bytes") or 0)
        return manifest

    def abort_checkpoint(self, tag: str, reason: str) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE checkpoints SET state='ABORTED', error=? WHERE tag=?",
                    (str(reason), str(tag)),
                )

    def delete_checkpoint(self, tag: str) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute("UPDATE checkpoints SET state='DELETING' WHERE tag=?", (str(tag),))
                self._conn.execute("DELETE FROM chunks WHERE tag=?", (str(tag),))
                self._conn.execute("DELETE FROM operations WHERE tag=?", (str(tag),))
                self._conn.execute("DELETE FROM checkpoints WHERE tag=?", (str(tag),))

    def list_tags(self, committed_only: bool = True) -> list[str]:
        query = "SELECT tag FROM checkpoints"
        params: tuple[Any, ...] = ()
        if committed_only:
            query += " WHERE state='COMMITTED'"
        query += " ORDER BY tag"
        with self._lock:
            return [str(row[0]) for row in self._conn.execute(query, params).fetchall()]
