"""Public RACER API.

The user-facing surface is `racer.init`, `racer.store`, and `racer.load`.
RACER accepts either raw `Dict[int, torch.uint8 Tensor]` packets or
`Dict[int, state_dict]` checkpoint payloads keyed by train rank.
"""

from __future__ import annotations

from typing import Any

from .config import RacerConfig
from .context import RacerContext
from .csd import CheckpointStorageDaemonClient
from .handles import RepairHandle, StoreHandle

_DEFAULT_CONTEXT: RacerContext | None = None


def _validate_csd_backend(requested_backend: str, chunk_storage: Any) -> None:
    capabilities_fn = getattr(chunk_storage, "capabilities", None)
    if capabilities_fn is None:
        raise ValueError("CSD storage requires a client with a capabilities() method")
    caps = dict(capabilities_fn())
    actual_backend = str(caps.get("backend", caps.get("storage_backend", ""))).lower()
    requested = str(requested_backend).lower().replace("-", "_")

    if requested in {"csd_native_pinned", "daemon_native_pinned"}:
        if actual_backend != "native_pinned" or not bool(caps.get("cuda_native_pinned", False)):
            raise ValueError(
                "storage_backend='csd_native_pinned' requires a CSD daemon with "
                f"backend=native_pinned; got backend={actual_backend or '<unknown>'}, "
                f"capabilities={caps}"
            )
        if not (bool(caps.get("supports_cuda_ipc", False)) and bool(caps.get("supports_async_copy", False))):
            raise ValueError(
                "storage_backend='csd_native_pinned' requires CUDA IPC async copy support; "
                f"capabilities={caps}"
            )
    elif requested in {"csd_egm", "daemon_egm"}:
        if actual_backend != "egm" or not bool(caps.get("supports_egm_native_transport", False)):
            raise ValueError(
                "storage_backend='csd_egm' requires a daemon-owned EGM backend with "
                f"native handle transport; got backend={actual_backend or '<unknown>'}, "
                f"capabilities={caps}"
            )
    else:
        raise ValueError(
            f"unsupported RACER storage_backend={requested_backend!r}; "
            "only daemon-owned csd_native_pinned and csd_egm are valid production paths"
        )


def init(
    k: int,
    m: int,
    train_ranks: list[int],
    spare_ranks: list[int],
    buffer_size: int = 64 * 1024 * 1024,
    optimize_cauchy: bool = False,
    storage_backend: str = "csd_native_pinned",
    storage_options: dict[str, Any] | None = None,
) -> RacerContext:
    global _DEFAULT_CONTEXT
    config = RacerConfig(
        k=k,
        m=m,
        train_ranks=tuple(train_ranks),
        spare_ranks=tuple(spare_ranks),
        buffer_size=buffer_size,
        optimize_cauchy=optimize_cauchy,
    )
    options = dict(storage_options or {})
    normalized_backend = str(storage_backend).lower().replace("-", "_")
    if normalized_backend in {"csd_native_pinned", "daemon_native_pinned", "csd_egm", "daemon_egm"}:
        client = options.pop("client", None)
        if client is not None:
            chunk_storage = client
        else:
            address = options.pop("address", None)
            if address is None:
                host = options.pop("host", "127.0.0.1")
                port = options.pop("port", None)
                if port is None:
                    raise ValueError("CSD storage requires storage_options with client, address, or host/port")
                address = (host, int(port))
            chunk_storage = CheckpointStorageDaemonClient(
                address,
                authkey=options.pop("authkey", None),
            )
        _validate_csd_backend(normalized_backend, chunk_storage)
    else:
        raise ValueError(
            f"unsupported RACER storage_backend={storage_backend!r}. "
            "RACER checkpoint storage is intentionally restricted to daemon-owned "
            "csd_native_pinned or csd_egm; in-process CUDA, CPU pinned, fd/mmap, "
            "and other fallback paths are disabled."
        )
    _DEFAULT_CONTEXT = RacerContext(config, chunk_storage=chunk_storage)
    return _DEFAULT_CONTEXT


def get_context() -> RacerContext:
    if _DEFAULT_CONTEXT is None:
        raise RuntimeError("racer.init(...) must be called before using the default context")
    return _DEFAULT_CONTEXT


def store(
    obj,
    tag: str | None = None,
    context: RacerContext | None = None,
    async_op: bool = False,
) -> StoreHandle:
    ctx = context if context is not None else get_context()
    return ctx.store(obj, tag=tag, async_op=async_op)


def load(
    tag: str | None = None,
    failed_train_ranks: list[int] | None = None,
    survivor_rows: list[int] | None = None,
    requested_train_ranks: list[int] | None = None,
    replacement_mapping: dict[int, int] | None = None,
    context: RacerContext | None = None,
):
    ctx = context if context is not None else get_context()
    return ctx.load(
        tag=tag,
        failed_train_ranks=failed_train_ranks,
        survivor_rows=survivor_rows,
        requested_train_ranks=requested_train_ranks,
        replacement_mapping=replacement_mapping,
    )


def repair(
    tag: str | None = None,
    failed_train_ranks: list[int] | None = None,
    replacement_mapping: dict[int, int] | None = None,
    context: RacerContext | None = None,
    async_op: bool = False,
) -> RepairHandle:
    ctx = context if context is not None else get_context()
    return ctx.repair(
        tag=tag,
        failed_train_ranks=failed_train_ranks,
        replacement_mapping=replacement_mapping,
        async_op=async_op,
    )
