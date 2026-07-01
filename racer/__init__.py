"""RACER public package exports."""

from .api import get_context, init, load, repair, store
from .config import RacerConfig
from .context import RacerContext
from .handles import RepairHandle, StoreHandle
from .stats import LoadResult, LoadStats, StoreStats

_LAZY_CSD_EXPORTS = {
    "CheckpointStorageDaemonClient",
    "NativePinnedMemoryBackend",
    "EgmBackend",
    "export_cuda_ipc_view",
    "start_checkpoint_storage_daemon",
}

__all__ = [
    "RacerConfig",
    "RacerContext",
    "RepairHandle",
    "StoreHandle",
    "CheckpointStorageDaemonClient",
    "NativePinnedMemoryBackend",
    "EgmBackend",
    "export_cuda_ipc_view",
    "start_checkpoint_storage_daemon",
    "StoreStats",
    "LoadStats",
    "LoadResult",
    "get_context",
    "init",
    "store",
    "load",
    "repair",
]


def __getattr__(name: str):
    if name in _LAZY_CSD_EXPORTS:
        from . import csd

        value = getattr(csd, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
