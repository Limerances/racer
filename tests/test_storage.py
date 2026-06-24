import pytest

import racer
from racer import storage


def test_legacy_storage_classes_are_not_public_exports():
    assert not hasattr(racer, "InProcessCudaStorage")
    assert not hasattr(racer, "CpuPinnedStorage")
    assert not hasattr(racer, "EgmStorage")
    assert not hasattr(racer, "FdMmapHostBackend")


@pytest.mark.parametrize(
    "cls",
    [
        storage.InProcessCudaStorage,
        storage.CpuPinnedStorage,
        storage.EgmStorage,
        storage.InProcessStorage,
    ],
)
def test_legacy_storage_constructors_raise(cls):
    with pytest.raises(RuntimeError, match="daemon-owned"):
        cls()


@pytest.mark.parametrize("backend", ["cuda", "cuda_legacy", "cpu_pinned", "egm", "csd_pinned", "fd_mmap_host"])
def test_init_rejects_non_daemon_native_storage_backends(backend):
    with pytest.raises(ValueError, match="unsupported RACER storage_backend"):
        racer.init(
            k=2,
            m=1,
            train_ranks=[0, 1, 2],
            spare_ranks=[3],
            storage_backend=backend,
        )
