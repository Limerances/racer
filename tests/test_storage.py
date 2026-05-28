import pytest
import torch

from racer.storage import CpuPinnedStorage, EgmStorage, FileMMapStorage, InProcessCudaStorage


def test_in_process_cuda_storage_api_cpu_tensor_compat():
    storage = InProcessCudaStorage()
    tensor = torch.arange(16, dtype=torch.uint8)
    storage.put("t", "c0", tensor, {"owner_rank": 0})
    storage.put_manifest("t", {"train_ranks": [0], "spare_ranks": [1]})
    loaded = storage.get("t", "c0")
    loaded.fill_(99)
    assert torch.equal(storage.get("t", "c0"), tensor)
    assert storage.get_metadata("t", "c0")["owner_rank"] == 0
    assert storage.list_chunks("t") == ["c0"]
    assert storage.get_manifest("t")["spare_ranks"] == [1]


def test_cpu_pinned_storage_copies_to_cpu():
    storage = CpuPinnedStorage()
    tensor = torch.arange(32, dtype=torch.uint8)
    storage.put("t", "c0", tensor, {"owner_rank": 0})
    loaded = storage.get("t", "c0")
    assert loaded.device.type == "cpu"
    assert torch.equal(loaded, tensor)


def test_file_mmap_storage_persists_chunks_and_manifest(tmp_path):
    storage = FileMMapStorage(tmp_path)
    tensor = torch.arange(64, dtype=torch.uint8).view(8, 8)
    storage.put("tag0", "chunk0", tensor, {"owner_rank": 2})
    storage.put_manifest("tag0", {"E": [[1]], "train_ranks": [2], "spare_ranks": [3]})

    reopened = FileMMapStorage(tmp_path)
    assert reopened.list_chunks("tag0") == ["chunk0"]
    assert reopened.get_metadata("tag0", "chunk0")["owner_rank"] == 2
    assert torch.equal(reopened.get("tag0", "chunk0"), tensor)
    assert reopened.get_manifest("tag0")["train_ranks"] == [2]


def test_egm_storage_is_declared_but_not_implemented():
    with pytest.raises(NotImplementedError):
        EgmStorage()
