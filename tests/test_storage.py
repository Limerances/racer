import pytest
import torch

from racer.storage import EgmStorage, InProcessCudaStorage


def test_in_process_cuda_storage_rejects_host_tensor():
    storage = InProcessCudaStorage()
    tensor = torch.arange(16, dtype=torch.uint8)
    with pytest.raises(ValueError, match="CUDA tensors"):
        storage.put("t", "c0", tensor, {"owner_rank": 0})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_in_process_cuda_storage_preserves_cuda_tensor_isolation():
    storage = InProcessCudaStorage()
    tensor = torch.arange(16, dtype=torch.uint8, device="cuda:0")
    storage.put("t", "c0", tensor, {"owner_rank": 0})
    storage.put_manifest("t", {"train_ranks": [0], "spare_ranks": [1]})
    loaded = storage.get("t", "c0")
    assert loaded.device.type == "cuda"
    loaded.fill_(99)
    assert torch.equal(storage.get("t", "c0"), tensor)
    assert storage.get_metadata("t", "c0")["owner_rank"] == 0
    assert storage.list_chunks("t") == ["c0"]
    assert storage.get_manifest("t")["spare_ranks"] == [1]


def test_egm_storage_is_declared_but_not_implemented():
    with pytest.raises(NotImplementedError):
        EgmStorage()
