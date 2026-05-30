import pytest
import torch

from racer.utils import uint8_view_no_serialize


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_uint8_view_no_serialize_reinterprets_tensor_storage_on_cuda():
    tensor = torch.arange(16, dtype=torch.float32, device="cuda:0")
    payload = uint8_view_no_serialize(tensor)
    assert payload.dtype == torch.uint8
    assert payload.device.type == "cuda"
    assert payload.numel() == tensor.numel() * tensor.element_size()
    payload[0] ^= 0xFF
    assert uint8_view_no_serialize(tensor)[0] == payload[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_uint8_view_no_serialize_rejects_noncontiguous_cuda_tensor():
    tensor = torch.arange(16, dtype=torch.float32, device="cuda:0").view(4, 4).t()
    with pytest.raises(ValueError, match="contiguous"):
        uint8_view_no_serialize(tensor)
