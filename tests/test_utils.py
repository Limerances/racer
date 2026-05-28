import pytest
import torch

from racer.utils import uint8_view_no_serialize


def test_uint8_view_no_serialize_reinterprets_tensor_storage():
    tensor = torch.arange(16, dtype=torch.float32)
    payload = uint8_view_no_serialize(tensor)
    assert payload.dtype == torch.uint8
    assert payload.numel() == tensor.numel() * tensor.element_size()
    payload[0] ^= 0xFF
    assert uint8_view_no_serialize(tensor)[0] == payload[0]


def test_uint8_view_no_serialize_rejects_noncontiguous():
    tensor = torch.arange(16, dtype=torch.float32).view(4, 4).t()
    with pytest.raises(ValueError, match="contiguous"):
        uint8_view_no_serialize(tensor)
