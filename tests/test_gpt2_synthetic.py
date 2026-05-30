import pytest
import torch

from racer.gpt2_synthetic import (
    estimate_rank_nbytes,
    make_rank_state,
    profile_config,
    state_dict_byte_equal,
    tensor_count,
)
from racer.state_dict_codec import flatten_state_dict, unflatten_state_dict


def test_gpt2_124m_tp4_estimate_is_checkpoint_scale():
    config = profile_config("gpt2-124m", tensor_parallel=4, dtype="bf16", include_optimizer=True)
    assert tensor_count(config) > 500
    rank_mib = estimate_rank_nbytes(config) / 1024**2
    assert 350 < rank_mib < 650


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_gpt2_synthetic_state_has_many_named_tensors_and_roundtrips_on_cuda():
    config = profile_config("megatron-test-tp4", include_optimizer=True)
    state = make_rank_state(0, config, device="cuda:0", fill=True, max_tensors=24)

    assert len(state) == 24
    assert any(key.startswith("model.decoder.layers.0") for key in state)
    assert any(key.startswith("optimizer.state.") for key in state)

    flat = flatten_state_dict(0, state, target_device="cuda:0")
    recovered = unflatten_state_dict(flat.metadata, flat.payload, target_device="cuda:0")
    assert state_dict_byte_equal(state, recovered)
    assert all(isinstance(tensor, torch.Tensor) for tensor in recovered.values())
    assert all(tensor.device.type == "cuda" for tensor in recovered.values())
