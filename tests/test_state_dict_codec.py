import pytest
import torch

from racer.state_dict_codec import flatten_state_dict, unflatten_state_dict


def assert_equal_state_dict(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert left[key].dtype == right[key].dtype
        assert left[key].shape == right[key].shape
        assert str(right[key].device).startswith("cuda")
        assert torch.equal(right[key].to(left[key].device), left[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_flatten_unflatten_multi_dtype_state_dict_cuda() -> None:
    state = {
        "fp32": torch.randn(2, 3, dtype=torch.float32, device="cuda:0"),
        "fp16": torch.randn(4, dtype=torch.float16, device="cuda:0"),
        "bf16": torch.randn(5, dtype=torch.bfloat16, device="cuda:0"),
        "i64": torch.arange(6, dtype=torch.int64, device="cuda:0"),
        "u8": torch.arange(7, dtype=torch.uint8, device="cuda:0"),
    }

    flat = flatten_state_dict(0, state, target_device="cuda:0")
    recovered = unflatten_state_dict(flat.metadata, flat.payload, target_device="cuda:0")

    assert flat.metadata.source_train_rank == 0
    assert flat.metadata.payload_nbytes == flat.payload.numel()
    assert_equal_state_dict(state, recovered)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_flatten_records_non_contiguous_metadata_and_restores_values() -> None:
    state = {"view": torch.arange(12, dtype=torch.float32, device="cuda:0").reshape(3, 4).t()}
    assert not state["view"].is_contiguous()

    flat = flatten_state_dict(1, state, target_device="cuda:0")
    recovered = unflatten_state_dict(flat.metadata, flat.payload, target_device="cuda:0")

    assert flat.metadata.tensors[0].was_contiguous is False
    assert torch.equal(recovered["view"], state["view"])
    assert recovered["view"].is_contiguous()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_flatten_restores_requires_grad_for_float_tensor() -> None:
    state = {"weight": torch.randn(3, dtype=torch.float32, device="cuda:0", requires_grad=True)}

    flat = flatten_state_dict(0, state, target_device="cuda:0")
    recovered = unflatten_state_dict(flat.metadata, flat.payload, target_device="cuda:0")

    assert recovered["weight"].requires_grad
    assert torch.equal(recovered["weight"], state["weight"])
