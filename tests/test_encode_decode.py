import pytest
import torch

import racer


def _cuda_obj(ranks: list[int], nbytes: int = 4096) -> dict[int, torch.Tensor]:
    return {
        rank: torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=f"cuda:{rank}")
        for rank in ranks
    }


@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="requires train GPUs 0-3 plus spare GPU 4")
def test_cuda_store_load_5gpu_default_layout():
    ctx = racer.init(
        k=3,
        m=1,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
    )
    obj = {
        rank: torch.arange(rank * 19, rank * 19 + 4096, dtype=torch.uint8, device=f"cuda:{rank}")
        for rank in [0, 1, 2, 3]
    }
    racer.store(obj, tag="cuda", context=ctx)
    recovered = racer.load(tag="cuda", failed_train_ranks=[0], context=ctx)
    assert recovered[0].device.index == 4
    assert torch.equal(recovered[0].to(obj[0].device), obj[0])


@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="requires train GPUs 0-3 plus spare GPU 4")
def test_cuda_store_load_large_random_buffer_5gpu():
    ctx = racer.init(
        k=3,
        m=1,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
    )
    obj = _cuda_obj([0, 1, 2, 3], nbytes=2 * 1024 * 1024)
    racer.store(obj, tag="cuda_large", context=ctx)
    recovered = racer.load(tag="cuda_large", failed_train_ranks=[0], context=ctx)
    assert recovered[0].device.index == 4
    assert torch.equal(recovered[0].to(obj[0].device), obj[0])


@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="requires train GPUs 0-3 plus spare GPU 4")
def test_load_returns_clone_for_direct_checkpoint_rows():
    ctx = racer.init(
        k=3,
        m=1,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
    )
    obj = _cuda_obj([0, 1, 2, 3])
    racer.store(obj, tag="alias", context=ctx)
    loaded = racer.load(tag="alias", context=ctx)
    loaded[0].fill_(99)
    reloaded = racer.load(tag="alias", context=ctx)
    assert torch.equal(reloaded[0], obj[0])


@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="requires train GPUs 0-3 plus spare GPU 4")
def test_cuda_store_load_k2_m2_two_failures():
    ctx = racer.init(
        k=2,
        m=2,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
    )
    obj = _cuda_obj([0, 1, 2, 3])
    racer.store(obj, tag="two", context=ctx)
    recovered = racer.load(tag="two", failed_train_ranks=[0, 1], context=ctx)
    assert torch.equal(recovered[0].to(obj[0].device), obj[0])
    assert torch.equal(recovered[1].to(obj[1].device), obj[1])
