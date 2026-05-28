import pytest
import torch

import racer


def _cpu_obj():
    return {
        0: torch.arange(0, 17, dtype=torch.uint8),
        1: torch.arange(31, 48, dtype=torch.uint8),
        2: torch.arange(77, 94, dtype=torch.uint8),
        3: torch.arange(101, 118, dtype=torch.uint8),
    }


def test_cpu_store_load_w_mod_k_not_zero_and_recover_first_row():
    ctx = racer.init(
        k=3,
        m=1,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
        backend="cpu",
        storage_backend="in_process_cpu",
        async_op=False,
    )
    obj = _cpu_obj()
    racer.store(obj, tag="t0", context=ctx, async_op=False)
    recovered = racer.load(tag="t0", failed_train_ranks=[0], context=ctx)
    assert list(recovered) == [0]
    assert torch.equal(recovered[0].cpu(), obj[0])


def test_cpu_store_load_recover_tail_rank_without_decoding_data_row():
    ctx = racer.init(
        k=3,
        m=1,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
        backend="cpu",
        storage_backend="in_process_cpu",
        async_op=False,
    )
    obj = _cpu_obj()
    racer.store(obj, tag="tail", context=ctx, async_op=False)
    recovered = racer.load(tag="tail", failed_train_ranks=[3], context=ctx)
    assert torch.equal(recovered[3].cpu(), obj[3])


def test_cpu_store_load_k2_m2_two_failures():
    ctx = racer.init(
        k=2,
        m=2,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
        backend="cpu",
        storage_backend="in_process_cpu",
        async_op=False,
    )
    obj = _cpu_obj()
    racer.store(obj, tag="two", context=ctx, async_op=False)
    recovered = racer.load(tag="two", failed_train_ranks=[0, 1], context=ctx)
    assert torch.equal(recovered[0].cpu(), obj[0])
    assert torch.equal(recovered[1].cpu(), obj[1])


def test_load_returns_clone_for_direct_checkpoint_rows():
    ctx = racer.init(
        k=3,
        m=1,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
        backend="cpu",
        storage_backend="in_process_cpu",
        async_op=False,
    )
    obj = _cpu_obj()
    racer.store(obj, tag="alias", context=ctx, async_op=False)
    loaded = racer.load(tag="alias", context=ctx)
    loaded[0].fill_(99)
    reloaded = racer.load(tag="alias", context=ctx)
    assert torch.equal(reloaded[0], obj[0])


@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="requires train GPUs 0-3 plus spare GPU 4")
def test_cuda_store_load_5gpu_default_layout():
    ctx = racer.init(
        k=3,
        m=1,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
        backend="cuda",
        storage_backend="in_process_cuda",
        async_op=False,
    )
    obj = {
        rank: torch.arange(rank * 19, rank * 19 + 4096, dtype=torch.uint8, device=f"cuda:{rank}")
        for rank in [0, 1, 2, 3]
    }
    racer.store(obj, tag="cuda", context=ctx, async_op=False)
    recovered = racer.load(tag="cuda", failed_train_ranks=[0], context=ctx)
    assert recovered[0].device.index == 4
    assert torch.equal(recovered[0].cpu(), obj[0].cpu())


@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="requires train GPUs 0-3 plus spare GPU 4")
def test_cuda_store_load_large_random_buffer_5gpu():
    ctx = racer.init(
        k=3,
        m=1,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
        backend="cuda",
        storage_backend="in_process_cuda",
        async_op=False,
    )
    nbytes = 2 * 1024 * 1024
    obj = {
        rank: torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=f"cuda:{rank}")
        for rank in [0, 1, 2, 3]
    }
    racer.store(obj, tag="cuda_large", context=ctx, async_op=False)
    recovered = racer.load(tag="cuda_large", failed_train_ranks=[0], context=ctx)
    assert recovered[0].device.index == 4
    assert torch.equal(recovered[0].cpu(), obj[0].cpu())


@pytest.mark.skipif(torch.cuda.device_count() < 3, reason="requires train GPUs 0-2")
def test_cuda_store_load_without_spare_uses_train_compute_fallback():
    ctx = racer.init(
        k=2,
        m=1,
        train_ranks=[0, 1, 2],
        spare_ranks=[],
        backend="cuda",
        storage_backend="in_process_cuda",
        async_op=False,
    )
    obj = {
        rank: torch.randint(0, 256, (4096,), dtype=torch.uint8, device=f"cuda:{rank}")
        for rank in [0, 1, 2]
    }
    racer.store(obj, tag="cuda_no_spare", context=ctx, async_op=False)
    recovered = racer.load(tag="cuda_no_spare", failed_train_ranks=[1], context=ctx)
    assert recovered[1].device.index == 1
    assert torch.equal(recovered[1].cpu(), obj[1].cpu())
