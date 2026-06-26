import pytest
import torch

import racer


def _cuda_obj(ranks: list[int], nbytes: int = 4096) -> dict[int, torch.Tensor]:
    return {
        rank: torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=f"cuda:{rank}")
        for rank in ranks
    }


@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="requires train GPUs 0-3 plus spare GPU 4")
def test_cuda_store_load_5gpu_default_layout(native_csd_context_factory):
    ctx, _daemon = native_csd_context_factory(
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
def test_cuda_store_load_large_random_buffer_5gpu(native_csd_context_factory):
    ctx, _daemon = native_csd_context_factory(
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
def test_load_returns_clone_for_direct_checkpoint_rows(native_csd_context_factory):
    ctx, _daemon = native_csd_context_factory(
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
def test_async_store_wait_publishes_checkpoint(native_csd_context_factory):
    ctx, _daemon = native_csd_context_factory(
        k=3,
        m=1,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
    )
    obj = _cuda_obj([0, 1, 2, 3])

    handle = racer.store(obj, tag="async_store", async_op=True, context=ctx)
    assert handle.async_op
    handle.wait()

    assert handle.done()
    assert handle.stats is not None
    loaded = racer.load(tag="async_store", requested_train_ranks=[2], context=ctx)
    assert torch.equal(loaded[2], obj[2])


def test_cpu_pinned_storage_backend_is_rejected():
    with pytest.raises(ValueError, match="unsupported RACER storage_backend"):
        racer.init(
            k=3,
            m=1,
            train_ranks=[0, 1, 2, 3],
            spare_ranks=[4],
            storage_backend="cpu_pinned",
        )


@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="requires train GPUs 0-3 plus spare GPU 4")
def test_cuda_store_load_k2_m2_two_failures(native_csd_context_factory):
    ctx, _daemon = native_csd_context_factory(
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


@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="requires train GPUs 0-3 plus spare GPU 4")
def test_failed_physical_owner_row_recovers_nonfailed_packet(native_csd_context_factory):
    ctx, _daemon = native_csd_context_factory(
        k=3,
        m=1,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
    )
    obj = _cuda_obj([0, 1, 2, 3], nbytes=4096)
    racer.store(obj, tag="owner_row", context=ctx)

    recovered = racer.load(
        tag="owner_row",
        failed_train_ranks=[0],
        requested_train_ranks=[3],
        context=ctx,
    )

    assert torch.equal(recovered[3].to(obj[3].device), obj[3])
    assert ctx.last_load_profile["failed_owner_rows"] == [0]
    assert ctx.last_load_profile["ec_decode_ms"] > 0


@pytest.mark.skipif(torch.cuda.device_count() < 6, reason="requires train GPUs 0-3 plus spare GPUs 4-5")
def test_replacement_mapping_controls_failed_rank_output_device(native_csd_context_factory):
    ctx, _daemon = native_csd_context_factory(
        k=3,
        m=1,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4, 5],
    )
    obj = _cuda_obj([0, 1, 2, 3], nbytes=4096)
    racer.store(obj, tag="replacement", context=ctx)

    recovered = racer.load(
        tag="replacement",
        failed_train_ranks=[0],
        replacement_mapping={0: 5},
        context=ctx,
    )

    assert recovered[0].device.index == 5
    assert torch.equal(recovered[0].to(obj[0].device), obj[0])


@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="requires train GPUs 0-3 plus spare GPU 4")
def test_repair_replaces_failed_owner_rows_and_allows_direct_reload(native_csd_context_factory):
    ctx, _daemon = native_csd_context_factory(
        k=3,
        m=1,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
    )
    obj = _cuda_obj([0, 1, 2, 3], nbytes=4096)
    racer.store(obj, tag="repair", context=ctx)

    handle = racer.repair(tag="repair", failed_train_ranks=[0], replacement_mapping={0: 4}, context=ctx)
    assert handle.repaired_rows
    assert set(handle.replacement_mapping.items()) == {(0, 4)}

    manifest = ctx.chunk_storage.get_manifest("repair")
    row0_chunks = [chunk for chunk in manifest["chunks"] if chunk["row"] == 0]
    assert row0_chunks
    assert all(chunk["owner_rank"] == 4 for chunk in row0_chunks)

    reloaded = racer.load(tag="repair", requested_train_ranks=[3], context=ctx)
    assert torch.equal(reloaded[3].to(obj[3].device), obj[3])


@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="requires train GPUs 0-3 plus spare GPU 4")
def test_async_repair_wait_updates_manifest(native_csd_context_factory):
    ctx, _daemon = native_csd_context_factory(
        k=3,
        m=1,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
    )
    obj = _cuda_obj([0, 1, 2, 3], nbytes=4096)
    racer.store(obj, tag="async_repair", context=ctx)

    handle = racer.repair(
        tag="async_repair",
        failed_train_ranks=[0],
        replacement_mapping={0: 4},
        async_op=True,
        context=ctx,
    )
    assert handle.async_op
    handle.wait()

    assert handle.done()
    assert handle.repaired_rows
    reloaded = racer.load(tag="async_repair", requested_train_ranks=[3], context=ctx)
    assert torch.equal(reloaded[3].to(obj[3].device), obj[3])
