import pytest
import torch

import racer
from racer.gpt2_synthetic import state_dict_byte_equal


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires train GPUs 0-2 plus spare GPU 3")
def test_store_load_state_dict_payload_cuda_failed_rank():
    ctx = racer.init(
        k=2,
        m=1,
        train_ranks=[0, 1, 2],
        spare_ranks=[3],
        backend="cuda",
        storage_backend="in_process_cuda",
        async_op=False,
    )
    states = {
        rank: {
            "model.weight": (torch.arange(12, dtype=torch.float32, device=f"cuda:{rank}").view(3, 4) + rank),
            "model.bias": torch.arange(4, dtype=torch.float16, device=f"cuda:{rank}") + rank,
            "optimizer.step": torch.tensor([rank], dtype=torch.int64, device=f"cuda:{rank}"),
        }
        for rank in [0, 1, 2]
    }

    handle = racer.store(states, tag="state", context=ctx, async_op=False)
    assert handle.stats["payload_kind"] == "state_dict"
    recovered = racer.load(tag="state", failed_train_ranks=[0], context=ctx)

    assert list(recovered) == [0]
    assert state_dict_byte_equal(states[0], recovered[0])
    assert ctx.last_load_profile["payload_kind"] == "state_dict"
    assert ctx.last_store_profile["bytes_total"] > 0


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires train GPUs 0-2 plus spare GPU 3")
def test_state_dict_manifest_metadata_is_json_serializable():
    ctx = racer.init(
        k=2,
        m=1,
        train_ranks=[0, 1, 2],
        spare_ranks=[3],
        backend="cuda",
        storage_backend="in_process_cuda",
        async_op=False,
    )
    states = {rank: {"x": torch.full((8,), rank, dtype=torch.uint8, device=f"cuda:{rank}")} for rank in [0, 1, 2]}
    racer.store(states, tag="state_manifest", context=ctx, async_op=False)

    import json

    metadata = ctx.storage.get("state_manifest").metadata
    json.dumps(metadata)
    assert metadata["payload_kind"] == "state_dict"
    assert sorted(metadata["rank_state_metadata"]) == ["0", "1", "2"]
