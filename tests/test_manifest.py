import pytest
import sqlite3
import torch

import racer
from racer import manifest as manifest_mod


@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="requires train GPUs 0-3 plus spare GPU 4")
def test_store_writes_manifest_with_train_owners_and_virtual_slots(native_csd_context_factory):
    ctx, _daemon = native_csd_context_factory(
        k=3,
        m=1,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
    )
    obj = {
        rank: torch.arange(rank * 11, rank * 11 + 7, dtype=torch.uint8, device=f"cuda:{rank}")
        for rank in [0, 1, 2, 3]
    }
    racer.store(obj, tag="manifest", context=ctx)
    manifest = ctx.chunk_storage.get_manifest("manifest")

    assert manifest["E"] == ctx.matrix
    assert manifest["train_ranks"] == [0, 1, 2, 3]
    assert manifest["spare_ranks"] == [4]
    assert manifest["elastic_layout"]["num_virtual_zero"] == 2
    assert [slot["slot_id"] for slot in manifest["virtual_slots"]] == [4, 5]
    assert set(manifest["chunk_owner"].values()) <= {0, 1, 2, 3}
    assert 4 not in manifest["chunk_owner"].values()
    assert len(ctx.chunk_storage.list_chunks("manifest")) == 8
    assert all("checksum" in chunk for chunk in manifest["chunks"])
    assert "strategy" not in manifest["routing_plan"]
    assert manifest["routing_cost"]["compute_bytes_on_train_ranks"] == 0
    assert manifest["routing_cost"]["compute_bytes_on_accelerators"] > 0
    assert manifest["routing_cost"]["skipped_virtual_zero_bytes"] > 0
    assert ctx.last_routing_plan is not None


@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="requires train GPUs 0-3 plus spare GPU 4")
def test_load_can_rebuild_checkpoint_from_chunk_storage_manifest(native_csd_context_factory):
    ctx, _daemon = native_csd_context_factory(
        k=3,
        m=1,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
    )
    obj = {
        rank: torch.arange(rank * 13, rank * 13 + 9, dtype=torch.uint8, device=f"cuda:{rank}")
        for rank in [0, 1, 2, 3]
    }
    racer.store(obj, tag="persistent", context=ctx)

    recovered = racer.load(tag="persistent", failed_train_ranks=[0], context=ctx)
    assert torch.equal(recovered[0].to(obj[0].device), obj[0])


def test_checkpoint_from_chunk_storage_requires_committed_daemon_resident_manifest():
    class FakeStorage:
        def get_manifest(self, tag):
            assert tag == "lost"
            return {
                "tag": "lost",
                "committed": True,
                "daemon_owned": True,
                "data_resident": False,
                "chunks": [],
            }

    with pytest.raises(RuntimeError, match="data_resident"):
        manifest_mod.checkpoint_from_chunk_storage(tag="lost", chunk_storage=FakeStorage())


@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="requires train GPUs 0-3 plus spare GPU 4")
def test_load_verifies_manifest_chunk_checksums(native_csd_context_factory, tmp_path):
    ctx, daemon = native_csd_context_factory(
        k=3,
        m=1,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
    )
    obj = {
        rank: torch.arange(rank * 17, rank * 17 + 32, dtype=torch.uint8, device=f"cuda:{rank}")
        for rank in [0, 1, 2, 3]
    }
    racer.store(obj, tag="checksum", context=ctx)

    manifest = daemon.client.get_manifest("checksum")
    original_checksum = manifest["chunks"][1]["checksum"]
    manifest["chunks"][1]["checksum"] = "bad"
    daemon.client.put_manifest("checksum", manifest)
    stored = daemon.client.get_manifest("checksum")
    assert stored["chunks"][1]["checksum"] == original_checksum

    sqlite_path = tmp_path / "metadata_0" / "csd_manifest.sqlite"
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            "UPDATE chunks SET checksum='bad' WHERE tag=? AND chunk_id=?",
            ("checksum", stored["chunks"][1]["chunk_id"]),
        )

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        racer.load(tag="checksum", failed_train_ranks=[0], context=ctx)
