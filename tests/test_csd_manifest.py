import pytest

from racer.csd_manifest import CsdManifestStore


def test_manifest_begin_uncommitted_is_not_loadable(tmp_path):
    store = CsdManifestStore(tmp_path / "csd.sqlite")
    store.begin_checkpoint(
        "t0",
        {"tag": "t0", "chunks": []},
        expected_chunks=0,
        backend="native_pinned",
    )

    with pytest.raises(KeyError, match="not committed"):
        store.manifest_for_tag("t0")
    assert store.list_tags(committed_only=True) == []
    assert store.list_tags(committed_only=False) == ["t0"]


def test_atomic_metadata_commit_is_idempotent_and_rejects_conflict(tmp_path):
    store = CsdManifestStore(tmp_path / "csd.sqlite")
    manifest = {
        "racer_manifest_kind": "megatron_tensor_tree",
        "checkpoint_tag": "checkpoint",
        "generation": "gen-1",
    }

    assert store.commit_metadata_checkpoint("meta", manifest, backend="egm") is True
    assert store.commit_metadata_checkpoint("meta", manifest, backend="egm") is False
    loaded = store.manifest_for_tag("meta")
    assert loaded["generation"] == "gen-1"
    assert loaded["expected_chunks"] == 0
    assert loaded["committed"] is True

    with pytest.raises(RuntimeError, match="different content"):
        store.commit_metadata_checkpoint(
            "meta",
            {**manifest, "generation": "gen-2"},
            backend="egm",
        )
    assert store.manifest_for_tag("meta")["generation"] == "gen-1"


def test_atomic_metadata_commit_completes_matching_legacy_writing_record(tmp_path):
    store = CsdManifestStore(tmp_path / "csd.sqlite")
    manifest = {"racer_manifest_kind": "marker", "generation": "gen-1"}
    store.begin_checkpoint("meta", manifest, expected_chunks=0, backend="egm")

    assert store.commit_metadata_checkpoint("meta", manifest, backend="egm") is True
    assert store.manifest_for_tag("meta")["checkpoint_state"] == "COMMITTED"


def test_atomic_metadata_commit_rejects_data_manifests(tmp_path):
    store = CsdManifestStore(tmp_path / "csd.sqlite")

    with pytest.raises(ValueError, match="cannot contain chunks"):
        store.commit_metadata_checkpoint(
            "data",
            {"chunks": [{"chunk_id": "c0"}]},
            backend="egm",
        )
    assert store.list_tags(committed_only=False) == []


def test_manifest_commit_requires_all_chunks_sealed(tmp_path):
    store = CsdManifestStore(tmp_path / "csd.sqlite")
    store.begin_checkpoint(
        "t1",
        {"tag": "t1", "chunks": [{"chunk_id": "c0", "row": 0, "owner_rank": 0}]},
        expected_chunks=1,
        backend="native_pinned",
    )
    store.reserve_chunk("t1", "c0", {"row": 0, "owner_rank": 0, "nbytes": 3}, backend="native_pinned")

    with pytest.raises(RuntimeError, match="sealed_chunks=0"):
        store.commit_checkpoint("t1")


def test_manifest_commit_succeeds_after_all_chunks_sealed(tmp_path):
    store = CsdManifestStore(tmp_path / "csd.sqlite")
    store.begin_checkpoint(
        "t2",
        {"tag": "t2", "chunks": [{"chunk_id": "c0", "row": 0, "owner_rank": 0}]},
        expected_chunks=1,
        backend="native_pinned",
    )
    store.reserve_chunk("t2", "c0", {"row": 0, "owner_rank": 0, "nbytes": 3}, backend="native_pinned")
    store.mark_chunk_copying("t2", "c0", "op0")
    store.seal_chunk(
        "t2",
        "c0",
        location={"backend": "native_pinned", "offset": 0, "nbytes": 3},
        checksum_type="sha256",
        checksum="abc",
        nbytes=3,
        valid_nbytes=3,
    )
    store.commit_checkpoint("t2")

    assert store.list_tags(committed_only=True) == ["t2"]
    manifest = store.manifest_for_tag("t2")
    assert manifest["committed"] is True
    assert manifest["checkpoint_state"] == "COMMITTED"
    assert manifest["chunks"][0]["checksum"] == "abc"


def test_manifest_update_preserves_sealed_checksum(tmp_path):
    store = CsdManifestStore(tmp_path / "csd.sqlite")
    manifest = {
        "tag": "sealed",
        "chunks": [{"chunk_id": "c0", "row": 0, "owner_rank": 0, "checksum": "stale-prewrite"}],
    }
    store.begin_checkpoint("sealed", manifest, expected_chunks=1, backend="native_pinned")
    store.reserve_chunk("sealed", "c0", {"row": 0, "owner_rank": 0, "nbytes": 3}, backend="native_pinned")
    store.mark_chunk_copying("sealed", "c0", "op0")
    store.seal_chunk(
        "sealed",
        "c0",
        location={"backend": "native_pinned", "offset": 0, "nbytes": 3},
        checksum_type="sha256",
        checksum="sealed-sha",
        nbytes=3,
        valid_nbytes=3,
    )
    store.update_manifest(
        "sealed",
        {
            "tag": "sealed",
            "chunks": [
                {
                    "chunk_id": "c0",
                    "row": 0,
                    "owner_rank": 0,
                    "checksum_type": "sample64",
                    "checksum": "stale-sampled",
                }
            ],
        },
    )
    store.commit_checkpoint("sealed")

    stored = store.manifest_for_tag("sealed")
    assert stored["chunks"][0]["checksum_type"] == "sha256"
    assert stored["chunks"][0]["checksum"] == "sealed-sha"


def test_manifest_reopens_committed_metadata(tmp_path):
    path = tmp_path / "csd.sqlite"
    store = CsdManifestStore(path)
    store.begin_checkpoint(
        "restart",
        {"tag": "restart", "chunks": [{"chunk_id": "c0", "row": 1, "owner_rank": 2}]},
        expected_chunks=1,
        backend="native_pinned",
    )
    store.reserve_chunk(
        "restart",
        "c0",
        {"row": 1, "owner_rank": 2, "writer_rank": 2, "nbytes": 7},
        backend="native_pinned",
    )
    store.mark_chunk_copying("restart", "c0", "op-restart")
    store.seal_chunk(
        "restart",
        "c0",
        location={"backend": "native_pinned", "segment_id": "seg0", "offset": 256, "nbytes": 7},
        checksum_type="sha256",
        checksum="deadbeef",
        nbytes=7,
        valid_nbytes=7,
    )
    store.commit_checkpoint("restart")
    store.close()

    reopened = CsdManifestStore(path)
    manifest = reopened.manifest_for_tag("restart")
    assert manifest["storage_backend"] == "native_pinned"
    assert manifest["chunks"][0]["location"]["segment_id"] == "seg0"
    assert manifest["total_valid_bytes"] == 7
