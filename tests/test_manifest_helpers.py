from racer import checksum, manifest
from racer.state_dict_codec import NonTensorMetadata, RankStateMetadata, TensorMetadata


def test_state_metadata_manifest_round_trip_preserves_tensor_and_non_tensor_fields():
    metadata = RankStateMetadata(
        source_train_rank=7,
        tensors=[
            TensorMetadata(
                key="layer.weight",
                dtype="torch.float16",
                shape=(2, 3),
                device="cuda:1",
                requires_grad=True,
                was_contiguous=False,
                offset=16,
                nbytes=12,
            )
        ],
        payload_nbytes=64,
        non_tensors=[
            NonTensorMetadata(key="step", value=42),
            NonTensorMetadata(key="flags", value={"fp8": False}),
        ],
    )

    encoded = manifest.state_metadata_to_manifest(metadata)
    decoded = manifest.state_metadata_from_manifest(encoded)

    assert decoded == metadata


def test_manifest_checksum_changes_when_chunk_checksum_changes():
    chunks = [{"checksum": "a"}, {"checksum": "b"}]
    original = checksum.manifest_checksum(chunks)

    chunks[1]["checksum"] = "c"

    assert checksum.manifest_checksum(chunks) != original
