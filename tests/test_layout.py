import torch

from racer import cauchy, codec_cpu
from racer.layout import ElasticLayout, RacerLayout


def test_layout_supports_w_mod_k_not_zero():
    layout = RacerLayout.build([0, 1, 2, 3], k=3)
    assert len(layout.stripes) == 2
    assert layout.stripes[0].data_ranks == (0, 1, 2)
    assert layout.stripes[1].data_ranks == (3, None, None)
    assert layout.locate_rank(3) == (1, 0)


def test_elastic_layout_k3_m1_virtual_zero_slots():
    layout = ElasticLayout.build([0, 1, 2, 3], [4], k=3, m=1)
    assert layout.q == 2
    assert layout.virtual_W == 6
    assert layout.num_virtual_zero == 2
    assert [slot.train_rank for slot in layout.reduction_groups[0]] == [0, 1, 2]
    assert [slot.train_rank for slot in layout.reduction_groups[1]] == [3, None, None]
    assert all(slot.train_rank != 4 for group in layout.data_groups for slot in group)
    assert [slot.slot_id for slot in layout.virtual_zero_slots()] == [4, 5]


def test_elastic_layout_k2_m2_no_virtual_zero():
    layout = ElasticLayout.build([0, 1, 2, 3], [4], k=2, m=2)
    assert layout.q == 2
    assert layout.virtual_W == 4
    assert layout.num_virtual_zero == 0
    assert [slot.train_rank for slot in layout.reduction_groups[0]] == [0, 1]
    assert [slot.train_rank for slot in layout.reduction_groups[1]] == [2, 3]


def test_elastic_layout_k4_m2_virtual_zero_slots():
    layout = ElasticLayout.build([0, 1, 2, 3, 4, 5], [6, 7], k=4, m=2)
    assert layout.q == 2
    assert layout.virtual_W == 8
    assert layout.num_virtual_zero == 2
    assert [slot.train_rank for slot in layout.reduction_groups[0]] == [0, 1, 2, 3]
    assert [slot.train_rank for slot in layout.reduction_groups[1]] == [4, 5, None, None]
    assert all(slot.train_rank not in {6, 7} for group in layout.data_groups for slot in group)


def test_elastic_layout_virtual_zero_cpu_codec_recovery():
    layout = ElasticLayout.build([0, 1, 2, 3], [4], k=3, m=1)
    E = cauchy.generate_systematic_matrix(k=3, m=1)
    packets = {
        rank: torch.arange(rank * 17, rank * 17 + 19, dtype=torch.uint8)
        for rank in layout.train_ranks
    }

    for group in layout.reduction_groups:
        chunks = [
            torch.zeros(0, dtype=torch.uint8) if slot.is_virtual_zero else packets[slot.train_rank]
            for slot in group
        ]
        code = codec_cpu.encode_cpu(chunks, E)
        for failed_row in range(4):
            survivors = [row for row in range(4) if row != failed_row]
            decoded = codec_cpu.decode_cpu([code[row] for row in survivors], survivors, E)
            for col, slot in enumerate(group):
                if not slot.is_virtual_zero:
                    expected = packets[slot.train_rank]
                    assert torch.equal(decoded[col][: expected.numel()], expected)
