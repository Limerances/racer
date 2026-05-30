from racer import cauchy
from racer.config import RacerConfig
from racer.layout import ElasticLayout
from racer.routing import SpareComputePlanner


def _config():
    return RacerConfig(
        k=3,
        m=1,
        train_ranks=(0, 1, 2, 3),
        spare_ranks=(4,),
    )


def test_spare_compute_plan_skips_virtual_zero_and_returns_parity_to_train_owner():
    config = _config()
    layout = ElasticLayout.build([0, 1, 2, 3], [4], k=3, m=1)
    E = cauchy.generate_systematic_matrix(3, 1)
    plan = SpareComputePlanner(config).plan(layout, E, chunk_nbytes=1024)

    assert all(not op.is_virtual_zero for op in plan.transfers)
    assert all(op.input_slot not in {4, 5} for op in plan.computes)
    assert all(op.rank == 4 for op in plan.computes)
    assert all(op.is_on_spare_rank for op in plan.computes)
    assert any(op.dst_rank == 3 and op.description == "parity result to train-rank chunk owner" for op in plan.transfers)
    assert plan.reductions[1].skipped_virtual_zero_slots == (4, 5)
    assert plan.cost.skipped_virtual_zero_bytes == 2 * 1024
    assert plan.cost.compute_bytes_on_train_ranks == 0
    assert plan.cost.compute_bytes_on_accelerators > 0
