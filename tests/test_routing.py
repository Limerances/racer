from racer import cauchy
from racer.config import RacerConfig
from racer.layout import ElasticLayout
from racer.routing import SpareComputePlanner, _CostAccumulator


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
    assert plan.cost.compute_bytes_on_spare_ranks == plan.cost.compute_bytes_on_accelerators
    assert plan.cost.xor_bytes_on_spare_ranks == plan.cost.xor_bytes_on_accelerators
    assert plan.cost.max_spare_rank_compute_bytes == plan.cost.max_spare_compute_bytes
    assert plan.cost.final_result_return_bytes == 2 * 1024


def test_cost_model_combines_compute_and_xor_per_rank():
    cost = _CostAccumulator()
    cost.compute(rank=4, num_bytes=100, is_spare=True)
    cost.xor(rank=5, num_bytes=90, is_spare=True)
    cost.compute(rank=5, num_bytes=20, is_spare=True)

    model = cost.build()

    assert model.max_spare_rank_compute_bytes == 110
    assert model.max_spare_compute_bytes == 110
