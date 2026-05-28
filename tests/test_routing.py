from racer import cauchy
from racer.config import RacerConfig
from racer.layout import ElasticLayout
from racer.routing import HybridPlanner, SpareComputePlanner, TrainingLocalPlanner


def _config(strategy="spare_compute"):
    return RacerConfig(
        k=3,
        m=1,
        train_ranks=(0, 1, 2, 3),
        spare_ranks=(4,),
        backend="cuda",
        storage_backend="in_process_cuda",
        routing_strategy=strategy,
    )


def test_spare_compute_plan_skips_virtual_zero_and_returns_parity_to_train_owner():
    config = _config("spare_compute")
    layout = ElasticLayout.build([0, 1, 2, 3], [4], k=3, m=1)
    E = cauchy.generate_systematic_matrix(3, 1)
    plan = SpareComputePlanner(config).plan(layout, E, chunk_nbytes=1024)

    assert plan.strategy == "spare_compute"
    assert all(not op.is_virtual_zero for op in plan.transfers)
    assert all(op.input_slot not in {4, 5} for op in plan.computes)
    assert all(op.rank == 4 for op in plan.computes)
    assert any(op.dst_rank == 3 and op.description == "parity result to train-rank chunk owner" for op in plan.transfers)
    assert plan.reductions[1].skipped_virtual_zero_slots == (4, 5)
    assert plan.cost.skipped_virtual_zero_bytes == 2 * 1024


def test_training_local_plan_computes_on_train_and_reduces_on_parity_owner():
    config = _config("training_local")
    layout = ElasticLayout.build([0, 1, 2, 3], [4], k=3, m=1)
    E = cauchy.generate_systematic_matrix(3, 1)
    plan = TrainingLocalPlanner(config).plan(layout, E, chunk_nbytes=2048)

    assert plan.strategy == "training_local"
    assert all(op.is_on_train_rank for op in plan.computes)
    assert {op.target_rank for op in plan.reductions} == {3}
    assert all(op.target_rank not in config.spare_ranks for op in plan.reductions)
    assert not any(op.description == "parity result to train-rank chunk owner" for op in plan.transfers)
    assert plan.cost.compute_bytes_on_train_ranks == 4 * 2048
    assert plan.cost.compute_bytes_on_accelerators == 0


def test_hybrid_plan_uses_simple_size_heuristic():
    config = _config("hybrid")
    layout = ElasticLayout.build([0, 1, 2, 3], [4], k=3, m=1)
    E = cauchy.generate_systematic_matrix(3, 1)
    small = HybridPlanner(config).plan(layout, E, chunk_nbytes=4096)
    large = HybridPlanner(config).plan(layout, E, chunk_nbytes=2 * 1024 * 1024)
    assert small.strategy == "training_local"
    assert large.strategy == "spare_compute"
