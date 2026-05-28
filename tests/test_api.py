import pytest
import torch

import racer
from racer.config import RacerConfig
from racer.routing import compute_device


def test_spare_not_in_e_matrix_invariant():
    with pytest.raises(ValueError, match="k \\+ m must equal len\\(train_ranks\\)"):
        racer.init(
            k=4,
            m=1,
            train_ranks=[0, 1, 2, 3],
            spare_ranks=[4],
            backend="cpu",
            storage_backend="in_process_cpu",
        )


def test_include_spares_in_train_not_implemented():
    with pytest.raises(NotImplementedError):
        racer.init(
            k=4,
            m=1,
            train_ranks=[0, 1, 2, 3],
            spare_ranks=[4],
            backend="cpu",
            storage_backend="in_process_cpu",
            include_spares_in_train=True,
        )


def test_store_rejects_spare_rank_key():
    ctx = racer.init(
        k=3,
        m=1,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
        backend="cpu",
        storage_backend="in_process_cpu",
    )
    obj = {
        0: torch.zeros(4, dtype=torch.uint8),
        1: torch.zeros(4, dtype=torch.uint8),
        2: torch.zeros(4, dtype=torch.uint8),
        3: torch.zeros(4, dtype=torch.uint8),
        4: torch.zeros(4, dtype=torch.uint8),
    }
    with pytest.raises(ValueError, match="spare ranks"):
        racer.store(obj, context=ctx, async_op=False)


def test_spare_compute_without_spare_falls_back_to_train_gpu():
    config = RacerConfig(
        k=2,
        m=1,
        train_ranks=(0, 1, 2),
        spare_ranks=(),
        backend="cuda",
        storage_backend="in_process_cuda",
    )
    assert compute_device(config) == torch.device("cuda:0")
