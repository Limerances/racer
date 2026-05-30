import pytest
import torch

import racer
from racer.config import RacerConfig
from racer.routing import compute_device


def test_spare_not_in_e_matrix_invariant():
    with pytest.raises(ValueError, match=r"k \+ m must equal len\(train_ranks\)"):
        RacerConfig(
            k=4,
            m=1,
            train_ranks=(0, 1, 2, 3),
            spare_ranks=(4,),
        )


def test_spare_compute_requires_spare_gpu():
    with pytest.raises(ValueError, match="spare_ranks"):
        RacerConfig(
            k=2,
            m=1,
            train_ranks=(0, 1, 2),
            spare_ranks=(),
        )


@pytest.mark.skipif(torch.cuda.device_count() < 5, reason="requires train GPUs 0-3 plus spare GPU 4")
def test_store_rejects_spare_rank_key():
    ctx = racer.init(
        k=3,
        m=1,
        train_ranks=[0, 1, 2, 3],
        spare_ranks=[4],
    )
    obj = {
        rank: torch.zeros(4, dtype=torch.uint8, device=f"cuda:{rank}")
        for rank in [0, 1, 2, 3, 4]
    }
    with pytest.raises(ValueError, match="spare ranks"):
        racer.store(obj, context=ctx)


def test_spare_compute_uses_spare_gpu():
    config = RacerConfig(
        k=2,
        m=1,
        train_ranks=(0, 1, 2),
        spare_ranks=(3,),
    )
    assert compute_device(config) == torch.device("cuda:3")
