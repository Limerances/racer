"""Pytest configuration for deterministic RACER correctness tests."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import racer


def pytest_runtest_setup() -> None:
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


@pytest.fixture
def native_csd_context_factory(tmp_path):
    daemons = []

    def make_context(
        *,
        k: int,
        m: int,
        train_ranks: list[int],
        spare_ranks: list[int],
        buffer_size: int = 64 * 1024 * 1024,
    ):
        try:
            daemon = racer.start_checkpoint_storage_daemon(
                metadata_dir=tmp_path / f"metadata_{len(daemons)}",
                backend="native_pinned",
                backend_options={
                    "segment_bytes": max(int(buffer_size), 8 * 1024 * 1024),
                    "device": 0,
                },
            )
            ctx = racer.init(
                k=k,
                m=m,
                train_ranks=train_ranks,
                spare_ranks=spare_ranks,
                buffer_size=buffer_size,
                storage_backend="csd_native_pinned",
                storage_options={"client": daemon.client},
            )
        except Exception as exc:
            pytest.skip(f"native_pinned CSD unavailable; no fallback backend is allowed: {exc}")
        daemons.append(daemon)
        return ctx, daemon

    try:
        yield make_context
    finally:
        for daemon in reversed(daemons):
            daemon.shutdown()
