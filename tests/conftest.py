"""Pytest configuration for deterministic RACER correctness tests."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_runtest_setup() -> None:
    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
