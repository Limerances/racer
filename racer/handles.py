"""Async operation handles returned by the RACER public API."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

import torch

from .utils import synchronize_devices


@dataclass
class StoreHandle:
    tag: str
    context: Any
    devices: set[torch.device]
    async_op: bool
    stats: dict[str, Any] | None = None
    _worker: threading.Thread | None = None
    _error: BaseException | None = None

    def wait(self) -> "StoreHandle":
        if self._worker is not None:
            self._worker.join()
            if self._error is not None:
                raise self._error
        synchronize_devices(self.devices)
        return self

    def done(self) -> bool:
        if self._worker is not None:
            return not self._worker.is_alive()
        if not self.async_op:
            return True
        for device in self.devices:
            if device.type == "cuda":
                with torch.cuda.device(device):
                    if not torch.cuda.current_stream(device).query():
                        return False
        return True


@dataclass
class RepairHandle:
    tag: str
    repaired_rows: dict[str, int]
    replacement_mapping: dict[int, int]
    stats: dict[str, Any]
    async_op: bool = False
    _worker: threading.Thread | None = None
    _error: BaseException | None = None

    def wait(self) -> "RepairHandle":
        if self._worker is not None:
            self._worker.join()
            if self._error is not None:
                raise self._error
        return self

    def done(self) -> bool:
        if self._worker is not None:
            return not self._worker.is_alive()
        return True
