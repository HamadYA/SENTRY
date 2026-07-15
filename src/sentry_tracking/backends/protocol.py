from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..types import BackendFrame, BBox, Hypothesis


@runtime_checkable
class BaselineBackend(Protocol):
    def initialize(self, frame: Any, bbox: Sequence[float], mask: np.ndarray | None = None) -> BackendFrame: ...

    def track(self, frame: Any) -> BackendFrame: ...


@runtime_checkable
class SENTRYBackend(BaselineBackend, Protocol):
    def generate_amg_hypotheses(self, frame: Any) -> list[Hypothesis]: ...

    def reverse_track_window(
        self,
        frames: Sequence[Any],
        bbox: Sequence[float] | None = None,
        mask: np.ndarray | None = None,
    ) -> dict[int, BBox | None]: ...

    def replace_current_memory(self, mask: np.ndarray) -> dict: ...
