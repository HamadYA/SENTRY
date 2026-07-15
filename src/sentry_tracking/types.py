from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, TypeAlias

import numpy as np

BBox: TypeAlias = list[float]


@dataclass
class Hypothesis:
    key: Hashable
    kind: str
    bbox: BBox | None
    mask: np.ndarray | None = None
    predicted_iou: float | None = None
    object_score: float | None = None
    template_similarity: float | None = None
    stability_score: float | None = None
    is_baseline: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendFrame:
    primary: Hypothesis
    decoder_hypotheses: list[Hypothesis] = field(default_factory=list)
    object_score: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackResult:
    bbox: BBox | None
    mask: np.ndarray | None
    source: str
    baseline_bbox: BBox | None
    selected_candidate: Hypothesis | None = None
    severe_failure: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
