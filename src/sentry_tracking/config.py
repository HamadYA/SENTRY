from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class SENTRYConfig:
    rescue_enabled: bool = True
    reverse_enabled: bool = True
    amg_enabled: bool = True
    association_mode: str = "simple"
    memory_policy: str = "severe_rescue"

    rescue_cooldown_frames: int = 5
    rescue_history_window: int = 10
    rescue_min_history: int = 3
    rescue_area_collapse: float = 0.10
    rescue_area_expand: float = 8.0
    rescue_candidate_min_area_ratio: float = 0.25
    rescue_candidate_max_area_ratio: float = 4.0
    rescue_center_jump_scale: float = 4.0
    rescue_moderate_area_min_ratio: float = 0.50
    rescue_moderate_area_max_ratio: float = 2.0
    rescue_low_confidence: float = 0.35

    reverse_frames: int = 9
    reverse_feature_cache_enabled: bool = True
    reverse_min_history: int = 3
    reverse_min_mean_iou: float = 0.65
    reverse_min_iou: float = 0.20
    reverse_min_coverage: float = 0.75
    reverse_min_margin: float = 0.15

    pool_match_min_iou: float = 0.05
    pool_rescue_min_iou: float = 0.40
    kalman_warning_iou: float = 0.40

    decoder_relative_iou_alpha: float = 0.80
    amg_template_similarity_beta: float = 0.80
    soft_nms_sigma: float = 0.01
    soft_nms_score_threshold: float = 0.25
    max_appearance_candidates: int = 5

    def __post_init__(self) -> None:
        if self.association_mode not in {"simple", "joint"}:
            raise ValueError("association_mode must be 'simple' or 'joint'")
        if self.memory_policy not in {"baseline", "severe_rescue"}:
            raise ValueError("memory_policy must be 'baseline' or 'severe_rescue'")
        if self.reverse_frames < 1 or self.reverse_min_history < 1:
            raise ValueError("reverse history settings must be positive")
        if self.rescue_cooldown_frames < 0:
            raise ValueError("rescue_cooldown_frames cannot be negative")
        if self.max_appearance_candidates < 1:
            raise ValueError("max_appearance_candidates must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SENTRYConfig":
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown SENTRY configuration keys: {sorted(unknown)}")
        return cls(**dict(values))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SENTRYConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise ValueError("SENTRY configuration must be a YAML mapping")
        return cls.from_mapping(values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
