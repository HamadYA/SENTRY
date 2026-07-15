from __future__ import annotations

from typing import Any

import numpy as np

from ..config import SENTRYConfig
from ..geometry import bbox_area, clean_bbox
from ..types import BBox


class RescueHistory:
    def __init__(self, config: SENTRYConfig):
        self.config = config
        self.accepted_areas: list[float] = []
        self.last_bbox: BBox | None = None
        self.last_mask: np.ndarray | None = None
        self.cooldown_until = -1
        self.recent_frames: list[Any] = []
        self.recent_bboxes: list[BBox | None] = []

    def median_area(self) -> float | None:
        if not self.accepted_areas:
            return None
        return float(np.median(self.accepted_areas[-self.config.rescue_history_window :]))

    def has_reliable_history(self) -> bool:
        median = self.median_area()
        return len(self.accepted_areas) >= self.config.rescue_min_history and median is not None and median > 0

    def in_cooldown(self, frame_idx: int) -> bool:
        return frame_idx <= self.cooldown_until

    def reverse_context(self):
        return list(self.recent_frames), list(self.recent_bboxes)

    def update(self, bbox, source: str, frame_idx: int, frame=None, mask=None) -> None:
        cleaned = clean_bbox(bbox)
        if cleaned is not None:
            area = bbox_area(cleaned)
            if area > 0:
                self.accepted_areas.append(area)
                self.last_bbox = cleaned
        if mask is not None and np.asarray(mask).sum() > 0:
            self.last_mask = (np.asarray(mask) > 0).astype(np.uint8).copy()
        if frame is not None:
            self.recent_frames.append(frame.copy() if hasattr(frame, "copy") else frame)
            self.recent_bboxes.append(cleaned)
            self.recent_frames = self.recent_frames[-self.config.reverse_frames :]
            self.recent_bboxes = self.recent_bboxes[-self.config.reverse_frames :]
        if source not in {"baseline", "sam2", "init"}:
            self.cooldown_until = frame_idx + self.config.rescue_cooldown_frames
