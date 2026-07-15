from __future__ import annotations

import numpy as np

from ..config import SENTRYConfig
from ..geometry import bbox_area, bbox_center, clean_bbox
from .history import RescueHistory


def severe_failure_gate(baseline_bbox, baseline_mask, history: RescueHistory, frame_idx: int, confidence, config):
    bbox = clean_bbox(baseline_bbox)
    median_area = history.median_area()
    reliable_history = history.has_reliable_history()
    area_ratio = bbox_area(bbox) / median_area if bbox is not None and median_area else None
    center_jump = None
    failure_reason = None
    if bbox is None:
        failure_reason = "empty"
    elif reliable_history:
        if (
            bbox_area(bbox) < config.rescue_area_collapse * median_area
            or bbox_area(bbox) > config.rescue_area_expand * median_area
        ):
            failure_reason = "area"
        elif history.last_bbox is not None:
            center_jump = float(np.linalg.norm(bbox_center(bbox) - bbox_center(history.last_bbox)))
            area_warning = area_ratio is not None and not (
                config.rescue_moderate_area_min_ratio <= area_ratio <= config.rescue_moderate_area_max_ratio
            )
            low_confidence = confidence is not None and confidence < config.rescue_low_confidence
            if center_jump > config.rescue_center_jump_scale * np.sqrt(median_area) and (
                area_warning or low_confidence
            ):
                failure_reason = "jump"
    diagnostics = {
        "mask_area": int(np.asarray(baseline_mask).sum()) if baseline_mask is not None else 0,
        "area_ratio": area_ratio,
        "center_jump": center_jump,
        "baseline_confidence": confidence,
        "failure_reason": failure_reason,
    }
    if not reliable_history and failure_reason != "empty":
        return False, diagnostics
    return failure_reason is not None and not history.in_cooldown(frame_idx), diagnostics
