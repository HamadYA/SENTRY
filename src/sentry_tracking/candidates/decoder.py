from __future__ import annotations

import numpy as np

from ..config import SENTRYConfig
from ..geometry import bbox_area, clip_bbox, mask_to_bbox
from ..types import Hypothesis


def filter_decoder_hypotheses(
    hypotheses: list[Hypothesis],
    object_score: float | None,
    median_area: float | None,
    image_width: int,
    image_height: int,
    config: SENTRYConfig,
) -> tuple[list[Hypothesis], dict]:
    finite_scores = [item.predicted_iou for item in hypotheses if item.predicted_iou is not None]
    diagnostics = {
        "alpha": config.decoder_relative_iou_alpha,
        "best_iou": max(finite_scores) if finite_scores else None,
        "object_score": object_score,
        "object_visible": object_score is None or object_score > 0,
        "retained_keys": [],
        "rejections": [],
    }
    if not finite_scores:
        diagnostics["rejections"].append({"key": None, "reason": "missing_ious"})
        return [], diagnostics

    best_iou = diagnostics["best_iou"]
    retained = []
    for hypothesis in hypotheses:
        score = hypothesis.predicted_iou
        if score is None or not np.isfinite(score):
            diagnostics["rejections"].append({"key": hypothesis.key, "reason": "invalid_iou"})
            continue
        if not score > config.decoder_relative_iou_alpha * best_iou:
            diagnostics["rejections"].append({"key": hypothesis.key, "reason": "below_alpha"})
            continue
        if object_score is not None and object_score <= 0:
            diagnostics["rejections"].append({"key": hypothesis.key, "reason": "occluded"})
            continue

        bbox = clip_bbox(hypothesis.bbox or mask_to_bbox(hypothesis.mask), image_width, image_height)
        if bbox is None:
            diagnostics["rejections"].append({"key": hypothesis.key, "reason": "empty_mask"})
            continue
        area_ratio = bbox_area(bbox) / median_area if median_area else None
        if area_ratio is not None and not (
            config.rescue_candidate_min_area_ratio <= area_ratio <= config.rescue_candidate_max_area_ratio
        ):
            diagnostics["rejections"].append({"key": hypothesis.key, "reason": "area_implausible"})
            continue

        metadata = dict(hypothesis.metadata)
        metadata.update(
            {
                "area_ratio": area_ratio,
                "mask_area": int(np.asarray(hypothesis.mask).sum()) if hypothesis.mask is not None else None,
                "ranking_score": score / best_iou if best_iou > 0 else 0.0,
            }
        )
        retained.append(
            Hypothesis(
                key=hypothesis.key,
                kind="decoder",
                bbox=bbox,
                mask=hypothesis.mask,
                predicted_iou=score,
                object_score=object_score,
                is_baseline=hypothesis.is_baseline,
                metadata=metadata,
            )
        )
        diagnostics["retained_keys"].append(hypothesis.key)
    retained.sort(key=lambda item: item.metadata["ranking_score"], reverse=True)
    return retained, diagnostics
