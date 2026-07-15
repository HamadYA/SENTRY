from __future__ import annotations

import numpy as np

from ..config import SENTRYConfig
from ..geometry import bbox_area, clip_bbox, mask_to_bbox
from ..types import Hypothesis


def filter_amg_hypotheses(
    hypotheses: list[Hypothesis],
    median_area: float | None,
    image_width: int,
    image_height: int,
    config: SENTRYConfig,
) -> tuple[list[Hypothesis], dict]:
    finite = [
        item
        for item in hypotheses
        if item.template_similarity is not None and np.isfinite(item.template_similarity)
    ]
    diagnostics = {
        "attempted": True,
        "proposal_count": len(hypotheses),
        "max_similarity": None,
        "similarity_threshold": None,
        "kept_keys": [],
        "rejections": [],
        "error": None,
    }
    if not finite:
        diagnostics["error"] = "no_finite_similarity"
        return [], diagnostics
    maximum = max(float(item.template_similarity) for item in finite)
    diagnostics["max_similarity"] = maximum
    if maximum <= 0:
        diagnostics["error"] = "non_positive_max_similarity"
        return [], diagnostics
    threshold = config.amg_template_similarity_beta * maximum
    diagnostics["similarity_threshold"] = threshold

    retained = []
    for hypothesis in sorted(finite, key=lambda item: float(item.template_similarity), reverse=True):
        similarity = float(hypothesis.template_similarity)
        if similarity < threshold:
            diagnostics["rejections"].append({"key": hypothesis.key, "reason": "below_beta"})
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
                "ranking_score": similarity / maximum,
            }
        )
        retained.append(
            Hypothesis(
                key=hypothesis.key,
                kind="amg",
                bbox=bbox,
                mask=hypothesis.mask,
                predicted_iou=hypothesis.predicted_iou,
                template_similarity=similarity,
                stability_score=hypothesis.stability_score,
                metadata=metadata,
            )
        )
        diagnostics["kept_keys"].append(hypothesis.key)
    return retained, diagnostics
