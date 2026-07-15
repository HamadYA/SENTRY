from __future__ import annotations

import math

from ..config import SENTRYConfig
from ..geometry import bbox_iou
from ..types import Hypothesis


def soft_nms(hypotheses: list[Hypothesis], config: SENTRYConfig) -> tuple[list[Hypothesis], dict]:
    pending = [(item, float(item.metadata.get("ranking_score", 0.0))) for item in hypotheses]
    kept: list[Hypothesis] = []
    post_scores = {}
    while pending and len(kept) < config.max_appearance_candidates:
        best_idx = max(range(len(pending)), key=lambda idx: pending[idx][1])
        best, score = pending.pop(best_idx)
        if score < config.soft_nms_score_threshold:
            break
        metadata = dict(best.metadata)
        metadata["pre_nms_score"] = metadata.get("ranking_score", score)
        metadata["post_nms_score"] = score
        best.metadata = metadata
        kept.append(best)
        post_scores[str(best.key)] = score

        updated = []
        for candidate, candidate_score in pending:
            overlap = bbox_iou(best.bbox, candidate.bbox) or 0.0
            decayed = candidate_score * math.exp(-(overlap * overlap) / config.soft_nms_sigma)
            if decayed > config.soft_nms_score_threshold:
                updated.append((candidate, decayed))
        pending = updated

    kept_keys = {item.key for item in kept}
    diagnostics = {
        "input_count": len(hypotheses),
        "kept_count": len(kept),
        "kept_keys": [item.key for item in kept],
        "suppressed_keys": [item.key for item in hypotheses if item.key not in kept_keys],
        "post_scores": post_scores,
    }
    return kept, diagnostics
