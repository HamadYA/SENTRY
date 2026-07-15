from __future__ import annotations

import numpy as np

from ..geometry import bbox_iou, clean_bbox


def empty_reverse_score():
    return {"mean_iou": None, "min_iou": None, "coverage": 0.0, "matched_frames": 0, "expected_frames": 0}


def reverse_trajectory_score(reverse_boxes, historical_bboxes):
    expected = sum(clean_bbox(bbox) is not None for bbox in historical_bboxes)
    overlaps = []
    for frame_idx, historical_bbox in enumerate(historical_bboxes):
        overlap = bbox_iou(reverse_boxes.get(frame_idx), historical_bbox)
        if overlap is not None:
            overlaps.append(overlap)
    return {
        "mean_iou": float(np.mean(overlaps)) if overlaps else None,
        "min_iou": float(np.min(overlaps)) if overlaps else None,
        "coverage": len(overlaps) / expected if expected else 0.0,
        "matched_frames": len(overlaps),
        "expected_frames": expected,
    }
