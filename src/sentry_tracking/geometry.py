from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .types import BBox


def clean_bbox(bbox: Sequence[float] | None) -> BBox | None:
    if bbox is None:
        return None
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if len(values) != 4 or not np.isfinite(values).all() or values[2] <= 0 or values[3] <= 0:
        return None
    return values


def bbox_area(bbox: Sequence[float] | None) -> float:
    cleaned = clean_bbox(bbox)
    return 0.0 if cleaned is None else cleaned[2] * cleaned[3]


def bbox_center(bbox: Sequence[float]) -> np.ndarray:
    cleaned = clean_bbox(bbox)
    if cleaned is None:
        raise ValueError("bbox must be valid")
    return np.asarray([cleaned[0] + cleaned[2] / 2, cleaned[1] + cleaned[3] / 2], dtype=np.float32)


def bbox_iou(first: Sequence[float] | None, second: Sequence[float] | None) -> float | None:
    a = clean_bbox(first)
    b = clean_bbox(second)
    if a is None or b is None:
        return None
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    intersection = max(0.0, min(ax2, bx2) - max(a[0], b[0])) * max(
        0.0, min(ay2, by2) - max(a[1], b[1])
    )
    union = bbox_area(a) + bbox_area(b) - intersection
    return intersection / union if union > 0 else None


def clip_bbox(bbox: Sequence[float] | None, width: int, height: int) -> BBox | None:
    cleaned = clean_bbox(bbox)
    if cleaned is None:
        return None
    x1 = min(max(cleaned[0], 0.0), float(width))
    y1 = min(max(cleaned[1], 0.0), float(height))
    x2 = min(max(cleaned[0] + cleaned[2], 0.0), float(width))
    y2 = min(max(cleaned[1] + cleaned[3], 0.0), float(height))
    return clean_bbox([x1, y1, x2 - x1, y2 - y1])


def mask_to_bbox(mask) -> BBox | None:
    if mask is None:
        return None
    array = np.asarray(mask) > 0
    rows, columns = np.where(array)
    if not len(rows):
        return None
    x1, x2 = int(columns.min()), int(columns.max()) + 1
    y1, y2 = int(rows.min()), int(rows.max()) + 1
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def frame_size(frame) -> tuple[int, int]:
    if hasattr(frame, "size") and isinstance(frame.size, tuple):
        return int(frame.size[0]), int(frame.size[1])
    array = np.asarray(frame)
    if array.ndim < 2:
        raise ValueError("frame must be a PIL image or HxW array")
    return int(array.shape[1]), int(array.shape[0])
