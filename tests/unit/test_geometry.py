import numpy as np

from sentry_tracking.geometry import bbox_iou, clip_bbox, mask_to_bbox


def test_geometry_helpers():
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:7, 3:9] = 1
    assert mask_to_bbox(mask) == [3.0, 2.0, 6.0, 5.0]
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert clip_bbox([-2, -3, 8, 9], 10, 10) == [0.0, 0.0, 6.0, 6.0]
