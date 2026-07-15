from __future__ import annotations

import numpy as np

from ..geometry import bbox_center, clean_bbox, clip_bbox, mask_to_bbox
from ..types import Hypothesis


class PositionKalmanFilter:
    def __init__(self, width_limit: int, height_limit: int):
        dt = 0.1
        self.width_limit = float(width_limit)
        self.height_limit = float(height_limit)
        self.state = np.zeros((4, 1), dtype=np.float64)
        self.transition = np.asarray(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64
        )
        self.control = np.asarray([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]], dtype=np.float64)
        self.acceleration = np.ones((2, 1), dtype=np.float64)
        self.measurement = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self.process_noise = np.asarray(
            [
                [dt**4 / 4, 0, dt**3 / 2, 0],
                [0, dt**4 / 4, 0, dt**3 / 2],
                [dt**3 / 2, 0, dt**2, 0],
                [0, dt**3 / 2, 0, dt**2],
            ],
            dtype=np.float64,
        )
        self.measurement_noise = np.diag([(1 / 20) ** 2, (1 / 20) ** 2])
        self.covariance = np.eye(4, dtype=np.float64)
        self.stable_position = None

    def predict(self) -> np.ndarray:
        if self.stable_position is not None:
            return self.stable_position.copy()
        previous = self.state.copy()
        self.state = self.transition @ self.state + self.control @ self.acceleration
        self.covariance = self.transition @ self.covariance @ self.transition.T + self.process_noise
        position = self.state[:2, 0]
        if not (0 < position[0] <= self.width_limit and 0 < position[1] <= self.height_limit):
            position = previous[:2, 0]
            self.stable_position = position.copy()
        return position.copy()

    def update(self, center) -> None:
        self.stable_position = None
        observed = np.asarray(center, dtype=np.float64).reshape(2, 1)
        innovation = self.measurement @ self.covariance @ self.measurement.T + self.measurement_noise
        gain = self.covariance @ self.measurement.T @ np.linalg.inv(innovation)
        self.state = np.round(self.state + gain @ (observed - self.measurement @ self.state))
        self.covariance = (np.eye(4) - gain @ self.measurement) @ self.covariance


class KalmanCandidate:
    def __init__(self):
        self.filter = None
        self.last_bbox = None
        self.last_mask = None

    @staticmethod
    def _translate_mask(mask, delta_x: float, delta_y: float):
        source = (np.asarray(mask) > 0).astype(np.uint8)
        height, width = source.shape
        shift_x, shift_y = int(round(delta_x)), int(round(delta_y))
        translated = np.zeros_like(source)
        sx1, sy1 = max(0, -shift_x), max(0, -shift_y)
        sx2, sy2 = min(width, width - shift_x), min(height, height - shift_y)
        if sx2 > sx1 and sy2 > sy1:
            translated[sy1 + shift_y : sy2 + shift_y, sx1 + shift_x : sx2 + shift_x] = source[sy1:sy2, sx1:sx2]
        return translated

    def initialize(self, bbox, image_width: int, image_height: int, mask=None) -> None:
        cleaned = clean_bbox(bbox)
        if cleaned is None:
            return
        self.filter = PositionKalmanFilter(image_width, image_height)
        self.filter.update(bbox_center(cleaned))
        self.last_bbox = cleaned
        if mask is not None and np.asarray(mask).sum() > 0:
            self.last_mask = (np.asarray(mask) > 0).astype(np.uint8).copy()

    def predict(self, image_width: int, image_height: int) -> Hypothesis | None:
        if self.filter is None or self.last_bbox is None:
            return None
        center = self.filter.predict()
        width, height = self.last_bbox[2:]
        bbox = clip_bbox([center[0] - width / 2, center[1] - height / 2, width, height], image_width, image_height)
        if bbox is None:
            return None
        mask = None
        if self.last_mask is not None:
            previous_center = bbox_center(self.last_bbox)
            mask = self._translate_mask(self.last_mask, center[0] - previous_center[0], center[1] - previous_center[1])
            bbox = clip_bbox(mask_to_bbox(mask), image_width, image_height) or bbox
        return Hypothesis(key="kalman", kind="kalman", bbox=bbox, mask=mask)

    def update(self, bbox, mask=None) -> None:
        cleaned = clean_bbox(bbox)
        if self.filter is None or cleaned is None:
            return
        self.filter.update(bbox_center(cleaned))
        self.last_bbox = cleaned
        if mask is not None and np.asarray(mask).sum() > 0:
            self.last_mask = (np.asarray(mask) > 0).astype(np.uint8).copy()
