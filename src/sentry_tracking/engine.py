from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np

from .association import TrajectoryPool
from .candidates import KalmanCandidate, filter_amg_hypotheses, filter_decoder_hypotheses, soft_nms
from .config import SENTRYConfig
from .geometry import clean_bbox, frame_size, mask_to_bbox
from .memory import MemoryController
from .rescue import RescueHistory, severe_failure_gate
from .reverse import empty_verification, verify_candidates
from .types import TrackResult


class BaselineTracker:
    def __init__(self, backend):
        self.backend = backend

    def initialize(self, frame, bbox, mask=None):
        started = perf_counter()
        output = self.backend.initialize(frame, bbox, mask)
        diagnostics = dict(output.diagnostics)
        diagnostics["timing_ms"] = {"backend_initialize": (perf_counter() - started) * 1000.0}
        return TrackResult(
            bbox=clean_bbox(bbox),
            mask=output.primary.mask,
            source="init",
            baseline_bbox=clean_bbox(bbox),
            diagnostics=diagnostics,
        )

    def track(self, frame):
        started = perf_counter()
        output = self.backend.track(frame)
        forward_ms = (perf_counter() - started) * 1000.0
        bbox = clean_bbox(output.primary.bbox) or mask_to_bbox(output.primary.mask)
        diagnostics = dict(output.diagnostics)
        diagnostics["timing_ms"] = {"backend_forward": forward_ms, "total": forward_ms}
        return TrackResult(
            bbox=bbox,
            mask=output.primary.mask,
            source="baseline",
            baseline_bbox=bbox,
            diagnostics=diagnostics,
        )


class SENTRYTracker:
    def __init__(self, backend, config: SENTRYConfig | None = None):
        self.backend = backend
        self.config = config or SENTRYConfig()
        configure_cache = getattr(self.backend, "configure_reverse_feature_cache", None)
        if callable(configure_cache):
            configure_cache(
                self.config.reverse_feature_cache_enabled,
                max_frames=self.config.reverse_frames + 1,
            )
        self.history = RescueHistory(self.config)
        self.pool = TrajectoryPool(self.config)
        self.kalman = KalmanCandidate()
        self.memory = MemoryController(self.config.memory_policy)
        self.frame_idx = -1
        self.initialized = False

    def initialize(self, frame: Any, bbox, mask=None) -> TrackResult:
        total_started = perf_counter()
        bbox = clean_bbox(bbox)
        if bbox is None:
            raise ValueError("SENTRY requires a valid initialization box")
        backend_started = perf_counter()
        output = self.backend.initialize(frame, bbox, mask)
        backend_ms = (perf_counter() - backend_started) * 1000.0
        accepted_mask = output.primary.mask
        width, height = frame_size(frame)
        self.history = RescueHistory(self.config)
        self.pool = TrajectoryPool(self.config)
        self.kalman = KalmanCandidate()
        self.memory = MemoryController(self.config.memory_policy)
        self.frame_idx = 0
        self.history.update(bbox, "init", 0, frame=frame, mask=accepted_mask)
        self.kalman.initialize(bbox, width, height, mask=accepted_mask)
        self.initialized = True
        return TrackResult(
            bbox=bbox,
            mask=accepted_mask,
            source="init",
            baseline_bbox=bbox,
            diagnostics={
                "frame_idx": 0,
                "memory_update": {"policy": self.config.memory_policy, "attempted": False, "applied": False},
                "backend": output.diagnostics,
                "timing_ms": {
                    "backend_initialize": backend_ms,
                    "total": (perf_counter() - total_started) * 1000.0,
                },
            },
        )

    def track(self, frame: Any) -> TrackResult:
        total_started = perf_counter()
        if not self.initialized:
            raise RuntimeError("initialize() must be called before track()")
        self.frame_idx += 1
        width, height = frame_size(frame)
        kalman_started = perf_counter()
        kalman = self.kalman.predict(width, height)
        kalman_predict_ms = (perf_counter() - kalman_started) * 1000.0
        backend_started = perf_counter()
        output = self.backend.track(frame)
        backend_forward_ms = (perf_counter() - backend_started) * 1000.0
        baseline = output.primary
        baseline.bbox = clean_bbox(baseline.bbox) or mask_to_bbox(baseline.mask)
        gate_started = perf_counter()
        severe, failure = severe_failure_gate(
            baseline.bbox,
            baseline.mask,
            self.history,
            self.frame_idx,
            baseline.predicted_iou,
            self.config,
        )
        failure_gate_ms = (perf_counter() - gate_started) * 1000.0

        decoder, decoder_diagnostics = [], None
        amg, amg_diagnostics = [], {"attempted": False}
        nms_diagnostics = None
        candidates = []
        verification = empty_verification(self.config)
        candidate_preparation_ms = 0.0
        reverse_verification_ms = 0.0
        if self.config.reverse_enabled:
            candidates_started = perf_counter()
            decoder, decoder_diagnostics = filter_decoder_hypotheses(
                output.decoder_hypotheses,
                output.object_score,
                self.history.median_area(),
                width,
                height,
                self.config,
            )
            if self.config.amg_enabled:
                try:
                    raw_amg = self.backend.generate_amg_hypotheses(frame)
                    amg, amg_diagnostics = filter_amg_hypotheses(
                        raw_amg, self.history.median_area(), width, height, self.config
                    )
                except Exception as error:
                    amg_diagnostics = {"attempted": True, "error": f"{type(error).__name__}: {error}"}
            filtered, nms_diagnostics = soft_nms(decoder + amg, self.config)
            candidates = [item for item in filtered if not item.is_baseline]
            candidate_preparation_ms = (perf_counter() - candidates_started) * 1000.0
            reverse_started = perf_counter()
            verification = verify_candidates(
                self.backend,
                self.history,
                self.pool,
                self.frame_idx,
                frame,
                baseline,
                kalman,
                candidates,
                self.config,
            )
            reverse_verification_ms = (perf_counter() - reverse_started) * 1000.0

        selection_started = perf_counter()
        candidate_ready = bool(candidates) or verification["kalman"]["attempted"]
        would_rescue = bool(severe and candidate_ready and verification["verified"])
        rescued = bool(self.config.rescue_enabled and would_rescue)
        selected = verification["selected_candidate"]
        saved_bbox, saved_mask, source = baseline.bbox, baseline.mask, "baseline"
        if rescued:
            saved_bbox = clean_bbox(selected.bbox)
            saved_mask = selected.mask if selected.mask is not None else baseline.mask
            source = f"sentry_{selected.kind}_rescue_{failure['failure_reason']}"
        selection_ms = (perf_counter() - selection_started) * 1000.0

        memory_started = perf_counter()
        memory_update = self.memory.admit(self.backend, rescued, saved_mask)
        memory_update_ms = (perf_counter() - memory_started) * 1000.0
        state_started = perf_counter()
        accepted_key = selected.key if rescued and selected.kind != "kalman" else baseline.key
        if rescued and selected.kind == "kalman":
            accepted_key = None
        self.pool.mark_target(accepted_key)
        self.history.update(saved_bbox, source, self.frame_idx, frame=frame, mask=saved_mask)
        self.kalman.update(saved_bbox, mask=saved_mask)
        state_update_ms = (perf_counter() - state_started) * 1000.0
        cache_stats = None
        get_cache_stats = getattr(self.backend, "reverse_feature_cache_stats", None)
        if callable(get_cache_stats):
            cache_stats = get_cache_stats()

        diagnostics = {
            "frame_idx": self.frame_idx,
            "baseline_bbox": baseline.bbox,
            "saved_bbox": saved_bbox,
            "source": source,
            "severe_failure": failure,
            "candidate_ready": candidate_ready,
            "would_rescue": would_rescue,
            "decoder": decoder_diagnostics,
            "amg": amg_diagnostics,
            "soft_nms": nms_diagnostics,
            "verification": verification,
            "reverse_feature_cache": cache_stats,
            "memory_update": memory_update,
            "backend": output.diagnostics,
            "timing_ms": {
                "kalman_predict": kalman_predict_ms,
                "backend_forward": backend_forward_ms,
                "failure_gate": failure_gate_ms,
                "candidate_preparation": candidate_preparation_ms,
                "reverse_verification": reverse_verification_ms,
                "selection": selection_ms,
                "memory_update": memory_update_ms,
                "state_update": state_update_ms,
                "total": (perf_counter() - total_started) * 1000.0,
            },
        }
        return TrackResult(
            bbox=saved_bbox,
            mask=saved_mask,
            source=source,
            baseline_bbox=baseline.bbox,
            selected_candidate=selected if rescued else None,
            severe_failure=failure["failure_reason"],
            diagnostics=diagnostics,
        )
