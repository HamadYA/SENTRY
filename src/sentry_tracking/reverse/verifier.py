from __future__ import annotations

from time import perf_counter

import numpy as np

from ..association import TrajectoryPool
from ..config import SENTRYConfig
from ..geometry import bbox_area, clean_bbox
from ..rescue import RescueHistory
from ..types import Hypothesis
from .scoring import empty_reverse_score, reverse_trajectory_score


def empty_verification(config: SENTRYConfig):
    return {
        "attempted": False,
        "verified": False,
        "candidate": empty_reverse_score(),
        "candidates": [],
        "baseline": empty_reverse_score(),
        "score_margin": None,
        "selected_key": None,
        "selected_kind": None,
        "selected_candidate": None,
        "kalman": {
            "warning_triggered": False,
            "attempted": False,
            "score": empty_reverse_score(),
            "score_margin": None,
            "plausible": False,
            "verified": False,
            "error": None,
        },
        "pool_was_ready": False,
        "pool": [],
        "association": {"mode": config.association_mode, "target_candidate_key": None},
        "timing_ms": {
            "baseline_reverse": 0.0,
            "candidate_reverse": 0.0,
            "kalman_reverse": 0.0,
            "association": 0.0,
            "total": 0.0,
        },
        "error": None,
    }


def _passes_thresholds(score, valid_history, config):
    return bool(
        valid_history >= config.reverse_min_history
        and score["mean_iou"] is not None
        and score["mean_iou"] >= config.reverse_min_mean_iou
        and score["min_iou"] is not None
        and score["min_iou"] >= config.reverse_min_iou
        and score["coverage"] >= config.reverse_min_coverage
    )


def _reverse(backend, frames, hypothesis: Hypothesis):
    if hypothesis.mask is not None and np.asarray(hypothesis.mask).sum() > 0:
        return backend.reverse_track_window(frames, mask=hypothesis.mask)
    return backend.reverse_track_window(frames, bbox=hypothesis.bbox)


def verify_candidates(
    backend,
    history: RescueHistory,
    pool: TrajectoryPool,
    frame_idx: int,
    current_frame,
    baseline: Hypothesis,
    kalman: Hypothesis | None,
    candidates: list[Hypothesis],
    config: SENTRYConfig,
):
    total_started = perf_counter()
    verification = empty_verification(config)
    timing = verification["timing_ms"]
    historical_frames, historical_bboxes = history.reverse_context()
    valid_history = sum(clean_bbox(bbox) is not None for bbox in historical_bboxes)
    if not historical_frames:
        verification["error"] = "no_history"
        timing["total"] = (perf_counter() - total_started) * 1000.0
        return verification
    frames = historical_frames + [current_frame]
    verification["attempted"] = True

    baseline_required = baseline.mask is not None and np.asarray(baseline.mask).sum() > 0
    baseline_reverse = None
    baseline_score = empty_reverse_score()
    if baseline_required:
        reverse_started = perf_counter()
        try:
            baseline_reverse = _reverse(backend, frames, baseline)
            baseline_score = reverse_trajectory_score(baseline_reverse, historical_bboxes)
        except Exception as error:
            verification["error"] = f"baseline_{type(error).__name__}: {error}"
        finally:
            timing["baseline_reverse"] = (perf_counter() - reverse_started) * 1000.0
    verification["baseline"] = baseline_score
    baseline_mean = baseline_score["mean_iou"]

    pool_inputs = []
    if baseline_reverse and baseline.bbox is not None:
        pool_inputs.append(
            {"key": baseline.key, "bbox": baseline.bbox, "reverse_boxes": baseline_reverse, "is_winner": True}
        )

    results = []
    by_key = {}
    for candidate in candidates:
        result = {
            "key": candidate.key,
            "kind": candidate.kind,
            "score": empty_reverse_score(),
            "score_margin": None,
            "pool_iou": None,
            "pool_matched": False,
            "verified": False,
            "reverse_ms": 0.0,
            "error": None,
        }
        reverse_started = perf_counter()
        try:
            reverse_boxes = _reverse(backend, frames, candidate)
            score = reverse_trajectory_score(reverse_boxes, historical_bboxes)
            result["score"] = score
            if score["mean_iou"] is not None and baseline_mean is not None:
                result["score_margin"] = score["mean_iou"] - baseline_mean
            pool_inputs.append(
                {"key": candidate.key, "bbox": candidate.bbox, "reverse_boxes": reverse_boxes, "is_winner": False}
            )
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            result["reverse_ms"] = (perf_counter() - reverse_started) * 1000.0
            timing["candidate_reverse"] += result["reverse_ms"]
        results.append(result)
        by_key[candidate.key] = result

    kalman_result = verification["kalman"]
    transient = []
    if kalman is not None and clean_bbox(kalman.bbox) is not None:
        kalman_result["attempted"] = True
        median = history.median_area()
        ratio = bbox_area(kalman.bbox) / median if median else None
        kalman_result["plausible"] = ratio is None or (
            config.rescue_candidate_min_area_ratio <= ratio <= config.rescue_candidate_max_area_ratio
        )
        kalman_result["warning_triggered"] = valid_history >= config.reverse_min_history and (
            baseline_mean is None or baseline_mean < config.kalman_warning_iou
        )
        reverse_started = perf_counter()
        try:
            reverse_boxes = _reverse(backend, frames, kalman)
            score = reverse_trajectory_score(reverse_boxes, historical_bboxes)
            margin = (
                score["mean_iou"] - baseline_mean
                if score["mean_iou"] is not None and baseline_mean is not None
                else None
            )
            margin_ok = not baseline_required or (margin is not None and margin >= config.reverse_min_margin)
            kalman_result.update(
                {
                    "score": score,
                    "score_margin": margin,
                    "verified": bool(
                        kalman_result["warning_triggered"]
                        and kalman_result["plausible"]
                        and _passes_thresholds(score, valid_history, config)
                        and margin_ok
                    ),
                }
            )
            transient.append(
                {"key": "kalman", "bbox": kalman.bbox, "reverse_boxes": reverse_boxes, "persistent": False}
            )
        except Exception as error:
            kalman_result["error"] = f"{type(error).__name__}: {error}"
        finally:
            timing["kalman_reverse"] = (perf_counter() - reverse_started) * 1000.0

    association_started = perf_counter()
    pool_metadata, pool_was_ready, association = pool.update(
        frame_idx, len(historical_frames), pool_inputs, historical_bboxes, transient
    )
    timing["association"] = (perf_counter() - association_started) * 1000.0
    verification["association"] = association
    verified = []
    for candidate in candidates:
        result = by_key[candidate.key]
        metadata = pool_metadata.get(candidate.key)
        if metadata:
            result["pool_iou"] = metadata["pool_iou"]
            result["pool_matched"] = metadata["matched"]
        margin_ok = not baseline_required or (
            result["score_margin"] is not None and result["score_margin"] >= config.reverse_min_margin
        )
        if config.association_mode == "joint":
            pool_ok = association["target_candidate_key"] == candidate.key
        else:
            pool_ok = not pool_was_ready or (
                result["pool_matched"]
                and result["pool_iou"] is not None
                and result["pool_iou"] >= config.pool_rescue_min_iou
            )
        result["verified"] = bool(
            _passes_thresholds(result["score"], valid_history, config) and margin_ok and pool_ok
        )
        if result["verified"]:
            verified.append((candidate, result))

    verification["candidates"] = results
    verification["pool_was_ready"] = pool_was_ready
    verification["pool"] = pool.summary()
    if valid_history < config.reverse_min_history and verification["error"] is None:
        verification["error"] = "insufficient_history_for_selection"
    if verified:
        candidate, result = max(
            verified,
            key=lambda item: (
                item[1]["score"]["mean_iou"],
                item[1]["pool_iou"] if item[1]["pool_iou"] is not None else -np.inf,
                item[0].template_similarity if item[0].template_similarity is not None else -np.inf,
                item[0].predicted_iou if item[0].predicted_iou is not None else -np.inf,
            ),
        )
        verification.update(
            {
                "verified": True,
                "candidate": result["score"],
                "score_margin": result["score_margin"],
                "selected_key": candidate.key,
                "selected_kind": candidate.kind,
                "selected_candidate": candidate,
            }
        )

    if config.association_mode == "joint":
        kalman_result["verified"] = bool(
            kalman_result["verified"] and association["target_candidate_key"] == "kalman"
        )
    if kalman_result["verified"]:
        selected_mean = verification["candidate"]["mean_iou"] if verification["verified"] else None
        kalman_mean = kalman_result["score"]["mean_iou"]
        if selected_mean is None or kalman_mean > selected_mean:
            verification.update(
                {
                    "verified": True,
                    "candidate": kalman_result["score"],
                    "score_margin": kalman_result["score_margin"],
                    "selected_key": "kalman",
                    "selected_kind": "kalman",
                    "selected_candidate": kalman,
                }
            )
    timing["total"] = (perf_counter() - total_started) * 1000.0
    return verification
