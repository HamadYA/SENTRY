import numpy as np

from sentry_tracking import BackendFrame, Hypothesis, SENTRYConfig, SENTRYTracker
from sentry_tracking.geometry import mask_to_bbox


def make_mask(x, y):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[y : y + 10, x : x + 10] = 1
    return mask


class FakeBackend:
    def __init__(self):
        self.frame_idx = 0
        self.memory_writes = []
        self.reverse_cache_config = None

    def configure_reverse_feature_cache(self, enabled, max_frames=None):
        self.reverse_cache_config = (enabled, max_frames)

    @staticmethod
    def _hypothesis(key, mask, score, baseline=False):
        return Hypothesis(
            key=key,
            kind="decoder",
            bbox=mask_to_bbox(mask),
            mask=mask,
            predicted_iou=score,
            object_score=1.0,
            is_baseline=baseline,
        )

    def initialize(self, frame, bbox, mask=None):
        target = make_mask(10, 10)
        return BackendFrame(primary=self._hypothesis("init", target, 1.0, True))

    def track(self, frame):
        self.frame_idx += 1
        if self.frame_idx < 3:
            target = make_mask(10, 10)
            primary = self._hypothesis("decoder_0", target, 0.9, True)
            return BackendFrame(primary=primary, decoder_hypotheses=[primary], object_score=1.0)
        wrong = make_mask(75, 75)
        correct = make_mask(10, 10)
        primary = self._hypothesis("decoder_0", wrong, 0.1, True)
        alternative = self._hypothesis("decoder_1", correct, 0.9)
        return BackendFrame(primary=primary, decoder_hypotheses=[primary, alternative], object_score=1.0)

    def generate_amg_hypotheses(self, frame):
        return []

    def reverse_track_window(self, frames, bbox=None, mask=None):
        prompt_bbox = mask_to_bbox(mask) if mask is not None else bbox
        return {index: list(prompt_bbox) for index in range(len(frames) - 1)}

    def replace_current_memory(self, mask):
        self.memory_writes.append(np.asarray(mask).copy())
        return {"applied": True}


def run_until_failure(memory_policy, rescue_enabled=True):
    backend = FakeBackend()
    config = SENTRYConfig(
        amg_enabled=False,
        memory_policy=memory_policy,
        rescue_enabled=rescue_enabled,
    )
    tracker = SENTRYTracker(backend, config)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    tracker.initialize(frame, [10, 10, 10, 10])
    tracker.track(frame)
    tracker.track(frame)
    return backend, tracker.track(frame)


def test_verified_severe_failure_rescues_output_without_default_memory_write():
    backend, result = run_until_failure("baseline")
    assert result.bbox == [10.0, 10.0, 10.0, 10.0]
    assert result.source == "sentry_decoder_rescue_jump"
    assert backend.memory_writes == []
    assert result.diagnostics["timing_ms"]["backend_forward"] >= 0
    assert result.diagnostics["timing_ms"]["reverse_verification"] >= 0
    assert result.diagnostics["verification"]["timing_ms"]["candidate_reverse"] >= 0


def test_severe_rescue_memory_policy_writes_only_accepted_mask():
    backend, result = run_until_failure("severe_rescue")
    assert result.source == "sentry_decoder_rescue_jump"
    assert len(backend.memory_writes) == 1
    assert mask_to_bbox(backend.memory_writes[0]) == [10.0, 10.0, 10.0, 10.0]
    assert result.diagnostics["memory_update"]["applied"] is True


def test_shadow_mode_logs_rescue_but_preserves_baseline_output_and_memory():
    backend, result = run_until_failure("severe_rescue", rescue_enabled=False)
    assert result.bbox == [75.0, 75.0, 10.0, 10.0]
    assert result.source == "baseline"
    assert result.diagnostics["would_rescue"] is True
    assert backend.memory_writes == []


def test_reverse_feature_cache_configuration_is_forwarded_to_backend():
    backend = FakeBackend()
    SENTRYTracker(
        backend,
        SENTRYConfig(reverse_feature_cache_enabled=True, reverse_frames=4),
    )

    assert backend.reverse_cache_config == (True, 5)
