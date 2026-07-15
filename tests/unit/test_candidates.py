import numpy as np

from sentry_tracking.candidates import filter_amg_hypotheses, filter_decoder_hypotheses, soft_nms
from sentry_tracking.config import SENTRYConfig
from sentry_tracking.types import Hypothesis


def mask_at(x, y, size=10):
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[y : y + size, x : x + size] = 1
    return mask


def hypothesis(key, kind, x, score=None, similarity=None, baseline=False):
    mask = mask_at(x, 10)
    return Hypothesis(
        key=key,
        kind=kind,
        bbox=[x, 10, 10, 10],
        mask=mask,
        predicted_iou=score,
        template_similarity=similarity,
        is_baseline=baseline,
    )


def test_decoder_relative_filter_and_visibility():
    config = SENTRYConfig()
    inputs = [
        hypothesis("a", "decoder", 10, score=0.9, baseline=True),
        hypothesis("b", "decoder", 30, score=0.8),
        hypothesis("c", "decoder", 50, score=0.5),
    ]
    retained, diagnostics = filter_decoder_hypotheses(inputs, 1.0, 100.0, 100, 100, config)
    assert [item.key for item in retained] == ["a", "b"]
    assert diagnostics["best_iou"] == 0.9
    hidden, _ = filter_decoder_hypotheses(inputs, -1.0, 100.0, 100, 100, config)
    assert hidden == []


def test_amg_relative_filter_and_soft_nms():
    config = SENTRYConfig()
    proposals = [
        hypothesis("amg_0", "amg", 10, similarity=0.9),
        hypothesis("amg_1", "amg", 10, similarity=0.85),
        hypothesis("amg_2", "amg", 50, similarity=0.5),
    ]
    retained, _ = filter_amg_hypotheses(proposals, 100.0, 100, 100, config)
    assert [item.key for item in retained] == ["amg_0", "amg_1"]
    filtered, diagnostics = soft_nms(retained, config)
    assert [item.key for item in filtered] == ["amg_0"]
    assert diagnostics["suppressed_keys"] == ["amg_1"]
