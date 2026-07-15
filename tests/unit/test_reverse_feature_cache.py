from collections import OrderedDict

import torch

from sentry_tracking.backends.sam2_runtime import SAM2Runtime


def _feature(value):
    image = torch.full((1, 3, 2, 2), float(value))
    backbone = {"backbone_fpn": [image], "vision_pos_enc": [image]}
    return image, backbone


def _runtime(frame_idx, features):
    runtime = SAM2Runtime.__new__(SAM2Runtime)
    runtime.frame_idx = frame_idx
    runtime.state = None
    runtime.reverse_feature_cache_enabled = True
    runtime._reverse_feature_cache_capacity = 3
    runtime._reverse_feature_cache = OrderedDict(features)
    runtime._reverse_cache_hits = 0
    runtime._reverse_cache_misses = 0
    runtime._last_reverse_cache = {"requests": 0, "hits": 0, "misses": 0}
    return runtime


def test_reverse_window_reindexes_and_reuses_forward_features():
    old = _feature(2)
    first = _feature(3)
    second = _feature(4)
    runtime = _runtime(4, [(2, old), (3, first), (4, second)])

    images, cached = runtime._reverse_window_inputs([object(), object()])

    assert cached[0] is first
    assert cached[1] is second
    assert images[0].data_ptr() == first[0][0].data_ptr()
    assert list(runtime._reverse_feature_cache) == [3, 4]
    assert runtime.reverse_feature_cache_stats()["hit_rate"] == 1.0


def test_reverse_window_computes_each_missing_feature_only_once():
    first = _feature(0)
    second = _feature(1)
    runtime = _runtime(1, [(0, first)])
    runtime._compute_reverse_feature = lambda frame: second

    _, cached = runtime._reverse_window_inputs([object(), object()])
    _, repeated = runtime._reverse_window_inputs([object(), object()])

    assert cached[1] is second
    assert repeated[1] is second
    stats = runtime.reverse_feature_cache_stats()
    assert stats["hits"] == 3
    assert stats["misses"] == 1
