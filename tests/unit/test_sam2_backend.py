import numpy as np
import pytest
import torch

from sentry_tracking.backends.factory import available_backends, create_backend
from sentry_tracking.backends.forked_sam2 import SAMURAIBackend
from sentry_tracking.backends.forked_sam2_runtime import FORK_SPECS, ForkedSAM2Runtime
from sentry_tracking.backends.sam2 import SAM2Backend


class FakeRuntime:
    def initialize(self, frame, bbox, mask=None):
        initial = np.zeros((20, 20), dtype=np.uint8)
        initial[2:7, 3:9] = 1
        return {"pred_mask": initial, "pred_bbox": [3, 2, 6, 5]}

    def track(self, frame):
        masks = torch.full((1, 3, 20, 20), -1.0)
        masks[0, 0, 2:7, 3:9] = 1
        masks[0, 1, 5:10, 10:15] = 1
        masks[0, 2, 12:17, 12:17] = 1
        return {
            "pred_mask": (masks[0, 0] > 0).to(torch.uint8).numpy(),
            "pred_bbox": [3, 2, 6, 5],
            "multimask_logits": masks,
            "multimask_ious": [[0.9, 0.8, 0.3]],
            "multimask_winner_idx": 0,
            "object_score_logits": [[2.0]],
        }

    def generate_amg(self, frame):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[4:9, 4:9] = 1
        return [
            {
                "index": 7,
                "mask": mask,
                "bbox": [4, 4, 5, 5],
                "template_similarity": 0.85,
                "predicted_iou": 0.8,
                "stability_score": 0.95,
            }
        ]

    def reverse_track_window(self, frames, bbox=None, mask=None):
        return {}

    def replace_current_memory(self, mask):
        return {"applied": True}


def test_sam2_backend_normalizes_decoder_and_amg_outputs():
    backend = SAM2Backend(runtime=FakeRuntime())
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    initialized = backend.initialize(frame, [3, 2, 6, 5])
    assert initialized.primary.bbox == [3.0, 2.0, 6.0, 5.0]
    output = backend.track(frame)
    assert output.primary.key == "decoder_0"
    assert len(output.decoder_hypotheses) == 3
    assert output.object_score == 2.0
    assert output.decoder_hypotheses[1].bbox == [10.0, 5.0, 5.0, 5.0]
    amg = backend.generate_amg_hypotheses(frame)
    assert amg[0].key == "amg_7"
    assert amg[0].template_similarity == 0.85


def test_backend_uses_native_winner_score_instead_of_maximum_iou():
    runtime = FakeRuntime()
    original_track = runtime.track

    def track(frame):
        output = original_track(frame)
        output["multimask_winner_idx"] = 1
        output["pred_mask"] = (output["multimask_logits"][0, 1] > 0).to(torch.uint8).numpy()
        output["pred_bbox"] = [10, 5, 5, 5]
        return output

    runtime.track = track
    output = SAMURAIBackend(runtime=runtime).track(np.zeros((20, 20, 3), dtype=np.uint8))

    assert output.primary.key == "decoder_1"
    assert output.primary.predicted_iou == pytest.approx(0.8)
    assert output.diagnostics["adapter"] == "samurai"


def test_fork_runtime_matches_the_host_selected_mask():
    runtime = ForkedSAM2Runtime.__new__(ForkedSAM2Runtime)
    masks = torch.zeros((1, 3, 4, 4))
    masks[0, 0, :2, :2] = 2
    masks[0, 1, 1:3, 1:3] = 2
    masks[0, 2, 2:, 2:] = 2
    capture = {"masks": masks, "ious": torch.tensor([[0.9, 0.8, 0.7]])}

    assert runtime._winner_index(capture, {"pred_masks": masks[:, 2:3]}) == 2


def test_fork_runtime_selects_the_matching_decoder_batch():
    runtime = ForkedSAM2Runtime.__new__(ForkedSAM2Runtime)
    masks = torch.zeros((2, 3, 4, 4))
    masks[0, 0, :2, :2] = 2
    masks[1, 1, 1:3, 1:3] = 3
    capture = {"masks": masks, "ious": torch.tensor([[0.9, 0.8, 0.7], [0.6, 0.5, 0.4]])}

    assert runtime._winner_selection(capture, {"pred_masks": masks[1:2, 1:2]}) == (1, 1)


@pytest.mark.parametrize(
    ("fork", "dataset", "expected"),
    (
        ("samurai", "lasot", "configs/samurai/sam2.1_hiera_t.yaml"),
        ("samite", "lasot", "configs/samite/samite_hiera_t.yaml"),
        ("him2sam", "lasot", "configs/him2sam/lasot/sam2.1_hiera_t.yaml"),
        ("him2sam", "lasot_ext", "configs/him2sam/lasotext/sam2.1_hiera_t.yaml"),
        ("him2sam", "didi", "configs/him2sam/vot/sam2.1_hiera_l.yaml"),
    ),
)
def test_fork_runtime_uses_explicit_native_model_config(fork, dataset, expected):
    runtime = ForkedSAM2Runtime.__new__(ForkedSAM2Runtime)
    runtime.spec = FORK_SPECS[fork]
    runtime.dataset = dataset
    tracker_name = "sam21-L" if dataset == "didi" else "sam21-T"

    assert runtime._default_model_config(tracker_name) == expected


def test_him2sam_rejects_an_unbundled_vot_model_scale():
    runtime = ForkedSAM2Runtime.__new__(ForkedSAM2Runtime)
    runtime.spec = FORK_SPECS["him2sam"]
    runtime.dataset = "didi"

    with pytest.raises(ValueError, match="available only for sam21-L"):
        runtime._default_model_config("sam21-T")


def test_fork_runtime_resets_reverse_only_model_state():
    class Filter:
        max_pre_iou = 0.5
        ct = 12
        debugflag = False

    class Predictor:
        kf_mean = np.ones(8)
        kf_covariance = np.eye(8)
        stable_frames = 9
        frame_cnt = 18
        history = {1: "old"}
        rvcot_mem_selection_highconf_frameidx = [3]
        rvcor_area_list = [100]
        rvcot_filter = Filter()

    predictor = Predictor()
    ForkedSAM2Runtime._reset_model_tracking_state(predictor)

    assert predictor.kf_mean is None
    assert predictor.kf_covariance is None
    assert predictor.stable_frames == 0
    assert predictor.history == {}
    assert predictor.rvcot_mem_selection_highconf_frameidx == []
    assert predictor.rvcor_area_list == []
    assert predictor.rvcot_filter.max_pre_iou == -1


def test_fork_runtime_uses_a_distinct_reusable_reverse_host():
    runtime = ForkedSAM2Runtime.__new__(ForkedSAM2Runtime)
    runtime.host = object()
    runtime.reverse_host = None
    created = []

    def new_host():
        created.append(object())
        return created[-1]

    runtime._new_host = new_host
    first = runtime._ensure_reverse_host()

    assert first is not runtime.host
    assert runtime._ensure_reverse_host() is first
    assert len(created) == 1


def test_release_exposes_every_sentry_backend():
    assert {"sam2", "samurai", "dam4sam", "samite", "him2sam"} <= set(available_backends())


@pytest.mark.parametrize("name", ("sam2", "samurai", "dam4sam", "samite", "him2sam"))
def test_factory_constructs_every_backend_without_replacing_its_runtime(name):
    runtime = FakeRuntime()
    backend = create_backend(name, runtime=runtime)

    assert backend.runtime is runtime
    assert backend.adapter_name == name
