from __future__ import annotations

import numpy as np

from ..geometry import clean_bbox, mask_to_bbox
from ..types import BackendFrame, Hypothesis


def _scalar(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    finite = array[np.isfinite(array)]
    return float(finite.reshape(-1)[0]) if finite.size else None


class SAM2Backend:
    def __init__(self, runtime=None, adapter_name="sam2", **runtime_kwargs):
        if runtime is None:
            from .sam2_runtime import SAM2Runtime

            runtime = SAM2Runtime(**runtime_kwargs)
        self.runtime = runtime
        self.adapter_name = str(adapter_name)

    def initialize(self, frame, bbox, mask=None):
        outputs = self.runtime.initialize(frame, bbox, mask)
        primary_mask = outputs.get("pred_mask")
        primary = Hypothesis(
            key="init",
            kind="decoder",
            bbox=clean_bbox(outputs.get("pred_bbox")) or mask_to_bbox(primary_mask),
            mask=primary_mask,
            is_baseline=True,
        )
        return BackendFrame(primary=primary, diagnostics={"adapter": self.adapter_name})

    def track(self, frame):
        outputs = self.runtime.track(frame)
        masks = outputs.get("multimask_logits")
        scores = outputs.get("multimask_ious")
        winner = outputs.get("multimask_winner_idx")
        object_score = _scalar(outputs.get("object_score_logits"))
        decoder = []
        if masks is not None:
            object_masks = masks[0]
            score_values = np.asarray(scores, dtype=np.float32)[0] if scores is not None else []
            for index in range(int(object_masks.shape[0])):
                logits = object_masks[index]
                if hasattr(logits, "detach"):
                    logits = logits.detach().float().cpu().numpy()
                candidate_mask = (np.asarray(logits) > 0).astype(np.uint8)
                decoder.append(
                    Hypothesis(
                        key=f"decoder_{index}",
                        kind="decoder",
                        bbox=mask_to_bbox(candidate_mask),
                        mask=candidate_mask,
                        predicted_iou=float(score_values[index]) if index < len(score_values) else None,
                        object_score=object_score,
                        is_baseline=index == winner,
                    )
                )
        primary_mask = outputs.get("pred_mask")
        primary_score = None
        if scores is not None:
            values = np.asarray(scores, dtype=np.float32)
            object_values = values[0] if values.ndim > 1 else values
            if winner is not None and 0 <= int(winner) < len(object_values):
                value = float(object_values[int(winner)])
                primary_score = value if np.isfinite(value) else None
            else:
                finite = object_values[np.isfinite(object_values)]
                primary_score = float(finite.max()) if finite.size else None
        primary = Hypothesis(
            key=f"decoder_{winner}" if winner is not None else "baseline",
            kind="decoder",
            bbox=clean_bbox(outputs.get("pred_bbox")) or mask_to_bbox(primary_mask),
            mask=primary_mask,
            predicted_iou=primary_score,
            object_score=object_score,
            is_baseline=True,
        )
        outputs["multimask_logits"] = None
        return BackendFrame(
            primary=primary,
            decoder_hypotheses=decoder,
            object_score=object_score,
            diagnostics={
                "adapter": self.adapter_name,
                "winner_key": primary.key,
                "multimask_ious": scores,
            },
        )

    def generate_amg_hypotheses(self, frame):
        hypotheses = []
        for proposal in self.runtime.generate_amg(frame):
            hypotheses.append(
                Hypothesis(
                    key=f"amg_{proposal['index']}",
                    kind="amg",
                    bbox=proposal["bbox"],
                    mask=proposal["mask"],
                    predicted_iou=proposal["predicted_iou"],
                    template_similarity=proposal["template_similarity"],
                    stability_score=proposal["stability_score"],
                )
            )
        return hypotheses

    def reverse_track_window(self, frames, bbox=None, mask=None):
        return self.runtime.reverse_track_window(frames, bbox=bbox, mask=mask)

    def configure_reverse_feature_cache(self, enabled, max_frames=None):
        return self.runtime.configure_reverse_feature_cache(enabled, max_frames=max_frames)

    def reverse_feature_cache_stats(self):
        return self.runtime.reverse_feature_cache_stats()

    def replace_current_memory(self, mask):
        return self.runtime.replace_current_memory(mask)
