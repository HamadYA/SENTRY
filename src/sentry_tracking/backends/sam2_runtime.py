from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f
import torchvision.transforms.functional as vision_f

from ..geometry import mask_to_bbox


MODEL_CONFIGS = {
    "sam21-L": "sam2.1_hiera_l.yaml",
    "sam21-B": "sam2.1_hiera_b+.yaml",
    "sam21-S": "sam2.1_hiera_s.yaml",
    "sam21-T": "sam2.1_hiera_t.yaml",
}
CHECKPOINTS = {
    "sam21-L": "sam2.1_hiera_large.pt",
    "sam21-B": "sam2.1_hiera_base_plus.pt",
    "sam21-S": "sam2.1_hiera_small.pt",
    "sam21-T": "sam2.1_hiera_tiny.pt",
}


def _release_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _activate_bundled_sam2(baseline_root: str | Path | None):
    root = Path(baseline_root) if baseline_root else _release_root() / "baseline" / "sam2"
    root = root.resolve()
    if not (root / "sam2" / "build_sam.py").exists():
        raise FileNotFoundError(f"SAM2 package not found under {root}")
    existing = sys.modules.get("sam2")
    if existing is not None:
        loaded_from = Path(existing.__file__).resolve().parent
        if loaded_from != root / "sam2":
            raise RuntimeError(
                f"A different SAM2 package is already loaded from {loaded_from}. "
                "Run each bundled baseline in a separate process."
            )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


class SAM2Runtime:
    """Enhanced vanilla SAM2 runtime used by the release adapter."""

    def __init__(
        self,
        tracker_name: str = "sam21-T",
        checkpoint: str | Path | None = None,
        model_config: str | None = None,
        device: str = "cuda:0",
        baseline_root: str | Path | None = None,
        dataset: str | None = None,
    ):
        del dataset
        self.baseline_root = _activate_bundled_sam2(baseline_root)
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2_video_predictor

        if tracker_name not in MODEL_CONFIGS and (checkpoint is None or model_config is None):
            raise ValueError(f"Unknown tracker '{tracker_name}'. Available: {', '.join(MODEL_CONFIGS)}")
        self.release_root = _release_root()
        self.checkpoint = str(checkpoint or self.release_root / "checkpoints" / CHECKPOINTS[tracker_name])
        self.model_config = model_config or MODEL_CONFIGS[tracker_name]
        self.device = torch.device(device)
        self.image_size = 1024
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None].to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None].to(self.device)
        self.predictor = build_sam2_video_predictor(
            self.model_config, self.checkpoint, device=self.device
        )
        self._amg_class = SAM2AutomaticMaskGenerator
        self.amg_generator = None
        self.template_feature = None
        self.frame_idx = -1
        self.state = None
        self.reverse_feature_cache_enabled = False
        self._reverse_feature_cache_capacity = 10
        self._reverse_feature_cache = OrderedDict()
        self._reverse_cache_hits = 0
        self._reverse_cache_misses = 0
        self._last_reverse_cache = {"requests": 0, "hits": 0, "misses": 0}

    def _new_state(self):
        return {
            "images": {},
            "num_frames": 0,
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
            "video_height": None,
            "video_width": None,
            "device": self.device,
            "storage_device": self.device,
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "consolidated_frame_inds": {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()},
            "tracking_has_started": False,
            "frames_already_tracked": {},
            "frames_tracked_per_obj": {},
        }

    def _prepare_image(self, image, state=None):
        state = state or self.state
        tensor = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).to(state["device"])
        tensor = tensor.permute(2, 0, 1).float() / 255.0
        tensor = vision_f.resize(tensor, (self.image_size, self.image_size))
        return (tensor - self.mean) / self.std

    def configure_reverse_feature_cache(self, enabled, max_frames=None):
        self.reverse_feature_cache_enabled = bool(enabled)
        if max_frames is not None:
            if int(max_frames) < 2:
                raise ValueError("reverse feature cache must retain at least two frames")
            self._reverse_feature_cache_capacity = int(max_frames)
        self._reset_reverse_feature_cache()
        return self.reverse_feature_cache_stats()

    def _reset_reverse_feature_cache(self):
        self._reverse_feature_cache.clear()
        self._reverse_cache_hits = 0
        self._reverse_cache_misses = 0
        self._last_reverse_cache = {"requests": 0, "hits": 0, "misses": 0}

    def _remember_current_feature(self):
        if not self.reverse_feature_cache_enabled or self.state is None:
            return
        feature = self.state.get("cached_features", {}).get(self.frame_idx)
        if feature is None:
            return
        self._reverse_feature_cache[self.frame_idx] = feature
        self._reverse_feature_cache.move_to_end(self.frame_idx)
        while len(self._reverse_feature_cache) > self._reverse_feature_cache_capacity:
            self._reverse_feature_cache.popitem(last=False)

    def _compute_reverse_feature(self, frame):
        prepared = self._prepare_image(frame)
        image = prepared.to(self.device).float().unsqueeze(0)
        backbone = self.predictor.forward_image(image)
        return image, backbone

    def _reverse_window_inputs(self, frames):
        if not self.reverse_feature_cache_enabled:
            self._last_reverse_cache = {"requests": 0, "hits": 0, "misses": 0}
            return {
                index: self._prepare_image(frame) for index, frame in enumerate(frames)
            }, {}

        global_start = self.frame_idx - len(frames) + 1
        images = {}
        cached_features = {}
        hits = 0
        misses = 0
        active_indices = set()
        for local_idx, frame in enumerate(frames):
            global_idx = global_start + local_idx
            active_indices.add(global_idx)
            feature = self._reverse_feature_cache.get(global_idx)
            if feature is None:
                feature = self._compute_reverse_feature(frame)
                self._reverse_feature_cache[global_idx] = feature
                misses += 1
            else:
                hits += 1
            self._reverse_feature_cache.move_to_end(global_idx)
            images[local_idx] = feature[0][0]
            cached_features[local_idx] = feature

        for global_idx in list(self._reverse_feature_cache):
            if global_idx not in active_indices:
                del self._reverse_feature_cache[global_idx]

        self._reverse_cache_hits += hits
        self._reverse_cache_misses += misses
        self._last_reverse_cache = {
            "requests": len(frames),
            "hits": hits,
            "misses": misses,
        }
        return images, cached_features

    def reverse_feature_cache_stats(self):
        requests = self._reverse_cache_hits + self._reverse_cache_misses
        return {
            "enabled": self.reverse_feature_cache_enabled,
            "capacity": self._reverse_feature_cache_capacity,
            "entries": len(self._reverse_feature_cache),
            "requests": requests,
            "hits": self._reverse_cache_hits,
            "misses": self._reverse_cache_misses,
            "hit_rate": self._reverse_cache_hits / requests if requests else None,
            "last_window": dict(self._last_reverse_cache),
        }

    @torch.inference_mode()
    def initialize(self, image, bbox, mask=None):
        self.frame_idx = 0
        self._reset_reverse_feature_cache()
        self.state = self._new_state()
        width, height = image.size
        self.state.update(
            {
                "images": {0: self._prepare_image(image, self.state)},
                "num_frames": 1,
                "video_height": height,
                "video_width": width,
            }
        )
        self.predictor.reset_state(self.state)
        self.predictor._get_image_feature(self.state, frame_idx=0, batch_size=1)
        if mask is None:
            mask = self._estimate_mask_from_box(bbox)
        _, _, logits = self.predictor.add_new_mask(
            inference_state=self.state, frame_idx=0, obj_id=0, mask=mask
        )
        prediction = (logits[0, 0] > 0).to(torch.uint8).cpu().numpy()
        self.template_feature = self._pool_cached_feature(prediction)
        self._remember_current_feature()
        self.state["images"].pop(0, None)
        return {"pred_mask": prediction, "pred_bbox": mask_to_bbox(prediction)}

    @torch.inference_mode()
    def track(self, image):
        self.frame_idx += 1
        self.state["num_frames"] += 1
        self.state["images"][self.frame_idx] = self._prepare_image(image)
        outputs = self.predictor.propagate_in_video(
            self.state,
            start_frame_idx=self.frame_idx,
            max_frame_num_to_track=0,
            return_multimasks=True,
        )
        for _, _, mask_logits, side_outputs in outputs:
            mask = (mask_logits[0, 0] > 0).to(torch.uint8).cpu().numpy()
            ious = side_outputs["multimask_ious"]
            iou_values = ious.detach().float().cpu().numpy().tolist() if ious is not None else None
            object_score = side_outputs["object_score_logits"]
            object_values = (
                object_score.detach().float().cpu().numpy().tolist() if object_score is not None else None
            )
            winner = int(np.argmax(iou_values[0])) if iou_values and iou_values[0] else None
            self._remember_current_feature()
            self.state["images"].pop(self.frame_idx, None)
            return {
                "pred_mask": mask,
                "pred_bbox": mask_to_bbox(mask),
                "multimask_logits": side_outputs["video_res_multimasks"],
                "multimask_ious": iou_values,
                "multimask_winner_idx": winner,
                "object_score_logits": object_values,
            }
        raise RuntimeError("SAM2 propagation produced no output")

    def _cached_embedding(self):
        cached = self.state.get("cached_features", {}).get(self.frame_idx)
        if cached is None:
            return None
        _, backbone = cached
        _, features, _, sizes = self.predictor._prepare_backbone_features(backbone)
        embedding = features[-1]
        if self.predictor.directly_add_no_mem_embed:
            embedding = embedding + self.predictor.no_mem_embed
        height, width = sizes[-1]
        return embedding.permute(1, 2, 0)[0].reshape(-1, height, width).float()

    def _pool_cached_feature(self, mask):
        embedding = self._cached_embedding()
        if embedding is None or mask is None:
            return None
        weights = torch.as_tensor(np.asarray(mask) > 0, dtype=torch.float32, device=embedding.device)[None, None]
        weights = torch_f.interpolate(weights, size=embedding.shape[-2:], mode="area")[0, 0]
        if weights.sum().item() <= 1e-6:
            return None
        pooled = (embedding * weights[None]).sum((1, 2)) / weights.sum()
        return torch_f.normalize(pooled, dim=0).detach()

    @torch.inference_mode()
    def generate_amg(self, image):
        if self.template_feature is None:
            return []
        embedding = self._cached_embedding()
        if embedding is None:
            return []
        if self.amg_generator is None:
            self.amg_generator = self._amg_class(
                model=self.predictor,
                points_per_side=32,
                points_per_batch=64,
                pred_iou_thresh=0.8,
                stability_score_thresh=0.95,
                crop_n_layers=0,
                output_mode="binary_mask",
            )
        annotations = self.amg_generator.generate(np.asarray(image.convert("RGB")))
        proposals = []
        for index, annotation in enumerate(annotations):
            mask = np.asarray(annotation["segmentation"], dtype=np.uint8)
            weights = torch.as_tensor(mask > 0, dtype=torch.float32, device=embedding.device)[None, None]
            weights = torch_f.interpolate(weights, size=embedding.shape[-2:], mode="area")[0, 0]
            if weights.sum().item() <= 1e-6:
                continue
            pooled = (embedding * weights[None]).sum((1, 2)) / weights.sum()
            similarity = torch.dot(torch_f.normalize(pooled, dim=0), self.template_feature).item()
            proposals.append(
                {
                    "index": index,
                    "mask": mask,
                    "bbox": mask_to_bbox(mask),
                    "template_similarity": float(similarity),
                    "predicted_iou": float(annotation.get("predicted_iou", 0.0)),
                    "stability_score": float(annotation.get("stability_score", 0.0)),
                }
            )
        return proposals

    @torch.inference_mode()
    def reverse_track_window(self, frames, bbox=None, mask=None):
        if len(frames) < 2:
            raise ValueError("Reverse tracking requires at least two frames")
        width, height = frames[0].size
        if any(frame.size != (width, height) for frame in frames):
            raise ValueError("All reverse frames must have equal dimensions")
        reverse_state = self._new_state()
        reverse_images, cached_features = self._reverse_window_inputs(frames)
        reverse_state.update(
            {
                "images": reverse_images,
                "cached_features": cached_features,
                "num_frames": len(frames),
                "video_height": height,
                "video_width": width,
            }
        )
        start = len(frames) - 1
        try:
            if mask is not None:
                prompt = (np.asarray(mask) > 0).astype(np.uint8)
                if prompt.shape != (height, width) or prompt.sum() == 0:
                    raise ValueError(f"Reverse prompt mask must be non-empty with shape {(height, width)}")
                self.predictor.add_new_mask(reverse_state, start, 0, prompt)
            else:
                x, y, box_width, box_height = [float(value) for value in bbox]
                x1, y1 = min(max(x, 0.0), width), min(max(y, 0.0), height)
                x2 = min(max(x + box_width, 0.0), width)
                y2 = min(max(y + box_height, 0.0), height)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError("Reverse prompt box is empty after clipping")
                box = np.asarray([x1, y1, x2, y2], dtype=np.float32)
                self.predictor.add_new_points_or_box(reverse_state, start, 0, box=box)
            reverse_boxes = {}
            for frame_idx, _, logits in self.predictor.propagate_in_video(
                reverse_state,
                start_frame_idx=start,
                max_frame_num_to_track=start,
                reverse=True,
                return_multimasks=False,
            ):
                reverse_mask = (logits[0, 0] > 0).to(torch.uint8).cpu().numpy()
                reverse_boxes[int(frame_idx)] = mask_to_bbox(reverse_mask)
            return reverse_boxes
        finally:
            reverse_state.clear()

    @torch.inference_mode()
    def replace_current_memory(self, mask):
        mask = (np.asarray(mask) > 0).astype(np.uint8)
        expected = (int(self.state["video_height"]), int(self.state["video_width"]))
        if mask.shape != expected or mask.sum() == 0:
            raise ValueError(f"Memory mask must be non-empty with shape {expected}")
        obj_idx = self.state["obj_id_to_idx"].get(0)
        current = self.state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"].get(self.frame_idx)
        if current is None:
            raise RuntimeError(f"No non-conditioning output for frame {self.frame_idx}")
        if self.frame_idx not in self.state["cached_features"]:
            raise RuntimeError(f"Frame {self.frame_idx} image features are not cached")

        tensor = torch.as_tensor(mask, dtype=torch.float32, device=self.device)[None, None]
        tensor = torch_f.interpolate(
            tensor,
            size=(self.predictor.image_size, self.predictor.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        high_logits = (tensor >= 0.5).float() * 20.0 - 10.0
        object_score = torch.full_like(current["object_score_logits"].to(self.device), 10.0)
        features, position = self.predictor._run_memory_encoder(
            self.state, self.frame_idx, 1, high_logits, object_score, True
        )
        low_logits = torch_f.interpolate(
            high_logits,
            size=current["pred_masks"].shape[-2:],
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        current["pred_masks"] = low_logits.to(self.state["storage_device"], dtype=current["pred_masks"].dtype)
        current["maskmem_features"] = features
        current["maskmem_pos_enc"] = position
        current["object_score_logits"] = object_score.to(self.state["storage_device"])
        return {"applied": True, "frame_idx": self.frame_idx, "mask_area": int(mask.sum())}

    @torch.inference_mode()
    def _estimate_mask_from_box(self, bbox):
        from sam2.utils.transforms import SAM2Transforms

        _, _, features, _, sizes = self.predictor._get_image_feature(self.state, 0, 1)
        box = np.asarray([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])[None]
        box = torch.as_tensor(box, dtype=torch.float32, device=features[0].device)
        transforms = SAM2Transforms(self.predictor.image_size, 0.0, 0.0, 0.0)
        box = transforms.transform_boxes(
            box,
            normalize=True,
            orig_hw=(self.state["video_height"], self.state["video_width"]),
        )
        points = box.reshape(-1, 2, 2)
        labels = torch.tensor([[2, 3]], dtype=torch.int32, device=box.device).repeat(box.size(0), 1)
        sparse, dense = self.predictor.sam_prompt_encoder(points=(points, labels), boxes=None, masks=None)
        high_res = []
        for index in range(2):
            _, batch, channels = features[index].shape
            high_res.append(features[index].permute(1, 2, 0).view(batch, channels, *sizes[index]))
        image_embedding = features[2]
        if self.predictor.directly_add_no_mem_embed:
            image_embedding = image_embedding + self.predictor.no_mem_embed
        _, batch, channels = image_embedding.shape
        image_embedding = image_embedding.permute(1, 2, 0).view(batch, channels, *sizes[2])
        low_masks, _, _, _ = self.predictor.sam_mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.predictor.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res,
        )
        masks = transforms.postprocess_masks(low_masks, (self.state["video_height"], self.state["video_width"]))
        return (masks[0, 0] > 0).to(torch.uint8).cpu().numpy()
