from __future__ import annotations

import importlib.util
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_f

from ..geometry import mask_to_bbox


CHECKPOINTS = {
    "sam21-L": "sam2.1_hiera_large.pt",
    "sam21-B": "sam2.1_hiera_base_plus.pt",
    "sam21-S": "sam2.1_hiera_small.pt",
    "sam21-T": "sam2.1_hiera_tiny.pt",
    "sam2-L": "sam2_hiera_large.pt",
    "sam2-B": "sam2_hiera_base_plus.pt",
    "sam2-S": "sam2_hiera_small.pt",
    "sam2-T": "sam2_hiera_tiny.pt",
}


@dataclass(frozen=True)
class ForkSpec:
    key: str
    directory: str
    model_configs: dict[str, str]


FORK_SPECS = {
    "samurai": ForkSpec(
        key="samurai",
        directory="SAMURAI",
        model_configs={
            "sam21-L": "configs/samurai/sam2.1_hiera_l.yaml",
            "sam21-B": "configs/samurai/sam2.1_hiera_b+.yaml",
            "sam21-S": "configs/samurai/sam2.1_hiera_s.yaml",
            "sam21-T": "configs/samurai/sam2.1_hiera_t.yaml",
        },
    ),
    "dam4sam": ForkSpec(
        key="dam4sam",
        directory="DAM4SAM",
        model_configs={
            "sam21-L": "sam21pp_hiera_l.yaml",
            "sam21-B": "sam21pp_hiera_b+.yaml",
            "sam21-S": "sam21pp_hiera_s.yaml",
            "sam21-T": "sam21pp_hiera_t.yaml",
            "sam2-L": "sam2pp_hiera_l.yaml",
            "sam2-B": "sam2pp_hiera_b+.yaml",
            "sam2-S": "sam2pp_hiera_s.yaml",
            "sam2-T": "sam2pp_hiera_t.yaml",
        },
    ),
    "samite": ForkSpec(
        key="samite",
        directory="SAMITE",
        model_configs={
            "sam21-L": "configs/samite/samite_hiera_l.yaml",
            "sam21-B": "configs/samite/samite_hiera_b+.yaml",
            "sam21-S": "configs/samite/samite_hiera_s.yaml",
            "sam21-T": "configs/samite/samite_hiera_t.yaml",
        },
    ),
    "him2sam": ForkSpec(
        key="him2sam",
        directory="HiM2SAM",
        model_configs={
            "sam21-L": "sam2.1_hiera_l.yaml",
            "sam21-B": "sam2.1_hiera_b+.yaml",
            "sam21-S": "sam2.1_hiera_s.yaml",
            "sam21-T": "sam2.1_hiera_t.yaml",
        },
    ),
}


def _release_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _activate_fork(spec: ForkSpec, baseline_root: str | Path | None) -> Path:
    root = Path(baseline_root) if baseline_root else _release_root() / "baseline" / spec.directory
    root = root.expanduser().resolve()
    expected_package = root / "sam2"
    if not (expected_package / "build_sam.py").is_file() or not (root / "tracker.py").is_file():
        raise FileNotFoundError(f"{spec.directory} tracker and SAM2 package were not found under {root}")

    existing = sys.modules.get("sam2")
    if existing is not None:
        loaded_file = getattr(existing, "__file__", None)
        loaded_from = Path(loaded_file).resolve().parent if loaded_file else None
        if loaded_from != expected_package:
            raise RuntimeError(
                f"A different SAM2 package is already loaded from {loaded_from}. "
                "Run each bundled backend in a separate process."
            )

    baseline_parent = root.parent
    for path in (baseline_parent, root):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    return root


def _load_tracker_class(spec: ForkSpec, root: Path):
    module_name = f"sentry_tracking._bundled_{spec.key}_tracker"
    module = sys.modules.get(module_name)
    if module is None:
        module_spec = importlib.util.spec_from_file_location(module_name, root / "tracker.py")
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Unable to load tracker module from {root / 'tracker.py'}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        try:
            module_spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    return module.SAMTracker


def _clone_feature(feature):
    # Backbone outputs are immutable during inference. Retaining their tensors is enough
    # to isolate inference-state dictionaries while avoiding another encoder pass.
    return feature


class ForkedSAM2Runtime:
    """SENTRY runtime that preserves a bundled SAM2-family tracker's forward path."""

    def __init__(
        self,
        fork: str,
        tracker_name: str = "sam21-T",
        checkpoint: str | Path | None = None,
        model_config: str | None = None,
        device: str = "cuda:0",
        baseline_root: str | Path | None = None,
        dataset: str | None = None,
    ):
        try:
            self.spec = FORK_SPECS[fork.strip().lower()]
        except KeyError as error:
            raise ValueError(f"Unknown SAM2 fork {fork!r}. Available: {', '.join(sorted(FORK_SPECS))}") from error
        if tracker_name not in self.spec.model_configs and (checkpoint is None or model_config is None):
            available = ", ".join(sorted(self.spec.model_configs))
            raise ValueError(f"Unknown tracker {tracker_name!r} for {self.spec.key}. Available: {available}")

        self.release_root = _release_root()
        self.baseline_root = _activate_fork(self.spec, baseline_root)
        self.tracker_class = _load_tracker_class(self.spec, self.baseline_root)
        self.tracker_name = tracker_name
        self.checkpoint = str(
            Path(checkpoint).expanduser()
            if checkpoint is not None
            else self.release_root / "checkpoints" / CHECKPOINTS[tracker_name]
        )
        self.dataset = dataset.strip().lower() if dataset else None
        self.model_config = model_config or self._default_model_config(tracker_name)
        self.device = str(device)
        self.host = self._new_host()
        self.predictor = self.host.predictor

        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        self._amg_class = SAM2AutomaticMaskGenerator
        self.amg_generator = None
        self.template_feature = None
        self.frame_idx = -1
        self._capture_decoder = False
        self._decoder_captures = []
        self._decoder_hook = self.predictor.sam_mask_decoder.register_forward_hook(self._decoder_hook_fn)

        self.reverse_host = None
        self.reverse_feature_cache_enabled = False
        self._reverse_feature_cache_capacity = 10
        self._reverse_feature_cache = OrderedDict()
        self._reverse_cache_hits = 0
        self._reverse_cache_misses = 0
        self._last_reverse_cache = {"requests": 0, "hits": 0, "misses": 0}

    def _default_model_config(self, tracker_name):
        config_name = self.spec.model_configs[tracker_name]
        if self.spec.key != "him2sam":
            return config_name

        if self.dataset == "lasot_ext":
            family = "lasotext"
        elif self.dataset == "didi" or (self.dataset and self.dataset.startswith("vot")):
            family = "vot"
        else:
            family = "lasot"
        if family == "vot" and tracker_name != "sam21-L":
            raise ValueError(
                "The bundled HiM2SAM VOT configuration is available only for sam21-L; "
                "select sam21-L or provide --model-config explicitly."
            )
        return f"configs/him2sam/{family}/{config_name}"

    def _new_host(self):
        return self.tracker_class(
            tracker_name=self.tracker_name,
            checkpoint=self.checkpoint,
            model_cfg=self.model_config,
            device=self.device,
        )

    def _decoder_hook_fn(self, _module, _inputs, output):
        if not self._capture_decoder or not isinstance(output, (tuple, list)) or len(output) < 4:
            return
        masks, ious, _, object_score = output[:4]
        if not isinstance(masks, torch.Tensor) or not isinstance(ious, torch.Tensor):
            return
        self._decoder_captures.append(
            {
                "masks": masks.detach(),
                "ious": ious.detach(),
                "object_score": object_score.detach() if isinstance(object_score, torch.Tensor) else None,
            }
        )

    @property
    def state(self):
        return getattr(self.host, "inference_state", None)

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
        self._reverse_feature_cache[self.frame_idx] = _clone_feature(feature)
        self._reverse_feature_cache.move_to_end(self.frame_idx)
        while len(self._reverse_feature_cache) > self._reverse_feature_cache_capacity:
            self._reverse_feature_cache.popitem(last=False)

    def _prepare_with_host(self, host, image):
        return host._prepare_image(image)

    def _compute_reverse_feature(self, frame):
        prepared = self._prepare_with_host(self.host, frame)
        image = prepared.to(torch.device(self.device)).float()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return image, self.predictor.forward_image(image)

    def _reverse_features(self, frames):
        if not self.reverse_feature_cache_enabled:
            self._last_reverse_cache = {"requests": 0, "hits": 0, "misses": 0}
            return {}

        global_start = self.frame_idx - len(frames) + 1
        features = {}
        hits = 0
        misses = 0
        for local_idx, frame in enumerate(frames[:-1]):
            global_idx = global_start + local_idx
            feature = self._reverse_feature_cache.get(global_idx)
            if feature is None:
                feature = self._compute_reverse_feature(frame)
                self._reverse_feature_cache[global_idx] = feature
                misses += 1
            else:
                hits += 1
            self._reverse_feature_cache.move_to_end(global_idx)
            features[local_idx] = _clone_feature(feature)
            while len(self._reverse_feature_cache) > self._reverse_feature_cache_capacity:
                self._reverse_feature_cache.popitem(last=False)

        self._reverse_cache_hits += hits
        self._reverse_cache_misses += misses
        self._last_reverse_cache = {"requests": len(frames) - 1, "hits": hits, "misses": misses}
        return features

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
        outputs = self.host.initialize(image, mask, bbox=bbox)
        prediction = self._normalize_prediction(outputs["pred_mask"], image)
        self.template_feature = self._pool_cached_feature(prediction)
        self._remember_current_feature()
        return {"pred_mask": prediction, "pred_bbox": mask_to_bbox(prediction)}

    @staticmethod
    def _normalize_prediction(mask, image):
        prediction = np.asarray(mask, dtype=np.uint8)
        expected_shape = (int(image.height), int(image.width))
        if prediction.shape != expected_shape:
            resized = torch_f.interpolate(
                torch.as_tensor(prediction, dtype=torch.float32)[None, None],
                size=expected_shape,
                mode="bilinear",
                align_corners=False,
            )
            prediction = (resized[0, 0] >= 0.5).to(torch.uint8).numpy()
        return prediction

    def _select_decoder_capture(self):
        if not self._decoder_captures:
            return None
        return max(
            enumerate(self._decoder_captures),
            key=lambda item: (int(item[1]["masks"].shape[1]), item[0]),
        )[1]

    def _current_output(self, host=None):
        host = host or self.host
        state = getattr(host, "inference_state", None)
        if state is None:
            return None
        frame_idx = int(host.frame_index)
        output = state.get("output_dict", {})
        return output.get("non_cond_frame_outputs", {}).get(frame_idx) or output.get(
            "cond_frame_outputs", {}
        ).get(frame_idx)

    def _winner_selection(self, capture, current):
        masks = capture["masks"].float()
        selected = current.get("pred_masks") if current else None
        if isinstance(selected, torch.Tensor):
            selected = selected[:1].to(masks.device).float()
            if selected.shape[-2:] != masks.shape[-2:]:
                selected = torch_f.interpolate(selected, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            differences = (masks - selected).abs().flatten(2).mean(dim=2)
            flat_index = int(torch.argmin(differences).item())
            batch_index, winner = divmod(flat_index, int(masks.shape[1]))
            if torch.isfinite(differences[batch_index, winner]):
                return batch_index, winner
        ious = capture["ious"]
        if not ious.numel():
            return 0, None
        flat_index = int(torch.argmax(ious).item())
        return divmod(flat_index, int(ious.shape[1]))

    def _winner_index(self, capture, current):
        return self._winner_selection(capture, current)[1]

    @torch.inference_mode()
    def track(self, image):
        self.frame_idx += 1
        self._decoder_captures = []
        self._capture_decoder = True
        try:
            outputs = self.host.track(image)
        finally:
            self._capture_decoder = False

        prediction = self._normalize_prediction(outputs["pred_mask"], image)
        capture = self._select_decoder_capture()
        current = self._current_output()
        self._remember_current_feature()
        result = {"pred_mask": prediction, "pred_bbox": mask_to_bbox(prediction)}
        if capture is not None:
            batch_index, winner = self._winner_selection(capture, current)
            masks = capture["masks"][batch_index : batch_index + 1]
            scores = capture["ious"][batch_index : batch_index + 1]
            _, video_masks = self.predictor._get_orig_video_res_output(self.state, masks)
            ious = scores.float().cpu().numpy().tolist()
            object_score = capture["object_score"]
            if isinstance(object_score, torch.Tensor) and object_score.shape[0] > batch_index:
                object_score = object_score[batch_index : batch_index + 1]
            result.update(
                {
                    "multimask_logits": video_masks,
                    "multimask_ious": ious,
                    "multimask_winner_idx": winner,
                    "object_score_logits": (
                        object_score.float().cpu().numpy().tolist() if object_score is not None else None
                    ),
                }
            )
        self._decoder_captures = []
        return result

    def _cached_embedding(self):
        if self.state is None:
            return None
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

    def _ensure_reverse_host(self):
        if self.reverse_host is None:
            self.reverse_host = self._new_host()
        return self.reverse_host

    @staticmethod
    def _reset_model_tracking_state(predictor):
        defaults = {
            "kf_mean": None,
            "kf_covariance": None,
            "stable_frames": 0,
            "frame_cnt": 0,
            "history": {},
            "rvcot_mem_selection_highconf_frameidx": [],
            "rvcor_area_list": [],
        }
        for name, value in defaults.items():
            if hasattr(predictor, name):
                setattr(predictor, name, value.copy() if isinstance(value, (dict, list)) else value)
        rvcot = getattr(predictor, "rvcot_filter", None)
        if rvcot is not None:
            for name, value in (("max_pre_iou", -1), ("ct", 0), ("debugflag", True)):
                if hasattr(rvcot, name):
                    setattr(rvcot, name, value)

    @torch.inference_mode()
    def reverse_track_window(self, frames, bbox=None, mask=None):
        if len(frames) < 2:
            raise ValueError("Reverse tracking requires at least two frames")
        width, height = frames[0].size
        if any(frame.size != (width, height) for frame in frames):
            raise ValueError("All reverse frames must have equal dimensions")
        prompt_mask = None
        if mask is not None:
            prompt_mask = (np.asarray(mask) > 0).astype(np.uint8)
            if prompt_mask.shape != (height, width) or prompt_mask.sum() == 0:
                raise ValueError(f"Reverse prompt mask must be non-empty with shape {(height, width)}")
            bbox = mask_to_bbox(prompt_mask)
        if bbox is None:
            raise ValueError("Reverse tracking requires a valid mask or box")

        reverse_host = self._ensure_reverse_host()
        self._reset_model_tracking_state(reverse_host.predictor)
        cached_features = self._reverse_features(frames)
        reverse_boxes = {}
        try:
            initialize_kwargs = {"bbox": bbox}
            if self.spec.key == "samite":
                initialize_kwargs["use_mask_prompt"] = prompt_mask is not None
            initialized = reverse_host.initialize(frames[-1], prompt_mask, **initialize_kwargs)
            reverse_boxes[len(frames) - 1] = mask_to_bbox(initialized["pred_mask"])
            for original_idx in range(len(frames) - 2, -1, -1):
                next_frame_idx = int(reverse_host.frame_index) + 1
                feature = cached_features.get(original_idx)
                if feature is not None:
                    reverse_host.inference_state.setdefault("cached_features", {})[next_frame_idx] = feature
                output = reverse_host.track(frames[original_idx])
                reverse_boxes[original_idx] = mask_to_bbox(output["pred_mask"])
            return reverse_boxes
        finally:
            state = getattr(reverse_host, "inference_state", None)
            if isinstance(state, dict):
                state.clear()
            reverse_host.inference_state = None

    @staticmethod
    def _write_output_mask(output, low_logits, features, position, object_score, obj_slice=None):
        if output is None:
            return
        output["pred_masks"] = low_logits if obj_slice is None else low_logits[obj_slice]
        output["maskmem_features"] = features if obj_slice is None else features[obj_slice]
        output["maskmem_pos_enc"] = (
            position if obj_slice is None else [value[obj_slice] for value in position]
        )
        output["object_score_logits"] = object_score if obj_slice is None else object_score[obj_slice]

    def _correct_host_motion(self, high_logits):
        predictor = self.predictor
        if not all(hasattr(predictor, name) for name in ("kf", "kf_mean", "kf_covariance")):
            return False
        positive = torch.argwhere(high_logits[0, 0] > 0)
        if positive.numel() == 0:
            return False
        y_min, x_min = positive.min(dim=0).values
        y_max, x_max = positive.max(dim=0).values
        xyxy = [float(x_min), float(y_min), float(x_max), float(y_max)]
        measurement = predictor.kf.xyxy_to_xyah(xyxy)
        if predictor.kf_mean is None or predictor.kf_covariance is None:
            predictor.kf_mean, predictor.kf_covariance = predictor.kf.initiate(measurement)
        else:
            predictor.kf_mean, predictor.kf_covariance = predictor.kf.update(
                predictor.kf_mean, predictor.kf_covariance, measurement
            )
        if hasattr(predictor, "stable_frames"):
            predictor.stable_frames = max(1, int(predictor.stable_frames))
        return True

    @torch.inference_mode()
    def replace_current_memory(self, mask):
        mask = (np.asarray(mask) > 0).astype(np.uint8)
        expected = (int(self.state["video_height"]), int(self.state["video_width"]))
        if mask.shape != expected or mask.sum() == 0:
            raise ValueError(f"Memory mask must be non-empty with shape {expected}")
        frame_idx = int(self.host.frame_index)
        global_output = self.state["output_dict"]["non_cond_frame_outputs"].get(frame_idx)
        if global_output is None:
            raise RuntimeError(f"No non-conditioning output for frame {frame_idx}")
        if frame_idx not in self.state["cached_features"]:
            raise RuntimeError(f"Frame {frame_idx} image features are not cached")

        tensor = torch.as_tensor(mask, dtype=torch.float32, device=torch.device(self.device))[None, None]
        tensor = torch_f.interpolate(
            tensor,
            size=(self.predictor.image_size, self.predictor.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        high_logits = (tensor >= 0.5).float() * 20.0 - 10.0
        current_score = global_output["object_score_logits"].to(torch.device(self.device))
        object_score = torch.full_like(current_score, 10.0)
        features, position = self.predictor._run_memory_encoder(
            self.state, frame_idx, 1, high_logits, object_score, True
        )
        low_logits = torch_f.interpolate(
            high_logits,
            size=global_output["pred_masks"].shape[-2:],
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).to(self.state["storage_device"], dtype=global_output["pred_masks"].dtype)
        features = features.to(self.state["storage_device"])
        object_score = object_score.to(self.state["storage_device"])
        self._write_output_mask(global_output, low_logits, features, position, object_score)

        obj_idx = self.state["obj_id_to_idx"].get(0)
        per_object = self.state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"].get(frame_idx)
        self._write_output_mask(per_object, low_logits, features, position, object_score, slice(0, 1))
        if hasattr(self.predictor, "curr_out"):
            self.predictor.curr_out = global_output
        motion_corrected = self._correct_host_motion(high_logits)
        return {
            "applied": True,
            "frame_idx": frame_idx,
            "mask_area": int(mask.sum()),
            "motion_corrected": motion_corrected,
        }
