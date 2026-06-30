import os
import cv2
import numpy as np


def keep_largest_component(mask):
    """
    Keeps only the largest connected component from a binary mask.
    
    Args:
    - mask (numpy array): 2D binary mask where object pixels are non-zero and background is 0.
    
    Returns:
    - filtered_mask (numpy array): Binary mask with only the largest connected component.
    """
    # Perform connected components analysis
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Find the index of the largest component (excluding background)
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip background (index 0)
    # Create a mask that contains only the largest component
    filtered_mask = np.zeros_like(mask)
    filtered_mask[labels == largest_component] = 1
    return filtered_mask

BASE_TRACKER_MODEL_CONFIGS = {
    "sam21-L": "sam2.1_hiera_l.yaml",
    "sam21-B": "sam2.1_hiera_b+.yaml",
    "sam21-S": "sam2.1_hiera_s.yaml",
    "sam21-T": "sam2.1_hiera_t.yaml",
    "sam2-L": "sam2_hiera_l.yaml",
    "sam2-B": "sam2_hiera_b+.yaml",
    "sam2-S": "sam2_hiera_s.yaml",
    "sam2-T": "sam2_hiera_t.yaml",
}

BACKEND_TRACKER_MODEL_CONFIGS = {
    "sam2": BASE_TRACKER_MODEL_CONFIGS,
    "SAMURAI": {
        "sam21-L": "sam2.1_hiera_l.yaml",
        "sam21-B": "sam2.1_hiera_b+.yaml",
        "sam21-S": "sam2.1_hiera_s.yaml",
        "sam21-T": "sam2.1_hiera_t.yaml",
    },
    "HiM2SAM": {
        "sam21-L": "sam2.1_hiera_l.yaml",
        "sam21-B": "sam2.1_hiera_b+.yaml",
        "sam21-S": "sam2.1_hiera_s.yaml",
        "sam21-T": "sam2.1_hiera_t.yaml",
    },
    "DAM4SAM": {
        "sam21-L": "sam21pp_hiera_l.yaml",
        "sam21-B": "sam21pp_hiera_b+.yaml",
        "sam21-S": "sam21pp_hiera_s.yaml",
        "sam21-T": "sam21pp_hiera_t.yaml",
        "sam2-L": "sam2pp_hiera_l.yaml",
        "sam2-B": "sam2pp_hiera_b+.yaml",
        "sam2-S": "sam2pp_hiera_s.yaml",
        "sam2-T": "sam2pp_hiera_t.yaml",
    },
    "SAMITE": {
        "sam21-L": "samite_hiera_l.yaml",
        "sam21-B": "samite_hiera_b+.yaml",
        "sam21-S": "samite_hiera_s.yaml",
        "sam21-T": "samite_hiera_t.yaml",
    },
}

TRACKER_MODEL_KEYS = {
    "sam21-L": "sam2.1_hiera_l",
    "sam21-B": "sam2.1_hiera_b+",
    "sam21-S": "sam2.1_hiera_s",
    "sam21-T": "sam2.1_hiera_t",
    "sam2-L": "sam2_hiera_l",
    "sam2-B": "sam2_hiera_b+",
    "sam2-S": "sam2_hiera_s",
    "sam2-T": "sam2_hiera_t",
}

TRACKER_CHECKPOINT_FILES = {
    "sam21-L": "sam2.1_hiera_large.pt",
    "sam21-B": "sam2.1_hiera_base_plus.pt",
    "sam21-S": "sam2.1_hiera_small.pt",
    "sam21-T": "sam2.1_hiera_tiny.pt",
    "sam2-L": "sam2_hiera_large.pt",
    "sam2-B": "sam2_hiera_base_plus.pt",
    "sam2-S": "sam2_hiera_small.pt",
    "sam2-T": "sam2_hiera_tiny.pt",
}


def _load_shared_config():
    config_path = os.environ.get("SAM2_CONFIG")
    if config_path is None:
        candidate = os.path.join(os.getcwd(), "config.yaml")
        if os.path.exists(candidate):
            config_path = candidate
    if config_path is None:
        candidate = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, "config.yaml")
        )
        if os.path.exists(candidate):
            config_path = candidate
    if config_path is None or not os.path.exists(config_path):
        return {}, None

    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    return config, os.path.dirname(os.path.abspath(config_path))


def _resolve_config_path(path, config_dir):
    if path is None:
        return None
    if os.path.isabs(path) or config_dir is None:
        return path
    return os.path.abspath(os.path.join(config_dir, path))


def _checkpoint_from_config(tracker_name):
    config, config_dir = _load_shared_config()

    weight_config = (config.get("weights") or {}).get(tracker_name)
    if isinstance(weight_config, dict):
        checkpoint = (
            weight_config.get("path")
            or weight_config.get("checkpoint")
            or weight_config.get("sam2_checkpoint")
        )
    else:
        checkpoint = weight_config
    if checkpoint:
        return _resolve_config_path(checkpoint, config_dir)

    model_key = TRACKER_MODEL_KEYS.get(tracker_name)
    model_config = (config.get("models") or {}).get(model_key, {})
    checkpoint = model_config.get("sam2_checkpoint") or model_config.get("checkpoint")
    return _resolve_config_path(checkpoint, config_dir)


def _fallback_checkpoint(tracker_name):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    return os.path.join(
        repo_root,
        "checkpoints",
        TRACKER_CHECKPOINT_FILES[tracker_name],
    )


def _infer_backend_name():
    backend_name = os.environ.get("SAM2_MODEL_NAME") or os.environ.get("SAM2_BACKEND")
    if backend_name:
        return backend_name

    cwd_name = os.path.basename(os.getcwd())
    if cwd_name in BACKEND_TRACKER_MODEL_CONFIGS:
        return cwd_name
    return "sam2"


def determine_tracker(tracker_name):
    backend_name = _infer_backend_name()
    model_configs = BACKEND_TRACKER_MODEL_CONFIGS.get(
        backend_name,
        BASE_TRACKER_MODEL_CONFIGS,
    )
    if tracker_name not in model_configs:
        available = ", ".join(sorted(model_configs))
        raise ValueError(
            f"Unknown tracker {tracker_name!r} for backend {backend_name!r}. "
            f"Available: {available}"
        )

    checkpoint = _checkpoint_from_config(tracker_name) or _fallback_checkpoint(tracker_name)
    model_cfg = model_configs[tracker_name]
    return checkpoint, model_cfg

def get_seq_names(dataset_path):
    list_path = os.path.join(dataset_path, 'list.txt')
    with open(list_path, 'r') as f:
        lines = f.readlines()
    seq_names = [line.strip() for line in lines]
    return seq_names

def compute_seq_perf(pred_masks_, gt, bounds, sequence_name):
    try:
        from vot.region import RegionType
        from vot.region.raster import calculate_overlaps
    except ImportError as exc:
        raise RuntimeError(
            "compute_seq_perf requires the VOT toolkit. Install the VOT Python "
            "package or avoid VOT-only evaluation helpers."
        ) from exc

    pred_masks = pred_masks_.copy()
    # bounds: tuple (width, height)
    # convert gt to bboxes:
    gt = gt[1:]
    pred_masks = pred_masks[1:]

    for i in range(len(gt)):
        gt_ = gt[i]
        if gt_.type is not RegionType.SPECIAL and not gt_.is_empty():
            if gt_ != RegionType.RECTANGLE:
                gt[i] = gt_.convert(RegionType.RECTANGLE)
                pred_masks[i] = pred_masks[i].convert(RegionType.RECTANGLE)    

    overlaps = calculate_overlaps(pred_masks, gt, bounds)
    overlaps_arr = np.array(overlaps)

    avg_overlap = overlaps_arr.mean()
    robustness = (overlaps_arr>0).sum() / len(gt)
    
    print('--------------------------------')
    print('Performance on %s:' % sequence_name)
    print('Average overlap: %.3f' % (avg_overlap))
    print('Robustness: %.2f' % (robustness))
    print('--------------------------------')

    return (sequence_name, avg_overlap, robustness)
