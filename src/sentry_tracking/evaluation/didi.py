from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from ..diagnostics import JSONLLogger
from ..geometry import mask_to_bbox
from .box import format_duration, write_boxes
from .datasets import read_sequence_names


def _load_vot_api():
    try:
        from vot.dataset import load_dataset
        from vot.region.io import read_trajectory, write_trajectory
        from vot.region.shapes import Mask
    except ImportError as error:
        raise RuntimeError(
            "DiDi evaluation requires the VOT toolkit. Install the 'vot-toolkit' package "
            "before selecting --dataset didi."
        ) from error
    return load_dataset, read_trajectory, write_trajectory, Mask


def _initial_mask(sequence, image, read_trajectory):
    mask_path = Path(sequence.metadata("root")) / "first_frame_segm.txt"
    if not mask_path.is_file():
        raise FileNotFoundError(f"DiDi first-frame segmentation not found: {mask_path}")
    trajectory = read_trajectory(str(mask_path))
    if not trajectory:
        raise ValueError(f"DiDi first-frame segmentation is empty: {mask_path}")
    mask = trajectory[0].rasterize((0, 0, image.width - 1, image.height - 1))
    return (np.asarray(mask) > 0).astype(np.uint8)


def evaluate_didi(
    tracker_builder,
    dataset_root,
    output_dir,
    sequence=None,
    debug_log=None,
    overwrite=False,
    progress_every=100,
):
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative")
    root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    names = read_sequence_names(root, "didi")
    if sequence is not None:
        if sequence not in names:
            raise ValueError(f"Sequence '{sequence}' is not part of the DiDi evaluation split")
        names = [sequence]

    load_dataset, read_trajectory, write_trajectory, Mask = _load_vot_api()
    dataset = load_dataset(str(root))
    logger = JSONLLogger(debug_log)
    pending = []
    for name in names:
        mask_output = output_root / name / f"{name}.txt"
        box_output = output_root / f"{name}.txt"
        if overwrite or not (mask_output.exists() and box_output.exists()):
            pending.append((name, mask_output, box_output))

    skipped = len(names) - len(pending)
    print(
        f"DiDi evaluation split: {len(names)} sequence(s), "
        f"{len(pending)} pending, {skipped} already complete.",
        flush=True,
    )
    processed = []
    benchmark_started = perf_counter()
    for sequence_number, (name, mask_output, box_output) in enumerate(pending, start=1):
        sequence_started = perf_counter()
        vot_sequence = dataset[name]
        frame_count = len(vot_sequence)
        if frame_count < 1:
            raise ValueError(f"DiDi sequence has no frames: {name}")
        print(
            f"[{sequence_number}/{len(pending)}] {name}: {frame_count} frames, building tracker...",
            flush=True,
        )
        build_started = perf_counter()
        tracker = tracker_builder()
        build_seconds = perf_counter() - build_started
        predictions = []
        masks = []
        tracking_started = perf_counter()
        for frame_idx in range(frame_count):
            frame_path = Path(vot_sequence.frame(frame_idx).filename())
            with Image.open(frame_path) as image_file:
                frame = image_file.convert("RGB")
            tracker_started = perf_counter()
            if frame_idx == 0:
                mask = _initial_mask(vot_sequence, frame, read_trajectory)
                bbox = mask_to_bbox(mask)
                if bbox is None:
                    raise ValueError(f"DiDi initialization mask is empty: {name}")
                result = tracker.initialize(frame, bbox, mask=mask)
                saved_mask = mask
                saved_bbox = bbox
            else:
                result = tracker.track(frame)
                saved_mask = (
                    (np.asarray(result.mask) > 0).astype(np.uint8)
                    if result.mask is not None
                    else np.zeros((frame.height, frame.width), dtype=np.uint8)
                )
                saved_bbox = result.bbox
            tracker_call_ms = (perf_counter() - tracker_started) * 1000.0
            predictions.append(saved_bbox)
            masks.append(Mask(saved_mask))
            logger.write(
                {
                    "dataset": "didi",
                    "sequence": name,
                    "frame_idx": frame_idx,
                    "bbox": saved_bbox,
                    "source": result.source,
                    "diagnostics": result.diagnostics,
                }
            )
            frame_number = frame_idx + 1
            if frame_number == 1 or frame_number == frame_count or (
                progress_every and frame_number % progress_every == 0
            ):
                elapsed = perf_counter() - tracking_started
                fps = frame_number / elapsed if elapsed > 0 else 0.0
                remaining = (frame_count - frame_number) / fps if fps > 0 else 0.0
                print(
                    f"[{sequence_number}/{len(pending)}] {name} "
                    f"{frame_number}/{frame_count} ({100.0 * frame_number / frame_count:.1f}%) "
                    f"{fps:.3f} FPS, ETA {format_duration(remaining)}, "
                    f"tracker {tracker_call_ms:.1f} ms",
                    flush=True,
                )

        mask_output.parent.mkdir(parents=True, exist_ok=True)
        write_trajectory(str(mask_output), masks)
        write_boxes(box_output, predictions)
        processed.append(name)
        tracking_seconds = perf_counter() - tracking_started
        print(
            f"[{sequence_number}/{len(pending)}] {name}: saved masks to {mask_output} "
            f"and boxes to {box_output} in {format_duration(perf_counter() - sequence_started)} "
            f"(tracking {frame_count / tracking_seconds:.3f} FPS, "
            f"build {format_duration(build_seconds)}).",
            flush=True,
        )

    print(
        f"DiDi run complete: {len(processed)} processed in "
        f"{format_duration(perf_counter() - benchmark_started)}.",
        flush=True,
    )
    return processed
