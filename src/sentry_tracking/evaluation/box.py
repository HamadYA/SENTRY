from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter

from PIL import Image

from ..diagnostics import JSONLLogger
from .datasets import dataset_label, load_box_sequence, select_box_sequence_names


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def average_timing(totals, counts, key):
    count = counts.get(key, 0)
    return totals.get(key, 0.0) / count if count else None


def write_boxes(path: Path, boxes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for bbox in boxes:
            values = bbox if bbox is not None else [0, 0, 0, 0]
            handle.write(",".join(f"{float(value):.6f}" for value in values) + "\n")


def evaluate_box_dataset(
    tracker_builder,
    dataset_name,
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
    logger = JSONLLogger(debug_log)
    sequence_names = select_box_sequence_names(dataset_name, root, sequence=sequence)
    pending = []
    for name in sequence_names:
        output_path = output_root / f"{name}.txt"
        if overwrite or not output_path.exists():
            pending.append((name, output_path))

    label = dataset_label(dataset_name)
    skipped = len(sequence_names) - len(pending)
    print(
        f"{label} evaluation split: {len(sequence_names)} sequence(s), "
        f"{len(pending)} pending, {skipped} already complete.",
        flush=True,
    )

    processed = []
    benchmark_started = perf_counter()
    for sequence_number, (sequence_name, output_path) in enumerate(pending, start=1):
        sequence_started = perf_counter()
        print(
            f"[{sequence_number}/{len(pending)}] {sequence_name}: discovering frames...",
            flush=True,
        )
        discovery_started = perf_counter()
        item = load_box_sequence(dataset_name, root, sequence_name)
        discovery_seconds = perf_counter() - discovery_started
        print(
            f"[{sequence_number}/{len(pending)}] {item.name}: "
            f"{len(item.frames)} frames discovered in {format_duration(discovery_seconds)}, "
            "building tracker...",
            flush=True,
        )
        build_started = perf_counter()
        tracker = tracker_builder()
        build_seconds = perf_counter() - build_started
        print(
            f"[{sequence_number}/{len(pending)}] {item.name}: "
            f"tracker ready in {format_duration(build_seconds)}.",
            flush=True,
        )
        predictions = []
        timing_totals = defaultdict(float)
        timing_counts = defaultdict(int)
        tracking_started = perf_counter()
        for frame_idx, frame_path in enumerate(item.frames):
            frame_started = perf_counter()
            image_started = perf_counter()
            with Image.open(frame_path) as image_file:
                frame = image_file.convert("RGB")
            image_load_ms = (perf_counter() - image_started) * 1000.0
            tracker_started = perf_counter()
            result = tracker.initialize(frame, item.initial_bbox) if frame_idx == 0 else tracker.track(frame)
            tracker_call_ms = (perf_counter() - tracker_started) * 1000.0
            frame_total_ms = (perf_counter() - frame_started) * 1000.0

            timing = result.diagnostics.setdefault("timing_ms", {})
            timing.update(
                {
                    "image_load": image_load_ms,
                    "tracker_call": tracker_call_ms,
                    "frame_total": frame_total_ms,
                }
            )
            for key, value in timing.items():
                if isinstance(value, (int, float)):
                    timing_totals[key] += float(value)
                    timing_counts[key] += 1
            predictions.append(result.bbox)
            logger.write(
                {
                    "dataset": dataset_name,
                    "sequence": item.name,
                    "frame_idx": frame_idx,
                    "bbox": result.bbox,
                    "source": result.source,
                    "diagnostics": result.diagnostics,
                }
            )
            frame_number = frame_idx + 1
            report_frame = bool(
                frame_number == 1
                or frame_number == len(item.frames)
                or (progress_every and frame_number % progress_every == 0)
            )
            if report_frame:
                elapsed = perf_counter() - tracking_started
                fps = frame_number / elapsed if elapsed > 0 else 0.0
                remaining = (len(item.frames) - frame_number) / fps if fps > 0 else 0.0
                stages = []
                for stage_label, key in (
                    ("I/O", "image_load"),
                    ("forward", "backend_forward"),
                    ("candidates", "candidate_preparation"),
                    ("reverse", "reverse_verification"),
                ):
                    average = average_timing(timing_totals, timing_counts, key)
                    if average is not None:
                        stages.append(f"{stage_label} {average:.1f} ms")
                cache = result.diagnostics.get("reverse_feature_cache")
                if cache and cache.get("enabled") and cache.get("hit_rate") is not None:
                    stages.append(f"feature cache {100.0 * cache['hit_rate']:.1f}%")
                stage_text = " | " + ", ".join(stages) if stages else ""
                print(
                    f"[{sequence_number}/{len(pending)}] {item.name} "
                    f"{frame_number}/{len(item.frames)} ({100.0 * frame_number / len(item.frames):.1f}%) "
                    f"{fps:.3f} FPS, ETA {format_duration(remaining)}{stage_text}",
                    flush=True,
                )
        write_started = perf_counter()
        write_boxes(output_path, predictions)
        write_seconds = perf_counter() - write_started
        processed.append(item.name)
        tracking_seconds = perf_counter() - tracking_started
        print(
            f"[{sequence_number}/{len(pending)}] {item.name}: saved {output_path} "
            f"in {format_duration(perf_counter() - sequence_started)} "
            f"(tracking {len(item.frames) / tracking_seconds:.3f} FPS, "
            f"discover {format_duration(discovery_seconds)}, "
            f"build {format_duration(build_seconds)}, write {format_duration(write_seconds)}).",
            flush=True,
        )
    print(
        f"{label} run complete: {len(processed)} processed in "
        f"{format_duration(perf_counter() - benchmark_started)}.",
        flush=True,
    )
    return processed
