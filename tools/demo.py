#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentry_tracking.backends import available_backends, build_tracker
from sentry_tracking.config import SENTRYConfig


def main():
    parser = argparse.ArgumentParser(description="Track one object in a video with SENTRY")
    parser.add_argument("--video", required=True)
    parser.add_argument("--init-bbox", nargs=4, type=float, required=True, metavar=("X", "Y", "W", "H"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method", choices=("baseline", "sentry"), default="sentry")
    parser.add_argument("--backend", choices=available_backends(), default="sam2")
    parser.add_argument("--tracker-name", default="sam21-T")
    parser.add_argument("--sentry-config", default=str(ROOT / "configs/sentry/default.yaml"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--model-config")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline-root")
    args = parser.parse_args()

    config = SENTRYConfig.from_yaml(args.sentry_config) if args.method == "sentry" else None
    tracker = build_tracker(
        method=args.method,
        backend=args.backend,
        config=config,
        tracker_name=args.tracker_name,
        checkpoint=args.checkpoint,
        model_config=args.model_config,
        device=args.device,
        baseline_root=args.baseline_root,
    )
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    boxes = []
    frame_idx = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(rgb)
            result = tracker.initialize(frame, args.init_bbox) if frame_idx == 0 else tracker.track(frame)
            boxes.append(result.bbox or [0, 0, 0, 0])
            if writer is None:
                fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
                height, width = bgr.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_dir / "tracking.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
            if result.bbox is not None:
                x, y, width, height = [int(round(value)) for value in result.bbox]
                cv2.rectangle(bgr, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(bgr, result.source, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            writer.write(bgr)
            frame_idx += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    with (output_dir / "boxes.txt").open("w", encoding="utf-8") as handle:
        for bbox in boxes:
            handle.write(",".join(f"{float(value):.6f}" for value in bbox) + "\n")
    print(f"Tracked {len(boxes)} frames. Results: {output_dir}")


if __name__ == "__main__":
    main()
