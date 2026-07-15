from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from sentry_tracking.evaluation import evaluate_lasot
from sentry_tracking.types import TrackResult


class ConstantTracker:
    def initialize(self, frame, bbox, mask=None):
        return TrackResult(list(bbox), None, "init", list(bbox), diagnostics={"ok": True})

    def track(self, frame):
        return TrackResult([1, 2, 3, 4], None, "baseline", [1, 2, 3, 4], diagnostics={"ok": True})


def test_lasot_runner_writes_common_result_format_and_jsonl():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "testing_set.txt").write_text("airplane-1\n", encoding="utf-8")
        sequence = root / "airplane" / "airplane-1"
        image_dir = sequence / "img"
        image_dir.mkdir(parents=True)
        (sequence / "groundtruth.txt").write_text("1,2,3,4\n1,2,3,4\n", encoding="utf-8")
        Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(image_dir / "00000001.jpg")
        Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(image_dir / "00000002.jpg")
        output = root / "outputs"
        debug = root / "debug.jsonl"
        progress = StringIO()
        with redirect_stdout(progress):
            processed = evaluate_lasot(ConstantTracker, root, output, debug_log=debug, progress_every=1)
        assert processed == ["airplane-1"]
        assert len((output / "airplane-1.txt").read_text(encoding="utf-8").splitlines()) == 2
        assert len(debug.read_text(encoding="utf-8").splitlines()) == 2
        assert "airplane-1 2/2" in progress.getvalue()
        assert "FPS" in progress.getvalue()
        assert "I/O" in progress.getvalue()


def test_lasot_runner_uses_only_official_testing_split():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "testing_set.txt").write_text("airplane-1\n", encoding="utf-8")
        for name in ("airplane-1", "airplane-2"):
            image_dir = root / "airplane" / name / "img"
            image_dir.mkdir(parents=True)
            (image_dir.parent / "groundtruth.txt").write_text("1,2,3,4\n", encoding="utf-8")
            Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(image_dir / "00000001.jpg")

        processed = evaluate_lasot(ConstantTracker, root, root / "outputs")

        assert processed == ["airplane-1"]
        assert not (root / "outputs" / "airplane-2.txt").exists()


def test_lasot_runner_loads_only_pending_sequences_after_printing_summary():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "testing_set.txt").write_text("airplane-1\nairplane-2\n", encoding="utf-8")

        pending_sequence = root / "airplane" / "airplane-2"
        image_dir = pending_sequence / "img"
        image_dir.mkdir(parents=True)
        (pending_sequence / "groundtruth.txt").write_text("1,2,3,4\n", encoding="utf-8")
        Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(image_dir / "00000001.jpg")

        output = root / "outputs"
        output.mkdir()
        (output / "airplane-1.txt").write_text("1,2,3,4\n", encoding="utf-8")

        progress = StringIO()
        with redirect_stdout(progress):
            processed = evaluate_lasot(ConstantTracker, root, output)

        report = progress.getvalue()
        assert processed == ["airplane-2"]
        assert "2 sequence(s), 1 pending, 1 already complete" in report
        assert report.index("evaluation split") < report.index("airplane-2: discovering frames")
        assert "airplane-1: discovering frames" not in report
