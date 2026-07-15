from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from sentry_tracking.evaluation import evaluate_dataset
from sentry_tracking.types import TrackResult


class ConstantTracker:
    def initialize(self, frame, bbox, mask=None):
        return TrackResult(list(bbox), mask, "init", list(bbox), diagnostics={})

    def track(self, frame):
        mask = np.zeros((frame.height, frame.width), dtype=np.uint8)
        mask[2:6, 1:4] = 1
        return TrackResult([1, 2, 3, 4], mask, "baseline", [1, 2, 3, 4], diagnostics={})


def _save_frame(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((10, 12, 3), dtype=np.uint8)).save(path)


def test_generic_box_evaluator_dispatches_tnl2k(tmp_path):
    (tmp_path / "list.txt").write_text("video-1\n", encoding="utf-8")
    sequence = tmp_path / "video-1"
    (sequence / "groundtruth.txt").parent.mkdir(parents=True, exist_ok=True)
    (sequence / "groundtruth.txt").write_text("1,2,3,4\n", encoding="utf-8")
    _save_frame(sequence / "imgs" / "1.jpg")
    _save_frame(sequence / "imgs" / "2.jpg")
    output = tmp_path / "results"

    processed = evaluate_dataset(ConstantTracker, "tnl2k", tmp_path, output)

    assert processed == ["video-1"]
    lines = (output / "video-1.txt").read_text(encoding="utf-8").splitlines()
    assert lines == ["1.000000,2.000000,3.000000,4.000000"] * 2


class FakeRegion:
    def rasterize(self, bounds):
        mask = np.zeros((10, 12), dtype=np.uint8)
        mask[2:6, 1:4] = 1
        return mask


class FakeMask:
    def __init__(self, mask):
        self.mask = np.asarray(mask)


class FakeFrame:
    def __init__(self, path):
        self.path = path

    def filename(self):
        return str(self.path)


class FakeSequence:
    def __init__(self, root, frames):
        self.root = root
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def frame(self, index):
        return FakeFrame(self.frames[index])

    def metadata(self, key):
        assert key == "root"
        return str(self.root)


def test_didi_evaluator_writes_masks_and_boxes(tmp_path):
    name = "didi-1"
    (tmp_path / "list.txt").write_text(f"{name}\n", encoding="utf-8")
    sequence_root = tmp_path / name
    sequence_root.mkdir()
    (sequence_root / "first_frame_segm.txt").write_text("fake\n", encoding="utf-8")
    frames = [sequence_root / "1.jpg", sequence_root / "2.jpg"]
    for frame in frames:
        _save_frame(frame)
    fake_sequence = FakeSequence(sequence_root, frames)

    def load_dataset(path):
        assert Path(path) == tmp_path
        return {name: fake_sequence}

    def read_trajectory(path):
        assert Path(path) == sequence_root / "first_frame_segm.txt"
        return [FakeRegion()]

    def write_trajectory(path, masks):
        assert all(isinstance(mask, FakeMask) for mask in masks)
        Path(path).write_text("mask\n" * len(masks), encoding="utf-8")

    api = (load_dataset, read_trajectory, write_trajectory, FakeMask)
    output = tmp_path / "results"
    with patch("sentry_tracking.evaluation.didi._load_vot_api", return_value=api):
        processed = evaluate_dataset(ConstantTracker, "didi", tmp_path, output)

    assert processed == [name]
    assert len((output / name / f"{name}.txt").read_text(encoding="utf-8").splitlines()) == 2
    assert len((output / f"{name}.txt").read_text(encoding="utf-8").splitlines()) == 2
