from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from sentry_tracking.evaluation.datasets import BOX_DATASETS, load_box_sequences


def _write_frame(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(path)


def _make_dataset(root: Path, dataset_name: str):
    name = "airplane-1" if dataset_name in {"lasot", "lasot_ext"} else "sequence-1"
    index = "testing_set.txt" if dataset_name in {"lasot", "lasot_ext"} else "list.txt"
    (root / index).write_text(f"{name}\n", encoding="utf-8")

    if dataset_name in {"lasot", "lasot_ext"}:
        sequence_root = root / "airplane" / name
        frame_dir = sequence_root / "img"
        groundtruth = sequence_root / "groundtruth.txt"
    elif dataset_name == "got_10k":
        sequence_root = root / name
        frame_dir = sequence_root
        groundtruth = sequence_root / "groundtruth.txt"
    elif dataset_name == "trackingnet":
        frame_dir = root / "frames" / name
        groundtruth = root / "anno" / f"{name}.txt"
    elif dataset_name == "tnl2k":
        sequence_root = root / name
        frame_dir = sequence_root / "imgs"
        groundtruth = sequence_root / "groundtruth.txt"
    elif dataset_name == "latot":
        sequence_root = root / name
        frame_dir = sequence_root / "img"
        groundtruth = sequence_root / f"{name}.txt"
    else:
        sequence_root = root / name
        frame_dir = sequence_root / "img"
        groundtruth = sequence_root / "groundtruth_rect.txt"

    groundtruth.parent.mkdir(parents=True, exist_ok=True)
    groundtruth.write_text("1, 2, 3, 4\n", encoding="utf-8")
    _write_frame(frame_dir / "10.jpg")
    _write_frame(frame_dir / "2.jpg")
    return name


@pytest.mark.parametrize("dataset_name", BOX_DATASETS)
def test_box_dataset_layouts_and_numeric_frame_order(tmp_path, dataset_name):
    name = _make_dataset(tmp_path, dataset_name)

    sequences = load_box_sequences(dataset_name, tmp_path)

    assert [item.name for item in sequences] == [name]
    assert sequences[0].initial_bbox == (1.0, 2.0, 3.0, 4.0)
    assert [path.name for path in sequences[0].frames] == ["2.jpg", "10.jpg"]


def test_sequence_filter_must_reference_official_index(tmp_path):
    _make_dataset(tmp_path, "lasot")

    with pytest.raises(ValueError, match="not part of the LaSOT evaluation split"):
        load_box_sequences("lasot", tmp_path, sequence="airplane-2")
