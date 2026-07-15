from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re


BOX_DATASETS = (
    "lasot",
    "lasot_ext",
    "got_10k",
    "trackingnet",
    "tnl2k",
    "latot",
    "otb",
)
DATASETS = BOX_DATASETS + ("didi",)

DATASET_LABELS = {
    "lasot": "LaSOT",
    "lasot_ext": "LaSOT Extension",
    "got_10k": "GOT-10k",
    "trackingnet": "TrackingNet",
    "tnl2k": "TNL2K",
    "latot": "LaTOT",
    "otb": "OTB",
    "didi": "DiDi",
}

_INDEX_FILES = {
    "lasot": "testing_set.txt",
    "lasot_ext": "testing_set.txt",
    "got_10k": "list.txt",
    "trackingnet": "list.txt",
    "tnl2k": "list.txt",
    "latot": "list.txt",
    "otb": "list.txt",
    "didi": "list.txt",
}


@dataclass(frozen=True)
class BoxSequence:
    name: str
    frames: tuple[Path, ...]
    initial_bbox: tuple[float, float, float, float]
    groundtruth_path: Path


def dataset_label(dataset_name: str) -> str:
    return DATASET_LABELS.get(dataset_name, dataset_name)


def read_sequence_names(root: str | Path, dataset_name: str) -> list[str]:
    root = Path(root).expanduser().resolve()
    try:
        index_name = _INDEX_FILES[dataset_name]
    except KeyError as error:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {', '.join(DATASETS)}") from error
    index_path = root / index_name
    if not index_path.is_file():
        raise FileNotFoundError(
            f"{dataset_label(dataset_name)} sequence index not found: {index_path}"
        )
    with index_path.open("r", encoding="utf-8-sig") as handle:
        names = [line.strip() for line in handle if line.strip()]
    if not names:
        raise ValueError(f"{dataset_label(dataset_name)} sequence index is empty: {index_path}")
    if len(names) != len(set(names)):
        raise ValueError(f"{dataset_label(dataset_name)} sequence index contains duplicate names: {index_path}")
    return names


def _select_names(names: list[str], sequence: str | None, dataset_name: str) -> list[str]:
    if sequence is None:
        return names
    if sequence not in names:
        raise ValueError(
            f"Sequence '{sequence}' is not part of the {dataset_label(dataset_name)} evaluation split"
        )
    return [sequence]


def _read_initial_bbox(path: Path) -> tuple[float, float, float, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth file not found: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        line = next((line.strip() for line in handle if line.strip()), "")
    values = [value for value in re.split(r"[\s,]+", line) if value]
    if len(values) < 4:
        raise ValueError(f"Invalid first ground-truth entry in {path}: {line!r}")
    try:
        bbox = tuple(float(value) for value in values[:4])
    except ValueError as error:
        raise ValueError(f"Invalid first ground-truth entry in {path}: {line!r}") from error
    if not all(math.isfinite(value) for value in bbox) or bbox[2] <= 0 or bbox[3] <= 0:
        raise ValueError(f"Invalid initialization box in {path}: {bbox}")
    return bbox


def _natural_key(path: Path):
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", path.name)
    )


def _frame_paths(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {directory}")
    frames = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=_natural_key,
    )
    if not frames:
        raise FileNotFoundError(f"No image frames found under {directory}")
    return tuple(frames)


def _lasot_sequence(root: Path, name: str) -> tuple[Path, Path]:
    category = name.split("-", 1)[0]
    candidates = (root / category / name, root / name)
    sequence_root = next((path for path in candidates if (path / "groundtruth.txt").is_file()), None)
    if sequence_root is None:
        raise FileNotFoundError(f"LaSOT sequence not found under {root}: {name}")
    return sequence_root / "img", sequence_root / "groundtruth.txt"


def _sequence_layout(dataset_name: str, root: Path, name: str) -> tuple[Path, Path]:
    if dataset_name in {"lasot", "lasot_ext"}:
        return _lasot_sequence(root, name)
    if dataset_name == "got_10k":
        sequence_root = root / name
        return sequence_root, sequence_root / "groundtruth.txt"
    if dataset_name == "trackingnet":
        return root / "frames" / name, root / "anno" / f"{name}.txt"
    if dataset_name == "tnl2k":
        sequence_root = root / name
        return sequence_root / "imgs", sequence_root / "groundtruth.txt"
    if dataset_name == "latot":
        sequence_root = root / name
        return sequence_root / "img", sequence_root / f"{name}.txt"
    if dataset_name == "otb":
        sequence_root = root / name
        return sequence_root / "img", sequence_root / "groundtruth_rect.txt"
    raise ValueError(f"Dataset '{dataset_name}' is not a box-tracking dataset")


def select_box_sequence_names(
    dataset_name: str,
    root: str | Path,
    sequence: str | None = None,
) -> list[str]:
    if dataset_name not in BOX_DATASETS:
        raise ValueError(f"Unknown box dataset '{dataset_name}'. Available: {', '.join(BOX_DATASETS)}")
    return _select_names(read_sequence_names(root, dataset_name), sequence, dataset_name)


def load_box_sequence(dataset_name: str, root: str | Path, name: str) -> BoxSequence:
    if dataset_name not in BOX_DATASETS:
        raise ValueError(f"Unknown box dataset '{dataset_name}'. Available: {', '.join(BOX_DATASETS)}")
    root = Path(root).expanduser().resolve()
    frame_dir, groundtruth_path = _sequence_layout(dataset_name, root, name)
    return BoxSequence(
        name=name,
        frames=_frame_paths(frame_dir),
        initial_bbox=_read_initial_bbox(groundtruth_path),
        groundtruth_path=groundtruth_path,
    )


def load_box_sequences(
    dataset_name: str,
    root: str | Path,
    sequence: str | None = None,
) -> list[BoxSequence]:
    names = select_box_sequence_names(dataset_name, root, sequence)
    return [load_box_sequence(dataset_name, root, name) for name in names]
