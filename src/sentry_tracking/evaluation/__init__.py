from .box import evaluate_box_dataset
from .datasets import BOX_DATASETS, DATASETS
from .didi import evaluate_didi
from .lasot import evaluate_lasot


def evaluate_dataset(tracker_builder, dataset_name, dataset_root, output_dir, **kwargs):
    if dataset_name == "didi":
        return evaluate_didi(tracker_builder, dataset_root, output_dir, **kwargs)
    if dataset_name in BOX_DATASETS:
        return evaluate_box_dataset(
            tracker_builder,
            dataset_name,
            dataset_root,
            output_dir,
            **kwargs,
        )
    raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {', '.join(DATASETS)}")


__all__ = [
    "BOX_DATASETS",
    "DATASETS",
    "evaluate_box_dataset",
    "evaluate_dataset",
    "evaluate_didi",
    "evaluate_lasot",
]
