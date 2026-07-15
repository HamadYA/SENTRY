from __future__ import annotations

from .box import evaluate_box_dataset


def evaluate_lasot(
    tracker_builder,
    dataset_root,
    output_dir,
    sequence=None,
    debug_log=None,
    overwrite=False,
    progress_every=100,
):
    return evaluate_box_dataset(
        tracker_builder,
        "lasot",
        dataset_root,
        output_dir,
        sequence=sequence,
        debug_log=debug_log,
        overwrite=overwrite,
        progress_every=progress_every,
    )
