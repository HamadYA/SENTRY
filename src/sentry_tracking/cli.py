from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from .backends import build_tracker
from .config import SENTRYConfig
from .evaluation import DATASETS, evaluate_dataset
from .paths import PathConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.yaml"
DEFAULT_SENTRY_CONFIG = PROJECT_ROOT / "configs" / "sentry" / "default.yaml"


def build_parser():
    parser = argparse.ArgumentParser(description="Run baseline or SENTRY tracking")
    parser.add_argument("--method", choices=("baseline", "sentry"), default="sentry")
    parser.add_argument(
        "--backend",
        default="sam2",
        help="Built-ins: sam2, samurai, dam4sam, samite, him2sam; or module:Class",
    )
    parser.add_argument("--dataset", choices=DATASETS, default="lasot")
    parser.add_argument(
        "--paths-config",
        help="Local dataset/checkpoint path YAML (defaults to configs/paths.yaml when present)",
    )
    parser.add_argument("--dataset-root", help="Override the configured dataset root")
    parser.add_argument("--output-dir", help="Override the configured output directory")
    parser.add_argument("--sequence")
    parser.add_argument(
        "--sentry-config",
        help="SENTRY policy YAML (defaults to configs/sentry/default.yaml)",
    )
    parser.add_argument("--tracker-name", default="sam21-T")
    parser.add_argument("--checkpoint")
    parser.add_argument("--model-config")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--baseline-root")
    parser.add_argument("--debug-log")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        metavar="N",
        help="Report frame timing every N frames; use 0 to disable periodic frame reports",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_paths_config(parser, requested_path):
    if requested_path:
        path = Path(requested_path).expanduser()
        if not path.is_file():
            parser.error(f"paths configuration does not exist: {path}")
    elif DEFAULT_PATHS_CONFIG.is_file():
        path = DEFAULT_PATHS_CONFIG
    else:
        return PathConfig(), None

    try:
        return PathConfig.from_yaml(path, base_dir=PROJECT_ROOT), path.resolve()
    except (OSError, ValueError) as error:
        parser.error(str(error))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    paths, paths_file = _load_paths_config(parser, args.paths_config)
    dataset_root = args.dataset_root or paths.datasets.get(args.dataset)
    output_dir = args.output_dir or paths.outputs.get(args.dataset)
    checkpoint = args.checkpoint or paths.checkpoints.get(args.tracker_name)
    if dataset_root is None:
        parser.error(
            f"no root configured for dataset '{args.dataset}'; use --dataset-root or add it to configs/paths.yaml"
        )
    if output_dir is None:
        parser.error(
            f"no output configured for dataset '{args.dataset}'; use --output-dir or add it to configs/paths.yaml"
        )

    if paths_file is not None:
        print(f"Using paths configuration: {paths_file}")
    config = None
    if args.method == "sentry":
        config_path = Path(args.sentry_config).expanduser() if args.sentry_config else DEFAULT_SENTRY_CONFIG
        if not config_path.is_file():
            parser.error(f"SENTRY configuration does not exist: {config_path}")
        config = SENTRYConfig.from_yaml(config_path)
    tracker_builder = partial(
        build_tracker,
        method=args.method,
        backend=args.backend,
        config=config,
        dataset=args.dataset,
        tracker_name=args.tracker_name,
        checkpoint=checkpoint,
        model_config=args.model_config,
        device=args.device,
        baseline_root=args.baseline_root,
    )
    processed = evaluate_dataset(
        tracker_builder,
        args.dataset,
        dataset_root,
        output_dir,
        sequence=args.sequence,
        debug_log=args.debug_log,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )
    print(f"Processed {len(processed)} sequence(s).")


if __name__ == "__main__":
    main()
