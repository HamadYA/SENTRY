from sentry_tracking.cli import DEFAULT_SENTRY_CONFIG, build_parser
from sentry_tracking.evaluation import DATASETS


def test_cli_exposes_every_release_dataset():
    parser = build_parser()

    for dataset_name in DATASETS:
        assert parser.parse_args(["--dataset", dataset_name]).dataset == dataset_name

    assert DEFAULT_SENTRY_CONFIG.name == "default.yaml"
