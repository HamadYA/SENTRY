from pathlib import Path

from sentry_tracking.paths import PathConfig
from sentry_tracking.evaluation import DATASETS


ROOT = Path(__file__).resolve().parents[2]


def test_path_config_resolves_relative_entries_from_project_root(tmp_path):
    config_path = tmp_path / "paths.yaml"
    config_path.write_text(
        """
datasets:
  lasot: datasets/LaSOT
checkpoints:
  sam21-T: checkpoints/tiny.pt
outputs:
  lasot: outputs/lasot
""".strip(),
        encoding="utf-8",
    )

    paths = PathConfig.from_yaml(config_path, base_dir=tmp_path)

    assert paths.datasets["lasot"] == str((tmp_path / "datasets" / "LaSOT").resolve())
    assert paths.checkpoints["sam21-T"] == str((tmp_path / "checkpoints" / "tiny.pt").resolve())
    assert paths.outputs["lasot"] == str((tmp_path / "outputs" / "lasot").resolve())


def test_public_paths_example_covers_every_dataset():
    paths = PathConfig.from_yaml(ROOT / "configs" / "paths.example.yaml", base_dir=ROOT)

    assert set(paths.datasets) == set(DATASETS)
    assert set(paths.outputs) == set(DATASETS)
