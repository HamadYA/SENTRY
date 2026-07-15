from pathlib import Path

from sentry_tracking.config import SENTRYConfig


ROOT = Path(__file__).resolve().parents[2]


def test_python_defaults_match_canonical_yaml():
    canonical = SENTRYConfig.from_yaml(ROOT / "configs" / "sentry" / "default.yaml")

    assert canonical == SENTRYConfig()
    assert canonical.amg_enabled is True
    assert canonical.memory_policy == "severe_rescue"
    assert canonical.reverse_feature_cache_enabled is True


def test_conservative_config_preserves_native_memory_policy():
    conservative = SENTRYConfig.from_yaml(ROOT / "configs" / "sentry" / "conservative.yaml")

    assert conservative.amg_enabled is False
    assert conservative.memory_policy == "baseline"
    assert conservative.reverse_feature_cache_enabled is True


def test_feature_cache_is_enabled_in_every_public_policy():
    for name in ("default.yaml", "conservative.yaml", "shadow.yaml"):
        config = SENTRYConfig.from_yaml(ROOT / "configs" / "sentry" / name)
        assert config.reverse_feature_cache_enabled is True
