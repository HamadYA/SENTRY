from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class PathConfig:
    """Machine-local paths used by benchmark runners."""

    datasets: dict[str, str] = field(default_factory=dict)
    checkpoints: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def _read_section(values: Mapping[str, Any], name: str, base_dir: Path) -> dict[str, str]:
        section = values.get(name, {}) or {}
        if not isinstance(section, Mapping):
            raise ValueError(f"Path configuration section '{name}' must be a mapping")

        paths = {}
        for key, value in section.items():
            if value is None or str(value).strip() == "":
                continue
            if not isinstance(value, (str, Path)):
                raise ValueError(f"Path configuration value '{name}.{key}' must be a path string")
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = base_dir / path
            paths[str(key)] = str(path.resolve())
        return paths

    @classmethod
    def from_yaml(cls, path: str | Path, base_dir: str | Path | None = None) -> "PathConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, Mapping):
            raise ValueError("Path configuration must be a YAML mapping")

        unknown = set(values) - {"datasets", "checkpoints", "outputs"}
        if unknown:
            raise ValueError(f"Unknown path configuration sections: {sorted(unknown)}")

        root = Path(base_dir).expanduser().resolve() if base_dir else config_path.parent
        return cls(
            datasets=cls._read_section(values, "datasets", root),
            checkpoints=cls._read_section(values, "checkpoints", root),
            outputs=cls._read_section(values, "outputs", root),
        )
