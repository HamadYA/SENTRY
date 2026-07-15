from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np


def to_jsonable(value):
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "sum": float(value.sum())}
    if isinstance(value, np.generic):
        return value.item()
    return value


class JSONLLogger:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None

    def write(self, record) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(to_jsonable(record), sort_keys=True) + "\n")
