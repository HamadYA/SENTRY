from __future__ import annotations

from importlib import import_module

from ..config import SENTRYConfig
from ..engine import BaselineTracker, SENTRYTracker

_BACKENDS = {}
_BUILTIN_BACKENDS = {
    "sam2": ("sentry_tracking.backends.sam2", "SAM2Backend"),
    "samurai": ("sentry_tracking.backends.forked_sam2", "SAMURAIBackend"),
    "dam4sam": ("sentry_tracking.backends.forked_sam2", "DAM4SAMBackend"),
    "samite": ("sentry_tracking.backends.forked_sam2", "SAMITEBackend"),
    "him2sam": ("sentry_tracking.backends.forked_sam2", "HiM2SAMBackend"),
}


def register_backend(name, builder, replace=False):
    key = name.strip().lower()
    if not key:
        raise ValueError("backend name cannot be empty")
    if key in _BACKENDS and not replace:
        raise ValueError(f"backend '{key}' is already registered")
    _BACKENDS[key] = builder


def available_backends():
    return tuple(sorted(set(_BACKENDS) | set(_BUILTIN_BACKENDS)))


def create_backend(name="sam2", **kwargs):
    key = name.strip().lower()
    builder = _BACKENDS.get(key)
    if builder is None and key in _BUILTIN_BACKENDS:
        module_name, attribute = _BUILTIN_BACKENDS[key]
        builder = getattr(import_module(module_name), attribute)
    if builder is None and ":" in name:
        module_name, attribute = name.rsplit(":", 1)
        builder = getattr(import_module(module_name), attribute)
    if builder is None:
        raise ValueError(f"Unknown backend '{name}'. Available: {', '.join(available_backends())}")
    return builder(**kwargs)


def build_tracker(
    method="sentry",
    backend="sam2",
    config: SENTRYConfig | None = None,
    dataset=None,
    **backend_kwargs,
):
    if dataset is not None and backend.strip().lower() in _BUILTIN_BACKENDS:
        backend_kwargs.setdefault("dataset", dataset)
    instance = create_backend(backend, **backend_kwargs)
    if method == "baseline":
        return BaselineTracker(instance)
    if method == "sentry":
        return SENTRYTracker(instance, config)
    raise ValueError("method must be 'baseline' or 'sentry'")
