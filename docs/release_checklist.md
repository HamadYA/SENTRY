# Release Checklist

- Run `python -m compileall -q src/sentry_tracking tools`.
- Run `python -m pytest -q` in the development environment.
- Compare `--method baseline` output with the unmodified baseline runner.
- Compare structured SENTRY output with the research reference implementation.
- Run baseline-memory and severe-rescue-memory ablations separately.
- Confirm every memory write corresponds to an accepted severe rescue in JSONL diagnostics.
- Evaluate all reported datasets with fixed public configuration files.
- Keep dataset paths, checkpoints, generated results, and debug logs out of Git.
- Retain licenses and notices for SAM2 and every bundled baseline.
- Verify README commands from a fresh environment before tagging the release.
