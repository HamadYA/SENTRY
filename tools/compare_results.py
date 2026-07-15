#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_boxes(path):
    return np.loadtxt(path, delimiter=",").reshape(-1, 4)


def main():
    parser = argparse.ArgumentParser(description="Compare two tracker result directories")
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()
    reference_root, candidate_root = Path(args.reference), Path(args.candidate)
    files = sorted(reference_root.glob("*.txt"))
    changed_frames = 0
    compared_frames = 0
    missing = []
    for reference_path in files:
        candidate_path = candidate_root / reference_path.name
        if not candidate_path.exists():
            missing.append(reference_path.name)
            continue
        reference, candidate = load_boxes(reference_path), load_boxes(candidate_path)
        if reference.shape != candidate.shape:
            raise SystemExit(f"Shape mismatch for {reference_path.name}: {reference.shape} vs {candidate.shape}")
        changed_frames += int((~np.isclose(reference, candidate, atol=args.atol).all(axis=1)).sum())
        compared_frames += len(reference)
    print(f"Compared {compared_frames} frames; changed frames: {changed_frames}; missing sequences: {len(missing)}")
    if missing:
        print("Missing: " + ", ".join(missing))


if __name__ == "__main__":
    main()
