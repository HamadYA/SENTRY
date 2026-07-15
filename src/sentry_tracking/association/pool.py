from __future__ import annotations

from itertools import combinations, permutations

import numpy as np

from ..config import SENTRYConfig
from ..geometry import bbox_iou, clean_bbox


class TrajectoryPool:
    def __init__(self, config: SENTRYConfig):
        self.config = config
        self.entries = []
        self.next_track_id = 0

    @staticmethod
    def trajectory_iou(first, second):
        overlaps = [bbox_iou(first[idx], second[idx]) for idx in sorted(set(first) & set(second))]
        overlaps = [value for value in overlaps if value is not None]
        return float(np.mean(overlaps)) if overlaps else None

    def _simple_assignment(self, matrix):
        old_count = len(matrix)
        current_count = len(matrix[0]) if old_count else 0
        best_score, best_pairs = 0.0, []
        for count in range(1, min(old_count, current_count) + 1):
            for old_indices in combinations(range(old_count), count):
                for current_indices in permutations(range(current_count), count):
                    pairs = list(zip(old_indices, current_indices))
                    scores = [matrix[old][current] for old, current in pairs]
                    if any(score is None or score < self.config.pool_match_min_iou for score in scores):
                        continue
                    if sum(scores) > best_score:
                        best_score, best_pairs = float(sum(scores)), pairs
        return best_pairs

    @staticmethod
    def _joint_assignment(matrix):
        rows = len(matrix)
        columns = len(matrix[0]) if rows else 0
        size = max(rows, columns)
        if not size:
            return []
        padded = np.zeros((size, size), dtype=np.float32)
        for row_idx, row in enumerate(matrix):
            for column_idx, score in enumerate(row):
                if score is not None and np.isfinite(score):
                    padded[row_idx, column_idx] = score
        best_columns, best_key = None, None
        for assignment in permutations(range(size)):
            scores = [float(padded[row, column]) for row, column in enumerate(assignment)]
            key = (sum(scores), scores[0] if rows else 0.0)
            if best_key is None or key > best_key:
                best_columns, best_key = assignment, key
        return [
            (row, column, float(padded[row, column]))
            for row, column in enumerate(best_columns)
            if row < rows and column < columns
        ]

    def update(self, frame_idx, history_length, candidates, target_history, transient_candidates=None):
        previous = self.entries
        current = []
        for candidate in list(candidates) + list(transient_candidates or []):
            if not candidate.get("reverse_boxes"):
                continue
            trajectory = {
                frame_idx - history_length + local_idx: clean_bbox(bbox)
                for local_idx, bbox in candidate["reverse_boxes"].items()
                if local_idx < history_length and clean_bbox(bbox) is not None
            }
            trajectory[frame_idx] = clean_bbox(candidate["bbox"])
            current.append(
                {
                    "candidate_key": candidate["key"],
                    "is_winner": candidate.get("is_winner", False),
                    "is_target": candidate.get("is_winner", False),
                    "persistent": candidate.get("persistent", True),
                    "trajectory": trajectory,
                    "track_id": None,
                    "pool_iou": None,
                }
            )

        association = {
            "mode": self.config.association_mode,
            "target_candidate_key": None,
            "score_matrix": [],
            "assignments": [],
        }
        matched_current = set()
        if self.config.association_mode == "joint":
            old_neighbors = [entry for entry in previous if not entry.get("is_target", False)]
            target = {
                frame_idx - history_length + idx: clean_bbox(bbox)
                for idx, bbox in enumerate(target_history)
                if clean_bbox(bbox) is not None
            }
            rows = [{"trajectory": target}] + old_neighbors
            matrix = [[self.trajectory_iou(row["trajectory"], item["trajectory"]) for item in current] for row in rows]
            assignments = self._joint_assignment(matrix)
            for row_idx, current_idx, score in assignments:
                item = current[current_idx]
                if row_idx == 0:
                    association["target_candidate_key"] = item["candidate_key"]
                elif score >= self.config.pool_match_min_iou and item["persistent"]:
                    item["track_id"] = old_neighbors[row_idx - 1]["track_id"]
                    item["pool_iou"] = score
                    matched_current.add(current_idx)
            association.update({"score_matrix": matrix, "assignments": assignments})
        else:
            persistent = [item for item in current if item["persistent"]]
            matrix = [
                [self.trajectory_iou(old["trajectory"], item["trajectory"]) for item in persistent]
                for old in previous
            ]
            for old_idx, current_idx in self._simple_assignment(matrix):
                item = persistent[current_idx]
                item["track_id"] = previous[old_idx]["track_id"]
                item["pool_iou"] = matrix[old_idx][current_idx]
                matched_current.add(current.index(item))
            association["score_matrix"] = matrix

        for idx, item in enumerate(current):
            if item["persistent"] and idx not in matched_current:
                item["track_id"] = self.next_track_id
                self.next_track_id += 1
        self.entries = [item for item in current if item["persistent"]]
        metadata = {
            item["candidate_key"]: {
                "track_id": item["track_id"],
                "pool_iou": item["pool_iou"],
                "matched": item["pool_iou"] is not None,
            }
            for item in self.entries
        }
        return metadata, bool(previous), association

    def mark_target(self, key) -> None:
        for entry in self.entries:
            entry["is_target"] = entry["candidate_key"] == key

    def summary(self):
        return [
            {
                "track_id": item["track_id"],
                "candidate_key": item["candidate_key"],
                "is_target": item.get("is_target", False),
                "pool_iou": item["pool_iou"],
                "trajectory_length": len(item["trajectory"]),
            }
            for item in self.entries
        ]
