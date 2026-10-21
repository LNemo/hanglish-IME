"""Conservative student overrides for an existing semantic ranker.

This adapter keeps the teacher's scores byte-for-byte unless a student changes
the winner with enough margin while the teacher itself is uncertain.  It is an
experimental evaluation component; promotion remains an external decision.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from hanglish.models.semantic_ranker import SemanticRanker


def _finite_nonnegative(value: object, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def _bundle_identity(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise ValueError(f"semantic-ranker checkpoint directory is missing: {path}")
    files = sorted(item for item in path.iterdir() if item.is_file())
    if not files:
        raise ValueError(f"semantic-ranker checkpoint directory is empty: {path}")
    return {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in files
    }


def _validate_scores(values: object, count: int, source: str) -> list[float]:
    if (not isinstance(values, (list, tuple)) or len(values) != count
            or any(isinstance(value, bool)
                   or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in values)):
        raise RuntimeError(f"{source} semantic ranker returned invalid scores")
    return [float(value) for value in values]


class GuardedSemanticRanker:
    """Use a student only for low-margin teacher disagreements.

    Candidate groups are defined by exact context equality.  Duplicate
    candidate strings inside one group are evaluated once, preventing repeated
    copies from becoming the artificial runner-up in a margin calculation.
    """

    def __init__(self, teacher, student, *, max_teacher_margin: float,
                 min_student_margin: float,
                 teacher_checkpoint: str | Path | None = None,
                 student_checkpoint: str | Path | None = None) -> None:
        if not callable(getattr(teacher, "score_pairs", None)):
            raise TypeError("teacher must provide score_pairs")
        if not callable(getattr(student, "score_pairs", None)):
            raise TypeError("student must provide score_pairs")
        self.teacher = teacher
        self.student = student
        self.max_teacher_margin = _finite_nonnegative(
            max_teacher_margin, "max_teacher_margin")
        self.min_student_margin = _finite_nonnegative(
            min_student_margin, "min_student_margin")
        self.teacher_checkpoint = (Path(teacher_checkpoint).resolve()
                                   if teacher_checkpoint is not None else None)
        self.student_checkpoint = (Path(student_checkpoint).resolve()
                                   if student_checkpoint is not None else None)
        if (self.teacher_checkpoint is None) != (self.student_checkpoint is None):
            raise ValueError("teacher and student checkpoint paths must be provided together")
        self.artifact_paths = {
            name: str(path) for name, path in (
                ("teacher", self.teacher_checkpoint),
                ("student", self.student_checkpoint),
            ) if path is not None
        }
        self.artifact_hashes = {
            name: _bundle_identity(path) for name, path in (
                ("teacher", self.teacher_checkpoint),
                ("student", self.student_checkpoint),
            ) if path is not None
        }
        self.metadata = {
            "schema": "guarded-semantic-ranker-v1",
            "experimental": True,
            "max_teacher_margin": self.max_teacher_margin,
            "min_student_margin": self.min_student_margin,
            "artifact_paths": dict(self.artifact_paths),
            "artifact_hashes": {name: dict(values)
                                for name, values in self.artifact_hashes.items()},
        }

    @classmethod
    def load(cls, teacher_checkpoint: str | Path,
             student_checkpoint: str | Path, *, max_teacher_margin: float,
             min_student_margin: float) -> "GuardedSemanticRanker":
        teacher_path = Path(teacher_checkpoint).resolve()
        student_path = Path(student_checkpoint).resolve()
        teacher = SemanticRanker.load(teacher_path)
        student = SemanticRanker.load(student_path)
        return cls(
            teacher, student,
            max_teacher_margin=max_teacher_margin,
            min_student_margin=min_student_margin,
            teacher_checkpoint=teacher_path,
            student_checkpoint=student_path,
        )

    def score_pairs(self, contexts: list[str], candidates: list[str]) -> list[float]:
        if not isinstance(contexts, list) or not isinstance(candidates, list):
            raise TypeError("contexts and candidates must be lists")
        if len(contexts) != len(candidates):
            raise ValueError("contexts and candidates must have equal length")
        if not contexts:
            return []

        teacher_scores = _validate_scores(
            self.teacher.score_pairs(contexts, candidates), len(candidates), "teacher")
        groups: dict[str, list[int]] = {}
        for index, context in enumerate(contexts):
            groups.setdefault(context, []).append(index)

        eligible: list[tuple[list[int], list[int]]] = []
        student_contexts: list[str] = []
        student_candidates: list[str] = []
        for context, indices in groups.items():
            distinct: dict[str, int] = {}
            for index in indices:
                distinct.setdefault(candidates[index], index)
            unique_indices = list(distinct.values())
            if len(unique_indices) < 2:
                continue
            teacher_top = sorted((teacher_scores[index] for index in unique_indices), reverse=True)
            if teacher_top[0] - teacher_top[1] > self.max_teacher_margin:
                continue
            eligible.append((indices, unique_indices))
            student_contexts.extend([context] * len(unique_indices))
            student_candidates.extend(candidates[index] for index in unique_indices)

        if not student_candidates:
            return teacher_scores
        student_flat = _validate_scores(
            self.student.score_pairs(student_contexts, student_candidates),
            len(student_candidates), "student")

        result = list(teacher_scores)
        cursor = 0
        for indices, unique_indices in eligible:
            count = len(unique_indices)
            student_scores = student_flat[cursor:cursor + count]
            cursor += count
            teacher_unique = [teacher_scores[index] for index in unique_indices]
            teacher_order = sorted(range(count), key=lambda i: (-teacher_unique[i], i))
            student_order = sorted(range(count), key=lambda i: (-student_scores[i], i))
            teacher_margin = (teacher_unique[teacher_order[0]]
                              - teacher_unique[teacher_order[1]])
            student_margin = (student_scores[student_order[0]]
                              - student_scores[student_order[1]])
            if (teacher_order[0] == student_order[0]
                    or teacher_margin > self.max_teacher_margin
                    or student_margin < self.min_student_margin):
                continue
            mapped = [value - student_scores[student_order[0]]
                      + teacher_unique[teacher_order[0]] for value in student_scores]
            by_candidate = {
                candidates[index]: mapped[position]
                for position, index in enumerate(unique_indices)
            }
            for index in indices:
                result[index] = by_candidate[candidates[index]]
        if cursor != len(student_flat):
            raise RuntimeError("student semantic ranker scores were not fully consumed")
        return result


__all__ = ["GuardedSemanticRanker"]
