"""Pure phonetic compatibility features for frozen candidate-choice rows.

The first feature is a source-length-weighted average of per-edge, per-token
teacher-forced log probabilities.  It is a calibration feature, not a path log
probability.  Multiple alignment alternatives are averaged uniformly; that
average is deliberately not presented as a marginalized probability.

Copy edges are the reference category.  Unsupported conversions remain present
in their paths and contribute only to their explicit source-fraction feature.
The returned base scores are copied unchanged, so applying zero residual weights
preserves the frozen candidate scores exactly.
"""
from __future__ import annotations

import math
import re


FEATURE_NAMES = (
    "length_weighted_mean_jamo_eos_log_probability",
    "supported_source_fraction",
    "unsupported_conversion_source_fraction",
    "forward_mismatch_supported_source_fraction",
)

_ROW_SCHEMA = "phonetic-likelihood-fit-v1-row"
_STATUSES = {
    "copy", "supported", "unsupported_ascii", "unsupported_shape", "length", "empty",
}
_LOWER_ASCII = re.compile(r"[a-z]+\Z")
_HANGUL = re.compile(r"[가-힣]+\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _expected_status(source: str, target: str) -> str:
    if not source or not target:
        return "empty"
    if source == target:
        return "copy"
    if _LOWER_ASCII.fullmatch(source) is None:
        return "unsupported_ascii"
    if _HANGUL.fullmatch(target) is None:
        return "unsupported_shape"
    if len(source) > 48 or len(target) > 24:
        return "length"
    return "supported"


def _score_for(edge: dict, scores_by_pair_id: dict) -> dict:
    pair_id = edge.get("pair_id")
    _require(isinstance(pair_id, str) and pair_id, "supported edge pair ID is missing")
    score = scores_by_pair_id.get(pair_id)
    _require(isinstance(score, dict), "supported edge score is missing")
    _require(set(score) == {
        "pair_id", "source", "target", "log_probability", "token_count",
        "mean_log_probability",
    }, "supported edge score schema differs")
    _require(score["pair_id"] == pair_id
             and score["source"] == edge["source"]
             and score["target"] == edge["target"],
             "supported edge score identity differs")
    total, mean = score["log_probability"], score["mean_log_probability"]
    tokens = score["token_count"]
    _require(_finite_number(total) and total <= 0
             and _finite_number(mean) and mean <= 0
             and type(tokens) is int and tokens == 3 * len(edge["target"]) + 1
             and mean == total / tokens,
             "supported edge score value differs")
    return score


def _validate_edge(edge: object, source: str) -> None:
    _require(isinstance(edge, dict), "candidate edge must be an object")
    status = edge.get("status")
    expected_keys = {"input_start", "input_end", "source", "target", "status"}
    if status == "supported":
        expected_keys |= {"pair_id", "romanize_match"}
    _require(status in _STATUSES and set(edge) == expected_keys,
             "candidate edge schema or status differs")
    start, end = edge["input_start"], edge["input_end"]
    _require(type(start) is int and type(end) is int and 0 <= start < end <= len(source)
             and edge["source"] == source[start:end]
             and isinstance(edge["target"], str),
             "candidate edge source offset differs")
    _require(status == _expected_status(edge["source"], edge["target"]),
             "candidate edge classification differs")
    if status == "supported":
        _require(type(edge["romanize_match"]) is bool,
                 "supported edge forward-match flag differs")


def _path_features(source: str, candidate: str, path: dict, edges: list,
                   scores_by_pair_id: dict) -> list[float]:
    _require(isinstance(path, dict) and set(path) == {"path_id", "edge_indices"}
             and isinstance(path["path_id"], str) and path["path_id"],
             "candidate path schema differs")
    indices = path["edge_indices"]
    _require(isinstance(indices, list) and indices, "candidate path is empty")
    cursor = 0
    pieces = []
    features = [0.0] * len(FEATURE_NAMES)
    for index in indices:
        _require(type(index) is int and 0 <= index < len(edges),
                 "candidate path edge index differs")
        edge = edges[index]
        _require(edge["input_start"] == cursor, "candidate path source coverage is not contiguous")
        cursor = edge["input_end"]
        pieces.append(edge["target"])
        weight = (edge["input_end"] - edge["input_start"]) / len(source)
        if edge["status"] == "supported":
            score = _score_for(edge, scores_by_pair_id)
            features[0] += weight * score["mean_log_probability"]
            features[1] += weight
            if edge["romanize_match"] is False:
                features[3] += weight
        elif edge["status"] != "copy":
            features[2] += weight
    _require(cursor == len(source), "candidate path does not cover the full source")
    _require("".join(pieces) == candidate, "candidate path target differs")
    _require(all(math.isfinite(value) for value in features),
             "candidate path feature is nonfinite")
    return features


def project_row(row: dict, scores_by_pair_id: dict) -> dict:
    """Project one frozen FIT row to four candidate-aligned numeric features.

    Candidate and positive-index order is preserved.  Each candidate's feature
    vector is the arithmetic mean over its complete alignment alternatives.
    ``scores_by_pair_id`` must contain exact captured likelihood rows for every
    supported edge used by the row.
    """
    _require(isinstance(row, dict) and set(row) == {
        "schema", "id", "input", "component_id", "candidates", "base_scores",
        "positive_indices", "selected_indices", "edges", "candidate_paths",
    } and row.get("schema") == _ROW_SCHEMA, "FIT likelihood row schema differs")
    _require(isinstance(scores_by_pair_id, dict), "scores must be keyed by pair ID")
    identifier, source = row["id"], row["input"]
    _require(isinstance(identifier, str) and identifier
             and isinstance(source, str) and source, "FIT row identity differs")
    candidates, base_scores = row["candidates"], row["base_scores"]
    positives, alternatives = row["positive_indices"], row["candidate_paths"]
    _require(isinstance(candidates, list) and candidates
             and len(set(candidates)) == len(candidates)
             and all(isinstance(candidate, str) and candidate for candidate in candidates)
             and isinstance(base_scores, list) and len(base_scores) == len(candidates)
             and all(_finite_number(score) for score in base_scores)
             and isinstance(alternatives, list) and len(alternatives) == len(candidates),
             "FIT candidate domain differs")
    _require(isinstance(positives, list) and positives and len(positives) == len(set(positives))
             and all(type(index) is int and 0 <= index < len(candidates) for index in positives),
             "FIT positive indices differ")
    edges = row["edges"]
    _require(isinstance(edges, list) and edges, "FIT row edges are missing")
    for edge in edges:
        _validate_edge(edge, source)

    features = []
    seen_path_ids = set()
    for candidate, paths in zip(candidates, alternatives):
        _require(isinstance(paths, list) and paths, "candidate has no alignment alternatives")
        path_values = []
        for path in paths:
            path_id = path.get("path_id") if isinstance(path, dict) else None
            _require(path_id not in seen_path_ids, "candidate path ID is duplicated")
            values = _path_features(source, candidate, path, edges, scores_by_pair_id)
            seen_path_ids.add(path_id)
            path_values.append(values)
        vector = [sum(values[index] for values in path_values) / len(path_values)
                  for index in range(len(FEATURE_NAMES))]
        _require(all(math.isfinite(value) for value in vector),
                 "candidate feature is nonfinite")
        features.append(vector)

    return {
        "id": identifier,
        "input": source,
        "component_id": row["component_id"],
        "selected_indices": list(row["selected_indices"]),
        "candidates": list(candidates),
        "base_scores": list(base_scores),
        "positive_indices": list(positives),
        "features": features,
        "alternative_path_counts": list(map(len, alternatives)),
        "total_paths": len(seen_path_ids),
    }


__all__ = ["FEATURE_NAMES", "project_row"]
