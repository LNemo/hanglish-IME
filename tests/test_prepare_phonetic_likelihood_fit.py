import json
from pathlib import Path

import pytest

from scripts import prepare_phonetic_likelihood_fit as prep


def selected_row():
    return {
        "id": "train-1", "input": "gana", "component_id": "component-1",
        "candidates": ["가나", "가나나"], "base_scores": [1.0, -2.0],
        "positive_indices": [0], "selected_indices": [0, 1],
        "candidate_path_counts": [2, 1], "selected_path_expansion_count": 0,
    }


def correspondence_row():
    return {
        "schema": "source-correspondence-v1", "id": "train-1",
        "input": "gana", "component_id": "component-1",
        "reference_path_id": "p1",
        "edges": [
            {"input_start": 0, "input_end": 2, "text": "가"},
            {"input_start": 2, "input_end": 4, "text": "나"},
            {"input_start": 0, "input_end": 4, "text": "가나"},
            {"input_start": 2, "input_end": 4, "text": "나나"},
        ],
        "paths": [
            {"path_id": "p1", "text": "가나", "edge_indices": [0, 1]},
            {"path_id": "p2", "text": "가나", "edge_indices": [2]},
            {"path_id": "p3", "text": "가나나", "edge_indices": [0, 3]},
        ],
    }


def test_classifies_explicit_domain_without_truncation():
    assert prep.classify_pair("ga", "가") == "supported"
    assert prep.classify_pair("ga", "ga") == "copy"
    assert prep.classify_pair("", "가") == "empty"
    assert prep.classify_pair("GA", "가") == "unsupported_ascii"
    assert prep.classify_pair("ga", "가!") == "unsupported_shape"
    assert prep.classify_pair("g" * 49, "가") == "length"
    assert prep.classify_pair("ga", "가" * 25) == "length"


def test_projects_all_alternative_paths_and_deduplicates_requests():
    requests = {}
    row, counts = prep.project_row(
        selected_row(), correspondence_row(), requests,
        lambda target: {"가": "ga", "나": "na", "가나": "gana",
                        "나나": "nana"}[target])
    assert [len(paths) for paths in row["candidate_paths"]] == [2, 1]
    assert row["positive_indices"] == [0]
    assert row["base_scores"] == [1.0, -2.0]
    assert counts["paths"] == 3
    assert set(requests) == {edge["pair_id"] for edge in row["edges"]}
    assert all(edge["status"] == "supported" for edge in row["edges"])


def test_copy_and_unsupported_edges_remain_in_paths_but_not_requests():
    chosen = selected_row() | {"input": "ab", "candidates": ["a가", "AB"],
                               "candidate_path_counts": [1, 1]}
    record = correspondence_row() | {
        "input": "ab",
        "edges": [
            {"input_start": 0, "input_end": 1, "text": "a"},
            {"input_start": 1, "input_end": 2, "text": "가"},
            {"input_start": 0, "input_end": 2, "text": "AB"},
        ],
        "paths": [
            {"path_id": "copy-convert", "text": "a가", "edge_indices": [0, 1]},
            {"path_id": "unsupported", "text": "AB", "edge_indices": [2]},
        ],
    }
    requests = {}
    row, _ = prep.project_row(chosen, record, requests, lambda _: "b")
    assert [edge["status"] for edge in row["edges"]] == ["copy", "supported",
                                                          "unsupported_shape"]
    assert len(requests) == 1
    assert sum(len(paths) for paths in row["candidate_paths"]) == 2


def test_rejects_noncontiguous_or_missing_candidate_paths():
    broken = correspondence_row()
    broken["paths"][0]["edge_indices"] = [1]
    with pytest.raises(ValueError, match="contiguously"):
        prep.project_row(selected_row(), broken, {}, lambda _: "")

    missing = correspondence_row()
    missing["paths"] = missing["paths"][:2]
    with pytest.raises(ValueError, match="missing"):
        prep.project_row(selected_row(), missing, {}, lambda _: "")


def test_pair_ids_are_stable_and_unambiguous():
    first = prep.pair_id("ab", "가")
    assert first == prep.pair_id("ab", "가")
    assert first != prep.pair_id("a", "브가")
