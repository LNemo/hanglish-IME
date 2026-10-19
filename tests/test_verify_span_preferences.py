from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_span_preferences.py"
SPEC = importlib.util.spec_from_file_location("verify_span_preferences", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def compact(source, variants, row_id="row", component="component"):
    edges, edge_map, paths = [], {}, []
    for parts in variants:
        cursor = 0
        indices = []
        for source_part, text in parts:
            start, end = cursor, cursor + len(source_part)
            assert source[start:end] == source_part
            key = (start, end, text)
            if key not in edge_map:
                edge_map[key] = len(edges)
                edges.append({"input_start": start, "input_end": end, "text": text})
            indices.append(edge_map[key])
            cursor = end
        assert cursor == len(source)
        identity = [source, [[edges[index]["input_start"], edges[index]["input_end"],
                              edges[index]["text"]] for index in indices]]
        paths.append({"path_id": "alignment-v1-" + verifier.digest(identity),
                      "text": "".join(edges[index]["text"] for index in indices),
                      "edge_indices": indices})
    return {"schema": verifier.PATH_SCHEMA, "id": row_id,
            "component_id": component, "input": source, "edges": edges,
            "paths": paths, "reference_path_id": paths[0]["path_id"]}


def selection(source, candidates, positives=(0,)):
    return {"id": "row", "input": source, "component_id": "component",
            "arms": {"b_union": {"candidates": candidates,
                                   "positive_indices": list(positives)}}}


def groups(record, indexes):
    return [[verifier.expand_path(record, record["paths"][index])]
            for index in indexes]


def test_reconstructs_local_change_with_complete_path_witness_offsets():
    source = "testreul hamnida"
    record = compact(source, [
        [("test", "test"), ("reul", "를"), (" ", " "), ("hamnida", "합니다")],
        [("test", "test"), ("reul", "를"), (" ", " "), ("hamnida", "함니다")],
    ])
    candidates = [path["text"] for path in record["paths"]]
    result = verifier.expected_row(selection(source, candidates), record)
    pair, = result["preferences"]
    assert (pair["input_start"], pair["input_end"], pair["source"]) == (9, 16, "hamnida")
    assert pair["prefix"] == "test를 " and pair["suffix"] == ""
    assert pair["witnesses"][0]["preferred"]["span_indices"] == [3]
    assert result["counts"]["eligible_path_pairs"] == 1


def test_rejects_every_nonlocal_or_ambiguous_exclusion_class():
    multi = compact("a b", [[("a", "아"), (" ", " "), ("b", "비")],
                             [("a", "어"), (" ", " "), ("b", "바")]])
    values = [path["text"] for path in multi["paths"]]
    _, counts = verifier.reconstruct_preferences("a b", values, [0], groups(multi, [0, 1]))
    assert counts["excluded_multiple_or_zero_changes"] == 1

    whole = compact("ab", [[("ab", "아비")], [("ab", "아바")]])
    values = [path["text"] for path in whole["paths"]]
    _, counts = verifier.reconstruct_preferences("ab", values, [0], groups(whole, [0, 1]))
    assert counts["excluded_whole_source"] == 1

    segmented = compact("abc", [[("a", "아"), ("bc", "비시")],
                                 [("a", "어"), ("b", "비"), ("c", "시")]])
    values = [path["text"] for path in segmented["paths"]]
    _, counts = verifier.reconstruct_preferences("abc", values, [0], groups(segmented, [0, 1]))
    assert counts["excluded_outside_segmentation_changes"] == 1

    ambiguous = compact("abc", [[("a", "A"), ("b", "B"), ("c", "C")],
                                [("ab", "AB"), ("c", "C")],
                                [("a", "X"), ("b", "B"), ("c", "C")]])
    paths = [verifier.expand_path(ambiguous, value) for value in ambiguous["paths"]]
    result, counts = verifier.reconstruct_preferences(
        "abc", ["ABC", "XBC"], [0], [[paths[0], paths[1]], [paths[2]]])
    assert result == []
    assert counts["excluded_ambiguous_interval_path_pairs"] == 2


@pytest.mark.parametrize("mutation", ["preference", "path_id", "span_index", "accepted_label"])
def test_exact_row_check_rejects_tampered_labels_paths_and_indices(mutation):
    source = "a b"
    record = compact(source, [[("a", "아"), (" ", " "), ("b", "비")],
                              [("a", "어"), (" ", " "), ("b", "비")]])
    candidates = [path["text"] for path in record["paths"]]
    selected = selection(source, candidates)
    expected = verifier.expected_row(selected, record)
    actual = deepcopy(expected)
    if mutation == "preference":
        actual["preferences"][0]["preferred_text"] = "조작"
    elif mutation == "path_id":
        actual["preferences"][0]["witnesses"][0]["preferred"]["path_id"] = "bad"
    elif mutation == "span_index":
        actual["preferences"][0]["witnesses"][0]["preferred"]["span_indices"] = [99]
    else:
        changed_selection = selection(source, candidates, positives=(1,))
        expected = verifier.expected_row(changed_selection, record)
    with pytest.raises(ValueError, match="preference row differs"):
        verifier.validate_rows([actual], [expected])


def test_exact_row_check_rejects_row_preference_and_witness_omissions():
    source = "a b"
    record = compact(source, [[("a", "아"), (" ", " "), ("b", "비")],
                              [("a", "어"), (" ", " "), ("b", "비")]])
    candidates = [path["text"] for path in record["paths"]]
    expected = verifier.expected_row(selection(source, candidates), record)
    with pytest.raises(ValueError, match="omission"):
        verifier.validate_rows([], [expected])
    for key in ("preferences",):
        actual = deepcopy(expected); actual[key] = []
        with pytest.raises(ValueError, match="preference row differs"):
            verifier.validate_rows([actual], [expected])
    actual = deepcopy(expected); actual["preferences"][0]["witnesses"] = []
    with pytest.raises(ValueError, match="preference row differs"):
        verifier.validate_rows([actual], [expected])


@pytest.mark.parametrize("field,value", [
    ("edge_index", True), ("input_start", 1), ("input_end", 99),
    ("text", None), ("path_id", "alignment-v1-bad")])
def test_compact_path_tampering_is_rejected(field, value):
    record = compact("ab", [[("a", "아"), ("b", "비")],
                            [("a", "어"), ("b", "비")]])
    changed = deepcopy(record)
    if field == "edge_index":
        changed["paths"][0]["edge_indices"][0] = value
    elif field == "path_id":
        changed["paths"][0][field] = value
    else:
        changed["edges"][0][field] = value
    with pytest.raises(ValueError):
        verifier.expand_path(changed, changed["paths"][0])


def test_aggregate_recomputes_direction_conflicts_actions_and_ascii_flags():
    source = "testx z"
    first = compact(source, [[("testx", "test"), (" ", " "), ("z", "지")],
                             [("testx", "테스트"), (" ", " "), ("z", "지")]])
    candidates = [path["text"] for path in first["paths"]]
    row1 = verifier.expected_row(selection(source, candidates), first)
    row2 = deepcopy(row1)
    row2["id"] = "row2"; row2["component_id"] = "component2"
    pair = row2["preferences"][0]
    pair["preferred_text"], pair["dispreferred_text"] = (
        pair["dispreferred_text"], pair["preferred_text"])
    summary = verifier.aggregate([row1, row2])
    assert summary["counts"]["rows"] == 2
    assert summary["local_pairs_with_both_preferences"] == 1
    assert summary["contextual_pairs_with_both_preferences"] == 1
    assert summary["flags"]["preferred:unresolved_ascii_witnesses"] == 2


def test_verifier_source_contains_no_model_torch_or_producer_imports():
    source = SCRIPT.read_text()
    assert "import torch" not in source
    assert "from hanglish" not in source
    assert "from scripts" not in source


def test_alternate_artifact_directory_is_not_authorized(tmp_path):
    with pytest.raises(ValueError, match="alternate output"):
        verifier.verify(tmp_path)
