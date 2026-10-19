import copy

import pytest

from scripts import analyze_phonetic_choice_production_pool as analysis


def segment(source, text, kind="rule"):
    return [{"input_start": 0, "input_end": len(source), "source": source,
             "text": text, "kind": kind}]


def response(source, paths):
    candidates = analysis.render_public(source, paths)
    return {"result": candidates[0]["text"], "candidates": candidates}


def cached_row():
    source = "source"
    base = [
        ("부모A", 3.0, segment(source, "부모A")),
        ("추가", 2.5, segment(source, "추가")),
        ("부모B", 2.0, segment(source, "부모B")),
    ]
    features = [
        {"text": text, "base_score": score, "segments": segments,
         "features": [float(index), 0.0, 0.0, 0.0]}
        for index, (text, score, segments) in enumerate(base)
    ]
    production = [base[0], base[2]]
    return {"id": "row", "request": {"input": source, "context": ""},
            "parent_pool": ["부모A", "부모B"], "base_paths": base,
            "parent": response(source, production),
            "proposed": {"sentence_ranking_applied": True,
                         "phonetic_choice": {"rows": features}}}


def test_derive_row_filters_union_extras_and_reuses_saved_features():
    row = cached_row()
    derived = analysis.derive_row(row, [1.0, 0.0, 0.0, 0.0])

    assert [path[0] for path in derived["production_paths"]] == ["부모A", "부모B"]
    assert [path[0] for path in derived["calibrated_paths"]] == ["부모B", "부모A"]
    assert derived["derived"]["result"] == "부모B"
    assert all(candidate["text"] != "추가" for candidate in derived["derived"]["candidates"])


def test_derive_row_rejects_missing_or_duplicate_production_membership():
    row = cached_row(); row["parent_pool"] = ["부모A", "없는후보"]
    with pytest.raises(ValueError, match="membership"):
        analysis.derive_row(row, [0.0] * 4)


def test_unranked_row_passes_through_exactly():
    parent = {"result": "raw", "candidates": [{"text": "raw", "score": None,
        "segments": [{"input_start": 0, "input_end": 3, "source": "raw",
                      "text": "raw", "kind": "raw"}]}]}
    row = {"id": "raw", "request": {"input": "raw", "context": ""},
           "parent_pool": [], "base_paths": [], "parent": parent,
           "proposed": {"sentence_ranking_applied": False,
                        "phonetic_choice": {"rows": []}}}
    derived = analysis.derive_row(row, [0.0] * 4)
    assert derived["derived"] == parent
    assert derived["ranked"] is False


def test_metrics_blocks_regression_even_with_a_correction():
    rows, labels = [], []
    for index, (old, new) in enumerate((("gold", "wrong"), ("wrong", "gold"))):
        source = f"src{index}"
        rows.append({"id": str(index), "request": {"input": source, "context": ""},
                     "parent": {"result": old, "candidates": [{"text": old}]},
                     "derived": {"result": new, "candidates": [{"text": new}]}})
        labels.append({"id": str(index), "input": source, "context": "", "split": "dev",
                       "expected": ["gold"]})
    # The production evaluator uses 316 rows. Repeat neutral rows to preserve
    # that fixed denominator without adding any additional changes.
    for index in range(2, 316):
        source = f"src{index}"
        response_row = {"result": "gold", "candidates": [{"text": "gold"}]}
        rows.append({"id": str(index), "request": {"input": source, "context": ""},
                     "parent": copy.deepcopy(response_row), "derived": copy.deepcopy(response_row)})
        labels.append({"id": str(index), "input": source, "context": "", "split": "dev",
                       "expected": ["gold"]})
    result = analysis.metrics(rows, labels)
    assert result["corrections"] == 1
    assert result["regressions"] == 1
    assert result["strict_gate"] is False
