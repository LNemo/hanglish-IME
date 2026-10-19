from copy import deepcopy
import math

import pytest

from scripts.evaluate_lexical_boundary_fit import display, summarize


def path(text, score):
    return {"text": text, "score": score, "segments": [{"input_start": 0,
        "input_end": 5, "source": "roman", "text": text, "kind": "morphology"}]}


def cohort():
    rows, captures = [], []
    for index in range(512):
        base = [path("base", 2.0)]
        extras = [path("new", 1.0)]
        child_scores = [2.0, 3.0] if index == 0 else [2.0, 1.0]
        parent = {"candidates": display(base, [2.0], "roman"), "result": "base"}
        child = {"candidates": display(base + extras, child_scores, "roman"),
                 "result": "new" if index == 0 else "base"}
        source = {"id": f"r{index:04d}", "input": "roman", "context": "context"}
        rows.append({**source, **deepcopy(parent), "expected": ["new" if index == 0 else "base"]})
        captures.append({**source, "parent": parent, "child": child,
            "parent_raw": {"result": "roman"}, "child_raw": {"result": "roman"},
            "parent_pool": deepcopy(base), "trace": {"preserved_pool": deepcopy(base),
                "accepted_new_extras": extras},
            "parent_sentence_scored": True, "child_sentence_scored": True,
            "parent_score_inputs": ["base"], "parent_scores": [2.0],
            "child_score_inputs": ["base", "new"], "child_scores": child_scores})
    return rows, captures


def test_actual_score_replay_counts_first_choice_and_top5_gains():
    rows, captures = cohort()
    report = summarize(rows, captures)
    assert report["counts"]["before_top1"] == report["counts"]["before_top5"] == 511
    assert report["counts"]["top1"] == report["counts"]["top5"] == 512
    assert report["correction_ids"] == report["top5_gain_ids"] == ["r0000"]
    assert report["gate"]["passed"] is True
    assert report["promotion_allowed"] is False


def test_one_regression_rejects_even_with_a_correction():
    rows, captures = cohort()
    c = captures[1]
    c["child_scores"] = [2.0, 3.0]
    c["child"]["candidates"] = display(c["parent_pool"] + c["trace"]["accepted_new_extras"],
                                        c["child_scores"], "roman")
    c["child"]["result"] = "new"
    report = summarize(rows, captures)
    assert report["gate"]["passed"] is False
    assert report["regression_ids"] == ["r0001"]
    assert report["top5_loss_ids"] == []


@pytest.mark.parametrize("mutate", [
    lambda c: c["trace"]["preserved_pool"][0].update(score=99.0),
    lambda c: c.update(child_score_inputs=["new", "base"]),
    lambda c: c.update(child_scores=[2.0]),
    lambda c: c.update(child_scores=[2.0, math.nan]),
    lambda c: c["child"]["candidates"][0].update(score=123.0),
    lambda c: c.update(child_raw={"result": "changed"}),
    lambda c: c.update(child_sentence_scored=False),
    lambda c: c["child"].update(result="inconsistent"),
])
def test_tampered_capture_cannot_be_reported_as_valid(mutate):
    rows, captures = cohort()
    mutate(captures[511])
    with pytest.raises(ValueError):
        summarize(rows, captures)


def test_incomplete_or_reordered_cohort_rejected():
    rows, captures = cohort()
    with pytest.raises(ValueError, match="complete fixed FIT512"):
        summarize(rows, captures[:-1])
    captures[-1]["id"] = captures[0]["id"]
    with pytest.raises(ValueError, match="identity or ordering"):
        summarize(rows, captures)


def test_every_prediction_replayed_before_any_label_access():
    class TargetTrap(dict):
        def __getitem__(self, key):
            if key == "expected":
                raise AssertionError("labels accessed before integrity failure")
            return super().__getitem__(key)
    rows, captures = cohort()
    captures[-1]["child_scores"] = [2.0, float("inf")]
    with pytest.raises(ValueError, match="score vector"):
        summarize([TargetTrap(row) for row in rows], captures)


def test_display_keeps_original_raw_and_whole_candidate_segments():
    pool = [path(str(i), float(i)) for i in range(8)]
    result = display(pool, [float(i) for i in range(8)], "roman")
    assert [c["text"] for c in result] == ["7", "6", "5", "4", "roman"]
    assert result[0]["segments"] == pool[7]["segments"]
    assert result[-1]["score"] is None


def test_unscored_branch_must_be_exact_parent_response():
    rows, captures = cohort()
    c = captures[-1]
    c.update(parent_sentence_scored=False, child_sentence_scored=False,
             child_score_inputs=None, child_scores=None, child=deepcopy(c["parent"]))
    c["trace"]["accepted_new_extras"] = []
    assert summarize(rows, captures)["gate"]["passed"]
    c["child"]["result"] = "new"
    with pytest.raises(ValueError, match="unsupported parent branch changed"):
        summarize(rows, captures)
