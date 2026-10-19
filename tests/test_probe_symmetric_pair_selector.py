import copy
import json

import pytest

from scripts.probe_symmetric_pair_selector import (
    assign_folds, candidate_union, pair_views, relation, choose_pair,
)


def candidate(text, score=1., kind="neural"):
    return {"text": text, "score": score, "segments": [{
        "input_start": 0, "input_end": 6, "source": "imnida",
        "text": text, "kind": kind}]}


def capture():
    return {"input": "imnida", "parent": {"candidates": [candidate("임니다", 3.),
        candidate("입니다", 2.), candidate("imnida", None, "raw")]},
        "child": {"candidates": [candidate("입니다", 4.), candidate("임니다", 3.),
            candidate("imnida", None, "raw")]}}


def test_union_preserves_both_origin_objects_without_synthetic_score():
    source = capture(); before = copy.deepcopy(source)
    current, alternatives = candidate_union(source)
    assert source == before and len(alternatives) == 1
    alternative = alternatives[0]
    assert alternative["origins"]["parent"]["score"] == 2.
    assert alternative["origins"]["child"]["score"] == 4.
    assert alternative["origins"]["parent"]["rank"] == 1
    assert alternative["origins"]["child"]["rank"] == 0
    assert current["origins"]["parent"]["object"] == source["parent"]["candidates"][0]
    assert "score" not in alternative and "rank" not in alternative


def test_protected_candidate_changes_are_excluded():
    source = capture()
    source["parent"]["candidates"][0]["segments"][0]["kind"] = "mixed"
    assert candidate_union(source)[1] == []


def test_swapping_moves_complete_candidate_metadata_and_no_labels_enter_text():
    current, alternatives = candidate_union(capture())
    row = {"input": "imnida", "context": "context", "expected": ["secret gold"],
           "group_id": "private-group", "fold": 3, "id": "private-row"}
    ab, ba = map(json.loads, pair_views(row, current, alternatives[0]))
    assert ab["first"] == ba["second"] and ab["second"] == ba["first"]
    assert ab["roman_input"] == ba["roman_input"] == "imnida"
    assert ab["first"]["policy_role"] == "alternative"
    assert ab["second"]["policy_role"] == "current"
    changed = {**row, "expected": [], "group_id": "other", "fold": 1, "id": "other"}
    assert pair_views(row, current, alternatives[0]) == pair_views(changed, current, alternatives[0])
    assert "secret gold" not in json.dumps(ab)


def test_context_bounds_do_not_truncate_source_or_candidates():
    current, alternatives = candidate_union(capture())
    row = {"input": "x"*2000, "context": "A"*1000+"B"*1000}
    a, b = map(json.loads, pair_views(row, current, alternatives[0]))
    assert a["roman_input"] == row["input"]
    assert a["context"] == "A"*64+"[…중략…]"+"B"*64
    assert a["first"]["text"] == "입니다" and b["first"]["text"] == "임니다"


def test_component_closure_unites_groups_and_duplicate_source_transitively():
    rows = [
        {"id": "a", "group_id": "doc1", "input": " a ", "context": "x"},
        {"id": "b", "group_id": "doc1", "input": "b", "context": "y"},
        {"id": "c", "group_id": "doc2", "input": "a", "context": "x"},
        {"id": "d", "group_id": "doc2", "input": "d", "context": "z"},
    ]
    folds = assign_folds(rows)
    assert all(item == folds[0] for item in folds)
    reordered = assign_folds(list(reversed(rows)))
    assert reordered == list(reversed(folds))


@pytest.mark.parametrize("expected,answer", [(["입니다"], 1), (["임니다"], -1),
    (["입니다", "임니다"], 0), (["different"], 0)])
def test_relation_distinguishes_informative_from_both_or_neither(expected, answer):
    assert relation("임니다", "입니다", expected) == answer


def test_decisions_keep_all_ties_and_choose_unique_positive_maximum():
    assert choose_pair([], []) is None
    assert choose_pair([4, 1], [0., -1.]) is None
    assert choose_pair([4, 1], [.2, .2]) is None
    assert choose_pair([4, 1], [.2, .3]) == 1
    with pytest.raises(ValueError): choose_pair([1], [float("nan")])
    with pytest.raises(ValueError): choose_pair([1], [])
