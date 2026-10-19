from copy import deepcopy

from scripts.evaluate_contextual_character_dev import aggregate, evaluate_row
from scripts.verify_contextual_character_dev_metrics import (
    independent_outcome, independent_summary,
)
from tests.test_evaluate_contextual_character_dev import fixture


def test_independent_recount_matches_but_does_not_import_evaluator_helpers():
    parent, capture, label = fixture()
    parent["predictions"]["expanded_calibrated"] = [{"text": "오답."}]
    parent["predictions"]["expanded_character"] = [{"text": "오답."}]
    expected = evaluate_row(parent, capture, label)
    actual = independent_outcome(parent, capture, label)
    assert actual == expected
    assert independent_summary([actual]) == aggregate([expected])


def test_independent_recount_accepts_raw_reserve_and_rejects_changed_fallback():
    parent, capture, label = fixture()
    source = parent["request"]["input"]
    capture["predictions"] = {"old_contextual": [{"text": source}],
                              "expanded_contextual": [{"text": source}]}
    independent_outcome(parent, capture, label)

    parent["scoring"] = {"ranked": False, "original": [], "novel": [],
                         "fallback_public": [{"text": source}]}
    capture["contextual_scores"] = {"old_contextual": [], "expanded_contextual": []}
    changed = deepcopy(capture)
    changed["predictions"]["expanded_contextual"] = [{"text": "바뀜"}]
    try:
        independent_outcome(parent, changed, label)
    except ValueError as error:
        assert "fallback" in str(error)
    else:
        raise AssertionError("changed no-pool fallback was accepted")


def test_independent_strata_recount_keeps_empty_buckets_and_no_stratum_gates():
    parent, capture, label = fixture(source="가" * 90, answer="정답")
    row = independent_outcome(parent, capture, label)
    summary = independent_summary([row])
    assert summary["strata"]["source_utf8_length"]["gt_256"]["branches"]["old_contextual"]["rows"] == 1
    assert summary["strata"]["source_utf8_length"]["le_64"]["branches"]["old_contextual"]["rows"] == 0
    assert "quality_gates" not in summary["strata"]["source_utf8_length"]["gt_256"]

