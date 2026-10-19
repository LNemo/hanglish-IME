from copy import deepcopy

import pytest

from scripts.audit_semantic_context_learning import interventions, summarize


def test_donor_pairing_is_train_only_same_input_different_sense_and_order_independent():
    rows = [{"id": str(i), "input": roman, "expected": [target], "context": str(i), "split": "train"}
            for i, (roman, target) in enumerate((("bak", "박"), ("bak", "밖"), ("bit", "빛"), ("bit", "빚")))]
    first = {r["id"]: r for r in interventions(rows)}
    assert first == {r["id"]: r for r in interventions(rows[::-1])}
    original = {r["id"]: r for r in rows}
    for r in first.values():
        assert original[r["donor_id"]]["input"] == r["input"]
        assert r["expected"] != r["donor_expected"]
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="TRAIN"):
        interventions(rows)


def test_donor_pairing_rejects_missing_contrast():
    with pytest.raises(ValueError, match="missing"):
        interventions([{"id": "1", "input": "bak", "expected": ["박"], "context": "ctx", "split": "train"}])


def test_summary_distinguishes_logit_learning_from_runtime_bias_and_context_sensitivity():
    def observation(result, scores):
        return {"response": {"result": result}, "calls": [{"candidates": ["박", "밖"], "scores": scores}],
                "context_crop_changed": False, "token_budget_exceeded": False}
    row = {"arm": "parent", "expected": ["밖"], "donor_expected": ["박"],
           "own": observation("박", [0., 1.]), "donor": observation("박", [1., 0.]),
           "blank": observation("박", [1., 0.])}
    trained = deepcopy(row)
    trained["arm"] = "augmented"
    trained["own"]["response"]["result"] = "밖"
    report = summarize([row, trained])
    assert report["parent"]["own_logits_only_top1"] == 1
    assert report["parent"]["bias_losses"] == 1
    assert report["parent"]["both_contexts_correct"] == 0
    assert report["augmented"]["both_contexts_correct"] == 1
    assert report["augmented"]["own_differs_from_blank"] == 1
