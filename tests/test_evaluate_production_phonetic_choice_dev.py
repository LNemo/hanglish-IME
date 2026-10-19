import copy

import pytest

from scripts import evaluate_production_phonetic_choice_dev as evaluator
from tests.test_evaluate_phonetic_choice_dev import ranked_fixture, comparison_rows


def production_fixture():
    source, previous, response, paths = ranked_fixture()
    before = previous["proposed"]
    before.pop("phonetic_word_union")
    before["version"] = evaluator.PRODUCTION_VERSION
    response.pop("phonetic_word_union")
    response["input"] = source
    response["phonetic_choice"]["parent_version"] = evaluator.PRODUCTION_VERSION
    for row, path in zip(response["phonetic_choice"]["rows"], paths):
        row["segments"] = copy.deepcopy(path[2])
    return source, {"parent": before, "parent_pool": [p[0] for p in paths]}, response, paths


def test_production_capture_preserves_full_pool_and_raw_public_slot():
    source, frozen, response, paths = production_fixture()
    evaluator.verify_capture(source, frozen, response, paths, [0.] * 4)


@pytest.mark.parametrize("mutation", ["extra_proposals", "pool_order", "score", "segments", "input", "parent"])
def test_capture_rejects_generation_drift_and_trace_corruption(mutation):
    source, frozen, response, paths = production_fixture()
    if mutation == "extra_proposals":
        response["phonetic_word_union"] = {}
    elif mutation == "pool_order":
        frozen["parent_pool"].reverse()
    elif mutation == "score":
        response["phonetic_choice"]["rows"][0]["adjusted_score"] += .1
    elif mutation == "segments":
        response["phonetic_choice"]["rows"][0]["segments"][0]["source"] = "def"
    elif mutation == "input":
        response["input"] = "def"
    else:
        response["phonetic_choice"]["parent_version"] = "wrong"
    with pytest.raises(ValueError):
        evaluator.verify_capture(source, frozen, response, paths, [0.] * 4)


def test_unranked_input_remains_unchanged():
    source, frozen, response, _ = production_fixture()
    for value in (frozen["parent"], response):
        value["sentence_ranking_applied"] = False
        value["ctc_word"]["prediction"] = None
    frozen["parent_pool"] = []
    response["phonetic_choice"]["rows"] = []
    evaluator.verify_capture(source, frozen, response, [], [0.] * 4)
    response["candidates"][0]["score"] += 1
    with pytest.raises(ValueError, match="unranked"):
        evaluator.verify_capture(source, frozen, response, [], [0.] * 4)


def test_aggregate_requires_improvement_and_rejects_net_gain_with_losses():
    rows, labels = comparison_rows()
    result = evaluator.aggregate(rows, labels)
    assert result["parent"]["counts"]["top1"] == 262
    assert result["proposed"]["counts"]["top1"] == 274
    assert result["quality_gate"] is True
    rows[0]["proposed"].update(result="wrong", candidates=[{"text": "wrong"}])
    result = evaluator.aggregate(rows, labels)
    assert result["proposed"]["counts"]["top1"] > 262
    assert result["regressions"] == result["top5_losses"] == 1
    assert result["quality_gate"] is False


def test_aggregate_rejects_label_order_and_external_context():
    rows, labels = comparison_rows()
    with pytest.raises(ValueError, match="identities"):
        evaluator.aggregate(rows, list(reversed(labels)))
    labels[0]["context"] = "hint"
    with pytest.raises(ValueError, match="input"):
        evaluator.aggregate(rows, labels)
