from copy import deepcopy

import pytest

from scripts.evaluate_contextual_character_dev import ARMS, aggregate, evaluate_row


def fixture(source="testreul handa.", answer="test를 한다."):
    parent = {
        "id": "dev-1", "request": {"input": source, "context": ""},
        "source_row_index": 0,
        "scoring": {"ranked": True, "original": [{"text": answer}], "novel": [],
                    "fallback_public": None},
        "predictions": {arm: [{"text": answer}] for arm in ARMS[:5]},
    }
    capture = {
        "id": "dev-1", "request": parent["request"], "dataset_split": "dev",
        "source_row_index": 0,
        "contextual_scores": {"old_contextual": [0.5], "expanded_contextual": [0.5]},
        "predictions": {"old_contextual": [{"text": answer}],
                        "expanded_contextual": [{"text": answer}]},
    }
    label = {"id": "dev-1", "input": source, "context": "", "split": "dev",
             "expected": [answer]}
    return parent, capture, label


def test_seven_arms_and_separate_production_and_matched_pool_gates():
    parent, capture, label = fixture()
    for arm in ARMS[:5]:
        parent["predictions"][arm] = [{"text": "오답."}]
    row = evaluate_row(parent, capture, label)
    result = aggregate([row])
    assert result["branches"]["expanded_contextual"]["first"] == 1
    assert result["quality_gates"]["production"]["expanded_contextual"]
    assert result["quality_gates"]["matched_pool"]["expanded_contextual"] == {
        "vs_calibrated": True, "vs_character": True}

    parent2, capture2, label2 = fixture()
    parent2["id"] = capture2["id"] = label2["id"] = "dev-2"
    parent2["scoring"]["original"].append({"text": "오답."})
    capture2["contextual_scores"] = {"old_contextual": [0.5, 0.4],
                                    "expanded_contextual": [0.5, 0.4]}
    capture2["predictions"]["old_contextual"] = [{"text": "오답."}]
    capture2["predictions"]["expanded_contextual"] = [{"text": "오답."}]
    loss = evaluate_row(parent2, capture2, label2)
    result = aggregate([row, row, loss])
    assert not result["quality_gates"]["production"]["expanded_contextual"]
    assert not result["quality_gates"]["matched_pool"]["expanded_contextual"]["vs_calibrated"]


def test_utf8_length_and_gold_ascii_strata_are_diagnostics_without_gates():
    short = evaluate_row(*fixture(source="a" * 64, answer="정답"))
    medium = evaluate_row(*fixture(source="가" * 22, answer="정답"))  # 66 UTF-8 bytes.
    long = evaluate_row(*fixture(source="a" * 129, answer="정답"))
    very_long = evaluate_row(*fixture(source="가" * 86, answer="정답"))  # 258 bytes.
    ascii_gold = evaluate_row(*fixture(source="test", answer="test"))
    result = aggregate([short, medium, long, very_long, ascii_gold])
    strata = result["strata"]
    assert [strata["source_utf8_length"][key]["branches"]["current_production"]["rows"]
            for key in ("le_64", "65_128", "129_256", "gt_256")] == [2, 1, 1, 1]
    assert strata["gold_ascii"]["gold_ascii"]["branches"]["current_production"]["rows"] == 1
    assert "quality_gates" not in strata["source_utf8_length"]["le_64"]


def test_no_pool_fallback_and_raw_reserve_outside_scored_pool_are_valid():
    parent, capture, label = fixture()
    source = parent["request"]["input"]
    capture["predictions"]["old_contextual"] = [{"text": source}]
    capture["predictions"]["expanded_contextual"] = [{"text": source}]
    evaluate_row(parent, capture, label)

    parent["scoring"] = {"ranked": False, "original": [], "novel": [],
                         "fallback_public": [{"text": source}]}
    capture["contextual_scores"] = {"old_contextual": [], "expanded_contextual": []}
    row = evaluate_row(parent, capture, label)
    assert not row["ranked"]


def test_contextual_score_pool_mismatch_is_rejected():
    parent, capture, label = fixture()
    capture["contextual_scores"]["old_contextual"] = []
    with pytest.raises(ValueError, match="score pool"):
        evaluate_row(parent, capture, label)


def test_old_scores_must_be_exact_expanded_prefix():
    parent, capture, label = fixture()
    parent["scoring"]["novel"] = [{"text": "다른 후보", "status": "scored"}]
    capture["contextual_scores"] = {"old_contextual": [0.5],
                                    "expanded_contextual": [0.4, 0.3]}
    with pytest.raises(ValueError, match="prefix"):
        evaluate_row(parent, capture, label)


def test_multi_reference_and_unsupported_novel_do_not_inflate_pool_coverage():
    parent, capture, label = fixture()
    label["expected"].append("다른 정답")
    parent["scoring"]["original"] = []
    parent["scoring"]["novel"] = [{"text": "다른 정답", "status": "unsupported"}]
    capture["contextual_scores"] = {"old_contextual": [], "expanded_contextual": []}
    capture["predictions"] = {"old_contextual": [{"text": parent["request"]["input"]}],
                              "expanded_contextual": [{"text": parent["request"]["input"]}]}
    row = evaluate_row(parent, capture, label)
    assert not row["arms"]["expanded_contextual"]["pool_covered"]
