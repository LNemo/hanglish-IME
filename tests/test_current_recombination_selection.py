from copy import deepcopy

import pytest

from scripts.current_recombination_selection import (
    calibrated_score, compare_row, render_public, validate_model,
)


def segment(source, text):
    return [{"input_start": 0, "input_end": len(source), "source": source,
             "text": text, "kind": "fixture"}]


def candidate(source, text, score, feature, status="preserved"):
    return {"text": text, "sentence_score": score, "features": [feature, 0.0, 0.0, 0.0],
            "segments": segment(source, text), "status": status,
            "score_source": "captured_original" if status == "preserved" else None}


def model(weight=2.0):
    return {"schema": "phonetic-score-scale-model-v1", "arm": "learned_scaled",
            "parameters": [1.0, weight, 0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0, 1.0]}


def scored():
    source = "raw source"
    return {"schema": "current-recombination-scoring-row-v1", "id": "row-1",
            "component_id": "component-1", "request": {"input": source, "context": ""},
            "ranked": True,
            "original": [candidate(source, "old-best", 3.0, -2.0),
                         candidate(source, "old-gold", 2.0, 2.0)],
            "novel": [candidate(source, "novel-gold", 4.0, 1.0, "scored"),
                      {"proposal_id": "unsupported", "text": "unsupported", "segments": [],
                       "edge_ids": [], "features": [0.0] * 4, "status": "unsupported",
                       "reason": "sentence_ranker_can_score_false", "sentence_score": None}],
            "fallback_public": None, "original_score_parity": None}


def test_four_arms_capture_actual_rank_changes_without_mutating_scores():
    row = scored(); frozen = deepcopy(row)
    result = compare_row(row, ["novel-gold"], model())
    assert result["arms"]["old_sentence"]["first"] is False
    assert result["arms"]["expanded_sentence"]["first"] is True
    assert result["arms"]["old_calibrated"]["public"][0]["text"] == "old-gold"
    assert result["arms"]["expanded_calibrated"]["public"][0]["text"] == "novel-gold"
    assert result["transitions"]["candidate_expansion_sentence"]["corrections"] == 1
    assert result["transitions"]["candidate_expansion_calibrated"]["corrections"] == 1
    assert row == frozen


def test_unsupported_candidates_are_counted_but_never_ranked():
    result = compare_row(scored(), ["unsupported"], model())
    assert result["counts"]["unsupported_novel"] == 1
    assert result["counts"]["unsupported_reasons"] == {"sentence_ranker_can_score_false": 1}
    assert all(not arm["pool_coverage"] for arm in result["arms"].values())
    assert all("unsupported" not in [item["text"] for item in arm["public"]]
               for arm in result["arms"].values())


def test_no_pool_keeps_nonraw_fallback_identical_in_all_arms():
    source = "roman"
    fallback = [{"text": "로만", "score": None, "segments": segment(source, "로만")},
                {"text": source, "score": None, "segments": segment(source, source)}]
    row = {"schema": "current-recombination-scoring-row-v1", "id": "empty",
           "request": {"input": source, "context": ""}, "ranked": False,
           "original": [], "novel": [], "fallback_public": fallback}
    result = compare_row(row, ["로만"], model())
    assert all(arm["public"] == fallback and arm["first"] and not arm["pool_coverage"]
               for arm in result["arms"].values())
    assert all(not any(transition.values()) for transition in result["transitions"].values())


def test_raw_slot_and_tie_order_match_public_contract_even_if_raw_is_sixth():
    source = "raw"
    pool = [{"text": str(index), "score": 1.0, "segments": []} for index in reversed(range(5))]
    pool.append({"text": source, "score": 0.0, "segments": segment(source, source)})
    public = render_public(source, pool)
    assert [row["text"] for row in public] == ["0", "1", "2", "3", "raw"]
    assert public[-1]["score"] is None and public[-1]["segments"][0]["kind"] == "raw"


@pytest.mark.parametrize("change", ["parameters", "scales", "alpha", "nan", "bool"])
def test_rejects_invalid_calibration_models(change):
    value = model()
    if change == "parameters": value["parameters"] = value["parameters"][:4]
    elif change == "scales": value["scale"][0] = 0.0
    elif change == "alpha": value["parameters"][0] = -0.1
    elif change == "nan": value["parameters"][2] = float("nan")
    else: value["parameters"][0] = True
    with pytest.raises(ValueError, match="calibration model"):
        validate_model(value)


def test_ties_sort_by_text_before_reserved_raw_replacement():
    source = "source"
    row = scored(); row["request"]["input"] = source
    for candidate_row in row["original"] + row["novel"][:1]:
        candidate_row["sentence_score"] = 1.0
    result = compare_row(row, ["novel-gold"], model(weight=0.0))
    texts = [item["text"] for item in result["arms"]["expanded_sentence"]["public"]]
    assert texts[:3] == ["novel-gold", "old-best", "old-gold"]


def test_calibration_uses_runtime_feature_division_before_weight_multiplication():
    weight = -4.898619485211566
    feature = -0.09129825816118142
    scale = 0.3125436271078933
    row = {"sentence_score": 0.0, "features": [feature, 0.0, 0.0, 0.0]}
    value = calibrated_score(row, {"parameters": [0.0, weight, 0.0, 0.0, 0.0],
                                   "scale": [scale, 1.0, 1.0, 1.0]})
    assert value == weight * (feature / scale)
    assert value != weight * feature / scale
