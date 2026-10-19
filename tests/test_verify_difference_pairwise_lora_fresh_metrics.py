from copy import deepcopy
import json

import pytest

from scripts.difference_pairwise_lora_evaluation import resolve_selection
from scripts.difference_pairwise_lora_inference import prepare_inference_selection
from scripts import verify_difference_pairwise_lora_fresh_metrics as audit


@pytest.fixture
def cohort():
    rows, predictions, targets = [], [], []
    for index in range(888):
        role = "fresh_natural" if index < 512 else "challenge_preservation"
        candidates = [{"text": text, "score": float(2 - i), "segments": [{
            "input_start": 0, "input_end": 6, "source": "imnida", "text": text,
            "kind": "morphology"}]} for i, text in enumerate(("임니다", "입니다"))]
        row = {"id": f"r{index}", "group_id": f"g{index % 387}" if index < 512 else f"p{index}",
               "origin": "synthetic", "role": role, "input": "imnida", "context": "",
               "candidates": candidates,
               "parent_response": {"input": "imnida", "context": "", "raw": False,
                   "version": audit.VERSION, "candidates": candidates, "result": "임니다"},
               "raw_bypass_response": {"input": "imnida", "context": "", "raw": True,
                   "version": audit.VERSION, "result": "imnida", "candidates": [{"text": "imnida"}]}}
        selection = prepare_inference_selection(role, row)
        repaired = index < 40
        logits = {p["pair_id"]: [0.0, 1.0 if repaired else -1.0] for p in selection["pairs"]}
        prediction = resolve_selection(selection, logits)
        rows.append(row)
        predictions.append(prediction)
        targets.append({"id": row["id"], "group_id": row["group_id"], "role": role,
                        "expected": ["입니다" if repaired else "임니다"]})
    # This is the boundary that the existing object-identity TRAIN checker
    # cannot validate: each document is independently serialized/deserialized.
    return tuple(json.loads(json.dumps(part, ensure_ascii=False)) for part in (rows, predictions, targets))


def test_independent_replay_counts_gain_and_preservation(cohort):
    result = audit.recompute_metrics(*cohort)
    fresh = result["roles"]["fresh_natural"]
    assert (fresh["rows"], fresh["before_top1"], fresh["top1"], fresh["corrections"], fresh["regressions"]) == (512, 472, 512, 40, 0)
    assert (fresh["actions"], fresh["fallbacks"]) == (40, 472)
    assert result["roles"]["challenge_preservation"]["top1"] == 376
    assert result["bootstrap"]["groups"] == 387
    assert result["bootstrap"]["lower_index"] == 49
    assert result["bootstrap"]["lower_2_5_percentile"] > 0
    assert result["gate"]["passed"] is True
    proof = audit.verify_reported_metrics(*cohort, result)
    assert proof["verification_passed"] is True
    assert proof["execution_verified"] is False
    assert proof["promotion_allowed"] is False


def test_independent_and_producer_metrics_match_on_json_roundtrip(cohort):
    from scripts.difference_pairwise_lora_fresh_metrics import score_closed_predictions

    reported = score_closed_predictions(*cohort)
    assert audit.verify_reported_metrics(*cohort, reported)["quality_gate_passed"] is True


@pytest.mark.parametrize("index,passed", [(80, True), (800, False)])
def test_allowed_but_wrong_switch_is_counted_by_both_implementations(cohort, index, passed):
    from scripts.difference_pairwise_lora_fresh_metrics import score_closed_predictions

    rows, predictions, targets = cohort
    selection = prepare_inference_selection(rows[index]["role"], rows[index])
    predictions[index] = json.loads(json.dumps(resolve_selection(
        selection, {p["pair_id"]: [0.0, 3.0] for p in selection["pairs"]})))
    report = score_closed_predictions(*cohort)
    assert report["roles"][rows[index]["role"]]["regressions"] == 1
    assert report["gate"]["passed"] is passed
    assert audit.verify_reported_metrics(*cohort, report)["quality_gate_passed"] is passed


@pytest.mark.parametrize("field", ["top1", "gain", "bootstrap", "gate"])
def test_forged_metrics_rejected(cohort, field):
    report = audit.recompute_metrics(*cohort)
    if field == "top1":
        report["roles"]["fresh_natural"]["top1"] -= 1
    elif field == "gain":
        report["roles"]["fresh_natural"]["net_gain"] += 1
    elif field == "bootstrap":
        report["bootstrap"]["lower_2_5_percentile"] += 0.1
    else:
        report["gate"]["promotion_allowed"] = True
    with pytest.raises(ValueError, match="reported fresh metrics differ"):
        audit.verify_reported_metrics(*cohort, report)


@pytest.mark.parametrize("field", ["result", "candidate", "raw", "pair_id", "margin", "logit", "extra_pair", "raw_mode"])
def test_tampered_prediction_rejected_before_target_access(cohort, field):
    rows, predictions, targets = cohort

    class TargetTrap(dict):
        def get(self, *args):
            raise AssertionError("target opened before all prediction replay")

        def __getitem__(self, key):
            raise AssertionError("target opened before all prediction replay")

    value = predictions[-1]
    if field == "result":
        value["result"] = "입니다"
    elif field == "candidate":
        value["candidates"][0]["segments"][0]["input_end"] = 5
    elif field == "raw":
        value["raw_bypass_response"]["result"] = "changed"
    elif field == "pair_id":
        value["pair_scores"][0]["pair_id"] = "wrong"
    elif field == "margin":
        value["pair_scores"][0]["margin"] = 100.0
    elif field == "logit":
        value["pair_scores"][0]["keep_logit"] = float("nan")
    elif field == "extra_pair":
        value["pair_scores"].append(deepcopy(value["pair_scores"][0]))
    else:
        value["raw_mode"] = True
    with pytest.raises(ValueError):
        audit.recompute_metrics(rows, predictions, [TargetTrap(t) for t in targets])


def test_candidate_absence_remains_in_denominator_and_cannot_be_forged(cohort):
    rows, predictions, targets = cohort
    targets[100]["expected"] = ["존재하지 않는 후보"]
    result = audit.recompute_metrics(*cohort)
    assert result["roles"]["fresh_natural"]["candidate_absent"] == 1
    assert result["roles"]["fresh_natural"]["rows"] == 512
    assert result["roles"]["fresh_natural"]["top1"] == 511
    predictions[100]["result"] = targets[100]["expected"][0]
    with pytest.raises(ValueError, match="does not exactly replay"):
        audit.recompute_metrics(*cohort)


def test_preservation_failure_is_valid_quality_rejection(cohort):
    targets = cohort[2]
    targets[-1]["expected"] = ["입니다"]
    result = audit.recompute_metrics(*cohort)
    assert result["gate"]["fresh_passed"] is True
    assert result["roles"]["challenge_preservation"]["top1"] == 375
    assert result["gate"]["passed"] is False
    assert audit.verify_reported_metrics(*cohort, result)["quality_gate_passed"] is False


@pytest.mark.parametrize("corruption", ["short", "duplicate", "role", "target_order", "answers", "missing_source"])
def test_complete_cohort_and_target_identity_required(cohort, corruption):
    rows, predictions, targets = cohort
    if corruption == "short":
        predictions.pop()
    elif corruption == "duplicate":
        rows[-1]["id"] = rows[-2]["id"]
    elif corruption == "role":
        rows[-1]["role"] = "fresh_natural"
    elif corruption == "target_order":
        targets[0], targets[1] = targets[1], targets[0]
    elif corruption == "missing_source":
        del rows[-1]["parent_response"]
    else:
        targets[0]["expected"] = ["입니다", "입니다"]
    with pytest.raises(ValueError):
        audit.recompute_metrics(*cohort)
