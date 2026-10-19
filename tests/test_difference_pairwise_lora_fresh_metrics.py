from copy import deepcopy
import json

import pytest

from scripts.difference_pairwise_lora_evaluation import resolve_selection
from scripts.difference_pairwise_lora_fresh_metrics import (
    FRESH_ROLE,
    PRESERVATION_ROLE,
    SCHEMA,
    VERSION,
    score_closed_predictions,
)
from scripts.difference_pairwise_lora_inference import prepare_inference_selection


def _candidate(source, text, score):
    return {
        "text": text,
        "score": score,
        "segments": [{
            "input_start": 0,
            "input_end": len(source),
            "source": source,
            "text": text,
            "kind": "morphology",
        }],
    }


def _row(index, role, *, parent_correct):
    source = "imnida"
    texts = ("입니다", "임니다") if parent_correct else ("임니다", "입니다")
    candidates = [_candidate(source, text, 2.0 - position)
                  for position, text in enumerate(texts)]
    context = f"문서 {index}의 앞 문맥"
    parent = {
        "input": source,
        "context": context,
        "version": VERSION,
        "raw": False,
        "result": candidates[0]["text"],
        "candidates": candidates,
    }
    raw = {
        "input": source,
        "context": context,
        "version": VERSION,
        "raw": True,
        "result": source,
        "candidates": [{"text": source}],
    }
    return {
        "role": role,
        "id": f"row-{index:04d}",
        "group_id": f"fresh-group-{index % 387:03d}" if role == FRESH_ROLE
                    else f"preserve-group-{index:03d}",
        "input": source,
        "context": context,
        "parent_response": parent,
        "candidates": candidates,
        "raw_bypass_response": raw,
    }


def _prediction(row, *, switch):
    selection = prepare_inference_selection(row["role"], row)
    logits = {
        pair["pair_id"]: ([0.0, 1.0] if switch else [1.0, 0.0])
        for pair in selection["pairs"]
    }
    return resolve_selection(selection, logits)


def _cohort(*, fresh_mode="gain"):
    rows = []
    predictions = []
    targets = []
    for index in range(888):
        role = FRESH_ROLE if index < 512 else PRESERVATION_ROLE
        if role == PRESERVATION_ROLE:
            parent_correct, switch = True, False
        elif fresh_mode == "gain":
            parent_correct, switch = False, True
        elif fresh_mode == "keep":
            parent_correct, switch = False, False
        elif fresh_mode == "gain_with_regression":
            if index < 40:
                parent_correct, switch = False, True
            elif index == 40:
                parent_correct, switch = True, True
            else:
                parent_correct, switch = True, False
        else:
            raise AssertionError(fresh_mode)
        row = _row(index, role, parent_correct=parent_correct)
        rows.append(row)
        predictions.append(_prediction(row, switch=switch))
        targets.append({
            "id": row["id"],
            "group_id": row["group_id"],
            "role": role,
            "expected": ["입니다"],
        })
    return rows, predictions, targets


@pytest.fixture(scope="module")
def full_gain():
    return _cohort()


def test_full_gain_passes_with_exact_denominators_and_no_deployment_authority(full_gain):
    result = score_closed_predictions(*full_gain)
    fresh = result["roles"][FRESH_ROLE]
    preservation = result["roles"][PRESERVATION_ROLE]
    assert result["schema"] == SCHEMA
    assert fresh == {
        "rows": 512,
        "before_top1": 0,
        "top1": 512,
        "before_top5": 512,
        "top5": 512,
        "corrections": 512,
        "regressions": 0,
        "net_gain": 512,
        "candidate_present": 512,
        "candidate_absent": 0,
        "actions": 512,
        "fallbacks": 0,
        "reasons": {"switch_margin": 512},
    }
    assert preservation["rows"] == preservation["before_top1"] \
        == preservation["top1"] == 376
    assert preservation["actions"] == 0
    assert preservation["fallbacks"] == 376
    assert preservation["reasons"] == {"all_keep": 376}
    assert result["bootstrap"]["rows"] == 512
    assert result["bootstrap"]["groups"] == 387
    assert result["gate"] == {
        "passed": True,
        "fresh_passed": True,
        "preservation_all_correct": True,
        "invariants_preserved": True,
        "downstream_evaluation_allowed": True,
        "promotion_allowed": False,
    }


def test_all_keep_fails_fresh_gain_gate():
    result = score_closed_predictions(*_cohort(fresh_mode="keep"))
    assert result["roles"][FRESH_ROLE]["net_gain"] == 0
    assert result["roles"][FRESH_ROLE]["fallbacks"] == 512
    assert result["gate"]["fresh_passed"] is False
    assert result["gate"]["passed"] is False


def test_fresh_top1_regression_is_allowed_when_net_and_interval_pass():
    result = score_closed_predictions(*_cohort(fresh_mode="gain_with_regression"))
    fresh = result["roles"][FRESH_ROLE]
    assert fresh["corrections"] == 40
    assert fresh["regressions"] == 1
    assert fresh["net_gain"] == 39
    assert result["bootstrap"]["lower_2_5_percentile"] > 0
    assert result["gate"]["passed"] is True


def test_a_single_preservation_loss_rejects_the_gate(full_gain):
    rows, predictions, targets = deepcopy(full_gain)
    predictions[512] = _prediction(rows[512], switch=True)
    result = score_closed_predictions(rows, predictions, targets)
    assert result["roles"][PRESERVATION_ROLE]["regressions"] == 1
    assert result["gate"]["preservation_all_correct"] is False
    assert result["gate"]["passed"] is False


def test_candidate_absent_answer_cannot_be_manufactured(full_gain):
    rows, predictions, targets = deepcopy(full_gain)
    targets[0]["expected"] = ["후보에 없는 정답"]
    result = score_closed_predictions(rows, predictions, targets)
    fresh = result["roles"][FRESH_ROLE]
    assert fresh["candidate_absent"] == 1
    assert fresh["top1"] == 511
    assert fresh["top5"] == 511


def test_json_round_trip_predictions_replay_exactly(full_gain):
    rows, predictions, targets = full_gain
    round_tripped = json.loads(json.dumps(predictions, ensure_ascii=False))
    assert score_closed_predictions(rows, round_tripped, targets)["gate"]["passed"] is True


class TargetTrap(dict):
    def get(self, *_args, **_kwargs):
        raise AssertionError("target opened before all predictions were validated")

    def __iter__(self):
        raise AssertionError("target opened before all predictions were validated")


def test_every_prediction_is_validated_before_any_target_field_is_opened(full_gain):
    rows, predictions, targets = deepcopy(full_gain)
    predictions[-1]["complete"] = False
    trapped = [TargetTrap(target) for target in targets]
    with pytest.raises(ValueError, match="replay exactly"):
        score_closed_predictions(rows, predictions, trapped)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda prediction: prediction["pair_scores"][0].update(margin=99.0),
        lambda prediction: prediction["pair_scores"].append(
            deepcopy(prediction["pair_scores"][0])),
        lambda prediction: prediction["pair_scores"][0].update(keep_logit=float("nan")),
        lambda prediction: prediction["candidates"][0].update(text="변조"),
        lambda prediction: prediction["raw_bypass_response"].update(result="변조"),
        lambda prediction: prediction.update(raw_mode=True),
        lambda prediction: prediction.update(result="변조"),
    ],
)
def test_score_candidate_raw_mode_and_result_tampering_are_rejected(full_gain, mutation):
    rows, predictions, targets = deepcopy(full_gain)
    mutation(predictions[0])
    with pytest.raises(ValueError):
        score_closed_predictions(rows, predictions, targets)


def test_incomplete_missing_and_extra_pair_scores_are_rejected(full_gain):
    rows, predictions, targets = deepcopy(full_gain)
    predictions[0]["pair_scores"] = []
    with pytest.raises(ValueError, match="incomplete"):
        score_closed_predictions(rows, predictions, targets)

    rows, predictions, targets = deepcopy(full_gain)
    predictions[0]["pair_scores"][0]["pair_id"] = "extra-pair"
    with pytest.raises(ValueError, match="incomplete"):
        score_closed_predictions(rows, predictions, targets)


def test_source_cohort_and_target_order_are_exact(full_gain):
    rows, predictions, targets = deepcopy(full_gain)
    rows[386]["group_id"] = rows[0]["group_id"]
    predictions[386] = _prediction(rows[386], switch=True)
    with pytest.raises(ValueError, match="document groups"):
        score_closed_predictions(rows, predictions, targets)

    rows, predictions, targets = deepcopy(full_gain)
    targets[0], targets[1] = targets[1], targets[0]
    with pytest.raises(ValueError, match="target identity"):
        score_closed_predictions(rows, predictions, targets)


@pytest.mark.parametrize(
    "expected",
    [[], [""], ["입니다", "입니다"], [7]],
)
def test_target_answers_must_be_nonempty_unique_strings(full_gain, expected):
    rows, predictions, targets = deepcopy(full_gain)
    targets[0]["expected"] = expected
    with pytest.raises(ValueError, match="target identity"):
        score_closed_predictions(rows, predictions, targets)
