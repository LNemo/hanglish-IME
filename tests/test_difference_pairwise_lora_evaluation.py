from copy import deepcopy

import pytest

from scripts.difference_pairwise_lora_evaluation import (
    canonical,
    digest,
    prepare_selection,
    resolve_selection,
    score_predictions,
)
from scripts.prepare_difference_pairwise_lora_data import external_context_window


def candidate(text, score=1.0, *, source="imnida", kind="neural", marker=None):
    value = {
        "text": text,
        "score": score,
        "segments": [{
            "input_start": 0,
            "input_end": len(source),
            "source": source,
            "text": text,
            "kind": kind,
        }],
    }
    if marker is not None:
        value["marker"] = marker
    return value


def row(row_id="r1", expected=None, candidates=None):
    candidates = candidates or [
        candidate("입이다", 3.0, marker={"rank": 0}),
        candidate("입니다", 2.0, marker={"rank": 1}),
        candidate("임니다", 1.0, marker={"rank": 2}),
    ]
    return {
        "id": row_id,
        "group_id": "g",
        "input": "imnida",
        "context": "문맥",
        "expected": expected or ["입니다"],
        "result": candidates[0]["text"],
        "candidates": candidates,
        "raw_bypass_response": {"result": "imnida", "candidates": [{"text": "imnida"}]},
    }


def pair(role, value, index):
    current, alternative = value["candidates"][0], value["candidates"][index]
    identity = {
        "role": role,
        "row_id": value["id"],
        "current_source_index": 0,
        "alternative_source_index": index,
    }
    return {
        "pair_id": digest(canonical(identity)),
        "role": role,
        "row_id": value["id"],
        "source": value["input"],
        "external_context": external_context_window(value["context"], value["input"]),
        "current_source_index": 0,
        "alternative_source_index": index,
        "candidate_count": len(value["candidates"]),
        "raw_mode": False,
        "candidate_objects_sha256": digest(canonical(value["candidates"])),
        "raw_response_sha256": digest(canonical(value["raw_bypass_response"])),
        "current_candidate_sha256": digest(canonical(current)),
        "alternative_candidate_sha256": digest(canonical(alternative)),
        "current": {"text": current["text"], "score": current["score"],
                    "kinds": [item["kind"] for item in current["segments"]]},
        "alternative": {"text": alternative["text"], "score": alternative["score"],
                        "kinds": [item["kind"] for item in alternative["segments"]]},
    }


def prepared(value=None, role="natural_fit"):
    value = value or row()
    pairs = [pair(role, value, index) for index in range(1, len(value["candidates"]))]
    return value, pairs, prepare_selection(role, value, pairs)


def logits(pairs, margins):
    return {item["pair_id"]: [0.0, margin] for item, margin in zip(pairs, margins)}


def test_source_only_prediction_is_unchanged_when_labels_mutate():
    value, pairs, selection = prepared()
    before = resolve_selection(selection, logits(pairs, [2.0, 1.0]))
    value["expected"] = ["완전히 다른 정답"]
    after = resolve_selection(selection, logits(pairs, [2.0, 1.0]))
    assert before["chosen_index"] == after["chosen_index"] == 1
    assert before["pair_scores"] == after["pair_scores"]


def test_source_only_boundary_never_reads_expected_field():
    class LabelTrap(dict):
        def __getitem__(self, key):
            if key == "expected":
                raise AssertionError("selection opened labels")
            return super().__getitem__(key)

        def get(self, key, default=None):
            if key == "expected":
                raise AssertionError("selection opened labels")
            return super().get(key, default)

    plain = row()
    pairs = [pair("natural_fit", plain, 1), pair("natural_fit", plain, 2)]
    trapped = LabelTrap(plain)
    result = resolve_selection(
        prepare_selection("natural_fit", trapped, pairs), logits(pairs, [2.0, 1.0])
    )
    assert result["chosen_index"] == 1


def test_scores_every_pair_and_largest_positive_margin_wins():
    value, pairs, selection = prepared()
    result = resolve_selection(selection, logits(pairs, [0.2, 1.5]))
    assert result["complete"] is True
    assert result["expected_pair_count"] == result["scored_pair_count"] == 2
    assert result["chosen_index"] == 2
    assert result["reason"] == "switch_margin"


def test_all_keep_and_exact_margin_tie_rules():
    value, pairs, selection = prepared()
    assert resolve_selection(selection, logits(pairs, [0.0, -1.0]))["chosen_index"] == 0
    tied = resolve_selection(selection, logits(pairs, [1.0, 1.0]))
    assert tied["chosen_index"] == 1


def test_reorder_uses_exact_objects_and_preserves_unaffected_order_and_fields():
    value, pairs, selection = prepared()
    original = list(value["candidates"])
    snapshots = [canonical(item) for item in original]
    result = resolve_selection(selection, logits(pairs, [0.1, 2.0]))
    assert [item is original[index] for item, index in zip(result["candidates"], [2, 0, 1])] == [True] * 3
    assert [canonical(item) for item in original] == snapshots
    assert result["raw_bypass_response"] is value["raw_bypass_response"]


@pytest.mark.parametrize("bad", [None, [1.0], [1.0, 2.0, 3.0], [0.0, float("nan")],
                                  [-1e308, 1e308], [False, 1.0]])
def test_missing_extra_or_malformed_scores_fall_back_and_are_incomplete(bad):
    value, pairs, selection = prepared()
    values = logits(pairs, [1.0, 1.0])
    if bad is None:
        values.pop(pairs[0]["pair_id"])
    else:
        values[pairs[0]["pair_id"]] = bad
    result = resolve_selection(selection, values)
    assert result["chosen_index"] == 0
    assert result["complete"] is False
    assert result["reason"] == "incomplete_pair_scores"
    with pytest.raises(ValueError, match="incomplete diagnostic"):
        score_predictions([{"role": "natural_fit", "row": value, "stratum": "s"}], [result], expected_rows=1)


def test_extra_score_falls_back_and_is_reported():
    _, pairs, selection = prepared()
    values = logits(pairs, [1.0, 1.0])
    values["unknown"] = [0.0, 9.0]
    result = resolve_selection(selection, values)
    assert result["complete"] is False
    assert result["failures"] == [{"reason": "extra_pair_logits", "pair_ids": ["unknown"]}]


def test_non_string_score_key_is_safe_and_explicit():
    _, pairs, selection = prepared()
    values = logits(pairs, [1.0, 1.0])
    values[1] = [0.0, 9.0]
    result = resolve_selection(selection, values)
    assert result["complete"] is False
    assert result["chosen_index"] == 0
    assert result["failures"][0] == {"reason": "malformed_pair_logit_keys"}


@pytest.mark.parametrize("index", [True, 0, -1, 3, "1"])
def test_malformed_qualified_pair_indices_are_rejected(index):
    value = row()
    pairs = [pair("natural_fit", value, 1), pair("natural_fit", value, 2)]
    pairs[0]["alternative_source_index"] = index
    with pytest.raises(ValueError, match="index"):
        prepare_selection("natural_fit", value, pairs)


def test_missing_or_duplicate_qualified_pair_is_rejected():
    value = row()
    first = pair("natural_fit", value, 1)
    with pytest.raises(ValueError, match="cover every permitted"):
        prepare_selection("natural_fit", value, [first])
    with pytest.raises(ValueError, match="duplicated"):
        prepare_selection("natural_fit", value, [first, deepcopy(first)])


def test_pair_context_window_must_bind_the_source_row():
    value = row()
    pairs = [pair("natural_fit", value, 1), pair("natural_fit", value, 2)]
    pairs[0]["external_context"] = {"strategy": "complete", "text": "다른 문맥", "original_chars": 5}
    with pytest.raises(ValueError, match="bind the source row"):
        prepare_selection("natural_fit", value, pairs)


def test_protected_candidate_and_raw_mode_never_switch():
    protected = row(candidates=[candidate("test", 2, kind="raw"), candidate("테스트", 1)])
    selection = prepare_selection("challenge_fit", protected, [])
    result = resolve_selection(selection, {})
    assert result["chosen_index"] == 0
    assert result["reason"] == "no_permitted_non_parent_alternative"
    raw_selection = prepare_selection("challenge_fit", row(), [], raw_mode=True)
    raw_result = resolve_selection(raw_selection, {})
    assert raw_result["complete"] is True
    assert raw_result["reason"] == "raw_mode"
    assert raw_result["chosen_index"] == 0


def test_winning_switch_is_guarded_again_and_next_allowed_margin_can_win(monkeypatch):
    value, pairs, selection = prepared()
    from scripts import difference_pairwise_lora_evaluation as module

    original = module.proposal_allowed
    monkeypatch.setattr(module, "proposal_allowed", lambda source, parent, proposal:
                        False if proposal is value["candidates"][1] else original(source, parent, proposal))
    result = resolve_selection(selection, logits(pairs, [5.0, 2.0]))
    assert result["chosen_index"] == 2
    assert result["rejected_switch_indices"] == [1]


def test_metrics_replay_exact_decisions_and_count_gate_strata_and_absence():
    corrected = row("corrected", ["입니다"])
    c_pairs = [pair("natural_fit", corrected, 1), pair("natural_fit", corrected, 2)]
    c_prediction = resolve_selection(
        prepare_selection("natural_fit", corrected, c_pairs), logits(c_pairs, [2.0, -1.0])
    )
    absent = row("absent", ["없음"])
    a_pairs = [pair("natural_fit", absent, 1), pair("natural_fit", absent, 2)]
    a_prediction = resolve_selection(
        prepare_selection("natural_fit", absent, a_pairs), logits(a_pairs, [-1.0, -2.0])
    )
    report = score_predictions(
        [
            {"role": "natural_fit", "row": corrected, "stratum": "repair"},
            {"role": "natural_fit", "row": absent, "stratum": "absent"},
        ],
        [c_prediction, a_prediction],
        expected_rows=2,
        minimum_net_gain=1,
    )
    assert report["gate"]["passed"] is True
    assert report["gate"]["net_top1_gain"] == 1
    assert report["gate"]["corrections"] == 1
    assert report["gate"]["regressions"] == 0
    assert report["gate"]["candidate_absent"] == 1
    assert report["gate"]["recoverable_errors"] == 1
    assert report["strata"]["repair"]["corrections"] == 1


def test_metrics_reject_tampered_decision_or_candidate_object():
    value, pairs, selection = prepared()
    prediction = resolve_selection(selection, logits(pairs, [2.0, -1.0]))
    item = {"role": "natural_fit", "row": value, "stratum": "s"}
    prediction["chosen_index"] = 2
    with pytest.raises(ValueError, match="replay"):
        score_predictions([item], [prediction], expected_rows=1)
    prediction = resolve_selection(selection, logits(pairs, [2.0, -1.0]))
    prediction["candidates"] = deepcopy(prediction["candidates"])
    with pytest.raises(ValueError, match="preservation"):
        score_predictions([item], [prediction], expected_rows=1)


def test_grammatical_but_wrong_allowed_switch_is_counted_as_regression():
    value = row(expected=["입이다"])
    pairs = [pair("natural_fit", value, 1), pair("natural_fit", value, 2)]
    prediction = resolve_selection(
        prepare_selection("natural_fit", value, pairs), logits(pairs, [2.0, -1.0])
    )
    report = score_predictions(
        [{"role": "natural_fit", "row": value, "stratum": "corruption"}],
        [prediction],
        expected_rows=1,
        minimum_net_gain=0,
    )
    assert report["gate"]["passed"] is False
    assert report["gate"]["regressions"] == 1
    assert report["gate"]["net_top1_gain"] == -1


def test_train_gate_requires_exact_row_count_and_net_five_not_perfect_fit():
    values, predictions, items = [], [], []
    for index in range(6):
        value = row(f"r{index}")
        pairs = [pair("natural_fit", value, 1), pair("natural_fit", value, 2)]
        prediction = resolve_selection(
            prepare_selection("natural_fit", value, pairs),
            logits(pairs, [2.0 if index < 5 else -1.0, -2.0]),
        )
        values.append(value)
        predictions.append(prediction)
        items.append({"role": "natural_fit", "row": value, "stratum": "s"})
    report = score_predictions(items, predictions, expected_rows=6)
    assert report["gate"]["passed"] is True
    assert report["metrics"]["all"]["top1"] == 5
    assert report["metrics"]["all"]["rows"] == 6
    assert score_predictions(items, predictions, expected_rows=7)["gate"]["passed"] is False
