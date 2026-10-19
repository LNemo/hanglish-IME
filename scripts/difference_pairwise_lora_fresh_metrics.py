"""Post-inference metrics for the fixed pairwise LoRA fresh cohort.

This helper validates every source-bound prediction before it consults any
accepted answer.  It does not prove that captures were closed or that model
references were released before target access; the evaluation runner and its
independent verifier must establish that ordering.
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Mapping, Sequence

from scripts.difference_pairwise_lora_evaluation import (
    SCHEMA as PREDICTION_SCHEMA,
    canonical,
    resolve_selection,
)
from scripts.difference_pairwise_lora_inference import prepare_inference_selection
from scripts.long_context_lora_fresh_metrics import paired_document_bootstrap


SCHEMA = "difference-pairwise-lora-fresh-metrics-v1"
VERSION = "neural-1-867f13596ed5"
FRESH_ROLE = "fresh_natural"
PRESERVATION_ROLE = "challenge_preservation"
FRESH_ROWS = 512
FRESH_GROUPS = 387
PRESERVATION_ROWS = 376
TOTAL_ROWS = FRESH_ROWS + PRESERVATION_ROWS


def _validate_parent_row(row: Mapping, role: str) -> None:
    try:
        row_id = row["id"]
        group_id = row["group_id"]
        source = row["input"]
        context = row["context"]
        parent = row["parent_response"]
        candidates = row["candidates"]
        raw = row["raw_bypass_response"]
    except (KeyError, TypeError) as error:
        raise ValueError("fresh source row fields are missing") from error
    if (
        not isinstance(row_id, str)
        or not row_id
        or not isinstance(group_id, str)
        or not group_id
        or not isinstance(source, str)
        or not source
        or not isinstance(context, str)
        or not isinstance(parent, dict)
        or not isinstance(candidates, list)
        or not 1 <= len(candidates) <= 5
        or any(
            not isinstance(candidate, dict)
            or not isinstance(candidate.get("text"), str)
            or not isinstance(candidate.get("segments"), list)
            for candidate in candidates
        )
        or not isinstance(raw, dict)
    ):
        raise ValueError("fresh source row fields are malformed")
    try:
        candidates_match = canonical(parent.get("candidates")) == canonical(candidates)
    except (TypeError, ValueError) as error:
        raise ValueError("fresh parent candidates are malformed") from error
    if (
        parent.get("input") != source
        or parent.get("context") != context
        or parent.get("version") != VERSION
        or parent.get("raw") is not False
        or parent.get("result") != candidates[0].get("text")
        or not candidates_match
        or raw.get("input") != source
        or raw.get("context") != context
        or raw.get("version") != VERSION
        or raw.get("raw") is not True
        or raw.get("result") != source
    ):
        raise ValueError("fresh parent response does not bind its source row")
    expected_role = FRESH_ROLE if role == FRESH_ROLE else PRESERVATION_ROLE
    if role != expected_role:
        raise ValueError("fresh source role differs")


def _validate_and_replay_prediction(role: str, row: Mapping, prediction: Mapping) -> None:
    if not isinstance(prediction, Mapping):
        raise ValueError("fresh prediction is malformed")
    prepared = prepare_inference_selection(role, row)
    scores = prediction.get("pair_scores")
    if not isinstance(scores, list):
        raise ValueError("fresh prediction pair scores are malformed")
    logits_by_pair_id = {}
    for score in scores:
        if not isinstance(score, dict):
            raise ValueError("fresh prediction pair score is malformed")
        pair_id = score.get("pair_id")
        keep, switch = score.get("keep_logit"), score.get("switch_logit")
        if (
            not isinstance(pair_id, str)
            or not pair_id
            or pair_id in logits_by_pair_id
            or type(keep) not in (int, float)
            or type(switch) not in (int, float)
            or not math.isfinite(keep)
            or not math.isfinite(switch)
            or not math.isfinite(float(switch) - float(keep))
        ):
            raise ValueError("fresh prediction pair score differs, duplicates, or is nonfinite")
        logits_by_pair_id[pair_id] = [keep, switch]
    replayed = resolve_selection(prepared, logits_by_pair_id)
    if (
        replayed.get("schema") != PREDICTION_SCHEMA
        or replayed.get("complete") is not True
        or replayed.get("failures") != []
    ):
        raise ValueError("incomplete fresh pair scores cannot support evaluation")
    try:
        matches = canonical(dict(prediction)) == canonical(replayed)
    except (TypeError, ValueError) as error:
        raise ValueError("fresh prediction evidence is malformed") from error
    if not matches:
        raise ValueError("fresh prediction does not replay exactly from source evidence")


def _blank_role_metrics() -> dict:
    return {
        "rows": 0,
        "before_top1": 0,
        "top1": 0,
        "before_top5": 0,
        "top5": 0,
        "corrections": 0,
        "regressions": 0,
        "net_gain": 0,
        "candidate_present": 0,
        "candidate_absent": 0,
        "actions": 0,
        "fallbacks": 0,
        "reasons": {},
    }


def score_closed_predictions(
    rows: Sequence[Mapping], predictions: Sequence[Mapping], targets: Sequence[Mapping]
) -> dict:
    """Validate a complete closed prediction set, then open and score targets."""
    if (
        not isinstance(rows, (list, tuple))
        or not isinstance(predictions, (list, tuple))
        or len(rows) != TOTAL_ROWS
        or len(predictions) != TOTAL_ROWS
    ):
        raise ValueError("complete 888-row fresh predictions are required")

    # This entire pass is target-blind.  A target mapping can therefore trap
    # every access and prove malformed predictions fail before label access.
    seen_ids = set()
    fresh_groups = set()
    expected_roles = [FRESH_ROLE] * FRESH_ROWS + [PRESERVATION_ROLE] * PRESERVATION_ROWS
    for row, prediction, role in zip(rows, predictions, expected_roles):
        if not isinstance(row, Mapping) or row.get("role") != role:
            raise ValueError("fresh source cohort role or order differs")
        _validate_parent_row(row, role)
        row_id = row["id"]
        if row_id in seen_ids:
            raise ValueError("fresh source identities are duplicated")
        seen_ids.add(row_id)
        if role == FRESH_ROLE:
            fresh_groups.add(row["group_id"])
        _validate_and_replay_prediction(role, row, prediction)
    if len(seen_ids) != TOTAL_ROWS or len(fresh_groups) != FRESH_GROUPS:
        raise ValueError("fresh source cohort denominator or document groups differ")

    if not isinstance(targets, (list, tuple)) or len(targets) != TOTAL_ROWS:
        raise ValueError("complete 888-row fresh targets are required")

    roles = {FRESH_ROLE: _blank_role_metrics(), PRESERVATION_ROLE: _blank_role_metrics()}
    reason_counts = {FRESH_ROLE: Counter(), PRESERVATION_ROLE: Counter()}
    outcomes = []
    for row, prediction, target, role in zip(rows, predictions, targets, expected_roles):
        if not isinstance(target, Mapping):
            raise ValueError("fresh target is malformed")
        expected = target.get("expected")
        if (
            set(target) != {"id", "group_id", "role", "expected"}
            or target.get("id") != row["id"]
            or target.get("group_id") != row["group_id"]
            or target.get("role") != role
            or not isinstance(expected, list)
            or not expected
            or any(not isinstance(value, str) or not value for value in expected)
            or len(set(expected)) != len(expected)
        ):
            raise ValueError("fresh target identity, order, role, or answers differ")
        accepted = set(expected)
        before_candidates = row["candidates"]
        after_candidates = prediction["candidates"]
        before = before_candidates[0]["text"] in accepted
        after = after_candidates[0]["text"] in accepted
        before_top5 = any(candidate["text"] in accepted for candidate in before_candidates[:5])
        after_top5 = any(candidate["text"] in accepted for candidate in after_candidates[:5])
        present = any(candidate["text"] in accepted for candidate in before_candidates)
        values = roles[role]
        values["rows"] += 1
        values["before_top1"] += int(before)
        values["top1"] += int(after)
        values["before_top5"] += int(before_top5)
        values["top5"] += int(after_top5)
        values["corrections"] += int(after and not before)
        values["regressions"] += int(before and not after)
        values["candidate_present"] += int(present)
        values["candidate_absent"] += int(not present)
        action = prediction["reason"] == "switch_margin"
        values["actions"] += int(action)
        values["fallbacks"] += int(not action)
        reason_counts[role][prediction["reason"]] += 1
        if role == FRESH_ROLE:
            outcomes.append(
                {
                    "id": row["id"],
                    "group_id": row["group_id"],
                    "before_correct": before,
                    "after_correct": after,
                }
            )

    for role, values in roles.items():
        values["net_gain"] = values["top1"] - values["before_top1"]
        values["reasons"] = dict(sorted(reason_counts[role].items()))
    bootstrap = paired_document_bootstrap(outcomes)
    fresh = roles[FRESH_ROLE]
    preservation = roles[PRESERVATION_ROLE]
    invariants = all(values["before_top5"] == values["top5"] for values in roles.values())
    fresh_passed = (
        fresh["rows"] == FRESH_ROWS
        and bootstrap["rows"] == FRESH_ROWS
        and bootstrap["groups"] == FRESH_GROUPS
        and bootstrap["gain_and_interval_passed"]
    )
    preservation_all_correct = (
        preservation["rows"] == PRESERVATION_ROWS
        and preservation["before_top1"] == preservation["top1"] == PRESERVATION_ROWS
    )
    passed = fresh_passed and preservation_all_correct and invariants
    return {
        "schema": SCHEMA,
        "roles": roles,
        "bootstrap": bootstrap,
        "gate": {
            "passed": passed,
            "fresh_passed": fresh_passed,
            "preservation_all_correct": preservation_all_correct,
            "invariants_preserved": invariants,
            "downstream_evaluation_allowed": passed,
            "promotion_allowed": False,
        },
    }
