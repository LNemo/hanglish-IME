"""Leakage-free pairwise resolution and post-inference quality metrics.

Selection consumes only a frozen row, its qualified pair records, and the two
KEEP/SWITCH logits for every permitted alternative.  Accepted answers are read
only by :func:`score_predictions`, after inference is complete.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence

from scripts.final_choice_policy import (
    finite_candidate_indices,
    proposal_allowed,
    reorder_candidates,
)
from scripts.prepare_difference_pairwise_lora_data import external_context_window


SCHEMA = "difference-pairwise-lora-evaluation-v1"


def canonical(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pair_id(role: str, row_id: str, alternative_index: int) -> str:
    identity = {
        "role": role,
        "row_id": row_id,
        "current_source_index": 0,
        "alternative_source_index": alternative_index,
    }
    return digest(canonical(identity))


def _required_fields(row: dict) -> tuple[str, str, str, list, dict]:
    try:
        row_id, source = row["id"], row["input"]
        context = row["context"]
        candidates, raw = row["candidates"], row["raw_bypass_response"]
    except (KeyError, TypeError) as error:
        raise ValueError("row source fields are missing") from error
    if (
        not isinstance(row_id, str)
        or not row_id
        or not isinstance(source, str)
        or not isinstance(context, str)
        or not isinstance(candidates, list)
        or not candidates
        or not isinstance(raw, dict)
    ):
        raise ValueError("row source fields are malformed")
    return row_id, source, context, candidates, raw


def _permitted_indices(source: str, candidates: list) -> list[int]:
    finite = finite_candidate_indices(candidates)
    if not finite or finite[0] != 0:
        return []
    return [
        index
        for index in finite
        if index and proposal_allowed(source, candidates[0], candidates[index])
    ]


def _validate_pair(role: str, row: dict, pair: dict, index: int) -> None:
    row_id, source, context, candidates, raw = _required_fields(row)
    if (
        not isinstance(pair, dict)
        or pair.get("pair_id") != _pair_id(role, row_id, index)
        or pair.get("role") != role
        or pair.get("row_id") != row_id
        or pair.get("source") != source
        or pair.get("external_context") != external_context_window(context, source)
        or pair.get("current_source_index") != 0
        or pair.get("alternative_source_index") != index
        or pair.get("candidate_count") != len(candidates)
        or pair.get("raw_mode") is not False
        or pair.get("candidate_objects_sha256") != digest(canonical(candidates))
        or pair.get("raw_response_sha256") != digest(canonical(raw))
        or pair.get("current_candidate_sha256") != digest(canonical(candidates[0]))
        or pair.get("alternative_candidate_sha256") != digest(canonical(candidates[index]))
    ):
        raise ValueError("qualified pair does not bind the source row")
    current, alternative = pair.get("current"), pair.get("alternative")
    if not isinstance(current, dict) or not isinstance(alternative, dict):
        raise ValueError("qualified pair candidate summaries are malformed")
    for summary, candidate in ((current, candidates[0]), (alternative, candidates[index])):
        expected = {
            "text": candidate.get("text"),
            "score": candidate.get("score"),
            "kinds": [segment.get("kind") for segment in candidate.get("segments", [])],
        }
        if any(summary.get(field) != value for field, value in expected.items()):
            raise ValueError("qualified pair candidate summary differs")


def prepare_selection(
    role: str, row: dict, qualified_pairs: Sequence[dict], *, raw_mode: bool = False
) -> dict:
    """Bind every source-permitted alternative without consulting row labels."""
    row_id, source, context, candidates, raw = _required_fields(row)
    if not isinstance(role, str) or not role or type(raw_mode) is not bool:
        raise ValueError("role and raw mode must be explicit")
    if not isinstance(qualified_pairs, (list, tuple)):
        raise ValueError("qualified pairs must be an ordered sequence")
    permitted = [] if raw_mode else _permitted_indices(source, candidates)
    by_index: dict[int, dict] = {}
    for pair in qualified_pairs:
        index = pair.get("alternative_source_index") if isinstance(pair, dict) else None
        if type(index) is not int or not 0 < index < len(candidates) or index in by_index:
            raise ValueError("qualified pair alternative index is malformed or duplicated")
        by_index[index] = pair
    if sorted(by_index) != permitted:
        raise ValueError("qualified pairs do not cover every permitted alternative exactly")
    for index in permitted:
        _validate_pair(role, row, by_index[index], index)
    return {
        "role": role,
        "row_id": row_id,
        "source": source,
        "context": context,
        "raw_mode": raw_mode,
        "alternative_indices": permitted,
        "pairs": [by_index[index] for index in permitted],
        "candidate_snapshot": canonical(candidates),
        "raw_snapshot": canonical(raw),
        "candidates": candidates,
        "raw_bypass_response": raw,
    }


def _parse_logits(value) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        return None
    keep, switch = float(value[0]), float(value[1])
    if not math.isfinite(switch - keep):
        return None
    return keep, switch


def _choose(prepared: dict, parsed: list[dict]) -> tuple[int, int | None, str, list[int]]:
    if prepared["raw_mode"]:
        return 0, None, "raw_mode", []
    positive = sorted(
        (item for item in parsed if item["margin"] > 0),
        key=lambda item: (-item["margin"], item["alternative_source_index"]),
    )
    if not positive:
        reason = "no_permitted_non_parent_alternative" if not parsed else "all_keep"
        return 0, None, reason, []
    rejected = []
    candidates = prepared["candidates"]
    for item in positive:
        index = item["alternative_source_index"]
        if proposal_allowed(prepared["source"], candidates[0], candidates[index]):
            return index, index, "switch_margin", rejected
        rejected.append(index)
    return 0, positive[0]["alternative_source_index"], "all_switches_rejected", rejected


def resolve_selection(prepared: dict, logits_by_pair_id: Mapping[str, Sequence[float]]) -> dict:
    """Resolve complete pair logits, falling back safely on invalid evidence."""
    if not isinstance(logits_by_pair_id, Mapping):
        raise ValueError("pair logits must be a mapping")
    candidates, raw = prepared["candidates"], prepared["raw_bypass_response"]
    failures: list[dict] = []
    if canonical(candidates) != prepared["candidate_snapshot"]:
        failures.append({"reason": "candidate_snapshot_changed"})
    if canonical(raw) != prepared["raw_snapshot"]:
        failures.append({"reason": "raw_snapshot_changed"})
    expected_ids = [pair["pair_id"] for pair in prepared["pairs"]]
    if any(not isinstance(key, str) for key in logits_by_pair_id):
        failures.append({"reason": "malformed_pair_logit_keys"})
        supplied_ids = set()
    else:
        supplied_ids = set(logits_by_pair_id)
    missing = sorted(set(expected_ids) - supplied_ids)
    extra = sorted(supplied_ids - set(expected_ids))
    if missing:
        failures.append({"reason": "missing_pair_logits", "pair_ids": missing})
    if extra:
        failures.append({"reason": "extra_pair_logits", "pair_ids": extra})
    parsed = []
    for pair in prepared["pairs"]:
        pair_id = pair["pair_id"]
        if pair_id not in logits_by_pair_id:
            continue
        logits = _parse_logits(logits_by_pair_id[pair_id])
        if logits is None:
            failures.append({"reason": "malformed_or_nonfinite_pair_logits", "pair_id": pair_id})
            continue
        keep, switch = logits
        parsed.append(
            {
                "pair_id": pair_id,
                "alternative_source_index": pair["alternative_source_index"],
                "keep_logit": keep,
                "switch_logit": switch,
                "margin": switch - keep,
            }
        )
    complete = not failures
    if complete:
        chosen, proposed, reason, rejected = _choose(prepared, parsed)
    else:
        chosen, proposed, reason, rejected = 0, None, "incomplete_pair_scores", []
    ordered = reorder_candidates(candidates, chosen)
    return {
        "schema": SCHEMA,
        "role": prepared["role"],
        "id": prepared["row_id"],
        "source": prepared["source"],
        "context": prepared["context"],
        "raw_mode": prepared["raw_mode"],
        "complete": complete,
        "reason": reason,
        "failures": failures,
        "chosen_index": chosen,
        "proposed_index": proposed,
        "rejected_switch_indices": rejected,
        "expected_pair_count": len(expected_ids),
        "scored_pair_count": len(parsed),
        "pair_scores": parsed,
        "result": ordered[0].get("text"),
        "candidates": ordered,
        "raw_bypass_response": raw,
        "candidate_sha256_before": digest(prepared["candidate_snapshot"]),
        "raw_sha256_before": digest(prepared["raw_snapshot"]),
    }


def _replay_prediction(role: str, row: dict, prediction: dict) -> None:
    row_id, source, _context, candidates, raw = _required_fields(row)
    permitted = [] if prediction.get("raw_mode") is True else _permitted_indices(source, candidates)
    expected_ids = [_pair_id(role, row_id, index) for index in permitted]
    scores = prediction.get("pair_scores")
    if not isinstance(scores, list) or len(scores) != len(expected_ids):
        raise ValueError("diagnostic pair scores are incomplete")
    parsed = []
    for item, index, pair_id in zip(scores, permitted, expected_ids):
        if (
            not isinstance(item, dict)
            or item.get("pair_id") != pair_id
            or item.get("alternative_source_index") != index
        ):
            raise ValueError("diagnostic pair score binding differs")
        logits = _parse_logits([item.get("keep_logit"), item.get("switch_logit")])
        if logits is None or item.get("margin") != logits[1] - logits[0]:
            raise ValueError("diagnostic pair score is malformed or nonfinite")
        parsed.append(item)
    replay = {
        "raw_mode": prediction.get("raw_mode"),
        "source": source,
        "candidates": candidates,
    }
    chosen, proposed, reason, rejected = _choose(replay, parsed)
    if (
        prediction.get("chosen_index") != chosen
        or prediction.get("proposed_index") != proposed
        or prediction.get("reason") != reason
        or prediction.get("rejected_switch_indices") != rejected
        or prediction.get("expected_pair_count") != len(expected_ids)
        or prediction.get("scored_pair_count") != len(expected_ids)
    ):
        raise ValueError("diagnostic selection does not replay exactly")
    expected_order = reorder_candidates(candidates, chosen)
    actual_order = prediction.get("candidates")
    if (
        not isinstance(actual_order, list)
        or len(actual_order) != len(expected_order)
        or any(actual is not expected for actual, expected in zip(actual_order, expected_order))
        or prediction.get("result") != expected_order[0].get("text")
        or prediction.get("raw_bypass_response") is not raw
        or digest(canonical(candidates)) != prediction.get("candidate_sha256_before")
        or digest(canonical(raw)) != prediction.get("raw_sha256_before")
    ):
        raise ValueError("candidate or raw object preservation failed")


def _blank_counts() -> dict:
    return dict.fromkeys(
        (
            "rows",
            "before_top1",
            "top1",
            "before_top5",
            "top5",
            "corrections",
            "regressions",
            "actions",
            "fallbacks",
            "candidate_present",
            "candidate_absent",
            "recoverable_errors",
        ),
        0,
    )


def score_predictions(
    items: Sequence[dict],
    predictions: Sequence[dict],
    *,
    expected_rows: int = 640,
    minimum_net_gain: int = 5,
) -> dict:
    """Open labels only after inference and calculate the TRAIN quality gate."""
    if (
        not isinstance(items, (list, tuple))
        or not items
        or len(items) != len(predictions)
        or type(expected_rows) is not int
        or expected_rows <= 0
        or type(minimum_net_gain) is not int
    ):
        raise ValueError("diagnostic rows must be complete and gate values explicit")
    metrics: dict[str, dict] = {}
    strata: dict[str, dict] = {}
    outcomes = []
    identities = set()
    for item, prediction in zip(items, predictions):
        if not isinstance(item, dict) or not isinstance(prediction, dict):
            raise ValueError("diagnostic row or prediction is malformed")
        role, row = item.get("role"), item.get("row")
        if not isinstance(role, str) or not isinstance(row, dict):
            raise ValueError("diagnostic identity is missing")
        identity = (role, row.get("id"))
        if identity in identities or prediction.get("role") != role or prediction.get("id") != row.get("id"):
            raise ValueError("diagnostic prediction identity differs or duplicates")
        identities.add(identity)
        if prediction.get("schema") != SCHEMA or prediction.get("complete") is not True or prediction.get("failures") != []:
            raise ValueError("incomplete diagnostic scores cannot support a quality claim")
        _replay_prediction(role, row, prediction)
        expected = row.get("expected")
        if (
            not isinstance(expected, list)
            or not expected
            or any(not isinstance(value, str) or not value for value in expected)
        ):
            raise ValueError("diagnostic accepted answers are malformed")
        candidates = row["candidates"]
        if row.get("result") != candidates[0].get("text"):
            raise ValueError("diagnostic baseline is not candidate zero")
        expected_set = set(expected)
        before = row["result"] in expected_set
        after = prediction["result"] in expected_set
        before5 = any(candidate.get("text") in expected_set for candidate in candidates[:5])
        after5 = any(candidate.get("text") in expected_set for candidate in prediction["candidates"][:5])
        present = any(candidate.get("text") in expected_set for candidate in candidates)
        permitted = _permitted_indices(row["input"], candidates)
        recoverable = (not before) and any(candidates[index].get("text") in expected_set for index in permitted)
        stratum = item.get("stratum")
        if not isinstance(stratum, str) or not stratum:
            raise ValueError("diagnostic stratum is missing")
        increments = {
            "rows": 1,
            "before_top1": int(before),
            "top1": int(after),
            "before_top5": int(before5),
            "top5": int(after5),
            "corrections": int(after and not before),
            "regressions": int(before and not after),
            "actions": int(prediction["chosen_index"] != 0),
            "fallbacks": int(prediction["chosen_index"] == 0),
            "candidate_present": int(present),
            "candidate_absent": int(not present),
            "recoverable_errors": int(recoverable),
        }
        for collection, key in ((metrics, "all"), (metrics, role), (strata, stratum)):
            counts = collection.setdefault(key, _blank_counts())
            for name, value in increments.items():
                counts[name] += value
        outcomes.append(
            {
                "role": role,
                "id": row["id"],
                "stratum": stratum,
                "before_top1": before,
                "top1": after,
                "before_top5": before5,
                "top5": after5,
                "corrected": after and not before,
                "regressed": before and not after,
                "candidate_present": present,
                "recoverable_error": recoverable,
                "chosen_index": prediction["chosen_index"],
                "reason": prediction["reason"],
            }
        )
    total = metrics["all"]
    net = total["top1"] - total["before_top1"]
    top5_losses = [item["id"] for item in outcomes if item["before_top5"] and not item["top5"]]
    gate = {
        "passed": len(items) == expected_rows and net >= minimum_net_gain and not top5_losses,
        "expected_rows": expected_rows,
        "minimum_net_top1_gain": minimum_net_gain,
        "all_rows_accounted": len(items) == expected_rows,
        "net_top1_gain": net,
        "corrections": total["corrections"],
        "regressions": total["regressions"],
        "candidate_absent": total["candidate_absent"],
        "recoverable_errors": total["recoverable_errors"],
        "zero_top5_losses": not top5_losses,
        "top5_loss_ids": top5_losses,
        "candidate_objects_preserved": True,
        "raw_responses_preserved": True,
        "protected_policy_replayed": True,
    }
    return {"schema": SCHEMA, "metrics": metrics, "strata": strata, "outcomes": outcomes, "gate": gate}
