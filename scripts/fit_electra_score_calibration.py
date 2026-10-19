"""Cross-fitted nonnegative calibration of saved baseline and ELECTRA scores."""
from __future__ import annotations

import math
from collections import defaultdict

import torch


RECIPE = {"folds": 4, "learning_rate": 0.05, "steps": 200, "l2": 0.01,
          "initial_weights": [1.0, 0.0, 0.0], "dtype": "float64"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate(rows):
    _require(isinstance(rows, list) and rows, "nonempty calibration rows required")
    ids, partitions = set(), {"component": {}, "group_id": {}}
    for row in rows:
        candidates, baseline, plausibility = (row.get("candidate_texts"),
                                               row.get("baseline_scores"),
                                               row.get("plausibility"))
        _require(isinstance(row.get("id"), str) and row["id"] not in ids,
                 "row ids must be unique strings")
        ids.add(row["id"])
        _require(type(row.get("fold")) is int and 0 <= row["fold"] < RECIPE["folds"],
                 "fold must be in range")
        _require(isinstance(row.get("component"), str) and row["component"],
                 "component is required")
        _require(row.get("source_role") in {"natural_fit", "challenge_fit"},
                 "source role differs")
        _require(isinstance(row.get("current_result"), str), "current result is required")
        _require(isinstance(candidates, list) and candidates
                 and all(isinstance(value, str) for value in candidates)
                 and len(set(candidates)) == len(candidates),
                 "candidate texts must be unique strings")
        _require(isinstance(baseline, list) and len(baseline) == len(candidates)
                 and all(type(value) in (int, float) and math.isfinite(value)
                         for value in baseline), "baseline scores differ")
        _require(isinstance(row.get("expected"), list) and row["expected"]
                 and all(isinstance(value, str) and value for value in row["expected"]),
                 "expected values differ")
        if plausibility is not None:
            _require(isinstance(plausibility, list) and len(plausibility) == len(candidates),
                     "plausibility rows differ")
            for value in plausibility:
                _require(isinstance(value, dict)
                         and {"mean_original_logprob", "changed_original_logprob"}
                         <= set(value)
                         and all(type(value[key]) in (int, float)
                                 and math.isfinite(value[key])
                                 for key in ("mean_original_logprob",
                                             "changed_original_logprob")),
                         "plausibility value differs")
        for key in partitions:
            if key == "group_id" and key not in row:
                continue
            value = row.get(key)
            _require(isinstance(value, str) and value, f"{key} must be a nonempty string")
            previous = partitions[key].setdefault(value, row["fold"])
            _require(previous == row["fold"], f"{key} crosses folds")
    _require({row["fold"] for row in rows} == set(range(RECIPE["folds"])),
             "all four folds are required")


def _features(row):
    if row["plausibility"] is None:
        return None
    values = torch.tensor([
        [base, item["mean_original_logprob"], item["changed_original_logprob"]]
        for base, item in zip(row["baseline_scores"], row["plausibility"])
    ], dtype=torch.float64)
    return values - values[0]


def _positives(row):
    accepted = set(row["expected"])
    return [index for index, text in enumerate(row["candidate_texts"])
            if text in accepted]


def _eligible(row):
    return len(row["candidate_texts"]) >= 2 and bool(_positives(row)) \
        and row["plausibility"] is not None


def _pack(rows):
    _require(rows, "eligible packed rows required")
    maximum = max(len(row["candidate_texts"]) for row in rows)
    features = torch.zeros((len(rows), maximum, 3), dtype=torch.float64)
    valid = torch.zeros((len(rows), maximum), dtype=torch.bool)
    positive = torch.zeros((len(rows), maximum), dtype=torch.bool)
    for index, row in enumerate(rows):
        size = len(row["candidate_texts"])
        features[index, :size] = _features(row)
        valid[index, :size] = True
        positive[index, _positives(row)] = True
    return features, valid, positive


def _packed_objective(weights, packed):
    features, valid, positive = packed
    scores = features.matmul(weights)
    negative_infinity = torch.tensor(float("-inf"), dtype=scores.dtype)
    denominator = torch.logsumexp(scores.masked_fill(~valid, negative_infinity), dim=1)
    numerator = torch.logsumexp(scores.masked_fill(~positive, negative_infinity), dim=1)
    data_loss = (denominator - numerator).mean()
    anchor = weights.new_tensor(RECIPE["initial_weights"])
    return data_loss + RECIPE["l2"] * torch.square(weights - anchor).sum()


def _objective(weights, rows):
    """Small reference-compatible wrapper; training packs rows only once."""
    return _packed_objective(weights, _pack(rows))


def _fit_weights(rows):
    eligible = [row for row in rows if _eligible(row)]
    weights = torch.tensor(RECIPE["initial_weights"], dtype=torch.float64,
                           requires_grad=True)
    if eligible:
        packed = _pack(eligible)
        optimizer = torch.optim.Adam([weights], lr=RECIPE["learning_rate"])
        for _ in range(RECIPE["steps"]):
            optimizer.zero_grad()
            _packed_objective(weights, packed).backward()
            optimizer.step()
            with torch.no_grad():
                weights.clamp_(min=0.0)
        loss = float(_packed_objective(weights, packed).detach())
    else:
        loss = None
    return weights.detach(), eligible, loss


def _choose(row, weights):
    candidates = row["candidate_texts"]
    if len(candidates) == 1:
        return candidates[0], None
    features = _features(row)
    if features is None:
        return row["current_result"], None
    scores = features.mv(weights).tolist()
    maximum = max(scores)
    winners = [index for index, score in enumerate(scores) if score == maximum]
    selected = min(winners, key=lambda index: candidates[index])
    return candidates[selected], scores


def _metrics(outcomes):
    result = {}
    for role in ("all", "natural_fit", "challenge_fit"):
        selected = outcomes if role == "all" else [row for row in outcomes
                                                    if row["source_role"] == role]
        result[role] = {
            "rows": len(selected),
            "baseline_top1": sum(row["baseline_correct"] for row in selected),
            "calibrated_top1": sum(row["calibrated_correct"] for row in selected),
            "corrections": sum(row["correction"] for row in selected),
            "regressions": sum(row["regression"] for row in selected),
            "changes": sum(row["changed"] for row in selected),
            "none_gold": sum(row["none_gold"] for row in selected),
        }
    return result


def evaluate_calibration(rows):
    """Fit four training-only calibrators and report held-fold choices."""
    _validate(rows)
    folds, outcomes = [], []
    for held in range(RECIPE["folds"]):
        train = [row for row in rows if row["fold"] != held]
        weights, eligible, loss = _fit_weights(train)
        folds.append({"fold": held, "weights": weights.tolist(),
                      "train_rows": len(train), "train_eligible_rows": len(eligible),
                      "train_row_ids": [row["id"] for row in eligible],
                      "final_loss": loss})
        for row in rows:
            if row["fold"] != held:
                continue
            choice, scores = _choose(row, weights)
            accepted = set(row["expected"])
            baseline_ok, calibrated_ok = (row["current_result"] in accepted,
                                           choice in accepted)
            outcomes.append({
                "id": row["id"], "fold": held, "source_role": row["source_role"],
                "baseline_choice": row["current_result"], "calibrated_choice": choice,
                "calibrated_scores": scores, "baseline_correct": baseline_ok,
                "calibrated_correct": calibrated_ok,
                "correction": not baseline_ok and calibrated_ok,
                "regression": baseline_ok and not calibrated_ok,
                "changed": choice != row["current_result"],
                "none_gold": not bool(_positives(row)),
            })
    outcomes.sort(key=lambda row: row["id"])
    return {
        "schema": "electra-score-calibration-v1", "recipe": dict(RECIPE),
        "rows": len(rows), "folds": folds, "oof_outcomes": outcomes,
        "metrics": _metrics(outcomes),
        "correction_ids": [row["id"] for row in outcomes if row["correction"]],
        "regression_ids": [row["id"] for row in outcomes if row["regression"]],
        "change_ids": [row["id"] for row in outcomes if row["changed"]],
        "generalization_claimed": False, "deployment_recommended": False,
    }


__all__ = ["RECIPE", "evaluate_calibration"]
