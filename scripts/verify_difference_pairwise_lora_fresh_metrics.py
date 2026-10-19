"""Independent CPU replay of serialized pairwise fresh predictions and metrics.

No producer resolver, metric, pair-builder, or bootstrap function is imported.
This checks metric evidence only. The eventual execution verifier must also
prove artifact lineage, prompt/logit binding, resources, capture closure and
model release before target access. This module neither loads files/models nor
authorizes deployment.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import random

from scripts.final_choice_policy import proposal_allowed

SCHEMA = "difference-pairwise-lora-fresh-metrics-v1"
VERSION = "neural-1-867f13596ed5"
ROLES = ("fresh_natural", "challenge_preservation")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def replay_one(row, prediction):
    candidates, raw = row["candidates"], row["raw_bypass_response"]
    parent = row["parent_response"]
    need(isinstance(candidates, list) and 1 <= len(candidates) <= 5
         and all(isinstance(c, dict) and isinstance(c.get("text"), str)
                 and isinstance(c.get("segments"), list) for c in candidates),
         "parent candidates malformed")
    need(isinstance(parent, dict) and isinstance(raw, dict)
         and canonical(parent.get("candidates")) == canonical(candidates)
         and parent.get("result") == candidates[0]["text"], "parent candidate snapshot differs")
    for response, raw_mode in ((parent, False), (raw, True)):
        need(response.get("input") == row["input"]
             and response.get("context") == row["context"]
             and response.get("version") == VERSION and response.get("raw") is raw_mode,
             "parent response source or mode differs")
    need(raw.get("result") == row["input"], "raw result differs")
    finite = []
    for index, candidate in enumerate(candidates):
        score = candidate.get("score")
        if score is not None:
            need(type(score) in (int, float) and math.isfinite(score), "candidate score nonfinite")
            finite.append(index)
    allowed = [] if not finite or finite[0] != 0 else [
        i for i in finite if i and proposal_allowed(row["input"], candidates[0], candidates[i])]
    need(isinstance(prediction, dict), "prediction malformed")
    records = prediction.get("pair_scores")
    need(isinstance(records, list) and len(records) == len(allowed), "incomplete pair scores")
    scores = []
    for index, record in zip(allowed, records):
        need(isinstance(record, dict), "pair score malformed")
        identity = {"role": row["role"], "row_id": row["id"],
                    "current_source_index": 0, "alternative_source_index": index}
        left, right = record.get("keep_logit"), record.get("switch_logit")
        need(all(type(x) in (int, float) and math.isfinite(x) for x in (left, right)),
             "pair logits nonfinite")
        left, right = float(left), float(right)
        need(math.isfinite(right - left), "pair margin nonfinite")
        expected = {"pair_id": digest(identity), "alternative_source_index": index,
                    "keep_logit": left, "switch_logit": right, "margin": right - left}
        need(canonical(record) == canonical(expected), "pair identity or margin differs")
        scores.append(expected)
    positive = sorted((s for s in scores if s["margin"] > 0),
                      key=lambda s: (-s["margin"], s["alternative_source_index"]))
    chosen = positive[0]["alternative_source_index"] if positive else 0
    reason = "switch_margin" if positive else (
        "all_keep" if allowed else "no_permitted_non_parent_alternative")
    ordered = [candidates[chosen]] + [c for i, c in enumerate(candidates) if i != chosen]
    replayed = {"schema": "difference-pairwise-lora-evaluation-v1", "role": row["role"],
        "id": row["id"], "source": row["input"], "context": row["context"],
        "raw_mode": False, "complete": True, "reason": reason, "failures": [],
        "chosen_index": chosen, "proposed_index": chosen if positive else None,
        "rejected_switch_indices": [], "expected_pair_count": len(allowed),
        "scored_pair_count": len(allowed), "pair_scores": scores,
        "result": ordered[0]["text"], "candidates": ordered, "raw_bypass_response": raw,
        "candidate_sha256_before": digest(candidates), "raw_sha256_before": digest(raw)}
    need(canonical(prediction) == canonical(replayed), "prediction does not exactly replay")
    return replayed


def independent_bootstrap(outcomes):
    documents = defaultdict(lambda: [0, 0])
    corrections = regressions = 0
    for group, before, after in outcomes:
        documents[group][0] += int(after) - int(before)
        documents[group][1] += 1
        corrections += int(after and not before)
        regressions += int(before and not after)
    groups = [documents[k] for k in sorted(documents)]
    rng = random.Random(20260915)
    samples = []
    for _ in range(2000):
        drawn = rng.choices(groups, k=len(groups))
        samples.append(sum(d[0] for d in drawn) / sum(d[1] for d in drawn))
    samples.sort()
    gain = corrections - regressions
    return {"rows": len(outcomes), "groups": len(groups), "corrections": corrections,
        "regressions": regressions, "net_gain": gain, "accuracy_delta": gain / len(outcomes),
        "resamples": 2000, "seed": 20260915, "lower_index": 49,
        "lower_2_5_percentile": samples[49], "gain_and_interval_passed": gain >= 8 and samples[49] > 0}


def recompute_metrics(rows, predictions, targets):
    need(all(isinstance(x, (list, tuple)) and len(x) == 888 for x in (rows, predictions, targets)),
         "all 888 rows required")
    seen, groups, replayed = set(), set(), []
    # Validate all prediction evidence, including the last row, before touching
    # any target record. Input/target file hashes belong to the execution layer.
    for index, (row, prediction) in enumerate(zip(rows, predictions)):
        role = ROLES[0] if index < 512 else ROLES[1]
        need(isinstance(row, dict) and row.get("role") == role,
             "source role or order differs")
        need(all(k in row for k in ("candidates", "parent_response", "raw_bypass_response")),
             "parent source fields missing")
        need(all(isinstance(row.get(k), str) and row[k] for k in ("id", "group_id", "input"))
             and isinstance(row.get("context"), str) and row["id"] not in seen,
             "source identity malformed or duplicated")
        seen.add(row["id"])
        if index < 512:
            groups.add(row["group_id"])
        replayed.append(replay_one(row, prediction))
    need(len(groups) == 387, "fresh group accounting differs")
    fields = ("rows", "before_top1", "top1", "before_top5", "top5", "corrections",
              "regressions", "net_gain", "candidate_present", "candidate_absent", "actions", "fallbacks")
    roles = {role: {**dict.fromkeys(fields, 0), "reasons": Counter()} for role in ROLES}
    outcomes = []
    for row, prediction, target in zip(rows, replayed, targets):
        need(isinstance(target, dict) and set(target) == {"id", "group_id", "role", "expected"}
             and all(target.get(k) == row[k] for k in ("id", "group_id", "role")),
             "target identity or order differs")
        answers = target.get("expected")
        need(isinstance(answers, list) and bool(answers)
             and all(isinstance(a, str) and a for a in answers)
             and len(set(answers)) == len(answers), "target answer set malformed")
        before = row["candidates"][0]["text"] in answers
        after = prediction["result"] in answers
        before5 = any(c["text"] in answers for c in row["candidates"])
        after5 = any(c["text"] in answers for c in prediction["candidates"])
        need(before5 == after5, "top5 preservation differs")
        counters = roles[row["role"]]
        counters["rows"] += 1
        for name, condition in (("before_top1", before), ("top1", after),
                ("before_top5", before5), ("top5", after5),
                ("corrections", after and not before), ("regressions", before and not after),
                ("candidate_present", before5), ("candidate_absent", not before5),
                ("actions", prediction["reason"] == "switch_margin"),
                ("fallbacks", prediction["reason"] != "switch_margin")):
            counters[name] += int(condition)
        counters["net_gain"] += int(after) - int(before)
        counters["reasons"][prediction["reason"]] += 1
        if row["role"] == ROLES[0]:
            outcomes.append((row["group_id"], before, after))
    for counters in roles.values():
        counters["reasons"] = dict(sorted(counters["reasons"].items()))
    bootstrap = independent_bootstrap(outcomes)
    fresh = bootstrap["gain_and_interval_passed"]
    preservation = roles[ROLES[1]]["before_top1"] == roles[ROLES[1]]["top1"] == 376
    passed = fresh and preservation
    return {"schema": SCHEMA, "roles": roles, "bootstrap": bootstrap,
            "gate": {"passed": passed, "fresh_passed": fresh,
                     "preservation_all_correct": preservation, "invariants_preserved": True,
                     "downstream_evaluation_allowed": passed, "promotion_allowed": False}}


def verify_reported_metrics(rows, predictions, targets, reported):
    expected = recompute_metrics(rows, predictions, targets)
    need(canonical(reported) == canonical(expected), "reported fresh metrics differ from independent replay")
    return {"schema": "difference-pairwise-lora-fresh-metrics-verification-v1",
            "verification_passed": True, "quality_gate_passed": expected["gate"]["passed"],
            "metrics_sha256": digest(expected), "execution_verified": False,
            "promotion_allowed": False}
