"""Describe frozen TRAIN calibration signal; never read evaluation examples.

This is a diagnostic, not a parameter search or a new accuracy evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / ".omx/artifacts/phonetic-choice-features-v1"
TRAINING = ROOT / ".omx/artifacts/phonetic-choice-calibration-training-v1"
PINS = {
    FEATURES / "manifest.json": "b884847fc4fc5f69d20acb95628135f82ba5fa6596479e540600f29e59517a84",
    FEATURES / "rows.jsonl": "20071a1874e057f346b4d3f131767c3a13182806576c98ac773c09f389ebccec",
    FEATURES / "independent-verification.json": "9eed6ed3642be4fe34a5741886f8c9b9a36e8654e88eb0fec27e4eda77baef32",
    TRAINING / "weights.json": "930e3ef942061e6d459d7527c0e073fc6ad5626d59f5170e9b11df8499b39f4b",
    TRAINING / "independent-verification.json": "7181d0ca4e783c0cf40e10ed28f26d4e2401e49829f5aade13f2123d5da06a83",
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(values):
    if not values:
        return {"count": 0}
    values = sorted(values)
    return {"count": len(values), "min": values[0], "max": values[-1],
            "p50": values[math.ceil(.5 * len(values)) - 1],
            "p95": values[math.ceil(.95 * len(values)) - 1]}


def audit(rows, weights):
    need(len(weights) == 4 and all(type(w) in (int, float) and math.isfinite(w) for w in weights),
         "invalid four weights")
    counts = Counter(rows=0, candidates=0, base_correct=0, calibrated_correct=0,
                     corrections=0, regressions=0, base_errors=0,
                     errors_with_all_positives_blocked_by_identical_features=0,
                     errors_best_gold_same_features_as_winner=0,
                     errors_phonetic_likelihood_prefers_best_gold=0,
                     errors_phonetic_likelihood_ties_best_gold=0,
                     errors_residual_moves_best_gold_toward_winner=0)
    feature_differences = [0] * 4
    gaps, improvements, ranges = [], [], []
    seen = set()
    for row in rows:
        need(isinstance(row.get("id"), str) and row["id"] not in seen, "duplicate or invalid row")
        seen.add(row["id"])
        candidates, base, features = row["candidates"], row["base_scores"], row["features"]
        positive = set(row["positive_indices"])
        size = len(candidates)
        need(size > 0 and len(base) == len(features) == size and len(set(candidates)) == size
             and positive and all(type(i) is int and 0 <= i < size for i in positive),
             "invalid row domain")
        need(all(type(v) in (int, float) and math.isfinite(v) for v in base)
             and all(len(f) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in f)
                     for f in features), "invalid numeric feature")
        residual = [sum(w * v for w, v in zip(weights, f)) for f in features]
        adjusted = [b + r for b, r in zip(base, residual)]
        choose = lambda indices, scores: min(indices, key=lambda i: (-scores[i], candidates[i]))
        before, after = choose(range(size), base), choose(range(size), adjusted)
        old, new = before in positive, after in positive
        counts.update(rows=1, candidates=size, base_correct=int(old), calibrated_correct=int(new),
                      corrections=int(not old and new), regressions=int(old and not new))
        ranges.append(max(residual) - min(residual))
        if old:
            continue
        counts["base_errors"] += 1
        gold = choose(positive, base)
        delta = [g - b for g, b in zip(features[gold], features[before])]
        for index, value in enumerate(delta):
            feature_differences[index] += value != 0
        counts["errors_best_gold_same_features_as_winner"] += features[gold] == features[before]
        counts["errors_phonetic_likelihood_prefers_best_gold"] += delta[0] > 0
        counts["errors_phonetic_likelihood_ties_best_gold"] += delta[0] == 0
        movement = residual[gold] - residual[before]
        counts["errors_residual_moves_best_gold_toward_winner"] += movement > 0
        gaps.append(base[before] - base[gold])
        improvements.append(movement)
        # A wrong candidate with exactly the same feature vector and a better
        # base rank dominates that positive for every possible four-weight fit.
        best_wrong_by_feature = {}
        for i in range(size):
            if i in positive:
                continue
            vector = tuple(features[i])
            old_index = best_wrong_by_feature.get(vector)
            if old_index is None or (-base[i], candidates[i]) < (-base[old_index], candidates[old_index]):
                best_wrong_by_feature[vector] = i
        blocked = []
        for i in positive:
            blocker = best_wrong_by_feature.get(tuple(features[i]))
            blocked.append(blocker is not None and
                           (-base[blocker], candidates[blocker]) < (-base[i], candidates[i]))
        counts["errors_with_all_positives_blocked_by_identical_features"] += all(blocked)
    need(counts["rows"] > 0, "empty TRAIN rows")
    return {"counts": dict(counts), "base_error_feature_difference_counts": feature_differences,
            "base_error_gold_gap": summarize(gaps),
            "base_error_gold_residual_advantage": summarize(improvements),
            "all_row_residual_range": summarize(ranges)}


def run(output):
    output = Path(output)
    need(not output.exists(), "audit output exists")
    for path, digest in PINS.items():
        need(sha(path) == digest, "frozen TRAIN artifact changed")
    manifest = json.loads((FEATURES / "manifest.json").read_text())
    weights = json.loads((TRAINING / "weights.json").read_text())["weights"]
    with (FEATURES / "rows.jsonl").open() as stream:
        metrics = audit((json.loads(line) for line in stream), weights)
    need(metrics["counts"]["rows"] == manifest["counts"]["rows"] == 1483
         and metrics["counts"]["candidates"] == manifest["counts"]["candidates"] == 202603,
         "TRAIN denominator differs")
    result = {"schema": "phonetic-calibration-fit-signal-audit-v1", "completed": True,
              "scope": "TRAIN-only descriptive diagnostic; no parameter search",
              "inputs": {str(p.relative_to(ROOT)): h for p, h in PINS.items()},
              "producer_sha256": sha(Path(__file__)), "weights": weights, **metrics,
              "model_loaded": False, "training_performed": False,
              "evaluation_cases_read": False, "production_changed": False,
              "limitations": ["FIT candidates predate the current phonetic candidate union.",
                              "Identical-feature blocking is a sufficient obstruction, not a complete feasibility test.",
                              "In-sample ranking does not measure generalization or new candidate coverage."]}
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), ensure_ascii=False))
