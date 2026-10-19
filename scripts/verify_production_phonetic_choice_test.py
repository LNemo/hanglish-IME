#!/usr/bin/env python3
"""Pure independent verifier for the frozen TEST1000 regression run."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re

from scripts import verify_production_phonetic_choice_dev as pure


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".omx/artifacts/production-phonetic-choice-test-v1"
REFERENCE = ROOT / ".omx/artifacts/ctc-union-runtime-test-v1"
DEV_RUN = ROOT / ".omx/artifacts/production-phonetic-choice-dev-v1"
CALIBRATION = ROOT / ".omx/artifacts/phonetic-choice-calibration-training-v1"
TEST = ROOT / "data/challenge/sentence-v3/test.jsonl"
PRODUCER = ROOT / "scripts/evaluate_production_phonetic_choice_test.py"
ADAPTER = ROOT / "hanglish/core/production_phonetic_choice_converter.py"
PURE_HELPER = ROOT / "scripts/verify_production_phonetic_choice_dev.py"
PRODUCER_SHA256 = "adfaa64401dd32fc5b5744899ca332fd03a3d144e0362eafd1f278f2bd766caf"
ADAPTER_SHA256 = "961aabb7feaadbd14c1cb1b93c98d132c47173e134484400ce8bccbb83015892"
PURE_HELPER_SHA256 = "366d463af2fefc24f7bbf69c6aa52f46e52467335021c442a14eb1a38328003c"
RUN_SCHEMA = "production-phonetic-choice-test-v1"
SCHEMA = "production-phonetic-choice-test-independent-verification-v1"
ROWS = 1000
LIMITS = dict(pure.LIMITS)
REFERENCE_HASHES = {
    REFERENCE / "report.json": "3800c78273dddddbd2118dceccbf9d91880976a3db993435964d2c3324b5ed3a",
    REFERENCE / "capture.jsonl": "d3f17017848e25212abd0ff3cd0c36eced7480c09667d8ce9e4d5416c552d404",
}
DEV_HASHES = {
    DEV_RUN / "report.json": "479ad53464e3c2ef978002fa5616ddaf9afadac936941a859ae5e01f3422b5cc",
    DEV_RUN / "capture.jsonl": "f562b5e813484fa49e2d7bb7d231bde63dd40201120eb0bae2d1043f7f4dc432",
    DEV_RUN / "capture.complete.json": "4ff2664bc0b79762864cef0bef065a287195381ef7f7ed0d63b60c891982a4d4",
    DEV_RUN / "completion.json": "2ed7af7624459493d8b527db40501426f247fc715674388bc9e178a238f444c6",
    DEV_RUN / "independent-verification.json": "a7c64c74f7fd756ef53834be326cb96e41927d472a7131020181bd811462977a",
}
WEIGHTS = CALIBRATION / "weights.json"
WEIGHTS_SHA256 = "930e3ef942061e6d459d7527c0e073fc6ad5626d59f5170e9b11df8499b39f4b"
ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def reference_rows():
    report = pure.read(REFERENCE / "report.json")
    pure.need(report.get("completed") is True and report.get("split") == "test"
              and report.get("input_contract") == "roman_sentence_only"
              and report.get("gold_not_passed_to_predictor") is True
              and report.get("capture_sha256")
                  == REFERENCE_HASHES[REFERENCE / "capture.jsonl"]
              and report.get("proposed", {}).get("counts")
                  == {"rows": 1000, "top1": 953, "top5": 972, "errors": 0},
              "frozen TEST production reference differs")
    rows = pure.read_exact_jsonl(REFERENCE / "capture.jsonl", ROWS)
    seen = set()
    for row in rows:
        request = row.get("request")
        pure.need(isinstance(row.get("id"), str) and row["id"] not in seen
                  and isinstance(request, dict)
                  and request == {"input": request.get("input"), "context": ""}
                  and isinstance(row.get("proposed"), dict)
                  and row["proposed"].get("input") == request["input"],
                  "frozen TEST source identity differs")
        seen.add(row["id"])
    return rows


def verify_probe(group, row, expected):
    source = expected["request"]["input"]
    pure.need(set(row) == {"id", "request", "parent", "proposed", "parent_paths",
                           "base_paths", "exact_full_path_parity"}
              and row["id"] == group["probe_id"]
              and row["request"] == {"input": source, "context": ""}
              and row["exact_full_path_parity"] is True
              and row["parent_paths"] == row["base_paths"],
              "fresh TEST zero probe full path parity differs")
    parent = expected["proposed"]
    pure.verify_parent_response(source, parent, row["parent"])
    derived = {"parent": parent,
               "parent_pool": [path[0] for path in row["base_paths"]]}
    return pure.verify_response(source, derived, row["proposed"],
                                row["base_paths"], [0.0] * 4)


def branch_metrics(rows, labels, branch):
    totals, categories, outcomes = Counter(), defaultdict(Counter), []
    for row, label in zip(rows, labels):
        texts = [candidate["text"] for candidate in row[branch]["candidates"][:5]]
        expected = set(label["expected"])
        top1 = bool(texts) and texts[0] in expected
        top5 = bool(expected & set(texts))
        counts = {"rows": 1, "top1": int(top1), "top5": int(top5), "errors": 0}
        totals.update(counts)
        categories[label.get("category", "uncategorized")].update(counts)
        outcomes.append((top1, top5))
    return {"counts": dict(totals),
            "categories": {key: dict(value) for key, value in categories.items()},
            "passed": totals["top1"] == len(rows)}, outcomes


def aggregate(rows, labels):
    pure.need(len(rows) == len(labels) == ROWS, "TEST denominator differs")
    parent, before = branch_metrics(rows, labels, "parent")
    proposed, after = branch_metrics(rows, labels, "proposed")
    changes, stages, english = Counter(), Counter(), Counter()
    for row, label, old, new in zip(rows, labels, before, after):
        pure.need(row["id"] == label.get("id")
                  and row["request"] == {"input": label.get("input"), "context": ""}
                  and label.get("context", "") == "" and label.get("split") == "test",
                  "late TEST label identity differs")
        changes["corrections"] += not old[0] and new[0]
        changes["regressions"] += old[0] and not new[0]
        changes["top5_losses"] += old[1] and not new[1]
        gold = set(label["expected"])
        if row["proposed"].get("sentence_ranking_applied"):
            pool_hit = bool(gold & set(row["actual_base_pool"]))
            public_hit = bool(gold & {
                candidate["text"] for candidate in row["proposed"]["candidates"]})
            selected = row["proposed"]["result"] in gold
            stages["ranked_rows"] += 1
            stages["actual_base_pool_gold"] += pool_hit
            stages["public_top5_gold"] += public_hit
            stages["selected_gold"] += selected
            stages["actual_pool_to_public_loss"] += pool_hit and not public_hit
            stages["actual_pool_to_first_loss"] += pool_hit and not selected
        source_literal = pure.literal_signature(row["request"]["input"])
        english["literal_signature_regressions"] += (
            pure.literal_signature(row["parent"]["result"]) == source_literal
            and pure.literal_signature(row["proposed"]["result"]) != source_literal)
        signatures = {tuple(ASCII_WORD.findall(text)) for text in gold}
        if any(signatures):
            english["rows_with_reference_ascii_words"] += 1
            old_match = tuple(ASCII_WORD.findall(row["parent"]["result"])) in signatures
            new_match = tuple(ASCII_WORD.findall(row["proposed"]["result"])) in signatures
            english["parent_exact_signature"] += old_match
            english["proposed_exact_signature"] += new_match
            english["signature_regressions"] += old_match and not new_match
    improvement = bool(changes["corrections"] and not changes["regressions"]
                       and not changes["top5_losses"])
    quality = bool(improvement and not english["signature_regressions"]
                   and not english["literal_signature_regressions"])
    return {"parent": parent, "proposed": proposed,
            "corrections": changes["corrections"],
            "regressions": changes["regressions"],
            "top5_losses": changes["top5_losses"],
            "improvement_gate": improvement, "quality_gate": quality,
            "english_ascii_signature": dict(english),
            "actual_base_pool_stage_counts": dict(stages)}


def expected_operations(requests, groups):
    operations = ["authenticate_dev_then_test_sources", "load_models"]
    operations += [name + ":" + group["probe_id"] for group in groups
                   for name in ("probe_production", "probe_zero")]
    operations += ["convert:" + identifier for identifier, _ in requests]
    operations += ["close_capture", "late_test_label_join",
                   "write_report", "write_completion"]
    return operations


def verify(directory):
    directory = Path(directory)
    proof_path = directory / "independent-verification.json"
    pure.need(directory.is_dir() and not proof_path.exists(), "run/proof state invalid")
    for name in ("failure.json", "failure-report.json", "resource-watchdog-failure.json"):
        pure.need(not (directory / name).exists(), "failure artifact exists: " + name)
    required = ("report.json", "completion.json", "capture.jsonl", "capture.complete.json",
                "probe.jsonl", "probe-schedule.json", "qualification.json", "pins.json",
                "resources.jsonl")
    pure.need(all((directory / name).is_file() for name in required),
              "terminal TEST run files missing")
    pure.need(pure.sha(PRODUCER) == PRODUCER_SHA256
              and pure.sha(ADAPTER) == ADAPTER_SHA256
              and pure.sha(PURE_HELPER) == PURE_HELPER_SHA256,
              "TEST producer, adapter, or pure helper changed")
    pure.need(all(path.is_file() and pure.sha(path) == digest
                  for path, digest in {**REFERENCE_HASHES, **DEV_HASHES,
                                       WEIGHTS: WEIGHTS_SHA256}.items()),
              "frozen reference, DEV gate, or calibration changed")

    report = pure.read(directory / "report.json")
    completion = pure.read(directory / "completion.json")
    closure = pure.read(directory / "capture.complete.json")
    pins = pure.read(directory / "pins.json")
    capture_sha = pure.sha(directory / "capture.jsonl")
    pure.need(closure == {"schema": RUN_SCHEMA + "-closure", "rows": ROWS,
                          "capture_sha256": capture_sha, "source_only": True,
                          "labels_opened": False}, "TEST source-only closure differs")
    pure.need(report.get("schema") == RUN_SCHEMA and report.get("completed") is True
              and report.get("split") == "test"
              and report.get("capture_sha256") == capture_sha
              and report.get("closure_sha256") == pure.sha(directory / "capture.complete.json")
              and report.get("pins") == pins, "terminal TEST report differs")
    pure.need(completion.get("completed") is True
              and completion.get("report_sha256") == pure.sha(directory / "report.json"),
              "terminal TEST completion differs")
    test_digest = pins.get(pure.path_key(TEST))
    pure.need(pins.get(pure.path_key(PRODUCER)) == PRODUCER_SHA256
              and pins.get(pure.path_key(ADAPTER)) == ADAPTER_SHA256
              and isinstance(test_digest, str), "critical TEST pins differ")
    for raw, digest in pins.items():
        path = Path(raw) if Path(raw).is_absolute() else ROOT / raw
        if path.resolve() != TEST.resolve():
            pure.need(path.is_file() and pure.sha(path) == digest,
                      "runtime pin changed: " + raw)

    dev_identity = report.get("dev_qualification")
    pure.need(dev_identity == {
        "report_sha256": DEV_HASHES[DEV_RUN / "report.json"],
        "capture_sha256": DEV_HASHES[DEV_RUN / "capture.jsonl"],
        "closure_sha256": DEV_HASHES[DEV_RUN / "capture.complete.json"],
        "completion_sha256": DEV_HASHES[DEV_RUN / "completion.json"],
        "proof_sha256": DEV_HASHES[DEV_RUN / "independent-verification.json"],
        "version": report.get("version"), "weights": report.get("weights")},
        "qualified DEV identity differs")
    frozen_rows = reference_rows()
    requests = [(row["id"], row["request"]["input"]) for row in frozen_rows]
    groups = pure.probe_groups([(identifier, source, None)
                                for identifier, source in requests])
    pure.need(pure.read(directory / "probe-schedule.json") == {
        "groups": groups,
        "selection": "16 source-length balanced groups; longest source per group"},
        "TEST probe schedule differs")

    frozen = {row["id"]: row for row in frozen_rows}
    probes = pure.read_exact_jsonl(directory / "probe.jsonl", 16)
    for group, row in zip(groups, probes):
        verify_probe(group, row, frozen[group["probe_id"]])
    qualification = pure.read(directory / "qualification.json")
    timings = qualification.get("probes")
    pure.need(qualification.get("qualified") is True
              and isinstance(timings, list) and len(timings) == 16
              and report.get("zero_probe_timing") == timings,
              "TEST qualification differs")
    weighted = 0.0
    for group, timing, probe in zip(groups, timings, probes):
        pure.need(timing.get("id") == group["probe_id"]
                  and timing.get("group_size") == group["count"]
                  and pure.finite(timing.get("zero_seconds"))
                  and timing["zero_seconds"] >= 0
                  and timing.get("production_engine_ms") == probe["parent"]["engine_ms"]
                  and timing.get("zero_engine_ms") == probe["proposed"]["engine_ms"]
                  and pure.finite(timing.get("production_engine_ms"))
                  and timing["production_engine_ms"] >= 0
                  and pure.finite(timing.get("zero_engine_ms"))
                  and timing["zero_engine_ms"] >= 0,
                  "TEST qualification timing differs")
        weighted += timing["zero_seconds"] * group["count"]
    elapsed, projected = qualification.get("elapsed_seconds"), qualification.get("projected_seconds")
    pure.need(pure.finite(elapsed) and elapsed >= 0 and pure.finite(projected)
              and projected == 1.3 * (elapsed + weighted) + 60.0
              and projected <= LIMITS["whole_seconds"]
              and qualification.get("projection")
                 == "1.3*(elapsed+sum(zero_seconds*group_size))+60",
              "TEST projection differs")

    weights_record = pure.read(WEIGHTS)
    weights = weights_record.get("weights")
    pure.need(weights_record.get("feature_names") == list(pure.FEATURE_NAMES)
              and weights == report.get("weights"), "TEST weights differ")
    capture_rows = pure.read_exact_jsonl(directory / "capture.jsonl", ROWS)
    rows, census, artifacts = [], Counter(), None
    for expected, row in zip(frozen_rows, capture_rows):
        source = expected["request"]["input"]
        pure.need(set(row) == {"id", "request", "parent", "proposed",
                               "actual_base_pool", "base_paths"}
                  and row["id"] == expected["id"]
                  and row["request"] == expected["request"]
                  and row["parent"] == expected["proposed"]
                  and row["actual_base_pool"] == [path[0] for path in row["base_paths"]],
                  "TEST capture/source/base-pool identity differs")
        derived = {"parent": expected["proposed"],
                   "parent_pool": row["actual_base_pool"]}
        census.update(pure.verify_response(source, derived, row["proposed"],
                                           row["base_paths"], weights,
                                           report.get("version")))
        pure.need(pure.finite(row["proposed"].get("engine_ms"))
                  and row["proposed"]["engine_ms"] >= 0,
                  "TEST calibrated engine timing differs")
        current = row["proposed"]["phonetic_choice"]["artifacts"]
        artifacts = current if artifacts is None else artifacts
        pure.need(current == artifacts, "TEST choice artifacts changed across requests")
        for raw, digest in current.items():
            path = Path(raw) if Path(raw).is_absolute() else ROOT / raw
            pure.need(pins.get(pure.path_key(path)) == digest,
                      "TEST choice artifact is not pinned: " + raw)
        rows.append(row)

    # Only now authenticate and parse TEST labels.  All terminal state,
    # source-only rows, paths, scores, features, and public output are closed.
    pure.need(pure.sha(TEST) == test_digest, "deferred TEST label pin differs")
    labels = pure.read_exact_jsonl(TEST, ROWS)
    metrics = aggregate(rows, labels)
    pure.need(metrics["parent"]["counts"]
              == {"rows": 1000, "top1": 953, "top5": 972, "errors": 0},
              "frozen TEST production metrics differ")
    for key, value in metrics.items():
        pure.need(report.get(key) == value, "reported TEST metric differs: " + key)

    expected_latency = {
        "actual_calibrated": pure.latency([row["proposed"]["engine_ms"] for row in rows]),
        "historical_production": pure.latency([row["parent"]["engine_ms"] for row in rows]),
        "likelihood": pure.latency([row["proposed"]["phonetic_choice"]["likelihood_ms"]
                                    for row in rows]),
        "calibration_without_likelihood": pure.latency([
            row["proposed"]["phonetic_choice"]["calibration_ms"] for row in rows]),
    }
    pure.need(report.get("latency") == expected_latency
              and report.get("input_contract") == "roman_sentence_only"
              and report.get("timing_scope")
                 == "actual calibrated runtime; historical production timings are not fresh paired measurements"
              and report.get("limits") == LIMITS
              and report.get("actual_zero_probe_rows") == 16
              and report.get("full_zero_shadow_parity_rows") == ROWS
              and report.get("parent_version") == pure.PRODUCTION_VERSION
              and report.get("production_changed") is False
              and report.get("test_accessed") is True
              and report.get("labels_opened_after_capture") is True
              and report.get("gold_not_passed_to_predictor") is True
              and report.get("repeatedly_seen_regression_set") is True
              and report.get("fully_blind_generalization") is False
              and report.get("performance_tuning_allowed") is False
              and report.get("historical_full_parent_paths_available") is False,
              "TEST report identity/timing contract differs")

    operations = expected_operations(requests, groups)
    pure.need(len(operations) == 1038, "TEST operation count differs")
    resources = pure.verify_resources(
        list(pure.jsonl(directory / "resources.jsonl")), operations)
    pure.need(completion == {"completed": True,
                             "report_sha256": pure.sha(directory / "report.json"),
                             "resource_after_report": resources["final"]},
              "TEST completion/resource linkage differs")
    pure.need(report.get("probe_sha256") == pure.sha(directory / "probe.jsonl")
              and report.get("qualification_sha256")
                  == pure.sha(directory / "qualification.json"),
              "TEST probe/qualification hash differs")

    proof = {
        "schema": SCHEMA, "verified": True,
        "report_sha256": pure.sha(directory / "report.json"),
        "completion_sha256": pure.sha(directory / "completion.json"),
        "capture_sha256": capture_sha,
        "closure_sha256": pure.sha(directory / "capture.complete.json"),
        "probe_sha256": pure.sha(directory / "probe.jsonl"),
        "qualification_sha256": pure.sha(directory / "qualification.json"),
        "producer_sha256": PRODUCER_SHA256, "adapter_sha256": ADAPTER_SHA256,
        "pure_helper_sha256": PURE_HELPER_SHA256,
        "verifier_sha256": pure.sha(Path(__file__)),
        "counts": {"rows": ROWS, "probe_rows": 16, "paths": census["paths"],
                   "unique_pair_scores_per_request_sum": census["pairs"],
                   "supported_edge_occurrences": census["supported_edges"],
                   "operations": resources["operations"]},
        "metrics": metrics, "latency": expected_latency,
        "resources": resources, "limits": LIMITS,
        "contracts_verified": {
            "dev_gate_authenticated_before_test_authorization": True,
            "source_only_capture_closed_before_test_labels": True,
            "frozen_production_public_reference": True,
            "historical_full_parent_pool_not_claimed": True,
            "actual_base_paths_and_zero_shadow_public_all_rows": True,
            "fresh_zero_probe_direct_full_path_parity": True,
            "independent_score_feature_residual_and_public_replay": True,
            "independent_test_metrics_ascii_literal_stage_counts_and_gate": True,
            "pins_completion_and_1038_atomic_operations": True,
            "model_loaded_or_inference_run_by_verifier": False,
        },
        "limitations": [
            "Recorded neural likelihood values are validated but not rerun.",
            "Historical TEST captures do not contain full precision parent paths; only public production output and fresh 16-probe full paths are compared.",
            "TEST1000 has been repeatedly observed and is regression evidence, not blind generalization evidence.",
            "Passing this proof does not itself approve production deployment or web behavior.",
        ],
    }
    with proof_path.open("xb") as stream:
        stream.write(pure.canonical(proof))
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    print(json.dumps(verify(args.run), ensure_ascii=False, sort_keys=True))
