"""Independently verify the closed numeric-composition DEV evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

from scripts import verify_numeric_composition_downstream as trusted


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "numeric-composition-development-proof-v1"
REPORT_SCHEMA = "numeric-composition-development-v1"
RUNTIME_VERSION = "neural-1-867f13596ed5"
PROFILE = {"cpu_threads": 2, "rss_limit_bytes": 3 * 1024**3,
           "time_limit_seconds": 1800.0, "progress_interval": 32}
SUITE_COUNTS = {"natural_dev": 128, "original_dev": 140,
                "selection_dev": 60, "reasoning_dev": 40}
TOTAL_ROWS = 368
DOWNSTREAM_PROOF = ROOT / ".omx/artifacts/numeric-composition-downstream-v1-proof.json"
DOWNSTREAM_PROOF_SHA256 = "b7777ebe42950dc5e264e728c0f47c894f06b9024fa82ac954a3c2c58b25fd7c"
DOWNSTREAM_VERIFIER_SHA256 = "0aed52466a7153466dab6b31657aa278a4bb169d4cf71da041ad7147af6fceea"
INTEGRATION = ROOT / ".omx/artifacts/numeric-structure-integration-v1/report.json"
INTEGRATION_SHA256 = "77c04334af817e15bc4c5ccc9d27b1d6ab9f9f6a759b4207e752762ae49d3aab"
REFERENCE = ROOT / ".omx/artifacts/numeric-structure-development-v1"
REFERENCE_REPORT_SHA256 = "750e777f784e3bbdd2e79246b8abc68532e4840dc18e6c9f035da3e62a9b14e0"
REFERENCE_CASES_SHA256 = "2baaccfbca4d7f4370a9c0b61c548fa9d4e33c50440c38c6dabdcd67d2ca920c"
PRODUCER_SHA256 = "bfaeb972d26775fa20f97aa47d79020dd4b0aaa6b19fb136b0bf88d047663564"
SUITES = (
    ("natural_dev", ROOT / "data/natural_context/current-context-v1/natural_dev.jsonl",
     ROOT / "data/natural_context/current-context-v1/manifest.json"),
    ("original_dev", ROOT / "data/challenge/dev.jsonl",
     ROOT / ".omx/artifacts/challenge-dev-natural-v3-deployed.json"),
    ("selection_dev", ROOT / "data/challenge/selection-dev-v1.1/cases.jsonl",
     ROOT / ".omx/artifacts/selection-dev-natural-v3-deployed.json"),
    ("reasoning_dev", ROOT / "data/challenge/reasoning-dev-v1.1/cases.jsonl",
     ROOT / ".omx/artifacts/reasoning-dev-natural-v3-deployed.json"),
)
DOWNSTREAM_FILES = {
    "report": (".omx/artifacts/numeric-composition-downstream-v1/report.json",
               "209df79c60e01f50855ddaeb9ca245f06b27649bbabbb9d722988994343ccbd9", None),
    "sources": (".omx/artifacts/numeric-composition-downstream-v1/sources.jsonl",
                "e8fefa187dd3b87809ac4d8a2b88add86b6f34cd87c5c552580b922d00b55ff6", 888),
    "targets": (".omx/artifacts/numeric-composition-downstream-v1/targets.jsonl",
                "957f784665edca86162e0b677472b1e2f4ce716edd7991e87992da6568c0182f", 888),
    "captures": (".omx/artifacts/numeric-composition-downstream-v1/captures.jsonl",
                 "d4639c041330262ac8444a047e6d841f2512c8ef3f893222969750a5e5b2ddae", 888),
    "preservation_baselines": (
        ".omx/artifacts/numeric-composition-downstream-v1/preservation_baselines.jsonl",
        "e95119f0e0ac17c4383bdeaddfd451f0434ece1b6b9f84a58649aafcac2d7580", 376),
}
REFERENCE_TOP1 = {"natural_dev": 81, "original_dev": 140,
                  "selection_dev": 60, "reasoning_dev": 34}
REFERENCE_BEFORE = {"natural_dev": 76, "original_dev": 140,
                    "selection_dev": 60, "reasoning_dev": 34}
FLAG_FIELDS = trusted.FLAG_FIELDS


VerificationError = trusted.VerificationError
need = trusted.need
canonical = trusted.canonical
digest = trusted.digest
file_digest = trusted.file_digest
read_json = trusted.read_json
read_jsonl = trusted.read_jsonl
path_key = trusted.path_key
valid_hash_map = trusted.valid_hash_map
merge_hashes = trusted.merge_hashes
verify_current_hashes = trusted.verify_current_hashes


def _validate_authority():
    """Bind accepted fresh evidence and DEV metadata without parsing DEV targets."""
    need(file_digest(DOWNSTREAM_PROOF) == DOWNSTREAM_PROOF_SHA256,
         "numeric downstream proof changed")
    envelope = read_json(DOWNSTREAM_PROOF)
    payload = envelope.get("payload")
    need(envelope.get("schema") == "numeric-composition-downstream-proof-v1"
         and isinstance(payload, dict) and envelope.get("digest") == digest(canonical(payload))
         and payload.get("verification_passed") is True
         and payload.get("quality_gate_passed") is True
         and payload.get("downstream_evaluation_allowed") is True
         and payload.get("resource_valid_experiment") is True
         and payload.get("labels_read_only_after_complete_capture_preflight") is True
         and payload.get("fresh_targets_read") is True
         and payload.get("development_final_targets_read") is False
         and payload.get("promotion_allowed") is False and payload.get("deployed") is False
         and payload.get("runtime_version") == RUNTIME_VERSION
         and payload.get("verifier") == {
             "path": "scripts/verify_numeric_composition_downstream.py",
             "sha256": DOWNSTREAM_VERIFIER_SHA256},
         "numeric downstream proof is not a valid DEV entry")
    for field, (name, expected, rows) in DOWNSTREAM_FILES.items():
        item = payload.get(field)
        need(isinstance(item, dict) and item.get("path") == name
             and item.get("sha256") == expected and (rows is None or item.get("rows") == rows)
             and file_digest(ROOT / name) == expected,
             f"numeric downstream {field} lineage differs")
    need(file_digest(ROOT / payload["verifier"]["path"]) == DOWNSTREAM_VERIFIER_SHA256,
         "numeric downstream verifier changed")
    downstream_report = read_json(ROOT / DOWNSTREAM_FILES["report"][0])
    need(downstream_report.get("completed") is True
         and downstream_report.get("terminal_status") == "success"
         and downstream_report.get("captures_closed") is True
         and downstream_report.get("models_released_before_target_join") is True
         and downstream_report.get("targets_joined") is True
         and downstream_report.get("quality_gate_passed") is True
         and downstream_report.get("downstream_evaluation_allowed") is True
         and downstream_report.get("promotion_allowed") is False
         and downstream_report.get("deployed") is False
         and downstream_report.get("metrics") == payload.get("metrics")
         and valid_hash_map(downstream_report.get("source_hashes_before"))
         and downstream_report["source_hashes_before"]
            == downstream_report.get("source_hashes_after")
         and digest(canonical(downstream_report["source_hashes_before"]))
            == payload.get("source_hashes_sha256"),
         "numeric downstream report semantics differ")
    fresh = payload.get("metrics", {}).get("roles", {}).get("fresh_natural", {})
    protected = payload.get("metrics", {}).get("roles", {}).get("challenge_preservation", {})
    bootstrap = payload.get("metrics", {}).get("bootstrap", {})
    need(fresh.get("rows") == 512 and fresh.get("before_top1") == 318
         and fresh.get("top1") == 336 and fresh.get("before_top5") == 331
         and fresh.get("top5") == 350 and len(fresh.get("corrections", [])) == 18
         and fresh.get("regressions") == [] and fresh.get("top5_losses") == []
         and protected.get("rows") == 376 and protected.get("top1") == 376
         and protected.get("top5") == 376 and protected.get("regressions") == []
         and bootstrap.get("net_gain") == 18
         and bootstrap.get("gain_and_interval_passed") is True,
         "numeric downstream fixed quality result differs")
    verify_current_hashes(downstream_report["source_hashes_before"])

    need(file_digest(INTEGRATION) == INTEGRATION_SHA256,
         "numeric integration report changed")
    integration = read_json(INTEGRATION)
    weights = integration.get("accepted_weight_hashes")
    need(integration.get("passed") is True
         and integration.get("scope") == "post-integration actual runtime parity"
         and integration.get("integrated_runtime_version") == RUNTIME_VERSION
         and integration.get("accepted_development_report_sha256") == REFERENCE_REPORT_SHA256
         and integration.get("prototype_artifact_hashes", {}).get(
             ".omx/artifacts/numeric-structure-development-v1/cases.jsonl")
             == REFERENCE_CASES_SHA256
         and integration.get("prototype_artifact_hashes", {}).get(
             ".omx/artifacts/numeric-structure-development-v1/report.json")
             == REFERENCE_REPORT_SHA256
         and valid_hash_map(weights) and len(weights) == 16
         and weights == payload.get("model_artifacts")
         and integration.get("integrated_runtime_source_hashes")
            == integration.get("runtime_source_hashes_after")
         and integration.get("rows") == 1880
         and integration.get("all_candidate_objects_equal") is True
         and integration.get("all_sentence_pools_equal") is True
         and integration.get("all_sentence_scores_within_tolerance") is True
         and integration.get("all_versions_equal") is True
         and integration.get("deployed") is False,
         "accepted current production reference differs")
    reference_report_path = REFERENCE / "report.json"
    need(file_digest(reference_report_path) == REFERENCE_REPORT_SHA256,
         "accepted numeric DEV report changed")
    reference = read_json(reference_report_path)
    need(reference.get("passed") is True
         and reference.get("eligible_for_further_evaluation") is True
         and reference.get("cases_sha256") == REFERENCE_CASES_SHA256
         and valid_hash_map(reference.get("source_hashes"))
         and reference["source_hashes"] == reference.get("source_hashes_after")
         and {name: reference.get("suites", {}).get(name, {}).get("rows")
              for name in SUITE_COUNTS} == SUITE_COUNTS
         and reference.get("final_data_read") is False and reference.get("deployed") is False,
         "accepted numeric DEV identity report differs")
    for name in SUITE_COUNTS:
        suite = reference["suites"][name]
        need(suite.get("before_top1") == REFERENCE_BEFORE[name]
             and suite.get("top1") == REFERENCE_TOP1[name]
             and suite.get("regressions") == [] and suite.get("top5_regressions") == [],
             "accepted numeric DEV reference metrics differ")
    source_hashes = dict(downstream_report["source_hashes_before"])
    merge_hashes(source_hashes, integration["integrated_runtime_source_hashes"])
    merge_hashes(source_hashes, weights)
    fixed = [DOWNSTREAM_PROOF, ROOT / payload["verifier"]["path"], INTEGRATION,
             reference_report_path, ROOT / "scripts/evaluate_numeric_composition_downstream.py",
             ROOT / "scripts/evaluate_numeric_structure_development.py",
             ROOT / "scripts/evaluate_numeric_composition_development.py"]
    fixed.extend(ROOT / values[0] for values in DOWNSTREAM_FILES.values())
    for path in fixed:
        merge_hashes(source_hashes, {path_key(path): file_digest(path)})
    need(source_hashes.get("scripts/evaluate_numeric_composition_development.py")
         == PRODUCER_SHA256, "numeric DEV producer changed")
    for _suite, corpus, identity in SUITES:
        expected_corpus = reference["source_hashes"].get(path_key(corpus))
        expected_identity = reference["source_hashes"].get(path_key(identity))
        need(file_digest(corpus) == expected_corpus and file_digest(identity) == expected_identity,
             "frozen DEV corpus or identity hash differs")
        merge_hashes(source_hashes, {path_key(corpus): expected_corpus,
                                     path_key(identity): expected_identity})
    merge_hashes(source_hashes, {path_key(REFERENCE / "cases.jsonl"): REFERENCE_CASES_SHA256})
    verify_current_hashes(source_hashes)
    return {"proof": payload, "integration": integration, "reference_report": reference,
            "source_hashes": dict(sorted(source_hashes.items())), "weights": weights}


def _validate_capture(source, capture, weights):
    need(all(capture.get(name) == source[name] for name in source),
         "capture source identity differs")
    for name in ("parent", "child", "parent_raw", "child_raw"):
        response, raw = capture.get(name), name.endswith("raw")
        trusted._validate_response(response, source["input"], source["context"],
                                   raw=raw, model_artifacts=weights)
        need(set(response) == {"input", "context", "engine", "mode", "raw", "version",
             "candidates", "result", "ranker", "model_artifacts", "context_used",
             "external_context_used", "ranking_applied", "sentence_ranking_applied",
             "warnings"} and response.get("mode") == "batch"
             and all(type(response.get(flag)) is bool for flag in
                     ("context_used", "external_context_used", "ranking_applied",
                      "sentence_ranking_applied"))
             and response.get("context_used") == response.get("external_context_used")
             and isinstance(response.get("warnings"), list)
             and all(isinstance(value, str) for value in response["warnings"])
             and ((raw and response["context_used"] is False
                   and response["ranking_applied"] is False
                   and response["sentence_ranking_applied"] is False)
                  or (not raw and response.get("ranker") == "hybrid")),
             "complete response flags differ")
    need(canonical(capture["parent_raw"]) == canonical(capture["child_raw"]),
         "raw response changed")
    for name in ("parent_ms", "child_ms"):
        value = capture.get(name)
        need(type(value) in (int, float) and math.isfinite(value) and value >= 0,
             "capture timing is malformed")
    for name in ("parent_pool", "parent_base"):
        need(isinstance(capture.get(name), list) and capture[name], f"path pool missing: {name}")
        for path in capture[name]:
            trusted._validate_path(path, source["input"])
    for flag in ("parent_sentence_scored", "child_sentence_scored"):
        need(type(capture.get(flag)) is bool, "sentence scoring flag is malformed")
    need(capture["parent_sentence_scored"]
            is capture["parent"].get("sentence_ranking_applied")
         and capture["child_sentence_scored"]
            is capture["child"].get("sentence_ranking_applied"),
         "capture sentence scoring flags differ from responses")
    trace = capture.get("trace")
    lists = ("base", "accepted_old_extras", "preserved_pool", "production_scoring_pool",
             "generated_new_extras", "accepted_new_extras", "scoring_pool")
    need(isinstance(trace, dict) and all(isinstance(trace.get(name), list) for name in lists)
         and type(trace.get("eligible_piece_count")) is int and trace["eligible_piece_count"] >= 0
         and type(trace.get("retained_combination_count")) is int
         and trace["retained_combination_count"] >= 0
         and type(trace.get("composition_bypassed")) is bool
         and type(trace.get("activated")) is bool
         and type(trace.get("production_activated")) is bool
         and type(trace.get("base_scoreable")) is bool
         and isinstance(trace.get("reason"), str) and trace["reason"],
         "numeric composition trace is incomplete")
    need(trace["production_activated"] is capture["parent_sentence_scored"]
         and trace["activated"] is capture["child_sentence_scored"],
         "trace activation flags differ from captured scoring")
    for name in lists:
        for path in trace[name]:
            trusted._validate_path(path, source["input"])
    preserved = trace["base"] + trace["accepted_old_extras"]
    need(canonical(trace["base"]) == canonical(capture["parent_base"])
         and canonical(trace["preserved_pool"]) == canonical(preserved)
         and canonical(trace["scoring_pool"][:len(preserved)]) == canonical(preserved),
         "original base/preserved pool changed")
    if capture["parent_sentence_scored"]:
        need(canonical(capture["parent_pool"]) == canonical(preserved)
             and canonical(trace["production_scoring_pool"]) == canonical(preserved),
             "original scored full pool changed")
    else:
        need(trace["production_scoring_pool"] == [], "unexpected production scoring pool")
    if capture["child_sentence_scored"]:
        need(capture.get("child_score_inputs") == [row["text"] for row in trace["scoring_pool"]],
             "actual scorer inputs differ from trace")
    else:
        need(capture.get("child_score_inputs") is None, "unexpected child scorer inputs")


def _preflight(directory, authority):
    """Validate all source-only evidence before the target loader is called."""
    directory = Path(directory).resolve()
    report_path = directory / "report.json"
    report_sha, report = file_digest(report_path), read_json(report_path)
    need(report.get("schema") == REPORT_SCHEMA and report.get("completed") is True
         and report.get("terminal_status") == "success"
         and report.get("captures_closed") is True
         and report.get("models_released_before_target_join") is True
         and report.get("targets_joined") is True and report.get("model_loaded") is False
         and report.get("captured_rows") == TOTAL_ROWS and report.get("rows") == TOTAL_ROWS
         and report.get("suite_counts") == SUITE_COUNTS and report.get("profile") == PROFILE
         and report.get("runtime_version") == RUNTIME_VERSION
         and report.get("model_artifacts") == authority["weights"]
         and report.get("target_access") == {"development_target_bytes_accessed": True}
         and report.get("final_data_read") is False
         and report.get("promotion_allowed") is False and report.get("deployed") is False,
         "numeric DEV report is incomplete or has the wrong scope")
    files = report.get("files")
    need(isinstance(files, dict) and set(files) == {"sources.jsonl", "targets.jsonl",
         "parent_references.jsonl", "captures.jsonl"}, "numeric DEV file inventory differs")
    for name, expected in files.items():
        need(file_digest(directory / name) == expected, f"closed DEV file hash differs: {name}")
    expected_hashes = authority["source_hashes"]
    need(len(expected_hashes) == 146
         and report.get("source_hashes_before") == expected_hashes
         and report.get("source_hashes_after") == expected_hashes,
         "numeric DEV source hash inventory differs")
    verify_current_hashes(expected_hashes)

    sources, captures = read_jsonl(directory / "sources.jsonl"), read_jsonl(directory / "captures.jsonl")
    references = read_jsonl(directory / "parent_references.jsonl")
    need(len(sources) == len(captures) == len(references) == TOTAL_ROWS,
         "exact 368 DEV source captures are required")
    expected_roles = [name for name, count in SUITE_COUNTS.items() for _ in range(count)]
    ids = set()
    for source, capture, reference, role in zip(sources, captures, references, expected_roles):
        need(set(source) == {"id", "group_id", "origin", "input", "context"}
             and all(isinstance(source.get(name), str) for name in source)
             and source["id"] and source["group_id"] and source["input"]
             and source["id"] not in ids and source["origin"] == role,
             "numeric DEV source-only row differs")
        ids.add(source["id"])
        need(isinstance(reference, dict) and set(reference) == {
            "id", "suite", "candidates", "sentence_pool", "flags"}
             and reference["id"] == source["id"] and reference["suite"] == role
             and "expected" not in reference, "numeric DEV parent reference differs")
        _validate_capture(source, capture, authority["weights"])
        need(canonical(reference["candidates"]) == canonical(capture["parent"]["candidates"]),
             "parent candidates differ from accepted DEV reference")
        expected_pool = ([row["text"] for row in capture["parent_pool"]]
                         if capture["parent_sentence_scored"] else None)
        need(reference["sentence_pool"] == expected_pool,
             "parent sentence pool differs from accepted DEV reference")
        comparable = ("ranker", "context_used", "ranking_applied",
                      "sentence_ranking_applied", "warnings")
        need(all(reference["flags"].get(name) == capture["parent"].get(name)
                 for name in comparable), "parent flags differ from accepted DEV reference")

    observations = report.get("resource_observations")
    need(isinstance(observations, list) and len(observations) == TOTAL_ROWS + 1,
         "numeric DEV resource observation count differs")
    previous = -1.0
    for index, observation in enumerate(observations):
        need(isinstance(observation, dict), "numeric DEV resource observation differs")
        stage = f"development_capture_{index+1}" if index < TOTAL_ROWS \
            else "after_development_model_release"
        elapsed, rss = observation.get("elapsed_seconds"), observation.get("rss_peak_bytes")
        need(observation.get("stage") == stage
             and type(elapsed) in (int, float) and math.isfinite(elapsed)
             and previous <= elapsed <= PROFILE["time_limit_seconds"]
             and type(rss) is int and 0 <= rss <= PROFILE["rss_limit_bytes"],
             "numeric DEV resource observation differs")
        previous = elapsed
    need(report.get("terminal_resources") == observations[-1],
         "numeric DEV terminal resources differ")
    elapsed = report.get("elapsed_seconds")
    need(type(elapsed) in (int, float) and math.isfinite(elapsed)
         and previous <= elapsed <= PROFILE["time_limit_seconds"],
         "numeric DEV elapsed time differs")
    created_at = report.get("created_at")
    need(isinstance(created_at, str), "numeric DEV timestamp differs")
    try:
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise VerificationError("numeric DEV timestamp differs") from error
    warm = captures[1:]; percentile = math.ceil(.95 * len(warm)) - 1
    latency = {"first_parent_ms": captures[0]["parent_ms"],
        "first_child_ms": captures[0]["child_ms"],
        "warm_parent_p95_ms": sorted(row["parent_ms"] for row in warm)[percentile],
        "warm_child_p95_ms": sorted(row["child_ms"] for row in warm)[percentile],
        "scope": "engine calls after model load; not browser or cold model loading"}
    need(report.get("latency") == latency, "numeric DEV reported latency differs")
    need(file_digest(report_path) == report_sha, "numeric DEV report changed during preflight")
    return directory, report_path, report_sha, report, sources, captures, expected_hashes


def _default_target_loader(directory, authority, sources):
    """Open only the four DEV suites after the source-only preflight."""
    targets = read_jsonl(directory / "targets.jsonl")
    saved_references = read_jsonl(directory / "parent_references.jsonl")
    reference_rows = read_jsonl(REFERENCE / "cases.jsonl")
    grouped = {name: [] for name in SUITE_COUNTS}
    for row in reference_rows:
        need(row.get("suite") in grouped, "accepted numeric DEV reference suite differs")
        grouped[row["suite"]].append(row)
    need({name: len(rows) for name, rows in grouped.items()} == SUITE_COUNTS,
         "accepted numeric DEV reference counts differ")
    offset = 0
    for suite, corpus_path, _identity_path in SUITES:
        corpus = read_jsonl(corpus_path)
        need(len(corpus) == SUITE_COUNTS[suite], f"frozen {suite} row count differs")
        for row, reference in zip(corpus, grouped[suite]):
            source, target = sources[offset], targets[offset]
            need(all(row.get(name) == reference.get(name) for name in
                     ("id", "input", "context", "expected"))
                 and source["id"] == row.get("id") and source["origin"] == suite
                 and source["input"] == row.get("input") and source["context"] == row.get("context")
                 and target == {"id": source["id"], "role": suite,
                                "expected": row.get("expected")},
                 f"frozen {suite} source/target/reference binding differs")
            group = row.get("group_id")
            expected_group = group if isinstance(group, str) and group else f"{suite}:{row['id']}"
            need(source["group_id"] == expected_group, f"frozen {suite} group binding differs")
            _validate_original_reference_binding(reference, saved_references[offset])
            offset += 1
    need(offset == TOTAL_ROWS and len(targets) == len(saved_references) == TOTAL_ROWS,
         "numeric DEV target rows are incomplete")
    return targets


def _validate_original_reference_binding(original, saved):
    """Bind the saved target-free reference to the accepted DEV reference row."""
    need(isinstance(original, dict) and isinstance(saved, dict),
         "accepted DEV parent reference is malformed")
    projected = {"id": original.get("id"), "suite": original.get("suite"),
        "candidates": original.get("candidates"),
        "sentence_pool": original.get("sentence_pool"), "flags": original.get("flags", {})}
    need(canonical(saved) == canonical(projected),
         "saved parent reference differs from accepted DEV reference")
    return True


def _score(targets, captures):
    suites = {name: {key: ([] if key in ("corrections", "regressions", "top5_gains", "top5_losses") else 0)
                    for key in ("rows", "before_top1", "top1", "before_top5", "top5",
                                "before_pool", "pool", "corrections", "regressions",
                                "top5_gains", "top5_losses")}
              for name in SUITE_COUNTS}
    for target, capture in zip(targets, captures):
        role, expected = target["role"], set(target["expected"])
        parent, child, trace = capture["parent"], capture["child"], capture["trace"]
        before, after = parent["result"] in expected, child["result"] in expected
        before5 = any(row["text"] in expected for row in parent["candidates"])
        after5 = any(row["text"] in expected for row in child["candidates"])
        child_pool = trace["scoring_pool"] if capture["child_sentence_scored"] else child["candidates"]
        values = suites[role]; values["rows"] += 1
        for name, condition in (("before_top1", before), ("top1", after),
                ("before_top5", before5), ("top5", after5),
                ("before_pool", any(row["text"] in expected for row in capture["parent_pool"])),
                ("pool", any(row["text"] in expected for row in child_pool))):
            values[name] += int(condition)
        for condition, name in ((after and not before, "corrections"),
                (before and not after, "regressions"), (after5 and not before5, "top5_gains"),
                (before5 and not after5, "top5_losses")):
            if condition:
                values[name].append(target["id"])
    protected = suites["original_dev"]["top1"] == 140 and suites["selection_dev"]["top1"] == 60
    top5 = all(not row["top5_losses"] for row in suites.values())
    passed = (suites["natural_dev"]["top1"] >= 81 and protected
              and suites["reasoning_dev"]["top1"] > 34 and top5)
    return {"suites": suites, "gate": {"passed": passed,
        "natural_at_least_81": suites["natural_dev"]["top1"] >= 81,
        "original_140_of_140": suites["original_dev"]["top1"] == 140,
        "selection_60_of_60": suites["selection_dev"]["top1"] == 60,
        "reasoning_above_34_of_40": suites["reasoning_dev"]["top1"] > 34,
        "protected_preserved": protected, "top5_preserved": top5,
        "raw_preserved": True, "original_pool_preserved": True,
        "scorer_trace_preserved": True, "final_evaluation_allowed": passed,
        "promotion_allowed": False}}


def verify_evaluation(directory, target_loader=None, authority_loader=None):
    authority = (authority_loader or _validate_authority)()
    (directory, report_path, report_sha, report, sources, captures,
     source_hashes) = _preflight(directory, authority)
    targets = (target_loader or _default_target_loader)(directory, authority, sources)
    need(isinstance(targets, list) and len(targets) == TOTAL_ROWS,
         "numeric DEV target loader result differs")
    expected_roles = [name for name, count in SUITE_COUNTS.items() for _ in range(count)]
    for source, target, role in zip(sources, targets, expected_roles):
        need(isinstance(target, dict) and set(target) == {"id", "role", "expected"}
             and target.get("id") == source["id"] and target.get("role") == role
             and isinstance(target.get("expected"), list) and target["expected"]
             and all(isinstance(value, str) and value for value in target["expected"]),
             "numeric DEV target identity/order or accepted answers differ")
    metrics = _score(targets, captures)
    need(report.get("metrics") == metrics, "declared numeric DEV metrics differ")
    quality = metrics["gate"]["passed"]
    need(report.get("quality_gate_passed") is quality
         and report.get("final_evaluation_allowed") is quality,
         "declared numeric DEV quality decision differs")
    need(file_digest(report_path) == report_sha
         and report.get("source_hashes_after") == source_hashes,
         "numeric DEV report or source chain changed during verification")
    for name, expected in report["files"].items():
        need(file_digest(directory / name) == expected,
             f"closed DEV evidence changed during verification: {name}")
    verify_current_hashes(source_hashes)
    return {"stage": "numeric_composition_development", "verification_passed": True,
        "quality_gate_passed": quality, "final_evaluation_allowed": quality,
        "promotion_allowed": False, "deployed": False, "resource_valid_experiment": True,
        "runtime_version": RUNTIME_VERSION, "model_artifacts": authority["weights"],
        "report": {"path": path_key(report_path), "sha256": report_sha},
        "sources": {"path": path_key(directory / "sources.jsonl"),
                    "sha256": report["files"]["sources.jsonl"], "rows": TOTAL_ROWS},
        "targets": {"path": path_key(directory / "targets.jsonl"),
                    "sha256": report["files"]["targets.jsonl"], "rows": TOTAL_ROWS},
        "captures": {"path": path_key(directory / "captures.jsonl"),
                     "sha256": report["files"]["captures.jsonl"], "rows": TOTAL_ROWS},
        "parent_references": {"path": path_key(directory / "parent_references.jsonl"),
            "sha256": report["files"]["parent_references.jsonl"], "rows": TOTAL_ROWS},
        "source_hashes_sha256": digest(canonical(source_hashes)), "metrics": metrics,
        "labels_read_only_after_complete_capture_preflight": True,
        "development_targets_read": True, "final_fixed_integration_targets_read": False,
        "verifier": {"path": path_key(Path(__file__)), "sha256": file_digest(Path(__file__))}}


def write_proof(path, payload):
    path = Path(path)
    need(not path.exists(), "proof output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"schema": SCHEMA, "payload": payload, "digest": digest(canonical(payload))}
    path.write_bytes(canonical(envelope) + b"\n")
    return envelope


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = verify_evaluation(args.evaluation)
        write_proof(args.output, payload)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        print(json.dumps({"schema": SCHEMA, "verification_passed": False,
                          "error": str(error)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"schema": SCHEMA, "verification_passed": True,
        "quality_gate_passed": payload["quality_gate_passed"],
        "final_evaluation_allowed": payload["final_evaluation_allowed"],
        "promotion_allowed": False, "output": str(args.output)}, sort_keys=True))
    return 0 if payload["quality_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
