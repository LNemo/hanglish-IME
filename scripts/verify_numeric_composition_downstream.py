"""Independently verify the closed numeric-composition downstream evaluation."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import sys


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "numeric-composition-downstream-proof-v1"
REPORT_SCHEMA = "numeric-composition-downstream-v1"
RUNTIME_VERSION = "neural-1-867f13596ed5"
FRESH_ROWS, FRESH_GROUPS, PRESERVATION_ROWS, TOTAL_ROWS = 512, 387, 376, 888
RESAMPLES, SEED, MINIMUM_NET_GAIN = 2000, 20260915, 8
PROFILE = {"cpu_threads": 2, "rss_limit_bytes": 3 * 1024**3,
           "time_limit_seconds": 1800.0, "progress_interval": 32}
HANDOFF = ROOT / ".omx/plans/numeric-composition-downstream-handoff-v1.json"
HANDOFF_SHA256 = "bff235188624cab16ad7f103014c20d8bbc35e83e834da648508a28d44818fee"
FIT_PROOF = ROOT / ".omx/artifacts/numeric-composition-fit-v1-proof-v2.json"
FIT_PROOF_SHA256 = "c0b5c6f59b35b28122f165ed1327f25c07f8cca0ef26197bf473aed52ff3d933"
DATA = ROOT / ".omx/artifacts/long-context-lora-data-v1-review2"
DATA_MANIFEST_SHA256 = "cb4558249aa9acefb246abca9275ba8020f39cee27eb9d4143dfb94f266ecd06"
FRESH_SELECTION_SHA256 = "fb1257833c99acc406a831336f535d9ef3a786fc38bb08a84902598e5fe2287d"
FRESH_RAW = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
FRESH_RAW_SHA256 = "904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b"
PRESERVATION = ROOT / "data/natural_context/final-choice-v1/challenge_preservation"
PRESERVATION_MANIFEST_SHA256 = "c4f587d42ac0838ce7eb2f31881bc182d8f1e0615bab47965f35da032cbaa133"
PRESERVATION_PARTITION_SHA256 = "e997f06558bdb5633446171d35912140fb34b55066c13fa13b668939d6347846"
RUNTIME_REPORT = ROOT / ".omx/artifacts/numeric-structure-integration-v1/report.json"
RUNTIME_REPORT_SHA256 = "77c04334af817e15bc4c5ccc9d27b1d6ab9f9f6a759b4207e752762ae49d3aab"
FRESH_PARTITION = ROOT / ".omx/artifacts/finite-final-choice-partition-v3/manifest.json"
FRESH_PARTITION_SHA256 = "8f699052b3e8d6e07dcaf59dc8696cc13a8ffb15b4e1bb5df62571e906ae3e61"
PRIOR_SELECTIONS = (
    "data/natural_context/final-choice-v1/natural_fit/selection.jsonl",
    "data/natural_context/final-choice-v1/natural_calibration/selection.jsonl",
    "data/natural_context/final-choice-v1/natural_holdout/selection.jsonl",
    "data/natural_context/final-choice-v1/challenge_fit/selection.jsonl",
    "data/natural_context/final-choice-v1/challenge_preservation/selection.jsonl",
    "data/natural_context/balanced-final-choice-eval-v2/selection.jsonl",
    ".omx/artifacts/finite-final-choice-evaluation-v3/selection.jsonl",
)
HISTORICAL_QWEN = (
    ".omx/artifacts/lora-smoke-source/train.jsonl",
    ".omx/artifacts/lora-smoke-source/valid.jsonl",
    ".omx/artifacts/lora-expanded-source/train.jsonl",
    ".omx/artifacts/lora-balanced-source/train.jsonl",
    ".omx/artifacts/lora-balanced-source/valid.jsonl",
)
LIMITATIONS = {
    "group_collision_check": "unavailable: historical messages have no group IDs",
    "id_collision_check": "unavailable: historical messages have no IDs",
    "semantic_family_independence": "unestablished",
    "upstream_pretraining_overlap": "unknown",
}
FLAG_FIELDS = ("engine", "mode", "raw", "version", "ranker", "model_artifacts",
               "context_used", "external_context_used", "ranking_applied",
               "sentence_ranking_applied", "warnings")


class VerificationError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise VerificationError(message)


def canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                          separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise VerificationError("value is not canonical-JSON encodable") from error


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
    except OSError as error:
        raise VerificationError(f"required file is unavailable: {path}") from error
    return value.hexdigest()


def read_json(path):
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSON: {path}") from error
    need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def read_jsonl(path):
    try:
        rows = [json.loads(line) for line in Path(path).read_bytes().splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSONL: {path}") from error
    need(all(isinstance(row, dict) for row in rows), f"JSONL objects required: {path}")
    return rows


def path_key(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def valid_hash_map(value):
    return (isinstance(value, dict) and value
            and all(isinstance(name, str) and name and isinstance(sha, str)
                    and len(sha) == 64 and not set(sha) - set("0123456789abcdef")
                    for name, sha in value.items()))


def merge_hashes(target, values):
    need(isinstance(values, dict), "source hash inventory is malformed")
    for name, value in values.items():
        need(isinstance(name, str) and name and isinstance(value, str),
             "source hash inventory is malformed")
        need(name not in target or target[name] == value, f"source hash conflict: {name}")
        target[name] = value


def verify_current_hashes(values):
    need(valid_hash_map(values), "source hash inventory is malformed")
    for name, expected in values.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        need(file_digest(path) == expected, f"source changed: {name}")


def _validate_authority():
    """Validate all target-independent authority before evaluation labels are parsed."""
    need(file_digest(HANDOFF) == HANDOFF_SHA256, "downstream handoff changed")
    handoff = read_json(HANDOFF)
    need(handoff.get("schema") == "numeric-composition-downstream-handoff-v1"
         and handoff.get("terminal_planning_state") == "complete"
         and handoff.get("review") == "CLEAR"
         and handoff.get("model_concurrency_allowed") is False
         and handoff.get("promotion_allowed") is False
         and handoff.get("goal_complete") is False,
         "downstream handoff is not executable")
    for name, expected in handoff.get("files", {}).items():
        need(file_digest(ROOT / name) == expected, f"handoff-bound file changed: {name}")

    need(file_digest(FIT_PROOF) == FIT_PROOF_SHA256, "FIT proof-v2 changed")
    fit_envelope = read_json(FIT_PROOF)
    fit = fit_envelope.get("payload")
    need(fit_envelope.get("schema") == "numeric-composition-fit-proof-v2"
         and isinstance(fit, dict) and fit_envelope.get("digest") == digest(canonical(fit))
         and fit.get("verification_passed") is True
         and fit.get("quality_gate_passed") is True
         and fit.get("downstream_evaluation_allowed") is True
         and fit.get("promotion_allowed") is False and fit.get("deployed") is False,
         "FIT proof-v2 semantics differ")
    fit_report_path = ROOT / fit.get("report", {}).get("path", "")
    fit_captures_path = ROOT / fit.get("captures", {}).get("path", "")
    fit_sources_path = ROOT / fit.get("sources", {}).get("path", "")
    fit_verifier_path = ROOT / fit.get("verifier", {}).get("path", "")
    need(file_digest(fit_report_path) == fit.get("report", {}).get("sha256")
         and file_digest(fit_captures_path) == fit.get("captures", {}).get("sha256")
         and file_digest(fit_sources_path) == fit.get("sources", {}).get("sha256")
         and file_digest(fit_verifier_path) == fit.get("verifier", {}).get("sha256"),
         "FIT proof-v2 bound files differ")
    fit_report = read_json(fit_report_path)
    need(valid_hash_map(fit_report.get("source_hashes_before"))
         and fit_report["source_hashes_before"] == fit_report.get("source_hashes_after")
         and digest(canonical(fit_report["source_hashes_before"]))
             == fit.get("source_hashes_sha256"), "FIT source hash chain differs")

    manifest_path, selection_path = DATA / "manifest.json", DATA / "fresh_selection.jsonl"
    need(file_digest(manifest_path) == DATA_MANIFEST_SHA256
         and file_digest(selection_path) == FRESH_SELECTION_SHA256,
         "frozen fresh partition changed")
    manifest, selection = read_json(manifest_path), read_jsonl(selection_path)
    fresh = manifest.get("fresh", {})
    need(manifest.get("historical_qwen_limitations") == LIMITATIONS
         and manifest.get("file_hashes", {}).get("fresh_selection.jsonl")
             == FRESH_SELECTION_SHA256
         and fresh.get("rows") == FRESH_ROWS and fresh.get("groups") == FRESH_GROUPS
         and fresh.get("whole_group_selection") is True
         and fresh.get("target_fields_copied") is False
         and fresh.get("selection_fields") == ["id", "group_id"]
         and len(fresh.get("group_ids", [])) == FRESH_GROUPS
         and fresh.get("excluded_groups") == {},
         "frozen fresh partition contract differs")
    need(len(selection) == FRESH_ROWS
         and all(set(row) == {"id", "group_id"} for row in selection)
         and len({row["id"] for row in selection}) == FRESH_ROWS
         and len({row["group_id"] for row in selection}) == FRESH_GROUPS
         and list(dict.fromkeys(row["group_id"] for row in selection)) == fresh["group_ids"],
         "frozen fresh identities/groups differ")

    preservation_manifest_path = PRESERVATION / "manifest.json"
    need(file_digest(preservation_manifest_path) == PRESERVATION_MANIFEST_SHA256,
         "preservation manifest changed")
    preservation = read_json(preservation_manifest_path)
    need(preservation.get("completed") is True and preservation.get("rows") == PRESERVATION_ROWS
         and preservation.get("runtime_version") == RUNTIME_VERSION
         and preservation.get("role") == "challenge_preservation"
         and preservation.get("scope") == "fresh actual parent TRAIN capture for final-choice learning"
         and preservation.get("expected_rows") == PRESERVATION_ROWS
         and preservation.get("partition_manifest_sha256") == PRESERVATION_PARTITION_SHA256
         and len(preservation.get("partition_ordered_ids", [])) == PRESERVATION_ROWS
         and len(set(preservation["partition_ordered_ids"])) == PRESERVATION_ROWS
         and preservation.get("selection_sha256") == file_digest(PRESERVATION / "selection.jsonl")
         and preservation.get("selection_sha256_before_inference")
             == preservation.get("selection_sha256")
         and preservation.get("cases_sha256") == file_digest(PRESERVATION / "cases.jsonl")
         and valid_hash_map(preservation.get("source_hashes"))
         and preservation["source_hashes"] == preservation.get("source_hashes_after")
         and preservation.get("training_run") is False
         and preservation.get("new_selector_inference_run") is False
         and preservation.get("development_data_read") is False
         and preservation.get("final_data_read") is False
         and preservation.get("deployed") is False,
         "preservation manifest bindings differ")
    need(file_digest(RUNTIME_REPORT) == RUNTIME_REPORT_SHA256,
         "production runtime report changed")
    runtime = read_json(RUNTIME_REPORT)
    model_artifacts = runtime.get("accepted_weight_hashes")
    need(runtime.get("integrated_runtime_version") == RUNTIME_VERSION
         and valid_hash_map(model_artifacts)
         and model_artifacts == preservation.get("model_artifact_hashes"),
         "production runtime/model identity differs")

    expected = {}
    for values in (manifest.get("source_hashes"), fit_report["source_hashes_before"],
                   runtime.get("integrated_runtime_source_hashes"), model_artifacts,
                   preservation["source_hashes"]):
        merge_hashes(expected, values)
    fixed = [FIT_PROOF, HANDOFF, manifest_path, selection_path, RUNTIME_REPORT, FRESH_RAW,
             preservation_manifest_path, PRESERVATION / "cases.jsonl",
             PRESERVATION / "selection.jsonl", ROOT / "scripts/numeric_composition_runtime.py",
             ROOT / "hanglish/core/numeric_composition.py",
             ROOT / "scripts/prepare_long_context_lora_data.py",
             ROOT / "scripts/prepare_long_context_lora_fresh_capture.py",
             ROOT / "scripts/evaluate_final_choice_holdout.py",
             ROOT / "scripts/final_choice_capture.py",
             ROOT / "scripts/audit_final_choice_candidate_stages.py",
             ROOT / "scripts/long_context_lora_fresh_metrics.py",
             ROOT / "scripts/prepare_final_choice_data.py",
             ROOT / "hanglish/core/neural_converter.py", ROOT / "hanglish/core/phonology.py",
             ROOT / "scripts/evaluate_numeric_composition_downstream.py"]
    fixed.extend(ROOT / name for name in handoff["files"])
    for path in fixed:
        merge_hashes(expected, {path_key(path): file_digest(path)})
    need(expected.get(path_key(FRESH_RAW)) == FRESH_RAW_SHA256,
         "fresh raw source binding differs")
    verify_current_hashes(expected)
    return {"selection": selection, "fresh_manifest": fresh,
            "preservation_manifest": preservation, "model_artifacts": model_artifacts,
            "source_hashes": dict(sorted(expected.items())), "fit_proof_sha256": FIT_PROOF_SHA256}


def _validate_segments(candidate, source, *, rounded=False):
    need(isinstance(candidate, dict) and isinstance(candidate.get("text"), str),
         "candidate object is malformed")
    score = candidate.get("score")
    need(score is None or (type(score) in (int, float) and math.isfinite(score)
                           and (not rounded or round(score, 4) == score)),
         "candidate score is malformed or nonfinite")
    segments = candidate.get("segments")
    need(isinstance(segments, list) and segments, "candidate segments are missing")
    cursor, text = 0, []
    for segment in segments:
        need(isinstance(segment, dict), "candidate segment is malformed")
        start, end = segment.get("input_start"), segment.get("input_end")
        need(type(start) is int and type(end) is int and start == cursor and start < end <= len(source)
             and segment.get("source") == source[start:end]
             and isinstance(segment.get("text"), str)
             and isinstance(segment.get("kind"), str) and segment["kind"],
             "candidate segment does not align to source")
        if segment["kind"] == "raw":
            need(segment["text"] == segment["source"], "raw segment changed source bytes")
        cursor, text = end, text + [segment["text"]]
    need(cursor == len(source) and candidate["text"] == "".join(text),
         "candidate text differs from complete segments")


def _validate_response(response, source, context, *, raw, model_artifacts):
    need(isinstance(response, dict) and response.get("input") == source
         and response.get("context") == context and response.get("engine") == "neural"
         and response.get("raw") is raw and response.get("version") == RUNTIME_VERSION,
         "response identity/runtime differs")
    candidates = response.get("candidates")
    need(isinstance(candidates, list) and 1 <= len(candidates) <= 5
         and response.get("result") == candidates[0].get("text"), "response top-five differs")
    for candidate in candidates:
        _validate_segments(candidate, source, rounded=True)
        if not raw and candidate.get("score") is None:
            need(candidate == {"text": source, "score": None, "segments": [{
                "input_start": 0, "input_end": len(source), "source": source,
                "text": source, "kind": "raw"}]},
                "None score is reserved for the complete original raw candidate")
    need(len({row["text"] for row in candidates}) == len(candidates),
         "response contains duplicate candidates")
    if raw:
        expected = {"text": source, "score": 0.0, "segments": [{"input_start": 0,
            "input_end": len(source), "source": source, "text": source, "kind": "raw"}]}
        need(candidates == [expected] and response.get("result") == source
             and response.get("model_artifacts") == {} and response.get("ranker") is None,
             "raw bypass response differs")
    else:
        need(response.get("model_artifacts") == model_artifacts,
             "response model artifacts differ")


def _validate_path(value, source):
    need(isinstance(value, dict) and set(value) == {"text", "score", "segments"},
         "instrumented path object differs")
    _validate_segments(value, source)


def _preflight(directory, authority):
    """Validate closed source-only evidence; do not parse targets here."""
    directory = Path(directory).resolve()
    report_path = directory / "report.json"
    report_sha, report = file_digest(report_path), read_json(report_path)
    need(report.get("schema") == REPORT_SCHEMA and report.get("completed") is True
         and report.get("terminal_status") == "success" and report.get("deployed") is False
         and report.get("promotion_allowed") is False and report.get("captures_closed") is True
         and report.get("models_released_before_target_join") is True
         and report.get("targets_joined") is True and report.get("model_loaded") is False
         and report.get("captured_rows") == TOTAL_ROWS and report.get("rows") == TOTAL_ROWS
         and report.get("fresh_rows") == FRESH_ROWS and report.get("fresh_groups") == FRESH_GROUPS
         and report.get("preservation_rows") == PRESERVATION_ROWS
         and report.get("runtime_version") == RUNTIME_VERSION
         and report.get("model_artifacts") == authority["model_artifacts"]
         and report.get("historical_qwen_limitations") == LIMITATIONS
         and report.get("profile") == PROFILE,
         "downstream report is incomplete or has the wrong scope")
    need(report.get("target_access") == {"fresh_target_bytes_accessed": True,
                                         "preservation_target_bytes_accessed": True},
         "target access closure differs")
    files = report.get("files")
    need(isinstance(files, dict) and set(files) == {"sources.jsonl", "targets.jsonl",
         "preservation_baselines.jsonl", "captures.jsonl"}, "report file inventory differs")
    for name, expected in files.items():
        need(file_digest(directory / name) == expected, f"closed file hash differs: {name}")
    expected_hashes = authority["source_hashes"]
    need(len(expected_hashes) >= 126 and report.get("source_hashes_before") == expected_hashes
         and report.get("source_hashes_after") == expected_hashes,
         "downstream source hash inventory differs")
    verify_current_hashes(expected_hashes)

    sources = read_jsonl(directory / "sources.jsonl")
    captures = read_jsonl(directory / "captures.jsonl")
    baselines = read_jsonl(directory / "preservation_baselines.jsonl")
    need(len(sources) == len(captures) == TOTAL_ROWS and len(baselines) == PRESERVATION_ROWS,
         "exact 888 captures and 376 baselines are required")
    ids, fresh_groups = set(), set()
    baseline_by_id = {row.get("id"): row for row in baselines}
    need(len(baseline_by_id) == PRESERVATION_ROWS and None not in baseline_by_id,
         "preservation baseline identities differ")
    for index, (source, capture) in enumerate(zip(sources, captures)):
        need(set(source) == {"id", "group_id", "origin", "input", "context"}
             and all(isinstance(source.get(name), str) for name in source)
             and source["id"] and source["group_id"] and source["input"]
             and source["id"] not in ids, "source-only row is malformed")
        ids.add(source["id"])
        if index < FRESH_ROWS:
            need(source["origin"] == "fresh_natural"
                 and source["id"] == authority["selection"][index]["id"]
                 and source["group_id"] == authority["selection"][index]["group_id"],
                 "fresh selection identity/order differs")
            fresh_groups.add(source["group_id"])
        else:
            expected_id = authority["preservation_manifest"]["partition_ordered_ids"][index-FRESH_ROWS]
            need(source["id"] == expected_id, "preservation identity/order differs")
        need(all(capture.get(name) == source[name] for name in source),
             "capture source identity differs")
        for name in ("parent", "child", "parent_raw", "child_raw"):
            _validate_response(capture.get(name), source["input"], source["context"],
                               raw=name.endswith("raw"), model_artifacts=authority["model_artifacts"])
        need(canonical(capture["parent_raw"]) == canonical(capture["child_raw"]),
             "raw response changed")
        for timing in ("parent_ms", "child_ms"):
            need(type(capture.get(timing)) in (int, float) and math.isfinite(capture[timing])
                 and capture[timing] >= 0, "capture timing is malformed")
        for flag in ("parent_sentence_scored", "child_sentence_scored"):
            need(type(capture.get(flag)) is bool, "sentence scoring flag is malformed")
        need(capture["parent_sentence_scored"]
                is capture["parent"].get("sentence_ranking_applied")
             and capture["child_sentence_scored"]
                is capture["child"].get("sentence_ranking_applied"),
             "capture sentence scoring flags differ from responses")
        for name in ("parent_pool", "parent_base"):
            need(isinstance(capture.get(name), list) and capture[name], f"path pool missing: {name}")
            for path in capture[name]:
                _validate_path(path, source["input"])
        trace = capture.get("trace")
        list_fields = ("base", "accepted_old_extras", "preserved_pool",
                       "production_scoring_pool", "generated_new_extras",
                       "accepted_new_extras", "scoring_pool")
        need(isinstance(trace, dict) and all(isinstance(trace.get(name), list) for name in list_fields)
             and type(trace.get("eligible_piece_count")) is int
             and trace["eligible_piece_count"] >= 0
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
        for name in list_fields:
            for path in trace[name]:
                _validate_path(path, source["input"])
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
            need(trace["production_scoring_pool"] == [],
                 "unexpected production scoring pool")
        if capture["child_sentence_scored"]:
            need(capture.get("child_score_inputs") == [row["text"] for row in trace["scoring_pool"]],
                 "actual scorer inputs differ from trace")
        else:
            need(capture.get("child_score_inputs") is None, "unexpected child scorer inputs")
        if index >= FRESH_ROWS:
            baseline = baseline_by_id.get(source["id"])
            need(isinstance(baseline, dict) and baseline.get("runtime_version") == RUNTIME_VERSION
                 and baseline.get("result") == capture["parent"]["result"]
                 and canonical(baseline.get("candidates")) == canonical(capture["parent"]["candidates"])
                 and canonical(baseline.get("response_flags"))
                    == canonical({name: capture["parent"].get(name) for name in FLAG_FIELDS})
                 and canonical(baseline.get("raw_bypass_response"))
                    == canonical(capture["parent_raw"]),
                 "protected production baseline changed")
    need(len(fresh_groups) == FRESH_GROUPS, "fresh document group count differs")

    observations = report.get("resource_observations")
    need(isinstance(observations, list) and len(observations) == TOTAL_ROWS + 1,
         "resource observation count differs")
    prior_elapsed = -1.0
    for index, observation in enumerate(observations):
        expected_stage = f"capture_{index+1}" if index < TOTAL_ROWS else "after_model_release"
        need(isinstance(observation, dict), "resource observation differs")
        elapsed, rss = observation.get("elapsed_seconds"), observation.get("rss_peak_bytes")
        need(observation.get("stage") == expected_stage
             and type(elapsed) in (int, float) and math.isfinite(elapsed)
             and prior_elapsed <= elapsed <= PROFILE["time_limit_seconds"]
             and type(rss) is int and 0 <= rss <= PROFILE["rss_limit_bytes"],
             "resource observation differs")
        prior_elapsed = elapsed
    need(report.get("terminal_resources") == observations[-1],
         "terminal resource observation differs")
    elapsed = report.get("elapsed_seconds")
    need(type(elapsed) in (int, float) and math.isfinite(elapsed)
         and prior_elapsed <= elapsed <= PROFILE["time_limit_seconds"],
         "report elapsed time differs")
    created_at = report.get("created_at")
    need(isinstance(created_at, str), "report creation timestamp is malformed")
    try:
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise VerificationError("report creation timestamp is malformed") from error
    latency = report.get("latency", {})
    warm_parent = sorted(row["parent_ms"] for row in captures[1:])
    warm_child = sorted(row["child_ms"] for row in captures[1:])
    percentile = math.ceil(.95 * len(warm_child)) - 1
    need(latency == {"first_parent_ms": captures[0]["parent_ms"],
                     "first_child_ms": captures[0]["child_ms"],
                     "warm_parent_p95_ms": warm_parent[percentile],
                     "warm_child_p95_ms": warm_child[percentile],
                     "scope": "engine calls after model load; not browser or cold model loading"},
         "reported engine latency differs")
    need(file_digest(report_path) == report_sha, "report changed during preflight")
    return directory, report_path, report_sha, report, sources, captures, expected_hashes


def _default_target_loader(directory, authority, sources):
    """Open labels only after preflight and bind them to the original frozen sources."""
    targets = read_jsonl(directory / "targets.jsonl")
    need(len(targets) == TOTAL_ROWS, "downstream target rows are incomplete")
    raw_rows = read_jsonl(FRESH_RAW)
    by_id = {row.get("id"): row for row in raw_rows}
    need(len(by_id) == len(raw_rows) and None not in by_id, "fresh raw identities differ")
    from hanglish.core.phonology import romanize_word
    _replay_fresh_partition(raw_rows, authority, romanize_word)
    preservation_rows = read_jsonl(PRESERVATION / "selection.jsonl")
    preservation_cases = read_jsonl(PRESERVATION / "cases.jsonl")
    downstream_baselines = {row.get("id"): row for row in
                            read_jsonl(directory / "preservation_baselines.jsonl")}
    preservation_by_id = {row.get("id"): row for row in preservation_rows}
    need(len(preservation_rows) == PRESERVATION_ROWS
         and len(preservation_by_id) == PRESERVATION_ROWS
         and [row.get("id") for row in preservation_rows]
            == authority["preservation_manifest"]["partition_ordered_ids"]
         and [row.get("id") for row in preservation_cases]
            == authority["preservation_manifest"]["partition_ordered_ids"],
         "protected preservation rows differ")
    cases_by_id = {row["id"]: row for row in preservation_cases}
    for index, (source, target) in enumerate(zip(sources, targets)):
        role = "fresh_natural" if index < FRESH_ROWS else "challenge_preservation"
        need(set(target) == {"id", "group_id", "role", "expected"}
             and target.get("id") == source["id"] and target.get("group_id") == source["group_id"]
             and target.get("role") == role and isinstance(target.get("expected"), list)
             and target["expected"] and all(isinstance(value, str) and value for value in target["expected"]),
             "target identity/order or accepted answers differ")
        original = by_id.get(source["id"]) if index < FRESH_ROWS else preservation_by_id.get(source["id"])
        need(isinstance(original, dict) and original.get("group_id") == source["group_id"],
             "original target source identity differs")
        if index < FRESH_ROWS:
            need(original.get("split") == "train" and isinstance(original.get("text"), str)
                 and romanize_word(original["text"]) == source["input"]
                 and original.get("left_context") == source["context"]
                 and target["expected"] == [original["text"]],
                 "fresh source/target replay differs")
        else:
            need(all(original.get(name) == source[name]
                     for name in ("id", "group_id", "origin", "input", "context"))
                 and target["expected"] == original.get("expected"),
                 "preservation source/target replay differs")
            case = cases_by_id[source["id"]]
            baseline = downstream_baselines.get(source["id"])
            need(all(case.get(name) == source[name]
                     for name in ("id", "group_id", "origin", "input", "context"))
                 and case.get("expected") == target["expected"]
                 and case.get("runtime_version") == RUNTIME_VERSION
                 and isinstance(baseline, dict)
                 and case.get("result") == baseline.get("result")
                 and canonical(case.get("candidates")) == canonical(baseline.get("candidates"))
                 and canonical(case.get("response_flags"))
                    == canonical(baseline.get("response_flags"))
                 and _validate_original_raw_binding(
                    case.get("raw_bypass_response"), baseline.get("raw_bypass_response")),
                 "preservation frozen case binding differs")
    return targets


def _validate_original_raw_binding(original, closed):
    """Validate the one intentional format difference in protected raw evidence."""
    need(isinstance(original, dict) and isinstance(closed, dict),
         "preservation raw evidence is malformed")
    engine_ms = original.get("engine_ms")
    need(type(engine_ms) in (int, float) and math.isfinite(engine_ms) and engine_ms >= 0,
         "preservation original raw timing is malformed")
    without_timing = {name: value for name, value in original.items() if name != "engine_ms"}
    need(canonical(without_timing) == canonical(closed),
         "preservation raw evidence differs beyond engine_ms")
    return True


def _parse_historical_prompt(content):
    prefix, middle, marker = "로마자 입력: ", "\n문맥: ", "\n후보:"
    need(isinstance(content, str) and content.startswith(prefix)
         and middle in content and marker in content, "historical prompt is malformed")
    source, rest = content[len(prefix):].split(middle, 1)
    context, _ = rest.split(marker, 1)
    return source, "" if context == "(없음)" else context


def _replay_fresh_partition(raw_rows, authority, romanizer):
    """Independently replay reserve, collision exclusions, and whole-group selection."""
    need(file_digest(FRESH_PARTITION) == FRESH_PARTITION_SHA256,
         "fresh partition manifest changed")
    partition = read_json(FRESH_PARTITION)
    remaining_ids, remaining_groups = partition.get("remaining_ids"), partition.get("remaining_group_ids")
    need(isinstance(remaining_ids, list) and len(remaining_ids) == 2077
         and len(set(remaining_ids)) == 2077
         and isinstance(remaining_groups, list) and len(remaining_groups) == 1607
         and len(set(remaining_groups)) == 1607 and len(raw_rows) == 9002,
         "fresh reserve partition differs")
    collision_ids, collision_groups, collision_pairs = set(), set(), set()
    for name in PRIOR_SELECTIONS:
        for row in read_jsonl(ROOT / name):
            need(all(isinstance(row.get(key), str) and row[key]
                     for key in ("id", "group_id", "input"))
                 and isinstance(row.get("context"), str), "prior selection source fields differ")
            collision_ids.add(row["id"]); collision_groups.add(row["group_id"])
            collision_pairs.add((row["input"], row["context"]))
    counts = []
    for name in HISTORICAL_QWEN:
        rows = read_jsonl(ROOT / name); counts.append(len(rows))
        for row in rows:
            messages = row.get("messages")
            need(isinstance(messages, list) and len(messages) == 3
                 and isinstance(messages[1], dict), "historical Qwen messages differ")
            collision_pairs.add(_parse_historical_prompt(messages[1].get("content")))
    need(counts == [58, 8, 298, 588, 140], "historical Qwen row counts differ")

    remaining_id_set, remaining_group_set = set(remaining_ids), set(remaining_groups)
    by_group, seen = defaultdict(list), set()
    for row in raw_rows:
        identity, group = row.get("id"), row.get("group_id")
        need(isinstance(identity, str) and identity and identity not in seen
             and isinstance(group, str) and group, "fresh raw identity/group differs")
        seen.add(identity)
        need((identity in remaining_id_set) == (group in remaining_group_set),
             "fresh reserve contains a partial group")
        if identity in remaining_id_set:
            text, context = row.get("text"), row.get("left_context")
            need(isinstance(text, str) and text and isinstance(context, str),
                 "fresh reserve source fields differ")
            by_group[group].append({"id": identity, "group_id": group,
                                    "input": romanizer(text), "context": context})
    need(remaining_id_set <= seen and set(by_group) == remaining_group_set,
         "fresh reserve join is incomplete")
    excluded = {}
    for group, rows in by_group.items():
        reasons = set()
        if any(row["id"] in collision_ids for row in rows):
            reasons.add("id")
        if group in collision_groups:
            reasons.add("known_group")
        if any((row["input"], row["context"]) in collision_pairs for row in rows):
            reasons.add("exact_input_context")
        if reasons:
            excluded[group] = sorted(reasons)
    admissible = set(by_group) - set(excluded)
    ordered = sorted(admissible, key=lambda group: (
        digest(("long-context-lora-eval-v1-2026-09-15\0" + group).encode()), group))
    selected_groups, selected = [], []
    for group in ordered:
        if len(selected) >= FRESH_ROWS:
            break
        selected_groups.append(group); selected.extend(by_group[group])
    declared = authority["fresh_manifest"]
    need(excluded == declared.get("excluded_groups")
         and len(admissible) == declared.get("admissible_groups") == 1607
         and sum(len(by_group[group]) for group in admissible)
            == declared.get("admissible_rows") == 2077
         and selected_groups == declared.get("group_ids")
         and [{"id": row["id"], "group_id": row["group_id"]} for row in selected]
            == authority["selection"], "fresh collision/selection replay differs")


def _bootstrap(outcomes):
    groups, ids = defaultdict(lambda: [0, 0]), set()
    corrections = regressions = 0
    for row in outcomes:
        identity, group = row["id"], row["group_id"]
        before, after = row["before_correct"], row["after_correct"]
        need(identity not in ids and isinstance(group, str) and group
             and type(before) is bool and type(after) is bool,
             "bootstrap outcomes are malformed")
        ids.add(identity); groups[group][0] += int(after) - int(before); groups[group][1] += 1
        corrections += int(after and not before); regressions += int(before and not after)
    ordered = [groups[name] for name in sorted(groups)]
    rng, deltas = random.Random(SEED), []
    for _ in range(RESAMPLES):
        sample = rng.choices(ordered, k=len(ordered))
        deltas.append(sum(item[0] for item in sample) / sum(item[1] for item in sample))
    deltas.sort(); lower_index = math.floor(.025 * (RESAMPLES - 1))
    gain = corrections - regressions
    return {"rows": len(ids), "groups": len(groups), "corrections": corrections,
            "regressions": regressions, "net_gain": gain, "accuracy_delta": gain / len(ids),
            "resamples": RESAMPLES, "seed": SEED, "lower_index": lower_index,
            "lower_2_5_percentile": deltas[lower_index],
            "gain_and_interval_passed": gain >= MINIMUM_NET_GAIN and deltas[lower_index] > 0}


def _score(targets, captures):
    roles = {name: {key: ([] if key in ("corrections", "regressions", "top5_gains", "top5_losses") else 0)
                    for key in ("rows", "before_top1", "top1", "before_top5", "top5",
                                "before_pool", "pool", "corrections", "regressions",
                                "top5_gains", "top5_losses")}
             for name in ("fresh_natural", "challenge_preservation")}
    outcomes = []
    for target, capture in zip(targets, captures):
        role, expected = target["role"], set(target["expected"])
        parent, child, trace = capture["parent"], capture["child"], capture["trace"]
        before, after = parent["result"] in expected, child["result"] in expected
        before5 = any(row["text"] in expected for row in parent["candidates"])
        after5 = any(row["text"] in expected for row in child["candidates"])
        child_pool = trace["scoring_pool"] if capture["child_sentence_scored"] else child["candidates"]
        values = roles[role]; values["rows"] += 1
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
        if role == "fresh_natural":
            outcomes.append({"id": target["id"], "group_id": target["group_id"],
                             "before_correct": before, "after_correct": after})
    bootstrap = _bootstrap(outcomes)
    fresh, protected = roles["fresh_natural"], roles["challenge_preservation"]
    invariants = not fresh["top5_losses"] and not protected["top5_losses"]
    all_protected = protected["before_top1"] == protected["top1"] == PRESERVATION_ROWS
    passed = (fresh["rows"] == FRESH_ROWS and protected["rows"] == PRESERVATION_ROWS
              and bootstrap["rows"] == FRESH_ROWS and bootstrap["groups"] == FRESH_GROUPS
              and bootstrap["gain_and_interval_passed"] and invariants and all_protected)
    return {"roles": roles, "bootstrap": bootstrap,
            "gate": {"passed": passed, "invariants_preserved": invariants,
                     "raw_preserved": True, "original_pool_preserved": True,
                     "scorer_trace_preserved": True,
                     "preservation_all_correct": all_protected,
                     "downstream_evaluation_allowed": passed, "promotion_allowed": False}}


def verify_evaluation(directory, target_loader=None, authority_loader=None):
    authority = (authority_loader or _validate_authority)()
    (directory, report_path, report_sha, report, sources, captures,
     source_hashes) = _preflight(directory, authority)
    targets = (target_loader or _default_target_loader)(directory, authority, sources)
    need(isinstance(targets, list) and len(targets) == TOTAL_ROWS,
         "target loader result is malformed")
    for index, (source, target) in enumerate(zip(sources, targets)):
        expected_role = "fresh_natural" if index < FRESH_ROWS else "challenge_preservation"
        need(isinstance(target, dict)
             and set(target) == {"id", "group_id", "role", "expected"}
             and target.get("id") == source["id"]
             and target.get("group_id") == source["group_id"]
             and target.get("role") == expected_role
             and isinstance(target.get("expected"), list) and target["expected"]
             and all(isinstance(value, str) and value for value in target["expected"]),
             "target loader identity/order or accepted answers differ")
    metrics = _score(targets, captures)
    need(report.get("metrics") == metrics, "declared downstream metrics differ")
    quality = metrics["gate"]["passed"]
    need(report.get("quality_gate_passed") is quality
         and report.get("downstream_evaluation_allowed") is quality,
         "declared downstream quality decision differs")
    need(file_digest(report_path) == report_sha
         and report.get("source_hashes_after") == source_hashes,
         "report or source chain changed during verification")
    for name, expected in report["files"].items():
        need(file_digest(directory / name) == expected,
             f"closed evidence changed during verification: {name}")
    verify_current_hashes(source_hashes)
    return {"stage": "numeric_composition_downstream", "verification_passed": True,
            "quality_gate_passed": quality, "downstream_evaluation_allowed": quality,
            "promotion_allowed": False, "deployed": False,
            "resource_valid_experiment": True, "runtime_version": RUNTIME_VERSION,
            "model_artifacts": authority["model_artifacts"],
            "report": {"path": path_key(report_path), "sha256": report_sha},
            "captures": {"path": path_key(directory / "captures.jsonl"),
                         "sha256": report["files"]["captures.jsonl"], "rows": TOTAL_ROWS},
            "sources": {"path": path_key(directory / "sources.jsonl"),
                        "sha256": report["files"]["sources.jsonl"], "rows": TOTAL_ROWS},
            "targets": {"path": path_key(directory / "targets.jsonl"),
                        "sha256": report["files"]["targets.jsonl"], "rows": TOTAL_ROWS},
            "preservation_baselines": {
                "path": path_key(directory / "preservation_baselines.jsonl"),
                "sha256": report["files"]["preservation_baselines.jsonl"],
                "rows": PRESERVATION_ROWS},
            "source_hashes_sha256": digest(canonical(source_hashes)), "metrics": metrics,
            "labels_read_only_after_complete_capture_preflight": True,
            "fresh_targets_read": True,
            "development_final_targets_read": False,
            "verifier": {"path": path_key(Path(__file__)),
                         "sha256": file_digest(Path(__file__))}}


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
                      "downstream_evaluation_allowed": payload["downstream_evaluation_allowed"],
                      "promotion_allowed": False, "output": str(args.output)}, sort_keys=True))
    return 0 if payload["quality_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
