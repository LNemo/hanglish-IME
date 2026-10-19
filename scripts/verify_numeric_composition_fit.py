"""Independently verify the isolated numeric-composition FIT experiment."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "numeric-composition-fit-proof-v2"
REPORT_SCHEMA = "numeric-composition-fit-v1"
ROWS = 512
RUNTIME_VERSION = "neural-1-867f13596ed5"
AUDIT = ROOT / ".omx/artifacts/final-choice-candidate-stages-v1"
AUDIT_REPORT_SHA256 = "528f4fbc1ebb2bbc9e22e18b85b4af6e0a205a9a5fe1a64bbe266b52ba46b615"
AUDIT_CASES_SHA256 = "860da6746310eee896008280a8ad6eac72736b35db16a289aa02a9a5f45264c6"
FIT_MANIFEST_SHA256 = "fc22b1635cc496ed70aa757750944726d583d640f2d9f3a598e380f3e262a1c9"
PLAN_FILES = {
    ".omx/plans/numeric-composition-successor-v1.md":
        "ab841eacba7df4beaa5f67f030010e8a45c81556c1f77de8c23b7c9a88a7b3b1",
    ".omx/plans/numeric-composition-successor-test-spec-v1.md":
        "1e07201fb73997017b5aea063fa370af9c63df0e27330d559b4ad3b3da692ddc",
    ".omx/plans/numeric-composition-successor-handoff-v1.json":
        "9caccf1697a078eb309136e35d7bafff66dff5cdf3c3c28e700b3964ea6d02bc",
}
RUNNER_FILES = (
    "scripts/numeric_composition_runtime.py",
    "hanglish/core/numeric_composition.py",
    "scripts/evaluate_numeric_composition_fit.py",
    ".omx/plans/numeric-composition-successor-handoff-v1.json",
)
FLAG_FIELDS = (
    "engine", "mode", "raw", "version", "ranker", "model_artifacts",
    "context_used", "external_context_used", "ranking_applied",
    "sentence_ranking_applied", "warnings",
)


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
        rows = [json.loads(line) for line in Path(path).read_bytes().splitlines()]
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


def _valid_hash_map(value):
    return (isinstance(value, dict) and value
            and all(isinstance(name, str) and name
                    and isinstance(sha, str) and len(sha) == 64
                    and all(char in "0123456789abcdef" for char in sha)
                    for name, sha in value.items()))


def _validate_authority():
    for name, expected in PLAN_FILES.items():
        need(file_digest(ROOT / name) == expected, f"approved authority changed: {name}")
    handoff = read_json(ROOT / ".omx/plans/numeric-composition-successor-handoff-v1.json")
    need(handoff.get("schema") == "numeric-composition-successor-handoff-v1"
         and handoff.get("terminal_planning_state") == "complete"
         and handoff.get("authorized") is True
         and handoff.get("architect") == "APPROVE"
         and handoff.get("critic") == "APPROVE"
         and handoff.get("sequential_reviews") is True
         and handoff.get("goal_complete") is False,
         "approved numeric-composition handoff differs")
    for name, expected in handoff.get("files", {}).items():
        need(file_digest(ROOT / name) == expected, f"handoff-bound file changed: {name}")


def _expected_source_hashes():
    need(file_digest(AUDIT / "report.json") == AUDIT_REPORT_SHA256
         and file_digest(AUDIT / "cases.jsonl") == AUDIT_CASES_SHA256,
         "frozen candidate-stage audit changed")
    prior = read_json(AUDIT / "report.json")
    hashes = prior.get("source_hashes")
    need(_valid_hash_map(hashes) and hashes == prior.get("source_hashes_after"),
         "candidate-stage source chain differs")
    expected = dict(hashes)
    for name in RUNNER_FILES:
        expected[name] = file_digest(ROOT / name)
    return dict(sorted(expected.items()))


def _validate_segments(candidate, source, *, rounded_score=False):
    need(isinstance(candidate, dict) and isinstance(candidate.get("text"), str),
         "candidate object is malformed")
    score = candidate.get("score")
    need(score is None or (type(score) in (int, float) and math.isfinite(score)
                           and (not rounded_score or round(score, 4) == score)),
         "candidate score is malformed or nonfinite")
    segments = candidate.get("segments")
    need(isinstance(segments, list) and segments, "candidate segments are missing")
    cursor, output = 0, []
    for segment in segments:
        need(isinstance(segment, dict), "candidate segment is malformed")
        start, end = segment.get("input_start"), segment.get("input_end")
        need(type(start) is int and type(end) is int and start == cursor
             and start < end <= len(source) and segment.get("source") == source[start:end]
             and isinstance(segment.get("text"), str)
             and isinstance(segment.get("kind"), str) and segment["kind"],
             "candidate segment does not align to source")
        if segment["kind"] == "raw":
            need(segment["text"] == segment["source"], "raw segment changed source bytes")
        cursor = end
        output.append(segment["text"])
    need(cursor == len(source) and candidate["text"] == "".join(output),
         "candidate text differs from complete segments")


def _validate_response(response, source, context, *, raw, model_artifacts=None):
    need(isinstance(response, dict) and response.get("input") == source
         and response.get("context") == context and response.get("engine") == "neural"
         and response.get("raw") is raw and response.get("version") == RUNTIME_VERSION,
         "captured response identity or runtime differs")
    candidates = response.get("candidates")
    need(isinstance(candidates, list) and 1 <= len(candidates) <= 5
         and response.get("result") == candidates[0].get("text"),
         "captured response top-five differs")
    for candidate in candidates:
        _validate_segments(candidate, source, rounded_score=True)
    need(len({candidate["text"] for candidate in candidates}) == len(candidates),
         "captured response contains duplicate texts")
    if raw:
        expected = {"text": source, "score": 0.0, "segments": [{
            "input_start": 0, "input_end": len(source), "source": source,
            "text": source, "kind": "raw",
        }]}
        need(candidates == [expected] and response["result"] == source
             and response.get("model_artifacts") == {} and response.get("ranker") is None,
             "raw bypass response differs")
    else:
        need(_valid_hash_map(response.get("model_artifacts")),
             "captured model artifact identity is malformed")
        if model_artifacts is not None:
            need(response["model_artifacts"] == model_artifacts,
                 "captured model artifacts differ from frozen FIT")


def _validate_path(path, source):
    need(isinstance(path, dict) and set(path) == {"text", "score", "segments"},
         "instrumented path object differs")
    _validate_segments(path, source)


def _preflight(directory):
    directory = Path(directory).resolve()
    report_path = directory / "report.json"
    captures_path = directory / "captures.jsonl"
    sources_path = directory / "sources.jsonl"
    report_sha = file_digest(report_path)
    report = read_json(report_path)
    need(report.get("schema") == REPORT_SCHEMA and report.get("completed") is True
         and report.get("deployed") is False
         and report.get("scope") == "first512 natural FIT only"
         and report.get("failure") is None
         and report.get("runtime_version") == RUNTIME_VERSION
         and report.get("rows_captured_before_accuracy") == ROWS,
         "FIT report is incomplete or has the wrong scope")
    need(type(report.get("elapsed_seconds")) in (int, float)
         and math.isfinite(report["elapsed_seconds"]) and report["elapsed_seconds"] >= 0
         and isinstance(report.get("created_at"), str), "report timing metadata differs")
    try:
        datetime.fromisoformat(report["created_at"].replace("Z", "+00:00"))
    except ValueError as error:
        raise VerificationError("report creation timestamp is malformed") from error
    need(report.get("captures_sha256") == file_digest(captures_path)
         and report.get("sources_sha256") == file_digest(sources_path),
         "capture or source file hash chain differs")
    expected_sources = _expected_source_hashes()
    need(report.get("source_hashes_before") == expected_sources
         and report.get("source_hashes_after") == expected_sources,
         "FIT source hash chain differs")
    for name, expected in expected_sources.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        need(file_digest(path) == expected, f"FIT source changed: {name}")

    sources, captures = read_jsonl(sources_path), read_jsonl(captures_path)
    need(len(sources) == len(captures) == ROWS, "exactly 512 source captures are required")
    identities = set()
    for source, capture in zip(sources, captures):
        need(set(source) == {"id", "input", "context"}
             and all(isinstance(source.get(key), str) for key in source),
             "source-only row is malformed")
        need(source["id"] and source["input"] and source["id"] not in identities,
             "source identity is empty or duplicated")
        identities.add(source["id"])
        need(all(capture.get(key) == source[key] for key in source),
             "capture source identity differs")
        for name in ("parent", "child", "parent_raw", "child_raw"):
            need(isinstance(capture.get(name), dict), f"capture response missing: {name}")
        _validate_response(capture["parent"], source["input"], source["context"], raw=False)
        _validate_response(capture["child"], source["input"], source["context"], raw=False)
        _validate_response(capture["parent_raw"], source["input"], source["context"], raw=True)
        _validate_response(capture["child_raw"], source["input"], source["context"], raw=True)
        need(canonical(capture["parent_raw"]) == canonical(capture["child_raw"]),
             "raw response changed")
        for timing in ("parent_ms", "child_ms"):
            need(type(capture.get(timing)) in (int, float)
                 and math.isfinite(capture[timing]) and capture[timing] >= 0,
                 "capture timing is malformed")
        for flag in ("parent_sentence_scored", "child_sentence_scored"):
            need(type(capture.get(flag)) is bool, "sentence scoring flag is malformed")
        for name in ("parent_pool", "parent_base"):
            need(isinstance(capture.get(name), list) and capture[name],
                 f"instrumented path pool missing: {name}")
            for path in capture[name]:
                _validate_path(path, source["input"])
        trace = capture.get("trace")
        need(isinstance(trace, dict)
             and all(isinstance(trace.get(name), list) for name in (
                 "base", "accepted_old_extras", "accepted_new_extras", "scoring_pool"))
             and type(trace.get("eligible_piece_count")) is int
             and trace["eligible_piece_count"] >= 0,
             "numeric composition trace is incomplete")
        for name in ("base", "accepted_old_extras", "accepted_new_extras", "scoring_pool"):
            for path in trace[name]:
                _validate_path(path, source["input"])
        need(canonical(trace["base"]) == canonical(capture["parent_base"]),
             "original base pool objects changed")
        if capture["parent_sentence_scored"]:
            preserved = trace["base"] + trace["accepted_old_extras"]
            need(canonical(preserved) == canonical(capture["parent_pool"])
                 and canonical(trace["scoring_pool"][:len(preserved)]) == canonical(preserved),
                 "scored old pool objects or order changed")
        score_inputs = capture.get("child_score_inputs")
        if capture["child_sentence_scored"]:
            need(score_inputs == [path["text"] for path in trace["scoring_pool"]],
                 "actual sentence scorer inputs differ from trace")
        else:
            need(score_inputs is None, "unexpected child sentence scorer inputs")
    # No FIT row or accepted answer has been loaded before this return.
    return directory, report_path, report_sha, report, sources, captures, expected_sources


def _default_fit_loader():
    from scripts import audit_final_choice_candidate_stages as prior
    from scripts.train_final_choice_selector import load_fit_role

    loaded = load_fit_role(prior.FIT, FIT_MANIFEST_SHA256, "natural_fit")
    return {"rows": loaded["rows"][:ROWS], "manifest": loaded["manifest"]}


def _flags(response):
    return {name: response.get(name) for name in FLAG_FIELDS}


def _empty_counts():
    return dict.fromkeys(("rows", "before_top1", "top1", "before_top5", "top5",
                          "before_pool", "pool", "eligible_rows", "new_path_rows"), 0)


def _score(rows, captures):
    counts = _empty_counts()
    corrections, regressions, top5_gains, top5_losses = [], [], [], []
    for row, capture in zip(rows, captures):
        expected = row.get("expected")
        need(isinstance(expected, list) and expected
             and all(isinstance(value, str) and value for value in expected),
             "FIT accepted answers are malformed")
        expected_set = set(expected)
        parent, child, trace = capture["parent"], capture["child"], capture["trace"]
        before = parent["result"] in expected_set
        after = child["result"] in expected_set
        before5 = any(candidate["text"] in expected_set for candidate in parent["candidates"])
        after5 = any(candidate["text"] in expected_set for candidate in child["candidates"])
        child_pool = trace["scoring_pool"] if capture["child_sentence_scored"] else child["candidates"]
        values = {
            "rows": 1, "before_top1": before, "top1": after,
            "before_top5": before5, "top5": after5,
            "before_pool": any(path["text"] in expected_set for path in capture["parent_pool"]),
            "pool": any(path["text"] in expected_set for path in child_pool),
            "eligible_rows": trace["eligible_piece_count"] > 0,
            "new_path_rows": bool(trace["accepted_new_extras"]),
        }
        for name, value in values.items():
            counts[name] += int(value)
        if after and not before:
            corrections.append(row["id"])
        if before and not after:
            regressions.append(row["id"])
        if after5 and not before5:
            top5_gains.append(row["id"])
        if before5 and not after5:
            top5_losses.append(row["id"])
    gate = {
        "passed": len(rows) == ROWS and not regressions and not top5_losses and bool(top5_gains),
        "original_pool_preserved": True,
        "raw_preserved": True,
        "zero_first_choice_losses": not regressions,
        "zero_top5_losses": not top5_losses,
        "at_least_one_new_top5": bool(top5_gains),
    }
    return {"counts": counts, "correction_ids": corrections,
            "regression_ids": regressions, "top5_gain_ids": top5_gains,
            "top5_loss_ids": top5_losses, "gate": gate}


def verify_evaluation(directory, fit_loader=None):
    """Verify evidence first, then open the frozen FIT labels and score it."""
    _validate_authority()
    (directory, report_path, report_sha, report, sources, captures,
     source_hashes) = _preflight(directory)
    loaded = (fit_loader or _default_fit_loader)()
    need(isinstance(loaded, dict) and isinstance(loaded.get("rows"), list)
         and isinstance(loaded.get("manifest"), dict), "FIT loader result is malformed")
    rows = loaded["rows"]
    need(len(rows) >= ROWS, "frozen natural FIT first512 is incomplete")
    rows = rows[:ROWS]
    model_artifacts = loaded["manifest"].get("model_artifact_hashes")
    need(_valid_hash_map(model_artifacts), "frozen FIT model artifacts are malformed")
    for source, capture, row in zip(sources, captures, rows):
        need(all(row.get(key) == source[key] for key in ("id", "input", "context")),
             "frozen FIT source identity/order differs")
        need(row.get("runtime_version") == RUNTIME_VERSION
             and row.get("result") == capture["parent"]["result"]
             and canonical(row.get("candidates")) == canonical(capture["parent"]["candidates"])
             and canonical(row.get("response_flags")) == canonical(_flags(capture["parent"])),
             "production baseline candidate objects or flags differ from frozen FIT")
        _validate_response(capture["parent"], source["input"], source["context"],
                           raw=False, model_artifacts=model_artifacts)
        _validate_response(capture["child"], source["input"], source["context"],
                           raw=False, model_artifacts=model_artifacts)
    metrics = _score(rows, captures)
    for name in ("counts", "correction_ids", "regression_ids", "top5_gain_ids",
                 "top5_loss_ids", "gate"):
        need(report.get(name) == metrics[name], f"declared FIT {name} differs")
    child_timings = sorted(capture["child_ms"] for capture in captures[1:])
    warm_p95 = child_timings[math.ceil(.95 * len(child_timings)) - 1]
    need(report.get("first_child_ms") == captures[0]["child_ms"]
         and report.get("warm_child_p95_ms") == warm_p95
         and report.get("latency_scope") ==
             "engine calls after model load; not browser or cold model loading",
         "reported FIT latency differs")
    need(file_digest(report_path) == report_sha
         and report.get("source_hashes_after") == source_hashes,
         "FIT report or source chain changed during verification")
    quality_passed = metrics["gate"]["passed"]
    return {
        "stage": "numeric_composition_fit",
        "verification_passed": True,
        "quality_gate_passed": quality_passed,
        "downstream_evaluation_allowed": quality_passed,
        "promotion_allowed": False,
        "resource_valid_experiment": True,
        "report": {"path": path_key(report_path), "sha256": report_sha},
        "captures": {"path": path_key(directory / "captures.jsonl"),
                     "sha256": report["captures_sha256"], "rows": ROWS},
        "sources": {"path": path_key(directory / "sources.jsonl"),
                    "sha256": report["sources_sha256"], "rows": ROWS},
        "runtime_version": RUNTIME_VERSION,
        "model_artifacts": model_artifacts,
        "source_hashes_sha256": digest(canonical(source_hashes)),
        "metrics": metrics,
        "labels_read_only_after_complete_capture_preflight": True,
        "deployed": False,
        "fresh_development_final_read": False,
        "verifier": {"path": path_key(Path(__file__)),
                     "sha256": file_digest(Path(__file__))},
    }


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
    except VerificationError as error:
        print(json.dumps({"schema": SCHEMA, "verification_passed": False,
                          "error": str(error)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"schema": SCHEMA, "verification_passed": True,
                      "quality_gate_passed": payload["quality_gate_passed"],
                      "downstream_evaluation_allowed":
                          payload["downstream_evaluation_allowed"],
                      "promotion_allowed": payload["promotion_allowed"],
                      "output": str(args.output)}, sort_keys=True))
    return 0 if payload["quality_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
