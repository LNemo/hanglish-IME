"""Run the fixed DEV gate for the isolated numeric-composition runtime.

Every prediction and parent-reference check is closed before target records are
reopened. Passing this evaluator permits later evaluation only; it never
promotes or deploys the experimental runtime.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from scripts import evaluate_numeric_composition_downstream as downstream


ROOT = downstream.ROOT
SCHEMA = "numeric-composition-development-v1"
RUNTIME_VERSION = downstream.RUNTIME_VERSION
PROFILE = downstream.PROFILE
SUITE_COUNTS = {
    "natural_dev": 128,
    "original_dev": 140,
    "selection_dev": 60,
    "reasoning_dev": 40,
}
TOTAL_ROWS = sum(SUITE_COUNTS.values())
DOWNSTREAM_PROOF = ROOT / ".omx/artifacts/numeric-composition-downstream-v1-proof.json"
DOWNSTREAM_PROOF_SHA256 = "b7777ebe42950dc5e264e728c0f47c894f06b9024fa82ac954a3c2c58b25fd7c"
DOWNSTREAM_VERIFIER_SHA256 = "0aed52466a7153466dab6b31657aa278a4bb169d4cf71da041ad7147af6fceea"
INTEGRATION = ROOT / ".omx/artifacts/numeric-structure-integration-v1/report.json"
INTEGRATION_SHA256 = "77c04334af817e15bc4c5ccc9d27b1d6ab9f9f6a759b4207e752762ae49d3aab"
REFERENCE = ROOT / ".omx/artifacts/numeric-structure-development-v1"
REFERENCE_REPORT_SHA256 = "750e777f784e3bbdd2e79246b8abc68532e4840dc18e6c9f035da3e62a9b14e0"
REFERENCE_CASES_SHA256 = "2baaccfbca4d7f4370a9c0b61c548fa9d4e33c50440c38c6dabdcd67d2ca920c"
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


def _path(path):
    return str(Path(path).resolve().relative_to(ROOT))


def _merge(target, additions):
    for name, value in additions.items():
        if name in target and target[name] != value:
            raise ValueError(f"development source hash conflict: {name}")
        target[name] = value


def validate_downstream_entry():
    """Validate the complete fresh proof before any DEV record is opened."""
    if downstream.file_digest(DOWNSTREAM_PROOF) != DOWNSTREAM_PROOF_SHA256:
        raise ValueError("numeric downstream proof changed")
    proof = json.loads(DOWNSTREAM_PROOF.read_bytes())
    payload = proof.get("payload")
    expected_files = {
        "report": (".omx/artifacts/numeric-composition-downstream-v1/report.json",
                   "209df79c60e01f50855ddaeb9ca245f06b27649bbabbb9d722988994343ccbd9"),
        "sources": (".omx/artifacts/numeric-composition-downstream-v1/sources.jsonl",
                    "e8fefa187dd3b87809ac4d8a2b88add86b6f34cd87c5c552580b922d00b55ff6"),
        "targets": (".omx/artifacts/numeric-composition-downstream-v1/targets.jsonl",
                    "957f784665edca86162e0b677472b1e2f4ce716edd7991e87992da6568c0182f"),
        "captures": (".omx/artifacts/numeric-composition-downstream-v1/captures.jsonl",
                     "d4639c041330262ac8444a047e6d841f2512c8ef3f893222969750a5e5b2ddae"),
        "preservation_baselines": (
            ".omx/artifacts/numeric-composition-downstream-v1/preservation_baselines.jsonl",
            "e95119f0e0ac17c4383bdeaddfd451f0434ece1b6b9f84a58649aafcac2d7580"),
    }
    if (proof.get("schema") != "numeric-composition-downstream-proof-v1"
            or not isinstance(payload, dict)
            or proof.get("digest") != downstream.digest_value(payload)
            or payload.get("verification_passed") is not True
            or payload.get("quality_gate_passed") is not True
            or payload.get("downstream_evaluation_allowed") is not True
            or payload.get("resource_valid_experiment") is not True
            or payload.get("labels_read_only_after_complete_capture_preflight") is not True
            or payload.get("fresh_targets_read") is not True
            or payload.get("development_final_targets_read") is not False
            or payload.get("promotion_allowed") is not False
            or payload.get("deployed") is not False
            or payload.get("runtime_version") != RUNTIME_VERSION
            or payload.get("verifier") != {
                "path": "scripts/verify_numeric_composition_downstream.py",
                "sha256": DOWNSTREAM_VERIFIER_SHA256,
            }):
        raise ValueError("numeric downstream proof is not a valid DEV entry")
    for field, (path, expected) in expected_files.items():
        item = payload.get(field)
        expected_rows = (None if field == "report" else
                         downstream.PRESERVATION_ROWS
                         if field == "preservation_baselines" else downstream.TOTAL_ROWS)
        if (not isinstance(item, dict) or item.get("path") != path
                or item.get("sha256") != expected
                or (expected_rows is not None and item.get("rows") != expected_rows)
                or downstream.file_digest(ROOT / path) != expected):
            raise ValueError(f"numeric downstream {field} lineage differs")
    if downstream.file_digest(ROOT / payload["verifier"]["path"]) != DOWNSTREAM_VERIFIER_SHA256:
        raise ValueError("numeric downstream verifier changed")
    report = json.loads((ROOT / expected_files["report"][0]).read_bytes())
    if (report.get("completed") is not True
            or report.get("terminal_status") != "success"
            or report.get("captures_closed") is not True
            or report.get("models_released_before_target_join") is not True
            or report.get("targets_joined") is not True
            or report.get("quality_gate_passed") is not True
            or report.get("downstream_evaluation_allowed") is not True
            or report.get("promotion_allowed") is not False
            or report.get("deployed") is not False
            or report.get("metrics") != payload.get("metrics")
            or report.get("source_hashes_before") != report.get("source_hashes_after")
            or downstream.digest_value(report.get("source_hashes_before"))
               != payload.get("source_hashes_sha256")):
        raise ValueError("numeric downstream report semantics differ")
    fresh = payload.get("metrics", {}).get("roles", {}).get("fresh_natural", {})
    preservation = payload.get("metrics", {}).get("roles", {}).get(
        "challenge_preservation", {})
    bootstrap = payload.get("metrics", {}).get("bootstrap", {})
    if (fresh.get("rows") != 512 or fresh.get("before_top1") != 318
            or fresh.get("top1") != 336 or fresh.get("before_top5") != 331
            or fresh.get("top5") != 350 or len(fresh.get("corrections", [])) != 18
            or fresh.get("regressions") != [] or fresh.get("top5_losses") != []
            or preservation.get("rows") != 376 or preservation.get("top1") != 376
            or preservation.get("top5") != 376 or preservation.get("regressions") != []
            or bootstrap.get("net_gain") != 18
            or bootstrap.get("gain_and_interval_passed") is not True):
        raise ValueError("numeric downstream fixed quality result differs")
    downstream.verify_hashes(report["source_hashes_before"])

    if downstream.file_digest(INTEGRATION) != INTEGRATION_SHA256:
        raise ValueError("numeric integration report changed")
    integration = json.loads(INTEGRATION.read_bytes())
    weights = integration.get("accepted_weight_hashes")
    if (integration.get("passed") is not True
            or integration.get("scope") != "post-integration actual runtime parity"
            or integration.get("integrated_runtime_version") != RUNTIME_VERSION
            or integration.get("accepted_development_report_sha256")
               != REFERENCE_REPORT_SHA256
            or integration.get("prototype_artifact_hashes", {}).get(
                ".omx/artifacts/numeric-structure-development-v1/cases.jsonl")
               != REFERENCE_CASES_SHA256
            or integration.get("prototype_artifact_hashes", {}).get(
                ".omx/artifacts/numeric-structure-development-v1/report.json")
               != REFERENCE_REPORT_SHA256
            or not isinstance(weights, dict) or len(weights) != 16
            or weights != payload.get("model_artifacts")
            or integration.get("integrated_runtime_source_hashes")
               != integration.get("runtime_source_hashes_after")
            or integration.get("rows") != 1880
            or integration.get("all_candidate_objects_equal") is not True
            or integration.get("all_sentence_pools_equal") is not True
            or integration.get("all_sentence_scores_within_tolerance") is not True
            or integration.get("all_versions_equal") is not True
            or integration.get("deployed") is not False):
        raise ValueError("accepted current production reference differs")
    source_hashes = dict(report["source_hashes_before"])
    _merge(source_hashes, integration["integrated_runtime_source_hashes"])
    _merge(source_hashes, weights)
    fixed = [DOWNSTREAM_PROOF, ROOT / payload["verifier"]["path"], INTEGRATION,
             REFERENCE / "report.json", ROOT / "scripts/evaluate_numeric_composition_downstream.py",
             ROOT / "scripts/evaluate_numeric_structure_development.py", Path(__file__)]
    fixed.extend(ROOT / path for path, _ in expected_files.values())
    for path in fixed:
        _merge(source_hashes, {_path(path): downstream.file_digest(path)})
    if downstream.file_digest(REFERENCE / "report.json") != REFERENCE_REPORT_SHA256:
        raise ValueError("accepted numeric DEV report changed")
    reference_report = json.loads((REFERENCE / "report.json").read_bytes())
    if (reference_report.get("passed") is not True
            or reference_report.get("eligible_for_further_evaluation") is not True
            or reference_report.get("cases_sha256") != REFERENCE_CASES_SHA256
            or reference_report.get("source_hashes")
               != reference_report.get("source_hashes_after")
            or {name: reference_report.get("suites", {}).get(name, {}).get("rows")
                for name in SUITE_COUNTS} != SUITE_COUNTS
            or reference_report.get("final_data_read") is not False
            or reference_report.get("deployed") is not False):
        raise ValueError("accepted numeric DEV identity report differs")
    fixed_reference_top1 = {"natural_dev": 81, "original_dev": 140,
                            "selection_dev": 60, "reasoning_dev": 34}
    if (any(reference_report["suites"][name].get("top1") != correct
            or reference_report["suites"][name].get("regressions") != []
            or reference_report["suites"][name].get("top5_regressions") != []
            for name, correct in fixed_reference_top1.items())):
        raise ValueError("accepted numeric DEV reference metrics differ")
    downstream.verify_hashes(source_hashes)
    return {"proof": payload, "integration": integration,
            "reference_report": reference_report,
            "source_hashes": dict(sorted(source_hashes.items())), "weights": weights}


def load_and_split_development(entry, access):
    """Open the four DEV suites only after validate_downstream_entry succeeds."""
    access["development_target_bytes_accessed"] = True
    reference_raw = (REFERENCE / "cases.jsonl").read_bytes()
    if hashlib.sha256(reference_raw).hexdigest() != REFERENCE_CASES_SHA256:
        raise ValueError("accepted numeric DEV cases changed")
    reference_rows = [json.loads(line) for line in reference_raw.splitlines() if line]
    grouped = {name: [] for name in SUITE_COUNTS}
    for row in reference_rows:
        if row.get("suite") not in grouped:
            raise ValueError("accepted numeric DEV case has an unknown suite")
        grouped[row["suite"]].append(row)
    if {name: len(rows) for name, rows in grouped.items()} != SUITE_COUNTS:
        raise ValueError("accepted numeric DEV case counts differ")

    sources, targets, references, hashes = [], [], [], {
        _path(REFERENCE / "cases.jsonl"): REFERENCE_CASES_SHA256,
    }
    for suite, corpus_path, identity_path in SUITES:
        corpus_raw, identity_raw = corpus_path.read_bytes(), identity_path.read_bytes()
        expected_corpus = entry["reference_report"]["source_hashes"].get(_path(corpus_path))
        expected_identity = entry["reference_report"]["source_hashes"].get(_path(identity_path))
        if (hashlib.sha256(corpus_raw).hexdigest() != expected_corpus
                or hashlib.sha256(identity_raw).hexdigest() != expected_identity):
            raise ValueError(f"frozen {suite} corpus or identity changed")
        rows = [json.loads(line) for line in corpus_raw.splitlines() if line]
        if len(rows) != SUITE_COUNTS[suite]:
            raise ValueError(f"frozen {suite} row count differs")
        for row, reference in zip(rows, grouped[suite]):
            if (any(row.get(name) != reference.get(name)
                    for name in ("id", "input", "context", "expected"))
                    or not isinstance(row.get("expected"), list) or not row["expected"]):
                raise ValueError(f"frozen {suite} source/target identity differs")
            group = row.get("group_id")
            if not isinstance(group, str) or not group:
                group = f"{suite}:{row['id']}"
            sources.append({"id": row["id"], "group_id": group, "origin": suite,
                            "input": row["input"], "context": row["context"]})
            targets.append({"id": row["id"], "role": suite,
                            "expected": deepcopy(row["expected"])})
            references.append({"id": row["id"], "suite": suite,
                "candidates": deepcopy(reference["candidates"]),
                "sentence_pool": deepcopy(reference["sentence_pool"]),
                "flags": deepcopy(reference.get("flags", {}))})
        hashes[_path(corpus_path)] = hashlib.sha256(corpus_raw).hexdigest()
        hashes[_path(identity_path)] = hashlib.sha256(identity_raw).hexdigest()
        del corpus_raw, identity_raw
    if (len(sources) != TOTAL_ROWS or len({row["id"] for row in sources}) != TOTAL_ROWS):
        raise ValueError("numeric DEV source accounting differs")
    downstream.verify_hashes(hashes)
    return sources, targets, references, dict(sorted(hashes.items()))


def validate_parent_reference(capture, reference):
    if (capture["id"] != reference.get("id")
            or capture["parent"].get("candidates") != reference.get("candidates")):
        raise ValueError("current 867f parent candidates differ from accepted reference")
    expected_pool = reference.get("sentence_pool")
    actual_pool = ([row["text"] for row in capture["parent_pool"]]
                   if capture["parent_sentence_scored"] else None)
    if actual_pool != expected_pool:
        raise ValueError("current 867f parent sentence pool differs from accepted reference")
    comparable = ("ranker", "context_used", "ranking_applied",
                  "sentence_ranking_applied", "warnings")
    if any(capture["parent"].get(name) != reference.get("flags", {}).get(name)
           for name in comparable):
        raise ValueError("current 867f parent flags differ from accepted reference")


def capture_development(bundle, source, reference, weights):
    capture = downstream.capture_source(bundle, source, None, weights)
    validate_parent_reference(capture, reference)
    return capture


def score_development(sources, targets, captures):
    if len(sources) != TOTAL_ROWS or len(targets) != TOTAL_ROWS or len(captures) != TOTAL_ROWS:
        raise ValueError("numeric DEV captures are incomplete")
    suites = {name: {"rows": 0, "before_top1": 0, "top1": 0,
        "before_top5": 0, "top5": 0, "before_pool": 0, "pool": 0,
        "corrections": [], "regressions": [], "top5_gains": [], "top5_losses": []}
        for name in SUITE_COUNTS}
    seen = set()
    for source, target, capture in zip(sources, targets, captures):
        downstream.validate_capture_replay(source, capture)
        identity, role = source["id"], target.get("role")
        expected = target.get("expected")
        if (identity in seen or target.get("id") != identity or capture.get("id") != identity
                or role != source.get("origin") or role not in suites
                or not isinstance(expected, list) or not expected
                or any(not isinstance(value, str) or not value for value in expected)):
            raise ValueError("numeric DEV target join differs")
        seen.add(identity)
        parent, child, trace = capture["parent"], capture["child"], capture["trace"]
        before, after = parent["result"] in expected, child["result"] in expected
        before5 = any(row["text"] in expected for row in parent["candidates"])
        after5 = any(row["text"] in expected for row in child["candidates"])
        child_pool = trace["scoring_pool"] if capture["child_sentence_scored"] \
            else child["candidates"]
        values = suites[role]
        values["rows"] += 1
        for name, condition in (("before_top1", before), ("top1", after),
                ("before_top5", before5), ("top5", after5),
                ("before_pool", any(row["text"] in expected for row in capture["parent_pool"])),
                ("pool", any(row["text"] in expected for row in child_pool))):
            values[name] += int(condition)
        for condition, name in ((after and not before, "corrections"),
                (before and not after, "regressions"),
                (after5 and not before5, "top5_gains"),
                (before5 and not after5, "top5_losses")):
            if condition:
                values[name].append(identity)
    if {name: row["rows"] for name, row in suites.items()} != SUITE_COUNTS:
        raise ValueError("numeric DEV suite denominators differ")
    protected = (suites["original_dev"]["top1"] == 140
                 and suites["selection_dev"]["top1"] == 60)
    top5_preserved = all(not row["top5_losses"] for row in suites.values())
    passed = (suites["natural_dev"]["top1"] >= 81 and protected
              and suites["reasoning_dev"]["top1"] > 34 and top5_preserved)
    return {"suites": suites, "gate": {"passed": passed,
        "natural_at_least_81": suites["natural_dev"]["top1"] >= 81,
        "original_140_of_140": suites["original_dev"]["top1"] == 140,
        "selection_60_of_60": suites["selection_dev"]["top1"] == 60,
        "reasoning_above_34_of_40": suites["reasoning_dev"]["top1"] > 34,
        "protected_preserved": protected, "top5_preserved": top5_preserved,
        "raw_preserved": True, "original_pool_preserved": True,
        "scorer_trace_preserved": True, "final_evaluation_allowed": passed,
        "promotion_allowed": False}}


def execute_pipeline(output, entry, loader, model_loader, capture_one,
                     model_releaser=downstream.release_models,
                     resource_observer=downstream.resources):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started, bundle = time.monotonic(), None
    report = {"schema": SCHEMA, "completed": False, "terminal_status": "running",
        "profile": PROFILE, "runtime_version": RUNTIME_VERSION, "deployed": False,
        "promotion_allowed": False, "final_data_read": False,
        "target_access": {"development_target_bytes_accessed": False},
        "captures_closed": False, "models_released_before_target_join": False,
        "targets_joined": False, "captured_rows": 0, "resource_observations": []}
    try:
        sources, targets, references, target_hashes = loader(entry, report["target_access"])
        if (len(sources) != TOTAL_ROWS or len(targets) != TOTAL_ROWS
                or len(references) != TOTAL_ROWS
                or any(set(row) != {"id", "group_id", "origin", "input", "context"}
                       or not all(isinstance(row.get(name), str)
                                  and (name == "context" or bool(row[name])) for name in row)
                       for row in sources)
                or any(set(row) != {"id", "role", "expected"}
                       or not isinstance(row.get("expected"), list)
                       or not row["expected"]
                       for row in targets)
                or any("expected" in row for row in references)
                or [row.get("id") for row in sources] != [row.get("id") for row in targets]
                or [row.get("id") for row in sources] != [row.get("id") for row in references]):
            raise ValueError("numeric DEV split accounting differs")
        hashes = dict(entry["source_hashes"])
        _merge(hashes, target_hashes)
        downstream.verify_hashes(hashes)
        report["source_hashes_before"] = dict(sorted(hashes.items()))
        report["files"] = {
            "sources.jsonl": downstream.write_jsonl(output / "sources.jsonl", sources),
            "targets.jsonl": downstream.write_jsonl(output / "targets.jsonl", targets),
            "parent_references.jsonl": downstream.write_jsonl(
                output / "parent_references.jsonl", references),
        }
        references_by_id = {row["id"]: row for row in references}
        del targets, references
        loaded = time.monotonic()
        bundle = model_loader(entry["weights"])
        report["model_load_seconds"] = time.monotonic() - loaded
        report["model_loaded"] = True
        captures = []
        with (output / "captures.jsonl").open("xb") as stream:
            for index, source in enumerate(sources, 1):
                capture = capture_one(bundle, source, references_by_id[source["id"]],
                                      entry["weights"])
                stream.write(downstream.canonical(capture) + b"\n")
                stream.flush()
                captures.append(capture)
                report["captured_rows"] = index
                report["resource_observations"].append(
                    resource_observer(started, f"development_capture_{index}"))
        report["captures_closed"] = True
        report["files"]["captures.jsonl"] = downstream.file_digest(output / "captures.jsonl")
        model_releaser(bundle)
        report["model_loaded"] = False
        report["models_released_before_target_join"] = True
        terminal = resource_observer(started, "after_development_model_release")
        report["terminal_resources"] = terminal
        report["resource_observations"].append(terminal)
        downstream.verify_hashes(hashes)
        reopened_targets = downstream.read_jsonl(output / "targets.jsonl")
        report["targets_joined"] = True
        metrics = score_development(sources, reopened_targets, captures)
        downstream.verify_hashes(hashes)
        warm = captures[1:]
        report.update({"completed": True, "terminal_status": "success",
            "rows": TOTAL_ROWS, "suite_counts": SUITE_COUNTS,
            "model_artifacts": entry["weights"],
            "source_hashes_after": dict(sorted(hashes.items())), "metrics": metrics,
            "quality_gate_passed": metrics["gate"]["passed"],
            "final_evaluation_allowed": metrics["gate"]["passed"],
            "promotion_allowed": False, "deployed": False,
            "latency": {"first_parent_ms": captures[0]["parent_ms"],
                "first_child_ms": captures[0]["child_ms"],
                "warm_parent_p95_ms": sorted(row["parent_ms"] for row in warm)[
                    math.ceil(.95 * len(warm)) - 1],
                "warm_child_p95_ms": sorted(row["child_ms"] for row in warm)[
                    math.ceil(.95 * len(warm)) - 1],
                "scope": "engine calls after model load; not browser or cold model loading"},
            "elapsed_seconds": time.monotonic() - started,
            "created_at": datetime.now(timezone.utc).isoformat()})
    except BaseException as error:
        if bundle is not None and report.get("model_loaded"):
            try:
                model_releaser(bundle)
                report["model_loaded"] = False
                report["models_released_on_failure"] = True
            except BaseException as release_error:
                report["models_released_on_failure"] = False
                report["release_failure"] = f"{type(release_error).__name__}: {release_error}"
        partial = output / "captures.jsonl"
        if partial.exists():
            report["partial_captures_sha256"] = downstream.file_digest(partial)
        report.update({"completed": False, "terminal_status": "failed",
            "failure": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.monotonic() - started,
            "created_at": datetime.now(timezone.utc).isoformat()})
    (output / "report.json").write_bytes(downstream.canonical(report) + b"\n")
    return report


def run(output):
    entry = validate_downstream_entry()
    return execute_pipeline(output, entry, load_and_split_development,
                            downstream.load_models, capture_development)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        default=ROOT / ".omx/artifacts/numeric-composition-development-v1")
    result = run(parser.parse_args().output)
    print(json.dumps({"completed": result["completed"],
        "quality_gate_passed": result.get("quality_gate_passed"),
        "final_evaluation_allowed": result.get("final_evaluation_allowed"),
        "promotion_allowed": False}, sort_keys=True))
    raise SystemExit(0 if result["completed"] else 1)
