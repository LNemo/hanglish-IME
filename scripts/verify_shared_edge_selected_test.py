#!/usr/bin/env python3
"""Independent verifier for the DEV-selected shared-edge 1,000-case TEST run.

The source-only proposal and prediction closure is authenticated before the
canonical TEST target is opened.  Neural forwards and tokenization are not
repeated; proposal generation is deterministic, source-only, and replayed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "shared-edge-selected-test-independent-verification-v1"
RUN_SCHEMA = "shared-edge-selected-test-evaluation-v1"
ROWS = 1000
SELECTED = "shared_edge"
BASELINE = {"top1": 953, "top5": 972}
RUN_PLAN = ROOT / ".omx/plans/shared-edge-residual-v1.md"
TRAINING = ROOT / ".omx/artifacts/shared-edge-residual-training-v1"
DEV_RUN = ROOT / ".omx/artifacts/shared-edge-residual-dev-v1"
RUNTIME = ROOT / ".omx/artifacts/extended-runtime-test-v1"
DEV_REPORT_SHA256 = "65138dc85c01480cb8d9c8ab67a9a288c77252934f9db3f7841b65b0cf147dad"
DEV_PROOF_SHA256 = "6506d584f276da6311f1c464fa0374c88faa4a0fe123a32565dc75ee2f8a4d55"
DEV_HELPER = ROOT / ".omx/artifacts/verify-source-span-pair-dev-v1.py"
DEV_HELPER_SHA256 = "77373cca43d198a4fd47946cb8584e9dcb6089583a7a7ff2a47f48c1de2baa1d"
SHARED_HELPER = ROOT / "scripts/verify_shared_edge_dev.py"
SHARED_HELPER_SHA256 = "b479b474c479c72bbf2ccd6b7d101dd0becc1bb6027d9814efbc7127b5f9b617"
PRODUCER = ROOT / "scripts/evaluate_shared_edge_selected_test.py"
PRODUCER_SHA256 = "d2222dfd2146071c09117a5af644b8dc3c96b431a38b7ece2c711756cafb549b"
GENERATOR = ROOT / "hanglish/core/source_edge_recombination.py"
GENERATOR_SHA256 = "c4e7d7c59c3edb8623b28dc933f429b220b4d17ff58416d47449eeaa4d78d119"
SELECTED_RESIDUAL_SHA256 = "a7282d5fba58b2e2bd3b01211cd0397d2a53fba6d695044495ba34c3f26c957d"
TEST_SHA256 = "1dcf804460b40c445abf688dc47b452937d8911ac156b8520141fad25b973d06"
RUNTIME_HASHES = {
    "report.json": "a4a1bb33ec2756fee2ed9833eda126fdf7cd27caa591414c92250af35439a402",
    "capture.jsonl": "3d7b12d458411bd8caa6b34ea47eb024df15d5bbaa0467b58ac7d3b9a30ce010",
    "capture.complete.json": "c6ee6af86852544945aad63066bd1b4b52ecb44e0c87772b76960b6d22c4a62d",
    "independent-verification.json": "8384213f9faf07283ce54b0abe64cdda04328e2a7936dd065ad6a3ae896327e5",
    "pins.json": "2590fe9bd42a5e7658f443ca96f7b2bf509db1534ac990590e245836803212b0",
}
LIMITS = {"whole_seconds": 1800.0, "update_seconds": 60.0,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSONL at {path}:{number}") from error
            require(isinstance(value, dict), f"JSONL row is not an object at {path}:{number}")
            rows.append(value)
    return rows


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def load_module(path, name, expected_hash):
    require(sha(path) == expected_hash, "independent helper identity differs: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load independent helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPER = load_module(DEV_HELPER, "source_span_pair_dev_offline_helper", DEV_HELPER_SHA256)
SHARED = load_module(SHARED_HELPER, "shared_edge_dev_offline_helper", SHARED_HELPER_SHA256)
GENERATOR_MODULE = load_module(GENERATOR, "source_edge_recombination_offline_replay",
                               GENERATOR_SHA256)


def validate_authorization(run, report):
    dev_report_path, dev_proof_path = DEV_RUN / "report.json", DEV_RUN / "independent-verification.json"
    require(sha(dev_report_path) == DEV_REPORT_SHA256 and sha(dev_proof_path) == DEV_PROOF_SHA256,
            "shared-edge DEV selection artifacts differ")
    dev_report, dev_proof = read(dev_report_path), read(dev_proof_path)
    expected_counts = {"captured_runtime": (264, 284), "parent": (263, 280),
                       "source_local_control": (271, 288), "shared_edge": (272, 288)}
    require(dev_report.get("schema") == "shared-edge-residual-dev-evaluation-v1"
            and dev_report.get("completed") is True and dev_report.get("rows") == 316
            and dev_report.get("decision", {}).get("selected_for_1000_case_regression") == SELECTED
            and dev_report.get("decision", {}).get("eligible_for_1000_case_regression") is True
            and dev_proof.get("schema") == "shared-edge-residual-dev-independent-verification-v1"
            and dev_proof.get("verified") is True
            and dev_proof.get("report_sha256") == DEV_REPORT_SHA256
            and dev_proof.get("decision") == dev_report.get("decision")
            and dev_proof.get("verifier_sha256") == SHARED_HELPER_SHA256,
            "shared-edge DEV independent selection proof differs")
    for arm, (top1, top5) in expected_counts.items():
        counts = dev_report["metrics"][arm]["counts"]
        require(counts.get("top1") == top1 and counts.get("top5") == top5,
                "shared-edge DEV selected metric differs: " + arm)

    training_report_path = TRAINING / "report.json"
    training_proof_path = TRAINING / "independent-verification.json"
    training_report, training_proof = read(training_report_path), read(training_proof_path)
    residual = TRAINING / "residual.safetensors"
    require(sha(training_report_path) == "481ff33a1ae6e9f0e8f3d5c24566f5e076a42d29dd02145e1ad964fd6a06e5da"
            and sha(training_proof_path) == "e4d48a8be5373198e83a11bc88793f6262520f6a2b3b075da2d3a1d8e28cf0e1"
            and training_report.get("completed") is True
            and training_proof.get("verified") is True
            and training_proof.get("report_sha256") == sha(training_report_path)
            and dev_report.get("training_report_sha256") == sha(training_report_path)
            and sha(residual) == SELECTED_RESIDUAL_SHA256
            and training_proof.get("counts", {}).get("updates") == 2966,
            "selected shared-edge training binding differs")

    for name, digest in RUNTIME_HASHES.items():
        require(sha(RUNTIME / name) == digest, "captured TEST artifact differs: " + name)
    runtime_proof = read(RUNTIME / "independent-verification.json")
    runtime_closure = read(RUNTIME / "capture.complete.json")
    runtime_pins = read(RUNTIME / "pins.json")
    require(runtime_proof.get("verified") is True and runtime_proof.get("split") == "test"
            and runtime_proof.get("identity", {}).get("report_sha256") == RUNTIME_HASHES["report.json"]
            and runtime_proof.get("metrics", {}).get("proposed")
                == {"errors": 0, "rows": ROWS, **BASELINE}
            and runtime_proof.get("contracts", {}).get("source_only_runtime") is True
            and runtime_proof.get("contracts", {}).get("capture_closed_before_target_join") is True
            and runtime_closure == {"capture_sha256": RUNTIME_HASHES["capture.jsonl"],
                "rows": ROWS, "schema": "extended-separated-runtime-source-closure-v1",
                "source_only": True, "split": "test", "targets_joined": False,
                "test_opened": True}, "captured TEST source proof differs")
    targets = [(Path(raw), digest) for raw, digest in runtime_pins.items()
               if Path(raw).name == "test.jsonl"
               and "data/challenge/sentence-v3" in Path(raw).as_posix()]
    require(len(targets) == 1 and targets[0][1] == TEST_SHA256,
            "canonical TEST target pin differs")
    target_path = targets[0][0]

    expected = dict(training_report["pins"])
    expected[str(training_report_path)] = sha(training_report_path)
    expected[str(training_proof_path)] = sha(training_proof_path)
    expected.update({str(TRAINING / relative): digest
                     for relative, digest in training_report["outputs"].items()})
    expected.update(runtime_pins)
    expected.update({str(RUNTIME / name): digest for name, digest in RUNTIME_HASHES.items()})
    dev_relied = [ROOT / "scripts/evaluate_shared_edge_residual.py",
        ROOT / "scripts/train_shared_edge_residual.py", ROOT / "hanglish/models/shared_edge_residual.py",
        ROOT / "scripts/source_span_resources.py", ROOT / "hanglish/models/span_token_bridge.py",
        ROOT / "hanglish/core/source_correspondence.py"]
    expected[str(dev_report_path)] = DEV_REPORT_SHA256; expected[str(dev_proof_path)] = DEV_PROOF_SHA256
    for path in dev_relied:
        digest = dev_report.get("pins", {}).get(str(path))
        require(isinstance(digest, str) and sha(path) == digest,
                "DEV-evaluated dependency differs: " + str(path))
        expected[str(path)] = digest
    tracked = [PRODUCER, RUN_PLAN, GENERATOR, ROOT / "scripts/evaluate_shared_edge_residual.py",
        ROOT / "scripts/evaluate_source_span_selected_test.py",
        ROOT / "scripts/score_source_edge_recombination.py", ROOT / "scripts/source_span_resources.py",
        ROOT / "hanglish/models/shared_edge_residual.py", ROOT / "hanglish/models/source_span_residual.py",
        ROOT / "hanglish/models/span_token_bridge.py", ROOT / "hanglish/core/source_correspondence.py"]
    expected.update({str(path): sha(path) for path in tracked})
    recorded = report.get("pins"); require(isinstance(recorded, dict), "selected TEST pins absent")
    extras = set(recorded) - set(expected)
    require(len(extras) == 1, "selected TEST authorization pin inventory differs")
    authorization_key = extras.pop(); authorization_path = Path(authorization_key)
    authorization = read(authorization_path)
    expected_auth = {"schema": "shared-edge-selected-test-prelaunch-authorization-v1",
        "rows": ROWS, "limits": LIMITS, "target_cache_bytes": 256 * 1024**2,
        "evaluator_sha256": PRODUCER_SHA256,
        "training_report_sha256": sha(training_report_path),
        "training_proof_sha256": sha(training_proof_path),
        "selected_residual_sha256": SELECTED_RESIDUAL_SHA256,
        "dev_report_sha256": DEV_REPORT_SHA256, "dev_proof_sha256": DEV_PROOF_SHA256,
        "runtime_artifacts": RUNTIME_HASHES,
        "runtime_source_authorization_sha256": HELPER.canonical_hash(runtime_pins),
        "source_generator_sha256": GENERATOR_SHA256,
        "model_sha256": sha(ROOT / "hanglish/models/shared_edge_residual.py")}
    require(authorization == expected_auth and recorded[authorization_key] == sha(authorization_path),
            "selected TEST prelaunch authorization differs")
    expected[authorization_key] = recorded[authorization_key]; expected = dict(sorted(expected.items()))
    require(read(run / "pins.json") == expected == recorded, "selected TEST pin inventory differs")
    for raw, digest in expected.items():
        path = Path(raw)
        if path == target_path:
            continue
        require(path.is_file() and sha(path) == digest, "selected TEST pin changed: " + raw)
    require(report.get("training_report_sha256") == sha(training_report_path)
            and report.get("training_proof_sha256") == sha(training_proof_path)
            and report.get("selected_residual_sha256") == SELECTED_RESIDUAL_SHA256
            and report.get("dev_report_sha256") == DEV_REPORT_SHA256
            and report.get("dev_proof_sha256") == DEV_PROOF_SHA256
            and report.get("runtime_report_sha256") == RUNTIME_HASHES["report.json"]
            and report.get("runtime_proof_sha256") == RUNTIME_HASHES["independent-verification.json"],
            "selected TEST report authorization differs")
    return target_path, {"training_report_sha256": sha(training_report_path),
        "training_proof_sha256": sha(training_proof_path), "dev_report_sha256": DEV_REPORT_SHA256,
        "dev_proof_sha256": DEV_PROOF_SHA256, "selected_residual_sha256": SELECTED_RESIDUAL_SHA256,
        "pins": len(expected), "domain_helper_sha256": DEV_HELPER_SHA256,
        "shared_helper_sha256": SHARED_HELPER_SHA256, "source_generator_sha256": GENERATOR_SHA256}

def replay_proposal(runtime):
    identifier, request, response = runtime.get("id"), runtime.get("request"), runtime.get("proposed")
    source = request.get("input") if isinstance(request, dict) else None
    require(isinstance(identifier, str) and isinstance(source, str)
            and request == {"input": source, "context": ""}
            and isinstance(response, dict) and response.get("input") == source
            and response.get("context") == "", "captured TEST source row differs")
    generated = GENERATOR_MODULE.propose(source, response["pipeline"]["pool"],
                                         response["candidates"])
    return {"id": identifier, "request": deepcopy(request), **generated}


def verify_prediction(saved, runtime, proposal, cached):
    expected_proposal = replay_proposal(runtime)
    require(proposal == expected_proposal, "fresh source-edge proposal replay differs")
    texts, representatives, groups, path_ids = HELPER.reconstruct_domain(runtime, proposal)
    source = runtime["request"]["input"]
    proposal_ids = [row.get("proposal_id") for row in proposal.get("proposals", [])]
    candidate_ids = [row.get("candidate_id") or row.get("proposal_id")
                     for row in representatives]
    targets = SHARED.isolated_targets(groups)
    required = {"id", "request", "proposal_schema_version", "proposal_ids",
        "candidate_texts", "candidate_ids", "path_ids", "isolated_target_texts",
        "target_cache", "support", "base_scores", "scores", "latency_ms",
        "shared_edge_audit", "responses", "protected_paths", "protected_responses"}
    require(set(saved) == required and saved["id"] == runtime["id"]
            and saved["request"] == runtime["request"]
            and saved["proposal_schema_version"] == proposal.get("schema_version")
            and saved["proposal_ids"] == proposal_ids and saved["candidate_texts"] == texts
            and saved["candidate_ids"] == candidate_ids and saved["path_ids"] == path_ids
            and saved["isolated_target_texts"] == targets
            and all(isinstance(value, str) for value in proposal_ids + candidate_ids),
            "selected TEST source/proposal/candidate/path/target identity differs")
    misses = [target for target in targets if target not in cached]
    require(saved["target_cache"] == {"requested": len(targets),
            "hits": len(targets) - len(misses), "misses": len(misses)},
            "selected TEST target cache accounting differs")
    latency = saved["latency_ms"]
    require(set(latency) == {"captured_runtime", "source_edge_proposal", "parent_features",
                             "target_extraction", SELECTED}
            and all(finite(value) and value >= 0 for value in latency.values())
            and latency["captured_runtime"] == float(runtime["elapsed_ms"]),
            "selected TEST latency differs")
    original = runtime["proposed"]
    require(saved["responses"].get("captured_runtime") == HELPER.compact_response(original),
            "captured TEST response differs")
    support = saved["support"]
    require(set(support) == {"supported", "reason"} and type(support["supported"]) is bool,
            "selected TEST support contract differs")
    if support["supported"]:
        require(support["reason"] is None and isinstance(saved["scores"], list)
                and isinstance(saved["base_scores"], list)
                and len(saved["scores"]) == len(saved["base_scores"]) == len(texts)
                and all(finite(value) for value in saved["scores"] + saved["base_scores"]),
                "selected TEST supported scores differ")
        cached.update(misses)
        residuals = SHARED.fp32_path_audit(source, groups, path_ids,
                                            saved["shared_edge_audit"])
        require(saved["scores"] == [SHARED.f32(base + residual)
                for base, residual in zip(saved["base_scores"], residuals)],
                "selected TEST FP32 shared-edge scores differ")
        expected = HELPER.apply_policy(original, representatives, texts, saved["scores"],
                                       source, SELECTED)
        require(saved["responses"].get(SELECTED) == HELPER.compact_response(expected),
                "selected TEST ranked response differs")
    else:
        require(support["reason"] in {"source_exceeds_2048_bytes", "empty_nonraw_pool",
                                      "full_candidate_exceeds_256_tokens",
                                      "isolated_target_exceeds_256_tokens"}
                and saved["scores"] == [] and saved["base_scores"] == []
                and saved["shared_edge_audit"] is None
                and saved["responses"].get(SELECTED) == HELPER.compact_response(original),
                "selected TEST fallback differs")
        if len(source.encode("utf-8")) > 2048:
            require(support["reason"] == "source_exceeds_2048_bytes", "long source reason differs")
        if not texts:
            require(support["reason"] == "empty_nonraw_pool", "empty pool reason differs")
    require(set(saved["responses"]) == {"captured_runtime", SELECTED},
            "selected TEST response branch inventory differs")
    expected_paths = HELPER.protected_paths(source, groups)
    expected_responses = {arm: HELPER.protected_response(source, saved["responses"][arm])
                          for arm in ("captured_runtime", SELECTED)}
    require(saved["protected_paths"] == expected_paths
            and saved["protected_responses"] == expected_responses,
            "selected TEST protected-literal evidence differs")
    return support["supported"]

def summarize(predictions, labels, arm):
    require(arm in ("captured_runtime", SELECTED)
            and len(predictions) == len(labels) == ROWS
            and [row["id"] for row in predictions] == [row.get("id") for row in labels],
            "TEST metric identity differs")
    counts, categories, outcomes, latency = Counter(), defaultdict(Counter), [], []
    for row, label in zip(predictions, labels):
        require(row["request"] == {"input": label.get("input"), "context": ""},
                "late TEST source identity differs")
        expected = label.get("expected")
        require(isinstance(expected, list) and expected
                and all(isinstance(value, str) for value in expected), "TEST labels differ")
        texts = [candidate.get("text") for candidate in row["responses"][arm].get("candidates", [])[:5]
                 if isinstance(candidate, dict) and isinstance(candidate.get("text"), str)]
        top1 = bool(texts and texts[0] in expected)
        top5 = any(text in expected for text in texts)
        covered = bool(set(row["candidate_texts"] + [label["input"]]).intersection(expected))
        protected = row["protected_responses"][arm]["verified"]
        supported = row["support"]["supported"]
        values = {"rows": 1, "errors": 0, "top1": int(top1), "top5": int(top5),
                  "full_pool_covered": int(covered), "supported": int(supported),
                  "fallback": int(not supported), "protected_exact": int(protected)}
        counts.update(values)
        categories[label.get("category", "uncategorized")].update(values)
        outcomes.append({"id": row["id"], "actual": texts, "expected": expected,
                         "top1": top1, "top5": top5, "full_pool_covered": covered,
                         "supported": supported, "protected_exact": protected})
        latency.append(row["latency_ms"]["captured_runtime"] if arm == "captured_runtime"
                       else sum(row["latency_ms"][key] for key in
                                ("source_edge_proposal", "parent_features",
                                 "target_extraction", SELECTED)))
    latency.sort()
    return {"counts": dict(counts),
            "categories": {key: dict(value) for key, value in sorted(categories.items())},
            "latency": {"measured_rows": len(latency), "total_ms": sum(latency),
                        "mean_ms": sum(latency) / len(latency), "max_ms": max(latency),
                        "p95_ms": latency[math.ceil(.95 * len(latency)) - 1]},
            "outcomes": outcomes}


def paired_delta(before, after):
    left = {row["id"]: row for row in before["outcomes"]}
    right = {row["id"]: row for row in after["outcomes"]}
    require(set(left) == set(right), "TEST paired metric IDs differ")
    return {"top1_gain_ids": [key for key in left if right[key]["top1"] and not left[key]["top1"]],
            "top1_loss_ids": [key for key in left if left[key]["top1"] and not right[key]["top1"]],
            "top5_gain_ids": [key for key in left if right[key]["top5"] and not left[key]["top5"]],
            "top5_loss_ids": [key for key in left if left[key]["top5"] and not right[key]["top5"]],
            "protected_regression_ids": [key for key in left
                if left[key]["protected_exact"] and not right[key]["protected_exact"]]}


def decide(metrics):
    baseline, selected = metrics["captured_runtime"], metrics[SELECTED]
    delta = paired_delta(baseline, selected)
    base, counts = baseline["counts"], selected["counts"]
    require(base["top1"] == BASELINE["top1"] and base["top5"] == BASELINE["top5"],
            "captured 1,000-case baseline differs")
    gate = (counts["top1"] > BASELINE["top1"] and counts["top5"] >= BASELINE["top5"]
            and not delta["top1_loss_ids"] and not delta["top5_loss_ids"]
            and not delta["protected_regression_ids"] and counts["protected_exact"] == ROWS
            and counts["errors"] == 0)
    return {"selected_arm": SELECTED, "versus_captured_runtime": delta,
            "offline_1000_case_gate": gate,
            "gate_contract": {"top1_strictly_above": 953, "top5_at_least": 972,
                "no_top1_losses": True, "no_top5_losses": True,
                "no_protected_literal_failures": True, "zero_execution_errors": True},
            "production_qualified": False,
            "production_qualification_reason":
                "offline qualification still requires runtime integration and web/Tailscale verification"}


def expected_operations(predictions):
    names = ["load_parent_and_selected_residual"]; cached = set()
    for row in predictions:
        names.append("source_edge_proposal:" + row["id"])
        if row["support"]["supported"]:
            names.append("parent_features:" + row["id"])
            misses = [target for target in row["isolated_target_texts"] if target not in cached]
            if misses:
                names.append("target_extraction:" + row["id"]); cached.update(misses)
            names.append(SELECTED + ":" + row["id"])
    return names


def verify_resources(run, report, predictions):
    trace, expected = jsonl(run / "resources.jsonl"), expected_operations(predictions)
    baseline = report.get("baseline_swap_bytes")
    require(trace and type(baseline) is int and baseline >= 0, "TEST resource trace differs")
    next_index, current, previous = 0, None, -1.; final_seen = False; final_trace = None
    peaks = {"rss": 0, "swap": 0, "operation": 0.}
    for row in trace:
        require(set(row) == {"at", "elapsed_s", "rss_bytes", "swap_used_bytes",
                             "swap_growth_bytes", "operation_elapsed_s", "stage", "operation"}
                and row["stage"] in {"operation_start", "operation_end", "watchdog", "final"}
                and finite(row["elapsed_s"]) and row["elapsed_s"] >= previous
                and type(row["rss_bytes"]) is int and row["rss_bytes"] >= 0
                and type(row["swap_used_bytes"]) is int and row["swap_used_bytes"] >= 0
                and type(row["swap_growth_bytes"]) is int
                and row["swap_used_bytes"] - baseline == row["swap_growth_bytes"],
                "TEST resource observation differs")
        previous = row["elapsed_s"]
        require(row["elapsed_s"] <= LIMITS["whole_seconds"]
                and row["rss_bytes"] <= LIMITS["rss_bytes"]
                and row["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"],
                "TEST resource hard bound exceeded")
        if row["operation"] is None:
            require(row["stage"] in {"watchdog", "final"}
                    and row["operation_elapsed_s"] is None,
                    "TEST idle watchdog attribution differs")
        else:
            require(finite(row["operation_elapsed_s"])
                    and 0 <= row["operation_elapsed_s"] <= LIMITS["update_seconds"],
                    "TEST operation time bound exceeded")
        if row["stage"] == "final":
            require(not final_seen and current is None and next_index == len(expected),
                    "TEST final resource order differs")
            final_seen = True; final_trace = row
        elif row["stage"] == "operation_start":
            require(not final_seen and current is None and next_index < len(expected)
                    and row["operation"] == expected[next_index],
                    "TEST resource operation start order differs")
            current = row["operation"]; next_index += 1
        elif row["stage"] == "operation_end":
            require(not final_seen and current is not None and row["operation"] == current,
                    "TEST resource operation pairing differs")
            current = None
        else:
            require(row["operation"] == current, "TEST atomic watchdog transition differs")
        peaks["rss"] = max(peaks["rss"], row["rss_bytes"])
        peaks["swap"] = max(peaks["swap"], row["swap_growth_bytes"])
        if row["operation_elapsed_s"] is not None:
            peaks["operation"] = max(peaks["operation"], row["operation_elapsed_s"])
    require(current is None and next_index == len(expected) and final_seen,
            "TEST resource inventory/final sample differs")
    final = report.get("final_resources", {})
    require(final == final_trace and finite(report.get("elapsed_s"))
            and previous <= report["elapsed_s"] <= LIMITS["whole_seconds"],
            "TEST final resources differ")
    return {"observations": len(trace), "operation_pairs": len(expected),
            "supported_rows": sum(row["support"]["supported"] for row in predictions),
            "peak_rss_bytes": max(peaks["rss"], final["rss_bytes"]),
            "peak_swap_growth_bytes": max(peaks["swap"], final["swap_growth_bytes"]),
            "peak_operation_seconds": peaks["operation"],
            "atomic_operation_attribution_verified": True}

def verify(run):
    run = Path(run).resolve()
    require(sha(PRODUCER) == PRODUCER_SHA256, "frozen selected TEST producer changed")
    # Do not parse predictions, proposals, or TEST labels before this gate.
    report = read(run / "report.json")
    closure_path = run / "source-only.complete.json"
    closure = read(closure_path)
    proposal_path, prediction_path = run / "proposals.jsonl", run / "predictions.jsonl"
    proposal_sha, prediction_sha = sha(proposal_path), sha(prediction_path)
    require(report.get("schema") == RUN_SCHEMA and report.get("completed") is True
            and report.get("split") == "test" and report.get("rows") == ROWS
            and report.get("selected_arm") == SELECTED and report.get("limits") == LIMITS,
            "completed selected TEST report differs")
    require(closure == {"schema": RUN_SCHEMA + "-source-only-closure", "rows": ROWS,
        "proposals_sha256": proposal_sha, "predictions_sha256": prediction_sha,
        "source_only": True, "targets_joined": False, "test_opened": False,
        "selected_arm": SELECTED,
        "candidate_policy": "fresh_source_edge_recombination_then_top4_plus_raw",
        "source_byte_limit": 2048, "full_candidate_limit": 256,
        "isolated_target_limit": 256,
        "input_truncation": False}
        and report.get("proposals_sha256") == proposal_sha
        and report.get("predictions_sha256") == prediction_sha
        and report.get("closure_sha256") == sha(closure_path),
        "selected TEST source-only closure differs")
    require(report.get("scope") == {
        "source_only_streams_closed_before_target_join": True,
        "existing_exposed_1000_case_regression": True,
        "fresh_original_generators_run": False, "cached_pipeline_reused": True,
        "fresh_source_edge_proposals": True, "production_changed": False,
        "input_truncation": False, "isolated_target_vectors_cached_by_text_only": True,
        "edge_scores_not_cached_across_requests": True}, "selected TEST scope differs")
    target_path, identity = validate_authorization(run, report)

    runtime_rows = jsonl(RUNTIME / "capture.jsonl")
    proposals, predictions = jsonl(proposal_path), jsonl(prediction_path)
    require(len(runtime_rows) == len(proposals) == len(predictions) == ROWS,
            "selected TEST source-only denominator differs")
    cached = set()
    supported = [verify_prediction(saved, runtime, proposal, cached)
                 for saved, runtime, proposal in zip(predictions, runtime_rows, proposals)]
    require(len({row["id"] for row in predictions}) == ROWS,
            "selected TEST prediction IDs are duplicated")
    cache_summary = {"unique_texts": len(cached), "limit_bytes": 256 * 1024**2,
        "tensor_bytes": len(cached) * (256 * 4 + 1),
        "requests": sum(row["target_cache"]["requested"] for row in predictions),
        "hits": sum(row["target_cache"]["hits"] for row in predictions),
        "misses": sum(row["target_cache"]["misses"] for row in predictions)}
    require(cache_summary["tensor_bytes"] <= cache_summary["limit_bytes"]
            and report.get("target_cache") == cache_summary,
            "selected TEST target cache summary differs")

    # First TEST target byte access in this verifier, after the closure and replay.
    require(sha(target_path) == TEST_SHA256, "canonical TEST labels changed")
    labels = jsonl(target_path)
    metrics = {arm: summarize(predictions, labels, arm)
               for arm in ("captured_runtime", SELECTED)}
    require(metrics["captured_runtime"]["counts"]["top1"] == 953
            and metrics["captured_runtime"]["counts"]["top5"] == 972,
            "captured TEST baseline differs")
    require(report.get("metrics") == metrics, "reported TEST metrics differ")
    decision = decide(metrics)
    require(report.get("decision") == decision, "reported TEST gate differs")
    protected_rows = [row for row in predictions if HELPER.fixtures(row["request"]["input"])]
    protected_summary = {"rows": len(protected_rows),
        "fixture_counts": dict(Counter(fixture["kind"] for row in protected_rows
            for fixture in HELPER.fixtures(row["request"]["input"]))),
        "all_observed_candidate_paths_verified": all(
            row["protected_paths"]["verified"] for row in protected_rows),
        "response_failures": {arm: [row["id"] for row in protected_rows
            if not row["protected_responses"][arm]["verified"]]
            for arm in ("captured_runtime", SELECTED)},
        "production_runtime_preservation_verified": False}
    require(report.get("protected_literal_contract") == protected_summary,
            "reported TEST protected-literal summary differs")
    resources = verify_resources(run, report, predictions)
    outputs = {name: sha(run / name) for name in
               ("pins.json", "proposals.jsonl", "predictions.jsonl",
                "source-only.complete.json", "resources.jsonl")}
    require(report.get("outputs") == outputs
            and not (run / "failure.json").exists()
            and set(path.name for path in run.iterdir() if path.is_file()) <= {
                *outputs, "report.json", "independent-verification.json"},
            "selected TEST output inventory differs")
    return {"schema": SCHEMA, "verified": True,
        "evaluation_report_sha256": sha(run / "report.json"),
        "proposals_sha256": proposal_sha, "predictions_sha256": prediction_sha,
        "closure_sha256": sha(closure_path),
        "producer_sha256": sha(PRODUCER),
        "verifier_sha256": sha(Path(__file__)), "identity": identity,
        "counts": {"rows": ROWS, "supported": sum(supported),
                   "fallback": ROWS - sum(supported), "protected_rows": len(protected_rows),
                   "unique_target_texts": len(cached)},
        "metrics": {arm: metrics[arm]["counts"] for arm in metrics},
        "decision": decision, "resources": resources,
        "contracts": {"completed_report_and_source_only_closure_checked_before_labels": True,
            "fresh_source_edge_proposals_deterministically_replayed": True,
            "candidate_domain_all_paths_and_isolated_targets_reconstructed": True,
            "target_cache_sequence_replayed": True,
            "shared_edge_ids_scores_weighted_paths_winners_and_fp32_scores_replayed": True,
            "ranking_and_fallback_replayed": True,
            "protected_literals_replayed_from_spans": True,
            "metrics_categories_deltas_and_gate_recomputed": True,
            "existing_exposed_1000_case_regression": True,
            "test_labels_opened_only_after_closed_replay": True,
            "original_generators_rerun": False, "cached_pipeline_reused": True,
            "torch_imported": False, "tokenizer_instantiated": False,
            "neural_model_instantiated": False, "neural_forward_rerun": False,
            "production_changed": False, "production_qualified": False},
        "limitations": [
            "The selected neural checkpoint forward, edge scoring, and tokenizer support decision are not rerun.",
            "Fresh source-edge proposals are replayed, but the original candidate pipeline is a verified cached capture.",
            "The 1,000-case suite is an existing exposed regression rather than a blind generalization set.",
            "An offline gate cannot establish production runtime wiring, resource parity, localhost behavior, or access from another Tailscale peer."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    output = args.output or args.run / "independent-verification.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"verified": True, "counts": result["counts"],
                      "metrics": result["metrics"], "decision": result["decision"],
                      "resources": result["resources"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
