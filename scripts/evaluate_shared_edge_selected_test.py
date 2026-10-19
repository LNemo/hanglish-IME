"""Closed 1,000-case regression for the DEV-selected shared-edge residual."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_shared_edge_residual import (
    TARGET_CACHE_BYTES,
    common_support_reason,
    extend_target_cache,
    isolated_target_texts,
    validate_training_artifacts,
)
from scripts.evaluate_source_span_pair import (
    collect_candidate_groups, compact_public, protected_fixtures,
    response_protects_literals, verify_paths_protect_literals,
)
from scripts.evaluate_source_span_selected_test import (
    RUNTIME_HASHES, SOURCE_GENERATOR, SOURCE_GENERATOR_SHA256, TEST_SHA256,
    append_jsonl, iter_jsonl, make_proposal_row, validate_runtime_artifacts,
)
from scripts.preflight_source_span_residual import PARENT, PARENT_HASHES, require, sha, write
from scripts.score_source_edge_recombination import apply_policy


SCHEMA = "shared-edge-selected-test-evaluation-v1"
AUTHORIZATION_SCHEMA = "shared-edge-selected-test-prelaunch-authorization-v1"
SELECTED_ARM = "shared_edge"
EXPECTED_ROWS = 1_000
BASELINE = {"top1": 953, "top5": 972}
PLAN = ROOT / ".omx/plans/shared-edge-residual-v1.md"
DEFAULT_TRAINING = ROOT / ".omx/artifacts/shared-edge-residual-training-v1"
DEFAULT_DEV = ROOT / ".omx/artifacts/shared-edge-residual-dev-v1"
DEFAULT_RUNTIME = ROOT / ".omx/artifacts/extended-runtime-test-v1"
DEV_REPORT_SHA256 = "65138dc85c01480cb8d9c8ab67a9a288c77252934f9db3f7841b65b0cf147dad"
DEV_PROOF_SHA256 = "6506d584f276da6311f1c464fa0374c88faa4a0fe123a32565dc75ee2f8a4d55"
TRAINING_REPORT_SHA256 = "481ff33a1ae6e9f0e8f3d5c24566f5e076a42d29dd02145e1ad964fd6a06e5da"
TRAINING_PROOF_SHA256 = "e4d48a8be5373198e83a11bc88793f6262520f6a2b3b075da2d3a1d8e28cf0e1"
SELECTED_RESIDUAL_SHA256 = "a7282d5fba58b2e2bd3b01211cd0397d2a53fba6d695044495ba34c3f26c957d"
EVAL_LIMITS = {"whole_seconds": 1800.0, "update_seconds": 60.0,
               "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}


def finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def canonical_hash(value) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_dev_selection(directory: Path) -> dict:
    directory = Path(directory); report_path = directory / "report.json"
    proof_path = directory / "independent-verification.json"
    require(report_path.is_file() and proof_path.is_file()
            and sha(report_path) == DEV_REPORT_SHA256 and sha(proof_path) == DEV_PROOF_SHA256,
            "shared-edge DEV artifacts differ")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    expected = {"captured_runtime": (264, 284), "parent": (263, 280),
                "source_local_control": (271, 288), "shared_edge": (272, 288)}
    require(report.get("schema") == "shared-edge-residual-dev-evaluation-v1"
            and report.get("completed") is True and report.get("split") == "dev"
            and report.get("rows") == 316
            and report.get("scope", {}).get("test_opened") is False
            and report.get("scope", {}).get("production_changed") is False
            and report.get("decision", {}).get("selected_for_1000_case_regression") == SELECTED_ARM
            and report.get("decision", {}).get("eligible_for_1000_case_regression") is True,
            "shared-edge DEV selection gate differs")
    for arm, (top1, top5) in expected.items():
        counts = report.get("metrics", {}).get(arm, {}).get("counts", {})
        require(counts.get("top1") == top1 and counts.get("top5") == top5,
                "shared-edge DEV metric differs: " + arm)
    contracts = proof.get("contracts", {})
    require(proof.get("schema") == "shared-edge-residual-dev-independent-verification-v1"
            and proof.get("verified") is True
            and proof.get("report_sha256") == DEV_REPORT_SHA256
            and proof.get("decision") == report.get("decision")
            and proof.get("metrics", {}).get("shared_edge", {}).get("top1") == 272
            and proof.get("metrics", {}).get("shared_edge", {}).get("top5") == 288
            and contracts.get("metrics_categories_gain_loss_and_decision_recomputed") is True
            and contracts.get("historical_gold_bearing_dev_report_opened") is False
            and contracts.get("test_opened") is False,
            "shared-edge DEV independent proof differs")
    relied = [ROOT / "scripts/evaluate_shared_edge_residual.py",
        ROOT / "scripts/train_shared_edge_residual.py",
        ROOT / "hanglish/models/shared_edge_residual.py",
        ROOT / "scripts/source_span_resources.py",
        ROOT / "hanglish/models/span_token_bridge.py",
        ROOT / "hanglish/core/source_correspondence.py"]
    pins = {str(report_path): DEV_REPORT_SHA256, str(proof_path): DEV_PROOF_SHA256}
    for path in relied:
        digest = report.get("pins", {}).get(str(path))
        require(isinstance(digest, str) and sha(path) == digest,
                "DEV-evaluated dependency differs: " + str(path))
        pins[str(path)] = digest
    return {"report": report, "proof": proof, "pins": pins}


def validate_selected_training(dev_report: dict, training: Path) -> dict:
    auth = validate_training_artifacts(training)
    report_path = Path(auth["report_path"]); residual_path = Path(auth["residual_path"])
    require(sha(report_path) == TRAINING_REPORT_SHA256
            and sha(auth["proof_path"]) == TRAINING_PROOF_SHA256
            and sha(residual_path) == SELECTED_RESIDUAL_SHA256
            and dev_report.get("training_report_sha256") == TRAINING_REPORT_SHA256
            and auth["architecture"].get("schema") == "shared-source-edge-residual-v1",
            "DEV-selected shared-edge checkpoint differs")
    return auth


def authorization_payload(training: Path, dev: Path, runtime: Path) -> dict:
    dev_auth = validate_dev_selection(dev)
    training_auth = validate_selected_training(dev_auth["report"], training)
    runtime_auth = validate_runtime_artifacts(runtime)
    return {"schema": AUTHORIZATION_SCHEMA, "rows": EXPECTED_ROWS,
        "limits": EVAL_LIMITS, "target_cache_bytes": TARGET_CACHE_BYTES,
        "evaluator_sha256": sha(Path(__file__).resolve()),
        "training_report_sha256": sha(training_auth["report_path"]),
        "training_proof_sha256": sha(training_auth["proof_path"]),
        "selected_residual_sha256": sha(training_auth["residual_path"]),
        "dev_report_sha256": DEV_REPORT_SHA256, "dev_proof_sha256": DEV_PROOF_SHA256,
        "runtime_artifacts": RUNTIME_HASHES,
        "runtime_source_authorization_sha256": canonical_hash(runtime_auth["pins"]),
        "source_generator_sha256": SOURCE_GENERATOR_SHA256,
        "model_sha256": sha(ROOT / "hanglish/models/shared_edge_residual.py")}


def consume_authorization(path: Path, expected: dict) -> dict:
    path = Path(path).resolve(); require(path.is_file(), "TEST prelaunch authorization is absent")
    require(json.loads(path.read_text(encoding="utf-8")) == expected,
            "TEST prelaunch authorization differs")
    return {str(path): sha(path)}


def build_prediction(runtime_row: dict, proposal_row: dict, texts: list[str],
                     representatives: list[dict], groups: list[list], targets: list[str],
                     scores: list[float] | None, base_scores: list[float] | None,
                     reason: str | None,
                     latency_ms: dict[str, float], shared_audit: dict | None,
                     target_cache: dict) -> dict:
    source = runtime_row["request"]["input"]; original = runtime_row["proposed"]
    require(set(latency_ms) == {"captured_runtime", "source_edge_proposal",
            "parent_features", "target_extraction", SELECTED_ARM}
            and all(finite(value) and value >= 0 for value in latency_ms.values()),
            "TEST latency record differs")
    if reason is None:
        require(scores is not None and base_scores is not None
                and len(scores) == len(base_scores) == len(texts)
                and all(finite(value) for value in scores + base_scores)
                and isinstance(shared_audit, dict)
                and set(shared_audit) == {"edge_ids", "edge_scores", "residuals",
                                          "winning_path_ids"},
                "selected shared-edge outputs differ")
        applied = apply_policy(original, representatives, dict(zip(texts, scores)),
            supported=True, source=source, model_id=SELECTED_ARM)
        require(applied["applied"] is True, "selected shared-edge policy was not applied")
        selected = applied["response"]; score_values = list(map(float, scores))
        base_values = list(map(float, base_scores))
    else:
        require(scores is None and base_scores is None and shared_audit is None,
                "fallback prediction may not contain model outputs")
        selected = deepcopy(original); score_values = []; base_values = []
    proposal_ids = [row.get("proposal_id") for row in proposal_row.get("proposals", [])]
    candidate_ids = [row.get("candidate_id") or row.get("proposal_id") for row in representatives]
    path_ids = [[path.path_id for path in group] for group in groups]
    require(all(isinstance(value, str) for value in proposal_ids + candidate_ids)
            and len(candidate_ids) == len(texts)
            and all(group and len(group) == len(set(group)) for group in path_ids),
            "selected TEST candidate/path identities differ")
    require(set(target_cache) == {"requested", "hits", "misses"}
            and target_cache["requested"] == len(targets)
            and target_cache["hits"] + target_cache["misses"] == len(targets),
            "selected TEST target cache accounting differs")
    return {"id": runtime_row["id"], "request": deepcopy(runtime_row["request"]),
        "proposal_schema_version": proposal_row.get("schema_version"),
        "proposal_ids": proposal_ids, "candidate_texts": list(texts),
        "candidate_ids": candidate_ids, "path_ids": path_ids,
        "isolated_target_texts": list(targets), "target_cache": dict(target_cache),
        "support": {"supported": reason is None, "reason": reason},
        "scores": score_values, "base_scores": base_values,
        "latency_ms": dict(latency_ms),
        "shared_edge_audit": deepcopy(shared_audit),
        "responses": {"captured_runtime": compact_public(original),
                      SELECTED_ARM: compact_public(selected)},
        "protected_paths": verify_paths_protect_literals(source, groups),
        "protected_responses": {"captured_runtime": response_protects_literals(source, original),
                                SELECTED_ARM: response_protects_literals(source, selected)}}


def response_result(response: dict, expected: list[str]):
    texts = [row.get("text") for row in response.get("candidates", [])[:5]
             if isinstance(row, dict) and isinstance(row.get("text"), str)]
    return texts, bool(texts and texts[0] in expected), any(text in expected for text in texts)


def summarize(predictions: list[dict], labels: list[dict], arm: str) -> dict:
    require(arm in ("captured_runtime", SELECTED_ARM)
            and len(predictions) == len(labels) == EXPECTED_ROWS
            and [row["id"] for row in predictions] == [row.get("id") for row in labels],
            "TEST metric denominator or identity differs")
    counts = Counter(); categories = defaultdict(Counter); outcomes = []; latency = []
    for row, label in zip(predictions, labels):
        require(row["request"] == {"input": label.get("input"), "context": ""},
                "late TEST source identity differs")
        expected = label.get("expected")
        require(isinstance(expected, list) and expected
                and all(isinstance(value, str) for value in expected),
                "TEST expected values differ")
        texts, top1, top5 = response_result(row["responses"][arm], expected)
        covered = bool(set(row["candidate_texts"] + [label["input"]]).intersection(expected))
        protected = row["protected_responses"][arm]["verified"]
        supported = row["support"]["supported"]
        values = {"rows": 1, "errors": 0, "top1": int(top1), "top5": int(top5),
            "full_pool_covered": int(covered), "supported": int(supported),
            "fallback": int(not supported), "protected_exact": int(protected)}
        counts.update(values); categories[label.get("category", "uncategorized")].update(values)
        outcomes.append({"id": row["id"], "actual": texts, "expected": expected,
            "top1": top1, "top5": top5, "full_pool_covered": covered,
            "supported": supported, "protected_exact": protected})
        keys = (("captured_runtime",) if arm == "captured_runtime" else
                ("source_edge_proposal", "parent_features", "target_extraction", SELECTED_ARM))
        latency.append(sum(row["latency_ms"][key] for key in keys))
    latency.sort()
    return {"counts": dict(counts),
        "categories": {key: dict(value) for key, value in sorted(categories.items())},
        "latency": {"measured_rows": len(latency), "total_ms": sum(latency),
            "mean_ms": sum(latency) / len(latency), "max_ms": max(latency),
            "p95_ms": latency[math.ceil(.95 * len(latency)) - 1]},
        "outcomes": outcomes}


def paired_delta(before: dict, after: dict) -> dict:
    left = {row["id"]: row for row in before["outcomes"]}
    right = {row["id"]: row for row in after["outcomes"]}
    require(set(left) == set(right), "TEST paired metric IDs differ")
    return {"top1_gain_ids": [key for key in left if right[key]["top1"] and not left[key]["top1"]],
        "top1_loss_ids": [key for key in left if left[key]["top1"] and not right[key]["top1"]],
        "top5_gain_ids": [key for key in left if right[key]["top5"] and not left[key]["top5"]],
        "top5_loss_ids": [key for key in left if left[key]["top5"] and not right[key]["top5"]],
        "protected_regression_ids": [key for key in left
            if left[key]["protected_exact"] and not right[key]["protected_exact"]]}


def decide(metrics: dict[str, dict]) -> dict:
    baseline, selected = metrics["captured_runtime"], metrics[SELECTED_ARM]
    delta = paired_delta(baseline, selected); counts = selected["counts"]
    require(baseline["counts"]["top1"] == BASELINE["top1"]
            and baseline["counts"]["top5"] == BASELINE["top5"],
            "captured 1,000-case baseline differs")
    passed = (counts["top1"] > BASELINE["top1"] and counts["top5"] >= BASELINE["top5"]
        and not delta["top1_loss_ids"] and not delta["top5_loss_ids"]
        and not delta["protected_regression_ids"]
        and counts["protected_exact"] == EXPECTED_ROWS and counts["errors"] == 0)
    return {"selected_arm": SELECTED_ARM, "versus_captured_runtime": delta,
        "offline_1000_case_gate": passed,
        "gate_contract": {"top1_strictly_above": 953, "top5_at_least": 972,
            "no_top1_losses": True, "no_top5_losses": True,
            "no_protected_literal_failures": True, "zero_execution_errors": True},
        "production_qualified": False,
        "production_qualification_reason":
            "offline qualification still requires runtime integration and web/Tailscale verification"}


def run(training: Path, dev: Path, runtime: Path, output: Path, authorization: Path) -> dict:
    output = Path(output); output.mkdir(parents=True, exist_ok=False); started = time.monotonic()
    try:
        dev_auth = validate_dev_selection(dev)
        training_auth = validate_selected_training(dev_auth["report"], training)
        runtime_auth = validate_runtime_artifacts(runtime)
        authorization_pins = consume_authorization(
            authorization, authorization_payload(training, dev, runtime))
        from safetensors.torch import load_file
        import torch
        from hanglish.models.sentence_ranker import SentenceRanker
        from hanglish.models.shared_edge_residual import FrozenEdgeFeatures, SharedEdgeResidualRanker
        from hanglish.models.span_token_bridge import encode_aligned_candidates
        from scripts.preflight_pko_seq2seq import system_swap_bytes
        from scripts.source_span_resources import AtomicResourceWatchdog

        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        pins = dict(training_auth["pins"]); pins.update(training_auth["artifact_pins"])
        pins.update(runtime_auth["pins"]); pins.update(runtime_auth["artifact_pins"])
        pins.update(dev_auth["pins"]); pins.update(authorization_pins)
        tracked = [Path(__file__).resolve(), PLAN, SOURCE_GENERATOR,
            ROOT / "scripts/evaluate_shared_edge_residual.py",
            ROOT / "scripts/evaluate_source_span_selected_test.py",
            ROOT / "scripts/score_source_edge_recombination.py",
            ROOT / "scripts/source_span_resources.py",
            ROOT / "hanglish/models/shared_edge_residual.py",
            ROOT / "hanglish/models/source_span_residual.py",
            ROOT / "hanglish/models/span_token_bridge.py",
            ROOT / "hanglish/core/source_correspondence.py"]
        pins.update({str(path): sha(path) for path in tracked})
        write(output / "pins.json", dict(sorted(pins.items())))
        baseline_swap = system_swap_bytes()
        watchdog = AtomicResourceWatchdog(started, baseline_swap, output / "resources.jsonl",
            interval_s=2.0, limits=EVAL_LIMITS)
        predictions = []; target_vectors = {}; target_flags = {}
        with watchdog:
            with watchdog.operation("load_parent_and_selected_residual", update=True):
                parent = SentenceRanker.load(PARENT); model = SharedEdgeResidualRanker(parent)
                require(model.architecture == training_auth["architecture"],
                        "selected shared-edge architecture differs")
                model.load_residual_state_dict(load_file(training_auth["residual_path"])); model.eval()
            proposal_path = output / "proposals.jsonl"; prediction_path = output / "predictions.jsonl"
            seen = set()
            with (proposal_path.open("x", encoding="utf-8") as proposal_stream,
                  prediction_path.open("x", encoding="utf-8") as prediction_stream):
                for runtime_row in iter_jsonl(Path(runtime) / "capture.jsonl"):
                    identifier = runtime_row.get("id")
                    require(isinstance(identifier, str) and identifier not in seen,
                            "captured TEST ID is missing or duplicated")
                    seen.add(identifier); before = time.monotonic()
                    with watchdog.operation("source_edge_proposal:" + identifier, update=True):
                        proposal_row = make_proposal_row(runtime_row)
                    proposal_ms = (time.monotonic() - before) * 1000; append_jsonl(proposal_stream, proposal_row)
                    texts, representatives, groups = collect_candidate_groups(runtime_row, proposal_row)
                    source = runtime_row["request"]["input"]; targets = isolated_target_texts(groups)
                    reason = common_support_reason(source, texts, targets, lambda values: (
                        encode_aligned_candidates(parent.tokenizer, values, 256), True)[1])
                    captured_ms = runtime_row.get("elapsed_ms")
                    require(finite(captured_ms) and captured_ms >= 0, "captured TEST latency differs")
                    latency = {"captured_runtime": float(captured_ms),
                        "source_edge_proposal": proposal_ms, "parent_features": 0.0,
                        "target_extraction": 0.0, SELECTED_ARM: 0.0}
                    misses = [text for text in targets if text not in target_vectors]
                    cache_counts = {"requested": len(targets), "hits": len(targets) - len(misses),
                                    "misses": len(misses)}
                    scores = base_scores = audit = None
                    if reason is None:
                        before = time.monotonic()
                        with watchdog.operation("parent_features:" + identifier, update=True):
                            base = model.extract_features(texts)
                        latency["parent_features"] = (time.monotonic() - before) * 1000
                        if misses:
                            before = time.monotonic()
                            with watchdog.operation("target_extraction:" + identifier, update=True):
                                extend_target_cache(model, misses, target_vectors, target_flags)
                            latency["target_extraction"] = (time.monotonic() - before) * 1000
                        edge_features = FrozenEdgeFeatures(tuple(targets),
                            torch.stack([target_vectors[text] for text in targets]),
                            torch.stack([target_flags[text] for text in targets]))
                        before = time.monotonic()
                        with watchdog.operation(SELECTED_ARM + ":" + identifier, update=True), torch.no_grad():
                            result = model(source, groups, base_features=base, edge_features=edge_features)
                        latency[SELECTED_ARM] = (time.monotonic() - before) * 1000
                        scores = result["scores"].tolist()
                        base_scores = result["base_scores"].tolist()
                        audit = {"edge_ids": [list(edge) for edge in result["edge_ids"]],
                            "edge_scores": result["edge_scores"].tolist(),
                            "residuals": result["residuals"].tolist(),
                            "winning_path_ids": [list(value) for value in result["winning_path_ids"]]}
                    prediction = build_prediction(runtime_row, proposal_row, texts, representatives,
                        groups, targets, scores, base_scores, reason, latency, audit, cache_counts)
                    append_jsonl(prediction_stream, prediction); predictions.append(prediction)
                    if len(predictions) % 25 == 0:
                        print(json.dumps({"captured": len(predictions), "rows": EXPECTED_ROWS,
                                          "elapsed_s": time.monotonic() - started}), flush=True)
                require(len(predictions) == EXPECTED_ROWS, "selected TEST denominator differs")
                os.fsync(proposal_stream.fileno()); os.fsync(prediction_stream.fileno())
            proposal_sha, prediction_sha = sha(proposal_path), sha(prediction_path)
            closure = {"schema": SCHEMA + "-source-only-closure", "rows": EXPECTED_ROWS,
                "proposals_sha256": proposal_sha, "predictions_sha256": prediction_sha,
                "source_only": True, "targets_joined": False, "test_opened": False,
                "selected_arm": SELECTED_ARM,
                "candidate_policy": "fresh_source_edge_recombination_then_top4_plus_raw",
                "source_byte_limit": 2048, "full_candidate_limit": 256,
                "isolated_target_limit": 256, "input_truncation": False}
            write(output / "source-only.complete.json", closure); closure_sha = sha(output / "source-only.complete.json")
            require(sha(runtime_auth["target_path"]) == runtime_auth["target_sha256"],
                    "canonical TEST target changed before late join")
            labels = list(iter_jsonl(runtime_auth["target_path"]))
            metrics = {arm: summarize(predictions, labels, arm)
                       for arm in ("captured_runtime", SELECTED_ARM)}
            decision = decide(metrics)
            protected_rows = [row for row in predictions if protected_fixtures(row["request"]["input"])]
            protected_summary = {"rows": len(protected_rows),
                "fixture_counts": dict(Counter(fixture["kind"] for row in protected_rows
                    for fixture in protected_fixtures(row["request"]["input"]))),
                "all_observed_candidate_paths_verified": all(row["protected_paths"]["verified"]
                                                               for row in protected_rows),
                "response_failures": {arm: [row["id"] for row in protected_rows
                    if not row["protected_responses"][arm]["verified"]]
                    for arm in ("captured_runtime", SELECTED_ARM)},
                "production_runtime_preservation_verified": False}
            final_resources = watchdog.sample_and_append("final")
            require(final_resources.get("stage") == "final"
                    and final_resources.get("operation") is None
                    and final_resources.get("operation_elapsed_s") is None
                    and finite(final_resources.get("elapsed_s"))
                    and 0 <= final_resources["elapsed_s"] <= EVAL_LIMITS["whole_seconds"]
                    and type(final_resources.get("rss_bytes")) is int
                    and 0 <= final_resources["rss_bytes"] <= EVAL_LIMITS["rss_bytes"]
                    and type(final_resources.get("swap_used_bytes")) is int
                    and type(final_resources.get("swap_growth_bytes")) is int
                    and final_resources["swap_used_bytes"] - baseline_swap
                        == final_resources["swap_growth_bytes"]
                    and final_resources["swap_growth_bytes"]
                        <= EVAL_LIMITS["swap_growth_bytes"],
                    "final TEST resource observation differs")
        for raw, digest in pins.items():
            require(Path(raw).is_file() and sha(Path(raw)) == digest,
                    "selected TEST evaluation pin changed: " + raw)
        elapsed = time.monotonic() - started
        require(elapsed <= EVAL_LIMITS["whole_seconds"], "whole TEST evaluation time exceeded")
        outputs = {name: sha(output / name) for name in ("pins.json", "proposals.jsonl",
            "predictions.jsonl", "source-only.complete.json", "resources.jsonl")}
        report = {"schema": SCHEMA, "completed": True, "split": "test",
            "rows": EXPECTED_ROWS, "selected_arm": SELECTED_ARM,
            "scope": {"source_only_streams_closed_before_target_join": True,
                "existing_exposed_1000_case_regression": True,
                "fresh_original_generators_run": False, "cached_pipeline_reused": True,
                "fresh_source_edge_proposals": True, "production_changed": False,
                "input_truncation": False, "isolated_target_vectors_cached_by_text_only": True,
                "edge_scores_not_cached_across_requests": True},
            "metrics": metrics, "decision": decision,
            "protected_literal_contract": protected_summary,
            "target_cache": {"unique_texts": len(target_vectors),
                "limit_bytes": TARGET_CACHE_BYTES,
                "tensor_bytes": sum(value.numel() * value.element_size()
                    for value in [*target_vectors.values(), *target_flags.values()]),
                "requests": sum(row["target_cache"]["requested"] for row in predictions),
                "hits": sum(row["target_cache"]["hits"] for row in predictions),
                "misses": sum(row["target_cache"]["misses"] for row in predictions)},
            "proposals_sha256": proposal_sha, "predictions_sha256": prediction_sha,
            "closure_sha256": closure_sha,
            "training_report_sha256": TRAINING_REPORT_SHA256,
            "training_proof_sha256": TRAINING_PROOF_SHA256,
            "selected_residual_sha256": SELECTED_RESIDUAL_SHA256,
            "dev_report_sha256": DEV_REPORT_SHA256, "dev_proof_sha256": DEV_PROOF_SHA256,
            "runtime_report_sha256": RUNTIME_HASHES["report.json"],
            "runtime_proof_sha256": RUNTIME_HASHES["independent-verification.json"],
            "test_labels_sha256": runtime_auth["target_sha256"],
            "pins": dict(sorted(pins.items())), "outputs": outputs,
            "limits": EVAL_LIMITS, "baseline_swap_bytes": baseline_swap,
            "final_resources": final_resources, "elapsed_s": elapsed,
            "limitations": ["The 1,000-case suite is exposed and not blind generalization",
                "Original generators are not rerun; the verified cached pipeline is reused",
                "A passing offline gate still requires runtime/API/web/Tailscale verification"]}
        write(output / "report.json", report)
        print(json.dumps({"completed": True,
            "metrics": {arm: value["counts"] for arm, value in metrics.items()},
            "decision": decision}, ensure_ascii=False), flush=True)
        return report
    except BaseException as error:
        write(output / "failure.json", {"schema": SCHEMA, "completed": False,
            "error_type": type(error).__name__, "error": str(error),
            "elapsed_s": time.monotonic() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--dev", type=Path, default=DEFAULT_DEV)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        require(args.authorization is None, "prepare-only does not consume authorization")
        write(args.output, authorization_payload(args.training, args.dev, args.runtime))
    else:
        require(args.authorization is not None, "TEST prelaunch authorization required")
        run(args.training, args.dev, args.runtime, args.output, args.authorization)
