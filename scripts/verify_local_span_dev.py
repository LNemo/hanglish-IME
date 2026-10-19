#!/usr/bin/env python3
"""Independent verifier for the closed local-span DEV evaluation.

The completed report and source-only closure are authenticated before parsing
DEV labels.  The historical gold-bearing DEV report is never opened.  Pure
domain/policy helpers come from a hash-pinned offline verifier; no evaluator,
Torch, tokenizer, or model code is imported and no neural forward is rerun.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".omx/artifacts/local-span-objective-dev-v1"
TRAINING = ROOT / ".omx/artifacts/local-span-objective-training-v1"
CONTROL_TRAINING = ROOT / ".omx/artifacts/source-span-paired-training-v1"
CONTROL_DEV = ROOT / ".omx/artifacts/source-span-paired-dev-v1"
RUNTIME = ROOT / ".omx/artifacts/extended-runtime-dev-v2"
PROPOSALS = ROOT / ".omx/artifacts/source-edge-recombination-dev-v1"
DEV = ROOT / "data/challenge/sentence-v3/dev.jsonl"
PLAN = ROOT / ".omx/plans/local-span-objective-v1.md"
HELPER_PATH = ROOT / ".omx/artifacts/verify-source-span-pair-dev-v1.py"
HELPER_SHA256 = "77373cca43d198a4fd47946cb8584e9dcb6089583a7a7ff2a47f48c1de2baa1d"
PRODUCER = ROOT / "scripts/evaluate_local_span_objective.py"
PRODUCER_SHA256 = "f421a7e6cf43ec34e3145612aedbcce8d04e31826f4959dedca485131778efd2"
SCHEMA = "local-span-objective-dev-independent-verification-v1"
RUN_SCHEMA = "local-span-objective-dev-evaluation-v1"
TRAIN_SCHEMA = "local-span-objective-training-v1"
TRAIN_PROOF_SCHEMA = "local-span-objective-training-independent-verification-v1"
ROWS = 316
ARMS = ("captured_runtime", "parent", "source_local_control", "local_aux")
MODELS = ("parent", "source_local_control", "local_aux")
EVAL_LIMITS = {"whole_seconds": 1800.0, "update_seconds": 60.0,
               "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
DEV_SHA256 = "593e1a032b54bf4d2c541da6754cfc68daf759cdcee7f8fa1f0a746459910ac9"
CONTROL_PREDICTIONS_SHA256 = "152cbab38ea99cd5f5e83967b2d528be004d772364db8eaaa16e292ac220e5a9"
CONTROL_CLOSURE_SHA256 = "523bdf989b0575fc439ffcd7d6818bf759c00a96d6536cb85dacc94293983be8"
RUNTIME_HASHES = {
    "report.json": "64fa054990b0dc78bde0bb20e5de0ee89fa7ae910ff8fc281cb16bf607387227",
    "capture.jsonl": "4e329278d18ed4ac0081be2dd2ac41261a1762e68c7a5691f778f07ce897b2f0",
    "capture.complete.json": "d8982c836832e07ba7b2bff07b01841f275154e350adfb4e232419f829ac92e1",
    "independent-verification.json": "42d16c9669af47bc3483c79eb1d171d70ec7d2daee2bbc01a198fb65c0c9ca91",
}
PROPOSAL_HASHES = {
    "report.json": "43bf273157242cad834b97af62b5576dfc5be71f6debfa4691f2f04b6371fb5b",
    "capture.jsonl": "f4d9cab9e9aebc00a788c71fc81dc5e8c2e732e0ec41ac3f858bf4c8f7ad8cc6",
    "capture.complete.json": "9c0e66295e7e43bf0e85212155bc0941010707ac2f110f09a07fa1bdb44398fa",
    "independent-verification.json": "4d2dc1968861b6577f8fa1cc1b2a4d6fbb698b709371d33ad2446cab50a8a30d",
    "pins.json": "45c40f36c520bff05a23d49ba92dea00b54be428f41a7d2e35e6adea3c295cc3",
}
OUTPUTS = {"pins.json", "predictions.jsonl", "predictions.complete.json",
           "resources.jsonl"}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    values = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSONL at {path}:{number}") from error
    return values


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _helper():
    need(sha(HELPER_PATH) == HELPER_SHA256, "frozen pure DEV helper changed")
    spec = importlib.util.spec_from_file_location("frozen_source_span_dev_verifier", HELPER_PATH)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def authenticate_training():
    report_path, proof_path = TRAINING / "report.json", TRAINING / "independent-verification.json"
    report, proof = read(report_path), read(proof_path)
    expected_contracts = {
        "fit_only": True, "hold_opened": False, "dev_opened": False,
        "test_opened": False, "production_changed": False,
        "historical_control_all_scores_exact": True,
        "score_fp32_base_plus_residual_replayed": True,
        "local_softplus_losses_replayed_from_saved_margins": True,
    }
    need(report.get("schema") == TRAIN_SCHEMA and report.get("completed") is True
         and report.get("rows") == 1483 and report.get("updates") == 2966
         and report.get("parent_before") == report.get("parent_after")
         and proof.get("schema") == TRAIN_PROOF_SCHEMA and proof.get("verified") is True
         and proof.get("report_sha256") == sha(report_path)
         and proof.get("producer_sha256")
             == sha(ROOT / "scripts/train_local_span_objective.py")
         and proof.get("counts") == {"selected_fit_rows": 1483, "preferences": 6889,
                                    "updates": 2966, "reload_rows": 16}
         and all(proof.get("contracts", {}).get(key) == value
                 for key, value in expected_contracts.items())
         and proof.get("tensors", {}).get("parent_unchanged") is True
         and proof.get("tensors", {}).get("initial_matches_historical_control") is True
         and proof.get("resources", {}).get("atomic_operation_attribution_verified") is True,
         "local training report/proof differs")
    need(report.get("contracts") == {"fit_only": True, "dev_opened": False,
         "test_opened": False, "hold_opened": False, "production_changed": False},
         "local training access contract differs")
    pins = report.get("pins")
    need(isinstance(pins, dict) and pins and read(TRAINING / "pins.json") == pins,
         "local training pin manifest differs")
    need(all(Path(raw).is_file() and sha(raw) == digest for raw, digest in pins.items()),
         "local training pinned input changed")
    outputs = report.get("outputs")
    need(isinstance(outputs, dict) and set(outputs) == {
        "pins.json", "selection.json", "resources.jsonl", "control-parity.jsonl",
        "initial.safetensors", "updates.jsonl", "after.jsonl", "residual.safetensors",
        "reload.json", "checkpoint.json"}, "local training output manifest differs")
    need(all((TRAINING / name).is_file() and sha(TRAINING / name) == digest
             for name, digest in outputs.items()), "local training output changed")
    checkpoint = read(TRAINING / "checkpoint.json")
    need(checkpoint.get("schema") == TRAIN_SCHEMA and checkpoint.get("arm") == "local_aux"
         and checkpoint.get("pins") == pins
         and checkpoint.get("residual_sha256") == sha(TRAINING / "residual.safetensors")
         and checkpoint.get("initial_sha256") == sha(TRAINING / "initial.safetensors")
         and checkpoint.get("selection_sha256") == sha(TRAINING / "selection.json"),
         "local training checkpoint differs")
    artifact_pins = {str(report_path): sha(report_path), str(proof_path): sha(proof_path)}
    artifact_pins.update({str(TRAINING / name): digest for name, digest in outputs.items()})
    return report, checkpoint, artifact_pins


def authenticate_control_training():
    report_path = CONTROL_TRAINING / "report.json"
    proof_path = CONTROL_TRAINING / "independent-verification.json"
    report, proof = read(report_path), read(proof_path)
    need(report.get("schema") == "source-span-paired-training-v1"
         and report.get("completed") is True and report.get("rows") == 1483
         and report.get("parent_before") == report.get("parent_after")
         and proof.get("schema") == "source-span-paired-training-independent-verification-v1"
         and proof.get("verified") is True
         and proof.get("run_report_sha256") == sha(report_path),
         "historical control training proof differs")
    need(all(Path(raw).is_file() and sha(raw) == digest
             for raw, digest in report.get("pins", {}).items()),
         "historical control training pin changed")
    need(all((CONTROL_TRAINING / name).is_file()
             and sha(CONTROL_TRAINING / name) == digest
             for name, digest in report.get("outputs", {}).items()),
         "historical control training output changed")
    checkpoint = read(CONTROL_TRAINING / "source_local/checkpoint.json")
    need(checkpoint.get("schema") == "source-span-paired-training-v1"
         and checkpoint.get("arm") == "source_local"
         and checkpoint.get("residual_sha256")
             == sha(CONTROL_TRAINING / "source_local/residual.safetensors"),
         "historical control checkpoint differs")
    artifacts = {str(report_path): sha(report_path), str(proof_path): sha(proof_path)}
    artifacts.update({str(CONTROL_TRAINING / name): digest
                      for name, digest in report["outputs"].items()})
    return report, checkpoint, artifacts


def authenticate_sources():
    need(all(sha(RUNTIME / name) == digest for name, digest in RUNTIME_HASHES.items()),
         "runtime source artifact changed")
    need(all(sha(PROPOSALS / name) == digest for name, digest in PROPOSAL_HASHES.items()),
         "proposal source artifact changed")
    proposal_report, proposal_proof = read(PROPOSALS / "report.json"), read(
        PROPOSALS / "independent-verification.json")
    need(proposal_report.get("completed") is True and proposal_report.get("rows") == ROWS
         and proposal_report.get("capture_sha256") == PROPOSAL_HASHES["capture.jsonl"]
         and proposal_proof.get("verified") is True
         and proposal_proof.get("contracts", {}).get("source_only") is True,
         "proposal proof differs")
    expected = {str((RUNTIME / name).resolve()): digest
                for name, digest in RUNTIME_HASHES.items()}
    expected.update({str((PROPOSALS / name).resolve()): digest
                     for name, digest in PROPOSAL_HASHES.items()})
    for raw, digest in proposal_report.get("pins", {}).items():
        path = Path(raw); path = path if path.is_absolute() else ROOT / path
        need(path.is_file() and sha(path) == digest, "proposal pin changed: " + raw)
        expected[str(path.resolve())] = digest
    pure = ROOT / ".omx/artifacts/verify-source-edge-recombination-v1.py"
    expected[str(pure)] = proposal_proof["identity"]["verifier_sha256"]
    expected[str(DEV)] = DEV_SHA256
    return expected


def expected_pins(training_report, training_artifacts, control_artifacts, source_pins):
    expected = dict(training_report["pins"])
    expected.update(training_artifacts)
    expected.update(source_pins)
    expected.update(control_artifacts)
    expected[str(CONTROL_DEV / "predictions.jsonl")] = CONTROL_PREDICTIONS_SHA256
    expected[str(CONTROL_DEV / "predictions.complete.json")] = CONTROL_CLOSURE_SHA256
    tracked = [PRODUCER, PLAN, ROOT / "scripts/evaluate_source_span_pair.py",
               ROOT / "scripts/score_source_edge_recombination.py",
               ROOT / "scripts/train_local_span_objective.py",
               ROOT / "scripts/train_source_span_pair.py", ROOT / "scripts/source_span_resources.py",
               ROOT / "hanglish/models/source_span_residual.py",
               ROOT / "hanglish/models/span_token_bridge.py",
               ROOT / "hanglish/core/source_correspondence.py"]
    expected.update({str(path): sha(path) for path in tracked})
    return dict(sorted(expected.items()))


def verify_prediction(saved, runtime, proposal, historical, helper):
    texts, representatives, groups, path_ids = helper.reconstruct_domain(runtime, proposal)
    source = runtime["request"]["input"]
    need(set(saved) == {"id", "request", "candidate_texts", "path_ids", "support",
                        "scores", "latency_ms", "responses", "full_response_sha256",
                        "protected_paths", "protected_responses"}
         and saved["id"] == runtime["id"] and saved["request"] == runtime["request"]
         and saved["candidate_texts"] == texts and saved["path_ids"] == path_ids,
         "prediction source, domain, or paths differ")
    need(saved["id"] == historical.get("id")
         and saved["request"] == historical.get("request")
         and saved["candidate_texts"] == historical.get("candidate_texts")
         and saved["path_ids"] == historical.get("path_ids")
         and saved["support"] == historical.get("support"),
         "historical control source/domain replay differs")
    latency = saved["latency_ms"]
    need(isinstance(latency, dict)
         and set(latency) == {"captured_runtime", "parent_features",
                             "source_local_control", "local_aux"}
         and all(finite(value) and value >= 0 for value in latency.values())
         and latency["captured_runtime"] == float(runtime["elapsed_ms"]),
         "prediction latency differs")
    original = runtime["proposed"]
    need(saved["responses"].get("captured_runtime") == helper.compact_response(original)
         and saved["full_response_sha256"].get("captured_runtime")
             == helper.canonical_hash(original), "captured response differs")
    support = saved["support"]
    need(set(support) == {"supported", "reason"} and type(support["supported"]) is bool,
         "common support contract differs")
    if support["supported"]:
        need(support["reason"] is None and set(saved["scores"]) == set(MODELS),
             "supported score map differs")
        for arm in MODELS:
            scores = saved["scores"][arm]
            need(isinstance(scores, list) and len(scores) == len(texts)
                 and all(finite(value) for value in scores),
                 "supported branch scores differ: " + arm)
            expected = helper.apply_policy(original, representatives, texts, scores, source, arm)
            need(saved["responses"].get(arm) == helper.compact_response(expected)
                 and saved["full_response_sha256"].get(arm)
                     == helper.canonical_hash(expected), "ranked response differs: " + arm)
    else:
        need(support["reason"] in {"source_exceeds_2048_bytes", "empty_nonraw_pool",
                                   "candidate_exceeds_256_paired_tokens"}
             and saved["scores"] == {}, "fallback support or scores differ")
        if len(source.encode("utf-8")) > 2048:
            need(support["reason"] == "source_exceeds_2048_bytes",
                 "long-source fallback reason differs")
        if not texts:
            need(support["reason"] == "empty_nonraw_pool", "empty fallback reason differs")
        for arm in MODELS:
            need(saved["responses"].get(arm) == helper.compact_response(original)
                 and saved["full_response_sha256"].get(arm)
                     == helper.canonical_hash(original), "fallback changed response: " + arm)
    need(set(saved["responses"]) == set(ARMS)
         and set(saved["full_response_sha256"]) == set(ARMS),
         "response branch inventory differs")
    need(saved["scores"].get("parent") == historical.get("scores", {}).get("parent")
         and saved["scores"].get("source_local_control")
             == historical.get("scores", {}).get("source_local")
         and saved["responses"]["source_local_control"]
             == historical.get("responses", {}).get("source_local")
         and saved["full_response_sha256"]["source_local_control"]
             == historical.get("full_response_sha256", {}).get("source_local"),
         "historical control score/response replay differs")
    paths = helper.protected_paths(source, groups)
    responses = {arm: helper.protected_response(source, saved["responses"][arm]) for arm in ARMS}
    need(saved["protected_paths"] == paths
         and saved["protected_responses"] == responses,
         "protected literal evidence differs")
    return support["supported"]


def response_counts(response, expected):
    texts = [candidate.get("text") for candidate in response.get("candidates", [])[:5]
             if isinstance(candidate, dict)]
    return texts, bool(texts and texts[0] in expected), any(text in expected for text in texts)


def summarize(predictions, labels, arm, rows=ROWS):
    need(arm in ARMS and len(predictions) == len(labels) == rows
         and [row["id"] for row in predictions] == [row.get("id") for row in labels],
         "DEV metric identity differs")
    counts, categories, outcomes, latency = Counter(), defaultdict(Counter), [], []
    for row, label in zip(predictions, labels):
        need(row["request"] == {"input": label.get("input"), "context": ""},
             "late DEV source identity differs")
        expected = label.get("expected")
        need(isinstance(expected, list) and expected, "DEV expected values differ")
        texts, top1, top5 = response_counts(row["responses"][arm], expected)
        covered = bool(set(row["candidate_texts"] + [label["input"]]).intersection(expected))
        protected = row["protected_responses"][arm]["verified"]
        supported = row["support"]["supported"]
        values = {"rows": 1, "top1": int(top1), "top5": int(top5),
                  "full_pool_covered": int(covered), "supported": int(supported),
                  "fallback": int(not supported), "protected_exact": int(protected)}
        counts.update(values); categories[label.get("category", "uncategorized")].update(values)
        key = ("captured_runtime" if arm == "captured_runtime"
               else "parent_features" if arm == "parent" else arm)
        measured = row["latency_ms"][key]
        if arm in ("source_local_control", "local_aux"):
            measured += row["latency_ms"]["parent_features"]
        latency.append(measured)
        outcomes.append({"id": row["id"], "actual": texts, "expected": expected,
                         "top1": top1, "top5": top5, "full_pool_covered": covered,
                         "supported": supported, "protected_exact": protected})
    latency.sort()
    return {"counts": dict(counts),
            "categories": {key: dict(value) for key, value in sorted(categories.items())},
            "latency": {"measured_rows": len(latency), "total_ms": sum(latency),
                        "mean_ms": sum(latency) / len(latency), "max_ms": max(latency),
                        "p95_ms": latency[min(len(latency) - 1,
                                               math.ceil(.95 * len(latency)) - 1)]},
            "outcomes": outcomes}


def paired_delta(before, after):
    left, right = ({row["id"]: row for row in value["outcomes"]}
                   for value in (before, after))
    need(set(left) == set(right), "paired metric IDs differ")
    return {"top1_gain_ids": [key for key in left if right[key]["top1"] and not left[key]["top1"]],
            "top1_loss_ids": [key for key in left if left[key]["top1"] and not right[key]["top1"]],
            "top5_gain_ids": [key for key in left if right[key]["top5"] and not left[key]["top5"]],
            "top5_loss_ids": [key for key in left if left[key]["top5"] and not right[key]["top5"]],
            "protected_regression_ids": [key for key in left
                if left[key]["protected_exact"] and not right[key]["protected_exact"]]}


def decide(metrics):
    need(set(metrics) == set(ARMS), "evaluation metric branches differ")
    baselines = ("source_local_control", "captured_runtime")
    comparisons = {baseline: paired_delta(metrics[baseline], metrics["local_aux"])
                   for baseline in baselines}
    counts = metrics["local_aux"]["counts"]
    eligible = all(counts["top1"] > metrics[baseline]["counts"]["top1"]
                   and counts["top5"] >= metrics[baseline]["counts"]["top5"]
                   and not comparisons[baseline]["protected_regression_ids"]
                   for baseline in baselines)
    return {"evaluated_arm": "local_aux", "historical_control_eligible": False,
            "versus": comparisons, "eligible_for_1000_case_regression": eligible,
            "selected_for_1000_case_regression": "local_aux" if eligible else None,
            "production_qualified": False,
            "production_qualification_reason":
                "DEV eligibility still requires the fixed 1,000-case and runtime gates"}


def expected_operations(predictions):
    names = ["load_parent_and_residuals"]
    for row in predictions:
        if row["support"]["supported"]:
            names.extend(("parent_features:" + row["id"],
                          "source_local_control:" + row["id"],
                          "local_aux:" + row["id"]))
    return names


def verify_resources(report, predictions, path):
    trace = jsonl(path); baseline = report.get("baseline_swap_bytes")
    need(trace and type(baseline) is int and baseline >= 0, "resource trace/baseline differs")
    expected, next_index, current, previous = expected_operations(predictions), 0, None, -1.
    peak_rss = peak_swap = peak_operation = 0.
    final_seen = False
    final_trace = None
    for index, row in enumerate(trace):
        need(set(row) == {"at", "elapsed_s", "rss_bytes", "swap_used_bytes",
                          "swap_growth_bytes", "operation_elapsed_s", "stage", "operation"}
             and row["stage"] in {"operation_start", "operation_end", "watchdog", "final"}
             and finite(row["elapsed_s"]) and row["elapsed_s"] >= previous
             and type(row["rss_bytes"]) is int and row["rss_bytes"] >= 0
             and type(row["swap_used_bytes"]) is int and row["swap_used_bytes"] >= 0
             and type(row["swap_growth_bytes"]) is int
             and row["swap_used_bytes"] - baseline == row["swap_growth_bytes"],
             "resource observation differs")
        previous = row["elapsed_s"]
        need(row["elapsed_s"] <= EVAL_LIMITS["whole_seconds"]
             and row["rss_bytes"] <= EVAL_LIMITS["rss_bytes"]
             and row["swap_growth_bytes"] <= EVAL_LIMITS["swap_growth_bytes"],
             "resource hard bound exceeded")
        if row["operation"] is None:
            need(row["stage"] in {"watchdog", "final"}
                 and row["operation_elapsed_s"] is None,
                 "idle watchdog attribution differs")
        else:
            need(finite(row["operation_elapsed_s"])
                 and 0 <= row["operation_elapsed_s"] <= EVAL_LIMITS["update_seconds"],
                 "resource operation time bound exceeded")
        if row["stage"] == "final":
            need(not final_seen and current is None
                 and next_index == len(expected), "final resource sample order differs")
            final_seen = True
            final_trace = row
        elif row["stage"] == "operation_start":
            need(not final_seen and current is None and next_index < len(expected)
                 and row["operation"] == expected[next_index],
                 "resource operation start order differs")
            current = row["operation"]; next_index += 1
        elif row["stage"] == "operation_end":
            need(not final_seen and current is not None and row["operation"] == current,
                 "resource operation end pairing differs")
            current = None
        else:
            need(row["operation"] == current, "atomic watchdog attribution differs")
        peak_rss = max(peak_rss, row["rss_bytes"])
        peak_swap = max(peak_swap, row["swap_growth_bytes"])
        if row["operation_elapsed_s"] is not None:
            peak_operation = max(peak_operation, row["operation_elapsed_s"])
    need(current is None and next_index == len(expected) and final_seen,
         "resource operation inventory/final sample differs")
    final = report.get("final_resources", {})
    need(final == final_trace and final.get("stage") == "final" and final.get("operation") is None
         and final.get("operation_elapsed_s") is None and finite(final.get("elapsed_s"))
         and final["elapsed_s"] <= EVAL_LIMITS["whole_seconds"]
         and type(final.get("rss_bytes")) is int and final["rss_bytes"] <= EVAL_LIMITS["rss_bytes"]
         and type(final.get("swap_used_bytes")) is int
         and type(final.get("swap_growth_bytes")) is int
         and final["swap_used_bytes"] - baseline == final["swap_growth_bytes"]
         and final["swap_growth_bytes"] <= EVAL_LIMITS["swap_growth_bytes"]
         and finite(report.get("elapsed_s"))
         and previous <= report["elapsed_s"] <= EVAL_LIMITS["whole_seconds"],
         "final resource observation differs")
    return {"observations": len(trace), "operation_pairs": len(expected),
            "supported_rows": sum(row["support"]["supported"] for row in predictions),
            "peak_rss_bytes": max(peak_rss, final["rss_bytes"]),
            "peak_swap_growth_bytes": max(peak_swap, final["swap_growth_bytes"]),
            "peak_operation_seconds": peak_operation,
            "atomic_operation_attribution_verified": True}


def verify(run=RUN):
    run = Path(run).resolve()
    need(run == RUN.resolve(), "alternate DEV run is not authorized")
    need(sha(PRODUCER) == PRODUCER_SHA256, "frozen DEV producer changed")
    helper = _helper()
    # No prediction or DEV label parse before this completed report/closure gate.
    report = read(run / "report.json")
    closure = read(run / "predictions.complete.json")
    prediction_path = run / "predictions.jsonl"; capture_sha = sha(prediction_path)
    need(report.get("schema") == RUN_SCHEMA and report.get("completed") is True
         and report.get("split") == "dev" and report.get("rows") == ROWS
         and report.get("limits") == EVAL_LIMITS, "completed DEV report differs")
    need(closure == {"schema": RUN_SCHEMA + "-source-only-closure", "rows": ROWS,
         "predictions_sha256": capture_sha, "source_only": True,
         "targets_joined": False, "candidate_policy": "recombined_pool_256",
         "common_support_rule": True, "input_truncation": False}
         and report.get("capture_sha256") == capture_sha
         and report.get("closure_sha256") == sha(run / "predictions.complete.json"),
         "source-only prediction closure differs")
    need(report.get("scope") == {
        "source_only_predictions_closed_before_target_join": True,
        "dev_is_exposed_development": True, "test_opened": False,
        "production_changed": False, "input_truncation": False}
         and report.get("dev_labels_sha256") == DEV_SHA256,
         "DEV scope differs")

    training_report, training_checkpoint, training_artifacts = authenticate_training()
    control_report, control_checkpoint, control_artifacts = authenticate_control_training()
    need(training_checkpoint.get("architecture") == control_checkpoint.get("architecture")
         and training_checkpoint.get("initial_sha256")
             == control_checkpoint.get("initial_sha256")
         and training_report.get("initial_hashes")
             == control_report.get("arms", {}).get("source_local", {}).get("initial_hashes"),
         "local/control checkpoint initialization differs")
    source_pins = authenticate_sources()
    pins = expected_pins(training_report, training_artifacts, control_artifacts, source_pins)
    need(read(run / "pins.json") == pins == report.get("pins"), "DEV pin inventory differs")
    need(all(Path(raw).is_file() and sha(raw) == digest for raw, digest in pins.items()),
         "DEV pinned input changed")
    need(report.get("training_report_sha256") == sha(TRAINING / "report.json")
         and report.get("control_training_report_sha256")
             == sha(CONTROL_TRAINING / "report.json"),
         "DEV checkpoint identity differs")
    need(sha(CONTROL_DEV / "predictions.jsonl") == CONTROL_PREDICTIONS_SHA256
         and sha(CONTROL_DEV / "predictions.complete.json") == CONTROL_CLOSURE_SHA256
         and read(CONTROL_DEV / "predictions.complete.json") == {
             "candidate_policy": "recombined_pool_256", "common_support_rule": True,
             "input_truncation": False, "predictions_sha256": CONTROL_PREDICTIONS_SHA256,
             "rows": ROWS, "schema": "source-span-paired-dev-evaluation-v1-source-only-closure",
             "source_only": True, "targets_joined": False},
         "historical source-only DEV closure differs")

    predictions = jsonl(prediction_path)
    runtime_rows, proposal_rows = jsonl(RUNTIME / "capture.jsonl"), jsonl(PROPOSALS / "capture.jsonl")
    historical_rows = jsonl(CONTROL_DEV / "predictions.jsonl")
    need(len(predictions) == len(runtime_rows) == len(proposal_rows)
         == len(historical_rows) == ROWS, "source-only DEV denominator differs")
    supported = [verify_prediction(saved, runtime, proposal, historical, helper)
                 for saved, runtime, proposal, historical in zip(
                     predictions, runtime_rows, proposal_rows, historical_rows)]
    need(len({row["id"] for row in predictions}) == ROWS, "DEV prediction IDs duplicate")

    # This is the first DEV target parse, after closure and all source-only replay.
    need(sha(DEV) == DEV_SHA256, "DEV labels changed")
    labels = jsonl(DEV)
    metrics = {arm: summarize(predictions, labels, arm) for arm in ARMS}
    need(metrics["captured_runtime"]["counts"]["top1"] == 264
         and metrics["captured_runtime"]["counts"]["top5"] == 284,
         "captured runtime baseline differs")
    need(report.get("metrics") == metrics, "reported DEV metrics differ")
    decision = decide(metrics)
    need(report.get("decision") == decision, "reported DEV decision differs")
    protected_rows = [row for row in predictions if helper.fixtures(row["request"]["input"])]
    protected = {"rows": len(protected_rows),
        "fixture_counts": dict(Counter(fixture["kind"] for row in protected_rows
            for fixture in helper.fixtures(row["request"]["input"]))),
        "all_observed_candidate_paths_verified": all(
            row["protected_paths"]["verified"] for row in protected_rows),
        "production_runtime_preservation_verified": False}
    protected["response_failures"] = {arm: [row["id"] for row in protected_rows
        if not row["protected_responses"][arm]["verified"]] for arm in ARMS}
    need(report.get("protected_literal_contract") == protected,
         "protected literal summary differs")
    need(report.get("limitations") == [
        "DEV metrics do not establish blind generalization",
        "protected literals are verified on offline candidate paths",
        "captured and rescored latency are not an apples-to-apples benchmark",
        "the historical control is descriptive and is never selectable in this experiment"],
         "DEV report limitations differ")
    resources = verify_resources(report, predictions, run / "resources.jsonl")
    need(not (run / "failure.json").exists()
         and {path.name for path in run.iterdir() if path.is_file()}
             <= OUTPUTS | {"report.json", "independent-verification.json"},
         "DEV output inventory differs")
    return {"schema": SCHEMA, "verified": True,
        "report_sha256": sha(run / "report.json"), "predictions_sha256": capture_sha,
        "closure_sha256": sha(run / "predictions.complete.json"),
        "producer_sha256": sha(PRODUCER), "verifier_sha256": sha(Path(__file__)),
        "counts": {"rows": ROWS, "supported": sum(supported),
                   "fallback": ROWS - sum(supported), "protected_rows": len(protected_rows)},
        "metrics": {arm: metrics[arm]["counts"] for arm in ARMS},
        "decision": decision, "resources": resources,
        "identity": {"training_report_sha256": sha(TRAINING / "report.json"),
                     "training_proof_sha256": sha(TRAINING / "independent-verification.json"),
                     "control_training_report_sha256": sha(CONTROL_TRAINING / "report.json"),
                     "control_predictions_sha256": CONTROL_PREDICTIONS_SHA256,
                     "pins": len(pins)},
        "contracts": {
            "completed_report_and_source_only_closure_checked_before_labels": True,
            "historical_gold_bearing_dev_report_opened": False,
            "historical_control_source_domain_scores_and_responses_replayed": True,
            "captured_candidate_domain_and_all_path_ids_reconstructed": True,
            "scores_finite_and_common_support_replayed": True,
            "top4_nonraw_plus_raw_policy_replayed": True,
            "fallback_exact_response_replayed": True,
            "protected_literals_replayed_from_spans": True,
            "metrics_categories_gain_loss_and_decision_recomputed": True,
            "dev_is_exposed_development": True, "test_opened": False,
            "production_changed": False, "torch_imported": False,
            "tokenizer_instantiated": False, "neural_model_instantiated": False,
            "neural_forward_rerun": False, "production_qualified": False},
        "limitations": [
            "Neural feature extraction and residual forwards are not repeated.",
            "Candidate token-length support is authenticated from the closed evaluator artifact; source-byte and empty-pool fallbacks are independently derived.",
            "This is exposed DEV metric replay, not blind generalization or the required 1,000-case regression.",
            "Offline path preservation does not prove production runtime wiring or access from a different Tailscale peer."],
    }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path); args = parser.parse_args()
    result = verify(args.run); output = args.output or args.run / "independent-verification.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"verified": True, "counts": result["counts"],
                      "metrics": result["metrics"], "decision": result["decision"],
                      "resources": result["resources"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
