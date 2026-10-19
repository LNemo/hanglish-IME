#!/usr/bin/env python3
"""Independent verifier for the closed shared-edge DEV evaluation.

The evaluator's source-only closure is authenticated before DEV labels are
opened.  Domain and policy reconstruction use a hash-pinned pure verifier;
Torch, tokenizer, evaluator, and model modules are never imported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".omx/artifacts/shared-edge-residual-dev-v1"
TRAINING = ROOT / ".omx/artifacts/shared-edge-residual-training-v1"
CONTROL_TRAINING = ROOT / ".omx/artifacts/source-span-paired-training-v1"
CONTROL_DEV = ROOT / ".omx/artifacts/source-span-paired-dev-v1"
RUNTIME = ROOT / ".omx/artifacts/extended-runtime-dev-v2"
PROPOSALS = ROOT / ".omx/artifacts/source-edge-recombination-dev-v1"
DEV = ROOT / "data/challenge/sentence-v3/dev.jsonl"
PLAN = ROOT / ".omx/plans/shared-edge-residual-v1.md"
HELPER_PATH = ROOT / ".omx/artifacts/verify-source-span-pair-dev-v1.py"
HELPER_SHA256 = "77373cca43d198a4fd47946cb8584e9dcb6089583a7a7ff2a47f48c1de2baa1d"
PRODUCER = ROOT / "scripts/evaluate_shared_edge_residual.py"
# Set once the producer is frozen; the verifier refuses every other program.
PRODUCER_SHA256 = "6e799a54fa7f99ad3aded2a9c82c6eb54d0a8dd0b5e01650bccacf53c9e88f66"
SCHEMA = "shared-edge-residual-dev-independent-verification-v1"
RUN_SCHEMA = "shared-edge-residual-dev-evaluation-v1"
TRAIN_SCHEMA = "shared-edge-residual-training-v1"
TRAIN_PROOF_SCHEMA = "shared-edge-residual-training-independent-verification-v1"
ROWS = 316
ARMS = ("captured_runtime", "parent", "source_local_control", "shared_edge")
MODELS = ("parent", "source_local_control", "shared_edge")
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
OUTPUTS = {"pins.json", "predictions.jsonl", "predictions.complete.json", "resources.jsonl"}


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


def f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _helper():
    need(sha(HELPER_PATH) == HELPER_SHA256, "frozen pure DEV helper changed")
    spec = importlib.util.spec_from_file_location("frozen_source_span_dev_verifier", HELPER_PATH)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    return helper


def authenticate_training():
    report_path, proof_path = TRAINING / "report.json", TRAINING / "independent-verification.json"
    report, proof = read(report_path), read(proof_path)
    required_contracts = {
        "authorized_fit_inventory_independently_reconstructed": True,
        "all_target_cache_shapes_flags_metadata_and_hashes_verified": True,
        "historical_initial_and_parent_frozen": True,
        "exact_CE_only_update_order_and_losses_verified": True,
        "all_final_edges_paths_weights_contributions_and_winners_replayed": True,
        "shared_edge_contributions_equal_across_paths": True,
        "fp32_base_plus_residual_and_fit_metrics_replayed": True,
        "reload_scores_and_edge_scores_verified": True,
        "closed_output_and_pin_hashes_verified": True,
        "fit_only": True, "hold_opened": False, "dev_opened": False,
        "test_opened": False, "production_changed": False,
        "neural_forward_rerun": False,
    }
    need(report.get("schema") == TRAIN_SCHEMA and report.get("completed") is True
         and report.get("rows") == 1483 and report.get("updates") == 2966
         and report.get("parent_before") == report.get("parent_after")
         and proof.get("schema") == TRAIN_PROOF_SCHEMA and proof.get("verified") is True
         and proof.get("report_sha256") == sha(report_path)
         and proof.get("producer_sha256") == sha(ROOT / "scripts/train_shared_edge_residual.py")
         and proof.get("counts") == {"rows": 1483, "targets": 16814, "edges": 47282,
                                     "candidate_paths": 11793, "updates": 2966,
                                     "reload_rows": 16}
         and all(proof.get("contracts", {}).get(key) == value
                 for key, value in required_contracts.items())
         and proof.get("tensors", {}).get("trainable_parameters") == 71073
         and proof.get("tensors", {}).get("parent_unchanged") is True
         and proof.get("tensors", {}).get("initial_matches_historical") is True
         and proof.get("resources", {}).get("atomic_operation_attribution_verified") is True,
         "shared-edge training report/proof differs")
    pins = report.get("pins"); outputs = report.get("outputs")
    expected_outputs = {"pins.json", "selection.json", "targets.json", "resources.jsonl",
        "initial.safetensors", "tokens.json", "target-features.safetensors",
        "target-features.json", "before.jsonl", "updates.jsonl", "after.jsonl",
        "residual.safetensors", "reload.json", "checkpoint.json"}
    need(isinstance(pins, dict) and pins and read(TRAINING / "pins.json") == pins
         and all(Path(raw).is_file() and sha(raw) == digest for raw, digest in pins.items()),
         "shared-edge training pins differ")
    need(isinstance(outputs, dict) and set(outputs) == expected_outputs
         and all((TRAINING / name).is_file() and sha(TRAINING / name) == digest
                 for name, digest in outputs.items()), "shared-edge training outputs differ")
    checkpoint = read(TRAINING / "checkpoint.json")
    need(checkpoint.get("schema") == TRAIN_SCHEMA and checkpoint.get("arm") == "shared_edge"
         and checkpoint.get("pins") == pins
         and checkpoint.get("residual_sha256") == sha(TRAINING / "residual.safetensors")
         and checkpoint.get("initial_sha256") == sha(TRAINING / "initial.safetensors")
         and checkpoint.get("selection_sha256") == sha(TRAINING / "selection.json"),
         "shared-edge checkpoint differs")
    artifacts = {str(report_path): sha(report_path), str(proof_path): sha(proof_path)}
    artifacts.update({str(TRAINING / name): digest for name, digest in outputs.items()})
    return report, checkpoint, artifacts


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
    need(all(Path(raw).is_file() and sha(raw) == digest for raw, digest in report.get("pins", {}).items())
         and all((CONTROL_TRAINING / name).is_file()
                 and sha(CONTROL_TRAINING / name) == digest
                 for name, digest in report.get("outputs", {}).items()),
         "historical control training artifact changed")
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
    report, proof = read(PROPOSALS / "report.json"), read(PROPOSALS / "independent-verification.json")
    need(report.get("completed") is True and report.get("rows") == ROWS
         and report.get("capture_sha256") == PROPOSAL_HASHES["capture.jsonl"]
         and proof.get("verified") is True
         and proof.get("contracts", {}).get("source_only") is True,
         "proposal proof differs")
    expected = {str((RUNTIME / name).resolve()): digest for name, digest in RUNTIME_HASHES.items()}
    expected.update({str((PROPOSALS / name).resolve()): digest
                     for name, digest in PROPOSAL_HASHES.items()})
    for raw, digest in report.get("pins", {}).items():
        path = Path(raw); path = path if path.is_absolute() else ROOT / path
        need(path.is_file() and sha(path) == digest, "proposal pin changed: " + raw)
        expected[str(path.resolve())] = digest
    pure = ROOT / ".omx/artifacts/verify-source-edge-recombination-v1.py"
    expected[str(pure)] = proof["identity"]["verifier_sha256"]
    expected[str(DEV)] = DEV_SHA256
    return expected


def expected_pins(training_report, training_artifacts, control_artifacts, source_pins):
    expected = dict(training_report["pins"]); expected.update(training_artifacts)
    expected.update(source_pins); expected.update(control_artifacts)
    expected[str(CONTROL_DEV / "predictions.jsonl")] = CONTROL_PREDICTIONS_SHA256
    expected[str(CONTROL_DEV / "predictions.complete.json")] = CONTROL_CLOSURE_SHA256
    tracked = [PRODUCER, PLAN, ROOT / "scripts/evaluate_local_span_objective.py",
        ROOT / "scripts/score_source_edge_recombination.py",
        ROOT / "scripts/train_shared_edge_residual.py", ROOT / "scripts/source_span_resources.py",
        ROOT / "hanglish/models/shared_edge_residual.py",
        ROOT / "hanglish/models/source_span_residual.py",
        ROOT / "hanglish/models/span_token_bridge.py",
        ROOT / "hanglish/core/source_correspondence.py"]
    expected.update({str(path): sha(path) for path in tracked})
    return dict(sorted(expected.items()))


def isolated_targets(groups):
    return sorted({span["text"] for group in groups for path in group
                   for span in path["segments"]})


def edge_id(span):
    return (span["input_start"], span["input_end"], span["text"])


def ulp_sum_tolerance(contributions):
    need(contributions and all(finite(value) for value in contributions),
         "contribution tolerance domain differs")
    magnitude = max(1., sum(abs(value) for value in contributions))
    ulp = 2. ** (math.frexp(magnitude)[1] - 24)
    levels = math.ceil(math.log2(len(contributions))) if len(contributions) > 1 else 0
    return 4 * (levels + 1) * ulp


def fp32_path_audit(source, groups, path_ids, audit):
    need(isinstance(audit, dict)
         and set(audit) == {"edge_ids", "edge_scores", "residuals", "winning_path_ids"},
         "shared-edge audit fields differ")
    edges = sorted({edge_id(span) for group in groups for path in group
                    for span in path["segments"]})
    need(audit.get("edge_ids") == [list(edge) for edge in edges], "shared edge IDs differ")
    scores = audit.get("edge_scores")
    need(isinstance(scores, list) and len(scores) == len(edges)
         and all(finite(value) for value in scores), "shared edge scores differ")
    edge_scores = dict(zip(edges, map(f32, scores)))
    winners, residuals = [], []
    denominator = len(source)
    need(denominator > 0, "empty supported source")
    for group, candidate_path_ids in zip(groups, path_ids):
        path_values = []
        for path, path_id in zip(group, candidate_path_ids):
            contributions = []
            for span in path["segments"]:
                weight = f32((span["input_end"] - span["input_start"]) / denominator)
                contributions.append(f32(edge_scores[edge_id(span)] * weight))
            value = f32(sum(contributions))
            path_values.append((path_id, value, contributions))
        best = max(value for _, value, _ in path_values)
        residuals.append(best)
        winners.append([path_id for path_id, value, _ in path_values if value == best])
        recorded = audit.get("residuals", [])[len(residuals) - 1]
        winning_contributions = next(parts for _, value, parts in path_values if value == best)
        need(finite(recorded)
             and abs(recorded - sum(winning_contributions))
                 <= ulp_sum_tolerance(winning_contributions),
             "shared candidate residual differs")
        residuals[-1] = recorded
    need(audit.get("winning_path_ids") == winners, "shared winning path IDs differ")
    return residuals


def verify_prediction(saved, runtime, proposal, historical, helper, cached):
    texts, representatives, groups, path_ids = helper.reconstruct_domain(runtime, proposal)
    source = runtime["request"]["input"]; targets = isolated_targets(groups)
    required = {"id", "request", "candidate_texts", "path_ids", "isolated_target_texts",
                "target_cache", "support", "scores", "latency_ms", "shared_edge_audit",
                "responses", "full_response_sha256", "protected_paths", "protected_responses"}
    need(set(saved) == required and saved["id"] == runtime["id"]
         and saved["request"] == runtime["request"] and saved["candidate_texts"] == texts
         and saved["path_ids"] == path_ids and saved["isolated_target_texts"] == targets,
         "prediction source, domain, paths, or isolated targets differ")
    need(saved["id"] == historical.get("id") and saved["request"] == historical.get("request")
         and texts == historical.get("candidate_texts") and path_ids == historical.get("path_ids"),
         "historical control source/domain replay differs")
    misses = [target for target in targets if target not in cached]
    cache = {"requested": len(targets), "hits": len(targets) - len(misses), "misses": len(misses)}
    need(saved["target_cache"] == cache, "target cache accounting differs")
    latency = saved["latency_ms"]
    need(isinstance(latency, dict)
         and set(latency) == {"captured_runtime", "parent_features", "target_extraction",
                              "source_local_control", "shared_edge"}
         and all(finite(value) and value >= 0 for value in latency.values())
         and latency["captured_runtime"] == float(runtime["elapsed_ms"]),
         "prediction latency differs")
    original = runtime["proposed"]
    need(saved["responses"].get("captured_runtime") == helper.compact_response(original)
         and saved["full_response_sha256"].get("captured_runtime") == helper.canonical_hash(original),
         "captured response differs")
    support = saved["support"]
    need(set(support) == {"supported", "reason"} and type(support["supported"]) is bool,
         "common support contract differs")
    if support["supported"]:
        need(support["reason"] is None and set(saved["scores"]) == set(MODELS),
             "supported score map differs")
        cached.update(misses)
        residuals = fp32_path_audit(source, groups, path_ids, saved["shared_edge_audit"])
        parent_scores = saved["scores"]["parent"]
        need(len(parent_scores) == len(texts)
             and saved["scores"]["shared_edge"]
                 == [f32(base + residual) for base, residual in zip(parent_scores, residuals)],
             "shared-edge FP32 branch scores differ")
        for arm in MODELS:
            scores = saved["scores"][arm]
            need(isinstance(scores, list) and len(scores) == len(texts)
                 and all(finite(value) for value in scores), "supported branch scores differ: " + arm)
            expected = helper.apply_policy(original, representatives, texts, scores, source, arm)
            need(saved["responses"].get(arm) == helper.compact_response(expected)
                 and saved["full_response_sha256"].get(arm) == helper.canonical_hash(expected),
                 "ranked response differs: " + arm)
        need(historical.get("support", {}).get("supported") is True
             and parent_scores == historical.get("scores", {}).get("parent")
             and saved["scores"]["source_local_control"]
                 == historical.get("scores", {}).get("source_local")
             and saved["responses"]["source_local_control"]
                 == historical.get("responses", {}).get("source_local")
             and saved["full_response_sha256"]["source_local_control"]
                 == historical.get("full_response_sha256", {}).get("source_local"),
             "historical control supported replay differs")
    else:
        need(support["reason"] in {"source_exceeds_2048_bytes", "empty_nonraw_pool",
                                   "full_candidate_exceeds_256_tokens",
                                   "isolated_target_exceeds_256_tokens"}
             and saved["scores"] == {} and saved["shared_edge_audit"] is None,
             "fallback support, scores, or audit differs")
        if len(source.encode("utf-8")) > 2048:
            need(support["reason"] == "source_exceeds_2048_bytes", "long-source fallback differs")
        if not texts:
            need(support["reason"] == "empty_nonraw_pool", "empty fallback differs")
        for arm in MODELS:
            need(saved["responses"].get(arm) == helper.compact_response(original)
                 and saved["full_response_sha256"].get(arm) == helper.canonical_hash(original),
                 "fallback changed response: " + arm)
    need(set(saved["responses"]) == set(ARMS)
         and set(saved["full_response_sha256"]) == set(ARMS), "response branch inventory differs")
    paths = helper.protected_paths(source, groups)
    responses = {arm: helper.protected_response(source, saved["responses"][arm]) for arm in ARMS}
    need(saved["protected_paths"] == paths and saved["protected_responses"] == responses,
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
        expected = label.get("expected"); need(isinstance(expected, list) and expected,
                                                "DEV expected values differ")
        texts, top1, top5 = response_counts(row["responses"][arm], expected)
        covered = bool(set(row["candidate_texts"] + [label["input"]]).intersection(expected))
        protected = row["protected_responses"][arm]["verified"]
        supported = row["support"]["supported"]
        values = {"rows": 1, "top1": int(top1), "top5": int(top5),
                  "full_pool_covered": int(covered), "supported": int(supported),
                  "fallback": int(not supported), "protected_exact": int(protected)}
        counts.update(values); categories[label.get("category", "uncategorized")].update(values)
        if arm == "captured_runtime": measured = row["latency_ms"]["captured_runtime"]
        elif arm == "parent": measured = row["latency_ms"]["parent_features"]
        elif arm == "source_local_control":
            measured = row["latency_ms"]["parent_features"] + row["latency_ms"][arm]
        else:
            measured = (row["latency_ms"]["parent_features"]
                        + row["latency_ms"]["target_extraction"] + row["latency_ms"][arm])
        latency.append(measured)
        outcomes.append({"id": row["id"], "actual": texts, "expected": expected,
                         "top1": top1, "top5": top5, "full_pool_covered": covered,
                         "supported": supported, "protected_exact": protected})
    latency.sort()
    return {"counts": dict(counts),
        "categories": {key: dict(value) for key, value in sorted(categories.items())},
        "latency": {"measured_rows": len(latency), "total_ms": sum(latency),
                    "mean_ms": sum(latency) / len(latency), "max_ms": max(latency),
                    "p95_ms": latency[min(len(latency) - 1, math.ceil(.95 * len(latency)) - 1)]},
        "outcomes": outcomes}


def paired_delta(before, after):
    left, right = ({row["id"]: row for row in value["outcomes"]} for value in (before, after))
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
    comparisons = {baseline: paired_delta(metrics[baseline], metrics["shared_edge"])
                   for baseline in baselines}
    counts = metrics["shared_edge"]["counts"]
    eligible = all(counts["top1"] > metrics[baseline]["counts"]["top1"]
                   and counts["top5"] >= metrics[baseline]["counts"]["top5"]
                   and not comparisons[baseline]["protected_regression_ids"]
                   for baseline in baselines)
    return {"evaluated_arm": "shared_edge", "historical_control_eligible": False,
            "versus": comparisons, "eligible_for_1000_case_regression": eligible,
            "selected_for_1000_case_regression": "shared_edge" if eligible else None,
            "production_qualified": False,
            "production_qualification_reason":
                "DEV eligibility still requires the fixed 1,000-case and runtime gates"}


def expected_operations(predictions):
    names = ["load_parent_and_residuals"]
    cached = set()
    for row in predictions:
        if row["support"]["supported"]:
            names.append("parent_features:" + row["id"])
            misses = [target for target in row["isolated_target_texts"] if target not in cached]
            if misses:
                names.append("target_extraction:" + row["id"]); cached.update(misses)
            names.extend(("source_local_control:" + row["id"], "shared_edge:" + row["id"]))
    return names


def verify_resources(report, predictions, path):
    trace = jsonl(path); baseline = report.get("baseline_swap_bytes")
    need(trace and type(baseline) is int and baseline >= 0, "resource trace/baseline differs")
    expected, next_index, current, previous = expected_operations(predictions), 0, None, -1.
    peak_rss = peak_swap = peak_operation = 0.; final_seen = False; final_trace = None
    for row in trace:
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
            need(row["stage"] in {"watchdog", "final"} and row["operation_elapsed_s"] is None,
                 "idle watchdog attribution differs")
        else:
            need(finite(row["operation_elapsed_s"])
                 and 0 <= row["operation_elapsed_s"] <= EVAL_LIMITS["update_seconds"],
                 "resource operation time bound exceeded")
        if row["stage"] == "final":
            need(not final_seen and current is None and next_index == len(expected),
                 "final resource sample order differs")
            final_seen = True; final_trace = row
        elif row["stage"] == "operation_start":
            need(not final_seen and current is None and next_index < len(expected)
                 and row["operation"] == expected[next_index], "resource operation start order differs")
            current = row["operation"]; next_index += 1
        elif row["stage"] == "operation_end":
            need(not final_seen and current is not None and row["operation"] == current,
                 "resource operation end pairing differs")
            current = None
        else:
            need(row["operation"] == current, "atomic watchdog attribution differs")
        peak_rss = max(peak_rss, row["rss_bytes"]); peak_swap = max(peak_swap, row["swap_growth_bytes"])
        if row["operation_elapsed_s"] is not None:
            peak_operation = max(peak_operation, row["operation_elapsed_s"])
    need(current is None and next_index == len(expected) and final_seen,
         "resource operation inventory/final sample differs")
    final = report.get("final_resources", {})
    need(final == final_trace and finite(report.get("elapsed_s"))
         and previous <= report["elapsed_s"] <= EVAL_LIMITS["whole_seconds"],
         "final resource observation differs")
    return {"observations": len(trace), "operation_pairs": len(expected),
            "supported_rows": sum(row["support"]["supported"] for row in predictions),
            "peak_rss_bytes": int(max(peak_rss, final["rss_bytes"])),
            "peak_swap_growth_bytes": int(max(peak_swap, final["swap_growth_bytes"])),
            "peak_operation_seconds": peak_operation,
            "atomic_operation_attribution_verified": True}


def verify(run=RUN):
    run = Path(run).resolve(); need(run == RUN.resolve(), "alternate DEV run is not authorized")
    need(sha(PRODUCER) == PRODUCER_SHA256, "frozen DEV producer changed")
    helper = _helper()
    report = read(run / "report.json"); closure = read(run / "predictions.complete.json")
    prediction_path = run / "predictions.jsonl"; capture_sha = sha(prediction_path)
    need(report.get("schema") == RUN_SCHEMA and report.get("completed") is True
         and report.get("split") == "dev" and report.get("rows") == ROWS
         and report.get("limits") == EVAL_LIMITS, "completed DEV report differs")
    need(closure == {"schema": RUN_SCHEMA + "-source-only-closure", "rows": ROWS,
         "predictions_sha256": capture_sha, "source_only": True, "targets_joined": False,
         "candidate_policy": "recombined_pool_256", "common_support_rule": True,
         "full_candidate_limit": 256, "isolated_target_limit": 256,
         "source_byte_limit": 2048, "input_truncation": False}
         and report.get("capture_sha256") == capture_sha
         and report.get("closure_sha256") == sha(run / "predictions.complete.json"),
         "source-only prediction closure differs")
    need(report.get("scope") == {
        "source_only_predictions_closed_before_target_join": True,
        "dev_is_exposed_development": True, "test_opened": False,
        "production_changed": False, "input_truncation": False,
        "isolated_target_vectors_cached_by_text_only": True,
        "edge_scores_not_cached_across_requests": True}
         and report.get("dev_labels_sha256") == DEV_SHA256, "DEV scope differs")
    training_report, training_checkpoint, training_artifacts = authenticate_training()
    control_report, control_checkpoint, control_artifacts = authenticate_control_training()
    need(training_checkpoint.get("initial_sha256") == control_checkpoint.get("initial_sha256")
         and training_report.get("initial_hashes")
             == control_report.get("arms", {}).get("source_local", {}).get("initial_hashes"),
         "shared-edge/control initialization differs")
    source_pins = authenticate_sources()
    pins = expected_pins(training_report, training_artifacts, control_artifacts, source_pins)
    recorded_pins = report.get("pins")
    need(isinstance(recorded_pins, dict) and read(run / "pins.json") == recorded_pins,
         "DEV pin manifest differs")
    authorization_keys = set(recorded_pins) - set(pins)
    need(len(authorization_keys) == 1, "DEV authorization pin inventory differs")
    authorization_key = authorization_keys.pop(); authorization_path = Path(authorization_key)
    authorization = read(authorization_path)
    expected_authorization = {
        "schema": "shared-edge-residual-dev-prelaunch-authorization-v1", "rows": ROWS,
        "limits": EVAL_LIMITS, "target_cache_bytes": 256 * 1024**2,
        "evaluator_sha256": PRODUCER_SHA256,
        "training_report_sha256": sha(TRAINING / "report.json"),
        "training_proof_sha256": sha(TRAINING / "independent-verification.json"),
        "control_training_report_sha256": sha(CONTROL_TRAINING / "report.json"),
        "source_authorization_sha256": helper.canonical_hash(source_pins),
        "historical_predictions_sha256": CONTROL_PREDICTIONS_SHA256,
        "historical_closure_sha256": CONTROL_CLOSURE_SHA256,
        "model_sha256": sha(ROOT / "hanglish/models/shared_edge_residual.py")}
    need(authorization == expected_authorization
         and recorded_pins[authorization_key] == sha(authorization_path),
         "DEV prelaunch authorization differs")
    pins[authorization_key] = recorded_pins[authorization_key]
    pins = dict(sorted(pins.items()))
    need(pins == recorded_pins, "DEV pin inventory differs")
    need(all(Path(raw).is_file() and sha(raw) == digest for raw, digest in pins.items()),
         "DEV pinned input changed")
    need(report.get("training_report_sha256") == sha(TRAINING / "report.json")
         and report.get("control_training_report_sha256") == sha(CONTROL_TRAINING / "report.json")
         and report.get("historical_control_predictions_sha256") == CONTROL_PREDICTIONS_SHA256
         and report.get("historical_control_closure_sha256") == CONTROL_CLOSURE_SHA256,
         "DEV checkpoint/control identity differs")
    need(sha(CONTROL_DEV / "predictions.jsonl") == CONTROL_PREDICTIONS_SHA256
         and sha(CONTROL_DEV / "predictions.complete.json") == CONTROL_CLOSURE_SHA256,
         "historical source-only DEV capture differs")
    predictions = jsonl(prediction_path)
    runtime_rows = jsonl(RUNTIME / "capture.jsonl"); proposal_rows = jsonl(PROPOSALS / "capture.jsonl")
    historical_rows = jsonl(CONTROL_DEV / "predictions.jsonl")
    need(len(predictions) == len(runtime_rows) == len(proposal_rows)
         == len(historical_rows) == ROWS, "source-only DEV denominator differs")
    cached = set(); supported = [verify_prediction(*values, helper, cached) for values in zip(
        predictions, runtime_rows, proposal_rows, historical_rows)]
    need(len({row["id"] for row in predictions}) == ROWS, "DEV prediction IDs duplicate")
    cache_summary = {"unique_texts": len(cached), "limit_bytes": 256 * 1024**2,
        "tensor_bytes": len(cached) * (256 * 4 + 1),
        "requests": sum(row["target_cache"]["requested"] for row in predictions),
        "hits": sum(row["target_cache"]["hits"] for row in predictions),
        "misses": sum(row["target_cache"]["misses"] for row in predictions)}
    need(cache_summary["tensor_bytes"] <= cache_summary["limit_bytes"],
         "target cache byte limit exceeded")
    need(report.get("target_cache") == cache_summary, "reported target cache differs")
    need(sha(DEV) == DEV_SHA256, "DEV labels changed")
    labels = jsonl(DEV); metrics = {arm: summarize(predictions, labels, arm) for arm in ARMS}
    need(metrics["captured_runtime"]["counts"]["top1"] == 264
         and metrics["captured_runtime"]["counts"]["top5"] == 284,
         "captured runtime baseline differs")
    need(report.get("metrics") == metrics, "reported DEV metrics differ")
    decision = decide(metrics); need(report.get("decision") == decision, "reported DEV decision differs")
    protected_rows = [row for row in predictions if helper.fixtures(row["request"]["input"])]
    protected = {"rows": len(protected_rows),
        "fixture_counts": dict(Counter(fixture["kind"] for row in protected_rows
            for fixture in helper.fixtures(row["request"]["input"]))),
        "all_observed_candidate_paths_verified": all(
            row["protected_paths"]["verified"] for row in protected_rows),
        "production_runtime_preservation_verified": False}
    protected["response_failures"] = {arm: [row["id"] for row in protected_rows
        if not row["protected_responses"][arm]["verified"]] for arm in ARMS}
    need(report.get("protected_literal_contract") == protected, "protected literal summary differs")
    need(report.get("limitations") == [
        "DEV metrics do not establish blind generalization",
        "the shared-edge bundle changes target context, position features, and normalization",
        "captured and rescored latency are not an apples-to-apples benchmark",
        "the historical control is descriptive and cannot be selected"],
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
                   "fallback": ROWS - sum(supported), "protected_rows": len(protected_rows),
                   "unique_target_texts": len(cached)},
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
            "historical_control_supported_scores_and_responses_replayed": True,
            "captured_candidate_domain_all_paths_and_isolated_targets_reconstructed": True,
            "target_cache_sequence_replayed": True,
            "shared_edge_ids_scores_weighted_paths_winners_and_fp32_scores_replayed": True,
            "top4_nonraw_plus_raw_policy_replayed": True,
            "fallback_exact_response_replayed": True,
            "protected_literals_replayed_from_spans": True,
            "metrics_categories_gain_loss_and_decision_recomputed": True,
            "dev_is_exposed_development": True, "test_opened": False,
            "production_changed": False, "torch_imported": False,
            "tokenizer_instantiated": False, "neural_model_instantiated": False,
            "neural_forward_rerun": False, "production_qualified": False},
        "limitations": [
            "Neural feature extraction, edge scoring, and residual forwards are not repeated.",
            "Candidate and isolated-target token support is authenticated from the closed evaluator artifact; source-byte and empty-pool fallbacks are independently derived.",
            "This is exposed DEV metric replay, not blind generalization or the required 1,000-case regression.",
            "Offline path preservation does not prove production runtime wiring or access from a different Tailscale peer."]}


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
