"""Evaluate the fixed 128-row TRAIN context-consensus pilot."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.probe_consensus_context_resources import (
    ADAPTER_SHA256, MODEL_REVISION, MODEL_WEIGHT_SHA256, MlxBackend,
    PLAN_SHA256, RUNTIME_VERSION, digest, file_digest, load_inputs,
    path_key, preserve_decision, prompt_record, system_message,
    validate_frozen_sources, verify_hashes, rss_bytes,
)
from scripts.probe_consensus_context_resources import finite_distribution


OUTPUT = ROOT / ".omx/artifacts/consensus-context-train-pilot-v1"
RESOURCE_REPORT = ROOT / ".omx/artifacts/consensus-context-resource-smoke-v1/report.json"
RESOURCE_REPORT_SHA256 = "f3d98ee2354612b7def36c45291d655243b5445525dac125ac4016ecc2df8030"
RESOURCE_VERIFICATION = ROOT / ".omx/artifacts/consensus-context-work-v1/resource-verification.json"
RESOURCE_VERIFICATION_SHA256 = "069db9e664d5728e0c4cfba2bb7b829770fb801151a151cb41b58963838338c0"
RESOURCE_HELPER_SHA256 = "eb189141b33a880003b18cb879811c7813d0a41cda0fef4d2059c3772dd50cb9"
PILOT_MANIFEST_SHA256 = "3e9f84fa3c6defc1075824a984e90945895c4ace8c9609521c328172cd013f86"
LINEAGE_SHA256 = "1720d99b7e6904eb1289cc35d89bcadf01767be5a08fec24c56b3ded6c40d183"
PROFILE = {"rows": 128, "max_tokens": 256, "time_limit_seconds": 1800,
           "rss_limit_bytes": 6 * 1024**3, "mlx_peak_limit_bytes": 6 * 1024**3}
STRATA = (("natural", 0, 64), ("challenge_single", 64, 96),
          ("challenge_sentence", 96, 128))


def read_json_once(path, expected, label):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"{label} hash differs")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return raw, value


def validate_resource_smoke(report_path, report_sha, verification_path):
    if report_sha != RESOURCE_REPORT_SHA256:
        raise ValueError("only the accepted resource smoke may authorize the pilot")
    report_raw, report = read_json_once(report_path, report_sha, "resource smoke")
    verification_raw, verification = read_json_once(
        verification_path, RESOURCE_VERIFICATION_SHA256, "resource verification")
    profile = report.get("profile")
    if (report.get("schema") != "consensus-context-resource-smoke-v1"
            or report.get("completed") is not True or report.get("resource_only") is not True
            or report.get("rows") != 4 or report.get("plan_sha256") != PLAN_SHA256
            or report.get("pilot_manifest_sha256") != PILOT_MANIFEST_SHA256
            or report.get("lineage_report_sha256") != LINEAGE_SHA256
            or report.get("runtime_version") != RUNTIME_VERSION
            or report.get("base_model_sha256") != MODEL_WEIGHT_SHA256
            or report.get("adapter_sha256") != ADAPTER_SHA256
            or report.get("base_revision") != MODEL_REVISION
            or report.get("source_hashes") != report.get("source_hashes_after")
            or report.get("source_hashes", {}).get(
                "scripts/probe_consensus_context_resources.py") != RESOURCE_HELPER_SHA256
            or report.get("candidate_objects_and_raw_bypass_preserved") is not True
            or report.get("identical_prompt_tokens_and_choices") is not True
            or report.get("base_released_before_adapter_load") is not True
            or report.get("expected_or_labels_read") is not False
            or report.get("accuracy_measured") is not False
            or report.get("deployment_eligible") is not False
            or not isinstance(profile, dict)
            or report.get("elapsed_seconds", math.inf) > profile.get("time_limit_seconds", -1)
            or report.get("rss_peak_bytes", math.inf) > profile.get("rss_limit_bytes", -1)
            or report.get("mlx_peak_bytes", math.inf) > profile.get("mlx_peak_limit_bytes", -1)):
        raise ValueError("resource smoke did not establish the fixed feasibility contract")
    if (verification.get("report_sha256") != report_sha
            or verification.get("rows") != 4 or verification.get("forward_passes") != 8
            or verification.get("candidate_and_logit_mapping_recomputed") is not True
            or verification.get("raw_verified") is not True
            or verification.get("accuracy_measured") is not False
            or verification.get("deployed") is not False):
        raise ValueError("resource smoke verification differs")
    verify_hashes(report["source_hashes"])
    return {"report_raw": report_raw, "report": report,
            "verification_raw": verification_raw, "verification": verification}


def require_resources(started, backend=None):
    from scripts.probe_consensus_context_resources import rss_bytes
    if time.monotonic() - started > PROFILE["time_limit_seconds"]:
        raise RuntimeError("consensus pilot time bound exceeded")
    if rss_bytes() > PROFILE["rss_limit_bytes"]:
        raise RuntimeError("consensus pilot RSS bound exceeded")
    if backend is not None and backend.peak() > PROFILE["mlx_peak_limit_bytes"]:
        raise RuntimeError("consensus pilot MLX bound exceeded")


def infer_all_stage(backend, records, *, adapter, expected=None, started=None):
    """Tokenize every row and infer every source-only eligible row."""
    model, tokenizer = backend.load_model(adapter=adapter)
    try:
        require_resources(started, backend)
        digit_ids = backend.digit_ids(tokenizer)
        outcomes = []
        for index, record in enumerate(records):
            tokens = backend.tokens(tokenizer, record["messages"])
            if expected is not None:
                prior = expected[index]
                if tokens != prior["tokens"] or record["choices"] != prior["choices"]:
                    raise RuntimeError("base and adapter prompt tokens or choices differ")
            if len(record["choices"]) < 2:
                outcomes.append({"record_index": index, "choices": record["choices"],
                                 "tokens": tokens, "eligible": False,
                                 "reason": "insufficient_choices", "latency_seconds": 0.0,
                                 "logits": [], "probabilities": [],
                                 "prediction_source_index": None})
            elif len(tokens) > PROFILE["max_tokens"]:
                outcomes.append({"record_index": index, "choices": record["choices"],
                                 "tokens": tokens, "eligible": False,
                                 "reason": "prompt_too_long", "latency_seconds": 0.0,
                                 "logits": [], "probabilities": [],
                                 "prediction_source_index": None})
            else:
                require_resources(started, backend)
                tick = time.monotonic()
                logits = backend.infer(model, tokens, digit_ids, len(record["choices"]))
                latency = time.monotonic() - tick
                require_resources(started, backend)
                if len(logits) != len(record["choices"]):
                    raise RuntimeError("choice logit vector cardinality differs")
                probabilities, selected = finite_distribution(logits)
                outcomes.append({"record_index": index, "choices": record["choices"],
                                 "tokens": tokens, "eligible": True,
                                 "reason": "inferred", "latency_seconds": latency,
                                 "logits": logits, "probabilities": probabilities,
                                 "prediction_source_index": record["choices"][selected][0]})
            require_resources(started, backend)
        if len(outcomes) != PROFILE["rows"]:
            raise RuntimeError("pilot stage did not account for every row")
        return outcomes
    finally:
        del model, tokenizer
        backend.clear()


def evaluate_rows(records, label_rows, base, trained):
    if (len(records) != PROFILE["rows"] or len(label_rows) != len(records)
            or len(base) != len(records) or len(trained) != len(records)):
        raise ValueError("pilot inference result cardinality differs")
    outcomes = []
    for index, (record, label_row, left, right) in enumerate(
            zip(records, label_rows, base, trained)):
        if (left["record_index"] != index or right["record_index"] != index
                or left["eligible"] != right["eligible"]
                or left["reason"] != right["reason"]
                or left["tokens"] != right["tokens"]
                or left["choices"] != right["choices"]
                or label_row.get("id") != record["id"]):
            raise RuntimeError("paired pilot inference differs structurally")
        if left["eligible"]:
            decision, raw = preserve_decision(
                record, left["prediction_source_index"], right["prediction_source_index"])
        else:
            decision, raw = preserve_decision(record, 0, 0)
            if decision["chosen_index"] != 0:
                raise RuntimeError("ineligible pilot row did not retain parent")
        before = record["candidates"]
        after = decision["candidates"]
        expected = label_row.get("expected")  # First consulted after both stages.
        if (not isinstance(expected, list) or not expected
                or not all(isinstance(text, str) and text for text in expected)):
            raise ValueError("pilot expected labels are malformed")
        before_top1 = before[0]["text"] in expected
        top1 = after[0]["text"] in expected
        before_top5 = any(candidate["text"] in expected for candidate in before)
        top5 = any(candidate["text"] in expected for candidate in after)
        outcomes.append({
            "id": record["id"], "role": record["role"], "record_index": index,
            "eligible": left["eligible"], "ineligible_reason": None if left["eligible"] else left["reason"],
            "choices": left["choices"], "tokens": left["tokens"],
            "base_logits": left["logits"], "base_probabilities": left["probabilities"],
            "base_prediction_source_index": left["prediction_source_index"],
            "base_latency_seconds": left["latency_seconds"],
            "trained_logits": right["logits"], "trained_probabilities": right["probabilities"],
            "trained_prediction_source_index": right["prediction_source_index"],
            "trained_latency_seconds": right["latency_seconds"],
            "chosen_index": decision["chosen_index"], "reason": decision["reason"],
            "result": decision["result"], "before_candidates": deepcopy(before),
            "candidates": deepcopy(after), "raw_bypass_response": raw["response"],
            "raw_bypass_decision": {key: raw["decision"][key]
                                    for key in ("chosen_index", "result", "reason")},
            "candidate_objects_and_relative_order_preserved": True,
            "raw_candidate_list_identity_preserved": raw["actual_candidate_list_identity_preserved"],
            "before_top1": before_top1, "top1": top1,
            "before_top5": before_top5, "top5": top5,
            "corrected": top1 and not before_top1,
            "regressed": before_top1 and not top1,
            "top5_lost": before_top5 and not top5,
            "labels_used_after_both_inference_stages": True,
        })
    return outcomes


def summarize(rows):
    base_latencies = sorted(row["base_latency_seconds"] for row in rows if row["eligible"])
    trained_latencies = sorted(row["trained_latency_seconds"] for row in rows if row["eligible"])
    def percentiles(values):
        if not values:
            return {"p50_seconds": None, "p95_seconds": None}
        middle = len(values) // 2
        median = (values[middle] if len(values) % 2
                  else (values[middle - 1] + values[middle]) / 2)
        return {"p50_seconds": median,
                "p95_seconds": values[max(0, math.ceil(0.95 * len(values)) - 1)]}
    return {"rows": len(rows), "eligible": sum(row["eligible"] for row in rows),
            "ineligible": sum(not row["eligible"] for row in rows),
            "before_top1": sum(row["before_top1"] for row in rows),
            "top1": sum(row["top1"] for row in rows),
            "before_top5": sum(row["before_top5"] for row in rows),
            "top5": sum(row["top5"] for row in rows),
            "corrections": [row["id"] for row in rows if row["corrected"]],
            "regressions": [row["id"] for row in rows if row["regressed"]],
            "top5_losses": [row["id"] for row in rows if row["top5_lost"]],
            "actions": sum(row["chosen_index"] != 0 for row in rows),
            "base_inference_seconds": sum(row["base_latency_seconds"] for row in rows),
            "trained_inference_seconds": sum(row["trained_latency_seconds"] for row in rows),
            "base_latency": percentiles(base_latencies),
            "trained_latency": percentiles(trained_latencies)}


def pilot_gate(metrics):
    all_metrics = metrics["all"]
    losses = any(value["regressions"] or value["top5_losses"] for value in metrics.values())
    accounted = all_metrics["rows"] == PROFILE["rows"]
    passed = accounted and not losses
    return {"passed": passed, "eligible_for_fresh_evaluation": passed,
            "all_rows_accounted": accounted, "zero_first_choice_regressions":
            not any(value["regressions"] for value in metrics.values()),
            "zero_top5_losses": not any(value["top5_losses"] for value in metrics.values()),
            "net_top1_gain": all_metrics["top1"] - all_metrics["before_top1"],
            "actions": all_metrics["actions"]}


def run(args, backend_factory=MlxBackend):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.plan_sha256 != PLAN_SHA256 or digest(args.plan.read_bytes()) != PLAN_SHA256:
        raise ValueError("consensus pilot plan differs")
    resource = validate_resource_smoke(
        args.resource_report, args.resource_report_sha256, args.resource_verification)
    inputs = load_inputs(args.data, args.manifest_sha256,
                         args.lineage_report, args.lineage_sha256)
    if args.manifest_sha256 != PILOT_MANIFEST_SHA256 or args.lineage_sha256 != LINEAGE_SHA256:
        raise ValueError("pilot data or lineage identity differs")
    runner_raw = Path(__file__).read_bytes()
    before, _, _ = validate_frozen_sources(
        inputs, data_dir=args.data, manifest_sha=args.manifest_sha256,
        lineage_path=args.lineage_report, lineage_sha=args.lineage_sha256,
        runner_sha=RESOURCE_HELPER_SHA256, plan_path=args.plan)
    before.update({path_key(args.resource_report): RESOURCE_REPORT_SHA256,
                   path_key(args.resource_verification): RESOURCE_VERIFICATION_SHA256,
                   "scripts/evaluate_consensus_context_pilot.py": digest(runner_raw)})
    verify_hashes(before)
    require_resources(started)
    system = system_message()
    records = [prompt_record(item, system) for item in inputs["rows"]]
    backend = backend_factory()
    base = infer_all_stage(backend, records, adapter=False, started=started)
    require_resources(started, backend)
    trained = infer_all_stage(backend, records, adapter=True, expected=base, started=started)
    require_resources(started, backend)
    label_rows = [item["row"] for item in inputs["rows"]]
    outcomes = evaluate_rows(records, label_rows, base, trained)
    strata = {name: summarize(outcomes[start:end]) for name, start, end in STRATA}
    metrics = {"all": summarize(outcomes), **strata}
    gate = pilot_gate(metrics)
    after, _, _ = validate_frozen_sources(
        inputs, data_dir=args.data, manifest_sha=args.manifest_sha256,
        lineage_path=args.lineage_report, lineage_sha=args.lineage_sha256,
        runner_sha=RESOURCE_HELPER_SHA256, plan_path=args.plan)
    after.update({path_key(args.resource_report): RESOURCE_REPORT_SHA256,
                  path_key(args.resource_verification): RESOURCE_VERIFICATION_SHA256,
                  "scripts/evaluate_consensus_context_pilot.py": digest(runner_raw)})
    verify_hashes(after)
    if before != after or file_digest(Path(__file__)) != digest(runner_raw):
        raise RuntimeError("pilot evaluator inputs changed during execution")
    require_resources(started, backend)
    args.output.mkdir(parents=True)
    (args.output / "source.py").write_bytes(runner_raw)
    report = {
        "schema": "consensus-context-train-pilot-evaluation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "completed": True,
        "scope": "fixed 128-row TRAIN diagnostic; no development, final, or deployment claim",
        "profile": PROFILE, "plan_sha256": PLAN_SHA256,
        "pilot_manifest_sha256": args.manifest_sha256,
        "lineage_report_sha256": args.lineage_sha256,
        "resource_report_sha256": RESOURCE_REPORT_SHA256,
        "resource_verification_sha256": RESOURCE_VERIFICATION_SHA256,
        "runtime_version": RUNTIME_VERSION, "base_revision": MODEL_REVISION,
        "base_model_sha256": MODEL_WEIGHT_SHA256, "adapter_sha256": ADAPTER_SHA256,
        "metrics": metrics, "gate": gate, "outcomes": outcomes,
        "elapsed_seconds": time.monotonic() - started,
        "rss_peak_bytes": rss_bytes(),
        "mlx_peak_bytes": backend.peak(),
        "versions": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm")},
        "source_hashes": before, "source_hashes_after": after,
        "labels_not_used_in_prompt_candidate_selection_or_predictions": True,
        "labels_used_only_after_both_inference_stages": True,
        "candidate_objects_raw_and_source_index_mappings_preserved": True,
        "all_rows_accounted_including_ineligible": True,
        "fixed_policy_no_tuning": True, "training_run": False,
        "family_group_independence_established": False,
        "development_data_read": False, "final_data_read": False,
        "deployed": False,
    }
    report_raw = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    require_resources(started, backend)
    (args.output / "report.json").write_bytes(report_raw)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--lineage-report", type=Path, required=True)
    parser.add_argument("--lineage-sha256", required=True)
    parser.add_argument("--resource-report", type=Path, default=RESOURCE_REPORT)
    parser.add_argument("--resource-report-sha256", required=True)
    parser.add_argument("--resource-verification", type=Path, default=RESOURCE_VERIFICATION)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
