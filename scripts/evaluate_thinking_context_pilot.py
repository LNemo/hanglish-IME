"""Evaluate the fixed 128-row TRAIN completed-thinking selector pilot."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import probe_thinking_context_resources as thinking
from scripts.probe_consensus_context_resources import (
    MODEL_REVISION, MODEL_WEIGHT_SHA256, RUNTIME_VERSION, digest, file_digest,
    load_inputs, path_key, prompt_record, rss_bytes, system_message,
    validate_frozen_sources, verify_hashes,
)


OUTPUT = ROOT / ".omx/artifacts/thinking-context-train-pilot-v1"
RESOURCE_REPORT = ROOT / ".omx/artifacts/thinking-context-resource-v1/report.json"
RESOURCE_REPORT_SHA256 = "770a9aafe717bbd192d150fd0b2ce232e83227a7b18bba0cc4b2b028297871ea"
RESOURCE_VERIFICATION = ROOT / ".omx/artifacts/thinking-context-work-v1/resource-verification.json"
RESOURCE_VERIFICATION_SHA256 = "ec1af16e977d1209633c2733432fe6ef4ecccde155eb75ded6fa52ce849e8ce7"
RESOURCE_SOURCE_SHA256 = "5ac13782120d8db455bdd763a697da01ff456fdd9733c647673b69a924d2ff5a"
PLAN_SHA256 = "b54e1c5ba570442b5dc96c85bf1d1802e846a4193401e7826c55dc3d3decfff2"
PILOT_MANIFEST_SHA256 = "3e9f84fa3c6defc1075824a984e90945895c4ace8c9609521c328172cd013f86"
LINEAGE_SHA256 = "1720d99b7e6904eb1289cc35d89bcadf01767be5a08fec24c56b3ded6c40d183"
PROFILE = {"rows": 128, "max_prompt_tokens": 1024,
           "max_generated_tokens": 2048, "row_time_limit_seconds": 120,
           "time_limit_seconds": 7200, "rss_limit_bytes": 6 * 1024**3,
           "mlx_peak_limit_bytes": 6 * 1024**3,
           "cache_limit_bytes": 256 * 1024**2}
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


def validate_resource(report_path, report_sha, verification_path):
    if report_sha != RESOURCE_REPORT_SHA256:
        raise ValueError("only the accepted thinking resource probe may authorize the pilot")
    report_raw, report = read_json_once(report_path, report_sha, "thinking resource report")
    proof_raw, proof = read_json_once(
        verification_path, RESOURCE_VERIFICATION_SHA256, "thinking resource verification")
    if (report.get("schema") != "thinking-context-resource-probe-v1"
            or report.get("completed") is not True
            or report.get("resource_feasible") is not True
            or report.get("valid_completed_reasoning_rows") != 3
            or report.get("rows") != 4 or report.get("plan_sha256") != PLAN_SHA256
            or report.get("pilot_manifest_sha256") != PILOT_MANIFEST_SHA256
            or report.get("lineage_report_sha256") != LINEAGE_SHA256
            or report.get("runtime_version") != RUNTIME_VERSION
            or report.get("base_revision") != MODEL_REVISION
            or report.get("base_model_sha256") != MODEL_WEIGHT_SHA256
            or report.get("source_hashes") != report.get("source_hashes_after")
            or report.get("source_hashes", {}).get(
                "scripts/probe_thinking_context_resources.py") != RESOURCE_SOURCE_SHA256
            or report.get("versions") != thinking.VERSIONS
            or report.get("versions_after") != thinking.VERSIONS
            or report.get("sampler") != thinking.SAMPLER
            or report.get("expected_or_labels_read") is not False
            or report.get("accuracy_measured") is not False
            or report.get("adapter_loaded") is not False
            or report.get("deployment_eligible") is not False):
        raise ValueError("thinking resource report gate differs")
    if (proof.get("report_sha256") != report_sha
            or proof.get("rows_verified") != 4
            or proof.get("valid_completed_reasoning_rows") != 3
            or proof.get("resource_feasible") is not True
            or proof.get("source_only_selection_and_tokenization_recomputed") is not True
            or proof.get("seed_markers_logits_candidate_raw_bytes_verified") is not True
            or proof.get("accuracy_measured") is not False
            or proof.get("model_rerun") is not False
            or proof.get("deployed") is not False):
        raise ValueError("thinking resource verification gate differs")
    verify_hashes(report["source_hashes"])
    return {"report_raw": report_raw, "report": report,
            "proof_raw": proof_raw, "proof": proof}


def require_resources(started, backend=None, row_started=None):
    if time.monotonic() - started > PROFILE["time_limit_seconds"]:
        raise RuntimeError("thinking pilot whole time bound exceeded")
    if rss_bytes() > PROFILE["rss_limit_bytes"]:
        raise RuntimeError("thinking pilot RSS bound exceeded")
    if backend is not None and backend.peak() > PROFILE["mlx_peak_limit_bytes"]:
        raise RuntimeError("thinking pilot MLX bound exceeded")
    if (row_started is not None
            and time.monotonic() - row_started > PROFILE["row_time_limit_seconds"]):
        return False
    return True


def run_reasoning_row(backend, model, tokenizer, contract, record, started):
    """Run one row under the pilot's independent 7200/120-second clocks."""
    row_started = time.monotonic()
    seed = thinking.seed_for(record["source"], record["context"])
    emitted, state, timed_out = [], None, False
    generator = backend.start_generation(model, record["prompt_tokens"], seed)
    try:
        while True:
            if not require_resources(started, backend, row_started):
                state = {"terminal": True, "valid": False, "reason": "row_timeout"}
                timed_out = True
                break
            try:
                token, _ = next(generator)
            except StopIteration:
                break
            require_resources(started, backend)
            emitted.append(int(token))
            if not require_resources(started, backend, row_started):
                state = {"terminal": True, "valid": False, "reason": "row_timeout"}
                timed_out = True
                break
            state = thinking.inspect_reasoning(
                emitted, contract["opener"], contract["closer"], contract["eos"],
                lambda values: backend.decode(tokenizer, values))
            if state["terminal"]:
                break
    finally:
        backend.close_generation(generator)
    if not require_resources(started, backend, row_started):
        state = {"terminal": True, "valid": False, "reason": "row_timeout"}
        timed_out = True
    if state is None or not state.get("terminal"):
        state = {"terminal": True, "valid": False, "reason": "token_limit_incomplete"}
    prediction, logits, probabilities = None, [], []
    if state["valid"] and not timed_out:
        full_tokens = record["prompt_tokens"] + emitted + contract["newline"]
        if not require_resources(started, backend, row_started):
            state = {"terminal": True, "valid": False, "reason": "row_timeout"}
            timed_out = True
        else:
            logits = backend.score(
                model, full_tokens, contract["digits"], len(record["choices"]))
        if not require_resources(started, backend, row_started):
            state = {"terminal": True, "valid": False, "reason": "row_timeout"}
            timed_out, logits = True, []
        if (not timed_out and (len(logits) != len(record["choices"])
                or not all(type(value) in (int, float) and math.isfinite(value)
                           for value in logits))):
            raise RuntimeError("thinking pilot final logits are malformed")
        if not timed_out:
            high = max(logits)
            weights = [math.exp(value - high) for value in logits]
            probabilities = [value / sum(weights) for value in weights]
            position = max(range(len(logits)), key=lambda index: (logits[index], -index))
            prediction = record["choices"][position][0]
    decision = thinking.single_model_decision(record, prediction)
    return {"seed": seed, "prompt_tokens": record["prompt_tokens"],
            "emitted_tokens": emitted, "prompt_token_count": len(record["prompt_tokens"]),
            "generated_token_count": len(emitted), "reasoning_valid": state["valid"],
            "completion_reason": state["reason"], "digit_logits": logits,
            "digit_probabilities": probabilities,
            "prediction_source_index": prediction, "decision": decision,
            "latency_seconds": time.monotonic() - row_started,
            "upstream_may_prefetch_one_unused_token": True}


def fallback_row(record, reason):
    decision = thinking.single_model_decision(record, None)
    return {"seed": thinking.seed_for(record["source"], record["context"]),
            "prompt_tokens": record["prompt_tokens"], "emitted_tokens": [],
            "prompt_token_count": len(record["prompt_tokens"]),
            "generated_token_count": 0, "reasoning_valid": False,
            "completion_reason": reason, "digit_logits": [],
            "digit_probabilities": [], "prediction_source_index": None,
            "decision": decision, "latency_seconds": 0.0,
            "upstream_may_prefetch_one_unused_token": False}


def attach_labels(records, label_rows, inference):
    """Consult TRAIN labels only after all 128 source-only inference outcomes exist."""
    if len(records) != PROFILE["rows"] or len(label_rows) != len(records) or len(inference) != len(records):
        raise ValueError("thinking pilot row cardinality differs")
    outcomes = []
    for index, (record, label_row, result) in enumerate(zip(records, label_rows, inference)):
        if label_row.get("id") != record["id"]:
            raise RuntimeError("thinking pilot label identity differs")
        expected = label_row.get("expected")
        if (not isinstance(expected, list) or not expected
                or not all(isinstance(value, str) and value for value in expected)):
            raise ValueError("thinking pilot expected labels are malformed")
        decision = result["decision"]
        before, after = record["candidates"], decision["candidates"]
        before_top1 = before[0]["text"] in expected
        top1 = after[0]["text"] in expected
        before_top5 = any(candidate["text"] in expected for candidate in before)
        top5 = any(candidate["text"] in expected for candidate in after)
        outcomes.append({"id": record["id"], "role": record["role"],
                         "record_index": index,
                         **{key: value for key, value in result.items() if key != "decision"},
                         "chosen_index": decision["chosen_index"],
                         "decision_reason": decision["reason"],
                         "result": decision["result"],
                         "before_candidates": deepcopy(before),
                         "candidates": deepcopy(after),
                         "raw_bypass_response": decision["raw_bypass_response"],
                         "raw_canonical_json_before": decision["raw_canonical_json_before"],
                         "raw_canonical_json_after": decision["raw_canonical_json_after"],
                         "raw_sha256_before": decision["raw_sha256_before"],
                         "raw_sha256_after": decision["raw_sha256_after"],
                         "before_top1": before_top1, "top1": top1,
                         "before_top5": before_top5, "top5": top5,
                         "corrected": top1 and not before_top1,
                         "regressed": before_top1 and not top1,
                         "top5_lost": before_top5 and not top5,
                         "labels_used_after_all_128_inference_rows": True})
    return outcomes


def summarize(rows):
    latencies = sorted(row["latency_seconds"] for row in rows
                       if row["generated_token_count"] > 0)
    def percentile(fraction):
        return (latencies[max(0, math.ceil(fraction * len(latencies)) - 1)]
                if latencies else None)
    reasons = Counter(row["completion_reason"] for row in rows)
    budget_fallbacks = (reasons["token_limit_incomplete"]
                        + reasons["prompt_too_long"])
    return {"rows": len(rows),
            "before_top1": sum(row["before_top1"] for row in rows),
            "top1": sum(row["top1"] for row in rows),
            "before_top5": sum(row["before_top5"] for row in rows),
            "top5": sum(row["top5"] for row in rows),
            "corrections": [row["id"] for row in rows if row["corrected"]],
            "regressions": [row["id"] for row in rows if row["regressed"]],
            "top5_losses": [row["id"] for row in rows if row["top5_lost"]],
            "actions": sum(row["chosen_index"] != 0 for row in rows),
            "valid_completed_reasoning": sum(row["reasoning_valid"] for row in rows),
            "fallbacks": sum(not row["reasoning_valid"] for row in rows),
            "completion_rate": (sum(row["reasoning_valid"] for row in rows) / len(rows)
                                if rows else 0.0),
            "completion_reasons": dict(sorted(reasons.items())),
            "budget_fallbacks": budget_fallbacks,
            "budget_fallback_rate": budget_fallbacks / len(rows) if rows else 0.0,
            "row_timeout_fallbacks": reasons["row_timeout"],
            "generated_tokens": sum(row["generated_token_count"] for row in rows),
            "latency_p50_seconds": percentile(0.5),
            "latency_p95_seconds": percentile(0.95),
            "inference_seconds": sum(row["latency_seconds"] for row in rows)}


def pilot_gate(metrics):
    all_rows = metrics["all"]
    exact_counts = (all_rows["rows"] == 128 and metrics["natural"]["rows"] == 64
                    and metrics["challenge_single"]["rows"] == 32
                    and metrics["challenge_sentence"]["rows"] == 32)
    regressions = any(value["regressions"] for value in metrics.values())
    top5_losses = any(value["top5_losses"] for value in metrics.values())
    gain = all_rows["top1"] - all_rows["before_top1"]
    passed = exact_counts and gain >= 1 and not regressions and not top5_losses and all_rows["actions"] > 0
    return {"passed": passed, "eligible_for_fresh_evaluation": passed,
            "all_rows_accounted": exact_counts, "net_top1_gain": gain,
            "corrections": len(all_rows["corrections"]),
            "zero_first_choice_regressions": not regressions,
            "zero_top5_losses": not top5_losses,
            "actions": all_rows["actions"]}


def run(args, backend_factory=thinking.ThinkingBackend):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.plan_sha256 != PLAN_SHA256 or digest(args.plan.read_bytes()) != PLAN_SHA256:
        raise ValueError("thinking pilot plan differs")
    validate_resource(args.resource_report, args.resource_report_sha256,
                      args.resource_verification)
    inputs = load_inputs(args.data, args.manifest_sha256,
                         args.lineage_report, args.lineage_sha256)
    if args.manifest_sha256 != PILOT_MANIFEST_SHA256 or args.lineage_sha256 != LINEAGE_SHA256:
        raise ValueError("thinking pilot data identity differs")
    runner_raw = Path(__file__).read_bytes()
    before, _, _ = validate_frozen_sources(
        inputs, data_dir=args.data, manifest_sha=args.manifest_sha256,
        lineage_path=args.lineage_report, lineage_sha=args.lineage_sha256,
        runner_sha=thinking.RESOURCE_HELPER_SHA256, plan_path=thinking.SELECTION_PLAN)
    before.update({path_key(args.plan): PLAN_SHA256,
                   path_key(args.resource_report): RESOURCE_REPORT_SHA256,
                   path_key(args.resource_verification): RESOURCE_VERIFICATION_SHA256,
                   "scripts/probe_thinking_context_resources.py": RESOURCE_SOURCE_SHA256,
                   "scripts/evaluate_thinking_context_pilot.py": digest(runner_raw)})
    before.update(thinking.package_sources())
    versions = thinking.require_versions()
    verify_hashes(before)
    require_resources(started)
    system = system_message()
    records = [thinking.snapshot_raw_record(prompt_record(item, system))
               for item in inputs["rows"]]
    backend = backend_factory()
    model, tokenizer = backend.load_model()
    try:
        require_resources(started, backend)
        contract = backend.token_contract(tokenizer)
        prepared = []
        for record in records:
            tokens = backend.tokens(tokenizer, record["messages"])
            decoded = backend.decode(tokenizer, tokens)
            if "<think>\n\n</think>" in decoded or "<think></think>" in decoded:
                raise RuntimeError("thinking template inserted a closed empty block")
            prepared.append({**record, "prompt_tokens": tokens})
        inference = []
        valid_count = fallback_count = 0
        for index, record in enumerate(prepared):
            if len(record["choices"]) < 2:
                outcome = fallback_row(record, "insufficient_choices")
            elif len(record["prompt_tokens"]) > PROFILE["max_prompt_tokens"]:
                outcome = fallback_row(record, "prompt_too_long")
            else:
                outcome = run_reasoning_row(
                    backend, model, tokenizer, contract, record, started)
            inference.append(outcome)
            valid_count += int(outcome["reasoning_valid"])
            fallback_count += int(not outcome["reasoning_valid"])
            print(json.dumps({"stage": "thinking_pilot_row", "completed_rows": index + 1,
                              "total_rows": PROFILE["rows"],
                              "valid_completed": valid_count,
                              "fallbacks": fallback_count,
                              "elapsed_seconds": time.monotonic() - started},
                             sort_keys=True), flush=True)
            require_resources(started, backend)
    finally:
        del model, tokenizer
        gc.collect()
        backend.release()
    label_rows = [item["row"] for item in inputs["rows"]]
    outcomes = attach_labels(prepared, label_rows, inference)
    metrics = {"all": summarize(outcomes)}
    metrics.update({name: summarize(outcomes[start:end])
                    for name, start, end in STRATA})
    gate = pilot_gate(metrics)
    after, _, _ = validate_frozen_sources(
        inputs, data_dir=args.data, manifest_sha=args.manifest_sha256,
        lineage_path=args.lineage_report, lineage_sha=args.lineage_sha256,
        runner_sha=thinking.RESOURCE_HELPER_SHA256, plan_path=thinking.SELECTION_PLAN)
    after.update({path_key(args.plan): PLAN_SHA256,
                  path_key(args.resource_report): RESOURCE_REPORT_SHA256,
                  path_key(args.resource_verification): RESOURCE_VERIFICATION_SHA256,
                  "scripts/probe_thinking_context_resources.py": RESOURCE_SOURCE_SHA256,
                  "scripts/evaluate_thinking_context_pilot.py": digest(runner_raw)})
    after.update(thinking.package_sources())
    versions_after = thinking.require_versions()
    verify_hashes(after)
    if before != after or versions != versions_after or file_digest(Path(__file__)) != digest(runner_raw):
        raise RuntimeError("thinking pilot inputs changed during execution")
    require_resources(started, backend)
    args.output.mkdir(parents=True)
    (args.output / "source.py").write_bytes(runner_raw)
    report = {"schema": "thinking-context-train-pilot-evaluation-v1",
              "created_at": datetime.now(timezone.utc).isoformat(), "completed": True,
              "scope": "fixed 128-row TRAIN diagnostic; no downstream or deployment claim",
              "profile": PROFILE, "sampler": thinking.SAMPLER,
              "plan_sha256": PLAN_SHA256,
              "resource_report_sha256": RESOURCE_REPORT_SHA256,
              "resource_verification_sha256": RESOURCE_VERIFICATION_SHA256,
              "pilot_manifest_sha256": args.manifest_sha256,
              "lineage_report_sha256": args.lineage_sha256,
              "runtime_version": RUNTIME_VERSION, "base_revision": MODEL_REVISION,
              "base_model_sha256": MODEL_WEIGHT_SHA256,
              "metrics": metrics, "gate": gate, "outcomes": outcomes,
              "versions": versions, "versions_after": versions_after,
              "source_hashes": before, "source_hashes_after": after,
              "elapsed_seconds": time.monotonic() - started,
              "rss_peak_bytes": rss_bytes(), "mlx_peak_bytes": backend.peak(),
              "labels_not_used_in_selection_prompt_generation_or_prediction": True,
              "labels_used_only_after_all_128_inference_rows": True,
              "all_candidate_raw_and_source_mappings_preserved": True,
              "fixed_recipe_no_tuning": True, "adapter_loaded": False,
              "training_run": False, "development_data_read": False,
              "final_data_read": False, "deployed": False}
    report_raw = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    require_resources(started, backend)
    (args.output / "report.json").write_bytes(report_raw)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=thinking.PILOT_DATA)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--lineage-report", type=Path, default=thinking.LINEAGE)
    parser.add_argument("--lineage-sha256", required=True)
    parser.add_argument("--resource-report", type=Path, default=RESOURCE_REPORT)
    parser.add_argument("--resource-report-sha256", required=True)
    parser.add_argument("--resource-verification", type=Path, default=RESOURCE_VERIFICATION)
    parser.add_argument("--plan", type=Path, default=thinking.PLAN)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
