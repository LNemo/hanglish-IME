"""Single-use, TRAIN-gated worker for the complete pairwise fresh cohort.

Invoke through run_difference_pairwise_lora_fresh.py. The supervisor owns the
external deadline; this worker binds source projection, qualified prompts and
the actual local MLX callbacks. Quality results still require the independent
terminal execution verifier and do not authorize production deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import prepare_difference_pairwise_lora_fresh as bridge
from scripts.difference_pairwise_lora_evaluation import canonical
from scripts.difference_pairwise_lora_fresh_execution import execute
from scripts.difference_pairwise_lora_fresh_runtime import (
    LIMITS, VERSIONS, load_verified_runtime, qualify_rows,
)
from scripts.difference_pairwise_lora_inference import prepare_inference_selection

OUTPUT = ROOT / ".omx/artifacts/difference-pairwise-lora-fresh-v1"
LIMIT_SECONDS = 21600.0
SCHEMA = "difference-pairwise-lora-fresh-worker-v1"
TARGET_PATH = bridge.PARENT + "/targets.jsonl"
TARGET_SHA256 = "957f784665edca86162e0b677472b1e2f4ce716edd7991e87992da6568c0182f"
BUDGET_PATH = ".omx/artifacts/difference-pairwise-lora-budget-v1-proof.json"
BUDGET_SHA256 = "e59fe3cf9293f2df7cfa19e7e9e198f4691489d41c11215efb19877141888d6a"
FRESH_SOURCES = (
    "scripts/prepare_difference_pairwise_lora_fresh.py",
    "scripts/difference_pairwise_lora_inference.py",
    "scripts/difference_pairwise_lora_fresh_metrics.py",
    "scripts/verify_difference_pairwise_lora_fresh_metrics.py",
    "scripts/difference_pairwise_lora_fresh_execution.py",
    "scripts/difference_pairwise_lora_fresh_runtime.py",
    "scripts/evaluate_difference_pairwise_lora_fresh.py",
    "scripts/run_difference_pairwise_lora_fresh.py",
)
sha = bridge.sha


def write_new(path, value):
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def estimate_work(pair_count, timing):
    """Retain the measured probe's 1.5 safety factor and 300-second overhead."""
    if type(pair_count) is not int or pair_count < 0:
        raise ValueError("invalid complete-cohort pair count")
    inference = timing.get("pair_inference_p95")
    finalization = timing.get("save_reload_finalization_seconds")
    if not _finite(inference) or inference <= 0 or not _finite(finalization):
        raise ValueError("verified timing observations required")
    seconds = 1.5 * (pair_count * inference + finalization) + 300.0
    if not _finite(seconds) or seconds > LIMIT_SECONDS:
        raise ValueError("complete fresh inference estimate exceeds six-hour cap")
    return {"pair_count": pair_count, "pair_inference_p95": inference,
            "finalization_allowance_seconds": finalization, "safety_factor": 1.5,
            "overhead_seconds": 300.0, "estimate_seconds": seconds,
            "limit_seconds": LIMIT_SECONDS}


def read_inputs(training):
    """Reconstruct parent-only rows independently of the projected JSONL."""
    manifest_path = bridge.OUTPUT / "manifest.json"
    inputs_path = bridge.OUTPUT / "inputs.jsonl"
    manifest = json.loads(manifest_path.read_bytes())
    if (manifest.get("schema") != bridge.SCHEMA
            or manifest.get("training") != training
            or manifest.get("runtime_version") != bridge.VERSION
            or [manifest.get(k) for k in ("rows", "fresh_rows", "fresh_groups", "preservation_rows")]
                != [888, 512, 387, 376]
            or any(manifest.get(k) is not False for k in
                   ("target_files_read", "model_loaded", "child_predictions_used",
                    "inference_authorized", "promotion_allowed"))
            or manifest.get("historical_targets_exposed_in_numeric_lane") is not True
            or manifest.get("historical_semantic_family_independence") != "unestablished"
            or manifest.get("upstream_pretraining_overlap") != "unknown"):
        raise ValueError("fresh input manifest contract differs")
    bridge.verify_pins(bridge.PINS)
    parent_proof = json.loads((ROOT / (bridge.PARENT + "-proof.json")).read_bytes())["payload"]
    runtime = json.loads((ROOT / ".omx/artifacts/numeric-structure-integration-v1/report.json").read_bytes())
    if (parent_proof.get("verification_passed") is not True
            or parent_proof.get("runtime_version") != bridge.VERSION
            or runtime.get("integrated_runtime_version") != bridge.VERSION
            or runtime.get("accepted_weight_hashes") != parent_proof.get("model_artifacts")):
        raise ValueError("verified parent provenance differs")
    expected_pins = {**bridge.PINS, **runtime["integrated_runtime_source_hashes"],
                     **parent_proof["model_artifacts"],
                     "scripts/prepare_difference_pairwise_lora_fresh.py": sha(bridge.__file__)}
    if manifest.get("source_hashes") != expected_pins:
        raise ValueError("fresh source inventory differs")
    bridge.verify_pins(expected_pins)
    rows = bridge.project_parent_rows(
        bridge.read_jsonl(ROOT / bridge.PARENT / "sources.jsonl"),
        bridge.read_jsonl(ROOT / bridge.PARENT / "captures.jsonl"),
        bridge.read_jsonl(ROOT / ".omx/artifacts/long-context-lora-data-v1-review2/fresh_selection.jsonl"),
        parent_proof["model_artifacts"],
    )
    expected_bytes = b"".join(canonical(row) + b"\n" for row in rows)
    if (inputs_path.read_bytes() != expected_bytes
            or hashlib.sha256(expected_bytes).hexdigest() != manifest.get("inputs_sha256")):
        raise ValueError("fresh parent projection does not replay exactly")
    return rows, manifest


def worker_identity():
    invocation = Path(sys.executable).absolute()
    expected = ROOT / ".venv-mlx/bin/python"
    packages = {name: importlib.metadata.version(name) for name in VERSIONS}
    if invocation != expected or packages != VERSIONS:
        raise ValueError("use the qualified .venv-mlx Python invocation and packages")
    return {"invocation_path": str(invocation), "resolved_path": str(invocation.resolve()),
            "resolved_sha256": sha(invocation.resolve()), "package_versions": packages}


def prepare_launch():
    # TRAIN replay is first: an unfinished/failed fit must not open fresh files,
    # create an output, import MLX/transformers, or load a tokenizer.
    training = bridge.require_training_proof()
    identity = worker_identity()
    if not bridge.OUTPUT.exists():
        bridge.prepare()
    rows, manifest = read_inputs(training)
    if sha(ROOT / BUDGET_PATH) != BUDGET_SHA256:
        raise ValueError("verified timing proof changed")
    budget_envelope = json.loads((ROOT / BUDGET_PATH).read_bytes())
    budget = budget_envelope["payload"]
    if (budget_envelope.get("digest") != hashlib.sha256(canonical(budget)).hexdigest()
            or budget.get("passed") is not True):
        raise ValueError("verified timing proof envelope differs")
    count = sum(len(prepare_inference_selection(row["role"], row)["pairs"]) for row in rows)
    estimate = estimate_work(count, budget["timing_summary"])
    training_proof = json.loads(bridge.TRAIN_PROOF.read_bytes())["payload"]
    train_report_path = ROOT / training_proof["sole_output"] / "report.json"
    train_report = json.loads(train_report_path.read_bytes())
    pins = {**manifest["source_hashes"], **training_proof["source_hashes"],
            **{path: sha(ROOT / path) for path in FRESH_SOURCES},
            str(bridge.TRAIN_PROOF.relative_to(ROOT)): training["sha256"],
            str(train_report_path.relative_to(ROOT)): training_proof["artifacts"]["report.json"],
            BUDGET_PATH: BUDGET_SHA256,
            str((bridge.OUTPUT / "manifest.json").relative_to(ROOT)): sha(bridge.OUTPUT / "manifest.json"),
            str((bridge.OUTPUT / "inputs.jsonl").relative_to(ROOT)): manifest["inputs_sha256"]}
    for path_key, hash_key in (("path", "sha256"), ("config_path", "config_sha256")):
        pins[train_report["adapter"][path_key]] = train_report["adapter"][hash_key]
    if TARGET_PATH in pins:
        raise ValueError("fresh targets must not enter the pre-inference read inventory")
    bridge.verify_pins(pins)
    return {"schema": "difference-pairwise-lora-fresh-launch-plan-v1",
            "training": training, "source_hashes": pins, "worker_python": identity,
            "budget": estimate, "limits": LIMITS,
            "inputs": str(bridge.OUTPUT.relative_to(ROOT)),
            "targets": {"path": TARGET_PATH, "sha256": TARGET_SHA256},
            "adapter_tensor_hashes": train_report["final_trainable_tensor_hashes"],
            "historical_targets_exposed_in_numeric_lane": True,
            "promotion_allowed": False, "resume_allowed": False}


def run_worker(worker_claim, start_monotonic, *, monotonic=time.monotonic):
    launch_path = OUTPUT / "launch.json"
    launch = json.loads(launch_path.read_bytes())
    if (launch.get("schema") != "difference-pairwise-lora-fresh-launch-v1"
            or not isinstance(worker_claim, str) or not worker_claim
            or launch.get("worker_claim") != worker_claim
            or launch.get("consumed") is not True or launch.get("released") is not False
            or launch.get("sole_output") != str(OUTPUT.relative_to(ROOT))
            or not _finite(start_monotonic) or launch.get("start_monotonic") != start_monotonic
            or not 0 <= monotonic() - start_monotonic < LIMIT_SECONDS):
        raise ValueError("single-use supervised worker launch differs")
    write_new(OUTPUT / "worker-start.json", {
        "schema": "difference-pairwise-lora-fresh-worker-start-v1",
        "worker_claim": worker_claim, "pid": os.getpid(),
        "launch_sha256": sha(launch_path), "start_monotonic": monotonic(),
        "supervisor_start_monotonic": start_monotonic,
    })
    report = {"schema": SCHEMA, "completed": False, "terminal_status": "running",
              "quality_gate_passed": False, "execution_verified": False,
              "promotion_allowed": False, "resume_allowed": False,
              "worker_claim": worker_claim, "launch_sha256": sha(launch_path),
              "target_accessed": False, "model_loaded": False, "model_released": False}
    runtime = None
    error = None
    try:
        plan = prepare_launch()
        if canonical(plan) != canonical(launch.get("plan")):
            raise ValueError("launch plan or source snapshot changed before worker")
        rows, _ = read_inputs(plan["training"])
        runtime = load_verified_runtime()
        runtime.started = start_monotonic
        # Resource observations share the external process clock; baseline is
        # captured by the actual runtime before qualification and model load.
        qualification_before = runtime.checkpoint("before_qualification")
        qualified = qualify_rows(rows, runtime.tokenizer)
        if len(qualified) != plan["budget"]["pair_count"]:
            raise ValueError("qualified pair count differs from launch estimate")
        qualification_after = runtime.checkpoint("after_qualification")
        remaining_estimate = 1.5 * (len(qualified) * plan["budget"]["pair_inference_p95"]
                                  + plan["budget"]["finalization_allowance_seconds"])
        if monotonic() - start_monotonic + remaining_estimate > LIMIT_SECONDS:
            raise ValueError("qualification consumed the available inference budget")
        write_new(OUTPUT / "qualifications.json", qualified)
        qualification_hash = sha(OUTPUT / "qualifications.json")
        report["qualification"] = {"sha256": qualification_hash, "pairs": len(qualified),
                                   "before": qualification_before, "after": qualification_after}
        pins = {**plan["source_hashes"],
                str((OUTPUT / "qualifications.json").relative_to(ROOT)): qualification_hash,
                str(launch_path.relative_to(ROOT)): report["launch_sha256"]}

        def validate_sources(stage):
            runtime.validate_sources(stage)
            bridge.verify_pins(pins)
            return {"passed": True, "stage": stage, "source_hashes": pins}

        def load_targets():
            # execute() closes/reopens both streams, releases the model and
            # validates resources/sources before invoking this sole label read.
            report["target_accessed"] = True
            path = ROOT / TARGET_PATH
            body = path.read_bytes()
            if hashlib.sha256(body).hexdigest() != TARGET_SHA256:
                raise ValueError("frozen fresh target file differs")
            targets = [json.loads(line) for line in body.splitlines()]
            report["target_sha256"] = TARGET_SHA256
            return targets

        result = execute(rows, qualified, OUTPUT / "capture", score_pair=runtime.score_pair,
                         release_model=runtime.release_model, load_targets=load_targets,
                         checkpoint=runtime.checkpoint, validate_sources=validate_sources,
                         monotonic=monotonic)
        validate_sources("worker_finished")
        report["capture_report_sha256"] = sha(OUTPUT / "capture/report.json")
        report["completed"] = result["completed"] is True
        report["quality_gate_passed"] = result["metrics"]["gate"]["passed"] is True
        report["terminal_status"] = result["terminal_status"]
        report["verified_adapter_tensor_hashes"] = dict(runtime.tensor_hashes)
    except BaseException as exc:
        error = exc
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if runtime is not None:
            if not runtime.released:
                try:
                    evidence = runtime.release_model()
                    if not isinstance(evidence, dict) or evidence.get("released") is not True:
                        raise ValueError("worker cleanup failed to release model")
                except BaseException as exc:
                    error = error or exc
                    report["cleanup_error"] = {"type": type(exc).__name__, "message": str(exc)}
            report["model_loaded"] = runtime.model_loaded
            report["model_released"] = runtime.released and runtime.model is None
            report["resource_baseline"] = runtime.baseline
            report["resource_records"] = runtime.resource_records
        report["elapsed_seconds"] = monotonic() - start_monotonic
        if (not _finite(report["elapsed_seconds"]) or report["elapsed_seconds"] > LIMIT_SECONDS
                or (report["completed"] and not report["model_released"])):
            error = error or RuntimeError("worker lifetime or model release bound exceeded")
        if error is not None:
            report["completed"] = report["quality_gate_passed"] = False
            report["terminal_status"] = "failed"
        capture_path = OUTPUT / "capture/report.json"
        if capture_path.is_file():
            report["capture_report_sha256"] = sha(capture_path)
        write_new(OUTPUT / "worker-report.json", report)
    return 1 if error is not None else (0 if report["quality_gate_passed"] else 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-claim", required=True)
    parser.add_argument("--start-monotonic", required=True, type=float)
    args = parser.parse_args()
    try:
        return run_worker(args.worker_claim, args.start_monotonic)
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
