"""Evaluate one accepted adapter on frozen fresh and preservation captures."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
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
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from scripts.long_context_lora_evaluation import prepare_diagnostic, apply_logits, score_predictions
from scripts.long_context_lora_fresh_metrics import paired_document_bootstrap
from scripts.long_context_lora_model import score_choices, require_trainable_contract, tensor_hashes
from scripts.long_context_lora_resources import observe_resources, ResourceObservationError
from scripts import probe_long_context_lora_resources as resources

OUTPUT = ROOT / ".omx/artifacts/long-context-lora-fresh-evaluation-v1"
LIMITS = {**resources.LIMITS, "whole_seconds": 21600.0}


def sha(path):
    return resources.sha256_file(Path(path))


def merge_pins(target, additions):
    for name, digest in additions.items():
        if name in target and target[name] != digest:
            raise ValueError(f"evaluation source pin conflict: {name}")
        target[name] = digest


def verify_pins(pins):
    if (not isinstance(pins, dict) or not pins
            or any(not isinstance(name, str) or not name or ".." in Path(name).parts
                   or not isinstance(digest, str) or len(digest) != 64
                   or set(digest) - set("0123456789abcdef") for name, digest in pins.items())):
        raise ValueError("evaluation source inventory is malformed")
    for name, digest in pins.items():
        if not (ROOT / name).is_file() or sha(ROOT / name) != digest:
            raise ValueError(f"evaluation source changed: {name}")


def load_capture(directory, manifest_sha, training, training_proof_sha):
    from scripts.prepare_long_context_lora_fresh_capture import (
        DATA, PRESERVATION, RUNTIME_REPORT, RUNTIME_REPORT_SHA256, PROFILE)
    if sha(RUNTIME_REPORT) != RUNTIME_REPORT_SHA256:
        raise ValueError("pinned parent runtime report differs")
    runtime = json.loads(RUNTIME_REPORT.read_text())
    path = directory / "manifest.json"
    if sha(path) != manifest_sha:
        raise ValueError("fresh capture manifest hash differs")
    manifest = json.loads(path.read_text())
    if (manifest.get("schema") != "long-context-lora-fresh-capture-v1"
            or manifest.get("completed") is not True or manifest.get("terminal_status") != "success"
            or manifest.get("runtime_version") != "neural-1-867f13596ed5"
            or manifest.get("rows") != 888
            or manifest.get("natural_rows") != 512 or manifest.get("natural_groups") != 387
            or manifest.get("preservation_rows") != 376
            or manifest.get("training_report_sha256") != sha(training["report_path"])
            or manifest.get("training_proof_sha256") != training_proof_sha
            or manifest.get("model_artifact_hashes") != runtime["accepted_weight_hashes"]
            or manifest.get("capture_profile") != PROFILE
            or manifest.get("deployed") is not False
            or any(manifest.get(key) is not True for key in (
                "model_inference_run", "fresh_labels_read", "preservation_labels_read",
                "preservation_reused_without_new_parent_prediction",
                "full_candidate_objects_preserved", "raw_bypass_preserved"))
            or manifest.get("source_hashes") != manifest.get("source_hashes_after")
            or set(manifest.get("file_hashes", {})) != {"items.jsonl", "selection.jsonl"}):
        raise ValueError("fresh capture contract or training identity differs")
    verify_pins(manifest["source_hashes"])
    from scripts import prepare_long_context_lora_fresh_capture as capture
    preservation_manifest_path = PRESERVATION / "manifest.json"
    if sha(preservation_manifest_path) != capture.PRESERVATION_MANIFEST_SHA256:
        raise ValueError("pinned preservation manifest differs")
    preservation_manifest = json.loads(preservation_manifest_path.read_text())
    required = dict(training["source_hashes"])
    for more in (runtime["integrated_runtime_source_hashes"], runtime["accepted_weight_hashes"],
                 preservation_manifest["source_hashes"]):
        merge_pins(required, more)
    fixed = {capture.DATA_MANIFEST: capture.DATA_MANIFEST_SHA256,
             DATA / "fresh_selection.jsonl": capture.FRESH_SELECTION_SHA256,
             capture.FRESH_RAW: capture.FRESH_RAW_SHA256,
             RUNTIME_REPORT: RUNTIME_REPORT_SHA256,
             preservation_manifest_path: capture.PRESERVATION_MANIFEST_SHA256,
             PRESERVATION / "cases.jsonl": preservation_manifest["cases_sha256"],
             PRESERVATION / "selection.jsonl": preservation_manifest["selection_sha256"],
             capture.TRAINING_VERIFIER: capture.TRAINING_VERIFIER_SHA256,
             training["report_path"]: training["proof"]["report_sha256"],
             training["proof_path"]: training_proof_sha}
    for path in (ROOT / "scripts/prepare_long_context_lora_fresh_capture.py",
                 ROOT / "scripts/prepare_final_choice_data.py", ROOT / "scripts/final_choice_capture.py",
                 ROOT / "scripts/evaluate_final_choice_holdout.py"):
        fixed[path] = sha(path)
    merge_pins(required, {resources.relative_path(path): digest for path, digest in fixed.items()})
    if any(manifest["source_hashes"].get(name) != digest for name, digest in required.items()):
        raise ValueError("fresh capture required source inventory is incomplete")
    observed = manifest.get("capture_resources")
    if (not isinstance(observed, dict) or set(observed) != {"stage", "elapsed_seconds", "rss_peak_bytes"}
            or observed["stage"] != "terminal"
            or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                   for value in (observed["elapsed_seconds"], observed["rss_peak_bytes"], manifest.get("elapsed_seconds")))
            or observed["rss_peak_bytes"] > PROFILE["rss_limit_bytes"]
            or not observed["elapsed_seconds"] <= manifest["elapsed_seconds"] <= PROFILE["time_limit_seconds"]):
        raise ValueError("fresh capture resource evidence differs")
    for name, digest in manifest["file_hashes"].items():
        if sha(directory / name) != digest:
            raise ValueError("fresh capture payload hash differs")
    items = [json.loads(line) for line in (directory / "items.jsonl").read_text().splitlines() if line]
    if (len(items) != 888 or len({item["row"]["id"] for item in items}) != 888
            or Counter(item["role"] for item in items) != {
                "fresh_natural": 512, "challenge_preservation": 376}):
        raise ValueError("fresh capture identities or role counts differ")
    natural = [item for item in items if item["role"] == "fresh_natural"]
    if len({item["row"]["group_id"] for item in natural}) != 387:
        raise ValueError("fresh capture document groups differ")
    selection = [json.loads(line) for line in (directory / "selection.jsonl").read_text().splitlines() if line]
    fixed_selection = [json.loads(line) for line in (DATA / "fresh_selection.jsonl").read_text().splitlines() if line]
    if ([{"id": row["id"], "group_id": row["group_id"]} for row in selection] != fixed_selection
            or len(selection) != len(natural)):
        raise ValueError("fresh capture differs from pre-training selected identities")
    identity_keys = ("id", "group_id", "origin", "input", "context", "expected")
    for item, source in zip(natural, selection):
        if (item["stratum"] != "natural"
                or {key: item["row"][key] for key in identity_keys} != source):
            raise ValueError("fresh capture row differs from source selection")
    preserved = [item for item in items if item["role"] == "challenge_preservation"]
    original = [json.loads(line) for line in (PRESERVATION / "cases.jsonl").read_text().splitlines() if line]
    if ([item["row"] for item in preserved] != original
            or any(item["stratum"] != "preservation" for item in preserved)):
        raise ValueError("preservation capture objects differ from the frozen original")
    pins = dict(manifest["source_hashes"])
    merge_pins(pins, {resources.relative_path(path): manifest_sha})
    merge_pins(pins, {resources.relative_path(directory / name): digest
                      for name, digest in manifest["file_hashes"].items()})
    verify_pins(pins)
    return items, manifest, pins


def score_fresh(items, predictions):
    """Called only after all predictions have been saved and model released."""
    combined = score_predictions(items, predictions)
    split = {}
    for role in ("fresh_natural", "challenge_preservation"):
        pairs = [(item, prediction) for item, prediction in zip(items, predictions) if item["role"] == role]
        if len(pairs) != (512 if role == "fresh_natural" else 376):
            raise ValueError("fresh evaluation role denominator differs")
        split[role] = score_predictions([x[0] for x in pairs], [x[1] for x in pairs])
    outcomes = [{"id": item["row"]["id"], "group_id": item["row"]["group_id"],
                 "before_correct": item["row"]["result"] in item["row"]["expected"],
                 "after_correct": prediction["result"] in item["row"]["expected"]}
                for item, prediction in zip(items, predictions) if item["role"] == "fresh_natural"]
    bootstrap = paired_document_bootstrap(outcomes)
    invariants = all(combined["gate"][key] is True for key in (
        "all_rows_accounted", "zero_top5_losses", "raw_preserved", "candidates_preserved"))
    preservation = split["challenge_preservation"]["metrics"]["all"]
    preserved = preservation["before_top1"] == preservation["top1"] == 376
    passed = (bootstrap["rows"] == 512 and bootstrap["groups"] == 387
              and bootstrap["gain_and_interval_passed"] and invariants and preserved)
    return {"combined": combined, "suites": split, "bootstrap": bootstrap,
            "gate": {"passed": passed, "preservation_all_correct": preserved,
                     "invariants_preserved": invariants,
                     "eligible_for_development": passed}}


def finalize_failure(report, report_path, started, mx, checkpoint):
    """Observe cleanup without replacing the inference exception being raised."""
    report["model_references_cleared"] = True
    report["cleanup_errors"] = []
    try:
        gc.collect()
        if mx is not None:
            mx.synchronize()
            report["synchronized_after_failure"] = True
    except Exception as error:
        report["cleanup_errors"].append(f"{type(error).__name__}: {error}")
    if callable(checkpoint):
        try:
            report["failure_terminal_resources"] = checkpoint("failure_after_model_release")
        except Exception as error:
            report["cleanup_errors"].append(f"resource observation: {type(error).__name__}: {error}")
    report["elapsed_seconds"] = time.monotonic() - started
    try:
        report_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    except Exception as error:
        print(f"Failure report could not be written: {type(error).__name__}: {error}", file=sys.stderr)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    # The bridge performs all training gates before any heldout capture is read.
    from scripts.prepare_long_context_lora_fresh_capture import validate_training_proof
    started = time.monotonic()
    code_paths = (Path(__file__), ROOT / "scripts/long_context_lora_fresh_metrics.py",
                  ROOT / "scripts/prepare_long_context_lora_fresh_capture.py")
    code_pins = {resources.relative_path(path): sha(path) for path in code_paths}
    training = validate_training_proof(args.training_proof, args.training_proof_sha256)
    training["proof_path"] = args.training_proof
    pins = dict(training["source_hashes"])
    merge_pins(pins, {resources.relative_path(args.training_proof): args.training_proof_sha256,
                     resources.relative_path(training["report_path"]): training["proof"]["report_sha256"]})
    adapter_dir = training["report_path"].parent / "adapter"
    merge_pins(pins, {resources.relative_path(adapter_dir / "adapters.safetensors"): training["report"]["adapter"]["sha256"],
                     resources.relative_path(adapter_dir / "adapter_config.json"): training["report"]["adapter"]["config_sha256"]})
    verify_pins(pins)
    items, manifest, capture_pins = load_capture(args.capture, args.capture_manifest_sha256,
                                               training, args.training_proof_sha256)
    merge_pins(pins, capture_pins)
    merge_pins(pins, code_pins)
    versions = {name: importlib.metadata.version(name) for name in resources.PACKAGE_VERSIONS}
    if versions != resources.PACKAGE_VERSIONS:
        raise ValueError("MLX evaluation versions differ")
    verify_pins(pins)
    args.output.mkdir(parents=True)
    progress = args.output / "progress.jsonl"
    report_path = args.output / "report.json"
    report = {"schema": "long-context-lora-fresh-evaluation-v1", "completed": False,
              "evaluation_gate_passed": False, "deployed": False, "limits": LIMITS,
              "source_hashes": dict(sorted(pins.items())), "versions": versions,
              "development_final_read": False, "training_report_sha256": sha(training["report_path"]),
              "capture_manifest_sha256": args.capture_manifest_sha256,
              "label_scoring_started_after_model_release": False,
              "labels_scored_after_model_release": False,
              "resource_observations": []}
    model = tokenizer = mx = checkpoint = values = None
    finalized = False
    try:
        baseline = resources.system_resources()
        report["system_baseline"] = baseline
        import mlx.core as mx_module
        from mlx_lm.utils import load
        from mlx_lm.tuner.utils import load_adapters
        mx = mx_module
        mx.set_memory_limit(LIMITS["mlx_peak_bytes"])
        mx.set_cache_limit(LIMITS["cache_bytes"])

        def checkpoint(stage):
            try:
                settled, observation = observe_resources(stage=stage, mx=mx,
                    started=started, baseline=baseline, limits=LIMITS,
                    system_reader=resources.system_resources, rss_reader=resources.rss_bytes)
            except ResourceObservationError as error:
                report["resource_observations"].append(error.record)
                raise
            report["resource_observations"].append(observation)
            resources.write_progress(progress, "resource_observation", observation=observation)
            return settled

        loaded_at = time.monotonic()
        model, tokenizer = load(str(resources.MODEL_DIR), tokenizer_config={"trust_remote_code": False})
        model.freeze()
        load_adapters(model, str(training["report_path"].parent / "adapter"))
        if tensor_hashes(require_trainable_contract(model)) != training["report"]["saved_trainable_tensor_hashes"]:
            raise ValueError("loaded final adapter tensor hashes differ")
        model.eval()
        checkpoint("final_adapter_loaded")
        report["model_load_seconds"] = time.monotonic() - loaded_at
        records = prepare_diagnostic(items, tokenizer)
        checkpoint("source_prompts_prepared")
        predictions, timing = [], []
        path = args.output / "predictions.jsonl"
        with path.open("x") as stream:
            for index, record in enumerate(records, 1):
                row_started = time.monotonic()
                logits = None
                if record["eligible"]:
                    values = score_choices(mx, model, record["tokens"], record["digit_ids"])
                    mx.eval(values)
                    logits = values.tolist()
                    del values
                prediction = apply_logits(record, logits)
                predictions.append(prediction)
                stream.write(json.dumps(prediction, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                checkpoint(f"prediction_{index}")
                timing.append(time.monotonic() - row_started)
        del model, tokenizer
        model = tokenizer = None
        gc.collect()
        checkpoint("model_released_before_labels")
        report["label_scoring_started_after_model_release"] = True
        metrics = score_fresh(items, predictions)
        report["labels_scored_after_model_release"] = True
        verify_pins(pins)
        report.update({"metrics": metrics, "rows": len(predictions),
            "eligible_rows": sum(record["eligible"] for record in records),
            "row_seconds": timing, "row_p95_seconds": sorted(timing)[math.ceil(.95 * len(timing)) - 1],
            "timing_scope": "scoring/policy/prediction serialization/resource observation; excludes model load and prompt preparation; not web end-to-end",
            "predictions_sha256": sha(path), "source_hashes_after": dict(sorted(pins.items())),
            "labels_scored_after_model_release": True})
        report_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        gc.collect()
        report["system_terminal"] = checkpoint("terminal_after_serialization")
        report.update(completed=True, terminal_status="success",
                      evaluation_gate_passed=metrics["gate"]["passed"],
                      elapsed_seconds=time.monotonic() - started)
        report_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        if time.monotonic() - started > LIMITS["whole_seconds"]:
            raise RuntimeError("fresh evaluation whole time exceeded")
        finalized = True
        return report
    except BaseException as error:
        report.update(completed=False, evaluation_gate_passed=False, terminal_status="failed",
                      failure=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        if not finalized:
            model = tokenizer = values = None
            finalize_failure(report, report_path, started, mx, checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-proof", type=Path, required=True)
    parser.add_argument("--training-proof-sha256", required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--capture-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    result = run(parser.parse_args())
    print(json.dumps({"completed": result["completed"], "evaluation_gate_passed": result["evaluation_gate_passed"]}))


if __name__ == "__main__":
    main()
