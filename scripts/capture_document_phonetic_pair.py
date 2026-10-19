#!/usr/bin/env python3
"""Authorize and capture fixed-model source-only word predictions.

``authorize`` is model-free. ``capture`` replays the complete authorization
before importing Torch, then performs source-only generation for one arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_observed_phonetic_pair as base

INPUTS = ROOT / ".omx/artifacts/document-phonetic-inputs-v1/inputs.jsonl"
INPUTS_SHA = "c9aee2f9a1ebd4b9ec5754c5145c4c2ddfd8c1eae6846fb6dc2774c8ff38376b"
TRAINING_PROOF_SCHEMA = "observed-phonetic-training-independent-v1"
AUTHORIZATION_SCHEMA = "document-phonetic-capture-authorization-v1"
TRAINING_VERIFIER = ROOT / "scripts/verify_observed_phonetic_training.py"
PARAMETER_FINITE_CHECKS = sum(
    update == 1 or update % 128 == 0 or update == base.UPDATES
    for update in range(1, base.UPDATES + 1)
)


def original_swap_baseline(resource: dict) -> int:
    """Recover the system swap baseline from a terminal resource sample."""
    base.require(isinstance(resource, dict), "training resource sample missing")
    used = resource.get("swap_used_bytes")
    growth = resource.get("swap_growth_bytes")
    base.require(type(used) is int and used >= 0 and type(growth) is int,
                 "training swap evidence malformed")
    baseline = used - growth
    base.require(baseline >= 0, "training swap baseline is negative")
    return baseline


def _finite_nonnegative(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def authenticate_training(directory: Path, arm: str, prepared: dict,
                          prepared_sha: str) -> dict:
    """Validate a completed training arm without loading its checkpoint."""
    directory = Path(directory)
    report_path = directory / "report.json"
    completion_path = directory / "completion.json"
    checkpoint_path = directory / "phonetic_transducer.pt"
    metadata_path = directory / "phonetic_transducer.json"
    updates_path = directory / "updates.jsonl"
    proof_path = directory / "independent-verification.json"
    report = base.read_json(report_path)
    completion = base.read_json(completion_path)
    metadata = base.read_json(metadata_path)
    proof = base.read_json(proof_path)
    base.require(
        report.get("schema") == "observed-phonetic-pair-training-v1"
        and report.get("completed") is True and report.get("arm") == arm
        and report.get("recipe") == prepared["recipe"]
        and report.get("model_config") == prepared["model_config"] == base.CONFIG
        and report.get("prepared_sha256") == prepared_sha
        and report.get("evaluation_inputs") == prepared["evaluation_inputs"]
        and report.get("source_hashes") == prepared["source_hashes"]
        and report.get("source_hashes_after") == prepared["source_hashes"]
        and report.get("schedule") == prepared["arms"][arm]
        and report.get("updates_completed") == base.UPDATES
        and report.get("parameter_finiteness_checks") == PARAMETER_FINITE_CHECKS
        and report.get("checkpoint_selection") == "final_fixed_update_only"
        and report.get("limits") == base.LIMITS
        and report.get("evaluation_targets_read") is False
        and report.get("production_changed") is False
        and all(_finite_nonnegative(report.get(name))
                for name in ("mean_loss", "final_loss", "max_gradient_norm")),
        "training contract differs")
    base.verify_pins(report["source_hashes"])
    base.require(
        completion.get("completed") is True
        and completion.get("schema") == "observed-phonetic-pair-completion-v1"
        and completion.get("report_sha256") == base.sha256_file(report_path)
        and base.resource_within_limits(completion.get("resource_after_report"))
        and base.resource_within_limits(report.get("final_resource")),
        "training completion differs")
    original_swap_baseline(report["final_resource"])
    original_swap_baseline(completion["resource_after_report"])
    initial = report.get("initial", {})
    final = report.get("final", {})
    initial_path = Path(initial.get("path", ""))
    base.require(
        final.get("reload_exact") is True
        and final.get("tensor_sha256") == final.get("reload_tensor_sha256")
        and final.get("tensor_sha256") != initial.get("tensor_sha256")
        and len(initial.get("checkpoint_sha256", "")) == 64
        and len(initial.get("tensor_sha256", "")) == 64
        and initial_path.is_file()
        and base.sha256_file(initial_path) == initial["checkpoint_sha256"]
        and base.sha256_file(checkpoint_path) == final.get("checkpoint_sha256")
        and base.sha256_file(updates_path) == report.get("updates_sha256"),
        "checkpoint/update evidence differs")
    base.require(
        metadata == {
            "format_version": 1,
            "checkpoint_sha256": final["checkpoint_sha256"],
            "config": prepared["model_config"],
            "arm": arm,
            "initial_checkpoint_sha256": initial["checkpoint_sha256"],
            "initial_tensor_sha256": initial["tensor_sha256"],
            "final_tensor_sha256": final["tensor_sha256"],
        }, "checkpoint metadata differs")
    base.require(
        proof.get("schema") == TRAINING_PROOF_SCHEMA
        and proof.get("verified") is True
        and proof.get("arm") == arm
        and proof.get("report_sha256") == base.sha256_file(report_path)
        and proof.get("completion_sha256") == base.sha256_file(completion_path)
        and proof.get("checkpoint_sha256") == final["checkpoint_sha256"]
        and proof.get("checkpoint_metadata_sha256") == base.sha256_file(metadata_path)
        and proof.get("updates_sha256") == report["updates_sha256"]
        and proof.get("prepared_sha256") == prepared_sha
        and proof.get("probe_report_sha256") == report.get("probe_report_sha256")
        and proof.get("probe_completion_sha256") == report.get("probe_completion_sha256")
        and proof.get("verifier_sha256") == base.sha256_file(TRAINING_VERIFIER)
        and proof.get("counts") == {
            "pairs": 51361 if arm == "control" else 84345,
            "updates": base.UPDATES,
            "presentations": base.UPDATES * base.BATCH_SIZE,
            "parameter_finiteness_checks": PARAMETER_FINITE_CHECKS,
        }
        and proof.get("schedule") == {
            key: value for key, value in report["schedule"].items()
            if key not in {"probe_beam_inputs", "probe_update_positions"}
        }
        and proof.get("training_aggregates") == {
            "updates": base.UPDATES,
            "mean_loss": report["mean_loss"],
            "final_loss": report["final_loss"],
            "max_gradient_norm": report["max_gradient_norm"],
            "parameter_finiteness_checks": PARAMETER_FINITE_CHECKS,
        }
        and proof.get("limits") == base.LIMITS,
        "independent training proof differs")
    contracts = proof.get("contracts_verified")
    required_contracts = {
        "prepared_v3_and_source_pins": True,
        "accepted_probe_and_shared_initial": True,
        "schedule_indices_costs_and_exposure": True,
        "finite_update_ledger": True,
        "finiteness_flags_exact": True,
        "final_changed_and_reload_recorded_exact": True,
        "checkpoint_file_and_metadata_hashes": True,
        "atomic_resources_and_finalization": True,
        "failure_artifact_absent": True,
        "evaluation_targets_read": False,
        "neural_or_checkpoint_load_performed": False,
    }
    base.require(isinstance(contracts, dict)
                 and all(contracts.get(key) is value
                         for key, value in required_contracts.items()),
                 "independent training proof contracts differ")
    resources = proof.get("resources")
    base.require(
        isinstance(resources, dict)
        and isinstance(resources.get("main"), dict)
        and isinstance(resources.get("finalization"), dict)
        and resources["main"].get("operations") == base.UPDATES + 2
        and resources["main"].get("final") == report["final_resource"]
        and resources["finalization"].get("operations") == 2
        and resources["finalization"].get("final")
            == completion["resource_after_report"],
        "independent training resource proof differs")
    for name in ("failure.json", "failure-report.json", "resource-watchdog-failure.json"):
        base.require(not (directory / name).exists(), "training failure artifact exists")
    return report


def load_inputs() -> list[dict]:
    base.require(base.sha256_file(INPUTS) == INPUTS_SHA, "source requests differ")
    rows = base.read_jsonl(INPUTS)
    base.require(len(rows) == 10166
                 and len({r["id"] for r in rows}) == len(rows)
                 and all(set(r) == {"id", "input"}
                         and r["id"] == hashlib.sha256(r["input"].encode()).hexdigest()
                         for r in rows), "source-only request contract differs")
    return rows


def _configure_grouped():
    from scripts import train_observed_phonetic_pair_grouped as grouped
    if base.RECIPE.get("inference_operation") != "balanced-contiguous-groups-v1":
        grouped.configure()
    return grouped


def _artifact_hashes(directory: Path) -> dict[str, str]:
    names = ("report.json", "completion.json", "phonetic_transducer.pt",
             "phonetic_transducer.json", "updates.jsonl",
             "independent-verification.json")
    return {base.path_key(Path(directory) / name):
            base.sha256_file(Path(directory) / name) for name in names}


def build_authorization(arm: str, prepared_dir: Path, training: Path,
                        other_training: Path) -> dict:
    """Build the exact pure authorization consumed by ``capture``."""
    base.require(arm in base.ARMS, "unknown arm")
    grouped = _configure_grouped()
    prepared, _, prepared_pins = base.load_prepared(prepared_dir)
    prepared_path = Path(prepared_dir) / "prepared.json"
    prepared_sha = base.sha256_file(prepared_path)
    current = authenticate_training(training, arm, prepared, prepared_sha)
    other_arm = "augmented" if arm == "control" else "control"
    other = authenticate_training(other_training, other_arm, prepared, prepared_sha)
    base.require(current["initial"] == other["initial"], "paired initialization differs")
    rows = load_inputs()
    groups = grouped.balanced_groups(rows)
    training_artifacts = {}
    for name, directory, report in ((arm, Path(training), current),
                                    (other_arm, Path(other_training), other)):
        training_artifacts[name] = {
            "directory": str(directory.resolve()),
            "report_sha256": base.sha256_file(directory / "report.json"),
            "completion_sha256": base.sha256_file(directory / "completion.json"),
            "checkpoint_sha256": report["final"]["checkpoint_sha256"],
            "metadata_sha256": base.sha256_file(directory / "phonetic_transducer.json"),
            "updates_sha256": report["updates_sha256"],
            "proof_sha256": base.sha256_file(directory / "independent-verification.json"),
        }
    pins = dict(prepared_pins)
    for path in (Path(__file__), TRAINING_VERIFIER, INPUTS, prepared_path,
                 Path(current["initial"]["path"])):
        pins[base.path_key(path)] = base.sha256_file(path)
    pins.update(_artifact_hashes(training))
    pins.update(_artifact_hashes(other_training))
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "authorized": True,
        "model_loaded": False,
        "model_inference": False,
        "arm": arm,
        "prepared_sha256": prepared_sha,
        "prepared_source_hashes": prepared_pins,
        "evaluation_inputs": prepared["evaluation_inputs"],
        "input_count": len(rows),
        "inputs_sha256": INPUTS_SHA,
        "input_bytes": INPUTS.stat().st_size,
        "group_count": len(groups),
        "group_sizes": [len(group) for group in groups],
        "model_config": prepared["model_config"],
        "recipe": prepared["recipe"],
        "parameter_finiteness_checks": PARAMETER_FINITE_CHECKS,
        "paired_initial_checkpoint_sha256": current["initial"]["checkpoint_sha256"],
        "paired_initial_tensor_sha256": current["initial"]["tensor_sha256"],
        "training_artifacts": training_artifacts,
        "limits": base.LIMITS,
        "production_changed": False,
        "source_hashes": pins,
    }


def write_authorization(arm: str, prepared_dir: Path, training: Path,
                        other_training: Path, output: Path) -> dict:
    output = Path(output)
    base.require(not output.exists(), "authorization output already exists")
    authorization = build_authorization(arm, prepared_dir, training, other_training)
    output.parent.mkdir(parents=True, exist_ok=True)
    base.write_json(output, authorization)
    return authorization


def validate_authorization(path: Path, arm: str, prepared_dir: Path,
                           training: Path, other_training: Path) -> dict:
    supplied = base.read_json(path)
    expected = build_authorization(arm, prepared_dir, training, other_training)
    base.require(supplied == expected, "capture authorization differs")
    base.verify_pins(expected["source_hashes"])
    return expected


def run(arm: str, prepared_dir: Path, training: Path, other_training: Path,
        authorization: Path, output: Path) -> dict:
    capture_started = time.monotonic()
    base.require(not output.exists(), "capture output already exists")
    authorized = validate_authorization(
        authorization, arm, prepared_dir, training, other_training)
    grouped = _configure_grouped()
    prepared, _, pins = base.load_prepared(prepared_dir)
    prepared_sha = base.sha256_file(prepared_dir / "prepared.json")
    current = authenticate_training(training, arm, prepared, prepared_sha)
    other = authenticate_training(
        other_training, "augmented" if arm == "control" else "control",
        prepared, prepared_sha)
    base.require(current["initial"] == other["initial"], "paired initialization differs")
    rows = load_inputs()
    groups = grouped.balanced_groups(rows)
    training_completion = base.read_json(training / "completion.json")
    training_elapsed = training_completion["resource_after_report"]["elapsed_s"]
    base.require(_finite_nonnegative(training_elapsed), "training elapsed time malformed")
    started = capture_started - training_elapsed
    baseline = original_swap_baseline(current["final_resource"])
    pins = {**pins, **authorized["source_hashes"]}
    pins[base.path_key(authorization)] = base.sha256_file(authorization)
    output.mkdir(parents=True)
    modules = base.torch_modules()
    base.configure_torch(modules["torch"])
    group_rows = []
    with modules["AtomicResourceWatchdog"](
            started, baseline, output / "resources.jsonl", limits=base.LIMITS) as watchdog:
        with watchdog.operation("load_final_model", update=True):
            model = modules["PhoneticTransducer"].load(training / "phonetic_transducer.pt")
        initial_tensor = base.tensor_hash(model.network.state_dict())
        base.require(initial_tensor == current["final"]["tensor_sha256"], "loaded final tensors differ")
        for index, group in enumerate(groups):
            observation = grouped.capture_group(
                model, group, watchdog, output / "predictions.jsonl", f"capture_group_{index}")
            group_rows.append(observation)
            base.append_jsonl(output / "groups.jsonl", observation)
        base.require(base.tensor_hash(model.network.state_dict()) == initial_tensor, "capture changed weights")
        base.verify_pins(pins)
        final_resource = watchdog.sample_and_append("final_before_report")
    predictions = base.read_jsonl(output / "predictions.jsonl")
    base.require([{k: r[k] for k in ("id", "input")} for r in predictions] == rows,
                 "captured sources differ")
    report = {
        "schema": "document-phonetic-capture-v1", "completed": True, "arm": arm,
        "input_manifest_sha256": base.FIXED_HASHES[base.EVALUATION_MANIFEST],
        "inputs_sha256": INPUTS_SHA,
        "predictions_sha256": base.sha256_file(output / "predictions.jsonl"),
        "checkpoint_sha256": current["final"]["checkpoint_sha256"],
        "training_report_sha256": base.sha256_file(training / "report.json"),
        "training_proof_sha256": base.sha256_file(training / "independent-verification.json"),
        "authorization_sha256": base.sha256_file(authorization),
        "paired_initial_checkpoint_sha256": current["initial"]["checkpoint_sha256"],
        "prediction_count": len(predictions), "beam_size": 8,
        "group_count": len(groups), "groups_sha256": base.sha256_file(output / "groups.jsonl"),
        "training_elapsed_s": training_elapsed,
        "capture_elapsed_s_before_report": time.monotonic() - capture_started,
        "reference_targets_read": False, "source_hashes": pins,
        "final_resource": final_resource, "limits": base.LIMITS,
        "quality_evaluation": False, "production_changed": False,
    }
    base.finalize_report(output, report, modules, started, baseline)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)
    for command in ("authorize", "capture"):
        child = commands.add_parser(command)
        child.add_argument("--arm", choices=base.ARMS, required=True)
        for key in ("prepared", "training", "other-training", "output"):
            child.add_argument(f"--{key}", type=Path, required=True)
        if command == "capture":
            child.add_argument("--authorization", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "authorize":
        result = write_authorization(
            args.arm, args.prepared, args.training, args.other_training, args.output)
        print(json.dumps({"authorized": result["authorized"], "arm": result["arm"]}))
        return
    existed = args.output.exists()
    try:
        result = run(args.arm, args.prepared, args.training, args.other_training,
                     args.authorization, args.output)
    except BaseException as error:
        if not existed and args.output.is_dir():
            base.write_json(args.output / "failure-report.json", {
                "completed": False, "error": f"{type(error).__name__}: {error}"})
        raise
    print(json.dumps({k: result[k] for k in ("completed", "arm", "prediction_count")}))


if __name__ == "__main__":
    main()
