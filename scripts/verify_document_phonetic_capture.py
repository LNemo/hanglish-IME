#!/usr/bin/env python3
"""Pure independent verifier for a closed document phonetic capture."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_document_phonetic_pair as evaluation
from scripts import verify_observed_phonetic_training as training_verifier


INPUTS = ROOT / ".omx/artifacts/document-phonetic-inputs-v1/inputs.jsonl"
PREPARED = ROOT / ".omx/artifacts/observed-phonetic-pair-prepared-v3/prepared.json"
AUTHORIZATION_SCHEMA = "document-phonetic-capture-authorization-v1"
PROOF_SCHEMA = "document-phonetic-capture-independent-v1"
MAX_GROUP_SIZE = 32


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    values = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                require(isinstance(value, dict), f"invalid row {number}: {path}")
                values.append(value)
    return values


def path_key(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def balanced_groups(values: list[dict], maximum: int = MAX_GROUP_SIZE) -> list[list[dict]]:
    require(type(maximum) is int and maximum > 0, "group maximum invalid")
    if not values:
        return []
    count = math.ceil(len(values) / maximum)
    small, extra = divmod(len(values), count)
    sizes = [small + (index < extra) for index in range(count)]
    groups, offset = [], 0
    for size in sizes:
        groups.append(values[offset:offset + size])
        offset += size
    require(offset == len(values) and max(sizes) <= maximum
            and max(sizes) - min(sizes) <= 1, "balanced groups differ")
    return groups


def beam_work_units(length: int) -> int:
    require(type(length) is int and length > 0, "beam input length invalid")
    return length + 2 + min(3 * length + 4, 73)


def verify_groups(rows: list[dict], groups: list[list[dict]]) -> dict:
    require(len(rows) == len(groups), "capture group count differs")
    total_work = 0
    for index, (row, group) in enumerate(zip(rows, groups)):
        sources = [value["input"] for value in group]
        work = sum(beam_work_units(len(source)) for source in sources)
        require(isinstance(row, dict)
                and set(row) == {"inputs", "work_units", "seconds", "count"}
                and row["inputs"] == sources and row["count"] == len(group)
                and row["work_units"] == work
                and type(row["seconds"]) in (int, float)
                and math.isfinite(row["seconds"])
                and 0 <= row["seconds"] <= evaluation.LIMITS["update_seconds"],
                f"capture group differs: {index}")
        total_work += work
    return {"groups": len(groups), "inputs": sum(map(len, groups)),
            "work_units": total_work,
            "size_histogram": {str(size): sum(len(group) == size for group in groups)
                               for size in sorted({len(group) for group in groups})}}


def verify_authorization(authorization_path: Path, arm: str, report: dict,
                         requests: list[dict], groups: list[list[dict]],
                         training_dir: Path) -> dict:
    authorization = read_json(authorization_path)
    prepared = read_json(PREPARED)
    pins = authorization.get("source_hashes")
    require(authorization.get("schema") == AUTHORIZATION_SCHEMA
            and authorization.get("authorized") is True
            and authorization.get("model_loaded") is False
            and authorization.get("model_inference") is False
            and authorization.get("arm") == arm
            and authorization.get("input_count") == len(requests)
            and authorization.get("evaluation_inputs") == len(requests)
            and authorization.get("inputs_sha256") == evaluation.INPUTS_SHA256
            and authorization.get("input_bytes") == INPUTS.stat().st_size
            and authorization.get("group_count") == len(groups)
            and authorization.get("group_sizes") == list(map(len, groups))
            and authorization.get("limits") == evaluation.LIMITS
            and authorization.get("production_changed") is False
            and authorization.get("prepared_sha256") == sha256_file(PREPARED)
            and authorization.get("prepared_source_hashes") == prepared.get("source_hashes")
            and authorization.get("recipe") == prepared.get("recipe")
            and authorization.get("model_config") == prepared.get("model_config")
            and isinstance(pins, dict) and pins,
            "capture authorization contract differs")
    for relative, wanted in pins.items():
        path = ROOT / relative
        require(path.is_file() and sha256_file(path) == wanted,
                f"authorization pin differs: {relative}")
    expected_report_pins = {**pins, path_key(authorization_path): sha256_file(authorization_path)}
    require(report.get("authorization_sha256") == sha256_file(authorization_path)
            and report.get("source_hashes") == expected_report_pins,
            "capture report does not bind authorization")
    artifacts = authorization.get("training_artifacts", {}).get(arm)
    require(isinstance(artifacts, dict)
            and Path(artifacts.get("directory", "")).resolve() == training_dir.resolve(),
            "authorized training directory differs")
    expected = {
        "report_sha256": sha256_file(training_dir / "report.json"),
        "completion_sha256": sha256_file(training_dir / "completion.json"),
        "checkpoint_sha256": sha256_file(training_dir / "phonetic_transducer.pt"),
        "metadata_sha256": sha256_file(training_dir / "phonetic_transducer.json"),
        "updates_sha256": sha256_file(training_dir / "updates.jsonl"),
        "proof_sha256": sha256_file(training_dir / "independent-verification.json"),
    }
    require(all(artifacts.get(key) == value for key, value in expected.items()),
            "authorized training artifact differs")
    require(report.get("checkpoint_sha256") == expected["checkpoint_sha256"]
            and report.get("training_report_sha256") == expected["report_sha256"]
            and report.get("training_proof_sha256") == expected["proof_sha256"]
            and report.get("paired_initial_checkpoint_sha256")
                == authorization.get("paired_initial_checkpoint_sha256")
            and report.get("quality_evaluation") is False
            and report.get("production_changed") is False,
            "capture training hashes differ")
    proof = read_json(training_dir / "independent-verification.json")
    training_report = read_json(training_dir / "report.json")
    require(proof.get("schema") == "observed-phonetic-training-independent-v1"
            and proof.get("verified") is True and proof.get("arm") == arm
            and proof.get("checkpoint_sha256") == expected["checkpoint_sha256"]
            and training_report.get("initial", {}).get("checkpoint_sha256")
                == authorization.get("paired_initial_checkpoint_sha256")
            and training_report.get("initial", {}).get("tensor_sha256")
                == authorization.get("paired_initial_tensor_sha256"),
            "training proof contract differs")
    return authorization


def require_no_failures(directory: Path) -> None:
    for name in ("failure.json", "failure-report.json", "resource-watchdog-failure.json"):
        require(not (Path(directory) / name).exists(), f"capture failure artifact exists: {name}")


def verify_combined_timing(training_elapsed: object, capture_elapsed: object,
                           main_final_elapsed: object,
                           finalization_final_elapsed: object) -> None:
    values = (training_elapsed, capture_elapsed, main_final_elapsed,
              finalization_final_elapsed)
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                for value in values), "combined capture timing differs")
    require(main_final_elapsed >= training_elapsed
            and capture_elapsed >= main_final_elapsed - training_elapsed
            and finalization_final_elapsed >= training_elapsed + capture_elapsed
            and finalization_final_elapsed >= main_final_elapsed,
            "combined capture timing differs")


def verify(capture_dir: Path, authorization_path: Path, training_dir: Path) -> dict:
    capture_dir, authorization_path, training_dir = map(
        Path, (capture_dir, authorization_path, training_dir))
    proof_path = capture_dir / "independent-verification.json"
    require(capture_dir.is_dir() and not proof_path.exists(), "capture/proof state invalid")
    require_no_failures(capture_dir)
    requests, _ = evaluation.load_input_requests()
    groups = balanced_groups(requests)
    report = read_json(capture_dir / "report.json")
    arm = report.get("arm")
    require(arm in {"control", "augmented"}, "capture arm differs")
    authenticated = evaluation.authenticate_capture(capture_dir, arm, requests)
    authorization = verify_authorization(
        authorization_path, arm, report, requests, groups, training_dir)
    group_path = capture_dir / "groups.jsonl"
    group_rows = read_jsonl(group_path)
    group_evidence = verify_groups(group_rows, groups)
    require(report.get("group_count") == len(groups)
            and report.get("groups_sha256") == sha256_file(group_path),
            "capture group hash differs")

    expected_operations = ["load_final_model"] + [
        f"capture_group_{index}" for index in range(len(groups))]
    main_resources = training_verifier.verify_resource_trace(
        read_jsonl(capture_dir / "resources.jsonl"), expected_operations,
        "final_before_report")
    finalization_resources = training_verifier.verify_resource_trace(
        read_jsonl(capture_dir / "resources-finalize.jsonl"),
        ["write_report", "write_completion"], "after_report",
        final_after_operations=1, terminal_stages=("after_completion",))
    training_report = read_json(training_dir / "report.json")
    training_completion = read_json(training_dir / "completion.json")
    training_baseline = (training_report["final_resource"]["swap_used_bytes"]
                         - training_report["final_resource"]["swap_growth_bytes"])
    require(training_baseline >= 0
            and main_resources["baseline_swap_bytes"] == training_baseline
            and finalization_resources["baseline_swap_bytes"] == training_baseline,
            "capture/training swap baseline differs")
    require(main_resources["final"] == report.get("final_resource")
            and finalization_resources["final"]
                == read_json(capture_dir / "completion.json").get("resource_after_report"),
            "capture terminal resources differ")
    training_elapsed = training_completion.get("resource_after_report", {}).get("elapsed_s")
    capture_elapsed = report.get("capture_elapsed_s_before_report")
    require(report.get("training_elapsed_s") == training_elapsed,
            "reported training elapsed differs")
    verify_combined_timing(
        training_elapsed, capture_elapsed, main_resources["final"]["elapsed_s"],
        finalization_resources["final"]["elapsed_s"])

    proof = {
        "schema": PROOF_SCHEMA, "verified": True, "arm": arm,
        "report_sha256": authenticated["report_sha256"],
        "completion_sha256": authenticated["completion_sha256"],
        "predictions_sha256": authenticated["predictions_sha256"],
        "groups_sha256": sha256_file(group_path),
        "authorization_sha256": sha256_file(authorization_path),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "training_report_sha256": report["training_report_sha256"],
        "training_proof_sha256": report["training_proof_sha256"],
        "verifier_sha256": sha256_file(Path(__file__)),
        "counts": {"requests": len(requests), "predictions": len(authenticated["predictions"]),
                   "groups": len(groups), "maximum_group_size": MAX_GROUP_SIZE,
                   "beam_size": report["beam_size"]},
        "groups": group_evidence,
        "resources": {"main": main_resources, "finalization": finalization_resources,
                      "training_swap_baseline_bytes": training_baseline},
        "limits": evaluation.LIMITS,
        "contracts_verified": {
            "source_only_input_identity_order_and_hash": True,
            "authorization_and_all_source_pins": True,
            "training_checkpoint_and_independent_proof": True,
            "balanced_group_partition_and_work_counts": True,
            "beam8_unique_forward_valid_finite_descending_outputs": True,
            "atomic_resources_and_combined_time": True,
            "failure_artifacts_absent": True,
            "reference_targets_read": False,
            "performance_metrics_computed": False,
            "neural_model_loaded_or_run_by_verifier": False,
        },
        "limitations": [
            "The verifier authenticates recorded capture output and resource evidence without rerunning the model.",
            "No reserve references or quality metrics are read or computed.",
        ],
    }
    require(authorization["arm"] == proof["arm"], "proof authorization arm differs")
    proof_path.write_bytes(canonical(proof) + b"\n")
    return proof


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--capture", type=Path, required=True)
    value.add_argument("--authorization", type=Path, required=True)
    value.add_argument("--training", type=Path, required=True)
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(verify(args.capture, args.authorization, args.training), sort_keys=True))
