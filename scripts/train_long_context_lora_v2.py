"""Checkpointed v2 trainer for the reviewed long-context LoRA experiment.

The source-free checkpoint proof is validated before this module reads FIT rows
or imports MLX.  A failed segment is durable and is never resumed inline.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping
import uuid


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import probe_long_context_lora_resources as resource_v1
from scripts import train_long_context_lora as v1
from scripts.long_context_lora_checkpoint_v2 import (
    ADAM_CONFIG, ADAPTER_CONFIG, CheckpointStore, FINAL_UPDATE,
    cumulative_active_seconds, make_run_identity,
    restore_model_optimizer, validate_adapter_tensors,
)
from scripts.long_context_lora_model import (
    add_materialized_gradients, apply_materialized_update,
    average_and_clip_gradients, configure_fresh_lora, create_optimizer,
    materialize_microbatch, mlx_log_mass_loss, require_matching_gradient_tree,
    require_trainable_contract, score_choices,
)
from scripts.long_context_lora_resources import ResourceObservationError, observe_resources


OUTPUT = ROOT / ".omx/artifacts/long-context-lora-train-v2"
RESOURCE_REPORT = ROOT / ".omx/artifacts/long-context-lora-resource-checkpoint-v2/report.json"
RESOURCE_VERIFIER = ROOT / ".omx/artifacts/long-context-lora-work-v2/verify_checkpoint_probe.py"
RESOURCE_RUNNER = ROOT / "scripts/probe_long_context_lora_checkpoint_v2.py"
CHECKPOINT_HELPER = ROOT / "scripts/long_context_lora_checkpoint_v2.py"
CHECKPOINT_HELPER_SHA256 = "2f5928b6d02d03cb22e9142e7bdfdcd212dcc090e5f7a1230f8f94c437806810"
PLAN = ROOT / ".omx/plans/long-context-lora-checkpoint-v2.md"
PLAN_SHA256 = "cf50d05dff1df2c384e7b4a0b5ab7ae968e7daccbbcf3a71464ee74bb9e88a5f"
TEST_SPEC = ROOT / ".omx/plans/long-context-lora-checkpoint-v2-test-spec.md"
TEST_SPEC_SHA256 = "d340bf79d1855b449153e89c97978dcec5356e203c0ec28d317aaa2c852d1d12"
HANDOFF = ROOT / ".omx/plans/long-context-lora-checkpoint-v2-handoff.json"
HANDOFF_SHA256 = "122616dfea1506cf4dfa9a14cbf3f511bb2d8aad7150647b6ad8d7bd24202b5a"
LIMITS = {**resource_v1.LIMITS, "whole_seconds": 21600.0}
EXPECTED_ROWS = 2771
EXPECTED_UPDATES = 693
EXPECTED_DIAGNOSTIC_ROWS = 640
CLOSE_RESERVE_SECONDS = 1.0
PROBE_INDICES = (0, 923, 1846, 2770)


class TrainingV2Error(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise TrainingV2Error(f"required regular JSON file is unavailable: {path}")
    seen: list[str] = []

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise TrainingV2Error(f"duplicate JSON key: {key}")
            value[key] = item
            seen.append(key)
        return value

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=pairs,
                           parse_constant=lambda item: (_ for _ in ()).throw(
                               TrainingV2Error(f"nonfinite JSON value: {item}")))
    except TrainingV2Error:
        raise
    except Exception as error:
        raise TrainingV2Error(f"invalid JSON file: {path}") from error
    if not isinstance(value, dict):
        raise TrainingV2Error(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write((json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     indent=2, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    try:
        if exclusive:
            os.link(temporary, path)
            temporary.unlink()
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_resource_proof(resource_proof: Path, explicit_sha256: str,
                            *, report_path: Path = RESOURCE_REPORT,
                            verifier_path: Path = RESOURCE_VERIFIER,
                            runner_path: Path = RESOURCE_RUNNER,
                            helper_path: Path = CHECKPOINT_HELPER) -> dict[str, Any]:
    """Validate the independent source-free proof without importing MLX or FIT data."""
    if not isinstance(explicit_sha256, str) or len(explicit_sha256) != 64:
        raise TrainingV2Error("an explicit resource-proof SHA256 is required")
    if sha256_file(resource_proof) != explicit_sha256:
        raise TrainingV2Error("resource-proof file hash differs from the explicit pin")
    proof = _read_json(resource_proof)
    required = {
        "passed", "report_sha256", "verifier_sha256", "runner_sha256",
        "helper_sha256", "actual_fit_training_run", "source_files_checked",
        "main_updates", "restart_branches", "restart_parity_exact",
        "original_global_swap_baseline_preserved", "runtime_projection",
        "retained_payload_pairs_byte_verified",
        "historical_pruned_payload_bytes_independently_verified",
    }
    if set(proof) != required:
        raise TrainingV2Error("resource-proof schema differs")
    current = {
        "report_sha256": sha256_file(report_path),
        "verifier_sha256": sha256_file(verifier_path),
        "runner_sha256": sha256_file(runner_path),
        "helper_sha256": sha256_file(helper_path),
    }
    if any(proof[key] != digest for key, digest in current.items()):
        raise TrainingV2Error("resource proof does not bind current report/verifier/runner/helper")
    projection = proof["runtime_projection"]
    if (proof["passed"] is not True or proof["actual_fit_training_run"] is not False
            or proof["main_updates"] != 20 or proof["restart_branches"] != 3
            or proof["restart_parity_exact"] is not True
            or proof["original_global_swap_baseline_preserved"] is not True
            or proof["retained_payload_pairs_byte_verified"] != 8
            or proof["historical_pruned_payload_bytes_independently_verified"] is not False
            or type(proof["source_files_checked"]) is not int
            or proof["source_files_checked"] < 100
            or not isinstance(projection, dict)
            or projection.get("within_21600_seconds") is not True
            or type(projection.get("estimated_training_seconds")) not in (int, float)
            or not math.isfinite(projection["estimated_training_seconds"])
            or projection["estimated_training_seconds"] > LIMITS["whole_seconds"]):
        raise TrainingV2Error("source-free checkpoint/resource gate did not pass")
    report = _read_json(report_path)
    if (report.get("schema") != "long-context-lora-resource-checkpoint-v2"
            or report.get("completed") is not True
            or report.get("resource_feasible") is not True
            or report.get("terminal_status") != "success"
            or report.get("real_fit_rows_read") is not False
            or report.get("real_training_run") is not False
            or report.get("runtime_estimate") != projection):
        raise TrainingV2Error("source-free checkpoint report differs from its passing proof")
    if current["helper_sha256"] != CHECKPOINT_HELPER_SHA256:
        raise TrainingV2Error("reviewed checkpoint helper source has changed")
    return {"proof": proof, "proof_sha256": explicit_sha256,
            "report": report, "paths": {"proof": str(resource_proof),
                                         "report": str(report_path),
                                         "verifier": str(verifier_path),
                                         "runner": str(runner_path),
                                         "helper": str(helper_path)}, **current}


def capture_source_inventory(bundle: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    report = bundle["report"]
    hashes = copy.deepcopy(report.get("source_hashes"))
    versions = copy.deepcopy(report.get("versions"))
    if (not isinstance(hashes, dict) or len(hashes) < 100
            or hashes != report.get("source_hashes_after")
            or versions != resource_v1.PACKAGE_VERSIONS
            or versions != report.get("versions_after")):
        raise TrainingV2Error("source-free report provenance differs")
    fixed = {
        CHECKPOINT_HELPER: CHECKPOINT_HELPER_SHA256,
        RESOURCE_VERIFIER: bundle["verifier_sha256"],
        RESOURCE_RUNNER: bundle["runner_sha256"],
        Path(bundle["paths"]["proof"]): bundle["proof_sha256"],
        RESOURCE_REPORT: bundle["report_sha256"],
        PLAN: PLAN_SHA256, TEST_SPEC: TEST_SPEC_SHA256, HANDOFF: HANDOFF_SHA256,
        Path(__file__): sha256_file(Path(__file__)),
        ROOT / "scripts/train_long_context_lora.py": sha256_file(
            ROOT / "scripts/train_long_context_lora.py"),
    }
    for path, expected in fixed.items():
        actual = sha256_file(path)
        if actual != expected:
            raise TrainingV2Error(f"fixed v2 training source differs: {path}")
        hashes[resource_v1.relative_path(path)] = actual
    for name, expected in hashes.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_symlink() or sha256_file(path) != expected:
            raise TrainingV2Error(f"training source inventory changed: {name}")
    return dict(sorted(hashes.items())), dict(sorted(versions.items()))


def validate_schedule_binding(data: Mapping[str, Any], run: Mapping[str, Any],
                              checkpoint_snapshot: Mapping[str, Any] | None = None) -> None:
    batches = data.get("batches")
    records = data.get("records")
    manifest = data.get("manifest")
    if (not isinstance(batches, list) or len(batches) != EXPECTED_UPDATES
            or not isinstance(records, list) or len(records) != EXPECTED_ROWS
            or not isinstance(manifest, dict)
            or manifest.get("file_hashes", {}).get("schedule.jsonl")
            != run["schedule"]["sha256"]
            or run["schedule"] != {
                "sha256": run["schedule"]["sha256"], "microbatches": EXPECTED_ROWS,
                "updates": EXPECTED_UPDATES, "accumulation": 4,
                "final_update_microbatches": 3}):
        raise TrainingV2Error("reviewed FIT schedule identity differs")
    cursor = 0
    for update, batch in enumerate(batches, 1):
        expected_count = 3 if update == EXPECTED_UPDATES else 4
        if len(batch) != expected_count:
            raise TrainingV2Error("reviewed FIT batch or final tail differs")
        for entry in batch:
            record = records[cursor]
            cursor += 1
            if (entry != {"role": entry.get("role"), "id": entry.get("id"),
                          "microbatch": cursor, "update": update}
                    or (record.get("role"), record.get("id"))
                    != (entry["role"], entry["id"])):
                raise TrainingV2Error("FIT row order differs from the immutable schedule")
    if cursor != EXPECTED_ROWS:
        raise TrainingV2Error("FIT schedule cursor is incomplete")
    for item in (checkpoint_snapshot or {}).get("manifests", []):
        payload = item.get("payload", {})
        update = payload.get("update")
        if type(update) is not int or not 1 <= update <= EXPECTED_UPDATES:
            raise TrainingV2Error("checkpoint manifest update is outside the FIT schedule")
        expected = batches[update - 1]
        actual = payload.get("batch_records")
        if (not isinstance(actual, list) or len(actual) != len(expected)
                or any((row.get("role"), row.get("id"), row.get("microbatch"),
                        row.get("update"))
                       != (entry["role"], entry["id"], entry["microbatch"], entry["update"])
                       for row, entry in zip(actual, expected))):
            raise TrainingV2Error("checkpoint batch identities differ from pinned FIT schedule")
        next_index = min(update * 4, EXPECTED_ROWS)
        next_expected = None if next_index == EXPECTED_ROWS else {
            "role": records[next_index]["role"], "id": records[next_index]["id"]}
        if payload.get("next_expected") != next_expected:
            raise TrainingV2Error("checkpoint next identity differs from pinned FIT schedule")


def load_and_validate_schedule_only(data_manifest: Mapping[str, Any],
                                    run: Mapping[str, Any],
                                    checkpoint_snapshot: Mapping[str, Any],
                                    *, data_dir: Path = v1.DATA_DIR) -> dict[str, Any]:
    """Bind checkpoints to the source-only schedule before reading FIT targets."""
    schedule_path = data_dir / "schedule.jsonl"
    expected = data_manifest.get("file_hashes", {}).get("schedule.jsonl")
    if not isinstance(expected, str) or sha256_file(schedule_path) != expected:
        raise TrainingV2Error("reviewed source-only schedule file differs")
    schedule = v1.read_jsonl(schedule_path)
    batches = v1.validate_schedule(schedule)
    data = {"batches": batches,
            "records": [{"role": row["role"], "id": row["id"]} for row in schedule],
            "manifest": {"file_hashes": {"schedule.jsonl": expected}}}
    validate_schedule_binding(data, run, checkpoint_snapshot)
    return data


def _prior_active_seconds(events: list[Mapping[str, Any]], current_segment: str) -> float:
    return cumulative_active_seconds(events, 0.0, current_segment)


def validate_prior_close_proofs(output: Path, events: list[Mapping[str, Any]],
                                current_segment: str) -> None:
    terminals = [event for event in events if event.get("kind") == "segment_terminal"
                 and event.get("segment_id") != current_segment
                 and str(event.get("segment_id", "")).startswith("segment-")]
    for event in terminals:
        path = output / "segments" / event["segment_id"] / "report.json"
        report = _read_json(path)
        close = report.get("close_proof", {})
        expected_report_status = ("success" if event.get("status") == "complete"
                                  else event.get("status"))
        if (report.get("schema") != "long-context-lora-train-v2-segment"
                or report.get("segment_id") != event["segment_id"]
                or report.get("terminal_status") != expected_report_status
                or report.get("completed") != (event.get("status") == "complete")
                or close.get("reserve_seconds") != CLOSE_RESERVE_SECONDS
                or type(close.get("actual_close_seconds")) not in (int, float)
                or not 0 <= close["actual_close_seconds"] <= CLOSE_RESERVE_SECONDS
                or close.get("completed_within_reserve") is not True
                or close.get("charged_terminal_seconds")
                != event.get("terminal_segment_seconds")):
            raise TrainingV2Error("prior segment terminal close proof differs")


def _base_paths(output: Path) -> dict[str, Path]:
    base = output / "base-diagnostic"
    return {"root": base, "parent": base / "parent_predictions.jsonl",
            "base": base / "base_predictions.jsonl", "manifest": base / "manifest.json"}


def _load_jsonl(path: Path) -> list[dict]:
    if path.is_symlink() or not path.is_file():
        raise TrainingV2Error(f"prediction artifact is unavailable: {path}")
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def _score_diagnostic(mx, model, diagnostic_records, digit_ids, checkpoint, label):
    from scripts.long_context_lora_evaluation import apply_logits

    predictions = []
    model.eval()
    for index, record in enumerate(diagnostic_records, 1):
        logits = None
        if record.get("eligible") is True:
            count = len(record["choices"])
            values = score_choices(mx, model, record["tokens"], digit_ids[:count])
            mx.eval(values)
            checkpoint(f"{label}_diagnostic_{index}")
            logits = values.tolist()
            if len(logits) != count or not all(math.isfinite(float(value)) for value in logits):
                raise TrainingV2Error("diagnostic logits are malformed")
        predictions.append(apply_logits(record, logits))
    return predictions


def _load_model_and_diagnostic(mx, data):
    from mlx_lm.utils import load
    from scripts.long_context_lora_evaluation import prepare_diagnostic

    model, tokenizer = load(str(resource_v1.MODEL_DIR),
                            tokenizer_config={"trust_remote_code": False})
    if not bool(model.args.tie_word_embeddings):
        raise TrainingV2Error("pinned base unexpectedly has an untied output head")
    _, digit_ids = resource_v1.exact_prompts(tokenizer)
    for count, expected in data["manifest"]["digit_continuation_ids_by_candidate_count"].items():
        if digit_ids[:int(count)] != expected:
            raise TrainingV2Error("prepared digit continuation IDs differ")
    diagnostic = prepare_diagnostic(data["diagnostic_items"], tokenizer)
    if len(diagnostic) != EXPECTED_DIAGNOSTIC_ROWS or any(
            set(record) & {"expected", "category", "outcome"} for record in diagnostic):
        raise TrainingV2Error("diagnostic preparation is not source-only and complete")
    return model, tokenizer, digit_ids, diagnostic


def _load_base_diagnostic(output):
    paths = _base_paths(output)
    manifest = _read_json(paths["manifest"])
    if (manifest.get("schema") != "long-context-lora-base-diagnostic-v2"
            or manifest.get("rows") != EXPECTED_DIAGNOSTIC_ROWS
            or manifest.get("v1_prediction_cache_read") is not False
            or sha256_file(paths["parent"]) != manifest.get("parent_sha256")
            or sha256_file(paths["base"]) != manifest.get("base_sha256")):
        raise TrainingV2Error("fresh-start base diagnostic provenance differs")
    parent, base = _load_jsonl(paths["parent"]), _load_jsonl(paths["base"])
    if len(parent) != EXPECTED_DIAGNOSTIC_ROWS or len(base) != EXPECTED_DIAGNOSTIC_ROWS:
        raise TrainingV2Error("base diagnostic row count differs")
    return parent, base, manifest


def _export_final_adapter(checkpoint, attempt: Path) -> tuple[Path, dict[str, str]]:
    adapter_dir = attempt / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=False)
    source_adapter = checkpoint.path / "adapter.safetensors"
    source_config = checkpoint.path / "adapter_config.json"
    adapter = adapter_dir / "adapters.safetensors"
    config = adapter_dir / "adapter_config.json"
    shutil.copyfile(source_adapter, adapter)
    shutil.copyfile(source_config, config)
    for path in (adapter, config):
        _fsync_file(path)
    if sha256_file(adapter) != checkpoint.manifest["files"]["adapter.safetensors"]["sha256"]:
        raise TrainingV2Error("final adapter differs from checkpoint 693")
    if sha256_file(config) != checkpoint.manifest["files"]["adapter_config.json"]["sha256"]:
        raise TrainingV2Error("final adapter configuration differs from checkpoint 693")
    _fsync_directory(adapter_dir)
    _fsync_directory(attempt)
    return adapter_dir, {"adapter_sha256": sha256_file(adapter),
                         "config_sha256": sha256_file(config)}


def _next_finalization_attempt(output: Path) -> Path:
    existing = sorted(output.glob("finalization-attempt-*"))
    path = output / f"finalization-attempt-{len(existing) + 1:04d}"
    path.mkdir(exist_ok=False)
    return path


def validate_run_request(mode: str, output: Path, segment_id: str) -> None:
    if mode not in {"fresh", "resume"}:
        raise ValueError("mode must be fresh or resume")
    if (not isinstance(segment_id, str) or not segment_id.startswith("segment-")
            or not segment_id.removeprefix("segment-").isdigit()):
        raise TrainingV2Error("segment id must be segment-qualified")
    number = int(segment_id.removeprefix("segment-"))
    if mode == "fresh":
        if segment_id != "segment-0001":
            raise TrainingV2Error("fresh v2 training must begin at segment-0001")
        if output.exists():
            raise FileExistsError(f"v2 logical-run root already exists: {output}")
    elif not output.is_dir() or output.is_symlink() or number < 2:
        raise TrainingV2Error("resume requires an existing v2 root and segment number >=2")


def run(*, mode: str, output: Path, resource_proof: Path,
        resource_proof_sha256: str, segment_id: str) -> dict[str, Any]:
    """Execute one fresh or resumed segment. There is no inline retry path."""
    validate_run_request(mode, output, segment_id)
    proof = validate_resource_proof(resource_proof, resource_proof_sha256)
    sources, versions = capture_source_inventory(proof)
    data_manifest_path = v1.DATA_DIR / "manifest.json"
    if sha256_file(data_manifest_path) != v1.DATA_MANIFEST_SHA256:
        raise TrainingV2Error("reviewed training-data manifest differs before FIT access")
    data_manifest = _read_json(data_manifest_path)
    schedule_sha = data_manifest.get("file_hashes", {}).get("schedule.jsonl")
    if not isinstance(schedule_sha, str) or len(schedule_sha) != 64:
        raise TrainingV2Error("reviewed schedule hash is unavailable")
    if mode == "fresh":
        baseline = resource_v1.system_resources()
        run_identity = make_run_identity(
            logical_run_id=f"long-context-lora-fit-v2-{uuid.uuid4().hex}", mode="fit",
            baseline_swap_used_bytes=baseline["swap_used_bytes"],
            schedule_sha256=schedule_sha, microbatches=EXPECTED_ROWS,
            updates=EXPECTED_UPDATES, final_update_microbatches=3,
            source_hashes=sources,
            config_hashes={"plan": PLAN_SHA256, "test_spec": TEST_SPEC_SHA256,
                           "handoff": HANDOFF_SHA256,
                           "data_manifest": v1.DATA_MANIFEST_SHA256,
                           "resource_proof": resource_proof_sha256},
            model_hashes={"SOURCE.json": resource_v1.PINNED_ARTIFACT_SHA256[
                              resource_v1.MODEL_SOURCE],
                          "model.safetensors": resource_v1.MODEL_WEIGHT_SHA256},
            package_versions=versions, created_at=datetime.now(timezone.utc).isoformat())
        store = CheckpointStore.create(output, run_identity)
        preflight = None
    else:
        store = CheckpointStore.open(output)
        current = resource_v1.system_resources()
        preflight = {"swap_used_bytes": current["swap_used_bytes"],
                     "system_free_percent": current["free_percent"]}
    report = {
        "schema": "long-context-lora-train-v2-segment", "mode": mode,
        "segment_id": segment_id, "completed": False, "training_gate_passed": False,
        "deployed": False, "auto_resume": False, "resource_proof": {
            "path": str(resource_proof), "sha256": resource_proof_sha256,
            "report_sha256": proof["report_sha256"],
            "verifier_sha256": proof["verifier_sha256"],
            "runner_sha256": proof["runner_sha256"],
            "helper_sha256": proof["helper_sha256"]},
        "source_hashes": sources, "versions": versions, "limits": LIMITS,
        "fresh_data_read_for_evaluation": False,
        "development_or_final_read_for_evaluation": False,
        "production_modified": False, "v1_prediction_cache_read": False,
        "resource_observations": [], "updates": [],
    }
    segment_dir = output / "segments" / segment_id
    report_path = segment_dir / "report.json"
    model = tokenizer = optimizer = value_and_grad = loss_fn = mx = None
    attempt = None
    segment_started = time.monotonic()
    terminal_written = False
    segment_output_ready = False
    with store.segment(segment_id, resume=mode == "resume",
                       preflight_metrics=preflight) as segment:
        try:
            segment_dir.mkdir(parents=True, exist_ok=False)
            segment_output_ready = True
            snapshot = store.audit_snapshot()
            run_identity = snapshot["run"]
            report.update({
                "logical_run_id": run_identity["logical_run_id"],
                "run_json_sha256": snapshot["run_json_sha256"],
                "system_baseline": run_identity["baseline"],
                "planned_updates": EXPECTED_UPDATES,
                "training_rows": EXPECTED_ROWS,
                "resume_preflight": preflight,
            })
            if run_identity["source_hashes"] != sources or run_identity["package_versions"] != versions:
                raise TrainingV2Error("resume source or package identity differs")
            validate_prior_close_proofs(output, [item["payload"] for item in snapshot["events"]],
                                        segment_id)
            prior_seconds = _prior_active_seconds(
                [item["payload"] for item in snapshot["events"]], segment_id)
            logical_started = time.monotonic() - prior_seconds
            report["prior_terminal_seconds"] = prior_seconds
            schedule_only = load_and_validate_schedule_only(
                data_manifest, run_identity, snapshot)
            report["source_only_schedule_verified_before_fit"] = True

            import mlx.core as mlx_core
            import mlx.nn as nn
            from mlx_lm.tuner.utils import load_adapters
            from mlx_lm.utils import load
            from scripts.long_context_lora_evaluation import score_predictions

            mx = mlx_core
            mx.set_memory_limit(LIMITS["mlx_peak_bytes"])
            mx.set_cache_limit(LIMITS["cache_bytes"])
            baseline = {"swap_used_bytes": run_identity["baseline"]["swap_used_bytes"]}

            def checkpoint(stage: str, update_started: float | None = None):
                try:
                    settled, observation = observe_resources(
                        stage=stage, mx=mx, started=logical_started, baseline=baseline,
                        limits=LIMITS, system_reader=resource_v1.system_resources,
                        rss_reader=resource_v1.rss_bytes)
                except ResourceObservationError as error:
                    report["resource_observations"].append(error.record)
                    report["failure_resource_observation"] = error.record
                    raise
                report["resource_observations"].append(observation)
                if update_started is not None and time.monotonic() - update_started > 120.0:
                    raise TrainingV2Error("optimizer update exceeded 120 seconds")
                return settled

            restored = None
            if mode == "resume":
                restored = store.load_eligible(mx)
                validate_schedule_binding(schedule_only, run_identity, store.audit_snapshot())
                checkpoint("eligible_checkpoint_validated_before_model")
            data = v1.load_and_validate_inputs()
            validate_schedule_binding(data, run_identity, store.audit_snapshot())
            report["fit_rows_read"] = True
            model, tokenizer, digit_ids, diagnostic = _load_model_and_diagnostic(mx, data)
            checkpoint("model_loaded")
            if mode == "fresh":
                from scripts.long_context_lora_evaluation import apply_logits
                parent_predictions = [apply_logits(record, None) for record in diagnostic]
                base_predictions = _score_diagnostic(
                    mx, model, diagnostic, digit_ids, checkpoint, "base")
                initial = configure_fresh_lora(mx, model)
                initial_hashes = {
                    name: item["sha256"] for name, item in validate_adapter_tensors(initial).items()}
                paths = _base_paths(output)
                paths["root"].mkdir(exist_ok=False)
                parent_sha = v1.write_jsonl(paths["parent"], parent_predictions)
                base_sha = v1.write_jsonl(paths["base"], base_predictions)
                _atomic_json(paths["manifest"], {
                    "schema": "long-context-lora-base-diagnostic-v2",
                    "rows": 640, "parent_sha256": parent_sha, "base_sha256": base_sha,
                    "initial_adapter_hashes": initial_hashes,
                    "v1_prediction_cache_read": False}, exclusive=True)
                optimizer = create_optimizer()
                start_update = 1
                cursor = 0
            else:
                assert restored is not None
                initial = configure_fresh_lora(mx, model)
                initial_inventory = validate_adapter_tensors(initial)
                del initial
                parent_predictions, base_predictions, base_manifest = _load_base_diagnostic(output)
                if ({name: item["sha256"] for name, item in initial_inventory.items()}
                        != base_manifest.get("initial_adapter_hashes")):
                    raise TrainingV2Error("resume fresh-LoRA initialization differs")
                if restored.manifest["update"] == FINAL_UPDATE:
                    restore_model_optimizer(mx, model, None, restored)
                    optimizer = None
                    start_update = FINAL_UPDATE + 1
                else:
                    optimizer = create_optimizer()
                    restore_model_optimizer(mx, model, optimizer, restored)
                    start_update = restored.manifest["update"] + 1
                cursor = restored.manifest["completed_microbatches"]

            if start_update <= EXPECTED_UPDATES:
                model.train()

                def loss_fn(active_model, record):
                    scores = score_choices(mx, active_model, record["tokens"],
                                            digit_ids[:len(record["choices"])])
                    return mlx_log_mass_loss(mx, scores, record["allowed_gold_positions"])

                value_and_grad = nn.value_and_grad(model, loss_fn)
                for update_index in range(start_update, EXPECTED_UPDATES + 1):
                    update_started = time.monotonic()
                    schedule_batch = data["batches"][update_index - 1]
                    accumulated, losses = None, []
                    for entry in schedule_batch:
                        record = data["records"][cursor]
                        cursor += 1
                        if (record["role"], record["id"]) != (entry["role"], entry["id"]):
                            raise TrainingV2Error("training cursor differs from pinned schedule")
                        loss, gradients = value_and_grad(model, record)
                        materialize_microbatch(mx, loss, gradients)
                        flat = require_matching_gradient_tree(model, gradients)
                        if not bool(mx.isfinite(loss).item()) or any(
                                not bool(mx.all(mx.isfinite(value)).item()) for value in flat.values()):
                            raise TrainingV2Error("training loss or gradient is nonfinite")
                        accumulated = add_materialized_gradients(mx, accumulated, gradients)
                        checkpoint(f"update_{update_index}_microbatch_{len(losses) + 1}",
                                   update_started)
                        losses.append(float(loss.item()))
                    clipped, norm = average_and_clip_gradients(mx, accumulated, len(schedule_batch))
                    checkpoint(f"update_{update_index}_averaged_clipped", update_started)
                    apply_materialized_update(mx, model, optimizer, clipped)
                    next_index = min(update_index * 4, EXPECTED_ROWS)
                    next_schedule = None if next_index == EXPECTED_ROWS else {
                        "role": data["records"][next_index]["role"],
                        "id": data["records"][next_index]["id"]}
                    result = segment.publish_checkpoint(
                        mx=mx, adapter_tensors=require_trainable_contract(model),
                        optimizer_state=optimizer.state, update=update_index,
                        completed_microbatches=cursor, next_schedule=next_schedule,
                        batch_records=schedule_batch, losses=losses,
                        adapter_config=ADAPTER_CONFIG, optimizer_config=ADAM_CONFIG,
                        recipe=run_identity["recipe"],
                        elapsed_seconds=time.monotonic() - logical_started,
                        update_elapsed_before_checkpoint=time.monotonic() - update_started,
                        resource_callback=lambda stage, u=update_index, s=update_started:
                            checkpoint(f"update_{u}_checkpoint_{stage}", s))
                    validate_schedule_binding(
                        data, run_identity,
                        {"manifests": [{"payload": result["manifest"]}]})
                    update_record = {
                        "update": update_index, "microbatches": len(schedule_batch),
                        "batch_records": schedule_batch, "microbatch_losses": losses,
                        "gradient_norm_before_clip": float(norm.item()),
                        "seconds": result["update_seconds"],
                        "checkpoint_timings": result["timings"],
                        "checkpoint_payload_bytes": result["payload_bytes"],
                        "checkpoint_manifest_sha256": result["pointer"]["manifest_sha256"],
                    }
                    report["updates"].append(update_record)
                    report["completed_updates"] = update_index
                    report["completed_microbatches"] = cursor

            latest = store.load_eligible(mx)
            if latest.manifest["update"] != FINAL_UPDATE or cursor != EXPECTED_ROWS:
                raise TrainingV2Error("finalization requires eligible checkpoint 693")
            model = tokenizer = optimizer = value_and_grad = loss_fn = None
            gc.collect()
            mx.synchronize()
            checkpoint("training_model_released")

            attempt = _next_finalization_attempt(output)
            model, tokenizer, digit_ids, diagnostic = _load_model_and_diagnostic(mx, data)
            configure_fresh_lora(mx, model)
            restore_model_optimizer(mx, model, None, latest)
            before_probes = []
            for index in PROBE_INDICES:
                record = data["records"][index]
                values = score_choices(mx, model, record["tokens"],
                                        digit_ids[:len(record["choices"])])
                mx.eval(values)
                checkpoint(f"save_probe_{index}")
                logits = values.tolist()
                v1.validate_reload_logits(logits, logits)
                before_probes.append({"index": index, "role": record["role"],
                                      "id": record["id"],
                                      "tokens_sha256": v1.canonical_token_digest(
                                          record["tokens"]),
                                      "digit_logits": logits})
            trained_predictions = _score_diagnostic(
                mx, model, diagnostic, digit_ids, checkpoint, "trained")
            model = tokenizer = None
            gc.collect()
            checkpoint("checkpoint_model_released")

            adapter_dir, adapter_hashes = _export_final_adapter(latest, attempt)
            model, tokenizer = load(str(resource_v1.MODEL_DIR),
                                    tokenizer_config={"trust_remote_code": False})
            model.freeze()
            load_adapters(model, str(adapter_dir))
            reloaded_inventory = validate_adapter_tensors(require_trainable_contract(model))
            if reloaded_inventory != latest.manifest["adapter_inventory"]:
                raise TrainingV2Error("final adapter reload inventory differs from checkpoint 693")
            model.eval()
            reload_checks = []
            for expected in before_probes:
                record = data["records"][expected["index"]]
                values = score_choices(mx, model, record["tokens"],
                                        digit_ids[:len(record["choices"])])
                mx.eval(values)
                checkpoint(f"reload_probe_{expected['index']}")
                actual = values.tolist()
                delta, close = v1.validate_reload_logits(expected["digit_logits"], actual)
                if not close or actual.index(max(actual)) != expected["digit_logits"].index(
                        max(expected["digit_logits"])):
                    raise TrainingV2Error("checkpoint693 final-adapter reload probe differs")
                reload_checks.append({**expected, "actual_logits": actual,
                                      "max_absolute_delta": delta})
            trained_sha = v1.write_jsonl(attempt / "trained_predictions.jsonl",
                                         trained_predictions)
            _fsync_file(attempt / "trained_predictions.jsonl")
            model = tokenizer = None
            gc.collect()
            checkpoint("evaluation_model_released")

            parent_metrics = score_predictions(data["diagnostic_items"], parent_predictions)
            base_metrics = score_predictions(data["diagnostic_items"], base_predictions)
            trained_metrics = score_predictions(data["diagnostic_items"], trained_predictions)
            train_gate = v1.require_train_gate(parent_metrics, trained_metrics)
            if train_gate["passed"] is not True:
                raise TrainingV2Error("TRAIN640 net-gain gate did not pass")
            after, versions_after = capture_source_inventory(proof)
            if after != sources or versions_after != versions:
                raise TrainingV2Error("training sources changed during the segment")
            report.update({
                "completed": True, "terminal_status": "success",
                "training_gate_passed": train_gate["passed"],
                "source_hashes_after": after, "versions_after": versions_after,
                "final_checkpoint": latest.manifest,
                "final_adapter": adapter_hashes,
                "reload_checks": reload_checks,
                "trained_predictions_sha256": trained_sha,
                "diagnostic_metrics": {"cached_parent": parent_metrics,
                                       "fresh_base": base_metrics,
                                       "final_adapter": trained_metrics},
                "train_gate": train_gate,
                "source_inventory_unchanged": True,
                "all_diagnostic_labels_consulted_after_model_release": True,
            })
            _atomic_json(attempt / "manifest.json", {
                "schema": "long-context-lora-finalization-attempt-v2",
                "segment_id": segment_id, "checkpoint_update": FINAL_UPDATE,
                "checkpoint_manifest_sha256": sha256_file(latest.path / "manifest.json"),
                "adapter": adapter_hashes,
                "trained_predictions_sha256": trained_sha,
                "reload_checks": reload_checks,
                "diagnostic_metrics": report["diagnostic_metrics"],
                "train_gate": train_gate,
                "source_hashes_after": after,
            }, exclusive=True)
            _fsync_directory(attempt)
            checkpoint("terminal_pre_report")
            _atomic_json(report_path, report)
            close_started = time.monotonic()
            charged = time.monotonic() - segment_started + CLOSE_RESERVE_SECONDS
            if charged > LIMITS["whole_seconds"]:
                raise TrainingV2Error("whole logical-run active time exceeded during final report")
            segment.finish("complete", charged, close_reserve_seconds=CLOSE_RESERVE_SECONDS)
            terminal_written = True
            actual_close = time.monotonic() - close_started
            report["close_proof"] = {
                "reserve_seconds": CLOSE_RESERVE_SECONDS,
                "actual_close_seconds": actual_close,
                "completed_within_reserve": actual_close <= CLOSE_RESERVE_SECONDS,
                "charged_terminal_seconds": charged}
            if actual_close > CLOSE_RESERVE_SECONDS:
                segment.append_event("terminal_close_reserve_exceeded",
                                     actual_close_seconds=actual_close)
                raise TrainingV2Error("terminal close exceeded its conservative reserve")
            _atomic_json(report_path, report)
            return report
        except BaseException as error:
            model = tokenizer = optimizer = value_and_grad = loss_fn = None
            gc.collect()
            if mx is not None:
                try:
                    mx.synchronize()
                except Exception as cleanup_error:
                    report["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
            report.update({"completed": False, "training_gate_passed": False,
                           "terminal_status": "resource_failure" if isinstance(
                               error, ResourceObservationError) else "failed",
                           "failure_type": type(error).__name__, "failure": str(error),
                           "elapsed_seconds": time.monotonic() - segment_started})
            try:
                after, versions_after = capture_source_inventory(proof)
                report["source_hashes_after"] = after
                report["versions_after"] = versions_after
                report["source_inventory_unchanged"] = (
                    after == sources and versions_after == versions)
            except Exception as source_error:
                report["source_inventory_unchanged"] = False
                report["source_verification_error"] = str(source_error)
            if attempt is not None and attempt.is_dir() and not (attempt / "manifest.json").exists():
                try:
                    _atomic_json(attempt / "failure.json", {
                        "schema": "long-context-lora-finalization-failure-v2",
                        "segment_id": segment_id, "completed": False,
                        "failure_type": type(error).__name__, "failure": str(error),
                    }, exclusive=True)
                    _fsync_directory(attempt)
                except Exception as attempt_error:
                    report["finalization_attempt_record_error"] = str(attempt_error)
            try:
                pointer = store._pointer()
                if (pointer is not None and pointer.get("status") == "eligible"
                        and pointer.get("update") == FINAL_UPDATE):
                    report["terminal_status"] = "finalization_required"
            except Exception as pointer_error:
                report["pointer_status_error"] = str(pointer_error)
            if segment_output_ready:
                _atomic_json(report_path, report)
            if not terminal_written:
                close_started = time.monotonic()
                charged = time.monotonic() - segment_started + CLOSE_RESERVE_SECONDS
                segment.finish(report["terminal_status"], charged,
                               close_reserve_seconds=CLOSE_RESERVE_SECONDS)
                actual_close = time.monotonic() - close_started
                report["close_proof"] = {
                    "reserve_seconds": CLOSE_RESERVE_SECONDS,
                    "actual_close_seconds": actual_close,
                    "completed_within_reserve": actual_close <= CLOSE_RESERVE_SECONDS,
                    "charged_terminal_seconds": charged}
                if actual_close > CLOSE_RESERVE_SECONDS:
                    segment.append_event("terminal_close_reserve_exceeded",
                                         actual_close_seconds=actual_close)
                if segment_output_ready:
                    _atomic_json(report_path, report)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fresh", "resume"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resource-proof", type=Path, required=True)
    parser.add_argument("--resource-proof-sha256", required=True)
    parser.add_argument("--segment-id", required=True)
    args = parser.parse_args()
    result = run(mode=args.mode, output=args.output,
                 resource_proof=args.resource_proof,
                 resource_proof_sha256=args.resource_proof_sha256,
                 segment_id=args.segment_id)
    print(json.dumps({"completed": result["completed"],
                      "training_gate_passed": result["training_gate_passed"],
                      "output": str(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
