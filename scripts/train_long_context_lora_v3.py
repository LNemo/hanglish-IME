"""Checkpointed v3 trainer for the reviewed overnight long-context LoRA fit.

The reviewed resource qualification is validated before this module reads FIT rows
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
from scripts.long_context_lora_checkpoint_v3 import (
    ADAM_CONFIG, ADAPTER_CONFIG, CheckpointStore, FINAL_UPDATE,
    cumulative_active_seconds, make_run_identity,
    restore_model_optimizer, validate_adapter_tensors,
)
from scripts.long_context_lora_model import (
    add_materialized_gradients, configure_fresh_lora, create_optimizer,
    mlx_log_mass_loss, require_trainable_contract, score_choices,
)
from scripts.long_context_lora_model_v2 import (
    apply_materialized_update, average_and_clip_gradients,
    materialize_microbatch, read_numeric_scalar,
)
from scripts.long_context_lora_resources import ResourceObservationError, observe_resources


OUTPUT = ROOT / ".omx/artifacts/long-context-lora-train-v3"
QUALIFICATION = ROOT / ".omx/artifacts/long-context-lora-work-v4/qualification.json"
QUALIFIER = ROOT / ".omx/artifacts/long-context-lora-work-v4/qualify_resources.py"
QUALIFIER_SHA256 = "bc08d55976dfd6c8e4b79217bcdc7996db70d046d1891d213e1fcb5ba2dc7549"
RESOURCE_REPORT = ROOT / ".omx/artifacts/long-context-lora-resource-checkpoint-v3/report.json"
RESOURCE_AUDIT = ROOT / ".omx/artifacts/long-context-lora-work-v3/execution-audit.json"
RESOURCE_RUNNER = ROOT / "scripts/probe_long_context_lora_checkpoint_v3.py"
GROUPED_MODEL = ROOT / "scripts/long_context_lora_model_v2.py"
CHECKPOINT_HELPER = ROOT / "scripts/long_context_lora_checkpoint_v3.py"
CHECKPOINT_HELPER_SHA256 = "8bc94bc90976fa3a49c9ced11a5081742e7d214901665f40360dc31480ecfa89"
FROZEN_HELPER = ROOT / "scripts/long_context_lora_checkpoint_v2.py"
FROZEN_HELPER_SHA256 = "2f5928b6d02d03cb22e9142e7bdfdcd212dcc090e5f7a1230f8f94c437806810"
GROUPED_MODEL_SHA256 = "e3cba5f1df433978306a6a8ceab446cc664d26ef1ce9cf4fa67af14f9edb148a"
CONTEXT = ROOT / ".omx/context/long-context-lora-budget-v4.md"
CONTEXT_SHA256 = "7efb74c7065b1ffdd5e9653e7f9ee570a45c302596065ac0f8fa87cf7bf6a0f0"
PLAN = ROOT / ".omx/plans/long-context-lora-budget-v4.md"
PLAN_SHA256 = "77a4b19f59aed48a241c41110ff81fc341c7500d1ee4c1d115d62fb6127b9e5e"
TEST_SPEC = ROOT / ".omx/plans/long-context-lora-budget-v4-test-spec.md"
TEST_SPEC_SHA256 = "685d1542cce8c1c1b86be73c8b4f96f50f415a6fb0d87e214cbb006da332b67a"
ARCHITECT_REVIEW = ROOT / ".omx/plans/long-context-lora-budget-v4-architect-review.md"
ARCHITECT_REVIEW_SHA256 = "db5ac1acb3d9e3162d98be260d079414b01d69b4d0ee089d949bdd520221b02e"
CRITIC_REVIEW = ROOT / ".omx/plans/long-context-lora-budget-v4-critic-review.md"
CRITIC_REVIEW_SHA256 = "43414de7194061e149fa8835e0e372bd5855507b7193960ca6d76060cc27d1b3"
HANDOFF = ROOT / ".omx/plans/long-context-lora-budget-v4-handoff.json"
HANDOFF_SHA256 = "dbb8152c665453a82cd42a7a56439dea0eeff36e1df0571ac195d6561aad946a"
LIMITS = {**resource_v1.LIMITS, "whole_seconds": 28800.0}
EXPECTED_ROWS = 2771
EXPECTED_UPDATES = 693
EXPECTED_DIAGNOSTIC_ROWS = 640
CLOSE_RESERVE_SECONDS = 1.0
PROBE_INDICES = (0, 923, 1846, 2770)
EXPECTED_PREDICATE_READS = 4157
EXPECTED_NUMERIC_READS = 3464


class TrainingV2Error(RuntimeError):
    pass


class PostEligibilityUpdateError(TrainingV2Error):
    def __init__(self, update: int, prior_cursor: int, cause: BaseException):
        super().__init__(f"eligible update {update} failed full-update settlement: {cause}")
        self.update = update
        self.prior_cursor = prior_cursor


def mark_post_eligibility_failure(report: dict[str, Any],
                                  error: PostEligibilityUpdateError) -> None:
    report.update({
        "completed": False,
        "failure_stage": "post_eligibility_update_settlement",
        "resume_permitted": False,
        "rejected_eligible_update": error.update,
        "completed_updates": error.update - 1,
        "completed_microbatches": error.prior_cursor,
    })


def settle_update_record(*, report: dict[str, Any], update_record: dict[str, Any],
                         update_index: int, cursor: int, update_started: float,
                         checkpoint, clock=time.monotonic) -> float:
    """Finish caller bookkeeping and enforce the complete update wall time."""
    report["updates"].append(update_record)
    report["completed_updates"] = update_index
    report["completed_microbatches"] = cursor
    checkpoint(f"update_{update_index}_settled", update_started)
    elapsed = clock() - update_started
    if not math.isfinite(elapsed) or elapsed < 0 or elapsed > LIMITS["update_seconds"]:
        raise TrainingV2Error("optimizer update exceeded 120 seconds")
    update_record["seconds"] = elapsed
    return elapsed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_helper_successor(frozen_path: Path = FROZEN_HELPER,
                              successor_path: Path = CHECKPOINT_HELPER) -> None:
    frozen = frozen_path.read_text()
    successor = successor_path.read_text()
    if (frozen.count("21600.0") != 2 or successor != frozen.replace(
            "21600.0", "28800.0") or successor.count("28800.0") != 2):
        raise TrainingV2Error("checkpoint helper successor differs beyond two FIT budget literals")


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


def _atomic_json(path: Path, value: Mapping[str, Any], *, exclusive: bool = False,
                 before_replace=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write((json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     indent=2, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    try:
        if before_replace is not None:
            before_replace()
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


def validate_resource_qualification(resource_qualification: Path, explicit_sha256: str,
                                    *, qualifier_path: Path = QUALIFIER,
                                    report_path: Path = RESOURCE_REPORT,
                                    helper_path: Path = CHECKPOINT_HELPER) -> dict[str, Any]:
    """Validate policy-v4 qualification before importing MLX or reading FIT data."""
    if not isinstance(explicit_sha256, str) or len(explicit_sha256) != 64:
        raise TrainingV2Error("an explicit resource-qualification SHA256 is required")
    if sha256_file(resource_qualification) != explicit_sha256:
        raise TrainingV2Error("resource-qualification file hash differs from the explicit pin")
    qualifier_sha = sha256_file(qualifier_path)
    if qualifier_sha != QUALIFIER_SHA256:
        raise TrainingV2Error("reviewed resource qualifier source differs")
    proof = _read_json(resource_qualification)
    required = {
        "schema", "qualification_passed", "qualified_for_one_fresh_fit",
        "historical_probe_passed", "historical_failure_stage",
        "historical_process_returncode", "historical_budget_seconds",
        "qualified_budget_seconds", "observed_estimated_training_seconds",
        "runtime_estimate", "reason", "physical_limits_unchanged_and_passed",
        "parity_and_restart_evidence_passed", "source_files_verified",
        "resource_observations_verified", "durable_snapshots_verified",
        "stores_verified", "retained_payload_pairs_byte_verified",
        "real_fit_rows_read", "labels_read", "mlx_imported", "qwen_model_loaded",
        "actual_training_run", "artifact_sha256", "qualifier_sha256",
        "checkpoint_helper_exact_two_literal_transform",
    }
    if set(proof) != required:
        raise TrainingV2Error("resource-qualification schema differs")
    artifact_paths = {
        "report": report_path,
        "process_audit": RESOURCE_AUDIT,
        "failure_audit": ROOT / ".omx/artifacts/long-context-lora-work-v3/failure-audit-v2.json",
        "parent_verifier": ROOT / ".omx/artifacts/long-context-lora-work-v3/verify_checkpoint_probe.py",
        "checkpoint_helper_v2": FROZEN_HELPER,
        "checkpoint_helper_v3": helper_path,
        ".omx/context/long-context-lora-budget-v4.md": CONTEXT,
        ".omx/plans/long-context-lora-budget-v4.md": PLAN,
        ".omx/plans/long-context-lora-budget-v4-test-spec.md": TEST_SPEC,
        ".omx/plans/long-context-lora-budget-v4-architect-review.md": ARCHITECT_REVIEW,
        ".omx/plans/long-context-lora-budget-v4-critic-review.md": CRITIC_REVIEW,
        ".omx/plans/long-context-lora-budget-v4-handoff.json": HANDOFF,
    }
    artifacts = proof.get("artifact_sha256")
    if (not isinstance(artifacts, dict) or set(artifacts) != set(artifact_paths)
            or any(artifacts[name] != sha256_file(path) for name, path in artifact_paths.items())):
        raise TrainingV2Error("resource qualification does not bind current artifacts")
    projection = proof["runtime_estimate"]
    false_fields = ("historical_probe_passed", "real_fit_rows_read", "labels_read",
                    "mlx_imported", "qwen_model_loaded", "actual_training_run")
    if (proof["schema"] != "long-context-lora-resource-qualification-v4"
            or proof["qualification_passed"] is not True
            or proof["qualified_for_one_fresh_fit"] is not True
            or any(proof[name] is not False for name in false_fields)
            or proof["historical_failure_stage"] != "runtime_projection"
            or proof["historical_process_returncode"] != 1
            or proof["historical_budget_seconds"] != 21600.0
            or proof["qualified_budget_seconds"] != 28800.0
            or proof["observed_estimated_training_seconds"] != 23894.80004841287
            or proof["physical_limits_unchanged_and_passed"] is not True
            or proof["parity_and_restart_evidence_passed"] is not True
            or proof["checkpoint_helper_exact_two_literal_transform"] is not True
            or proof["source_files_verified"] != 113
            or proof["resource_observations_verified"] != 497
            or proof["durable_snapshots_verified"] != 6
            or proof["stores_verified"] != 5
            or proof["retained_payload_pairs_byte_verified"] != 10
            or proof["qualifier_sha256"] != qualifier_sha
            or proof["reason"] != ("reviewed one-fit policy v4 accepts the pinned estimate "
                "under 28800 seconds while preserving the failed 21600-second v3 result")
            or not isinstance(projection, dict)
            or projection.get("within_21600_seconds") is not False
            or projection.get("estimated_training_seconds") != 23894.80004841287
            or projection["estimated_training_seconds"] > LIMITS["whole_seconds"]):
        raise TrainingV2Error("policy-v4 resource qualification did not pass")
    report = _read_json(report_path)
    if (report.get("schema") != "long-context-lora-resource-checkpoint-v3"
            or report.get("completed") is not False
            or report.get("resource_feasible") is not False
            or report.get("terminal_status") != "failed"
            or report.get("failure_stage") != "runtime_projection"
            or report.get("real_fit_rows_read") is not False
            or report.get("real_training_run") is not False
            or report.get("runtime_estimate") != projection):
        raise TrainingV2Error("historical source-free report differs from qualification")
    helper_sha = sha256_file(helper_path)
    if helper_sha != CHECKPOINT_HELPER_SHA256:
        raise TrainingV2Error("reviewed checkpoint helper source has changed")
    return {"qualification": proof, "qualification_sha256": explicit_sha256,
            "report": report, "paths": {"qualification": str(resource_qualification),
                                         "qualifier": str(qualifier_path),
                                         "report": str(report_path),
                                         "helper": str(helper_path)},
            "report_sha256": sha256_file(report_path),
            "qualifier_sha256": qualifier_sha, "helper_sha256": helper_sha}


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
        FROZEN_HELPER: FROZEN_HELPER_SHA256,
        GROUPED_MODEL: GROUPED_MODEL_SHA256,
        QUALIFIER: bundle["qualifier_sha256"],
        Path(bundle["paths"]["qualification"]): bundle["qualification_sha256"],
        RESOURCE_REPORT: bundle["report_sha256"],
        CONTEXT: CONTEXT_SHA256, PLAN: PLAN_SHA256, TEST_SPEC: TEST_SPEC_SHA256,
        ARCHITECT_REVIEW: ARCHITECT_REVIEW_SHA256,
        CRITIC_REVIEW: CRITIC_REVIEW_SHA256, HANDOFF: HANDOFF_SHA256,
        Path(__file__): sha256_file(Path(__file__)),
        ROOT / "scripts/train_long_context_lora.py": sha256_file(
            ROOT / "scripts/train_long_context_lora.py"),
    }
    for path, expected in fixed.items():
        actual = sha256_file(path)
        if actual != expected:
            raise TrainingV2Error(f"fixed v3 training source differs: {path}")
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
        if records and "tokens" in records[0]:
            first = (update - 1) * 4
            expected_full = [training_batch_record(entry, records[first + offset], manifest)
                             for offset, entry in enumerate(expected)]
            if actual != expected_full:
                raise TrainingV2Error("checkpoint token/digit/gold binding differs")
        next_index = min(update * 4, EXPECTED_ROWS)
        next_expected = None if next_index == EXPECTED_ROWS else {
            "role": records[next_index]["role"], "id": records[next_index]["id"]}
        if payload.get("next_expected") != next_expected:
            raise TrainingV2Error("checkpoint next identity differs from pinned FIT schedule")


def training_batch_record(entry: Mapping[str, Any], record: Mapping[str, Any],
                          manifest: Mapping[str, Any]) -> dict[str, Any]:
    choices = record.get("choices")
    tokens = record.get("tokens")
    gold = record.get("allowed_gold_positions")
    digit_map = manifest.get("digit_continuation_ids_by_candidate_count", {})
    digits = digit_map.get(str(len(choices) if isinstance(choices, list) else -1))
    if (not isinstance(tokens, list) or not tokens
            or not isinstance(choices, list) or not 2 <= len(choices) <= 5
            or not isinstance(gold, list) or not gold or not isinstance(digits, list)
            or (record.get("role"), record.get("id"))
            != (entry.get("role"), entry.get("id"))):
        raise TrainingV2Error("training batch payload identity differs")
    return {**dict(entry), "tokens_sha256": v1.canonical_token_digest(tokens),
            "candidate_count": len(choices), "digit_ids": list(digits),
            "allowed_gold_positions": list(gold)}


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
    prior_charged = 0.0
    for event in terminals:
        path = output / "segments" / event["segment_id"] / "report.json"
        report = _read_json(path)
        close = report.get("close_reservation", {})
        receipt_path = output / "segments" / event["segment_id"] / "completion-receipt.json"
        receipt = _read_json(receipt_path)
        terminal_digest = event.get("digest")
        if terminal_digest is None:
            from scripts.long_context_lora_checkpoint_v3 import canonical_digest
            terminal_digest = canonical_digest(event)
        expected_outcome = report.get("terminal_outcome")
        active = close.get("current_active_observed_seconds")
        current = close.get("current_charged_seconds")
        cumulative = close.get("cumulative_charged_seconds")
        if (report.get("schema") != "long-context-lora-train-v3-segment"
                or report.get("segment_id") != event["segment_id"]
                or report.get("completed") is not False
                or report.get("completion_pending") is not True
                or report.get("ledger_terminal_status") != event.get("status")
                or report.get("resume_permitted", True) is not True
                or close.get("terminal_event_digest") != terminal_digest
                or close.get("reserve_seconds") != CLOSE_RESERVE_SECONDS
                or type(active) not in (int, float) or not math.isfinite(active) or active < 0
                or type(current) not in (int, float) or not math.isfinite(current)
                or current != active + CLOSE_RESERVE_SECONDS
                or current != event.get("terminal_segment_seconds")
                or close.get("prior_charged_seconds") != prior_charged
                or type(cumulative) not in (int, float) or not math.isfinite(cumulative)
                or cumulative != prior_charged + current
                or cumulative > LIMITS["whole_seconds"]
                or receipt != {
                    "schema": "long-context-lora-train-v3-completion-receipt",
                    "segment_id": event["segment_id"],
                    "report_sha256": sha256_file(path),
                    "terminal_event_digest": terminal_digest,
                    "ledger_terminal_status": event.get("status"),
                    "terminal_outcome": expected_outcome,
                    "finalized": True,
                    "training_succeeded": expected_outcome == "success",
                    "reserve_seconds": CLOSE_RESERVE_SECONDS,
                    "observed_close_through_unlock_seconds": receipt.get(
                        "observed_close_through_unlock_seconds"),
                    "current_charged_seconds": close.get("current_charged_seconds"),
                    "prior_charged_seconds": close.get("prior_charged_seconds"),
                    "cumulative_charged_seconds": close.get("cumulative_charged_seconds"),
                }
                or type(receipt["observed_close_through_unlock_seconds"])
                not in (int, float)
                or not 0 <= receipt["observed_close_through_unlock_seconds"]
                <= CLOSE_RESERVE_SECONDS):
            raise TrainingV2Error("prior segment terminal close proof differs")
        prior_charged = cumulative


def predicate_read_accounting(output: Path, events: list[Mapping[str, Any]],
                              current_segment: str,
                              current: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    names = ("predicate_host_reads", "numeric_host_reads")
    actual = {name: 0 for name in names}
    terminals = [event for event in events if event.get("kind") == "segment_terminal"
                 and event.get("segment_id") != current_segment]
    for event in terminals:
        prior = _read_json(output / "segments" / event["segment_id"] / "report.json").get(
            "predicate_reads_segment")
        if (not isinstance(prior, dict) or set(prior) != set(names)
                or any(type(prior[name]) is not int or prior[name] < 0 for name in names)):
            raise TrainingV2Error("prior segment predicate accounting differs")
        for name in names:
            actual[name] += prior[name]
    if (set(current) != set(names)
            or any(type(current[name]) is not int or current[name] < 0 for name in names)):
        raise TrainingV2Error("current segment predicate accounting differs")
    for name in names:
        actual[name] += current[name]
    expected = {"predicate_host_reads": EXPECTED_PREDICATE_READS,
                "numeric_host_reads": EXPECTED_NUMERIC_READS}
    if any(actual[name] < expected[name] for name in names):
        raise TrainingV2Error("logical grouped predicate accounting is incomplete")
    return {"expected_logical": expected, "actual_cumulative": actual,
            "interrupted_extra": {name: actual[name] - expected[name] for name in names}}


def finish_segment_locked(*, segment, report: dict, report_path: Path,
                          ledger_status: str, terminal_outcome: str,
                          segment_started: float, prior_seconds: float,
                          clock=time.monotonic, enforce_budget: bool = True) -> dict:
    close_started = clock()
    observed = close_started - segment_started
    if not math.isfinite(observed) or observed < 0:
        raise TrainingV2Error("current segment clock differs")
    charged = observed + CLOSE_RESERVE_SECONDS
    cumulative = prior_seconds + charged
    if enforce_budget and cumulative > LIMITS["whole_seconds"]:
        raise TrainingV2Error("whole logical-run active time exceeded during final report")
    terminal = segment.finish(
        ledger_status, charged, close_reserve_seconds=CLOSE_RESERVE_SECONDS)
    report.update({
        "completed": False, "completion_pending": True,
        "terminal_outcome": terminal_outcome,
        "ledger_terminal_status": ledger_status,
        "elapsed_seconds": observed,
        "close_reservation": {
            "reserve_seconds": CLOSE_RESERVE_SECONDS,
            "prior_charged_seconds": prior_seconds,
            "current_active_observed_seconds": observed,
            "current_charged_seconds": charged,
            "cumulative_charged_seconds": cumulative,
            "terminal_event_digest": terminal["digest"],
            "reservation_scope": "terminal event, locked report, unlock, and completion receipt",
        },
    })
    _atomic_json(report_path, report)
    return {"close_started": close_started, "terminal": terminal,
            "report_sha256": sha256_file(report_path)}


def commit_segment_receipt(*, report_path: Path, report: Mapping[str, Any],
                           closing: Mapping[str, Any], clock=time.monotonic,
                           writer=_atomic_json) -> dict:
    receipt_path = report_path.parent / "completion-receipt.json"
    failure_path = report_path.parent / "completion-receipt-failure.json"
    close = report["close_reservation"]
    if sha256_file(report_path) != closing["report_sha256"]:
        raise TrainingV2Error("locked segment report changed before receipt")
    observed = clock() - closing["close_started"]
    receipt = {
        "schema": "long-context-lora-train-v3-completion-receipt",
        "segment_id": report["segment_id"],
        "report_sha256": closing["report_sha256"],
        "terminal_event_digest": closing["terminal"]["digest"],
        "ledger_terminal_status": report["ledger_terminal_status"],
        "terminal_outcome": report["terminal_outcome"],
        "finalized": True,
        "training_succeeded": report["terminal_outcome"] == "success",
        "reserve_seconds": CLOSE_RESERVE_SECONDS,
        "observed_close_through_unlock_seconds": observed,
        "current_charged_seconds": close["current_charged_seconds"],
        "prior_charged_seconds": close["prior_charged_seconds"],
        "cumulative_charged_seconds": close["cumulative_charged_seconds"],
    }

    def before_replace() -> None:
        elapsed = clock() - closing["close_started"]
        if not math.isfinite(elapsed) or elapsed < 0 or elapsed > CLOSE_RESERVE_SECONDS:
            raise TrainingV2Error("segment close exceeded its conservative reserve")

    try:
        writer(receipt_path, receipt, exclusive=True, before_replace=before_replace)
        if clock() - closing["close_started"] > CLOSE_RESERVE_SECONDS:
            raise TrainingV2Error("segment close exceeded its conservative reserve")
    except BaseException as error:
        if receipt_path.exists():
            receipt_path.unlink()
            _fsync_directory(receipt_path.parent)
        try:
            writer(failure_path, {
                "schema": "long-context-lora-train-v3-completion-receipt-failure",
                "segment_id": report["segment_id"], "completed": False,
                "report_sha256": sha256_file(report_path),
                "terminal_event_digest": closing["terminal"]["digest"],
                "failure_type": type(error).__name__, "failure": str(error),
            }, exclusive=True)
        except BaseException:
            pass
        raise
    return receipt


def _base_paths(output: Path) -> dict[str, Path]:
    base = output / "base-diagnostic"
    return {"root": base, "parent": base / "parent_predictions.jsonl",
            "base": base / "base_predictions.jsonl", "manifest": base / "manifest.json"}


def _load_jsonl(path: Path) -> list[dict]:
    if path.is_symlink() or not path.is_file():
        raise TrainingV2Error(f"prediction artifact is unavailable: {path}")
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def diagnostic_identity_sha256(records: list[Mapping[str, Any]]) -> str:
    identities = [{"position": index, "role": row.get("role"), "id": row.get("id")}
                  for index, row in enumerate(records, 1)]
    if (len(identities) != EXPECTED_DIAGNOSTIC_ROWS
            or any(not isinstance(item["role"], str) or not item["role"]
                   or not isinstance(item["id"], str) or not item["id"]
                   for item in identities)
            or len({(item["role"], item["id"]) for item in identities}) != len(identities)):
        raise TrainingV2Error("diagnostic source identity/order differs")
    return _canonical_sha256(identities)


def validate_prediction_replay(records: list[Mapping[str, Any]], predictions: list[dict],
                               *, parent: bool) -> None:
    from scripts.long_context_lora_evaluation import apply_logits, canonical

    if len(records) != EXPECTED_DIAGNOSTIC_ROWS or len(predictions) != len(records):
        raise TrainingV2Error("diagnostic prediction row count differs")
    for record, prediction in zip(records, predictions):
        logits = prediction.get("logits")
        if parent and logits is not None:
            raise TrainingV2Error("parent diagnostic unexpectedly contains logits")
        try:
            expected = apply_logits(record, logits)
        except Exception as error:
            raise TrainingV2Error("diagnostic prediction source replay failed") from error
        # JSONL persistence normalizes tuples to arrays; compare the canonical
        # JSON value while still replaying every field from the source record.
        if canonical(prediction) != canonical(expected):
            raise TrainingV2Error("diagnostic prediction identity/candidate/raw replay differs")


def _write_base_diagnostic(segment, output: Path, diagnostic_records: list[dict],
                           parent_predictions: list[dict], base_predictions: list[dict],
                           initial_hashes: Mapping[str, str]) -> dict:
    validate_prediction_replay(diagnostic_records, parent_predictions, parent=True)
    validate_prediction_replay(diagnostic_records, base_predictions, parent=False)
    paths = _base_paths(output)
    paths["root"].mkdir(exist_ok=False)
    parent_sha = v1.write_jsonl(paths["parent"], parent_predictions)
    base_sha = v1.write_jsonl(paths["base"], base_predictions)
    for path in (paths["parent"], paths["base"]):
        _fsync_file(path)
    manifest = {
        "schema": "long-context-lora-base-diagnostic-v3",
        "rows": EXPECTED_DIAGNOSTIC_ROWS,
        "diagnostic_identity_sha256": diagnostic_identity_sha256(diagnostic_records),
        "parent_sha256": parent_sha, "base_sha256": base_sha,
        "initial_adapter_hashes": dict(initial_hashes),
        "v1_prediction_cache_read": False,
    }
    _atomic_json(paths["manifest"], manifest, exclusive=True)
    manifest_sha = sha256_file(paths["manifest"])
    event = segment.append_event(
        "base_diagnostic_bound", rows=EXPECTED_DIAGNOSTIC_ROWS,
        manifest_sha256=manifest_sha,
        diagnostic_identity_sha256=manifest["diagnostic_identity_sha256"],
        parent_sha256=parent_sha, base_sha256=base_sha)
    return {"manifest": manifest, "manifest_sha256": manifest_sha,
            "event_digest": event["digest"]}


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


def _load_base_diagnostic(output, diagnostic_records, events):
    paths = _base_paths(output)
    manifest = _read_json(paths["manifest"])
    required = {"schema", "rows", "diagnostic_identity_sha256", "parent_sha256",
                "base_sha256", "initial_adapter_hashes", "v1_prediction_cache_read"}
    bindings = [event for event in events if event.get("kind") == "base_diagnostic_bound"]
    commits = [event for event in events if event.get("kind") == "checkpoint_committed"]
    if (set(manifest) != required
            or manifest.get("schema") != "long-context-lora-base-diagnostic-v3"
            or manifest.get("rows") != EXPECTED_DIAGNOSTIC_ROWS
            or manifest.get("v1_prediction_cache_read") is not False
            or manifest.get("diagnostic_identity_sha256")
            != diagnostic_identity_sha256(diagnostic_records)
            or sha256_file(paths["parent"]) != manifest.get("parent_sha256")
            or sha256_file(paths["base"]) != manifest.get("base_sha256")
            or len(bindings) != 1
            or (commits and bindings[0].get("sequence", 0) >= commits[0].get("sequence", 0))
            or bindings[0] != {
                "sequence": bindings[0].get("sequence"),
                "prior_event_digest": bindings[0].get("prior_event_digest"),
                "logical_run_id": bindings[0].get("logical_run_id"),
                "segment_id": bindings[0].get("segment_id"),
                "kind": "base_diagnostic_bound", "rows": EXPECTED_DIAGNOSTIC_ROWS,
                "manifest_sha256": sha256_file(paths["manifest"]),
                "diagnostic_identity_sha256": manifest.get("diagnostic_identity_sha256"),
                "parent_sha256": manifest.get("parent_sha256"),
                "base_sha256": manifest.get("base_sha256"),
            }):
        raise TrainingV2Error("fresh-start base diagnostic provenance differs")
    parent, base = _load_jsonl(paths["parent"]), _load_jsonl(paths["base"])
    validate_prediction_replay(diagnostic_records, parent, parent=True)
    validate_prediction_replay(diagnostic_records, base, parent=False)
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


def require_durable_train_gate(report: Mapping[str, Any], attempt_manifest: Path) -> None:
    attempt = _read_json(attempt_manifest)
    if (attempt.get("evaluation_completed") is not True
            or attempt.get("diagnostic_metrics") != report.get("diagnostic_metrics")
            or attempt.get("train_gate") != report.get("train_gate")
            or attempt.get("quality_outcome") != report.get("quality_outcome")):
        raise TrainingV2Error("durable TRAIN640 outcome differs")
    if report.get("train_gate", {}).get("passed") is not True:
        raise TrainingV2Error("TRAIN640 net-gain gate did not pass")


def validate_run_request(mode: str, output: Path, segment_id: str) -> None:
    if mode not in {"fresh", "resume"}:
        raise ValueError("mode must be fresh or resume")
    if (not isinstance(segment_id, str) or not segment_id.startswith("segment-")
            or not segment_id.removeprefix("segment-").isdigit()):
        raise TrainingV2Error("segment id must be segment-qualified")
    number = int(segment_id.removeprefix("segment-"))
    if mode == "fresh":
        if segment_id != "segment-0001":
            raise TrainingV2Error("fresh v3 training must begin at segment-0001")
        if output.exists():
            raise FileExistsError(f"v3 logical-run root already exists: {output}")
    elif not output.is_dir() or output.is_symlink() or number < 2:
        raise TrainingV2Error("resume requires an existing v3 root and segment number >=2")


def run(*, mode: str, output: Path, resource_qualification: Path,
        resource_qualification_sha256: str, segment_id: str) -> dict[str, Any]:
    """Execute one fresh or resumed segment. There is no inline retry path."""
    validate_run_request(mode, output, segment_id)
    validate_helper_successor()
    proof = validate_resource_qualification(resource_qualification, resource_qualification_sha256)
    sources, versions = capture_source_inventory(proof)
    data_manifest_path = v1.DATA_DIR / "manifest.json"
    if sha256_file(data_manifest_path) != v1.DATA_MANIFEST_SHA256:
        raise TrainingV2Error("reviewed training-data manifest differs before FIT access")
    data_manifest = _read_json(data_manifest_path)
    schedule_sha = data_manifest.get("file_hashes", {}).get("schedule.jsonl")
    if not isinstance(schedule_sha, str) or len(schedule_sha) != 64:
        raise TrainingV2Error("reviewed schedule hash is unavailable")
    # The reviewed qualification/source boundary is outside the fit budget.
    # Fresh/resume resource preflight, store creation/open, and locking are inside it.
    segment_started = time.monotonic()
    if mode == "fresh":
        baseline = resource_v1.system_resources()
        run_identity = make_run_identity(
            logical_run_id=f"long-context-lora-fit-v3-{uuid.uuid4().hex}", mode="fit",
            baseline_swap_used_bytes=baseline["swap_used_bytes"],
            schedule_sha256=schedule_sha, microbatches=EXPECTED_ROWS,
            updates=EXPECTED_UPDATES, final_update_microbatches=3,
            source_hashes=sources,
            config_hashes={"plan": PLAN_SHA256, "test_spec": TEST_SPEC_SHA256,
                           "handoff": HANDOFF_SHA256,
                           "data_manifest": v1.DATA_MANIFEST_SHA256,
                           "resource_qualification": resource_qualification_sha256},
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
        # This pre-lock read is deliberately fail-closed: a missing immutable
        # receipt must reject resume before Segment.__enter__ appends resume_start.
        validate_prior_close_proofs(output, store._events(), segment_id)
    report = {
        "schema": "long-context-lora-train-v3-segment", "mode": mode,
        "segment_id": segment_id, "completed": False, "training_gate_passed": False,
        "deployed": False, "auto_resume": False, "resume_permitted": True,
        "resource_qualification": {
            "path": str(resource_qualification), "sha256": resource_qualification_sha256,
            "report_sha256": proof["report_sha256"],
            "qualifier_sha256": proof["qualifier_sha256"],
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
    terminal_written = False
    segment_output_ready = False
    prior_seconds = 0.0
    pending_error = None
    closing = None
    quality_outcome_durable = False
    predicate_counters = {"predicate_host_reads": 0, "numeric_host_reads": 0}
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
            logical_started = segment_started - prior_seconds
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
                report["base_diagnostic_binding"] = _write_base_diagnostic(
                    segment, output, diagnostic, parent_predictions, base_predictions,
                    initial_hashes)
                optimizer = create_optimizer()
                start_update = 1
                cursor = 0
            else:
                assert restored is not None
                initial = configure_fresh_lora(mx, model)
                initial_inventory = validate_adapter_tensors(initial)
                del initial
                parent_predictions, base_predictions, base_manifest = _load_base_diagnostic(
                    output, diagnostic, [item["payload"] for item in snapshot["events"]])
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
                        materialize_microbatch(
                            mx, model, loss, gradients, counters=predicate_counters)
                        accumulated = add_materialized_gradients(mx, accumulated, gradients)
                        checkpoint(f"update_{update_index}_microbatch_{len(losses) + 1}",
                                   update_started)
                        losses.append(read_numeric_scalar(loss, counters=predicate_counters))
                    clipped, norm = average_and_clip_gradients(
                        mx, accumulated, len(schedule_batch), counters=predicate_counters)
                    checkpoint(f"update_{update_index}_averaged_clipped", update_started)
                    apply_materialized_update(
                        mx, model, optimizer, clipped, counters=predicate_counters)
                    next_index = min(update_index * 4, EXPECTED_ROWS)
                    next_schedule = None if next_index == EXPECTED_ROWS else {
                        "role": data["records"][next_index]["role"],
                        "id": data["records"][next_index]["id"]}
                    checkpoint_batch = [
                        training_batch_record(entry, data["records"][
                            (update_index - 1) * 4 + position], data["manifest"])
                        for position, entry in enumerate(schedule_batch)]
                    result = segment.publish_checkpoint(
                        mx=mx, adapter_tensors=require_trainable_contract(model),
                        optimizer_state=optimizer.state, update=update_index,
                        completed_microbatches=cursor, next_schedule=next_schedule,
                        batch_records=checkpoint_batch, losses=losses,
                        adapter_config=ADAPTER_CONFIG, optimizer_config=ADAM_CONFIG,
                        recipe=run_identity["recipe"],
                        elapsed_seconds=time.monotonic() - logical_started,
                        update_elapsed_before_checkpoint=time.monotonic() - update_started,
                        resource_callback=lambda stage, u=update_index, s=update_started:
                            checkpoint(f"update_{u}_checkpoint_{stage}", s))
                    try:
                        validate_schedule_binding(
                            data, run_identity,
                            {"manifests": [{"payload": result["manifest"]}]})
                        update_record = {
                            "update": update_index, "microbatches": len(schedule_batch),
                            "batch_records": checkpoint_batch, "microbatch_losses": losses,
                            "gradient_norm_before_clip": read_numeric_scalar(
                                norm, counters=predicate_counters),
                            "seconds": None,
                            "helper_checkpoint_inclusive_seconds": result["update_seconds"],
                            "checkpoint_timings": result["timings"],
                            "checkpoint_payload_bytes": result["payload_bytes"],
                            "checkpoint_manifest_sha256": result["pointer"]["manifest_sha256"],
                        }
                        settle_update_record(
                            report=report, update_record=update_record,
                            update_index=update_index, cursor=cursor,
                            update_started=update_started, checkpoint=checkpoint)
                    except BaseException as error:
                        if ("update_record" in locals()
                                and update_record.get("update") == update_index):
                            update_record["eligible_but_rejected"] = True
                        raise PostEligibilityUpdateError(
                            update_index, cursor - len(schedule_batch), error) from error

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
            after, versions_after = capture_source_inventory(proof)
            if after != sources or versions_after != versions:
                raise TrainingV2Error("training sources changed during the segment")
            quality_outcome = "passed" if train_gate["passed"] is True else "quality_rejected"
            predicate_accounting = predicate_read_accounting(
                output, [item["payload"] for item in snapshot["events"]], segment_id,
                predicate_counters)
            report.update({
                "training_gate_passed": train_gate["passed"],
                "quality_outcome": quality_outcome,
                "source_hashes_after": after, "versions_after": versions_after,
                "final_checkpoint": latest.manifest,
                "final_adapter": adapter_hashes,
                "reload_checks": reload_checks,
                "trained_predictions_sha256": trained_sha,
                "diagnostic_metrics": {"fresh_parent": parent_metrics,
                                       "fresh_base": base_metrics,
                                       "final_adapter": trained_metrics},
                "train_gate": train_gate,
                "source_inventory_unchanged": True,
                "all_diagnostic_labels_consulted_after_model_release": True,
                "predicate_reads_segment": dict(predicate_counters),
                "predicate_read_accounting": predicate_accounting,
            })
            attempt_record = {
                "schema": "long-context-lora-finalization-attempt-v3",
                "segment_id": segment_id, "checkpoint_update": FINAL_UPDATE,
                "evaluation_completed": True, "quality_outcome": quality_outcome,
                "checkpoint_manifest_sha256": sha256_file(latest.path / "manifest.json"),
                "adapter": adapter_hashes,
                "parent_predictions_sha256": sha256_file(_base_paths(output)["parent"]),
                "base_predictions_sha256": sha256_file(_base_paths(output)["base"]),
                "trained_predictions_sha256": trained_sha,
                "reload_checks": reload_checks,
                "diagnostic_metrics": report["diagnostic_metrics"],
                "train_gate": train_gate,
                "source_hashes_after": after,
                "predicate_read_accounting": predicate_accounting,
            }
            _atomic_json(attempt / "manifest.json", attempt_record, exclusive=True)
            _fsync_directory(attempt)
            quality_outcome_durable = True
            require_durable_train_gate(report, attempt / "manifest.json")
            checkpoint("terminal_pre_report")
            report["resume_permitted"] = False
            closing = finish_segment_locked(
                segment=segment, report=report, report_path=report_path,
                ledger_status="complete", terminal_outcome="success",
                segment_started=segment_started, prior_seconds=prior_seconds)
            terminal_written = True
        except BaseException as error:
            model = tokenizer = optimizer = value_and_grad = loss_fn = None
            gc.collect()
            if mx is not None:
                try:
                    mx.synchronize()
                except Exception as cleanup_error:
                    report["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
            report.update({"completed": False, "training_gate_passed": False,
                           "failure_type": type(error).__name__, "failure": str(error),
                           "elapsed_seconds": time.monotonic() - segment_started,
                           "predicate_reads_segment": dict(predicate_counters)})
            if isinstance(error, PostEligibilityUpdateError):
                mark_post_eligibility_failure(report, error)
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
                        "schema": "long-context-lora-finalization-failure-v3",
                        "segment_id": segment_id, "completed": False,
                        "failure_type": type(error).__name__, "failure": str(error),
                    }, exclusive=True)
                    _fsync_directory(attempt)
                except Exception as attempt_error:
                    report["finalization_attempt_record_error"] = str(attempt_error)
            ledger_status = ("resource_failure" if isinstance(error, ResourceObservationError)
                             else "failed")
            terminal_outcome = ledger_status
            if quality_outcome_durable:
                if report.get("quality_outcome") == "quality_rejected":
                    ledger_status, terminal_outcome = "failed", "quality_rejected"
                    report["resume_permitted"] = False
            elif not isinstance(error, PostEligibilityUpdateError):
                try:
                    pointer = store._pointer()
                    if (pointer is not None and pointer.get("status") == "eligible"
                            and pointer.get("update") == FINAL_UPDATE):
                        ledger_status = terminal_outcome = "finalization_required"
                except Exception as pointer_error:
                    report["pointer_status_error"] = str(pointer_error)
            terminal_written = terminal_written or bool(getattr(segment, "_terminal", False))
            if not terminal_written:
                closing = finish_segment_locked(
                    segment=segment, report=report, report_path=report_path,
                    ledger_status=ledger_status, terminal_outcome=terminal_outcome,
                    segment_started=segment_started, prior_seconds=prior_seconds,
                    enforce_budget=False)
                terminal_written = True
            pending_error = (error, error.__traceback__)

    if closing is None:
        if pending_error is not None:
            error, traceback = pending_error
            raise error.with_traceback(traceback)
        raise TrainingV2Error("segment ended without durable terminal evidence")
    receipt = commit_segment_receipt(
        report_path=report_path, report=report, closing=closing)
    result = {**report, "completion_pending": False, "finalized": True,
              "completed": receipt["training_succeeded"],
              "completion_receipt": receipt}
    if pending_error is not None:
        error, traceback = pending_error
        raise error.with_traceback(traceback)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fresh", "resume"), required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resource-qualification", type=Path, required=True)
    parser.add_argument("--resource-qualification-sha256", required=True)
    parser.add_argument("--segment-id", required=True)
    args = parser.parse_args()
    result = run(mode=args.mode, output=args.output,
                 resource_qualification=args.resource_qualification,
                 resource_qualification_sha256=args.resource_qualification_sha256,
                 segment_id=args.segment_id)
    print(json.dumps({"finalized": result["finalized"],
                      "completed": result["completed"],
                      "training_gate_passed": result["training_gate_passed"],
                      "output": str(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
