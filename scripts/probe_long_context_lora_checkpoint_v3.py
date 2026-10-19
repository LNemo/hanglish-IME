"""Run the grouped-predicate v3 source-free checkpoint/resource gate once."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Callable
from datetime import datetime, timezone


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import probe_long_context_lora_resources as v1
from scripts import probe_long_context_lora_checkpoint_v2 as frozen_v2
from scripts.long_context_lora_model import (
    ADAM_BETAS, ADAM_EPS, CLIP_NORM, LEARNING_RATE, LORA_CONFIG, LORA_LAYERS,
    SEED, add_materialized_gradients, configure_fresh_lora, create_optimizer,
    mlx_log_mass_loss, ordinary_forward_choice_scores,
    require_matching_gradient_tree, require_trainable_contract, score_choices,
    tensor_hashes,
)
from scripts.long_context_lora_model_v2 import (
    apply_materialized_update, average_and_clip_gradients,
    materialize_microbatch, read_numeric_scalar,
)
from scripts.long_context_lora_resources import (
    ResourceObservationError, observe_resources,
)


DEFAULT_OUTPUT = ROOT / ".omx/artifacts/long-context-lora-resource-checkpoint-v3"
PLAN = ROOT / ".omx/plans/long-context-lora-validation-v3.md"
TEST_SPEC = ROOT / ".omx/plans/long-context-lora-validation-v3-test-spec.md"
HANDOFF = ROOT / ".omx/plans/long-context-lora-validation-v3-handoff.json"
ARCHITECT_REVIEW = ROOT / ".omx/plans/long-context-lora-validation-v3-architect-review.md"
CRITIC_REVIEW = ROOT / ".omx/plans/long-context-lora-validation-v3-critic-review.md"
CONTEXT = ROOT / ".omx/context/long-context-lora-validation-v3.md"
FAILED_V1_REPORT = ROOT / ".omx/artifacts/long-context-lora-train-v1/report.json"
FAILED_V2_REPORT = ROOT / ".omx/artifacts/long-context-lora-resource-checkpoint-v2/report.json"
RESOURCE_V2 = ROOT / "scripts/probe_long_context_lora_resources_v2.py"
CHECKPOINT_HELPER = ROOT / "scripts/long_context_lora_checkpoint_v2.py"
FROZEN_V2_PRODUCER = ROOT / "scripts/probe_long_context_lora_checkpoint_v2.py"
MODEL_V2 = ROOT / "scripts/long_context_lora_model_v2.py"
MODEL_V2_TEST = ROOT / "tests/test_long_context_lora_model_v2.py"
PROBE_V3_TEST = ROOT / "tests/test_probe_long_context_lora_checkpoint_v3.py"
SMALL_MLX_VERIFIER = ROOT / ".omx/artifacts/long-context-lora-work-v3/verify_grouped_predicates_small.py"
SMALL_MLX_PROOF = ROOT / ".omx/artifacts/long-context-lora-work-v3/grouped-predicates-small.json"
SMALL_MLX_PROOF_SHA256 = "a478f6013482d48231207d3098d5a0c5369d718e3b367976ac329b333efc62ee"
SMALL_MLX_VERIFIER_SHA256 = "0854ffdc0ebe4846be7fc6f03494c243783a484993ff6ab252ee8a4b39d96097"
MLX_UTILS = ROOT / ".venv-mlx/lib/python3.12/site-packages/mlx/utils.py"

FIXED_INPUTS = {
    PLAN: "2610c73cddf12f0766c31db3f6df6cfc55b9bc49e4a71efc89619640860093f6",
    TEST_SPEC: "5406e458074ca335b56f5da413c0816bc28446bde88a58f6f0ffd0b9feefcbb9",
    HANDOFF: "d9b1f96bd5b7fd2d21ab77e6e26b15bd286bc483fe0b9bdd9986f7f4b0d421ba",
    ARCHITECT_REVIEW: "e7550571cbff2bc5b1c39145dc55e0a9179e546b9253ccf3dc46fdf84d9d2c5b",
    CRITIC_REVIEW: "3183296b3564fcff8ea3dc1490292039d71930063ab359c28d44ac928e99af7e",
    CONTEXT: "743281f8c74f550988364638d85fe47e1092e02a4099cba5192569b72bc42a8f",
    FAILED_V1_REPORT: "2b5791dcfc340c56aba2e4cda8a7111c944f32ce21182160a21b7d1bb1777a43",
    RESOURCE_V2: "ccf482a074ba9ff688c43ee329fbd81513316af94d28e8ee746cfb9bbf0d898d",
    FAILED_V2_REPORT: "65f6519cfabee72310fe217cd630b17849400c25186d647e627f96827a810f1c",
    FROZEN_V2_PRODUCER: "8874e19849fa3a7400237efd651e64004c538eb889281b105c7d4a34a7ebf709",
    CHECKPOINT_HELPER: "2f5928b6d02d03cb22e9142e7bdfdcd212dcc090e5f7a1230f8f94c437806810",
    MLX_UTILS: "9f68b6fac3c5b2459e1e61b915f93d19934cbbab47649c968d37c45d84f4df9f",
    ROOT / "scripts/long_context_lora_resources.py":
        "b4cce78d89f5622521c4a60f4e95827a2994f95a37741577880c423051850a2a",
    ROOT / "scripts/long_context_lora_model.py":
        "eb9b6c23ef6f639773e34782f04e1499810a448eee3a688a3385d9ef29c12015",
}

SYNTHETIC_LIMITS = dict(v1.LIMITS)
MAIN_UPDATES = 20
PARITY_UPDATES = 4
CHECKPOINT_UPDATE = 2
REFERENCE_UPDATE = 4
INTERRUPT_AFTER_MICROBATCHES = (1, 2, 3)
ACCUMULATION = 4
SCORER_TIMINGS = 40
LOGIT_ATOL = 1e-5
LOGIT_RTOL = 1e-5
SEGMENT_CLOSE_RESERVE_SECONDS = 1.0
FINAL_REPORT_CLOSE_RESERVE_SECONDS = 1.0
EXPECTED_OBSERVATIONS = 497
EXPECTED_CLOSURES = 8
EXPECTED_PACKAGES = {"mlx": "0.31.2", "mlx-lm": "0.31.3", "transformers": "5.12.1"}

ADAPTER_CONFIG = {
    "fine_tune_type": "lora", "num_layers": LORA_LAYERS,
    "lora_parameters": LORA_CONFIG,
}
OPTIMIZER_CONFIG = {
    "learning_rate": LEARNING_RATE,
    "betas": list(ADAM_BETAS), "eps": ADAM_EPS,
    "bias_correction": True,
}
RECIPE = {
    "seed": SEED,
    "lora": {"rank": LORA_CONFIG["rank"], "scale": LORA_CONFIG["scale"],
             "dropout": LORA_CONFIG["dropout"], "layers": LORA_LAYERS,
             "keys": list(LORA_CONFIG["keys"])},
    "adam": OPTIMIZER_CONFIG, "accumulation": ACCUMULATION,
    "clip_norm": CLIP_NORM, "loss": "finite_choice_log_mass_float32",
    "order": "immutable_schedule_no_shuffle",
    "post_initialization_random_consumers": [],
    "synthetic_only": True,
}


def p95(values) -> float:
    return v1.p95(values)


def checkpoint_runtime_estimate(update_times, scorer_times) -> dict:
    """Use the unchanged v1 projection with checkpoint-inclusive update timings."""
    if len(update_times) != MAIN_UPDATES or len(scorer_times) != SCORER_TIMINGS:
        raise ValueError("runtime estimate requires exactly 20 updates and 40 scorer timings")
    result = v1.runtime_estimate(update_times, scorer_times)
    if result["formula"] != "1.5*(693*update_p95+1280*probe_scoring_p95)+300":
        raise RuntimeError("frozen v1 runtime formula differs")
    return {**result, "update_timing_scope": "gradient plus checkpoint transaction and retention",
            "checkpoint_frequency": "every optimizer update"}


def require_runtime_projection(estimate: dict) -> None:
    if estimate.get("within_21600_seconds") is not True:
        raise RuntimeError("checkpoint-inclusive projected training runtime exceeds 21600 seconds")


def require_main_phase(summary: dict) -> None:
    if summary.get("fresh_checkpoint_reload", {}).get("passed") is not True:
        raise RuntimeError("main checkpoint source-only scorer reload parity failed")


def require_branch_phase(branch: dict) -> None:
    if (branch.get("parity", {}).get("passed") is not True
            or branch.get("trajectory_parity", {}).get("passed") is not True):
        raise RuntimeError("checkpoint restart parity failed")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict, *, exclusive: bool = False,
                 before_replace: Callable[[], None] | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", closefd=False) as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)
            stream.write("\n")
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
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return _sha256_file(path)


def commit_final_report(report_path: Path, report: dict, started: float, *,
                        clock=time.monotonic,
                        writer: Callable[..., str] = _atomic_json) -> None:
    """Bind success to a report-fsync measurement and reserved receipt commit."""
    if report.get("completed") is not True:
        report["elapsed_seconds"] = clock() - started
        writer(report_path, report)
        return
    receipt_path = report_path.parent / "completion-receipt.json"
    try:
        before_commit = clock() - started
        if before_commit < 0 or not math.isfinite(before_commit):
            raise RuntimeError("final report clock differs")
        report["final_report_close"] = {
            "reserve_seconds": FINAL_REPORT_CLOSE_RESERVE_SECONDS,
            "active_elapsed_observed_seconds": before_commit,
            "charged_elapsed_seconds": SYNTHETIC_LIMITS["whole_seconds"],
            "completion_receipt_required": True,
            "reservation_scope": (
                "report rename/directory fsync and completion receipt atomic write"),
        }
        report["completion_receipt_path"] = receipt_path.name
        report["elapsed_seconds"] = before_commit

        def report_precommit() -> None:
            if clock() - started + FINAL_REPORT_CLOSE_RESERVE_SECONDS \
                    > SYNTHETIC_LIMITS["whole_seconds"]:
                raise RuntimeError(
                    "whole resource probe time bound exceeded before final report commit")

        report_sha256 = writer(report_path, report, before_replace=report_precommit)
        through_report_fsync = clock() - started
        if (through_report_fsync < before_commit
                or not math.isfinite(through_report_fsync)):
            raise RuntimeError("final report clock differs")
        receipt = {
            "schema": "long-context-lora-v3-completion-receipt",
            "report_sha256": report_sha256,
            "final_report_close": report["final_report_close"],
            "report_elapsed_through_fsync_seconds": through_report_fsync,
        }

        def receipt_precommit() -> None:
            if clock() - started + FINAL_REPORT_CLOSE_RESERVE_SECONDS \
                    > SYNTHETIC_LIMITS["whole_seconds"]:
                raise RuntimeError(
                    "whole resource probe time bound exceeded before completion receipt")

        writer(receipt_path, receipt, exclusive=True,
               before_replace=receipt_precommit)
    except BaseException as error:
        if receipt_path.exists():
            receipt_path.unlink()
            descriptor = os.open(receipt_path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        report.update({
            "completed": False, "resource_feasible": False,
            "terminal_status": "failed", "failure_stage": "final_report_commit",
            "failure_type": type(error).__name__, "failure": str(error),
            "real_fit_launch_allowed_by_runtime_estimate": False,
        })
        try:
            report["elapsed_seconds"] = clock() - started
            writer(report_path, report)
        except BaseException:
            pass
        raise


def persist_phase(output: Path, report_path: Path, report: dict, phase: str,
                  payload: dict) -> dict:
    sequence = len(report.setdefault("durable_phases", [])) + 1
    prior = None if sequence == 1 else report["durable_phases"][-1]["sha256"]
    snapshot = {"schema": "long-context-lora-phase-snapshot-v3",
                "sequence": sequence, "phase": phase,
                "prior_snapshot_sha256": prior, "payload": payload}
    relative = Path("phase-snapshots") / f"{sequence:04d}-{phase}.json"
    digest = _atomic_json(output / relative, snapshot, exclusive=True)
    entry = {"sequence": sequence, "phase": phase, "path": str(relative),
             "sha256": digest}
    report["durable_phases"].append(entry)
    _atomic_json(report_path, report)
    return entry


def persist_observe_and_gate(
        output: Path, report_path: Path, report: dict, phase: str, payload: dict,
        *, persistence_gate: str, observation_stage: str, validation_gate: str,
        observe: Callable[[str], object], validate: Callable[[], None],
        persister: Callable[[Path, Path, dict, str, dict], dict] = persist_phase,
) -> dict:
    """Durably preserve a completed phase before resource and parity gates."""
    report["active_gate"] = persistence_gate
    entry = persister(output, report_path, report, phase, payload)
    observe(observation_stage)
    report["active_gate"] = validation_gate
    validate()
    return entry


def max_abs_delta(left, right) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("logit vectors must be nonempty and equal length")
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def logits_match(left, right) -> tuple[bool, float]:
    delta = max_abs_delta(left, right)
    close = all(abs(float(a) - float(b)) <= LOGIT_ATOL + LOGIT_RTOL * abs(float(b))
                for a, b in zip(left, right))
    return close and left.index(max(left)) == right.index(max(right)), delta


def exact_state_match(reference: dict, actual: dict) -> dict:
    """Compare the durable update-4 resume contract without numeric tolerance."""
    exact_fields = (
        "adapter_tensor_hashes", "optimizer_tensor_hashes", "optimizer_step",
        "cursor", "microbatch_losses", "mean_loss", "gradient_norm_before_clip",
    )
    mismatches = [name for name in exact_fields if reference.get(name) != actual.get(name)]
    reference_logits = reference.get("source_only_logits", [])
    actual_logits = actual.get("source_only_logits", [])
    if len(reference_logits) != 4 or len(actual_logits) != 4:
        mismatches.append("source_only_logits_count")
        logit_checks = []
    else:
        logit_checks = []
        for expected, observed in zip(reference_logits, actual_logits):
            same, delta = logits_match(expected["logits"], observed["logits"])
            same = same and expected["prompt_sha256"] == observed["prompt_sha256"]
            logit_checks.append({"passed": same, "max_absolute_delta": delta,
                                 "prompt_sha256": observed["prompt_sha256"]})
        if not all(item["passed"] for item in logit_checks):
            mismatches.append("source_only_logits")
    return {"passed": not mismatches, "exact_fields": list(exact_fields),
            "mismatches": mismatches, "logit_checks": logit_checks,
            "logit_atol": LOGIT_ATOL, "logit_rtol": LOGIT_RTOL}


def compact_update_trajectory(updates: list[dict]) -> list[dict]:
    fields = ("update", "cursor", "microbatch_losses", "mean_loss",
              "gradient_norm_before_clip")
    return [{field: row[field] for field in fields} for row in updates]


def exact_trajectory_match(reference: list[dict], actual: list[dict]) -> dict:
    if len(reference) != PARITY_UPDATES or len(actual) != PARITY_UPDATES:
        return {"passed": False, "mismatched_updates": [],
                "reason": "trajectory must contain exactly four updates"}
    expected = compact_update_trajectory(reference)
    observed = compact_update_trajectory(actual)
    mismatches = [index + 1 for index, (left, right) in enumerate(zip(expected, observed))
                  if left != right]
    return {"passed": not mismatches, "mismatched_updates": mismatches,
            "reference": expected, "actual": observed}


def synthetic_descriptor() -> dict:
    return {
        "source_free": True, "fit_files_read": False, "labels_read": False,
        "v1_prediction_cache_read": False, "old_adapter_loaded": False,
        "shapes": [list(shape) for shape in v1.SHAPES],
        "main": {"updates": MAIN_UPDATES, "microbatches": MAIN_UPDATES * ACCUMULATION,
                 "checkpoint_every_update": True},
        "restart": {"branches": len(INTERRUPT_AFTER_MICROBATCHES),
                    "logical_updates_per_branch": PARITY_UPDATES,
                    "logical_microbatches_per_branch": PARITY_UPDATES * ACCUMULATION,
                    "restore_checkpoint_update": CHECKPOINT_UPDATE,
                    "interrupt_update": CHECKPOINT_UPDATE + 1,
                    "discarded_partial_microbatches": list(INTERRUPT_AFTER_MICROBATCHES),
                    "resumes_per_branch": 1},
        "original_reference": {"updates": 4, "checkpoint_every_update": True,
                               "direct_frozen_v2_calls": True},
    }


def _update_stages(prefix: str, update: int) -> list[str]:
    return [*[f"{prefix}update_{update}_microbatch_{position}" for position in range(1, 5)],
            f"{prefix}update_{update}_averaged_clipped",
            *[f"{prefix}update_{update}_checkpoint_{stage}" for stage in
              ("post_materialize", "post_reload", "precommit", "postcommit",
               "final_eligibility")],
            f"{prefix}update_{update}_eligible"]


def expected_observation_stages() -> list[str]:
    stages = ["tiny_math", "reference_model_load", "reference_warm_1024_backward"]
    for update in range(1, 5):
        stages.extend(_update_stages("reference_", update))
    stages.extend([f"reference_update4_{shape[0]}" for shape in v1.SHAPES])
    stages.extend(["reference_model_released", "reference_phase_persisted",
                   "main_model_load", "warm_1024_backward"])
    for update in range(1, MAIN_UPDATES + 1):
        stages.extend(_update_stages("", update))
        if update == REFERENCE_UPDATE:
            stages.extend([f"main_update4_reference_{shape[0]}" for shape in v1.SHAPES])
    stages.extend([f"main_ordinary_forward_{shape[0]}" for shape in v1.SHAPES])
    stages.extend([f"main_inference_{shape[0]}_{repeat}" for shape in v1.SHAPES
                   for repeat in range(1, v1.INFERENCE_REPETITIONS + 1)])
    stages.extend(["main_trained_model_released",
                   *[f"main_fresh_reload_{shape[0]}" for shape in v1.SHAPES],
                   "main_model_released", "main_phase_persisted"])
    for interrupted in INTERRUPT_AFTER_MICROBATCHES:
        prefix = f"restart_{interrupted}_"
        stages.append(f"{prefix}initial_model_load")
        for update in (1, 2):
            stages.extend(_update_stages("", update))
        stages.extend(f"interrupted_update_3_microbatch_{position}"
                      for position in range(1, interrupted + 1))
        stages.extend([f"{prefix}interrupted_model_released",
                       f"{prefix}restored_model_load"])
        for update in (3, 4):
            stages.extend(_update_stages("", update))
        stages.extend([*[f"{prefix}update4_{shape[0]}" for shape in v1.SHAPES],
                       f"{prefix}model_released", f"{prefix}phase_persisted"])
    stages.extend(["projection_phase_persisted", "terminal"])
    return stages


def resume_preflight_metrics(system_metrics: dict) -> dict:
    if (set(system_metrics) != {"free_percent", "swap_used_bytes"}
            or type(system_metrics["swap_used_bytes"]) is not int
            or type(system_metrics["free_percent"]) not in (int, float)
            or not math.isfinite(system_metrics["free_percent"])):
        raise ValueError("system resource metrics differ")
    return {"swap_used_bytes": system_metrics["swap_used_bytes"],
            "system_free_percent": float(system_metrics["free_percent"])}


def reserved_terminal_seconds(segment_started: float, close_started: float) -> float:
    elapsed = close_started - segment_started
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("segment close clock differs")
    return elapsed + SEGMENT_CLOSE_RESERVE_SECONDS


def validate_segment_close(close_started: float, clock=time.monotonic) -> float:
    elapsed = clock() - close_started
    if not math.isfinite(elapsed) or elapsed < 0:
        raise RuntimeError("segment close clock differs")
    if elapsed > SEGMENT_CLOSE_RESERVE_SECONDS:
        raise RuntimeError("segment terminal fsync/audit/unlock exceeded close reserve")
    return elapsed


def _append_terminal_to_audit(audit: dict, terminal: dict,
                              close_elapsed: float) -> dict:
    result = dict(audit)
    result["events"] = [*audit["events"], terminal]
    result["terminal_close"] = {
        "reserve_seconds": SEGMENT_CLOSE_RESERVE_SECONDS,
        "actual_seconds": close_elapsed,
        "within_reserve": close_elapsed <= SEGMENT_CLOSE_RESERVE_SECONDS,
    }
    return result


def _closure_record(run_relative_path: str, segment_id: str, terminal: dict,
                    actual_close_seconds: float) -> dict:
    charged = terminal["payload"]["terminal_segment_seconds"]
    return {"run_relative_path": run_relative_path, "segment_id": segment_id,
            "reserve_seconds": SEGMENT_CLOSE_RESERVE_SECONDS,
            "actual_close_seconds": actual_close_seconds,
            "charged_terminal_seconds": charged,
            "terminal_event_digest": terminal["digest"],
            "completed_within_reserve": actual_close_seconds <= SEGMENT_CLOSE_RESERVE_SECONDS}


def capture_provenance() -> tuple[dict, dict]:
    hashes, versions = v1.capture_provenance()
    if versions != EXPECTED_PACKAGES:
        raise RuntimeError(f"package versions differ: {versions}")
    if (_sha256_file(SMALL_MLX_PROOF) != SMALL_MLX_PROOF_SHA256
            or _sha256_file(SMALL_MLX_VERIFIER) != SMALL_MLX_VERIFIER_SHA256):
        raise RuntimeError("small actual MLX proof provenance differs")
    small = json.loads(SMALL_MLX_PROOF.read_text())
    small_sources = small.get("source_sha256", {})
    if (small.get("passed") is not True or small.get("actual_mlx") is not True
            or small.get("qwen_model_loaded") is not False
            or small.get("fit_rows_read") is not False
            or small.get("predicate_reads_per_group") != 1
            or small_sources.get(v1.relative_path(MODEL_V2)) != v1.sha256_file(MODEL_V2)
            or small_sources.get(v1.relative_path(SMALL_MLX_VERIFIER))
            != v1.sha256_file(SMALL_MLX_VERIFIER)):
        raise RuntimeError("small actual MLX grouped-predicate proof differs")
    handoff = json.loads(HANDOFF.read_text())
    artifacts = handoff.get("planning_artifacts", {})
    pins = handoff.get("sha256", {})
    if (handoff.get("planning_state") != "complete"
            or artifacts.get("plan") != str(PLAN.relative_to(ROOT))
            or artifacts.get("test_spec") != str(TEST_SPEC.relative_to(ROOT))
            or pins.get("plan") != FIXED_INPUTS[PLAN]
            or pins.get("test_spec") != FIXED_INPUTS[TEST_SPEC]
            or handoff.get("ralplan_architect_review", {}).get("verdict") != "APPROVE"
            or handoff.get("ralplan_critic_review", {}).get("verdict") != "APPROVE"
            or handoff.get("ralplan_critic_review", {}).get("after_architect") is not True
            or handoff.get("ralplan_execution_handoff", {}).get("authorized") is not True
            or handoff.get("actual_qwen_launch_allowed") is not False
            or handoff.get("actual_fit_launch_allowed") is not False):
        raise RuntimeError("approved validation-v3 handoff differs")
    for path, expected in FIXED_INPUTS.items():
        actual = v1.sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"fixed validation-v3 input differs: {v1.relative_path(path)}")
        hashes[v1.relative_path(path)] = actual
    for path in (Path(__file__), CHECKPOINT_HELPER, MODEL_V2, MODEL_V2_TEST,
                 PROBE_V3_TEST, SMALL_MLX_VERIFIER, SMALL_MLX_PROOF):
        if not path.is_file():
            raise RuntimeError(f"validation-v3 source is missing: {v1.relative_path(path)}")
        hashes[v1.relative_path(path)] = v1.sha256_file(path)
    return dict(sorted(hashes.items())), dict(sorted(versions.items()))


def _flat_optimizer_state(optimizer) -> dict:
    from mlx.utils import tree_flatten

    return dict(tree_flatten(optimizer.state))


def _optimizer_step(optimizer) -> int:
    flat = _flat_optimizer_state(optimizer)
    matches = [value for key, value in flat.items() if key == "step" or key.endswith(".step")]
    if len(matches) != 1:
        raise RuntimeError("optimizer state has no unique root step")
    return int(matches[0].item())


def _cursor(update: int) -> dict:
    return {"completed_updates": update,
            "completed_microbatches": update * ACCUMULATION,
            "next_update": update + 1,
            "next_microbatch": 1}


def _schedule_identity(update: int, total_updates: int) -> dict | None:
    if update >= total_updates:
        return None
    return {"role": "synthetic",
            "id": f"update-{update + 1:04d}-microbatch-0001"}


def _batch_record(update: int, position: int, row: dict) -> dict:
    if type(update) is not int or update < 1 or type(position) is not int \
            or not 1 <= position <= ACCUMULATION:
        raise ValueError("synthetic batch position differs")
    return {
        "role": "synthetic",
        "id": f"update-{update:04d}-microbatch-{position:04d}",
        "update": update,
        "microbatch": (update - 1) * ACCUMULATION + position,
        "position": position, "prompt_sha256": row["prompt_sha256"],
        "prompt_tokens": row["prompt_token_count"],
    }


def _source_logits(mx, model, rows, digit_ids, observe=None,
                   stage_prefix: str = "source_logits") -> list[dict]:
    was_training = bool(getattr(model, "training", True))
    model.eval()
    records = []
    try:
        for row in rows:
            scores = score_choices(mx, model, row["prompt_tokens"],
                                    digit_ids[:row["candidate_count"]])
            mx.eval(scores)
            values = [float(value) for value in scores.tolist()]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError("source-only logits are nonfinite")
            records.append({"prompt_sha256": row["prompt_sha256"],
                            "prompt_token_count": row["prompt_token_count"],
                            "candidate_count": row["candidate_count"],
                            "logits": values, "argmax": values.index(max(values))})
            if observe is not None:
                observe(f"{stage_prefix}_{row['prompt_token_count']}")
    finally:
        if was_training:
            model.train()
    return records


def _state_record(mx, model, optimizer, update_record, rows, digit_ids,
                  observe=None, stage_prefix: str = "source_logits") -> dict:
    adapter = require_trainable_contract(model)
    optimizer_flat = _flat_optimizer_state(optimizer)
    mx.eval(adapter, optimizer.state)
    return {
        "adapter_tensor_hashes": tensor_hashes(adapter),
        "adapter_tensor_hash_domain": "runtime mlx dtype + shape + raw bytes",
        "optimizer_tensor_hashes": tensor_hashes(optimizer_flat),
        "optimizer_tensor_hash_domain": "runtime mlx dtype + shape + raw bytes",
        "optimizer_step": _optimizer_step(optimizer),
        "cursor": _cursor(update_record["update"]),
        "microbatch_losses": update_record["microbatch_losses"],
        "mean_loss": update_record["mean_loss"],
        "gradient_norm_before_clip": update_record["gradient_norm_before_clip"],
        "source_only_logits": _source_logits(
            mx, model, rows, digit_ids, observe, stage_prefix),
        "model_mode_restored_to_train": bool(getattr(model, "training", True)),
    }


def _new_model(mx):
    from mlx_lm.utils import load

    model, tokenizer = load(str(v1.MODEL_DIR), tokenizer_config={"trust_remote_code": False})
    if not bool(model.args.tie_word_embeddings):
        raise RuntimeError("pinned base unexpectedly has an untied output head")
    rows, digit_ids = v1.exact_prompts(tokenizer)
    initial = configure_fresh_lora(mx, model)
    initial_hashes = tensor_hashes(initial)
    optimizer = create_optimizer()
    model.train()
    return model, tokenizer, optimizer, rows, digit_ids, initial_hashes


def _make_value_and_grad(mx, nn, model, digit_ids):
    def loss_fn(active_model, row):
        scores = score_choices(mx, active_model, row["prompt_tokens"],
                                digit_ids[:row["candidate_count"]])
        return mlx_log_mass_loss(mx, scores, row["allowed_gold_positions"])

    return loss_fn, nn.value_and_grad(model, loss_fn)

def _microbatch(mx, model, value_and_grad, row, counters):
    loss, gradients = value_and_grad(model, row)
    materialize_microbatch(mx, model, loss, gradients, counters=counters)
    return read_numeric_scalar(loss, counters=counters), gradients


def _publish_update(*, mx, model, optimizer, value_and_grad, rows, update,
                    total_updates: int, segment, observe: Callable,
                    update_started: float, segment_started: float,
                    counters: dict[str, int]) -> tuple[dict, object]:
    accumulated = None
    losses = []
    batch_records = []
    for position, row in enumerate(rows, 1):
        loss, gradients = _microbatch(mx, model, value_and_grad, row, counters)
        accumulated = add_materialized_gradients(mx, accumulated, gradients)
        observe(f"update_{update}_microbatch_{position}", update_started)
        losses.append(loss)
        batch_records.append(_batch_record(update, position, row))
    clipped, norm = average_and_clip_gradients(
        mx, accumulated, ACCUMULATION, counters=counters)
    observe(f"update_{update}_averaged_clipped", update_started)
    apply_materialized_update(mx, model, optimizer, clipped, counters=counters)
    publish_started = time.monotonic()
    result = segment.publish_checkpoint(
        mx=mx, adapter_tensors=require_trainable_contract(model),
        optimizer_state=optimizer.state, update=update,
        completed_microbatches=update * ACCUMULATION,
        next_schedule=_schedule_identity(update, total_updates), batch_records=batch_records,
        losses=losses, adapter_config=ADAPTER_CONFIG,
        optimizer_config=OPTIMIZER_CONFIG, recipe=RECIPE,
        elapsed_seconds=time.monotonic() - segment_started,
        update_elapsed_before_checkpoint=time.monotonic() - update_started,
        resource_callback=lambda stage: observe(
            f"update_{update}_checkpoint_{stage}", update_started),
    )
    checkpoint_seconds = time.monotonic() - publish_started
    observe(f"update_{update}_eligible", update_started)
    elapsed = time.monotonic() - update_started
    summary = {
        "manifest": result["manifest"], "pointer": result["pointer"],
        "path": str(result["path"]), "timings": result["timings"],
        "resource_observations": result["resource_observations"],
        "payload_bytes": result["payload_bytes"],
        "update_seconds": result["update_seconds"],
    }
    record = {
        "update": update, "microbatches": ACCUMULATION,
        "microbatch_losses": losses, "mean_loss": sum(losses) / len(losses),
        "gradient_norm_before_clip": read_numeric_scalar(norm, counters=counters),
        "seconds": elapsed, "checkpoint_seconds": checkpoint_seconds,
        "checkpoint": summary,
        "cursor": _cursor(update),
    }
    return record, result


def _interrupt_partial(mx, model, value_and_grad, rows, count, observe, update_started,
                       counters):
    accumulated = None
    losses = []
    for position, row in enumerate(rows[:count], 1):
        loss, gradients = _microbatch(mx, model, value_and_grad, row, counters)
        accumulated = add_materialized_gradients(mx, accumulated, gradients)
        observe(f"interrupted_update_3_microbatch_{position}", update_started)
        losses.append(loss)
    return {"update": 3, "computed_microbatches": count, "losses": losses,
            "checkpoint_published": False, "durable_cursor_remains": _cursor(2),
            "partial_accumulator_discarded": True}


def _run_identity(make_run_identity, *, logical_run_id, baseline, source_hashes,
                  updates: int):
    schedule = {"updates": updates, "accumulation": ACCUMULATION,
                "shapes": [list(shape) for shape in v1.SHAPES]}
    return make_run_identity(
        logical_run_id=logical_run_id, mode="synthetic",
        baseline_swap_used_bytes=baseline["swap_used_bytes"],
        schedule_sha256=v1.hashlib.sha256(
            json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        microbatches=updates * ACCUMULATION, updates=updates,
        final_update_microbatches=ACCUMULATION,
        source_hashes=source_hashes,
        config_hashes={"plan": FIXED_INPUTS[PLAN], "test_spec": FIXED_INPUTS[TEST_SPEC],
                       "handoff": FIXED_INPUTS[HANDOFF]},
        model_hashes={"SOURCE.json": v1.PINNED_ARTIFACT_SHA256[v1.MODEL_SOURCE],
                      "model.safetensors": v1.MODEL_WEIGHT_SHA256},
        package_versions=EXPECTED_PACKAGES,
        created_at=datetime.now(timezone.utc).isoformat(),
        recipe_extra={"synthetic_only": True},
    )


def _reference_state_without_logits(mx, model, optimizer, record) -> dict:
    adapter = require_trainable_contract(model)
    optimizer_flat = frozen_v2._flat_optimizer_state(optimizer)
    mx.eval(adapter, optimizer.state)
    return {
        "adapter_tensor_hashes": tensor_hashes(adapter),
        "optimizer_tensor_hashes": tensor_hashes(optimizer_flat),
        "optimizer_step": frozen_v2._optimizer_step(optimizer),
        "cursor": record["cursor"],
        "microbatch_losses": record["microbatch_losses"],
        "mean_loss": record["mean_loss"],
        "gradient_norm_before_clip": record["gradient_norm_before_clip"],
    }


def _run_original_reference(*, mx, nn, CheckpointStore, make_run_identity,
                            output, baseline, source_hashes, observe, progress):
    store = CheckpointStore.create(
        output / "reference",
        _run_identity(make_run_identity, logical_run_id="synthetic-reference",
                      baseline=baseline, source_hashes=source_hashes,
                      updates=PARITY_UPDATES))
    segment_started = time.monotonic()

    def original_observe(stage, update_started=None):
        return observe(f"reference_{stage}", update_started)

    with store.segment("segment-0001", resume=False) as segment:
        model, tokenizer, optimizer, rows, digit_ids, initial_hashes = frozen_v2._new_model(mx)
        loss_fn, value_and_grad = frozen_v2._make_value_and_grad(mx, nn, model, digit_ids)
        observe("reference_model_load")
        warm_loss, warm_gradients = frozen_v2._microbatch(
            mx, model, value_and_grad, rows[-1])
        warm_nonzero = any(bool(mx.any(value != 0).item()) for value in
                           require_matching_gradient_tree(model, warm_gradients).values())
        if not warm_nonzero:
            raise RuntimeError("reference warm gradient is entirely zero")
        observe("reference_warm_1024_backward")
        updates, states, update4 = [], [], None
        for update in range(1, PARITY_UPDATES + 1):
            record, _ = frozen_v2._publish_update(
                mx=mx, model=model, optimizer=optimizer, value_and_grad=value_and_grad,
                rows=rows, update=update, total_updates=PARITY_UPDATES,
                segment=segment, observe=original_observe,
                update_started=time.monotonic(), segment_started=segment_started)
            state = _reference_state_without_logits(mx, model, optimizer, record)
            updates.append(record)
            states.append({"update": update, **state})
            v1.write_progress(progress, "reference_optimizer_update", **record)
            if update == REFERENCE_UPDATE:
                update4 = frozen_v2._state_record(
                    mx, model, optimizer, record, rows, digit_ids,
                    original_observe, "update4")
        if update4 is None:
            raise AssertionError("original update-4 state was not captured")
        del model, tokenizer, optimizer, value_and_grad, loss_fn, warm_gradients
        model = tokenizer = optimizer = value_and_grad = loss_fn = None
        gc.collect()
        mx.synchronize()
        observe("reference_model_released")
        close_started = time.monotonic()
        audit = store.audit_snapshot()
        terminal = segment.finish(
            "complete", reserved_terminal_seconds(segment_started, close_started),
            completed_updates=PARITY_UPDATES,
            terminal_close_reserve_seconds=SEGMENT_CLOSE_RESERVE_SECONDS)
    close_elapsed = validate_segment_close(close_started)
    audit = _append_terminal_to_audit(audit, terminal, close_elapsed)
    return {
        "updates": updates, "states": states, "update4_reference": update4,
        "initial_trainable_tensor_hashes": initial_hashes,
        "warm_1024_backward": {"completed": True, "update_applied": False,
                                "loss": warm_loss, "finite_gradients": True,
                                "nonzero_gradient": True},
        "checkpoint_chain": audit,
        "segment_closure": _closure_record(
            "reference", "segment-0001", terminal, close_elapsed),
        "reference_function_identity": {
            "module": "scripts.probe_long_context_lora_checkpoint_v2",
            "producer_sha256": FIXED_INPUTS[FROZEN_V2_PRODUCER],
            "functions": ["_new_model", "_microbatch", "_publish_update", "_state_record"],
            "frozen_model_sha256": FIXED_INPUTS[ROOT / "scripts/long_context_lora_model.py"],
        },
    }


def _run_main(*, mx, nn, store, observe, progress, started, counters):
    segment_started = time.monotonic()
    with store.segment("segment-0001", resume=False) as segment:
        model, tokenizer, optimizer, rows, digit_ids, initial_hashes = _new_model(mx)
        loss_fn, value_and_grad = _make_value_and_grad(mx, nn, model, digit_ids)
        observe("main_model_load")
        warm_loss, warm_gradients = _microbatch(
            mx, model, value_and_grad, rows[-1], counters)
        warm_nonzero = any(
            bool(mx.any(value != 0).item())
            for value in require_matching_gradient_tree(model, warm_gradients).values())
        if not warm_nonzero:
            raise RuntimeError("warm 1024-token gradient is entirely zero")
        warm = {"completed": True, "update_applied": False, "loss": warm_loss,
                "finite_gradients": True, "nonzero_gradient": True}
        del warm_gradients
        observe("warm_1024_backward")

        updates = []
        states = []
        reference = None
        for update in range(1, MAIN_UPDATES + 1):
            update_started = time.monotonic()
            record, _ = _publish_update(
                mx=mx, model=model, optimizer=optimizer,
                value_and_grad=value_and_grad, rows=rows, update=update,
                total_updates=MAIN_UPDATES, segment=segment, observe=observe,
                update_started=update_started, segment_started=segment_started,
                counters=counters)
            updates.append(record)
            if update <= PARITY_UPDATES:
                states.append({"update": update,
                               **_reference_state_without_logits(
                                   mx, model, optimizer, record)})
            v1.write_progress(progress, "main_optimizer_update", **record)
            if update == REFERENCE_UPDATE:
                reference = _state_record(
                    mx, model, optimizer, record, rows, digit_ids, observe,
                    "main_update4_reference")
        if reference is None:
            raise AssertionError("update-4 reference was not captured")

        final_adapter_hashes = tensor_hashes(require_trainable_contract(model))
        model.eval()
        ordinary_checks = []
        final_logits = []
        for row in rows:
            selected = score_choices(
                mx, model, row["prompt_tokens"], digit_ids[:row["candidate_count"]])
            ordinary = ordinary_forward_choice_scores(
                mx, model, row["prompt_tokens"], digit_ids[:row["candidate_count"]])
            mx.eval(selected, ordinary)
            left, right = selected.tolist(), ordinary.tolist()
            delta = max_abs_delta(left, right)
            same_argmax = left.index(max(left)) == right.index(max(right))
            if delta > v1.FULL_FORWARD_TOLERANCE or not same_argmax:
                raise RuntimeError("final-hidden scorer differs from ordinary forward")
            ordinary_checks.append({
                "prompt_sha256": row["prompt_sha256"],
                "prompt_token_count": row["prompt_token_count"],
                "candidate_count": row["candidate_count"],
                "max_absolute_digit_delta": delta, "same_argmax": same_argmax,
                "selected_scorer_logits": left, "ordinary_forward_logits": right,
            })
            final_logits.append({"prompt_sha256": row["prompt_sha256"],
                                 "prompt_token_count": row["prompt_token_count"],
                                 "candidate_count": row["candidate_count"],
                                 "logits": left, "argmax": left.index(max(left))})
            observe(f"main_ordinary_forward_{row['prompt_token_count']}")

        scorer_times = []
        for row in rows:
            for repetition in range(v1.INFERENCE_REPETITIONS):
                timing_started = time.monotonic()
                scores = score_choices(
                    mx, model, row["prompt_tokens"], digit_ids[:row["candidate_count"]])
                mx.eval(scores)
                observe(f"main_inference_{row['prompt_token_count']}_{repetition + 1}")
                scorer_times.append(time.monotonic() - timing_started)

        restored = store.load_eligible(mx)
        del model, tokenizer, optimizer, value_and_grad, loss_fn, scores, selected, ordinary
        model = tokenizer = optimizer = value_and_grad = loss_fn = None
        gc.collect()
        mx.synchronize()
        observe("main_trained_model_released")

        model, tokenizer, optimizer, reload_rows, reload_digits, reload_initial = _new_model(mx)
        if reload_rows != rows or reload_digits != digit_ids or reload_initial != initial_hashes:
            raise RuntimeError("main fresh reload initialization identity differs")
        from scripts.long_context_lora_checkpoint_v2 import restore_model_optimizer
        restored_state = restore_model_optimizer(mx, model, optimizer, restored)
        if restored_state != {"finalization_only": False,
                              "next_update": MAIN_UPDATES + 1, "train_mode": True}:
            raise RuntimeError("main checkpoint restore cursor differs")
        reloaded_hashes = tensor_hashes(require_trainable_contract(model))
        if reloaded_hashes != final_adapter_hashes:
            raise RuntimeError("main checkpoint adapter hashes differ after fresh restore")
        reload_logits = _source_logits(
            mx, model, reload_rows, reload_digits, observe, "main_fresh_reload")
        reload_checks = []
        for expected, actual in zip(final_logits, reload_logits):
            passed, delta = logits_match(expected["logits"], actual["logits"])
            passed = passed and expected["prompt_sha256"] == actual["prompt_sha256"]
            reload_checks.append({"prompt_sha256": actual["prompt_sha256"],
                                  "prompt_token_count": actual["prompt_token_count"],
                                  "candidate_count": actual["candidate_count"],
                                  "expected_logits": expected["logits"],
                                  "actual_logits": actual["logits"],
                                  "expected_argmax": expected["argmax"],
                                  "actual_argmax": actual["argmax"],
                                  "passed": passed, "max_absolute_delta": delta})
        reload_passed = all(item["passed"] for item in reload_checks)

        del model, tokenizer, optimizer, restored, value_and_grad, loss_fn
        model = tokenizer = optimizer = value_and_grad = loss_fn = None
        gc.collect()
        mx.synchronize()
        observe("main_model_released")
        close_started = time.monotonic()
        audit = store.audit_snapshot()
        terminal = segment.finish(
            "complete",
            terminal_segment_seconds=reserved_terminal_seconds(
                segment_started, close_started),
            completed_updates=MAIN_UPDATES,
            terminal_close_reserve_seconds=SEGMENT_CLOSE_RESERVE_SECONDS)
    close_elapsed = validate_segment_close(close_started)
    audit = _append_terminal_to_audit(audit, terminal, close_elapsed)
    return {
        "warm_1024_backward": warm,
        "summary": {
            "updates": updates,
            "states": states,
            "update_timings_seconds": [row["seconds"] for row in updates],
            "checkpoint_timings_seconds": [row["checkpoint_seconds"] for row in updates],
            "initial_trainable_tensor_hashes": initial_hashes,
            "final_trainable_tensor_hashes": final_adapter_hashes,
            "reloaded_trainable_tensor_hashes": reloaded_hashes,
            "tensor_hash_domain": "runtime mlx dtype + shape + raw bytes",
            "update4_reference": reference,
            "ordinary_forward_diagnostic": ordinary_checks,
            "fresh_checkpoint_reload": {"checks": reload_checks,
                                         "atol": LOGIT_ATOL, "rtol": LOGIT_RTOL,
                                         "passed": reload_passed},
            "checkpoint_chain": audit,
        },
        "reference": reference, "scorer_times": scorer_times,
        "initial_hashes": initial_hashes,
        "segment_closure": _closure_record(
            "main", "segment-0001", terminal, close_elapsed),
    }


def _run_restart_branch(*, mx, nn, CheckpointStore, restore_model_optimizer,
                        make_run_identity, output, interrupted_after, baseline,
                        source_hashes, initial_hashes, reference,
                        reference_updates, observe, counters):
    branch_started = time.monotonic()
    branch_root = output / f"restart-after-{interrupted_after}"
    identity = _run_identity(
        make_run_identity, logical_run_id=f"synthetic-restart-{interrupted_after}",
        baseline=baseline, source_hashes=source_hashes, updates=PARITY_UPDATES)
    store = CheckpointStore.create(branch_root, identity)
    prefix_updates = []
    first_segment_started = time.monotonic()
    with store.segment("segment-0001", resume=False) as segment:
        model, tokenizer, optimizer, rows, digit_ids, branch_initial = _new_model(mx)
        if branch_initial != initial_hashes:
            raise RuntimeError("restart branch initialization identity differs")
        loss_fn, value_and_grad = _make_value_and_grad(mx, nn, model, digit_ids)
        observe(f"restart_{interrupted_after}_initial_model_load")
        for update in range(1, CHECKPOINT_UPDATE + 1):
            record, _ = _publish_update(
                mx=mx, model=model, optimizer=optimizer,
                value_and_grad=value_and_grad, rows=rows, update=update,
                total_updates=PARITY_UPDATES, segment=segment, observe=observe,
                update_started=time.monotonic(), segment_started=first_segment_started,
                counters=counters)
            prefix_updates.append(record)
        partial = _interrupt_partial(
            mx, model, value_and_grad, rows, interrupted_after, observe,
            time.monotonic(), counters)
        del model, tokenizer, optimizer, value_and_grad, loss_fn
        model = tokenizer = optimizer = value_and_grad = loss_fn = None
        gc.collect()
        mx.synchronize()
        observe(f"restart_{interrupted_after}_interrupted_model_released")
        first_close_started = time.monotonic()
        first_terminal = segment.finish(
            "failed",
            terminal_segment_seconds=reserved_terminal_seconds(
                first_segment_started, first_close_started),
            completed_updates=CHECKPOINT_UPDATE,
            interrupted_after_microbatches=interrupted_after,
            terminal_close_reserve_seconds=SEGMENT_CLOSE_RESERVE_SECONDS)
    first_close_elapsed = validate_segment_close(first_close_started)

    resumed_store = CheckpointStore.open(branch_root)
    resumed_updates = []
    second_segment_started = time.monotonic()
    with resumed_store.segment(
            "segment-0002", resume=True,
            preflight_metrics=resume_preflight_metrics(v1.system_resources())) as segment:
        restored = resumed_store.load_eligible(mx)
        if restored.manifest["update"] != CHECKPOINT_UPDATE:
            raise RuntimeError("restart branch did not restore checkpoint update 2")
        model, tokenizer, optimizer, rows, digit_ids, branch_initial = _new_model(mx)
        restore_result = restore_model_optimizer(mx, model, optimizer, restored)
        if restore_result != {"finalization_only": False, "next_update": 3,
                              "train_mode": True}:
            raise RuntimeError("restart branch restore cursor differs")
        loss_fn, value_and_grad = _make_value_and_grad(mx, nn, model, digit_ids)
        observe(f"restart_{interrupted_after}_restored_model_load")
        for update in range(CHECKPOINT_UPDATE + 1, PARITY_UPDATES + 1):
            record, _ = _publish_update(
                mx=mx, model=model, optimizer=optimizer,
                value_and_grad=value_and_grad, rows=rows, update=update,
                total_updates=PARITY_UPDATES, segment=segment, observe=observe,
                update_started=time.monotonic(), segment_started=second_segment_started,
                counters=counters)
            resumed_updates.append(record)
        actual = _state_record(
            mx, model, optimizer, resumed_updates[-1], rows, digit_ids, observe,
            f"restart_{interrupted_after}_update4")
        parity = exact_state_match(reference, actual)
        trajectory = exact_trajectory_match(
            reference_updates, [*prefix_updates, *resumed_updates])
        del model, tokenizer, optimizer, value_and_grad, loss_fn, restored
        model = tokenizer = optimizer = value_and_grad = loss_fn = None
        gc.collect()
        mx.synchronize()
        observe(f"restart_{interrupted_after}_model_released")
        second_close_started = time.monotonic()
        audit = resumed_store.audit_snapshot()
        second_terminal = segment.finish(
            "complete",
            terminal_segment_seconds=reserved_terminal_seconds(
                second_segment_started, second_close_started),
            completed_updates=PARITY_UPDATES,
            terminal_close_reserve_seconds=SEGMENT_CLOSE_RESERVE_SECONDS)
    second_close_elapsed = validate_segment_close(second_close_started)
    audit = _append_terminal_to_audit(audit, second_terminal, second_close_elapsed)
    return {
        "interrupted_after_microbatches": interrupted_after,
        "resume_count": 1, "prefix_updates": prefix_updates,
        "partial_update": partial, "resumed_updates": resumed_updates,
        "updates": [*prefix_updates, *resumed_updates],
        "final_state": actual, "parity": parity,
        "trajectory_parity": trajectory,
        "checkpoint_chain": audit,
        "segment_closures": [
            _closure_record(f"restart-after-{interrupted_after}", "segment-0001",
                            first_terminal, first_close_elapsed),
            _closure_record(f"restart-after-{interrupted_after}", "segment-0002",
                            second_terminal, second_close_elapsed),
        ],
        "elapsed_seconds": time.monotonic() - branch_started,
    }


def run(output: Path = DEFAULT_OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"checkpoint resource artifact already exists: {output}")
    output.mkdir(parents=True)
    progress = output / "progress.jsonl"
    report_path = output / "report.json"
    started = time.monotonic()
    observations = []
    counters = {"predicate_host_reads": 0, "numeric_host_reads": 0}
    stage, active_gate = "initialization", "initialization"
    model = tokenizer = optimizer = value_and_grad = loss_fn = mx = observe = None
    report = {
        "schema": "long-context-lora-resource-checkpoint-v3",
        "completed": False, "resource_feasible": False,
        "synthetic_only": True, "real_fit_rows_read": False,
        "labels_read": False, "v1_prediction_cache_read": False,
        "real_training_run": False, "old_adapter_loaded": False,
        "production_modified": False, "limits": SYNTHETIC_LIMITS,
        "synthetic_descriptor": synthetic_descriptor(),
        "resource_observations": observations, "durable_phases": [],
        "last_resource_stage": None, "active_gate": active_gate,
        "system_swap_attribution": False,
    }

    try:
        before, versions = capture_provenance()
        baseline = v1.system_resources()
        report.update({"source_hashes": before, "versions": versions,
                       "system_baseline": baseline})
        v1.write_progress(progress, "provenance_verified", files=len(before))

        import mlx.core as mlx_core
        import mlx.nn as nn
        from scripts.long_context_lora_checkpoint_v2 import (
            CheckpointStore, make_run_identity, restore_model_optimizer,
        )
        mx = mlx_core
        mx.set_memory_limit(SYNTHETIC_LIMITS["mlx_peak_bytes"])
        mx.set_cache_limit(SYNTHETIC_LIMITS["cache_bytes"])

        def observe(name: str, update_started: float | None = None):
            nonlocal stage
            stage = name
            try:
                settled, record = observe_resources(
                    stage=name, mx=mx, started=started, baseline=baseline,
                    limits=SYNTHETIC_LIMITS, system_reader=v1.system_resources,
                    rss_reader=v1.rss_bytes,
                )
            except ResourceObservationError as error:
                observations.append(error.record)
                report["failure_resource_observation"] = error.record
                raise
            observations.append(record)
            report["last_resource_stage"] = name
            if update_started is not None:
                elapsed = time.monotonic() - update_started
                if elapsed > SYNTHETIC_LIMITS["update_seconds"]:
                    raise RuntimeError("optimizer update time bound exceeded")
            return settled

        report["tiny_math_validation"] = v1.validate_tiny_math(mx)
        observe("tiny_math")

        active_gate = "reference_execution"
        report["active_gate"] = active_gate
        reference_result = _run_original_reference(
            mx=mx, nn=nn, CheckpointStore=CheckpointStore,
            make_run_identity=make_run_identity, output=output, baseline=baseline,
            source_hashes=before, observe=observe, progress=progress)
        report["reference"] = reference_result
        report["segment_closures"] = [reference_result["segment_closure"]]
        active_gate = "reference_persistence"
        persist_observe_and_gate(
            output, report_path, report, "reference", reference_result,
            persistence_gate=active_gate,
            observation_stage="reference_phase_persisted",
            validation_gate="reference_persisted",
            observe=observe, validate=lambda: None)

        main_root = output / "main"
        main_identity = _run_identity(
            make_run_identity, logical_run_id="synthetic-main",
            baseline=baseline, source_hashes=before, updates=MAIN_UPDATES)
        main_store = CheckpointStore.create(main_root, main_identity)
        active_gate = "main_execution"
        report["active_gate"] = active_gate
        main_result = _run_main(
            mx=mx, nn=nn, store=main_store, observe=observe,
            progress=progress, started=started, counters=counters)
        report["warm_1024_backward"] = main_result["warm_1024_backward"]
        main_summary = main_result["summary"]
        optimized_reference = main_result["reference"]
        scorer_times = main_result["scorer_times"]
        initial_hashes = main_result["initial_hashes"]
        main_updates = main_summary["updates"]
        report.update({"main": main_summary,
                       "scorer_timings_seconds": scorer_times})
        report["segment_closures"].append(main_result["segment_closure"])
        state_checks = [
            {"update": update,
             "passed": reference_result["states"][update - 1]
                       == main_summary["states"][update - 1]}
            for update in range(1, PARITY_UPDATES + 1)
        ]
        update4_check = exact_state_match(
            reference_result["update4_reference"], optimized_reference)
        report["original_vs_grouped_parity"] = {
            "passed": all(item["passed"] for item in state_checks)
                      and update4_check["passed"],
            "per_update_exact_state": state_checks,
            "update4_logits_and_state": update4_check,
        }

        def validate_main_and_reference() -> None:
            require_main_phase(main_summary)
            if report["original_vs_grouped_parity"]["passed"] is not True:
                raise RuntimeError("original and grouped four-update states differ")

        active_gate = "main_persistence"
        persist_observe_and_gate(
            output, report_path, report, "main", main_summary,
            persistence_gate=active_gate,
            observation_stage="main_phase_persisted",
            validation_gate="main_and_reference_parity",
            observe=observe, validate=validate_main_and_reference)

        branches = []
        for interrupted_after in INTERRUPT_AFTER_MICROBATCHES:
            active_gate = f"restart_{interrupted_after}_execution"
            report["active_gate"] = active_gate
            branch = _run_restart_branch(
                mx=mx, nn=nn, CheckpointStore=CheckpointStore,
                restore_model_optimizer=restore_model_optimizer,
                make_run_identity=make_run_identity, output=output,
                interrupted_after=interrupted_after, baseline=baseline,
                source_hashes=before, initial_hashes=initial_hashes,
                reference=optimized_reference,
                reference_updates=main_updates[:PARITY_UPDATES],
                observe=observe, counters=counters)
            branches.append(branch)
            report["branches"] = list(branches)
            report["segment_closures"].extend(branch["segment_closures"])
            active_gate = f"restart_{interrupted_after}_persistence"
            persist_observe_and_gate(
                output, report_path, report,
                f"restart-{interrupted_after}", branch,
                persistence_gate=active_gate,
                observation_stage=f"restart_{interrupted_after}_phase_persisted",
                validation_gate=f"restart_{interrupted_after}_parity",
                observe=observe,
                validate=lambda branch=branch: require_branch_phase(branch))

        active_gate = "runtime_projection"
        report["active_gate"] = active_gate
        estimate = checkpoint_runtime_estimate(
            [item["seconds"] for item in main_updates], scorer_times)
        report["runtime_estimate"] = estimate
        report["predicate_reads"] = {
            "optimized_observed": dict(counters),
            "optimized_expected": {"predicate_host_reads": 199,
                                   "numeric_host_reads": 167},
            "original_reference_expected_static": {
                "predicate_host_reads": 421, "numeric_host_reads": 21},
            "predicate_scope": "loss+16 gradients per microbatch; norm+16 clipped gradients and 16 adapters per update",
            "numeric_scope": "loss and norm values read separately for reporting",
            "other_host_reads": "state hashes, optimizer step, warm nonzero-gradient, and logits diagnostics excluded",
        }
        projection_payload = {
            "runtime_estimate": estimate,
            "scorer_timings_seconds": scorer_times,
            "predicate_reads": report["predicate_reads"],
        }
        active_gate = "projection_persistence"
        persist_observe_and_gate(
            output, report_path, report, "projection", projection_payload,
            persistence_gate=active_gate,
            observation_stage="projection_phase_persisted",
            validation_gate="runtime_projection",
            observe=observe,
            validate=lambda: require_runtime_projection(estimate))
        if counters != {"predicate_host_reads": 199, "numeric_host_reads": 167}:
            raise RuntimeError("grouped predicate host-read accounting differs")
        if len(report["segment_closures"]) != EXPECTED_CLOSURES:
            raise RuntimeError("segment closure count differs")
        expected_stages = expected_observation_stages()
        if ([item.get("stage") for item in observations] != expected_stages[:-1]
                or len(observations) != EXPECTED_OBSERVATIONS - 1):
            raise RuntimeError("resource observation grammar differs before terminal")
        after, versions_after = capture_provenance()
        if before != after or versions != versions_after:
            raise RuntimeError("pinned sources or package versions changed during probe")
        terminal = observe("terminal")
        if [item.get("stage") for item in observations] != expected_stages:
            raise RuntimeError("terminal resource observation grammar differs")
        report.update({
            "completed": True, "resource_feasible": True,
            "terminal_status": "success", "source_hashes_after": after,
            "versions_after": versions_after, "provenance_unchanged": True,
            "main": main_summary, "branches": branches,
            "active_gate": "terminal",
            "real_fit_launch_allowed_by_runtime_estimate": True,
            "system_terminal": terminal,
            "elapsed_seconds": time.monotonic() - started,
        })
        v1.write_progress(progress, "terminal_success", elapsed_seconds=report["elapsed_seconds"])
        return report
    except BaseException as error:
        active_gate = report.get("active_gate", active_gate)
        failed_at = stage if isinstance(error, ResourceObservationError) else active_gate
        if callable(observe) and "failure_resource_observation" not in report:
            try:
                report["failure_resources"] = observe("failure_terminal")
            except BaseException as observation_error:
                report["failure_observation_error"] = (
                    f"{type(observation_error).__name__}: {observation_error}")
        report.update({"terminal_status": "failed", "failure_stage": failed_at,
                       "active_gate": active_gate,
                       "failure_type": type(error).__name__, "failure": str(error),
                       "elapsed_seconds": time.monotonic() - started,
                       "resource_feasible": False, "completed": False,
                       "real_fit_launch_allowed_by_runtime_estimate": False})
        try:
            v1.write_progress(progress, "terminal_failure", failure_stage=failed_at,
                              failure_type=type(error).__name__, failure=str(error))
        except Exception as progress_error:
            report["failure_progress_error"] = f"{type(progress_error).__name__}: {progress_error}"
        raise
    finally:
        late_failure = None
        model = tokenizer = optimizer = value_and_grad = loss_fn = None
        gc.collect()
        if mx is not None:
            try:
                mx.synchronize()
            except Exception as cleanup_error:
                report["terminal_cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
                if report.get("completed") is True:
                    late_failure = "terminal MLX cleanup failed"
        if "source_hashes" in report and "source_hashes_after" not in report:
            try:
                after, versions_after = capture_provenance()
                report["source_hashes_after"] = after
                report["versions_after"] = versions_after
                report["provenance_unchanged"] = (report["source_hashes"] == after
                                                   and report["versions"] == versions_after)
            except Exception as provenance_error:
                report["after_provenance_failure"] = (
                    f"{type(provenance_error).__name__}: {provenance_error}")
        serialization_started = time.monotonic()
        rendered = json.dumps(report, sort_keys=True, indent=2) + "\n"
        report["report_serialization_seconds"] = time.monotonic() - serialization_started
        if late_failure is not None:
            report.update({"completed": False, "resource_feasible": False,
                           "terminal_status": "failed",
                           "failure_stage": "terminal_cleanup_and_serialization",
                           "failure_type": "RuntimeError", "failure": late_failure,
                           "real_fit_launch_allowed_by_runtime_estimate": False,
                           "elapsed_seconds": time.monotonic() - started})
        commit_final_report(report_path, report, started)
        if late_failure is not None:
            raise RuntimeError(late_failure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({"completed": report["completed"],
                      "resource_feasible": report["resource_feasible"],
                      "runtime_estimate": report["runtime_estimate"],
                      "output": str(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
