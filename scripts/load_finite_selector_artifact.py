"""Model-free validator for a completed finite-candidate selector v3 bundle."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.neural_converter import file_sha256
from hanglish.models.sentence_ranker import REQUIRED_ARTIFACTS
from scripts.train_finite_final_choice_selector import (
    EVALUATION_PARTITION_SHA256, FIT_MANIFESTS, LINEAGE_SHA256,
    PLAN_SHA256, PREFLIGHT_SHA256, RECIPE, RUNTIME_VERSION,
    SCHEDULE_SHA256, WARM_METADATA_SHA256, WARM_WEIGHT_SHA256,
)


TRAINER_SHA256 = "c6bf9046d70b4b6f488bbb64a28e0f2409a0e25549f2d78342366e4a5132b591"
EXPECTED_ELIGIBILITY = {
    "examples": 2791, "parent_correct": 2712, "parent_error": 79,
    "roles": {"challenge_fit": 1444, "natural_fit": 1347},
    "excluded": {"candidate_budget": 13, "no_eligible_positive": 688},
}


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _explicit_hash(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or set(value) - set("0123456789abcdef")):
        raise ValueError(f"explicit {label} SHA-256 is required")


def _valid_hashes(values):
    return (isinstance(values, dict) and bool(values)
            and all(isinstance(name, str) and name
                    and isinstance(digest, str) and len(digest) == 64
                    and not set(digest) - set("0123456789abcdef")
                    for name, digest in values.items()))


def _valid_inference_bounds(metadata):
    return (type(metadata.get("max_length")) is int
            and metadata["max_length"] == RECIPE["max_length"]
            and type(metadata.get("inference_batch_size")) is int
            and metadata["inference_batch_size"] == RECIPE["inference_batch_size"])


def _read_json(path, expected, label):
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected:
        raise ValueError(f"{label} hash differs")
    return raw, json.loads(raw)


def _validate_fit_result(value, *, expected_rows=2791):
    if not isinstance(value, dict):
        raise ValueError("finite selector FIT result shape differs")
    outcomes = value.get("outcomes")
    if (value.get("rows") != expected_rows or type(value.get("correct")) is not int
            or not 0 <= value["correct"] <= expected_rows
            or not isinstance(outcomes, list) or len(outcomes) != expected_rows):
        raise ValueError("finite selector FIT result shape differs")
    identities = []
    for row in outcomes:
        if (not isinstance(row, dict)
                or row.get("role") not in FIT_MANIFESTS
                or not isinstance(row.get("id"), str) or not row["id"]
                or type(row.get("parent_correct")) is not bool
                or type(row.get("chosen_finite_index")) is not int
                or row["chosen_finite_index"] < 0
                or type(row.get("chosen_source_index")) is not int
                or row["chosen_source_index"] < 0
                or type(row.get("correct")) is not bool):
            raise ValueError("finite selector FIT outcome is malformed")
        identities.append((row["role"], row["id"]))
    if (len(identities) != len(set(identities))
            or sum(row["correct"] for row in outcomes) != value["correct"]):
        raise ValueError("finite selector FIT outcome arithmetic differs")
    if expected_rows == 2791 and (
            Counter(row["role"] for row in outcomes)
            != Counter({"natural_fit": 1347, "challenge_fit": 1444})
            or Counter(row["parent_correct"] for row in outcomes)
            != Counter({True: 2712, False: 79})):
        raise ValueError("finite selector FIT outcome role or parent shape differs")
    return [(row["role"], row["id"], row["parent_correct"]) for row in outcomes]


def _validate_schedule(report_path, report):
    path = report_path.parent / "schedule.jsonl"
    raw = path.read_bytes()
    if sha256_bytes(raw) != SCHEDULE_SHA256:
        raise ValueError("finite selector schedule file hash differs")
    rows = [json.loads(line) for line in raw.splitlines() if line]
    if rows != report.get("schedule_ids") or len(rows) != 904:
        raise ValueError("finite selector reported schedule differs from file")
    preserves, repairs = [], []
    for step, row in enumerate(rows, 1):
        examples = row.get("examples") if isinstance(row, dict) else None
        if (row.get("step") != step or not isinstance(examples, list) or len(examples) != 4
                or examples[0].get("parent_correct") is not False
                or any(example.get("parent_correct") is not True for example in examples[1:])
                or any(example.get("role") not in FIT_MANIFESTS
                       or not isinstance(example.get("id"), str) or not example["id"]
                       for example in examples)):
            raise ValueError("finite selector schedule structure differs")
        repairs.append(f"{examples[0]['role']}:{examples[0]['id']}")
        preserves.extend((example["role"], example["id"]) for example in examples[1:])
    repeat_counts = Counter(repairs)
    if (len(preserves) != len(set(preserves)) or len(preserves) != 2712
            or len(repeat_counts) != 79
            or Counter(repeat_counts.values()) != Counter({12: 35, 11: 44})
            or report.get("repair_repeats") != dict(sorted(repeat_counts.items()))):
        raise ValueError("finite selector schedule coverage differs")
    return raw


def load_selector(report_path, explicit_report_sha, explicit_checkpoint_sha,
                  runtime_report):
    """Validate and return a v3 selector bundle without loading any model."""
    _explicit_hash(explicit_report_sha, "selector report")
    _explicit_hash(explicit_checkpoint_sha, "selector checkpoint")
    report_raw, report = _read_json(report_path, explicit_report_sha,
                                    "finite selector report")
    checkpoint = report_path.parent / "checkpoint"
    metadata_raw = (checkpoint / "metadata.json").read_bytes()
    metadata = json.loads(metadata_raw)
    schedule_raw = _validate_schedule(report_path, report)

    partition = report.get("fresh_evaluation_partition")
    resources = (report.get("peak_rss_bytes"), report.get("elapsed_seconds"),
                 report.get("training_seconds"), report.get("mean_loss"),
                 report.get("save_load_score_max_delta"))
    if (report.get("schema") != "finite-candidate-roberta-final-choice-selector-v3"
            or report.get("scope") !=
            "FIT-only finite-candidate RoBERTa selector; no deployment decision"
            or report.get("recipe") != RECIPE
            or report.get("runtime_version") != RUNTIME_VERSION
            or report.get("plan_sha256") != PLAN_SHA256
            or not isinstance(partition, dict)
            or not isinstance(partition.get("path"), str) or not partition["path"]
            or partition != {
                "path": partition.get("path"),
                "sha256": EVALUATION_PARTITION_SHA256, "rows": 1024, "groups": 785}
            or report.get("fit_manifest_sha256") != FIT_MANIFESTS
            or report.get("preflight_sha256") != PREFLIGHT_SHA256
            or report.get("warm_lineage_sha256") != LINEAGE_SHA256
            or report.get("warm_checkpoint_metadata_sha256") != WARM_METADATA_SHA256
            or report.get("warm_checkpoint_artifact_hashes", {}).get("model.safetensors")
               != WARM_WEIGHT_SHA256
            or report.get("eligibility") != EXPECTED_ELIGIBILITY
            or report.get("schedule_sha256") != SCHEDULE_SHA256
            or report.get("optimizer_steps") != 904
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in resources)
            or not 0 <= resources[0] <= RECIPE["rss_limit_bytes"]
            or not 0 < resources[1] <= RECIPE["whole_run_seconds_limit"]
            or not 0 < resources[2] <= RECIPE["whole_run_seconds_limit"]
            or resources[3] < 0 or not 0 <= resources[4] <= 1e-4
            or report.get("checkpoint_sha256") != explicit_checkpoint_sha
            or file_sha256(checkpoint / "model.safetensors") != explicit_checkpoint_sha
            or report.get("finite_candidate_view") is not True
            or report.get("model_quality_gate_passed") is not False
            or report.get("evaluation_cases_read") is not False
            or report.get("development_data_read") is not False
            or report.get("final_data_read") is not False
            or report.get("eligible_for_deployment") is not False
            or report.get("deployed") is not False):
        raise ValueError("completed finite selector v3 report contract differs")
    captured = report.get("captured_production_fit")
    if (not isinstance(captured, dict) or captured.get("all_rows") != 3492
            or type(captured.get("all_top1")) is not int
            or not 0 <= captured["all_top1"] <= 3492
            or captured.get("eligible_rows") != 2791
            or captured.get("eligible_top1") != 2712
            or captured.get("source") !=
            "captured production parent responses; not warm RoBERTa scores"):
        raise ValueError("captured production FIT summary differs")
    warm_ids = _validate_fit_result(report.get("warm_roberta_fit"))
    trained_ids = _validate_fit_result(report.get("trained_roberta_fit"))
    if warm_ids != trained_ids:
        raise ValueError("warm and trained FIT identities differ")
    scheduled = report["schedule_ids"]
    scheduled_preserve = {(example["role"], example["id"])
                          for row in scheduled for example in row["examples"][1:]}
    scheduled_repairs = {(row["examples"][0]["role"], row["examples"][0]["id"])
                         for row in scheduled}
    fit_preserve = {(role, identity) for role, identity, correct in warm_ids if correct}
    fit_repairs = {(role, identity) for role, identity, correct in warm_ids if not correct}
    if scheduled_preserve != fit_preserve or scheduled_repairs != fit_repairs:
        raise ValueError("finite selector schedule and FIT identities differ")

    sources = report.get("source_hashes")
    runtime_sources = runtime_report.get("integrated_runtime_source_hashes")
    weights = runtime_report.get("accepted_weight_hashes")
    if (report.get("source_hashes_after") != sources or not _valid_hashes(sources)
            or runtime_report.get("passed") is not True
            or runtime_report.get("integrated_runtime_version") != RUNTIME_VERSION
            or runtime_report.get("runtime_source_hashes_after") != runtime_sources
            or not _valid_hashes(runtime_sources) or not _valid_hashes(weights)
            or len(weights) != 16
            or any(sources.get(name) != digest
                   for name, digest in {**runtime_sources, **weights}.items())
            or sources.get("scripts/train_finite_final_choice_selector.py")
               != TRAINER_SHA256):
        raise ValueError("finite selector runtime or source lineage differs")
    for name, digest in sources.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if file_sha256(path) != digest:
            raise ValueError(f"finite selector source changed: {name}")

    artifacts = metadata.get("artifact_hashes")
    warm_artifacts = report.get("warm_checkpoint_artifact_hashes")
    expected_warm = {
        name: sources.get(f".omx/artifacts/roberta-context-ranker-v1/checkpoint/{name}")
        for name in REQUIRED_ARTIFACTS
    }
    parent_metadata = metadata.get("parent_metadata")
    if (metadata.get("format_version") != 1
            or metadata.get("experiment_schema") != report["schema"]
            or metadata.get("training") != RECIPE
            or not _valid_inference_bounds(metadata)
            or metadata.get("frozen_source_hashes") != sources
            or metadata.get("parent_checkpoint_sha256") != WARM_WEIGHT_SHA256
            or metadata.get("parent_checkpoint_metadata_sha256") != WARM_METADATA_SHA256
            or metadata.get("parent_runtime_version") != RUNTIME_VERSION
            or not isinstance(parent_metadata, dict)
            or parent_metadata.get("artifact_hashes") != warm_artifacts
            or parent_metadata.get("max_length") != 96
            or metadata.get("fit_captures") != {
                role: {"path": metadata.get("fit_captures", {}).get(role, {}).get("path"),
                       "manifest_sha256": digest}
                for role, digest in FIT_MANIFESTS.items()}
            or metadata.get("finite_plan", {}).get("sha256") != PLAN_SHA256
            or metadata.get("fresh_evaluation_partition") !=
               report["fresh_evaluation_partition"]
            or metadata.get("warm_lineage_sha256") != LINEAGE_SHA256
            or metadata.get("preflight_sha256") != PREFLIGHT_SHA256
            or metadata.get("schedule_sha256") != SCHEDULE_SHA256
            or metadata.get("training_eligibility") != EXPECTED_ELIGIBILITY
            or metadata.get("finite_candidate_view") is not True
            or metadata.get("all_case_weight") != 1.
            or metadata.get("evaluation_cases_read") is not False
            or metadata.get("development_data_read") is not False
            or metadata.get("final_data_read") is not False
            or metadata.get("eligible_for_deployment") is not False
            or metadata.get("deployed") is not False
            or not isinstance(warm_artifacts, dict)
            or set(warm_artifacts) != REQUIRED_ARTIFACTS
            or warm_artifacts != expected_warm
            or warm_artifacts.get("model.safetensors") != WARM_WEIGHT_SHA256
            or not isinstance(artifacts, dict) or set(artifacts) != REQUIRED_ARTIFACTS
            or artifacts.get("model.safetensors") != explicit_checkpoint_sha):
        raise ValueError("finite selector checkpoint metadata differs")
    observed = {}
    for name, digest in artifacts.items():
        observed[name] = file_sha256(checkpoint / name)
        if observed[name] != digest:
            raise ValueError(f"finite selector checkpoint artifact differs: {name}")
    return {
        "report_path": report_path, "report_raw": report_raw, "report": report,
        "metadata_raw": metadata_raw, "metadata": metadata,
        "schedule_raw": schedule_raw, "checkpoint": checkpoint,
        "checkpoint_hashes": observed,
    }


__all__ = ["load_selector"]
