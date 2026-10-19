"""Capture fresh EVAL-v2 parent candidates after the balanced selector freezes."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.core.phonology import romanize_word
from hanglish.models.sentence_ranker import REQUIRED_ARTIFACTS
from scripts.final_choice_capture import capture_case
from scripts.partition_balanced_selector_holdout import select_ids
from scripts.prepare_final_choice_data import (
    capture_to_directory,
    read_runtime_evidence,
    require_resources,
)
from scripts.train_balanced_final_choice_selector import (
    EVALUATION_PARTITION_SHA256,
    EXPECTED_BATCHES,
    EXPECTED_PARENT_CORRECT,
    EXPECTED_PARENT_ERROR,
    FIT_MANIFEST_SHA256,
    PARENT_WEIGHT,
    PLAN_SHA256,
    RECIPE,
    read_evaluation_partition,
)


OUTPUT = ROOT / "data/natural_context/balanced-final-choice-eval-v2"
RUNTIME_EVIDENCE_SHA256 = "77c04334af817e15bc4c5ccc9d27b1d6ab9f9f6a759b4207e752762ae49d3aab"
RUNTIME_VERSION = "neural-1-867f13596ed5"
SELECTOR_REPORT_SHA256 = "f04c61ffc7fc03c5f2215f5a82b6b248c1d236dd969a8cae46863dfccf4005af"
SELECTOR_CHECKPOINT_SHA256 = "bd76d6b2aefd9826a7a34ac06366cfd0fa47a7bff78e3968f0a507bb02731028"
TRAINER_SHA256 = "808fc72567a21a62cb8ead9ef43ecbe1cbe8b377beeaf6bc40d2f73950ce9c9b"
POLICY_SHA256 = "0a0afb0c0f73023f62d300917b05987585e62afeb7dc33a16ae563a8dd518d6e"
CAPTURE_SHA256 = "80bad648b1c9d46a9483fafabd61f3b82736853735189697b3d12c6c500926fa"
BUILDER_SHA256 = "983353a6b8a95aaa5693f506a992e2680f21c31892e0ee8c539a014084ad9335"
PREPARER_V1_SHA256 = "41da4d13418480de47034ba414d7c9f49c6b1473b833f487ece13e87f0eff402"
RAW = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
RAW_SHA256 = "904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b"
PARTITION_V1 = ROOT / ".omx/artifacts/final-choice-partition-v1/manifest.json"
PARTITION_V1_SHA256 = "e997f06558bdb5633446171d35912140fb34b55066c13fa13b668939d6347846"
PLAN = ROOT / ".omx/artifacts/balanced-final-choice-plan-v2.md"
PROFILE = {
    "name": "balanced-final-choice-fresh-eval-parent-capture-v2",
    "rows": 1025,
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 10800,
    "progress_interval": 32,
}
FROZEN_DEPENDENCIES = {
    "scripts/train_balanced_final_choice_selector.py": TRAINER_SHA256,
    "scripts/balanced_final_choice_policy.py": POLICY_SHA256,
    "scripts/final_choice_capture.py": CAPTURE_SHA256,
    "scripts/prepare_final_choice_data.py": PREPARER_V1_SHA256,
    "scripts/partition_balanced_selector_holdout.py": BUILDER_SHA256,
    ".omx/artifacts/balanced-final-choice-plan-v2.md": PLAN_SHA256,
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _path_key(path):
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _valid_hashes(values):
    return (isinstance(values, dict) and bool(values)
            and all(isinstance(name, str) and name
                    and isinstance(digest, str) and len(digest) == 64
                    and not set(digest) - set("0123456789abcdef")
                    for name, digest in values.items()))


def _read_json(path, expected_sha):
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha:
        raise ValueError(f"balanced selector artifact hash differs: {path}")
    return raw, json.loads(raw)


def _read_jsonl(path, expected_sha):
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha:
        raise ValueError(f"balanced selector artifact hash differs: {path}")
    return raw, [json.loads(line) for line in raw.splitlines() if line]


def _schedule_bytes(records):
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
                    for row in records)


def _valid_fit_summary(summary, expected_correct):
    if (not isinstance(summary, dict) or summary.get("rows") != 2473
            or summary.get("correct") != expected_correct
            or not isinstance(summary.get("outcomes"), list)
            or len(summary["outcomes"]) != summary["rows"]):
        return False
    identities = []
    for row in summary["outcomes"]:
        if (not isinstance(row, dict)
                or row.get("role") not in FIT_MANIFEST_SHA256
                or not isinstance(row.get("id"), str) or not row["id"]
                or type(row.get("parent_correct")) is not bool
                or type(row.get("correct")) is not bool
                or type(row.get("chosen_finite_index")) is not int
                or type(row.get("chosen_source_index")) is not int):
            return False
        identities.append((row["role"], row["id"]))
    return (len(identities) == len(set(identities))
            and sum(row["correct"] for row in summary["outcomes"])
            == expected_correct)


def load_balanced_selector_artifact(report_path, explicit_sha, runtime):
    """Validate the frozen v2 direct selector without loading its model."""
    if explicit_sha != SELECTOR_REPORT_SHA256:
        raise ValueError("fresh EVAL requires the frozen successful v2 selector report")
    report_raw, report = _read_json(report_path, explicit_sha)
    checkpoint = report_path.parent / "checkpoint"
    metadata_raw = (checkpoint / "metadata.json").read_bytes()
    metadata = json.loads(metadata_raw)
    schedule_raw = (report_path.parent / "schedule.jsonl").read_bytes()
    schedule = [json.loads(line) for line in schedule_raw.splitlines() if line]
    schedule_ids = report.get("schedule_ids")
    repeats = report.get("parent_error_repeats")
    expected_eligibility = {
        "natural_fit": {"rows": 2048, "eligible_with_positive": 1029,
                        "missing_eligible_positive": 376, "ineligible": 643,
                        "parent_correct": 971, "parent_error": 58},
        "challenge_fit": {"rows": 1444, "eligible_with_positive": 1444,
                          "missing_eligible_positive": 0, "ineligible": 0,
                          "parent_correct": 1444, "parent_error": 0},
    }
    if (report.get("experiment_schema") != "balanced-final-choice-direct-selector-v2"
            or report.get("scope") !=
            "balanced FIT-only direct final-choice selector; no learned gate or deployment decision"
            or report.get("recipe") != RECIPE
            or report.get("plan_sha256") != PLAN_SHA256
            or report.get("fresh_evaluation_partition", {}).get("sha256")
               != EVALUATION_PARTITION_SHA256
            or report["fresh_evaluation_partition"].get("rows") != PROFILE["rows"]
            or report.get("fit_manifest_sha256") != FIT_MANIFEST_SHA256
            or report.get("parent_checkpoint_sha256") != PARENT_WEIGHT
            or report.get("runtime_version") != runtime["integrated_runtime_version"]
            or report.get("source_hashes") != report.get("source_hashes_after")
            or report.get("checkpoint_sha256")
               != file_sha256(checkpoint / "model.safetensors")
            or report.get("checkpoint_sha256") != SELECTOR_CHECKPOINT_SHA256
            or report.get("optimizer_steps") != EXPECTED_BATCHES
            or report.get("parent_correct_once") != EXPECTED_PARENT_CORRECT
            or report.get("parent_error_unique") != EXPECTED_PARENT_ERROR
            or report.get("parent_error_draws") != EXPECTED_BATCHES
            or report.get("eligibility") != expected_eligibility
            or not isinstance(repeats, dict) or len(repeats) != EXPECTED_PARENT_ERROR
            or sum(repeats.values()) != EXPECTED_BATCHES
            or not isinstance(schedule_ids, list) or len(schedule_ids) != EXPECTED_BATCHES
            or schedule != schedule_ids
            or sha256_bytes(schedule_raw) != report.get("schedule_sha256")
            or _schedule_bytes(schedule_ids) != schedule_raw
            or type(report.get("save_load_score_max_delta")) not in (int, float)
            or not math.isfinite(report["save_load_score_max_delta"])
            or report["save_load_score_max_delta"] > 1e-4
            or type(report.get("mean_loss")) not in (int, float)
            or not math.isfinite(report["mean_loss"]) or report["mean_loss"] < 0
            or type(report.get("training_seconds")) not in (int, float)
            or not math.isfinite(report["training_seconds"])
            or report["training_seconds"] <= 0
            or not _valid_fit_summary(report.get("parent_fit"), EXPECTED_PARENT_CORRECT)
            or not _valid_fit_summary(report.get("trained_fit"), 2457)
            or report.get("balanced_direct_selector") is not True
            or report.get("compatible_with_v1_calibrator") is not False
            or report.get("separate_learned_gate") is not False
            or report.get("fresh_evaluation_cases_read") is not False
            or report.get("calibration_data_read") is not False
            or report.get("holdout_data_read") is not False
            or report.get("preservation_data_read") is not False
            or report.get("development_data_read") is not False
            or report.get("final_data_read") is not False
            or report.get("eligible_for_deployment") is not False
            or report.get("deployed") is not False):
        raise ValueError("completed frozen balanced selector report is required")
    parent_outcomes = report["parent_fit"]["outcomes"]
    trained_outcomes = report["trained_fit"]["outcomes"]
    if ([(row["role"], row["id"], row["parent_correct"])
         for row in parent_outcomes]
            != [(row["role"], row["id"], row["parent_correct"])
                for row in trained_outcomes]):
        raise ValueError("balanced selector FIT outcome identities differ")
    preserve_ids, repair_ids = [], []
    for step, record in enumerate(schedule_ids, 1):
        examples = record.get("examples") if isinstance(record, dict) else None
        if (record.get("step") != step or not isinstance(examples, list)
                or not 2 <= len(examples) <= 4
                or examples[0].get("parent_correct") is not False
                or any(example.get("parent_correct") is not True for example in examples[1:])):
            raise ValueError("balanced selector schedule structure differs")
        repair_ids.append(f"{examples[0].get('role')}:{examples[0].get('id')}")
        preserve_ids.extend((example.get("role"), example.get("id"))
                            for example in examples[1:])
    if (len(preserve_ids) != EXPECTED_PARENT_CORRECT
            or len(set(preserve_ids)) != EXPECTED_PARENT_CORRECT
            or len(set(repair_ids)) != EXPECTED_PARENT_ERROR
            or dict(sorted(Counter(repair_ids).items())) != repeats):
        raise ValueError("balanced selector schedule coverage differs")
    sources = report.get("source_hashes")
    weights = runtime.get("accepted_weight_hashes")
    runtime_sources = runtime.get("integrated_runtime_source_hashes")
    if (not _valid_hashes(sources) or not _valid_hashes(weights) or len(weights) != 16
            or runtime.get("integrated_runtime_version") != RUNTIME_VERSION
            or runtime.get("runtime_source_hashes_after") != runtime_sources
            or any(sources.get(name) != digest
                   for name, digest in {**runtime_sources, **weights}.items())
            or sources.get("scripts/train_balanced_final_choice_selector.py")
               != TRAINER_SHA256):
        raise ValueError("balanced selector runtime or source lineage differs")
    if (metadata.get("format_version") != 1
            or metadata.get("experiment_schema") != report["experiment_schema"]
            or metadata.get("training") != RECIPE
            or metadata.get("parent_checkpoint_sha256") != PARENT_WEIGHT
            or metadata.get("parent_runtime_version") != RUNTIME_VERSION
            or metadata.get("frozen_source_hashes") != sources
            or metadata.get("schedule_sha256") != report["schedule_sha256"]
            or metadata.get("balanced_plan", {}).get("sha256") != PLAN_SHA256
            or metadata.get("fresh_evaluation_partition", {}).get("sha256")
               != EVALUATION_PARTITION_SHA256
            or metadata.get("balanced_direct_selector") is not True
            or metadata.get("compatible_with_v1_calibrator") is not False
            or metadata.get("separate_learned_gate") is not False
            or metadata.get("fresh_evaluation_cases_read", False) is not False
            or metadata.get("max_length") != RECIPE["max_length"]
            or metadata.get("inference_batch_size") != RECIPE["inference_batch_size"]
            or metadata.get("calibration_data_read") is not False
            or metadata.get("holdout_data_read") is not False
            or metadata.get("preservation_data_read") is not False
            or metadata.get("development_data_read") is not False
            or metadata.get("final_data_read") is not False
            or metadata.get("deployed") is not False):
        raise ValueError("balanced selector checkpoint metadata differs")
    artifacts = metadata.get("artifact_hashes")
    if not _valid_hashes(artifacts) or set(artifacts) != REQUIRED_ARTIFACTS:
        raise ValueError("balanced selector checkpoint requires all six model files")
    for name, digest in artifacts.items():
        if file_sha256(checkpoint / name) != digest:
            raise ValueError("balanced selector checkpoint artifact differs")
    for name, digest in sources.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if file_sha256(path) != digest:
            raise ValueError(f"balanced selector training source changed: {name}")
    return {"report_path": report_path, "report_raw": report_raw,
            "report": report, "metadata_raw": metadata_raw,
            "metadata": metadata, "schedule_raw": schedule_raw, "checkpoint": checkpoint}


def read_evaluation_sources(partition):
    """Read the pinned TRAIN bytes once, then reproduce metadata-only selection."""
    raw_bytes = RAW.read_bytes()
    partition_bytes = PARTITION_V1.read_bytes()
    plan_bytes = PLAN.read_bytes()
    if (sha256_bytes(raw_bytes) != RAW_SHA256
            or sha256_bytes(partition_bytes) != PARTITION_V1_SHA256
            or sha256_bytes(plan_bytes) != PLAN_SHA256):
        raise ValueError("fresh EVAL source bytes differ")
    raw_rows = [json.loads(line) for line in raw_bytes.splitlines() if line]
    old_partition = json.loads(partition_bytes)
    metadata = [{"id": row.get("id"), "group_id": row.get("group_id")}
                for row in raw_rows]
    reproduced = select_ids(metadata, old_partition["natural"])
    for key in ("eval_ids", "eval_group_ids", "remaining_ids", "remaining_group_ids"):
        if reproduced[key] != partition[key]:
            raise ValueError("fresh EVAL partition does not reproduce from source metadata")
    by_id = {row.get("id"): row for row in raw_rows}
    if len(by_id) != len(raw_rows):
        raise ValueError("fresh EVAL raw source IDs are duplicated")
    sources = []
    for source_id in partition["eval_ids"]:
        row = by_id.get(source_id)
        if (not isinstance(row, dict) or row.get("split") != "train"
                or not isinstance(row.get("text"), str) or not row["text"]
                or not isinstance(row.get("left_context"), str)
                or not isinstance(row.get("group_id"), str) or not row["group_id"]):
            raise ValueError("fresh EVAL source row is malformed")
        sources.append({"id": source_id, "group_id": row["group_id"],
                        "origin": "natural_eval_v2",
                        "input": romanize_word(row["text"]),
                        "context": row["left_context"], "expected": [row["text"]]})
    if ([row["id"] for row in sources] != partition["eval_ids"]
            or {row["group_id"] for row in sources} != set(partition["eval_group_ids"])):
        raise ValueError("fresh EVAL source order or groups differ")
    snapshots = {
        _path_key(RAW): RAW_SHA256,
        _path_key(PARTITION_V1): PARTITION_V1_SHA256,
        _path_key(PLAN): PLAN_SHA256,
    }
    return sources, snapshots


def merge_hashes(target, additions, label):
    for name, digest in additions.items():
        if name in target and target[name] != digest:
            raise ValueError(f"conflicting {label} hash: {name}")
        target[name] = digest


def provenance(selector, partition_path, partition_raw, runtime_path, runtime_raw,
               runner_sha, source_snapshots):
    hashes = dict(selector["report"]["source_hashes"])
    merge_hashes(hashes, source_snapshots, "fresh source")
    additions = {
        _path_key(partition_path): sha256_bytes(partition_raw),
        _path_key(runtime_path): sha256_bytes(runtime_raw),
        _path_key(selector["report_path"]):
            sha256_bytes(selector["report_raw"]),
        _path_key(selector["checkpoint"] / "metadata.json"):
            sha256_bytes(selector["metadata_raw"]),
        _path_key(selector["checkpoint"].parent / "schedule.jsonl"):
            sha256_bytes(selector["schedule_raw"]),
        "scripts/train_balanced_final_choice_selector.py": TRAINER_SHA256,
        "scripts/balanced_final_choice_policy.py": POLICY_SHA256,
        "scripts/final_choice_capture.py": CAPTURE_SHA256,
        "scripts/partition_balanced_selector_holdout.py": BUILDER_SHA256,
        "scripts/prepare_balanced_selector_evaluation.py": runner_sha,
    }
    for name, digest in selector["metadata"]["artifact_hashes"].items():
        additions[_path_key(selector["checkpoint"] / name)] = digest
    merge_hashes(hashes, additions, "fresh evaluation dependency")
    return dict(sorted(hashes.items()))


def verify_hashes(hashes):
    for name, expected in hashes.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if file_sha256(path) != expected:
            raise ValueError(f"fresh evaluation input changed: {name}")


def validate_frozen_dependencies():
    verify_hashes(FROZEN_DEPENDENCIES)


def prepare(args, *, runtime_loader=read_runtime_evidence,
            selector_loader=load_balanced_selector_artifact,
            source_loader=read_evaluation_sources,
            converter_factory=NeuralConverter, capture=capture_case,
            frozen_validator=validate_frozen_dependencies,
            directory_capture=capture_to_directory):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    runner_source = Path(__file__).read_bytes()
    runner_sha = sha256_bytes(runner_source)
    if args.runtime_evidence_sha256 != RUNTIME_EVIDENCE_SHA256:
        raise ValueError("fresh EVAL requires the accepted numeric runtime evidence")
    runtime_raw, runtime = runtime_loader(
        args.runtime_evidence, args.runtime_evidence_sha256
    )
    if (runtime.get("passed") is not True
            or runtime.get("integrated_runtime_version") != RUNTIME_VERSION
            or len(runtime.get("accepted_weight_hashes", {})) != 16):
        raise ValueError("fresh EVAL runtime identity differs")
    partition_raw = args.evaluation_partition.read_bytes()
    if (args.evaluation_partition_sha256 != EVALUATION_PARTITION_SHA256
            or sha256_bytes(partition_raw) != EVALUATION_PARTITION_SHA256):
        raise ValueError("fresh EVAL partition differs")
    partition = read_evaluation_partition(
        args.evaluation_partition, args.evaluation_partition_sha256, []
    )[1]
    selector = selector_loader(
        args.selector_report, args.selector_report_sha256, runtime
    )
    frozen_validator()
    # Raw target-bearing TRAIN rows and the actual converter are touched only
    # after every selector/runtime/partition gate above succeeds.
    sources, source_snapshots = source_loader(partition)
    if len(sources) != PROFILE["rows"]:
        raise ValueError("fresh EVAL source count differs")
    hashes = provenance(
        selector, args.evaluation_partition, partition_raw,
        args.runtime_evidence, runtime_raw, runner_sha, source_snapshots
    )
    merge_hashes(hashes, runtime["integrated_runtime_source_hashes"], "runtime source")
    merge_hashes(hashes, runtime["accepted_weight_hashes"], "runtime model")
    before = dict(sorted(hashes.items()))
    verify_hashes(before)
    require_resources(started, "pre-inference validation")

    torch.set_num_threads(PROFILE["threads"])
    os.environ["OMP_NUM_THREADS"] = str(PROFILE["threads"])
    converter = converter_factory(ranker_kind="hybrid", external_lexicon=True)
    converter.load_models()
    if (converter.version != RUNTIME_VERSION
            or converter.model_artifacts != runtime["accepted_weight_hashes"]):
        raise ValueError("actual parent converter differs from frozen runtime")
    selection_raw = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
        for row in sources
    )
    manifest_base = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "fresh actual parent capture for balanced final-choice EVAL-v2",
        "role": "natural_evaluation_v2", "expected_rows": PROFILE["rows"],
        "evaluation_partition_sha256": sha256_bytes(partition_raw),
        "partition_ordered_ids": list(partition["eval_ids"]),
        "partition_ordered_group_ids": list(partition["eval_group_ids"]),
        "selection_sha256_before_inference": sha256_bytes(selection_raw),
        "train_inputs": {name: {"path": name, "sha256": digest}
                         for name, digest in source_snapshots.items()},
        "runtime_evidence_path": str(args.runtime_evidence),
        "runtime_evidence_sha256": sha256_bytes(runtime_raw),
        "runtime_version": RUNTIME_VERSION,
        "runtime_source_hashes": runtime["integrated_runtime_source_hashes"],
        "model_artifact_hashes": runtime["accepted_weight_hashes"],
        "selector_report_sha256": sha256_bytes(selector["report_raw"]),
        "selector_checkpoint_sha256": selector["report"]["checkpoint_sha256"],
        "selector_inference_run": False, "source_hashes": before,
        "profile": PROFILE,
    }
    manifest = directory_capture(
        args.output, converter, sources, runtime_version=RUNTIME_VERSION,
        model_artifacts=runtime["accepted_weight_hashes"],
        manifest_base=manifest_base, capture=capture, started=started,
    )
    if (manifest.get("completed") is not True
            or manifest.get("rows") != PROFILE["rows"]
            or manifest.get("new_selector_inference_run") is not False
            or manifest.get("development_data_read") is not False
            or manifest.get("final_data_read") is not False
            or manifest.get("deployed") is not False):
        raise RuntimeError("fresh EVAL capture did not complete within its declared scope")
    after = provenance(
        selector, args.evaluation_partition, partition_raw,
        args.runtime_evidence, runtime_raw, runner_sha, source_snapshots
    )
    merge_hashes(after, runtime["integrated_runtime_source_hashes"], "runtime source")
    merge_hashes(after, runtime["accepted_weight_hashes"], "runtime model")
    after = dict(sorted(after.items()))
    verify_hashes(after)
    if (after != before or converter.version != RUNTIME_VERSION
            or converter.model_artifacts != runtime["accepted_weight_hashes"]
            or file_sha256(args.output / "selection.jsonl") != sha256_bytes(selection_raw)
            or file_sha256(Path(__file__)) != runner_sha):
        raise RuntimeError("fresh EVAL sources, runtime, or selection changed")
    manifest["source_hashes_after"] = after
    manifest["selector_artifact_validated_without_inference"] = True
    source_path = args.output / "source.py"
    source_path.write_bytes(runner_source)
    if file_sha256(source_path) != runner_sha:
        raise RuntimeError("fresh EVAL capture source snapshot differs")
    require_resources(started, "completed manifest")
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"rows": manifest["rows"],
                      "diagnostics": manifest["diagnostics"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-report", type=Path, required=True)
    parser.add_argument("--selector-report-sha256", required=True)
    parser.add_argument("--evaluation-partition", type=Path, required=True)
    parser.add_argument("--evaluation-partition-sha256", required=True)
    parser.add_argument("--runtime-evidence", type=Path, required=True)
    parser.add_argument("--runtime-evidence-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    prepare(parser.parse_args())
