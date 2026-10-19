#!/usr/bin/env python3
"""Pure verifier for a completed observed-phonetic training arm."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / ".omx/artifacts/phonetic-supported-data-v1/pairs.jsonl"
DELTA = ROOT / ".omx/artifacts/observed-phonetic-delta-v1/delta.jsonl"
FROZEN_INITIAL = ROOT / ".omx/artifacts/observed-phonetic-control-probe-v1/initial.pt"
CONTROL_PROBE_V3 = ROOT / ".omx/artifacts/observed-phonetic-control-probe-v3"
BASE_SHA256 = "cf3e43f75420c17fbe8e1ec118a29a0f6a71c0e55cedbdbae5d7ef3668c84a84"
DELTA_SHA256 = "bc299116e618ef37c044d8ac0f3061a55e46efdc1b4f1d0c8d9dc1b8ab655e1a"
INITIAL_SHA256 = "f8b7afe557ef256307373e784b470a3bcc4cedbceb399a10dc0eb019dc3974ea"
INITIAL_TENSOR_SHA256 = "e5be11cadf430c22927377056d57facd3b7eed631d64314de6f1ce2671c3ae37"
SEED = 20260921
UPDATES = 3954
BATCH_SIZE = 128
LIMITS = {"whole_seconds": 1800.0, "update_seconds": 60.0,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
CONFIG = {"embedding_dim": 80, "encoder_hidden": 160, "decoder_hidden": 320,
          "dropout": 0.1, "max_roman_length": 48}
RECIPE = {
    "seed": SEED, "updates": UPDATES, "batch_size": BATCH_SIZE,
    "presentations": UPDATES * BATCH_SIZE, "optimizer": "AdamW",
    "learning_rate": 0.002, "weight_decay": 1e-5, "label_smoothing": 0.02,
    "gradient_clip_norm": 1.0, "beam_size": 8, "cpu_threads": 2,
    "interop_threads": 1, "deterministic_algorithms": True, "dtype": "float32",
    "projection_multiplier": 1.3, "projection_margin_seconds": 60.0,
    "probe_updates": 16, "probe_beam_inputs": 128,
    "decoding_execution": "batched-active-beams-v1",
    "inference_operation": "balanced-contiguous-groups-v1", "maximum_group_size": 32,
}


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
    result = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                require(isinstance(value, dict), f"invalid row {number}: {path}")
                result.append(value)
    return result


def cyclic_schedule(size: int, *, seed: int = SEED, updates: int = UPDATES,
                    batch_size: int = BATCH_SIZE) -> list[list[int]]:
    require(type(size) is int and size > 0 and type(updates) is int and updates > 0
            and type(batch_size) is int and batch_size > 0, "schedule arguments invalid")
    rng, result, pending = random.Random(seed), [], []
    while len(result) < updates:
        permutation = list(range(size)); rng.shuffle(permutation); pending.extend(permutation)
        while len(pending) >= batch_size and len(result) < updates:
            result.append(pending[:batch_size]); del pending[:batch_size]
    return result


def batch_cost(records: list[dict], indices: list[int]) -> int:
    selected = [records[index] for index in indices]
    return len(indices) * (max(len(row["roman"]) + 2 for row in selected)
                           + 2 * max(3 * len(row["surface"]) + 1 for row in selected))


def schedule_summary(records: list[dict], schedule: list[list[int]]) -> dict:
    exposure = Counter(index for batch in schedule for index in batch)
    require(set(exposure) == set(range(len(records))), "schedule omits a pair")
    costs = [batch_cost(records, batch) for batch in schedule]
    return {"pairs": len(records), "updates": len(schedule),
            "presentations": sum(map(len, schedule)),
            "schedule_sha256": hashlib.sha256(canonical(schedule)).hexdigest(),
            "exposure_histogram": {str(key): value for key, value in sorted(
                Counter(exposure.values()).items())},
            "min_exposure": min(exposure.values()), "max_exposure": max(exposure.values()),
            "total_padded_cost": sum(costs), "min_batch_cost": min(costs),
            "max_batch_cost": max(costs)}


def finiteness_check_expected(update: int, total: int = UPDATES) -> bool:
    return update == 1 or update % 128 == 0 or update == total


def verify_updates(rows: list[dict], records: list[dict], schedule: list[list[int]], *,
                   batch_size: int = BATCH_SIZE) -> dict:
    require(len(rows) == len(schedule), "update ledger length differs")
    losses, gradients, finiteness = [], [], 0
    required = {"update", "loss", "gradient_norm", "compute_seconds", "padded_cost",
                "pair_indices", "parameter_finiteness_checked"}
    for update, (row, indices) in enumerate(zip(rows, schedule), 1):
        require(set(row) == required and row["update"] == update,
                f"update row schema/order differs: {update}")
        require(row["pair_indices"] == indices and len(indices) == batch_size,
                f"update pair indices differ: {update}")
        require(row["padded_cost"] == batch_cost(records, indices),
                f"update padded cost differs: {update}")
        for field in ("loss", "gradient_norm", "compute_seconds"):
            require(type(row[field]) in (int, float) and math.isfinite(row[field])
                    and row[field] >= 0, f"non-finite update {field}: {update}")
        expected = finiteness_check_expected(update, len(schedule))
        require(row["parameter_finiteness_checked"] is expected,
                f"parameter finiteness flag differs: {update}")
        finiteness += expected; losses.append(row["loss"]); gradients.append(row["gradient_norm"])
    return {"updates": len(rows), "mean_loss": sum(losses) / len(losses),
            "final_loss": losses[-1], "max_gradient_norm": max(gradients),
            "parameter_finiteness_checks": finiteness}


def resource_row_valid(row: object, limits: dict = LIMITS) -> bool:
    return (isinstance(row, dict) and set(row) == {
        "at", "elapsed_s", "operation", "operation_elapsed_s", "rss_bytes", "stage",
        "swap_growth_bytes", "swap_used_bytes"}
        and type(row["elapsed_s"]) in (int, float) and math.isfinite(row["elapsed_s"])
        and 0 <= row["elapsed_s"] <= limits["whole_seconds"]
        and type(row["rss_bytes"]) is int and 0 <= row["rss_bytes"] <= limits["rss_bytes"]
        and type(row["swap_growth_bytes"]) is int
        and row["swap_growth_bytes"] <= limits["swap_growth_bytes"]
        and type(row["swap_used_bytes"]) is int and row["swap_used_bytes"] >= 0)


def verify_resource_trace(rows: list[dict], expected_operations: list[str],
                          final_stage: str, *, final_after_operations: int | None = None,
                          terminal_stages: tuple[str, ...] = ()) -> dict:
    require(rows, "resource trace empty")
    if final_after_operations is None:
        final_after_operations = len(expected_operations)
    baseline_swap = None
    previous, current, starts, ends, final_rows, final_seen = -1.0, None, [], [], [], False
    for row in rows:
        require(resource_row_valid(row), "resource row invalid")
        row_baseline = row["swap_used_bytes"] - row["swap_growth_bytes"]
        require(type(row_baseline) is int and row_baseline >= 0,
                "resource swap baseline invalid")
        if baseline_swap is None:
            baseline_swap = row_baseline
        require(row_baseline == baseline_swap, "resource swap baseline changed")
        datetime.fromisoformat(row["at"])
        require(row["elapsed_s"] >= previous, "resource elapsed time regressed")
        previous = row["elapsed_s"]
        stage, operation = row["stage"], row["operation"]
        if stage == "operation_start":
            require(current is None and isinstance(operation, str),
                    "resource operation start is not atomic")
            current = operation; starts.append(operation)
        elif stage == "operation_end":
            require(current == operation
                    and type(row["operation_elapsed_s"]) in (int, float)
                    and math.isfinite(row["operation_elapsed_s"])
                    and 0 <= row["operation_elapsed_s"] <= LIMITS["update_seconds"],
                    "resource operation end is invalid")
            ends.append(operation); current = None
        elif stage == "watchdog":
            require(operation in (None, current), "watchdog operation pointer differs")
        elif stage == final_stage:
            require(not final_seen and current is None and operation is None
                    and row["operation_elapsed_s"] is None
                    and len(starts) == final_after_operations,
                    "final resource row invalid")
            final_seen = True; final_rows.append(row)
        elif stage in terminal_stages:
            require(final_seen and current is None and operation is None
                    and row["operation_elapsed_s"] is None
                    and len(starts) == len(expected_operations),
                    "post-final terminal resource row invalid")
        else:
            raise ValueError(f"unexpected resource stage: {stage}")
    require(current is None and starts == expected_operations == ends,
            "resource operation sequence differs")
    require(len(final_rows) == 1, "unique final resource row missing")
    return {"rows": len(rows), "operations": len(starts), "final": final_rows[0],
            "baseline_swap_bytes": baseline_swap,
            "peak_rss_bytes": max(row["rss_bytes"] for row in rows),
            "peak_swap_growth_bytes": max(row["swap_growth_bytes"] for row in rows),
            "peak_operation_seconds": max(
                (row["operation_elapsed_s"] for row in rows if row["stage"] == "operation_end"),
                default=0.0)}


def require_same_swap_baseline(main: dict, finalization: dict) -> int:
    """Require both independently finalized watchdog traces to share one baseline."""
    main_baseline = main.get("baseline_swap_bytes")
    finalization_baseline = finalization.get("baseline_swap_bytes")
    require(type(main_baseline) is int and main_baseline >= 0
            and finalization_baseline == main_baseline,
            "main/finalization swap baseline differs")
    return main_baseline


def require_no_failure_artifacts(run: Path) -> None:
    for name in ("failure.json", "failure-report.json",
                 "resource-watchdog-failure.json"):
        require(not (Path(run) / name).exists(), f"training failure artifact exists: {name}")


def load_records(arm: str) -> list[dict]:
    require(arm in {"control", "augmented"}, "unknown arm")
    require(sha256_file(BASE) == BASE_SHA256 and sha256_file(DELTA) == DELTA_SHA256,
            "training pair source hash differs")
    control = [{"roman": row["roman"], "surface": row["surface"]}
               for row in read_jsonl(BASE) if row.get("split") == "train"]
    delta = [{"roman": row["roman"], "surface": row["surface"]}
             for row in read_jsonl(DELTA)]
    require(len(control) == 51361 and len(delta) == 32984
            and len({(x["roman"], x["surface"]) for x in control + delta}) == 84345,
            "training pair inventory differs")
    return control if arm == "control" else control + delta


def validate_prepared(path: Path, arm: str, records: list[dict], schedule: list[list[int]]) -> dict:
    prepared = read_json(path)
    require(prepared.get("schema") == "observed-phonetic-pair-prepared-v1"
            and prepared.get("completed") is True and prepared.get("recipe") == RECIPE
            and prepared.get("model_config") == CONFIG
            and prepared.get("evaluation_inputs") == 10166
            and prepared.get("contracts") == {
                "model_loaded": False, "model_inference": False, "training_run": False,
                "document_reserve_targets_read": False, "fixed_compute_paired_arms": True},
            "prepared-v3 contract differs")
    pins = prepared.get("source_hashes")
    require(isinstance(pins, dict) and pins, "prepared source pins missing")
    for relative, wanted in pins.items():
        require((ROOT / relative).is_file() and sha256_file(ROOT / relative) == wanted,
                f"prepared source pin differs: {relative}")
    summary = schedule_summary(records, schedule)
    actual_arm = prepared.get("arms", {}).get(arm)
    require(isinstance(actual_arm, dict), "prepared arm missing")
    for key, value in summary.items():
        require(actual_arm.get(key) == value, f"prepared schedule differs: {key}")
    return prepared


def validate_probe(directory: Path, arm: str, prepared: dict,
                   prepared_sha256: str) -> tuple[dict, dict]:
    report_path, completion_path = directory / "report.json", directory / "completion.json"
    report, completion = read_json(report_path), read_json(completion_path)
    require(report.get("schema") == "observed-phonetic-pair-grouped-probe-v1"
            and report.get("completed") is True and report.get("accepted") is True
            and report.get("arm") == arm and report.get("prepared_sha256") == prepared_sha256
            and report.get("recipe") == RECIPE and report.get("limits") == LIMITS
            and report.get("source_hashes") == prepared["source_hashes"]
            and report.get("source_hashes_after") == prepared["source_hashes"]
            and report.get("initial", {}).get("checkpoint_sha256") == INITIAL_SHA256
            and report.get("initial", {}).get("tensor_sha256") == INITIAL_TENSOR_SHA256
            and report.get("initial", {}).get("created_by_this_probe") is False
            and report.get("projection", {}).get("projected_total_seconds", math.inf)
                <= LIMITS["whole_seconds"]
            and report.get("evaluation_targets_read") is False
            and report.get("training_run") is False,
            "accepted grouped probe contract differs")
    initial_path = Path(report["initial"].get("path", ""))
    require(initial_path.resolve() == FROZEN_INITIAL.resolve()
            and sha256_file(initial_path) == INITIAL_SHA256,
            "probe shared initial differs")
    control_probe_sha = report["initial"].get("control_probe_report_sha256")
    if arm == "control":
        require(control_probe_sha is None, "control probe unexpectedly names a shared report")
    else:
        require((CONTROL_PROBE_V3 / "report.json").is_file()
                and (CONTROL_PROBE_V3 / "completion.json").is_file()
                and control_probe_sha == sha256_file(CONTROL_PROBE_V3 / "report.json")
                and read_json(CONTROL_PROBE_V3 / "completion.json").get("report_sha256")
                    == control_probe_sha,
                "augmented probe does not authenticate accepted control probe")
    prediction_path = Path(report.get("probe_predictions_path", ""))
    predictions = read_jsonl(prediction_path) if prediction_path.is_file() else []
    require(prediction_path.is_file()
            and sha256_file(prediction_path) == report.get("probe_predictions_sha256")
            and len(predictions) == report.get("beam_inputs_run")
            and [row.get("input") for row in predictions]
                == [row.get("input") for row in prepared["arms"][arm]["probe_beam_inputs"]],
            "accepted probe prediction closure differs")
    require(completion.get("schema") == "observed-phonetic-pair-completion-v1"
            and completion.get("completed") is True
            and completion.get("report_sha256") == sha256_file(report_path)
            and resource_row_valid(report.get("final_resource"))
            and resource_row_valid(completion.get("resource_after_report")),
            "probe completion/resources differ")
    return report, completion


def verify(run: Path, prepared_path: Path, probe_dir: Path) -> dict:
    run, prepared_path, probe_dir = Path(run), Path(prepared_path), Path(probe_dir)
    proof_path = run / "independent-verification.json"
    require(run.is_dir() and not proof_path.exists(), "training run/proof state invalid")
    require_no_failure_artifacts(run)
    report_path, completion_path = run / "report.json", run / "completion.json"
    update_path, checkpoint_path = run / "updates.jsonl", run / "phonetic_transducer.pt"
    metadata_path = run / "phonetic_transducer.json"
    report, completion, metadata = read_json(report_path), read_json(completion_path), read_json(metadata_path)
    arm = report.get("arm")
    records = load_records(arm)
    schedule = cyclic_schedule(len(records))
    prepared = validate_prepared(prepared_path, arm, records, schedule)
    prepared_sha = sha256_file(prepared_path)
    probe, probe_completion = validate_probe(probe_dir, arm, prepared, prepared_sha)
    updates = read_jsonl(update_path)
    update_evidence = verify_updates(updates, records, schedule)
    require(report.get("schema") == "observed-phonetic-pair-training-v1"
            and report.get("completed") is True and report.get("prepared_sha256") == prepared_sha
            and report.get("probe_report_sha256") == sha256_file(probe_dir / "report.json")
            and report.get("probe_completion_sha256") == sha256_file(probe_dir / "completion.json")
            and report.get("source_hashes") == prepared["source_hashes"]
            and report.get("source_hashes_after") == prepared["source_hashes"]
            and report.get("recipe") == RECIPE and report.get("model_config") == CONFIG
            and report.get("schedule") == prepared["arms"][arm]
            and report.get("updates_completed") == UPDATES
            and report.get("updates_sha256") == sha256_file(update_path)
            and report.get("limits") == LIMITS
            and report.get("evaluation_targets_read") is False
            and report.get("checkpoint_selection") == "final_fixed_update_only"
            and report.get("production_changed") is False,
            "training report contract differs")
    for field in ("mean_loss", "final_loss", "max_gradient_norm",
                  "parameter_finiteness_checks"):
        require(report.get(field) == update_evidence[field], f"training aggregate differs: {field}")
    require(report["parameter_finiteness_checks"] == 32, "finiteness check count differs")
    initial, final = report.get("initial", {}), report.get("final", {})
    require(initial.get("checkpoint_sha256") == INITIAL_SHA256
            and initial.get("tensor_sha256") == INITIAL_TENSOR_SHA256
            and Path(initial.get("path", "")).resolve() == FROZEN_INITIAL.resolve()
            and sha256_file(FROZEN_INITIAL) == INITIAL_SHA256,
            "training initial differs")
    checkpoint_sha = sha256_file(checkpoint_path)
    require(final.get("checkpoint_sha256") == checkpoint_sha
            and final.get("reload_exact") is True
            and final.get("tensor_sha256") == final.get("reload_tensor_sha256")
            and isinstance(final.get("tensor_sha256"), str)
            and len(final["tensor_sha256"]) == 64
            and final["tensor_sha256"] != INITIAL_TENSOR_SHA256,
            "final/reload checkpoint evidence differs")
    require(metadata == {"format_version": 1, "checkpoint_sha256": checkpoint_sha,
            "config": CONFIG, "arm": arm, "initial_checkpoint_sha256": INITIAL_SHA256,
            "initial_tensor_sha256": INITIAL_TENSOR_SHA256,
            "final_tensor_sha256": final["tensor_sha256"]},
            "checkpoint metadata differs")

    main_rows, final_rows = read_jsonl(run / "resources.jsonl"), read_jsonl(
        run / "resources-finalize.jsonl")
    operations = [f"optimizer_update_{update}" for update in range(1, UPDATES + 1)]
    operations += ["save_final_checkpoint", "reload_final_checkpoint"]
    main_resources = verify_resource_trace(main_rows, operations, "final_before_report")
    finalize_resources = verify_resource_trace(
        final_rows, ["write_report", "write_completion"], "after_report",
        final_after_operations=1, terminal_stages=("after_completion",))
    named_after_completion = [row for row in final_rows if row["stage"] == "after_completion"]
    require(len(named_after_completion) == 1 and named_after_completion[0]["operation"] is None,
            "after-completion resource row differs")
    require_same_swap_baseline(main_resources, finalize_resources)
    require(main_resources["final"] == report.get("final_resource"),
            "reported final resource differs")
    require(completion.get("schema") == "observed-phonetic-pair-completion-v1"
            and completion.get("completed") is True
            and completion.get("report_sha256") == sha256_file(report_path)
            and completion.get("resource_after_report") == finalize_resources["final"],
            "training completion differs")

    proof = {
        "schema": "observed-phonetic-training-independent-v1", "verified": True,
        "arm": arm, "report_sha256": sha256_file(report_path),
        "completion_sha256": sha256_file(completion_path),
        "checkpoint_sha256": checkpoint_sha, "checkpoint_metadata_sha256": sha256_file(metadata_path),
        "updates_sha256": sha256_file(update_path), "prepared_sha256": prepared_sha,
        "probe_report_sha256": sha256_file(probe_dir / "report.json"),
        "probe_completion_sha256": sha256_file(probe_dir / "completion.json"),
        "verifier_sha256": sha256_file(Path(__file__)),
        "counts": {"pairs": len(records), "updates": len(updates),
                   "presentations": len(updates) * BATCH_SIZE,
                   "parameter_finiteness_checks": update_evidence["parameter_finiteness_checks"]},
        "schedule": schedule_summary(records, schedule), "training_aggregates": update_evidence,
        "resources": {"main": main_resources, "finalization": finalize_resources},
        "limits": LIMITS,
        "contracts_verified": {
            "prepared_v3_and_source_pins": True, "accepted_probe_and_shared_initial": True,
            "schedule_indices_costs_and_exposure": True, "finite_update_ledger": True,
            "finiteness_flags_exact": True, "final_changed_and_reload_recorded_exact": True,
            "checkpoint_file_and_metadata_hashes": True, "atomic_resources_and_finalization": True,
            "failure_artifact_absent": True, "evaluation_targets_read": False,
            "neural_or_checkpoint_load_performed": False,
        },
        "limitations": [
            "The verifier authenticates recorded tensor hashes and reload equality without loading tensors.",
            "Finite losses, gradients, and scheduled finiteness flags do not independently replay neural updates.",
            "This is training-integrity evidence, not document-reserve performance or production approval.",
        ],
    }
    proof_path.write_bytes(canonical(proof) + b"\n")
    return proof


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--run", type=Path, required=True)
    result.add_argument("--prepared", type=Path, required=True)
    result.add_argument("--probe", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    print(json.dumps(verify(args.run, args.prepared, args.probe), sort_keys=True))
