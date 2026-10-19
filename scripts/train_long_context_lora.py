"""Train the approved fresh long-context finite-choice LoRA adapter."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.long_context_lora_model import (
    LORA_CONFIG, LORA_LAYERS, SEED, add_materialized_gradients,
    apply_materialized_update, average_and_clip_gradients, configure_fresh_lora,
    create_optimizer, materialize_microbatch, mlx_log_mass_loss,
    require_matching_gradient_tree, require_trainable_contract, save_adapter,
    score_choices, tensor_hashes,
)
from scripts.long_context_lora_resources import ResourceObservationError, observe_resources
from scripts import probe_long_context_lora_resources as resource_v1
from scripts import probe_long_context_lora_resources_v2 as resource_v2


OUTPUT = ROOT / ".omx/artifacts/long-context-lora-train-v1"
DATA_DIR = ROOT / ".omx/artifacts/long-context-lora-data-v1-review2"
DATA_MANIFEST_SHA256 = "cb4558249aa9acefb246abca9275ba8020f39cee27eb9d4143dfb94f266ecd06"
RESOURCE_REPORT = ROOT / ".omx/artifacts/long-context-lora-resource-v2/report.json"
RESOURCE_REPORT_SHA256 = "5adf75f71cd196a00ae086d3b36f655ff6113c65998b114363082bff64f9e7e5"
RESOURCE_PROOF = ROOT / ".omx/artifacts/long-context-lora-work-v1/resource-verification-v2.json"
RESOURCE_PROOF_SHA256 = "88cb17a250acf6855cad896c944835ce25d4c87cce0e2af7a4e5874a3e952874"
RESOURCE_VERIFIER = ROOT / ".omx/artifacts/long-context-lora-work-v1/verify_resource.py"
RESOURCE_VERIFIER_SHA256 = "90ef6da72580fff4eac702ee86df16b7542ae04c8872f5ab276d1bbc543f36d8"
DATA_PROOF = ROOT / ".omx/artifacts/long-context-lora-work-v1/data-verification-final.json"
DATA_PROOF_SHA256 = "5ca4e86255c88e0496ebe53ac5bf19951c247548d27eb62faa2ea9deeafb1194"
DATA_VERIFIER = ROOT / ".omx/artifacts/long-context-lora-work-v1/verify_data.py"
DATA_VERIFIER_SHA256 = "e88f69cbfcbf2ae11631c3e7cf4052ce4374c4051cae90beb1c9be5686baa2e9"
OBSERVER_SHA256 = "b4cce78d89f5622521c4a60f4e95827a2994f95a37741577880c423051850a2a"
MODEL_HELPER_SHA256 = "eb9b6c23ef6f639773e34782f04e1499810a448eee3a688a3385d9ef29c12015"
EXPECTED_ROWS = 2771
EXPECTED_UPDATES = 693
EXPECTED_DIAGNOSTIC_ROWS = 640
ACCUMULATION = 4
TRAIN_LIMITS = {**resource_v1.LIMITS, "whole_seconds": 21600.0}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line]


def canonical_token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def validate_schedule(schedule: list[dict], expected_rows: int = EXPECTED_ROWS,
                      accumulation: int = ACCUMULATION) -> list[list[dict]]:
    if len(schedule) != expected_rows:
        raise ValueError(f"training schedule must contain exactly {expected_rows} rows")
    identities, batches = set(), []
    for index, entry in enumerate(schedule, 1):
        expected_update = (index - 1) // accumulation + 1
        if (not isinstance(entry, dict) or set(entry) != {"id", "microbatch", "role", "update"}
                or entry.get("microbatch") != index
                or entry.get("update") != expected_update
                or entry.get("role") not in {"natural_fit", "challenge_fit"}
                or not isinstance(entry.get("id"), str) or not entry["id"]):
            raise ValueError(f"malformed training schedule entry at microbatch {index}")
        identity = (entry["role"], entry["id"])
        if identity in identities:
            raise ValueError("training schedule contains a duplicate identity")
        identities.add(identity)
        if not batches or batches[-1][0]["update"] != expected_update:
            batches.append([])
        batches[-1].append(entry)
    tail = expected_rows % accumulation or accumulation
    if (len(batches) != math.ceil(expected_rows / accumulation)
            or any(len(batch) != accumulation for batch in batches[:-1])
            or len(batches[-1]) != tail):
        raise ValueError("training schedule accumulation or tail differs")
    return batches


def validate_resource_gate(report: dict, proof: dict) -> None:
    if (report.get("schema") != "long-context-lora-resource-v2"
            or report.get("completed") is not True
            or report.get("resource_feasible") is not True
            or report.get("real_training_launch_allowed_by_runtime_estimate") is not True
            or report.get("real_fit_rows_read") is not False
            or report.get("real_training_run") is not False
            or report.get("source_hashes", {}).get("scripts/long_context_lora_resources.py")
            != OBSERVER_SHA256):
        raise ValueError("reviewed resource report did not pass the fixed gate")
    estimate = report.get("runtime_estimate", {}).get("estimated_training_seconds")
    if type(estimate) not in (int, float) or not math.isfinite(estimate) or estimate > 21600:
        raise ValueError("reviewed resource runtime estimate does not authorize training")
    if (proof.get("passed") is not True
            or proof.get("training_resource_gate_passed") is not True
            or proof.get("report_sha256") != RESOURCE_REPORT_SHA256
            or proof.get("verifier_sha256") != RESOURCE_VERIFIER_SHA256
            or proof.get("updates") != 20 or proof.get("microbatches") != 80
            or proof.get("adapter_tensors") != 16
            or proof.get("adapter_scalars") != 327680
            or proof.get("fresh_reload_parity") is not True
            or proof.get("actual_fit_training_run") is not False
            or proof.get("estimated_training_seconds") != estimate):
        raise ValueError("independent resource verification differs")


def load_and_validate_inputs(data_dir: Path = DATA_DIR) -> dict:
    manifest_path = data_dir / "manifest.json"
    if resource_v1.sha256_file(manifest_path) != DATA_MANIFEST_SHA256:
        raise ValueError("reviewed training data manifest differs")
    manifest = json.loads(manifest_path.read_text())
    data_proof = json.loads(DATA_PROOF.read_text())
    if (resource_v1.sha256_file(DATA_PROOF) != DATA_PROOF_SHA256
            or resource_v1.sha256_file(DATA_VERIFIER) != DATA_VERIFIER_SHA256
            or data_proof != {
                "passed": True, "manifest_sha256": DATA_MANIFEST_SHA256,
                "verifier_sha256": DATA_VERIFIER_SHA256,
                "source_hashes_checked": 108, "training_records": EXPECTED_ROWS,
                "updates": EXPECTED_UPDATES, "diagnostic_rows": EXPECTED_DIAGNOSTIC_ROWS,
                "fresh_rows": 512, "fresh_groups": 387,
                "full_candidate_objects_equal": True, "model_loaded": False,
                "fresh_labels_read": False, "development_final_read": False,
                "boundary": "Exact source/file/record/schedule/candidate accounting; no independent raw-corpus overlap recomputation.",
            }):
        raise ValueError("independent reviewed-data proof differs")
    expected_files = {
        "tokens.jsonl", "prompts.jsonl", "targets.jsonl", "diagnostic.jsonl",
        "schedule.jsonl", "training_examples.jsonl", "fresh_selection.jsonl",
    }
    if (manifest.get("schema") != "long-context-lora-data-v1"
            or manifest.get("completed") is not True
            or manifest.get("model_loaded") is not False
            or manifest.get("training_run") is not False
            or manifest.get("eligible_rows") != EXPECTED_ROWS
            or manifest.get("training_schedule") != {
                "salt": "long-context-lora-train-v1-2026-09-15",
                "microbatches": EXPECTED_ROWS, "updates": EXPECTED_UPDATES,
                "final_update_microbatches": 3,
            }
            or manifest.get("diagnostic", {}).get("rows") != EXPECTED_DIAGNOSTIC_ROWS
            or set(manifest.get("file_hashes", {})) != expected_files
            or manifest.get("source_hashes") != manifest.get("source_hashes_after")):
        raise ValueError("reviewed training data contract differs")
    for name, expected in manifest["file_hashes"].items():
        if resource_v1.sha256_file(data_dir / name) != expected:
            raise ValueError(f"reviewed training data file differs: {name}")
    for raw_path, expected in manifest["source_hashes"].items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        if resource_v1.sha256_file(path) != expected:
            raise ValueError(f"reviewed data source differs: {raw_path}")

    examples = read_jsonl(data_dir / "training_examples.jsonl")
    tokens = read_jsonl(data_dir / "tokens.jsonl")
    targets = read_jsonl(data_dir / "targets.jsonl")
    schedule = read_jsonl(data_dir / "schedule.jsonl")
    diagnostic = read_jsonl(data_dir / "diagnostic.jsonl")
    batches = validate_schedule(schedule)
    if not (len(examples) == len(tokens) == len(targets) == EXPECTED_ROWS
            and len(diagnostic) == EXPECTED_DIAGNOSTIC_ROWS):
        raise ValueError("reviewed data row counts differ")
    joined = {}
    for example, token_row, target in zip(examples, tokens, targets):
        identity = (example.get("role"), example.get("id"))
        if (identity != (token_row.get("role"), token_row.get("id"))
                or identity != (target.get("role"), target.get("id"))
                or identity in joined):
            raise ValueError("training example/token/target identities differ")
        choices = example.get("choices")
        token_values = token_row.get("tokens")
        gold = target.get("allowed_gold_positions")
        if (not isinstance(choices, list) or not 2 <= len(choices) <= 5
                or not isinstance(token_values, list) or not token_values
                or len(token_values) > 1024
                or any(type(value) is not int for value in token_values)
                or token_row.get("tokens_sha256") != canonical_token_digest(token_values)
                or not isinstance(gold, list) or not gold
                or len(set(gold)) != len(gold)
                or any(type(value) is not int or not 0 <= value < len(choices) for value in gold)):
            raise ValueError(f"malformed training payload: {identity}")
        joined[identity] = {**example, "tokens": token_values,
                            "allowed_gold_positions": gold}
    ordered = []
    for entry in schedule:
        identity = (entry["role"], entry["id"])
        if identity not in joined:
            raise ValueError("schedule identity is missing from training payloads")
        ordered.append(joined.pop(identity))
    if joined:
        raise ValueError("training payload contains unscheduled identities")
    return {"manifest": manifest, "records": ordered, "batches": batches,
            "diagnostic_items": diagnostic}


def capture_provenance(data: dict) -> tuple[dict, dict]:
    hashes, versions = resource_v2.capture_provenance()
    fixed = {
        RESOURCE_REPORT: RESOURCE_REPORT_SHA256,
        RESOURCE_PROOF: RESOURCE_PROOF_SHA256,
        RESOURCE_VERIFIER: RESOURCE_VERIFIER_SHA256,
        DATA_PROOF: DATA_PROOF_SHA256,
        DATA_VERIFIER: DATA_VERIFIER_SHA256,
        DATA_DIR / "manifest.json": DATA_MANIFEST_SHA256,
        ROOT / "scripts/long_context_lora_resources.py": OBSERVER_SHA256,
        ROOT / "scripts/long_context_lora_model.py": MODEL_HELPER_SHA256,
    }
    for name, expected in data["manifest"]["file_hashes"].items():
        fixed[DATA_DIR / name] = expected
    for raw_path, expected in data["manifest"]["source_hashes"].items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        key = resource_v1.relative_path(path)
        if key in hashes and hashes[key] != expected:
            raise ValueError(f"conflicting training source pin: {key}")
        if path in fixed and fixed[path] != expected:
            raise ValueError(f"conflicting fixed training pin: {key}")
        fixed[path] = expected
    for path, expected in fixed.items():
        actual = resource_v1.sha256_file(path)
        if actual != expected:
            raise ValueError(f"fixed training input differs: {resource_v1.relative_path(path)}")
        hashes[resource_v1.relative_path(path)] = actual
    evaluation_path = ROOT / "scripts/long_context_lora_evaluation.py"
    for path in (Path(__file__), evaluation_path):
        hashes[resource_v1.relative_path(path)] = resource_v1.sha256_file(path)
    expected_versions = resource_v1.PACKAGE_VERSIONS
    actual_versions = {name: importlib.metadata.version(name) for name in expected_versions}
    if actual_versions != expected_versions or versions != expected_versions:
        raise ValueError("training package versions differ")
    return dict(sorted(hashes.items())), versions


def require_train_gate(parent: dict, trained: dict) -> dict:
    """Require the exposed TRAIN net-gain and preservation contract."""
    for label, result in (("parent", parent), ("trained", trained)):
        metrics = result.get("metrics", {}).get("all", {}) if isinstance(result, dict) else {}
        gate = result.get("gate", {}) if isinstance(result, dict) else {}
        if metrics.get("rows") != EXPECTED_DIAGNOSTIC_ROWS:
            raise ValueError(f"{label} diagnostic metrics do not account all rows")
        if gate.get("candidates_preserved") is not True:
            raise ValueError(f"{label} candidate-object invariant failed")
        if gate.get("raw_preserved") is not True:
            raise ValueError(f"{label} raw-bypass invariant failed")
        if gate.get("zero_top5_losses") is not True:
            raise ValueError(f"{label} top5 invariant failed")
        if gate.get("all_rows_accounted") is not True:
            raise ValueError(f"{label} row-accounting invariant failed")
    parent_correct = parent["metrics"]["all"].get("top1")
    trained_correct = trained["metrics"]["all"].get("top1")
    gain = trained["gate"].get("net_top1_gain")
    if (type(parent_correct) is not int or type(trained_correct) is not int
            or type(gain) is not int
            or trained["metrics"]["all"].get("before_top1") != parent_correct
            or gain != trained_correct - parent_correct):
        raise ValueError("diagnostic first-choice gain is inconsistent")
    passed = gain >= 5 and trained["gate"].get("passed") is True
    return {"passed": passed, "net_gain": gain, "required_net_gain": 5,
            "parent_correct": parent_correct, "trained_correct": trained_correct,
            "corrections": trained["gate"].get("corrections"),
            "regressions": trained["gate"].get("regressions")}


def write_jsonl(path: Path, rows: list[dict]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as stream:
        for row in rows:
            raw = (json.dumps(row, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":")) + "\n").encode()
            stream.write(raw)
            digest.update(raw)
        stream.flush()
    return digest.hexdigest()


def validate_reload_logits(expected: list, actual: list) -> tuple[float, bool]:
    """Validate finite same-shape probe vectors before computing reload parity."""
    if (not expected or len(actual) != len(expected)
            or not all(type(value) in (int, float) and math.isfinite(float(value))
                       for value in [*expected, *actual])):
        raise RuntimeError("adapter probe logits are malformed")
    maximum_delta = max(
        abs(float(left) - float(right)) for left, right in zip(actual, expected))
    close = all(
        abs(float(left) - float(right)) <= 1e-5 + 1e-5 * abs(float(right))
        for left, right in zip(actual, expected))
    return maximum_delta, close


def mark_failed(report: dict, error: BaseException, elapsed_seconds: float) -> None:
    """Make every failure terminal even if it occurs after success assembly."""
    report.update({
        "completed": False, "training_gate_passed": False,
        "terminal_status": "failed", "failure_type": type(error).__name__,
        "failure": str(error), "elapsed_seconds": elapsed_seconds,
    })


def require_whole_time(started: float, now: float | None = None) -> float:
    elapsed = (time.monotonic() if now is None else now) - started
    if elapsed > TRAIN_LIMITS["whole_seconds"]:
        raise RuntimeError("whole-run time bound exceeded during terminal report write")
    return elapsed


def run(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise FileExistsError(f"training artifact already exists: {output}")
    output.mkdir(parents=True)
    progress = output / "progress.jsonl"
    report_path = output / "report.json"
    started = time.monotonic()
    observations = []
    report = {
        "schema": "long-context-lora-train-v1", "completed": False,
        "training_gate_passed": False, "deployed": False,
        "fresh_data_read_for_evaluation": False,
        "development_or_final_read_for_evaluation": False,
        "resource_adapter_loaded": False, "old_adapter_loaded": False,
        "auto_resume": False, "limits": TRAIN_LIMITS,
        "resource_observations": observations,
    }
    model = tokenizer = mx = checkpoint = optimizer = value_and_grad = loss_fn = None
    report_finalized = False
    try:
        data = load_and_validate_inputs()
        resource_report = json.loads(RESOURCE_REPORT.read_text())
        resource_proof = json.loads(RESOURCE_PROOF.read_text())
        validate_resource_gate(resource_report, resource_proof)
        before, versions = capture_provenance(data)
        baseline = resource_v1.system_resources()
        report.update({"source_hashes": before, "versions": versions,
                       "system_baseline": baseline,
                       "training_rows": EXPECTED_ROWS,
                       "planned_updates": EXPECTED_UPDATES})
        resource_v1.write_progress(progress, "inputs_verified",
                                   source_hashes=len(before), rows=EXPECTED_ROWS)

        from scripts.long_context_lora_evaluation import (
            apply_logits, prepare_diagnostic, score_predictions,
        )
        import mlx.core as mlx_core
        import mlx.nn as nn
        from mlx_lm.tuner.utils import load_adapters
        from mlx_lm.utils import load
        mx = mlx_core
        mx.set_memory_limit(TRAIN_LIMITS["mlx_peak_bytes"])
        mx.set_cache_limit(TRAIN_LIMITS["cache_bytes"])

        def checkpoint(stage: str, update_started: float | None = None):
            try:
                settled, record = observe_resources(
                    stage=stage, mx=mx, started=started, baseline=baseline,
                    limits=TRAIN_LIMITS, system_reader=resource_v1.system_resources,
                    rss_reader=resource_v1.rss_bytes)
            except ResourceObservationError as error:
                observations.append(error.record)
                report["failure_resource_observation"] = error.record
                settled = error.record.get("post") or error.record.get("pre")
                if settled is not None:
                    report["failure_resources"] = settled
                resource_v1.write_progress(
                    progress, "resource_observation_failed", observation=error.record)
                raise
            observations.append(record)
            resource_v1.write_progress(progress, "resource_observation", observation=record)
            if (update_started is not None
                    and time.monotonic() - update_started > TRAIN_LIMITS["update_seconds"]):
                record["errors"].append("optimizer update time bound exceeded")
                report["failure_resource_observation"] = record
                report["failure_resources"] = settled
                raise RuntimeError("optimizer update time bound exceeded")
            return settled

        model, tokenizer = load(
            str(resource_v1.MODEL_DIR), tokenizer_config={"trust_remote_code": False})
        if not bool(model.args.tie_word_embeddings):
            raise RuntimeError("pinned base unexpectedly has an untied output head")
        _, digit_ids = resource_v1.exact_prompts(tokenizer)
        for count, expected in data["manifest"]["digit_continuation_ids_by_candidate_count"].items():
            if digit_ids[:int(count)] != expected:
                raise RuntimeError("prepared digit continuation IDs differ")
        diagnostic_records = prepare_diagnostic(data["diagnostic_items"], tokenizer)
        if (len(diagnostic_records) != EXPECTED_DIAGNOSTIC_ROWS
                or any(set(record) & {"expected", "category", "outcome"}
                       for record in diagnostic_records)
                or any(record.get("eligible") is True
                       and record.get("digit_ids") != digit_ids[:len(record["choices"])]
                       for record in diagnostic_records)):
            raise RuntimeError("diagnostic preparation is not source-only or complete")
        checkpoint("fresh_base_loaded")

        def score_diagnostic(active_model, label: str):
            predictions = []
            active_model.eval()
            for index, record in enumerate(diagnostic_records, 1):
                logits = None
                if record.get("eligible") is True:
                    count = len(record["choices"])
                    values = score_choices(
                        mx, active_model, record["tokens"], digit_ids[:count])
                    mx.eval(values)
                    checkpoint(f"{label}_diagnostic_{index}")
                    logits = values.tolist()
                    if len(logits) != count or not all(
                            math.isfinite(float(value)) for value in logits):
                        raise RuntimeError("diagnostic logits are malformed")
                prediction = apply_logits(record, logits)
                predictions.append(prediction)
            return predictions

        parent_predictions = [apply_logits(record, None) for record in diagnostic_records]
        parent_sha = write_jsonl(output / "parent_predictions.jsonl", parent_predictions)
        base_predictions = score_diagnostic(model, "base")
        base_sha = write_jsonl(output / "base_predictions.jsonl", base_predictions)
        resource_v1.write_progress(progress, "base_diagnostic_complete",
                                   rows=len(base_predictions))

        initial = configure_fresh_lora(mx, model)
        initial_hashes = tensor_hashes(initial)
        del initial
        checkpoint("fresh_lora_initialized")
        optimizer = create_optimizer()
        model.train()

        def loss_fn(active_model, record):
            scores = score_choices(
                mx, active_model, record["tokens"],
                digit_ids[:len(record["choices"])])
            return mlx_log_mass_loss(mx, scores, record["allowed_gold_positions"])

        value_and_grad = nn.value_and_grad(model, loss_fn)
        update_records, cursor = [], 0
        for update_index, schedule_batch in enumerate(data["batches"], 1):
            update_started = time.monotonic()
            accumulated, losses, identities = None, [], []
            for schedule_entry in schedule_batch:
                record = data["records"][cursor]
                cursor += 1
                identity = (record["role"], record["id"])
                if identity != (schedule_entry["role"], schedule_entry["id"]):
                    raise RuntimeError("training cursor differs from reviewed schedule")
                loss, gradients = value_and_grad(model, record)
                materialize_microbatch(mx, loss, gradients)
                flat_gradients = require_matching_gradient_tree(model, gradients)
                if not bool(mx.isfinite(loss).item()) or any(
                        not bool(mx.all(mx.isfinite(value)).item())
                        for value in flat_gradients.values()):
                    raise RuntimeError("training loss or gradient is nonfinite")
                accumulated = add_materialized_gradients(mx, accumulated, gradients)
                checkpoint(f"update_{update_index}_microbatch_{len(losses) + 1}",
                           update_started)
                losses.append(float(loss.item()))
                identities.append({"role": identity[0], "id": identity[1]})
            actual_count = len(schedule_batch)
            clipped, norm = average_and_clip_gradients(
                mx, accumulated, actual_count)
            checkpoint(f"update_{update_index}_averaged_clipped", update_started)
            apply_materialized_update(mx, model, optimizer, clipped)
            resources = checkpoint(f"update_{update_index}_updated", update_started)
            elapsed = time.monotonic() - update_started
            if elapsed > TRAIN_LIMITS["update_seconds"]:
                raise RuntimeError("optimizer update time bound exceeded")
            update_record = {
                "update": update_index, "microbatches": actual_count,
                "identities": identities, "microbatch_losses": losses,
                "mean_loss": sum(losses) / actual_count,
                "gradient_norm_before_clip": float(norm.item()),
                "seconds": elapsed, "resources": resources,
            }
            update_records.append(update_record)
            report.update({"completed_microbatches": cursor,
                           "completed_updates": update_index,
                           "updates": update_records})
            resource_v1.write_progress(progress, "optimizer_update", **update_record)
        if cursor != EXPECTED_ROWS or len(update_records) != EXPECTED_UPDATES:
            raise RuntimeError("training did not consume the exact reviewed epoch")

        final = require_trainable_contract(model)
        final_hashes = tensor_hashes(final)
        if initial_hashes == final_hashes:
            raise RuntimeError("trained adapter did not change")
        model.eval()
        probe_indices = (0, 923, 1846, 2770)
        probe_before_reload = []
        for index in probe_indices:
            record = data["records"][index]
            values = score_choices(
                mx, model, record["tokens"], digit_ids[:len(record["choices"])])
            mx.eval(values)
            checkpoint(f"save_probe_{index}")
            logits = values.tolist()
            validate_reload_logits(logits, logits)
            probe_before_reload.append({
                "index": index, "id": record["id"], "role": record["role"],
                "tokens_sha256": canonical_token_digest(record["tokens"]),
                "digit_logits": logits,
            })

        adapter_dir = output / "adapter"
        adapter_dir.mkdir()
        adapter_path = adapter_dir / "adapters.safetensors"
        saved_hashes = save_adapter(mx, model, adapter_path)
        if saved_hashes != final_hashes:
            raise RuntimeError("saved adapter tensors differ from final trainable tensors")
        adapter_config = {
            "fine_tune_type": "lora", "num_layers": LORA_LAYERS,
            "lora_parameters": LORA_CONFIG,
        }
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(adapter_config, sort_keys=True, indent=2) + "\n")
        checkpoint("final_adapter_saved")
        adapter_sha = resource_v1.sha256_file(adapter_path)

        del model, tokenizer, optimizer, value_and_grad, loss_fn, accumulated
        del clipped, final, gradients, flat_gradients, loss, values
        model = tokenizer = optimizer = value_and_grad = loss_fn = None
        gc.collect()
        checkpoint("training_model_released")

        model, tokenizer = load(
            str(resource_v1.MODEL_DIR), tokenizer_config={"trust_remote_code": False})
        model.freeze()
        load_adapters(model, str(adapter_dir))
        reloaded = require_trainable_contract(model)
        reloaded_hashes = tensor_hashes(reloaded)
        if reloaded_hashes != saved_hashes:
            raise RuntimeError("reloaded adapter tensors differ from saved tensors")
        model.eval()
        _, fresh_digit_ids = resource_v1.exact_prompts(tokenizer)
        if fresh_digit_ids != digit_ids:
            raise RuntimeError("fresh tokenizer digit identity differs")
        reload_checks = []
        for expected in probe_before_reload:
            record = data["records"][expected["index"]]
            values = score_choices(
                mx, model, record["tokens"], digit_ids[:len(record["choices"])])
            mx.eval(values)
            checkpoint(f"reload_probe_{expected['index']}")
            actual = values.tolist()
            maximum_delta, close = validate_reload_logits(
                expected["digit_logits"], actual)
            if not close or actual.index(max(actual)) != expected["digit_logits"].index(
                    max(expected["digit_logits"])):
                raise RuntimeError("fresh base plus final adapter probe parity failed")
            reload_checks.append({**expected, "actual_logits": actual,
                                  "max_absolute_delta": maximum_delta})

        trained_predictions = score_diagnostic(model, "trained")
        trained_sha = write_jsonl(output / "trained_predictions.jsonl", trained_predictions)
        resource_v1.write_progress(progress, "trained_diagnostic_complete",
                                   rows=len(trained_predictions))
        del model, tokenizer, reloaded
        model = tokenizer = None
        gc.collect()
        checkpoint("evaluation_model_released")

        parent_metrics = score_predictions(data["diagnostic_items"], parent_predictions)
        base_metrics = score_predictions(data["diagnostic_items"], base_predictions)
        trained_metrics = score_predictions(data["diagnostic_items"], trained_predictions)
        train_gate = require_train_gate(parent_metrics, trained_metrics)
        after, versions_after = capture_provenance(data)
        if before != after or versions != versions_after:
            raise RuntimeError("training sources or package versions changed")
        report.update({
            "terminal_status": "finalizing",
            "source_hashes_after": after, "versions_after": versions_after,
            "seed": SEED, "completed_microbatches": cursor,
            "completed_updates": len(update_records), "updates": update_records,
            "initial_trainable_tensor_hashes": initial_hashes,
            "final_trainable_tensor_hashes": final_hashes,
            "saved_trainable_tensor_hashes": saved_hashes,
            "reloaded_trainable_tensor_hashes": reloaded_hashes,
            "adapter": {
                "sha256": adapter_sha,
                "config_sha256": resource_v1.sha256_file(
                    adapter_dir / "adapter_config.json"),
                "only_16_trainable_tensors": True,
            },
            "reload_checks": reload_checks,
            "prediction_artifacts": {
                "parent_predictions.jsonl": parent_sha,
                "base_predictions.jsonl": base_sha,
                "trained_predictions.jsonl": trained_sha,
            },
            "diagnostic_metrics": {
                "cached_parent": parent_metrics,
                "fresh_base": base_metrics,
                "final_adapter": trained_metrics,
            },
            "train_gate": train_gate,
            "all_diagnostic_labels_consulted_after_model_release": True,
            "elapsed_seconds": time.monotonic() - started,
        })
        # Materialize the full report and release runtime objects before the terminal
        # observer, so their cleanup and serialization costs remain inside the cap.
        report_path.write_text(json.dumps(report, ensure_ascii=False,
                                          sort_keys=True, indent=2) + "\n")
        optimizer = value_and_grad = loss_fn = None
        gc.collect()
        mx.synchronize()
        terminal_resources = checkpoint("terminal_after_cleanup_and_serialization")
        report["system_terminal"] = terminal_resources
        report.update({
            "completed": True, "terminal_status": "success",
            "training_gate_passed": train_gate["passed"],
        })
        resource_v1.write_progress(
            progress, "terminal_success", training_gate_passed=train_gate["passed"])
        report["elapsed_seconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, ensure_ascii=False,
                                          sort_keys=True, indent=2) + "\n")
        require_whole_time(started)
        report_finalized = True
        return report
    except BaseException as error:
        if callable(checkpoint) and "failure_resource_observation" not in report:
            try:
                report["failure_resources"] = checkpoint("failure_terminal")
            except BaseException as observation_error:
                report["failure_observation_error"] = (
                    f"{type(observation_error).__name__}: {observation_error}")
        mark_failed(report, error, time.monotonic() - started)
        resource_v1.write_progress(progress, "terminal_failure",
                                   failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        if not report_finalized:
            model = tokenizer = optimizer = value_and_grad = loss_fn = None
            gc.collect()
            if mx is not None:
                try:
                    mx.synchronize()
                except Exception as error:
                    report["terminal_cleanup_error"] = f"{type(error).__name__}: {error}"
            if "source_hashes" in report and "source_hashes_after" not in report:
                try:
                    after, versions_after = capture_provenance(data)
                    report["source_hashes_after"] = after
                    report["versions_after"] = versions_after
                    report["provenance_unchanged"] = (
                        report["source_hashes"] == after
                        and report["versions"] == versions_after)
                except Exception as provenance_error:
                    report["after_provenance_failure"] = str(provenance_error)
            report_path.write_text(json.dumps(report, ensure_ascii=False,
                                              sort_keys=True, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({"completed": report["completed"],
                      "training_gate_passed": report["training_gate_passed"],
                      "output": str(args.output)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
