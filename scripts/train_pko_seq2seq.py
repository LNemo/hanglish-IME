"""Bounded TRAIN-only pko-T5 pilot with resumable, pinned execution.

The resource preflight authorizes a small pilot, not a production replacement.
This program fine-tunes only the frozen FIT partition, measures teacher-forced
loss on the frozen TRAIN inner-validation partition, and leaves DEV/TEST and
the currently deployed converter untouched.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import threading
import time
from datetime import datetime, timezone


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import preflight_pko_seq2seq as preflight


SCHEMA = "pko-t5-small-bounded-pilot-v1"
PREFLIGHT_SCHEMA = "pko-t5-small-resource-preflight-v1"
PREFLIGHT_SOURCE_SHA256 = "89bc4bd2a9875ace9eb010ca443168df4db84ed34772342dd55923e2ab7518f2"
PREFLIGHT_REPORT_SHA256 = "cb56c8dbcebf34ebb1abc062aab71da3a9b6b9fdc354d446a73cb2046ac02c69"
PREFLIGHT_PROOF_SHA256 = "eb4bf40976a902a39342895421aee7484652264bbd04483e8d4b354d259d0781"
PARTITION_SHA256 = "8bc60b9d04f9c82c823fdb6bbecc98401d1c0a947e542b018def44fe6ac30c23"
PARTITION_PROOF_SHA256 = "55c359a7ded8891d5e2df944e1b1d519b224abe2257695ad3d2e5fd70c150c6b"
PLAN = ROOT / ".omx/plans/pko-training-pilot-v1.md"
PLAN_SHA256 = "f9bccc016cba1e9c6e97c9c20e5a18d23c82a829a1b0e83e7274d415a597a32f"
MODEL_DIR = ROOT / "checkpoints/pko-t5-small-pretrained"
PARTITION = ROOT / ".omx/artifacts/pko-training-partition-v1.json"
PARTITION_PROOF = ROOT / ".omx/artifacts/pko-training-partition-verification-v1.json"
SEED = 20260921
FIT_ROWS = 6916
INNER_ROWS = 788
MAX_TOKENS = 512
GRADIENT_ACCUMULATION = 8
CHECKPOINT_INTERVAL = 100
LIMITS = {
    "rss_bytes": 6 * 1024**3,
    "swap_growth_bytes": 512 * 1024**2,
    "whole_seconds": 3600.0,
    "update_seconds": 120.0,
}


class ResourceExceeded(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict, lock: threading.Lock | None = None) -> None:
    manager = lock if lock is not None else contextlib.nullcontext()
    with manager:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def finite(value, *, positive: bool = False) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def _qualification_report(path: Path) -> tuple[Path, dict]:
    path = Path(path)
    report_path = path / "report.json" if path.is_dir() else path
    if not report_path.is_file():
        raise ValueError("qualification must contain report.json")
    return report_path, json.loads(report_path.read_text())


def expected_preflight_pins(model_dir: Path = MODEL_DIR) -> dict[str, str]:
    if sha256_file(preflight.Path(preflight.__file__)) != PREFLIGHT_SOURCE_SHA256:
        raise ValueError("pinned pko preflight source differs")
    if (sha256_file(PARTITION) != PARTITION_SHA256
            or sha256_file(PARTITION_PROOF) != PARTITION_PROOF_SHA256):
        raise ValueError("pinned pko TRAIN partition evidence differs")
    _, model_pins = preflight.verify_model_dir(model_dir)
    expected = {preflight.path_key(path): digest
                for path, digest in preflight.EXPECTED_FILES.items()}
    expected[preflight.path_key(Path(preflight.__file__))] = PREFLIGHT_SOURCE_SHA256
    expected.update(model_pins)
    return dict(sorted(expected.items()))


def validate_qualification(path: Path, model_dir: Path = MODEL_DIR) -> dict:
    """Recompute every preflight fact that authorizes this training run."""
    report_path, report = _qualification_report(path)
    report_sha = sha256_file(report_path)
    proof_path = report_path.parent / "independent-verification.json"
    if (report_sha != PREFLIGHT_REPORT_SHA256 or not proof_path.is_file()
            or sha256_file(proof_path) != PREFLIGHT_PROOF_SHA256):
        raise ValueError("frozen independently verified pko preflight v3 is required")
    proof = json.loads(proof_path.read_text())
    expected_pins = expected_preflight_pins(model_dir)
    schedule = report.get("schedule_estimate")
    training = report.get("training")
    checkpoint = report.get("checkpoint")
    policy = report.get("tokenizer_audit", {}).get("policy")
    scope = report.get("scope")
    model = report.get("model")
    if (report.get("schema") != PREFLIGHT_SCHEMA or report.get("completed") is not True
            or report.get("qualifies") is not True or report.get("pins") != expected_pins):
        raise ValueError("matching completed pko resource preflight is required")
    if (proof.get("verified") is not True
            or proof.get("identity", {}).get("report_sha256") != report_sha
            or proof.get("identity", {}).get("frozen_source_sha256")
            != PREFLIGHT_SOURCE_SHA256
            or proof.get("counts") != {"train_rows": 7704, "fit_rows": 6916,
                                       "inner_validation_rows": 788,
                                       "dev_source_rows": 316,
                                       "selected_fit_rows": 8, "updates": 16,
                                       "generated_train_rows": 8}
            or proof.get("pins") != expected_pins
            or proof.get("contracts", {}).get("test_opened") is not False
            or proof.get("contracts", {}).get("dev_inference_performed") is not False
            or proof.get("decision")
            != "resource preflight qualifies the bounded 5204-microstep pilot"):
        raise ValueError("independent pko preflight verification differs")
    if (not isinstance(model, dict) or model.get("repo_id") != preflight.MODEL_REPO
            or model.get("revision") != preflight.MODEL_REVISION
            or model.get("manifest_sha256") != preflight.MODEL_MANIFEST_SHA256
            or not isinstance(model.get("license_note"), str)):
        raise ValueError("qualified official model identity differs")
    expected_scope = {
        "train_rows": preflight.TRAIN_ROWS, "dev_source_rows": preflight.DEV_ROWS,
        "probe_training_partition": "fit", "probe_fit_rows": FIT_ROWS,
        "dev_file_parsed_including_label_fields": True, "dev_labels_file_read": True,
        "dev_targets_consumed": False, "dev_labels_accessed_or_retained": False,
        "dev_labels_not_passed_to_model": True, "dev_inference_performed": False,
        "test_opened": False, "production_changed": False,
    }
    if scope != expected_scope:
        raise ValueError("qualified data-access scope differs")
    if (not isinstance(policy, dict) or policy.get("official_normalizer") != "NFKC"
            or policy.get("adapted_normalizer") is not None
            or policy.get("adapted_split_special_tokens") is not True
            or policy.get("vocabulary_changed") is not False
            or not all(isinstance(policy.get(key), str) and len(policy[key]) == 64
                       for key in ("official_backend_sha256", "adapted_backend_sha256"))):
        raise ValueError("lossless tokenizer qualification differs")
    if (not isinstance(training, dict) or training.get("precision") != "float32"
            or training.get("device") != "cpu" or training.get("threads") != 2
            or training.get("batch_size") != 1 or training.get("learning_rate") != 1e-4
            or training.get("weight_decay") != .01 or training.get("clip_norm") != 1.0
            or training.get("parameters_changed") is not True
            or len(training.get("updates", [])) != preflight.UPDATES):
        raise ValueError("qualified optimizer probe differs")
    for update in training["updates"]:
        if (not finite(update.get("loss")) or not finite(update.get("gradient_norm"))
                or not finite(update.get("elapsed_s"), positive=True)
                or update["elapsed_s"] > LIMITS["update_seconds"]):
            raise ValueError("qualified optimizer timing differs")
    parity = checkpoint.get("output_parity") if isinstance(checkpoint, dict) else None
    if (not isinstance(checkpoint, dict) or checkpoint.get("format") != "safetensors"
            or checkpoint.get("restored_parameters_exact") is not True
            or checkpoint.get("restored_parameter_sha256")
            != training.get("after_parameter_sha256")
            or not isinstance(parity, dict) or parity.get("logits_allclose") is not True
            or parity.get("argmax_same") is not True
            or parity.get("logit_atol") != 1e-5 or parity.get("logit_rtol") != 1e-6
            or not finite(parity.get("max_logit_delta"))
            or parity["max_logit_delta"] > parity["logit_atol"]
            or not finite(checkpoint.get("elapsed_s"), positive=True)):
        raise ValueError("qualified safe checkpoint evidence differs")
    if (not isinstance(schedule, dict) or schedule.get("fit_rows") != FIT_ROWS
            or schedule.get("heldout_inner_validation_rows") != INNER_ROWS
            or schedule.get("pilot_budget_s") != LIMITS["whole_seconds"]
            or not isinstance(schedule.get("bounded_pilot_updates"), int)
            or isinstance(schedule.get("bounded_pilot_updates"), bool)
            or not 0 < schedule["bounded_pilot_updates"] <= FIT_ROWS
            or not finite(schedule.get("update_p95_s"), positive=True)
            or not finite(schedule.get("estimated_dev_generation_s"))
            or not finite(schedule.get("measured_checkpoint_reload_s"), positive=True)
            or not finite(schedule.get("estimated_bounded_pilot_s"))
            or schedule["estimated_bounded_pilot_s"] > LIMITS["whole_seconds"]):
        raise ValueError("qualified bounded-pilot schedule differs")
    resources = report.get("resources", {}).get("final")
    if (not isinstance(resources, dict)
            or resources.get("rss_bytes", LIMITS["rss_bytes"] + 1) > LIMITS["rss_bytes"]
            or resources.get("swap_growth_bytes", LIMITS["swap_growth_bytes"] + 1)
            > LIMITS["swap_growth_bytes"]
            or resources.get("elapsed_s", LIMITS["whole_seconds"] + 1)
            > preflight.LIMITS["whole_seconds"]):
        raise ValueError("qualified resource evidence differs")
    # Hash the exact terminal authorization only after all replay checks pass.
    return {"path": str(report_path.resolve()), "sha256": report_sha,
            "proof_path": str(proof_path.resolve()),
            "proof_sha256": PREFLIGHT_PROOF_SHA256,
            "report": report, "expected_pins": expected_pins}


def load_train_partition() -> tuple[list[dict], list[dict], dict]:
    """Read only TRAIN files and independently replay the frozen split."""
    for path, digest in preflight.EXPECTED_FILES.items():
        if path == preflight.BASE / "dev.jsonl":
            continue
        if sha256_file(path) != digest:
            raise ValueError(f"pinned TRAIN input differs: {preflight.path_key(path)}")
    rows = []
    for path in (preflight.BASE / "train.jsonl", preflight.EXTRA / "train.jsonl"):
        for raw in path.read_bytes().splitlines():
            row = json.loads(raw)
            if (row.get("split") != "train"
                    or row.get("input_contract") != "roman_sentence_only"
                    or not isinstance(row.get("id"), str)
                    or not isinstance(row.get("input"), str)
                    or not isinstance(row.get("expected"), list)
                    or len(row["expected"]) < 1
                    or not isinstance(row["expected"][0], str)):
                raise ValueError("TRAIN row contract differs")
            rows.append({"id": row["id"], "source": row["input"],
                         "target": row["expected"][0]})
    fit_ids = preflight.fit_partition_ids(rows)
    fit = [row for row in rows if row["id"] in fit_ids]
    inner = [row for row in rows if row["id"] not in fit_ids]
    partition = json.loads(PARTITION.read_text())
    group_splits, group_ids = {}, {}
    for group in partition["groups"]:
        for row_id in group["ids"]:
            group_splits[row_id] = group["split"]
            group_ids[row_id] = group["group_sha256"]
    fit_id_set = {row["id"] for row in fit}
    proof = {
        "rows": len(rows), "fit_rows": len(fit), "inner_validation_rows": len(inner),
        "unique_ids": len({row["id"] for row in rows}),
        "fit_inner_id_overlap": len({row["id"] for row in fit}
                                    & {row["id"] for row in inner}),
        "group_split_mismatches": sum(
            group_splits[row["id"]] != ("fit" if row["id"] in fit_id_set
                                        else "inner_validation")
            for row in rows),
        "fit_group_inner_group_overlap": len(
            {group_ids[row["id"]] for row in fit}
            & {group_ids[row["id"]] for row in inner}),
    }
    if proof != {"rows": 7704, "fit_rows": FIT_ROWS,
                 "inner_validation_rows": INNER_ROWS, "unique_ids": 7704,
                 "fit_inner_id_overlap": 0, "group_split_mismatches": 0,
                 "fit_group_inner_group_overlap": 0}:
        raise ValueError("TRAIN partition replay differs")
    return fit, inner, proof


def shuffled_schedule(fit: list[dict]) -> tuple[list[dict], dict]:
    order = list(fit)
    random.Random(SEED).shuffle(order)
    ids = [row["id"] for row in order]
    proof = {
        "seed": SEED, "rows": len(ids), "unique_ids": len(set(ids)),
        "ordered_ids_sha256": canonical_sha(ids), "contains_targets": False,
    }
    if len(ids) != FIT_ROWS or len(set(ids)) != FIT_ROWS:
        raise ValueError("shuffled FIT schedule differs")
    return order, proof


def choose_microsteps(schedule: dict, *, elapsed_s: float,
                      baseline_validation_s: float) -> dict:
    """Fix the pilot length before learning, reserving all later work."""
    for name, value in (("elapsed_s", elapsed_s),
                        ("baseline_validation_s", baseline_validation_s)):
        if not finite(value):
            raise ValueError(f"invalid {name}")
    microstep_cost = schedule["update_p95_s"] * 1.25
    fixed_reserves = {
        "later_dev_generation_s": schedule["estimated_dev_generation_s"],
        "post_training_inner_validation_s": baseline_validation_s * 1.25,
        "final_checkpoint_reload_s": schedule["measured_checkpoint_reload_s"] * 1.25,
        "fixed_safety_margin_s": 60.0,
    }
    # A recovery save contains the model plus two AdamW moment buffers. Charge
    # three preflight checkpoint durations for every scheduled recovery save.
    recovery_checkpoint_s = schedule["measured_checkpoint_reload_s"] * 3 * 1.25
    cap = min(FIT_ROWS, schedule["bounded_pilot_updates"])
    microsteps = cap
    while microsteps > 0:
        optimizer_updates = math.ceil(microsteps / GRADIENT_ACCUMULATION)
        recovery_saves = optimizer_updates // CHECKPOINT_INTERVAL
        estimate = (elapsed_s + sum(fixed_reserves.values())
                    + microsteps * microstep_cost
                    + recovery_saves * recovery_checkpoint_s)
        if estimate <= LIMITS["whole_seconds"]:
            break
        microsteps -= 1
    by_budget = microsteps
    reserves = dict(fixed_reserves)
    reserves["periodic_recovery_checkpoints_s"] = (
        math.ceil(microsteps / GRADIENT_ACCUMULATION)
        // CHECKPOINT_INTERVAL * recovery_checkpoint_s)
    result = {
        "fixed_before_learning": True, "gradient_accumulation": GRADIENT_ACCUMULATION,
        "qualification_bounded_pilot_updates": schedule["bounded_pilot_updates"],
        "budget_limited_microsteps": by_budget, "microsteps": microsteps,
        "optimizer_updates": math.ceil(microsteps / GRADIENT_ACCUMULATION),
        "last_accumulation_size": (microsteps % GRADIENT_ACCUMULATION
                                   or GRADIENT_ACCUMULATION),
        "estimated_microstep_s": microstep_cost, "reserved": reserves,
        "estimated_total_s": (elapsed_s + sum(reserves.values())
                              + microsteps * microstep_cost),
    }
    if microsteps <= 0 or result["estimated_total_s"] > LIMITS["whole_seconds"]:
        raise ResourceExceeded("measured validation leaves no bounded training budget")
    return result


def accumulation_groups(rows: list[dict], microsteps: int,
                        size: int = GRADIENT_ACCUMULATION) -> list[list[dict]]:
    if (not isinstance(microsteps, int) or isinstance(microsteps, bool)
            or not 0 < microsteps <= len(rows) or size <= 0):
        raise ValueError("invalid fixed microstep schedule")
    selected = rows[:microsteps]
    return [selected[offset:offset + size]
            for offset in range(0, microsteps, size)]


def loss_scale_for_group(group: list[dict]) -> float:
    if not group:
        raise ValueError("empty gradient-accumulation group")
    return 1.0 / len(group)


def tokenizer_encode(tokenizer, text: str):
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True,
                        truncation=False, split_special_tokens=True)
    if encoded["input_ids"].shape[1] > MAX_TOKENS:
        raise RuntimeError("training would require forbidden truncation")
    return encoded


def teacher_forced_loss(model, tokenizer, rows: list[dict], watchdog,
                        *, stage: str) -> dict:
    import torch
    model.eval()
    total, tokens = 0.0, 0
    started = time.monotonic()
    with torch.inference_mode():
        for index, row in enumerate(rows):
            source = tokenizer_encode(tokenizer, row["source"])
            target = tokenizer_encode(tokenizer, row["target"])
            count = int(target["input_ids"].numel())
            with watchdog.operation(f"{stage}_{index + 1}"):
                loss = model(**source, labels=target["input_ids"]).loss
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite inner-validation loss")
            total += float(loss) * count
            tokens += count
    elapsed = time.monotonic() - started
    return {"rows": len(rows), "target_tokens": tokens,
            "token_weighted_loss": total / tokens, "elapsed_s": elapsed}


def _json_random_state(value):
    if isinstance(value, tuple):
        return [_json_random_state(item) for item in value]
    return value


def _tuple_random_state(value):
    if isinstance(value, list):
        return tuple(_tuple_random_state(item) for item in value)
    return value


class ResourceWatchdog:
    """Five-second sampler that writes terminal failure before hard exit."""

    def __init__(self, started: float, baseline_swap: int, output: Path,
                 *, interval_s: float = 5.0):
        self.started, self.baseline_swap = started, baseline_swap
        self.output, self.interval_s = Path(output), interval_s
        self.operation_started = None
        self.operation_name = None
        self.failure = None
        self.stop_event, self.lock = threading.Event(), threading.Lock()
        self.thread, self.old_handler = None, None

    def sample(self, stage: str) -> dict:
        now = time.monotonic()
        observation = preflight.evaluate_observation(
            elapsed_s=now - self.started,
            operation_elapsed_s=(None if self.operation_started is None
                                 else now - self.operation_started),
            rss=preflight.rss_bytes(), swap=preflight.system_swap_bytes(),
            baseline_swap=self.baseline_swap, limits=LIMITS)
        return observation | {"stage": stage, "operation": self.operation_name,
                              "at": datetime.now(timezone.utc).isoformat()}

    def _signal(self, signum, frame):  # pragma: no cover - OS signal delivery
        raise ResourceExceeded(self.failure or "training watchdog interrupted execution")

    def terminal_failure_payload(self, error: BaseException) -> dict:
        return {
            "schema": SCHEMA, "completed": False, "resumable": False,
            "error_type": type(error).__name__, "error": str(error),
            "watchdog_failure": self.failure,
            "cumulative_elapsed_s": time.monotonic() - self.started,
            "original_baseline_swap_bytes": self.baseline_swap,
            "test_opened": False, "dev_opened": False,
            "production_changed": False,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    def _monitor(self):
        while not self.stop_event.wait(self.interval_s):
            try:
                append_jsonl(self.output / "resources.jsonl", self.sample("watchdog"), self.lock)
            except BaseException as error:
                self.failure = f"{type(error).__name__}: {error}"
                write_json(self.output / "failure.json",
                           self.terminal_failure_payload(error))
                os.kill(os.getpid(), signal.SIGUSR1)
                if not self.stop_event.wait(5.0):  # pragma: no cover - hard fail-safe
                    os._exit(124)
                return

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("training watchdog requires the main thread")
        self.old_handler = signal.signal(signal.SIGUSR1, self._signal)
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=max(1.0, self.interval_s + 1.0))
        signal.signal(signal.SIGUSR1, self.old_handler)
        if exc is None and self.failure:
            raise ResourceExceeded(self.failure)
        return False

    @contextlib.contextmanager
    def operation(self, name: str, *, update: bool = False):
        if self.operation_name is not None:
            raise RuntimeError("watchdog operations may not nest")
        self.operation_name = name
        self.operation_started = time.monotonic() if update else None
        append_jsonl(self.output / "resources.jsonl", self.sample("operation_start"), self.lock)
        try:
            yield
            append_jsonl(self.output / "resources.jsonl", self.sample("operation_end"), self.lock)
        finally:
            self.operation_started = None
            self.operation_name = None


def save_recovery(output: Path, model, optimizer, state: dict,
                  identity: dict, tokenizer) -> dict:
    import torch
    existing = output / "recovery.json"
    if existing.is_file():
        slot = 1 - int(json.loads(existing.read_text())["slot"])
    else:
        slot = 0
    checkpoint = output / f"recovery-model-{slot}"
    checkpoint.mkdir(exist_ok=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    checkpoint_files = {}
    for path in sorted(checkpoint.iterdir()):
        if path.is_file():
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            checkpoint_files[path.name] = sha256_file(path)
    optimizer_path = output / f"recovery-optimizer-{slot}.pt"
    temporary = optimizer_path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save({"optimizer": optimizer.state_dict(),
                    "torch_rng_state": torch.get_rng_state()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, optimizer_path)
    persisted_state = dict(state)
    updates_path = output / "updates.jsonl"
    persisted_state["updates_log_bytes"] = (
        updates_path.stat().st_size if updates_path.exists() else 0)
    pointer = {
        "schema": SCHEMA, "resumable": True, "slot": slot,
        "model_dir": checkpoint.name, "optimizer_path": optimizer_path.name,
        "artifacts": {"model_files": checkpoint_files,
                      "optimizer_sha256": sha256_file(optimizer_path)},
        "identity": identity, "state": persisted_state,
        "python_random_state": _json_random_state(random.getstate()),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "recovery.json", pointer)
    return pointer


def load_recovery(output: Path, model, optimizer, identity: dict):
    import torch
    from transformers import T5ForConditionalGeneration
    pointer = json.loads((output / "recovery.json").read_text())
    if pointer.get("schema") != SCHEMA or pointer.get("resumable") is not True:
        raise ValueError("invalid pko recovery pointer")
    if pointer.get("identity") != identity:
        raise ValueError("pko recovery identity differs")
    checkpoint = output / pointer["model_dir"]
    expected_files = pointer.get("artifacts", {}).get("model_files")
    observed_files = ({path.name: sha256_file(path) for path in sorted(checkpoint.iterdir())
                       if path.is_file()} if checkpoint.is_dir() else None)
    if (not isinstance(expected_files, dict) or observed_files != expected_files
            or sha256_file(output / pointer["optimizer_path"])
            != pointer.get("artifacts", {}).get("optimizer_sha256")):
        raise ValueError("pko recovery artifact identity differs")
    restored = T5ForConditionalGeneration.from_pretrained(
        str(checkpoint), local_files_only=True,
        trust_remote_code=False, weights_only=True, torch_dtype="float32")
    model.load_state_dict(restored.state_dict())
    payload = torch.load(output / pointer["optimizer_path"], weights_only=True)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng_state"])
    random.setstate(_tuple_random_state(pointer["python_random_state"]))
    state = pointer["state"]
    log_path = output / "updates.jsonl"
    log_bytes = state.get("updates_log_bytes")
    if (not isinstance(log_bytes, int) or isinstance(log_bytes, bool) or log_bytes < 0
            or not log_path.is_file() or log_path.stat().st_size < log_bytes):
        raise ValueError("pko recovery update-log cursor differs")
    # Discard only updates newer than the last complete checkpoint; their model
    # and optimizer state was deliberately not authorized for recovery.
    with log_path.open("r+b") as stream:
        stream.truncate(log_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    return state


def prepare_output(output: Path, resume: bool) -> Path:
    output = Path(output)
    if resume:
        if (not output.is_dir() or not (output / "recovery.json").is_file()
                or (output / "report.json").exists()):
            raise ValueError("resume requires an unfinished pko pilot recovery pointer")
        failure = output / "failure.json"
        if failure.exists():
            index = 1
            while (output / f"failure-attempt-{index}.json").exists():
                index += 1
            prior = output / f"failure-attempt-{index}.json"
            os.replace(failure, prior)
    else:
        output.mkdir(parents=True, exist_ok=False)
    return output


def run(qualification: Path, output: Path, *, resume: bool = False,
        model_dir: Path = MODEL_DIR) -> dict:
    import torch
    from transformers import T5ForConditionalGeneration, T5TokenizerFast

    authorization = validate_qualification(qualification, model_dir)
    fit, inner, partition_proof = load_train_partition()
    schedule_rows, schedule_proof = shuffled_schedule(fit)
    output = prepare_output(output, resume)
    lock = (output / ".run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("pko pilot output is already owned by a live process")

    source_sha = sha256_file(Path(__file__))
    if sha256_file(PLAN) != PLAN_SHA256:
        raise ValueError("pinned pko training pilot plan differs")
    pins = dict(authorization["expected_pins"])
    pins.update({
        preflight.path_key(Path(__file__)): source_sha,
        preflight.path_key(Path(authorization["path"])): authorization["sha256"],
        preflight.path_key(Path(authorization["proof_path"])): authorization["proof_sha256"],
        preflight.path_key(PARTITION): PARTITION_SHA256,
        preflight.path_key(PARTITION_PROOF): PARTITION_PROOF_SHA256,
        preflight.path_key(PLAN): PLAN_SHA256,
    })
    identity = {
        "architecture": "paust-pko-t5-small-full-finetune",
        "official_model_revision": preflight.MODEL_REVISION,
        "official_weight_sha256": preflight.WEIGHT_SHA256,
        "fresh_start_from_official_weights": True,
        "seed": SEED, "pins": dict(sorted(pins.items())),
        "schedule_sha256": schedule_proof["ordered_ids_sha256"],
        "fit_rows": FIT_ROWS, "inner_validation_rows": INNER_ROWS,
        "optimizer": {"type": "AdamW", "learning_rate": 1e-4,
                      "weight_decay": .01, "clip_norm": 1.0,
                      "gradient_accumulation": GRADIENT_ACCUMULATION},
    }
    schedule_payload = {
        "schema": SCHEMA, "proof": schedule_proof,
        "partition_proof": partition_proof,
        "ids": [row["id"] for row in schedule_rows],
        "captured_before_learning": True, "contains_targets": False,
    }
    schedule_path = output / "schedule.json"
    if resume:
        if (not schedule_path.is_file()
                or json.loads(schedule_path.read_text()) != schedule_payload):
            raise ValueError("captured pko FIT schedule differs on resume")
    else:
        write_json(schedule_path, schedule_payload)
    started = time.monotonic()
    baseline_swap = preflight.system_swap_bytes()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    random.seed(SEED)
    state = {"microsteps": 0, "optimizer_updates": 0, "next_group": 0,
             "training_loss_sum": 0.0, "training_target_tokens": 0,
             "peak_rss_bytes": 0, "training_elapsed_s": 0.0,
             "total_elapsed_s": 0.0, "baseline_swap_bytes": baseline_swap}

    try:
        with ResourceWatchdog(started, baseline_swap, output) as watchdog:
            tokenizer = T5TokenizerFast.from_pretrained(
                str(model_dir), local_files_only=True, trust_remote_code=False)
            preflight.adapt_lossless_tokenizer(tokenizer)
            with watchdog.operation("load_fresh_official_model"):
                model = T5ForConditionalGeneration.from_pretrained(
                    str(model_dir), local_files_only=True, trust_remote_code=False,
                    weights_only=True, torch_dtype="float32")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
            if resume:
                state = load_recovery(output, model, optimizer, identity)
                if (not finite(state.get("total_elapsed_s"))
                        or not isinstance(state.get("baseline_swap_bytes"), int)):
                    raise ValueError("pko recovery resource cursor differs")
                attempts = [json.loads(path.read_text()) for path in
                            sorted(output.glob("failure-attempt-*.json"))]
                historical_elapsed = max(
                    [state["total_elapsed_s"]]
                    + [attempt.get("cumulative_elapsed_s", 0) for attempt in attempts])
                historical_baselines = {
                    attempt.get("original_baseline_swap_bytes") for attempt in attempts
                    if isinstance(attempt.get("original_baseline_swap_bytes"), int)}
                historical_baselines.add(state["baseline_swap_bytes"])
                if len(historical_baselines) != 1:
                    raise ValueError("pko recovery original swap baseline differs")
                # Charge this process and all earlier attempts to one 3,600-second
                # and one swap-growth budget.
                watchdog.started = started - historical_elapsed
                watchdog.baseline_swap = historical_baselines.pop()
                watchdog.sample("resume_budget_replay")
                if (not isinstance(state.get("baseline_validation"), dict)
                        or not isinstance(state.get("fixed_plan"), dict)):
                    raise ValueError("pko recovery lacks the pre-learning fixed plan")
                baseline, fixed = state["baseline_validation"], state["fixed_plan"]
                previous = json.loads((output / "fixed-plan.json").read_text())
                if previous != fixed:
                    raise ValueError("fixed pilot plan differs on resume")
            else:
                # The complete baseline is intentionally measured before any learning.
                baseline = teacher_forced_loss(model, tokenizer, inner, watchdog,
                                               stage="inner_validation_before")
                fixed = choose_microsteps(
                    authorization["report"]["schedule_estimate"],
                    elapsed_s=time.monotonic() - started,
                    baseline_validation_s=baseline["elapsed_s"])
                state["baseline_validation"] = baseline
                state["fixed_plan"] = fixed
                write_json(output / "fixed-plan.json", fixed)
            groups = accumulation_groups(schedule_rows, fixed["microsteps"])

            training_started = time.monotonic()
            for group_index in range(state["next_group"], len(groups)):
                group = groups[group_index]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                group_loss, group_tokens = 0.0, 0
                group_started = time.monotonic()
                with watchdog.operation(f"optimizer_update_{group_index + 1}", update=True):
                    for row in group:
                        source = tokenizer_encode(tokenizer, row["source"])
                        target = tokenizer_encode(tokenizer, row["target"])
                        count = int(target["input_ids"].numel())
                        loss = model(**source, labels=target["input_ids"]).loss
                        if not torch.isfinite(loss):
                            raise RuntimeError("nonfinite pko training loss")
                        (loss * loss_scale_for_group(group)).backward()
                        group_loss += float(loss.detach()) * count
                        group_tokens += count
                    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(gradient):
                        raise RuntimeError("nonfinite pko training gradient")
                    optimizer.step()
                state["microsteps"] += len(group)
                state["optimizer_updates"] += 1
                state["next_group"] = group_index + 1
                state["training_loss_sum"] += group_loss
                state["training_target_tokens"] += group_tokens
                state["training_elapsed_s"] += time.monotonic() - group_started
                state["total_elapsed_s"] = time.monotonic() - watchdog.started
                state["peak_rss_bytes"] = max(state["peak_rss_bytes"], preflight.rss_bytes())
                append_jsonl(output / "updates.jsonl", {
                    "optimizer_update": state["optimizer_updates"],
                    "microsteps": state["microsteps"], "ids": [row["id"] for row in group],
                    "accumulation_size": len(group),
                    "loss_divisor": len(group), "target_tokens": group_tokens,
                    "token_weighted_loss": group_loss / group_tokens,
                    "gradient_norm": float(gradient),
                    "elapsed_s": time.monotonic() - group_started,
                })
                if state["optimizer_updates"] % CHECKPOINT_INTERVAL == 0:
                    save_recovery(output, model, optimizer, state, identity, tokenizer)
                if state["optimizer_updates"] % 10 == 0:
                    print(json.dumps({"phase": "pko_bounded_pilot",
                                      "optimizer_update": state["optimizer_updates"],
                                      "microsteps": state["microsteps"],
                                      "fixed_microsteps": fixed["microsteps"]}), flush=True)

            if (state["microsteps"] != fixed["microsteps"]
                    or state["optimizer_updates"] != fixed["optimizer_updates"]
                    or state["next_group"] != len(groups)):
                raise RuntimeError("fixed pilot schedule did not complete exactly once")
            trained_fingerprint = preflight.parameter_fingerprint(model)
            after = teacher_forced_loss(model, tokenizer, inner, watchdog,
                                        stage="inner_validation_after")
            checkpoint = output / "model"
            with watchdog.operation("save_final_model"):
                model.save_pretrained(checkpoint, safe_serialization=True)
                tokenizer.save_pretrained(checkpoint)
            final_files = {path.name: sha256_file(path) for path in checkpoint.iterdir()
                           if path.is_file()}
            # Drop AdamW moments, the live model, and the last autograd references
            # before loading the independent persisted copy under the 6GiB bound.
            del optimizer, model, loss, gradient, source, target
            gc.collect()
            with watchdog.operation("reload_final_model"):
                restored = T5ForConditionalGeneration.from_pretrained(
                    str(checkpoint), local_files_only=True, trust_remote_code=False,
                    weights_only=True, torch_dtype="float32")
                restored_tokenizer = T5TokenizerFast.from_pretrained(
                    str(checkpoint), local_files_only=True, trust_remote_code=False)
                preflight.adapt_lossless_tokenizer(restored_tokenizer)
            restored_fingerprint = preflight.parameter_fingerprint(restored)
            if restored_fingerprint != trained_fingerprint:
                raise RuntimeError("final safetensors parameter fingerprint differs after reload")
            if (preflight.tokenizer_fingerprint(restored_tokenizer)
                    != preflight.tokenizer_fingerprint(tokenizer)):
                raise RuntimeError("final tokenizer policy differs after reload")
            for probe in (inner[0]["source"], inner[0]["target"], "㈜3.3㎡14㎝",
                          "literal<extra_id_0>tag", "<pad>", "한글"):
                ids = preflight._token_ids(restored_tokenizer, probe)
                if restored_tokenizer.decode(
                        ids, skip_special_tokens=True,
                        clean_up_tokenization_spaces=False) != probe:
                    raise RuntimeError("final tokenizer is not lossless after reload")
            observation = watchdog.sample("final")

        if sha256_file(Path(__file__)) != source_sha:
            raise RuntimeError("training source changed during execution")
        if validate_qualification(qualification, model_dir)["sha256"] != authorization["sha256"]:
            raise RuntimeError("qualification changed during execution")
        if (sha256_file(PARTITION) != PARTITION_SHA256
                or sha256_file(PARTITION_PROOF) != PARTITION_PROOF_SHA256):
            raise RuntimeError("partition evidence changed during execution")
        report = {
            "schema": SCHEMA, "completed": True,
            "artifact_status": "trained_candidate_waiting_for_source_only_DEV_evaluation",
            "promoted": False, "production_changed": False,
            "scope": {"train_rows": FIT_ROWS + INNER_ROWS, "fit_rows": FIT_ROWS,
                      "inner_validation_rows": INNER_ROWS, "dev_opened": False,
                      "test_opened": False, "gold_used_only_as_teacher_forced_TRAIN_labels": True},
            "identity": identity, "qualification": {
                "path": authorization["path"], "sha256": authorization["sha256"],
                "independent_verification_path": authorization["proof_path"],
                "independent_verification_sha256": authorization["proof_sha256"]},
            "partition_proof": partition_proof, "schedule": schedule_proof,
            "fixed_plan": fixed,
            "training": {"microsteps": state["microsteps"],
                         "optimizer_updates": state["optimizer_updates"],
                         "target_tokens": state["training_target_tokens"],
                         "token_weighted_loss": (state["training_loss_sum"]
                                                 / state["training_target_tokens"]),
                         "elapsed_s": state["training_elapsed_s"],
                         "last_accumulation_size": len(groups[-1]),
                         "last_accumulation_divisor": len(groups[-1])},
            "inner_validation": {"before": baseline, "after": after,
                                 "loss_change": (after["token_weighted_loss"]
                                                 - baseline["token_weighted_loss"]),
                                 "selection_claim": False},
            "model": {"path": str(checkpoint.resolve()), "format": "safetensors",
                      "parameter_sha256": trained_fingerprint,
                      "restored_parameter_sha256": restored_fingerprint,
                      "reload_parameters_exact": True,
                      "tokenizer_policy_reload_exact": True,
                      "files": dict(sorted(final_files.items()))},
            "resources": {"limits": LIMITS,
                          "baseline_swap_bytes": watchdog.baseline_swap,
                          "watchdog_interval_s": 5.0,
                          "watchdog_hard_exit_grace_s": 5.0, "final": observation,
                          "trace_sha256": sha256_file(output / "resources.jsonl")},
            "elapsed_s": observation["elapsed_s"],
            "limitations": [
                "Teacher-forced TRAIN holdout loss does not prove conversion quality.",
                "The candidate must complete source-only DEV capture and evaluation before promotion.",
            ],
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(output / "metadata.json", {
            "schema": SCHEMA, "input_contract": "roman_sentence_only",
            "model_repo": preflight.MODEL_REPO, "model_revision": preflight.MODEL_REVISION,
            "license_treatment": "CC-BY-4.0 pending upstream metadata reconciliation",
            "tokenizer_policy": {"normalizer": None, "split_special_tokens": True,
                                 "truncation": False},
            "pins": identity["pins"], "fixed_plan": fixed,
        })
        write_json(output / "report.json", report)
        return report
    except BaseException as error:
        resumable = False
        existing_pointer = output / "recovery.json"
        resource_failure = isinstance(error, (ResourceExceeded, preflight.ResourceExceeded))
        if resource_failure:
            # Saving another ~383MB checkpoint after crossing RSS/swap/time bounds
            # could worsen the violation. Preserve only the last complete pointer.
            resumable = existing_pointer.is_file()
        elif "model" in locals() and "optimizer" in locals() and state["optimizer_updates"] > 0:
            try:
                save_recovery(output, model, optimizer, state, identity, tokenizer)
                resumable = True
            except BaseException:
                resumable = (output / "recovery.json").is_file()
        elif existing_pointer.is_file():
            resumable = True
        pointer_state = (json.loads(existing_pointer.read_text())["state"]
                         if resumable else {})
        cumulative_elapsed = (
            time.monotonic() - watchdog.started if "watchdog" in locals()
            else time.monotonic() - started)
        original_baseline_swap = (
            watchdog.baseline_swap if "watchdog" in locals() else baseline_swap)
        failure_payload = {
                "schema": SCHEMA, "completed": False, "resumable": resumable,
                "recovery_pointer": ("recovery.json" if resumable else None),
                "checkpointed_optimizer_updates": pointer_state.get("optimizer_updates", 0),
                "observed_optimizer_updates": state.get("optimizer_updates", 0),
                "uncheckpointed_optimizer_updates": (
                    state.get("optimizer_updates", 0)
                    - pointer_state.get("optimizer_updates", 0)),
                "cumulative_elapsed_s": cumulative_elapsed,
                "original_baseline_swap_bytes": original_baseline_swap,
                "error_type": type(error).__name__, "error": str(error),
                "dev_opened": False, "test_opened": False,
                "production_changed": False,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        failure_path = output / "failure.json"
        if failure_path.exists():
            prior_failure = json.loads(failure_path.read_text())
            failure_payload["watchdog_failure"] = prior_failure.get("watchdog_failure")
        write_json(failure_path, failure_payload)
        raise
    finally:
        lock.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification", type=Path, required=True,
                        help="completed pko resource-preflight v2 directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(args.qualification, args.output, resume=args.resume)
    print(json.dumps({key: report[key] for key in
                      ("completed", "artifact_status", "promoted", "elapsed_s")},
                     ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
