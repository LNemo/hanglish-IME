"""TRAIN-only PKO stage A: finish the previously unseen FIT tail once.

This is a bounded experiment, not a production training entry point.  It
reconstructs the exact step-620 AdamW state from the immutable step-600
recovery checkpoint, then trains only the 1,963 FIT rows not seen by the first
pilot.  DEV and TEST are never opened.
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
import shutil
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from statistics import median


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_pko_generation as generation
from scripts import preflight_pko_seq2seq as preflight
from scripts import probe_pko_source_conditioning as diagnostic
from scripts import train_pko_seq2seq as trainer


SCHEMA = "pko-unseen-fit-continuation-v1-20260921"
RUN = ROOT / ".omx/artifacts/pko-t5-pilot-v1"
DIAGNOSTIC = ROOT / ".omx/artifacts/pko-source-conditioning-v1"
PLAN = ROOT / ".omx/plans/pko-unseen-fit-continuation-v1.md"
SEEN_MICROSTEPS = 4_953
RECOVERY_MICROSTEPS = 4_800
REPLAY_MICROSTEPS = 153
UNSEEN_MICROSTEPS = 1_963
REPLAY_UPDATES = 20
CONTINUATION_UPDATES = 246
GRADIENT_ACCUMULATION = 8
CHECKPOINT_INTERVAL = 100
FINAL_PARENT_PARAMETER_SHA256 = (
    "119c409f18712706ebfb1baeb8cbf8755e80ec61d59c2e6400a83bca417b54af")
LOSS_RTOL = 1e-6
LOSS_ATOL = 2e-6
GRADIENT_RTOL = 2e-6
GRADIENT_ATOL = 2e-6
LIMITS = {"rss_bytes": 6 * 1024**3, "swap_growth_bytes": 512 * 1024**2,
          "whole_seconds": 900.0, "update_seconds": 120.0}

PINS = {
    PLAN: "c3b11b146a4f07f7d8e6a422eac1268786f735ec47dc0c89ef07fc360af71ee9",
    Path(trainer.__file__): "7cb34f7cde2ce8a493e2f0bf75793329e63096f8baea9ac42fc7f02a55663113",
    Path(preflight.__file__): "89bc4bd2a9875ace9eb010ca443168df4db84ed34772342dd55923e2ab7518f2",
    Path(diagnostic.__file__): "4d27624169c56396d62da9087f6414990a80c3557196ca5bc39ee24a826c0bbb",
    Path(generation.__file__): "e545cd6e193c905c5ff1db985653783f84839644e572f1e592bf70808e90e640",
    RUN / "report.json": "093456fcd7e695abf5ed84ecd71b5240a696b166c2668b39cb7a0edbce6615e9",
    RUN / "independent-verification.json":
        "963c6bd132c1979c752e43bdf3de4de3addadc11faf1eee6c89f22e848e91848",
    RUN / "recovery.json": "38505bf2fc2ba9bc39225f1797bbbc852ce05296d3664b2d043f2f0c5c476def",
    RUN / "schedule.json": "de423aa4f6eee2d96e18aca2cc2fa02974cfd526a792e7f2445723615e3846cd",
    RUN / "updates.jsonl": "b6aff505484418d3bf02f2a449ffb247afecc51ec351187f71bb0fd32c998ee4",
    RUN / "recovery-optimizer-1.pt":
        "ffe683cb362dd9861ebe6e23e9568edd7bf804b7a30ef0f31a03153ae53fb1db",
    RUN / "recovery-model-1/config.json":
        "a7e58e092c4bd9b6fa63bdac420e47f56bc8628971b91c07a1d20d211f9bb072",
    RUN / "recovery-model-1/generation_config.json":
        "8145d7eecabff8e16a9876617a6d52728e9b8fbe24c426e6bf9ebbd6bfb87737",
    RUN / "recovery-model-1/model.safetensors":
        "bdd6c16ac1b14b7cb90a15faf884dc6a8cc1022da0c3cd317cffc2ffff662e8b",
    RUN / "recovery-model-1/special_tokens_map.json":
        "9dd2846a446460500ae8204f70afef81847b1adb817cfbd42b2cc4c8d98e682d",
    RUN / "recovery-model-1/tokenizer.json":
        "6445b45a7e5e70cc8a6fa0f217075c02bf83dc6c3045628e2f3beb08ab9ea924",
    RUN / "recovery-model-1/tokenizer_config.json":
        "610822b34a6106447a845f5bafd36c6e45c784cccf45f44669e82dc0c3659175",
    DIAGNOSTIC / "report.json":
        "e3803ee8feb52557a76f6d9ecbf733cd764b1b37f873139ecd6558a714cddf2a",
    DIAGNOSTIC / "independent-verification.json":
        "48a57291fc5cd7b40b38f2087da25f2d3f00d49224fc99a7e2b29361fa78237e",
    DIAGNOSTIC / "selection.json":
        "406d517f623a9ec1cb266e967f825c4df4bb261f75ae47adf3a533bd0a3627f6",
    DIAGNOSTIC / "capture.jsonl":
        "ece70d00628fcf9427925cc08eca6c40dbe089b438310d62e73288d53ec81156",
}


class ResourceExceeded(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    return preflight.sha256_file(Path(path))


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


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line]


def close(value, expected, *, rtol, atol) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value)
            and math.isclose(value, expected, rel_tol=rtol, abs_tol=atol))


def groups(rows: list[dict], size: int = GRADIENT_ACCUMULATION) -> list[list[dict]]:
    require(bool(rows) and size > 0, "empty continuation schedule")
    return [rows[offset:offset + size] for offset in range(0, len(rows), size)]


def _tuple_state(value):
    if isinstance(value, list):
        return tuple(_tuple_state(item) for item in value)
    return value


def _json_state(value):
    if isinstance(value, tuple):
        return [_json_state(item) for item in value]
    return value


def validate_replay_contract(schedule_rows: list[dict], journals: list[dict]) -> list[list[dict]]:
    replay_rows = schedule_rows[RECOVERY_MICROSTEPS:SEEN_MICROSTEPS]
    replay_groups = groups(replay_rows)
    expected = journals[600:620]
    require(len(replay_rows) == REPLAY_MICROSTEPS, "replay microstep count differs")
    require(len(replay_groups) == REPLAY_UPDATES and len(expected) == REPLAY_UPDATES,
            "replay optimizer update count differs")
    for index, (group, record) in enumerate(zip(replay_groups, expected), start=601):
        ids = [row["id"] for row in group]
        require(record.get("optimizer_update") == index, "replay journal step differs")
        require(record.get("ids") == ids, "replay journal IDs differ")
        require(record.get("accumulation_size") == len(group)
                and record.get("loss_divisor") == len(group),
                "replay accumulation contract differs")
    require(len(replay_groups[-1]) == 1, "replay final accumulation differs")
    return replay_groups


def stage_gate(before_conditioning: dict, after_conditioning: dict,
               before_generation: dict, after_generation: dict) -> dict:
    decrease = ((before_conditioning["normal_token_nll"]
                 - after_conditioning["normal_token_nll"])
                / before_conditioning["normal_token_nll"])
    floor = before_conditioning["median_row_delta"] * .9
    new_exact = sum(after and not before for before, after in zip(
        before_generation["correct_flags"], after_generation["correct_flags"]))
    literal_regressions = sum(before and not after for before, after in zip(
        before_generation["literal_flags"], after_generation["literal_flags"]))
    checks = {
        "normal_nll_decrease_at_least_10pct": decrease >= .10,
        "normal_source_wins_at_least_42_of_64":
            after_conditioning["normal_lower_rows"] >= 42,
        "median_source_delta_retains_90pct":
            after_conditioning["median_row_delta"] >= floor,
        "at_least_4_new_exact": new_exact >= 4,
        "literal_signature_regressions_zero": literal_regressions == 0,
    }
    return {"promising_for_next_train_only_stage": all(checks.values()),
            "checks": checks, "normal_nll_relative_decrease": decrease,
            "median_delta_floor": floor, "new_exact": new_exact,
            "literal_signature_regressions": literal_regressions,
            "runtime_promotion": False,
            "literal_scope": "signature diagnostic only; not protected-span preservation"}


def character_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def score_generation(captures: list[dict], targets: dict[str, str]) -> dict:
    require(len(captures) == 64 and len({row["id"] for row in captures}) == 64,
            "generation capture denominator differs")
    correct_flags, literal_flags, eos_flags = [], [], []
    edits = characters = 0
    outcomes = []
    for row in captures:
        require(row.get("reason") in {
            "complete", "max_tokens_without_eos", "input_too_long"},
            "generation reason differs")
        require(isinstance(row.get("token_ids"), list)
                and all(isinstance(value, int) and not isinstance(value, bool)
                        for value in row["token_ids"])
                and isinstance(row.get("generated_text"), str)
                and isinstance(row.get("eos_completed"), bool),
                "generation capture payload differs")
        require((row["reason"] == "complete") == row["eos_completed"],
                "generation EOS relationship differs")
        target = targets[row["id"]]
        prediction = row["generated_text"]
        correct = row["reason"] == "complete" and prediction == target
        literal = generation.literal_signature(row["source"]) == generation.literal_signature(
            prediction)
        edits += character_distance(prediction, target)
        characters += len(target)
        correct_flags.append(correct); literal_flags.append(literal)
        eos_flags.append(row["eos_completed"])
        outcomes.append({"id": row["id"], "correct": correct,
                         "eos_completed": row["eos_completed"],
                         "literal_signature_exact": literal})
    return {"rows": 64, "whole_exact": sum(correct_flags),
            "eos_complete": sum(eos_flags),
            "literal_signature_exact": sum(literal_flags),
            "character_error_rate": edits / max(1, characters),
            "character_edits": edits, "target_characters": characters,
            "correct_flags": correct_flags, "literal_flags": literal_flags,
            "outcomes": outcomes,
            "labels_joined_after_closed_capture": True,
            "literal_scope": "uppercase/digit/punctuation signature; not protected-span guarantee"}


class StageWatchdog(preflight.ResourceWatchdog):
    def _monitor(self):
        while not self.stop_event.wait(self.interval_s):
            try:
                append_jsonl(self.trace, self._sample("watchdog"), self.lock)
            except BaseException as error:
                self.failure = f"{type(error).__name__}: {error}"
                write_json(self.trace.parent / "report.json", {
                    "schema": SCHEMA, "completed": False,
                    "error_type": type(error).__name__, "error": str(error),
                    "watchdog_failure": self.failure,
                    "dev_opened": False, "test_opened": False,
                    "production_changed": False,
                    "runtime_promotion": False,
                })
                os.kill(os.getpid(), signal.SIGUSR1)
                if not self.stop_event.wait(5.0):  # pragma: no cover - hard fail-safe
                    os._exit(124)
                return


def prepare() -> dict:
    """Model-free authorization and exact schedule construction."""
    for path, digest in PINS.items():
        require(path.is_file() and sha256_file(path) == digest,
                f"frozen continuation input differs: {path}")
    training_report = json.loads((RUN / "report.json").read_text())
    training_proof = json.loads((RUN / "independent-verification.json").read_text())
    diagnostic_report = json.loads((DIAGNOSTIC / "report.json").read_text())
    diagnostic_proof = json.loads(
        (DIAGNOSTIC / "independent-verification.json").read_text())
    require(training_report.get("completed") is True
            and training_proof.get("verified") is True,
            "parent pilot evidence is incomplete")
    require(diagnostic_report.get("completed") is True
            and diagnostic_proof.get("verified") is True
            and diagnostic_report.get("metrics", {}).get("source_signal_gate") is True
            and diagnostic_report["metrics"].get("normal_lower_rows") == 64,
            "source-conditioning gate did not qualify stage A")
    require(diagnostic_proof.get("identity", {}).get("report_sha256")
            == PINS[DIAGNOSTIC / "report.json"],
            "source-conditioning proof does not bind the report")

    fit, inner, partition_proof = trainer.load_train_partition()
    fit_by_id = {row["id"]: row for row in fit}
    inner_by_id = {row["id"]: row for row in inner}
    schedule_payload = json.loads((RUN / "schedule.json").read_text())
    schedule_ids = schedule_payload.get("ids")
    require(isinstance(schedule_ids, list) and len(schedule_ids) == 6916
            and len(set(schedule_ids)) == 6916 and set(schedule_ids) == set(fit_by_id),
            "parent FIT schedule identity differs")
    schedule_rows = [fit_by_id[row_id] for row_id in schedule_ids]
    unseen = schedule_rows[SEEN_MICROSTEPS:]
    require(len(unseen) == UNSEEN_MICROSTEPS
            and not ({row["id"] for row in unseen}
                     & set(schedule_ids[:SEEN_MICROSTEPS]))
            and not ({row["id"] for row in unseen} & set(inner_by_id)),
            "unseen FIT continuation set differs")
    continuation_groups = groups(unseen)
    require(len(continuation_groups) == CONTINUATION_UPDATES
            and len(continuation_groups[-1]) == 3,
            "continuation accumulation schedule differs")

    journals = read_jsonl(RUN / "updates.jsonl")
    require(len(journals) == 620, "parent update journal differs")
    replay_groups = validate_replay_contract(schedule_rows, journals)
    recovery = json.loads((RUN / "recovery.json").read_text())
    state = recovery.get("state", {})
    require(recovery.get("slot") == 1
            and recovery.get("model_dir") == "recovery-model-1"
            and recovery.get("optimizer_path") == "recovery-optimizer-1.pt"
            and state.get("optimizer_updates") == 600
            and state.get("microsteps") == RECOVERY_MICROSTEPS
            and state.get("next_group") == 600,
            "parent recovery cursor differs")
    require(recovery.get("identity") == training_report.get("identity"),
            "parent recovery identity differs from terminal training report")
    require(recovery.get("artifacts", {}).get("optimizer_sha256")
            == PINS[RUN / "recovery-optimizer-1.pt"],
            "parent recovery optimizer identity differs")
    observed_recovery_files = {
        path.name: sha256_file(path) for path in sorted((RUN / "recovery-model-1").iterdir())
        if path.is_file()}
    require(observed_recovery_files == recovery.get("artifacts", {}).get("model_files"),
            "parent recovery model identity differs")

    selection = json.loads((DIAGNOSTIC / "selection.json").read_text())
    pairs = selection.get("pairs")
    require(isinstance(pairs, list) and len(pairs) == 64
            and len({row["id"] for row in pairs}) == 64
            and all(row["id"] in inner_by_id for row in pairs),
            "diagnostic64 selection differs")
    frozen_conditioning = read_jsonl(DIAGNOSTIC / "capture.jsonl")
    require([row["id"] for row in frozen_conditioning]
            == [row["id"] for row in pairs], "diagnostic capture order differs")
    input_pins = dict(PINS)
    for path, digest in preflight.EXPECTED_FILES.items():
        if path != preflight.BASE / "dev.jsonl":
            require(path.is_file() and sha256_file(path) == digest,
                    f"pinned TRAIN-only input differs: {path}")
            input_pins[path] = digest
    return {
        "training_report": training_report, "diagnostic_report": diagnostic_report,
        "recovery": recovery, "schedule_ids": schedule_ids,
        "schedule_rows": schedule_rows, "replay_groups": replay_groups,
        "replay_journals": journals[600:620], "unseen_rows": unseen,
        "continuation_groups": continuation_groups, "pairs": pairs,
        "diagnostic_targets": {row["id"]: inner_by_id[row["id"]]["target"]
                               for row in pairs},
        "frozen_conditioning": frozen_conditioning,
        "partition_proof": partition_proof,
        "pins": {preflight.path_key(path): digest
                 for path, digest in sorted(input_pins.items(), key=lambda item: str(item[0]))},
    }


def optimizer_update(model, optimizer, tokenizer, group: list[dict]) -> dict:
    import torch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum, tokens = 0.0, 0
    for row in group:
        source = trainer.tokenizer_encode(tokenizer, row["source"])
        target = trainer.tokenizer_encode(tokenizer, row["target"])
        count = int(target["input_ids"].numel())
        loss = model(**source, labels=target["input_ids"]).loss
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite continuation loss")
        (loss * trainer.loss_scale_for_group(group)).backward()
        loss_sum += float(loss.detach()) * count
        tokens += count
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(gradient):
        raise RuntimeError("nonfinite continuation gradient")
    optimizer.step()
    return {"ids": [row["id"] for row in group], "accumulation_size": len(group),
            "loss_divisor": len(group), "target_tokens": tokens,
            "token_weighted_loss": loss_sum / tokens,
            "gradient_norm": float(gradient)}


def validate_replayed_update(actual: dict, expected: dict) -> None:
    require(actual["ids"] == expected.get("ids"), "replayed update IDs differ")
    require(actual["accumulation_size"] == expected.get("accumulation_size")
            and actual["loss_divisor"] == expected.get("loss_divisor")
            and actual["target_tokens"] == expected.get("target_tokens"),
            "replayed update accounting differs")
    require(close(actual["token_weighted_loss"], expected.get("token_weighted_loss"),
                  rtol=LOSS_RTOL, atol=LOSS_ATOL),
            "replayed loss exceeds tiny FP32 tolerance")
    require(close(actual["gradient_norm"], expected.get("gradient_norm"),
                  rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL),
            "replayed gradient exceeds tiny FP32 tolerance")


def conditioning_capture(model, tokenizer, pairs: list[dict], targets: dict[str, str],
                         watchdog, path: Path, *, stage: str) -> tuple[dict, list[dict]]:
    import torch
    model.eval()
    captures = []
    with torch.inference_mode():
        for pair in pairs:
            labels = trainer.tokenizer_encode(tokenizer, targets[pair["id"]])["input_ids"]
            row = dict(pair, target_tokens=int(labels.numel()))
            for field, key in (("source", "normal_nll"),
                               ("shuffled_source", "shuffled_nll")):
                encoded = trainer.tokenizer_encode(tokenizer, pair[field])
                row[field + "_tokens"] = int(encoded["input_ids"].numel())
                with watchdog.operation(f"{stage}:{pair['id']}:{key}", update=True):
                    row[key] = float(model(**encoded, labels=labels).loss)
            captures.append(row)
            append_jsonl(path, row)
    summary = diagnostic.summarize(captures)
    summary["median_row_delta"] = median(
        row["shuffled_nll"] - row["normal_nll"] for row in captures)
    return summary, captures


def validate_frozen_conditioning(actual: list[dict], expected: list[dict]) -> None:
    require(len(actual) == len(expected) == 64, "frozen conditioning denominator differs")
    for observed, frozen in zip(actual, expected):
        require(observed["id"] == frozen["id"], "frozen conditioning ID differs")
        for key in ("normal_nll", "shuffled_nll"):
            require(close(observed[key], frozen[key], rtol=LOSS_RTOL, atol=LOSS_ATOL),
                    f"reconstructed {key} exceeds tiny FP32 tolerance")


def generation_capture(model, tokenizer, pairs: list[dict], watchdog,
                       path: Path, *, stage: str) -> tuple[list[dict], str]:
    captures = []
    for pair in pairs:
        started = time.monotonic()
        with watchdog.operation(f"{stage}:{pair['id']}:greedy", update=True):
            result = generation.generate_source(model, tokenizer, pair["source"])
        row = {"id": pair["id"], "source": pair["source"], **result,
               "latency_s": time.monotonic() - started}
        captures.append(row)
        append_jsonl(path, row)
    raw_sha = sha256_file(path)
    write_json(path.with_suffix(".complete.json"), {
        "rows": len(captures), "sha256": raw_sha,
        "closed_before_label_join": True})
    return captures, raw_sha


def save_checkpoint(output: Path, model, optimizer, tokenizer, state: dict,
                    identity: dict, *, final: bool = False) -> dict:
    import torch
    existing = output / "checkpoint.json"
    slot = (1 - json.loads(existing.read_text())["slot"]) if existing.exists() else 0
    model_dir = output / f"checkpoint-model-{slot}"
    model_dir.mkdir(exist_ok=True)
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    model_files = {path.name: sha256_file(path) for path in sorted(model_dir.iterdir())
                   if path.is_file()}
    optimizer_path = output / f"checkpoint-optimizer-{slot}.pt"
    temporary = optimizer_path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save({"optimizer": optimizer.state_dict(),
                    "torch_rng_state": torch.get_rng_state()}, stream)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, optimizer_path)
    pointer = {"schema": SCHEMA, "slot": slot, "final": final,
               "model_dir": model_dir.name, "optimizer_path": optimizer_path.name,
               "model_files": model_files,
               "optimizer_sha256": sha256_file(optimizer_path),
               "state": state, "identity": identity,
               "python_random_state": _json_state(random.getstate()),
               "saved_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(existing, pointer)
    return pointer


def prepare_output(output: Path) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    return output


def run(output: Path) -> dict:
    import torch
    from transformers import T5ForConditionalGeneration, T5TokenizerFast

    started, baseline_swap = time.monotonic(), preflight.system_swap_bytes()
    authorization = prepare()
    output = prepare_output(output)
    lock = (output / ".run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("continuation output is already owned")
    source_sha = sha256_file(Path(__file__))
    source_snapshot = output / "continuation-source.py"
    shutil.copyfile(Path(__file__), source_snapshot)
    with source_snapshot.open("rb") as stream:
        os.fsync(stream.fileno())
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.manual_seed(trainer.SEED); random.seed(trainer.SEED)
    identity = {"parent_parameter_sha256": FINAL_PARENT_PARAMETER_SHA256,
                "recovery_optimizer_update": 600,
                "replayed_parent_microsteps": REPLAY_MICROSTEPS,
                "continuation_microsteps": UNSEEN_MICROSTEPS,
                "continuation_updates": CONTINUATION_UPDATES,
                "seed": trainer.SEED,
                "pins": {**authorization["pins"],
                         preflight.path_key(Path(__file__)): source_sha}}
    state = {"continuation_updates": 0, "continuation_microsteps": 0,
             "next_group": 0, "training_loss_sum": 0.0,
             "training_target_tokens": 0}
    write_json(output / "schedule.json", {
        "schema": SCHEMA,
        "ids": [row["id"] for row in authorization["unseen_rows"]],
        "rows": UNSEEN_MICROSTEPS, "unique_ids": UNSEEN_MICROSTEPS,
        "parent_schedule_slice": [SEEN_MICROSTEPS, 6916],
        "disjoint_from_parent_seen": True, "disjoint_from_inner": True,
        "captured_before_learning": True})
    try:
        with StageWatchdog(started, baseline_swap, output / "resources.jsonl",
                           limits=LIMITS, interval_s=5.0) as watchdog:
            with watchdog.operation("load_step600_recovery", update=True):
                tokenizer = T5TokenizerFast.from_pretrained(
                    str(RUN / "recovery-model-1"), local_files_only=True,
                    trust_remote_code=False)
                preflight.adapt_lossless_tokenizer(tokenizer)
                model = T5ForConditionalGeneration.from_pretrained(
                    str(RUN / "recovery-model-1"), local_files_only=True,
                    trust_remote_code=False, weights_only=True, torch_dtype="float32")
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
                payload = torch.load(RUN / "recovery-optimizer-1.pt", weights_only=True)
                optimizer.load_state_dict(payload["optimizer"])
                torch.set_rng_state(payload["torch_rng_state"])
                random.setstate(_tuple_state(authorization["recovery"]["python_random_state"]))
            replay_records = []
            for offset, (group, expected) in enumerate(zip(
                    authorization["replay_groups"], authorization["replay_journals"]),
                    start=601):
                begin = time.monotonic()
                with watchdog.operation(f"replay_optimizer_update_{offset}", update=True):
                    actual = optimizer_update(model, optimizer, tokenizer, group)
                validate_replayed_update(actual, expected)
                actual.update({"optimizer_update": offset,
                               "elapsed_s": time.monotonic() - begin})
                replay_records.append(actual)
                append_jsonl(output / "replay.jsonl", actual)
            replay_parameter_sha = preflight.parameter_fingerprint(model)
            require(replay_parameter_sha == FINAL_PARENT_PARAMETER_SHA256,
                    "exact parent parameter fingerprint was not reconstructed")

            before_conditioning, before_rows = conditioning_capture(
                model, tokenizer, authorization["pairs"],
                authorization["diagnostic_targets"], watchdog,
                output / "conditioning-before.jsonl", stage="before")
            validate_frozen_conditioning(before_rows, authorization["frozen_conditioning"])
            before_captures, before_capture_sha = generation_capture(
                model, tokenizer, authorization["pairs"], watchdog,
                output / "generation-before.jsonl", stage="before")
            before_generation = score_generation(
                before_captures, authorization["diagnostic_targets"])

            training_started = time.monotonic()
            for index, group in enumerate(authorization["continuation_groups"], start=1):
                begin = time.monotonic()
                with watchdog.operation(f"continuation_optimizer_update_{index}", update=True):
                    update = optimizer_update(model, optimizer, tokenizer, group)
                state["continuation_updates"] = index
                state["continuation_microsteps"] += len(group)
                state["next_group"] = index
                state["training_loss_sum"] += (
                    update["token_weighted_loss"] * update["target_tokens"])
                state["training_target_tokens"] += update["target_tokens"]
                update.update({"optimizer_update": index,
                               "elapsed_s": time.monotonic() - begin})
                append_jsonl(output / "updates.jsonl", update)
                if index % CHECKPOINT_INTERVAL == 0:
                    with watchdog.operation(f"checkpoint_{index}", update=True):
                        save_checkpoint(output, model, optimizer, tokenizer, state,
                                        identity, final=False)
                if index % 25 == 0:
                    print(json.dumps({"phase": "pko_unseen_fit", "update": index,
                                      "updates": CONTINUATION_UPDATES}), flush=True)
            training_elapsed = time.monotonic() - training_started
            require(state["continuation_updates"] == CONTINUATION_UPDATES
                    and state["continuation_microsteps"] == UNSEEN_MICROSTEPS
                    and state["next_group"] == CONTINUATION_UPDATES,
                    "continuation schedule did not complete exactly once")

            after_conditioning, after_rows = conditioning_capture(
                model, tokenizer, authorization["pairs"],
                authorization["diagnostic_targets"], watchdog,
                output / "conditioning-after.jsonl", stage="after")
            after_captures, after_capture_sha = generation_capture(
                model, tokenizer, authorization["pairs"], watchdog,
                output / "generation-after.jsonl", stage="after")
            after_generation = score_generation(
                after_captures, authorization["diagnostic_targets"])
            decision = stage_gate(before_conditioning, after_conditioning,
                                  before_generation, after_generation)
            final_parameter_sha = preflight.parameter_fingerprint(model)
            with watchdog.operation("save_final_state", update=True):
                final_pointer = save_checkpoint(output, model, optimizer, tokenizer,
                                                state, identity, final=True)
                final_model = output / "model"
                model.save_pretrained(final_model, safe_serialization=True)
                tokenizer.save_pretrained(final_model)
            final_files = {path.name: sha256_file(path) for path in sorted(final_model.iterdir())
                           if path.is_file()}
            tokenizer_sha = preflight.tokenizer_fingerprint(tokenizer)
            del model, optimizer, payload
            gc.collect()
            with watchdog.operation("reload_final_model", update=True):
                restored = T5ForConditionalGeneration.from_pretrained(
                    str(final_model), local_files_only=True, trust_remote_code=False,
                    weights_only=True, torch_dtype="float32")
                restored_tokenizer = T5TokenizerFast.from_pretrained(
                    str(final_model), local_files_only=True, trust_remote_code=False)
                preflight.adapt_lossless_tokenizer(restored_tokenizer)
            restored_sha = preflight.parameter_fingerprint(restored)
            require(restored_sha == final_parameter_sha,
                    "final continuation parameters differ after reload")
            require(preflight.tokenizer_fingerprint(restored_tokenizer) == tokenizer_sha,
                    "final continuation tokenizer policy differs after reload")
            for probe in (authorization["pairs"][0]["source"],
                          authorization["diagnostic_targets"][authorization["pairs"][0]["id"]],
                          "㈜3.3㎡14㎝", "literal<extra_id_0>tag", "<pad>", "한글"):
                token_ids = preflight._token_ids(restored_tokenizer, probe)
                require(restored_tokenizer.decode(
                    token_ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False) == probe,
                    "final continuation tokenizer is not lossless")
            require(sha256_file(Path(__file__)) == source_sha,
                    "continuation source changed during execution")
            for raw_path, digest in identity["pins"].items():
                path = ROOT / raw_path if not Path(raw_path).is_absolute() else Path(raw_path)
                require(sha256_file(path) == digest,
                        f"continuation input changed during execution: {raw_path}")
            observation = watchdog._sample("final")
        before_marker = output / "generation-before.complete.json"
        after_marker = output / "generation-after.complete.json"
        report = {
            "schema": SCHEMA, "completed": True,
            "artifact_status": "train_only_stage_a_candidate",
            "promoted": False, "runtime_promotion": False,
            "production_changed": False,
            "scope": {"fit_rows": 6916, "parent_seen_rows": SEEN_MICROSTEPS,
                      "newly_exposed_fit_rows": UNSEEN_MICROSTEPS,
                      "diagnostic_inner_rows": 64,
                      "dev_opened": False, "test_opened": False,
                      "generation_source_only": True,
                      "targets_joined_after_each_complete_generation_capture": True},
            "identity": identity,
            "replay": {"microsteps": REPLAY_MICROSTEPS, "updates": REPLAY_UPDATES,
                       "loss_rtol": LOSS_RTOL, "loss_atol": LOSS_ATOL,
                       "gradient_rtol": GRADIENT_RTOL,
                       "gradient_atol": GRADIENT_ATOL,
                       "exact_parent_parameter_sha256": replay_parameter_sha,
                       "journal_sha256": sha256_file(output / "replay.jsonl")},
            "training": {"microsteps": state["continuation_microsteps"],
                         "optimizer_updates": state["continuation_updates"],
                         "last_accumulation_size": 3,
                         "target_tokens": state["training_target_tokens"],
                         "token_weighted_loss": (state["training_loss_sum"]
                                                 / state["training_target_tokens"]),
                         "elapsed_s": training_elapsed,
                         "updates_sha256": sha256_file(output / "updates.jsonl")},
            "conditioning": {"before": before_conditioning,
                             "after": after_conditioning,
                             "before_sha256": sha256_file(
                                 output / "conditioning-before.jsonl"),
                             "after_sha256": sha256_file(
                                 output / "conditioning-after.jsonl")},
            "generation": {"configuration": {"num_beams": 1, "do_sample": False,
                                               "max_new_tokens": 512,
                                               "input_truncation": False},
                           "before": before_generation, "after": after_generation,
                           "before_capture_sha256": before_capture_sha,
                           "after_capture_sha256": after_capture_sha,
                           "before_closure_marker_sha256": sha256_file(before_marker),
                           "after_closure_marker_sha256": sha256_file(after_marker),
                           "all_captures_closed_before_label_join": True},
            "decision": decision,
            "literal_diagnostic": "signature only; not a protected-span guarantee",
            "model": {"path": str(final_model.resolve()), "format": "safetensors",
                      "parameter_sha256": final_parameter_sha,
                      "restored_parameter_sha256": restored_sha,
                      "reload_parameters_exact": True,
                      "tokenizer_policy_reload_exact": True,
                      "files": final_files},
            "checkpoint": {"pointer_sha256": sha256_file(output / "checkpoint.json"),
                           "final": final_pointer},
            "source_snapshot": {"path": source_snapshot.name,
                                "sha256": sha256_file(source_snapshot)},
            "resources": {"limits": LIMITS,
                          "prior_frozen_diagnostic_operation_limit_s": 60.0,
                          "continuation_operation_limit_s": 120.0,
                          "baseline_swap_bytes": baseline_swap, "final": observation,
                          "trace_sha256": sha256_file(output / "resources.jsonl")},
            "elapsed_s": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(output / "report.json", report)
        return report
    except BaseException as error:
        if not (output / "report.json").exists():
            write_json(output / "report.json", {
                "schema": SCHEMA, "completed": False,
                "error_type": type(error).__name__, "error": str(error),
                "checkpoint_available": (output / "checkpoint.json").exists(),
                "restart_supported": False, "dev_opened": False,
                "test_opened": False, "production_changed": False,
                "runtime_promotion": False,
                "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        lock.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        authorization = prepare()
        print(json.dumps({"prepared": True,
                          "replay_microsteps": sum(map(len, authorization["replay_groups"])),
                          "unseen_fit_rows": len(authorization["unseen_rows"]),
                          "continuation_updates": len(authorization["continuation_groups"]),
                          "diagnostic_rows": len(authorization["pairs"]),
                          "model_loaded": False, "dev_opened": False,
                          "test_opened": False}, sort_keys=True))
        return
    if args.output is None:
        parser.error("--output is required unless --self-test is used")
    report = run(args.output)
    print(json.dumps({"completed": report["completed"],
                      "decision": report["decision"],
                      "elapsed_s": report["elapsed_s"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
