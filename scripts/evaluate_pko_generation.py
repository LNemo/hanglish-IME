"""Source-only greedy DEV evaluation for a completed pko-T5 training candidate."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
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
from hanglish.core.ctc_word_proposals import literal_signature
from scripts import preflight_pko_seq2seq as preflight


SCHEMA = "pko-t5-small-dev-generation-v1"
TRAINING_SCHEMA = "pko-t5-small-bounded-pilot-v1"
TRAINER = ROOT / "scripts/train_pko_seq2seq.py"
EXPECTED_TRAINER_SHA256 = "7cb34f7cde2ce8a493e2f0bf75793329e63096f8baea9ac42fc7f02a55663113"
# Filled only after the training run and its independent proof are terminal.
# A missing binding deliberately keeps DEV evaluation fail-closed.
EXPECTED_TRAINING_REPORT_SHA256 = "093456fcd7e695abf5ed84ecd71b5240a696b166c2668b39cb7a0edbce6615e9"
EXPECTED_TRAINING_PROOF_SHA256 = "963c6bd132c1979c752e43bdf3de4de3addadc11faf1eee6c89f22e848e91848"
TRAINING_PROOF_NAME = "independent-verification.json"
PREFLIGHT_SOURCE_SHA256 = "89bc4bd2a9875ace9eb010ca443168df4db84ed34772342dd55923e2ab7518f2"
DEV = ROOT / "data/challenge/sentence-v3/dev.jsonl"
DEV_SHA256 = "593e1a032b54bf4d2c541da6754cfc68daf759cdcee7f8fa1f0a746459910ac9"
DEV_ROWS = 316
MAX_TOKENS = 512
LIMITS = {"rss_bytes": 6 * 1024**3, "swap_growth_bytes": 512 * 1024**2,
          "whole_seconds": 900.0, "update_seconds": 900.0}
TRAIN_LIMITS = {"rss_bytes": 6 * 1024**3, "swap_growth_bytes": 512 * 1024**2,
                "whole_seconds": 3600.0, "update_seconds": 120.0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_jsonl(path: Path, value: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def finite(value, *, positive=False) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def capture_row_count(path: Path) -> int:
    return (sum(bool(line) for line in Path(path).read_bytes().splitlines())
            if Path(path).is_file() else 0)


class EvaluationWatchdog(preflight.ResourceWatchdog):
    """Preflight resource sampler with evaluator-specific terminal evidence."""

    def __init__(self, started, baseline_swap, trace, capture_path, **kwargs):
        super().__init__(started, baseline_swap, trace, **kwargs)
        self.capture_path = Path(capture_path)
        self.targets_joined = False

    def _failure_report(self, error):
        write_json(self.trace.parent / "report.json", {
            "schema": SCHEMA, "completed": False, "promoted": False,
            "error_type": type(error).__name__, "error": str(error),
            "watchdog_failure": self.failure,
            "capture_rows": capture_row_count(self.capture_path),
            "scope": {"source_only_inference": True,
                      "dev_file_parsed_including_label_fields": True,
                      "targets_joined": self.targets_joined,
                      "labels_used_as_model_context": False,
                      "test_opened": False},
            "production_changed": False,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    def _monitor(self):
        while not self.stop_event.wait(self.interval_s):
            try:
                observed = self._sample("watchdog")
                preflight.append_jsonl(self.trace, observed, self.lock)
            except BaseException as error:
                self.failure = f"{type(error).__name__}: {error}"
                self._failure_report(error)
                os.kill(os.getpid(), signal.SIGUSR1)
                if not self.stop_event.wait(5.0):  # pragma: no cover - hard fail-safe
                    os._exit(124)
                return


def validate_training_run(training_run: Path) -> dict:
    training_run = Path(training_run).resolve()
    report_path, metadata_path = training_run / "report.json", training_run / "metadata.json"
    proof_path = training_run / TRAINING_PROOF_NAME
    if (EXPECTED_TRAINING_REPORT_SHA256 is None
            or EXPECTED_TRAINING_PROOF_SHA256 is None):
        raise ValueError("terminal pko training report/proof hashes are not frozen")
    if not report_path.is_file() or not metadata_path.is_file():
        raise ValueError("completed pko training report and metadata are required")
    if (sha256_file(report_path) != EXPECTED_TRAINING_REPORT_SHA256
            or not proof_path.is_file()
            or sha256_file(proof_path) != EXPECTED_TRAINING_PROOF_SHA256):
        raise ValueError("pinned pko training report or proof differs")
    report, metadata = json.loads(report_path.read_text()), json.loads(metadata_path.read_text())
    proof = json.loads(proof_path.read_text())
    identity, model_record = report.get("identity"), report.get("model")
    qualification = report.get("qualification")
    scope = report.get("scope")
    if (report.get("schema") != TRAINING_SCHEMA or report.get("completed") is not True
            or report.get("artifact_status")
            != "trained_candidate_waiting_for_source_only_DEV_evaluation"
            or report.get("promoted") is not False
            or report.get("production_changed") is not False
            or scope != {"train_rows": 7704, "fit_rows": 6916,
                         "inner_validation_rows": 788, "dev_opened": False,
                         "test_opened": False,
                         "gold_used_only_as_teacher_forced_TRAIN_labels": True}):
        raise ValueError("completed unexposed pko candidate is required")
    if (not isinstance(identity, dict)
            or identity.get("architecture") != "paust-pko-t5-small-full-finetune"
            or identity.get("official_model_revision") != preflight.MODEL_REVISION
            or identity.get("official_weight_sha256") != preflight.WEIGHT_SHA256
            or identity.get("fresh_start_from_official_weights") is not True
            or identity.get("fit_rows") != 6916
            or identity.get("inner_validation_rows") != 788
            or identity.get("optimizer") != {
                "type": "AdamW", "learning_rate": 1e-4, "weight_decay": .01,
                "clip_norm": 1.0, "gradient_accumulation": 8}):
        raise ValueError("pko training artifact identity differs")
    trainer_sha = sha256_file(TRAINER)
    trainer_key = preflight.path_key(TRAINER)
    if (trainer_sha != EXPECTED_TRAINER_SHA256
            or identity.get("pins", {}).get(trainer_key) != EXPECTED_TRAINER_SHA256
            or sha256_file(Path(preflight.__file__)) != PREFLIGHT_SOURCE_SHA256):
        raise ValueError("pinned pko trainer or preflight source differs")
    if (metadata.get("schema") != TRAINING_SCHEMA
            or metadata.get("input_contract") != "roman_sentence_only"
            or metadata.get("model_repo") != preflight.MODEL_REPO
            or metadata.get("model_revision") != preflight.MODEL_REVISION
            or metadata.get("pins") != identity.get("pins")
            or metadata.get("tokenizer_policy")
            != {"normalizer": None, "split_special_tokens": True, "truncation": False}):
        raise ValueError("pko training metadata differs")
    identity_pins = identity.get("pins")
    if not isinstance(identity_pins, dict) or not identity_pins:
        raise ValueError("pko training source pins are missing")
    for raw_path, digest in identity_pins.items():
        path = Path(raw_path) if Path(raw_path).is_absolute() else ROOT / raw_path
        if (not isinstance(digest, str) or len(digest) != 64
                or not path.is_file() or sha256_file(path) != digest):
            raise ValueError(f"pko training source pin differs: {raw_path}")
    fixed, training = report.get("fixed_plan"), report.get("training")
    if (not isinstance(fixed, dict) or fixed.get("fixed_before_learning") is not True
            or fixed.get("gradient_accumulation") != 8
            or not isinstance(fixed.get("microsteps"), int)
            or not 0 < fixed["microsteps"] <= 6916
            or fixed.get("optimizer_updates") != math.ceil(fixed["microsteps"] / 8)
            or not finite(fixed.get("estimated_total_s"), positive=True)
            or fixed["estimated_total_s"] > TRAIN_LIMITS["whole_seconds"]
            or not isinstance(training, dict)
            or training.get("microsteps") != fixed["microsteps"]
            or training.get("optimizer_updates") != fixed["optimizer_updates"]
            or not finite(training.get("target_tokens"), positive=True)
            or not finite(training.get("token_weighted_loss"))
            or not finite(training.get("elapsed_s"), positive=True)):
        raise ValueError("pko fixed training schedule or completion differs")
    inner = report.get("inner_validation")
    if (not isinstance(inner, dict) or inner.get("selection_claim") is not False
            or not all(isinstance(inner.get(stage), dict)
                       and inner[stage].get("rows") == 788
                       and finite(inner[stage].get("target_tokens"), positive=True)
                       and finite(inner[stage].get("token_weighted_loss"))
                       and finite(inner[stage].get("elapsed_s"), positive=True)
                       for stage in ("before", "after"))):
        raise ValueError("pko inner-validation evidence differs")
    model_dir = training_run / "model"
    if (not isinstance(model_record, dict) or model_record.get("format") != "safetensors"
            or Path(model_record.get("path", "")).resolve() != model_dir.resolve()
            or not isinstance(model_record.get("parameter_sha256"), str)
            or len(model_record["parameter_sha256"]) != 64
            or model_record.get("restored_parameter_sha256")
            != model_record.get("parameter_sha256")
            or model_record.get("reload_parameters_exact") is not True
            or model_record.get("tokenizer_policy_reload_exact") is not True
            or not isinstance(model_record.get("files"), dict)
            or "model.safetensors" not in model_record["files"]
            or "pytorch_model.bin" in model_record["files"]):
        raise ValueError("pko trained model record differs")
    actual_files = {path.name: sha256_file(path) for path in model_dir.iterdir()
                    if path.is_file()}
    if actual_files != model_record["files"]:
        raise ValueError("pko trained model files changed")
    if not isinstance(qualification, dict):
        raise ValueError("pko qualification binding is missing")
    qualification_pins = {}
    for path_key, sha_key in (("path", "sha256"),
                              ("independent_verification_path",
                               "independent_verification_sha256")):
        path = Path(qualification.get(path_key, ""))
        digest = qualification.get(sha_key)
        if (not path.is_file() or not isinstance(digest, str)
                or len(digest) != 64 or sha256_file(path) != digest):
            raise ValueError("pko qualification evidence changed")
        qualification_pins[preflight.path_key(path)] = digest
    resources = report.get("resources")
    observation = resources.get("final") if isinstance(resources, dict) else None
    if (not isinstance(resources, dict) or resources.get("limits") != TRAIN_LIMITS
            or not isinstance(observation, dict)
            or observation.get("rss_bytes", TRAIN_LIMITS["rss_bytes"] + 1)
            > TRAIN_LIMITS["rss_bytes"]
            or observation.get("swap_growth_bytes", TRAIN_LIMITS["swap_growth_bytes"] + 1)
            > TRAIN_LIMITS["swap_growth_bytes"]
            or observation.get("elapsed_s", TRAIN_LIMITS["whole_seconds"] + 1)
            > TRAIN_LIMITS["whole_seconds"]):
        raise ValueError("pko training resource completion differs")
    proof_identity, proof_counts = proof.get("identity"), proof.get("counts")
    proof_model, contracts = proof.get("model"), proof.get("contracts")
    updates_path, resources_path = training_run / "updates.jsonl", training_run / "resources.jsonl"
    expected_proof_identity = {
        "report_sha256": EXPECTED_TRAINING_REPORT_SHA256,
        "metadata_sha256": sha256_file(metadata_path),
        "updates_sha256": sha256_file(updates_path),
        "resources_sha256": sha256_file(resources_path),
        "frozen_trainer_sha256": EXPECTED_TRAINER_SHA256,
        "qualification_report_sha256": qualification["sha256"],
        "qualification_proof_sha256": qualification["independent_verification_sha256"],
    }
    expected_contracts = {
        "all_schedule_ids_unique_fit": True,
        "no_inner_ids_in_updates": True,
        "journal_exact_report_aggregates": True,
        "loss_scaling_and_divisors_valid": True,
        "resources_within_limits": True,
        "source_only_contract": True,
        "dev_opened": False, "test_opened": False,
        "production_changed": False, "model_loaded_by_verifier": False,
        "model_inference_by_verifier": False,
    }
    if (proof.get("verified") is not True
            or not isinstance(proof.get("scope"), str) or not proof["scope"]
            or proof_identity != expected_proof_identity
            or proof_counts != {
                "train_rows": 7704, "fit_rows": 6916, "inner_validation_rows": 788,
                "schedule_rows": 6916, "fixed_microsteps": fixed["microsteps"],
                "optimizer_updates": training["optimizer_updates"],
                "last_accumulation_size": training.get("last_accumulation_size")}
            or not isinstance(proof_model, dict)
            or proof_model.get("parameter_count") != 95_628_672
            or proof_model.get("safetensors_header_parameter_count") != 95_628_672
            or proof_model.get("file_sha256") != model_record["files"]
            or proof_model.get("reported_parameter_sha256")
            != model_record["parameter_sha256"]
            or proof_model.get("reported_reload_parameters_exact") is not True
            or proof_model.get("reported_tokenizer_policy_reload_exact") is not True
            or proof.get("training", {}).get("groups") != training["optimizer_updates"]
            or proof.get("training", {}).get("microsteps") != training["microsteps"]
            or proof.get("training", {}).get("target_tokens") != training["target_tokens"]
            or proof.get("training", {}).get("token_weighted_loss")
            != training["token_weighted_loss"]
            or proof.get("training", {}).get("last_accumulation_size")
            != training.get("last_accumulation_size")
            or proof.get("training", {}).get("inner_validation") != inner
            or proof.get("resources", {}).get("final") != observation
            or proof.get("resources", {}).get("peak_rss_bytes", TRAIN_LIMITS["rss_bytes"] + 1)
            > TRAIN_LIMITS["rss_bytes"]
            or proof.get("resources", {}).get(
                "max_swap_growth_bytes", TRAIN_LIMITS["swap_growth_bytes"] + 1)
            > TRAIN_LIMITS["swap_growth_bytes"]
            or contracts != expected_contracts
            or proof.get("pins") != identity_pins
            or proof.get("decision")
            != "trained candidate is internally valid and ready only for source-only DEV evaluation"):
        raise ValueError("independent pko training proof differs")
    config = json.loads((model_dir / "config.json").read_text())
    if (config.get("model_type") != "t5"
            or config.get("architectures") != ["T5ForConditionalGeneration"]
            or config.get("is_encoder_decoder") is not True
            or config.get("decoder_start_token_id") != 0
            or config.get("eos_token_id") != 1):
        raise ValueError("trained model is not the pinned T5 architecture")
    return {
        "training_run": str(training_run), "report": report, "metadata": metadata,
        "model_dir": model_dir, "pins": {
            preflight.path_key(TRAINER): trainer_sha,
            preflight.path_key(report_path): sha256_file(report_path),
            preflight.path_key(metadata_path): sha256_file(metadata_path),
            preflight.path_key(proof_path): EXPECTED_TRAINING_PROOF_SHA256,
            **{preflight.path_key(model_dir / name): digest
               for name, digest in actual_files.items()},
            **qualification_pins,
        },
    }


def read_dev_sources(path: Path = DEV, *, expected_sha: str = DEV_SHA256,
                     expected_rows: int = DEV_ROWS) -> list[dict]:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("pinned DEV input differs")
    rows = [json.loads(line) for line in raw.splitlines() if line]
    projected = []
    for row in rows:
        if (row.get("split") != "dev" or row.get("input_contract") != "roman_sentence_only"
                or not isinstance(row.get("id"), str) or not isinstance(row.get("input"), str)):
            raise ValueError("DEV source row differs")
        projected.append({"id": row["id"], "source": row["input"]})
    if len(projected) != expected_rows or len({row["id"] for row in projected}) != expected_rows:
        raise ValueError("DEV source denominator differs")
    return projected


def load_dev_targets(path: Path = DEV, *, expected_sha: str = DEV_SHA256,
                     expected_rows: int = DEV_ROWS) -> dict[str, list[str]]:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("DEV changed before late target join")
    targets = {}
    for row in (json.loads(line) for line in raw.splitlines() if line):
        expected = row.get("expected")
        if (row.get("split") != "dev" or not isinstance(row.get("id"), str)
                or not isinstance(expected, list) or not expected
                or any(not isinstance(value, str) for value in expected)):
            raise ValueError("DEV target row differs")
        targets[row["id"]] = expected
    if len(targets) != expected_rows:
        raise ValueError("DEV target denominator differs")
    return targets


def generate_source(model, tokenizer, source: str) -> dict:
    import torch
    encoded = tokenizer(source, return_tensors="pt", add_special_tokens=True,
                        truncation=False, split_special_tokens=True)
    input_tokens = int(encoded["input_ids"].shape[1])
    if input_tokens > MAX_TOKENS:
        return {"generated_text": "", "token_ids": [], "reason": "input_too_long",
                "eos_completed": False, "input_tokens": input_tokens}
    with torch.inference_mode():
        sequences = model.generate(**encoded, max_new_tokens=MAX_TOKENS, num_beams=1,
                                   do_sample=False, early_stopping=False)
    token_ids = [int(value) for value in sequences[0].tolist()]
    eos = tokenizer.eos_token_id in token_ids[1:]
    text = tokenizer.decode(token_ids, skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)
    return {"generated_text": text, "token_ids": token_ids,
            "reason": "complete" if eos else "max_tokens_without_eos",
            "eos_completed": eos, "input_tokens": input_tokens}


def capture_sources(sources: list[dict], predictor, path: Path) -> list[dict]:
    captures = []
    for row in sources:
        started = time.monotonic()
        result = predictor(row["source"])
        capture = {"id": row["id"], "source": row["source"], **result,
                   "latency_s": time.monotonic() - started}
        captures.append(capture)
        append_jsonl(path, capture)
    return captures


def score_closed_capture(capture_path: Path, sources: list[dict], target_loader) -> tuple[dict, list[dict]]:
    raw = Path(capture_path).read_bytes()
    capture_sha = hashlib.sha256(raw).hexdigest()
    captures = [json.loads(line) for line in raw.splitlines() if line]
    if (len(captures) != len(sources)
            or [(row.get("id"), row.get("source")) for row in captures]
            != [(row["id"], row["source"]) for row in sources]):
        raise ValueError("source-only capture is incomplete or reordered")
    for row in captures:
        if (not isinstance(row.get("generated_text"), str)
                or not isinstance(row.get("token_ids"), list)
                or any(not isinstance(value, int) or isinstance(value, bool)
                       for value in row["token_ids"])
                or row.get("reason") not in {
                    "complete", "max_tokens_without_eos", "input_too_long"}
                or not isinstance(row.get("eos_completed"), bool)
                or (row["reason"] == "complete") != row["eos_completed"]
                or not isinstance(row.get("input_tokens"), int)
                or isinstance(row.get("input_tokens"), bool) or row["input_tokens"] <= 0
                or not isinstance(row.get("latency_s"), (int, float))
                or isinstance(row.get("latency_s"), bool)
                or not math.isfinite(row["latency_s"]) or row["latency_s"] < 0):
            raise ValueError("source-only capture payload is malformed")
        token_ids = row["token_ids"]
        if row["reason"] == "input_too_long":
            valid = (row["input_tokens"] > MAX_TOKENS and token_ids == []
                     and row["generated_text"] == "")
        elif row["reason"] == "complete":
            valid = (row["input_tokens"] <= MAX_TOKENS and 2 <= len(token_ids) <= 513
                     and token_ids[0] == 0 and token_ids[-1] == 1)
        else:
            valid = (row["input_tokens"] <= MAX_TOKENS and len(token_ids) == 513
                     and token_ids[0] == 0 and 1 not in token_ids[1:])
        if not valid:
            raise ValueError("capture token/reason/EOS relationship differs")
    targets = target_loader()
    if sha256_file(capture_path) != capture_sha:
        raise ValueError("source-only capture changed during late target join")
    outcomes, correct = [], 0
    complete = literal_exact = raw_text_match = 0
    for capture in captures:
        expected = targets[capture["id"]]
        is_complete = capture.get("reason") == "complete" and capture.get("eos_completed") is True
        text_match = capture.get("generated_text") in expected
        exact = is_complete and text_match
        literal_ok = literal_signature(capture["source"]) == literal_signature(
            capture.get("generated_text", ""))
        correct += int(exact); complete += int(is_complete)
        raw_text_match += int(text_match); literal_exact += int(literal_ok)
        outcomes.append({"id": capture["id"], "correct": exact,
                         "raw_text_match": text_match, "eos_completed": is_complete,
                         "literal_signature_exact": literal_ok})
    rows = len(sources)
    metrics = {"rows": rows, "whole_exact": correct, "raw_text_match": raw_text_match,
               "eos_complete": complete, "incomplete_or_invalid": rows - complete,
               "literal_signature_exact": literal_exact,
               "all_captures_closed_before_target_join": True,
               "capture_sha_unchanged_after_target_join": True}
    return metrics, outcomes


def load_runtime(model_dir: Path):
    from transformers import T5ForConditionalGeneration, T5TokenizerFast
    tokenizer = T5TokenizerFast.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False)
    preflight.adapt_lossless_tokenizer(tokenizer)
    model = T5ForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        weights_only=True, torch_dtype="float32")
    model.eval()
    return model, tokenizer


def run(training_run: Path, output: Path, *, runtime_loader=load_runtime,
        source_loader=read_dev_sources, target_loader=load_dev_targets) -> dict:
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_snapshot = output / "evaluator-source.py"
    shutil.copyfile(Path(__file__), source_snapshot)
    with source_snapshot.open("rb") as stream:
        os.fsync(stream.fileno())
    source_sha = sha256_file(Path(__file__))
    try:
        authorization = validate_training_run(training_run)  # must precede model load
        sources = source_loader()
    except BaseException as error:
        write_json(output / "report.json", {"schema": SCHEMA, "completed": False,
            "error_type": type(error).__name__, "error": str(error),
            "capture_rows": 0, "test_opened": False, "production_changed": False})
        raise
    started, baseline_swap = time.monotonic(), preflight.system_swap_bytes()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    capture_path = output / "capture.jsonl"
    try:
        with EvaluationWatchdog(
                started, baseline_swap, output / "resources.jsonl", capture_path,
                limits=LIMITS) as watchdog:
            with watchdog.operation("load_trained_model"):
                model, tokenizer = runtime_loader(authorization["model_dir"])
            runtime_parameter_sha = preflight.parameter_fingerprint(model)
            if runtime_parameter_sha != authorization["report"]["model"]["parameter_sha256"]:
                raise ValueError("loaded pko parameter fingerprint differs")
            def predict(source):
                with watchdog.operation("greedy_generation"):
                    return generate_source(model, tokenizer, source)
            captures = capture_sources(sources, predict, capture_path)
            capture_sha = sha256_file(capture_path)
            write_json(output / "capture-complete.json", {
                "rows": len(captures), "sha256": capture_sha,
                "closed_before_target_join": True,
            })
            del predict, model
            gc.collect()
            metrics, outcomes = score_closed_capture(capture_path, sources, target_loader)
            watchdog.targets_joined = True
            observation = watchdog._sample("final")
        if sha256_file(Path(__file__)) != source_sha:
            raise RuntimeError("evaluator source changed during execution")
        for path_name, digest in authorization["pins"].items():
            path = ROOT / path_name if not Path(path_name).is_absolute() else Path(path_name)
            if sha256_file(path) != digest:
                raise RuntimeError(f"training artifact changed during evaluation: {path_name}")
        report = {
            "schema": SCHEMA, "completed": True, "promoted": False,
            "production_changed": False,
            "training_artifact": {"path": authorization["training_run"],
                                  "report_sha256": authorization["pins"][preflight.path_key(
                                      Path(authorization["training_run"]) / "report.json")]},
            "scope": {"dev_rows": len(sources),
                      "dev_file_parsed_including_label_fields": True,
                      "source_projection_used_for_inference": ["id", "input"],
                      "targets_joined_after_complete_capture": True,
                      "model_released_before_target_join": True,
                      "labels_used_as_model_context": False, "test_opened": False},
            "generation": {"num_beams": 1, "do_sample": False,
                           "max_new_tokens": MAX_TOKENS, "input_truncation": False},
            "metrics": metrics, "outcomes": outcomes,
            "literal_diagnostic": "uppercase/digit/punctuation signature equality; not a protected-span guarantee",
            "pins": {**authorization["pins"], preflight.path_key(DEV): DEV_SHA256,
                     preflight.path_key(Path(__file__)): source_sha},
            "capture": {"path": capture_path.name, "sha256": capture_sha,
                        "rows": len(captures)},
            "runtime_parameter_sha256": runtime_parameter_sha,
            "source_snapshot": {"path": source_snapshot.name,
                                "sha256": sha256_file(source_snapshot)},
            "resources": {"limits": LIMITS, "baseline_swap_bytes": baseline_swap,
                          "final": observation,
                          "trace_sha256": sha256_file(output / "resources.jsonl")},
            "elapsed_s": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(output / "report.json", report)
        return report
    except BaseException as error:
        write_json(output / "report.json", {"schema": SCHEMA, "completed": False,
            "error_type": type(error).__name__, "error": str(error),
            "capture_rows": (len(captures) if "captures" in locals() else 0),
            "test_opened": False, "production_changed": False})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.training_run, args.output)
    print(json.dumps({"completed": report["completed"], "metrics": report["metrics"],
                      "elapsed_s": report["elapsed_s"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
