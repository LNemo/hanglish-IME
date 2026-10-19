"""Bounded training and source-only DEV evaluation for SentenceCTC."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from hanglish.core.phonology import romanize_word
from hanglish.models.sentence_ctc import (
    CTC_VOCAB, MAX_TARGET_TOKENS, SentenceCTC, SentenceCTCConfig, training_batch,
)
from scripts.evaluate_ensemble_regression import rss_bytes
from scripts.preflight_sentence_ctc import estimate as preflight_estimate, frame_audit
from scripts.sentence_data import DEFAULT_CORPUS, read_split
from scripts.sentence_transducer_recovery import restore_checkpoint, save_checkpoint
from scripts.train_sentence_transducer import (
    BATCH, CHECKPOINT_INTERVAL, LENGTH_BINS, MAX_EVAL_SECONDS, MAX_RSS,
    MAX_TRAIN_SECONDS, SEED, batches, check_pins, read_training,
    sha, write_json,
)


EXPECTED_TRAINING_ROWS = 6680
EXPECTED_PARAMETERS = 1_002_757
EPOCHS = 8
EXTRA = ROOT / "data/challenge/sentence-train-extra-v2"
PREFLIGHT = ROOT / ".omx/artifacts/sentence-ctc-preflight-v1/report.json"
LATIN = re.compile(r"[A-Za-z]+")
NUMERIC = re.compile(r"\d+(?:[.,]\d+)*|[%㎡㎢㎖㎞]")
RECOVERY_SCOPE = (
    "optimizer/model/RNG/cursor recovery during training; interruption after final "
    "model serialization or during DEV capture requires a new output directory")


def preflight_pins(extra=EXTRA):
    paths = [
        ROOT / "scripts/preflight_sentence_ctc.py",
        ROOT / "hanglish/models/sentence_ctc.py",
        ROOT / "hanglish/models/sentence_transducer.py",
        ROOT / "scripts/train_sentence_transducer.py",
        ROOT / "scripts/sentence_data.py",
        ROOT / "scripts/evaluate_ensemble_regression.py",
        DEFAULT_CORPUS / "train.jsonl",
        DEFAULT_CORPUS / "dev.jsonl",
        DEFAULT_CORPUS / "manifest.json",
        extra / "train.jsonl",
        extra / "manifest.json",
    ]
    return {str(path): sha(path) for path in paths}


def input_pins(extra=EXTRA):
    pins = preflight_pins(extra)
    for path in (
        Path(__file__), ROOT / "scripts/sentence_transducer_recovery.py",
        ROOT / "hanglish/core/phonology.py",
    ):
        pins[str(path)] = sha(path)
    return pins


def _finite(value, *, positive=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def _finite_signed(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _validate_generation_response(response, config):
    if (not isinstance(response, dict)
            or response.get("reason") not in {
                "complete", "empty_output", "invalid_utf8", "invalid_jamo", "no_valid_path"}
            or not isinstance(response.get("labels"), list)
            or len(response["labels"]) > MAX_TARGET_TOKENS
            or any(isinstance(label, bool) or not isinstance(label, int)
                   or not 1 <= label < CTC_VOCAB
                   for label in response["labels"])
            or not isinstance(response.get("generated_tokens"), int)
            or response["generated_tokens"] != len(response["labels"])
            or response.get("frames") != 3 * (config.max_source_tokens - 2)
            or response.get("decode_mode") != "ctc_prefix_beam"):
        raise ValueError("maximum CTC generation response differs")
    if (response["reason"] == "complete") != isinstance(response.get("text"), str):
        raise ValueError("maximum CTC generation completion payload differs")
    score = response.get("log_probability")
    if score is not None and not _finite_signed(score):
        raise ValueError("maximum CTC beam score is invalid")
    beam = response.get("beam")
    if (not isinstance(beam, dict) or beam.get("frames") != response["frames"]
            or beam.get("beam_size") != config.beam_size
            or beam.get("token_shortlist") != config.token_shortlist
            or beam.get("max_label_tokens") != MAX_TARGET_TOKENS
            or not all(isinstance(beam.get(key), bool) for key in (
                "shortlist_pruned", "beam_pruned", "approximate_pruning"))
            or not isinstance(beam.get("repeat_tokens_restored"), int)
            or beam["repeat_tokens_restored"] < 0):
        raise ValueError("maximum CTC beam audit differs")
    greedy = response.get("greedy")
    if (not isinstance(greedy, dict)
            or greedy.get("reason") not in {
                "complete", "empty_output", "invalid_utf8", "invalid_jamo"}
            or not isinstance(greedy.get("labels"), list)
            or greedy.get("generated_tokens") != len(greedy["labels"])
            or (greedy["reason"] == "complete") != isinstance(greedy.get("text"), str)):
        raise ValueError("maximum CTC greedy diagnosis differs")


def qualify(path, rows, config=SentenceCTCConfig(), extra=EXTRA):
    """Recompute every bounded preflight claim used to authorize training."""
    path = Path(path)
    report = json.loads(path.read_text())
    expected_pins = preflight_pins(extra)
    if (report.get("completed") is not True
            or report.get("phase") != "ctc_preflight"
            or report.get("config") != asdict(config)
            or report.get("training_rows") != len(rows)
            or report.get("pins") != expected_pins):
        raise ValueError("matching completed CTC preflight is required")
    expected_audit = frame_audit(rows)
    if (report.get("frame_audit") != expected_audit
            or expected_audit["invalid"] != 0
            or expected_audit["rows"] != len(rows)
            or expected_audit["max_source_tokens"] > config.max_source_tokens
            or expected_audit["max_target_tokens"] > MAX_TARGET_TOKENS):
        raise ValueError("CTC frame feasibility audit differs")
    if (report.get("parameters") != EXPECTED_PARAMETERS
            or report.get("reload_max_logit_delta") != 0):
        raise ValueError("CTC parameter or serialization evidence differs")
    peak, elapsed = report.get("peak_rss_bytes"), report.get("elapsed_s")
    if (not _finite(peak, positive=True) or peak > MAX_RSS
            or not _finite(elapsed, positive=True) or elapsed > 180):
        raise ValueError("CTC preflight resource evidence exceeds its bound")

    real = report.get("real_update")
    if (not isinstance(real, dict) or real.get("rows") != BATCH
            or not all(_finite(real.get(key), positive=True)
                       for key in ("elapsed_s", "loss", "gradient_norm"))
            or not isinstance(real.get("source_tokens"), int)
            or not 3 <= real["source_tokens"] <= config.max_source_tokens
            or not isinstance(real.get("max_target_tokens"), int)
            or not 1 <= real["max_target_tokens"] <= MAX_TARGET_TOKENS
            or real.get("max_input_frames") != 3 * (real["source_tokens"] - 2)):
        raise ValueError("real CTC optimizer update evidence differs")

    updates = report.get("updates")
    if not isinstance(updates, list) or len(updates) != len(LENGTH_BINS):
        raise ValueError("four CTC length-bin updates are required")
    for step, length in zip(updates, LENGTH_BINS):
        if (step.get("kind") != "synthetic_length_bin"
                or step.get("rows") != BATCH or step.get("source_tokens") != length
                or step.get("max_target_tokens") != length - 1
                or step.get("max_input_frames") != 3 * (length - 2)
                or not all(_finite(step.get(key), positive=True)
                           for key in ("elapsed_s", "loss", "gradient_norm"))):
            raise ValueError("CTC synthetic length-bin evidence differs")

    generation = report.get("generation_probe")
    if (not isinstance(generation, dict) or generation.get("batch_size") != 8
            or generation.get("source_tokens") != config.max_source_tokens
            or not _finite(generation.get("elapsed_s"), positive=True)
            or not isinstance(generation.get("responses"), list)
            or len(generation["responses"]) != 8):
        raise ValueError("maximum CTC generation evidence differs")
    for response in generation["responses"]:
        _validate_generation_response(response, config)

    estimated_8epoch = preflight_estimate(updates, rows, epochs=EPOCHS)
    dev_rows = json.loads((DEFAULT_CORPUS / "manifest.json").read_text())["counts"]["dev"]
    estimated_dev = generation["elapsed_s"] * math.ceil(dev_rows / 8) * 1.25
    if (not math.isclose(report.get("estimated_8epoch_training_s", math.nan),
                         estimated_8epoch, rel_tol=0, abs_tol=1e-9)
            or not math.isclose(report.get("estimated_dev_generation_s", math.nan),
                                estimated_dev, rel_tol=0, abs_tol=1e-9)
            or estimated_8epoch > MAX_TRAIN_SECONDS or estimated_dev > MAX_EVAL_SECONDS
            or report.get("qualifies") is not True):
        raise ValueError("recomputed CTC training or DEV estimate does not qualify")
    check_pins(expected_pins)
    return report


def optimizer_step(model, optimizer, rows):
    model.train()
    source, labels, input_lengths, target_lengths = training_batch(
        [(row["input"], row["expected"][0]) for row in rows], model.config)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    loss = model.loss(source, labels, input_lengths, target_lengths)
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite CTC loss")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(gradient_norm):
        raise RuntimeError("nonfinite CTC gradient")
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm),
        "elapsed_s": time.monotonic() - started,
        "rows": len(rows),
        "source_tokens": int(source.shape[1]),
        "max_input_frames": int(input_lengths.max()),
        "max_target_tokens": int(target_lengths.max()),
        "nonpadding_target_tokens": int(target_lengths.sum()),
    }


def recovery_identity(rows, pins, config, qualification_path, epochs):
    schedules = [
        [[row["id"] for row in group] for group in batches(rows, epoch)]
        for epoch in range(epochs)
    ]
    return {
        "architecture": "sentence_ctc_v1",
        "pins": pins,
        "config": asdict(config),
        "seed": SEED,
        "batch_size": BATCH,
        "epochs": epochs,
        "qualification_sha256": sha(qualification_path),
        "training_rows": len(rows),
        "row_ids_sha256": hashlib.sha256(json.dumps(
            [row["id"] for row in rows]).encode()).hexdigest(),
        "schedule_sha256": hashlib.sha256(json.dumps(schedules).encode()).hexdigest(),
    }


def validate_cursor(state, rows, epochs):
    epoch, batch_index = state["next_epoch"], state["next_batch"]
    if (not isinstance(epoch, int) or isinstance(epoch, bool)
            or not 0 <= epoch <= epochs
            or not isinstance(batch_index, int) or isinstance(batch_index, bool)):
        raise ValueError("invalid CTC recovery cursor")
    if epoch == epochs:
        if batch_index != 0:
            raise ValueError("completed CTC cursor has a pending batch")
    elif not 0 <= batch_index < len(batches(rows, epoch)):
        raise ValueError("CTC recovery batch cursor exceeds schedule")
    expected = sum(len(batches(rows, index)) for index in range(epoch)) + batch_index
    if expected != state["updates"] or len(state["epochs_report"]) != epoch:
        raise ValueError("CTC recovery cursor/update accounting differs")
    if batch_index == 0 and (state["epoch_loss_sum"] != 0
                             or state["epoch_token_sum"] != 0):
        raise ValueError("new CTC epoch has uncommitted loss totals")


def _resource_guard(training_elapsed, peak_rss):
    if training_elapsed > MAX_TRAIN_SECONDS or peak_rss > MAX_RSS:
        raise RuntimeError("CTC training exceeded 1800seconds/3GiB")


def _sequence_metrics(text, reference, pattern):
    expected = pattern.findall(reference)
    return bool(expected), text is not None and pattern.findall(text) == expected


def evaluate(model, output):
    rows = read_split(DEFAULT_CORPUS, "dev")
    inputs = [{"id": row["id"], "input": row["input"]} for row in rows]
    predictions = []
    started = time.monotonic()
    with (output / "dev-predictions.jsonl").open("x") as stream:
        for offset in range(0, len(inputs), 8):
            items = inputs[offset:offset + 8]
            responses = model.generate([item["input"] for item in items])
            if len(responses) != len(items):
                raise RuntimeError("CTC DEV generation batch count differs")
            for item, response in zip(items, responses):
                prediction = {**item, "response": response}
                predictions.append(prediction)
                stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            stream.flush()
            elapsed = time.monotonic() - started
            if elapsed > MAX_EVAL_SECONDS or rss_bytes() > MAX_RSS:
                raise RuntimeError("CTC DEV generation exceeded 900seconds/3GiB")
            if len(predictions) % 40 == 0:
                print(json.dumps({"phase": "ctc_dev_generation",
                                  "captured": len(predictions), "total": len(inputs)}),
                      flush=True)

    outcomes = []
    literal_rows = literal_exact = numeric_rows = numeric_exact = 0
    reasons = Counter()
    greedy_reasons = Counter()
    for prediction, row in zip(predictions, rows):
        response = prediction["response"]
        greedy = response["greedy"]
        reference = row["expected"][0]
        has_literal, literal_match = _sequence_metrics(response["text"], reference, LATIN)
        has_numeric, numeric_match = _sequence_metrics(response["text"], reference, NUMERIC)
        literal_rows += has_literal
        literal_exact += has_literal and literal_match
        numeric_rows += has_numeric
        numeric_exact += has_numeric and numeric_match
        reasons[response["reason"]] += 1
        greedy_reasons[greedy["reason"]] += 1
        outcomes.append({
            "id": row["id"],
            "expected": row["expected"],
            "text": response["text"],
            "reason": response["reason"],
            "correct": response["text"] in row["expected"],
            "forward_equal": (response["text"] is not None
                              and romanize_word(response["text"]) == row["input"]),
            "greedy_text": greedy["text"],
            "greedy_reason": greedy["reason"],
            "greedy_correct": greedy["text"] in row["expected"],
        })
    return {
        "rows": len(rows),
        "whole_correct": sum(item["correct"] for item in outcomes),
        "valid_complete": reasons["complete"],
        "forward_equal": sum(item["forward_equal"] for item in outcomes),
        "greedy_whole_correct": sum(item["greedy_correct"] for item in outcomes),
        "greedy_valid_complete": greedy_reasons["complete"],
        "reasons": dict(reasons),
        "greedy_reasons": dict(greedy_reasons),
        "literal_exact": {"rows": literal_rows, "correct": literal_exact},
        "numeric_exact": {"rows": numeric_rows, "correct": numeric_exact},
        "elapsed_s": time.monotonic() - started,
        "labels_joined_after_complete_capture": True,
        "labels_opened_after_complete_capture": False,
        "gold_not_passed_to_predictor": True,
        "outcomes": outcomes,
    }


def train(output, rows, pins, config, qualification_path, epochs=EPOCHS,
          resume=False, extra=EXTRA):
    if not 1 <= epochs <= 20:
        raise ValueError("CTC epochs must be between 1 and 20")
    qualified = qualify(qualification_path, rows, config, extra)
    estimated = preflight_estimate(qualified["updates"], rows, epochs=epochs)
    if estimated > MAX_TRAIN_SECONDS:
        raise ValueError("planned CTC training exceeds the 1800-second estimate bound")
    model = SentenceCTC(config)
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    if (qualified.get("parameters") is not None
            and qualified["parameters"] != model_parameters):
        raise ValueError("qualified CTC parameter count differs from training model")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
    identity = recovery_identity(rows, pins, config, qualification_path, epochs)
    state = {
        "updates": 0, "next_epoch": 0, "next_batch": 0,
        "epochs_report": [], "epoch_loss_sum": 0., "epoch_token_sum": 0,
        "training_elapsed_s": 0., "peak_rss_bytes": 0,
        "discarded_tail_updates": 0,
    }
    if resume:
        state = restore_checkpoint(output, model, optimizer, identity)
        validate_cursor(state, rows, epochs)
    _resource_guard(state["training_elapsed_s"], state["peak_rss_bytes"])
    prior_elapsed = state["training_elapsed_s"]
    started = time.monotonic()
    with (output / "updates.jsonl").open("a" if resume else "x") as stream:
        for epoch in range(state["next_epoch"], epochs):
            schedule = batches(rows, epoch)
            first = state["next_batch"] if epoch == state["next_epoch"] else 0
            for batch_index in range(first, len(schedule)):
                group = schedule[batch_index]
                step = optimizer_step(model, optimizer, group)
                state["updates"] += 1
                state["training_elapsed_s"] = prior_elapsed + time.monotonic() - started
                state["peak_rss_bytes"] = max(state["peak_rss_bytes"], rss_bytes())
                step.update({
                    "step": state["updates"], "epoch": epoch + 1,
                    "ids": [row["id"] for row in group],
                    "training_elapsed_s": state["training_elapsed_s"],
                    "peak_rss_bytes": state["peak_rss_bytes"],
                })
                stream.write(json.dumps(step) + "\n")
                stream.flush()
                state["epoch_loss_sum"] += step["loss"] * step["nonpadding_target_tokens"]
                state["epoch_token_sum"] += step["nonpadding_target_tokens"]
                if batch_index + 1 == len(schedule):
                    state["epochs_report"].append({
                        "epoch": epoch + 1,
                        "target_normalized_ctc_loss": (
                            state["epoch_loss_sum"] / state["epoch_token_sum"]),
                        "tokens": state["epoch_token_sum"],
                    })
                    state.update({"next_epoch": epoch + 1, "next_batch": 0,
                                  "epoch_loss_sum": 0., "epoch_token_sum": 0})
                else:
                    state.update({"next_epoch": epoch, "next_batch": batch_index + 1})
                if (state["updates"] % CHECKPOINT_INTERVAL == 0
                        or state["next_batch"] == 0):
                    save_checkpoint(output, model, optimizer, identity, state, stream)
                _resource_guard(state["training_elapsed_s"], state["peak_rss_bytes"])
                if state["updates"] % 50 == 0:
                    print(json.dumps({"phase": "ctc_training", "step": state["updates"],
                                      "epoch": epoch + 1, "loss": step["loss"],
                                      "elapsed_s": state["training_elapsed_s"]}), flush=True)

    expected_updates = sum(len(batches(rows, epoch)) for epoch in range(epochs))
    if state["updates"] != expected_updates or state["next_epoch"] != epochs:
        raise RuntimeError("CTC training schedule did not complete exactly once")
    check_pins(pins)
    model.save(output / "model.pt", {
        "input_contract": "roman_sentence_only", "pins": pins,
        "qualification_sha256": sha(qualification_path), "seed": SEED,
        "epochs": epochs, "updates": state["updates"],
        "training_rows": len(rows), "warm_start": False,
        "recovery_scope": RECOVERY_SCOPE,
    })
    model = SentenceCTC.load(output / "model.pt")
    training_elapsed = prior_elapsed + time.monotonic() - started
    training_peak_rss = max(state["peak_rss_bytes"], rss_bytes())
    _resource_guard(training_elapsed, training_peak_rss)
    metrics = evaluate(model, output)
    check_pins(pins)
    report = {
        "completed": True,
        "phase": "ctc_trained_and_evaluated",
        "input_contract": "roman_sentence_only",
        "config": asdict(config),
        "pins": pins,
        "qualification_sha256": sha(qualification_path),
        "qualification_recomputed": True,
        "epochs": state["epochs_report"],
        "updates": state["updates"],
        "expected_updates": expected_updates,
        "training_rows": len(rows),
        "parameters": model_parameters,
        "seed": SEED,
        "batch_size": BATCH,
        "optimizer": {"kind": "AdamW", "learning_rate": 3e-4,
                      "weight_decay": .01, "gradient_clip_norm": 1.},
        "row_ids_sha256": identity["row_ids_sha256"],
        "schedule_sha256": identity["schedule_sha256"],
        "training_elapsed_s": training_elapsed,
        "estimated_training_s": estimated,
        "dev": metrics,
        "peak_rss_bytes": max(training_peak_rss, rss_bytes()),
        "resource_limits": {"training_seconds": MAX_TRAIN_SECONDS,
                            "dev_seconds": MAX_EVAL_SECONDS,
                            "rss_bytes": MAX_RSS, "cpu_threads": 2},
        "test_accessed": False,
        "discarded_tail_updates": state.get("discarded_tail_updates", 0),
        "resumed": resume,
        "warm_start": False,
        "recovery_scope": RECOVERY_SCOPE,
        "production_changed": False,
        "promoted": False,
        "model_sha256": sha(output / "model.pt"),
        "model_metadata_sha256": sha(output / "model.json"),
        "predictions_sha256": sha(output / "dev-predictions.jsonl"),
    }
    write_json(output / "report.json", report)
    print(json.dumps({
        "completed": True, "updates": state["updates"],
        "whole_correct": metrics["whole_correct"], "rows": metrics["rows"],
        "valid_complete": metrics["valid_complete"],
        "greedy_whole_correct": metrics["greedy_whole_correct"],
        "training_s": training_elapsed, "generation_s": metrics["elapsed_s"],
    }), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra", type=Path, default=EXTRA)
    parser.add_argument("--qualification", type=Path, default=PREFLIGHT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    random.seed(SEED)
    rows = read_training(args.extra)
    if len(rows) != EXPECTED_TRAINING_ROWS:
        raise ValueError("CTC control requires the pinned 6680-row TRAIN union")
    # Independently apply the CTC-specific frame and target-length audit.
    for row in rows:
        training_batch([(row["input"], row["expected"][0])])
    pins = input_pins(args.extra)
    config = SentenceCTCConfig()
    if args.resume:
        if (not (args.output / "recovery.pt").is_file()
                or (args.output / "report.json").exists()
                or (args.output / "failure.json").exists()
                or (args.output / "model.pt").exists()
                or (args.output / "dev-predictions.jsonl").exists()):
            raise ValueError("resume requires an interrupted training-phase CTC checkpoint")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    lock = (args.output / ".run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("CTC training output is already owned by a live process")
    try:
        train(args.output, rows, pins, config, args.qualification, args.epochs,
              args.resume, args.extra)
    except Exception as error:
        write_json(args.output / "failure.json", {
            "completed": False, "error_type": type(error).__name__,
            "error": str(error), "pins": pins,
            "recovery_scope": RECOVERY_SCOPE,
            "production_changed": False,
        })
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
