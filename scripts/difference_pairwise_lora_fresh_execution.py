"""Bounded, target-closed execution core for pairwise LoRA fresh evaluation.

This module deliberately does not load MLX, a tokenizer, an adapter, source
artifacts, or accepted answers.  The outer worker must independently replay
tokenizer qualification, prove model and adapter identity, and enforce the
actual resource and source-hash gates.  Here, a qualification record is bound
to the complete reconstructed pair and checked structurally; its token IDs are
not treated as proof that a trusted tokenizer produced them.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import hashlib
import json
import math
import os
import time

from scripts.difference_pairwise_lora_evaluation import canonical, resolve_selection
from scripts.difference_pairwise_lora_fresh_metrics import (
    FRESH_GROUPS,
    FRESH_ROLE,
    FRESH_ROWS,
    PRESERVATION_ROLE,
    PRESERVATION_ROWS,
    TOTAL_ROWS,
    score_closed_predictions,
)
from scripts.difference_pairwise_lora_inference import prepare_inference_selection
from scripts.prepare_difference_pairwise_lora_data import MAX_PROMPT_TOKENS
from scripts.verify_difference_pairwise_lora_fresh_metrics import verify_reported_metrics


SCHEMA = "difference-pairwise-lora-fresh-execution-v1"
PAIR_INFERENCE_SCHEMA = "difference-pairwise-lora-fresh-pair-inference-v1"
QUALIFICATION_FIELDS = {
    "pair_id",
    "pair_sha256",
    "messages",
    "tokens",
    "prompt_tokens",
    "digit_continuation_ids",
    "rendered_prefix",
}
DIGIT_IDS = {"1": 16, "2": 17}


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value) -> str:
    return _digest_bytes(canonical(value))


def _safe_error(error: BaseException) -> dict:
    return {"type": type(error).__name__, "message": str(error)}


def _now(monotonic: Callable[[], float]) -> float:
    value = monotonic()
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("monotonic clock returned a nonfinite value")
    return float(value)


def _passed_evidence(callback: Callable[[str], Mapping], stage: str) -> dict:
    evidence = callback(stage)
    if not isinstance(evidence, Mapping) or evidence.get("passed") is not True:
        raise ValueError(f"{stage} callback did not return passing evidence")
    evidence = dict(evidence)
    canonical(evidence)
    return evidence


def _released_evidence(callback: Callable[[], Mapping]) -> dict:
    evidence = callback()
    if not isinstance(evidence, Mapping) or evidence.get("released") is not True:
        raise ValueError("model release callback did not return released evidence")
    evidence = dict(evidence)
    canonical(evidence)
    return evidence


class _ExclusiveJsonl:
    def __init__(self, path: Path):
        self.path = path
        self.stream = path.open("xb")
        self.count = 0
        self.byte_count = 0
        self.hasher = hashlib.sha256()
        self.closed = False

    def write(self, value: Mapping) -> dict:
        payload = canonical(value)
        framed = payload + b"\n"
        self.stream.write(framed)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.hasher.update(framed)
        self.byte_count += len(framed)
        self.count += 1
        reopened = json.loads(payload)
        if canonical(reopened) != payload:
            raise ValueError("serialized JSONL record does not round-trip exactly")
        return reopened

    def close(self) -> None:
        if self.closed:
            return
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        self.closed = True

    def integrity(self) -> dict:
        return {
            "records": self.count,
            "bytes": self.byte_count,
            "sha256": self.hasher.hexdigest(),
        }


def _close_writer(writer: _ExclusiveJsonl | None) -> BaseException | None:
    if writer is None:
        return None
    try:
        writer.close()
    except BaseException as error:  # preserve closure failure in the terminal report
        return error
    return None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_report(path: Path, report: Mapping) -> None:
    payload = canonical(report)
    with path.open("xb") as stream:
        stream.write(payload + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _reopen_canonical_jsonl(path: Path, expected: Mapping) -> list[dict]:
    payload = path.read_bytes()
    if (
        len(payload) != expected.get("bytes")
        or _digest_bytes(payload) != expected.get("sha256")
        or (payload and not payload.endswith(b"\n"))
    ):
        raise ValueError(f"closed {path.name} byte integrity differs")
    lines = payload.splitlines()
    if len(lines) != expected.get("records"):
        raise ValueError(f"closed {path.name} record count differs")
    records = []
    for line in lines:
        try:
            record = json.loads(line)
        except (TypeError, ValueError) as error:
            raise ValueError(f"closed {path.name} JSON differs") from error
        if not isinstance(record, dict) or canonical(record) != line:
            raise ValueError(f"closed {path.name} canonical record differs")
        records.append(record)
    return records


def _validate_qualification(pair: Mapping, value: Mapping) -> dict:
    if not isinstance(value, Mapping) or set(value) != QUALIFICATION_FIELDS:
        raise ValueError("qualified prompt fields differ")
    record = dict(value)
    tokens = record.get("tokens")
    messages = record.get("messages")
    rendered = record.get("rendered_prefix")
    if (
        record.get("pair_id") != pair.get("pair_id")
        or record.get("pair_sha256") != _digest(pair)
        or not isinstance(messages, list)
        or not messages
        or any(not isinstance(item, dict) for item in messages)
        or not isinstance(tokens, list)
        or not tokens
        or any(type(token) is not int for token in tokens)
        or len(tokens) > MAX_PROMPT_TOKENS
        or record.get("prompt_tokens") != len(tokens)
        or record.get("digit_continuation_ids") != DIGIT_IDS
        or not isinstance(rendered, str)
        or not rendered
    ):
        raise ValueError("qualified prompt does not bind the pair or token contract")
    canonical(record)
    return record


def _prepare_all(rows: Sequence[Mapping], qualified_by_pair_id: Mapping) -> list[dict]:
    if not isinstance(rows, (list, tuple)) or len(rows) != TOTAL_ROWS:
        raise ValueError("exactly 888 source rows are required")
    if not isinstance(qualified_by_pair_id, Mapping) or any(
        not isinstance(key, str) or not key for key in qualified_by_pair_id
    ):
        raise ValueError("qualified prompts must be a pair-id mapping")
    expected_roles = [FRESH_ROLE] * FRESH_ROWS + [PRESERVATION_ROLE] * PRESERVATION_ROWS
    selections = []
    seen_rows = set()
    expected_pairs = {}
    fresh_groups = set()
    for row, role in zip(rows, expected_roles):
        if not isinstance(row, Mapping) or row.get("role") != role:
            raise ValueError("source cohort role or order differs")
        row_id = row.get("id")
        if not isinstance(row_id, str) or not row_id or row_id in seen_rows:
            raise ValueError("source row identity is malformed or duplicated")
        seen_rows.add(row_id)
        if role == FRESH_ROLE:
            fresh_groups.add(row.get("group_id"))
        selection = prepare_inference_selection(role, row)
        for pair in selection["pairs"]:
            pair_id = pair["pair_id"]
            if pair_id in expected_pairs:
                raise ValueError("permitted pair identity is duplicated")
            expected_pairs[pair_id] = pair
        selections.append(selection)
    if len(fresh_groups) != FRESH_GROUPS or None in fresh_groups:
        raise ValueError("fresh source group denominator differs")
    supplied = set(qualified_by_pair_id)
    expected = set(expected_pairs)
    if supplied != expected:
        raise ValueError("qualified prompts do not cover every permitted pair exactly")
    for pair_id, pair in expected_pairs.items():
        _validate_qualification(pair, qualified_by_pair_id[pair_id])
    return selections


def _parse_logits(value) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("scorer must return exactly KEEP and SWITCH logits")
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        raise ValueError("scorer returned malformed or nonfinite logits")
    keep, switch = float(value[0]), float(value[1])
    if not math.isfinite(switch - keep):
        raise ValueError("scorer returned a nonfinite logit margin")
    return keep, switch


def execute(
    rows: Sequence[Mapping],
    qualified_by_pair_id: Mapping[str, Mapping],
    output: Path | str,
    *,
    score_pair: Callable[[Mapping, Mapping], Sequence[float]],
    release_model: Callable[[], Mapping],
    load_targets: Callable[[], Sequence[Mapping]],
    checkpoint: Callable[[str], Mapping],
    validate_sources: Callable[[str], Mapping],
    monotonic: Callable[[], float] = time.monotonic,
) -> dict:
    """Run complete target-blind scoring, release the model, then score labels.

    Callbacks make the ordering testable.  ``checkpoint`` and
    ``validate_sources`` must return JSON evidence containing ``passed: true``;
    ``release_model`` must return JSON evidence containing ``released: true``.
    The outer worker remains responsible for proving those callback results.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = _now(monotonic)
    report = {
        "schema": SCHEMA,
        "completed": False,
        "terminal_status": "running",
        "execution_verified": False,
        "promotion_allowed": False,
        "target_accessed": False,
        "pair_stream_closed": False,
        "prediction_stream_closed": False,
        "output_directory_fsynced": False,
        "closed_streams_reopened_and_verified": False,
        "model_release_succeeded": False,
        "final_source_check_succeeded": False,
        "final_resource_check_succeeded": False,
        "pair_inference_count": 0,
        "prediction_count": 0,
        "events": [],
    }
    pair_writer = prediction_writer = None
    predictions = []
    terminal_error: BaseException | None = None

    def event(stage: str, **values) -> None:
        report["events"].append(
            {"sequence": len(report["events"]) + 1, "stage": stage,
             "elapsed_seconds": _now(monotonic) - started, **values}
        )

    try:
        report["initial_source_check"] = _passed_evidence(validate_sources, "before_scoring")
        report["initial_resource_check"] = _passed_evidence(checkpoint, "before_scoring")
        selections = _prepare_all(rows, qualified_by_pair_id)
        report["qualified_pair_count"] = sum(len(item["pairs"]) for item in selections)
        event("qualification_complete", rows=len(selections), pairs=report["qualified_pair_count"])
        pair_writer = _ExclusiveJsonl(output / "pair_inferences.jsonl")
        prediction_writer = _ExclusiveJsonl(output / "predictions.jsonl")
        for row_index, selection in enumerate(selections):
            logits_by_pair_id = {}
            for pair in selection["pairs"]:
                qualification = _validate_qualification(
                    pair, qualified_by_pair_id[pair["pair_id"]]
                )
                score_started = _now(monotonic)
                keep, switch = _parse_logits(score_pair(pair, qualification))
                elapsed = _now(monotonic) - score_started
                if elapsed < 0:
                    raise ValueError("monotonic clock moved backwards during inference")
                resource_evidence = _passed_evidence(
                    checkpoint, f"pair_{report['pair_inference_count'] + 1}_scored"
                )
                event(
                    "pair_scored",
                    pair=report["pair_inference_count"] + 1,
                    pair_id=pair["pair_id"],
                    resource=resource_evidence,
                )
                qualification_sha256 = _digest(qualification)
                inference = {
                    "schema": PAIR_INFERENCE_SCHEMA,
                    "sequence": report["pair_inference_count"] + 1,
                    "pair_id": pair["pair_id"],
                    "pair_sha256": qualification["pair_sha256"],
                    "qualification_sha256": qualification_sha256,
                    "row_id": selection["row_id"],
                    "role": selection["role"],
                    "alternative_source_index": pair["alternative_source_index"],
                    "prompt_tokens": qualification["prompt_tokens"],
                    "digit_continuation_ids": DIGIT_IDS,
                    "keep_logit": keep,
                    "switch_logit": switch,
                    "margin": switch - keep,
                    "inference_seconds": elapsed,
                    "resource_checkpoint": resource_evidence,
                }
                pair_writer.write(inference)
                report["pair_inference_count"] += 1
                logits_by_pair_id[pair["pair_id"]] = [keep, switch]
            prediction = resolve_selection(selection, logits_by_pair_id)
            if prediction.get("complete") is not True or prediction.get("failures") != []:
                raise ValueError("complete pair scores did not resolve successfully")
            round_tripped = prediction_writer.write(prediction)
            if canonical(round_tripped) != canonical(prediction):
                raise ValueError("serialized prediction differs from resolved selection")
            predictions.append(round_tripped)
            report["prediction_count"] = row_index + 1
            evidence = _passed_evidence(checkpoint, f"prediction_{row_index + 1}")
            event("prediction_serialized", row=row_index + 1, resource=evidence)
    except BaseException as error:
        terminal_error = error
    finally:
        pair_close_error = _close_writer(pair_writer)
        prediction_close_error = _close_writer(prediction_writer)
        report["pair_stream_closed"] = pair_writer is not None and pair_close_error is None
        report["prediction_stream_closed"] = (
            prediction_writer is not None and prediction_close_error is None
        )
        if report["pair_stream_closed"]:
            report["pair_stream_integrity"] = pair_writer.integrity()
        if report["prediction_stream_closed"]:
            report["prediction_stream_integrity"] = prediction_writer.integrity()
        if terminal_error is None:
            terminal_error = pair_close_error or prediction_close_error
        elif pair_close_error or prediction_close_error:
            report["secondary_close_errors"] = [
                _safe_error(error)
                for error in (pair_close_error, prediction_close_error)
                if error is not None
            ]
        if pair_writer is not None or prediction_writer is not None:
            try:
                _fsync_directory(output)
                report["output_directory_fsynced"] = True
            except BaseException as error:
                if terminal_error is None:
                    terminal_error = error
                else:
                    report["secondary_directory_fsync_error"] = _safe_error(error)
        try:
            report["model_release_evidence"] = _released_evidence(release_model)
            report["model_release_succeeded"] = True
            event("model_released")
        except BaseException as error:
            if terminal_error is None:
                terminal_error = error
            else:
                report["secondary_release_error"] = _safe_error(error)

    if terminal_error is None:
        try:
            if (
                report["prediction_count"] != TOTAL_ROWS
                or len(predictions) != TOTAL_ROWS
                or report["pair_inference_count"] != report["qualified_pair_count"]
                or not report["pair_stream_closed"]
                or not report["prediction_stream_closed"]
                or not report["output_directory_fsynced"]
                or not report["model_release_succeeded"]
            ):
                raise RuntimeError("closed prediction execution invariants differ")
            report["final_resource_check"] = _passed_evidence(checkpoint, "after_model_release")
            report["final_resource_check_succeeded"] = True
            report["final_source_check"] = _passed_evidence(
                validate_sources, "after_model_release"
            )
            report["final_source_check_succeeded"] = True
            event("closed_release_and_final_checks_complete")
            _reopen_canonical_jsonl(
                output / "pair_inferences.jsonl", report["pair_stream_integrity"]
            )
            reopened_predictions = _reopen_canonical_jsonl(
                output / "predictions.jsonl", report["prediction_stream_integrity"]
            )
            if len(reopened_predictions) != len(predictions) or any(
                canonical(reopened) != canonical(buffered)
                for reopened, buffered in zip(reopened_predictions, predictions)
            ):
                raise ValueError("closed predictions differ from serialized selections")
            predictions = reopened_predictions
            report["closed_streams_reopened_and_verified"] = True
            report["files"] = {
                "pair_inferences.jsonl": report["pair_stream_integrity"]["sha256"],
                "predictions.jsonl": report["prediction_stream_integrity"]["sha256"],
            }
            event("closed_streams_verified")
            report["target_accessed"] = True
            event("target_access_started")
            targets = load_targets()
            event("targets_loaded")
            metrics = score_closed_predictions(rows, predictions, targets)
            independent = verify_reported_metrics(rows, predictions, targets, metrics)
            report["metrics"] = metrics
            report["metrics_verification"] = independent
            report["completed"] = True
            report["terminal_status"] = "quality_passed" if metrics["gate"]["passed"] else "quality_rejected"
            event("metrics_verified", quality_gate_passed=metrics["gate"]["passed"])
        except BaseException as error:
            terminal_error = error

    if terminal_error is not None:
        report["terminal_status"] = "failed"
        report["error"] = _safe_error(terminal_error)
    report["elapsed_seconds"] = _now(monotonic) - started
    try:
        _write_report(output / "report.json", report)
    except BaseException as report_error:
        if terminal_error is None:
            raise
        report["report_write_error"] = _safe_error(report_error)
    if terminal_error is not None:
        raise terminal_error
    return report
