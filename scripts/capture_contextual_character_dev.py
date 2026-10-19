#!/usr/bin/env python3
"""Capture contextual-character predictions over the frozen 316-row DEV pool.

This producer never opens DEV labels and never regenerates candidates.  It
authenticates the independently verified parent capture, then stores only the
new contextual scores and public predictions while retaining the parent as a
pinned reference.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.current_recombination_selection import (  # noqa: E402
    calibrated_score,
    render_public,
    validate_model,
)
from scripts.train_character_edge_full_fit import ResourceGuard  # noqa: E402
from scripts.train_contextual_character import RECIPE  # noqa: E402


SCHEMA = "contextual-character-dev-capture-v1"
ROWS = 316
PARENT = ROOT / ".omx/artifacts/character-edge-dev-full-v1"
CALIBRATION = ROOT / ".omx/artifacts/phonetic-score-scale-training-v1/model.json"
MODEL_SOURCE = ROOT / "hanglish/models/contextual_character_residual.py"
PARENT_MODEL = ROOT / "hanglish/models/character_edge_residual.py"
SOURCE_CORRESPONDENCE = ROOT / "hanglish/core/source_correspondence.py"
TRAINING_MATH = ROOT / "scripts/train_contextual_character.py"
FULL_TRAINING = ROOT / "scripts/train_contextual_character_full.py"
TRAINING_VERIFIER = ROOT / "scripts/verify_contextual_character_fit.py"
SELECTION_HELPER = ROOT / "scripts/current_recombination_selection.py"
RESOURCE_HELPER = ROOT / "scripts/train_character_edge_full_fit.py"
PLAN = ROOT / ".omx/plans/contextual-character-dev-v1.md"
PRODUCER = Path(__file__).resolve()

PARENT_CAPTURE_SHA = "8985ec4f50b0151987e69cad63a3033203d1407af68cbc9c250fe82c7a7858f1"
PARENT_REPORT_SHA = "ea29e58810b9543205d4ec94b244e5aa9a1138f74a4acb95182e3c4f13a69614"
PARENT_PROOF_SHA = "453bb4a9993d3d083c7f0e0406cde8d5015faa27df36f73f243c2f52ce1998cf"
CALIBRATION_SHA = "e07462d50da0b67304a82a9a261068ccc84b8055a99b5d70af56db530d245838"
EXPECTED_COUNTS = {
    "ranked_rows": 312,
    "no_pool_rows": 4,
    "old_candidates": 47694,
    "novel_candidates": 27831,
    "scored_novel": 27831,
    "unsupported_novel": 0,
}
TRAINING_COUNTS = {
    "ranked_rows": 1629,
    "no_pool_rows": 13,
    "unsupported_candidates": 31,
    "gold_present_rows": 1571,
    "missing_gold_rows": 71,
}
LIMITS = {
    "whole_seconds": 600.0,
    "row_seconds": 30.0,
    "rss_bytes": 2 * 1024**3,
    "swap_growth_bytes": 512 * 1024**2,
    "output_bytes": 128 * 1024**2,
}
OUTPUT_HASHES = ("before.jsonl", "after.jsonl", "model.pt", "resources.jsonl")


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def write_json(path: Path, value, *, byte_budget: int | None = None) -> None:
    encoded = _json_bytes(value)
    if byte_budget is not None:
        require(len(encoded) <= byte_budget, "serialized output reached the hard byte bound")
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded.decode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _merge_pins(target: dict[str, str], additions: dict[str, str]) -> None:
    for raw_path, expected in additions.items():
        path = str(Path(raw_path).resolve())
        require(isinstance(expected, str) and len(expected) == 64,
                "artifact pin digest differs")
        require(path not in target or target[path] == expected,
                "artifact pin collision differs")
        target[path] = expected


def verify_pins(pins: dict[str, str]) -> None:
    require(bool(pins), "artifact pin inventory is empty")
    for raw_path, expected in pins.items():
        path = Path(raw_path)
        require(path.is_file() and digest(path) == expected,
                "frozen input differs: " + raw_path)


def _authenticate_parent() -> tuple[dict, dict[str, str]]:
    """Authenticate the complete source-only parent and all of its pins."""
    report_path = PARENT / "report.json"
    proof_path = PARENT / "independent-verification.json"
    capture_path = PARENT / "capture.jsonl"
    closure_path = PARENT / "capture.complete.json"
    resources_path = PARENT / "resources.jsonl"
    report = read_json(report_path)
    proof = read_json(proof_path)
    hashes = report.get("hashes")
    require(
        digest(report_path) == PARENT_REPORT_SHA
        and digest(proof_path) == PARENT_PROOF_SHA
        and report.get("completed") is True
        and report.get("mode") == "full"
        and report.get("dataset_split") == "dev"
        and report.get("rows") == ROWS
        and report.get("counts") == EXPECTED_COUNTS
        and report.get("capture_sha256") == PARENT_CAPTURE_SHA
        and report.get("labels_opened") is False
        and report.get("source_only") is True
        and isinstance(hashes, dict)
        and hashes == {
            "capture.jsonl": PARENT_CAPTURE_SHA,
            "capture.complete.json": digest(closure_path),
            "resources.jsonl": digest(resources_path),
        }
        and digest(capture_path) == PARENT_CAPTURE_SHA,
        "frozen DEV parent report or outputs differ",
    )
    require(
        proof.get("verified") is True
        and proof.get("mode") == "full"
        and proof.get("rows") == ROWS
        and proof.get("counts") == {
            "original": EXPECTED_COUNTS["old_candidates"],
            "novel": EXPECTED_COUNTS["novel_candidates"],
            "scored_novel": EXPECTED_COUNTS["scored_novel"],
            "unsupported_novel": EXPECTED_COUNTS["unsupported_novel"],
            "no_pool": EXPECTED_COUNTS["no_pool_rows"],
        }
        and proof.get("report_sha256") == PARENT_REPORT_SHA
        and proof.get("capture_sha256") == PARENT_CAPTURE_SHA
        and proof.get("labels_opened") is False
        and proof.get("source_only") is True
        and proof.get("character_inference_replayed") is True,
        "independently verified DEV parent differs",
    )
    parent_pins = report.get("pins")
    require(isinstance(parent_pins, dict) and parent_pins
            and proof.get("pins") == parent_pins,
            "DEV parent pin inventory differs")
    pins: dict[str, str] = {}
    _merge_pins(pins, parent_pins)
    _merge_pins(pins, {
        str(report_path): PARENT_REPORT_SHA,
        str(proof_path): PARENT_PROOF_SHA,
        str(capture_path): PARENT_CAPTURE_SHA,
        str(closure_path): hashes["capture.complete.json"],
        str(resources_path): hashes["resources.jsonl"],
    })
    verify_pins(pins)
    return report, pins


def _authenticate_training(training: Path) -> tuple[dict, dict, dict[str, str]]:
    """Require a completed and independently replayed full FIT checkpoint."""
    training = training.resolve()
    require(training.is_dir(), "training directory is missing")
    report_path = training / "report.json"
    proof_path = training / "independent-verification.json"
    report = read_json(report_path)
    proof = read_json(proof_path)
    hashes = report.get("hashes")
    require(
        report.get("completed") is True
        and report.get("scope") == "full"
        and report.get("rows") == 1642
        and report.get("recipe") == RECIPE
        and report.get("parameter_count") == 31681
        and report.get("updates") == 6284
        and report.get("counts") == TRAINING_COUNTS
        and report.get("dev_opened") is False
        and report.get("test_opened") is False
        and report.get("production_changed") is False
        and report.get("model_source_sha256") == digest(MODEL_SOURCE)
        and report.get("runner_sha256") == digest(FULL_TRAINING)
        and isinstance(hashes, dict)
        and set(hashes) == set(OUTPUT_HASHES)
        and all((training / name).is_file()
                and digest(training / name) == hashes[name]
                for name in OUTPUT_HASHES),
        "completed contextual full FIT report differs",
    )
    report_sha = digest(report_path)
    require(
        proof.get("verified") is True
        and proof.get("scope") == "full"
        and proof.get("rows") == 1642
        and proof.get("report_sha256") == report_sha
        and proof.get("checkpoint_sha256") == hashes["model.pt"]
        and proof.get("before_sha256") == hashes["before.jsonl"]
        and proof.get("after_sha256") == hashes["after.jsonl"]
        and proof.get("resources_sha256") == hashes["resources.jsonl"]
        and proof.get("candidate_chunk") == 32
        and proof.get("dev_opened") is False
        and proof.get("test_opened") is False,
        "independent full FIT checkpoint proof differs",
    )
    training_pins = report.get("pins")
    require(isinstance(training_pins, dict) and training_pins,
            "training pin inventory differs")
    pins: dict[str, str] = {}
    _merge_pins(pins, training_pins)
    _merge_pins(pins, {
        str(report_path): report_sha,
        str(proof_path): digest(proof_path),
        **{str(training / name): hashes[name] for name in OUTPUT_HASHES},
    })
    verify_pins(pins)
    return report, proof, pins


def authenticate(training: Path) -> dict:
    """Collect and verify parent, training, source, and producer evidence."""
    parent_report, pins = _authenticate_parent()
    training_report, training_proof, training_pins = _authenticate_training(training)
    _merge_pins(pins, training_pins)
    _merge_pins(pins, {
        str(CALIBRATION): CALIBRATION_SHA,
        str(MODEL_SOURCE): digest(MODEL_SOURCE),
        str(PARENT_MODEL): digest(PARENT_MODEL),
        str(SOURCE_CORRESPONDENCE): digest(SOURCE_CORRESPONDENCE),
        str(TRAINING_MATH): digest(TRAINING_MATH),
        str(FULL_TRAINING): digest(FULL_TRAINING),
        str(TRAINING_VERIFIER): digest(TRAINING_VERIFIER),
        str(SELECTION_HELPER): digest(SELECTION_HELPER),
        str(RESOURCE_HELPER): digest(RESOURCE_HELPER),
        str(PLAN): digest(PLAN),
        str(PRODUCER): digest(PRODUCER),
    })
    verify_pins(pins)
    return {
        "pins": pins,
        "parent_report": parent_report,
        "training_report": training_report,
        "training_proof": training_proof,
    }


def _candidate_pool(rows: list[dict], calibration: dict) -> tuple[list[dict], list[float]]:
    candidates = []
    baseline = []
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("text"), str)
                and row["text"] and isinstance(row.get("segments"), list),
                "parent candidate differs")
        candidates.append({"text": row["text"], "segments": deepcopy(row["segments"])})
        baseline.append(float(calibrated_score(row, calibration)))
    require(len({row["text"] for row in candidates}) == len(candidates),
            "parent candidate texts are duplicated")
    return candidates, baseline


def _score_chunks(model, source: str, candidates: list[dict],
                  baseline: list[float], chunk_size: int = 32) -> list[float]:
    """Score one expanded pool using bounded, order-preserving chunks."""
    require(type(chunk_size) is int and chunk_size == 32,
            "contextual DEV candidate chunk differs")
    require(len(candidates) == len(baseline) and candidates,
            "contextual expanded pool differs")
    values: list[float] = []
    for start in range(0, len(candidates), chunk_size):
        prepared = model.prepare(
            source,
            candidates[start:start + chunk_size],
            baseline[start:start + chunk_size],
        )
        result = model(prepared)
        scores = result.get("scores") if isinstance(result, dict) else None
        require(scores is not None and len(scores) == len(candidates[start:start + chunk_size]),
                "contextual score chunk differs")
        values.extend(float(value) for value in scores)
    require(len(values) == len(candidates) and all(math.isfinite(value) for value in values),
            "contextual scores are nonfinite or misaligned")
    return values


def predict_row(parent: dict, model, calibration: dict, index: int) -> dict:
    """Purely add two contextual arms to one authenticated parent row."""
    validate_model(calibration)
    request = parent.get("request") if isinstance(parent, dict) else None
    source = request.get("input") if isinstance(request, dict) else None
    scoring = parent.get("scoring") if isinstance(parent, dict) else None
    predictions = parent.get("predictions") if isinstance(parent, dict) else None
    require(
        isinstance(parent.get("id"), str)
        and request == {"input": source, "context": ""}
        and isinstance(source, str) and source
        and parent.get("dataset_split") == "dev"
        and parent.get("source_row_index") == index
        and isinstance(scoring, dict)
        and scoring.get("id") == parent["id"]
        and scoring.get("request") == request
        and isinstance(predictions, dict),
        "parent DEV row identity differs",
    )
    ranked = scoring.get("ranked")
    original, novel = scoring.get("original"), scoring.get("novel")
    require(type(ranked) is bool and isinstance(original, list)
            and isinstance(novel, list), "parent scored pool differs")

    contextual_scores = {"old_contextual": [], "expanded_contextual": []}
    contextual_predictions: dict[str, list[dict]] = {}
    if not ranked:
        fallback = scoring.get("fallback_public")
        require(not original and not novel and isinstance(fallback, list) and fallback
                and predictions.get("old_calibrated") == fallback
                and predictions.get("expanded_calibrated") == fallback,
                "parent no-pool fallback differs")
        contextual_predictions = {
            "old_contextual": deepcopy(fallback),
            "expanded_contextual": deepcopy(fallback),
        }
    else:
        require(original and scoring.get("fallback_public") is None
                and all(row.get("status") == "preserved" for row in original)
                and all(row.get("status") == "scored" for row in novel),
                "ranked parent candidate status differs")
        expanded_rows = original + novel
        expanded_candidates, expanded_baseline = _candidate_pool(
            expanded_rows, calibration)
        old_count = len(original)
        old_candidates = expanded_candidates[:old_count]
        old_baseline = expanded_baseline[:old_count]
        require(
            predictions.get("old_calibrated") == render_public(source, [
                dict(candidate, score=score)
                for candidate, score in zip(old_candidates, old_baseline)
            ])
            and predictions.get("expanded_calibrated") == render_public(source, [
                dict(candidate, score=score)
                for candidate, score in zip(expanded_candidates, expanded_baseline)
            ]),
            "parent calibrated public predictions differ",
        )
        expanded_scores = _score_chunks(
            model, source, expanded_candidates, expanded_baseline, 32)
        old_scores = expanded_scores[:old_count]
        contextual_scores = {
            "old_contextual": old_scores,
            "expanded_contextual": expanded_scores,
        }
        contextual_predictions = {
            "old_contextual": render_public(source, [
                dict(candidate, score=score)
                for candidate, score in zip(old_candidates, old_scores)
            ]),
            "expanded_contextual": render_public(source, [
                dict(candidate, score=score)
                for candidate, score in zip(expanded_candidates, expanded_scores)
            ]),
        }
    return {
        "id": parent["id"],
        "request": deepcopy(request),
        "dataset_split": "dev",
        "source_row_index": index,
        "contextual_scores": contextual_scores,
        "predictions": contextual_predictions,
    }


def _load_model(training: Path):
    import torch
    from hanglish.models.contextual_character_residual import (
        ContextualCharacterResidualRanker,
    )

    torch.set_num_threads(RECIPE["threads"])
    torch.set_num_interop_threads(RECIPE["interop_threads"])
    torch.manual_seed(RECIPE["seed"])
    torch.use_deterministic_algorithms(True)
    model = ContextualCharacterResidualRanker(
        byte_dim=16, hidden=32, head_hidden=32)
    state = torch.load(training / "model.pt", map_location="cpu", weights_only=True)
    require(isinstance(state, dict), "contextual checkpoint state differs")
    model.load_state_dict(state, strict=True)
    require(sum(parameter.numel() for parameter in model.parameters()) == 31681
            and all(parameter.device.type == "cpu"
                    and bool(torch.isfinite(parameter).all())
                    for parameter in model.parameters()),
            "contextual checkpoint tensors differ")
    model.eval()
    return model, torch


def _observe(guard: ResourceGuard, stage: str) -> dict:
    row = guard.observe(stage)
    require(row["seconds"] <= LIMITS["whole_seconds"],
            "contextual DEV capture time limit exceeded")
    return row


def _count_parent(parent: dict, counts: dict[str, int]) -> None:
    scoring = parent["scoring"]
    if scoring["ranked"]:
        counts["ranked_rows"] += 1
    else:
        counts["no_pool_rows"] += 1
    counts["old_candidates"] += len(scoring["original"])
    counts["novel_candidates"] += len(scoring["novel"])
    counts["scored_novel"] += sum(
        row.get("status") == "scored" for row in scoring["novel"])
    counts["unsupported_novel"] += sum(
        row.get("status") == "unsupported" for row in scoring["novel"])


def run(training: Path, output: Path) -> None:
    training = training.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    guard = None
    captured = 0
    try:
        authenticated = authenticate(training)
        pins = authenticated["pins"]
        guard = ResourceGuard(output)
        _observe(guard, "authenticated")
        calibration = read_json(CALIBRATION)
        validate_model(calibration)
        model, torch = _load_model(training)
        _observe(guard, "model-loaded")

        counts = {key: 0 for key in EXPECTED_COUNTS}
        capture_bytes = 0
        with ((PARENT / "capture.jsonl").open(encoding="utf-8") as source_stream,
              (output / "capture.jsonl").open("x", encoding="utf-8") as target_stream):
            for index, line in enumerate(source_stream):
                require(line.strip(), "blank parent capture row differs")
                started = time.monotonic()
                parent = json.loads(line)
                with torch.no_grad():
                    result = predict_row(parent, model, calibration, index)
                encoded = (json.dumps(result, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
                require(capture_bytes + len(encoded) <= LIMITS["output_bytes"],
                        "capture output reached the hard 128 MiB bound")
                target_stream.write(encoded.decode("utf-8"))
                target_stream.flush()
                os.fsync(target_stream.fileno())
                capture_bytes += len(encoded)
                captured += 1
                _count_parent(parent, counts)
                guard.require_row_time(started)
                if captured % 16 == 0:
                    _observe(guard, f"row-{captured}")
                    require(sum(path.stat().st_size for path in output.iterdir())
                            <= LIMITS["output_bytes"],
                            "capture outputs reached the hard 128 MiB bound")
                    print(json.dumps({"captured": captured, "rows": ROWS}), flush=True)

        require(captured == ROWS and counts == EXPECTED_COUNTS,
                "DEV parent row count or denominators differ")
        verify_pins(pins)
        _observe(guard, "inputs-reverified")
        completed = _observe(guard, "complete")
        capture_sha = digest(output / "capture.jsonl")
        resources_sha = digest(output / "resources.jsonl")
        report = {
            "schema": SCHEMA,
            "completed": True,
            "rows": captured,
            "dataset_split": "dev",
            "counts": counts,
            "parent_directory": str(PARENT.resolve()),
            "training_directory": str(training),
            "pins": pins,
            "hashes": {
                "capture.jsonl": capture_sha,
                "resources.jsonl": resources_sha,
            },
            "limits": LIMITS,
            "resources": {
                "seconds": completed["seconds"],
                "max_rss_bytes": guard.peak_rss,
                "max_swap_growth_bytes": guard.peak_swap_growth,
                "output_bytes": capture_bytes,
            },
            "candidate_chunk": 32,
            "labels_opened": False,
            "test_opened": False,
            "production_changed": False,
        }
        remaining = LIMITS["output_bytes"] - sum(
            path.stat().st_size for path in output.iterdir())
        write_json(output / "report.json", report, byte_budget=remaining)
    except Exception as error:
        if guard is not None:
            try:
                _observe(guard, "failed")
            except Exception:
                pass
        write_json(output / "failure.json", {
            "schema": SCHEMA,
            "completed": False,
            "captured_rows": captured,
            "parent_directory": str(PARENT.resolve()),
            "training_directory": str(training),
            "labels_opened": False,
            "test_opened": False,
            "production_changed": False,
            "error_type": type(error).__name__,
            "error": str(error),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.training, args.output)


if __name__ == "__main__":
    main()
