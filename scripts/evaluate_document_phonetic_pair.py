#!/usr/bin/env python3
"""Join closed source-only phonetic captures to aggregate document-reserve metrics."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.phonology import romanize_word


INPUTS = ROOT / ".omx/artifacts/document-phonetic-inputs-v1"
RESERVE = ROOT / ".omx/artifacts/document-reserve-v2"
RAW = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
BASE = ROOT / ".omx/artifacts/phonetic-supported-data-v1/pairs.jsonl"
DELTA = ROOT / ".omx/artifacts/observed-phonetic-delta-v1/delta.jsonl"
INPUT_MANIFEST_SHA256 = "da5ed2f782ca5272627dc01b90312cc5e61fba3272aba8cd4a0515784bc6fd87"
INPUTS_SHA256 = "c9aee2f9a1ebd4b9ec5754c5145c4c2ddfd8c1eae6846fb6dc2774c8ff38376b"
RESERVE_SHA256 = "7d62027d11dcf1320b2a1bac7bcb83e1a31d31ce521808bbe4d1faadf6d4b810"
RAW_SHA256 = "904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b"
BASE_SHA256 = "cf3e43f75420c17fbe8e1ec118a29a0f6a71c0e55cedbdbae5d7ef3668c84a84"
DELTA_SHA256 = "bc299116e618ef37c044d8ac0f3061a55e46efdc1b4f1d0c8d9dc1b8ab655e1a"
HANGUL = re.compile(r"[가-힣]+")
SUPPORTED = re.compile(r"[a-z]{1,48}\Z")
PAIR_STRATA = ("seen_control", "delta_only", "unseen_augmented")
REFERENCE_PATHS = {str((RESERVE / "selection.jsonl").relative_to(ROOT)),
                   str(RAW.relative_to(ROOT))}
LIMITS = {"whole_seconds": 1800.0, "update_seconds": 60.0,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2}


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
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"invalid JSONL row {number}: {path}")
            result.append(value)
    return result


def ratio(numerator: int, denominator: int) -> dict:
    return {"correct": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def resource_valid(row: object, limits: dict) -> bool:
    return (isinstance(row, dict)
            and type(row.get("elapsed_s")) in (int, float)
            and math.isfinite(row["elapsed_s"]) and 0 <= row["elapsed_s"] <= limits["whole_seconds"]
            and type(row.get("rss_bytes")) is int and 0 <= row["rss_bytes"] <= limits["rss_bytes"]
            and type(row.get("swap_growth_bytes")) is int
            and row["swap_growth_bytes"] <= limits["swap_growth_bytes"])


def load_input_requests() -> tuple[list[dict], dict]:
    """Load only the closed source request artifact; no reference files are touched."""
    require(sha256_file(INPUTS / "manifest.json") == INPUT_MANIFEST_SHA256,
            "input manifest hash differs")
    require(sha256_file(INPUTS / "inputs.jsonl") == INPUTS_SHA256,
            "source-only inputs hash differs")
    manifest = read_json(INPUTS / "manifest.json")
    requests = read_jsonl(INPUTS / "inputs.jsonl")
    require(manifest.get("schema") == "document-phonetic-inputs-v1"
            and manifest.get("completed") is True
            and manifest.get("inputs_sha256") == INPUTS_SHA256
            and manifest.get("counts", {}).get("unique_inputs") == 10166
            and manifest.get("performance_labels_joined") is False,
            "input manifest contract differs")
    require(len(requests) == 10166 and all(set(row) == {"id", "input"} for row in requests),
            "source-only request rows differ")
    for row in requests:
        require(row["id"] == hashlib.sha256(row["input"].encode()).hexdigest()
                and SUPPORTED.fullmatch(row["input"]) is not None,
                "source-only request identity differs")
    require(len({row["id"] for row in requests}) == len(requests)
            and len({row["input"] for row in requests}) == len(requests),
            "duplicate source-only request")
    return requests, manifest


def validate_outputs(source: str, outputs: object,
                     romanize: Callable[[str], str] = romanize_word) -> list[dict]:
    require(isinstance(outputs, list) and len(outputs) <= 8, "capture outputs exceed beam8")
    seen, previous = set(), math.inf
    for output in outputs:
        require(isinstance(output, dict) and set(output) == {"text", "score"},
                "capture output schema differs")
        text, score = output["text"], output["score"]
        require(isinstance(text, str) and 1 <= len(text) <= 24
                and HANGUL.fullmatch(text) is not None and text not in seen,
                "capture output text invalid or duplicated")
        require(type(score) in (int, float) and math.isfinite(score) and score <= previous,
                "capture output scores are not finite descending")
        require(romanize(text) == source, "capture output is not forward-valid")
        seen.add(text); previous = score
    return outputs


def authenticate_capture(directory: Path, arm: str, requests: list[dict]) -> dict:
    """Authenticate a complete source-only capture before any reference is opened."""
    directory = Path(directory)
    report_path, completion_path = directory / "report.json", directory / "completion.json"
    predictions_path = directory / "predictions.jsonl"
    require(report_path.is_file() and completion_path.is_file() and predictions_path.is_file(),
            f"incomplete {arm} capture files")
    report, completion = read_json(report_path), read_json(completion_path)
    require(report.get("schema") == "document-phonetic-capture-v1"
            and report.get("completed") is True and report.get("arm") == arm
            and report.get("input_manifest_sha256") == INPUT_MANIFEST_SHA256
            and report.get("inputs_sha256") == INPUTS_SHA256
            and report.get("prediction_count") == len(requests)
            and report.get("beam_size") == 8
            and report.get("reference_targets_read") is False,
            f"{arm} capture report contract differs")
    require(sha256_file(predictions_path) == report.get("predictions_sha256"),
            f"{arm} prediction hash differs")
    require(completion.get("schema") == "observed-phonetic-pair-completion-v1"
            and completion.get("completed") is True
            and completion.get("report_sha256") == sha256_file(report_path),
            f"{arm} completion does not authenticate report")
    limits = report.get("limits")
    require(limits == LIMITS and resource_valid(report.get("final_resource"), limits)
            and resource_valid(completion.get("resource_after_report"), limits),
            f"{arm} capture resources invalid")
    pins = report.get("source_hashes")
    require(isinstance(pins, dict) and pins, f"{arm} capture source pins missing")
    require(not (set(pins) & REFERENCE_PATHS),
            f"{arm} capture improperly pins reference files before closure")
    for relative, wanted in pins.items():
        path = ROOT / relative
        require(path.is_file() and sha256_file(path) == wanted,
                f"{arm} capture source pin differs: {relative}")
    for field in ("checkpoint_sha256", "training_report_sha256",
                  "paired_initial_checkpoint_sha256"):
        value = report.get(field)
        require(isinstance(value, str) and len(value) == 64
                and value in pins.values(), f"{arm} {field} is not source-bound")

    predictions = read_jsonl(predictions_path)
    require(len(predictions) == len(requests), f"{arm} capture is missing inputs")
    for expected, row in zip(requests, predictions):
        require(isinstance(row, dict)
                and set(row) == {"id", "input", "outputs", "latency_s"}
                and row.get("id") == expected["id"] and row.get("input") == expected["input"],
                f"{arm} capture identity is incomplete, duplicated, or reordered")
        latency = row["latency_s"]
        require(type(latency) in (int, float) and math.isfinite(latency) and latency >= 0,
                f"{arm} capture latency invalid")
        validate_outputs(row["input"], row["outputs"])
    require(len({row["id"] for row in predictions}) == len(predictions),
            f"{arm} duplicate capture IDs")
    return {"arm": arm, "report": report, "predictions": predictions,
            "report_sha256": sha256_file(report_path),
            "completion_sha256": sha256_file(completion_path),
            "predictions_sha256": sha256_file(predictions_path)}


def load_references(romanize: Callable[[str], str] = romanize_word) -> dict:
    """Open reserve references. Call only after both source-only captures close."""
    require(sha256_file(RESERVE / "selection.jsonl") == RESERVE_SHA256,
            "reserve selection hash differs")
    require(sha256_file(RAW) == RAW_SHA256, "raw reference source hash differs")
    selection, raw = read_jsonl(RESERVE / "selection.jsonl"), read_jsonl(RAW)
    require(len(selection) == 1046 and len({row.get("id") for row in selection}) == 1046,
            "reserve selection cardinality differs")
    by_id = {row.get("id"): row for row in raw}
    require(None not in by_id and len(by_id) == len(raw), "raw reference IDs differ")
    occurrences = []
    for selected in selection:
        row = by_id.get(selected["id"])
        require(row is not None and row.get("group_id") == selected.get("group_id"),
                "reserve identity/group mapping differs")
        text = row.get("text")
        require(isinstance(text, str) and text, "reserve reference text invalid")
        occurrences.extend((romanize(surface), surface) for surface in HANGUL.findall(text))
    pair_counts = Counter(occurrences)
    variants: dict[str, set[str]] = defaultdict(set)
    for source, surface in pair_counts:
        variants[source].add(surface)
    supported = {source for source in variants if SUPPORTED.fullmatch(source)}
    require(len(occurrences) == 15812 and len(pair_counts) == 10182
            and len(supported) == 10166
            and sum(count for pair, count in pair_counts.items()
                    if pair[0] not in supported) == 1
            and sum(pair[0] not in supported for pair in pair_counts) == 1,
            "reference projection differs from frozen census")
    return {"occurrences": occurrences, "pair_counts": pair_counts,
            "variants": dict(variants), "supported_inputs": supported,
            "source_hashes": {
                str((RESERVE / "selection.jsonl").relative_to(ROOT)): RESERVE_SHA256,
                str(RAW.relative_to(ROOT)): RAW_SHA256,
            }}


def load_training_strata() -> dict:
    require(sha256_file(BASE) == BASE_SHA256 and sha256_file(DELTA) == DELTA_SHA256,
            "training pair sources differ")
    base = read_jsonl(BASE)
    control = {(row["roman"], row["surface"]) for row in base if row.get("split") == "train"}
    delta = {(row["roman"], row["surface"]) for row in read_jsonl(DELTA)}
    require(len(control) == 51361 and len(delta) == 32984 and not (control & delta),
            "training pair inventories differ")
    return {"control_pairs": control, "delta_pairs": delta,
            "control_romans": {pair[0] for pair in control},
            "delta_romans": {pair[0] for pair in delta}}


def pair_stratum(pair: tuple[str, str], strata: dict) -> str:
    if pair in strata["control_pairs"]:
        return "seen_control"
    if pair in strata["delta_pairs"]:
        return "delta_only"
    return "unseen_augmented"


def roman_stratum(source: str, strata: dict) -> str:
    if source in strata["control_romans"]:
        return "seen_control"
    if source in strata["delta_romans"]:
        return "delta_only"
    return "unseen_augmented"


def _pair_metrics(pairs: list[tuple[str, str]], pair_counts: Counter,
                  predicted: dict[str, list[str]]) -> dict:
    top1 = sum(bool(predicted.get(source)) and predicted[source][0] == surface
               for source, surface in pairs)
    top8 = sum(surface in predicted.get(source, ()) for source, surface in pairs)
    occurrences = sum(pair_counts[pair] for pair in pairs)
    occurrence_top1 = sum(pair_counts[pair] for pair in pairs
                          if predicted.get(pair[0]) and predicted[pair[0]][0] == pair[1])
    occurrence_top8 = sum(pair_counts[pair] for pair in pairs
                          if pair[1] in predicted.get(pair[0], ()))
    return {"pairs": len(pairs), "occurrences": occurrences,
            "variant_recall_top1": ratio(top1, len(pairs)),
            "variant_recall_top8": ratio(top8, len(pairs)),
            "occurrence_weighted_top1": ratio(occurrence_top1, occurrences),
            "occurrence_weighted_top8": ratio(occurrence_top8, occurrences)}


def _roman_metrics(sources: list[str], variants: dict[str, set[str]],
                   pair_counts: Counter, predicted: dict[str, list[str]]) -> dict:
    top1 = sum(bool(predicted.get(source)) and predicted[source][0] in variants[source]
               for source in sources)
    top8 = sum(bool(set(predicted.get(source, ())) & variants[source]) for source in sources)
    return {"roman_keys": len(sources),
            "reference_pairs": sum(len(variants[source]) for source in sources),
            "occurrences": sum(count for (source, _), count in pair_counts.items()
                               if source in set(sources)),
            "acceptable_variant_top1": ratio(top1, len(sources)),
            "acceptable_variant_top8": ratio(top8, len(sources))}


def score_arm(predictions: list[dict], references: dict, strata: dict) -> dict:
    predicted = {row["input"]: [item["text"] for item in row["outputs"]]
                 for row in predictions}
    variants, pair_counts = references["variants"], references["pair_counts"]
    all_pairs, all_sources = sorted(pair_counts), sorted(variants)
    supported = references["supported_inputs"]
    require(set(predicted) == supported, "capture inputs differ from supported reference inputs")
    pair_groups = {name: [pair for pair in all_pairs if pair_stratum(pair, strata) == name]
                   for name in PAIR_STRATA}
    roman_groups = {name: [source for source in all_sources
                           if roman_stratum(source, strata) == name]
                    for name in PAIR_STRATA}
    overall_pairs = _pair_metrics(all_pairs, pair_counts, predicted)
    overall_romans = _roman_metrics(all_sources, variants, pair_counts, predicted)
    lengths = sorted({len(source) for source in all_sources})
    by_input_length = {}
    for length in lengths:
        length_sources = [source for source in all_sources if len(source) == length]
        length_pairs = [pair for pair in all_pairs if len(pair[0]) == length]
        by_input_length[str(length)] = {
            "roman_acceptable_variant": _roman_metrics(
                length_sources, variants, pair_counts, predicted),
            "all_spelling_variants": _pair_metrics(
                length_pairs, pair_counts, predicted),
        }
    return {
        "roman_acceptable_variant": overall_romans,
        "all_spelling_variants": overall_pairs,
        "candidate_coverage": overall_pairs["variant_recall_top8"],
        "empty_supported_outputs": sum(not predicted[source] for source in supported),
        "unsupported": {
            "roman_keys": len(set(variants) - supported),
            "reference_pairs": sum(source not in supported for source, _ in pair_counts),
            "occurrences": sum(count for (source, _), count in pair_counts.items()
                               if source not in supported),
        },
        "candidate_outputs": {"total": sum(map(len, predicted.values())),
                              "maximum_per_input": max(map(len, predicted.values()), default=0)},
        "by_exact_pair_stratum": {
            name: _pair_metrics(pair_groups[name], pair_counts, predicted) for name in PAIR_STRATA},
        "by_roman_key_stratum": {
            name: _roman_metrics(roman_groups[name], variants, pair_counts, predicted)
            for name in PAIR_STRATA},
        "by_input_length": by_input_length,
    }


def run(control_capture: Path, augmented_capture: Path, output: Path, *,
        request_loader=load_input_requests, capture_authenticator=authenticate_capture,
        reference_loader=load_references, strata_loader=load_training_strata) -> dict:
    output = Path(output)
    require(not output.exists(), f"output already exists: {output}")
    requests, input_manifest = request_loader()
    # Both calls and the cross-arm identity checks must finish before reference_loader is invoked.
    control = capture_authenticator(Path(control_capture), "control", requests)
    augmented = capture_authenticator(Path(augmented_capture), "augmented", requests)
    require(control["report"]["paired_initial_checkpoint_sha256"]
            == augmented["report"]["paired_initial_checkpoint_sha256"],
            "paired captures do not share initialization")
    require([(row["id"], row["input"]) for row in control["predictions"]]
            == [(row["id"], row["input"]) for row in augmented["predictions"]],
            "paired capture request identities differ")
    closed_hashes = {arm: {key: capture[key] for key in
                           ("report_sha256", "completion_sha256", "predictions_sha256")}
                     for arm, capture in (("control", control), ("augmented", augmented))}

    references = reference_loader()
    strata = strata_loader()
    metrics = {"control": score_arm(control["predictions"], references, strata),
               "augmented": score_arm(augmented["predictions"], references, strata)}
    for arm, capture in (("control", control), ("augmented", augmented)):
        directory = Path(control_capture if arm == "control" else augmented_capture)
        require(sha256_file(directory / "report.json") == closed_hashes[arm]["report_sha256"]
                and sha256_file(directory / "completion.json") == closed_hashes[arm]["completion_sha256"]
                and sha256_file(directory / "predictions.jsonl") == closed_hashes[arm]["predictions_sha256"],
                f"{arm} capture changed during late reference join")
    pair_inventory = Counter(pair_stratum(pair, strata) for pair in references["pair_counts"])
    roman_inventory = Counter(roman_stratum(source, strata) for source in references["variants"])
    occurrence_inventory = Counter()
    for pair, count in references["pair_counts"].items():
        occurrence_inventory[pair_stratum(pair, strata)] += count
    report = {
        "schema": "document-phonetic-pair-evaluation-v1", "completed": True,
        "quality_gate_applied": False, "checkpoint_selected": False,
        "capture_closed_before_reference_join": True,
        "capture_hashes_unchanged_after_reference_join": True,
        "input_manifest_sha256": INPUT_MANIFEST_SHA256,
        "inputs_sha256": INPUTS_SHA256,
        "capture_artifacts": closed_hashes,
        "checkpoints": {
            arm: {field: capture["report"][field] for field in
                  ("checkpoint_sha256", "training_report_sha256",
                   "paired_initial_checkpoint_sha256")}
            for arm, capture in (("control", control), ("augmented", augmented))},
        "reference_counts": input_manifest["counts"],
        "strata_inventory": {
            "exact_pairs": {name: pair_inventory[name] for name in PAIR_STRATA},
            "roman_keys": {name: roman_inventory[name] for name in PAIR_STRATA},
            "occurrences": {name: occurrence_inventory[name] for name in PAIR_STRATA},
        },
        "metrics": metrics,
        "source_hashes": {
            **references["source_hashes"], str(BASE.relative_to(ROOT)): BASE_SHA256,
            str(DELTA.relative_to(ROOT)): DELTA_SHA256,
            str(Path(__file__).resolve().relative_to(ROOT)): sha256_file(Path(__file__)),
        },
        "contracts": {
            "aggregate_only_no_examples": True, "beam_size": 8,
            "acceptable_variant_policy": "any exact reference spelling for the Roman key",
            "all_spelling_variant_denominator": True,
            "unsupported_reference_retained_in_denominators": True,
            "model_activity": False, "performance_tuning": False,
            "production_changed": False,
        },
        "limitations": [
            "Direct word candidate metrics are not sentence accuracy or English-copy evaluation.",
            "The document reserve is not an unseen lexical-family or pretraining-isolation claim.",
            "Results from one paired initialization do not establish seed-robust improvement.",
            "TRAIN exposure strata are descriptive and do not select a checkpoint or quality gate.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (temporary / "report.json").write_bytes(canonical(report) + b"\n")
        with (temporary / "report.json").open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except Exception:
        for child in temporary.iterdir():
            child.unlink()
        temporary.rmdir()
        raise
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--control-capture", type=Path, required=True)
    result.add_argument("--augmented-capture", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    result = run(args.control_capture, args.augmented_capture, args.output)
    print(json.dumps({"completed": result["completed"],
                      "quality_gate_applied": result["quality_gate_applied"]}))
