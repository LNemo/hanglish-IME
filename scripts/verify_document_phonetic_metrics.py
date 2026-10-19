#!/usr/bin/env python3
"""Independently verify aggregate document phonetic metrics.

The verifier authenticates both closed source-only captures before opening the
aggregate evaluation report or any reference/training-label artifact. It never
loads a model and does not import the producer evaluator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from hanglish.core.phonology import romanize_word


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / ".omx/artifacts/document-phonetic-inputs-v1"
RESERVE = ROOT / ".omx/artifacts/document-reserve-v2/selection.jsonl"
RAW = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
BASE = ROOT / ".omx/artifacts/phonetic-supported-data-v1/pairs.jsonl"
DELTA = ROOT / ".omx/artifacts/observed-phonetic-delta-v1/delta.jsonl"
EVALUATOR = ROOT / "scripts/evaluate_document_phonetic_pair.py"
INPUT_MANIFEST_SHA256 = "da5ed2f782ca5272627dc01b90312cc5e61fba3272aba8cd4a0515784bc6fd87"
INPUTS_SHA256 = "c9aee2f9a1ebd4b9ec5754c5145c4c2ddfd8c1eae6846fb6dc2774c8ff38376b"
RESERVE_SHA256 = "7d62027d11dcf1320b2a1bac7bcb83e1a31d31ce521808bbe4d1faadf6d4b810"
RAW_SHA256 = "904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b"
BASE_SHA256 = "cf3e43f75420c17fbe8e1ec118a29a0f6a71c0e55cedbdbae5d7ef3668c84a84"
DELTA_SHA256 = "bc299116e618ef37c044d8ac0f3061a55e46efdc1b4f1d0c8d9dc1b8ab655e1a"
HANGUL = re.compile(r"[가-힣]+")
SUPPORTED = re.compile(r"[a-z]{1,48}\Z")
STRATA = ("seen_control", "delta_only", "unseen_augmented")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                require(isinstance(row, dict), f"invalid JSONL row {number}: {path}")
                rows.append(row)
    return rows


def ratio(correct: int, denominator: int) -> dict:
    return {"correct": correct, "denominator": denominator,
            "rate": correct / denominator if denominator else None}


def load_source_requests() -> tuple[list[dict], dict]:
    manifest_path, input_path = INPUT_DIR / "manifest.json", INPUT_DIR / "inputs.jsonl"
    require(sha256_file(manifest_path) == INPUT_MANIFEST_SHA256
            and sha256_file(input_path) == INPUTS_SHA256,
            "source-only input artifacts differ")
    manifest, rows = read_json(manifest_path), read_jsonl(input_path)
    require(manifest.get("schema") == "document-phonetic-inputs-v1"
            and manifest.get("completed") is True
            and manifest.get("inputs_sha256") == INPUTS_SHA256
            and manifest.get("counts", {}).get("unique_inputs") == 10166
            and manifest.get("performance_labels_joined") is False,
            "source-only input manifest differs")
    require(len(rows) == 10166 and len({row.get("id") for row in rows}) == 10166,
            "source-only input rows differ")
    for row in rows:
        source = row.get("input")
        require(set(row) == {"id", "input"} and isinstance(source, str)
                and SUPPORTED.fullmatch(source) is not None
                and row["id"] == hashlib.sha256(source.encode()).hexdigest(),
                "source-only input identity differs")
    return rows, manifest


def authenticate_capture(directory: Path, arm: str, requests: list[dict]) -> dict:
    directory = Path(directory)
    report_path = directory / "report.json"
    completion_path = directory / "completion.json"
    prediction_path = directory / "predictions.jsonl"
    report, completion = read_json(report_path), read_json(completion_path)
    require(report.get("schema") == "document-phonetic-capture-v1"
            and report.get("completed") is True and report.get("arm") == arm
            and report.get("input_manifest_sha256") == INPUT_MANIFEST_SHA256
            and report.get("inputs_sha256") == INPUTS_SHA256
            and report.get("prediction_count") == len(requests)
            and report.get("beam_size") == 8
            and report.get("reference_targets_read") is False
            and report.get("quality_evaluation") is False
            and report.get("production_changed") is False,
            f"{arm} source-only capture contract differs")
    report_sha = sha256_file(report_path)
    completion_sha = sha256_file(completion_path)
    prediction_sha = sha256_file(prediction_path)
    require(prediction_sha == report.get("predictions_sha256")
            and completion.get("schema") == "observed-phonetic-pair-completion-v1"
            and completion.get("completed") is True
            and completion.get("report_sha256") == report_sha,
            f"{arm} capture closure differs")
    reference_paths = {str(RESERVE.relative_to(ROOT)), str(RAW.relative_to(ROOT))}
    pins = report.get("source_hashes")
    require(isinstance(pins, dict) and pins and not (set(pins) & reference_paths),
            f"{arm} capture improperly depends on references")
    for relative, expected in pins.items():
        path = ROOT / relative
        require(path.is_file() and sha256_file(path) == expected,
                f"{arm} source pin differs: {relative}")
    rows = read_jsonl(prediction_path)
    require(len(rows) == len(requests), f"{arm} prediction count differs")
    for expected, row in zip(requests, rows):
        require(set(row) == {"id", "input", "outputs", "latency_s"}
                and row["id"] == expected["id"] and row["input"] == expected["input"]
                and type(row["latency_s"]) in (int, float)
                and math.isfinite(row["latency_s"]) and row["latency_s"] >= 0,
                f"{arm} prediction identity differs")
        outputs, seen, previous = row["outputs"], set(), math.inf
        require(isinstance(outputs, list) and len(outputs) <= 8,
                f"{arm} output beam differs")
        for output in outputs:
            text, score = output.get("text"), output.get("score")
            require(set(output) == {"text", "score"}
                    and isinstance(text, str) and HANGUL.fullmatch(text) is not None
                    and 1 <= len(text) <= 24 and text not in seen
                    and romanize_word(text) == row["input"]
                    and type(score) in (int, float) and math.isfinite(score)
                    and score <= previous, f"{arm} output candidate differs")
            seen.add(text); previous = score
    return {"report": report, "predictions": rows,
            "report_sha256": report_sha, "completion_sha256": completion_sha,
            "predictions_sha256": prediction_sha}


def load_references() -> dict:
    require(sha256_file(RESERVE) == RESERVE_SHA256 and sha256_file(RAW) == RAW_SHA256,
            "reference source hash differs")
    selection, raw = read_jsonl(RESERVE), read_jsonl(RAW)
    require(len(selection) == 1046 and len({row.get("id") for row in selection}) == 1046,
            "reserve selection differs")
    raw_by_id = {row.get("id"): row for row in raw}
    require(None not in raw_by_id and len(raw_by_id) == len(raw), "raw IDs differ")
    occurrences = []
    for selected in selection:
        row = raw_by_id.get(selected["id"])
        require(row is not None and row.get("group_id") == selected.get("group_id"),
                "reserve mapping differs")
        text = row.get("text")
        require(isinstance(text, str) and text, "reserve reference text differs")
        occurrences.extend((romanize_word(surface), surface)
                           for surface in HANGUL.findall(text))
    counts = Counter(occurrences)
    variants = defaultdict(set)
    for source, surface in counts:
        variants[source].add(surface)
    supported = {source for source in variants if SUPPORTED.fullmatch(source)}
    require(len(occurrences) == 15812 and len(counts) == 10182
            and len(supported) == 10166
            and sum(count for pair, count in counts.items() if pair[0] not in supported) == 1,
            "reference census differs")
    return {"pair_counts": counts, "variants": dict(variants),
            "supported_inputs": supported}


def load_strata() -> dict:
    require(sha256_file(BASE) == BASE_SHA256 and sha256_file(DELTA) == DELTA_SHA256,
            "training source hash differs")
    control = {(row["roman"], row["surface"]) for row in read_jsonl(BASE)
               if row.get("split") == "train"}
    delta = {(row["roman"], row["surface"]) for row in read_jsonl(DELTA)}
    require(len(control) == 51361 and len(delta) == 32984 and not control & delta,
            "training strata differ")
    return {"control_pairs": control, "delta_pairs": delta,
            "control_sources": {pair[0] for pair in control},
            "delta_sources": {pair[0] for pair in delta}}


def pair_class(pair: tuple[str, str], strata: dict) -> str:
    if pair in strata["control_pairs"]:
        return "seen_control"
    if pair in strata["delta_pairs"]:
        return "delta_only"
    return "unseen_augmented"


def source_class(source: str, strata: dict) -> str:
    if source in strata["control_sources"]:
        return "seen_control"
    if source in strata["delta_sources"]:
        return "delta_only"
    return "unseen_augmented"


def spelling_metrics(pairs: list[tuple[str, str]], counts: Counter,
                     predictions: dict[str, list[str]]) -> dict:
    occurrence_total = sum(counts[pair] for pair in pairs)
    top1 = sum(bool(predictions.get(source)) and predictions[source][0] == surface
               for source, surface in pairs)
    top8 = sum(surface in predictions.get(source, ()) for source, surface in pairs)
    occurrence_top1 = sum(counts[pair] for pair in pairs
                          if predictions.get(pair[0]) and predictions[pair[0]][0] == pair[1])
    occurrence_top8 = sum(counts[pair] for pair in pairs
                          if pair[1] in predictions.get(pair[0], ()))
    return {"pairs": len(pairs), "occurrences": occurrence_total,
            "variant_recall_top1": ratio(top1, len(pairs)),
            "variant_recall_top8": ratio(top8, len(pairs)),
            "occurrence_weighted_top1": ratio(occurrence_top1, occurrence_total),
            "occurrence_weighted_top8": ratio(occurrence_top8, occurrence_total)}


def source_metrics(sources: list[str], variants: dict[str, set[str]], counts: Counter,
                   predictions: dict[str, list[str]]) -> dict:
    source_set = set(sources)
    top1 = sum(bool(predictions.get(source))
               and predictions[source][0] in variants[source] for source in sources)
    top8 = sum(bool(set(predictions.get(source, ())) & variants[source]) for source in sources)
    return {"roman_keys": len(sources),
            "reference_pairs": sum(len(variants[source]) for source in sources),
            "occurrences": sum(count for (source, _), count in counts.items()
                               if source in source_set),
            "acceptable_variant_top1": ratio(top1, len(sources)),
            "acceptable_variant_top8": ratio(top8, len(sources))}


def compute_metrics(rows: list[dict], references: dict, strata: dict) -> dict:
    predictions = {row["input"]: [candidate["text"] for candidate in row["outputs"]]
                   for row in rows}
    variants, counts = references["variants"], references["pair_counts"]
    pairs, sources = sorted(counts), sorted(variants)
    supported = references["supported_inputs"]
    require(set(predictions) == supported, "capture/reference supported inputs differ")
    pair_groups = {name: [pair for pair in pairs if pair_class(pair, strata) == name]
                   for name in STRATA}
    source_groups = {name: [source for source in sources
                            if source_class(source, strata) == name] for name in STRATA}
    all_spelling = spelling_metrics(pairs, counts, predictions)
    by_length = {}
    for length in sorted({len(source) for source in sources}):
        length_sources = [source for source in sources if len(source) == length]
        length_pairs = [pair for pair in pairs if len(pair[0]) == length]
        by_length[str(length)] = {
            "roman_acceptable_variant": source_metrics(
                length_sources, variants, counts, predictions),
            "all_spelling_variants": spelling_metrics(
                length_pairs, counts, predictions),
        }
    return {
        "roman_acceptable_variant": source_metrics(sources, variants, counts, predictions),
        "all_spelling_variants": all_spelling,
        "candidate_coverage": all_spelling["variant_recall_top8"],
        "empty_supported_outputs": sum(not predictions[source] for source in supported),
        "unsupported": {
            "roman_keys": len(set(variants) - supported),
            "reference_pairs": sum(source not in supported for source, _ in counts),
            "occurrences": sum(count for (source, _), count in counts.items()
                               if source not in supported),
        },
        "candidate_outputs": {"total": sum(map(len, predictions.values())),
                              "maximum_per_input": max(map(len, predictions.values()), default=0)},
        "by_exact_pair_stratum": {
            name: spelling_metrics(pair_groups[name], counts, predictions) for name in STRATA},
        "by_roman_key_stratum": {
            name: source_metrics(source_groups[name], variants, counts, predictions)
            for name in STRATA},
        "by_input_length": by_length,
    }


def verify(evaluation: Path, control_capture: Path, augmented_capture: Path,
           output: Path) -> dict:
    output = Path(output)
    require(not output.exists(), "verification output already exists")
    requests, input_manifest = load_source_requests()
    # No gold-bearing report or reference is opened above this boundary.
    captures = {
        "control": authenticate_capture(control_capture, "control", requests),
        "augmented": authenticate_capture(augmented_capture, "augmented", requests),
    }
    require(captures["control"]["report"]["paired_initial_checkpoint_sha256"]
            == captures["augmented"]["report"]["paired_initial_checkpoint_sha256"],
            "paired initialization differs")

    report_path = Path(evaluation) / "report.json"
    report = read_json(report_path)
    expected_capture_hashes = {
        arm: {key: value[key] for key in
              ("report_sha256", "completion_sha256", "predictions_sha256")}
        for arm, value in captures.items()}
    require(report.get("schema") == "document-phonetic-pair-evaluation-v1"
            and report.get("completed") is True
            and report.get("quality_gate_applied") is False
            and report.get("checkpoint_selected") is False
            and report.get("capture_closed_before_reference_join") is True
            and report.get("capture_hashes_unchanged_after_reference_join") is True
            and report.get("capture_artifacts") == expected_capture_hashes,
            "evaluation identity/closure differs")
    expected_checkpoints = {
        arm: {field: capture["report"][field] for field in
              ("checkpoint_sha256", "training_report_sha256",
               "paired_initial_checkpoint_sha256")}
        for arm, capture in captures.items()}
    require(report.get("checkpoints") == expected_checkpoints,
            "evaluation checkpoint identities differ")

    references, strata = load_references(), load_strata()
    recomputed = {arm: compute_metrics(capture["predictions"], references, strata)
                  for arm, capture in captures.items()}
    require(report.get("metrics") == recomputed, "aggregate metrics differ")
    for arm, directory in (("control", Path(control_capture)),
                           ("augmented", Path(augmented_capture))):
        require(sha256_file(directory / "report.json")
                    == expected_capture_hashes[arm]["report_sha256"]
                and sha256_file(directory / "completion.json")
                    == expected_capture_hashes[arm]["completion_sha256"]
                and sha256_file(directory / "predictions.jsonl")
                    == expected_capture_hashes[arm]["predictions_sha256"],
                f"{arm} capture changed during independent late join")
    pair_inventory = Counter(pair_class(pair, strata) for pair in references["pair_counts"])
    source_inventory = Counter(source_class(source, strata) for source in references["variants"])
    occurrence_inventory = Counter()
    for pair, count in references["pair_counts"].items():
        occurrence_inventory[pair_class(pair, strata)] += count
    expected_inventory = {
        "exact_pairs": {name: pair_inventory[name] for name in STRATA},
        "roman_keys": {name: source_inventory[name] for name in STRATA},
        "occurrences": {name: occurrence_inventory[name] for name in STRATA},
    }
    expected_sources = {
        str(RESERVE.relative_to(ROOT)): RESERVE_SHA256,
        str(RAW.relative_to(ROOT)): RAW_SHA256,
        str(BASE.relative_to(ROOT)): BASE_SHA256,
        str(DELTA.relative_to(ROOT)): DELTA_SHA256,
        str(EVALUATOR.relative_to(ROOT)): sha256_file(EVALUATOR),
    }
    require(report.get("strata_inventory") == expected_inventory
            and report.get("source_hashes") == expected_sources
            and report.get("reference_counts") == input_manifest["counts"],
            "evaluation inventory/source evidence differs")
    contracts = report.get("contracts", {})
    require(contracts.get("aggregate_only_no_examples") is True
            and contracts.get("beam_size") == 8
            and contracts.get("all_spelling_variant_denominator") is True
            and contracts.get("unsupported_reference_retained_in_denominators") is True
            and contracts.get("model_activity") is False
            and contracts.get("performance_tuning") is False
            and contracts.get("production_changed") is False,
            "evaluation contracts differ")
    require("examples" not in report and "errors" not in report,
            "evaluation contains item-level outcomes")
    proof = {
        "schema": "document-phonetic-pair-independent-verification-v1",
        "verified": True,
        "evaluation_report_sha256": sha256_file(report_path),
        "verifier_sha256": sha256_file(Path(__file__)),
        "capture_artifacts": expected_capture_hashes,
        "source_hashes": expected_sources,
        "metrics_sha256": hashlib.sha256(canonical(recomputed)).hexdigest(),
        "reference_counts": input_manifest["counts"],
        "contracts": {
            "captures_authenticated_before_report_or_references": True,
            "metrics_recomputed_without_producer_import": True,
            "unsupported_retained_in_denominators": True,
            "model_loaded": False, "model_inference": False,
            "checkpoint_selection": False, "production_changed": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        target = temporary / output.name
        target.write_bytes(canonical(proof) + b"\n")
        with target.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(target, output)
        temporary.rmdir()
    except Exception:
        for child in temporary.iterdir():
            child.unlink()
        temporary.rmdir()
        raise
    return proof


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--evaluation", type=Path, required=True)
    value.add_argument("--control-capture", type=Path, required=True)
    value.add_argument("--augmented-capture", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    result = verify(args.evaluation, args.control_capture, args.augmented_capture, args.output)
    print(json.dumps({"verified": result["verified"],
                      "evaluation_report_sha256": result["evaluation_report_sha256"]}))
