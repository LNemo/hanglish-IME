#!/usr/bin/env python3
"""Capture existing phonetic beams for every eligible natural TRAIN source chunk."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import resource
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.neural_converter import LETTERS, PROTECTED, TOKENS
from hanglish.core.phonology import romanize_word


ROWS = ROOT / ".omx/artifacts/semantic-pair-training-input-v2/rows.jsonl"
MANIFEST = ROOT / ".omx/artifacts/semantic-pair-training-input-v2/manifest.json"
CHECKPOINT = ROOT / "checkpoints/phonetic_transducer.pt"
METADATA = CHECKPOINT.with_suffix(".json")
OUTPUT = ROOT / ".omx/artifacts/phonetic-training-beam-census-v1"
ROWS_SHA256 = "6a1f7f61a7f7ffeb9209772559412931137d71ef0173d7429264d9cb53aed396"
MANIFEST_SHA256 = "3ab561d1a767e7a65b4706c2cd0f668366693536849e17f2f3e620fbcf5550a6"
CHECKPOINT_SHA256 = "d287b7992a75a96f2639e9520faba597bb34ebb0bcc502b85960cbb5a310ba43"
METADATA_SHA256 = "bc3946a84b8ecf5a7522ae63ea3bd2b2ad92745f9087d6ff8536f9d1424643da"
EXPECTED_SOURCE_ROWS = 2048
EXPECTED_UNIQUE_INPUTS = 18122
PROFILE = {
    "cpu_threads": 2,
    "rss_limit_bytes": 1024**3,
    "time_limit_seconds": 1800,
    "pilot_rows": 256,
    "pilot_multiplier": 1.3,
    "setup_margin_seconds": 60,
    "beam_size": 8,
}
DEPENDENCIES = (
    "hanglish/core/neural_converter.py",
    "hanglish/core/phonology.py",
    "hanglish/models/phonetic_transducer.py",
)
FORBIDDEN_SOURCE_FIELDS = frozenset({"expected", "accepted", "answers", "labels", "target"})


class ResourceGateError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_key(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def digest_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def runtime_chunk_records(text: str) -> tuple[tuple[str, bool], ...]:
    """Return eligible letter spans and their production protection state."""
    if not isinstance(text, str):
        raise ValueError("source input must be a string")
    if len(LETTERS.findall(text)) > 128:
        return ()
    found = []
    for match in TOKENS.finditer(text):
        chunk = match.group()
        if chunk.isspace():
            continue
        # Direct beam coverage deliberately includes eligible letter spans in
        # protected chunks.  The later target join can therefore distinguish a
        # model beam miss from the production PROTECTED gate without rerunning
        # the model.  Using the production matcher here freezes that boundary.
        protected = bool(PROTECTED.fullmatch(chunk))
        for part in LETTERS.finditer(chunk):
            source = part.group()
            if (source.isascii() and source.isalpha() and source.islower()
                    and 2 <= len(source) <= 40):
                found.append((source, protected))
    return tuple(found)


def runtime_chunks(text: str) -> tuple[str, ...]:
    return tuple(source for source, _ in runtime_chunk_records(text))


def load_source_rows(path: Path = ROWS, expected_sha: str = ROWS_SHA256,
                     expected_rows: int = EXPECTED_SOURCE_ROWS) -> list[dict]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("source-only rows SHA-256 differs")
    selected = []
    seen = set()
    for line in raw.splitlines():
        row = json.loads(line)
        if row.get("source_role") != "natural_fit":
            continue
        if FORBIDDEN_SOURCE_FIELDS.intersection(row):
            raise ValueError("source-only row contains a target field")
        identity, source = row.get("id"), row.get("input")
        if (not isinstance(identity, str) or not identity or identity in seen
                or not isinstance(source, str)):
            raise ValueError("natural TRAIN source row is malformed")
        seen.add(identity)
        selected.append({"id": identity, "input": source})
    if len(selected) != expected_rows:
        raise ValueError("natural TRAIN source row count differs")
    return selected


def quantile_indices(size: int, count: int) -> tuple[int, ...]:
    if size <= 0 or count <= 0:
        return ()
    count = min(size, count)
    if count == 1:
        return (0,)
    values = tuple(index * (size - 1) // (count - 1) for index in range(count))
    if len(set(values)) != count:
        raise ValueError("pilot quantiles are not unique")
    return values


def build_selection(rows: list[dict], *, expected_inputs: int = EXPECTED_UNIQUE_INPUTS,
                    pilot_rows: int | None = None) -> list[dict]:
    pilot_rows = PROFILE["pilot_rows"] if pilot_rows is None else pilot_rows
    membership: dict[str, set[str]] = {}
    protected_membership: dict[str, set[str]] = {}
    unprotected_membership: dict[str, set[str]] = {}
    for row in rows:
        for source, protected in runtime_chunk_records(row["input"]):
            membership.setdefault(source, set()).add(row["id"])
            destination = protected_membership if protected else unprotected_membership
            destination.setdefault(source, set()).add(row["id"])
    base = sorted(membership, key=lambda source: (len(source), source))
    if len(base) != expected_inputs:
        raise ValueError("unique eligible source input count differs")
    pilot_indexes = quantile_indices(len(base), pilot_rows)
    pilot = [base[index] for index in pilot_indexes]
    pilot_set = set(pilot)
    ordered = pilot + [source for source in base if source not in pilot_set]
    if len(ordered) != len(base) or set(ordered) != set(base):
        raise ValueError("capture order does not cover source inputs exactly once")
    return [
        {"index": index, "input": source, "row_ids": sorted(membership[source]),
         "protected_row_ids": sorted(protected_membership.get(source, ())),
         "unprotected_row_ids": sorted(unprotected_membership.get(source, ()))}
        for index, source in enumerate(ordered)
    ]


def canonical_jsonl(rows) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def selection_stats(rows: list[dict], selection: list[dict]) -> dict:
    fragmented = sum(len(LETTERS.findall(row["input"])) > 128 for row in rows)
    eligible_rows = sum(bool(runtime_chunks(row["input"])) for row in rows)
    return {
        "source_rows": len(rows),
        "rows_with_eligible_inputs": eligible_rows,
        "rows_without_eligible_inputs": len(rows) - eligible_rows,
        "fragmented_rows_bypassed": fragmented,
        "unique_inputs": len(selection),
        "min_input_length": min((len(row["input"]) for row in selection), default=None),
        "max_input_length": max((len(row["input"]) for row in selection), default=None),
        "protected_only_inputs": sum(
            bool(row["protected_row_ids"]) and not row["unprotected_row_ids"] for row in selection
        ),
        "mixed_protection_inputs": sum(
            bool(row["protected_row_ids"]) and bool(row["unprotected_row_ids"])
            for row in selection
        ),
        "unprotected_only_inputs": sum(
            bool(row["unprotected_row_ids"]) and not row["protected_row_ids"] for row in selection
        ),
        "protected_row_memberships": sum(len(row["protected_row_ids"]) for row in selection),
        "unprotected_row_memberships": sum(len(row["unprotected_row_ids"]) for row in selection),
    }


def source_pins(rows_path: Path = ROWS, manifest_path: Path = MANIFEST,
                checkpoint: Path = CHECKPOINT) -> dict[str, str]:
    paths = [Path(__file__), rows_path, manifest_path, checkpoint,
             checkpoint.with_suffix(".json"), *(ROOT / name for name in DEPENDENCIES)]
    return {path_key(path): file_sha256(path) for path in paths}


def validate_outputs(source: str, outputs, beam_size: int = PROFILE["beam_size"]) -> list[dict]:
    if not isinstance(outputs, (list, tuple)) or len(outputs) > beam_size:
        raise ValueError("transducer output violates the beam bound")
    records, seen = [], set()
    previous_score = float("inf")
    for item in outputs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("transducer output item is malformed")
        text, score = item
        if (not isinstance(text, str) or not text
                or any(not "가" <= char <= "힣" for char in text)
                or text in seen or romanize_word(text) != source):
            raise ValueError("transducer output is not unique forward-valid Hangul")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("transducer score is not finite numeric data")
        if score > previous_score:
            raise ValueError("transducer scores are not in non-increasing order")
        seen.add(text)
        records.append({"text": text, "score": float(score)})
        previous_score = score
    return records


def default_model_loader(checkpoint: Path):
    import torch
    from hanglish.models.phonetic_transducer import PhoneticTransducer

    torch.set_num_threads(PROFILE["cpu_threads"])
    torch.set_num_interop_threads(1)
    return PhoneticTransducer.load(checkpoint, device="cpu")


def _failure_payload(*, state, error, completed, capture_path, selection_sha,
                     ordered_sha, pins_start, pins_end, pilot, selection_summary,
                     elapsed_seconds, peak_rss_bytes, model_inference_run):
    return {
        "schema": "phonetic-training-beam-census-failure-v1",
        "completed": False,
        "state": state,
        "error_type": type(error).__name__,
        "error": str(error),
        "completed_inputs": completed,
        "capture_sha256": file_sha256(capture_path),
        "selection_sha256": selection_sha,
        "ordered_inputs_sha256": ordered_sha,
        "source_hashes_start": pins_start,
        "source_hashes_end": pins_end,
        "source_hashes_unchanged": pins_start == pins_end,
        "pilot": pilot,
        "selection": selection_summary,
        "selection_order": (
            "256 evenly spaced quantiles of (length,input), including endpoints; "
            "then remaining (length,input) order"
        ),
        "profile": PROFILE,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "peak_rss_bytes": peak_rss_bytes,
        "target_fields_read": False,
        "model_inference_run": model_inference_run,
        "training_run": False,
    }


def emit_progress(event: str, **values) -> None:
    print(json.dumps({"event": event, **values}, ensure_ascii=False, sort_keys=True), flush=True)


def run(output: Path = OUTPUT, *, rows_path: Path = ROWS,
        manifest_path: Path = MANIFEST, checkpoint: Path = CHECKPOINT,
        rows_sha: str = ROWS_SHA256, manifest_sha: str = MANIFEST_SHA256,
        checkpoint_sha: str = CHECKPOINT_SHA256, metadata_sha: str = METADATA_SHA256,
        expected_rows: int = EXPECTED_SOURCE_ROWS,
        expected_inputs: int = EXPECTED_UNIQUE_INPUTS,
        model_loader=default_model_loader, monotonic=time.monotonic,
        rss_reader=rss_bytes) -> dict:
    if output.exists():
        raise FileExistsError(output)
    started = monotonic()
    if file_sha256(manifest_path) != manifest_sha:
        raise ValueError("source-only manifest SHA-256 differs")
    if file_sha256(checkpoint) != checkpoint_sha:
        raise ValueError("phonetic checkpoint SHA-256 differs")
    if file_sha256(checkpoint.with_suffix(".json")) != metadata_sha:
        raise ValueError("phonetic checkpoint metadata SHA-256 differs")
    rows = load_source_rows(rows_path, rows_sha, expected_rows)
    selection = build_selection(rows, expected_inputs=expected_inputs)
    selection_summary = selection_stats(rows, selection)
    selection_raw = canonical_jsonl(selection)
    ordered_sha = digest_json([row["input"] for row in selection])
    del rows

    pins_start = source_pins(rows_path, manifest_path, checkpoint)
    output.mkdir(parents=True)
    selection_path = output / "selection.jsonl"
    selection_path.write_bytes(selection_raw)
    selection_sha = file_sha256(selection_path)
    pilot = None
    completed = 0
    outputs_count = 0
    empty_count = 0
    capture_path = output / "capture.jsonl"
    capture_path.touch(exist_ok=False)
    inference_started = None
    setup_seconds = None
    model = None
    generation_attempted = False
    try:
        model = model_loader(checkpoint)
        setup_seconds = monotonic() - started
        if (rss_reader() > PROFILE["rss_limit_bytes"]
                or monotonic() - started > PROFILE["time_limit_seconds"]):
            raise ResourceGateError("beam census exceeded its resource bound after model load")
        emit_progress("model_ready", setup_seconds=round(setup_seconds, 3),
                      inputs=len(selection), peak_rss_bytes=rss_reader())
        inference_started = monotonic()
        with capture_path.open("a") as stream:
            for selected in selection:
                generation_attempted = True
                records = validate_outputs(
                    selected["input"], model.generate(selected["input"], beam_size=PROFILE["beam_size"])
                )
                captured = {"index": selected["index"], "input": selected["input"],
                            "outputs": records}
                stream.write(json.dumps(captured, ensure_ascii=False, sort_keys=True,
                                        separators=(",", ":")) + "\n")
                stream.flush()
                completed += 1
                outputs_count += len(records)
                empty_count += not records
                elapsed = monotonic() - started
                if rss_reader() > PROFILE["rss_limit_bytes"] or elapsed > PROFILE["time_limit_seconds"]:
                    raise ResourceGateError("beam census exceeded its runtime resource bound")
                if completed == min(PROFILE["pilot_rows"], len(selection)):
                    generation_seconds = monotonic() - inference_started
                    projected = (setup_seconds + PROFILE["setup_margin_seconds"]
                                 + PROFILE["pilot_multiplier"] * generation_seconds
                                 / completed * len(selection))
                    pilot = {
                        "rows": completed,
                        "generation_seconds": round(generation_seconds, 3),
                        "setup_seconds": round(setup_seconds, 3),
                        "projected_total_seconds": round(projected, 3),
                        "includes_shortest_and_longest": len(selection) <= 1 or (
                            min(selection, key=lambda row: (len(row["input"]), row["input"]))["input"]
                            in {row["input"] for row in selection[:completed]}
                            and max(selection, key=lambda row: (len(row["input"]), row["input"]))["input"]
                            in {row["input"] for row in selection[:completed]}
                        ),
                        "extrapolation_limit": (
                            "Length-stratified quantiles reduce ordering bias but do not model "
                            "candidate-dependent beam branching; the 1.3 multiplier and setup margin "
                            "are conservative estimate adjustments. The elapsed-time guard is the bound."
                        ),
                    }
                    if projected > PROFILE["time_limit_seconds"]:
                        raise ResourceGateError("pilot projects beyond the beam census time bound")
                    emit_progress("pilot_qualified", **pilot, peak_rss_bytes=rss_reader())
                elif completed % 256 == 0:
                    emit_progress("beam_census_progress", completed=completed,
                                  total=len(selection), elapsed_seconds=round(elapsed, 3),
                                  peak_rss_bytes=rss_reader())
        pins_end = source_pins(rows_path, manifest_path, checkpoint)
        if pins_end != pins_start:
            raise RuntimeError("beam census sources changed during capture")
        if completed != len(selection):
            raise RuntimeError("beam census did not cover every selected input")
        if (rss_reader() > PROFILE["rss_limit_bytes"]
                or monotonic() - started > PROFILE["time_limit_seconds"]):
            raise ResourceGateError("beam census exceeded its resource bound during final validation")
    except Exception as error:
        pins_end = source_pins(rows_path, manifest_path, checkpoint)
        payload = _failure_payload(
            state="resource_failed" if isinstance(error, ResourceGateError) else "failed",
            error=error, completed=completed, capture_path=capture_path,
            selection_sha=selection_sha, ordered_sha=ordered_sha,
            pins_start=pins_start, pins_end=pins_end, pilot=pilot,
            selection_summary=selection_summary,
            elapsed_seconds=monotonic() - started,
            peak_rss_bytes=rss_reader(), model_inference_run=generation_attempted,
        )
        (output / "failure.json").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        )
        raise
    finally:
        model = None

    report = {
        "schema": "phonetic-training-beam-census-v1",
        "completed": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_role": "natural_fit",
        "source_rows": expected_rows,
        "selection": selection_summary,
        "selection_order": (
            "256 evenly spaced quantiles of (length,input), including endpoints; "
            "then remaining (length,input) order"
        ),
        "unique_inputs": len(selection),
        "captured_inputs": completed,
        "outputs": outputs_count,
        "empty_outputs": empty_count,
        "selection_sha256": selection_sha,
        "capture_sha256": file_sha256(capture_path),
        "ordered_inputs_sha256": ordered_sha,
        "source_hashes_start": pins_start,
        "source_hashes_end": pins_end,
        "source_hashes_unchanged": True,
        "checkpoint_sha256": pins_start[path_key(checkpoint)],
        "checkpoint_metadata_sha256": pins_start[path_key(checkpoint.with_suffix('.json'))],
        "profile": PROFILE,
        "pilot": pilot,
        "elapsed_seconds": round(monotonic() - started, 3),
        "inference_seconds": round(monotonic() - inference_started, 3),
        "peak_rss_bytes": rss_reader(),
        "torch_version": importlib.metadata.version("torch"),
        "target_fields_read": False,
        "model_inference_run": True,
        "training_run": False,
        "validation": {
            "all_selected_inputs_captured_once": True,
            "outputs_are_unique_forward_valid_hangul": True,
            "scores_are_finite_non_increasing": True,
            "maximum_outputs_per_input": PROFILE["beam_size"],
        },
        "interpretation_limits": [
            "The capture measures direct transducer beam membership only.",
            "No accepted Hangul targets are read or scored by this producer.",
            "Eligible letter spans inside production-protected chunks are included so a later join can measure that gate.",
            "Runtime dictionary gates, option merging, sentence composition, and ranking are not measured.",
        ],
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    if (rss_reader() > PROFILE["rss_limit_bytes"]
            or monotonic() - started > PROFILE["time_limit_seconds"]):
        (output / "report.json").unlink()
        error = ResourceGateError("beam census exceeded its resource bound while writing the report")
        payload = _failure_payload(
            state="resource_failed", error=error, completed=completed,
            capture_path=capture_path, selection_sha=selection_sha, ordered_sha=ordered_sha,
            pins_start=pins_start, pins_end=source_pins(rows_path, manifest_path, checkpoint),
            pilot=pilot, selection_summary=selection_summary,
            elapsed_seconds=monotonic() - started, peak_rss_bytes=rss_reader(),
            model_inference_run=generation_attempted,
        )
        (output / "failure.json").write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        )
        raise error
    emit_progress("beam_census_complete", completed=completed,
                  elapsed_seconds=report["elapsed_seconds"], peak_rss_bytes=report["peak_rss_bytes"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, sort_keys=True))
