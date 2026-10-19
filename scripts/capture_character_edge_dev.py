#!/usr/bin/env python3
"""Capture fixed character-edge predictions on the closed, source-only DEV union.

The ``current-*`` schemas used by the census/scoring helpers are legacy wire
names.  Every row produced here is explicitly ``dataset_split: dev`` and is
derived from the frozen DEV runtime capture; no FIT row or DEV label is read.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time

from scripts.current_fit_recombination_census import prepare_record, probe_groups
from scripts.current_recombination_scoring import prepare_row, score_prepared
from scripts.current_recombination_selection import calibrated_score, render_public
from scripts.evaluate_current_phonetic_choice_dev import FROZEN as DEV_FROZEN
from scripts.train_character_edge_full_fit import ResourceGuard


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".omx/artifacts/phonetic-choice-runtime-dev-v1"
COMPARATOR = ROOT / ".omx/artifacts/score-scale-runtime-dev-v1"
INTAKE = ROOT / ".omx/artifacts/current-phonetic-choice-dev-intake-v1.json"
CHARACTER = ROOT / ".omx/artifacts/character-edge-full-fit-v1"
CALIBRATION = ROOT / ".omx/artifacts/phonetic-score-scale-training-v1/model.json"
CHARACTER_SOURCE = ROOT / "hanglish/models/character_edge_residual.py"
GENERATOR = ROOT / "hanglish/core/source_edge_recombination.py"
CENSUS_HELPER = ROOT / "scripts/current_fit_recombination_census.py"
SCORING_HELPER = ROOT / "scripts/current_recombination_scoring.py"
SELECTION_HELPER = ROOT / "scripts/current_recombination_selection.py"
RESOURCE_HELPER = ROOT / "scripts/source_span_resources.py"
PLAN = ROOT / ".omx/plans/character-edge-dev-v1.md"
SENTENCE_CHECKPOINT = ROOT / "checkpoints/sentence_ranker"
PRODUCER = Path(__file__).resolve()
RESOURCE_GUARD_SOURCE = ROOT / "scripts/train_character_edge_full_fit.py"

SCHEMA = "character-edge-dev-capture-v1"
ROWS = 316
SOURCE_CAPTURE_SHA = "e0447efa305e1d4f61052866b1ee121a62b400f3adf163dfdeac1b0bb4734bf5"
COMPARATOR_CAPTURE_SHA = "9a47e0fb3174f0923f4eb39e0a7342316f0c3142dfee892b7d4f031afe2325e7"
CHARACTER_MODEL_SHA = "c1bc8fb3b8524ea5ed9462e0cc4f2ab28e9ccd4ef7e25623da9661d89e06794f"
CHARACTER_REPORT_SHA = "b3e1eb29e15a79b99217cb7dc20be0cc547ead30f3c204e3b11bc19feb1e6277"
CHARACTER_PROOF_SHA = "78ea1702f99a0ed8f319cdfb34c23c09943e6b3ef378a15b9644bc3093e5c6d4"
CHARACTER_SOURCE_SHA = "d30c8892efbd5e22c9731c138b761b5d78f49783b60e7c7b8db3497eae0bd841"
SENTENCE_SOURCE_SHA = "72ed3e7223aac215d692f96890ab642cf9c1da82d25c4485c5b901ff832bd1ca"
RESOURCE_GUARD_SOURCE_SHA = "7ccfa8b98aac5d651e54d7bc9343f64822b1cc58efa48ccc488e8909ca8c2ad1"
CALIBRATION_SHA = "e07462d50da0b67304a82a9a261068ccc84b8055a99b5d70af56db530d245838"
LIMITS = {"whole_seconds": 1800.0, "row_seconds": 30.0,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2,
          "output_bytes": 2 * 1024**3}
ARMS = ("current_production", "old_calibrated", "expanded_calibrated",
        "old_character", "expanded_character")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                require(isinstance(row, dict), f"non-object row in {path}")
                yield row


def index_jsonl(path: Path) -> list[dict]:
    """Index closed JSONL by byte offset while retaining only source identity."""
    result = []
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            request = row.get("request") if isinstance(row, dict) else None
            source = request.get("input") if isinstance(request, dict) else None
            require(isinstance(row.get("id"), str)
                    and request == {"input": source, "context": ""}
                    and isinstance(source, str) and source,
                    "indexed source identity differs")
            result.append({"id": row["id"], "input": source, "offset": offset})
    return result


def read_indexed(path: Path, entry: dict) -> dict:
    with path.open("rb") as stream:
        stream.seek(entry["offset"])
        row = json.loads(stream.readline())
    require(row.get("id") == entry["id"]
            and row.get("request") == {"input": entry["input"], "context": ""},
            "indexed row changed")
    return row


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def source_pins() -> dict[str, str]:
    """Reuse only the SOURCE/COMPARATOR/INTAKE pins from the frozen DEV screen."""
    allowed = (SOURCE.resolve(), COMPARATOR.resolve(), INTAKE.resolve())
    pins = {str(path.resolve()): expected for path, expected in DEV_FROZEN.items()
            if path.resolve() in allowed or SOURCE.resolve() in path.resolve().parents
            or COMPARATOR.resolve() in path.resolve().parents}
    require(pins.get(str((SOURCE / "capture.jsonl").resolve())) == SOURCE_CAPTURE_SHA
            and pins.get(str((COMPARATOR / "capture.jsonl").resolve()))
                == COMPARATOR_CAPTURE_SHA
            and pins.get(str(INTAKE.resolve())) is not None,
            "frozen DEV source/comparator pin selection differs")
    return pins


def authenticate() -> dict[str, str]:
    pins = source_pins()
    source_report = read_json(SOURCE / "report.json")
    runtime_pins = source_report.get("pins")
    require(isinstance(runtime_pins, dict), "source report runtime pins missing")
    sentence_prefix = str(SENTENCE_CHECKPOINT.resolve()) + "/"
    sentence_pins = {path: expected for path, expected in runtime_pins.items()
                     if path.startswith(sentence_prefix)}
    require(len(sentence_pins) == 7, "frozen sentence ranker pins differ")
    pins.update(sentence_pins)
    pins.update({
        str((CHARACTER / "model.pt").resolve()): CHARACTER_MODEL_SHA,
        str((CHARACTER / "report.json").resolve()): CHARACTER_REPORT_SHA,
        str((CHARACTER / "checkpoint-verification.json").resolve()): CHARACTER_PROOF_SHA,
        str(CALIBRATION.resolve()): CALIBRATION_SHA,
        str(CHARACTER_SOURCE.resolve()): CHARACTER_SOURCE_SHA,
        str((ROOT / "hanglish/models/sentence_ranker.py").resolve()): SENTENCE_SOURCE_SHA,
        str(RESOURCE_GUARD_SOURCE.resolve()): RESOURCE_GUARD_SOURCE_SHA,
        str(GENERATOR.resolve()): "c4e7d7c59c3edb8623b28dc933f429b220b4d17ff58416d47449eeaa4d78d119",
        str(CENSUS_HELPER.resolve()): "5247b6b9d9c48edb66a43cacc7e3f370f942fdeb3becd00bfa07f5a648878135",
        str(SCORING_HELPER.resolve()): "ac00ac7726fa33a2bcb22ffeebbe1ee58c96b072f839acd80c265fffd9730158",
        str(RESOURCE_HELPER.resolve()): "4bdf5f598677d1d5c94bcca39545015e7ae12370b0a91482aca1ba42e11a75bd",
    })
    # The selection helper and plan are intentionally pinned to their current,
    # reviewable source because the parent plan is created for this capture.
    pins[str(SELECTION_HELPER.resolve())] = digest(SELECTION_HELPER)
    pins[str(PLAN.resolve())] = digest(PLAN)
    pins[str(PRODUCER)] = digest(PRODUCER)
    for path, expected in pins.items():
        require(Path(path).is_file() and digest(Path(path)) == expected,
                "frozen input differs: " + path)
    report, proof = read_json(CHARACTER / "report.json"), read_json(
        CHARACTER / "checkpoint-verification.json")
    source_proof = read_json(SOURCE / "independent-verification.json")
    comparator_report = read_json(COMPARATOR / "report.json")
    comparator_proof = read_json(COMPARATOR / "independent-verification.json")
    require(report.get("completed") is True and report.get("hashes", {}).get("model.pt")
            == CHARACTER_MODEL_SHA and report.get("model_source_sha256") == CHARACTER_SOURCE_SHA
            and proof.get("verified") is True
            and proof.get("report_sha256") == CHARACTER_REPORT_SHA,
            "character checkpoint report/proof differs")
    require(source_report.get("completed") is True
            and source_report.get("capture_sha256") == SOURCE_CAPTURE_SHA
            and source_proof.get("verified") is True
            and source_proof.get("capture_sha256") == SOURCE_CAPTURE_SHA
            and comparator_report.get("completed") is True
            and comparator_report.get("artifacts", {}).get("capture.jsonl")
                == COMPARATOR_CAPTURE_SHA
            and comparator_report.get("labels_opened") is False
            and comparator_proof.get("verified") is True,
            "verified DEV source/comparator authorization differs")
    return pins


def raw_paths_public(source: str, paths: list) -> list[dict]:
    pool = [{"text": row[0], "score": float(row[1]), "segments": row[2]}
            for row in paths]
    return render_public(source, pool)


def adapt_source_row(row: dict) -> dict:
    """Adapt the DEV full-union capture to the frozen helper wire shape."""
    request, proposed, union = row.get("request"), row.get("proposed"), row.get("union_reference")
    source = request.get("input") if isinstance(request, dict) else None
    ranked = proposed.get("sentence_ranking_applied") if isinstance(proposed, dict) else None
    paths = row.get("base_paths")
    require(isinstance(row.get("id"), str) and request == {"input": source, "context": ""}
            and isinstance(source, str) and source and isinstance(paths, list)
            and type(ranked) is bool and isinstance(union, dict)
            and isinstance(proposed.get("phonetic_choice"), dict),
            "DEV source row differs")
    public = union.get("candidates")
    require(isinstance(public, list) and public
            and ((ranked and paths and raw_paths_public(source, paths) == public)
                 or (not ranked and not paths and proposed.get("candidates") == public)),
            "raw full-union public seed differs")
    return {"id": row["id"], "component_id": None,
            "request": deepcopy(request), "base_paths": deepcopy(paths),
            "response": {"input": source, "context": "",
                         "external_context_used": False,
                         "sentence_ranking_applied": ranked,
                         "candidates": deepcopy(public),
                         "phonetic_choice": deepcopy(proposed["phonetic_choice"])}}


def _pool(rows: list[dict], calibration: dict) -> tuple[list[dict], list[float]]:
    return ([{"text": row["text"], "segments": deepcopy(row["segments"])} for row in rows],
            [float(calibrated_score(row, calibration)) for row in rows])


def _character(model, source: str, rows: list[dict], calibration: dict):
    candidates, baseline = _pool(rows, calibration)
    result = model(model.prepare(source, candidates, baseline))["scores"]
    values = [float(value) for value in result]
    require(len(values) == len(rows) and all(math.isfinite(value) for value in values),
            "character score output differs")
    public = render_public(source, [dict(candidate, score=score)
                                    for candidate, score in zip(candidates, values)])
    return values, public


def capture_row(source_row: dict, comparator_row: dict, source_index: int,
                sentence_ranker, character_model, calibration: dict,
                generator=None) -> dict:
    original = adapt_source_row(source_row)
    kwargs = {} if generator is None else {"generator": generator}
    census = prepare_record(original,
        {"id": original["id"], "input": original["request"]["input"], "component_id": None},
        source_shard=0, source_row_index=source_index,
        capture_sha=SOURCE_CAPTURE_SHA, **kwargs)
    prepared = prepare_row(original, census, sentence_ranker)
    scored = score_prepared(prepared, sentence_ranker, verify_original_parity=False)
    require(scored.get("original_score_parity") is None,
            "original sentence scores were unexpectedly rescored")
    source, ranked = original["request"]["input"], scored["ranked"]
    actual = comparator_row.get("response", {}).get("candidates")
    require(comparator_row.get("id") == original["id"]
            and comparator_row.get("request") == original["request"]
            and isinstance(actual, list) and actual, "production comparator alignment differs")
    predictions = {"current_production": deepcopy(actual)}
    character_scores = {"old_character": [], "expanded_character": []}
    if not ranked:
        fallback = scored.get("fallback_public")
        require(fallback == original["response"]["candidates"]
                and not scored["original"] and not scored["novel"],
                "no-pool fallback differs")
        for name in ARMS[1:]:
            predictions[name] = deepcopy(fallback)
    else:
        originals = scored["original"]
        novel = [row for row in scored["novel"] if row["status"] == "scored"]
        old_candidates, old_baseline = _pool(originals, calibration)
        expanded_candidates, expanded_baseline = _pool(originals + novel, calibration)
        predictions["old_calibrated"] = render_public(source, [
            dict(candidate, score=score) for candidate, score in zip(old_candidates, old_baseline)])
        predictions["expanded_calibrated"] = render_public(source, [
            dict(candidate, score=score) for candidate, score in zip(expanded_candidates, expanded_baseline)])
        character_scores["old_character"], predictions["old_character"] = _character(
            character_model, source, originals, calibration)
        character_scores["expanded_character"], predictions["expanded_character"] = _character(
            character_model, source, originals + novel, calibration)
    require(set(predictions) == set(ARMS), "prediction arms differ")
    return {"id": original["id"], "request": deepcopy(original["request"]),
            "dataset_split": "dev", "source_row_index": source_index,
            "census": census, "scoring": scored,
            "character_scores": character_scores, "predictions": predictions}


def _load_models():
    import torch
    from hanglish.models.character_edge_residual import CharacterEdgeResidualRanker
    from hanglish.models.sentence_ranker import SentenceRanker
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.manual_seed(20260923); torch.use_deterministic_algorithms(True)
    sentence = SentenceRanker.load(SENTENCE_CHECKPOINT)
    require(sentence.max_length == 96 and sentence.batch_size == 8
            and next(sentence.model.parameters()).device.type == "cpu",
            "frozen sentence ranker metadata differs")
    character = CharacterEdgeResidualRanker(byte_dim=16, hidden=32, head_hidden=32)
    character.load_state_dict(torch.load(CHARACTER / "model.pt", map_location="cpu",
                                         weights_only=True))
    character.eval()
    return sentence, character, torch


def _probe_authorized(path: Path, pins: dict[str, str]) -> dict[str, str]:
    report = read_json(path / "report.json")
    closure = read_json(path / "capture.complete.json")
    proof = read_json(path / "independent-verification.json")
    hashes = {name: digest(path / name) for name in
              ("report.json", "capture.complete.json", "independent-verification.json")}
    require(report.get("completed") is True and report.get("mode") == "probe"
            and report.get("rows") == 16
            and report.get("qualification", {}).get("qualified") is True
            and closure.get("capture_sha256") == report.get("capture_sha256")
            and digest(path / "capture.jsonl") == report.get("capture_sha256")
            and report.get("hashes", {}).get("capture.complete.json")
                == digest(path / "capture.complete.json")
            and report.get("pins") == pins and report.get("labels_opened") is False
            and proof.get("verified") is True and proof.get("mode") == "probe"
            and proof.get("rows") == 16
            and proof.get("report_sha256") == hashes["report.json"]
            and proof.get("capture_sha256") == report.get("capture_sha256")
            and proof.get("pins") == pins
            and proof.get("labels_opened") is False
            and proof.get("source_only") is True,
            "qualified probe authorization differs")
    return {str((path / name).resolve()): value for name, value in hashes.items()}


def run(mode: str, output: Path, probe: Path | None = None) -> None:
    output.mkdir(parents=True, exist_ok=False)
    guard = None
    captured = 0
    try:
        pins = authenticate()
        probe_authorization = None
        if mode == "full":
            require(probe is not None and probe.is_dir(), "full mode requires --probe directory")
            probe_authorization = _probe_authorized(probe, pins)
        guard = ResourceGuard(output); guard.observe("authenticated")
        source_index = index_jsonl(SOURCE / "capture.jsonl")
        comparator_index = index_jsonl(COMPARATOR / "capture.jsonl")
        require(len(source_index) == len(comparator_index) == ROWS
                and [(r["id"], r["input"]) for r in source_index]
                    == [(r["id"], r["input"]) for r in comparator_index],
                "DEV source/comparator order differs")
        selected = source_index
        groups = None
        if mode == "probe":
            groups = probe_groups(source_index)
            chosen = {group["probe_id"] for group in groups}
            selected = [row for row in source_index if row["id"] in chosen]
        comparator_by_id = {row["id"]: row for row in comparator_index}
        load_started = time.monotonic()
        sentence, character, torch = _load_models()
        initialization = time.monotonic() - load_started
        calibration = read_json(CALIBRATION)
        times, sizes = {}, {}
        output_bytes = 0
        counts = {"ranked_rows": 0, "no_pool_rows": 0, "old_candidates": 0,
                  "novel_candidates": 0, "scored_novel": 0,
                  "unsupported_novel": 0}
        with (output / "capture.jsonl").open("x", encoding="utf-8") as stream:
            for entry in selected:
                started = time.monotonic()
                row = read_indexed(SOURCE / "capture.jsonl", entry)
                comparator = read_indexed(COMPARATOR / "capture.jsonl",
                                          comparator_by_id[entry["id"]])
                with torch.no_grad():
                    captured_row = capture_row(row, comparator,
                                               source_index.index(entry), sentence,
                                               character, calibration)
                encoded = json.dumps(captured_row, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False) + "\n"
                encoded_bytes = len(encoded.encode("utf-8"))
                require(output_bytes + encoded_bytes < LIMITS["output_bytes"],
                        "capture output reached the hard 2 GiB bound")
                stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
                output_bytes += encoded_bytes
                captured += 1; times[row["id"]] = time.monotonic() - started
                sizes[row["id"]] = encoded_bytes
                scoring = captured_row["scoring"]
                counts["ranked_rows" if scoring["ranked"] else "no_pool_rows"] += 1
                counts["old_candidates"] += len(scoring["original"])
                counts["novel_candidates"] += len(scoring["novel"])
                counts["scored_novel"] += sum(n["status"] == "scored" for n in scoring["novel"])
                counts["unsupported_novel"] += sum(n["status"] == "unsupported" for n in scoring["novel"])
                guard.require_row_time(started)
                guard.observe(f"row-{captured}")
                if mode == "probe" or captured % 16 == 0:
                    print(json.dumps({"captured": captured, "rows": len(selected)}), flush=True)
        capture_sha = digest(output / "capture.jsonl")
        projected_seconds = (1.3 * (initialization + sum(
            times[group["probe_id"]] * group["count"] for group in groups)) + 60.0
            if mode == "probe" else None)
        projected_bytes = (math.ceil(1.3 * sum(
            sizes[group["probe_id"]] * group["count"] for group in groups))
            if mode == "probe" else None)
        qualified = (mode == "full" or
                     projected_seconds <= LIMITS["whole_seconds"]
                     and projected_bytes < LIMITS["output_bytes"])
        require(qualified, "probe resource projection failed")
        for path, expected in pins.items():
            require(digest(Path(path)) == expected, "frozen input changed during capture")
        guard.observe("inputs-reverified")
        guard.observe("closing")
        resource_sha = digest(output / "resources.jsonl")
        closure = {"schema": SCHEMA + "-closure-v1", "completed": True,
                   "mode": mode, "dataset_split": "dev", "rows": captured,
                   "capture_sha256": capture_sha, "labels_opened": False,
                   "source_only": True, "capture_output_bytes": output_bytes}
        write_json(output / "capture.complete.json", closure)
        probes = ([{"id": group["probe_id"], "group_count": group["count"],
                    "maximum_source_length": group["maximum_source_length"],
                    "seconds": times[group["probe_id"]],
                    "serialized_bytes": sizes[group["probe_id"]]}
                   for group in groups] if mode == "probe" else None)
        report = {"schema": SCHEMA, "completed": True, "mode": mode,
                  "dataset_split": "dev", "rows": captured,
                  "source_rows": ROWS, "source_ids": [r["id"] for r in source_index],
                  "captured_ids": [r["id"] for r in selected],
                  "labels_opened": False, "source_only": True,
                  "legacy_namespace_note": "current-phonetic-fit IDs/schemas are helper implementation namespaces; all inputs are frozen DEV rows",
                  "capture_sha256": capture_sha, "pins": pins,
                  "capture_output_bytes": output_bytes,
                  "probe_authorization": probe_authorization,
                  "qualification": {"qualified": qualified,
                    "projected_seconds": projected_seconds,
                    "projected_output_bytes": projected_bytes,
                    "initialization_seconds": initialization,
                    "groups": groups, "probes": probes,
                    "formula": "1.3*(load_initialization+sum(probe_time*group_count))+60"},
                  "counts": counts,
                  "limits": LIMITS,
                  "resources": {"seconds": time.monotonic() - guard.started,
                                "max_rss_bytes": guard.peak_rss,
                                "max_swap_growth_bytes": guard.peak_swap_growth},
                  "hashes": {"capture.jsonl": capture_sha,
                             "capture.complete.json": digest(output / "capture.complete.json"),
                             "resources.jsonl": resource_sha}}
        write_json(output / "report.json", report)
    except Exception as error:
        if guard is not None:
            try: guard.observe("failed")
            except Exception: pass
        write_json(output / "failure.json", {"schema": SCHEMA, "completed": False,
                   "mode": mode, "captured_rows": captured, "labels_opened": False,
                   "error_type": type(error).__name__, "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("probe", "full"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--probe", type=Path)
    args = parser.parse_args()
    run(args.mode, args.output, args.probe)


if __name__ == "__main__":
    main()
