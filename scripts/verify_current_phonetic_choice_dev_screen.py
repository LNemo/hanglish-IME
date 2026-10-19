#!/usr/bin/env python3
"""Independently verify the cached current-choice DEV screen.

The verifier never imports the screen producer.  It authenticates its closed
source-only inputs, recomputes all three full-pool score branches with NumPy,
then opens the pinned DEV labels only after prediction closure is established.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import re

import numpy as np

from scripts.verify_production_phonetic_choice_dev import (
    verify_choice_row, verify_score_records,
)

ROOT = Path(__file__).resolve().parents[1]
PRODUCER = ROOT / "scripts/evaluate_current_phonetic_choice_dev.py"
WRAPPER = ROOT / "scripts/evaluate_current_phonetic_choice_dev_v2.py"
REFERENCE_PROOF = ROOT / ".omx/artifacts/phonetic-score-scale-dev-screen-v1/independent-verification.json"
TRAIN = ROOT / ".omx/artifacts/current-phonetic-choice-training-v1"
SOURCE = ROOT / ".omx/artifacts/phonetic-choice-runtime-dev-v1"
COMPARATOR = ROOT / ".omx/artifacts/score-scale-runtime-dev-v1"
DEV = ROOT / "data/challenge/sentence-v3/dev.jsonl"
DEPLOYED_MODEL = ROOT / ".omx/artifacts/phonetic-score-scale-training-v1/model.json"
INTAKE = ROOT / ".omx/artifacts/current-phonetic-choice-dev-intake-v1.json"
ROWS = 316
BRANCHES = ("current_production", "base_full", "deployed_full", "proposed")
TRANSITIONS = {
    "candidate_expansion": ("current_production", "deployed_full"),
    "coefficient_refit": ("deployed_full", "proposed"),
    "overall": ("current_production", "proposed"),
}
LIMITS = {"whole_seconds": 120.0, "update_seconds": 60.0,
          "rss_bytes": 1024**3, "swap_growth_bytes": 512 * 1024**2}
OPERATIONS = ["authenticate", "project_all", "close_predictions",
              "late_label_join", "write_report", "write_completion"]
MODEL_SHA256 = "3c71014f423b3487990b5d1d318dfcd1ee7b3a69b2900ad76e1b9adf81f83abf"
PRODUCER_SHA256 = "6141379bc626b6c57c481243e7fd167f40f04658ac2c019dc9cd6dbbbd9cfb07"
WRAPPER_SHA256 = "92acc04d8070ec32b190d93d16564a463774e463fa1f6f01c0c530ec906b8bdc"
REFERENCE_PROOF_SHA256 = "e1b23b2fc23fbacba83c56358207eaa810c4b80addab59481c3b04a47a8ef8f4"
FROZEN = {
    TRAIN / "report.json": "60cb1994bd28cd7786e82d07680cead2bc8f1f6610d37d38e4db25523c12b422",
    TRAIN / "independent-verification.json": "9724e2bf56a41e1bd171b974afb08bed5fb69f4862b92b51508cf7dc17ba5575",
    TRAIN / "model.json": MODEL_SHA256,
    TRAIN / "completion.json": "e2be1914f06f3da0c337f7e27fa8ed518155215beb01a178cbfbcf7c103199fa",
    SOURCE / "report.json": "a0072c79116da9acdbff83b373dccbb8c3ef8c26ac5164f575490941808d4473",
    SOURCE / "independent-verification.json": "f2e2eca118ef05c1397304d89ccea99e1d9ef28841fe109fcd04446d0928e6c4",
    SOURCE / "capture.jsonl": "e0447efa305e1d4f61052866b1ee121a62b400f3adf163dfdeac1b0bb4734bf5",
    SOURCE / "capture.complete.json": "e2f88f3383baf9c8f8cfebf643e02ecdf3e4652b4fc0d3b495004ebebdb6c023",
    SOURCE / "completion.json": "f62ac3ee9b2b8afed211d0c2ccf418af841ce3c041f5eb38f40d578462759bae",
    COMPARATOR / "report.json": "39b76bd5f59fbbe90c8ebbd65dfd15f9973b327746230eb997a215fd65aae959",
    COMPARATOR / "independent-verification.json": "60b8c4c16e00d36274e2f71760888ea5c12d519669ab4ad38cad5745dd0fd9c0",
    COMPARATOR / "capture.jsonl": "9a47e0fb3174f0923f4eb39e0a7342316f0c3142dfee892b7d4f031afe2325e7",
    COMPARATOR / "closure.json": "54ded14328a0d89adad6a5ae2222ad9c36b7881759200387934c6b3c89562bbe",
    COMPARATOR / "completion.json": "ee21fd5091160b3770496bdf5d471da17bc13c3ef6da12e4b94e7b36cc3e9618",
    INTAKE: "c7e656c117e8f46f2968cf401abc66408581ac6b328d85c1dbe0ce02223b451f",
}
CURRENT_VERSION = "experimental-score-scale-1-f6976e1b269d"
FULL_VERSION = "experimental-phonetic-choice-1-b11ca7b6d718"
FULL_PARENT_VERSION = "experimental-phonetic-word-union-1-84073edd0b9d"
ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]\Z")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    need(isinstance(value, dict), "expected JSON object: " + str(path))
    return value


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSONL at {path}:{number}") from error
            need(isinstance(value, dict), "JSONL row is not an object")
            yield value


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_model(model):
    parameters, scale = model.get("parameters"), model.get("scale")
    need(model.get("schema") == "phonetic-score-scale-model-v1"
         and model.get("arm") == "learned_scaled"
         and isinstance(parameters, list) and len(parameters) == 5
         and isinstance(scale, list) and len(scale) == 4
         and all(finite(x) for x in parameters + scale)
         and parameters[0] >= 0 and all(x > 0 for x in scale),
         "model contract differs")
    return np.asarray(parameters, dtype=np.float64), np.asarray(scale, dtype=np.float64)


def score_matrix(base_scores, features, models):
    """Return independent NumPy projections for named fixed coefficient models."""
    base = np.asarray(base_scores, dtype=np.float64)
    feature = np.asarray(features, dtype=np.float64)
    need(base.ndim == 1 and feature.shape == (base.size, 4)
         and np.isfinite(base).all() and np.isfinite(feature).all(),
         "score matrix shape or values differ")
    result = {"base_full": base.copy()}
    for name, model in models.items():
        parameters, scale = validate_model(model)
        design = np.column_stack((base, feature / scale))
        values = design @ parameters
        need(np.isfinite(values).all(), "projected scores are nonfinite")
        result[name] = values
    return result


def validate_segments(source, text, segments):
    need(isinstance(segments, list) and segments, "candidate segments missing")
    cursor, pieces = 0, []
    for segment in segments:
        start, end = segment.get("input_start"), segment.get("input_end")
        need(type(start) is int and type(end) is int and start == cursor
             and start < end <= len(source)
             and segment.get("source") == source[start:end]
             and isinstance(segment.get("text"), str)
             and isinstance(segment.get("kind"), str),
             "candidate source offsets differ")
        cursor = end
        pieces.append(segment["text"])
    need(cursor == len(source) and "".join(pieces) == text,
         "candidate segment reconstruction differs")


def compact_response(response):
    need(isinstance(response, dict)
         and set(("result", "candidates", "sentence_ranking_applied")) <= set(response),
         "response contract differs")
    candidates = response["candidates"]
    need(isinstance(candidates, list) and candidates
         and type(response["sentence_ranking_applied"]) is bool
         and response["result"] == candidates[0].get("text"),
         "response candidates differ")
    return {key: response[key] for key in
            ("result", "candidates", "sentence_ranking_applied")}


def raw_candidate(source):
    return {"text": source, "score": None, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": source, "kind": "raw"}]}


def render_public(source, paths):
    need(paths, "cannot render an empty ranked pool")
    ordered = sorted(paths, key=lambda row: (-row[1], row[0]))
    candidates = [{"text": text, "score": round(float(score), 4), "segments": segments}
                  for text, score, segments in ordered[:5]]
    if source not in {row["text"] for row in candidates}:
        candidates = candidates[:4] + [raw_candidate(source)]
    return candidates, [row[0] for row in ordered[:5]]


def _equal_score(actual, expected):
    return finite(actual) and math.isclose(actual, float(expected), rel_tol=0.0, abs_tol=1e-12)


def replay_row(source_row, comparator_row, prediction, deployed_model, proposed_model):
    """Authenticate and independently replay one source-only prediction row."""
    need(source_row.get("id") == comparator_row.get("id") == prediction.get("id")
         and source_row.get("request") == comparator_row.get("request") == prediction.get("request"),
         "source/comparator/prediction alignment differs")
    request = source_row["request"]
    source = request.get("input")
    need(request == {"input": source, "context": ""} and isinstance(source, str),
         "source-only request differs")
    need(prediction.get("kind") == "cached-full-union-fixed-coefficient-projection"
         and set(prediction.get("branches", {})) == set(BRANCHES),
         "prediction branch contract differs")
    current_response = comparator_row.get("response", {})
    need(current_response.get("version") == CURRENT_VERSION
         and current_response.get("input") == source
         and current_response.get("context") == ""
         and current_response.get("external_context_used") is False,
         "current capture identity differs")
    expected_current = {**compact_response(current_response), "latent_top5": None}
    need(prediction["branches"]["current_production"] == expected_current,
         "actual current production branch differs")
    for candidate in expected_current["candidates"]:
        validate_segments(source, candidate["text"], candidate["segments"])

    full = source_row.get("proposed", {})
    choice = full.get("phonetic_choice", {})
    need(full.get("version") == FULL_VERSION and full.get("input") == source
         and full.get("context") == "" and full.get("external_context_used") is False
         and choice.get("parent_version") == FULL_PARENT_VERSION,
         "full-union source identity differs")
    paths, feature_rows = source_row.get("base_paths"), choice.get("rows")
    need(isinstance(paths, list) and isinstance(feature_rows, list), "full pool missing")
    ranked = full.get("sentence_ranking_applied")
    need(type(ranked) is bool, "ranking flag differs")
    if not ranked:
        need(paths == [] and feature_rows == [] and prediction.get("score_rows") == [],
             "unranked pool differs")
        fallback = compact_response(full)
        need(fallback == compact_response(source_row.get("union_reference")),
             "unranked union fallback differs")
        for name in BRANCHES[1:]:
            need(prediction["branches"][name] == {**fallback, "latent_top5": []},
                 "unranked projected branch differs")
        return 0

    need(paths and len(paths) == len(feature_rows)
         and len({row[0] for row in paths}) == len(paths), "ranked pool differs")
    score_records = verify_score_records(choice.get("score_records"))
    vectors = [verify_choice_row(source, path, row, score_records)
               for path, row in zip(paths, feature_rows)]
    scores = score_matrix([row[1] for row in paths], vectors,
                          {"deployed_full": deployed_model, "proposed": proposed_model})
    recorded = prediction.get("score_rows")
    need(isinstance(recorded, list) and len(recorded) == len(paths),
         "recorded score row coverage differs")
    current_public = {row["text"] for row in expected_current["candidates"]}
    pools = {name: [] for name in BRANCHES[1:]}
    for index, (path, feature, frozen) in enumerate(zip(paths, vectors, recorded)):
        text, base, segments = path
        validate_segments(source, text, segments)
        need(frozen.get("text") == text and frozen.get("base_score") == base
             and frozen.get("features") == feature and frozen.get("segments") == segments
             and frozen.get("in_current_public") is (text in current_public)
             and set(frozen.get("scores", {})) == set(BRANCHES[1:]),
             "recorded feature row differs")
        for name in BRANCHES[1:]:
            value = scores[name][index]
            need(_equal_score(frozen["scores"][name], value), "recorded score differs")
            pools[name].append((text, float(value), segments))
    for name, pool in pools.items():
        public, latent = render_public(source, pool)
        expected = {"result": public[0]["text"], "candidates": public,
                    "sentence_ranking_applied": True, "latent_top5": latent}
        need(prediction["branches"][name] == expected,
             "independent branch replay differs: " + name)
    need(compact_response(source_row.get("union_reference")) ==
         compact_response(prediction["branches"]["base_full"]),
         "base full-union parity differs")
    return len(paths)


def ascii_words(text):
    return tuple(ASCII_WORD.findall(text))


def literal_signature(text):
    return "".join(char for char in text
                   if not ("a" <= char <= "z") and HANGUL.fullmatch(char) is None)


def evaluate_rows(predictions, labels, expected_rows=ROWS):
    empty = {"rows": 0, "top1": 0, "top5": 0, "ascii_reference_rows": 0,
             "ascii_exact_first": 0, "literal_exact_first": 0, "ranked_rows": 0,
             "scored_pool_coverage": 0, "latent_top5": 0}
    metrics = {name: dict(empty) for name in BRANCHES}
    metrics["current_production"]["scored_pool_coverage"] = None
    metrics["current_production"]["latent_top5"] = None
    transition_empty = {"corrections": 0, "regressions": 0, "top5_gains": 0,
                        "top5_losses": 0, "ascii_signature_regressions": 0,
                        "literal_signature_regressions": 0}
    transitions = {name: dict(transition_empty) for name in TRANSITIONS}
    count, seen = 0, set()
    for prediction, label in zip_longest(predictions, labels):
        need(prediction is not None and label is not None, "prediction/label denominator differs")
        identifier = prediction.get("id")
        need(identifier not in seen and identifier == label.get("id")
             and prediction.get("request") == {"input": label.get("input"), "context": ""}
             and label.get("context", "") == "" and label.get("split") == "dev"
             and isinstance(label.get("expected"), list) and label["expected"],
             "late-label identity differs")
        seen.add(identifier); count += 1
        gold = set(label["expected"])
        signatures = {ascii_words(text) for text in gold}
        has_ascii = any(signatures)
        source_literal = literal_signature(label["input"])
        score_rows = prediction.get("score_rows")
        need(isinstance(score_rows, list), "score rows differ")
        pool = {row.get("text") for row in score_rows}
        outcomes = {}
        for name in BRANCHES:
            branch = prediction.get("branches", {}).get(name)
            compact = compact_response(branch)
            texts = [row["text"] for row in compact["candidates"]]
            first, top5 = texts[0] in gold, bool(gold.intersection(texts))
            outcomes[name] = (first, top5, ascii_words(texts[0]), literal_signature(texts[0]))
            metric = metrics[name]
            metric["rows"] += 1; metric["top1"] += first; metric["top5"] += top5
            metric["ranked_rows"] += compact["sentence_ranking_applied"]
            metric["ascii_reference_rows"] += has_ascii
            metric["ascii_exact_first"] += has_ascii and outcomes[name][2] in signatures
            metric["literal_exact_first"] += outcomes[name][3] == source_literal
            if name != "current_production":
                latent = branch.get("latent_top5")
                need(isinstance(latent, list) and len(latent) <= 5
                     and len(set(latent)) == len(latent), "latent top-five differs")
                metric["scored_pool_coverage"] += bool(gold.intersection(pool))
                metric["latent_top5"] += bool(gold.intersection(latent))
        for name, (before, after) in TRANSITIONS.items():
            old, new = outcomes[before], outcomes[after]
            item = transitions[name]
            item["corrections"] += not old[0] and new[0]
            item["regressions"] += old[0] and not new[0]
            item["top5_gains"] += not old[1] and new[1]
            item["top5_losses"] += old[1] and not new[1]
            item["ascii_signature_regressions"] += has_ascii and old[2] in signatures and new[2] not in signatures
            item["literal_signature_regressions"] += old[3] == source_literal and new[3] != source_literal
    need(count == expected_rows and len(seen) == expected_rows, "DEV denominator differs")
    overall = transitions["overall"]
    gate = bool(metrics["proposed"]["top1"] > metrics["current_production"]["top1"]
                and metrics["proposed"]["top5"] >= metrics["current_production"]["top5"]
                and not any(overall[key] for key in ("regressions", "top5_losses",
                            "ascii_signature_regressions", "literal_signature_regressions")))
    return {"branches": metrics, "transitions": transitions,
            "quality_gate": gate, "actual_runtime_authorized": gate}


def verify_resources(rows, operations=OPERATIONS, limits=LIMITS):
    need(rows, "resource trace empty")
    starts, ends, current = [], [], None
    after_report = after_completion = None
    baseline, previous = None, -1.0
    for row in rows:
        need(finite(row.get("elapsed_s")) and previous <= row["elapsed_s"] <= limits["whole_seconds"]
             and type(row.get("rss_bytes")) is int and 0 <= row["rss_bytes"] <= limits["rss_bytes"]
             and type(row.get("swap_used_bytes")) is int and row["swap_used_bytes"] >= 0
             and type(row.get("swap_growth_bytes")) is int
             and row["swap_growth_bytes"] <= limits["swap_growth_bytes"], "resource row differs")
        datetime.fromisoformat(row["at"])
        previous = row["elapsed_s"]
        operation, elapsed = row.get("operation"), row.get("operation_elapsed_s")
        need((operation is None) is (elapsed is None), "resource operation attribution differs")
        if elapsed is not None:
            need(finite(elapsed) and 0 <= elapsed <= limits["update_seconds"], "operation limit differs")
        observed = row["swap_used_bytes"] - row["swap_growth_bytes"]
        need(observed >= 0, "swap baseline invalid")
        baseline = observed if baseline is None else baseline
        need(observed == baseline, "swap baseline changed")
        stage = row.get("stage")
        if stage == "operation_start":
            need(current is None and isinstance(operation, str), "operation start differs")
            current = operation; starts.append(operation)
        elif stage == "operation_end":
            need(current == operation, "operation end differs")
            current = None; ends.append(operation)
        elif stage == "watchdog":
            need(operation == current, "watchdog attribution differs")
        elif stage == "after_report":
            need(current is None and after_report is None and starts == operations[:-1], "after report differs")
            after_report = row
        elif stage == "after_completion":
            need(current is None and after_report is not None and after_completion is None
                 and starts == operations, "after completion differs")
            after_completion = row
        else:
            raise ValueError("unexpected resource stage: " + str(stage))
    need(starts == operations == ends and after_report and after_completion,
         "resource operation sequence differs")
    return {"rows": len(rows), "operations": len(starts), "final": after_report,
            "after_completion": after_completion, "baseline_swap_bytes": baseline,
            "peak_rss_bytes": max(row["rss_bytes"] for row in rows),
            "peak_swap_growth_bytes": max(row["swap_growth_bytes"] for row in rows)}


def _verify_upstream(directory, report_hash, capture_hash, closure_name, schema):
    report, proof = read_json(directory / "report.json"), read_json(directory / "independent-verification.json")
    closure, completion = read_json(directory / closure_name), read_json(directory / "completion.json")
    need(report.get("schema") == schema and report.get("completed") is True
         and proof.get("verified") is True and proof.get("report_sha256") == report_hash
         and proof.get("capture_sha256") == capture_hash
         and closure.get("rows") == ROWS and closure.get("labels_opened") is False
         and closure.get("capture_sha256") == capture_hash
         and completion.get("completed") is True and completion.get("report_sha256") == report_hash,
         "upstream terminal proof differs")
    return report


def verify(output, producer_sha256):
    output = Path(output).resolve()
    need(re.fullmatch(r"[0-9a-f]{64}", producer_sha256) is not None,
         "producer SHA-256 is invalid")
    proof_path = output / "independent-verification.json"
    need(output.is_dir() and not proof_path.exists(), "run/proof state invalid")
    need(not any((output / name).exists() for name in
                 ("failure.json", "failure-report.json", "resource-watchdog-failure.json")),
         "failure artifact exists")
    need(producer_sha256 == PRODUCER_SHA256
         and sha256_file(PRODUCER) == producer_sha256, "producer identity differs")
    need(sha256_file(WRAPPER) == WRAPPER_SHA256
         and sha256_file(REFERENCE_PROOF) == REFERENCE_PROOF_SHA256,
         "v2 reference correction identity differs")
    reference = read_json(REFERENCE_PROOF)
    reference_ascii = reference.get("comparisons", {}).get("parent", {}).get(
        "english_ascii_signature", {})
    need(reference.get("verified") is True
         and reference_ascii.get("rows_with_reference_ascii_words") == 59
         and reference_ascii.get("proposed_exact_signature") == 55,
         "deployed ASCII reference proof differs")
    launch_path = output.with_suffix(".launch.json")
    launch = read_json(launch_path)
    try:
        os.kill(launch.get("pid"), 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("producer PID is still present")
    need(launch.get("command") == [str(ROOT / ".venv/bin/python"), "-m",
                                   "scripts.evaluate_current_phonetic_choice_dev_v2",
                                   "--output", str(output)]
         and launch.get("output") == str(output)
         and launch.get("runner_sha256") == WRAPPER_SHA256,
         "v2 launch identity differs")
    for path, digest in FROZEN.items():
        need(path.is_file() and sha256_file(path) == digest, "frozen input differs: " + str(path))
    source_report = _verify_upstream(SOURCE, FROZEN[SOURCE / "report.json"],
                                     FROZEN[SOURCE / "capture.jsonl"], "capture.complete.json",
                                     "phonetic-choice-runtime-dev-v1")
    comparator_report = _verify_upstream(COMPARATOR, FROZEN[COMPARATOR / "report.json"],
                                         FROZEN[COMPARATOR / "capture.jsonl"], "closure.json",
                                         "score-scale-runtime-dev-v1")
    need(source_report.get("version") == FULL_VERSION
         and comparator_report.get("version") == CURRENT_VERSION,
         "upstream version differs")
    report, completion, closure, pins = (read_json(output / name) for name in
                                         ("report.json", "completion.json", "closure.json", "pins.json"))
    predictions_sha = sha256_file(output / "predictions.jsonl")
    need(report.get("schema") == "current-phonetic-choice-dev-screen-v1"
         and report.get("completed") is True and report.get("rows") == ROWS
         and report.get("branches") == list(BRANCHES)
         and report.get("model_sha256") == MODEL_SHA256
         and report.get("predictions_sha256") == predictions_sha
         and report.get("closure_sha256") == sha256_file(output / "closure.json")
         and report.get("pins") == pins and report.get("limits") == LIMITS
         and report.get("model_inference_performed") is False
         and report.get("training_performed") is False
         and report.get("test_accessed") is False
         and report.get("production_changed") is False
         and report.get("labels_opened_after_predictions") is True,
         "terminal report differs")
    need(closure == {"schema": "current-phonetic-choice-dev-prediction-closure-v1",
                     "closed": True, "rows": ROWS, "predictions_sha256": predictions_sha,
                     "labels_opened": False, "model_inference_performed": False},
         "prediction closure differs")
    need(completion.get("completed") is True
         and completion.get("report_sha256") == sha256_file(output / "report.json"),
         "completion differs")
    producer_key = str(PRODUCER.resolve())
    need(pins.get(producer_key) == producer_sha256
         and pins.get(str(WRAPPER.resolve())) == WRAPPER_SHA256
         and pins.get(str(REFERENCE_PROOF.resolve())) == REFERENCE_PROOF_SHA256,
         "producer/reference pin differs")
    need(all(pins.get(str(path.resolve())) == digest for path, digest in FROZEN.items()),
         "frozen input is absent from report pins")
    for name, digest in pins.items():
        path = Path(name)
        if path.resolve() != DEV.resolve():
            need(path.is_file() and sha256_file(path) == digest, "runtime pin differs: " + name)

    train_report = read_json(TRAIN / "report.json")
    train_proof = read_json(TRAIN / "independent-verification.json")
    proposed, deployed = read_json(TRAIN / "model.json"), read_json(DEPLOYED_MODEL)
    need(train_report.get("schema") == "current-phonetic-choice-training-v1"
         and train_report.get("completed") is True
         and train_report.get("qualification_passed") is True
         and train_report.get("full_fit_performed") is True
         and train_report.get("model") == proposed
         and train_proof.get("verified") is True
         and train_proof.get("qualification_passed") is True
         and train_proof.get("report_sha256") == FROZEN[TRAIN / "report.json"],
         "qualified TRAIN model identity differs")
    intake = read_json(INTAKE)
    artifacts = intake.get("shared_current_model_artifacts")
    need(intake.get("schema") == "current-phonetic-choice-dev-intake-v1"
         and intake.get("matched_count") == 32
         and isinstance(artifacts, list) and len(artifacts) == 32
         and len({row.get("path") for row in artifacts}) == 32
         and intake.get("dev_labels_opened") is False
         and intake.get("model_inference_run") is False,
         "DEV intake contract differs")
    validate_model(proposed); validate_model(deployed)
    predictions = list(jsonl(output / "predictions.jsonl"))
    paths = count = 0
    for source, current, prediction in zip_longest(
            jsonl(SOURCE / "capture.jsonl"), jsonl(COMPARATOR / "capture.jsonl"), predictions):
        need(source is not None and current is not None and prediction is not None,
             "source prediction denominator differs")
        paths += replay_row(source, current, prediction, deployed, proposed)
        count += 1
    need(count == ROWS and len({row["id"] for row in predictions}) == ROWS,
         "prediction coverage differs")
    need(sha256_file(output / "predictions.jsonl") == predictions_sha,
         "closed predictions changed before labels")
    need(pins.get(str(DEV.resolve())) == sha256_file(DEV), "DEV label pin differs")
    metrics = evaluate_rows(predictions, jsonl(DEV))
    need(report.get("metrics") == metrics
         and report.get("quality_gate") is metrics["quality_gate"]
         and report.get("actual_runtime_authorized") is metrics["actual_runtime_authorized"],
         "recorded metrics differ")
    resources = verify_resources(list(jsonl(output / "resources.jsonl")))
    need(completion.get("resource_after_report") == resources["final"],
         "completion resource binding differs")
    proof = {"schema": "current-phonetic-choice-dev-screen-independent-verification-v1",
             "verified": True, "report_sha256": sha256_file(output / "report.json"),
             "verifier_sha256": sha256_file(__file__), "producer_sha256": producer_sha256,
             "predictions_sha256": predictions_sha,
             "source_proof_sha256": FROZEN[SOURCE / "independent-verification.json"],
             "comparator_proof_sha256": FROZEN[COMPARATOR / "independent-verification.json"],
             "model_sha256": MODEL_SHA256, "counts": {"rows": count, "paths": paths},
             "launch_sha256": sha256_file(launch_path),
             "metrics": metrics, "quality_gate": metrics["quality_gate"], "resources": resources,
             "labels_opened_after_prediction_verification": True,
             "model_inference_performed": False,
             "scope": "All 316 cached source-only rows, actual current responses, full-union score matrices, public/raw slots, coverage/latent metrics, regressions, pins, closure, and resource trace.",
             "limitations": "Observed DEV only. This verifies a cached screen and does not authorize web promotion without the planned actual runtime gate."}
    with proof_path.open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--producer-sha256", required=True)
    result = verify(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("verified", "counts", "quality_gate")},
                     ensure_ascii=False))
