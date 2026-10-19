#!/usr/bin/env python3
"""Screen the frozen current choice model on the cached full-union DEV pool.

This program performs no model inference and fits no parameters.  It authenticates
two independently verified runtime captures, projects three fixed formulas over
the full-union candidate rows, closes those predictions, and only then opens DEV
labels.  The current-production branch is copied from the actual deployed runtime
capture; it is never reconstructed from the full-union pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / ".omx/artifacts/current-phonetic-choice-training-v1"
SOURCE = ROOT / ".omx/artifacts/phonetic-choice-runtime-dev-v1"
COMPARATOR = ROOT / ".omx/artifacts/score-scale-runtime-dev-v1"
INTAKE = ROOT / ".omx/artifacts/current-phonetic-choice-dev-intake-v1.json"
DEV = ROOT / "data/challenge/sentence-v3/dev.jsonl"
PLAN = ROOT / ".omx/plans/current-phonetic-choice-dev-v1.md"
DEPLOYED_MODEL = ROOT / ".omx/artifacts/phonetic-score-scale-training-v1/model.json"
RESOURCE_HELPER = ROOT / "scripts/source_span_resources.py"
NUMERICAL_CORE = ROOT / "scripts/phonetic_score_scale.py"

ROWS = 316
BRANCHES = ("current_production", "base_full", "deployed_full", "proposed")
TRANSITIONS = {
    "candidate_expansion": ("current_production", "deployed_full"),
    "coefficient_refit": ("deployed_full", "proposed"),
    "overall": ("current_production", "proposed"),
}
LIMITS = {"whole_seconds": 120.0, "update_seconds": 60.0,
          "rss_bytes": 1024**3, "swap_growth_bytes": 512 * 1024**2}

# Each digest names the exact already-verified evidence accepted by this screen.
FROZEN = {
    TRAIN / "report.json": "60cb1994bd28cd7786e82d07680cead2bc8f1f6610d37d38e4db25523c12b422",
    TRAIN / "independent-verification.json": "9724e2bf56a41e1bd171b974afb08bed5fb69f4862b92b51508cf7dc17ba5575",
    TRAIN / "model.json": "3c71014f423b3487990b5d1d318dfcd1ee7b3a69b2900ad76e1b9adf81f83abf",
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
    PLAN: "174e91dc75b2fa4a563c25fc7d5c234e0959d61095996723c18ec0c9e23f2d02",
}
CURRENT_VERSION = "experimental-score-scale-1-f6976e1b269d"
FULL_VERSION = "experimental-phonetic-choice-1-b11ca7b6d718"
FULL_PARENT_VERSION = "experimental-phonetic-word-union-1-84073edd0b9d"
CURRENT_REFERENCE = {"rows": ROWS, "top1": 267, "top5": 284,
                     "ascii_reference_rows": 59, "ascii_exact_first": 54}

# Local mappings deliberately mirror the frozen pure contracts without importing
# a producer: render_public -> evaluate_phonetic_score_scale_dev.render;
# score formula/model validation -> train_current_phonetic_choice.score_outcome.
HELPER_MAPPINGS = {
    "public_render": "scripts.evaluate_phonetic_score_scale_dev.render",
    "score_formula": "scripts.train_current_phonetic_choice.score_outcome",
}


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


def iter_jsonl(path):
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


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_jsonl(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                            allow_nan=False) + "\n")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_model(model):
    need(model.get("schema") == "phonetic-score-scale-model-v1"
         and model.get("arm") == "learned_scaled", "model contract differs")
    parameters, scale = model.get("parameters"), model.get("scale")
    need(isinstance(parameters, list) and isinstance(scale, list)
         and len(parameters) == 5 and len(scale) == 4
         and all(finite(value) for value in parameters + scale)
         and parameters[0] >= 0 and all(value > 0 for value in scale),
         "model coefficients differ")
    return parameters, scale


def raw_candidate(source):
    return {"text": source, "score": None,
            "segments": [{"input_start": 0, "input_end": len(source),
                          "source": source, "text": source, "kind": "raw"}]}


def render_public(source, scored_paths):
    """Render the runtime's rounded top-five and reserved raw-input slot."""
    ordered = sorted(scored_paths, key=lambda row: (-row[1], row[0]))
    need(ordered and len({row[0] for row in ordered}) == len(ordered),
         "empty or duplicate scored pool")
    result = [{"text": text, "score": round(score, 4), "segments": segments}
              for text, score, segments in ordered[:5]]
    if source not in [row["text"] for row in result]:
        result = result[:4] + [raw_candidate(source)]
    return result


def latent_top5(scored_paths):
    return [row[0] for row in sorted(scored_paths,
                                      key=lambda row: (-row[1], row[0]))[:5]]


def validate_segments(source, text, segments):
    need(isinstance(segments, list) and segments, "candidate segments missing")
    cursor, rendered = 0, []
    for segment in segments:
        start, end = segment.get("input_start"), segment.get("input_end")
        need(type(start) is int and type(end) is int
             and start == cursor and start < end <= len(source)
             and segment.get("source") == source[start:end]
             and isinstance(segment.get("text"), str)
             and isinstance(segment.get("kind"), str),
             "candidate source offsets differ")
        cursor = end; rendered.append(segment["text"])
    need(cursor == len(source) and "".join(rendered) == text,
         "candidate segment reconstruction differs")


def compact_response(response):
    required = ("result", "candidates", "sentence_ranking_applied")
    need(isinstance(response, dict) and all(key in response for key in required),
         "runtime response contract differs")
    candidates = response["candidates"]
    need(isinstance(candidates, list) and candidates
         and all(isinstance(row, dict) and isinstance(row.get("text"), str)
                 for row in candidates)
         and response["result"] == candidates[0]["text"]
         and type(response["sentence_ranking_applied"]) is bool,
         "runtime public candidates differ")
    return {key: response[key] for key in required}


def score_rows(source_row, deployed_model, proposed_model):
    """Authenticate source feature alignment and project the three full branches."""
    request = source_row.get("request")
    source = request.get("input") if isinstance(request, dict) else None
    need(request == {"input": source, "context": ""} and isinstance(source, str),
         "source-only request differs")
    response = source_row.get("proposed")
    need(isinstance(response, dict) and response.get("version") == FULL_VERSION
         and response.get("input") == source and response.get("context") == ""
         and response.get("external_context_used") is False,
         "full-union response identity differs")
    choice = response.get("phonetic_choice")
    need(isinstance(choice, dict)
         and choice.get("parent_version") == FULL_PARENT_VERSION,
         "full-union feature producer differs")
    paths, features = source_row.get("base_paths"), choice.get("rows")
    ranked = response.get("sentence_ranking_applied")
    need(type(ranked) is bool and isinstance(paths, list)
         and isinstance(features, list), "ranking contract differs")
    if not ranked:
        need(not paths and not features, "unranked row contains scored paths")
        fallback = compact_response(response)
        union = compact_response(source_row.get("union_reference"))
        need(fallback == union, "unranked full-union fallback differs")
        return [], {name: {**fallback, "latent_top5": []}
                    for name in ("base_full", "deployed_full", "proposed")}

    need(paths and len(paths) == len(features)
         and len({path[0] for path in paths}) == len(paths),
         "full-union candidate/feature coverage differs")
    deployed_parameters, deployed_scale = validate_model(deployed_model)
    proposed_parameters, proposed_scale = validate_model(proposed_model)
    projected = []
    branch_paths = {name: [] for name in ("base_full", "deployed_full", "proposed")}
    for path, feature in zip(paths, features):
        need(isinstance(path, list) and len(path) == 3
             and isinstance(path[0], str) and finite(path[1])
             and isinstance(path[2], list) and isinstance(feature, dict)
             and feature.get("text") == path[0]
             and feature.get("base_score") == path[1]
             and feature.get("segments") == path[2],
             "source path/feature alignment differs")
        validate_segments(source, path[0], path[2])
        values = feature.get("features")
        need(isinstance(values, list) and len(values) == 4
             and all(finite(value) for value in values), "feature vector differs")

        def apply(model_parameters, model_scale):
            result = model_parameters[0] * path[1] + sum(
                weight * (value / scale)
                for weight, value, scale in zip(model_parameters[1:], values, model_scale))
            need(math.isfinite(result), "projected score is nonfinite")
            return result

        scores = {"base_full": path[1],
                  "deployed_full": apply(deployed_parameters, deployed_scale),
                  "proposed": apply(proposed_parameters, proposed_scale)}
        projected.append({"text": path[0], "base_score": path[1],
                          "features": values, "segments": path[2],
                          "scores": scores})
        for name, score in scores.items():
            branch_paths[name].append((path[0], score, path[2]))
    branches = {}
    for name, values in branch_paths.items():
        candidates = render_public(source, values)
        branches[name] = {"result": candidates[0]["text"],
                          "candidates": candidates,
                          "sentence_ranking_applied": True,
                          "latent_top5": latent_top5(values)}
    union = compact_response(source_row.get("union_reference"))
    need(branches["base_full"]["result"] == union["result"]
         and branches["base_full"]["candidates"] == union["candidates"]
         and union["sentence_ranking_applied"] is True,
         "unadjusted full-union public replay differs")
    return projected, branches


def project_row(source_row, comparator_row, deployed_model, proposed_model):
    need(source_row.get("id") == comparator_row.get("id")
         and source_row.get("request") == comparator_row.get("request"),
         "source/comparator row alignment differs")
    source = source_row["request"]["input"]
    actual = comparator_row.get("response")
    need(isinstance(actual, dict) and actual.get("version") == CURRENT_VERSION
         and actual.get("input") == source and actual.get("context") == ""
         and actual.get("external_context_used") is False,
         "current production response identity differs")
    current = compact_response(actual)
    projected, branches = score_rows(source_row, deployed_model, proposed_model)
    branches = {"current_production": {**current, "latent_top5": None}, **branches}
    current_texts = {row["text"] for row in current["candidates"]}
    for row in projected:
        row["in_current_public"] = row["text"] in current_texts
    return {"id": source_row["id"], "request": source_row["request"],
            "branches": branches, "score_rows": projected,
            "kind": "cached-full-union-fixed-coefficient-projection"}


def ascii_words(text):
    return tuple(re.findall(r"[A-Za-z][A-Za-z0-9_]*", text))


def literal_signature(text):
    return "".join(character for character in text
                   if not ("a" <= character <= "z"
                           or re.fullmatch(r"[가-힣ᄀ-ᇿ㄰-㆏]", character)))


def empty_branch_metrics():
    return {"rows": 0, "top1": 0, "top5": 0,
            "ascii_reference_rows": 0, "ascii_exact_first": 0,
            "literal_exact_first": 0, "ranked_rows": 0,
            "scored_pool_coverage": 0, "latent_top5": 0}


def empty_transition_metrics():
    return {"corrections": 0, "regressions": 0, "top5_gains": 0,
            "top5_losses": 0, "ascii_signature_regressions": 0,
            "literal_signature_regressions": 0}


def evaluate(prediction_path, label_path):
    metrics = {name: empty_branch_metrics() for name in BRANCHES}
    metrics["current_production"]["scored_pool_coverage"] = None
    metrics["current_production"]["latent_top5"] = None
    transitions = {name: empty_transition_metrics() for name in TRANSITIONS}
    predictions, labels = iter_jsonl(prediction_path), iter_jsonl(label_path)
    count, seen = 0, set()
    while True:
        prediction, label = next(predictions, None), next(labels, None)
        need((prediction is None) == (label is None), "prediction/label denominator differs")
        if prediction is None:
            break
        identifier = prediction.get("id")
        need(identifier not in seen and identifier == label.get("id")
             and prediction.get("request") == {"input": label.get("input"), "context": ""}
             and label.get("context", "") == "" and label.get("split") == "dev"
             and isinstance(label.get("expected"), list) and label["expected"],
             "late-label identity differs")
        seen.add(identifier); count += 1
        gold = set(label["expected"])
        signatures = {ascii_words(text) for text in gold}
        has_ascii_reference = any(signatures)
        source_literal = literal_signature(prediction["request"]["input"])
        outcomes = {}
        branches = prediction.get("branches")
        score_rows = prediction.get("score_rows")
        need(isinstance(branches, dict) and set(branches) == set(BRANCHES),
             "prediction branches differ")
        need(isinstance(score_rows, list), "prediction score rows differ")
        scored_texts = {row.get("text") for row in score_rows}
        for name in BRANCHES:
            branch = compact_response(branches[name])
            texts = [row["text"] for row in branch["candidates"]]
            first, top5 = texts[0] in gold, bool(gold.intersection(texts))
            outcomes[name] = (first, top5, ascii_words(texts[0]),
                              literal_signature(texts[0]))
            summary = metrics[name]
            summary["rows"] += 1; summary["top1"] += first; summary["top5"] += top5
            summary["ranked_rows"] += branch["sentence_ranking_applied"]
            summary["ascii_reference_rows"] += has_ascii_reference
            summary["ascii_exact_first"] += has_ascii_reference and outcomes[name][2] in signatures
            summary["literal_exact_first"] += outcomes[name][3] == source_literal
            if name != "current_production":
                latent = branches[name].get("latent_top5")
                need(isinstance(latent, list) and len(latent) <= 5
                     and all(isinstance(text, str) for text in latent),
                     "latent top-five differs")
                summary["latent_top5"] += bool(gold.intersection(latent))
                summary["scored_pool_coverage"] += bool(gold.intersection(scored_texts))
        for transition, (before, after) in TRANSITIONS.items():
            old, new = outcomes[before], outcomes[after]
            summary = transitions[transition]
            summary["corrections"] += not old[0] and new[0]
            summary["regressions"] += old[0] and not new[0]
            summary["top5_gains"] += not old[1] and new[1]
            summary["top5_losses"] += old[1] and not new[1]
            summary["ascii_signature_regressions"] += (
                has_ascii_reference and old[2] in signatures and new[2] not in signatures)
            summary["literal_signature_regressions"] += (
                old[3] == source_literal and new[3] != source_literal)
    need(count == ROWS and len(seen) == ROWS, "DEV denominator differs")
    overall = transitions["overall"]
    quality_gate = bool(
        metrics["proposed"]["top1"] > metrics["current_production"]["top1"]
        and metrics["proposed"]["top5"] >= metrics["current_production"]["top5"]
        and not any(overall[key] for key in (
            "regressions", "top5_losses", "ascii_signature_regressions",
            "literal_signature_regressions")))
    return {"branches": metrics, "transitions": transitions,
            "quality_gate": quality_gate,
            "actual_runtime_authorized": quality_gate}


def _verify_terminal_artifact(directory, expected_schema, expected_report_sha,
                              expected_capture_sha, closure_name):
    report = read_json(directory / "report.json")
    proof = read_json(directory / "independent-verification.json")
    closure = read_json(directory / closure_name)
    completion = read_json(directory / "completion.json")
    need(report.get("schema") == expected_schema and report.get("completed") is True
         and proof.get("verified") is True
         and proof.get("report_sha256") == expected_report_sha
         and proof.get("capture_sha256") == expected_capture_sha
         and report.get("capture_sha256", report.get("artifacts", {}).get("capture.jsonl"))
             == expected_capture_sha
         and closure.get("rows") == ROWS and closure.get("labels_opened") is False
         and closure.get("capture_sha256") == expected_capture_sha
         and completion.get("completed") is True
         and completion.get("report_sha256") == expected_report_sha,
         "terminal runtime artifact differs: " + str(directory))
    return report, proof


def authorize():
    need(all(path.is_file() and sha256_file(path) == digest
             for path, digest in FROZEN.items()), "frozen DEV input differs")
    train_report = read_json(TRAIN / "report.json")
    train_proof = read_json(TRAIN / "independent-verification.json")
    train_model = read_json(TRAIN / "model.json")
    train_completion = read_json(TRAIN / "completion.json")
    need(train_report.get("schema") == "current-phonetic-choice-training-v1"
         and train_report.get("completed") is True
         and train_report.get("qualification_passed") is True
         and train_report.get("full_fit_performed") is True
         and train_report.get("model") == train_model
         and train_proof.get("verified") is True
         and train_proof.get("qualification_passed") is True
         and train_proof.get("report_sha256") == FROZEN[TRAIN / "report.json"]
         and train_completion.get("completed") is True
         and train_completion.get("report_sha256") == FROZEN[TRAIN / "report.json"],
         "qualified independently verified TRAIN differs")
    validate_model(train_model)
    deployed_model = read_json(DEPLOYED_MODEL)
    validate_model(deployed_model)

    source_report, _ = _verify_terminal_artifact(
        SOURCE, "phonetic-choice-runtime-dev-v1", FROZEN[SOURCE / "report.json"],
        FROZEN[SOURCE / "capture.jsonl"], "capture.complete.json")
    comparator_report, _ = _verify_terminal_artifact(
        COMPARATOR, "score-scale-runtime-dev-v1", FROZEN[COMPARATOR / "report.json"],
        FROZEN[COMPARATOR / "capture.jsonl"], "closure.json")
    need(source_report.get("version") == FULL_VERSION
         and comparator_report.get("version") == CURRENT_VERSION
         and comparator_report.get("rows") == ROWS
         and comparator_report.get("quality_gate") is True,
         "accepted runtime capture identity differs")

    intake = read_json(INTAKE)
    artifacts = intake.get("shared_current_model_artifacts")
    need(intake.get("schema") == "current-phonetic-choice-dev-intake-v1"
         and intake.get("matched_count") == 32
         and isinstance(artifacts, list) and len(artifacts) == 32
         and len({row.get("path") for row in artifacts}) == 32
         and intake.get("dev_labels_opened") is False
         and intake.get("model_inference_run") is False,
         "DEV intake evidence differs")
    source_runtime_pins = {
        str((Path(name) if Path(name).is_absolute() else ROOT / name).resolve()): digest
        for name, digest in source_report.get("pins", {}).items()
    }
    pins = {str(path.resolve()): digest for path, digest in FROZEN.items()}
    for row in artifacts:
        path = Path(row.get("path", ""))
        digest = row.get("sha256")
        need(path.is_absolute() and path.is_file() and isinstance(digest, str)
             and sha256_file(path) == digest
             and source_runtime_pins.get(str(path.resolve())) == digest,
             "one of 32 generator pins differs from the full-union DEV source")
        need(str(path.resolve()) not in pins or pins[str(path.resolve())] == digest,
             "generator pin collision")
        pins[str(path.resolve())] = digest
    for report in (source_report, comparator_report, train_report):
        for name, digest in report.get("pins", {}).items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            key = str(path.resolve())
            if path.resolve() != DEV.resolve():
                need(path.is_file() and sha256_file(path) == digest,
                     "inherited artifact pin differs: " + key)
            need(key not in pins or pins[key] == digest, "inherited pin collision: " + key)
            pins[key] = digest
    need(str(DEV.resolve()) in pins, "DEV label digest is not inherited")
    for path in (DEPLOYED_MODEL, RESOURCE_HELPER, NUMERICAL_CORE, Path(__file__)):
        key, digest = str(path.resolve()), sha256_file(path)
        need(key not in pins or pins[key] == digest, "local pin collision: " + key)
        pins[key] = digest
    return train_model, deployed_model, dict(sorted(pins.items()))


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        with AtomicResourceWatchdog(started, system_swap_bytes(),
                                    output / "resources.jsonl", limits=LIMITS,
                                    interval_s=0.5) as guard:
            with guard.operation("authenticate", update=True):
                proposed_model, deployed_model, pins = authorize()
                write_json(output / "pins.json", pins)
            with guard.operation("project_all", update=True):
                source_rows = iter_jsonl(SOURCE / "capture.jsonl")
                comparator_rows = iter_jsonl(COMPARATOR / "capture.jsonl")
                count, seen = 0, set()
                with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
                    while True:
                        source_row = next(source_rows, None)
                        comparator_row = next(comparator_rows, None)
                        need((source_row is None) == (comparator_row is None),
                             "source/comparator denominator differs")
                        if source_row is None:
                            break
                        projected = project_row(source_row, comparator_row,
                                                deployed_model, proposed_model)
                        need(projected["id"] not in seen, "duplicate DEV row ID")
                        seen.add(projected["id"]); count += 1
                        append_jsonl(stream, projected)
                    stream.flush(); os.fsync(stream.fileno())
                need(count == ROWS and len(seen) == ROWS, "DEV source coverage differs")
                prediction_sha = sha256_file(output / "predictions.jsonl")
            with guard.operation("close_predictions", update=True):
                write_json(output / "closure.json", {
                    "schema": "current-phonetic-choice-dev-prediction-closure-v1",
                    "closed": True, "rows": ROWS,
                    "predictions_sha256": prediction_sha, "labels_opened": False,
                    "model_inference_performed": False})
            with guard.operation("late_label_join", update=True):
                need(sha256_file(DEV) == pins[str(DEV.resolve())], "DEV label pin differs")
                metrics = evaluate(output / "predictions.jsonl", DEV)
                current = metrics["branches"]["current_production"]
                need(all(current.get(key) == value
                         for key, value in CURRENT_REFERENCE.items()),
                    "current deployed comparator metrics differ")
                need(sha256_file(output / "predictions.jsonl") == prediction_sha,
                     "closed predictions changed after labels opened")
            with guard.operation("write_report", update=True):
                report = {
                    "schema": "current-phonetic-choice-dev-screen-v1",
                    "completed": True, "rows": ROWS, "branches": list(BRANCHES),
                    "metrics": metrics, "quality_gate": metrics["quality_gate"],
                    "actual_runtime_authorized": metrics["actual_runtime_authorized"],
                    "model_sha256": FROZEN[TRAIN / "model.json"],
                    "predictions_sha256": prediction_sha,
                    "closure_sha256": sha256_file(output / "closure.json"),
                    "pins": pins, "limits": LIMITS, "helper_mappings": HELPER_MAPPINGS,
                    "model_inference_performed": False, "training_performed": False,
                    "test_accessed": False, "production_changed": False,
                    "labels_opened_after_predictions": True,
                    "limitations": [
                        "Cached actual DEV candidates; no runtime adapter or latency measurement.",
                        "DEV was previously observed and cannot establish unseen generalization.",
                        "ASCII/literal signatures are preservation proxies, not complete span-copy proofs.",
                    ],
                }
                write_json(output / "report.json", report)
            terminal = guard.sample_and_append("after_report")
            with guard.operation("write_completion", update=True):
                for name, digest in pins.items():
                    path = Path(name)
                    need(sha256_file(path) == digest, "final pin changed: " + name)
                write_json(output / "completion.json", {
                    "completed": True,
                    "report_sha256": sha256_file(output / "report.json"),
                    "resource_after_report": terminal})
            guard.sample_and_append("after_completion")
        return report
    except BaseException as error:
        failure = output / "failure.json"
        if not failure.exists():
            write_json(failure, {"completed": False,
                                 "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps({key: result[key] for key in
                      ("completed", "rows", "quality_gate")}, ensure_ascii=False))
