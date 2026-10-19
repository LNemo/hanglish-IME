#!/usr/bin/env python3
"""Independently replay a completed current-phonetic choice training run."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from scripts.verify_phonetic_score_scale import (
    check_resources,
    replay_updates,
    scale_for,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / ".omx/plans/current-phonetic-choice-training-v1.md"
PREPARED = ROOT / ".omx/artifacts/current-phonetic-train-inputs-v1"
SHARDS = tuple(ROOT / f".omx/artifacts/current-phonetic-train-shard-{index}-v1"
               for index in range(6))
DEPLOYED = ROOT / ".omx/artifacts/phonetic-score-scale-training-v1/model.json"
FEATURE_PREPARER = ROOT / "scripts/prepare_current_phonetic_train_features.py"
NUMERICAL_CORE = ROOT / "scripts/phonetic_score_scale.py"
INDEPENDENT_HELPER = ROOT / "scripts/verify_phonetic_score_scale.py"
RESOURCE_HELPER = ROOT / "scripts/source_span_resources.py"
PREFLIGHT = ROOT / "scripts/preflight_pko_seq2seq.py"

ROWS = 1642
STEPS = 200
BRANCHES = ("base_full", "deployed_full", "deployed_production", "proposed")
ANCHOR = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
LIMITS = {"whole_seconds": 600.0, "update_seconds": 60.0,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
CORE_SHA256 = "c40fe194cdedc3f34d0e6c420762ee0cab72d1f631e7676f1a5eb03b9f8f8015"
HELPER_SHA256 = "04b1b188a13a36f1a8502b0331a65b0b20c763c1608dc30e108da6fcd34daa28"
PLAN_SHA256 = "5ccefa8a2f553ac9c6a4f7cf27ffcea24568d47e2bc6dbd97caa61f4f9689cfc"
FEATURE_PREPARER_SHA256 = "0f005da20923705b0b5ad7d0966a0bb63aefde4b7dea476f565f5dbc459bd9b3"
DEPLOYED_SHA256 = "e07462d50da0b67304a82a9a261068ccc84b8055a99b5d70af56db530d245838"
PREPARED_HASHES = {
    "requests.json": "97b416a805b9da8dc15d4ff9178b3a1db2a98f57f961a0e369c51bc771e1762d",
    "manifest.json": "c135c4c7f35c763257bca3b4369ab15165b186b20692bf5d8c1d53a6ec337685",
    "independent-verification.json": "4b34c0622f33afc6ec02c16415119159857f4a8947143412e84290a2379efc14",
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    value = json_value(path)
    need(isinstance(value, dict), "expected JSON object: " + str(path))
    return value


def json_value(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSONL at {path}:{number}") from error
            need(isinstance(value, dict), "JSONL row is not an object")
            result.append(value)
    return result


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-8)


def fold_for(component_id):
    need(isinstance(component_id, str), "invalid component ID")
    value = hashlib.sha256(
        ("phonetic-score-scale-v1:" + component_id).encode()).hexdigest()
    return int(value, 16) % 4


def id_hash(identifiers):
    encoded = json.dumps(sorted(identifiers), ensure_ascii=False,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def process_absent(pid, probe=os.kill):
    need(type(pid) is int and pid > 0, "invalid launch PID")
    try:
        probe(pid, 0)
    except ProcessLookupError:
        return True
    raise ValueError("training process is still present")


def expected_launch_command(features, report_sha, output):
    return [str(ROOT / ".venv/bin/python"), "-m",
            "scripts.train_current_phonetic_choice",
            "--features", str(Path(features).resolve()),
            "--features-report-sha256", report_sha,
            "--output", str(Path(output).resolve())]


def verify_launch(features, report_sha, output, trainer_sha, pid_probe=os.kill):
    output = Path(output).resolve()
    launch_path = output.with_suffix(".launch.json")
    launch = read(launch_path)
    need(set(launch) == {"pid", "command", "output", "log",
                         "runner_sha256", "launched_at"}
         and launch.get("command") == expected_launch_command(
             features, report_sha, output)
         and launch.get("output") == str(output)
         and launch.get("log") == str(output.with_suffix(".log"))
         and launch.get("runner_sha256") == trainer_sha
         and Path(launch["log"]).is_file(), "launch receipt differs")
    datetime.fromisoformat(launch["launched_at"])
    process_absent(launch.get("pid"), pid_probe)
    return launch_path


def validate_feature_rows(rows, source_rows):
    need(len(rows) == len(source_rows) == ROWS, "feature row count differs")
    sources = {row.get("id"): row for row in source_rows}
    need(len(sources) == ROWS
         and all(set(row) == {"id", "input", "component_id"}
                 for row in source_rows), "prepared source domain differs")
    seen = set()
    for row in rows:
        identifier = row.get("id")
        source = sources.get(identifier)
        candidates = row.get("candidates")
        scores = row.get("base_scores")
        features = row.get("features")
        positives = row.get("positive_indices")
        production = row.get("production_indices")
        expected = row.get("expected")
        ranked = row.get("ranked")
        need(source is not None and identifier not in seen
             and row.get("input") == source["input"]
             and row.get("component_id") == source["component_id"]
             and type(ranked) is bool
             and type(row.get("training_eligible")) is bool
             and isinstance(expected, list) and expected
             and len(set(expected)) == len(expected)
             and all(isinstance(value, str) and value for value in expected)
             and isinstance(row.get("public_candidates"), list)
             and all(isinstance(value, str)
                     for value in row["public_candidates"])
             and isinstance(candidates, list) and isinstance(scores, list)
             and isinstance(features, list)
             and len(candidates) == len(scores) == len(features)
             and len(set(candidates)) == len(candidates)
             and all(isinstance(value, str) for value in candidates)
             and all(finite(value) for value in scores)
             and all(isinstance(vector, list) and len(vector) == 4
                     and all(finite(value) for value in vector)
                     for vector in features)
             and isinstance(positives, list)
             and len(set(positives)) == len(positives)
             and all(type(index) is int and 0 <= index < len(candidates)
                     for index in positives)
             and isinstance(production, list)
             and len(set(production)) == len(production)
             and all(type(index) is int and 0 <= index < len(candidates)
                     for index in production)
             and (not ranked or (candidates and production))
             and (ranked or not candidates), "feature row domain differs")
        recomputed = [index for index, text in enumerate(candidates)
                      if text in set(expected)]
        training_eligible = bool(ranked and recomputed)
        need(positives == recomputed
             and row["training_eligible"] is training_eligible,
             "feature eligibility differs")
        row["gradient_eligible"] = bool(
            training_eligible and len(positives) < len(candidates))
        seen.add(identifier)
    need(seen == set(sources), "prepared/feature ID domain differs")
    return [row for row in rows if row["gradient_eligible"]]


def verify_source_closure(closure, feature_report, prepared):
    evidence = closure.get("shards")
    need(closure.get("schema") == "current-phonetic-train-source-closure-v1"
         and closure.get("closed") is True and closure.get("rows") == ROWS
         and closure.get("labels_opened") is False
         and isinstance(evidence, list) and len(evidence) == len(SHARDS)
         and feature_report.get("inputs", {}).get("shards") == evidence,
         "source closure differs")
    need(isinstance(prepared.get("shards"), list)
         and len(prepared["shards"]) == len(SHARDS),
         "prepared shard partition differs")
    for shard, item in enumerate(evidence):
        directory = Path(SHARDS[shard])
        report = read(directory / "report.json")
        proof = read(directory / "independent-verification.json")
        expected = {
            "shard": shard, "rows": len(prepared["shards"][shard]),
            "capture_sha256": report.get("capture_sha256"),
            "report_sha256": sha(directory / "report.json"),
            "proof_sha256": sha(directory / "independent-verification.json"),
            "launch_sha256": sha(directory.with_suffix(".launch.json")),
            "completion_sha256": sha(directory / "completion.json"),
        }
        need(item == expected and report.get("completed") is True
             and report.get("mode") == "capture" and report.get("shard") == shard
             and report.get("rows") == expected["rows"]
             and report.get("ids") == prepared["shards"][shard]
             and report.get("labels_opened") is False
             and proof.get("verified") is True and proof.get("mode") == "capture"
             and proof.get("shard") == shard
             and proof.get("capture_sha256") == item["capture_sha256"]
             and proof.get("report_sha256") == item["report_sha256"]
             and proof.get("completion_sha256") == item["completion_sha256"]
             and proof.get("launch_sha256") == item["launch_sha256"],
             "closed shard evidence differs")
    need(sum(item["rows"] for item in evidence) == ROWS,
         "source closure row coverage differs")


def _logsumexp(values):
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def score_outcome(row, parameters, scale, indices=None):
    candidates = row["candidates"]
    expected = set(row["expected"])
    if not candidates:
        public = row["public_candidates"]
        return {"first": bool(public and public[0] in expected),
                "top5": bool(expected & set(public[:5])),
                "latent_top5": bool(expected & set(public[:5])),
                "covered": False, "cross_entropy": None,
                "selected_index": None, "public_top5": public[:5],
                "latent_top5_indices": [], "fallback_public": True}
    domain = list(range(len(candidates))) if indices is None else list(indices)
    need(domain, "scored candidate domain is empty")
    parameters = np.asarray(parameters, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    need(parameters.shape == (5,) and scale.shape == (4,)
         and np.isfinite(parameters).all() and np.isfinite(scale).all()
         and parameters[0] >= 0 and (scale > 0).all(), "score model differs")
    scores = {index: float(parameters[0] * row["base_scores"][index]
                    + np.dot(parameters[1:],
                             np.asarray(row["features"][index]) / scale))
              for index in domain}
    need(all(math.isfinite(value) for value in scores.values()),
         "nonfinite score")
    order = sorted(domain, key=lambda index: (-scores[index], candidates[index]))
    positives = [index for index in domain if candidates[index] in expected]
    latent = order[:5]
    latent_texts = [candidates[index] for index in latent]
    public = (latent_texts if row["input"] in latent_texts
              else [candidates[index] for index in order[:4]] + [row["input"]])
    cross_entropy = (_logsumexp([scores[index] for index in domain])
                     - _logsumexp([scores[index] for index in positives])
                     if positives else None)
    return {"first": order[0] in positives,
            "top5": bool(expected & set(public)),
            "latent_top5": bool(set(latent) & set(positives)),
            "covered": bool(positives), "cross_entropy": cross_entropy,
            "selected_index": order[0], "public_top5": public,
            "latent_top5_indices": latent, "fallback_public": False}


def evaluate_row(row, proposed_parameters, proposed_scale,
                 deployed_parameters, deployed_scale):
    branches = {
        "base_full": score_outcome(row, ANCHOR, np.ones(4)),
        "deployed_full": score_outcome(
            row, deployed_parameters, deployed_scale),
        "deployed_production": score_outcome(
            row, deployed_parameters, deployed_scale,
            row["production_indices"] if row["candidates"] else None),
        "proposed": score_outcome(row, proposed_parameters, proposed_scale),
    }
    before, after = branches["deployed_production"], branches["proposed"]
    return {"id": row["id"], "component_id": row["component_id"],
            "fold": fold_for(row["component_id"]), "ranked": row["ranked"],
            "training_eligible": row["training_eligible"],
            "gradient_eligible": row["gradient_eligible"],
            "branches": branches,
            "correction_vs_deployed_production": after["first"] and not before["first"],
            "regression_vs_deployed_production": before["first"] and not after["first"]}


def summarize(predictions):
    need(len(predictions) == ROWS
         and len({row["id"] for row in predictions}) == ROWS,
         "prediction coverage differs")
    result = {}
    for branch in BRANCHES:
        outcomes = [row["branches"][branch] for row in predictions]
        losses = [outcome["cross_entropy"] for outcome in outcomes
                  if outcome["cross_entropy"] is not None]
        coverage_ids = [row["id"] for row, outcome in zip(predictions, outcomes)
                        if outcome["covered"]]
        ce_ids = [row["id"] for row, outcome in zip(predictions, outcomes)
                  if outcome["cross_entropy"] is not None]
        result[branch] = {
            "rows": len(outcomes),
            "evaluated_id_sha256": id_hash([row["id"] for row in predictions]),
            "ranked_rows": sum(row["ranked"] for row in predictions),
            "scored_pool_rows": sum(not outcome["fallback_public"]
                                    for outcome in outcomes),
            "fallback_public_rows": sum(outcome["fallback_public"]
                                        for outcome in outcomes),
            "fallback_public_first": sum(
                outcome["fallback_public"] and outcome["first"]
                for outcome in outcomes),
            "fallback_public_top5": sum(
                outcome["fallback_public"] and outcome["top5"]
                for outcome in outcomes),
            "first": sum(outcome["first"] for outcome in outcomes),
            "top5": sum(outcome["top5"] for outcome in outcomes),
            "latent_top5": sum(outcome["latent_top5"] for outcome in outcomes),
            "coverage": sum(outcome["covered"] for outcome in outcomes),
            "coverage_denominator": sum(not outcome["fallback_public"]
                                        for outcome in outcomes),
            "coverage_id_sha256": id_hash(coverage_ids),
            "covered_cross_entropy_rows": len(losses),
            "covered_cross_entropy_id_sha256": id_hash(ce_ids),
            "covered_cross_entropy": (sum(losses) / len(losses)
                                      if losses else None),
        }

    def transition(before, after):
        return {"corrections": sum(row["branches"][after]["first"]
                                   and not row["branches"][before]["first"]
                                   for row in predictions),
                "regressions": sum(row["branches"][before]["first"]
                                   and not row["branches"][after]["first"]
                                   for row in predictions)}

    result["candidate_expansion_deployed_production_to_full"] = transition(
        "deployed_production", "deployed_full")
    result["refit_deployed_full_to_proposed"] = transition(
        "deployed_full", "proposed")
    result["versus_deployed_production"] = transition(
        "deployed_production", "proposed")
    return result


def qualified(summary):
    before, after = summary["base_full"], summary["proposed"]
    need(after["rows"] == before["rows"] == ROWS
         and after["evaluated_id_sha256"] == before["evaluated_id_sha256"]
         and after["covered_cross_entropy_id_sha256"]
             == before["covered_cross_entropy_id_sha256"],
         "OOF comparison domains differ")
    return bool(after["first"] > before["first"]
                and after["covered_cross_entropy"] is not None
                and before["covered_cross_entropy"] is not None
                and after["covered_cross_entropy"]
                    < before["covered_cross_entropy"])


def recursive_close(actual, expected, path="value"):
    if isinstance(expected, dict):
        need(isinstance(actual, dict) and set(actual) == set(expected),
             path + " fields differ")
        for key in expected:
            recursive_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        need(isinstance(actual, list) and len(actual) == len(expected),
             path + " length differs")
        for index, value in enumerate(expected):
            recursive_close(actual[index], value, f"{path}[{index}]")
    elif type(expected) is float:
        close(actual, expected)
    else:
        need(actual == expected, path + " differs")


def expected_operations(passed):
    operations = ["authenticate", "prepare_probe", "probe:1", "probe:2",
                  "probe:3", "qualify_resources"]
    for fold in range(4):
        operations += ([f"prepare_fold:{fold}"]
                       + [f"fit:{fold}:{step}" for step in range(1, STEPS + 1)]
                       + [f"evaluate_fold:{fold}"])
    operations.append("close_oof")
    if passed:
        operations += (["prepare_full_fit"]
                       + [f"full_fit:{step}" for step in range(1, STEPS + 1)]
                       + ["save_model"])
    operations += ["verify_pins", "write_report", "write_completion"]
    return operations


def verify(features, features_report_sha, output, trainer_sha,
           *, pid_probe=os.kill):
    features, output = Path(features).resolve(), Path(output).resolve()
    need(len(features_report_sha) == len(trainer_sha) == 64,
         "invalid required SHA")
    proof_path = output / "independent-verification.json"
    need(output.is_dir() and not proof_path.exists(),
         "output missing or independent proof already exists")
    need(not any((output / name).exists() for name in
                 ("failure.json", "resource-watchdog-failure.json")),
         "failed training run")
    trainer = ROOT / "scripts/train_current_phonetic_choice.py"
    need(sha(trainer) == trainer_sha, "trainer SHA differs")
    need(sha(features / "report.json") == features_report_sha,
         "feature report SHA differs")
    need(sha(NUMERICAL_CORE) == CORE_SHA256
         and sha(INDEPENDENT_HELPER) == HELPER_SHA256
         and sha(PLAN) == PLAN_SHA256
         and sha(FEATURE_PREPARER) == FEATURE_PREPARER_SHA256
         and sha(DEPLOYED) == DEPLOYED_SHA256,
         "fixed helper/model identity differs")
    launch_path = verify_launch(features, features_report_sha, output,
                                trainer_sha, pid_probe)

    feature_report = read(features / "report.json")
    feature_completion = read(features / "completion.json")
    feature_closure = features / "source-capture.complete.json"
    closure = read(feature_closure)
    need(feature_report.get("completed") is True
         and feature_report.get("schema") == "current-phonetic-train-features-v1"
         and feature_report.get("counts", {}).get("rows") == ROWS
         and feature_report.get("outputs", {}).get("rows.jsonl")
             == sha(features / "rows.jsonl")
         and feature_report.get("outputs", {}).get(
             "source-capture.complete.json") == sha(feature_closure)
         and feature_report.get("all_rows_preserved") is True
         and feature_report.get("positive_candidates_injected") is False
         and feature_report.get("non_fit_json_decoded") is False
         and feature_report.get("non_fit_target_values_used") is False
         and feature_report.get("dev_test_hold_reserve_labels_opened") is False
         and feature_completion == {"completed": True,
                                    "report_sha256": features_report_sha},
         "feature artifact closure differs")
    need(all(sha(PREPARED / name) == digest
             for name, digest in PREPARED_HASHES.items()),
         "prepared source identity differs")
    prepared = read(PREPARED / "requests.json")
    verify_source_closure(closure, feature_report, prepared)
    source_rows = prepared.get("rows")
    rows = jsonl(features / "rows.jsonl")
    eligible = validate_feature_rows(rows, source_rows)
    need(eligible and {fold_for(row["component_id"]) for row in eligible}
         == set(range(4)), "gradient fold coverage differs")

    report, completion = read(output / "report.json"), read(output / "completion.json")
    resource_records = jsonl(output / "resources.jsonl")
    after_report = [record for record in resource_records
                    if record.get("stage") == "after_report"]
    pins = read(output / "pins.json")
    need(report.get("schema") == "current-phonetic-choice-training-v1"
         and report.get("completed") is True
         and report.get("feature_report_sha256") == features_report_sha
         and report.get("feature_input_counts") == feature_report.get("counts")
         and report.get("pins") == pins
         and report.get("limits") == LIMITS
         and report.get("neural_model_loaded") is False
         and report.get("production_changed") is False
         and report.get("dev_hold_test_reserve_accessed") is False,
         "training report contract differs")
    need(completion.get("completed") is True
         and completion.get("report_sha256") == sha(output / "report.json")
         and set(completion) == {"completed", "report_sha256",
                                "resource_after_report"}
         and len(after_report) == 1
         and completion.get("resource_after_report") == after_report[0],
         "training completion differs")
    expected_pins = {
        str((features / name).resolve()): sha(features / name)
        for name in ("report.json", "completion.json", "rows.jsonl",
                     "source-capture.complete.json")}
    expected_pins.update({str((PREPARED / name).resolve()): digest
                          for name, digest in PREPARED_HASHES.items()})
    for path in (DEPLOYED, FEATURE_PREPARER, trainer, NUMERICAL_CORE, PLAN,
                 RESOURCE_HELPER, PREFLIGHT):
        expected_pins[str(path.resolve())] = sha(path)
    need(pins == dict(sorted(expected_pins.items())), "training pin set differs")
    need(pins[str(trainer.resolve())] == trainer_sha
         and pins[str(NUMERICAL_CORE.resolve())] == CORE_SHA256
         and pins[str(PLAN.resolve())] == PLAN_SHA256, "recipe pins differ")
    for path, digest in pins.items():
        need(Path(path).is_file() and sha(path) == digest,
             "runtime pin changed: " + path)

    artifacts = report.get("artifacts")
    need(isinstance(artifacts, dict), "artifact manifest missing")
    for name, digest in artifacts.items():
        need(Path(name).name == name and (output / name).is_file()
             and sha(output / name) == digest, "artifact hash differs: " + name)

    folds = [{"id": row["id"], "component_id": row["component_id"],
              "fold": fold_for(row["component_id"]),
              "training_eligible": row["training_eligible"],
              "gradient_eligible": row["gradient_eligible"]} for row in rows]
    need(json_value(output / "folds.json") == folds, "fold assignment differs")
    fold_all = {str(fold): sum(fold_for(row["component_id"]) == fold
                               for row in rows) for fold in range(4)}
    fold_eligible = {str(fold): sum(fold_for(row["component_id"]) == fold
                                    for row in eligible) for fold in range(4)}
    need(report.get("fold_all_rows") == fold_all
         and report.get("fold_eligible_rows") == fold_eligible,
         "fold counts differ")

    counts = {"rows": ROWS,
              "no_candidate_rows": sum(not row["candidates"] for row in rows),
              "no_positive_rows": sum(bool(row["candidates"])
                                       and not row["positive_indices"] for row in rows),
              "all_positive_rows": sum(bool(row["candidates"])
                  and len(row["positive_indices"]) == len(row["candidates"])
                  for row in rows),
              "feature_training_eligible_rows": sum(
                  row["training_eligible"] for row in rows),
              "gradient_eligible_rows": len(eligible),
              "gradient_ineligible_rows": ROWS - len(eligible),
              "candidates": sum(len(row["candidates"]) for row in rows)}

    probe = jsonl(output / "probe-updates.jsonl")
    probe_scale = scale_for(eligible, True)
    replay_updates(probe, eligible, probe_scale, True, 3)
    qualification = read(output / "qualification.json")
    durations = qualification.get("probe_durations_s")
    probe_ends = {record.get("operation"): record
                  for record in resource_records
                  if record.get("stage") == "operation_end"
                  and record.get("operation") in {"probe:1", "probe:2", "probe:3"}}
    qualify_starts = [record for record in resource_records
                      if record.get("stage") == "operation_start"
                      and record.get("operation") == "qualify_resources"]
    safe_durations = (durations if isinstance(durations, list)
                      and len(durations) == 3
                      and all(finite(value) for value in durations)
                      else [math.nan])
    implied_elapsed = ((qualification.get("projected_seconds", math.nan) - 60.0)
                       / 1.3 - max(safe_durations) * 1000.0)
    need(qualification.get("qualified") is True
         and qualification.get("maximum_real_fit_updates") == 1000
         and qualification.get("probe_state_reused") is False
         and qualification.get("projection")
             == "1.3*(elapsed+max_probe_seconds*1000)+60"
         and isinstance(durations, list) and len(durations) == 3
         and all(finite(value) and value >= 0 for value in durations)
         and finite(qualification.get("projected_seconds"))
         and qualification["projected_seconds"] <= LIMITS["whole_seconds"]
         and set(probe_ends) == {"probe:1", "probe:2", "probe:3"}
         and len(qualify_starts) == 1
         and all(durations[index - 1] + 1e-6
                 >= probe_ends[f"probe:{index}"]["operation_elapsed_s"]
                 for index in range(1, 4))
         and probe_ends["probe:3"]["elapsed_s"] - 1e-6 <= implied_elapsed
         and implied_elapsed <= qualify_starts[0]["elapsed_s"] + 1e-6,
         "resource qualification differs")

    models_file = read(output / "fold-models.json")
    models = models_file.get("models")
    need(models_file.get("schema") == "current-phonetic-choice-fold-models-v1"
         and isinstance(models, list) and [model.get("fold") for model in models]
             == list(range(4)), "fold model coverage differs")
    updates = jsonl(output / "updates.jsonl")
    need(len(updates) == 4 * STEPS, "fold update count differs")
    deployed = read(DEPLOYED)
    deployed_parameters = np.asarray(deployed["parameters"], dtype=np.float64)
    deployed_scale = np.asarray(deployed["scale"], dtype=np.float64)
    predictions = []
    for model in models:
        fold = model["fold"]
        training = [row for row in eligible
                    if fold_for(row["component_id"]) != fold]
        held = [row for row in rows if fold_for(row["component_id"]) == fold]
        need(not ({row["component_id"] for row in training}
                  & {row["component_id"] for row in held}),
             "component fold leakage")
        scale = scale_for(training, True)
        selected = [record for record in updates if record.get("fold") == fold]
        parameters = replay_updates(selected, training, scale, True, STEPS)
        expected_model = {"schema": "current-phonetic-choice-fold-model-v1",
                          "fold": fold, "arm": "learned_scaled",
                          "scale": scale.tolist(),
                          "parameters": parameters.tolist(), "steps": STEPS,
                          "training_eligible_rows": len(training),
                          "held_all_rows": len(held)}
        recursive_close(model, expected_model, f"fold_model[{fold}]")
        predictions.extend(evaluate_row(row, parameters, scale,
                                        deployed_parameters, deployed_scale)
                           for row in held)
    predictions.sort(key=lambda row: row["id"])
    recorded_predictions = jsonl(output / "predictions.jsonl")
    recursive_close(recorded_predictions, predictions, "predictions")
    summary = summarize(predictions)
    recursive_close(report.get("summaries"), summary, "report.summaries")
    decision = read(output / "oof-decision.json")
    passed = qualified(summary)
    need(decision.get("schema") == "current-phonetic-choice-oof-decision-v1"
         and decision.get("preselected_recipe") == "learned_scaled"
         and decision.get("qualification")
             == "proposed.first > base_full.first and proposed.covered_CE < base_full.covered_CE"
         and decision.get("qualification_passed") is passed
         and decision.get("deployment_gate") is False
         and decision.get("fold_reload_predictions_exact") is True,
         "OOF decision differs")
    recursive_close(decision.get("summaries"), summary, "decision.summaries")
    need(report.get("qualification_passed") is passed
         and report.get("full_fit_performed") is passed
         and report.get("qualification_is_deployment_gate") is False,
         "qualification gate differs")

    model = None
    if passed:
        scale = scale_for(eligible, True)
        parameters = replay_updates(jsonl(output / "full-fit-updates.jsonl"),
                                    eligible, scale, True, STEPS)
        model = read(output / "model.json")
        expected_model = {
            "schema": "phonetic-score-scale-model-v1", "arm": "learned_scaled",
            "scale": scale.tolist(), "parameters": parameters.tolist(),
            "steps": STEPS, "features_manifest_sha256": features_report_sha,
            "features_rows_sha256": sha(features / "rows.jsonl"),
            "numerical_core_sha256": CORE_SHA256,
            "recipe": "component-fold-learned-scaled-200-adam-anchor-v1",
            "training_source": "current-phonetic-train-features-v1",
            "training_rows": ROWS, "eligible_rows": len(eligible),
        }
        recursive_close(model, expected_model, "model")
        recursive_close(report.get("model"), model, "report.model")
        # Replay scores from the serialized model, not only its parameters.
        for row in rows:
            recursive_close(
                score_outcome(row, np.asarray(model["parameters"]),
                              np.asarray(model["scale"])),
                score_outcome(row, parameters, scale), "model_reload_score")
    else:
        need(report.get("model") is None
             and not (output / "model.json").exists()
             and not (output / "full-fit-updates.jsonl").exists(),
             "unexpected full-fit artifact")

    counts["completed_updates"] = 4 * STEPS + (STEPS if passed else 0)
    need(report.get("counts") == counts, "training counts differ")
    expected_artifacts = {"pins.json", "folds.json", "probe-updates.jsonl",
                          "qualification.json", "updates.jsonl",
                          "fold-models.json", "predictions.jsonl",
                          "oof-decision.json"}
    if passed:
        expected_artifacts |= {"full-fit-updates.jsonl", "model.json"}
    need(set(artifacts) == expected_artifacts, "artifact coverage differs")

    resources = check_resources(resource_records,
                                expected_operations(passed))
    proof = {
        "verified": True,
        "report_sha256": sha(output / "report.json"),
        "completion_sha256": sha(output / "completion.json"),
        "launch_sha256": sha(launch_path),
        "features_report_sha256": features_report_sha,
        "features_rows_sha256": sha(features / "rows.jsonl"),
        "features_completion_sha256": sha(features / "completion.json"),
        "features_source_closure_sha256": sha(feature_closure),
        "trainer_sha256": trainer_sha,
        "verifier_sha256": sha(__file__),
        "independent_helper_sha256": HELPER_SHA256,
        "plan_sha256": PLAN_SHA256,
        "counts": counts,
        "summaries": summary,
        "qualification_passed": passed,
        "resources": resources,
        "contracts_verified": {
            "terminal_exclusive_output_and_absent_pid": True,
            "exact_prepared_fit_source_domain": True,
            "feature_and_source_closure_hashes": True,
            "component_fold_isolation_and_scales": True,
            "all_fold_and_full_fit_adam_updates_replayed": True,
            "all_row_branch_predictions_and_metrics_replayed": True,
            "saved_model_scores_reloaded": passed,
            "model_or_neural_inference_run_by_verifier": False,
        },
        "limits": [
            "Gradient/loss values are independently recomputed at the first and last update of each fit; every Adam parameter transition is replayed.",
            "The verifier authenticates prepared FIT identities and the SHA-bound feature artifact; it does not open the mixed FIT/inner-HOLD container.",
            "This FIT component-held comparison is not unseen-language, DEV, TEST, reserve, web-runtime, or deployment evidence.",
            "Stored neural likelihood features are authenticated but the neural model is not recomputed.",
        ],
    }
    with proof_path.open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trainer-sha256", required=True)
    arguments = parser.parse_args()
    result = verify(arguments.features, arguments.features_report_sha256,
                    arguments.output, arguments.trainer_sha256)
    print(json.dumps({"verified": result["verified"],
                      "qualification_passed": result["qualification_passed"],
                      "resources": result["resources"]}, sort_keys=True))
