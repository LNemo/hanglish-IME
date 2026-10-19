#!/usr/bin/env python3
"""Train the fixed score-scale recipe on the latest captured FIT candidates."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts import phonetic_score_scale as core
from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes


PLAN = ROOT / ".omx/plans/current-phonetic-choice-training-v1.md"
PREPARED = ROOT / ".omx/artifacts/current-phonetic-train-inputs-v1"
SHARDS = tuple(ROOT / f".omx/artifacts/current-phonetic-train-shard-{i}-v1"
               for i in range(6))
DEPLOYED = ROOT / ".omx/artifacts/phonetic-score-scale-training-v1/model.json"
FEATURE_PREPARER = ROOT / "scripts/prepare_current_phonetic_train_features.py"
LIMITS = {"whole_seconds": 600.0, "update_seconds": 60.0,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
STEPS = 200
ROWS = 1642
FEATURE_PREPARER_SHA256 = "0f005da20923705b0b5ad7d0966a0bb63aefde4b7dea476f565f5dbc459bd9b3"
CORE_SHA256 = "c40fe194cdedc3f34d0e6c420762ee0cab72d1f631e7676f1a5eb03b9f8f8015"
DEPLOYED_SHA256 = "e07462d50da0b67304a82a9a261068ccc84b8055a99b5d70af56db530d245838"
PREPARED_HASHES = {
    "requests.json": "97b416a805b9da8dc15d4ff9178b3a1db2a98f57f961a0e369c51bc771e1762d",
    "manifest.json": "c135c4c7f35c763257bca3b4369ab15165b186b20692bf5d8c1d53a6ec337685",
    "independent-verification.json": "4b34c0622f33afc6ec02c16415119159857f4a8947143412e84290a2379efc14",
}
BRANCHES = ("base_full", "deployed_full", "deployed_production", "proposed")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    need(isinstance(value, dict), "expected JSON object")
    return value


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def append(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                            allow_nan=False) + "\n")
    stream.flush(); os.fsync(stream.fileno())


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


def id_hash(identifiers):
    import hashlib
    encoded = json.dumps(sorted(identifiers), ensure_ascii=False,
                         separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def authenticate(features, expected_report_sha):
    features = Path(features).resolve()
    need(isinstance(expected_report_sha, str) and len(expected_report_sha) == 64
         and sha(features / "report.json") == expected_report_sha,
         "feature report SHA differs")
    report = read(features / "report.json")
    completion = read(features / "completion.json")
    outputs = report.get("outputs", {})
    need(not (features / "failure.json").exists()
         and not (features / "resource-watchdog-failure.json").exists()
         and report.get("completed") is True
         and report.get("schema") == "current-phonetic-train-features-v1"
         and report.get("counts", {}).get("rows") == ROWS
         and report.get("all_rows_preserved") is True
         and report.get("positive_candidates_injected") is False
         and report.get("non_fit_json_decoded") is False
         and report.get("non_fit_target_values_used") is False
         and report.get("dev_test_hold_reserve_labels_opened") is False
         and outputs.get("rows.jsonl") == sha(features / "rows.jsonl")
         and outputs.get("source-capture.complete.json")
             == sha(features / "source-capture.complete.json")
         and completion == {"completed": True,
                            "report_sha256": expected_report_sha},
         "completed feature artifact differs")
    need(sha(FEATURE_PREPARER) == FEATURE_PREPARER_SHA256
         and sha(Path(core.__file__)) == CORE_SHA256
         and sha(DEPLOYED) == DEPLOYED_SHA256
         and all(sha(PREPARED / name) == digest
                 for name, digest in PREPARED_HASHES.items()),
         "fixed training input/code differs")
    prepared = read(PREPARED / "requests.json")
    source_rows = prepared.get("rows")
    need(isinstance(source_rows, list) and len(source_rows) == ROWS,
         "prepared source rows differ")
    sources = {row.get("id"): row for row in source_rows}
    need(len(sources) == ROWS
         and all(set(row) == {"id", "input", "component_id"}
                 for row in source_rows), "prepared source identity differs")
    closure = read(features / "source-capture.complete.json")
    shard_evidence = closure.get("shards")
    need(closure.get("schema") == "current-phonetic-train-source-closure-v1"
         and closure.get("closed") is True and closure.get("rows") == ROWS
         and closure.get("labels_opened") is False
         and isinstance(shard_evidence, list) and len(shard_evidence) == 6
         and report.get("inputs", {}).get("shards") == shard_evidence,
         "source-only feature closure differs")
    for shard, evidence in enumerate(shard_evidence):
        directory = SHARDS[shard]
        need(evidence == {"shard": shard,
             "rows": len(prepared["shards"][shard]),
             "capture_sha256": evidence.get("capture_sha256"),
             "report_sha256": sha(directory / "report.json"),
             "proof_sha256": sha(directory / "independent-verification.json"),
             "launch_sha256": sha(directory.with_suffix(".launch.json")),
             "completion_sha256": sha(directory / "completion.json")},
             "closed shard evidence differs")
        shard_report = read(directory / "report.json")
        shard_proof = read(directory / "independent-verification.json")
        need(shard_report.get("capture_sha256") == evidence["capture_sha256"]
             and shard_report.get("ids") == prepared["shards"][shard]
             and shard_proof.get("verified") is True
             and shard_proof.get("mode") == "capture"
             and shard_proof.get("shard") == shard
             and shard_proof.get("capture_sha256") == evidence["capture_sha256"],
             "closed shard proof linkage differs")
    rows = list(jsonl(features / "rows.jsonl"))
    validate_rows(rows, sources)
    deployed = read(DEPLOYED)
    need(deployed.get("schema") == "phonetic-score-scale-model-v1"
         and deployed.get("arm") == "learned_scaled"
         and deployed.get("steps") == STEPS,
         "deployed model contract differs")
    pins = {str((features / name).resolve()): sha(features / name)
            for name in ("report.json", "completion.json", "rows.jsonl",
                         "source-capture.complete.json")}
    for name, digest in PREPARED_HASHES.items():
        pins[str((PREPARED / name).resolve())] = digest
    for path in (DEPLOYED, FEATURE_PREPARER, Path(__file__), Path(core.__file__),
                 PLAN, ROOT / "scripts/source_span_resources.py",
                 ROOT / "scripts/preflight_pko_seq2seq.py"):
        need(path.is_file(), "required training source missing: " + str(path))
        pins[str(path.resolve())] = sha(path)
    return rows, deployed, dict(sorted(pins.items())), report


def validate_rows(rows, sources):
    need(isinstance(rows, list) and len(rows) == len(sources)
         and len({row.get("id") for row in rows}) == len(rows),
         "feature row coverage differs")
    seen = set()
    for row in rows:
        identifier = row.get("id")
        source = sources.get(identifier)
        candidates, scores, features = (row.get("candidates"),
                                        row.get("base_scores"), row.get("features"))
        positives, production = row.get("positive_indices"), row.get("production_indices")
        ranked = row.get("ranked")
        need(source is not None and identifier not in seen
             and row.get("input") == source.get("input")
             and row.get("component_id") == source.get("component_id")
             and type(ranked) is bool and type(row.get("training_eligible")) is bool
             and isinstance(row.get("expected"), list) and row["expected"]
             and all(isinstance(value, str) and value for value in row["expected"])
             and isinstance(row.get("public_candidates"), list)
             and all(isinstance(value, str) for value in row["public_candidates"])
             and isinstance(candidates, list) and isinstance(scores, list)
             and isinstance(features, list)
             and len(candidates) == len(scores) == len(features)
             and len(set(candidates)) == len(candidates)
             and all(isinstance(value, str) for value in candidates)
             and all(finite(value) for value in scores)
             and all(isinstance(vector, list) and len(vector) == 4
                     and all(finite(value) for value in vector) for vector in features)
             and isinstance(positives, list) and len(set(positives)) == len(positives)
             and all(type(index) is int and 0 <= index < len(candidates)
                     for index in positives)
             and isinstance(production, list) and len(set(production)) == len(production)
             and all(type(index) is int and 0 <= index < len(candidates)
                     for index in production)
             and (not ranked or (candidates and production))
             and (ranked or not candidates), "feature row identity/domain differs")
        expected_positives = [index for index, text in enumerate(candidates)
                              if text in set(row["expected"])]
        recomputed_training_eligible = bool(ranked and positives)
        need(positives == expected_positives
             and row["training_eligible"] is recomputed_training_eligible,
             "positive indices/training eligibility differ")
        row["gradient_eligible"] = bool(
            recomputed_training_eligible and len(positives) < len(candidates))
        seen.add(identifier)
    need(seen == set(sources), "prepared/feature IDs differ")


def eligible_arrays(rows):
    eligible = [row for row in rows if row["gradient_eligible"]]
    need(eligible, "no eligible training rows")
    base, features, valid, positives, folds = core.arrays(eligible)
    return eligible, base, features, valid, positives, folds


def projection(elapsed, durations):
    need(finite(elapsed) and elapsed >= 0 and len(durations) == 3
         and all(finite(value) and value >= 0 for value in durations),
         "resource probe timing differs")
    return 1.3 * (elapsed + max(durations) * 1000) + 60.0


def fit(matrix, valid, positives, steps=STEPS, on_update=None):
    parameters = core.ANCHOR.copy()
    first = np.zeros(5, dtype=np.float64)
    second = np.zeros(5, dtype=np.float64)
    records = []
    for step in range(1, steps + 1):
        parameters, first, second, record = core.adam_step(
            parameters, first, second, step, matrix, valid, positives, True)
        records.append(record)
        if on_update is not None:
            on_update(record)
    return parameters, records


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
                    + np.dot(parameters[1:], np.asarray(row["features"][index]) / scale))
              for index in domain}
    need(all(math.isfinite(value) for value in scores.values()),
         "nonfinite candidate score")
    order = sorted(domain, key=lambda index: (-scores[index], candidates[index]))
    positive = [index for index in domain if candidates[index] in expected]
    latent = order[:5]
    latent_texts = [candidates[index] for index in latent]
    if row["input"] in latent_texts:
        public_top5 = latent_texts
    else:
        public_top5 = [candidates[index] for index in order[:4]] + [row["input"]]
    cross_entropy = (_logsumexp([scores[index] for index in domain])
                     - _logsumexp([scores[index] for index in positive])
                     if positive else None)
    return {"first": order[0] in positive,
            "top5": bool(expected & set(public_top5)),
            "latent_top5": bool(set(latent) & set(positive)),
            "covered": bool(positive), "cross_entropy": cross_entropy,
            "selected_index": order[0], "public_top5": public_top5,
            "latent_top5_indices": latent,
            "fallback_public": False}


def evaluate_row(row, proposed_parameters, proposed_scale,
                 deployed_parameters, deployed_scale):
    branches = {
        "base_full": score_outcome(row, core.ANCHOR, np.ones(4)),
        "deployed_full": score_outcome(row, deployed_parameters, deployed_scale),
        "deployed_production": score_outcome(
            row, deployed_parameters, deployed_scale,
            row["production_indices"] if row["candidates"] else None),
        "proposed": score_outcome(row, proposed_parameters, proposed_scale),
    }
    before, after = branches["deployed_production"], branches["proposed"]
    return {"id": row["id"], "component_id": row["component_id"],
            "fold": core.fold_for(row["component_id"]),
            "ranked": row["ranked"],
            "training_eligible": row["training_eligible"],
            "gradient_eligible": row["gradient_eligible"],
            "branches": branches,
            "correction_vs_deployed_production": after["first"] and not before["first"],
            "regression_vs_deployed_production": before["first"] and not after["first"]}


def summarize(predictions):
    need(predictions and len({row["id"] for row in predictions}) == len(predictions),
         "prediction coverage differs")
    result = {}
    for branch in BRANCHES:
        outcomes = [row["branches"][branch] for row in predictions]
        losses = [row["cross_entropy"] for row in outcomes
                  if row["cross_entropy"] is not None]
        evaluated_ids = [row["id"] for row in predictions]
        coverage_ids = [row["id"] for row, outcome in zip(predictions, outcomes)
                        if outcome["covered"]]
        ce_ids = [row["id"] for row, outcome in zip(predictions, outcomes)
                  if outcome["cross_entropy"] is not None]
        result[branch] = {"rows": len(outcomes),
                          "evaluated_id_sha256": id_hash(evaluated_ids),
                          "ranked_rows": sum(row["ranked"] for row in predictions),
                          "scored_pool_rows": sum(not row["fallback_public"]
                                                  for row in outcomes),
                          "fallback_public_rows": sum(row["fallback_public"]
                                                      for row in outcomes),
                          "fallback_public_first": sum(
                              row["fallback_public"] and row["first"]
                              for row in outcomes),
                          "fallback_public_top5": sum(
                              row["fallback_public"] and row["top5"]
                              for row in outcomes),
                          "first": sum(row["first"] for row in outcomes),
                          "top5": sum(row["top5"] for row in outcomes),
                          "latent_top5": sum(row["latent_top5"] for row in outcomes),
                          "coverage": sum(row["covered"] for row in outcomes),
                          "coverage_denominator": sum(
                              not row["fallback_public"] for row in outcomes),
                          "coverage_id_sha256": id_hash(coverage_ids),
                          "covered_cross_entropy_rows": len(losses),
                          "covered_cross_entropy_id_sha256": id_hash(ce_ids),
                          "covered_cross_entropy": (sum(losses) / len(losses)
                                                    if losses else None)}
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


def run(features, expected_report_sha, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    completed_updates = 0
    try:
        with AtomicResourceWatchdog(started, system_swap_bytes(),
                                    output / "resources.jsonl", limits=LIMITS,
                                    interval_s=.5) as guard:
            with guard.operation("authenticate", update=True):
                rows, deployed, pins, feature_report = authenticate(
                    features, expected_report_sha)
                write(output / "pins.json", pins)
                eligible, base, values, valid, positives, folds = eligible_arrays(rows)
                all_indices = np.arange(len(eligible))
                deployed_parameters = np.asarray(deployed["parameters"], dtype=np.float64)
                deployed_scale = np.asarray(deployed["scale"], dtype=np.float64)
                write(output / "folds.json", [{"id": row["id"],
                    "component_id": row["component_id"],
                    "fold": core.fold_for(row["component_id"]),
                    "training_eligible": row["training_eligible"],
                    "gradient_eligible": row["gradient_eligible"]} for row in rows])
            with guard.operation("prepare_probe", update=True):
                probe_scale = core.feature_scale(values, valid, all_indices, True)
                probe_matrix = core.design(base, values, probe_scale)
            probe_parameters = core.ANCHOR.copy()
            probe_first = np.zeros(5); probe_second = np.zeros(5); durations = []
            with (output / "probe-updates.jsonl").open("x", encoding="utf-8") as stream:
                for step in range(1, 4):
                    tick = time.monotonic()
                    with guard.operation(f"probe:{step}", update=True):
                        probe_parameters, probe_first, probe_second, record = core.adam_step(
                            probe_parameters, probe_first, probe_second, step,
                            probe_matrix, valid, positives, True)
                        append(stream, record)
                    durations.append(time.monotonic() - tick)
            projected = projection(time.monotonic() - started, durations)
            with guard.operation("qualify_resources", update=True):
                write(output / "qualification.json", {"qualified": projected <= 600.0,
                    "projected_seconds": projected, "probe_durations_s": durations,
                    "projection": "1.3*(elapsed+max_probe_seconds*1000)+60",
                    "maximum_real_fit_updates": 1000,
                    "probe_state_reused": False})
                need(projected <= LIMITS["whole_seconds"],
                     "resource projection exceeds limit")
            del probe_matrix, probe_parameters, probe_first, probe_second
            models, predictions = [], []
            with (output / "updates.jsonl").open("x", encoding="utf-8") as updates:
                for fold in range(4):
                    training = np.flatnonzero(folds != fold)
                    need(len(training) and (folds == fold).any(),
                         "fold has no usable training/held rows")
                    with guard.operation(f"prepare_fold:{fold}", update=True):
                        scale = core.feature_scale(values, valid, training, True)
                        matrix = core.design(base[training], values[training], scale)
                    def record_update(record, current_fold=fold):
                        nonlocal completed_updates
                        append(updates, {"fold": current_fold, **record})
                        completed_updates += 1
                    parameters = core.ANCHOR.copy(); first = np.zeros(5); second = np.zeros(5)
                    for step in range(1, STEPS + 1):
                        with guard.operation(f"fit:{fold}:{step}", update=True):
                            parameters, first, second, record = core.adam_step(
                                parameters, first, second, step, matrix,
                                valid[training], positives[training], True)
                            record_update(record)
                    held_rows = [row for row in rows
                                 if core.fold_for(row["component_id"]) == fold]
                    model = {"schema": "current-phonetic-choice-fold-model-v1",
                             "fold": fold, "arm": "learned_scaled",
                             "scale": scale.tolist(),
                             "parameters": parameters.tolist(), "steps": STEPS,
                             "training_eligible_rows": len(training),
                             "held_all_rows": len(held_rows)}
                    models.append(model)
                    with guard.operation(f"evaluate_fold:{fold}", update=True):
                        predictions.extend(evaluate_row(
                            row, parameters, scale, deployed_parameters, deployed_scale)
                            for row in held_rows)
            with guard.operation("close_oof", update=True):
                predictions.sort(key=lambda row: row["id"])
                need(len(predictions) == ROWS, "all-row OOF coverage differs")
                write(output / "fold-models.json", {"schema":
                    "current-phonetic-choice-fold-models-v1", "models": models})
                with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
                    for row in predictions:
                        append(stream, row)
                reloaded = read(output / "fold-models.json")["models"]
                repeated = []
                for model in reloaded:
                    repeated.extend(evaluate_row(
                        row, np.asarray(model["parameters"]), np.asarray(model["scale"]),
                        deployed_parameters, deployed_scale) for row in rows
                        if core.fold_for(row["component_id"]) == model["fold"])
                repeated.sort(key=lambda row: row["id"])
                need(repeated == predictions, "fold model reload predictions differ")
                summary = summarize(predictions)
                passed = qualified(summary)
                write(output / "oof-decision.json", {
                    "schema": "current-phonetic-choice-oof-decision-v1",
                    "preselected_recipe": "learned_scaled", "summaries": summary,
                    "qualification": "proposed.first > base_full.first and proposed.covered_CE < base_full.covered_CE",
                    "qualification_passed": passed,
                    "deployment_gate": False, "fold_reload_predictions_exact": True})
            full_model = None
            if passed:
                with guard.operation("prepare_full_fit", update=True):
                    scale = core.feature_scale(values, valid, all_indices, True)
                    matrix = core.design(base, values, scale)
                parameters = core.ANCHOR.copy(); first = np.zeros(5); second = np.zeros(5)
                with (output / "full-fit-updates.jsonl").open("x", encoding="utf-8") as stream:
                    for step in range(1, STEPS + 1):
                        with guard.operation(f"full_fit:{step}", update=True):
                            parameters, first, second, record = core.adam_step(
                                parameters, first, second, step, matrix,
                                valid, positives, True)
                            append(stream, record); completed_updates += 1
                with guard.operation("save_model", update=True):
                    full_model = {"schema": "phonetic-score-scale-model-v1",
                        "arm": "learned_scaled", "scale": scale.tolist(),
                        "parameters": parameters.tolist(), "steps": STEPS,
                        "features_manifest_sha256": expected_report_sha,
                        "features_rows_sha256": sha(Path(features) / "rows.jsonl"),
                        "numerical_core_sha256": CORE_SHA256,
                        "recipe": "component-fold-learned-scaled-200-adam-anchor-v1",
                        "training_source": "current-phonetic-train-features-v1",
                        "training_rows": ROWS, "eligible_rows": len(eligible)}
                    write(output / "model.json", full_model)
                    reloaded_model = read(output / "model.json")
                    need(np.array_equal(
                        np.einsum("rcf,f->rc", matrix, parameters),
                        np.einsum("rcf,f->rc", matrix,
                                  np.asarray(reloaded_model["parameters"]))),
                        "full model reload differs")
            with guard.operation("verify_pins", update=True):
                need(all(sha(path) == digest for path, digest in pins.items()),
                     "training input pin changed")
            artifacts = {name: sha(output / name) for name in (
                "pins.json", "folds.json", "probe-updates.jsonl",
                "qualification.json", "updates.jsonl", "fold-models.json",
                "predictions.jsonl", "oof-decision.json")}
            if passed:
                artifacts.update({name: sha(output / name)
                                  for name in ("full-fit-updates.jsonl", "model.json")})
            report = {"schema": "current-phonetic-choice-training-v1",
                "completed": True, "feature_report_sha256": expected_report_sha,
                "counts": {"rows": ROWS,
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
                    "candidates": sum(len(row["candidates"]) for row in rows),
                    "completed_updates": completed_updates},
                "fold_all_rows": {str(fold): sum(core.fold_for(row["component_id"]) == fold
                                                  for row in rows) for fold in range(4)},
                "fold_eligible_rows": {str(fold): int((folds == fold).sum())
                                       for fold in range(4)},
                "summaries": summary, "qualification_passed": passed,
                "qualification_is_deployment_gate": False,
                "full_fit_performed": passed, "model": full_model,
                "artifacts": artifacts, "pins": pins, "limits": LIMITS,
                "feature_input_counts": feature_report["counts"],
                "neural_model_loaded": False, "production_changed": False,
                "dev_hold_test_reserve_accessed": False,
                "limitations": [
                    "This component-held FIT comparison is not unseen-language generalization.",
                    "Qualification permits only downstream DEV evaluation, not deployment.",
                ]}
            with guard.operation("write_report", update=True):
                write(output / "report.json", report)
            terminal = guard.sample_and_append("after_report")
            with guard.operation("write_completion", update=True):
                write(output / "completion.json", {"completed": True,
                    "report_sha256": sha(output / "report.json"),
                    "resource_after_report": terminal})
            guard.sample_and_append("after_completion")
        return report
    except BaseException as error:
        if output.is_dir() and not (output / "failure.json").exists():
            write(output / "failure.json", {"completed": False,
                "completed_updates": completed_updates,
                "error": f"{type(error).__name__}: {error}",
                "dev_hold_test_reserve_accessed": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.features, arguments.features_report_sha256,
                 arguments.output)
    print(json.dumps({key: result[key] for key in
                      ("completed", "counts", "summaries",
                       "qualification_passed", "full_fit_performed")},
                     ensure_ascii=False, sort_keys=True))
