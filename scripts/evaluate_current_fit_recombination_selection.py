#!/usr/bin/env python3
"""Late-label evaluation of six independently verified recombination score shards."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
from itertools import chain, zip_longest
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes

EVALUATOR = Path(__file__).resolve()
INTAKE_SCHEMA = "current-fit-recombination-selection-intake-v1"
HELPER = ROOT / "scripts/current_recombination_selection.py"
SCORING_CORE = ROOT / "scripts/current_recombination_scoring.py"
SCORING_RUNNER = ROOT / "scripts/capture_current_fit_recombination_scores.py"
SCORING_VERIFIER = ROOT / "scripts/verify_current_fit_recombination_scores.py"
PREPARED = ROOT / ".omx/artifacts/current-phonetic-train-inputs-v1/requests.json"
FEATURES = ROOT / ".omx/artifacts/current-phonetic-train-features-v1/rows.jsonl"
MODEL = ROOT / ".omx/artifacts/phonetic-score-scale-training-v1/model.json"
PREPARED_SHA256 = "97b416a805b9da8dc15d4ff9178b3a1db2a98f57f961a0e369c51bc771e1762d"
FEATURES_SHA256 = "4c13063656d9a108316b9ce1ceec394a61e2c45c2ba86e7a8c8361f7480d6cb4"
MODEL_SHA256 = "e07462d50da0b67304a82a9a261068ccc84b8055a99b5d70af56db530d245838"
LIMITS = {"whole_seconds": 300.0, "update_seconds": 30.0,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
ROWS = 1642
NAMES = ("report.json", "capture.jsonl", "capture.complete.json", "completion.json",
         "independent-verification.json", "pins.json", "resources.jsonl")
ARMS = ("old_sentence", "expanded_sentence", "old_calibrated", "expanded_calibrated")
TRANSITIONS = ("candidate_expansion_sentence", "candidate_expansion_calibrated",
               "calibration_old_pool", "calibration_expanded_pool", "overall")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return isinstance(value, str) and DIGEST.fullmatch(value) is not None


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    need(isinstance(value, dict), "expected JSON object: " + str(path))
    return value


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            need(line.strip(), f"blank JSONL line at {path}:{number}")
            value = json.loads(line)
            need(isinstance(value, dict), "JSONL row is not an object")
            yield value


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def append(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _merge_pin(pins, path, value):
    key = str(Path(path).resolve())
    need(digest(value) and (key not in pins or pins[key] == value), "pin collision: " + key)
    pins[key] = value


def _load_helper(helper_sha256):
    need(digest(helper_sha256) and HELPER.is_file() and sha(HELPER) == helper_sha256,
         "selection helper identity differs")
    spec = importlib.util.spec_from_file_location("_current_recombination_selection", HELPER)
    need(spec is not None and spec.loader is not None, "selection helper import differs")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    need(callable(getattr(module, "compare_row", None)), "selection helper API differs")
    return module


def authenticate(intake, intake_sha256, evaluator_sha256):
    intake = Path(intake)
    need(intake.is_absolute() and intake.is_file() and digest(intake_sha256)
         and sha(intake) == intake_sha256, "intake identity differs")
    need(digest(evaluator_sha256) and sha(EVALUATOR) == evaluator_sha256,
         "evaluator self identity differs")
    manifest = read(intake)
    entries = manifest.get("shards")
    core_sha, runner_sha = manifest.get("core_sha256"), manifest.get("runner_sha256")
    verifier_sha, helper_sha = manifest.get("verifier_sha256"), manifest.get("helper_sha256")
    need(manifest.get("schema") == INTAKE_SCHEMA and isinstance(entries, list) and len(entries) == 6
         and all(digest(value) for value in (core_sha, runner_sha, verifier_sha, helper_sha))
         and sha(SCORING_CORE) == core_sha and sha(SCORING_RUNNER) == runner_sha
         and sha(SCORING_VERIFIER) == verifier_sha and sha(HELPER) == helper_sha,
         "intake code identity differs")
    helper = _load_helper(helper_sha)
    need(sha(PREPARED) == PREPARED_SHA256, "prepared requests identity differs")
    prepared = read(PREPARED)
    source_rows, shard_ids = prepared.get("rows"), prepared.get("shards")
    need(isinstance(source_rows, list) and len(source_rows) == ROWS
         and isinstance(shard_ids, list) and len(shard_ids) == 6
         and sum(map(len, shard_ids)) == ROWS
         and len({row.get("id") for row in source_rows}) == ROWS
         and all(set(row) == {"id", "input", "component_id"} for row in source_rows),
         "prepared source schedule differs")
    pins = {}
    for path, value in ((intake, intake_sha256), (EVALUATOR, evaluator_sha256),
                        (SCORING_CORE, core_sha), (SCORING_RUNNER, runner_sha),
                        (SCORING_VERIFIER, verifier_sha), (HELPER, helper_sha),
                        (PREPARED, PREPARED_SHA256)):
        _merge_pin(pins, path, value)
    directories = []
    for shard, entry in enumerate(entries):
        directory, hashes = Path(entry.get("directory", "")), entry.get("hashes")
        need(entry.get("shard") == shard and directory.is_absolute() and directory.is_dir()
             and isinstance(hashes, dict) and set(hashes) == set(NAMES),
             "scoring shard inventory differs")
        need(not (directory / "failure.json").exists()
             and not (directory / "resource-watchdog-failure.json").exists(),
             "scoring shard failure artifact exists")
        for name, value in hashes.items():
            need(digest(value) and sha(directory / name) == value,
                 f"scoring shard {shard} hash differs: {name}")
            _merge_pin(pins, directory / name, value)
        report, closure, completion, proof, shard_pins = (read(directory / name) for name in
            ("report.json", "capture.complete.json", "completion.json",
             "independent-verification.json", "pins.json"))
        expected_ids = shard_ids[shard]
        need(report.get("schema") == "current-fit-recombination-scoring-shard-v1"
             and report.get("completed") is True and report.get("mode") == "capture"
             and report.get("shard") == shard and report.get("rows") == len(expected_ids)
             and report.get("ids") == expected_ids and report.get("row_counts") is not None
             and report.get("core_sha256") == core_sha and report.get("runner_sha256") == runner_sha
             and report.get("capture_sha256") == hashes["capture.jsonl"]
             and report.get("closure_sha256") == hashes["capture.complete.json"]
             and report.get("pins") == shard_pins and report.get("source_only") is True
             and report.get("fit_labels_opened") is False and report.get("dev_opened") is False
             and report.get("test_opened") is False and report.get("model_inference") is True
             and report.get("phonetic_model_inference") is False,
             "closed scoring report differs")
        need(closure.get("completed") is True and closure.get("mode") == "capture"
             and closure.get("shard") == shard and closure.get("rows") == len(expected_ids)
             and closure.get("ids") == expected_ids
             and closure.get("capture_sha256") == hashes["capture.jsonl"]
             and closure.get("runner_sha256") == runner_sha
             and closure.get("core_sha256") == core_sha and closure.get("labels_opened") is False,
             "closed scoring closure differs")
        need(completion.get("completed") is True
             and completion.get("report_sha256") == hashes["report.json"],
             "closed scoring completion differs")
        need(proof.get("schema") == "current-fit-recombination-scores-independent-verification-v1"
             and proof.get("verified") is True and proof.get("qualified_probe") is True
             and proof.get("mode") == "capture" and proof.get("shard") == shard
             and proof.get("rows") == len(expected_ids)
             and proof.get("report_sha256") == hashes["report.json"]
             and proof.get("capture_sha256") == hashes["capture.jsonl"]
             and proof.get("closure_sha256") == hashes["capture.complete.json"]
             and proof.get("runner_sha256") == runner_sha
             and proof.get("core_sha256") == core_sha
             and proof.get("verifier_sha256") == verifier_sha
             and proof.get("labels_opened") is False
             and proof.get("model_inference_replayed") is False,
             "closed scoring proof differs")
        counts = report["row_counts"]
        need(isinstance(counts, dict) and list(counts) == expected_ids
             and all(type(row.get("original")) is int and type(row.get("scored_novel")) is int
                     and type(row.get("unsupported_novel")) is int
                     and row.get("scored_novel") + row.get("unsupported_novel") == row.get("novel")
                     for row in counts.values()), "scoring row count closure differs")
        for path, value in shard_pins.items():
            need(Path(path).is_file() and sha(path) == value, "inherited runtime pin differs: " + path)
            _merge_pin(pins, path, value)
        directories.append(directory)
    for path, value in pins.items():
        need(Path(path).is_file() and sha(path) == value, "authenticated input pin differs: " + path)
    # Register fixed late inputs without opening either one before all six source-only
    # proofs above have authenticated successfully.
    _merge_pin(pins, FEATURES, FEATURES_SHA256)
    _merge_pin(pins, MODEL, MODEL_SHA256)
    return {"manifest": manifest, "helper": helper, "prepared": source_rows,
            "shard_ids": shard_ids, "directories": directories, "pins": dict(sorted(pins.items()))}


def validate_label(row, source):
    candidates, scores, features = (row.get("candidates"), row.get("base_scores"),
                                    row.get("features"))
    positives, production = row.get("positive_indices"), row.get("production_indices")
    ranked = row.get("ranked")
    need(row.get("id") == source.get("id") and row.get("input") == source.get("input")
         and row.get("component_id") == source.get("component_id")
         and type(ranked) is bool and type(row.get("training_eligible")) is bool
         and isinstance(row.get("expected"), list) and row["expected"]
         and all(isinstance(value, str) and value for value in row["expected"])
         and isinstance(row.get("public_candidates"), list)
         and all(isinstance(value, str) for value in row["public_candidates"])
         and isinstance(candidates, list) and isinstance(scores, list)
         and isinstance(features, list) and len(candidates) == len(scores) == len(features)
         and len(set(candidates)) == len(candidates)
         and all(isinstance(value, str) for value in candidates)
         and all(finite(value) for value in scores)
         and all(isinstance(vector, list) and len(vector) == 4
                 and all(finite(value) for value in vector) for vector in features)
         and isinstance(positives, list) and len(set(positives)) == len(positives)
         and all(type(index) is int and 0 <= index < len(candidates) for index in positives)
         and isinstance(production, list) and len(set(production)) == len(production)
         and all(type(index) is int and 0 <= index < len(candidates) for index in production)
         and (not ranked or (candidates and production)) and (ranked or not candidates),
         "late FIT label identity/domain differs")
    expected_positives = [index for index, text in enumerate(candidates)
                          if text in set(row["expected"])]
    need(positives == expected_positives
         and row["training_eligible"] is bool(ranked and positives),
         "late FIT positive indices/training eligibility differ")
    return {"id": row["id"], "input": row["input"],
            "component_id": row["component_id"], "expected": list(row["expected"])}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_outcome(outcome, expected_id, ranked):
    need(isinstance(outcome, dict) and outcome.get("id") == expected_id
         and outcome.get("request", {}).get("context") == ""
         and isinstance(outcome.get("counts"), dict)
         and set(outcome["counts"]) == {"original_candidates", "novel_candidates",
                                        "scored_novel", "unsupported_novel",
                                        "unsupported_reasons"}
         and all(type(outcome["counts"][name]) is int and outcome["counts"][name] >= 0
                 for name in ("original_candidates", "novel_candidates",
                              "scored_novel", "unsupported_novel"))
         and outcome["counts"]["scored_novel"] + outcome["counts"]["unsupported_novel"]
             == outcome["counts"]["novel_candidates"]
         and isinstance(outcome["counts"]["unsupported_reasons"], dict)
         and all(isinstance(reason, str) and reason and type(count) is int and count > 0
                 for reason, count in outcome["counts"]["unsupported_reasons"].items())
         and sum(outcome["counts"]["unsupported_reasons"].values())
             == outcome["counts"]["unsupported_novel"],
         "selection outcome identity/counts differ")
    arms, transitions = outcome.get("arms"), outcome.get("transitions")
    need(isinstance(arms, dict) and set(arms) == set(ARMS), "selection arms differ")
    for arm in arms.values():
        need(isinstance(arm, dict) and type(arm.get("first")) is bool
             and type(arm.get("top5")) is bool and type(arm.get("pool_coverage")) is bool
             and isinstance(arm.get("public"), list),
             "selection arm metrics differ")
    need(isinstance(transitions, dict) and set(transitions) == set(TRANSITIONS),
         "selection transitions differ")
    fields = {"corrections", "regressions", "top5_gains", "top5_losses"}
    for transition in transitions.values():
        need(isinstance(transition, dict) and set(transition) == fields
             and all(type(value) is int and value in (0, 1) for value in transition.values()),
             "selection transition metrics differ")
    need(type(ranked) is bool, "selection ranked state differs")
    outcome["ranked"] = ranked
    return outcome


def summarize(outcomes):
    need(outcomes and len(outcomes) == len({row["id"] for row in outcomes}),
         "selection outcome coverage differs")
    branches = {name: Counter(rows=0, first=0, top5=0, pool_covered=0) for name in ARMS}
    transitions = {name: Counter(corrections=0, regressions=0, top5_gains=0,
                                 top5_losses=0) for name in TRANSITIONS}
    candidates = Counter(rows=0, ranked_rows=0, no_pool_rows=0, original_candidates=0,
                         novel_candidates=0,
                         scored_novel=0, unsupported_novel=0)
    for row in outcomes:
        candidates["rows"] += 1; candidates["ranked_rows"] += row["ranked"]
        candidates["no_pool_rows"] += not row["ranked"]
        for name in ("original_candidates", "novel_candidates", "scored_novel",
                     "unsupported_novel"):
            candidates[name] += row["counts"][name]
        candidates.update({"unsupported_reason:" + reason: count
                           for reason, count in row["counts"]["unsupported_reasons"].items()})
        for name, arm in row["arms"].items():
            branches[name].update(rows=1, first=arm["first"], top5=arm["top5"],
                                  pool_covered=arm["pool_coverage"])
        for name, transition in row["transitions"].items(): transitions[name].update(transition)
    need(candidates["rows"] == candidates["ranked_rows"] + candidates["no_pool_rows"],
         "ranked/no-pool denominator differs")
    return {"branches": {name: dict(value) for name, value in branches.items()},
            "transitions": {name: dict(value) for name, value in transitions.items()},
            "candidate_counts": dict(candidates)}


def run(output, intake, intake_sha256, evaluator_sha256,
        *, watchdog_cls=AtomicResourceWatchdog):
    output = Path(output)
    need(output.is_absolute(), "output must be absolute")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); labels_opened = False
    try:
        with watchdog_cls(started, system_swap_bytes(), output / "resources.jsonl",
                          limits=LIMITS, interval_s=0.5) as guard:
            with guard.operation("authenticate_all_scoring_shards", update=False):
                auth = authenticate(intake, intake_sha256, evaluator_sha256)
                write(output / "pins.json", auth["pins"])
            with guard.operation("open_fit_labels_and_calibration_model", update=True):
                labels_opened = True
                need(sha(FEATURES) == FEATURES_SHA256 and sha(MODEL) == MODEL_SHA256,
                     "late FIT/model input identity differs")
                sources = {row["id"]: row for row in auth["prepared"]}
                labels, original_candidate_shas = {}, {}
                for row in jsonl(FEATURES):
                    identifier = row.get("id")
                    need(identifier in sources and identifier not in labels,
                         "FIT label ID coverage differs")
                    labels[identifier] = validate_label(row, sources[identifier])
                    original_candidate_shas[identifier] = canonical_sha([
                        {"text": text, "sentence_score": score, "features": vector}
                        for text, score, vector in zip(row["candidates"], row["base_scores"],
                                                       row["features"])])
                need(len(labels) == ROWS and set(labels) == set(sources),
                     "FIT label denominator differs")
                model = read(MODEL)
                need(model.get("schema") == "phonetic-score-scale-model-v1"
                     and model.get("arm") == "learned_scaled"
                     and isinstance(model.get("parameters"), list) and len(model["parameters"]) == 5
                     and isinstance(model.get("scale"), list) and len(model["scale"]) == 4
                     and all(finite(value) for value in model["parameters"] + model["scale"])
                     and model["parameters"][0] >= 0
                     and all(value > 0 for value in model["scale"]),
                     "deployed calibration model differs")
            captures = chain.from_iterable(jsonl(directory / "capture.jsonl")
                                           for directory in auth["directories"])
            outcomes, ids = [], []
            with (output / "outcomes.jsonl").open("x", encoding="utf-8") as stream:
                ordered_ids = [identifier for shard in auth["shard_ids"] for identifier in shard]
                for identifier, captured in zip_longest(ordered_ids, captures):
                    need(identifier is not None and captured is not None,
                         "score/label denominator differs")
                    label = labels[identifier]
                    with guard.operation("evaluate:" + identifier, update=True):
                        need(captured.get("id") == identifier == label["id"],
                             "score/label order differs")
                        scoring = captured.get("scoring")
                        need(isinstance(scoring, dict)
                             and scoring.get("request") == {"input": label["input"], "context": ""},
                             "scoring row/late label identity differs")
                        original_values = [{"text": row.get("text"),
                                            "sentence_score": row.get("sentence_score"),
                                            "features": row.get("features")}
                                           for row in scoring.get("original", [])]
                        need(canonical_sha(original_values) == original_candidate_shas[identifier],
                             "frozen original candidate values differ from FIT row")
                        outcome = validate_outcome(
                            auth["helper"].compare_row(scoring, label["expected"], model),
                            identifier, scoring.get("ranked"))
                        append(stream, outcome); outcomes.append(outcome); ids.append(identifier)
                stream.flush(); os.fsync(stream.fileno())
            need(ids == [identifier for shard in auth["shard_ids"] for identifier in shard]
                 and len(ids) == ROWS, "selection ID order/denominator differs")
            summary = summarize(outcomes)
            outcomes_sha = sha(output / "outcomes.jsonl")
            with guard.operation("write_closure", update=True):
                write(output / "outcomes.complete.json", {
                    "schema": "current-fit-recombination-selection-closure-v1",
                    "completed": True, "rows": ROWS, "ids": ids,
                    "outcomes_sha256": outcomes_sha, "fit_labels_opened": True,
                    "model_inference": False, "neural_training": False})
            closure_sha = sha(output / "outcomes.complete.json")
            with guard.operation("write_report", update=True):
                report = {"schema": "current-fit-recombination-selection-v1",
                          "completed": True, "rows": ROWS, "ids": ids,
                          "summary": summary, "outcomes_sha256": outcomes_sha,
                          "closure_sha256": closure_sha, "intake_sha256": intake_sha256,
                          "evaluator_sha256": evaluator_sha256,
                          "helper_sha256": auth["manifest"]["helper_sha256"],
                          "features_sha256": FEATURES_SHA256, "model_sha256": MODEL_SHA256,
                          "pins": auth["pins"], "limits": LIMITS,
                          "fit_labels_opened_after_all_six_score_proofs": True,
                          "source_only_scoring_authenticated": True,
                          "model_inference": False, "neural_training": False,
                          "dev_opened": False, "test_opened": False,
                          "production_changed": False}
                write(output / "report.json", report)
            with guard.operation("final_rehash", update=False):
                need(all(Path(path).is_file() and sha(path) == value
                         for path, value in auth["pins"].items()),
                     "authenticated input changed during late-label evaluation")
            terminal = guard.sample_and_append("after_final_rehash")
            with guard.operation("write_completion", update=True):
                write(output / "completion.json", {"completed": True,
                      "report_sha256": sha(output / "report.json"),
                      "outcomes_sha256": outcomes_sha, "closure_sha256": closure_sha,
                      "resource_after_final_rehash": terminal,
                      "all_inputs_rehashed_after_join": True})
            guard.sample_and_append("after_completion")
        return report
    except BaseException as error:
        failure = output / "failure.json"
        if not failure.exists():
            write(failure, {"completed": False, "fit_labels_opened": labels_opened,
                            "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--intake-sha256", required=True)
    parser.add_argument("--evaluator-sha256", required=True)
    args = parser.parse_args()
    need(args.intake.is_absolute(), "--intake must be absolute")
    result = run(args.output, args.intake, args.intake_sha256, args.evaluator_sha256)
    print(json.dumps({key: result[key] for key in ("completed", "rows", "summary")},
                     ensure_ascii=False))
