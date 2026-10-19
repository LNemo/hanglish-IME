#!/usr/bin/env python3
"""Independently verify one closed full FIT recombination scoring shard."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / ".omx/artifacts"
INTAKE = ART / "current-fit-recombination-coverage-intake-v1.json"
PREPARED = ART / "current-phonetic-train-inputs-v1/requests.json"
PROBE = ART / "current-fit-recombination-scoring-probe-v3"
CORE = ROOT / "scripts/current_recombination_scoring.py"
HELPER = ROOT / "scripts/verify_current_fit_recombination_scoring_probe.py"
SOURCE_DIRS = tuple(ART / f"current-phonetic-train-shard-{index}-v1" for index in range(6))
INTAKE_SHA256 = "64456f9616f3615d24b82c9163edd092fc0a93fe503c64d5b59784115176dbec"
PREPARED_SHA256 = "97b416a805b9da8dc15d4ff9178b3a1db2a98f57f961a0e369c51bc771e1762d"
LIMITS = {"whole_seconds": 900.0, "update_seconds": 30.0,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
CENSUS_NAMES = ("report.json", "capture.jsonl", "capture.complete.json", "completion.json",
                "independent-verification.json", "pins.json", "resources.jsonl")
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


def digest(value):
    return isinstance(value, str) and DIGEST.fullmatch(value) is not None


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def load_helper(helper_sha256):
    """Authenticate the pure verifier helper before executing any of its code."""
    need(digest(helper_sha256) and HELPER.is_file() and sha(HELPER) == helper_sha256,
         "helper identity differs")
    spec = importlib.util.spec_from_file_location("_recombination_scoring_probe_verifier", HELPER)
    need(spec is not None and spec.loader is not None, "helper import specification differs")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _merge_pin(expected, path, value):
    path = str(Path(path).resolve())
    need(digest(value) and (path not in expected or expected[path] == value),
         "source pin disagreement: " + path)
    expected[path] = value


def authenticate_probe(report_sha256, proof_sha256, core_sha256):
    need(digest(report_sha256) and digest(proof_sha256)
         and sha(PROBE / "report.json") == report_sha256
         and sha(PROBE / "independent-verification.json") == proof_sha256,
         "probe hash identity differs")
    report = read(PROBE / "report.json")
    proof = read(PROBE / "independent-verification.json")
    closure = read(PROBE / "capture.complete.json")
    completion = read(PROBE / "completion.json")
    capture_sha = sha(PROBE / "capture.jsonl")
    need(not (PROBE / "failure.json").exists()
         and not (PROBE / "resource-watchdog-failure.json").exists(),
         "probe failure artifact exists")
    need(report.get("completed") is True and report.get("qualified") is True
         and report.get("mode") == "probe" and report.get("rows") == 16
         and report.get("core_sha256") == core_sha256
         and report.get("capture_sha256") == capture_sha
         and report.get("closure_sha256") == sha(PROBE / "capture.complete.json")
         and report.get("limits") == LIMITS
         and isinstance(report.get("projected_shard_seconds"), list)
         and len(report["projected_shard_seconds"]) == 6
         and all(finite(value) and value <= 900 for value in report["projected_shard_seconds"])
         and report.get("model_metadata") == {"max_length": 96, "batch_size": 8,
                                                "device": "cpu", "threads": 2,
                                                "interop_threads": 1}
         and report.get("source_only") is True and report.get("fit_labels_opened") is False
         and report.get("dev_opened") is False and report.get("test_opened") is False,
         "qualified probe report differs")
    need(proof.get("verified") is True and proof.get("qualified") is True
         and proof.get("rows") == 16 and proof.get("report_sha256") == report_sha256
         and proof.get("capture_sha256") == capture_sha
         and proof.get("closure_sha256") == sha(PROBE / "capture.complete.json")
         and proof.get("core_sha256") == core_sha256
         and proof.get("labels_opened") is False
         and proof.get("model_inference_replayed") is False,
         "qualified probe proof differs")
    need(closure.get("completed") is True and closure.get("mode") == "probe"
         and closure.get("rows") == 16 and closure.get("capture_sha256") == capture_sha
         and closure.get("core_sha256") == core_sha256 and closure.get("labels_opened") is False,
         "probe closure differs")
    need(completion.get("completed") is True and completion.get("qualified") is True
         and completion.get("report_sha256") == report_sha256,
         "probe completion differs")
    pins = report.get("pins")
    need(isinstance(pins, dict), "probe pins missing")
    for path, value in pins.items():
        need(Path(path).is_file() and sha(path) == value, "probe runtime pin differs: " + path)
    for name in ("report.json", "capture.jsonl", "capture.complete.json", "completion.json",
                 "independent-verification.json", "resources.jsonl"):
        pins[str((PROBE / name).resolve())] = sha(PROBE / name)
    return report, pins


def authenticate_sources(shard):
    """Authenticate all six frozen source/census inventories, then stream one shard."""
    need(type(shard) is int and 0 <= shard < 6, "shard differs")
    need(sha(INTAKE) == INTAKE_SHA256 and sha(PREPARED) == PREPARED_SHA256,
         "frozen intake/prepared requests differ")
    manifest, prepared = read(INTAKE), read(PREPARED)
    rows, shards, entries = prepared.get("rows"), prepared.get("shards"), manifest.get("shards")
    need(isinstance(rows, list) and len(rows) == 1642 and isinstance(shards, list)
         and len(shards) == 6 and sum(map(len, shards)) == 1642
         and len({item for group in shards for item in group}) == 1642,
         "prepared source schedule differs")
    need(manifest.get("schema") == "current-fit-recombination-coverage-intake-v1"
         and isinstance(entries, list) and len(entries) == 6,
         "census intake differs")
    by_id = {row.get("id"): row for row in rows}
    need(len(by_id) == 1642 and set(by_id) == {item for group in shards for item in group},
         "prepared IDs differ")
    expected_pins = {str(INTAKE.resolve()): INTAKE_SHA256,
                     str(PREPARED.resolve()): PREPARED_SHA256}
    census_dirs = []
    for index, entry in enumerate(entries):
        directory, hashes = Path(entry.get("directory", "")), entry.get("hashes")
        need(entry.get("shard") == index and directory.is_absolute() and directory.is_dir()
             and isinstance(hashes, dict) and set(hashes) == set(CENSUS_NAMES),
             "census shard inventory differs")
        need(not (directory / "failure.json").exists()
             and not (directory / "resource-watchdog-failure.json").exists(),
             "census failure artifact exists")
        for name, value in hashes.items():
            need(digest(value) and sha(directory / name) == value,
                 f"census shard {index} hash differs: {name}")
            _merge_pin(expected_pins, directory / name, value)
        report, proof, closure, completion, pins = (read(directory / name) for name in
            ("report.json", "independent-verification.json", "capture.complete.json",
             "completion.json", "pins.json"))
        need(report.get("completed") is True and report.get("qualified") is True
             and report.get("mode") == "capture" and report.get("shard") == index
             and report.get("ids") == shards[index] and report.get("rows") == len(shards[index])
             and report.get("capture_sha256") == hashes["capture.jsonl"]
             and report.get("closure_sha256") == hashes["capture.complete.json"]
             and report.get("pins") == pins and report.get("source_only") is True
             and report.get("labels_opened") is False and report.get("fit_feature_files_opened") is False,
             "closed census report differs")
        need(proof.get("verified") is True and proof.get("qualified") is True
             and proof.get("mode") == "capture" and proof.get("shard") == index
             and proof.get("rows") == len(shards[index])
             and proof.get("report_sha256") == hashes["report.json"]
             and proof.get("capture_sha256") == hashes["capture.jsonl"]
             and proof.get("closure_sha256") == hashes["capture.complete.json"],
             "closed census proof differs")
        need(closure.get("completed") is True and closure.get("mode") == "capture"
             and closure.get("shard") == index and closure.get("rows") == len(shards[index])
             and closure.get("capture_sha256") == hashes["capture.jsonl"]
             and completion.get("completed") is True
             and completion.get("report_sha256") == hashes["report.json"],
             "closed census closure/completion differs")
        for path, value in pins.items(): _merge_pin(expected_pins, path, value)
        census_dirs.append(directory)
    for path, value in expected_pins.items():
        need(Path(path).is_file() and sha(path) == value, "source runtime pin differs: " + path)

    wanted = shards[shard]
    def selected():
        originals = jsonl(SOURCE_DIRS[shard] / "capture.jsonl")
        census_rows = jsonl(census_dirs[shard] / "capture.jsonl")
        for position, (identifier, original, census) in enumerate(
                zip_longest(wanted, originals, census_rows)):
            need(identifier is not None and original is not None and census is not None
                 and original.get("id") == census.get("id") == identifier
                 and original.get("component_id") == census.get("component_id")
                 and by_id[identifier].get("component_id") == original.get("component_id")
                 and by_id[identifier].get("input") == original.get("request", {}).get("input"),
                 "exact source shard/order/index differs")
            yield original, census, position
    return wanted, selected(), expected_pins


def verify_launch(output, shard, runner_sha256, core_sha256,
                  probe_report_sha256, probe_proof_sha256):
    launch_path = output.with_suffix(".launch.json")
    launch = read(launch_path)
    actual_command = launch.get("command")
    need(isinstance(actual_command, list) and len(actual_command) == 12,
         "launch identity differs")
    command = [str(ROOT / ".venv/bin/python"), str(Path(actual_command[1])),
               "--shard", str(shard), "--runner-sha256", runner_sha256,
               "--core-sha256", core_sha256, "--probe-report-sha256", probe_report_sha256,
               "--probe-proof-sha256", probe_proof_sha256]
    runner_path = Path(command[1])
    need(actual_command == command and runner_path.is_absolute()
         and sha(runner_path) == runner_sha256
         and launch.get("runner_sha256") == runner_sha256
         and launch.get("core_sha256") == core_sha256
         and launch.get("probe_report_sha256") == probe_report_sha256
         and launch.get("probe_proof_sha256") == probe_proof_sha256
         and launch.get("output") == str(output), "launch identity differs")
    need(type(launch.get("pid")) is int and launch["pid"] > 0, "launch PID differs")
    try:
        os.kill(launch["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("runner PID is still present")
    return launch_path, runner_path


def verify(output, runner_sha256, core_sha256, probe_report_sha256,
           probe_proof_sha256, helper_sha256):
    output = Path(output)
    need(output.is_absolute(), "output must be absolute")
    output = output.resolve()
    proof_path = output / "independent-verification.json"
    need(output.is_dir() and not proof_path.exists(), "run/proof state invalid")
    need(all(digest(value) for value in (runner_sha256, core_sha256, probe_report_sha256,
                                         probe_proof_sha256, helper_sha256)),
         "CLI identity digest differs")
    need(sha(CORE) == core_sha256, "core identity differs")
    helper = load_helper(helper_sha256)
    need(not (output / "failure.json").exists()
         and not (output / "resource-watchdog-failure.json").exists(),
         "failure artifact exists")
    report, closure, completion, pins = (read(output / name) for name in
        ("report.json", "capture.complete.json", "completion.json", "pins.json"))
    shard = report.get("shard")
    need(type(shard) is int and 0 <= shard < 6, "report shard differs")
    launch_path, runner_path = verify_launch(output, shard, runner_sha256, core_sha256,
                                             probe_report_sha256, probe_proof_sha256)
    probe_report, probe_pins = authenticate_probe(probe_report_sha256, probe_proof_sha256,
                                                   core_sha256)
    wanted, sources, source_pins = authenticate_sources(shard)
    capture_sha = sha(output / "capture.jsonl")
    closure_sha = sha(output / "capture.complete.json")
    report_sha = sha(output / "report.json")
    need(report.get("schema") == "current-fit-recombination-scoring-shard-v1"
         and report.get("completed") is True and report.get("mode") == "capture"
         and report.get("rows") == len(wanted) and report.get("ids") == wanted
         and report.get("runner_sha256") == runner_sha256
         and report.get("core_sha256") == core_sha256
         and report.get("capture_sha256") == capture_sha
         and report.get("capture_output_bytes") == (output / "capture.jsonl").stat().st_size
         and report.get("closure_sha256") == closure_sha and report.get("pins") == pins
         and report.get("limits") == LIMITS
         and report.get("probe_report_sha256") == probe_report_sha256
         and report.get("probe_proof_sha256") == probe_proof_sha256
         and report.get("model_metadata") == {"max_length": 96, "batch_size": 8,
                                                "device": "cpu", "threads": 2,
                                                "interop_threads": 1}
         and report.get("source_only") is True and report.get("fit_labels_opened") is False
         and report.get("dev_opened") is False and report.get("test_opened") is False
         and report.get("model_inference") is True
         and report.get("phonetic_model_inference") is False
         and report.get("neural_training") is False and report.get("production_changed") is False,
         "terminal report differs")
    need(closure == {"schema": "current-fit-recombination-scoring-closure-v1",
                     "mode": "capture", "shard": shard, "completed": True,
                     "rows": len(wanted), "ids": wanted, "capture_sha256": capture_sha,
                     "runner_sha256": runner_sha256, "core_sha256": core_sha256,
                     "labels_opened": False, "model_inference": True,
                     "phonetic_model_inference": False}, "capture closure differs")
    need(completion.get("completed") is True and completion.get("report_sha256") == report_sha,
         "completion differs")
    required_pins = dict(source_pins)
    required_pins.update(probe_pins)
    required_pins[str(runner_path.resolve())] = runner_sha256
    required_pins[str(CORE.resolve())] = core_sha256
    for path, value in required_pins.items():
        need(pins.get(path) == value, "required pin differs: " + path)
    for path, value in pins.items():
        need(Path(path).is_file() and sha(path) == value, "runtime pin differs: " + path)

    counts, observed = {}, []
    capture_rows = jsonl(output / "capture.jsonl")
    for expected, captured in zip_longest(sources, capture_rows):
        need(expected is not None and captured is not None, "scoring denominator differs")
        original, census, position = expected
        identifier = wanted[position]
        need(captured.get("id") == identifier and captured.get("source_shard") == shard
             and captured.get("source_row_index") == position,
             "exact scoring shard/order/index differs")
        summary = helper.verify_scoring_row(original, census, captured,
                                            original_parity_required=False)
        need(type(summary.get("novel")) is int and type(summary.get("scored_novel")) is int
             and type(summary.get("unsupported_novel")) is int
             and summary["scored_novel"] + summary["unsupported_novel"] == summary["novel"],
             "missing novel score status")
        counts[identifier] = summary
        observed.append(identifier)
    need(observed == wanted and report.get("row_counts") == counts, "row count summary differs")
    timings = report.get("timing_seconds")
    need(isinstance(timings, dict) and set(timings) == set(wanted)
         and all(finite(value) and value >= 0 for value in timings.values()),
         "row timing summary differs")
    operations = ["authenticate_sources", "load_frozen_sentence_ranker"]
    operations += ["score:" + identifier for identifier in wanted]
    operations += ["write_closure", "write_report", "final_rehash", "write_completion"]
    resources = helper.verify_resources(list(jsonl(output / "resources.jsonl")), operations)
    need(completion.get("resource_after_final_rehash") == resources["after_final_rehash"],
         "completion resource binding differs")
    proof = {"schema": "current-fit-recombination-scores-independent-verification-v1",
             "verified": True, "qualified_probe": True, "mode": "capture", "shard": shard,
             "rows": len(wanted), "report_sha256": report_sha, "capture_sha256": capture_sha,
             "closure_sha256": closure_sha, "launch_sha256": sha(launch_path),
             "runner_sha256": runner_sha256, "core_sha256": core_sha256,
             "probe_report_sha256": probe_report_sha256,
             "probe_proof_sha256": probe_proof_sha256,
             "helper_sha256": helper_sha256, "verifier_sha256": sha(__file__),
             "counts": counts, "resources": resources, "source_only": True,
             "labels_opened": False, "model_loaded": False,
             "model_inference_replayed": False,
             "scope": "Authenticated all frozen source pins and the qualified probe; verified exact shard IDs/order/index, independently recomputed original and novel phonetic features, cached original scores, recorded novel score coverage, closure, launch, completion, and resources.",
             "limitations": "Recorded novel sentence-ranker logits and tokenizer can_score decisions were not independently replayed; this verifier loads no neural model and makes no original-score replay claim."}
    with proof_path.open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--core-sha256", required=True)
    parser.add_argument("--probe-report-sha256", required=True)
    parser.add_argument("--probe-proof-sha256", required=True)
    parser.add_argument("--helper-sha256", required=True)
    args = parser.parse_args()
    result = verify(args.output, args.runner_sha256, args.core_sha256,
                    args.probe_report_sha256, args.probe_proof_sha256, args.helper_sha256)
    print(json.dumps({key: result[key] for key in ("verified", "shard", "rows")},
                     ensure_ascii=False))
