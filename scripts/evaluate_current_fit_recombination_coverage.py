#!/usr/bin/env python3
"""Evaluate six closed FIT recombination censuses after source authentication.

The intake manifest supplies the concrete hashes of the completed census runs.
Every source-only artifact is authenticated before this program opens FIT feature
rows or the already-published FIT-only reachability audit.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import chain, zip_longest
import json
import os
from pathlib import Path
import re
import time

from scripts.current_fit_recombination_coverage import compare_row, summarize
from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = Path(__file__).resolve()
PREPARED = ROOT / ".omx/artifacts/current-phonetic-train-inputs-v1/requests.json"
FEATURES = ROOT / ".omx/artifacts/current-phonetic-train-features-v1/rows.jsonl"
REACHABILITY = ROOT / ".omx/artifacts/current-fit-edge-reachability-v1.json"
PRODUCER = ROOT / "scripts/current_fit_recombination_census.py"
BASE_VERIFIER = ROOT / "scripts/verify_current_fit_recombination_census.py"
PROBE_VERIFIER = ROOT / "scripts/verify_current_fit_recombination_census_v2.py"
VERIFIER = ROOT / "scripts/verify_current_fit_recombination_census_v3.py"
HELPER = ROOT / "scripts/current_fit_recombination_coverage.py"
RESOURCE_HELPER = ROOT / "scripts/source_span_resources.py"

ROWS = 1642
SHARDS = 6
PRODUCER_SHA256 = "5247b6b9d9c48edb66a43cacc7e3f370f942fdeb3becd00bfa07f5a648878135"
BASE_VERIFIER_SHA256 = "93772051d01bf9213cc7bc540c4c86c1428f2910c66017282e0f1d9b0afdf2d9"
PROBE_VERIFIER_SHA256 = "47c3921ad01779fc816c6e4b2db35ff8b1b49b49dd94efcd9efc0653613b5ba5"
PREPARED_SHA256 = "97b416a805b9da8dc15d4ff9178b3a1db2a98f57f961a0e369c51bc771e1762d"
FEATURES_SHA256 = "4c13063656d9a108316b9ce1ceec394a61e2c45c2ba86e7a8c8361f7480d6cb4"
REACHABILITY_SHA256 = "53e5691bda9a9d3063b2c5fe4c402689161e308677b2a37e9351bd7294c947e4"
HELPER_SHA256 = "89d4df29c1795a6cabc5d9dce4c63fc1adf31ea2dae07efd8b3828ce9f42021a"
RESOURCE_HELPER_SHA256 = "4bdf5f598677d1d5c94bcca39545015e7ae12370b0a91482aca1ba42e11a75bd"
LIMITS = {"whole_seconds": 300.0, "update_seconds": 30.0,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
NAMES = ("report.json", "capture.jsonl", "capture.complete.json",
         "completion.json", "independent-verification.json", "pins.json",
         "resources.jsonl")


def need(value, message):
    if not value:
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
            need(line.strip(), f"blank JSONL line at {path}:{number}")
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
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def append_jsonl(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                            allow_nan=False) + "\n")


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _authenticate_run(entry, *, mode, shard, producer_sha, verifier_sha,
                      probe_hashes=None):
    directory = Path(entry.get("directory", ""))
    need(directory.is_absolute() and directory.is_dir(), "census directory differs")
    need(not (directory / "failure.json").exists()
         and not (directory / "resource-watchdog-failure.json").exists(),
         "census has failure evidence")
    hashes = entry.get("hashes")
    need(isinstance(hashes, dict) and set(hashes) == set(NAMES),
         "census hash inventory differs")
    actual = {}
    for name in NAMES:
        expected = hashes.get(name)
        path = directory / name
        need(_digest(expected) and path.is_file() and sha256_file(path) == expected,
             f"census {name} hash differs")
        actual[name] = expected
    report, closure, completion, proof = (read_json(directory / name) for name in
                                           ("report.json", "capture.complete.json",
                                            "completion.json", "independent-verification.json"))
    expected_rows = 16 if mode == "probe" else len(report.get("ids", []))
    need(report.get("schema") == "current-fit-recombination-census-v1"
         and report.get("completed") is True and report.get("qualified") is True
         and report.get("mode") == mode and report.get("shard") == shard
         and report.get("producer_sha256") == producer_sha
         and report.get("capture_sha256") == actual["capture.jsonl"]
         and report.get("closure_sha256") == actual["capture.complete.json"]
         and report.get("rows") == expected_rows
         and report.get("labels_opened") is False
         and report.get("fit_feature_files_opened") is False,
         "census report contract differs")
    need(closure.get("completed") is True and closure.get("mode") == mode
         and closure.get("shard") == shard and closure.get("rows") == expected_rows
         and closure.get("capture_sha256") == actual["capture.jsonl"]
         and closure.get("labels_opened") is False,
         "census closure differs")
    need(completion.get("completed") is True
         and completion.get("report_sha256") == actual["report.json"],
         "census completion differs")
    need(proof.get("schema") == "current-fit-recombination-census-independent-verification-v1"
         and proof.get("verified") is True and proof.get("qualified") is True
         and proof.get("mode") == mode and proof.get("shard") == shard
         and proof.get("rows") == expected_rows
         and proof.get("producer_sha256") == producer_sha
         and proof.get("verifier_sha256") == verifier_sha
         and proof.get("report_sha256") == actual["report.json"]
         and proof.get("capture_sha256") == actual["capture.jsonl"]
         and proof.get("closure_sha256") == actual["capture.complete.json"]
         and proof.get("labels_opened") is False,
         "independent census proof differs")
    if probe_hashes is not None:
        binding = report.get("probe_binding")
        need(isinstance(binding, dict)
             and all(binding.get(key) == probe_hashes[name]
                     for name, key in (("report.json", "report_sha256"),
                                       ("capture.jsonl", "capture_sha256"),
                                       ("capture.complete.json", "closure_sha256"),
                                       ("completion.json", "completion_sha256"),
                                       ("independent-verification.json", "proof_sha256"))),
             "capture probe binding differs")
    return directory, report, {str((directory / name).resolve()): digest
                               for name, digest in actual.items()}


def authenticate_manifest(path):
    path = Path(path).resolve()
    need(path.is_absolute() and path.is_file(), "intake manifest must be an existing absolute path")
    manifest = read_json(path)
    producer_sha = manifest.get("producer_sha256")
    need(manifest.get("schema") == "current-fit-recombination-coverage-intake-v1"
         and producer_sha == PRODUCER_SHA256
         and sha256_file(PRODUCER) == producer_sha,
         "intake producer differs")
    verifier_sha = manifest.get("verifier_sha256")
    need(_digest(verifier_sha) and VERIFIER.is_file()
         and sha256_file(VERIFIER) == verifier_sha
         and BASE_VERIFIER.is_file()
         and sha256_file(BASE_VERIFIER) == BASE_VERIFIER_SHA256
         and PROBE_VERIFIER.is_file()
         and sha256_file(PROBE_VERIFIER) == PROBE_VERIFIER_SHA256,
         "census verifier differs")
    for fixed, digest in ((PREPARED, PREPARED_SHA256), (HELPER, HELPER_SHA256),
                          (RESOURCE_HELPER, RESOURCE_HELPER_SHA256)):
        need(fixed.is_file() and sha256_file(fixed) == digest, "fixed input differs: " + str(fixed))
    probe_entry = manifest.get("probe")
    need(isinstance(probe_entry, dict), "probe intake missing")
    _, probe_report, probe_pins = _authenticate_run(
        probe_entry, mode="probe", shard=None, producer_sha=producer_sha,
        verifier_sha=PROBE_VERIFIER_SHA256)
    probe_hashes = probe_entry["hashes"]
    shard_entries = manifest.get("shards")
    need(isinstance(shard_entries, list) and len(shard_entries) == SHARDS,
         "six census shards required")
    directories, reports, pins = [], [], dict(probe_pins)
    for shard, entry in enumerate(shard_entries):
        need(isinstance(entry, dict) and entry.get("shard") == shard,
             "census shard order differs")
        directory, report, run_pins = _authenticate_run(
            entry, mode="capture", shard=shard, producer_sha=producer_sha,
            verifier_sha=verifier_sha, probe_hashes=probe_hashes)
        directories.append(directory); reports.append(report); pins.update(run_pins)
    prepared = read_json(PREPARED)
    rows, shard_ids = prepared.get("rows"), prepared.get("shards")
    need(isinstance(rows, list) and len(rows) == ROWS
         and isinstance(shard_ids, list) and len(shard_ids) == SHARDS
         and len({row.get("id") for row in rows}) == ROWS
         and all(set(row) == {"id", "input", "component_id"} for row in rows)
         and [report.get("ids") for report in reports] == shard_ids
         and sum(map(len, shard_ids)) == ROWS,
         "prepared FIT IDs/components differ from census reports")
    prepared_by_id = {row["id"]: row for row in rows}
    for shard, directory in enumerate(directories):
        expected_rows = (prepared_by_id[identifier] for identifier in shard_ids[shard])
        for expected, captured in zip_longest(expected_rows,
                                               iter_jsonl(directory / "capture.jsonl")):
            need(expected is not None and captured is not None
                 and captured.get("id") == expected["id"]
                 and captured.get("component_id") == expected["component_id"]
                 and captured.get("request") == {"input": expected["input"], "context": ""}
                 and captured.get("source_reference", {}).get("source_shard") == shard,
                 "prepared FIT source identity differs from census capture")
    pins.update({str(path): sha256_file(path), str(PREPARED.resolve()): PREPARED_SHA256,
                 str(EVALUATOR): sha256_file(EVALUATOR),
                 str(PRODUCER.resolve()): producer_sha,
                 str(VERIFIER.resolve()): verifier_sha,
                 str(BASE_VERIFIER.resolve()): BASE_VERIFIER_SHA256,
                 str(PROBE_VERIFIER.resolve()): PROBE_VERIFIER_SHA256,
                 str(HELPER.resolve()): HELPER_SHA256,
                 str(RESOURCE_HELPER.resolve()): RESOURCE_HELPER_SHA256})
    return {"manifest": manifest, "directories": directories, "reports": reports,
            "prepared": rows, "shard_ids": shard_ids, "pins": dict(sorted(pins.items())),
            "probe_report": probe_report}


def evaluate_rows(captures, features, reachability, output_stream):
    audit_rows = reachability.get("rows")
    need(reachability.get("schema") == "current-fit-edge-reachability-v1"
         and reachability.get("features_sha256") == FEATURES_SHA256
         and reachability.get("counts") == {"missing_gold_rows": 70,
                                             "reachable": 38, "unreachable": 32}
         and isinstance(audit_rows, list) and len(audit_rows) == 70,
         "FIT reachability audit differs")
    audit = {row.get("id"): row.get("reachable") for row in audit_rows}
    need(len(audit) == 70 and all(type(value) is bool for value in audit.values()),
         "FIT reachability rows differ")
    compared, ids = [], []
    for capture, feature in zip_longest(captures, features):
        need(capture is not None and feature is not None, "census/feature denominator differs")
        row = compare_row(capture, feature)
        append_jsonl(output_stream, row); compared.append(row); ids.append(row["id"])
    summary = summarize(compared)
    need(summary["rows"] == ROWS and summary["unranked"] == 13
         and len(ids) == len(set(ids)), "FIT coverage denominator differs")
    missing = [row for row in compared if row["ranked"] and not row["old_pool_covered"]]
    need(len(missing) == 70 and {row["id"] for row in missing} == set(audit),
         "reachability cohort differs from old-pool misses")
    newly = {row["id"] for row in missing if row["newly_covered"]}
    reachable = {identifier for identifier, value in audit.items() if value}
    need(newly <= reachable, "generated accepted output contradicts reachability audit")
    summary.update({"missing_old_pool_rows": len(missing),
                    "audit_reachable": len(reachable),
                    "audit_unreachable": len(audit) - len(reachable),
                    "newly_covered_reachable": len(newly),
                    "reachable_but_not_generated": len(reachable - newly),
                    "no_pool_rows": summary["unranked"]})
    return summary, ids


def run(output, manifest, *, label_reader=iter_jsonl, json_reader=read_json,
        watchdog_cls=AtomicResourceWatchdog):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        with watchdog_cls(started, system_swap_bytes(), output / "resources.jsonl",
                          limits=LIMITS, interval_s=0.5) as guard:
            with guard.operation("authenticate_closed_censuses", update=False):
                auth = authenticate_manifest(manifest)
                write_json(output / "pins.json", auth["pins"])
            # FIT labels are first touched only after every source closure/proof passed.
            with guard.operation("open_fit_labels_and_evaluate", update=False):
                need(sha256_file(FEATURES) == FEATURES_SHA256, "FIT feature rows differ")
                need(sha256_file(REACHABILITY) == REACHABILITY_SHA256,
                     "FIT reachability audit hash differs")
                capture_rows = chain.from_iterable(label_reader(
                    directory / "capture.jsonl") for directory in auth["directories"])
                with (output / "rows.jsonl").open("x", encoding="utf-8") as stream:
                    summary, ids = evaluate_rows(capture_rows, label_reader(FEATURES),
                                                 json_reader(REACHABILITY), stream)
                    stream.flush(); os.fsync(stream.fileno())
                need(ids == [identifier for shard in auth["shard_ids"] for identifier in shard],
                     "FIT feature rows differ from authenticated census order")
            rows_sha = sha256_file(output / "rows.jsonl")
            with guard.operation("write_report", update=True):
                report = {"schema": "current-fit-recombination-coverage-v1",
                          "completed": True, "rows": ROWS, "ids": ids,
                          "counts": summary, "rows_sha256": rows_sha,
                          "manifest_sha256": sha256_file(Path(manifest)),
                          "features_sha256": FEATURES_SHA256,
                          "reachability_sha256": REACHABILITY_SHA256,
                          "producer_sha256": PRODUCER_SHA256,
                          "evaluator_sha256": sha256_file(EVALUATOR),
                          "source_pins": auth["pins"], "limits": LIMITS,
                          "fit_labels_opened_after_all_census_closures": True,
                          "candidate_scoring": False, "model_loaded": False,
                          "model_inference": False, "dev_opened": False,
                          "test_opened": False, "production_changed": False,
                          "reachability_limit": "Capped search misses are inconclusive; reachable-but-not-generated is reported separately and no capped miss is called unreachable."}
                write_json(output / "report.json", report)
            with guard.operation("final_rehash", update=False):
                need(all(Path(path).is_file() and sha256_file(path) == digest
                         for path, digest in auth["pins"].items()),
                     "authenticated source input changed during FIT join")
                need(sha256_file(FEATURES) == FEATURES_SHA256
                     and sha256_file(REACHABILITY) == REACHABILITY_SHA256,
                     "FIT input changed during join")
            terminal = guard.sample_and_append("after_final_rehash")
            with guard.operation("write_completion", update=True):
                write_json(output / "completion.json", {
                    "completed": True, "report_sha256": sha256_file(output / "report.json"),
                    "rows_sha256": rows_sha, "resource_after_final_rehash": terminal,
                    "all_inputs_rehashed_after_join": True,
                    "features_sha256": FEATURES_SHA256,
                    "reachability_sha256": REACHABILITY_SHA256})
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    need(args.output.is_absolute() and args.manifest.is_absolute(),
         "--output and --manifest must be absolute")
    result = run(args.output, args.manifest)
    print(json.dumps({key: result[key] for key in ("completed", "rows", "counts")},
                     ensure_ascii=False))
