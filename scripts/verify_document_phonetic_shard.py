#!/usr/bin/env python3
"""Pure verifier for one closed source-only document phonetic shard.

The verifier opens no reference sentence or performance label and runs no
neural model.  It independently replays pre-calibration public assembly,
phonetic features, residual scores, calibrated ordering, resource boundaries,
and the terminal launch/report/closure chain.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

from scripts.verify_production_phonetic_choice_dev import (
    PRODUCTION_VERSION,
    finite,
    need,
    path_key,
    read,
    read_exact_jsonl,
    render_public,
    sha,
    verify_candidate,
    verify_resources,
    verify_response,
)


ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / ".omx/artifacts/document-sentence-shards-v1"
PLAN = ROOT / ".omx/plans/document-phonetic-choice-sharded-v1.md"
PRODUCER = ROOT / "scripts/capture_document_phonetic_shard.py"
RUNS = {
    0: ROOT / ".omx/artifacts/document-phonetic-choice-shard-0-v1",
    1: ROOT / ".omx/artifacts/document-phonetic-choice-shard-1-v1",
}
SCHEMA = "document-phonetic-shard-independent-verification-v1"
RUN_SCHEMA = "document-phonetic-shard-capture-v1"
BASELINE_KIND = "public_replay_of_actual_precalibration_paths"
EXPECTED_VERSION = "experimental-production-phonetic-choice-1-30e29bb96e4a"
EXPECTED_WEIGHTS = [
    0.09822412586490854,
    0.08018606995271488,
    -0.060145568052192455,
    -0.0019953930240048014,
]
LIMITS = {"whole_seconds": 1800.0, "update_seconds": 60.0,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
PARTITION_HASHES = {
    "manifest.json": "a09e13062f759ec8ac8d13b44344b63724e50f8d0302e4a945bfa13fdf797fc9",
    "shard-0.jsonl": "9c9e7f49f95d90f38d530cb0bc879eefe0be98c2fc9593e4aa24500129ffae48",
    "shard-1.jsonl": "3aeecc85a58186a4bbb9f0ac45edd52038488f31bc01b9749bb92aaddad04df5",
    "independent-verification.json":
        "f0eb4e1d4585b5c2d4873442186b896ad47f55c49324aa6c599f1cf3143376a3",
}
PLAN_SHA256 = "4b2c05e75e25c25b7f5f19679d9239f681f5e2d3fecdad16db133d0a543472b2"
UNRESOLVED = "__FILL_AFTER_CAPTURE_PRODUCER_FREEZE__"
PRODUCER_SHA256 = "412fb0bbdadab8931346ff529acbd04f6fb2d292d847d85b80c26e0b012bfbec"


def resolved_producer(value=PRODUCER_SHA256):
    need(isinstance(value, str) and len(value) == 64 and value != UNRESOLVED,
         "capture producer identity is not frozen")
    return value


def jsonl_strict(path, expected_rows):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            need(bool(line.strip()), f"blank JSONL row at {path}:{number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSONL at {path}:{number}") from error
            need(isinstance(value, dict), f"non-object JSONL at {path}:{number}")
            rows.append(value)
    need(len(rows) == expected_rows,
         f"JSONL row count differs: expected {expected_rows}, got {len(rows)}")
    return rows


def verify_baseline(source, baseline, proposed, base_paths):
    need(isinstance(baseline, dict) and set(baseline) == {
        "kind", "version", "candidates", "result", "sentence_ranking_applied"},
        "baseline schema differs")
    candidates = baseline.get("candidates")
    need(baseline.get("kind") == BASELINE_KIND
         and baseline.get("version") == PRODUCTION_VERSION
         and baseline.get("sentence_ranking_applied")
             == proposed.get("sentence_ranking_applied")
         and isinstance(candidates, list) and 1 <= len(candidates) <= 5
         and baseline.get("result") == candidates[0].get("text"),
         "baseline identity/public result differs")
    for candidate in candidates:
        verify_candidate(source, candidate)
    need(any(candidate["text"] == source for candidate in candidates),
         "baseline omitted raw source candidate")
    if proposed.get("sentence_ranking_applied") is True:
        need(isinstance(base_paths, list) and base_paths
             and render_public(source, base_paths) == candidates,
             "ranked baseline is not a replay of pre-calibration paths")
    else:
        choice = proposed.get("phonetic_choice")
        need(base_paths == [] and isinstance(choice, dict)
             and choice.get("rows") == [] and choice.get("score_records") == []
             and candidates == proposed.get("candidates"),
             "unranked baseline/calibration identity differs")


def verify_record(expected, record, weights=EXPECTED_WEIGHTS,
                  version=EXPECTED_VERSION):
    source = expected["input"]
    need(set(record) == {"id", "request", "baseline", "proposed", "base_paths"}
         and record.get("id") == expected["id"]
         and record.get("request") == {"input": source, "context": ""},
         "capture source/schema sequence differs")
    proposed = record["proposed"]
    paths = record["base_paths"]
    verify_baseline(source, record["baseline"], proposed, paths)
    proxy = dict(proposed, version=PRODUCTION_VERSION,
                 candidates=record["baseline"]["candidates"],
                 result=record["baseline"]["result"])
    frozen = {"parent": proxy, "parent_pool": [path[0] for path in paths]}
    census = verify_response(source, frozen, proposed, paths, weights, version)
    return census


def verify_launch(directory, shard, producer_sha):
    launch_path = directory.with_suffix(".launch.json")
    need(launch_path.is_file(), "launch receipt missing")
    launch = read(launch_path)
    expected_command = [
        str(ROOT / ".venv/bin/python"), str(PRODUCER.resolve()),
        "--output", str(directory), "--shard", str(shard),
    ]
    need(set(launch) == {
        "pid", "command", "output", "log", "runner_sha256", "launched_at"}
         and type(launch.get("pid")) is int and launch["pid"] > 0
         and launch.get("command") == expected_command
         and launch.get("output") == str(directory)
         and launch.get("log") == str(directory.with_suffix(".log"))
         and launch.get("runner_sha256") == producer_sha,
         "launch receipt identity differs")
    datetime.fromisoformat(launch["launched_at"])
    need(Path(launch["log"]).is_file(), "launch log missing")
    try:
        os.kill(launch["pid"], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("capture process is still present")
    return launch_path


def verify_pins(pins):
    need(isinstance(pins, dict) and pins, "runtime pins missing")
    for raw, digest in pins.items():
        need(isinstance(raw, str) and raw and isinstance(digest, str)
             and len(digest) == 64, "runtime pin schema differs")
        path = Path(raw) if Path(raw).is_absolute() else ROOT / raw
        need(path.is_file() and sha(path) == digest, "runtime pin changed: " + raw)


def verify(directory, shard):
    producer_sha = resolved_producer()
    need(type(shard) is int and shard in (0, 1), "invalid shard index")
    directory = Path(directory).resolve()
    need(directory == RUNS[shard].resolve(), "unexpected shard output directory")
    proof_path = directory / "independent-verification.json"
    need(directory.is_dir() and not proof_path.exists(), "run/proof state invalid")
    for name in ("failure.json", "resource-watchdog-failure.json"):
        need(not (directory / name).exists(), "failure artifact exists: " + name)
    required = ("report.json", "completion.json", "capture.jsonl",
                "capture.complete.json", "pins.json", "resources.jsonl")
    need(all((directory / name).is_file() for name in required),
         "terminal run files missing")
    need(PRODUCER.is_file() and sha(PRODUCER) == producer_sha,
         "capture producer identity differs")
    need(PLAN.is_file() and sha(PLAN) == PLAN_SHA256, "execution plan identity differs")
    need(all((SHARDS / name).is_file() and sha(SHARDS / name) == digest
             for name, digest in PARTITION_HASHES.items()),
         "frozen partition identity differs")
    partition_proof = read(SHARDS / "independent-verification.json")
    need(partition_proof.get("schema")
             == "document-sentence-shards-independent-verification-v1"
         and partition_proof.get("verified") is True
         and partition_proof.get("counts")
             == {"rows": 1046, "groups": 821, "shards": 2},
         "partition proof differs")

    launch_path = verify_launch(directory, shard, producer_sha)
    report = read(directory / "report.json")
    completion = read(directory / "completion.json")
    closure = read(directory / "capture.complete.json")
    pins = read(directory / "pins.json")
    capture_sha = sha(directory / "capture.jsonl")
    closure_sha = sha(directory / "capture.complete.json")
    report_sha = sha(directory / "report.json")
    completion_sha = sha(directory / "completion.json")
    source_rows = read_exact_jsonl(SHARDS / f"shard-{shard}.jsonl",
                                   (520, 526)[shard])
    row_count = len(source_rows)
    need(closure == {
        "schema": RUN_SCHEMA + "-closure", "shard": shard, "rows": row_count,
        "capture_sha256": capture_sha, "performance_labels_opened": False,
    }, "capture closure differs")
    need(set(completion) == {"completed", "report_sha256", "resource_after_report"}
         and completion.get("completed") is True
         and completion.get("report_sha256") == report_sha,
         "terminal completion differs")
    need(report.get("schema") == RUN_SCHEMA
         and report.get("completed") is True
         and report.get("shard") == shard and report.get("rows") == row_count
         and report.get("version") == EXPECTED_VERSION
         and report.get("weights") == EXPECTED_WEIGHTS
         and report.get("partition_hashes") == PARTITION_HASHES
         and report.get("pins") == pins and report.get("limits") == LIMITS
         and report.get("capture_sha256") == capture_sha
         and report.get("closure_sha256") == closure_sha
         and report.get("comparison") == BASELINE_KIND
         and report.get("parent_inference_run") is False
         and report.get("baseline_latency_measured") is False
         and report.get("performance_labels_opened") is False
         and report.get("accuracy_measured") is False
         and report.get("production_changed") is False,
         "terminal report identity/scope differs")
    verifier_sha = sha(Path(__file__))
    critical = {
        path_key(PRODUCER): producer_sha,
        path_key(Path(__file__)): verifier_sha,
        path_key(PLAN): PLAN_SHA256,
    }
    critical.update({path_key(SHARDS / name): digest
                     for name, digest in PARTITION_HASHES.items()})
    need(all(pins.get(raw) == digest for raw, digest in critical.items()),
         "critical runtime pins differ")
    verify_pins(pins)

    capture_rows = jsonl_strict(directory / "capture.jsonl", row_count)
    census = Counter()
    artifacts = None
    ranked = 0
    for expected, record in zip(source_rows, capture_rows):
        counts = verify_record(expected, record)
        census.update(counts)
        ranked += record["proposed"].get("sentence_ranking_applied") is True
        current = record["proposed"]["phonetic_choice"]["artifacts"]
        artifacts = current if artifacts is None else artifacts
        need(current == artifacts, "choice artifacts changed within shard")
        for raw, digest in current.items():
            path = Path(raw) if Path(raw).is_absolute() else ROOT / raw
            need(pins.get(path_key(path)) == digest,
                 "choice artifact is not runtime-pinned: " + raw)
    need(len({row["id"] for row in capture_rows}) == row_count,
         "capture identities are not unique")

    operations = ["authenticate_sources", "load_models",
                  *("predict:" + row["id"] for row in source_rows),
                  "close_capture", "write_report", "write_completion"]
    resources = verify_resources(
        read_exact_jsonl_unbounded(directory / "resources.jsonl"),
        operations,
    )
    need(completion["resource_after_report"] == resources["final"],
         "completion resource linkage differs")
    proof = {
        "schema": SCHEMA,
        "verified": True,
        "shard": shard,
        "rows": row_count,
        "producer_sha256": producer_sha,
        "verifier_sha256": verifier_sha,
        "plan_sha256": PLAN_SHA256,
        "launch_sha256": sha(launch_path),
        "report_sha256": report_sha,
        "capture_sha256": capture_sha,
        "closure_sha256": closure_sha,
        "completion_sha256": completion_sha,
        "partition_hashes": PARTITION_HASHES,
        "counts": {
            "rows": row_count,
            "ranked_rows": ranked,
            "unranked_rows": row_count - ranked,
            "paths": census["paths"],
            "unique_pair_scores_per_request_sum": census["pairs"],
            "supported_edge_occurrences": census["supported_edges"],
            "operations": len(operations),
        },
        "resources": resources,
        "contracts_verified": {
            "exact_source_shard_order_and_requests": True,
            "baseline_replayed_from_actual_precalibration_paths": True,
            "separate_parent_inference_or_latency_claimed": False,
            "independent_features_residual_order_and_public_replay": True,
            "all_runtime_pins_and_choice_artifacts": True,
            "atomic_resource_and_terminal_launch_contract": True,
            "model_tensor_or_neural_inference_run_by_verifier": False,
            "raw_references_or_performance_labels_deserialized": False,
        },
        "performance_labels_opened": False,
        "accuracy_measured": False,
        "limitations": [
            "The baseline is a public replay of actual pre-calibration paths, not a separately executed parent response.",
            "Recorded likelihood values are independently joined and replayed but are not recomputed by a model.",
            "This source-only shard proof contains no sentence-accuracy or deployment evidence.",
        ],
    }
    encoded = json.dumps(proof, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = proof_path.with_suffix(".json.tmp")
    need(not temporary.exists(), "temporary proof already exists")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, proof_path)
    return proof


def read_exact_jsonl_unbounded(path):
    """Strictly read a nonempty JSONL trace when row count is data-dependent."""
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            need(bool(line.strip()), f"blank JSONL row at {path}:{number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSONL at {path}:{number}") from error
            need(isinstance(row, dict), f"non-object JSONL at {path}:{number}")
            rows.append(row)
    need(rows, "resource trace empty")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=(0, 1), required=True)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.run, arguments.shard),
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
