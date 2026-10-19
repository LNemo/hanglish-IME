#!/usr/bin/env python3
"""Pure verifier for the current-phonetic TRAIN input and capture artifacts.

The verifier never imports the producer, Torch, a converter, or a neural model.
It authenticates the frozen FIT source domain, independently reconstructs the
partition and probes, and replays the recorded zero-weight feature/ranking
contract using hash-bound pure audit helpers.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re

from scripts.verify_production_phonetic_choice_dev import (
    jsonl,
    need,
    read,
    render_public,
    sha,
    verify_candidate,
    verify_choice_row,
    verify_resources,
    verify_score_records,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".omx/artifacts/source-correspondence-fit-v2"
PREPARED = ROOT / ".omx/artifacts/current-phonetic-train-inputs-v1"
PROBE = ROOT / ".omx/artifacts/current-phonetic-train-probe-v1"
PRODUCER = ROOT / "scripts/current_phonetic_train_pool.py"
PLAN = ROOT / ".omx/plans/current-phonetic-train-pool-v1.md"
PURE_HELPER = ROOT / "scripts/verify_production_phonetic_choice_dev.py"
OLD_RUNTIME = ROOT / ".omx/artifacts/phonetic-choice-runtime-dev-v1"
OLD_CHECKPOINT = ".omx/artifacts/phonetic-choice-calibration-training-v1/weights.json"
ROWS = 1642
SHARDS = 6
PROBES = 16
PARENT_VERSION = "experimental-phonetic-word-union-1-84073edd0b9d"
LIMITS = {"whole_seconds": 1800.0, "update_seconds": 60.0,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
FROZEN = {
    SOURCE / "fit-paths.jsonl": "f1cd66da23cacc6025e8aa2203944f664f0d74d916a0bde135cac65ef56afd8c",
    SOURCE / "report.json": "76672ccc2e56269a0b2c40701b8b10f9dc333c3d5ebeca274add7b305d3d65c0",
    SOURCE / "independent-verification.json": "449a8f286a60cc4a7c542612054c4ca762378caedfb0cbc9ca350801030f2de2",
    ROOT / ".omx/artifacts/observed-phonetic-augmented-training-v1/phonetic_transducer.pt":
        "f06b0a7cd6c5ccbdfaa376cc29f59e90b3fcf0ae3a157fd2c4f7efaa73b4180d",
    PURE_HELPER: "366d463af2fefc24f7bbf69c6aa52f46e52467335021c442a14eb1a38328003c",
    OLD_RUNTIME / "report.json": "a0072c79116da9acdbff83b373dccbb8c3ef8c26ac5164f575490941808d4473",
    OLD_RUNTIME / "capture.jsonl": "e0447efa305e1d4f61052866b1ee121a62b400f3adf163dfdeac1b0bb4734bf5",
    OLD_RUNTIME / "independent-verification.json":
        "f2e2eca118ef05c1397304d89ccea99e1d9ef28841fe109fcd04446d0928e6c4",
}
OLD_FROZEN = {
    OLD_RUNTIME / "report.json": "a0072c79116da9acdbff83b373dccbb8c3ef8c26ac5164f575490941808d4473",
    OLD_RUNTIME / "capture.jsonl": "e0447efa305e1d4f61052866b1ee121a62b400f3adf163dfdeac1b0bb4734bf5",
    OLD_RUNTIME / "independent-verification.json":
        "f2e2eca118ef05c1397304d89ccea99e1d9ef28841fe109fcd04446d0928e6c4",
    PURE_HELPER: "366d463af2fefc24f7bbf69c6aa52f46e52467335021c442a14eb1a38328003c",
}
HANGUL = re.compile(r"[\u3130-\u318f\uac00-\ud7a3\u1100-\u11ff]")


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def path_key(path):
    return str(Path(path).resolve())


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def write_proof(path, proof):
    path = Path(path)
    need(not path.exists(), "independent proof already exists")
    with path.open("xb") as stream:
        stream.write(canonical(proof))
        stream.flush()
        os.fsync(stream.fileno())


def source_rows(records):
    result, seen = [], set()
    for record in records:
        source = record.get("input")
        identifier = record.get("id")
        component = record.get("component_id")
        need(isinstance(source, str) and source and HANGUL.search(source) is None
             and isinstance(identifier, str) and identifier.startswith("s3-train-")
             and identifier not in seen
             and isinstance(component, str) and component,
             "FIT source domain differs")
        seen.add(identifier)
        result.append({"id": identifier, "input": source,
                       "component_id": component})
    return result


def probe_groups(rows):
    ordered = sorted(rows, key=lambda row: (len(row["input"]), row["id"]))
    small, extra = divmod(len(ordered), PROBES)
    result, offset = [], 0
    for index in range(PROBES):
        count = small + int(index < extra)
        group = ordered[offset:offset + count]
        need(group, "empty probe group")
        result.append({"ids": [row["id"] for row in group],
                       "probe_id": group[-1]["id"], "count": count,
                       "maximum_source_length": len(group[-1]["input"])})
        offset += count
    need(offset == len(rows), "probe partition differs")
    return result


def partition(rows, count=SHARDS):
    need(type(count) is int and count > 0, "invalid shard count")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["component_id"]].append(row)
    loads, assignment = [0] * count, {}
    ordered = sorted(grouped,
                     key=lambda component: (-sum(len(row["input"])
                                                   for row in grouped[component]),
                                            component))
    for component in ordered:
        index = min(range(count), key=lambda item: (loads[item], item))
        assignment[component] = index
        loads[index] += sum(len(row["input"]) for row in grouped[component])
    result = [[row["id"] for row in rows
               if assignment[row["component_id"]] == index]
              for index in range(count)]
    need(sorted(identifier for shard in result for identifier in shard)
         == sorted(row["id"] for row in rows), "shard row coverage differs")
    return result


def authenticate_frozen():
    need(all(path.is_file() and sha(path) == digest
             for path, digest in {**FROZEN, **OLD_FROZEN}.items()),
         "frozen source/helper/runtime differs")
    source_proof = read(SOURCE / "independent-verification.json")
    need(source_proof.get("verified") is True
         and source_proof.get("counts", {}).get("rows") == ROWS
         and source_proof.get("report_sha256") == sha(SOURCE / "report.json")
         and source_proof.get("capture_sha256") == sha(SOURCE / "fit-paths.jsonl"),
         "frozen FIT proof differs")


def frozen_sources():
    authenticate_frozen()
    rows = source_rows(jsonl(SOURCE / "fit-paths.jsonl"))
    need(len(rows) == ROWS, "FIT row count differs")
    return rows


def expected_artifacts():
    authenticate_frozen()
    first = next(jsonl(OLD_RUNTIME / "capture.jsonl"))
    response = first.get("proposed")
    choice = response.get("phonetic_choice", {}) if isinstance(response, dict) else {}
    old_models = response.get("model_artifacts") if isinstance(response, dict) else None
    old_choice = choice.get("artifacts")
    need(isinstance(old_models, dict) and len(old_models) == 33
         and isinstance(old_choice, dict) and len(old_choice) == 4
         and old_models.get(OLD_CHECKPOINT) == old_choice.get(OLD_CHECKPOINT),
         "old runtime artifact oracle differs")
    models = {key: value for key, value in old_models.items()
              if key != OLD_CHECKPOINT}
    choice_artifacts = {key: value for key, value in old_choice.items()
                        if key != OLD_CHECKPOINT}
    need(len(models) == 32 and len(choice_artifacts) == 3
         and set(choice_artifacts) <= set(models),
         "zero override artifact set differs")
    return models, choice_artifacts


def expected_version(choice_artifacts):
    payload = json.dumps({"parent": PARENT_VERSION,
                          "artifacts": choice_artifacts,
                          "weights": [0.0] * 4, "source": "override"},
                         sort_keys=True, separators=(",", ":"))
    return "experimental-phonetic-choice-1-" + hashlib.sha256(
        payload.encode()).hexdigest()[:12]


def verify_prepared(directory=PREPARED):
    directory = Path(directory).resolve()
    proof_path = directory / "independent-verification.json"
    need(directory.is_dir() and not proof_path.exists(),
         "prepared/proof state invalid")
    need((directory / "manifest.json").is_file()
         and (directory / "requests.json").is_file(),
         "prepared files missing")
    expected = frozen_sources()
    manifest = read(directory / "manifest.json")
    requests = read(directory / "requests.json")
    rows = requests.get("rows")
    need(rows == expected and source_rows(rows) == rows,
         "prepared FIT source/component order differs")
    groups = probe_groups(rows)
    shards = partition(rows)
    need(requests == {"rows": rows, "groups": groups, "shards": shards},
         "prepared requests/groups/shards differ")
    pins = manifest.get("pins")
    need(isinstance(pins, dict)
         and manifest == {
             "rows": ROWS,
             "components": len({row["component_id"] for row in rows}),
             "pins": pins,
             "requests_sha256": sha(directory / "requests.json"),
             "train_only": True,
             "previous_eligible_rows": 1483,
             "restored_rows": 159,
             "labels_opened": False,
         }, "prepared manifest differs")
    required = {path_key(path): digest for path, digest in FROZEN.items()}
    required[path_key(PRODUCER)] = sha(PRODUCER)
    required[path_key(PLAN)] = sha(PLAN)
    need(pins == required, "prepared pin set differs")
    proof = {
        "schema": "current-phonetic-train-inputs-independent-verification-v1",
        "verified": True,
        "manifest_sha256": sha(directory / "manifest.json"),
        "requests_sha256": sha(directory / "requests.json"),
        "verifier_sha256": sha(Path(__file__)),
        "counts": {"rows": ROWS, "components": manifest["components"],
                   "groups": PROBES, "shards": SHARDS,
                   "shard_rows": list(map(len, shards))},
        "contracts_verified": {
            "exact_frozen_fit_source_component_and_order": True,
            "independent_probe_and_component_partition": True,
            "fit_only_no_dev_test_hold_or_reserve_labels": True,
            "model_loaded_or_inference_run_by_verifier": False,
        },
    }
    write_proof(proof_path, proof)
    return proof


def expected_launch_command(directory, mode, shard=None):
    need(mode in ("probe", "capture"), "invalid runtime mode")
    need((mode == "capture") is (type(shard) is int and 0 <= shard < SHARDS),
         "invalid runtime shard")
    command = [str(ROOT / ".venv/bin/python"), "-m",
               "scripts.current_phonetic_train_pool", "--mode", mode,
               "--output", str(Path(directory).resolve())]
    if mode == "capture":
        command.extend(["--shard", str(shard)])
    return command


def verify_launch(directory, mode, shard, producer_sha):
    launch_path = Path(directory).with_suffix(".launch.json")
    need(launch_path.is_file(), "launch receipt missing")
    launch = read(launch_path)
    need(set(launch) == {"pid", "command", "output", "log",
                         "runner_sha256", "launched_at"}
         and type(launch.get("pid")) is int and launch["pid"] > 0
         and launch.get("command") == expected_launch_command(directory, mode, shard)
         and launch.get("output") == str(Path(directory).resolve())
         and launch.get("log") == str(Path(directory).with_suffix(".log"))
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


def verify_pins(pins, expected_models, mode):
    need(isinstance(pins, dict) and pins, "runtime pins missing")
    for raw, digest in pins.items():
        need(isinstance(raw, str) and isinstance(digest, str)
             and len(digest) == 64, "runtime pin schema differs")
        lowered = raw.lower()
        need("/dev.jsonl" not in lowered and "/test.jsonl" not in lowered
             and "inner-hold" not in lowered and "reserve" not in lowered,
             "forbidden label pin")
        path = Path(raw) if Path(raw).is_absolute() else ROOT / raw
        need(path.is_file() and sha(path) == digest,
             "runtime pin changed: " + raw)
    prepared_manifest = read(PREPARED / "manifest.json")
    required = dict(prepared_manifest["pins"])
    for path in (PREPARED / "requests.json", PREPARED / "manifest.json",
                 PREPARED / "independent-verification.json", PRODUCER, PLAN,
                 PURE_HELPER, ROOT / "scripts/source_span_resources.py",
                 ROOT / "scripts/preflight_pko_seq2seq.py",
                 ROOT / "scripts/evaluate_sentence_only.py",
                 ROOT / "scripts/evaluate_phonetic_choice_dev.py"):
        required[path_key(path)] = sha(path)
    for raw, digest in expected_models.items():
        required[path_key(ROOT / raw)] = digest
    if mode == "capture":
        for path in (PROBE / "report.json", PROBE / "independent-verification.json"):
            required[path_key(path)] = sha(path)
    need(pins == required, "runtime pin set differs")


def verify_record(expected, record, version, models, choice_artifacts):
    need(set(record) == {"id", "component_id", "request", "response",
                         "base_paths", "elapsed_s"}
         and record["id"] == expected["id"]
         and record["component_id"] == expected["component_id"]
         and record["request"] == {"input": expected["input"], "context": ""}
         and finite(record["elapsed_s"]) and record["elapsed_s"] >= 0,
         "capture source/schema differs")
    source, response, paths = expected["input"], record["response"], record["base_paths"]
    candidates = response.get("candidates")
    choice = response.get("phonetic_choice")
    need(response.get("input") == source and response.get("context") == ""
         and response.get("external_context_used") is False
         and response.get("version") == version
         and response.get("model_artifacts") == models
         and isinstance(candidates, list) and candidates
         and response.get("result") == candidates[0].get("text")
         and isinstance(choice, dict)
         and choice.get("parent_version") == PARENT_VERSION
         and choice.get("version") == version
         and choice.get("artifacts") == choice_artifacts
         and choice.get("effective_weights") == [0.0] * 4
         and choice.get("calibration_source") == "override",
         "zero runtime identity differs")
    for candidate in candidates:
        verify_candidate(source, candidate)
    if response.get("sentence_ranking_applied") is not True:
        need(paths == [] and choice.get("rows") == []
             and choice.get("score_records") == [], "unranked capture differs")
        return Counter(rows=1, unranked_rows=1)
    need(isinstance(paths, list) and paths
         and len({path[0] for path in paths}) == len(paths)
         and paths == sorted(paths, key=lambda path: (-path[1], path[0]))
         and [path[0] for path in paths]
             == response.get("phonetic_word_union", {}).get("union"),
         "complete union pool differs")
    for text, score, segments in paths:
        verify_candidate(source, {"text": text, "score": score,
                                  "segments": segments})
    scores = verify_score_records(choice.get("score_records"))
    rows = choice.get("rows")
    need(isinstance(rows, list) and len(rows) == len(paths),
         "choice row denominator differs")
    used, statuses = set(), Counter()
    for path, row in zip(paths, rows):
        features = verify_choice_row(source, path, row, scores)
        need(row.get("features") == features and row.get("residual") == 0
             and row.get("base_score") == path[1]
             and row.get("adjusted_score") == path[1],
             "zero score/feature replay differs")
        for edge in row["edges"]:
            statuses[edge["status"]] += 1
            if edge["status"] == "supported":
                used.add(edge["pair_id"])
    need(used == set(scores)
         and choice.get("supported_pair_count") == len(scores)
         and choice.get("edge_status_counts") == dict(sorted(statuses.items()))
         and choice.get("zero_weight_parity") is True
         and choice.get("zero_weight_score_parity") is True
         and choice.get("zero_weight_order_parity") is True
         and render_public(source, paths) == candidates
         and choice.get("public") == [row["text"] for row in candidates],
         "zero parity/public replay differs")
    return Counter(rows=1, ranked_rows=1, paths=len(paths), pairs=len(scores),
                   supported_edges=statuses["supported"])


def projected_seconds(groups, shards, timings, initialization):
    need(finite(initialization) and initialization >= 0
         and isinstance(timings, dict), "probe timing schema differs")
    probe_ids = {group["probe_id"] for group in groups}
    need(set(timings) == probe_ids
         and all(finite(value) and value >= 0 for value in timings.values()),
         "probe timings differ")
    by_id = {identifier: timings[group["probe_id"]]
             for group in groups for identifier in group["ids"]}
    return [1.3 * (initialization + sum(by_id[identifier]
                                       for identifier in shard)) + 60.0
            for shard in shards]


def verify_runtime(directory, mode=None, shard=None):
    directory = Path(directory).resolve()
    proof_path = directory / "independent-verification.json"
    need(directory.is_dir() and not proof_path.exists(), "run/proof state invalid")
    for name in ("failure.json", "resource-watchdog-failure.json"):
        need(not (directory / name).exists(), "failure artifact exists: " + name)
    required = ("capture.jsonl", "capture.complete.json", "report.json",
                "completion.json", "pins.json", "resources.jsonl")
    need(all((directory / name).is_file() for name in required),
         "terminal run files missing")
    report = read(directory / "report.json")
    mode = report.get("mode") if mode is None else mode
    shard = report.get("shard") if mode == "capture" and shard is None else shard
    need(mode in ("probe", "capture")
         and ((mode == "probe" and shard is None)
              or (mode == "capture" and type(shard) is int and 0 <= shard < SHARDS)),
         "runtime mode/shard differs")
    producer_sha = sha(PRODUCER)
    launch_path = verify_launch(directory, mode, shard, producer_sha)
    prepared_proof = read(PREPARED / "independent-verification.json")
    need(prepared_proof.get("verified") is True
         and prepared_proof.get("manifest_sha256") == sha(PREPARED / "manifest.json")
         and prepared_proof.get("requests_sha256") == sha(PREPARED / "requests.json"),
         "prepared proof differs")
    prepared = read(PREPARED / "requests.json")
    source = frozen_sources()
    need(prepared.get("rows") == source
         and prepared.get("groups") == probe_groups(source)
         and prepared.get("shards") == partition(source),
         "prepared source domain changed")
    ids = ([group["probe_id"] for group in prepared["groups"]]
           if mode == "probe" else prepared["shards"][shard])
    models, choice_artifacts = expected_artifacts()
    version = expected_version(choice_artifacts)
    capture_sha = sha(directory / "capture.jsonl")
    closure = read(directory / "capture.complete.json")
    pins = read(directory / "pins.json")
    completion = read(directory / "completion.json")
    need(closure == {"mode": mode, "shard": shard, "rows": len(ids),
                     "capture_sha256": capture_sha, "labels_opened": False},
         "capture closure differs")
    expected_projected = (projected_seconds(prepared["groups"], prepared["shards"],
                                            report.get("timings"),
                                            report.get("initialization_s"))
                          if mode == "probe" else None)
    qualified = (all(value <= LIMITS["whole_seconds"]
                     for value in expected_projected)
                 if expected_projected is not None else True)
    need(report == {
        "completed": True, "mode": mode, "shard": shard, "rows": len(ids),
        "ids": ids, "pins": pins, "capture_sha256": capture_sha,
        "closure_sha256": sha(directory / "capture.complete.json"),
        "initialization_s": report.get("initialization_s"),
        "timings": report.get("timings"),
        "projected_shard_seconds": expected_projected,
        "qualified": qualified, "version": version,
        "model_artifacts": models, "limits": LIMITS, "train_only": True,
        "labels_opened": False, "production_changed": False,
        "neural_training": False,
    }, "terminal report differs")
    need(completion == {"completed": True,
                        "report_sha256": sha(directory / "report.json")},
         "terminal completion differs")
    verify_pins(pins, models, mode)
    by_id = {row["id"]: row for row in source}
    rows = list(jsonl(directory / "capture.jsonl"))
    need(len(rows) == len(ids), "capture row count differs")
    census = Counter()
    for identifier, record in zip(ids, rows):
        census.update(verify_record(by_id[identifier], record, version,
                                    models, choice_artifacts))
    need(set(report["timings"]) == set(ids)
         and all(finite(report["timings"][row["id"]])
                 and report["timings"][row["id"]] >= row["elapsed_s"]
                 for row in rows),
         "reported timings are not capture-linked")
    if mode == "capture":
        probe_report = read(PROBE / "report.json")
        probe_proof = read(PROBE / "independent-verification.json")
        need(probe_report.get("qualified") is True
             and probe_proof.get("verified") is True
             and probe_proof.get("report_sha256") == sha(PROBE / "report.json"),
             "capture lacks qualified verified probe")
    operations = ["authenticate", "load_models"]
    operations += ["predict:" + identifier for identifier in ids]
    operations += ["write_report", "write_completion"]
    resources = verify_resources(list(jsonl(directory / "resources.jsonl")),
                                 operations)
    proof = {
        "schema": "current-phonetic-train-pool-independent-verification-v1",
        "verified": True, "mode": mode, "shard": shard,
        "report_sha256": sha(directory / "report.json"),
        "completion_sha256": sha(directory / "completion.json"),
        "capture_sha256": capture_sha,
        "closure_sha256": sha(directory / "capture.complete.json"),
        "launch_sha256": sha(launch_path), "producer_sha256": producer_sha,
        "verifier_sha256": sha(Path(__file__)),
        "pure_helper_sha256": OLD_FROZEN[PURE_HELPER],
        "version": version, "qualified": qualified,
        "projected_shard_seconds": expected_projected,
        "counts": {**dict(census), "operations": resources["operations"]},
        "resources": resources,
        "contracts_verified": {
            "exact_fit_source_component_group_shard_and_order": True,
            "source_only_requests_and_no_labels": True,
            "exact_terminal_launch_code_pins_models_and_artifact_set": True,
            "independent_zero_version_feature_score_order_and_public_replay": True,
            "probe_projection_recomputed_for_all_six_shards": mode == "probe",
            "capture_requires_qualified_verified_probe": mode == "capture",
            "atomic_resource_operation_sequence": True,
            "model_loaded_or_inference_run_by_verifier": False,
        },
        "limitations": [
            "Recorded teacher-forced likelihood values are structurally and relationally verified but are not recomputed by a neural model.",
            "This source-only proof contains no FIT label join, accuracy claim, neural training, or deployment evidence.",
        ],
    }
    write_proof(proof_path, proof)
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "probe", "capture"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard", type=int, choices=range(SHARDS))
    arguments = parser.parse_args()
    need((arguments.mode == "capture") == (arguments.shard is not None),
         "shard is required only for capture")
    result = (verify_prepared(arguments.output) if arguments.mode == "prepare"
              else verify_runtime(arguments.output, arguments.mode, arguments.shard))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
