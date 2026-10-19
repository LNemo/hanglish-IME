#!/usr/bin/env python3
"""Independently verify the source-only two-shard document partition.

This module deliberately does not import the partition producer, Torch, a
runtime converter, or any target-bearing dataset.  It reconstructs the 16
UTF-8 preflight strata, timing lookup, whole-document costs, and greedy
two-bin assignment from the frozen source-only inputs and selection records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".omx/artifacts/document-sentence-shards-v1"
INPUT_RUN = ROOT / ".omx/artifacts/document-sentence-inputs-v1"
RESERVE = ROOT / ".omx/artifacts/document-reserve-v2"
PREFLIGHT = ROOT / ".omx/artifacts/document-phonetic-choice-preflight-v1"
PLAN = ROOT / ".omx/plans/document-phonetic-choice-sharded-v1.md"
PRODUCER = ROOT / "scripts/prepare_document_sentence_shards.py"
SCHEMA = "document-sentence-shards-independent-verification-v1"
RUN_SCHEMA = "document-sentence-shards-v1"
UNRESOLVED = "__FILL_AFTER_PRODUCER_AND_ARTIFACT_FREEZE__"
EXPECTED_PRODUCER_SHA256 = "db55ca8979c8ab7b99ea06d19f49d710d3e617e78abf2ab1fdc6dcb662fd23b4"
EXPECTED_PLAN_SHA256 = "4b2c05e75e25c25b7f5f19679d9239f681f5e2d3fecdad16db133d0a543472b2"
EXPECTED_ARTIFACT_HASHES = {
    "manifest.json": "a09e13062f759ec8ac8d13b44344b63724e50f8d0302e4a945bfa13fdf797fc9",
    "shard-0.jsonl": "9c9e7f49f95d90f38d530cb0bc879eefe0be98c2fc9593e4aa24500129ffae48",
    "shard-1.jsonl": "3aeecc85a58186a4bbb9f0ac45edd52038488f31bc01b9749bb92aaddad04df5",
}
SOURCE_HASHES = {
    INPUT_RUN / "manifest.json":
        "acea9e1d4d4929253f0d4adfc2aea8ba7c877cf5093b0499961619a35a7632a3",
    INPUT_RUN / "inputs.jsonl":
        "536707f05503856f4f805bb9775bd9c5dc932c94783e669a5d62fe3023b0b1db",
    INPUT_RUN / "independent-verification.json":
        "c12e49e6f735da775e4c996ef0bea0ecdc699cb156d31bdf8ceef01b65d3935e",
    RESERVE / "selection.jsonl":
        "7d62027d11dcf1320b2a1bac7bcb83e1a31d31ce521808bbe4d1faadf6d4b810",
    PREFLIGHT / "schedule.json":
        "685d7b7310e6669839ff6b13498e46a12d26d7215213030460ad7de2dcb3e48a",
    PREFLIGHT / "report.json":
        "c4454712583618e3d213bd1f2baf631d4f82f459fb9e8de00358f0df54eff874",
    PREFLIGHT / "independent-verification.json":
        "46b678cf26b1768f3620b3220f06cb35ab554f8523b3c5940f5af78806e4809f",
}
LIMITS = {
    "per_shard_seconds": 1800.0,
    "combined_seconds": 3600.0,
    "loading_allowance_seconds": 60.0,
    "multiplier": 1.3,
    "trailing_allowance_seconds": 60.0,
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


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
    return rows


def digest_lines(values):
    payload = "".join(value + "\n" for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def byte_probe_groups(requests):
    """Reconstruct the preflight's 16 balanced UTF-8-length strata."""
    need(isinstance(requests, list) and len(requests) >= 16,
         "too few source requests")
    need(all(set(row) == {"id", "input"}
             and isinstance(row["id"], str) and row["id"]
             and isinstance(row["input"], str) and row["input"]
             for row in requests), "source-only request schema differs")
    need(len({row["id"] for row in requests}) == len(requests),
         "duplicate source request ID")
    ordered = sorted(
        requests,
        key=lambda row: (len(row["input"].encode("utf-8")), row["id"]),
    )
    small, extra = divmod(len(ordered), 16)
    groups, offset = [], 0
    for index in range(16):
        count = small + int(index < extra)
        group = ordered[offset:offset + count]
        groups.append({
            "ids": [row["id"] for row in group],
            "probe_id": group[-1]["id"],
            "count": count,
            "max_utf8_bytes": len(group[-1]["input"].encode("utf-8")),
        })
        offset += count
    need(offset == len(ordered), "UTF-8 strata do not cover every request")
    return groups


def request_costs(requests, schedule, timings):
    """Assign each request its stratum's measured learned-path duration."""
    expected = byte_probe_groups(requests)
    need(schedule == {
        "selection": "16 balanced UTF-8-length groups, longest source then ID",
        "groups": expected,
    }, "preflight byte schedule differs")
    need(isinstance(timings, list) and len(timings) == 16,
         "preflight timing rows differ")
    timings_by_probe = {}
    for timing in timings:
        need(isinstance(timing, dict)
             and isinstance(timing.get("id"), str) and timing["id"]
             and timing["id"] not in timings_by_probe,
             "preflight timing identity differs")
        timings_by_probe[timing["id"]] = timing
    costs = {}
    for group in expected:
        timing = timings_by_probe.get(group["probe_id"])
        need(isinstance(timing, dict)
             and timing.get("group_size") == group["count"]
             and finite(timing.get("learned_seconds"))
             and timing["learned_seconds"] > 0,
             "preflight learned timing identity differs")
        for identifier in group["ids"]:
            need(identifier not in costs, "duplicate request timing assignment")
            costs[identifier] = timing["learned_seconds"]
    need(len(costs) == len(requests), "request timing assignment is incomplete")
    need(set(timings_by_probe) == {group["probe_id"] for group in expected},
         "preflight timing probe domain differs")
    return costs


def projected_seconds(estimated_cost):
    need(finite(estimated_cost) and estimated_cost >= 0,
         "invalid estimated shard cost")
    return (LIMITS["multiplier"]
            * (LIMITS["loading_allowance_seconds"] + estimated_cost)
            + LIMITS["trailing_allowance_seconds"])


def partition(requests, selection, costs):
    """Independently apply the frozen whole-group greedy two-bin algorithm."""
    need(len(requests) == len(selection) == len(costs),
         "request/selection/cost denominator differs")
    request_ids = [row.get("id") for row in requests]
    selection_ids = [row.get("id") for row in selection]
    need(request_ids == selection_ids, "selection and request order differ")
    need(all(set(row) == {"id", "group_id"}
             and isinstance(row["id"], str) and row["id"]
             and isinstance(row["group_id"], str) and row["group_id"]
             for row in selection), "reserve selection schema differs")
    need(len(set(selection_ids)) == len(selection_ids), "duplicate selected ID")
    need(set(costs) == set(selection_ids)
         and all(finite(value) and value >= 0 for value in costs.values()),
         "request cost domain differs")

    group_costs = {}
    group_order = []
    for row in selection:
        group = row["group_id"]
        if group not in group_costs:
            group_costs[group] = 0.0
            group_order.append(group)
        group_costs[group] += costs[row["id"]]
    need(len(group_costs) == 821 or len(selection) != 1046,
         "whole evaluation group denominator differs")

    bin_groups = [set(), set()]
    bin_costs = [0.0, 0.0]
    for group, cost in sorted(group_costs.items(), key=lambda item: (-item[1], item[0])):
        index = 0 if bin_costs[0] <= bin_costs[1] else 1
        bin_groups[index].add(group)
        bin_costs[index] += cost

    shards = [[], []]
    shard_group_order = [[], []]
    seen_groups = [set(), set()]
    for request, selected in zip(requests, selection):
        memberships = [selected["group_id"] in groups for groups in bin_groups]
        need(sum(memberships) == 1, "document group assignment is not disjoint")
        index = memberships.index(True)
        shards[index].append(request)
        group = selected["group_id"]
        if group not in seen_groups[index]:
            seen_groups[index].add(group)
            shard_group_order[index].append(group)

    need(sum(len(rows) for rows in shards) == len(requests)
         and not (bin_groups[0] & bin_groups[1])
         and (bin_groups[0] | bin_groups[1]) == set(group_costs),
         "shard coverage/disjointness differs")
    metadata = []
    for index in range(2):
        estimate = bin_costs[index]
        projection = projected_seconds(estimate)
        metadata.append({
            "index": index,
            "file": f"shard-{index}.jsonl",
            "rows": len(shards[index]),
            "groups": len(bin_groups[index]),
            "estimated_cost_seconds": estimate,
            "projected_seconds": projection,
            "qualified": projection <= LIMITS["per_shard_seconds"],
            "ordered_ids_sha256": digest_lines([row["id"] for row in shards[index]]),
            "ordered_group_ids_sha256": digest_lines(shard_group_order[index]),
        })
    return shards, metadata


def resolved_identities(producer=EXPECTED_PRODUCER_SHA256,
                        plan=EXPECTED_PLAN_SHA256,
                        artifacts=EXPECTED_ARTIFACT_HASHES):
    values = [producer, plan, *artifacts.values()]
    need(set(artifacts) == {"manifest.json", "shard-0.jsonl", "shard-1.jsonl"}
         and all(isinstance(value, str) and len(value) == 64
                 and value != UNRESOLVED for value in values),
         "partition producer/artifact identities are not frozen")
    return producer, plan, dict(artifacts)


def _same_number(actual, expected):
    return finite(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)


def verify(directory=RUN):
    producer_sha, plan_sha, artifact_hashes = resolved_identities()
    directory = Path(directory)
    proof_path = directory / "independent-verification.json"
    need(directory.is_dir() and not proof_path.exists(), "run/proof state invalid")
    need(PRODUCER.is_file() and sha(PRODUCER) == producer_sha,
         "partition producer identity differs")
    need(PLAN.is_file() and sha(PLAN) == plan_sha, "partition plan identity differs")
    need(all(path.is_file() and sha(path) == digest
             for path, digest in SOURCE_HASHES.items()),
         "partition source identity differs")
    for name, digest in artifact_hashes.items():
        path = directory / name
        need(path.is_file() and sha(path) == digest,
             "partition artifact identity differs: " + name)

    input_manifest = read(INPUT_RUN / "manifest.json")
    input_proof = read(INPUT_RUN / "independent-verification.json")
    preflight_report = read(PREFLIGHT / "report.json")
    preflight_proof = read(PREFLIGHT / "independent-verification.json")
    schedule = read(PREFLIGHT / "schedule.json")
    need(input_manifest.get("schema") == "document-sentence-inputs-v1"
         and input_manifest.get("completed") is True
         and input_manifest.get("model_activity") is False
         and input_manifest.get("performance_labels_joined") is False
         and input_manifest.get("counts", {}).get("rows") == 1046
         and input_manifest.get("counts", {}).get("groups") == 821
         and input_proof.get("schema")
             == "document-sentence-inputs-independent-verification-v1"
         and input_proof.get("verified") is True
         and input_proof.get("contracts_verified", {}).get(
             "model_inference_training_or_http_activity") is False
         and input_proof.get("contracts_verified", {}).get(
             "no_filtering_context_group_reference_or_target_fields") is True,
         "source-only input proof differs")
    need(preflight_report.get("schema") == "document-phonetic-choice-preflight-v1"
         and preflight_report.get("completed") is True
         and preflight_report.get("input_rows") == 1046
         and preflight_report.get("probe_rows") == 16
         and preflight_report.get("accuracy_measured") is False
         and preflight_report.get("performance_labels_opened") is False
         and preflight_report.get("shared_qualified") is False
         and preflight_report.get("schedule_sha256") == SOURCE_HASHES[
             PREFLIGHT / "schedule.json"]
         and preflight_proof.get("schema")
             == "document-phonetic-choice-preflight-independent-verification-v1"
         and preflight_proof.get("verified") is True
         and preflight_proof.get("report_sha256") == SOURCE_HASHES[
             PREFLIGHT / "report.json"]
         and preflight_proof.get("schedule_sha256") == SOURCE_HASHES[
             PREFLIGHT / "schedule.json"],
         "preflight proof identity/scope differs")

    requests = jsonl(INPUT_RUN / "inputs.jsonl")
    selection = jsonl(RESERVE / "selection.jsonl")
    need(len(requests) == len(selection) == 1046,
         "source/selection denominator differs")
    costs = request_costs(requests, schedule, preflight_report.get("timings"))
    expected_shards, expected_meta = partition(requests, selection, costs)
    actual_shards = [jsonl(directory / f"shard-{index}.jsonl")
                     for index in range(2)]
    need(actual_shards == expected_shards,
         "shard membership, source text, or selection-relative order differs")
    for index, rows in enumerate(actual_shards):
        expected_meta[index]["inputs_sha256"] = sha(
            directory / f"shard-{index}.jsonl")
        need(len({row["id"] for row in rows}) == len(rows),
             "duplicate ID within shard")
    need({row["id"] for rows in actual_shards for row in rows}
             == {row["id"] for row in requests}
         and sum(len(rows) for rows in actual_shards) == 1046,
         "shards omit or duplicate source requests")
    need(all(row["qualified"] for row in expected_meta)
         and sum(row["projected_seconds"] for row in expected_meta)
             <= LIMITS["combined_seconds"],
         "two-shard time qualification failed")

    manifest = read(directory / "manifest.json")
    need(manifest.get("schema") == RUN_SCHEMA
         and manifest.get("completed") is True
         and manifest.get("counts") == {"rows": 1046, "groups": 821, "shards": 2}
         and manifest.get("limits") == LIMITS
         and manifest.get("model_activity") is False
         and manifest.get("performance_labels_joined") is False,
         "partition manifest identity/scope differs")
    actual_meta = manifest.get("shards")
    need(isinstance(actual_meta, list) and len(actual_meta) == 2,
         "partition manifest shard metadata differs")
    for actual, expected in zip(actual_meta, expected_meta):
        need(set(actual) == set(expected), "shard metadata schema differs")
        for key, value in expected.items():
            if key in {"estimated_cost_seconds", "projected_seconds"}:
                need(_same_number(actual.get(key), value),
                     "shard numeric metadata differs: " + key)
            else:
                need(actual.get(key) == value, "shard metadata differs: " + key)

    expected_sources = {
        str(path.relative_to(ROOT)): digest for path, digest in SOURCE_HASHES.items()
    }
    expected_sources[str(PLAN.relative_to(ROOT))] = plan_sha
    expected_sources[str(PRODUCER.relative_to(ROOT))] = producer_sha
    need(manifest.get("source_hashes") == expected_sources,
         "partition manifest source hashes differ")
    need(manifest.get("partition_contract") == {
        "cost_source": "preflight learned_seconds for each exact UTF-8 length bin",
        "document_group_cost": "selection-order sum of member input costs",
        "group_order": "descending cost, then ascending group_id",
        "placement": "smaller current bin cost, then lower shard index",
        "row_order": "original document-reserve-v2 selection order within each shard",
        "ordered_group_digest": "first-occurrence distinct groups in shard row order",
        "whole_document_groups": True,
        "all_rows_retained": True,
        "duplicates_retained": True,
        "labels_opened": False,
        "raw_references_deserialized": False,
    }, "partition contract differs")
    need(manifest.get("accuracy_measured") is False
         and manifest.get("limits_and_caveats") == [
             "Preflight timings are projections rather than execution-time upper bounds.",
             "The two qualified shards must run serially and retain independent 1,800-second watchdogs.",
             "No raw reference text or accuracy label was deserialized to construct this partition.",
         ], "partition scope/caveats differ")

    proof = {
        "schema": SCHEMA,
        "verified": True,
        "producer_sha256": producer_sha,
        "verifier_sha256": sha(Path(__file__)),
        "plan_sha256": plan_sha,
        "manifest_sha256": artifact_hashes["manifest.json"],
        "shards": expected_meta,
        "counts": {"rows": 1046, "groups": 821, "shards": 2},
        "contracts_verified": {
            "source_only_input_rows": True,
            "independent_utf8_strata_and_timing_lookup": True,
            "independent_whole_group_greedy_partition": True,
            "exact_coverage_disjointness_and_relative_order": True,
            "per_shard_and_combined_time_qualification": True,
            "model_tensor_or_neural_inference_run_by_verifier": False,
            "raw_references_or_performance_labels_deserialized": False,
            "partition_producer_imported": False,
        },
        "model_activity": False,
        "performance_labels_joined": False,
        "limitations": [
            "Timing projections inherit the preflight's length-stratified estimates and are not upper bounds.",
            "The partition verifier checks source-only IDs and Roman inputs; it never opens reference text.",
            "Partition qualification is execution authorization, not sentence-accuracy or deployment evidence.",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.run), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
