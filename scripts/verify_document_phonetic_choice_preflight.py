#!/usr/bin/env python3
"""Pure independent verifier for the document phonetic-choice preflight.

This verifier imports no preflight producer, runtime adapter, feature projector,
Torch, tensor library, or neural model.  Target-bearing pins may be hashed as
opaque bytes, but no accuracy label or raw reference is deserialized or joined.
"""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path

from scripts.verify_production_phonetic_choice_dev import (
    canonical, finite, jsonl, need, path_key, read, read_exact_jsonl,
    render_public, sha, verify_candidate, verify_resources, verify_response,
)


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".omx/artifacts/document-phonetic-choice-preflight-v1"
INPUTS = ROOT / ".omx/artifacts/document-sentence-inputs-v1"
TEST_RUN = ROOT / ".omx/artifacts/production-phonetic-choice-test-v1"
PRODUCER = ROOT / "scripts/preflight_document_phonetic_choice.py"
PLAN = ROOT / ".omx/plans/document-phonetic-choice-preflight-v1.md"
RUN_SCHEMA = "document-phonetic-choice-preflight-v1"
SCHEMA = "document-phonetic-choice-preflight-independent-verification-v1"
PRODUCTION_VERSION = "experimental-ctc-union-1-488fc89732af"
EXPECTED_VERSION = "experimental-production-phonetic-choice-1-30e29bb96e4a"
EXPECTED_WEIGHTS = [
    0.09822412586490854,
    0.08018606995271488,
    -0.060145568052192455,
    -0.0019953930240048014,
]
LIMITS = {"whole_seconds": 1800.0, "update_seconds": 60.0,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
UNRESOLVED = "__FILL_AFTER_INPUT_PROOF_AND_PRODUCER_FREEZE__"
EXPECTED_PRODUCER_SHA256 = "9497dfa40d8ea976a3f4b0e11b10c3a36096b907fb3f20d0f2e2203af508bee1"
EXPECTED_PLAN_SHA256 = "be9ff4a604fd997982a7ef9d8f27a65ebe9377723a382875bab27ec948e341bc"
INPUT_HASHES = {
    "manifest.json": "acea9e1d4d4929253f0d4adfc2aea8ba7c877cf5093b0499961619a35a7632a3",
    "inputs.jsonl": "536707f05503856f4f805bb9775bd9c5dc932c94783e669a5d62fe3023b0b1db",
    "independent-verification.json":
        "c12e49e6f735da775e4c996ef0bea0ecdc699cb156d31bdf8ceef01b65d3935e",
}
TEST_HASHES = {
    "report.json": "e8592ca1435c8ea1da37e7ec63a6ee3ad71ecc3bf2951541b5b791158de8360b",
    "completion.json": "b8a29e16601b5f533ea1f83dd21b68220ea08d2427272211639d1438f9be31e7",
    "independent-verification.json":
        "0ce8bc781d55ce2ac159447ad3e17082c331f17527ad3aa819cba7ab0865212c",
}


def resolved_identities(producer=EXPECTED_PRODUCER_SHA256,
                        inputs=INPUT_HASHES):
    values = [producer, *inputs.values()]
    need(set(inputs) == {"manifest.json", "inputs.jsonl",
                         "independent-verification.json"}
         and all(isinstance(value, str) and len(value) == 64
                 and value != UNRESOLVED for value in values),
         "preflight producer/input identities are not frozen")
    return producer, dict(inputs)


def byte_probe_groups(requests):
    need(isinstance(requests, list) and len(requests) >= 16,
         "too few source requests")
    ordered = sorted(requests,
                     key=lambda row: (len(row["input"].encode("utf-8")), row["id"]))
    small, extra = divmod(len(ordered), 16)
    groups, offset = [], 0
    for index in range(16):
        count = small + int(index < extra)
        group = ordered[offset:offset + count]
        groups.append({"ids": [row["id"] for row in group],
                       "probe_id": group[-1]["id"], "count": count,
                       "max_utf8_bytes": len(
                           group[-1]["input"].encode("utf-8"))})
        offset += count
    return groups


def projections(elapsed, timings):
    need(finite(elapsed) and elapsed >= 0 and len(timings) == 16,
         "projection inputs differ")
    paired = 1.3 * (elapsed + sum(
        (row["parent_seconds"] + row["learned_seconds"])
        * row["group_size"] for row in timings)) + 60.0
    shared = 1.3 * (elapsed + sum(
        row["learned_seconds"] * row["group_size"] for row in timings)) + 60.0
    return {"projected_paired_seconds": paired,
            "projected_shared_seconds": shared,
            "paired_qualified": paired <= LIMITS["whole_seconds"],
            "shared_qualified": shared <= LIMITS["whole_seconds"]}


def verify_parent(source, response, paths):
    need(response.get("input") == source and response.get("context") == ""
         and response.get("external_context_used") is False
         and response.get("version") == PRODUCTION_VERSION,
         "parent source/runtime identity differs")
    candidates = response.get("candidates")
    need(isinstance(candidates, list) and 1 <= len(candidates) <= 5
         and response.get("result") == candidates[0].get("text"),
         "parent public response differs")
    for candidate in candidates:
        verify_candidate(source, candidate)
    need(any(candidate["text"] == source for candidate in candidates),
         "parent public response omitted source")
    for name in ("ctc_word", "ctc_union"):
        stage = response.get(name)
        need(isinstance(stage, dict) and finite(stage.get("generation_ms"))
             and stage["generation_ms"] >= 0,
             "parent generation metadata differs: " + name)
    if response.get("sentence_ranking_applied") is True:
        need(isinstance(paths, list) and paths
             and len({path[0] for path in paths}) == len(paths)
             and paths == sorted(paths, key=lambda path: (-path[1], path[0]))
             and render_public(source, paths) == candidates,
             "parent complete path/public assembly differs")
    else:
        need(paths == [], "unranked parent unexpectedly exposed full paths")


def verify_probe(group, row, source, weights, expected_version):
    need(set(row) == {"id", "request", "parent", "zero", "proposed",
                      "parent_paths", "zero_base_paths", "base_paths"}
         and row["id"] == group["probe_id"]
         and row["request"] == {"input": source, "context": ""},
         "probe source/schema identity differs")
    parent_paths = row["parent_paths"]
    need(parent_paths == row["zero_base_paths"] == row["base_paths"],
         "parent/zero/learned complete paths differ")
    verify_parent(source, row["parent"], parent_paths)
    frozen = {"parent": row["parent"],
              "parent_pool": [path[0] for path in parent_paths]}
    zero_counts = verify_response(source, frozen, row["zero"],
                                  row["zero_base_paths"], [0.0] * 4)
    learned_counts = verify_response(source, frozen, row["proposed"],
                                     row["base_paths"], weights,
                                     expected_version)
    need(row["parent"]["candidates"] == row["zero"]["candidates"]
         and row["parent"]["result"] == row["zero"]["result"],
         "actual zero public parity differs")
    zero_choice = row["zero"]["phonetic_choice"]
    learned_choice = row["proposed"]["phonetic_choice"]
    need(zero_choice.get("score_records") == learned_choice.get("score_records"),
         "zero/learned likelihood records differ")
    need([item.get("features") for item in zero_choice.get("rows", [])]
         == [item.get("features") for item in learned_choice.get("rows", [])],
         "zero/learned feature vectors differ")
    return zero_counts, learned_counts


def operation_names(groups):
    operations = ["authenticate_source_inputs", "load_models"]
    operations += [name + ":" + group["probe_id"] for group in groups
                   for name in ("parent", "zero", "learned")]
    operations += ["close_capture", "write_report", "write_completion"]
    need(len(operations) == 53, "preflight operation count differs")
    return operations


def verify(directory=RUN):
    producer_sha, input_hashes = resolved_identities()
    directory = Path(directory)
    proof_path = directory / "independent-verification.json"
    need(directory.is_dir() and not proof_path.exists(), "run/proof state invalid")
    for name in ("failure.json", "failure-report.json",
                 "resource-watchdog-failure.json"):
        need(not (directory / name).exists(), "failure artifact exists: " + name)
    required = ("report.json", "completion.json", "capture.jsonl",
                "capture.complete.json", "schedule.json", "pins.json",
                "resources.jsonl")
    need(all((directory / name).is_file() for name in required),
         "terminal preflight files missing")
    need(sha(PRODUCER) == producer_sha and sha(PLAN) == EXPECTED_PLAN_SHA256,
         "preflight producer or plan changed")
    need(all(sha(INPUTS / name) == digest
             for name, digest in input_hashes.items()),
         "document input artifact changed")
    need(all(sha(TEST_RUN / name) == digest
             for name, digest in TEST_HASHES.items()),
         "qualified TEST artifact changed")

    report = read(directory / "report.json")
    completion = read(directory / "completion.json")
    closure = read(directory / "capture.complete.json")
    schedule = read(directory / "schedule.json")
    pins = read(directory / "pins.json")
    capture_sha = sha(directory / "capture.jsonl")
    need(closure == {"schema": RUN_SCHEMA + "-closure", "rows": 16,
                     "capture_sha256": capture_sha,
                     "performance_labels_opened": False},
         "source-only preflight closure differs")
    need(completion.get("completed") is True
         and completion.get("report_sha256") == sha(directory / "report.json"),
         "preflight completion differs")
    need(report.get("schema") == RUN_SCHEMA and report.get("completed") is True
         and report.get("capture_sha256") == capture_sha
         and report.get("closure_sha256") == sha(directory / "capture.complete.json")
         and report.get("schedule_sha256") == sha(directory / "schedule.json")
         and report.get("pins") == pins,
         "preflight report/terminal linkage differs")

    need(pins.get(path_key(PRODUCER)) == producer_sha
         and pins.get(path_key(PLAN)) == EXPECTED_PLAN_SHA256,
         "critical preflight pins differ")
    for raw, digest in pins.items():
        path = Path(raw) if Path(raw).is_absolute() else ROOT / raw
        need(path.is_file() and sha(path) == digest, "runtime pin changed: " + raw)

    input_manifest = read(INPUTS / "manifest.json")
    input_proof = read(INPUTS / "independent-verification.json")
    test_report = read(TEST_RUN / "report.json")
    test_proof = read(TEST_RUN / "independent-verification.json")
    need(input_manifest.get("completed") is True
         and input_manifest.get("inputs_sha256") == input_hashes["inputs.jsonl"]
         and input_proof.get("verified") is True
         and input_proof.get("manifest_sha256") == input_hashes["manifest.json"]
         and input_proof.get("inputs_sha256") == input_hashes["inputs.jsonl"],
         "independently verified document inputs differ")
    need(test_report.get("completed") is True and test_report.get("quality_gate") is True
         and test_report.get("version") == EXPECTED_VERSION
         and test_report.get("weights") == EXPECTED_WEIGHTS
         and test_proof.get("verified") is True
         and test_proof.get("report_sha256") == TEST_HASHES["report.json"],
         "qualified TEST runtime differs")

    requests = read_exact_jsonl(INPUTS / "inputs.jsonl", 1046)
    need(len({row.get("id") for row in requests}) == 1046
         and all(set(row) == {"id", "input"}
                 and isinstance(row["id"], str) and row["id"]
                 and isinstance(row["input"], str) and row["input"]
                 for row in requests),
         "document source-only requests differ")
    groups = byte_probe_groups(requests)
    need(schedule == {"groups": groups,
                      "selection":
                      "16 balanced UTF-8-length groups, longest source then ID"}
         and sum(group["count"] for group in groups) == 1046
         and len({identifier for group in groups for identifier in group["ids"]}) == 1046,
         "balanced UTF-8 probe schedule differs")
    sources = {row["id"]: row["input"] for row in requests}

    captures = read_exact_jsonl(directory / "capture.jsonl", 16)
    learned_census = Counter()
    zero_census = Counter()
    artifacts = None
    for group, row in zip(groups, captures):
        zero_counts, learned_counts = verify_probe(
            group, row, sources[group["probe_id"]], EXPECTED_WEIGHTS,
            EXPECTED_VERSION)
        zero_census.update(zero_counts)
        learned_census.update(learned_counts)
        zero_artifacts = row["zero"]["phonetic_choice"]["artifacts"]
        current = row["proposed"]["phonetic_choice"]["artifacts"]
        artifacts = current if artifacts is None else artifacts
        need(current == artifacts, "learned choice artifacts changed across probes")
        for raw, digest in {**zero_artifacts, **current}.items():
            path = Path(raw) if Path(raw).is_absolute() else ROOT / raw
            need(pins.get(path_key(path)) == digest,
                 "choice artifact is not runtime-pinned: " + raw)

    timings = report.get("timings")
    need(isinstance(timings, list) and len(timings) == 16,
         "preflight timing rows differ")
    for group, timing, capture in zip(groups, timings, captures):
        need(set(timing) == {"id", "group_size", "parent_seconds",
                             "learned_seconds", "parent_engine_ms",
                             "zero_engine_ms", "learned_engine_ms"}
             and timing["id"] == group["probe_id"]
             and timing["group_size"] == group["count"]
             and all(finite(timing[name]) and timing[name] >= 0 for name in (
                 "parent_seconds", "learned_seconds", "parent_engine_ms",
                 "zero_engine_ms", "learned_engine_ms"))
             and timing["parent_engine_ms"] == capture["parent"]["engine_ms"]
             and timing["zero_engine_ms"] == capture["zero"]["engine_ms"]
             and timing["learned_engine_ms"] == capture["proposed"]["engine_ms"],
             "preflight timing/capture identity differs")
    elapsed = report.get("elapsed_seconds_for_projection")
    expected_projection = projections(elapsed, timings)
    need(all(report.get(key) == value for key, value in expected_projection.items()),
         "paired/shared resource projection differs")

    resources = verify_resources(list(jsonl(directory / "resources.jsonl")),
                                 operation_names(groups))
    need(completion == {"completed": True,
                        "report_sha256": sha(directory / "report.json"),
                        "resource_after_report": resources["final"]},
         "completion/resource linkage differs")
    need(report.get("limits") == LIMITS and report.get("input_rows") == 1046
         and report.get("probe_rows") == 16
         and report.get("weights") == EXPECTED_WEIGHTS
         and report.get("version") == EXPECTED_VERSION
         and report.get("production_changed") is False
         and report.get("performance_labels_opened") is False
         and report.get("accuracy_measured") is False
         and report.get("timing_scope")
            == "16 fresh serial parent/zero/learned probes sharing frozen models; length-based projection is not an upper bound",
         "preflight report contract differs")

    proof = {
        "schema": SCHEMA,
        "verified": True,
        "report_sha256": sha(directory / "report.json"),
        "completion_sha256": sha(directory / "completion.json"),
        "capture_sha256": capture_sha,
        "closure_sha256": sha(directory / "capture.complete.json"),
        "schedule_sha256": sha(directory / "schedule.json"),
        "producer_sha256": producer_sha,
        "verifier_sha256": sha(Path(__file__)),
        "counts": {"input_rows": 1046, "probe_rows": 16,
                   "operations": resources["operations"],
                   "learned_paths": learned_census["paths"],
                   "learned_unique_pair_scores_per_request_sum":
                       learned_census["pairs"],
                   "learned_supported_edge_occurrences":
                       learned_census["supported_edges"],
                   "zero_paths": zero_census["paths"]},
        "projection": expected_projection,
        "resources": resources,
        "limits": LIMITS,
        "contracts_verified": {
            "source_only_1046_inputs_and_balanced_utf8_schedule": True,
            "all_16_parent_zero_learned_responses_and_full_paths": True,
            "exact_parent_zero_learned_base_path_parity": True,
            "independent_edge_pair_feature_residual_and_public_replay": True,
            "zero_public_score_and_order_parity": True,
            "qualified_test_input_proofs_and_all_runtime_pins": True,
            "paired_and_shared_projection_recomputed": True,
            "atomic_53_operation_resource_contract": True,
            "performance_labels_or_raw_references_deserialized": False,
            "accuracy_measured": False,
            "model_tensor_or_neural_inference_run_by_verifier": False,
        },
        "limitations": [
            "Recorded neural likelihood values are validated and replayed but not recomputed by a model.",
            "Length-stratified timing projections are estimates, not upper bounds.",
            "Target-bearing and raw-reference pins are hashed as opaque bytes but never deserialized or joined.",
            "This source-only preflight provides no sentence accuracy or deployment evidence.",
        ],
    }
    with proof_path.open("xb") as stream:
        stream.write(canonical(proof))
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    print(json.dumps(verify(parser.parse_args().run), ensure_ascii=False,
                     sort_keys=True))
