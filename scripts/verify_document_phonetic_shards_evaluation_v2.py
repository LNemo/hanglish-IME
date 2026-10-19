#!/usr/bin/env python3
"""Verify the V2 mixed-audit two-shard document evaluation.

Both captures used the frozen V1 runtime verifier pin.  Shard 0 was audited by
that V1 verifier, while shard 1 was audited by the versioned V2 verifier that
understands the mandatory predecessor launch argument.  References are opened
only after both proof chains are authenticated.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from scripts import verify_document_phonetic_shards_evaluation as helper


pure = helper.pure
aggregate = helper.aggregate
join_labels = helper.join_labels
validate_metrics = helper.validate_metrics

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".omx/artifacts/document-phonetic-shards-evaluation-v2"
REPORT = RUN / "report.json"
EVALUATOR = ROOT / "scripts/evaluate_document_phonetic_shards_v2.py"
METRIC_HELPER = Path(helper.__file__).resolve()
RAW = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
SELECTION = ROOT / ".omx/artifacts/document-reserve-v2/selection.jsonl"
SHARDS = ROOT / ".omx/artifacts/document-sentence-shards-v1"
CAPTURE_PRODUCER = ROOT / "scripts/capture_document_phonetic_shard.py"
RUNTIME_VERIFIER = ROOT / "scripts/verify_document_phonetic_shard.py"
AUDIT_VERIFIERS = {
    0: RUNTIME_VERIFIER,
    1: ROOT / "scripts/verify_document_phonetic_shard_v2.py",
}
CAPTURE_RUNS = [
    ROOT / ".omx/artifacts/document-phonetic-choice-shard-0-v1",
    ROOT / ".omx/artifacts/document-phonetic-choice-shard-1-v1",
]
SCHEMA = "document-phonetic-shards-evaluation-independent-verification-v2"
REPORT_SCHEMA = "document-phonetic-shards-evaluation-v2"
SHARD_PROOF_SCHEMA = "document-phonetic-shard-independent-verification-v1"
EVALUATOR_SHA256 = "9deaa671782926b11329cc5de44118215a195fc91314b323cf8ab68ccc95c1f9"
METRIC_HELPER_SHA256 = "0159d82a7b99032669145d9647830efa6a405aba96e247303ac4babe80f048cb"
CAPTURE_PRODUCER_SHA256 = "412fb0bbdadab8931346ff529acbd04f6fb2d292d847d85b80c26e0b012bfbec"
RUNTIME_VERIFIER_SHA256 = "4e2fb296fca6f1f5025c278dfefb9c68e52c8dc3fc43ed6d5669b94be779f41a"
AUDIT_VERIFIER_SHA256 = {
    0: RUNTIME_VERIFIER_SHA256,
    1: "fd0a1452219fa2a4e8716da30b649b03c6d35b3c15247df1af143b492fb5f335",
}
RAW_SHA256 = helper.RAW_SHA256
SELECTION_SHA256 = helper.SELECTION_SHA256
PARTITION_HASHES = dict(helper.PARTITION_HASHES)


def validate_report_scope(report):
    pure.need(isinstance(report, dict)
              and report.get("schema") == REPORT_SCHEMA
              and report.get("completed") is True
              and report.get("rows") == 1046 and report.get("groups") == 821
              and report.get("producer_sha256") == EVALUATOR_SHA256
              and report.get("reference_sha256") == RAW_SHA256
              and report.get("selection_sha256") == SELECTION_SHA256
              and report.get("audit_verifier_sha256_by_shard")
                  == {str(index): digest
                      for index, digest in AUDIT_VERIFIER_SHA256.items()}
              and report.get("runtime_verifier_sha256")
                  == RUNTIME_VERIFIER_SHA256
              and report.get("labels_opened_after_both_verified_captures") is True
              and report.get("exact_original_reference_strings") is True
              and report.get("production_changed") is False
              and report.get("baseline_is_reconstructed_public_output") is True
              and report.get("baseline_latency_measured") is False
              and report.get("fully_blind_generalization") is False,
              "V2 evaluation report identity/scope differs")


def validate_audit_identity(index, proof, shard_report, report):
    expected_rows = (520, 526)[index]
    pure.need(isinstance(proof, dict)
              and proof.get("schema") == SHARD_PROOF_SCHEMA
              and proof.get("verified") is True
              and proof.get("shard") == index
              and proof.get("rows") == expected_rows
              and proof.get("producer_sha256") == CAPTURE_PRODUCER_SHA256
              and proof.get("verifier_sha256") == AUDIT_VERIFIER_SHA256[index]
              and proof.get("partition_hashes") == PARTITION_HASHES
              and proof.get("performance_labels_opened") is False
              and proof.get("accuracy_measured") is False,
              "mixed shard audit proof differs")
    pins = shard_report.get("pins") if isinstance(shard_report, dict) else None
    pure.need(isinstance(pins, dict) and pins
              and pins.get(str(RUNTIME_VERIFIER.resolve()))
                  == RUNTIME_VERIFIER_SHA256
              and pins.get(str(CAPTURE_PRODUCER.resolve()))
                  == CAPTURE_PRODUCER_SHA256
              and report.get("runtime_verifier_sha256")
                  == RUNTIME_VERIFIER_SHA256,
              "frozen runtime verifier pin differs")
    if index == 1:
        pure.need(proof.get("runtime_verifier_path")
                      == str(RUNTIME_VERIFIER.resolve())
                  and proof.get("runtime_verifier_sha256")
                      == RUNTIME_VERIFIER_SHA256,
                  "V2 audit/runtime verifier distinction differs")


def authenticate_report(report_path, capture_runs=CAPTURE_RUNS):
    """Authenticate mixed audits and runtime evidence without opening labels."""
    report_path = Path(report_path)
    identities = {
        EVALUATOR: EVALUATOR_SHA256,
        METRIC_HELPER: METRIC_HELPER_SHA256,
        CAPTURE_PRODUCER: CAPTURE_PRODUCER_SHA256,
        RUNTIME_VERIFIER: RUNTIME_VERIFIER_SHA256,
        AUDIT_VERIFIERS[1]: AUDIT_VERIFIER_SHA256[1],
    }
    pure.need(report_path.is_file()
              and all(path.is_file() and pure.sha(path) == digest
                      for path, digest in identities.items()),
              "V2 evaluator or verifier source identity differs")
    pure.need(all((SHARDS / name).is_file()
                  and pure.sha(SHARDS / name) == digest
                  for name, digest in PARTITION_HASHES.items()),
              "source partition proof identity differs")
    # These references remain opaque hashes until both shard loops finish.
    pure.need(pure.sha(RAW) == RAW_SHA256
              and pure.sha(SELECTION) == SELECTION_SHA256,
              "reference identity differs")
    report = pure.read(report_path)
    validate_report_scope(report)
    evidence = report.get("shards")
    pure.need(isinstance(evidence, list) and len(evidence) == 2
              and len(capture_runs) == 2, "two shard proofs required")

    records = []
    requests = {}
    proof_hashes = []
    combined_elapsed = 0.0
    for index, (item, raw_directory) in enumerate(zip(evidence, capture_runs)):
        directory = Path(raw_directory).resolve()
        pure.need(not any((directory / name).exists() for name in
                          ("failure.json", "resource-watchdog-failure.json")),
                  "shard failure evidence exists")
        pure.need(isinstance(item, dict) and item.get("path") == str(directory),
                  "shard evidence path differs")
        proof_path = directory / "independent-verification.json"
        proof_hash = pure.sha(proof_path)
        proof = pure.read(proof_path)
        shard_report = pure.read(directory / "report.json")
        validate_audit_identity(index, proof, shard_report, report)
        expected_rows = (520, 526)[index]
        pure.need(item.get("proof_sha256") == proof_hash,
                  "shard proof hash differs")
        for field, name in (("report_sha256", "report.json"),
                            ("capture_sha256", "capture.jsonl"),
                            ("closure_sha256", "capture.complete.json"),
                            ("completion_sha256", "completion.json")):
            pure.need(item.get(field) == proof.get(field)
                      == pure.sha(directory / name),
                      "shard closure hash differs")
        pure.need(shard_report.get("completed") is True
                  and shard_report.get("shard") == index
                  and shard_report.get("rows") == expected_rows
                  and shard_report.get("partition_hashes") == PARTITION_HASHES
                  and shard_report.get("version") == report.get("version")
                  and shard_report.get("weights") == report.get("weights")
                  and shard_report.get("performance_labels_opened") is False
                  and shard_report.get("accuracy_measured") is False,
                  "closed shard report differs")
        pins = shard_report["pins"]
        pure.need(all(pins.get(str((SHARDS / name).resolve())) == digest
                      for name, digest in PARTITION_HASHES.items()),
                  "critical partition runtime pins differ")
        for raw, digest in pins.items():
            pure.need(isinstance(raw, str) and raw
                      and isinstance(digest, str) and len(digest) == 64
                      and pure.sha(raw) == digest,
                      "shard runtime pin changed")
        if index:
            predecessor = Path(capture_runs[0]) / "independent-verification.json"
            pure.need(pins.get(str(predecessor.resolve())) == proof_hashes[0],
                      "second shard predecessor proof differs")

        launch_path = directory.with_suffix(".launch.json")
        launch = pure.read(launch_path)
        command = [str(ROOT / ".venv/bin/python"), str(CAPTURE_PRODUCER),
                   "--output", str(directory), "--shard", str(index)]
        if index:
            command += ["--previous", str(Path(capture_runs[0]).resolve())]
        pure.need(proof.get("launch_sha256") == pure.sha(launch_path)
                  and launch.get("runner_sha256") == CAPTURE_PRODUCER_SHA256
                  and launch.get("output") == str(directory)
                  and launch.get("command") == command
                  and type(launch.get("pid")) is int and launch["pid"] > 0,
                  "terminal shard launch differs")
        try:
            os.kill(launch["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("shard capture process still present")

        shard_records = pure.read_exact_jsonl(
            directory / "capture.jsonl", expected_rows)
        shard_sources = pure.read_exact_jsonl(
            SHARDS / f"shard-{index}.jsonl", expected_rows)
        for expected, record in zip(shard_sources, shard_records):
            pure.need(record.get("id") == expected.get("id")
                      and record.get("request")
                          == {"input": expected.get("input"), "context": ""}
                      and record["id"] not in requests,
                      "capture source order or identity differs")
            requests[record["id"]] = record["request"]["input"]
        records.extend(shard_records)

        resources = proof.get("resources")
        completion = resources.get("after_completion", {}) \
            if isinstance(resources, dict) else {}
        elapsed = completion.get("elapsed_s")
        peak_rss = resources.get("peak_rss_bytes") \
            if isinstance(resources, dict) else None
        peak_swap = resources.get("peak_swap_growth_bytes") \
            if isinstance(resources, dict) else None
        pure.need(item.get("resources") == resources
                  and type(elapsed) in (int, float) and math.isfinite(elapsed)
                  and 0 <= elapsed <= 1800
                  and type(peak_rss) is int and 0 <= peak_rss <= 3 * 1024**3
                  and type(peak_swap) is int
                  and 0 <= peak_swap <= 512 * 1024**2,
                  "shard resource evidence differs")
        operations = ["authenticate_sources", "load_models",
                      *("predict:" + row["id"] for row in shard_sources),
                      "close_capture", "write_report", "write_completion"]
        replayed = pure.verify_resources(
            list(pure.jsonl(directory / "resources.jsonl")), operations)
        pure.need(replayed == resources, "shard resource trace differs")
        combined_elapsed += elapsed
        proof_hashes.append(proof_hash)

    pure.need(len(records) == len(requests) == 1046,
              "combined capture denominator differs")
    pure.need(combined_elapsed <= 3600, "combined shard time differs")
    source_proofs = {
        "evaluation_report_sha256": pure.sha(report_path),
        "evaluation_producer_sha256": EVALUATOR_SHA256,
        "metric_helper_sha256": METRIC_HELPER_SHA256,
        "runtime_verifier_sha256": RUNTIME_VERIFIER_SHA256,
        "audit_verifier_sha256_by_shard": {
            str(index): digest for index, digest in AUDIT_VERIFIER_SHA256.items()},
        "partition_proof_sha256": PARTITION_HASHES["independent-verification.json"],
        "shard_proof_sha256": proof_hashes,
        "raw_reference_sha256": RAW_SHA256,
        "selection_sha256": SELECTION_SHA256,
    }
    return report, records, requests, source_proofs


def verify(report_path=REPORT, output=None):
    report_path = Path(report_path)
    proof_path = Path(output) if output is not None else (
        report_path.parent / "independent-verification.json")
    pure.need(not proof_path.exists(), "evaluation proof already exists")
    report, records, requests, source_proofs = authenticate_report(report_path)
    selection = pure.read_exact_jsonl(SELECTION, 1046)
    raw_rows = list(pure.jsonl(RAW))
    labels = join_labels(selection, raw_rows, requests,
                         expected_rows=1046, expected_groups=821)
    metrics = aggregate(records, labels)
    validate_metrics(report, metrics)
    proof = {
        "schema": SCHEMA,
        "verified": True,
        "counts": {"rows": 1046, "groups": 821, "shards": 2},
        "metrics": metrics,
        "source_proofs": source_proofs,
        "verifier_sha256": pure.sha(Path(__file__)),
        "contracts_verified": {
            "both_terminal_mixed_audit_proofs_before_reference_deserialization": True,
            "frozen_v1_runtime_verifier_distinguished_from_v2_shard1_audit": True,
            "exact_original_reference_strings_without_normalization": True,
            "independent_first_top5_change_stage_ascii_literal_and_latency_replay": True,
            "individual_failures_not_emitted": True,
            "model_tensor_or_neural_inference_run_by_verifier": False,
        },
        "limitations": [
            "Roman inputs are synthetic outputs of the pinned current romanizer.",
            "This document reserve was partially exposed through prior word-level evaluation.",
            "Document grouping does not establish lexical-family or external-pretraining disjointness.",
        ],
    }
    with proof_path.open("xb") as stream:
        stream.write(pure.canonical(proof))
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(verify(arguments.report, arguments.output),
                     ensure_ascii=False, sort_keys=True))
