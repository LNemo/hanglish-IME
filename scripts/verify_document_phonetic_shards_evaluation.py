#!/usr/bin/env python3
"""Independently verify the closed two-shard document evaluation.

Shard proofs and all capture hashes are authenticated before exact references
are joined.  This verifier imports neither the evaluation producer nor a model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

from scripts import verify_production_phonetic_choice_dev as pure


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".omx/artifacts/document-phonetic-shards-evaluation-v1"
REPORT = RUN / "report.json"
EVALUATOR = ROOT / "scripts/evaluate_document_phonetic_shards.py"
SHARD_VERIFIER = ROOT / "scripts/verify_document_phonetic_shard.py"
CAPTURE_PRODUCER = ROOT / "scripts/capture_document_phonetic_shard.py"
RAW = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
SELECTION = ROOT / ".omx/artifacts/document-reserve-v2/selection.jsonl"
SHARDS = ROOT / ".omx/artifacts/document-sentence-shards-v1"
CAPTURE_RUNS = [
    ROOT / ".omx/artifacts/document-phonetic-choice-shard-0-v1",
    ROOT / ".omx/artifacts/document-phonetic-choice-shard-1-v1",
]
SCHEMA = "document-phonetic-shards-evaluation-independent-verification-v1"
REPORT_SCHEMA = "document-phonetic-shards-evaluation-v1"
SHARD_PROOF_SCHEMA = "document-phonetic-shard-independent-verification-v1"
UNRESOLVED = "__FILL_AFTER_EVALUATOR_FREEZE__"
EVALUATOR_SHA256 = "62930ba83ef23785bf3c57f4898bbad3703c788323234fc498358389de5e0875"
SHARD_VERIFIER_SHA256 = "4e2fb296fca6f1f5025c278dfefb9c68e52c8dc3fc43ed6d5669b94be779f41a"
CAPTURE_PRODUCER_SHA256 = "412fb0bbdadab8931346ff529acbd04f6fb2d292d847d85b80c26e0b012bfbec"
RAW_SHA256 = "904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b"
SELECTION_SHA256 = "7d62027d11dcf1320b2a1bac7bcb83e1a31d31ce521808bbe4d1faadf6d4b810"
PARTITION_HASHES = {
    "manifest.json": "a09e13062f759ec8ac8d13b44344b63724e50f8d0302e4a945bfa13fdf797fc9",
    "shard-0.jsonl": "9c9e7f49f95d90f38d530cb0bc879eefe0be98c2fc9593e4aa24500129ffae48",
    "shard-1.jsonl": "3aeecc85a58186a4bbb9f0ac45edd52038488f31bc01b9749bb92aaddad04df5",
    "independent-verification.json":
        "f0eb4e1d4585b5c2d4873442186b896ad47f55c49324aa6c599f1cf3143376a3",
}


def resolved_evaluator(value=EVALUATOR_SHA256):
    pure.need(isinstance(value, str) and len(value) == 64 and value != UNRESOLVED,
              "evaluation producer identity is not frozen")
    return value


def response_outcome(result, gold):
    pure.need(isinstance(result, dict)
              and isinstance(result.get("candidates"), list)
              and 1 <= len(result["candidates"]) <= 5,
              "public candidates missing or over budget")
    texts = []
    for candidate in result["candidates"]:
        pure.need(isinstance(candidate, dict)
                  and isinstance(candidate.get("text"), str),
                  "candidate text differs")
        texts.append(candidate["text"])
    pure.need(result.get("result") == texts[0], "public selection differs")
    return gold == texts[0], gold in texts[:5]


def aggregate(records, labels):
    """Pure exact-string replay independent of the evaluation producer."""
    pure.need(isinstance(labels, dict) and labels
              and all(isinstance(key, str) and key
                      and isinstance(value, str) and value
                      for key, value in labels.items()),
              "invalid or empty label")
    counts = {name: Counter(rows=0, top1=0, top5=0)
              for name in ("baseline", "proposed")}
    changes = Counter(corrections=0, regressions=0, top5_losses=0)
    stages = Counter(ranked_rows=0, ranked_pool_gold=0,
                     ranked_candidate_missing=0, ranked_selection_failures=0,
                     unranked_rows=0, unranked_first_failures=0)
    english = Counter(rows_with_reference_ascii_words=0,
                      baseline_exact_signature=0, proposed_exact_signature=0,
                      signature_regressions=0, literal_signature_regressions=0)
    seen = set()
    latencies = []
    for record in records:
        pure.need(isinstance(record, dict)
                  and isinstance(record.get("id"), str)
                  and record["id"] in labels and record["id"] not in seen,
                  "unknown or duplicate captured ID")
        identifier = record["id"]
        seen.add(identifier)
        request = record.get("request")
        pure.need(isinstance(request, dict)
                  and set(request) == {"input", "context"}
                  and isinstance(request["input"], str) and request["input"]
                  and request["context"] == "", "capture request differs")
        source = request["input"]
        gold = labels[identifier]
        outcomes = {}
        for branch in counts:
            outcomes[branch] = response_outcome(record.get(branch), gold)
            counts[branch].update(
                rows=1, top1=int(outcomes[branch][0]),
                top5=int(outcomes[branch][1]))
        before, after = outcomes["baseline"], outcomes["proposed"]
        changes["corrections"] += not before[0] and after[0]
        changes["regressions"] += before[0] and not after[0]
        changes["top5_losses"] += before[1] and not after[1]

        ranked = record["proposed"].get("sentence_ranking_applied")
        pure.need(type(ranked) is bool, "sentence-ranking marker differs")
        paths = record.get("base_paths")
        pure.need(isinstance(paths, list), "base path pool differs")
        if ranked:
            pool = set()
            for path in paths:
                pure.need(isinstance(path, (list, tuple)) and path
                          and isinstance(path[0], str), "base path differs")
                pool.add(path[0])
            in_pool = gold in pool
            stages["ranked_rows"] += 1
            stages["ranked_pool_gold"] += in_pool
            stages["ranked_candidate_missing"] += not in_pool
            stages["ranked_selection_failures"] += in_pool and not after[0]
        else:
            pure.need(paths == [], "unranked base paths differ")
            stages["unranked_rows"] += 1
            stages["unranked_first_failures"] += not after[0]

        expected_ascii = tuple(pure.ASCII_WORD.findall(gold))
        if expected_ascii:
            english["rows_with_reference_ascii_words"] += 1
            old = tuple(pure.ASCII_WORD.findall(
                record["baseline"]["result"])) == expected_ascii
            new = tuple(pure.ASCII_WORD.findall(
                record["proposed"]["result"])) == expected_ascii
            english["baseline_exact_signature"] += old
            english["proposed_exact_signature"] += new
            english["signature_regressions"] += old and not new
        literal = pure.literal_signature(source)
        english["literal_signature_regressions"] += (
            pure.literal_signature(record["baseline"]["result"]) == literal
            and pure.literal_signature(record["proposed"]["result"]) != literal)

        latency = record["proposed"].get("engine_ms")
        pure.need(type(latency) in (int, float)
                  and math.isfinite(latency) and latency >= 0,
                  "invalid engine latency")
        latencies.append(latency)

    pure.need(seen == set(labels) and seen, "aggregate denominator differs")
    improvement = bool(changes["corrections"] and not changes["regressions"]
                       and not changes["top5_losses"])
    ordered = sorted(latencies)
    return {
        **{name: dict(value) for name, value in counts.items()},
        **dict(changes),
        "stage_counts": dict(stages),
        "english_ascii_signature": dict(english),
        "improvement_gate": improvement,
        "quality_gate": bool(improvement and not english["signature_regressions"]
                             and not english["literal_signature_regressions"]),
        "actual_calibrated_latency": {
            "p50_ms": ordered[math.ceil(.5 * len(ordered)) - 1],
            "p95_ms": ordered[math.ceil(.95 * len(ordered)) - 1],
            "max_ms": ordered[-1],
        },
    }


def join_labels(selection, raw_rows, requests, *, expected_rows=None,
                expected_groups=None, romanize=None):
    """Join exact labels only after callers have authenticated both captures."""
    if romanize is None:
        from hanglish.core.phonology import romanize_word
        romanize = romanize_word
    pure.need(isinstance(selection, list) and isinstance(raw_rows, list)
              and isinstance(requests, dict) and requests,
              "late reference inputs differ")
    if expected_rows is not None:
        pure.need(len(selection) == expected_rows and len(requests) == expected_rows,
                  "late reference denominator differs")
    selected = {}
    groups = set()
    for row in selection:
        pure.need(isinstance(row, dict) and set(row) == {"id", "group_id"}
                  and isinstance(row["id"], str) and row["id"]
                  and isinstance(row["group_id"], str) and row["group_id"]
                  and row["id"] not in selected,
                  "selection schema or identity differs")
        selected[row["id"]] = row["group_id"]
        groups.add(row["group_id"])
    pure.need(set(selected) == set(requests), "reference selection differs")
    if expected_groups is not None:
        pure.need(len(groups) == expected_groups, "reference group count differs")

    labels = {}
    for row in raw_rows:
        pure.need(isinstance(row, dict) and isinstance(row.get("id"), str),
                  "raw reference schema differs")
        if row["id"] not in selected:
            continue
        text = row.get("text")
        pure.need(row["id"] not in labels
                  and row.get("group_id") == selected[row["id"]]
                  and isinstance(text, str) and text
                  and romanize(text) == requests[row["id"]],
                  "late reference alignment differs")
        labels[row["id"]] = text
    pure.need(set(labels) == set(requests), "reference join incomplete")
    return labels


def authenticate_report(report_path, capture_runs=CAPTURE_RUNS,
                        evaluator_sha=None):
    """Authenticate both terminal source-only proofs before any label read."""
    expected_evaluator = resolved_evaluator(
        EVALUATOR_SHA256 if evaluator_sha is None else evaluator_sha)
    report_path = Path(report_path)
    pure.need(report_path.is_file() and EVALUATOR.is_file()
              and pure.sha(EVALUATOR) == expected_evaluator,
              "evaluation producer identity differs")
    pure.need(all((SHARDS / name).is_file()
                  and pure.sha(SHARDS / name) == digest
                  for name, digest in PARTITION_HASHES.items()),
              "source partition proof identity differs")
    # Reference files are hashed as opaque bytes here and are not deserialized.
    pure.need(pure.sha(RAW) == RAW_SHA256
              and pure.sha(SELECTION) == SELECTION_SHA256,
              "reference identity differs")
    report = pure.read(report_path)
    pure.need(report.get("schema") == REPORT_SCHEMA
              and report.get("completed") is True
              and report.get("rows") == 1046 and report.get("groups") == 821
              and report.get("producer_sha256") == expected_evaluator
              and report.get("reference_sha256") == RAW_SHA256
              and report.get("selection_sha256") == SELECTION_SHA256
              and report.get("labels_opened_after_both_verified_captures") is True
              and report.get("exact_original_reference_strings") is True
              and report.get("production_changed") is False
              and report.get("baseline_is_reconstructed_public_output") is True
              and report.get("baseline_latency_measured") is False
              and report.get("fully_blind_generalization") is False,
              "evaluation report identity/scope differs")
    evidence = report.get("shards")
    pure.need(isinstance(evidence, list) and len(evidence) == 2
              and len(capture_runs) == 2, "two shard proofs required")

    records = []
    requests = {}
    proof_hashes = []
    combined_elapsed = 0.0
    for index, (item, directory) in enumerate(zip(evidence, map(Path, capture_runs))):
        directory = directory.resolve()
        pure.need(not any((directory / name).exists() for name in
                          ("failure.json", "resource-watchdog-failure.json")),
                  "shard failure evidence exists")
        pure.need(isinstance(item, dict)
                  and item.get("path") == str(directory),
                  "shard evidence path differs")
        proof_path = directory / "independent-verification.json"
        proof_hash = pure.sha(proof_path)
        proof = pure.read(proof_path)
        expected_rows = (520, 526)[index]
        pure.need(item.get("proof_sha256") == proof_hash
                  and proof.get("schema") == SHARD_PROOF_SCHEMA
                  and proof.get("verified") is True
                  and proof.get("shard") == index
                  and proof.get("rows") == expected_rows
                  and proof.get("partition_hashes") == PARTITION_HASHES
                  and proof.get("performance_labels_opened") is False
                  and proof.get("accuracy_measured") is False,
                  "terminal shard proof differs")
        pure.need(proof.get("verifier_sha256") == SHARD_VERIFIER_SHA256
                  == pure.sha(SHARD_VERIFIER)
                  and proof.get("producer_sha256") == CAPTURE_PRODUCER_SHA256
                  == pure.sha(CAPTURE_PRODUCER),
                  "shard producer/verifier identity differs")
        for field, name in (("report_sha256", "report.json"),
                            ("capture_sha256", "capture.jsonl"),
                            ("closure_sha256", "capture.complete.json"),
                            ("completion_sha256", "completion.json")):
            pure.need(item.get(field) == proof.get(field)
                      == pure.sha(directory / name),
                      "shard closure hash differs")
        resources = proof.get("resources")
        completion_resource = (resources.get("after_completion", {})
                               if isinstance(resources, dict) else {})
        elapsed = completion_resource.get("elapsed_s")
        peak_rss = resources.get("peak_rss_bytes") if isinstance(resources, dict) else None
        peak_swap = (resources.get("peak_swap_growth_bytes")
                     if isinstance(resources, dict) else None)
        pure.need(item.get("resources") == resources
                  and isinstance(resources, dict)
                  and isinstance(completion_resource, dict)
                  and type(elapsed) in (int, float) and math.isfinite(elapsed)
                  and 0 <= elapsed <= 1800
                  and type(peak_rss) is int and 0 <= peak_rss <= 3 * 1024**3
                  and type(peak_swap) is int
                  and 0 <= peak_swap <= 512 * 1024**2,
                  "shard resource evidence differs")
        combined_elapsed += elapsed
        shard_report = pure.read(directory / "report.json")
        pure.need(shard_report.get("completed") is True
                  and shard_report.get("shard") == index
                  and shard_report.get("rows") == expected_rows
                  and shard_report.get("partition_hashes") == PARTITION_HASHES
                  and shard_report.get("version") == report.get("version")
                  and shard_report.get("weights") == report.get("weights")
                  and shard_report.get("performance_labels_opened") is False
                  and shard_report.get("accuracy_measured") is False,
                  "closed shard report differs")
        pins = shard_report.get("pins")
        pure.need(isinstance(pins, dict) and pins
                  and pins.get(str(SHARD_VERIFIER.resolve()))
                      == SHARD_VERIFIER_SHA256
                  and pins.get(str(CAPTURE_PRODUCER.resolve()))
                      == CAPTURE_PRODUCER_SHA256
                  and all(pins.get(str((SHARDS / name).resolve())) == digest
                          for name, digest in PARTITION_HASHES.items()),
                  "critical shard runtime pins differ")
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
        operations = ["authenticate_sources", "load_models",
                      *("predict:" + row["id"] for row in shard_sources),
                      "close_capture", "write_report", "write_completion"]
        replayed_resources = pure.verify_resources(
            list(pure.jsonl(directory / "resources.jsonl")), operations)
        pure.need(replayed_resources == resources,
                  "shard resource trace differs")
        proof_hashes.append(proof_hash)
    pure.need(len(records) == len(requests) == 1046,
              "combined capture denominator differs")
    pure.need(combined_elapsed <= 3600, "combined shard time differs")
    source_proofs = {
        "evaluation_report_sha256": pure.sha(report_path),
        "evaluation_producer_sha256": expected_evaluator,
        "partition_proof_sha256": PARTITION_HASHES["independent-verification.json"],
        "shard_proof_sha256": proof_hashes,
        "raw_reference_sha256": RAW_SHA256,
        "selection_sha256": SELECTION_SHA256,
    }
    return report, records, requests, source_proofs


def validate_metrics(report, metrics):
    keys = ("baseline", "proposed", "corrections", "regressions",
            "top5_losses", "stage_counts", "english_ascii_signature",
            "improvement_gate", "quality_gate", "actual_calibrated_latency")
    pure.need(all(report.get(key) == metrics.get(key) for key in keys),
              "independent aggregate differs")


def verify(report_path=REPORT, output=None):
    report_path = Path(report_path)
    proof_path = Path(output) if output is not None else (
        report_path.parent / "independent-verification.json")
    pure.need(not proof_path.exists(), "evaluation proof already exists")
    report, records, requests, source_proofs = authenticate_report(report_path)
    # This is the first reference deserialization, after both terminal proofs.
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
            "both_terminal_shard_proofs_before_reference_deserialization": True,
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
