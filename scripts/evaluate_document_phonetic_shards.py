"""Join document references only after both complete shard captures are verified."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_production_phonetic_choice_dev import (
    need, sha, read, jsonl, ASCII_WORD, literal_signature, verify_resources,
)

SCHEMA = "document-phonetic-shards-evaluation-v1"
SHARDS = ROOT / ".omx/artifacts/document-sentence-shards-v1"
RAW = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
SELECTION = ROOT / ".omx/artifacts/document-reserve-v2/selection.jsonl"
RAW_SHA = "904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b"
SELECTION_SHA = "7d62027d11dcf1320b2a1bac7bcb83e1a31d31ce521808bbe4d1faadf6d4b810"
VERIFIER = ROOT / "scripts/verify_document_phonetic_shard.py"
CAPTURE_PRODUCER = ROOT / "scripts/capture_document_phonetic_shard.py"


def authenticate_runs(directories):
    """No reference deserialization is permitted in this function."""
    need(len(directories) == 2, "both shard directories required")
    evidence, requests = [], {}
    version = weights = partition = None
    for index, directory in enumerate(map(Path, directories)):
        need(not any((directory / name).exists() for name in
                     ("failure.json", "resource-watchdog-failure.json")),
             "capture has failure evidence")
        report, proof = read(directory / "report.json"), read(directory / "independent-verification.json")
        completion = read(directory / "completion.json")
        need(report.get("completed") is True and report.get("shard") == index
             and report.get("schema") == "document-phonetic-shard-capture-v1"
             and report.get("accuracy_measured") is False
             and report.get("performance_labels_opened") is False
             and proof.get("verified") is True
             and proof.get("schema") == "document-phonetic-shard-independent-verification-v1"
             and completion.get("completed") is True,
             "capture completion or independent verification differs")
        fields = {"report.json": "report_sha256", "capture.jsonl": "capture_sha256",
                  "capture.complete.json": "closure_sha256", "completion.json": "completion_sha256"}
        for name, field in fields.items():
            need(sha(directory / name) == proof.get(field), "closed artifact differs")
        need(completion["report_sha256"] == proof["report_sha256"], "completion binding differs")
        need(proof.get("shard") == index and proof.get("rows") == report["rows"]
             and proof.get("verifier_sha256") == sha(VERIFIER)
             == report["pins"].get(str(VERIFIER.resolve()))
             and report["pins"].get(str(CAPTURE_PRODUCER.resolve())) == sha(CAPTURE_PRODUCER),
             "capture producer/verifier identity differs")
        for path, digest in report["pins"].items():
            need(sha(path) == digest, "runtime pin changed")
        for name, digest in report["partition_hashes"].items():
            need(sha(SHARDS / name) == digest, "partition artifact changed")
        if index == 0:
            version, weights, partition = report["version"], report["weights"], report["partition_hashes"]
        need(report["version"] == version and report["weights"] == weights
             and report["partition_hashes"] == partition, "models or partitions differ")
        launch = read(directory.with_suffix(".launch.json"))
        expected_command = [str(ROOT / ".venv/bin/python"), str(CAPTURE_PRODUCER),
                            "--output", str(directory.resolve()), "--shard", str(index)]
        if index:
            expected_command += ["--previous", str(Path(directories[0]).resolve())]
            need(report["pins"].get(str((Path(directories[0]) / "independent-verification.json").resolve()))
                 == evidence[0]["proof_sha256"], "predecessor proof not bound by second run")
        need(proof.get("launch_sha256") == sha(directory.with_suffix(".launch.json"))
             and launch.get("runner_sha256") == sha(CAPTURE_PRODUCER)
             and launch.get("output") == str(directory.resolve())
             and launch.get("command") == expected_command, "launch identity differs")
        need(type(launch.get("pid")) is int and launch["pid"] > 0, "launch PID absent")
        try:
            os.kill(launch["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError("capture process still present")
        inputs = list(jsonl(SHARDS / f"shard-{index}.jsonl"))
        need(report["rows"] == len(inputs), "shard denominator changed")
        observed_ids = []
        for row in jsonl(directory / "capture.jsonl"):
            identifier = row["id"]
            need(identifier not in requests and row["request"]["context"] == "", "duplicate or contextual request")
            requests[identifier] = row["request"]["input"]
            observed_ids.append(identifier)
        need(observed_ids == [row["id"] for row in inputs]
             and all(requests[row["id"]] == row["input"] for row in inputs), "capture source order differs")
        resource = proof["resources"]
        replayed = verify_resources(list(jsonl(directory / "resources.jsonl")),
            ["authenticate_sources", "load_models", *("predict:" + r["id"] for r in inputs),
             "close_capture", "write_report", "write_completion"])
        need(resource == replayed, "independent resource summary differs")
        need(resource["after_completion"]["elapsed_s"] <= 1800
             and resource["peak_rss_bytes"] <= 3 * 1024**3
             and resource["peak_swap_growth_bytes"] <= 512 * 1024**2,
             "shard resource limit differs")
        evidence.append({"path": str(directory.resolve()), "proof_sha256": sha(directory / "independent-verification.json"),
                         **{field: proof[field] for field in fields.values()}, "resources": resource})
    need(len(requests) == 1046 and sum(r["resources"]["after_completion"]["elapsed_s"]
                                      for r in evidence) <= 3600, "combined denominator or time differs")
    return evidence, requests, version, weights


def load_references(requests):
    need(sha(RAW) == RAW_SHA and sha(SELECTION) == SELECTION_SHA, "reference identity differs")
    selection = list(jsonl(SELECTION))
    need(len(selection) == 1046 and len({r["group_id"] for r in selection}) == 821
         and {r["id"] for r in selection} == set(requests), "reference selection differs")
    selected = {row["id"]: row["group_id"] for row in selection}
    labels = {}
    from hanglish.core.phonology import romanize_word
    for row in jsonl(RAW):
        if row["id"] not in selected:
            continue
        need(row["id"] not in labels and row["group_id"] == selected[row["id"]]
             and isinstance(row.get("text"), str) and bool(row["text"])
             and romanize_word(row["text"]) == requests[row["id"]], "late reference alignment differs")
        labels[row["id"]] = row["text"]
    need(set(labels) == set(requests), "reference join incomplete")
    return labels


def aggregate(records, labels):
    need(isinstance(labels, dict) and labels
         and all(isinstance(key, str) and key and isinstance(value, str) and value
                 for key, value in labels.items()), "invalid or empty label")
    counts = {name: Counter(rows=0, top1=0, top5=0) for name in ("baseline", "proposed")}
    changes = Counter(corrections=0, regressions=0, top5_losses=0)
    stages = Counter(ranked_rows=0, ranked_pool_gold=0, ranked_candidate_missing=0,
                     ranked_selection_failures=0, unranked_rows=0, unranked_first_failures=0)
    english = Counter(rows_with_reference_ascii_words=0, baseline_exact_signature=0,
                      proposed_exact_signature=0, signature_regressions=0, literal_signature_regressions=0)
    seen, elapsed = set(), []
    for row in records:
        identifier = row["id"]
        need(identifier in labels and identifier not in seen, "unknown or duplicate captured ID")
        seen.add(identifier)
        gold, source = labels[identifier], row["request"]["input"]
        outcome = {}
        for branch in counts:
            result = row[branch]
            texts = [c["text"] for c in result["candidates"]]
            need(texts and result["result"] == texts[0], "public selection differs")
            outcome[branch] = (gold == texts[0], gold in texts[:5])
            counts[branch].update(rows=1, top1=int(outcome[branch][0]), top5=int(outcome[branch][1]))
        before, after = outcome["baseline"], outcome["proposed"]
        changes["corrections"] += not before[0] and after[0]
        changes["regressions"] += before[0] and not after[0]
        changes["top5_losses"] += before[1] and not after[1]
        if row["proposed"]["sentence_ranking_applied"]:
            in_pool = gold in {p[0] for p in row["base_paths"]}
            stages["ranked_rows"] += 1
            stages["ranked_pool_gold"] += in_pool
            stages["ranked_candidate_missing"] += not in_pool
            stages["ranked_selection_failures"] += in_pool and not after[0]
        else:
            stages["unranked_rows"] += 1
            stages["unranked_first_failures"] += not after[0]
        expected_ascii = tuple(ASCII_WORD.findall(gold))
        if expected_ascii:
            english["rows_with_reference_ascii_words"] += 1
            old = tuple(ASCII_WORD.findall(row["baseline"]["result"])) == expected_ascii
            new = tuple(ASCII_WORD.findall(row["proposed"]["result"])) == expected_ascii
            english["baseline_exact_signature"] += old
            english["proposed_exact_signature"] += new
            english["signature_regressions"] += old and not new
        literal = literal_signature(source)
        english["literal_signature_regressions"] += (
            literal_signature(row["baseline"]["result"]) == literal
            and literal_signature(row["proposed"]["result"]) != literal)
        ms = row["proposed"]["engine_ms"]
        need(type(ms) in (int, float) and math.isfinite(ms) and ms >= 0, "invalid engine latency")
        elapsed.append(ms)
    need(seen == set(labels) and seen, "aggregate denominator differs")
    improvement = bool(changes["corrections"] and not changes["regressions"] and not changes["top5_losses"])
    times = sorted(elapsed)
    return {**{name: dict(value) for name, value in counts.items()}, **dict(changes),
            "stage_counts": dict(stages), "english_ascii_signature": dict(english),
            "improvement_gate": improvement,
            "quality_gate": improvement and not english["signature_regressions"] and not english["literal_signature_regressions"],
            "actual_calibrated_latency": {"p50_ms": times[math.ceil(.5 * len(times))-1],
                                          "p95_ms": times[math.ceil(.95 * len(times))-1], "max_ms": times[-1]}}


def run(directories, output):
    need(not Path(output).exists(), "evaluation output exists")
    evidence, requests, version, weights = authenticate_runs(directories)
    # Both predictions and pure proofs have closed before this first label read.
    labels = load_references(requests)
    records = (row for directory in directories for row in jsonl(Path(directory) / "capture.jsonl"))
    metrics = aggregate(records, labels)
    report = {"schema": SCHEMA, "completed": True, "version": version, "weights": weights,
              "rows": 1046, "groups": 821, "shards": evidence, **metrics,
              "labels_opened_after_both_verified_captures": True,
              "exact_original_reference_strings": True, "production_changed": False,
              "baseline_is_reconstructed_public_output": True, "baseline_latency_measured": False,
              "fully_blind_generalization": False,
              "limitations": ["Synthetic Roman inputs; word-level evaluation previously exposed this reserve.",
                              "No lexical-family or external pretraining disjointness claim.",
                              "ASCII signatures are diagnostics, not complete copy-alignment validation."],
              "producer_sha256": sha(Path(__file__)), "reference_sha256": RAW_SHA,
              "selection_sha256": SELECTION_SHA}
    with Path(output).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return {key: report[key] for key in ("completed", "baseline", "proposed", "corrections", "regressions", "quality_gate")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs=2, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.shards, args.output)))
