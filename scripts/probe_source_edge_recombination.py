"""Model-free source-edge proposal capture followed by DEV coverage measurement."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INPUT = ROOT / ".omx/artifacts/extended-runtime-dev-v2"
DEV = ROOT / "data/challenge/sentence-v3/dev.jsonl"
PLAN = ROOT / ".omx/plans/source-edge-recombination-v1.md"
MODULE = ROOT / "hanglish/core/source_edge_recombination.py"
INPUT_HASHES = {
    "report.json": "64fa054990b0dc78bde0bb20e5de0ee89fa7ae910ff8fc281cb16bf607387227",
    "capture.jsonl": "4e329278d18ed4ac0081be2dd2ac41261a1762e68c7a5691f778f07ce897b2f0",
    "capture.complete.json": "d8982c836832e07ba7b2bff07b01841f275154e350adfb4e232419f829ac92e1",
    "independent-verification.json": "42d16c9669af47bc3483c79eb1d171d70ec7d2daee2bbc01a198fb65c0c9ca91",
}
SCHEMA = "source-edge-recombination-probe-v1"
LIMITS = {"whole_seconds": 300, "row_seconds": 30, "rss_bytes": 768 * 1024**2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def source_request(row, seen):
    identifier = row.get("id")
    require(isinstance(identifier, str) and identifier not in seen, "duplicate/missing ID")
    request = row.get("request", {})
    source = request.get("input")
    require(isinstance(source, str) and len(source.split()) >= 2
            and request == {"input": source, "context": ""}
            and row.get("proposed", {}).get("input") == source
            and row["proposed"].get("context") == "", "source-only request differs")
    seen.add(identifier)
    return identifier, source


def coverage(inputs, captures, labels):
    result = {"rows": 0, "baseline": 0, "union": 0, "gained_ids": [], "lost_ids": []}
    for original, capture, label in zip_longest(inputs, captures, labels):
        require(all(value is not None for value in (original, capture, label)),
                "coverage denominator differs")
        require(original["id"] == capture["id"] == label["id"]
                and original["request"] == capture["request"]
                == {"input": label["input"], "context": ""}, "coverage identity differs")
        before = {item["text"] for item in original["proposed"]["pipeline"]["pool"]}
        added = {item["text"] for item in capture["proposals"]}
        expected = set(label["expected"])
        old, new = bool(before & expected), bool((before | added) & expected)
        result["rows"] += 1; result["baseline"] += old; result["union"] += new
        if new and not old:
            result["gained_ids"].append(label["id"])
        if old and not new:
            result["lost_ids"].append(label["id"])
    return result


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def run(output):
    from hanglish.core.source_edge_recombination import propose

    for name, digest in INPUT_HASHES.items():
        require(sha(INPUT / name) == digest, f"input hash differs: {name}")
    parent = json.loads((INPUT / "report.json").read_text())
    proof = json.loads((INPUT / "independent-verification.json").read_text())
    require(parent["completed"] and proof["verified"]
            and proof["metrics"]["full_pool_coverage"]["proposed"] == 286,
            "input qualification differs")
    # The label file is not opened until all proposals have been closed.
    label_digest = parent["pins"][str(DEV)]
    tracked = [Path(__file__).resolve(), MODULE, PLAN]
    pins = {str(path): sha(path) for path in tracked}
    pins.update({str(INPUT / name): digest for name, digest in INPUT_HASHES.items()})
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write_json(output / "pins.json", pins)
    started = time.monotonic(); count = 0; proposal_count = 0; max_row_s = 0.0
    seen = set(); reasons = Counter()
    try:
        with (output / "capture.jsonl").open("x", encoding="utf-8") as stream:
            for original in rows(INPUT / "capture.jsonl"):
                identifier, source = source_request(original, seen)
                row_started = time.monotonic()
                response = original["proposed"]
                generated = propose(source, response["pipeline"]["pool"],
                                    response["candidates"])
                duration = time.monotonic() - row_started
                max_row_s = max(max_row_s, duration)
                proposals = generated["proposals"]
                require(len(proposals) <= 96, "proposal bound exceeded")
                record = {"id": identifier, "request": original["request"],
                          **generated, "elapsed_s": duration}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush(); count += 1; proposal_count += len(proposals)
                for seed in generated["diagnostics"]["seeds"]:
                    reasons[seed["termination_reason"]] += 1
                require(duration <= LIMITS["row_seconds"]
                        and time.monotonic() - started <= LIMITS["whole_seconds"]
                        and peak_rss() <= LIMITS["rss_bytes"], "probe resource limit exceeded")
                if count % 25 == 0:
                    print(json.dumps({"captured": count, "rows": 316,
                                      "elapsed_s": time.monotonic() - started}), flush=True)
            require(count == 316, "input denominator differs")
            os.fsync(stream.fileno())
        capture_digest = sha(output / "capture.jsonl")
        write_json(output / "capture.complete.json", {
            "rows": count, "capture_sha256": capture_digest,
            "source_only": True, "targets_joined": False})
        require(sha(DEV) == label_digest, "DEV target identity differs")
        measured = coverage(rows(INPUT / "capture.jsonl"), rows(output / "capture.jsonl"), rows(DEV))
        require(measured["baseline"] == 286, "baseline coverage differs")
        for path, digest in pins.items():
            require(sha(path) == digest, f"pinned input changed: {path}")
        require(time.monotonic() - started <= LIMITS["whole_seconds"]
                and peak_rss() <= LIMITS["rss_bytes"], "final probe resource limit exceeded")
        report = {"schema": SCHEMA, "completed": True, "rows": count,
                  "coverage": measured, "proposal_count": proposal_count,
                  "seed_termination_reasons": dict(reasons), "pins": pins,
                  "dev_labels_sha256": label_digest, "capture_sha256": capture_digest,
                  "closure_sha256": sha(output / "capture.complete.json"),
                  "candidate_improvement_gate": bool(measured["gained_ids"])
                  and not measured["lost_ids"], "labels_opened_after_capture": True,
                  "model_loaded": False, "model_inference": False, "test_opened": False,
                  "production_changed": False, "limits": LIMITS,
                  "elapsed_s": time.monotonic() - started, "max_row_s": max_row_s,
                  "peak_rss_bytes": peak_rss()}
        write_json(output / "report.json", report)
        return report
    except BaseException as error:
        write_json(output / "report.json", {"schema": SCHEMA, "completed": False,
                   "error": f"{type(error).__name__}: {error}", "captured_rows": count,
                   "model_loaded": False, "model_inference": False, "test_opened": False,
                   "production_changed": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); report = run(args.output)
    print(json.dumps({key: report[key] for key in (
        "completed", "coverage", "candidate_improvement_gate", "elapsed_s", "peak_rss_bytes")},
        ensure_ascii=False), flush=True)
