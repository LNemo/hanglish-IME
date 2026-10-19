"""Prepare a TRAIN-only source-edge recombination ranking dataset."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.source_edge_recombination import propose


CAPTURE = ROOT / ".omx/artifacts/roman-sentence-train-capture-v1/capture.jsonl"
CAPTURE_REPORT = ROOT / ".omx/artifacts/roman-sentence-train-capture-v1/report.json"
CAPTURE_PROOF = ROOT / ".omx/artifacts/roman-sentence-train-capture-v1/independent-verification.json"
TRAIN = ROOT / "data/challenge/sentence-v3/train.jsonl"
HELPER = ROOT / "hanglish/core/source_edge_recombination.py"
CAPTURE_SHA256 = "bfd88590669a550a7747df564a5792b9541f99e3255ab47d74c64e61d97f1c69"
REPORT_SHA256 = "f67fc884aacfa13122f363a0d00fe914b285fcd0fbc528b4fc535e362a64e64f"
PROOF_SHA256 = "d5cfaa7c2afc49bbe592408ebdcace1690f600c6434f2127e328488f00b22fec"
TRAIN_SHA256 = "e73ac3b3baf41075028c3b5629b4412336926faafdb8353b20c61a200bda4c42"
SCHEMA = "recombination-ranker-data-v1"
SPLIT_SALT = "recombination-ranker-v1"
EXPECTED_ROWS = 512
LIMITS = {"whole_seconds": 300, "row_seconds": 30,
          "rss_bytes": 768 * 1024**2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def normalized(text):
    require(isinstance(text, str), "normalized surface must be a string")
    return " ".join(text.split())


def ordered_unique(values):
    return list(dict.fromkeys(values))


def _stable_path_id(identifier, index, candidate):
    del candidate
    return f"saved-roman-train:{identifier}:{index}"


def _validate_capture_candidate(candidate, label):
    require(isinstance(candidate, dict)
            and isinstance(candidate.get("text"), str)
            and isinstance(candidate.get("segments"), list),
            f"{label} is malformed")


def prepare_record(row):
    """Generate a label-free recombination record from one saved capture."""
    require(isinstance(row, dict), "capture row is malformed")
    identifier, source = row.get("id"), row.get("input")
    require(isinstance(identifier, str) and identifier,
            "capture ID is missing")
    require(isinstance(source, str) and isinstance(row.get("result"), str),
            "capture source/result is malformed")
    candidates = row.get("candidates")
    displayed = row.get("displayed_candidates")
    require(isinstance(candidates, list) and candidates,
            "capture candidates are missing")
    require(isinstance(displayed, list),
            "displayed candidates are missing")
    for index, candidate in enumerate(candidates):
        _validate_capture_candidate(candidate, f"candidate {index}")
    for index, candidate in enumerate(displayed):
        _validate_capture_candidate(candidate, f"displayed candidate {index}")
    pool = []
    for index, candidate in enumerate(candidates):
        path_id = _stable_path_id(identifier, index, candidate)
        pool.append({"candidate_id": path_id, "path_id": path_id,
                     "pool_index": index, "text": candidate["text"],
                     "segments": candidate["segments"],
                     "origin": "saved_roman_train",
                     "origins": ["saved_roman_train"],
                     "score": candidate.get("score")})
    generated = propose(source, pool, displayed)
    return {"id": identifier, "request": {"input": source, "context": ""},
            "parent_result": row["result"], "pool": pool,
            "base_candidates": displayed, **generated}


def _leakage_keys(row):
    require(isinstance(row.get("id"), str) and row["id"],
            "metadata ID is missing")
    require(isinstance(row.get("group_id"), str) and row["group_id"],
            "metadata group_id is missing")
    require(isinstance(row.get("source_ids"), list)
            and all(isinstance(value, str) for value in row["source_ids"]),
            "metadata source_ids are malformed")
    require(isinstance(row.get("expected"), list) and row["expected"]
            and all(isinstance(value, str) for value in row["expected"]),
            "metadata expected targets are malformed")
    keys = {"group:" + row["group_id"],
            "input:" + normalized(row["input"])}
    keys.update("source:" + value for value in row["source_ids"])
    keys.update("target:" + normalized(value) for value in row["expected"])
    return keys


def assign_components(metadata):
    """Assign transitive identity/surface components to a deterministic split."""
    require(isinstance(metadata, list) and metadata,
            "metadata must be a nonempty list")
    ids = [row.get("id") for row in metadata]
    require(all(isinstance(value, str) and value for value in ids)
            and len(ids) == len(set(ids)),
            "metadata IDs must be nonempty and unique")
    ordered = sorted(metadata, key=lambda row: row["id"])
    parent = list(range(len(ordered)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    owners = {}
    for index, row in enumerate(ordered):
        for key in sorted(_leakage_keys(row)):
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index
    members = defaultdict(list)
    for index, row in enumerate(ordered):
        members[find(index)].append(row["id"])
    assignments = {}
    for component_members in members.values():
        component_members.sort()
        payload = json.dumps(component_members, ensure_ascii=False,
                             separators=(",", ":"))
        component_id = "component-" + hashlib.sha256(
            payload.encode()).hexdigest()[:20]
        split_hash = hashlib.sha256(
            (SPLIT_SALT + "\0" + component_id).encode()).hexdigest()
        split = "inner_hold" if int(split_hash, 16) % 5 == 0 else "fit"
        for identifier in component_members:
            assignments[identifier] = {
                "component_id": component_id, "split": split}
    return dict(sorted(assignments.items()))


def join_labels(source_records, label_rows):
    """Join TRAIN labels after source generation and create compact rows."""
    require(isinstance(source_records, list), "source records are malformed")
    source_ids = [row.get("id") for row in source_records]
    require(all(isinstance(value, str) and value for value in source_ids)
            and len(source_ids) == len(set(source_ids)),
            "source IDs must be nonempty and unique")
    wanted = set(source_ids)
    labels = {}
    seen_train_ids = set()
    for row in label_rows:
        identifier = row.get("id") if isinstance(row, dict) else None
        require(isinstance(identifier, str) and identifier
                and identifier not in seen_train_ids,
                "TRAIN IDs are missing or duplicate")
        seen_train_ids.add(identifier)
        if identifier in wanted:
            labels[identifier] = row
    require(set(labels) == wanted, "TRAIN target is missing")
    metadata = []
    compact = []
    for source_row in source_records:
        label = labels[source_row["id"]]
        require(label.get("split") == "train"
                and label.get("input_contract") == "roman_sentence_only"
                and label.get("input") == source_row["request"]["input"],
                "TRAIN source identity differs")
        metadata.append({key: label.get(key) for key in (
            "id", "input", "expected", "group_id", "source_ids")})
    assignments = assign_components(metadata)
    for source_row in source_records:
        label = labels[source_row["id"]]
        original = ordered_unique(item["text"] for item in source_row["pool"])
        novel = ordered_unique(item["text"] for item in source_row["proposals"])
        recombined = ordered_unique([*original, *novel])
        require(all(value not in original for value in novel),
                "recombination proposal is not novel")
        assignment = assignments[source_row["id"]]
        compact.append({
            "id": source_row["id"], "input": source_row["request"]["input"],
            "expected": list(label["expected"]),
            "group_id": label["group_id"],
            "source_ids": list(label["source_ids"]),
            "component_id": assignment["component_id"],
            "split": assignment["split"],
            "parent_result": source_row["parent_result"],
            "parent_candidates": [item["text"]
                                  for item in source_row["base_candidates"]],
            "original_candidates": original,
            "recombined_candidates": recombined,
            "novel_candidates": novel,
        })
    return compact


def _validate_upstream(report_path, capture_path, expected_sha, expected_rows):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    require(report.get("completed") is True
            and report.get("input_contract") == "roman_sentence_only"
            and report.get("split") == "train"
            and report.get("rows") == expected_rows,
            "saved TRAIN capture report differs")
    actual = sha256_file(capture_path)
    require(actual == expected_sha
            and report.get("sha256", {}).get("capture.jsonl") == actual,
            "saved TRAIN capture hash differs")
    return report


def _resource_check(started, row_started=None):
    require(time.monotonic() - started <= LIMITS["whole_seconds"]
            and peak_rss() <= LIMITS["rss_bytes"],
            "preparation resource limit exceeded")
    if row_started is not None:
        require(time.monotonic() - row_started <= LIMITS["row_seconds"],
                "row resource limit exceeded")


def run(output, *, capture_path=CAPTURE, report_path=CAPTURE_REPORT,
        proof_path=CAPTURE_PROOF,
        train_path=TRAIN, capture_sha=CAPTURE_SHA256,
        report_sha=REPORT_SHA256, proof_sha=PROOF_SHA256,
        train_sha=TRAIN_SHA256, expected_rows=EXPECTED_ROWS):
    """Write source-only capture, close it, then join canonical TRAIN labels."""
    started = time.monotonic()
    require(sha256_file(report_path) == report_sha,
            "saved TRAIN capture report hash differs")
    require(sha256_file(proof_path) == proof_sha,
            "saved TRAIN capture proof hash differs")
    proof = json.loads(Path(proof_path).read_text(encoding="utf-8"))
    require(proof.get("completed") is True and proof.get("passed") is True
            and proof.get("rows") == expected_rows
            and proof.get("model_inference_run") is False,
            "saved TRAIN capture proof differs")
    _validate_upstream(report_path, capture_path, capture_sha, expected_rows)
    tracked = [Path(capture_path), Path(report_path), Path(proof_path), HELPER,
               Path(__file__).resolve()]
    pins = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT)
            else str(path): sha256_file(path) for path in tracked}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_records = []
    seen = set()
    max_row_s = 0.0
    capture_output = output / "source-capture.jsonl"
    with capture_output.open("x", encoding="utf-8") as stream:
        for row in json_rows(capture_path):
            row_started = time.monotonic()
            record = prepare_record(row)
            record["elapsed_s"] = time.monotonic() - row_started
            require(record["id"] not in seen, "duplicate source capture ID")
            seen.add(record["id"])
            stream.write(json.dumps(record, ensure_ascii=False,
                                    allow_nan=False) + "\n")
            stream.flush()
            source_records.append({
                "id": record["id"], "request": record["request"],
                "parent_result": record["parent_result"],
                "base_candidates": [{"text": value["text"]}
                                    for value in record["base_candidates"]],
                "pool": [{"text": value["text"]} for value in record["pool"]],
                "proposals": [{"text": value["text"]}
                              for value in record["proposals"]],
            })
            max_row_s = max(max_row_s, time.monotonic() - row_started)
            _resource_check(started, row_started)
        require(len(source_records) == expected_rows,
                "saved source capture row count differs")
        os.fsync(stream.fileno())
    capture_output_sha = sha256_file(capture_output)
    closure = {"schema": SCHEMA + "-source-closure",
               "rows": len(source_records),
               "capture_sha256": capture_output_sha,
               "source_only": True, "targets_joined": False}
    closure_path = output / "capture.complete.json"
    write_json(closure_path, closure)

    # This is the first access to canonical TRAIN bytes. Source proposal
    # generation and its durable closure are complete before labels are read.
    require(sha256_file(train_path) == train_sha,
            "canonical TRAIN hash differs")
    pins[str(Path(train_path).relative_to(ROOT))
         if Path(train_path).is_relative_to(ROOT)
         else str(train_path)] = train_sha
    prepared = join_labels(source_records, json_rows(train_path))
    prepared_path = output / "prepared.jsonl"
    with prepared_path.open("x", encoding="utf-8") as stream:
        for row in prepared:
            stream.write(json.dumps(row, ensure_ascii=False,
                                    allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _resource_check(started)
    for name, digest in pins.items():
        path = ROOT / name if not Path(name).is_absolute() else Path(name)
        require(sha256_file(path) == digest, f"pinned input changed: {name}")
    counts = Counter(row["split"] for row in prepared)
    original_covered = sum(bool(set(row["expected"])
                                & set(row["original_candidates"]))
                           for row in prepared)
    recombined_covered = sum(bool(set(row["expected"])
                                  & set(row["recombined_candidates"]))
                             for row in prepared)
    outputs = {
        "source-capture.jsonl": {"sha256": capture_output_sha,
                                 "rows": len(source_records)},
        "capture.complete.json": {"sha256": sha256_file(closure_path),
                                  "rows": len(source_records)},
        "prepared.jsonl": {"sha256": sha256_file(prepared_path),
                           "rows": len(prepared)},
    }
    manifest = {
        "schema": SCHEMA, "completed": True, "rows": len(prepared),
        "counts": {"rows": len(prepared), "fit": counts["fit"],
                   "inner_hold": counts["inner_hold"],
                   "components": len({row["component_id"] for row in prepared}),
                   "novel_candidates": sum(len(row["novel_candidates"])
                                             for row in prepared)},
        "coverage": {"original": original_covered,
                     "recombined": recombined_covered,
                     "gained": recombined_covered - original_covered},
        "outputs": outputs, "pins": dict(sorted(pins.items())),
        "split_salt": SPLIT_SALT,
        "split_policy": "20% salted transitive identity/surface components",
        "source_only_capture_closed_before_labels": True,
        "labels_opened_after_capture": True,
        "model_loaded": False, "model_inference": False,
        "dev_accessed": False, "test_accessed": False,
        "production_changed": False, "limits": LIMITS,
        "elapsed_s": time.monotonic() - started,
        "max_row_s": max_row_s, "peak_rss_bytes": peak_rss(),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps({key: result[key] for key in
                      ("completed", "rows", "counts", "coverage")},
                     ensure_ascii=False), flush=True)
