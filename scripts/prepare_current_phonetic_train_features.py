#!/usr/bin/env python3
"""Join closed current-phonetic FIT captures with FIT targets only.

The mixed 2,048-row container is scanned as bytes first.  Only allowlisted FIT
lines are decoded as JSON; the 406 inner-HOLD lines are never JSON-decoded and
their target values are never used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / ".omx/artifacts/current-phonetic-train-inputs-v1"
SHARDS = tuple(ROOT / f".omx/artifacts/current-phonetic-train-shard-{i}-v1"
               for i in range(6))
MIXED = ROOT / ".omx/artifacts/recombination-fresh-runtime-v1/prepared.jsonl"
MIXED_REPORT = ROOT / ".omx/artifacts/recombination-fresh-runtime-v1/report.json"
MIXED_PROOF = ROOT / ".omx/artifacts/recombination-fresh-runtime-v1/independent-verification.json"
ROWS = 1642
MIXED_ROWS = 2048
INNER_HOLD_ROWS = 406
PRODUCER_SHA256 = "13bd819bf4e108400423518e69c97443ea58b4cf6b930e98fb546db763fffbfa"
VERIFIER_SHA256 = "3fd39342dcbd6a1c318deea4e80cf400d8a80319063ec2d19dc55b623e949a4d"
MIXED_SHA256 = "340f7611ff9f247e80b06dcbd48593b0c391572a0435f6ce7db477e789f6809e"
MIXED_REPORT_SHA256 = "d890bad55cbd52f3168e8a8fa7f9caefc88be448dc35af011e4e386e7871212d"
MIXED_PROOF_SHA256 = "d01213d92b20523eb29646cbb4b460342e91bd7c10d782b54ce03ff1ef054ca6"
PREPARED_HASHES = {
    "requests.json": "97b416a805b9da8dc15d4ff9178b3a1db2a98f57f961a0e369c51bc771e1762d",
    "manifest.json": "c135c4c7f35c763257bca3b4369ab15165b186b20692bf5d8c1d53a6ec337685",
    "independent-verification.json": "4b34c0622f33afc6ec02c16415119159857f4a8947143412e84290a2379efc14",
}
ID_PATTERN = re.compile(rb'"id"\s*:\s*"(s3-train-[0-9]+)"')
SPLIT_PATTERN = re.compile(rb'"split"\s*:\s*"(fit|inner_hold)"')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "expected JSON object: " + str(path))
    return value


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSONL at {path}:{number}") from error
            require(isinstance(value, dict), "JSONL row is not an object")
            yield value


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def append(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                            allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def process_absent(pid, probe=os.kill):
    require(type(pid) is int and pid > 0, "invalid launch PID")
    try:
        probe(pid, 0)
    except ProcessLookupError:
        return True
    raise ValueError("capture process is still present")


def authenticate_prepared():
    require(all(sha(PREPARED / name) == digest
                for name, digest in PREPARED_HASHES.items()),
            "prepared source artifact differs")
    requests = read(PREPARED / "requests.json")
    manifest = read(PREPARED / "manifest.json")
    proof = read(PREPARED / "independent-verification.json")
    rows, shards = requests.get("rows"), requests.get("shards")
    require(isinstance(rows, list) and len(rows) == ROWS
            and isinstance(shards, list) and len(shards) == len(SHARDS)
            and sum(map(len, shards)) == ROWS
            and len({row.get("id") for row in rows}) == ROWS
            and all(set(row) == {"id", "input", "component_id"}
                    and isinstance(row["id"], str)
                    and isinstance(row["input"], str) and row["input"]
                    and isinstance(row["component_id"], str)
                    for row in rows), "prepared source domain differs")
    require(manifest.get("rows") == ROWS
            and manifest.get("requests_sha256") == PREPARED_HASHES["requests.json"]
            and manifest.get("labels_opened") is False
            and proof.get("verified") is True
            and proof.get("manifest_sha256") == PREPARED_HASHES["manifest.json"]
            and proof.get("requests_sha256") == PREPARED_HASHES["requests.json"]
            and proof.get("verifier_sha256") == VERIFIER_SHA256
            and proof.get("counts", {}).get("rows") == ROWS
            and proof.get("counts", {}).get("shard_rows") == list(map(len, shards)),
            "prepared proof differs")
    by_id = {row["id"]: row for row in rows}
    require(sorted(identifier for shard in shards for identifier in shard)
            == sorted(by_id), "prepared shard coverage differs")
    return requests, by_id


def authenticate_mixed_metadata():
    require(sha(MIXED_REPORT) == MIXED_REPORT_SHA256
            and sha(MIXED_PROOF) == MIXED_PROOF_SHA256,
            "mixed FIT container metadata differs")
    report, proof = read(MIXED_REPORT), read(MIXED_PROOF)
    output = report.get("outputs", {}).get("prepared.jsonl", {})
    require(report.get("completed") is True
            and output == {"rows": MIXED_ROWS, "sha256": MIXED_SHA256}
            and proof.get("verified") is True
            and proof.get("report_sha256") == MIXED_REPORT_SHA256
            and proof.get("prepared_sha256") == MIXED_SHA256
            and proof.get("counts", {}).get("rows") == MIXED_ROWS,
            "mixed FIT container proof differs")


def expected_launch(directory, shard):
    return [str(ROOT / ".venv/bin/python"), "-m",
            "scripts.current_phonetic_train_pool", "--mode", "capture",
            "--output", str(Path(directory).resolve()), "--shard", str(shard)]


def iter_capture_rows(directory, expected_ids, sources):
    """Yield authenticated source rows without retaining a shard in memory."""
    stream = iter(jsonl(Path(directory) / "capture.jsonl"))
    for identifier in expected_ids:
        try:
            row = next(stream)
        except StopIteration as error:
            raise ValueError("shard capture count differs") from error
        source = sources[identifier]
        require(row.get("id") == identifier
                and row.get("component_id") == source["component_id"]
                and row.get("request") == {"input": source["input"], "context": ""},
                "shard capture source/order differs")
        yield row
    try:
        next(stream)
    except StopIteration:
        return
    raise ValueError("shard capture count differs")


def authenticate_shard(directory, shard, expected_ids, sources, pid_probe=os.kill):
    directory = Path(directory).resolve()
    require(not (directory / "failure.json").exists()
            and not (directory / "resource-watchdog-failure.json").exists(),
            "shard failure artifact exists")
    required = ("capture.jsonl", "capture.complete.json", "report.json",
                "completion.json", "pins.json", "independent-verification.json")
    require(all((directory / name).is_file() for name in required),
            "verified terminal shard files missing")
    launch_path = directory.with_suffix(".launch.json")
    launch = read(launch_path)
    require(set(launch) == {"pid", "command", "output", "log",
                            "runner_sha256", "launched_at"}
            and launch.get("command") == expected_launch(directory, shard)
            and launch.get("output") == str(directory)
            and launch.get("log") == str(directory.with_suffix(".log"))
            and launch.get("runner_sha256") == PRODUCER_SHA256
            and Path(launch["log"]).is_file(), "shard launch differs")
    datetime.fromisoformat(launch["launched_at"])
    process_absent(launch.get("pid"), pid_probe)
    report = read(directory / "report.json")
    proof = read(directory / "independent-verification.json")
    completion = read(directory / "completion.json")
    closure = read(directory / "capture.complete.json")
    pins = read(directory / "pins.json")
    capture_sha = sha(directory / "capture.jsonl")
    require(report.get("completed") is True and report.get("mode") == "capture"
            and report.get("shard") == shard and report.get("rows") == len(expected_ids)
            and report.get("ids") == expected_ids and report.get("pins") == pins
            and report.get("capture_sha256") == capture_sha
            and report.get("closure_sha256") == sha(directory / "capture.complete.json")
            and report.get("qualified") is True and report.get("labels_opened") is False,
            "shard report differs")
    require(proof.get("verified") is True and proof.get("mode") == "capture"
            and proof.get("shard") == shard
            and proof.get("producer_sha256") == PRODUCER_SHA256
            and proof.get("verifier_sha256") == VERIFIER_SHA256
            and proof.get("report_sha256") == sha(directory / "report.json")
            and proof.get("capture_sha256") == capture_sha
            and proof.get("closure_sha256") == sha(directory / "capture.complete.json")
            and proof.get("completion_sha256") == sha(directory / "completion.json")
            and proof.get("launch_sha256") == sha(launch_path)
            and proof.get("counts", {}).get("rows") == len(expected_ids),
            "shard independent proof differs")
    require(completion == {"completed": True,
                           "report_sha256": sha(directory / "report.json")}
            and closure == {"mode": "capture", "shard": shard,
                            "rows": len(expected_ids),
                            "capture_sha256": capture_sha,
                            "labels_opened": False},
            "shard closure/completion differs")
    critical = {
        str((PREPARED / name).resolve()): digest
        for name, digest in PREPARED_HASHES.items()
    }
    critical[str((ROOT / "scripts/current_phonetic_train_pool.py").resolve())] = PRODUCER_SHA256
    require(all(pins.get(path) == digest for path, digest in critical.items()),
            "shard critical pins differ")
    for raw, digest in pins.items():
        path = Path(raw) if Path(raw).is_absolute() else ROOT / raw
        require(path.is_file() and sha(path) == digest,
                "shard pin changed: " + raw)
    for _ in iter_capture_rows(directory, expected_ids, sources):
        pass
    return {
        "shard": shard, "rows": len(expected_ids), "capture_sha256": capture_sha,
        "report_sha256": sha(directory / "report.json"),
        "proof_sha256": sha(directory / "independent-verification.json"),
        "launch_sha256": sha(launch_path),
        "completion_sha256": sha(directory / "completion.json"),
    }


def scan_fit_labels(path, allowlist, *, expected_mixed_rows=None,
                    expected_inner_hold_rows=None):
    """Decode only allowlisted FIT lines; never decode inner-HOLD JSON."""
    expected_mixed_rows = (MIXED_ROWS if expected_mixed_rows is None
                           else expected_mixed_rows)
    expected_inner_hold_rows = (INNER_HOLD_ROWS if expected_inner_hold_rows is None
                                else expected_inner_hold_rows)
    labels, seen, splits = {}, set(), Counter()
    with Path(path).open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            identifiers = ID_PATTERN.findall(raw)
            split_matches = SPLIT_PATTERN.findall(raw)
            require(len(identifiers) == len(split_matches) == 1,
                    f"ambiguous mixed-container identity at line {number}")
            identifier = identifiers[0].decode("ascii")
            split = split_matches[0].decode("ascii")
            require(identifier not in seen, "duplicate mixed-container ID")
            seen.add(identifier)
            splits[split] += 1
            if identifier not in allowlist:
                require(split == "inner_hold", "non-allowlisted row is not inner-HOLD")
                continue
            require(split == "fit", "allowlisted FIT row has wrong split")
            try:
                row = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid allowlisted FIT JSON at line {number}") from error
            expected = row.get("expected")
            source = allowlist[identifier]
            require(row.get("id") == identifier and row.get("split") == "fit"
                    and row.get("input") == source["input"]
                    and row.get("component_id") == source["component_id"]
                    and isinstance(expected, list) and expected
                    and len(set(expected)) == len(expected)
                    and all(isinstance(value, str) and value for value in expected),
                    "allowlisted FIT target differs")
            labels[identifier] = list(expected)
    require(len(seen) == expected_mixed_rows and set(labels) == set(allowlist)
            and splits == {"fit": len(allowlist),
                           "inner_hold": expected_inner_hold_rows},
            "mixed FIT/inner-HOLD partition differs")
    return labels


def project_feature_row(capture, expected, shard):
    response = capture.get("response")
    paths = capture.get("base_paths")
    require(isinstance(response, dict) and isinstance(paths, list),
            "capture payload differs")
    ranked = response.get("sentence_ranking_applied") is True
    choice_rows = response.get("phonetic_choice", {}).get("rows")
    require(isinstance(choice_rows, list) and len(choice_rows) == len(paths),
            "feature/path denominator differs")
    candidates, scores, features = [], [], []
    for path, row in zip(paths, choice_rows):
        require(isinstance(path, list) and len(path) == 3
                and isinstance(path[0], str) and finite(path[1])
                and row.get("text") == path[0]
                and row.get("base_score") == path[1]
                and row.get("segments") == path[2]
                and isinstance(row.get("features"), list)
                and len(row["features"]) == 4
                and all(finite(value) for value in row["features"]),
                "captured candidate feature differs")
        candidates.append(path[0]); scores.append(path[1]); features.append(row["features"])
    require(len(set(candidates)) == len(candidates)
            and (ranked or not candidates), "ranked candidate domain differs")
    expected_set = set(expected)
    positives = [index for index, text in enumerate(candidates)
                 if text in expected_set]
    production = response.get("phonetic_word_union", {}).get("production_pool_texts")
    require(isinstance(production, list)
            and all(isinstance(text, str) for text in production),
            "production pool texts differ")
    production_set = set(production)
    production_indices = [index for index, text in enumerate(candidates)
                          if text in production_set]
    public = response.get("candidates")
    require(isinstance(public, list)
            and all(isinstance(row, dict) and isinstance(row.get("text"), str)
                    for row in public), "public candidates differ")
    return {
        "id": capture["id"], "input": capture["request"]["input"],
        "component_id": capture["component_id"], "source_shard": shard,
        "ranked": ranked, "training_eligible": ranked and bool(positives),
        "candidates": candidates, "base_scores": scores, "features": features,
        "positive_indices": positives, "production_indices": production_indices,
        "public_candidates": [row["text"] for row in public],
        "expected": list(expected),
    }


def run(output, *, shard_directories=SHARDS, mixed_path=MIXED, pid_probe=os.kill):
    output = Path(output)
    require(len(shard_directories) == len(SHARDS), "six shards are required")
    requests, sources = authenticate_prepared()
    authenticate_mixed_metadata()
    output.mkdir(parents=True, exist_ok=False)
    shard_evidence = []
    for shard, directory in enumerate(shard_directories):
        shard_evidence.append(authenticate_shard(
            directory, shard, requests["shards"][shard], sources, pid_probe))
    require(sum(item["rows"] for item in shard_evidence) == ROWS,
            "six-shard capture coverage differs")
    source_closure = {
        "schema": "current-phonetic-train-source-closure-v1",
        "closed": True, "rows": ROWS, "labels_opened": False,
        "shards": shard_evidence,
    }
    write(output / "source-capture.complete.json", source_closure)

    # The mixed file's bytes are authenticated only after the source-only
    # closure exists, and only allowlisted FIT lines are subsequently decoded.
    require(sha(mixed_path) == MIXED_SHA256, "mixed FIT container bytes differ")
    labels = scan_fit_labels(mixed_path, sources)
    counts = Counter(rows=ROWS)
    with (output / "rows.jsonl").open("x", encoding="utf-8") as stream:
        for shard, directory in enumerate(shard_directories):
            evidence = shard_evidence[shard]
            require(sha(Path(directory) / "capture.jsonl")
                    == evidence["capture_sha256"],
                    "shard capture changed before feature projection")
            for capture in iter_capture_rows(
                    directory, requests["shards"][shard], sources):
                row = project_feature_row(capture, labels[capture["id"]], shard)
                append(stream, row)
                counts["ranked"] += row["ranked"]
                counts["unranked"] += not row["ranked"]
                counts["covered"] += bool(row["positive_indices"])
                counts["uncovered"] += not row["positive_indices"]
                counts["train_eligible"] += row["training_eligible"]
                counts["full_pool_candidates"] += len(row["candidates"])
            require(sha(Path(directory) / "capture.jsonl")
                    == evidence["capture_sha256"],
                    "shard capture changed during feature projection")
    report = {
        "schema": "current-phonetic-train-features-v1", "completed": True,
        "counts": dict(counts),
        "inputs": {
            "prepared": PREPARED_HASHES,
            "mixed_report_sha256": MIXED_REPORT_SHA256,
            "mixed_proof_sha256": MIXED_PROOF_SHA256,
            "mixed_prepared_sha256": MIXED_SHA256,
            "shards": shard_evidence,
        },
        "outputs": {
            "source-capture.complete.json": sha(output / "source-capture.complete.json"),
            "rows.jsonl": sha(output / "rows.jsonl"),
        },
        "mixed_container_bytes_read": True,
        "non_fit_json_decoded": False,
        "non_fit_target_values_used": False,
        "fit_targets_joined_after_source_closure": True,
        "all_rows_preserved": True,
        "positive_candidates_injected": False,
        "neural_training": False, "weights_fitted": False,
        "dev_test_hold_reserve_labels_opened": False,
    }
    write(output / "report.json", report)
    write(output / "completion.json", {
        "completed": True, "report_sha256": sha(output / "report.json")})
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.output), ensure_ascii=False, sort_keys=True))
