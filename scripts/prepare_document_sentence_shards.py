"""Freeze two source-only document evaluation shards without opening labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / ".omx/artifacts/document-sentence-inputs-v1"
RESERVE = ROOT / ".omx/artifacts/document-reserve-v2"
PREFLIGHT = ROOT / ".omx/artifacts/document-phonetic-choice-preflight-v1"
PLAN = ROOT / ".omx/plans/document-phonetic-choice-sharded-v1.md"
OUTPUT = ROOT / ".omx/artifacts/document-sentence-shards-v1"

PINS = {
    INPUTS / "inputs.jsonl":
        "536707f05503856f4f805bb9775bd9c5dc932c94783e669a5d62fe3023b0b1db",
    INPUTS / "manifest.json":
        "acea9e1d4d4929253f0d4adfc2aea8ba7c877cf5093b0499961619a35a7632a3",
    INPUTS / "independent-verification.json":
        "c12e49e6f735da775e4c996ef0bea0ecdc699cb156d31bdf8ceef01b65d3935e",
    RESERVE / "selection.jsonl":
        "7d62027d11dcf1320b2a1bac7bcb83e1a31d31ce521808bbe4d1faadf6d4b810",
    PREFLIGHT / "schedule.json":
        "685d7b7310e6669839ff6b13498e46a12d26d7215213030460ad7de2dcb3e48a",
    PREFLIGHT / "report.json":
        "c4454712583618e3d213bd1f2baf631d4f82f459fb9e8de00358f0df54eff874",
    PREFLIGHT / "independent-verification.json":
        "46b678cf26b1768f3620b3220f06cb35ab554f8523b3c5940f5af78806e4809f",
    PLAN: "4b2c05e75e25c25b7f5f19679d9239f681f5e2d3fecdad16db133d0a543472b2",
}

LIMITS = {
    "per_shard_seconds": 1800.0,
    "combined_seconds": 3600.0,
    "loading_allowance_seconds": 60.0,
    "multiplier": 1.3,
    "trailing_allowance_seconds": 60.0,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def digest_lines(values):
    return hashlib.sha256(
        "".join(value + "\n" for value in values).encode()).hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"non-object JSON: {path}")
    return value


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()), f"blank JSONL row at {path}:{number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{number}") from error
            require(isinstance(row, dict), f"non-object JSONL row at {path}:{number}")
            rows.append(row)
    return rows


def byte_groups(requests):
    """Reproduce the frozen preflight's 16 balanced UTF-8 length bins."""
    require(len(requests) >= 16, "too few source requests")
    ordered = sorted(
        requests, key=lambda row: (len(row["input"].encode("utf-8")), row["id"]))
    small, extra = divmod(len(ordered), 16)
    groups = []
    offset = 0
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
    return groups


def projected_seconds(cost, limits=LIMITS):
    return (limits["multiplier"]
            * (limits["loading_allowance_seconds"] + cost)
            + limits["trailing_allowance_seconds"])


def partition(requests, selection, schedule, timings, limits=LIMITS):
    """Assign intact document groups to two greedy cost-balanced shards."""
    require(isinstance(requests, list) and isinstance(selection, list),
            "requests and selection must be lists")
    require(len(requests) == len(selection) and requests,
            "request and selection cardinality differs")
    request_ids = []
    group_by_id = {}
    for request, selected in zip(requests, selection):
        require(set(request) == {"id", "input"}
                and isinstance(request["id"], str) and request["id"]
                and isinstance(request["input"], str) and request["input"],
                "source-only request schema differs")
        require(set(selected) == {"id", "group_id"}
                and isinstance(selected["id"], str) and selected["id"]
                and isinstance(selected["group_id"], str) and selected["group_id"]
                and selected["id"] == request["id"],
                "selection identity or schema differs")
        require(request["id"] not in group_by_id, "duplicate request identity")
        request_ids.append(request["id"])
        group_by_id[request["id"]] = selected["group_id"]

    require(isinstance(schedule, list) and len(schedule) == 16,
            "preflight schedule must contain 16 groups")
    require(schedule == byte_groups(requests),
            "preflight UTF-8 length schedule differs")
    require(isinstance(timings, list) and len(timings) == 16,
            "preflight timings must contain 16 groups")
    timings_by_probe = {}
    for row in timings:
        require(isinstance(row, dict)
                and isinstance(row.get("id"), str)
                and isinstance(row.get("group_size"), int)
                and isinstance(row.get("learned_seconds"), (int, float))
                and math.isfinite(row["learned_seconds"])
                and row["learned_seconds"] > 0
                and row["id"] not in timings_by_probe,
                "invalid or duplicate timing row")
        timings_by_probe[row["id"]] = row

    cost_by_id = {}
    for group in schedule:
        timing = timings_by_probe.get(group["probe_id"])
        require(timing is not None and timing["group_size"] == group["count"],
                "schedule and timing group differ")
        for identifier in group["ids"]:
            require(identifier not in cost_by_id, "duplicate scheduled identity")
            cost_by_id[identifier] = float(timing["learned_seconds"])
    require(set(cost_by_id) == set(request_ids),
            "schedule does not cover the exact request denominator")

    group_costs = {}
    for identifier in request_ids:
        group = group_by_id[identifier]
        group_costs[group] = group_costs.get(group, 0.0) + cost_by_id[identifier]
    ordered_groups = sorted(group_costs, key=lambda group: (-group_costs[group], group))

    bins = [{"groups": set(), "cost": 0.0} for _ in range(2)]
    assignment = {}
    for group in ordered_groups:
        index = min(range(2), key=lambda candidate: (bins[candidate]["cost"], candidate))
        bins[index]["groups"].add(group)
        bins[index]["cost"] += group_costs[group]
        assignment[group] = index

    shard_rows = [[], []]
    shard_group_order = [[], []]
    seen_groups = [set(), set()]
    for request, selected in zip(requests, selection):
        index = assignment[selected["group_id"]]
        shard_rows[index].append(dict(request))
        if selected["group_id"] not in seen_groups[index]:
            seen_groups[index].add(selected["group_id"])
            shard_group_order[index].append(selected["group_id"])

    require(sum(len(rows) for rows in shard_rows) == len(requests),
            "partition dropped source requests")
    require(not (seen_groups[0] & seen_groups[1])
            and (seen_groups[0] | seen_groups[1]) == set(group_costs),
            "partition split or dropped document groups")
    require(set(row["id"] for rows in shard_rows for row in rows) == set(request_ids),
            "partition identity coverage differs")

    summaries = []
    for index, rows in enumerate(shard_rows):
        content = b"".join(canonical(row) + b"\n" for row in rows)
        cost = bins[index]["cost"]
        projection = projected_seconds(cost, limits)
        summaries.append({
            "index": index,
            "file": f"shard-{index}.jsonl",
            "rows": len(rows),
            "groups": len(seen_groups[index]),
            "estimated_cost_seconds": cost,
            "projected_seconds": projection,
            "qualified": projection <= limits["per_shard_seconds"],
            "inputs_sha256": hashlib.sha256(content).hexdigest(),
            "ordered_ids_sha256": digest_lines([row["id"] for row in rows]),
            "ordered_group_ids_sha256": digest_lines(shard_group_order[index]),
        })
    return shard_rows, summaries


def write_artifacts(output, shard_rows, manifest):
    output = Path(output)
    require(not output.exists(), f"output already exists: {output}")
    require(len(shard_rows) == 2 and len(manifest.get("shards", [])) == 2,
            "two shard outputs required")
    contents = []
    for index, rows in enumerate(shard_rows):
        require(rows and all(set(row) == {"id", "input"} for row in rows),
                "source-only shard schema differs")
        content = b"".join(canonical(row) + b"\n" for row in rows)
        require(hashlib.sha256(content).hexdigest()
                == manifest["shards"][index]["inputs_sha256"],
                "shard content digest differs")
        contents.append(content)
    output.mkdir(parents=True, exist_ok=False)
    for index, content in enumerate(contents):
        with (output / f"shard-{index}.jsonl").open("xb") as stream:
            stream.write(content)
    with (output / "manifest.json").open("xb") as stream:
        stream.write(canonical(manifest) + b"\n")


def run(output=OUTPUT):
    output = Path(output)
    require(not output.exists(), f"output already exists: {output}")
    for path, expected in PINS.items():
        require(path.is_file() and sha(path) == expected, f"source changed: {path}")

    input_manifest = read_json(INPUTS / "manifest.json")
    input_proof = read_json(INPUTS / "independent-verification.json")
    report = read_json(PREFLIGHT / "report.json")
    preflight_proof = read_json(PREFLIGHT / "independent-verification.json")
    schedule = read_json(PREFLIGHT / "schedule.json")
    requests = read_jsonl(INPUTS / "inputs.jsonl")
    selection = read_jsonl(RESERVE / "selection.jsonl")
    require(input_manifest.get("completed") is True
            and input_manifest.get("counts", {}).get("rows") == 1046
            and input_manifest.get("counts", {}).get("groups") == 821
            and input_proof.get("verified") is True,
            "source-only input qualification differs")
    require(report.get("completed") is True
            and report.get("input_rows") == 1046
            and report.get("probe_rows") == 16
            and report.get("schedule_sha256") == PINS[PREFLIGHT / "schedule.json"]
            and report.get("shared_qualified") is False
            and report.get("projected_shared_seconds", 0) > 1800.0,
            "failed whole-run preflight qualification differs")
    require(preflight_proof.get("verified") is True
            and preflight_proof.get("report_sha256") == PINS[PREFLIGHT / "report.json"]
            and preflight_proof.get("schedule_sha256") == PINS[PREFLIGHT / "schedule.json"]
            and preflight_proof.get("contracts_verified", {}).get(
                "performance_labels_or_raw_references_deserialized") is False,
            "independent preflight proof differs")
    require(schedule.get("selection")
            == "16 balanced UTF-8-length groups, longest source then ID",
            "preflight schedule contract differs")

    shard_rows, summaries = partition(
        requests, selection, schedule.get("groups"), report.get("timings"), LIMITS)
    require(len(requests) == 1046
            and len({row["group_id"] for row in selection}) == 821
            and all(row["qualified"] for row in summaries)
            and sum(row["projected_seconds"] for row in summaries)
                <= LIMITS["combined_seconds"],
            "two-shard execution plan does not qualify")

    producer = Path(__file__).resolve()
    source_hashes = {
        str(path.relative_to(ROOT)): digest for path, digest in PINS.items()
    }
    source_hashes[str(producer.relative_to(ROOT))] = sha(producer)
    manifest = {
        "schema": "document-sentence-shards-v1",
        "completed": True,
        "counts": {"rows": 1046, "groups": 821, "shards": 2},
        "shards": summaries,
        "limits": dict(LIMITS),
        "partition_contract": {
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
        },
        "source_hashes": source_hashes,
        "model_activity": False,
        "performance_labels_joined": False,
        "accuracy_measured": False,
        "limits_and_caveats": [
            "Preflight timings are projections rather than execution-time upper bounds.",
            "The two qualified shards must run serially and retain independent 1,800-second watchdogs.",
            "No raw reference text or accuracy label was deserialized to construct this partition.",
        ],
    }
    for raw, expected in source_hashes.items():
        require(sha(ROOT / raw) == expected, f"source changed during build: {raw}")
    write_artifacts(output, shard_rows, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    built = run(parser.parse_args().output)
    print(json.dumps({"counts": built["counts"], "shards": built["shards"]},
                     ensure_ascii=False, sort_keys=True))
