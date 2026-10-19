import ast
import copy
from pathlib import Path

import pytest

from scripts import verify_document_sentence_shards as verifier


def requests(count=32):
    return [
        {"id": f"row-{index:03d}", "input": "a " + ("x" * (index + 1))}
        for index in range(count)
    ]


def schedule_and_timings(rows):
    schedule = {
        "selection": "16 balanced UTF-8-length groups, longest source then ID",
        "groups": verifier.byte_probe_groups(rows),
    }
    timings = [
        {"id": group["probe_id"], "group_size": group["count"],
         "learned_seconds": float(index + 1)}
        for index, group in enumerate(schedule["groups"])
    ]
    return schedule, timings


def test_verifier_does_not_import_producer_neural_or_target_modules():
    tree = ast.parse(Path(verifier.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = (
        "prepare_document_sentence_shards", "torch", "hanglish", "phonetic",
        "sentence_data", "train.jsonl",
    )
    assert not any(item in name for name in imports for item in forbidden)


def test_byte_strata_cover_every_request_and_use_utf8_bytes():
    rows = requests(31)
    rows.append({"id": "multibyte", "input": "a ㄱ"})
    groups = verifier.byte_probe_groups(rows)
    assert len(groups) == 16
    assert sum(group["count"] for group in groups) == 32
    assert sorted(identifier for group in groups for identifier in group["ids"]) \
        == sorted(row["id"] for row in rows)
    by_id = {row["id"]: row for row in rows}
    assert all(group["max_utf8_bytes"] == len(
        by_id[group["probe_id"]]["input"].encode("utf-8")) for group in groups)


def test_request_costs_replays_schedule_and_rejects_timing_drift():
    rows = requests()
    schedule, timings = schedule_and_timings(rows)
    costs = verifier.request_costs(rows, schedule, timings)
    assert len(costs) == len(rows)
    assert costs[schedule["groups"][0]["ids"][0]] == 1.0
    assert costs[schedule["groups"][-1]["ids"][-1]] == 16.0

    damaged = copy.deepcopy(timings)
    damaged[0]["id"] = "wrong-probe"
    with pytest.raises(ValueError, match="timing identity"):
        verifier.request_costs(rows, schedule, damaged)
    damaged_schedule = copy.deepcopy(schedule)
    damaged_schedule["groups"][0]["ids"].reverse()
    with pytest.raises(ValueError, match="schedule"):
        verifier.request_costs(rows, damaged_schedule, timings)


def test_partition_keeps_groups_whole_and_source_relative_order():
    rows = requests(8)
    selection = [
        {"id": row["id"], "group_id": f"g-{index // 2}"}
        for index, row in enumerate(rows)
    ]
    costs = {row["id"]: float(index + 1) for index, row in enumerate(rows)}
    shards, metadata = verifier.partition(rows, selection, costs)
    assigned = {}
    position = {row["id"]: index for index, row in enumerate(rows)}
    group_by_id = {row["id"]: selected["group_id"]
                   for row, selected in zip(rows, selection)}
    for shard_index, shard in enumerate(shards):
        assert [position[row["id"]] for row in shard] == sorted(
            position[row["id"]] for row in shard)
        for row in shard:
            group = group_by_id[row["id"]]
            assigned.setdefault(group, shard_index)
            assert assigned[group] == shard_index
    assert sum(len(shard) for shard in shards) == 8
    assert sum(item["groups"] for item in metadata) == 4


def test_partition_tie_chooses_lower_bin_and_group_sort_is_deterministic():
    rows = [
        {"id": "a", "input": "a a"},
        {"id": "b", "input": "b b"},
        {"id": "c", "input": "c c"},
        {"id": "d", "input": "d d"},
    ]
    selection = [
        {"id": "a", "group_id": "z"},
        {"id": "b", "group_id": "a"},
        {"id": "c", "group_id": "y"},
        {"id": "d", "group_id": "b"},
    ]
    shards, _ = verifier.partition(rows, selection, {row["id"]: 1.0 for row in rows})
    assert [row["id"] for row in shards[0]] == ["b", "c"]
    assert [row["id"] for row in shards[1]] == ["a", "d"]


def test_projected_seconds_uses_both_allowances_and_limit():
    assert verifier.projected_seconds(100.0) == pytest.approx(1.3 * (60 + 100) + 60)
    boundary_cost = (1800.0 - 60.0) / 1.3 - 60.0
    assert verifier.projected_seconds(boundary_cost) == pytest.approx(1800.0)
    assert verifier.projected_seconds(boundary_cost + 0.01) > 1800.0


def test_resolved_identities_fail_closed_until_frozen():
    with pytest.raises(ValueError, match="not frozen"):
        verifier.resolved_identities(
            verifier.UNRESOLVED,
            verifier.UNRESOLVED,
            {name: verifier.UNRESOLVED
             for name in verifier.EXPECTED_ARTIFACT_HASHES},
        )
    assert verifier.resolved_identities() == (
        verifier.EXPECTED_PRODUCER_SHA256,
        verifier.EXPECTED_PLAN_SHA256,
        verifier.EXPECTED_ARTIFACT_HASHES,
    )
    hashes = {name: str(index) * 64 for index, name in enumerate(
        sorted(verifier.EXPECTED_ARTIFACT_HASHES), 1)}
    producer, plan, artifacts = verifier.resolved_identities(
        "a" * 64, "b" * 64, hashes)
    assert producer == "a" * 64
    assert plan == "b" * 64
    assert artifacts == hashes
