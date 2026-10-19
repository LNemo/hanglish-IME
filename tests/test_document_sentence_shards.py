import hashlib

import pytest

from scripts.prepare_document_sentence_shards import (
    LIMITS,
    byte_groups,
    canonical,
    partition,
    projected_seconds,
    write_artifacts,
)


def fixtures(groups=16):
    requests = []
    selection = []
    for index in range(groups):
        identifier = f"id-{index:02d}"
        requests.append({"id": identifier, "input": "a" * (index + 1)})
        selection.append({"id": identifier, "group_id": f"group-{index:02d}"})
    schedule = byte_groups(requests)
    timings = [{"id": group["probe_id"], "group_size": group["count"],
                "learned_seconds": float(index + 1)}
               for index, group in enumerate(schedule)]
    return requests, selection, schedule, timings


def test_partition_is_deterministic_greedy_and_preserves_relative_row_order():
    requests, selection, schedule, timings = fixtures()
    rows, summaries = partition(requests, selection, schedule, timings)
    assert [[row["id"] for row in shard] for shard in rows] == [
        ["id-00", "id-03", "id-04", "id-07", "id-08", "id-11", "id-12", "id-15"],
        ["id-01", "id-02", "id-05", "id-06", "id-09", "id-10", "id-13", "id-14"],
    ]
    assert [row["estimated_cost_seconds"] for row in summaries] == [68.0, 68.0]
    assert all(row["qualified"] for row in summaries)


def test_document_group_members_stay_together_and_costs_sum_in_selection_order():
    requests, selection, schedule, timings = fixtures()
    selection[0]["group_id"] = "shared"
    selection[15]["group_id"] = "shared"
    rows, summaries = partition(requests, selection, schedule, timings)
    located = [[row["id"] for row in shard if row["id"] in {"id-00", "id-15"}]
               for shard in rows]
    assert sorted(len(found) for found in located) == [0, 2]
    assert sum(row["groups"] for row in summaries) == 15
    assert sum(row["rows"] for row in summaries) == 16


def test_equal_cost_group_tie_uses_group_id_then_lower_shard_index():
    requests, selection, schedule, timings = fixtures()
    for row in timings:
        row["learned_seconds"] = 1.0
    rows, _ = partition(requests, selection, schedule, timings)
    assert [row["id"] for row in rows[0]][:2] == ["id-00", "id-02"]
    assert [row["id"] for row in rows[1]][:2] == ["id-01", "id-03"]


def test_projection_uses_load_multiplier_and_trailing_allowance():
    assert projected_seconds(10.0) == pytest.approx(1.3 * (60.0 + 10.0) + 60.0)
    assert projected_seconds(
        (LIMITS["per_shard_seconds"] - 60.0) / 1.3 - 60.0
    ) == pytest.approx(1800.0)


@pytest.mark.parametrize("mutation,message", [
    (lambda requests, selection, schedule, timings: requests.append(requests[0]),
     "cardinality"),
    (lambda requests, selection, schedule, timings: selection.__setitem__(
        0, {"id": "wrong", "group_id": "group-00"}), "identity"),
    (lambda requests, selection, schedule, timings: schedule[0]["ids"].__setitem__(
        0, "wrong"), "schedule"),
    (lambda requests, selection, schedule, timings: timings.__setitem__(
        0, {**timings[0], "group_size": 2}), "timing group"),
    (lambda requests, selection, schedule, timings: timings.__setitem__(
        0, {**timings[0], "learned_seconds": float("nan")}), "timing row"),
])
def test_invalid_identity_schedule_and_timing_are_rejected(mutation, message):
    requests, selection, schedule, timings = fixtures()
    mutation(requests, selection, schedule, timings)
    with pytest.raises(ValueError, match=message):
        partition(requests, selection, schedule, timings)


def test_write_artifacts_emits_only_id_and_input_and_refuses_overwrite(tmp_path):
    requests, selection, schedule, timings = fixtures()
    rows, summaries = partition(requests, selection, schedule, timings)
    manifest = {"shards": summaries}
    output = tmp_path / "shards"
    write_artifacts(output, rows, manifest)
    assert (output / "manifest.json").is_file()
    for index, shard in enumerate(rows):
        content = (output / f"shard-{index}.jsonl").read_bytes()
        assert hashlib.sha256(content).hexdigest() == summaries[index]["inputs_sha256"]
        assert content == b"".join(canonical(row) + b"\n" for row in shard)
        assert all(set(row) == {"id", "input"} for row in shard)
    with pytest.raises(ValueError, match="already exists"):
        write_artifacts(output, rows, manifest)


def test_write_artifacts_rejects_digest_mismatch_before_creating_output(tmp_path):
    requests, selection, schedule, timings = fixtures()
    rows, summaries = partition(requests, selection, schedule, timings)
    summaries[0]["inputs_sha256"] = "0" * 64
    output = tmp_path / "shards"
    with pytest.raises(ValueError, match="digest"):
        write_artifacts(output, rows, {"shards": summaries})
    assert not output.exists()
