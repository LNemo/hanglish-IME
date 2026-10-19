import pytest

from scripts.partition_balanced_selector_holdout import select_ids


def fixture():
    rows = [{"id": f"r{i}", "group_id": f"g{i // 2}"} for i in range(10)]
    partition = {"reserve_ids": [r["id"] for r in rows],
                 "reserve_group_ids": [f"g{i}" for i in range(5)],
                 "fit_ids": [], "calibration_ids": [], "holdout_ids": [],
                 "excluded_historical_group_sibling_ids": [],
                 "excluded_historical_group_ids": [], "calibration_group_ids": [],
                 "holdout_group_ids": []}
    return rows, partition


def test_deterministic_whole_groups_and_disjoint_remainder():
    rows, partition = fixture()
    result = select_ids(rows, partition, 3)
    assert result == select_ids(rows, partition, 3)
    assert len(result["eval_ids"]) == 4
    assert len(result["eval_group_ids"]) == 2
    assert set(result["eval_ids"]).isdisjoint(result["remaining_ids"])
    assert set(result["eval_ids"] + result["remaining_ids"]) == set(partition["reserve_ids"])
    for group in result["eval_group_ids"]:
        assert all(r["id"] in result["eval_ids"] for r in rows if r["group_id"] == group)


def test_rejects_exposed_group_overlap():
    rows, partition = fixture()
    partition["holdout_group_ids"] = ["g0"]
    with pytest.raises(ValueError, match="overlaps"):
        select_ids(rows, partition, 3)


def test_rejects_partial_group_and_missing_source():
    rows, partition = fixture()
    partition["reserve_ids"].remove("r0")
    with pytest.raises(ValueError, match="complete"):
        select_ids(rows, partition, 3)
    rows, partition = fixture()
    with pytest.raises(ValueError, match="mismatch"):
        select_ids(rows[1:], partition, 3)


def test_rejects_duplicate_ids_and_reordered_source():
    rows, partition = fixture()
    with pytest.raises(ValueError, match="duplicate"):
        select_ids(rows + [rows[0]], partition, 3)
    with pytest.raises(ValueError, match="order"):
        select_ids(list(reversed(rows)), partition, 3)


def test_labels_are_not_accessed():
    class MetadataOnly(dict):
        def __getitem__(self, key):
            assert key in ("id", "group_id")
            return super().__getitem__(key)
    rows, partition = fixture()
    assert select_ids([MetadataOnly(r) for r in rows], partition, 3)["eval_ids"]


@pytest.mark.parametrize("target", [0, -1, True, 11])
def test_rejects_invalid_or_unavailable_target(target):
    rows, partition = fixture()
    with pytest.raises(ValueError):
        select_ids(rows, partition, target)
