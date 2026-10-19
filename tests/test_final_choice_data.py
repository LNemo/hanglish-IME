import copy
import hashlib

import pytest

from scripts.partition_final_choice_data import (
    SALT,
    build_partition,
    read_inputs,
    salted_key,
    validate_partition,
)


def natural(source_id, group):
    return {"id": source_id, "group_id": group, "split": "train",
            "text": "문장", "left_context": "문맥"}


def historical(source_id, group):
    return {"id": source_id, "group_id": group, "split": "train"}


def challenge(source_id, family, source=None, target=None):
    return {"id": source_id, "family": family, "split": "train",
            "input": source or source_id, "expected": [target or (source_id + "정답")]}


def fixture_rows(overlap=False):
    challenge_rows = [challenge(f"c{i}", f"family-{i}") for i in range(20)]
    preserving = next(row for row in challenge_rows
                      if int(salted_key(
                          SALT,
                          row["family"],
                      )[0], 16) % 5 == 0)
    augmentation = challenge("a1", "augmentation", "new-source", "new-target")
    if overlap:
        augmentation["input"] = preserving["input"]
        augmentation["expected"] = list(preserving["expected"])
    return {
        "natural_raw_train": [
            natural("h1", "historical"), natural("sibling", "historical"),
            natural("a1", "group-a"), natural("a2", "group-a"),
            natural("b1", "group-b"), natural("c1", "group-c"),
            natural("c2", "group-c"), natural("d1", "group-d"),
        ],
        "historical_current_train": [historical("h1", "historical")],
        "challenge_train": challenge_rows,
        "augmentation_train": [augmentation],
    }


def test_deterministic_partitions_keep_whole_groups_and_exclude_historical_group():
    rows = fixture_rows()
    first = build_partition(rows, calibration_target=2, holdout_target=2)
    second = build_partition(copy.deepcopy(rows), calibration_target=2, holdout_target=2)
    assert first == second
    assert first["natural"]["fit_ids"] == ["h1"]
    assert first["natural"]["excluded_historical_group_ids"] == ["historical"]
    assert first["natural"]["excluded_historical_group_sibling_ids"] == ["sibling"]
    for role in ("calibration", "holdout", "reserve"):
        ids = set(first["natural"][f"{role}_ids"])
        groups = set(first["natural"][f"{role}_group_ids"])
        assert ids == {row["id"] for row in rows["natural_raw_train"]
                       if row["group_id"] in groups}
    assert not (set(first["challenge"]["fit_family_ids"])
                & set(first["challenge"]["preservation_family_ids"]))


def test_augmentation_is_fit_only_when_preservation_pair_guard_is_clear():
    clear = build_partition(fixture_rows(), calibration_target=2, holdout_target=2)
    assert clear["augmentation"]["included_in_fit"] is True
    assert clear["augmentation"]["fit_ids"] == ["a1"]
    blocked = build_partition(fixture_rows(overlap=True), calibration_target=2, holdout_target=2)
    assert blocked["augmentation"]["included_in_fit"] is False
    assert blocked["augmentation"]["fit_ids"] == []
    assert blocked["augmentation"]["omitted_ids"] == ["a1"]
    assert "omitted_exact_source_target_overlap" in blocked["augmentation"]["decision_reason"]


def test_duplicate_or_missing_historical_ids_are_rejected():
    rows = fixture_rows()
    rows["natural_raw_train"].append(copy.deepcopy(rows["natural_raw_train"][0]))
    with pytest.raises(ValueError, match="duplicate IDs"):
        build_partition(rows, calibration_target=2, holdout_target=2)
    rows = fixture_rows()
    rows["historical_current_train"][0]["id"] = "absent"
    with pytest.raises(ValueError, match="absent"):
        build_partition(rows, calibration_target=2, holdout_target=2)


def test_historical_group_lineage_mismatch_is_rejected():
    rows = fixture_rows()
    rows["historical_current_train"][0]["group_id"] = "wrong"
    with pytest.raises(ValueError, match="group lineage"):
        build_partition(rows, calibration_target=2, holdout_target=2)


def test_partition_tamper_and_group_leakage_are_rejected():
    rows = fixture_rows()
    partition = build_partition(rows, calibration_target=2, holdout_target=2)
    bad = copy.deepcopy(partition)
    bad["natural"]["holdout_ids"].append(bad["natural"]["calibration_ids"][0])
    with pytest.raises(ValueError, match="overlap|duplicate"):
        validate_partition(rows, bad, calibration_target=2, holdout_target=2)
    bad = copy.deepcopy(partition)
    family = bad["challenge"]["preservation_family_ids"][0]
    source_id = next(row["id"] for row in rows["challenge_train"] if row["family"] == family)
    bad["challenge"]["fit_ids"].append(source_id)
    with pytest.raises(ValueError, match="overlap"):
        validate_partition(rows, bad, calibration_target=2, holdout_target=2)
    bad = copy.deepcopy(partition)
    bad["challenge"]["fit_ids"].reverse()
    with pytest.raises(ValueError, match="hash split"):
        validate_partition(rows, bad, calibration_target=2, holdout_target=2)


def test_manifest_pins_preservation_as_evaluation_only():
    rows = fixture_rows()
    partition = build_partition(rows, calibration_target=2, holdout_target=2)
    assert partition["policy"]["challenge_fit_use"] == (
        "selector_and_distillation_model_optimization"
    )
    assert partition["policy"]["challenge_preservation_use"] == (
        "evaluation_only_never_model_optimization"
    )
    bad = copy.deepcopy(partition)
    bad["policy"]["challenge_preservation_use"] = "model_optimization"
    with pytest.raises(ValueError, match="policy"):
        validate_partition(rows, bad, calibration_target=2, holdout_target=2)


def test_hash_split_and_selection_ignore_labels():
    rows = fixture_rows()
    first = build_partition(rows, calibration_target=2, holdout_target=2)
    changed = copy.deepcopy(rows)
    for row in changed["natural_raw_train"]:
        row["text"] = "완전히 다른 학습 라벨"
    for row in changed["challenge_train"]:
        row["expected"] = ["다른 정답"]
    # Keep the explicit augmentation overlap guard clear after changing labels.
    second = build_partition(changed, calibration_target=2, holdout_target=2)
    assert first["natural"] == second["natural"]
    assert first["challenge"] == second["challenge"]


def test_read_inputs_hashes_exact_bytes_and_counts(tmp_path):
    path = tmp_path / "rows.jsonl"
    raw = b'{"id":"one"}\n'
    path.write_bytes(raw)
    inputs = {"natural_raw_train": (path, hashlib.sha256(raw).hexdigest(), 1)}
    rows, snapshots = read_inputs(inputs)
    assert rows["natural_raw_train"] == [{"id": "one"}]
    assert snapshots["natural_raw_train"]["sha256"] == hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="count"):
        read_inputs({"natural_raw_train": (path, hashlib.sha256(raw).hexdigest(), 2)})
    with pytest.raises(ValueError, match="hash"):
        read_inputs({"natural_raw_train": (path, "0" * 64, 1)})

@pytest.mark.parametrize('role', ['calibration', 'holdout'])
def test_augmentation_cannot_overlap_natural_protected_split(role):
    from hanglish.core.phonology import romanize_word
    rows = fixture_rows()
    for row in rows['natural_raw_train']:
        row['text'] = '자연 문장 ' + row['id']
    initial = build_partition(rows, calibration_target=2, holdout_target=2)
    chosen_id = initial['natural'][role + '_ids'][0]
    target = next(row['text'] for row in rows['natural_raw_train'] if row['id'] == chosen_id)
    rows['augmentation_train'][0].update(input=romanize_word(target), expected=[target])
    actual = build_partition(rows, calibration_target=2, holdout_target=2)
    assert actual['natural'] == initial['natural']
    assert actual['augmentation']['included_in_fit'] is False
    assert actual['augmentation']['protected_exact_source_target_overlap_counts']['natural_' + role] == 1
    assert actual['policy']['id_group_family_assignment_uses_labels'] is False
    assert actual['policy']['augmentation_guard_uses_labels_only_for_overlap_exclusion'] is True


def test_natural_within_group_id_reordering_is_rejected():
    rows = fixture_rows()
    partition = build_partition(rows, calibration_target=2, holdout_target=2)
    by_id = {row['id']: row['group_id'] for row in rows['natural_raw_train']}
    for role in ('calibration', 'holdout', 'reserve'):
        values = partition['natural'][role + '_ids']
        for index in range(len(values) - 1):
            if by_id[values[index]] == by_id[values[index + 1]]:
                values[index], values[index + 1] = values[index + 1], values[index]
                with pytest.raises(ValueError, match='within-group ID ordering'):
                    validate_partition(rows, partition, calibration_target=2, holdout_target=2)
                return
    pytest.fail('fixture has no multi-row group')
