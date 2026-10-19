import json
from pathlib import Path

import pytest

from scripts import prepare_observed_phonetic_delta as observed


def base_row(roman, surface, split="train", roots=(), family="family"):
    return {
        "roman": roman, "surface": surface, "split": split,
        "family_roots": [{"surface": root, "pos": pos} for root, pos in roots],
        "family_sha256": family,
    }


def source_row(row_id, expected, *, split="train", group="source-group", refs=()):
    return {
        "id": row_id, "group_id": group, "split": split,
        "expected": expected, "source_ids": list(refs), "source_row_ids": [],
    }


def reserve_fixture():
    reserve = [{"id": "reserve-1", "group_id": "reserve-group"}]
    raw = [{
        "id": "reserve-1", "group_id": "reserve-group", "text": "예약 문장",
        "source_ids": ["raw-source"],
        "provenance": [{"source_id": "raw-provenance"}],
    }]
    return reserve, raw


def eligible(surface):
    mapping = {"가": "same", "나": "same", "다": "old", "라": "held"}
    return (mapping[surface], surface) if surface in mapping else None


def fixture():
    rows = {"train-a.jsonl": [source_row("row-1", ["가 나 다 라"]) ]}
    base = [
        base_row("old", "다", roots=(("다", "NNG"),), family="train-family"),
        base_row("held", "보류", split="lexical_holdout", family="held-family"),
    ]
    analyses = {"다": {(('다', 'NNG'),)}, "라": {(('라', 'NNG'),)}}
    reserve, raw = reserve_fixture()
    return rows, base, analyses, reserve, raw


def test_retains_same_roman_pairs_and_accounts_existing_and_quarantine():
    rows, base, analyses, reserve, raw = fixture()
    accepted, rejected, counts, _, separation = observed.prepare(
        rows, base, analyses, reserve, raw, eligible=eligible)
    assert [(row["roman"], row["surface"]) for row in accepted] == [
        ("same", "가"), ("same", "나")]
    assert counts["eligible_pairs"] == 4
    assert counts["existing_observed_pairs"] == 1
    assert counts["proposed_pairs"] == 3
    assert counts["accepted_pairs"] == 2
    assert counts["quarantined_pairs"] == 1
    assert counts["same_roman_collision_keys"] == 1
    assert counts["same_roman_collision_pairs"] == 2
    assert len(rejected) == 1 and rejected[0]["held_splits"] == ["lexical_holdout"]
    assert not any(separation["overlaps"].values())


def test_rejects_non_train_source_and_duplicate_provenance():
    rows, base, analyses, reserve, raw = fixture()
    rows["train-a.jsonl"][0]["split"] = "dev"
    with pytest.raises(ValueError, match="non-TRAIN"):
        observed.prepare(rows, base, analyses, reserve, raw, eligible=eligible)

    rows, base, analyses, reserve, raw = fixture()
    rows["train-a.jsonl"].append(source_row("row-1", ["가"]))
    with pytest.raises(ValueError, match="duplicate provenance"):
        observed.prepare(rows, base, analyses, reserve, raw, eligible=eligible)


@pytest.mark.parametrize("mutation,match", [
    (lambda row: row.update(id="reserve-1"), "document reserve overlap"),
    (lambda row: row.update(group_id="reserve-group"), "document reserve overlap"),
    (lambda row: row.update(source_ids=["raw-source"]), "document reserve overlap"),
    (lambda row: row.update(expected=["예약 문장"]), "document reserve overlap"),
])
def test_rejects_reserve_identity_source_or_reference_leak(mutation, match):
    rows, base, analyses, reserve, raw = fixture()
    mutation(rows["train-a.jsonl"][0])
    with pytest.raises(ValueError, match=match):
        observed.prepare(rows, base, analyses, reserve, raw, eligible=eligible)


def test_run_refuses_existing_output_without_overwrite(tmp_path):
    output = tmp_path / "exists"
    output.mkdir()
    marker = output / "marker"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        observed.run(output, expected_hashes={})
    assert marker.read_text(encoding="utf-8") == "preserve"
