import hashlib
import json

import pytest

from scripts import select_recombination_training_sources as selection


def row(identifier, category="natural_sentence_context", *, group=None,
        sources=(), source=None, expected=None):
    return {"id": identifier, "input": source or identifier + " input",
            "expected": list(expected or [identifier + " target"]),
            "group_id": group or "group-" + identifier,
            "source_ids": list(sources), "category": category,
            "split": "train", "input_contract": "roman_sentence_only"}


def test_old_group_and_transitive_source_components_are_fully_excluded():
    rows = [row("old", group="shared"),
            row("a", group="shared", sources=["bridge"]),
            row("b", sources=["bridge"]),
            row("c", sources=["bridge"]), row("fresh")]
    chosen, assignments, diagnostics = selection.select(
        rows, ["old"], total_rows=1, natural_rows=1, other_rows=0)
    assert [value["id"] for value in chosen] == ["fresh"]
    assert assignments["old"]["component_id"] == assignments["a"]["component_id"]
    assert diagnostics["blocked_rows"] == 4


def test_target_surface_bridge_excludes_transitive_component():
    rows = [row("old", expected=["같은  문장"]),
            row("bridge", expected=["같은 문장", "다른 문장"]),
            row("linked", expected=["다른  문장"]), row("fresh")]
    chosen, assignments, diagnostics = selection.select(
        rows, ["old"], total_rows=1, natural_rows=1, other_rows=0)
    assert [value["id"] for value in chosen] == ["fresh"]
    assert len({assignments[value]["component_id"]
                for value in ("old", "bridge", "linked")}) == 1
    assert diagnostics["blocked_rows"] == 3


def test_selection_is_deterministic_exact_and_round_robins_other_categories():
    rows = [row("old"), *(row(f"n{i}") for i in range(6)),
            *(row(f"a{i}", "a") for i in range(4)),
            *(row(f"b{i}", "b") for i in range(4))]
    first = selection.select(rows, ["old"], total_rows=6,
                             natural_rows=4, other_rows=2)[0]
    second = selection.select(list(reversed(rows)), ["old"], total_rows=6,
                              natural_rows=4, other_rows=2)[0]
    assert [value["id"] for value in first] == [value["id"] for value in second]
    assert len(first) == 6
    assert {value["category"] for value in first
            if value["category"] != selection.NATURAL_CATEGORY} == {"a", "b"}


@pytest.mark.parametrize(("natural_count", "other_count"), [(6, 1), (1, 6)])
def test_quota_shortage_falls_back_to_opposite_pool(natural_count, other_count):
    rows = [row("old"),
            *(row(f"n{i}") for i in range(natural_count)),
            *(row(f"o{i}", "other") for i in range(other_count))]
    chosen, _, _ = selection.select(rows, ["old"], total_rows=5,
                                    natural_rows=3, other_rows=2)
    assert len(chosen) == 5


def test_insufficient_total_rows_is_rejected():
    with pytest.raises(ValueError, match="insufficient"):
        selection.select([row("old"), row("one")], ["old"],
                         total_rows=2, natural_rows=1, other_rows=1)


def test_run_writes_only_id_and_input_and_refuses_overwrite(tmp_path):
    train = tmp_path / "train.jsonl"
    rows = [row("old"), row("n1"), row("n2"), row("a", "a")]
    train.write_text("".join(json.dumps(value) + "\n" for value in rows))
    old = tmp_path / "old.json"
    old.write_text(json.dumps(["old"]) + "\n")
    output = tmp_path / "output"
    manifest = selection.run(
        output, train_path=train, old_selection_path=old,
        train_sha=hashlib.sha256(train.read_bytes()).hexdigest(),
        old_selection_sha=hashlib.sha256(old.read_bytes()).hexdigest(),
        expected_train_rows=4, expected_old_rows=1,
        total_rows=3, natural_rows=2, other_rows=1)
    written = [json.loads(line) for line in
               (output / "selection.jsonl").read_text().splitlines()]
    assert all(set(value) == {"id", "input"} for value in written)
    assert manifest["outputs"]["selection.jsonl"]["fields"] == ["id", "input"]
    assert len(manifest["partition"]) == 3
    assert manifest["targets_in_selection_artifact"] is False
    with pytest.raises(FileExistsError):
        selection.run(
            output, train_path=train, old_selection_path=old,
            train_sha=hashlib.sha256(train.read_bytes()).hexdigest(),
            old_selection_sha=hashlib.sha256(old.read_bytes()).hexdigest(),
            expected_train_rows=4, expected_old_rows=1,
            total_rows=3, natural_rows=2, other_rows=1)


def test_input_hashes_are_validated_before_output_creation(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text("{}\n")
    old = tmp_path / "old.json"
    old.write_text("[]\n")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="TRAIN hash"):
        selection.run(output, train_path=train, old_selection_path=old,
                      train_sha="wrong", old_selection_sha="wrong",
                      expected_train_rows=1, expected_old_rows=0,
                      total_rows=0, natural_rows=0, other_rows=0)
    assert not output.exists()
