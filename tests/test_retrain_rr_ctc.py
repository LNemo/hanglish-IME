import json

import pytest

from scripts import retrain_rr_ctc as runner


def _row(identity, source, target="한 문장"):
    return {
        "id": identity, "input": source, "expected": [target],
        "context": "", "context_role": "none", "split": "train",
        "input_contract": "roman_sentence_only", "literal": "ABC-123",
    }


def test_refresh_changes_only_generated_input_and_preserves_literals():
    original = _row("one", "old", target="값 ABC-123")
    rows, changes = runner.refresh_rows(
        [original], lambda _: "old", lambda _: "new", expected_changed=1)
    assert rows == [{**original, "input": "new"}]
    assert rows[0]["expected"] == ["값 ABC-123"]
    assert rows[0]["literal"] == "ABC-123"
    assert changes == [{"id": "one", "old_input": "old", "new_input": "new",
                        "target": "값 ABC-123"}]
    assert original["input"] == "old"


def test_refresh_rejects_changed_rule_row_not_generated_by_old_rules():
    with pytest.raises(ValueError, match="frozen old rules"):
        runner.refresh_rows([_row("one", "manually-edited")],
                            lambda _: "old", lambda _: "new")


def test_refresh_rejects_new_input_collision():
    rows = [_row("one", "old-one", "가"), _row("two", "old-two", "나")]
    with pytest.raises(ValueError, match="refreshed input collision"):
        runner.refresh_rows(rows, lambda target: "old-one" if target == "가" else "old-two",
                            lambda _: "same")


def test_real_training_unions_refresh_exact_expected_counts_without_held_overlap():
    from hanglish.core.phonology import romanize_word
    from scripts.sentence_data import read_split

    old = runner._load_old_romanizer()
    held, _ = runner.held_inputs()
    for name, spec in runner.ARMS.items():
        source = (read_split(runner.BASE_CORPUS, "train")
                  + read_split(spec["extra"], "train"))
        rows, changes = runner.refresh_rows(
            source, old, romanize_word, expected_changed=spec["changed"])
        assert len(rows) == spec["rows"]
        assert len({row["id"] for row in rows}) == len(rows)
        assert len({row["input"] for row in rows}) == len(rows)
        assert not ({row["input"] for row in source} & held)
        assert not ({row["input"] for row in rows} & held)
        assert all(before["expected"] == after["expected"]
                   for before, after in zip(source, rows))
        assert all(change["old_input"] != change["new_input"] for change in changes)


def test_validate_probe_rejects_wrong_arm_or_prepared_identity(tmp_path):
    limits = runner._limits(100.)
    report = {
        "schema": "rr-ctc-retrain-probe-v1", "completed": True,
        "accepted": True, "arm": "word", "prepared_sha256": "prepared",
        "recipe": runner.RECIPE, "limits": limits,
        "observations": [{}, {}], "prototype_discarded": True,
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    (tmp_path / "completion.json").write_text(json.dumps({
        "completed": True, "accepted": True,
        "report_sha256": runner.sha256_file(report_path)}))
    assert runner.validate_probe(tmp_path, "word", "prepared", limits) == report
    with pytest.raises(ValueError, match="matching accepted"):
        runner.validate_probe(tmp_path, "growth", "prepared", limits)
    with pytest.raises(ValueError, match="matching accepted"):
        runner.validate_probe(tmp_path, "word", "other", limits)


def test_projection_uses_measured_length_bins_and_includes_maximum(monkeypatch):
    short = {"tokens": 10}; medium = {"tokens": 20}; longest = {"tokens": 30}
    monkeypatch.setattr(runner, "schedule", lambda rows: [[[short], [medium], [longest]]])
    monkeypatch.setattr(runner, "_length", lambda row: row["tokens"])
    observations = [
        {"actual_max_tokens": 10, "seconds": 1.},
        {"actual_max_tokens": 20, "seconds": .5},
        {"actual_max_tokens": 30, "seconds": 2.},
    ]
    # Timings are made monotone: 1 + 1 + 2, then the 25% margin and 60s reserve.
    assert runner.project_training_seconds(observations, []) == 65.
    with pytest.raises(ValueError, match="maximum actual"):
        runner.project_training_seconds(observations[:-1], [])


def test_source_freeze_detects_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = tmp_path / "source.txt"
    path.write_text("before")
    pins = {"source.txt": runner.sha256_file(path)}
    path.write_text("after")
    with pytest.raises(ValueError, match="source changed"):
        runner.verify_hashes(pins)
