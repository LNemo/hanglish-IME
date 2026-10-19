import json

import pytest

from scripts import probe_symmetric_pair_semantic_selector as successor


def test_temporary_callbacks_restore_after_failure(monkeypatch, tmp_path):
    report = {"eligible_pairs": 1608, "pairs": 1608,
              "complete_roman_and_candidate_text_roundtrip": True,
              "source_hashes": {}, "views_sha256": "views"}
    (tmp_path / "report.json").write_text(json.dumps(report))
    monkeypatch.setattr(successor.layout, "OUTPUT", tmp_path)
    monkeypatch.setattr(successor.experiment, "sha",
        lambda path: successor.LAYOUT_REPORT_SHA if path.name == "report.json" else "views")
    old_prepare, old_views = successor.experiment.prepare, successor.experiment.pair_views

    def failed_run(output):
        assert successor.experiment.prepare is not old_prepare
        assert successor.experiment.pair_views is successor.layout.semantic_views
        raise RuntimeError("deliberate worker failure")

    monkeypatch.setattr(successor.experiment, "run", failed_run)
    with pytest.raises(RuntimeError, match="deliberate"):
        successor.run(tmp_path / "run")
    assert successor.experiment.prepare is old_prepare
    assert successor.experiment.pair_views is old_views


def test_changed_qualification_fails_before_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(successor.experiment, "sha", lambda path: "wrong")
    monkeypatch.setattr(successor.experiment, "run",
        lambda output: pytest.fail("must not start worker"))
    with pytest.raises(ValueError, match="qualification changed"):
        successor.run(tmp_path)
