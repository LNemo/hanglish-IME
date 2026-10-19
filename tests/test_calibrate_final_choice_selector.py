import copy
import hashlib
import json

import pytest

import scripts.calibrate_final_choice_selector as module


def segment(source, text, kind="syllable"):
    return {"input_start": 0, "input_end": len(source), "source": source,
            "text": text, "kind": kind}


def candidate(source, text, score):
    return {"text": text, "score": score, "segments": [segment(source, text)]}


class TracedRow(dict):
    def __init__(self, values):
        super().__init__(values)
        self.expected_reads = 0

    def __getitem__(self, key):
        if key == "expected":
            self.expected_reads += 1
        return super().__getitem__(key)


class FakeRanker:
    def __init__(self, scores):
        self.scores = scores

    def score_pairs(self, contexts, candidates):
        assert len(contexts) == len(candidates) == len(self.scores)
        return list(self.scores)

    def can_score(self, candidate):
        return True


def policy_row():
    source = "ga"
    candidates = [candidate(source, "카", 2.0), candidate(source, "가", 1.0),
                  {"text": source, "score": None,
                   "segments": [segment(source, source, "raw")]}]
    return TracedRow({
        "id": "row", "group_id": "group", "origin": "natural",
        "input": source, "context": "문맥", "expected": ["가"],
        "candidates": candidates, "finite_indices": [0, 1],
        "selector_eligible": True, "all_candidates_can_score": True,
        "candidate_can_score": [True, True, True],
    })


def test_scoring_uses_complete_candidates_and_reads_labels_after_policy():
    row = policy_row()

    class TracingRanker(FakeRanker):
        def score_pairs(self, contexts, candidates):
            assert row.expected_reads == 0
            assert candidates == ["카", "가", "ga"]
            return super().score_pairs(contexts, candidates)

    result = module.score_calibration_case(TracingRanker([.1, 2., -1.]), row)
    assert result["reason"] == "eligible_disagreement"
    assert result["proposal_index"] == 1
    assert result["parent_accepted"] is False
    assert result["proposal_accepted"] is True
    assert row.expected_reads == 1


def test_ineligible_row_is_not_scored_or_labeled():
    row = policy_row()
    row["selector_eligible"] = False
    ranker = FakeRanker([])
    result = module.score_calibration_case(ranker, row)
    assert result["reason"] == "capture_ineligible"
    assert result["selector_scores"] == []
    assert result["parent_accepted"] is None
    assert row.expected_reads == 0


def test_scoring_rejects_nonfinite_or_changed_budget_before_labels():
    row = policy_row()
    with pytest.raises(ValueError, match="complete CAL"):
        module.score_calibration_case(FakeRanker([.1, float("nan"), -1.]), row)
    assert row.expected_reads == 0

    class BudgetMismatch(FakeRanker):
        def can_score(self, candidate):
            return candidate != "ga"

    with pytest.raises(ValueError, match="budget differs"):
        module.score_calibration_case(BudgetMismatch([.1, 2., -1.]), row)
    assert row.expected_reads == 0


def test_calibration_rows_and_class_counts_use_only_eligible_disagreements():
    base = {"role": module.CALIBRATION_ROLE, "reason": "eligible_disagreement",
            "features": [0.] * 6}
    audit = [
        {**base, "group_id": "accept", "parent_accepted": False,
         "proposal_accepted": True},
        {**base, "group_id": "stay", "parent_accepted": True,
         "proposal_accepted": False},
        {**base, "group_id": "other", "reason": "parent_selected",
         "parent_accepted": True, "proposal_accepted": True},
    ]
    rows = module.calibration_rows(audit)
    assert len(rows) == 2
    assert module.class_counts(rows) == {
        "rows": 2, "accept_rows": 1, "stay_rows": 1,
        "accept_groups": 1, "stay_groups": 1,
    }


def test_calibration_loader_rejects_wrong_role_before_rows(monkeypatch, tmp_path):
    manifest = {"role": "natural_holdout", "cases_sha256": hashlib.sha256(b"").hexdigest(),
                "selection_sha256": hashlib.sha256(b"").hexdigest()}
    raw = json.dumps(manifest).encode()
    (tmp_path / "manifest.json").write_bytes(raw)
    (tmp_path / "cases.jsonl").write_bytes(b"")
    (tmp_path / "selection.jsonl").write_bytes(b"")
    with pytest.raises(ValueError, match="natural_calibration"):
        module.load_calibration_capture(tmp_path, hashlib.sha256(raw).hexdigest())


def test_selector_loader_rejects_stale_report_before_model_load(tmp_path):
    report = {"scope": "wrong"}
    raw = json.dumps(report).encode()
    (tmp_path / "report.json").write_bytes(raw)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text("{}")
    calibration = {"runtime_version": "runtime"}
    with pytest.raises(ValueError, match="completed FIT-only"):
        module.load_selector(tmp_path / "report.json", hashlib.sha256(raw).hexdigest(),
                             calibration)


def test_gate_insufficient_contract_has_no_successful_model(tmp_path):
    rows = [{"role": module.CALIBRATION_ROLE, "reason": "eligible_disagreement",
             "group_id": "one", "features": [0.] * 6,
             "parent_accepted": False, "proposal_accepted": True}]
    audit = [{"id": "one", **rows[0]}]
    gate, status, reason, parity, counts = module.fit_and_persist_gate(
        tmp_path, rows, audit
    )
    assert min(counts["accept_groups"], counts["stay_groups"]) < module.MIN_CLASS_GROUPS
    assert gate is None and status == "labels_insufficient" and parity is None
    assert "gate labels insufficient" in reason
    assert not (tmp_path / "gate.json").exists()


def test_gate_save_load_predictions_are_exact_for_all_calibration_features(tmp_path):
    rows = []
    for label in (0, 1):
        for index in range(module.MIN_CLASS_GROUPS):
            rows.append({"role": module.CALIBRATION_ROLE,
                         "reason": "eligible_disagreement",
                         "group_id": f"{label}-{index}",
                         "features": [float(label), float(index), 0., 0., 1., .1],
                         "parent_accepted": label == 0,
                         "proposal_accepted": label == 1})
    audit = [{"id": f"row-{index}", **row} for index, row in enumerate(rows)]
    gate, status, reason, parity, counts = module.fit_and_persist_gate(
        tmp_path, rows, audit
    )
    assert gate is not None and status == "completed" and reason is None
    assert parity == {"calibration_rows": len(audit), "feature_rows": len(rows),
                      "nonapplicable_rows": 0, "exact": True,
                      "gate_sha256": module.file_sha256(tmp_path / "gate.json")}
    assert counts["accept_groups"] == counts["stay_groups"] == module.MIN_CLASS_GROUPS
    assert all(type(row["gate_accept"]) is bool for row in audit)


def test_resource_bounds_fail_closed(monkeypatch):
    monkeypatch.setattr(module, "rss_bytes", lambda: module.PROFILE["rss_limit_bytes"] + 1)
    with pytest.raises(RuntimeError, match="resource bound"):
        module.require_resources(0., "fixture")
