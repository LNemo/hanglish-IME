import copy

import pytest

import scripts.evaluate_final_choice_holdout as module


def segment(source, text, kind="syllable"):
    return {"input_start": 0, "input_end": len(source), "source": source,
            "text": text, "kind": kind}


def candidate(source, text, score, kind="syllable"):
    return {"text": text, "score": score, "segments": [segment(source, text, kind)]}


class TracedRow(dict):
    def __init__(self, values):
        super().__init__(values)
        self.label_reads = 0

    def __getitem__(self, key):
        if key == "expected":
            self.label_reads += 1
        return super().__getitem__(key)


class Ranker:
    def __init__(self, scores):
        self.scores = scores

    def can_score(self, candidate):
        return True

    def score_pairs(self, contexts, candidates):
        return list(self.scores)


def row(expected=("가",)):
    source = "ga"
    candidates = [candidate(source, "카", 2.), candidate(source, "가", 1.),
                  candidate(source, source, None, "raw")]
    return TracedRow({"id": "id", "group_id": "group", "input": source,
                      "context": "문맥", "expected": list(expected),
                      "candidates": candidates, "finite_indices": [0, 1],
                      "selector_eligible": True, "all_candidates_can_score": True,
                      "candidate_can_score": [True, True, True],
                      "raw_bypass_response": {
                          "result": source,
                          "candidates": [candidate(source, source, 0., "raw")]}})


def accepting_gate():
    return {"schema": "fixture"}


def test_evaluation_scores_and_policy_before_labels_and_reorders_existing_objects(monkeypatch):
    value = row()
    original_ids = [id(item) for item in value["candidates"]]
    monkeypatch.setattr(module, "predict_accept", lambda gate, features: True)
    result = module.evaluate_case(Ranker([.1, 2., -1.]), accepting_gate(), value,
                                  "natural_holdout")
    assert value.label_reads == 1
    assert result["proposal_index"] == result["chosen_index"] == 1
    assert result["before_top1"] is False and result["top1"] is True
    assert [id(item) for item in value["candidates"]] == original_ids
    assert result["object_identity_preserved"] is True
    assert result["before_candidates"][1] == result["candidates"][0]
    assert result["raw_bypass_response"] == value["raw_bypass_response"]


def test_protected_change_is_rejected_without_gate_call(monkeypatch):
    value = row(expected=("일",))
    value["input"] = "1il"
    value["raw_bypass_response"] = {
        "result": "1il", "candidates": [candidate("1il", "1il", 0., "raw")]}
    value["candidates"] = [candidate("1il", "1일", 2., "numeric"),
                           candidate("1il", "일", 1., "numeric")]
    value["finite_indices"] = [0, 1]
    value["candidate_can_score"] = [True, True]
    monkeypatch.setattr(module, "predict_accept",
                        lambda *args: (_ for _ in ()).throw(AssertionError("gate called")))
    result = module.evaluate_case(Ranker([.1, 2.]), accepting_gate(), value,
                                  "challenge_preservation")
    assert result["reason"] == "protected_or_unaligned"
    assert result["chosen_index"] == 0 and result["protected_segments_preserved"] is True


def metric(before, after, *, corrections=(), regressions=(), actions=1):
    return {"rows": 1, "before_top1": before, "top1": after,
            "before_top5": 1, "top5": 1, "corrections": list(corrections),
            "regressions": list(regressions), "top5_losses": [],
            "eligible_disagreements": actions, "accepted_actions": actions,
            "missing_candidate_coverage": 0}


def test_holdout_gate_requires_natural_gain_no_regressions_and_action():
    suites = {"natural_holdout": metric(10, 12, corrections=("a", "b")),
              "challenge_preservation": metric(20, 20)}
    assert module.holdout_gate(suites)["passed"] is True
    suites["challenge_preservation"]["regressions"] = ["loss"]
    assert module.holdout_gate(suites)["passed"] is False
    suites["challenge_preservation"]["regressions"] = []
    suites["natural_holdout"]["accepted_actions"] = 0
    suites["challenge_preservation"]["accepted_actions"] = 0
    assert module.holdout_gate(suites)["inconclusive"] is True


def test_summary_retains_missing_candidate_coverage_and_row_regressions():
    rows = [{"id": "a", "before_top1": True, "top1": False,
             "before_top5": False, "top5": False, "gate_applied": True,
             "gate_accepted": True, "missing_candidate_coverage": True}]
    summary = module.summarize(rows)
    assert summary["regressions"] == ["a"]
    assert summary["missing_candidate_coverage"] == 1


def test_invalid_role_is_rejected_before_file_read(tmp_path):
    with pytest.raises(ValueError, match="only holdout"):
        module.load_evaluation_capture(tmp_path, "0" * 64, "natural_calibration")


def test_candidate_score_shape_and_budget_fail_before_labels(monkeypatch):
    value = row()
    with pytest.raises(ValueError, match="complete candidate"):
        module.evaluate_case(Ranker([float("nan"), 1., 0.]), accepting_gate(), value,
                             "natural_holdout")
    assert value.label_reads == 0

    class BadBudget(Ranker):
        def can_score(self, candidate):
            return candidate != "ga"

    with pytest.raises(ValueError, match="budget differs"):
        module.evaluate_case(BadBudget([0., 1., 2.]), accepting_gate(), value,
                             "natural_holdout")
    assert value.label_reads == 0


def test_resource_limit_fails_closed(monkeypatch):
    monkeypatch.setattr(module, "rss_bytes", lambda: module.PROFILE["rss_limit_bytes"] + 1)
    with pytest.raises(RuntimeError, match="resource bound"):
        module.require_resources(0., "fixture")


def test_raw_policy_exercises_raw_mode_and_bypasses_gate_before_labels(monkeypatch):
    value = row()
    real_policy = module.decision_features
    calls = []

    def policy(source, context, candidates, scores, *, raw=False):
        calls.append(raw)
        assert value.label_reads == 0
        return real_policy(source, context, candidates, scores, raw=raw)

    monkeypatch.setattr(module, 'decision_features', policy)
    monkeypatch.setattr(module, 'predict_accept',
                        lambda *args: (_ for _ in ()).throw(AssertionError('raw gate called')))
    evidence = module.verify_raw_policy(value)
    assert calls == [True]
    assert evidence['preserved'] and evidence['chosen_index'] == 0
    assert evidence['gate_applied'] is False
    assert value.label_reads == 0


def test_raw_policy_regression_is_rejected(monkeypatch):
    monkeypatch.setattr(module, 'decision_features',
                        lambda *args, **kwargs: {'reason': 'eligible_disagreement',
                                                'proposal_index': 1})
    with pytest.raises(RuntimeError, match='retain the parent'):
        module.verify_raw_policy(row())
