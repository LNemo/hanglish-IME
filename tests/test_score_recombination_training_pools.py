import json

import pytest

from scripts import score_recombination_training_pools as census


def row(**updates):
    value = {"id": "train-1", "input": "imnida test", "expected": ["입니다 테스트"],
             "group_id": "g1", "source_ids": ["s1"], "component_id": "c1",
             "split": "inner_hold", "parent_result": "임니다 테스트",
             "parent_candidates": ["임니다 테스트", "imnida test"],
             "original_candidates": ["임니다 테스트", "imnida test"],
             "recombined_candidates": ["임니다 테스트", "입니다 테스트", "imnida test"],
             "novel_candidates": ["입니다 테스트"]}
    value.update(updates)
    return value


class Ranker:
    def __init__(self, scores=None, unsupported=()):
        self.values = scores or {"임니다 테스트": 1.0, "입니다 테스트": 2.0}
        self.unsupported = set(unsupported)
        self.calls = []
        self.fit_calls = []

    def can_score(self, text):
        self.fit_calls.append(text)
        return text not in self.unsupported

    def score_pairs(self, contexts, candidates):
        self.calls.append((list(contexts), list(candidates)))
        return [self.values[value] for value in candidates]


def test_source_projection_excludes_raw_and_labels():
    projected = census.source_projection(row())
    assert "expected" not in projected
    assert projected["original_candidates"] == ["임니다 테스트"]
    assert projected["recombined_candidates"] == ["임니다 테스트", "입니다 테스트"]


def test_scores_shared_text_once_with_empty_context_and_no_label():
    ranker = Ranker()
    result = census.score_row(census.source_projection(row()), ranker)
    assert ranker.fit_calls == ["임니다 테스트", "입니다 테스트"]
    assert ranker.calls == [(["", ""], ["임니다 테스트", "입니다 테스트"])]
    assert result["score_texts"] == ["임니다 테스트", "입니다 테스트"]
    assert "입니다 테스트" not in ranker.calls[0][0]


def test_raw_is_reserved_as_last_slot():
    result = census.score_row(census.source_projection(row()), Ranker())
    assert result["recombined"]["result"] == "입니다 테스트"
    assert result["recombined"]["candidates"][-1] == "imnida test"
    assert result["recombined"]["candidates"].count("imnida test") == 1


def test_unsupported_union_falls_back_whole_without_mixed_scores():
    ranker = Ranker(unsupported={"입니다 테스트"})
    result = census.score_row(census.source_projection(row()), ranker)
    assert result["original"]["applied"] is True
    assert result["recombined"]["applied"] is False
    assert result["recombined"]["result"] == "임니다 테스트"
    assert result["recombined"]["candidates"] == ["임니다 테스트", "imnida test"]
    assert ranker.calls == [([""], ["임니다 테스트"])]


def test_fallback_preserves_displayed_parent_when_pool_order_differs():
    raw = row(parent_result="imnida test",
              parent_candidates=["imnida test", "표시 후보"],
              original_candidates=["풀1", "풀2", "풀3", "풀4", "표시 후보"],
              recombined_candidates=["풀1", "풀2", "풀3", "풀4", "표시 후보", "새 후보"],
              novel_candidates=["새 후보"])
    ranker = Ranker(scores={value: 1.0 for value in raw["recombined_candidates"]},
                    unsupported={"새 후보"})
    result = census.score_row(census.source_projection(raw), ranker)
    assert result["recombined"] == {"applied": False,
                                     "result": "imnida test",
                                     "candidates": ["imnida test", "표시 후보"]}


def test_equal_scores_use_text_as_stable_tie_breaker():
    ranker = Ranker(scores={"임니다 테스트": 1.0, "입니다 테스트": 1.0})
    result = census.score_row(census.source_projection(row()), ranker)
    assert result["recombined"]["candidates"][:2] == ["임니다 테스트", "입니다 테스트"]


def test_inconsistent_candidate_partition_is_rejected():
    with pytest.raises(ValueError, match="partition"):
        census.source_projection(row(novel_candidates=["없는 후보"]))


def capture_for(source, *, original, union, original_applied=True, union_applied=True):
    return {"id": source["id"], "input": source["input"],
            "split": source["split"], "component_id": source["component_id"],
            "score_texts": [], "scores": [], "novel_count": 1,
            "original": {"applied": original_applied, "result": original,
                         "candidates": [original, source["input"]]},
            "recombined": {"applied": union_applied, "result": union,
                           "candidates": [union, source["input"]]}}


def test_metrics_and_adequacy_gate_arithmetic():
    sources = []
    captures = []
    labels = []
    for index in range(64):
        raw = row(id=f"r{index}", component_id=f"c{index}")
        source = census.source_projection(raw)
        sources.append(source); labels.append(raw)
        union = "오답" if index < 16 else "입니다 테스트"
        captures.append(capture_for(source, original="임니다 테스트", union=union))
    metric = census.summarize(captures, sources, labels, "inner_hold")
    gate = census.adequacy_gate(metric)
    assert metric["counts"]["adequacy_eligible"] == 64
    assert len(metric["adequacy_eligible_parent_wrong_ids"]) == 16
    assert gate["passed"] is True


def test_adequacy_does_not_require_original_policy_support():
    raw = row()
    source = census.source_projection(raw)
    capture = capture_for(source, original="임니다 테스트", union="입니다 테스트",
                          original_applied=False, union_applied=True)
    metric = census.summarize([capture], [source], [raw], "inner_hold")
    assert metric["counts"]["adequacy_eligible"] == 1


def test_gate_rejects_support_or_parent_error_shortfall():
    inner = {"counts": {"adequacy_eligible": 63},
             "adequacy_eligible_parent_wrong_ids": [f"r{x}" for x in range(16)]}
    assert census.adequacy_gate(inner)["passed"] is False
    inner["counts"]["adequacy_eligible"] = 64
    inner["adequacy_eligible_parent_wrong_ids"].pop()
    assert census.adequacy_gate(inner)["passed"] is False


def test_validate_inputs_rejects_unverified_proof_before_model_import(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "prepared.jsonl").write_text("")
    (tmp_path / "source-capture.jsonl").write_text("")
    (tmp_path / "capture.complete.json").write_text("{}")
    proof = tmp_path / "proof.json"
    proof.write_text(json.dumps({"verified": False}))
    with pytest.raises(ValueError):
        census.validate_inputs(tmp_path, proof)


def test_self_test_is_model_free():
    assert census.self_test() == {"schema": census.SCHEMA, "self_test": True}


def test_frozen_parent_model_is_predeclared_and_present():
    pins = census.frozen_model_pins()
    assert set(pins.values()) == set(census.EXPECTED_MODEL.values())
