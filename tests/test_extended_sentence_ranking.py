from copy import deepcopy
import math

import pytest

from scripts import probe_extended_sentence_ranking as probe


def segment(text, marker="first"):
    return [{"input_start": 0, "input_end": len(text), "source": marker,
             "text": text, "kind": "neural"}]


def pool_candidate(text, score=1.0, marker="first"):
    return {"text": text, "score": score, "segments": segment(text, marker),
            "reason": "sentence_ranking_not_applied", "origin": "parent"}


def response(source, first="가 나"):
    raw = probe.canonical_raw(source)
    return {"input": source, "context": "", "result": first,
            "sentence_ranking_applied": False, "ranking_applied": True,
            "candidates": [{"text": first, "score": 1.0,
                            "segments": segment(first)}, raw]}


def capture(source="ga na", *, ranked=False, legacy_pool=None, expanded_pool=None):
    legacy = response(source)
    legacy["sentence_ranking_applied"] = ranked
    raw_pool = {**probe.canonical_raw(source), "reason": "appended_for_user",
                "origin": "raw_fallback"}
    old = list(legacy_pool or [pool_candidate("가 나"), pool_candidate("카 나")])
    new = list(expanded_pool or [*old, pool_candidate("가나")])
    return {"id": "row", "request": {"input": source, "context": ""},
            "legacy": legacy, "legacy_pipeline": {"pool": [*old, raw_pool]},
            "independent": {"pipeline": {"pool": [*new, deepcopy(raw_pool)]}}}


class Ranker:
    def __init__(self, scores=None, unsupported=()):
        self.scores = scores or {}
        self.unsupported = set(unsupported)
        self.calls = []

    def can_score(self, text):
        return text not in self.unsupported

    def score_pairs(self, contexts, candidates):
        self.calls.append((list(contexts), list(candidates)))
        return [self.scores.get(text, float(index))
                for index, text in enumerate(candidates)]


def test_dedupe_keeps_first_structural_path_and_excludes_every_exact_source_path():
    source = "ga na"
    first = pool_candidate("가 나", marker="first")
    second = pool_candidate("가 나", marker="second")
    same_source_but_structural = pool_candidate(source, marker="converted")
    raw = {**probe.canonical_raw(source), "reason": "appended_for_user"}
    result = probe.dedupe_pool([first, second, same_source_but_structural, raw], source)
    assert [row["text"] for row in result] == ["가 나"]
    assert result[0]["segments"][0]["source"] == "first"


def test_finite_scored_multisegment_exact_source_is_reserved_for_raw_slot():
    source = "ga na"
    exact_source = {"text": source, "score": 99.0,
                    "segments": [*segment("ga", "first"), *segment(" na", "second")],
                    "reason": "sentence_ranking_not_applied", "origin": "parent"}
    row = capture(source, legacy_pool=[exact_source, pool_candidate("가 나")])
    result = probe.capture_row(row, Ranker({"가 나": 1.0}))
    assert result["score_texts"] == ["가 나", "가나"]
    displayed = result["policies"]["parent_only_256"]["response"]["candidates"]
    assert [candidate["text"] for candidate in displayed] == ["가 나", source]
    assert displayed[-1] == probe.canonical_raw(source)


def test_shared_scoring_scores_each_text_once_with_empty_context():
    row = capture(
        legacy_pool=[pool_candidate("가 나"), pool_candidate("카 나")],
        expanded_pool=[pool_candidate("가 나", marker="independent"),
                       pool_candidate("다 라")])
    ranker = Ranker({"가 나": 3, "카 나": 2, "다 라": 1})
    result = probe.capture_row(row, ranker)
    assert ranker.calls == [(["", "", ""], ["가 나", "카 나", "다 라"])]
    assert result["score_texts"] == ["가 나", "카 나", "다 라"]
    assert result["policies"]["expanded_pool_256"]["rescored_pool"][0][
        "segments"][0]["source"] == "independent"


def test_unscoreable_policy_falls_back_wholly_without_mixing_scores():
    row = capture(
        legacy_pool=[pool_candidate("가 나"), pool_candidate("너무 긴 후보")],
        expanded_pool=[pool_candidate("가 나"), pool_candidate("다 라")])
    ranker = Ranker({"가 나": 9, "다 라": 8}, unsupported={"너무 긴 후보"})
    result = probe.capture_row(row, ranker)
    parent = result["policies"]["parent_only_256"]
    expanded = result["policies"]["expanded_pool_256"]
    assert parent["applied"] is False
    assert parent["response"] == row["legacy"]
    assert parent["rescored_pool"] == []
    assert expanded["applied"] is True
    assert ranker.calls[0][1] == ["가 나", "다 라"]


def test_supported_policy_uses_four_rescored_candidates_then_raw_last():
    source = "ga na"
    row = capture(source, legacy_pool=[pool_candidate(str(i)) for i in range(6)])
    result = probe.capture_row(row, Ranker({str(i): float(i) for i in range(6)}))
    response_value = result["policies"]["parent_only_256"]["response"]
    assert [candidate["text"] for candidate in response_value["candidates"]] == [
        "5", "4", "3", "2", source]
    assert response_value["candidates"][-1] == probe.canonical_raw(source)
    assert all(candidate["score"] is not None
               for candidate in response_value["candidates"][:-1])


def test_already_ranked_rows_remain_byte_shape_equal_for_both_policies():
    row = capture(ranked=True)
    result = probe.capture_row(row, Ranker())
    assert result["target_row"] is False
    assert result["score_texts"] == []
    for policy in probe.POLICIES:
        assert result["policies"][policy]["response"] == row["legacy"]
        assert result["policies"][policy]["reason"] == (
            "legacy_sentence_ranking_already_applied")


def test_all_312_non_target_rows_preserve_frozen_legacy_response():
    ranker = Ranker()
    for index in range(312):
        row = capture(source=f"ga na {index}", ranked=True)
        row["id"] = f"row-{index}"
        result = probe.capture_row(row, ranker)
        assert all(result["policies"][policy]["response"] == row["legacy"]
                   for policy in probe.POLICIES)
    assert ranker.calls == []


def test_empty_nonraw_pool_is_unsupported_and_preserves_legacy():
    source = "ga na"
    raw = {**probe.canonical_raw(source), "reason": "appended_for_user"}
    row = capture(source, legacy_pool=[raw], expanded_pool=[pool_candidate("가 나")])
    prepared = probe.prepare_rescore(row, lambda _: True)
    assert prepared["supported"] == {
        "parent_only_256": False, "expanded_pool_256": True}
    result = probe.capture_row(row, Ranker({"가 나": 1}))
    assert result["policies"]["parent_only_256"]["reason"] == "empty_nonraw_pool"
    assert result["policies"]["parent_only_256"]["response"] == row["legacy"]


def _summary(top1_ids=(), top5_ids=(), literal_ids=None):
    literal_ids = {"0", "1", "2", "3"} if literal_ids is None else literal_ids
    return {"counts": {"top1": len(top1_ids), "top5": len(top5_ids)},
            "outcomes": [{"id": str(index), "top1": str(index) in top1_ids,
                           "top5": str(index) in top5_ids,
                           "literal_signature_exact": str(index) in literal_ids}
                          for index in range(4)]}


def test_selection_requires_net_top1_without_regression_or_top5_loss():
    baseline = _summary(top1_ids={"0"}, top5_ids={"0", "1"})
    parent = _summary(top1_ids={"0", "1"}, top5_ids={"0", "1"})
    expanded = _summary(top1_ids={"1", "2"}, top5_ids={"0", "1", "2"})
    result = probe.compare_and_select(
        baseline, {"parent_only_256": parent, "expanded_pool_256": expanded})
    assert result["policies"]["parent_only_256"]["runtime_candidate_gate"] is True
    assert result["policies"]["expanded_pool_256"]["runtime_candidate_gate"] is False
    assert result["selected_for_runtime_evaluation"] == "parent_only_256"
    assert result["runtime_promotion"] is False


def test_literal_signature_regression_blocks_runtime_candidate_gate():
    baseline = _summary(top1_ids={"0"}, top5_ids={"0", "1"})
    parent = _summary(top1_ids={"0", "1"}, top5_ids={"0", "1"},
                      literal_ids={"0", "1", "2"})
    expanded = _summary(top1_ids={"0"}, top5_ids={"0", "1"})
    result = probe.compare_and_select(
        baseline, {"parent_only_256": parent, "expanded_pool_256": expanded})
    policy = result["policies"]["parent_only_256"]
    assert policy["literal_signature_regression_ids"] == ["3"]
    assert policy["runtime_candidate_gate"] is False
    assert result["selected_for_runtime_evaluation"] is None


def test_pool_coverage_is_distinct_from_visible_selection(monkeypatch):
    monkeypatch.setattr(probe, "EXPECTED_ROWS", 1)
    label = {"id": "row", "input": "ga na", "expected": ["정답"],
             "category": "test"}
    captured = capture()
    result = probe.capture_row(captured, Ranker({"가 나": 2, "카 나": 1, "가나": 0}))
    result["id"] = "row"
    structural = {"row": {"오답", "정답"}}
    summary = probe.summarize_policy([result], [label], "parent_only_256", structural)
    assert summary["counts"]["nonraw_pool_covered"] == 1
    assert summary["counts"]["full_pool_with_raw_covered"] == 1
    assert summary["counts"]["top1"] == 0


def test_full_pool_coverage_explicitly_includes_reserved_raw_source(monkeypatch):
    monkeypatch.setattr(probe, "EXPECTED_ROWS", 1)
    label = {"id": "row", "input": "ga na", "expected": ["ga na"],
             "category": "test"}
    captured = capture()
    result = probe.capture_row(captured, Ranker({"가 나": 2, "카 나": 1, "가나": 0}))
    result["id"] = "row"
    summary = probe.summarize_policy(
        [result], [label], "parent_only_256", {"row": {"오답"}})
    assert summary["counts"]["nonraw_pool_covered"] == 0
    assert summary["counts"]["full_pool_with_raw_covered"] == 1


def test_memory_light_text_sets_match_prior_deduped_pool_outputs():
    row = capture(
        legacy_pool=[pool_candidate("가 나", marker="first"),
                     pool_candidate("가 나", marker="duplicate"),
                     pool_candidate("카 나")],
        expanded_pool=[pool_candidate("가 나"), pool_candidate("다 라"),
                       pool_candidate("다 라", marker="duplicate")])
    full = probe.policy_pools(row)
    light = probe.policy_text_sets(row)
    assert light == {policy: {candidate["text"] for candidate in full[policy]}
                     for policy in probe.POLICIES}
    assert row["legacy_pipeline"]["pool"][0]["segments"][0]["source"] == "first"


def test_literal_signature_scope_tracks_uppercase_digits_and_punctuation():
    assert probe.literal_signature("API v2.0 test") == "API 2.0 "
    assert probe.literal_signature("API v2.0 테스트") == "API 2.0 "


def test_apply_policy_rejects_incomplete_or_nonfinite_scores():
    pool = [pool_candidate("가 나")]
    with pytest.raises(ValueError, match="incomplete"):
        probe.apply_policy(response("ga na"), pool, {}, supported=True, source="ga na")
    with pytest.raises(ValueError, match="malformed"):
        probe.apply_policy(response("ga na"), pool, {"가 나": math.nan},
                           supported=True, source="ga na")
