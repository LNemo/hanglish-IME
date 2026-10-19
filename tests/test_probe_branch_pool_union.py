from copy import deepcopy

import pytest

from scripts.probe_branch_pool_union import RecordingRanker, check_pools, compare_display


def record():
    def call(texts):
        return {"contexts": ["ctx"] * len(texts), "candidates": texts, "scores": [1.] * len(texts)}
    return {"context": "ctx", "ensemble_calls": [call(["old", "both"]), call(["both", "new"]), call(["new"])],
            "branch_pool_calls": [call(["old", "both", "new"])],
            "trace": {"fallback_reason": None, "baseline_pool": ["old", "both"],
                      "supported_pool": ["both", "new"], "union_pool": ["old", "both", "new"]}}


def test_pool_check_requires_exact_original_pools_and_no_missing_paths():
    value = record()
    check_pools(value)
    broken = deepcopy(value)
    broken["trace"]["baseline_pool"].reverse()
    with pytest.raises(ValueError, match="original model inputs"):
        check_pools(broken)
    broken = deepcopy(value)
    broken["trace"]["union_pool"].pop()
    with pytest.raises(ValueError, match="lost/reordered"):
        check_pools(broken)


def test_display_comparison_does_not_hide_score_drift_or_order_changes():
    old = {"candidates": [{"text": "a", "score": 1., "segments": []},
                          {"text": "raw", "score": None, "segments": []}]}
    new = deepcopy(old)
    new["candidates"][0]["score"] = 1.1
    comparison = compare_display(old, new)
    assert comparison["texts_equal"] and comparison["segments_equal"]
    assert not comparison["full_candidates_equal"]
    assert comparison["max_common_rounded_score_difference"] == pytest.approx(.1)
    new["candidates"].reverse()
    assert not compare_display(old, new)["texts_equal"]


def test_recorder_returns_original_scores_and_copies_mutable_arguments():
    class Ranker:
        def can_score(self, text):
            return True
        def score_pairs(self, contexts, candidates):
            return [float(len(c)) for c in candidates]
    recorder = RecordingRanker(Ranker())
    contexts, candidates = ["ctx"], ["word"]
    assert recorder.score_pairs(contexts, candidates) == [4.]
    contexts.clear()
    candidates.clear()
    assert recorder.calls == [{"contexts": ["ctx"], "candidates": ["word"], "scores": [4.]}]
