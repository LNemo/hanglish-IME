import copy

import pytest

import scripts.probe_constrained_sentence_union as module


def test_union_pool_preserves_parent_order_and_deduplicates_new_candidates():
    assert module.union_pool(["a", "b"], ["b", "c", "a", "d"], 4) == [
        "a", "b", "c", "d"
    ]
    assert module.union_pool(None, ["new"], 4) is None
    with pytest.raises(ValueError, match="bound"):
        module.union_pool(["a", "b"], ["c", "d", "e"], 4)


def test_visible_candidates_keeps_runtime_raw_append_semantics():
    ranked = [str(index) for index in range(6)]
    assert module.visible_candidates(ranked, "2") == ["0", "1", "2", "3", "4"]
    assert module.visible_candidates(ranked, "raw") == ["0", "1", "2", "3", "raw"]


class Ranker:
    def __init__(self, case=None):
        self.calls = []
        self.case = case

    def score_pairs(self, contexts, candidates):
        if self.case is not None:
            assert self.case.expected_reads == 0
        self.calls.append((contexts, candidates))
        return [0., 1., 2.]


class TraceCase(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expected_reads = 0

    def __getitem__(self, key):
        if key == "expected":
            self.expected_reads += 1
        return super().__getitem__(key)


def test_evaluate_case_scores_source_only_union_before_reading_targets():
    case = TraceCase({
        "id": "case", "input": "raw", "context": "ctx", "expected": ["new"],
        "before_candidates": ["parent", "raw"],
        "before_pool": ["parent", "second"],
        "new_pool": ["parent", "new"],
    })
    ranker = Ranker(case)

    result = module.evaluate_case(case, ranker)

    assert ranker.calls == [(["ctx", "ctx", "ctx"], ["parent", "second", "new"])]
    assert case.expected_reads > 0
    assert result["actual"] == ["new", "second", "parent", "raw"]
    assert result["union_pool"] == ["parent", "second", "new"]
    assert result["top1"] is True and result["union_pool_covered"] is True
    assert result["added_pool_candidates"] == 1


def test_parent_fallback_keeps_recorded_output_without_sentence_scoring():
    case = TraceCase({
        "id": "fallback", "input": "raw", "context": "", "expected": ["parent"],
        "before_candidates": ["parent", "raw"], "before_pool": None,
        "new_pool": ["new"],
    })
    ranker = Ranker()

    result = module.evaluate_case(case, ranker)

    assert ranker.calls == []
    assert result["actual"] == ["parent", "raw"]
    assert result["union_pool_size"] is None
    assert result["union_pool_covered"] is None
    assert result["sentence_ranking_applied"] is False


def test_summary_reports_individual_corrections_and_regressions():
    base = {
        "before_top5": True, "top5": True, "before_pool_covered": True,
        "new_pool_covered": True, "union_pool_covered": True,
        "first_choice_changed": True,
    }
    cases = [
        {**copy.copy(base), "id": "fix", "before_top1": False, "top1": True},
        {**copy.copy(base), "id": "loss", "before_top1": True, "top1": False},
    ]

    result = module.summarize(cases)
    assert result["corrections"] == ["fix"]
    assert result["regressions"] == ["loss"]
    assert result["first_choice_changes"] == 2
