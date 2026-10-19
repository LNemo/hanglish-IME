from copy import deepcopy
from functools import lru_cache
import threading
from types import SimpleNamespace

import pytest

from hanglish.core.branch_pool_converter import BranchPoolConverter, collect_pool, union_paths
from hanglish.core.lattice import WordOption
from hanglish.core.neural_converter import NeuralConverter


class Ranker:
    def __init__(self):
        self.calls = []
        self.invalid = False

    def can_score(self, text):
        return "oversized" not in text

    def score_pairs(self, contexts, candidates):
        self.calls.append((list(contexts), list(candidates)))
        return [float("nan") if self.invalid else float(t.count("병")) for t in candidates]


def setup():
    word, sentence = Ranker(), Ranker()
    branches = []
    for texts in (("갑", "을"), ("병", "정")):
        branch = NeuralConverter(external_lexicon=False, ranker_kind="sentence")
        branch.models_loaded = True
        branch.ranker, branch.sentence_ranker = word, sentence
        branch.model_artifacts = {}
        branch.word_options = lru_cache()(lambda source, learned, texts=texts: tuple(
            [WordOption(t, 2., "neural") for t in texts] + [WordOption(source, 0., "raw")]))
        branches.append(branch)
    models = SimpleNamespace(baseline=branches[0], supported=branches[1], lock=threading.RLock(),
                             version="fake", model_artifacts={}, load_models=lambda: None)
    return BranchPoolConverter(models, experimental=True), sentence


def test_collect_matches_actual_converter_sentence_input_without_scoring():
    converter, sentence = setup()
    for branch in (converter.models.baseline, converter.models.supported):
        branch.convert("roman roman", "ctx")
        actual = sentence.calls[-1][1]
        before = len(sentence.calls)
        extracted = collect_pool(branch, "roman roman", "ctx")
        assert extracted.reason is None
        assert [p[0] for p in extracted.paths] == actual
        assert len(sentence.calls) == before


def test_keeps_branch_beams_separate_and_scores_union_once():
    converter, sentence = setup()
    result = converter.convert("roman roman", "ctx")
    assert len(sentence.calls) == 1
    trace = converter.last_trace
    assert set(trace["union_pool"]) == set(trace["baseline_pool"]) | set(trace["supported_pool"])
    assert "갑 병" not in trace["union_pool"]
    assert "갑 을" in trace["union_pool"] and "병 정" in trace["union_pool"]
    assert result["result"] == "병 병"
    assert result["candidates"][-1]["text"] == "roman roman"


def test_duplicate_keeps_baseline_segments_and_priority():
    old = ("test를", 2., [{"kind": "mixed"}])
    new = ("test를", 99., [{"kind": "neural"}])
    assert union_paths([old], [new]) == [old]


def test_numeric_extras_match_actual_admission_and_reject_oversized_extra():
    converter, sentence = setup()
    branch = converter.models.baseline
    extras = [("추가", 0., [])]
    branch._numeric_sentence_extras = lambda *args: extras
    assert collect_pool(branch, "roman roman", "").paths[-1] == extras[0]
    extras[:] = [("oversized", 0., [])]
    pool = collect_pool(branch, "roman roman", "")
    assert pool.reason is None and "oversized" not in [p[0] for p in pool.paths]


def test_hybrid_single_token_falls_back_and_raw_does_not_load():
    converter, sentence = setup()
    converter.models.baseline.ranker_kind = "hybrid"
    expected = converter.models.baseline.convert("roman")
    result = converter.convert("roman")
    assert result["candidates"] == expected["candidates"]
    assert result["branch_pool_union"]["fallback_reason"] == "sentence_ranking_unavailable"
    assert not sentence.calls
    converter.models.load_models = lambda: pytest.fail("raw mode loaded models")
    result = converter.convert("https://example.com test", raw=True)
    assert result["result"] == "https://example.com test" and converter.last_trace is None


def test_trace_reset_and_protected_spans_preserved():
    converter, _ = setup()
    response = converter.convert("https://example.com roman")
    trace = converter.last_trace
    snapshot = deepcopy(trace)
    for candidate in response["candidates"]:
        assert candidate["text"].startswith("https://example.com ")
    converter.convert("next next")
    assert trace == snapshot and converter.last_trace is not trace


def test_invalid_scores_are_not_displayed():
    converter, sentence = setup()
    sentence.invalid = True
    with pytest.raises(ValueError, match="invalid union scores"):
        converter.convert("roman roman")


def test_explicit_experimental_mode():
    with pytest.raises(ValueError, match="experimental"):
        BranchPoolConverter(None)


def test_oversized_original_pool_skips_numeric_expansion():
    converter, _ = setup()
    branch = converter.models.baseline
    branch._numeric_sentence_extras = lambda *args: pytest.fail("expanded oversized pool")
    assert collect_pool(branch, "oversized roman", "").reason == "sentence_budget_exceeded"
