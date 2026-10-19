from functools import lru_cache
from types import SimpleNamespace
import threading

import pytest

from hanglish.core.lattice import WordOption
from hanglish.core.neural_converter import NeuralConverter
from hanglish.core.proposal_union_converter import ProposalUnionConverter, merge_options


def test_merge_preserves_parent_prefix_scores_protection_and_distinct_child_options():
    old = (WordOption("test를", 8., "mixed"), WordOption("testreul", 0., "raw"))
    new = (WordOption("test를", 99., "neural"), WordOption("새", 1., "neural"),
           WordOption("새", 2., "neural"))
    merged = merge_options(old, new)
    assert merged[:len(old)] == old
    assert merged[len(old):] == (new[1],)


class Ranker:
    def __init__(self):
        self.calls = []

    def can_score(self, text):
        return True

    def score_pairs(self, contexts, candidates):
        self.calls.append((list(contexts), list(candidates)))
        return [10. * text.count("새") for text in candidates]


def make_converter():
    baseline = NeuralConverter(external_lexicon=False, ranker_kind="sentence")
    supported = NeuralConverter(external_lexicon=False, ranker_kind="sentence")
    baseline.ranker, baseline.sentence_ranker = Ranker(), Ranker()
    baseline.models_loaded = True
    baseline.model_artifacts = {}
    baseline.word_options = lru_cache()(lambda source, learned: (
        WordOption("구", 2., "neural"), WordOption(source, 0., "raw")))
    supported.word_options = lru_cache()(lambda source, learned: (
        WordOption("새", 3., "neural"), WordOption(source, 0., "raw")))
    calls = []
    models = SimpleNamespace(baseline=baseline, supported=supported,
                             load_models=lambda: calls.append("load"),
                             version="test-models", model_artifacts={}, lock=threading.RLock())
    return ProposalUnionConverter(models, experimental=True), models, calls


def test_one_word_and_sentence_ranker_call_and_new_candidate_reaches_sentence_pool():
    engine, models, calls = make_converter()
    response = engine.convert("roman roman", "context")
    assert len(models.baseline.ranker.calls) == 1
    assert len(models.baseline.sentence_ranker.calls) == 1
    assert response["result"] == "새 새"
    assert "새 새" in engine.last_trace["sentence_pool"]
    assert engine.last_trace["words"][0]["before_ranking"] == ["구", "roman", "새"]
    assert response["version"].startswith("experimental-proposal-union-1-")
    assert calls == ["load"]


def test_raw_and_non_neural_bypass_loading_and_clear_trace():
    engine, models, calls = make_converter()
    response = engine.convert("testreul", raw=True)
    assert response["result"] == "testreul" and not calls
    engine.convert("roman roman")
    assert engine.last_trace
    response = engine.convert("imnida", raw=True)
    assert response["result"] == "imnida" and engine.last_trace is None
    response = engine.convert("roman", engine="rules")
    assert engine.last_trace is None and calls == ["load"]


def test_next_trace_does_not_mutate_previous_request_and_cache_is_independent():
    engine, models, _ = make_converter()
    engine.convert("roman roman")
    trace = engine.last_trace
    engine.convert("next")
    assert trace["words"][0]["source"] == "roman"
    assert engine.last_trace["words"][0]["source"] == "next"
    assert engine.word_options is not models.baseline.word_options
    assert engine.word_options is not models.supported.word_options


def test_explicit_experimental_opt_in():
    with pytest.raises(ValueError, match="experimental"):
        ProposalUnionConverter(None)
