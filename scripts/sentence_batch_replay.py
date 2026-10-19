"""Replay authenticated sentence-ranker call grouping for one frozen row."""
from __future__ import annotations

import math


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class SentenceBatchReplay:
    """Preserve captured call grouping while recomputing every score.

    The adapter handles one frozen row.  A call containing the complete ordered
    original text list and empty contexts is replayed using the captured call
    groups.  Disjoint novel-candidate calls pass through unchanged.
    """

    def __init__(self, ranker, original_texts, score_calls):
        _require(hasattr(ranker, "score_pairs") and hasattr(ranker, "can_score"),
                 "underlying ranker is not initialized")
        batch_size = getattr(ranker, "batch_size", None)
        _require(type(batch_size) is int and batch_size == 8,
                 "underlying ranker batch size differs")
        _require(isinstance(original_texts, list)
                 and all(isinstance(text, str) and text for text in original_texts)
                 and len(set(original_texts)) == len(original_texts),
                 "original text domain differs")
        _require(isinstance(score_calls, list), "score-call records must be a list")

        calls, covered = [], []
        for index, record in enumerate(score_calls):
            _require(isinstance(record, dict) and set(record) == {
                "contexts", "texts", "scores", "token_lengths", "batch_size"},
                f"score-call record {index} schema differs")
            contexts, texts = record["contexts"], record["texts"]
            scores, lengths = record["scores"], record["token_lengths"]
            _require(isinstance(contexts, list) and isinstance(texts, list)
                     and isinstance(scores, list) and isinstance(lengths, list)
                     and texts and len(contexts) == len(texts) == len(scores) == len(lengths),
                     f"score-call record {index} shape differs")
            _require(all(context == "" for context in contexts)
                     and all(isinstance(text, str) and text for text in texts),
                     f"score-call record {index} source-only context differs")
            _require(type(record["batch_size"]) is int
                     and record["batch_size"] == 8
                     and all(_finite(score) for score in scores)
                     and all(type(length) is int and length > 0 for length in lengths),
                     f"score-call record {index} numeric contract differs")
            calls.append((list(contexts), list(texts)))
            covered.extend(texts)

        _require(len(covered) == len(set(covered)),
                 "score-call records contain duplicate original texts")
        _require(set(covered) == set(original_texts) and len(covered) == len(original_texts),
                 "score-call records do not cover the original texts exactly once")
        _require(all(ranker.can_score(text) for text in original_texts),
                 "an original text is unsupported by the underlying ranker")
        self._ranker = ranker
        self._original_texts = tuple(original_texts)
        self._original_set = frozenset(original_texts)
        self._calls = tuple((tuple(contexts), tuple(texts)) for contexts, texts in calls)
        self.batch_size = batch_size

    def can_score(self, text):
        return self._ranker.can_score(text)

    def score_pairs(self, contexts, texts):
        _require(isinstance(contexts, list) and isinstance(texts, list),
                 "contexts and texts must be lists")
        _require(len(contexts) == len(texts),
                 "contexts and texts must have equal length")
        if not texts:
            return self._ranker.score_pairs(contexts, texts)
        if tuple(texts) == self._original_texts and all(context == "" for context in contexts):
            recomputed = {}
            for recorded_contexts, recorded_texts in self._calls:
                values = list(self._ranker.score_pairs(
                    list(recorded_contexts), list(recorded_texts)))
                _require(len(values) == len(recorded_texts)
                         and all(_finite(value) for value in values),
                         "underlying ranker returned invalid replay scores")
                recomputed.update(zip(recorded_texts, map(float, values)))
            _require(set(recomputed) == self._original_set,
                     "replayed scores do not cover the original texts")
            return [recomputed[text] for text in self._original_texts]
        _require(self._original_set.isdisjoint(texts),
                 "partial or reordered original score calls are ambiguous")
        return self._ranker.score_pairs(contexts, texts)


__all__ = ["SentenceBatchReplay"]
