import pytest

from scripts.sentence_batch_replay import SentenceBatchReplay


class GroupSensitiveRanker:
    batch_size = 8

    def __init__(self, unsupported=()):
        self.unsupported = set(unsupported)
        self.calls = []

    def can_score(self, text):
        return text not in self.unsupported

    def score_pairs(self, contexts, texts):
        self.calls.append((list(contexts), list(texts)))
        # The group size intentionally changes every score.
        return [float(ord(text[0]) + 100 * len(texts)) for text in texts]


def records():
    return [
        {"contexts": ["", ""], "texts": ["b", "a"],
         "scores": [999.0, 998.0], "token_lengths": [2, 2], "batch_size": 8},
        {"contexts": [""], "texts": ["c"],
         "scores": [997.0], "token_lengths": [2], "batch_size": 8},
    ]


def test_original_scores_replay_captured_groups_and_return_original_order():
    ranker = GroupSensitiveRanker()
    replay = SentenceBatchReplay(ranker, ["a", "b", "c"], records())
    values = replay.score_pairs(["", "", ""], ["a", "b", "c"])
    assert values == [297.0, 298.0, 199.0]
    assert ranker.calls == [(["", ""], ["b", "a"]), ([""], ["c"])]


def test_stored_scores_are_never_returned():
    ranker = GroupSensitiveRanker()
    replay = SentenceBatchReplay(ranker, ["a", "b", "c"], records())
    assert replay.score_pairs(["", "", ""], ["a", "b", "c"]) != [998.0, 999.0, 997.0]


def test_novel_and_empty_calls_delegate_unchanged_and_can_score_delegates():
    ranker = GroupSensitiveRanker(unsupported={"novel-x"})
    replay = SentenceBatchReplay(ranker, ["a", "b", "c"], records())
    assert replay.can_score("novel-x") is False
    assert replay.score_pairs(["", ""], ["novel-y", "novel-z"]) == [310.0, 310.0]
    assert replay.score_pairs([], []) == []
    assert ranker.calls[-2:] == [(["", ""], ["novel-y", "novel-z"]), ([], [])]


@pytest.mark.parametrize("mutate,match", [
    (lambda rows: rows[0].update(contexts=["context", ""]), "context"),
    (lambda rows: rows[0].update(scores=[1.0]), "shape"),
    (lambda rows: rows[0].update(token_lengths=[2, 0]), "numeric"),
    (lambda rows: rows[0].update(batch_size=4), "numeric"),
    (lambda rows: rows[1].update(texts=["a"]), "duplicate"),
    (lambda rows: rows.pop(), "cover"),
])
def test_rejects_malformed_duplicate_missing_or_context_records(mutate, match):
    rows = records(); mutate(rows)
    with pytest.raises(ValueError, match=match):
        SentenceBatchReplay(GroupSensitiveRanker(), ["a", "b", "c"], rows)


def test_rejects_unsupported_original_and_partial_or_reordered_original_calls():
    with pytest.raises(ValueError, match="unsupported"):
        SentenceBatchReplay(GroupSensitiveRanker({"b"}), ["a", "b", "c"], records())
    replay = SentenceBatchReplay(GroupSensitiveRanker(), ["a", "b", "c"], records())
    for texts in (["a", "b"], ["b", "a", "c"]):
        with pytest.raises(ValueError, match="ambiguous"):
            replay.score_pairs([""] * len(texts), list(texts))


def test_requires_exact_integer_batch_size_eight_on_ranker_and_records():
    for batch_size in (4, True):
        ranker = GroupSensitiveRanker()
        ranker.batch_size = batch_size
        with pytest.raises(ValueError, match="batch size"):
            SentenceBatchReplay(ranker, ["a", "b", "c"], records())
    rows = records()
    rows[0]["batch_size"] = True
    with pytest.raises(ValueError, match="numeric"):
        SentenceBatchReplay(GroupSensitiveRanker(), ["a", "b", "c"], rows)
