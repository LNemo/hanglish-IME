from types import SimpleNamespace

import pytest

from scripts.audit_recombination_tokenization import summarize


def encoding(ids, unknown=False):
    return SimpleNamespace(ids=ids, tokens=["[UNK]" if unknown else "known"])


def test_unknown_gold_is_not_counted_as_wrong():
    result = summarize([{"text": "희귀이름"}], [encoding([1], True)], {"희귀이름"}, "희귀이름")
    assert result["counts"]["gold_with_unknown"] == 1
    assert result["counts"]["wrong_winner_collides_with_gold"] == 0


def test_exact_ids_detect_collision_and_wrong_winner():
    cs = [{"text": text} for text in ("정답", "오답", "다른오답")]
    result = summarize(cs, [encoding([1], True), encoding([1], True), encoding([2], True)],
                       {"정답"}, "오답")
    assert result["counts"]["collision_groups"] == 1
    assert result["counts"]["mixed_gold_groups"] == 1
    assert result["counts"]["mixed_gold_candidates"] == 2
    assert result["counts"]["wrong_winner_collides_with_gold"] == 1


def test_collision_between_only_wrong_candidates_is_separate():
    result = summarize([{"text": "가"}, {"text": "나"}],
                       [encoding([3]), encoding([3])], {"다"}, "가")
    assert result["counts"]["collision_groups"] == 1
    assert result["counts"]["mixed_gold_groups"] == 0


def test_empty_pool_and_mismatched_token_count():
    assert summarize([], [], {"가"}, "가")["counts"]["candidates"] == 0
    with pytest.raises(ValueError, match="count"):
        summarize([{"text": "가"}], [], {"가"}, "가")
