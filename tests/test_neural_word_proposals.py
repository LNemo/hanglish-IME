from copy import deepcopy

import pytest

from hanglish.core.neural_word_proposals import aligned_words, sentence_extras


def paths(source):
    boundaries = []
    for index, token in enumerate(source.split(" ")):
        if index:
            boundaries.append((" ", " ", "raw"))
        text = {
            "banggeum": "방금", "geu": "그", "sarameun": "사람은",
            "pyeonggajaimnida": ("평가자입니다" if index == 3 else token),
        }.get(token, token)
        boundaries.append((token, text, "fixture"))

    result = []
    for variant in ("평가자입니다", "평가자임니다", "평가자이다", "평가자다"):
        cursor = 0
        segments = []
        for raw, text, kind in boundaries:
            if raw == "pyeonggajaimnida":
                text = variant
            segments.append({"input_start": cursor, "input_end": cursor + len(raw),
                             "source": raw, "text": text, "kind": kind})
            cursor += len(raw)
        result.append(("".join(item["text"] for item in segments),
                       float(10 - len(result)), segments))
    return result


def test_frozen_probe_policy_aligns_and_orders_word_replacements_exactly():
    source = "banggeum geu sarameun pyeonggajaimnida"
    generated = "방금 그 사라믄 평가지만다"
    parent = paths(source)
    frozen = deepcopy(parent)
    alignments = aligned_words(source, generated)
    assert [(item["source"], item["text"]) for item in alignments] == [
        ("banggeum", "방금"), ("geu", "그"), ("sarameun", "사라믄")]
    extras = sentence_extras(source, generated, parent)
    assert [item[0] for item in extras] == [
        "방금 그 사라믄 평가자입니다", "방금 그 사라믄 평가자임니다",
        "방금 그 사라믄 평가자이다", "방금 그 사라믄 평가자다",
    ]
    assert all(item[1] == 0.0 for item in extras)
    assert all(next(segment for segment in item[2]
                    if segment["source"] == "sarameun")["kind"]
               == "balanced_word_probe" for item in extras)
    assert parent == frozen


def test_repeated_alignment_is_source_leftmost_and_punctuation_round_trips():
    aligned = aligned_words("ga ga na", "가 나")
    assert [(item["source_chunk_index"], item["output_chunk_index"])
            for item in aligned] == [(0, 0), (2, 1)]
    assert [(item["source"], item["text"])
            for item in aligned_words("ga, na", "가, 나")] == [("ga,", "가,"), ("na", "나")]


def test_existing_spelling_is_not_proposed_and_duplicate_pool_is_rejected():
    source = "banggeum geu sarameun pyeonggajaimnida"
    parent = paths(source)
    assert sentence_extras(source, "방금 그 사람은 평가자입니다", parent) == []
    with pytest.raises(ValueError, match="duplicates"):
        sentence_extras(source, "방금 그 사라믄 평가자입니다",
                        [parent[0], parent[0], parent[2], parent[3]])


def test_limit_and_complete_segment_source_preservation():
    source = "banggeum geu sarameun pyeonggajaimnida"
    extras = sentence_extras(source, "방금 그 사라믄 평가지만다", paths(source), limit=2)
    assert len(extras) == 2
    for text, _, segments in extras:
        assert "".join(segment["source"] for segment in segments) == source
        assert "".join(segment["text"] for segment in segments) == text
