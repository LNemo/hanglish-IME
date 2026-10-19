import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / ".omx/artifacts/probe-balanced-word-candidates-v1.py"
SPEC = importlib.util.spec_from_file_location("balanced_word_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def candidate(source, left, right):
    return {"text": left + " " + right, "score": 0.0, "segments": [
        {"input_start": 0, "input_end": 1, "source": source[0:1],
         "text": left, "kind": "fixture"},
        {"input_start": 1, "input_end": 2, "source": source[1:2],
         "text": " ", "kind": "raw"},
        {"input_start": 2, "input_end": 3, "source": source[2:3],
         "text": right, "kind": "fixture"},
    ]}


def fixture(pool=None):
    source = "a b"
    pool = pool or [candidate(source, left, right) for left, right in
                    (("가", "나"), ("카", "다"), ("아", "라"), ("하", "마"))]
    capture = {"id": "dev-1", "input": source, "scored_pool": pool}
    prediction = {"id": "dev-1", "input": source,
                  "response": {"text": "각 낙", "reason": "complete",
                               "generated_tokens": 6}}
    return capture, prediction


def test_monotone_lcs_uses_source_leftmost_and_ignores_protected_chunks(monkeypatch):
    mapping = {"가": "a", "나": "b", "각,": "a,"}
    monkeypatch.setattr(probe, "romanize_word", mapping.__getitem__)
    aligned = probe.aligned_words("a a b 42 URL", "가 나 42 English")
    assert [(row["source_chunk_index"], row["output_chunk_index"])
            for row in aligned] == [(0, 0), (2, 1)]
    punctuated = probe.aligned_words("a, b", "각, 나")
    assert punctuated[0]["source"] == punctuated[0]["forward_roman"] == "a,"


def test_novel_alignment_excludes_forms_already_generated_for_exact_span(monkeypatch):
    mapping = {"각": "a", "낙": "b"}
    monkeypatch.setattr(probe, "romanize_word", mapping.__getitem__)
    pool = [candidate("a b", "각", right) for right in ("나", "다", "라", "마")]
    novel = probe.novel_alignments("a b", "각 낙", pool)
    assert [(row["source"], row["text"]) for row in novel] == [("b", "낙")]
    assert novel[0]["existing_span_forms"] == ["나", "다", "라", "마"]


def test_single_and_all_replacements_preserve_full_source_segments(monkeypatch):
    mapping = {"각": "a", "낙": "b"}
    monkeypatch.setattr(probe, "romanize_word", mapping.__getitem__)
    capture, prediction = fixture()
    result = probe.proposals_for_row(capture, prediction)
    assert "expected" not in result
    assert len(result["aligned_novel_words"]) == 2
    assert {row["mode"] for row in result["proposals"]} == {
        "single_span", "all_nonoverlapping"}
    assert len(result["proposals"]) == 9
    assert len({row["text"] for row in result["proposals"]}) == 9
    assert not ({row["text"] for row in result["proposals"]}
                & {row["text"] for row in capture["scored_pool"]})
    for row in result["proposals"]:
        probe.validate_segments("a b", row)


def test_proposal_cap_is_strict_and_does_not_delete_pool(monkeypatch):
    mapping = {"각": "a", "낙": "b"}
    monkeypatch.setattr(probe, "romanize_word", mapping.__getitem__)
    capture, prediction = fixture()
    before = [row.copy() for row in capture["scored_pool"]]
    result = probe.proposals_for_row(capture, prediction, limit=3)
    assert len(result["proposals"]) == 3
    assert result["original_pool_retained"] is True
    assert capture["scored_pool"] == before


def test_incomplete_or_protected_generation_yields_no_candidate(monkeypatch):
    monkeypatch.setattr(probe, "romanize_word", lambda value: value)
    capture, prediction = fixture()
    prediction["response"] = {"text": None, "reason": "token_limit",
                              "generated_tokens": 512}
    assert probe.proposals_for_row(capture, prediction)["proposals"] == []
    prediction["response"] = {"text": "English 42 URL/path", "reason": "complete",
                              "generated_tokens": 3}
    assert probe.proposals_for_row(capture, prediction)["proposals"] == []


def test_malformed_or_gapped_segment_coverage_is_rejected():
    broken = candidate("a b", "가", "나")
    broken["segments"][2]["input_start"] = 1
    with pytest.raises(ValueError, match="coverage"):
        probe.validate_segments("a b", broken)
