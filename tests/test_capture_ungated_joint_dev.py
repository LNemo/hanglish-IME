from copy import deepcopy

import pytest

from scripts.capture_ungated_joint_dev import (
    PARENT_VERSION, capture_row, selected_by_source,
)


def candidate(source, text, kind="lexicon"):
    return {"text": text, "score": 1.0, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": text, "kind": kind,
    }]}


def records(source="imnida", ranked=True):
    request = {"input": source, "context": ""}
    response = {"input": source, "context": "", "engine": "neural",
                "mode": "batch", "raw": False, "version": PARENT_VERSION,
                "sentence_ranking_applied": ranked,
                "candidates": [candidate(source, "입니다"), candidate(source, source, "raw")]}
    left = {"id": "d1", "request": request,
            "proposed": {"sentence_ranking_applied": ranked}}
    right = {"id": "d1", "request": request, "response": response}
    return left, right


def test_ranked_row_is_exact_passthrough_with_no_engine():
    source, comparator = records(ranked=True)
    row = capture_row(source, comparator, engine=None, inventory=("test",))
    assert row["selected_unranked"] is False
    assert row["passthrough_exact"] is True
    assert "response" not in row


def test_selection_comes_only_from_source_flag():
    source, _ = records(ranked=False)
    assert selected_by_source(source) is True
    source["proposed"]["sentence_ranking_applied"] = "false"
    with pytest.raises(ValueError, match="ranking flag"):
        selected_by_source(source)


class Ranker:
    def can_score(self, text):
        return text != "미지원"

    def score_pairs(self, contexts, texts):
        assert contexts == [""] * len(texts)
        return [3.0 for _ in texts]


class Model:
    def generate(self, word, beam_size=8):
        assert beam_size == 8
        return [("임니다", -0.1)] if word == "imnida" else []


class Engine:
    def __init__(self, response):
        import threading
        self.response = response
        self.lock = threading.RLock()
        self.sentence_ranker = Ranker()

    def convert(self, text):
        return deepcopy(self.response)

    def load_phonetic_model(self):
        return Model()


def test_unranked_row_keeps_fresh_baseline_trace_and_support_controls():
    source, comparator = records(ranked=False)
    engine = Engine(comparator["response"])
    row = capture_row(source, comparator, engine=engine, inventory=())
    assert row["selected_unranked"] is True
    assert row["fresh_baseline"] == comparator["response"]
    assert row["trace"]["baseline"] == comparator["response"]
    assert row["controls"]["parent_version"] == PARENT_VERSION
    assert row["controls"]["pool_size"] == (
        row["controls"]["supported"] + row["controls"]["unsupported"])


def test_identity_mismatch_is_rejected_before_inference():
    source, comparator = records()
    comparator["id"] = "other"
    with pytest.raises(ValueError, match="identity"):
        capture_row(source, comparator)
