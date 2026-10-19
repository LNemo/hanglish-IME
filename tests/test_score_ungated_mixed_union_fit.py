from scripts.score_ungated_mixed_union_fit import project_row, score_row


def raw(source):
    return {"text": source, "score": None, "segments": [{"input_start": 0,
        "input_end": len(source), "source": source, "text": source, "kind": "raw"}]}


def baseline(support="supported"):
    source = "a1"
    pool = [{"text": "기존", "support": support, "provenance": ["parent"]},
            {"text": "중복", "support": "supported", "provenance": ["joint"]}]
    return {"id": "x", "request": {"input": source, "context": ""},
            "fallback_public": [{"text": "fallback" + str(i), "segments": [{}]}
                                for i in range(4)] + [raw(source)],
            "supported_scores": [{"text": "기존", "score": 2.0},
                                 {"text": "중복", "score": 1.0}],
            "arms": {"original": {}, "roman_repaired": {
                "pool": pool, "supported_order": ["기존", "중복"],
                "public": [{"text": "기존"}, {"text": "중복"}, raw(source)]}}}


def mixed():
    return {"id": "x", "request": {"input": "a1", "context": ""},
            "proposals": [{"text": "중복", "token_support": {"supported": True}},
                          {"text": "추가", "token_support": {"supported": False}}]}


class Ranker:
    def can_score(self, text): return text != "추가" and text != "기존"
    def score_pairs(self, contexts, texts): return [1.0 for _ in texts]


def test_projection_preserves_baseline_dedups_and_keeps_unsupported():
    row = project_row(baseline(), mixed())
    assert [x["text"] for x in row["baseline"]] == ["기존", "중복"]
    assert [x["text"] for x in row["mixed_union"]] == ["기존", "중복", "추가"]
    assert row["mixed_union"][1]["provenance"] == ["joint", "mixed_boundary"]
    assert row["mixed_union"][-1]["support"] == "unsupported"


class StableRanker:
    def can_score(self, text): return text != "추가"
    def score_pairs(self, contexts, texts):
        values = {"기존": 2.0, "중복": 1.0}; return [values[x] for x in texts]


def test_score_preserves_raw_reserve_and_baseline_parity():
    projected = project_row(baseline(), mixed())
    row = score_row(projected, StableRanker())
    assert row["baseline_max_score_delta"] == 0
    assert row["arms"]["mixed_union"]["public"][-1]["text"] == "a1"
    assert row["arms"]["mixed_union"]["unsupported_texts"] == ["추가"]


def test_all_unsupported_uses_original_fallback():
    value = baseline("unsupported")
    value["arms"]["roman_repaired"]["pool"] = [value["arms"]["roman_repaired"]["pool"][0]]
    value["arms"]["roman_repaired"]["supported_order"] = []
    value["arms"]["roman_repaired"]["public"] = value["fallback_public"]
    value["supported_scores"] = []
    projected = project_row(value)
    row = score_row(projected, type("R", (), {"can_score": lambda s,t: False,
        "score_pairs": lambda s,c,t: []})())
    assert row["arms"]["baseline"]["fallback_used"] is True
    assert row["arms"]["baseline"]["public"] == value["fallback_public"]


def test_support_mismatch_is_rejected():
    projected = project_row(baseline(), mixed())
    import pytest
    with pytest.raises(ValueError, match="tokenizer support"):
        score_row(projected, Ranker())


def test_old_batch_is_replayed_exactly_and_additions_are_scored_separately():
    value = mixed()
    value["proposals"][-1]["token_support"]["supported"] = True
    projected = project_row(baseline(), value)

    class BatchSensitive:
        def __init__(self): self.calls = []
        def can_score(self, text): return True
        def score_pairs(self, contexts, texts):
            self.calls.append(list(texts))
            if texts == ["기존", "중복"]: return [2.0, 1.0]
            if texts == ["추가"]: return [3.0]
            return [99.0] * len(texts)

    ranker = BatchSensitive()
    row = score_row(projected, ranker)
    assert ranker.calls == [["기존", "중복"], ["추가"]]
    assert row["old_supported_score_count"] == 2
    assert row["additive_supported_score_count"] == 1
    assert row["arms"]["mixed_union"]["supported_order"] == ["추가", "기존", "중복"]
