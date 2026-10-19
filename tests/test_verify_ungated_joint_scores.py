from copy import deepcopy
from collections import UserDict

import pytest

from scripts import verify_ungated_joint_scores as verifier


class Tokenizer:
    def __init__(self, lengths):
        self.lengths = lengths
        self.calls = []

    def __call__(self, text, *, add_special_tokens, truncation):
        self.calls.append((text, add_special_tokens, truncation))
        return {"input_ids": list(range(self.lengths.get(text, 2)))}

    def num_special_tokens_to_add(self, *, pair):
        assert pair is True
        return 3


class Ranker:
    max_length = 96
    batch_size = 8

    def __init__(self, lengths, preferred=None):
        self.tokenizer = Tokenizer(lengths)
        self.preferred = preferred
        self.score_calls = []

    def score_pairs(self, contexts, texts):
        self.score_calls.append((list(contexts), list(texts)))
        return [1000.0 if text == self.preferred else -float(index)
                for index, text in enumerate(texts)]


class MappingTokenizer(Tokenizer):
    def __call__(self, text, *, add_special_tokens, truncation):
        value = super().__call__(
            text, add_special_tokens=add_special_tokens, truncation=truncation)
        return UserDict(value)


def source_candidate(text, ranker):
    tokens = ranker.tokenizer.lengths.get(text, 2) + 3
    return {"text": text, "sentence_tokens": tokens,
            "sentence_support": "supported" if tokens <= 96 else "unsupported"}


def segment(source, text, kind="fixture"):
    return {"input_start": 0, "input_end": len(source), "source": source,
            "text": text, "kind": kind}


def fixture(*, all_unsupported=False):
    source = "testreul handa"
    lengths = {}
    if all_unsupported:
        lengths = {f"parent-{index}": 94 for index in range(4)}
        lengths.update({f"single-{index}": 94 for index in range(96)})
        lengths.update({f"original-{index}": 94 for index in range(96)})
        lengths.update({f"repaired-{index}": 94 for index in range(96)})
    else:
        lengths["single-10"] = 94
    ranker = Ranker(lengths, preferred="repaired-7")
    parents = [source_candidate(f"parent-{index}", ranker) for index in range(4)]
    singles = [source_candidate(f"single-{index}", ranker) for index in range(96)]
    original = [source_candidate(f"original-{index}", ranker) for index in range(96)]
    repaired = [source_candidate("original-0" if index == 0 else f"repaired-{index}", ranker)
                for index in range(96)]
    fallback = [{"text": f"fallback-{index}", "score": -float(index),
                 "segments": [segment(source, f"fallback-{index}")]}
                for index in range(4)]
    fallback.append({"text": source, "score": None,
                     "segments": [segment(source, source, "raw")]})
    parent = {"id": "s3-train-fixture", "request": {"input": source, "context": ""},
              "fallback_public": fallback, "parents": parents, "proposals": singles}
    joint = {"id": parent["id"], "request": parent["request"],
             "arms": {"original": {"proposals": original},
                      "roman_repaired": {"proposals": repaired}}}
    return parent, joint, ranker


def test_recreates_unique_untruncated_logits_and_keeps_unsupported():
    parent, joint, ranker = fixture()
    row = verifier.recreate_scored_row(parent, joint, ranker)

    supported = ranker.score_calls[0][1]
    assert ranker.score_calls[0][0] == [""] * len(supported)
    assert len(supported) == len(set(supported)) == 4 + 96 + 96 + 95 - 1
    assert "single-10" not in supported
    assert row["unsupported_texts"] == ["single-10"]
    assert "single-10" in row["arms"]["original"]["unsupported_texts"]
    assert row["arms"]["roman_repaired"]["public"][0]["text"] == "repaired-7"
    assert row["arms"]["roman_repaired"]["public"][-1] == parent["fallback_public"][-1]
    assert ranker.tokenizer.calls
    assert all(not add and trunc is False
               for _, add, trunc in ranker.tokenizer.calls)


def test_token_count_accepts_batch_encoding_style_mapping_without_weakening_shape():
    ranker = Ranker({"테스트": 93})
    ranker.tokenizer = MappingTokenizer({"테스트": 93})
    assert verifier.token_count(ranker, "테스트") == 96
    assert ranker.tokenizer.calls == [("테스트", False, False)]


def test_score_comparison_allows_one_e_minus_six_only():
    parent, joint, ranker = fixture()
    fresh = verifier.recreate_scored_row(parent, joint, ranker)
    saved = deepcopy(fresh)
    saved["supported_scores"][0]["score"] += 0.9e-6
    saved["arms"]["original"]["public"][0]["sentence_score"] += 0.9e-6
    verifier.compare_scored(saved, fresh)
    saved["supported_scores"][0]["score"] += 0.2e-6
    with pytest.raises(ValueError, match="logits"):
        verifier.compare_scored(saved, fresh)


def test_all_unsupported_uses_exact_fallback_and_no_inference():
    parent, joint, ranker = fixture(all_unsupported=True)
    row = verifier.recreate_scored_row(parent, joint, ranker)
    assert ranker.score_calls == []
    assert row["supported_scores"] == []
    for arm in verifier.ARMS:
        assert row["arms"][arm]["fallback_used"] is True
        assert row["arms"][arm]["public"] == parent["fallback_public"]
        assert len(row["arms"][arm]["unsupported_texts"]) == 196


def test_independent_fit_metrics_and_transitions():
    parent, joint, ranker = fixture()
    ranker.preferred = "single-0"
    row = verifier.recreate_scored_row(parent, joint, ranker)
    label = {"id": parent["id"], "input": parent["request"]["input"],
             "expected": ["single-0"]}
    outcome = verifier.metric_row(row, label)
    summary = verifier.aggregate([outcome])
    assert summary["branches"]["captured_fallback"]["first"] == 0
    assert summary["branches"]["original"]["first"] == 1
    assert summary["branches"]["roman_repaired"]["top5"] == 1
    assert summary["transitions"]["fallback_to_original"]["corrections"] == 1


def test_fit_labels_cannot_open_before_score_verification(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text("{}\n")
    with pytest.raises(ValueError, match="cannot open"):
        verifier.load_labels_after_scores(
            path, verifier.digest(path), scores_verified=False)
