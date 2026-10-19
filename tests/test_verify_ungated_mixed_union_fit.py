from copy import deepcopy

import pytest

from scripts import verify_ungated_mixed_union_fit as verifier


class Tokenizer:
    def __init__(self, lengths=None):
        self.lengths = lengths or {}
        self.calls = []

    def __call__(self, text, *, add_special_tokens, truncation):
        self.calls.append((text, add_special_tokens, truncation))
        return {"input_ids": list(range(self.lengths.get(text, len(text))))}

    def num_special_tokens_to_add(self, *, pair):
        assert pair is True
        return 3


def candidate(text):
    return {"text": text, "sentence_tokens": len(text) + 3,
            "sentence_support": "supported"}


def sources(tokenizer=None):
    tokenizer = tokenizer or Tokenizer({"긴후보": 94})
    request = {"input": "raw1", "context": ""}
    raw = {"text": "raw1", "segments": [{"kind": "raw"}]}
    fallback = [{"text": text, "segments": [{"kind": "x"}]}
                for text in ("a", "b", "c", "d")] + [raw]
    parent = {"id": "r1", "request": request, "fallback_public": fallback,
              "parents": [candidate(text) for text in ("p1", "p2", "p3", "p4")],
              "proposals": [candidate(f"s{i}") for i in range(96)]}
    joint = {"id": "r1", "request": request, "arms": {"roman_repaired": {
        "proposals": [candidate(f"j{i}") for i in range(96)]}}}
    baseline = verifier.derive_baseline(parent, joint, tokenizer)
    old_scores = [{"text": row["text"], "score": float(index)}
                  for index, row in enumerate(baseline["pool"])]
    old_score_map = {row["text"]: row["score"] for row in old_scores}
    old = {"id": "r1", "request": request, "fallback_public": fallback,
           "supported_scores": old_scores,
           "arms": {"roman_repaired": verifier._arm(
               baseline["pool"], old_score_map, fallback, raw)}}
    mixed = {"id": "r1", "request": request, "proposal_count": 3,
             "proposals": [{"text": "p1", "token_support":
                 verifier.token_support(tokenizer, "p1")},
                 {"text": "새후보", "token_support":
                  verifier.token_support(tokenizer, "새후보")},
                 {"text": "긴후보", "token_support":
                  verifier.token_support(tokenizer, "긴후보")}]}
    expanded = verifier._merge((baseline["pool"], verifier.mixed_group(
        mixed, "r1", request, tokenizer)))
    scores = {row["text"]: float(index) for index, row in enumerate(baseline["pool"])}
    scores["새후보"] = 999.0
    supported = [row["text"] for row in expanded if row["support"] == "supported"]
    unsupported = [row["text"] for row in expanded if row["support"] == "unsupported"]
    saved = {"id": "r1", "request": request, "fallback_public": fallback,
             "supported_scores": [{"text": text, "score": scores[text]}
                                  for text in supported],
             "unsupported_texts": unsupported,
             "arms": {"baseline": verifier._arm(baseline["pool"], scores,
                 fallback, raw), "mixed_union": verifier._arm(expanded, scores,
                 fallback, raw)}, "baseline_max_score_delta": 0.0,
             "mixed_source_present": True,
             "old_supported_score_count": len(old_scores),
             "additive_supported_score_count": 1}
    return parent, joint, old, mixed, saved, tokenizer


def test_recreates_additive_union_and_retains_unsupported():
    parent, joint, old, mixed, saved, tokenizer = sources()
    counts = verifier.recreate_row(parent, joint, old, mixed, saved, tokenizer)
    assert counts["arms"]["mixed_union"]["pool"] \
        == counts["arms"]["baseline"]["pool"] + 2
    assert counts["total"]["unique_unsupported_unscored"] == 1
    duplicate = next(row for row in saved["arms"]["mixed_union"]["pool"]
                     if row["text"] == "p1")
    assert duplicate["provenance"] == ["parent", "mixed_boundary"]
    assert all(not special and truncation is False
               for _, special, truncation in tokenizer.calls)


def test_rejects_baseline_score_drift_over_tolerance():
    parent, joint, old, mixed, saved, tokenizer = sources()
    saved["supported_scores"][0]["score"] += 2e-5
    saved["arms"]["baseline"] = verifier._arm(
        verifier.derive_baseline(parent, joint, tokenizer)["pool"],
        {row["text"]: row["score"] for row in saved["supported_scores"]},
        saved["fallback_public"], saved["fallback_public"][-1])
    with pytest.raises(ValueError, match="parity"):
        verifier.recreate_row(parent, joint, old, mixed, saved, tokenizer)


def test_rejects_truncated_or_changed_mixed_support():
    parent, joint, old, mixed, saved, tokenizer = sources()
    mixed["proposals"][2]["token_support"]["supported"] = True
    with pytest.raises(ValueError, match="token annotation"):
        verifier.recreate_row(parent, joint, old, mixed, saved, tokenizer)


def test_row_without_mixed_source_keeps_baseline_pool():
    parent, joint, old, _, saved, tokenizer = sources()
    saved["mixed_source_present"] = False
    saved["additive_supported_score_count"] = 0
    saved["supported_scores"] = saved["supported_scores"][:-1]
    saved["arms"]["mixed_union"] = deepcopy(saved["arms"]["baseline"])
    saved["unsupported_texts"] = []
    counts = verifier.recreate_row(parent, joint, old, None, saved, tokenizer)
    assert counts["arms"]["mixed_union"]["pool"] \
        == counts["arms"]["baseline"]["pool"]
    assert counts["total"]["mixed_source_rows"] == 0
