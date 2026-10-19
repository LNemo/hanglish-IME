import hashlib
import json
import random

import pytest

from scripts.consensus_context_policy import choices, decide, prompt_choices


def candidate(source, text, kind="neural"):
    return {"text": text, "score": 1.0, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": text, "kind": kind,
    }]}


def test_consensus_moves_original_object_and_preserves_relative_order():
    source = "imnida"
    parent = candidate(source, "임니다")
    middle = candidate(source, "입니다")
    last = candidate(source, "입니닫")
    original = [parent, middle, last]
    result = decide(source, original, 1, 1)
    assert result == {
        "chosen_index": 1, "result": "입니다",
        "candidates": [middle, parent, last], "reason": "consensus_proposal",
    }
    assert result["candidates"][0] is middle
    assert result["candidates"][1] is parent
    assert result["candidates"][2] is last
    assert original == [parent, middle, last]


def test_disagreement_and_parent_consensus_preserve_exact_parent_list():
    source = "imnida"
    candidates = [candidate(source, "임니다"), candidate(source, "입니다")]
    disagreement = decide(source, candidates, 0, 1)
    parent = decide(source, candidates, 0, 0)
    assert disagreement["reason"] == "selector_disagreement"
    assert parent["reason"] == "parent_consensus"
    assert disagreement["candidates"] is candidates
    assert parent["candidates"] is candidates
    assert disagreement["result"] == parent["result"] == "임니다"


@pytest.mark.parametrize("malformation", ["protected", "unaligned"])
def test_forbidden_or_unaligned_consensus_is_rejected(malformation):
    source = "testreul"
    parent = candidate(source, "test를", kind="mixed")
    proposal = candidate(source, "테스트를")
    if malformation == "unaligned":
        proposal["segments"][0]["input_end"] -= 1
    candidates = [parent, proposal]
    result = decide(source, candidates, 1, 1)
    assert result["reason"] == "protected_or_unaligned"
    assert result["chosen_index"] == 0
    assert result["candidates"] is candidates


def test_raw_bypass_does_not_examine_predictions_or_candidates():
    candidates = [{"not": "validated in raw mode"}]
    result = decide("source", candidates, object(), object(), raw=True)
    assert result == {"chosen_index": 0, "result": "source",
                      "candidates": candidates, "reason": "raw_mode"}
    assert result["candidates"] is candidates


def test_choice_filter_preserves_source_indices_and_historical_prompt_shuffle():
    source, context = "imnida", "앞 문맥"
    candidates = [candidate(source, "임니다"), candidate(source, source, "raw"),
                  candidate(source, "입니다")]
    assert choices(source, candidates) == [(0, "임니다"), (2, "입니다")]
    expected = [(0, "임니다"), (2, "입니다")]
    seed = hashlib.sha256(
        json.dumps([source, context], ensure_ascii=False).encode()
    ).hexdigest()
    random.Random(seed).shuffle(expected)
    assert prompt_choices(source, context, candidates) == expected
    assert prompt_choices(source, context, candidates) == expected


def test_parent_source_is_retained_and_malformed_choices_or_indices_fail():
    source = "raw"
    candidates = [candidate(source, source, "raw"), candidate(source, "라오")]
    assert choices(source, candidates) == [(0, source), (1, "라오")]
    for bad in (True, -1, 2, "1"):
        with pytest.raises(ValueError, match="prediction"):
            decide(source, candidates, bad, 0)
    with pytest.raises(ValueError, match="unique"):
        choices(source, [candidate(source, source), candidate(source, source)])
    with pytest.raises(ValueError, match="nonempty"):
        choices(source, [{"text": ""}])
    with pytest.raises(ValueError, match="one to five"):
        choices(source, [candidate(source, str(index)) for index in range(6)])
