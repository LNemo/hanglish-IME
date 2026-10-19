from copy import deepcopy
import json

import pytest

from scripts.prepare_listwise_semantic_training import candidate_set, prepare_rows


def example():
    current = {"text": "현재", "score": 2, "segments": []}
    alternative = {"text": "대안", "score": 1, "segments": []}
    row = {"index": 0, "id": "r", "source_role": "natural_fit", "group_id": "g",
           "component": "c", "fold": 2, "input": "roman", "context": "context",
           "current_result": "현재", "pair_indices": [0],
           "finite_candidates": [{"source_index": 0, "object": current},
                                 {"source_index": 3, "object": alternative}]}
    pair = {"row_index": 0, "row_id": "r", "alternative_source_index": 3,
            "current": current, "alternative": alternative, "eligible_le_512": False}
    return row, [pair]


class Tokenizer:
    def __init__(self, lengths):
        self.lengths = lengths

    def __call__(self, views, **kwargs):
        assert kwargs == {"truncation": False, "padding": False, "add_special_tokens": True}
        assert len(views) == len(self.lengths)
        return {"input_ids": [[1] * length for length in self.lengths]}


def test_complete_objects_and_source_indices_survive_without_pair_token_filter():
    row, pairs = example()
    original = deepcopy((row, pairs))
    result = prepare_rows([row], pairs, Tokenizer([10, 12]))[0]
    assert result["source_indices"] == [0, 3]
    assert result["candidates"] == [pairs[0]["current"], pairs[0]["alternative"]]
    assert result["eligible"] is True
    assert (row, pairs) == original
    assert "expected" not in result and "positives" not in result


def test_one_long_candidate_bypasses_whole_row_without_losing_candidates():
    row, pairs = example()
    result = prepare_rows([row], pairs, Tokenizer([10, 513]))[0]
    assert not result["eligible"] and len(result["candidates"]) == 2
    assert result["fold"] == 2 and result["component"] == "c"
    assert json.loads(result["neural_views"][1])["candidate_text"] == "대안"


def test_candidate_source_mismatch_and_duplicate_fail_closed():
    row, pairs = example()
    pairs[0]["row_id"] = "other"
    with pytest.raises(ValueError, match="source"):
        candidate_set(row, pairs)
    row, pairs = example()
    row["pair_indices"] = [0, 0]
    with pytest.raises(ValueError, match="source"):
        candidate_set(row, pairs)


def test_current_only_row_is_preserved_for_none_supervision():
    row, pairs = example()
    row["pair_indices"] = []
    result = prepare_rows([row], pairs, Tokenizer([4]))[0]
    assert len(result["candidates"]) == 1 and result["eligible"]
