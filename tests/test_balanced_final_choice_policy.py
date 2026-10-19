from copy import deepcopy
import math

import pytest

from scripts.balanced_final_choice_policy import select_candidates


def candidates():
    def item(text, score, kind):
        return {"text": text, "score": score, "segments": [{"input_start": 0,
                "input_end": 6, "source": "imnida", "text": text, "kind": kind}]}
    return [item("임니다", 2., "syllable"), item("입니다", 1., "morphology"),
            item("imnida", None, "raw")]


class Ranker:
    def __init__(self, scores=(0., 1., 100.), too_long=None):
        self.scores, self.too_long, self.calls = scores, too_long, []

    def can_score(self, text):
        return text != self.too_long

    def score_pairs(self, contexts, texts):
        self.calls.append((contexts, texts))
        return list(self.scores)


def test_direct_choice_preserves_objects_scores_and_original_fallback():
    rows = candidates()
    before = deepcopy(rows)
    ranker = Ranker()
    result = select_candidates(ranker, "imnida", "앞 문맥", rows)
    assert result["chosen_index"] == 1
    assert result["result"] == "입니다"
    assert all(a is b for a, b in zip(result["candidates"], [rows[1], rows[0], rows[2]]))
    assert rows == before
    assert ranker.calls == [(["앞 문맥"] * 3, [r["text"] for r in rows])]


def test_raw_and_oversized_paths_make_no_scoring_call():
    rows = candidates()
    ranker = Ranker(too_long="imnida")
    result = select_candidates(ranker, "imnida", "", rows)
    assert result["decision"]["reason"] == "candidate_budget"
    assert result["candidates"] is rows
    result = select_candidates(None, "imnida", "", [rows[-1]], raw=True)
    assert result["decision"]["reason"] == "raw_mode"
    assert ranker.calls == []


def test_insufficient_finite_and_tie_keep_parent():
    rows = candidates()
    assert select_candidates(None, "imnida", "", [rows[0], rows[-1]])["chosen_index"] == 0
    assert select_candidates(Ranker((1., 1., 100.)), "imnida", "", rows)["chosen_index"] == 0


def test_forbidden_winner_does_not_fall_through_to_another_candidate():
    rows = candidates()
    bad = deepcopy(rows[1])
    bad["text"] = bad["segments"][0]["text"] = "테스트"
    rows.insert(1, bad)
    result = select_candidates(Ranker((0., 3., 2., 100.)), "imnida", "", rows)
    assert result["chosen_index"] == 0
    assert result["decision"]["reason"] == "protected_or_unaligned"


@pytest.mark.parametrize("scores", [(0., math.nan, 2.), (0., 1.), (0., math.inf, 2.)])
def test_invalid_residual_vector_fails_closed(scores):
    with pytest.raises(ValueError, match="residual"):
        select_candidates(Ranker(scores), "imnida", "", candidates())
