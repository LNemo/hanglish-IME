import math

import pytest
import torch

from scripts.verify_contextual_character_fit import (
    infer_chunks, metric, multi_positive_loss, public_texts, summarize,
)


def candidates(texts):
    return [{"text": text, "segments": [{
        "input_start": 0, "input_end": 1, "source": "x", "text": text,
    }]} for text in texts]


def test_public_render_reserves_raw_source_and_uses_text_tiebreak():
    pool = candidates(["다", "나", "가", "라", "마", "바"])
    assert public_texts("raw", pool, [2, 3, 3, 1, 0, -1]) == [
        "가", "나", "다", "라", "raw"]
    assert public_texts("다", pool, [2, 3, 3, 1, 0, -1]) == [
        "가", "나", "다", "라", "마"]


def test_multi_positive_loss_matches_torch_and_rejects_bad_indices():
    scores = [1.2, -.4, .7, 2.1]
    expected = torch.logsumexp(torch.tensor(scores, dtype=torch.float64), 0)
    expected -= torch.logsumexp(torch.tensor(scores, dtype=torch.float64)[[0, 2]], 0)
    assert multi_positive_loss(scores, [0, 2]) == pytest.approx(float(expected), abs=1e-12)
    with pytest.raises(ValueError, match="multi-positive"):
        multi_positive_loss(scores, [4])
    with pytest.raises(ValueError, match="multi-positive"):
        multi_positive_loss([math.inf], [0])


class FakeModel:
    def __init__(self):
        self.chunk_sizes = []

    def prepare(self, source, values, base_scores):
        self.chunk_sizes.append(len(values))
        return torch.tensor(base_scores, dtype=torch.float64)

    def __call__(self, prepared):
        return {"scores": prepared + 1.0}


def test_checkpoint_replay_is_fixed_to_chunks_of_32():
    model = FakeModel()
    row = {"input": "x", "candidates": candidates([str(i) for i in range(65)]),
           "base_scores": [float(i) for i in range(65)]}
    assert infer_chunks(model, row) == [float(i + 1) for i in range(65)]
    assert model.chunk_sizes == [32, 32, 1]
    with pytest.raises(ValueError, match="chunks of 32"):
        infer_chunks(model, row, 16)


def test_metric_handles_multiple_gold_and_no_pool_fallback():
    row = {"id": "r", "input": "raw", "expected": ["정답", "대안"],
           "candidates": candidates(["오답", "정답"]), "base_scores": [2., 1.],
           "positive_indices": [1], "fallback": None, "no_pool": False,
           "unsupported": 3}
    ranked = metric(row, row["base_scores"])
    assert ranked == {"id": "r", "first": False, "top5": True,
                      "pool_covered": True, "public": ["오답", "정답", "raw"],
                      "no_pool": False, "unsupported": 3}
    fallback = {**row, "candidates": [], "base_scores": [], "positive_indices": [],
                "fallback": [{"text": "대안"}], "no_pool": True}
    unranked = metric(fallback, [])
    assert unranked["first"] and unranked["top5"] and not unranked["pool_covered"]
    assert summarize([ranked, unranked]) == {
        "rows": 2, "first": 1, "top5": 2, "pool_covered": 1,
        "no_pool": 1, "unsupported": 6,
    }
