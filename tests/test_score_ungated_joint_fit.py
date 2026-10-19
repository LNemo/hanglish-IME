from copy import deepcopy

from scripts.score_ungated_joint_fit import aggregate, evaluate_row, score_row, source_projection


def candidate(text, support="supported"):
    return {"text": text, "sentence_support": support}


def segment(source, text, kind="neural"):
    return {"input_start": 0, "input_end": len(source), "source": source,
            "text": text, "kind": kind}


def fixture(*, all_unsupported=False, include_answer=False):
    source = "testreul handa"
    parent_texts = ["테스트를 한다", "테스트를 한디", "test를 한디", "testreul 한다"]
    fallback = [{"text": text, "segments": [segment(source, text)], "score": -float(i)}
                for i, text in enumerate(parent_texts)]
    raw = {"text": source, "segments": [segment(source, source, "raw")], "score": None}
    fallback.append(raw)
    status = "unsupported" if all_unsupported else "supported"
    parents = [candidate(text, status) for text in parent_texts]
    singles = [candidate("test를 한다" if include_answer and i == 0 else f"single-{i}", status)
               for i in range(96)]
    original_joint = [candidate(f"original-{i}", status) for i in range(96)]
    # One cross-arm duplicate proves that the shared text is scored only once.
    repaired_joint = [candidate("original-0" if i == 0 else f"repaired-{i}", status)
                      for i in range(96)]
    parent = {"id": "s3-train-1", "request": {"input": source, "context": ""},
              "fallback_public": fallback, "parents": parents, "proposals": singles}
    joint = {"id": parent["id"], "request": parent["request"],
             "arms": {"original": {"proposals": original_joint},
                      "roman_repaired": {"proposals": repaired_joint}}}
    return parent, joint


class Ranker:
    def __init__(self, unsupported=(), preferred=None):
        self.unsupported = set(unsupported)
        self.preferred = preferred
        self.fit_calls = []
        self.score_calls = []

    def can_score(self, text):
        self.fit_calls.append(text)
        return text not in self.unsupported

    def score_pairs(self, contexts, texts):
        self.score_calls.append((list(contexts), list(texts)))
        return [1000.0 if text == self.preferred else float(-index)
                for index, text in enumerate(texts)]


def test_scores_each_supported_text_once_across_both_cumulative_arms():
    projected = source_projection(*fixture())
    ranker = Ranker()
    result = score_row(projected, ranker)
    expected = []
    for arm in ("original", "roman_repaired"):
        for row in projected["arms"][arm]:
            if row["text"] not in expected:
                expected.append(row["text"])
    assert ranker.fit_calls == expected
    assert ranker.score_calls == [([""] * len(expected), expected)]
    assert [row["text"] for row in result["supported_scores"]] == expected
    assert len(result["arms"]["original"]["pool"]) == 4 + 96 + 96
    assert len(result["arms"]["roman_repaired"]["pool"]) == 4 + 96 + 96


def test_unsupported_candidates_are_retained_unscored_and_out_of_public_ranking():
    parent, joint = fixture()
    parent["proposals"][10]["sentence_support"] = "unsupported"
    unsupported = parent["proposals"][10]["text"]
    projected = source_projection(parent, joint)
    ranker = Ranker(unsupported={unsupported})
    result = score_row(projected, ranker)
    assert unsupported in result["unsupported_texts"]
    assert unsupported in result["arms"]["original"]["unsupported_texts"]
    assert unsupported not in ranker.score_calls[0][1]
    assert unsupported not in [row["text"] for row in result["arms"]["original"]["public"]]


def test_supported_public_is_best_four_plus_exact_raw_reserve():
    parent, joint = fixture(include_answer=True)
    raw = deepcopy(parent["fallback_public"][-1])
    result = score_row(source_projection(parent, joint), Ranker(preferred="test를 한다"))
    for arm in ("original", "roman_repaired"):
        public = result["arms"][arm]["public"]
        assert len(public) == 5
        assert public[0]["text"] == "test를 한다"
        assert public[-1] == raw
        assert result["fallback_public"] == parent["fallback_public"]


def test_no_supported_score_preserves_exact_five_slot_fallback():
    parent, joint = fixture(all_unsupported=True)
    all_texts = {row["text"] for row in parent["parents"] + parent["proposals"]}
    all_texts.update(row["text"] for arm in joint["arms"].values() for row in arm["proposals"])
    ranker = Ranker(unsupported=all_texts)
    result = score_row(source_projection(parent, joint), ranker)
    assert ranker.score_calls == []
    assert len(result["unsupported_texts"]) == len(all_texts)
    for arm in ("original", "roman_repaired"):
        assert result["arms"][arm]["fallback_used"] is True
        assert result["arms"][arm]["public"] == parent["fallback_public"]


def test_late_metrics_separate_selection_pool_coverage_and_ascii_preservation():
    parent, joint = fixture(include_answer=True)
    scored = score_row(source_projection(parent, joint), Ranker(preferred="test를 한다"))
    label = {"id": parent["id"], "input": parent["request"]["input"],
             "expected": ["test를 한다"]}
    outcome = evaluate_row(scored, label)
    summary = aggregate([outcome])
    fallback = summary["branches"]["captured_fallback"]
    repaired = summary["branches"]["roman_repaired"]
    assert (fallback["first"], fallback["top5"], fallback["ascii_exact"]) == (0, 0, 0)
    assert (repaired["first"], repaired["top5"], repaired["all_pool_covered"],
            repaired["scorable_pool_covered"], repaired["ascii_exact"]) == (1, 1, 1, 1, 1)
    transition = summary["transitions"]["fallback_to_roman_repaired"]
    assert transition["corrections"] == transition["top5_gains"] == 1
    assert transition["ascii_gains"] == 1 and transition["regressions"] == 0

