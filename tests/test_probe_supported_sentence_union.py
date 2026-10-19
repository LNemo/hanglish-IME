from copy import deepcopy
import json

import pytest

import scripts.probe_supported_sentence_union as module


def candidate(text, score, kind="neural", source="roman"):
    return {"text": text, "score": score, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": source if kind in {"raw", "english"} else text, "kind": kind,
    }]}


def candidates(prefix):
    values = [candidate(f"{prefix}{i}", float(4 - i)) for i in range(4)]
    values.append(candidate("roman", None, "raw"))
    return values


class Ranker:
    def __init__(self, scores=None, scoreable=True):
        self.scores = scores or []
        self.scoreable = scoreable
        self.calls = []

    def can_score(self, text):
        return self.scoreable

    def score_pairs(self, contexts, values):
        self.calls.append((contexts, values))
        return self.scores[:len(values)]


def row():
    artifacts = {"checkpoints/sentence_ranker/model.safetensors": "a" * 64}
    return {
        "id": "r", "group_id": "g", "input": "roman", "context": "ctx",
        "baseline_result": "a0", "baseline_candidates": candidates("a"),
        "new_result": "b0", "new_candidates": candidates("b"),
        "new_response_flags": {"sentence_ranking_applied": True,
                               "model_artifacts": artifacts},
    }, artifacts


def test_union_preserves_old_prefix_objects_and_deduplicates():
    old = candidates("a")
    new = [deepcopy(old[2]), *candidates("b")[:3], candidates("b")[-1]]
    union = module.candidate_union(old, new)
    assert [item["text"] for item in union[:4]] == [item["text"] for item in old[:4]]
    assert len(union) == 7 and len({item["text"] for item in union}) == 7
    assert union[0] == old[0] and union[0] is not old[0]


def test_fresh_scores_choose_by_score_then_text_and_restore_raw_slot(monkeypatch):
    value, artifacts = row()
    monkeypatch.setattr(module.time, "perf_counter", iter((1.0, 1.01)).__next__)
    ranker = Ranker([0.0, 1.0, 1.0, -1.0, 4.0, 3.0, 2.0, 1.5])
    result = module.score_case(value, ranker, artifacts)
    assert result["applicable"] is True
    assert [item["text"] for item in result["union_candidates"]] == [
        "b0", "b1", "b2", "b3", "roman"]
    assert result["union_result"] == "b0" and result["score_ms"] == pytest.approx(10.0)
    assert [item["text"] for item in result["union_pool"][:4]] == ["a0", "a1", "a2", "a3"]


def test_unscoreable_or_flag_disabled_preserves_exact_baseline():
    value, artifacts = row()
    ranker = Ranker(scoreable=False)
    result = module.score_case(value, ranker, artifacts)
    assert result["applicable"] is False and result["union_candidates"] == value["baseline_candidates"]
    assert not ranker.calls


def test_malformed_scores_fail_closed():
    value, artifacts = row()
    with pytest.raises(ValueError, match="malformed"):
        module.score_case(value, Ranker([float("nan")] * 8), artifacts)


def test_capture_contains_no_expected_and_flushes_source_only(tmp_path):
    value, artifacts = row()
    path = tmp_path / "capture.jsonl"
    rows = module.capture_scores(value if isinstance(value, list) else [value],
                                 Ranker(list(range(8))), artifacts, path)
    saved = json.loads(path.read_text().splitlines()[0])
    assert len(rows) == 1 and "expected" not in saved
    assert saved["union_pool"] and saved["baseline_candidates"] == value["baseline_candidates"]


def test_protected_english_segment_change_is_rejected():
    old = candidates("a")
    bad = candidates("b")
    bad[0] = candidate("changed", 1.0, "english", "test")
    bad[0]["segments"][0]["text"] = "changed"
    with pytest.raises(ValueError, match="protected"):
        module.candidate_union(old, bad)


def test_prelabel_pins_do_not_inherit_fit_label_files(monkeypatch):
    monkeypatch.setattr(module, "file_sha256", lambda path: "f" * 64)
    artifacts = {"checkpoints/sentence_ranker/model.safetensors": "f" * 64}
    pins = module.source_pins({"source_hashes": {
        "data/natural_context/final-choice-v1/natural_fit/cases.jsonl": "a" * 64,
        "data/natural_context/final-choice-v1/challenge_fit/cases.jsonl": "b" * 64,
    }}, artifacts)
    assert not any(name.endswith("cases.jsonl") for name in pins)
    assert module.key(module.RUNTIME / "report.json") not in pins


def test_visible_candidates_requires_original_raw_slot():
    with pytest.raises(ValueError, match="raw"):
        module.visible_candidates(candidates("a")[:4], candidate("roman", 0.0, "raw"))
