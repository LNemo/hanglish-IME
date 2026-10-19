from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading

import pytest

from hanglish.core.ensemble_converter import EnsembleConverter


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate(text, score, *, source="roman", kind="neural"):
    return {"text": text, "score": score, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": source if kind == "raw" else text, "kind": kind,
    }]}


def response(values, *, sentence=True, version="branch"):
    return {
        "input": "roman", "context": "ctx", "engine": "neural", "mode": "batch",
        "raw": False, "version": version, "candidates": deepcopy(values),
        "result": values[0]["text"], "ranker": "hybrid",
        "model_artifacts": {"checkpoints/sentence_ranker/model.safetensors": "a" * 64},
        "context_used": True, "external_context_used": True,
        "ranking_applied": True, "sentence_ranking_applied": sentence,
        "warnings": [], "engine_ms": 1.0,
    }


class Ranker:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def can_score(self, text):
        return text != "oversized"

    def score_pairs(self, contexts, candidates):
        self.calls.append((contexts, candidates))
        return self.scores[:len(candidates)]


class Cache:
    def cache_clear(self):
        pass


class Branch:
    def __init__(self, value, ranker):
        self.value = value
        self.ranker = object()
        self.sentence_ranker = ranker
        self.generator = object()
        self.numeric_units = ()
        self.lock = threading.RLock()
        self.word_options = Cache()
        self.model_artifacts = value["model_artifacts"]
        self.version = value["version"]
        self.loaded = 0
        self.calls = []

    def load_models(self):
        self.loaded += 1

    def convert(self, text, context="", engine="neural", mode="batch", raw=False):
        self.calls.append((text, context, engine, mode, raw))
        value = deepcopy(self.value)
        value.update({"input": text, "context": context, "engine": engine,
                      "mode": mode, "raw": raw})
        if raw:
            raw_candidate = candidate(text, None, source=text, kind="raw")
            value.update({"candidates": [raw_candidate], "result": text,
                          "sentence_ranking_applied": False, "model_artifacts": {}})
        return value


def files(tmp_path):
    checkpoint = tmp_path / "phonetic.pt"
    metadata = checkpoint.with_suffix(".json")
    checkpoint.write_bytes(b"weights")
    metadata.write_text(json.dumps({"format_version": 1}))
    return checkpoint, digest(checkpoint), digest(metadata)


def converter(tmp_path, old, new, scores):
    checkpoint, checkpoint_sha, metadata_sha = files(tmp_path)
    ranker = Ranker(scores)
    baseline = Branch(response(old), ranker)
    supported = Branch(response(new), ranker)
    supported.ranker = baseline.ranker
    return EnsembleConverter(
        checkpoint, checkpoint_sha256=checkpoint_sha, metadata_sha256=metadata_sha,
        experimental=True, baseline=baseline, supported=supported,
    ), ranker


def raw():
    return candidate("roman", None, source="roman", kind="raw")


def test_requires_explicit_experimental_mode_and_verified_hashes(tmp_path):
    checkpoint, checkpoint_sha, metadata_sha = files(tmp_path)
    with pytest.raises(ValueError, match="experimental"):
        EnsembleConverter(checkpoint, checkpoint_sha256=checkpoint_sha,
                          metadata_sha256=metadata_sha)
    with pytest.raises(ValueError, match="SHA"):
        EnsembleConverter(checkpoint, checkpoint_sha256="bad",
                          metadata_sha256=metadata_sha, experimental=True)


def test_union_rescores_old_prefix_then_new_and_keeps_raw_fifth(tmp_path):
    old = [candidate("old-a", 9), candidate("shared", 8), raw()]
    new = [candidate("new-a", 9), candidate("shared", 7), raw()]
    value, ranker = converter(tmp_path, old, new, [1.0, 2.0, 4.0])
    result = value.convert("roman", "ctx")
    assert ranker.calls[-1][1] == ["old-a", "shared", "new-a"]
    assert [item["text"] for item in result["candidates"]] == [
        "new-a", "shared", "old-a", "roman"]
    assert result["candidates"][-1]["score"] is None
    assert result["result"] == "new-a" and result["version"].startswith("experimental-ensemble-1-")
    assert value.last_baseline_response["result"] == "old-a"
    assert value.last_supported_response["result"] == "new-a"


def test_ties_sort_by_text_and_candidate_segments_are_preserved(tmp_path):
    old_candidate = candidate("z", 1)
    new_candidate = candidate("a", 1)
    value, _ = converter(tmp_path, [old_candidate, raw()], [new_candidate, raw()], [2., 2.])
    result = value.convert("roman", "ctx")
    assert [item["text"] for item in result["candidates"][:2]] == ["a", "z"]
    assert result["candidates"][0]["segments"] == new_candidate["segments"]


def test_sentence_flag_or_budget_failure_returns_exact_baseline(tmp_path):
    old = [candidate("old", 1), raw()]
    new = [candidate("oversized", 1), raw()]
    value, ranker = converter(tmp_path, old, new, [1., 2.])
    expected = deepcopy(value.baseline.value)
    result = value.convert("roman", "ctx")
    assert result["result"] == expected["result"]
    assert result["candidates"] == expected["candidates"]
    assert result["sentence_ranking_applied"] == expected["sentence_ranking_applied"]
    assert result["fallback_reason"] == "candidate_exceeds_sentence_budget"
    assert result["union_applied"] is False and result["ensemble_total_ms"] >= 0
    assert not ranker.calls
    value.supported.value["sentence_ranking_applied"] = False
    result = value.convert("roman", "ctx")
    assert result["candidates"] == expected["candidates"]
    assert result["fallback_reason"] == "branch_sentence_ranking_unavailable"


def test_raw_mode_bypasses_supported_branch_and_resets_snapshots(tmp_path):
    value, _ = converter(tmp_path, [candidate("old", 1), raw()],
                         [candidate("new", 1), raw()], [1., 2.])
    value.convert("roman", "ctx")
    before = len(value.supported.calls)
    result = value.convert("roman", "ctx", raw=True)
    assert result["raw"] is True and result["result"] == "roman"
    assert len(value.supported.calls) == before
    assert value.last_supported_response is None


def test_finite_original_raw_can_remain_top1_after_fresh_scoring(tmp_path):
    internal_raw = candidate("roman", -10000., source="roman", kind="raw")
    value, ranker = converter(
        tmp_path, [candidate("old", 1), internal_raw],
        [candidate("new", 1), deepcopy(internal_raw)], [1., 5., 2.])
    result = value.convert("roman", "ctx")
    assert result["result"] == "roman"
    assert result["candidates"][0] == candidate("roman", 5., source="roman", kind="raw")
    assert ranker.calls[-1][1] == ["old", "roman", "new"]


def test_finite_original_at_rank_five_is_not_duplicated_or_null(tmp_path):
    finite_original = candidate("roman", -10000., source="roman", kind="raw")
    old = [candidate(f"old-{i}", 4 - i) for i in range(4)] + [finite_original]
    new = [candidate("new", 1), raw()]
    value, _ = converter(tmp_path, old, new, [6., 5., 4., 3., 2., 1.])
    result = value.convert("roman", "ctx")
    assert [item["text"] for item in result["candidates"]] == [
        "old-0", "old-1", "old-2", "old-3", "roman"]
    assert result["candidates"][-1]["score"] == 2.
    assert sum(item["text"] == "roman" for item in result["candidates"]) == 1


def test_malformed_sentence_scores_fail_closed(tmp_path):
    value, _ = converter(tmp_path, [candidate("old", 1), raw()],
                         [candidate("new", 1), raw()], [float("nan"), 1.])
    with pytest.raises(ValueError, match="malformed"):
        value.convert("roman", "ctx")


def test_none_score_is_reserved_for_original_raw_candidate(tmp_path):
    malformed = candidate("not-raw", None)
    value, _ = converter(tmp_path, [malformed, raw()],
                         [candidate("new", 1), raw()], [1.])
    with pytest.raises(ValueError, match="reserved"):
        value.convert("roman", "ctx")


def test_load_rejects_changed_checkpoint(tmp_path):
    value, _ = converter(tmp_path, [candidate("old", 1), raw()],
                         [candidate("new", 1), raw()], [1., 2.])
    value.checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differ"):
        value.load_models()


def test_fake_branches_must_share_rankers_but_keep_locks_and_caches(tmp_path):
    value, _ = converter(tmp_path, [candidate("old", 1), raw()],
                         [candidate("new", 1), raw()], [1., 2.])
    value.supported.ranker = object()
    with pytest.raises(ValueError, match="share one word ranker"):
        value.load_models()


def test_segment_offsets_and_protected_text_are_validated(tmp_path):
    bad = candidate("changed", 1, source="test", kind="english")
    bad["segments"][0]["text"] = "changed"
    value, _ = converter(tmp_path, [candidate("old", 1), raw()], [bad, raw()], [1., 2.])
    with pytest.raises(ValueError, match="provenance"):
        value.convert("roman", "ctx")
