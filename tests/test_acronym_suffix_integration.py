import copy

import pytest

import scripts.verify_acronym_suffix_integration as module


def candidate(text, score=1.):
    return {"text": text, "score": score,
            "segments": [{"source": "x", "text": text, "kind": "mixed"}]}


def test_candidate_parity_preserves_order_objects_and_bounded_score_tolerance():
    expected = [candidate("A", 1.), candidate("B", 2.)]

    assert module.candidate_objects_equal(copy.deepcopy(expected), expected)
    close = copy.deepcopy(expected); close[0]["score"] += module.PROFILE["score_tolerance"] / 2
    assert module.candidate_objects_equal(close, expected)
    assert not module.candidate_objects_equal(list(reversed(expected)), expected)
    changed = copy.deepcopy(expected); changed[0]["segments"][0]["kind"] = "raw"
    assert not module.candidate_objects_equal(changed, expected)
    far = copy.deepcopy(expected); far[0]["score"] += module.PROFILE["score_tolerance"] * 2
    assert not module.candidate_objects_equal(far, expected)


def test_actual_shaped_raw_candidate_allows_matching_none_score_only():
    raw = {"text": "AIneun", "score": None,
           "segments": [{"input_start": 0, "input_end": 6,
                         "source": "AIneun", "text": "AIneun", "kind": "raw"}]}

    assert module.candidate_objects_equal([copy.deepcopy(raw)], [raw])
    scored = copy.deepcopy(raw); scored["score"] = 0.
    assert not module.candidate_objects_equal([scored], [raw])


def test_sentence_scores_require_same_shape_nullability_and_tolerance():
    assert module.score_lists_equal(None, None)
    assert not module.score_lists_equal([], None)
    assert module.score_lists_equal([1., 2.], [1.000001, 2.])
    assert not module.score_lists_equal([1.], [1., 2.])


class Recorder:
    candidates = None
    scores = None


class Converter:
    version = "integrated"

    def __init__(self, reference, recorder):
        self.reference, self.recorder = reference, recorder

    def convert(self, text, context):
        self.recorder.candidates = self.reference["sentence_pool"]
        self.recorder.scores = self.reference["sentence_scores"]
        return {"candidates": self.reference["candidates"],
                "sentence_ranking_applied": self.recorder.candidates is not None}


def test_case_verifier_matches_full_candidates_pool_and_scores_without_labels():
    reference = {"id": "id", "suite": "natural_dev", "input": "AIneun",
                 "context": "ctx", "candidates": [candidate("AI는")],
                 "sentence_pool": ["AI는"], "sentence_scores": [2.],
                 "expected": ["AI는"], "before_top1": False, "before_top5": False,
                 "before_sentence_pool_covered": False}
    recorder = Recorder()

    result = module.verify_case(Converter(reference, recorder), recorder, reference)

    assert result["candidate_objects_equal"] is True
    assert result["sentence_pool_equal"] is True


def test_case_verifier_rejects_pool_or_candidate_difference():
    reference = {"id": "id", "suite": "final_v22", "input": "AIneun",
                 "context": "", "candidates": [candidate("AI는")],
                 "sentence_pool": None, "sentence_scores": None,
                 "expected": ["AI는"], "before_top1": False, "before_top5": False,
                 "before_sentence_pool_covered": None}
    changed = copy.deepcopy(reference); changed["candidates"] = [candidate("AI은")]
    recorder = Recorder()
    with pytest.raises(ValueError, match="differs"):
        module.verify_case(Converter(changed, recorder), recorder, reference)


def test_weight_identity_rejects_missing_or_changed_checkpoint(monkeypatch, tmp_path):
    accepted = {"checkpoints/model.safetensors": "a" * 64}
    monkeypatch.setattr(module, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="missing"):
        module.current_weight_hashes(accepted)
    path = tmp_path / "checkpoints/model.safetensors"
    path.parent.mkdir(); path.write_bytes(b"changed")
    current = module.current_weight_hashes(accepted)
    assert current != accepted


def test_acceptance_snapshot_pins_both_explicit_reports_and_parent_source(monkeypatch):
    accepted = {"development_sha256": "dev", "final_sha256": "final",
                "train_sha256": module.TRAIN_PROBE_SHA256,
                "parent_source_sha256": module.PARENT_SOURCE_SHA256,
                "model_weights_changed": False, "parent_version": "parent",
                "final_metrics": {"rows": 1000}}
    class Snapshot:
        def read_text(self): return __import__("json").dumps(accepted)
    monkeypatch.setattr(module, "ACCEPTANCE", Snapshot())
    monkeypatch.setattr(module, "PARENT_SOURCE", object())
    monkeypatch.setattr(module, "file_sha256", lambda path: (
        module.ACCEPTANCE_SHA256 if path is module.ACCEPTANCE else module.PARENT_SOURCE_SHA256
    ))
    module.validate_acceptance_snapshot(
        "dev", "final", {"parent_version": "parent"}, {"metrics": {"rows": 1000}}
    )
    with pytest.raises(ValueError, match="explicit reports"):
        module.validate_acceptance_snapshot(
            "changed", "final", {"parent_version": "parent"},
            {"metrics": {"rows": 1000}},
        )


def test_final_artifact_must_be_passed_complete_and_regression_free(monkeypatch):
    outcomes = [{"id": str(index), "suite": "final_v22",
                 "before_top1": True, "top1": True,
                 "before_top5": True, "top5": True,
                 "before_sentence_pool_covered": True,
                 "sentence_pool_covered": True}
                for index in range(module.PROFILE["final_rows"])]
    metrics = module.summarize(outcomes)
    report = {"passed": True, "scope": "gated exposed final regression",
              "development_report_sha256": "dev", "training_probe_sha256": module.TRAIN_PROBE_SHA256,
              "preparation_manifest_sha256": module.PREPARATION_MANIFEST,
              "cases_sha256": "cases", "child_option_audit": {"sha256": "audit"},
              "source_snapshot_sha256": "source", "identity_snapshot_sha256": "identity",
              "deployed": False, "outcomes": outcomes, "metrics": metrics}
    module.validate_final(report, "final", "final", "dev", "cases", "audit", "source", "identity")
    report["passed"] = False
    with pytest.raises(ValueError):
        module.validate_final(report, "final", "final", "dev", "cases", "audit", "source", "identity")
