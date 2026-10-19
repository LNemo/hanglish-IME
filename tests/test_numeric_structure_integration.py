import copy

import pytest

import scripts.verify_numeric_structure_integration as module


def candidate(text, score=1.0):
    return {"text": text, "score": score, "segments": [
        {"input_start": 0, "input_end": 1, "source": "x",
         "text": text, "kind": "numeric"}
    ]}


def test_candidate_objects_require_exact_keys_order_segments_and_score_tolerance():
    expected = [candidate("일"), candidate("x", None)]
    assert module.candidate_objects_equal(copy.deepcopy(expected), expected)
    close = copy.deepcopy(expected)
    close[0]["score"] += module.PROFILE["score_tolerance"] / 2
    assert module.candidate_objects_equal(close, expected)
    assert not module.candidate_objects_equal(list(reversed(expected)), expected)
    changed = copy.deepcopy(expected)
    changed[0]["extra"] = True
    assert not module.candidate_objects_equal(changed, expected)
    changed = copy.deepcopy(expected)
    changed[0]["score"] += module.PROFILE["score_tolerance"] * 2
    assert not module.candidate_objects_equal(changed, expected)


def test_score_lists_require_shape_nullability_and_tolerance():
    assert module.score_lists_equal(None, None)
    assert not module.score_lists_equal([], None)
    assert module.score_lists_equal([1.0], [1.00001])
    assert not module.score_lists_equal([1.0], [1.1])
    assert not module.score_lists_equal([], [1.0])


class TraceReference(dict):
    def __init__(self, **values):
        super().__init__(**values)
        self.label_reads = 0

    def __getitem__(self, key):
        if key == "expected":
            self.label_reads += 1
        return super().__getitem__(key)


class Recorder:
    def __init__(self):
        self.candidates = self.scores = None

    def reset(self):
        self.candidates = self.scores = None


class Converter:
    def __init__(self, reference, recorder, weights):
        self.reference, self.recorder, self.weights = reference, recorder, weights

    def convert(self, text, context):
        assert self.reference.label_reads == 0
        self.recorder.candidates = list(self.reference["sentence_pool"])
        self.recorder.scores = list(self.reference["sentence_scores"])
        return {"version": "new", "model_artifacts": self.weights,
                "candidates": copy.deepcopy(self.reference["candidates"]),
                "sentence_ranking_applied": True}


def test_case_verifier_checks_objects_pool_scores_and_reads_labels_last():
    weights = {"weight": "a" * 64}
    reference = TraceReference(
        id="id", suite="natural_dev", input="x", context="ctx", expected=["일"],
        candidates=[candidate("일")], sentence_pool=["일"], sentence_scores=[2.0],
        before_top1=False, top1=True, before_top5=False, top5=True,
        before_sentence_pool_covered=False, sentence_pool_covered=True,
    )
    recorder = Recorder()
    result = module.verify_case(
        Converter(reference, recorder, weights), recorder, reference,
        "development", "new", weights,
    )
    assert reference.label_reads == 1
    assert result["candidate_objects_equal"] is True
    assert result["sentence_scores_within_tolerance"] is True


def test_train_case_uses_fixed_train_suite_without_reference_suite():
    weights = {"weight": "a" * 64}
    reference = TraceReference(
        id="train-id", input="x", context="ctx", expected=["일"],
        child_candidates=[candidate("일")], child_sentence_pool=["일"],
        child_sentence_scores=[2.0], parent_top1=False, child_top1=True,
        parent_top5=False, child_top5=True, parent_pool_covered=False,
        child_pool_covered=True,
    )
    recorder = Recorder()

    class TrainConverter(Converter):
        def convert(self, text, context):
            assert self.reference.label_reads == 0
            self.recorder.candidates = list(self.reference["child_sentence_pool"])
            self.recorder.scores = list(self.reference["child_sentence_scores"])
            return {"version": "new", "model_artifacts": self.weights,
                    "candidates": copy.deepcopy(self.reference["child_candidates"]),
                    "sentence_ranking_applied": True}

    result = module.verify_case(
        TrainConverter(reference, recorder, weights), recorder, reference,
        "train", "new", weights,
    )
    assert result["suite"] == "train"
    assert reference.label_reads == 1


def test_case_verifier_rejects_candidate_difference():
    weights = {"weight": "a" * 64}
    reference = TraceReference(
        id="id", suite="final_v22", input="x", context="", expected=["일"],
        candidates=[candidate("일")], sentence_pool=["일"], sentence_scores=[2.0],
        before_top1=True, top1=True, before_top5=True, top5=True,
        before_sentence_pool_covered=True, sentence_pool_covered=True,
    )
    recorder = Recorder()
    converter = Converter(reference, recorder, weights)
    converter.reference["candidates"] = [candidate("이")]
    expected = copy.deepcopy(reference)
    expected["candidates"] = [candidate("일")]
    with pytest.raises(ValueError, match="differs"):
        module.verify_case(converter, recorder, expected, "final", "new", weights)


def test_source_transition_allows_only_declared_core_change_and_new_module(
        tmp_path, monkeypatch):
    old = {"hanglish/core/neural_converter.py": module.PARENT_SOURCE_SHA256,
           "hanglish/core/other.py": "a" * 64}
    for relative, value in (("hanglish/core/neural_converter.py", b"new"),
                            ("hanglish/core/other.py", b"same"),
                            ("hanglish/core/numeric_structure.py", b"numeric")):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    old["hanglish/core/other.py"] = module.file_sha256(tmp_path / "hanglish/core/other.py")
    weights = {"checkpoints/model": "b" * 64}
    path = tmp_path / "checkpoints/model"
    path.parent.mkdir(parents=True); path.write_bytes(b"weight")
    weights["checkpoints/model"] = module.file_sha256(path)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    current, actual_weights = module.validate_source_transition(old, weights, "c" * 64)
    assert module.DECLARED_NEW_SOURCE in current and actual_weights == weights
    (tmp_path / "hanglish/core/other.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="undeclared"):
        module.validate_source_transition(old, weights, "c" * 64)


def test_acceptance_requires_exact_reports_parent_and_sixteen_weights():
    weights = {str(i): "a" * 64 for i in range(16)}
    reports = {name: {"source_hashes": weights} for name in module.REPORTS}
    accepted = {"reports": {name: {"sha256": values[1], "cases_sha256": values[3]}
                            for name, values in module.REPORTS.items()},
                "parent_runtime_version": module.PARENT_VERSION,
                "parent_source_sha256": module.PARENT_SOURCE_SHA256,
                "model_artifacts": weights,
                "score_tolerance": module.PROFILE["score_tolerance"],
                "policy": "fixed numeric overlay"}
    module.validate_acceptance(accepted, reports)
    accepted["reports"]["train"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="lineage"):
        module.validate_acceptance(accepted, reports)
