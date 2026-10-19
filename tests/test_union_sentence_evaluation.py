import copy

import pytest

import scripts.evaluate_union_sentence_ranker as module


def training_gate():
    return {
        "checkpoint_sha256": "weight", "eligible_for_runtime_dev": True,
        "regressions": [], "test_data_read": False,
        "parent": {"rows": 128, "correct": 70},
        "final": {"rows": 128, "correct": 71},
    }


def test_training_gate_requires_complete_natural_gain_and_no_regression():
    module.validate_training_gate(training_gate(), {"model.safetensors": "weight"})
    for mutation in ("weight", "eligible", "regression", "rows", "gain", "test"):
        gate = copy.deepcopy(training_gate())
        if mutation == "weight":
            gate["checkpoint_sha256"] = "changed"
        elif mutation == "eligible":
            gate["eligible_for_runtime_dev"] = False
        elif mutation == "regression":
            gate["regressions"] = ["case"]
        elif mutation == "rows":
            gate["final"]["rows"] = 127
        elif mutation == "gain":
            gate["final"]["correct"] = 70
        else:
            gate["test_data_read"] = True
        with pytest.raises(ValueError):
            module.validate_training_gate(gate, {"model.safetensors": "weight"})


def clean_results():
    return {
        name: {"rows": count, "before_top1": count - 1, "top1": count,
               "regressions": [], "parent_runtime_parity_rows": count}
        for name, count in module.DEV_COUNTS.items()
    }


@pytest.mark.parametrize("mutation", ["missing", "count", "regression", "no_gain", "parity"])
def test_runtime_gate_requires_all_four_regression_free_suites_and_natural_gain(mutation):
    results = clean_results()
    if mutation == "missing":
        del results["reasoning_dev"]
    elif mutation == "count":
        results["original_dev"]["rows"] = 139
    elif mutation == "regression":
        results["selection_dev"]["regressions"] = ["case"]
    elif mutation == "no_gain":
        results["natural_dev"]["top1"] = results["natural_dev"]["before_top1"]
    else:
        results["natural_dev"]["parent_runtime_parity_rows"] = 127
    assert module.runtime_development_passed(results) is False
    assert module.runtime_development_passed(clean_results()) is True


def test_visible_candidates_matches_runtime_raw_append_semantics():
    paths = [(str(index), float(index), []) for index in range(6)]

    assert module.visible_candidates(paths, "2") == ["0", "1", "2", "3", "4"]
    assert module.visible_candidates(paths, "raw") == ["0", "1", "2", "3", "raw"]


class TraceCase(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_reads = 0

    def __getitem__(self, key):
        if key == "expected":
            self.target_reads += 1
        return super().__getitem__(key)


def path(text, score=0.):
    return (text, score, [])


class Converter:
    def __init__(self, case, eligible=True):
        self.case = case
        self.ranker = object()
        self.sentence_ranker = object()
        self.eligible = eligible

    def convert(self, text, context):
        assert self.case.target_reads == 0
        candidates = ["parent", text] if self.eligible else [text]
        return {"candidates": [{"text": value} for value in candidates],
                "sentence_ranking_applied": self.eligible}


def test_evaluate_case_is_label_independent_and_checks_prepared_pool(monkeypatch):
    case = TraceCase({"id": "c", "input": "two words", "context": "ctx",
                      "expected": ["student"]})
    converter = Converter(case)
    parent_ranker = converter.ranker

    def fake_propose(actual, student, text, context):
        assert case.target_reads == 0
        return {"parent_paths": [path("parent"), path(text)],
                "union_paths": [path("parent"), path(text), path("student")],
                "sentence_eligible": True}

    def fake_choose(proposal, ranker, context):
        assert case.target_reads == 0
        paths = proposal["union_paths"]
        if ranker is converter.sentence_ranker:
            paths = [paths[0], *paths[1:]]
        else:
            paths = [next(item for item in paths if item[0] == "student"), *paths[:-1]]
        return {"paths": paths, "sentence_ranking_applied": True}

    monkeypatch.setattr(module, "propose", fake_propose)
    monkeypatch.setattr(module, "choose", fake_choose)
    prepared = {"id": "c", "input": "two words", "context": "ctx",
                "parent_candidates": ["parent", "two words"],
                "candidates": ["parent", "two words", "student"],
                "sentence_eligible": True}

    result = module.evaluate_case(converter, object(), object(), case, prepared)

    assert case.target_reads > 0
    assert converter.ranker is parent_ranker
    assert result["before_top1"] is False and result["top1"] is True
    assert result["before_sentence_pool_covered"] is False
    assert result["sentence_pool_covered"] is True
    assert result["parent_runtime_parity"] is True


def test_fallback_parity_keeps_raw_and_nullable_pool_coverage(monkeypatch):
    case = TraceCase({"id": "f", "input": "single", "context": "",
                      "expected": ["single"]})
    converter = Converter(case, eligible=False)

    monkeypatch.setattr(module, "propose", lambda *args: {
        "parent_paths": [path("single")], "union_paths": [path("single")],
        "sentence_eligible": False})
    monkeypatch.setattr(module, "choose", lambda proposal, ranker, context: {
        "paths": proposal.get("union_paths", proposal["parent_paths"]),
        "sentence_ranking_applied": False})

    result = module.evaluate_case(converter, object(), object(), case)

    assert result["actual"] == ["single"]
    assert result["before_top1"] is True and result["top1"] is True
    assert result["before_sentence_pool_covered"] is None
    assert result["sentence_pool_covered"] is None


def test_prepared_pool_mismatch_is_rejected_before_labels(monkeypatch):
    case = TraceCase({"id": "c", "input": "two words", "context": "ctx",
                      "expected": ["student"]})
    converter = Converter(case)
    monkeypatch.setattr(module, "propose", lambda *args: {
        "parent_paths": [path("parent"), path("two words")],
        "union_paths": [path("parent"), path("two words"), path("student")],
        "sentence_eligible": True})
    monkeypatch.setattr(module, "choose", lambda proposal, ranker, context: {
        "paths": proposal["union_paths"], "sentence_ranking_applied": True})
    prepared = {"id": "c", "input": "two words", "context": "ctx",
                "parent_candidates": ["changed"], "candidates": ["changed"],
                "sentence_eligible": True}

    with pytest.raises(ValueError, match="prepared data"):
        module.evaluate_case(converter, object(), object(), case, prepared)
    assert case.target_reads == 0
