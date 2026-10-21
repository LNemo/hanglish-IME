import math
from types import SimpleNamespace

import pytest

from scripts import evaluate_terminal_ne as evaluation


class CachedCall:
    def __init__(self):
        self.clears = 0

    def cache_clear(self):
        self.clears += 1


def fake_converter(index=None):
    endings = SimpleNamespace(ne={"gane": {"가네"}} if index is None else index)
    generator = SimpleNamespace(
        productive_endings=endings, options=CachedCall(), stem_options=CachedCall())
    return SimpleNamespace(generator=generator, word_options=CachedCall())


def case(case_id="case", suite="diagnostics", category="rule"):
    return {"id": case_id, "input": "gane", "context": "",
            "raw": False, "expected": ["가네"], "suite": suite,
            "category": category}


def metrics(**changes):
    value = {
        "baseline": {"counts": {"errors": 0}},
        "proposed": {"counts": {"errors": 0}},
        "corrections": [],
        "regressions": [],
        "top5_gains": ["diagnostic"],
        "top5_regressions": [],
    }
    value.update(changes)
    return value


def test_project_inputs_removes_all_labels_and_evaluation_metadata():
    assert evaluation.project_inputs([case()]) == [{
        "id": "case", "input": "gane", "context": "", "raw": False}]


def test_capture_arms_clears_all_caches_for_both_arms_and_restores_feature(tmp_path):
    converter = fake_converter()
    original_index = converter.generator.productive_endings.ne
    observations = []

    def capture_fn(current, inputs, path, guard, arm):
        observations.append((arm, dict(current.generator.productive_endings.ne), inputs))
        return [arm]

    baseline, proposed = evaluation.capture_arms(
        converter, [case()], tmp_path, object(), capture_fn=capture_fn)

    assert baseline == ["baseline"]
    assert proposed == ["proposed"]
    assert observations[0][0:2] == ("baseline", {})
    assert observations[1][0:2] == ("proposed", {"gane": {"가네"}})
    assert all("expected" not in row for _, _, inputs in observations for row in inputs)
    assert converter.generator.productive_endings.ne is original_index
    assert converter.word_options.clears == 3
    assert converter.generator.options.clears == 3
    assert converter.generator.stem_options.clears == 3


def test_capture_arms_restores_feature_and_caches_after_failure(tmp_path):
    converter = fake_converter()

    def capture_fn(current, inputs, path, guard, arm):
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        evaluation.capture_arms(
            converter, [case()], tmp_path, object(), capture_fn=capture_fn)

    assert converter.generator.productive_endings.ne == {"gane": {"가네"}}
    assert converter.word_options.clears == 2
    assert converter.generator.options.clears == 2
    assert converter.generator.stem_options.clears == 2


def test_capture_arms_rejects_an_empty_proposed_index(tmp_path):
    with pytest.raises(ValueError, match="index is empty"):
        evaluation.capture_arms(
            fake_converter({}), [case()], tmp_path, object(),
            capture_fn=lambda *args: [])


def test_adoption_gate_accepts_only_diagnostic_top5_gain_without_regressions():
    cases = [case("diagnostic"), case("test-gain", suite="test")]

    gate = evaluation.adoption_gate(
        metrics(top5_gains=["diagnostic", "test-gain"]), cases)

    assert gate["adopted"] is True
    assert gate["diagnostic_top5_gains"] == ["diagnostic"]
    assert all(gate["requirements"].values())


@pytest.mark.parametrize(
    "changes",
    [
        {"baseline": {"counts": {"errors": 1}}},
        {"proposed": {"counts": {"errors": 1}}},
        {"regressions": ["diagnostic"]},
        {"top5_regressions": ["diagnostic"]},
        {"top5_gains": ["test-gain"]},
        {"top5_gains": []},
    ],
)
def test_adoption_gate_rejects_every_strict_failure(changes):
    cases = [case("diagnostic"), case("test-gain", suite="test")]

    assert evaluation.adoption_gate(metrics(**changes), cases)["adopted"] is False


def test_grouped_latency_separates_suites_and_categories():
    cases = [case("a", suite="dev", category="first"),
             case("b", suite="dev", category="second"),
             case("c", suite="test", category="first")]
    captures = [{"elapsed_ms": 1.0}, {"elapsed_ms": 3.0}, {"elapsed_ms": 5.0}]

    suites = evaluation.grouped_latency(captures, cases, "suite")
    categories = evaluation.grouped_latency(captures, cases, "category")

    assert suites["dev"] == {"p50_ms": 1.0, "p95_ms": 3.0, "max_ms": 3.0}
    assert suites["test"]["max_ms"] == 5.0
    assert categories["first"]["p95_ms"] == 5.0


def test_grouped_quality_separates_suite_counts_after_capture():
    cases = [case("a", suite="dev"), case("b", suite="test")]
    captures = [
        {"response": {"candidates": [{"text": "가네"}]}},
        {"error": {"type": "failure"}},
    ]

    quality = evaluation.grouped_quality(captures, cases, "suite")

    assert quality["dev"] == {"rows": 1, "top1": 1, "top5": 1, "errors": 0}
    assert quality["test"] == {"rows": 1, "top1": 0, "top5": 0, "errors": 1}


@pytest.mark.parametrize(
    "rows",
    [[], [{"elapsed_ms": math.nan}], [{"elapsed_ms": -1.0}], [{}]],
)
def test_verify_numeric_captures_rejects_incomplete_or_invalid_rows(rows):
    with pytest.raises(ValueError):
        evaluation.verify_numeric_captures(rows, 1)


def test_load_cases_has_fixed_unique_suite_counts_and_pins():
    cases, pins = evaluation.load_cases()

    assert len(cases) == 1519
    assert len({row["id"] for row in cases}) == 1519
    assert sum(row["suite"] in {"natural", "original", "selection", "reasoning"}
               for row in cases) == 368
    assert sum(row["suite"] == "test" for row in cases) == 1000
    assert sum(row["suite"] == "diagnostics" for row in cases) == 151
    assert pins and all(len(value) == 64 for value in pins.values())


def test_additional_pins_cover_runtime_lexicon_and_watchdog_sources():
    pins = evaluation.additional_source_pins()

    assert any(path.endswith("hanglish/core/productive_endings.py") for path in pins)
    assert any(path.endswith("hanglish/core/workbench.py") for path in pins)
    assert any(path.endswith("scripts/preflight_pko_seq2seq.py") for path in pins)
    assert any(path.endswith("scripts/source_span_resources.py") for path in pins)
    assert any(path.endswith("mecab-ko-dic/NNBC.csv") for path in pins)
    assert any(path.endswith("mecab-ko-dic/SOURCE.json") for path in pins)
    assert any(path.endswith("data/morphology_lexicon.json") for path in pins)
    assert any(path.endswith("data/workbench_lexicon.json") for path in pins)
    assert any(path.endswith("data/common_english.txt") for path in pins)
    assert any(path.endswith("data/it_terms.txt") for path in pins)
    assert any(path.endswith("data/challenge/train_lexicon.json") for path in pins)
    assert any(path.endswith("predicate_boundary_inventory.json") for path in pins)
