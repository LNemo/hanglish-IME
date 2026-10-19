import copy

import pytest

from scripts import evaluate_phonetic_word_union_dev as evaluator


PARENT_VERSION = "experimental-ctc-union-1-488fc89732af"


def response(result, candidates=None, *, ranked=True, trace=None):
    candidates = candidates or [{"text": result, "score": 1.0, "segments": []}]
    value = {
        "version": PARENT_VERSION,
        "result": result,
        "candidates": candidates,
        "sentence_ranking_applied": ranked,
        "ctc_word": {"generation_requests": 0, "generation_ms": 1.0},
        "ctc_union": {"generation_requests": 0, "generation_ms": 2.0},
    }
    if trace is not None:
        value["phonetic_word_union"] = trace
    return value


def trace(parent_pool, generated=(), scoreable=(), union=None, public=None):
    generated_rows = [
        {"text": text, "source_start": 0, "source_end": 3,
         "source": "gim", "replacement": "김", "parent_index": 0}
        for text in generated]
    public = list(public if public is not None else parent_pool)
    generated_texts = [row["text"] for row in generated_rows]
    return {
        "experimental": True, "parent_version": PARENT_VERSION,
        "production_pool_texts": list(parent_pool),
        "generation_requests": int(bool(generated)),
        "skipped_known_english": [], "skipped_generation_sources": [],
        "generation_budget_exhausted": False,
        "generated_extras": generated_rows,
        "scoreable_extras": list(scoreable),
        "union": list(union if union is not None else [*parent_pool, *scoreable]),
        "public": public,
        "selected_extras": [text for text in generated_texts if text in public],
        "missing_extras": [text for text in generated_texts if text not in public],
        "generation_ms": 1.0, "artifacts": {},
    }


def capture(identifier, source, parent_result, proposed_result, stage_trace, *, ranked=True):
    parent = response(parent_result)
    proposed = response(
        proposed_result,
        [{"text": text, "score": float(5 - index), "segments": []}
         for index, text in enumerate(stage_trace["public"])],
        ranked=ranked, trace=stage_trace)
    return {"id": identifier, "request": {"input": source, "context": ""},
            "parent": parent, "proposed": proposed,
            "parent_pool": list(stage_trace["production_pool_texts"])}


def public_metrics(*, corrections=0, regressions=0, losses=0, improvement=True):
    summary = {"counts": {"rows": 2, "top1": 1, "top5": 1, "errors": 0},
               "outcomes": []}
    return {"parent": copy.deepcopy(summary), "proposed": copy.deepcopy(summary),
            "corrections": [f"c{i}" for i in range(corrections)],
            "regressions": [f"r{i}" for i in range(regressions)],
            "top5_losses": [f"l{i}" for i in range(losses)],
            "improvement_gate": improvement}


def test_aggregate_accounts_generation_pruning_selection_and_unranked_rows(monkeypatch):
    monkeypatch.setattr(evaluator, "compare", lambda rows, labels: public_metrics())
    ranked_trace = trace(
        ["부모"], generated=["김"], scoreable=[], union=["부모"], public=["부모"])
    unranked_trace = trace([], generated=[], scoreable=[], union=[], public=[])
    rows = [
        capture("ranked", "gim", "부모", "부모", ranked_trace),
        capture("unranked", "raw", "raw", "raw", unranked_trace, ranked=False),
    ]
    labels = [
        {"id": "ranked", "input": "gim", "context": "", "split": "dev",
         "expected": ["김"], "category": "word"},
        {"id": "unranked", "input": "raw", "context": "", "split": "dev",
         "expected": ["raw"], "category": "word"},
    ]

    result = evaluator.aggregate(rows, labels)

    expected = {
        "ranked_rows": 1,
        "parent_pool_correct": 0,
        "with_generated_correct": 1,
        "scoreable_union_correct": 0,
        "public_top5_correct": 0,
        "selected_correct": 0,
        "rows": 2,
        "generated_extras": 1,
        "scoreable_extras": 0,
        "generation_budget_exhausted_rows": 0,
    }
    assert {key: result["stage_counts"][key] for key in expected} == expected
    assert result["stage_counts"]["marginal_generated_gold_rows"] == 1
    assert result["stage_counts"]["generated_to_scoreable_gold_losses"] == 1
    assert result["stage_counts"]["union_to_public_gold_losses"] == 0


def test_aggregate_exposes_correction_regression_top5_gate(monkeypatch):
    monkeypatch.setattr(
        evaluator, "compare",
        lambda rows, labels: public_metrics(
            corrections=2, regressions=1, losses=1, improvement=False))
    stage_trace = trace(["부모"], public=["부모"])
    rows = [capture("a", "gim", "부모", "부모", stage_trace),
            capture("b", "han", "부모", "부모", stage_trace)]
    labels = [{"id": row["id"], "input": row["request"]["input"],
               "context": "", "split": "dev", "expected": ["부모"]}
              for row in rows]

    result = evaluator.aggregate(rows, labels)

    assert result["corrections"] == 2
    assert result["regressions"] == 1 and result["top5_losses"] == 1
    assert result["improvement_gate"] is False
    assert result["quality_gate"] is False


def test_ascii_signature_regression_blocks_quality_gate(monkeypatch):
    monkeypatch.setattr(
        evaluator, "compare",
        lambda rows, labels: public_metrics(corrections=1, improvement=True))
    stage_trace = trace(["test를"], scoreable=["테스트를"],
                        union=["test를", "테스트를"], public=["테스트를", "test를"])
    rows = [capture("a", "testreul", "test를", "테스트를", stage_trace)]
    labels = [{"id": "a", "input": "testreul", "context": "", "split": "dev",
               "expected": ["test를"]}]

    result = evaluator.aggregate(rows, labels)

    english = result["english_ascii_signature"]
    assert english["rows_with_reference_ascii_words"] == 1
    assert english["parent_exact_signature"] == 1
    assert english["proposed_exact_signature"] == 0
    assert english["signature_regressions"] == 1
    assert english["literal_signature_regressions"] == 0
    assert result["improvement_gate"] is True
    assert result["quality_gate"] is False


def test_literal_signature_regression_blocks_quality_gate(monkeypatch):
    monkeypatch.setattr(
        evaluator, "compare",
        lambda rows, labels: public_metrics(corrections=1, improvement=True))
    stage_trace = trace(["김-1"], scoreable=["김"],
                        union=["김-1", "김"], public=["김", "김-1"])
    rows = [capture("a", "gim-1", "김-1", "김", stage_trace)]
    labels = [{"id": "a", "input": "gim-1", "context": "", "split": "dev",
               "expected": ["김"]}]

    result = evaluator.aggregate(rows, labels)

    assert result["english_ascii_signature"]["literal_signature_regressions"] == 1
    assert result["improvement_gate"] is True
    assert result["quality_gate"] is False


def test_aggregate_rejects_source_identity_mismatch(monkeypatch):
    monkeypatch.setattr(evaluator, "compare", lambda rows, labels: public_metrics())
    stage_trace = trace(["부모"], public=["부모"])
    rows = [capture("a", "gim", "부모", "부모", stage_trace)]
    bad_labels = [{"id": "a", "input": "han", "context": "", "split": "dev",
                   "expected": ["부모"]}]
    with pytest.raises(ValueError, match="late label source contract"):
        evaluator.aggregate(rows, bad_labels)


def test_verify_pair_accepts_exact_stage_accounting(monkeypatch):
    monkeypatch.setattr(evaluator, "verify_runtime_contract", lambda *args: None)
    parent_pool = ["gim", "부모"]
    stage_trace = trace(parent_pool, generated=["김"], scoreable=["김"],
                        union=["gim", "부모", "김"], public=["김", "부모", "gim"])
    before = response("부모")
    after = response(
        "김", [{"text": text, "score": 3.0 - index, "segments": []}
              for index, text in enumerate(stage_trace["public"])], trace=stage_trace)
    evaluator.verify_pair("gim", before, after, parent_pool, PARENT_VERSION)


@pytest.mark.parametrize("mutation,match", [
    (lambda value: value["phonetic_word_union"].update(
        production_pool_texts=["changed"]), "complete production pool"),
    (lambda value: value["phonetic_word_union"]["generated_extras"][0].update(
        source_end=99), "proposal source offset"),
    (lambda value: value["phonetic_word_union"].update(
        union=["부모"]), "pool composition"),
    (lambda value: value["phonetic_word_union"].update(
        public=["different"]), "public trace"),
])
def test_verify_pair_rejects_pool_source_and_public_mismatches(
        monkeypatch, mutation, match):
    monkeypatch.setattr(evaluator, "verify_runtime_contract", lambda *args: None)
    parent_pool = ["부모"]
    stage_trace = trace(parent_pool, generated=["김"], scoreable=["김"],
                        union=["부모", "김"], public=["김", "부모"])
    before = response("부모")
    after = response(
        "김", [{"text": "김", "score": 2.0, "segments": []},
              {"text": "부모", "score": 1.0, "segments": []}], trace=stage_trace)
    mutation(after)
    with pytest.raises(ValueError, match=match):
        evaluator.verify_pair("gim", before, after, parent_pool, PARENT_VERSION)
