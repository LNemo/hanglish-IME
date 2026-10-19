import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".omx/artifacts/verify-fresh-recombination-census-v1.py"
SPEC = importlib.util.spec_from_file_location("fresh_census_verifier", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class Tokenizer:
    def __call__(self, text, **unused):
        return {"input_ids": list(range(300 if text == "long" else 2))}


def fixture():
    selected = {"id": "one", "input": "raw source"}
    candidate = lambda text: {"text": text}
    runtime = {"id": "one", "request": {"input": "raw source", "context": ""},
               "proposed": {"input": "raw source", "result": "parent",
                            "candidates": [candidate("parent"), candidate("raw source")],
                            "pipeline": {"pool": [candidate("a"), candidate("raw source")]}}}
    generated = {"id": "one", "request": runtime["request"],
                 "proposals": [candidate("long")]}
    assignment = {"component_id": "component", "split": "inner_hold"}
    return selected, runtime, generated, assignment


def test_source_projection_excludes_raw_but_compact_projection_preserves_it():
    selected, runtime, generated, assignment = fixture()
    source = verifier.reconstruct_source(selected, runtime, generated, assignment)
    prepared_stub = {"expected": ["target"], "group_id": "g",
                     "source_ids": ["s"]}
    compact = verifier.reconstruct_prepared(
        selected, runtime, generated, assignment, prepared_stub)
    assert source["original_candidates"] == ["a"]
    assert source["recombined_candidates"] == ["a", "long"]
    assert compact["original_candidates"] == ["a", "raw source"]
    assert compact["recombined_candidates"] == ["a", "raw source", "long"]


def test_score_verification_uses_exact_parent_fallback_for_unsupported_union():
    selected, runtime, generated, assignment = fixture()
    source = verifier.reconstruct_source(selected, runtime, generated, assignment)
    captured = {"id": "one", "input": "raw source", "split": "inner_hold",
                "component_id": "component", "score_texts": ["a"],
                "scores": [1.0], "novel_count": 1,
                "original": {"applied": True, "result": "a",
                             "candidates": ["a", "raw source"]},
                "recombined": {"applied": False, "result": "parent",
                               "candidates": ["parent", "raw source"]}}
    support = verifier.verify_score_capture(source, captured, Tokenizer(), 3)
    assert support == {"original": True, "recombined": False}


def test_score_verification_rejects_partial_union_scoring():
    selected, runtime, generated, assignment = fixture()
    source = verifier.reconstruct_source(selected, runtime, generated, assignment)
    captured = {"id": "one", "input": "raw source", "split": "inner_hold",
                "component_id": "component", "score_texts": ["a", "long"],
                "scores": [1.0, 2.0], "novel_count": 1,
                "original": {"applied": True, "result": "a",
                             "candidates": ["a", "raw source"]},
                "recombined": {"applied": True, "result": "long",
                               "candidates": ["long", "a", "raw source"]}}
    with pytest.raises(ValueError, match="score domain"):
        verifier.verify_score_capture(source, captured, Tokenizer(), 3)


def test_self_test_is_model_free():
    assert verifier.self_test() == {
        "schema": "fresh-recombination-census-independent-verification-v1",
        "self_test": True}
