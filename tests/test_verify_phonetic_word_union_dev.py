import copy
import subprocess
import sys

import pytest

from scripts import verify_phonetic_word_union_dev as module


def candidate(source, text=None, score=None):
    text = source if text is None else text
    return {"text": text, "score": score, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": text, "kind": "raw" if text == source else "lexicon"}]}


def response(source, texts):
    return {"candidates": [candidate(source, text, index) for index, text in enumerate(texts)],
            "result": texts[0], "engine_ms": 1.0}


def test_independent_aggregate_counts_quality_stages_and_ascii():
    rows = [
        {"id": "a", "request": {"input": "foo bar", "context": ""},
         "parent": response("foo bar", ["오답", "foo bar"]),
         "proposed": {**response("foo bar", ["정답", "foo bar"]),
                      "sentence_ranking_applied": True,
                      "phonetic_word_union": {"generated_extras": [{"text": "정답"}],
                          "scoreable_extras": ["정답"], "union": ["오답", "정답"],
                          "public": ["정답", "foo bar"], "generation_budget_exhausted": False,
                          "skipped_known_english": []}}, "parent_pool": ["오답"]},
        {"id": "b", "request": {"input": "test neun", "context": ""},
         "parent": response("test neun", ["test는", "test neun"]),
         "proposed": {**response("test neun", ["test는", "test neun"]),
                      "sentence_ranking_applied": True,
                      "phonetic_word_union": {"generated_extras": [], "scoreable_extras": [],
                          "union": ["test는"], "public": ["test는", "test neun"],
                          "generation_budget_exhausted": False,
                          "skipped_known_english": [{}]}}, "parent_pool": ["test는"]},
    ]
    labels = [{"id": "a", "input": "foo bar", "context": "", "split": "dev",
               "expected": ["정답"], "category": "x"},
              {"id": "b", "input": "test neun", "context": "", "split": "dev",
               "expected": ["test는"], "category": "y"}]
    metrics = module.aggregate(rows, labels)
    assert metrics["parent"]["counts"] == {"rows": 2, "top1": 1, "top5": 1, "errors": 0}
    assert metrics["proposed"]["counts"] == {"rows": 2, "top1": 2, "top5": 2, "errors": 0}
    assert metrics["corrections"] == 1 and metrics["regressions"] == 0
    assert metrics["quality_gate"] is True
    assert metrics["stage_counts"]["marginal_generated_gold_rows"] == 1
    assert metrics["english_ascii_signature"]["rows_with_reference_ascii_words"] == 1


def test_literal_signature_excludes_lowercase_and_all_korean_jamo_ranges():
    assert module.literal_signature("abc가ᄀ㄰_Z") == "_Z"


def resource(at, elapsed, stage, operation=None, operation_elapsed=None):
    return {"at": f"2026-09-21T00:00:{at:02d}+00:00", "elapsed_s": elapsed,
            "operation": operation, "operation_elapsed_s": operation_elapsed,
            "rss_bytes": 100, "stage": stage, "swap_growth_bytes": 10,
            "swap_used_bytes": 100}


def test_resources_require_atomic_exact_operations_and_stable_swap():
    rows = [resource(0, 0.0, "operation_start", "load_models", 0.0),
            resource(1, 0.1, "operation_end", "load_models", 0.1),
            resource(2, 0.2, "after_report"),
            resource(3, 0.3, "operation_start", "write_completion", 0.0),
            resource(4, 0.4, "operation_end", "write_completion", 0.1),
            resource(5, 0.5, "after_completion")]
    operations = ["load_models", "write_completion"]
    result = module.verify_resources(rows, operations)
    assert result["operations"] == 2 and result["baseline_swap_bytes"] == 90
    changed = copy.deepcopy(rows); changed[1]["swap_used_bytes"] = 101
    with pytest.raises(ValueError, match="baseline changed"):
        module.verify_resources(changed, operations)
    changed = copy.deepcopy(rows); changed[1]["operation_elapsed_s"] = 60.1
    with pytest.raises(ValueError, match="operation end"):
        module.verify_resources(changed, operations)
    changed = copy.deepcopy(rows); changed.pop()
    with pytest.raises(ValueError, match="operation sequence"):
        module.verify_resources(changed, operations)


def test_trace_rejects_invalid_generated_source_offsets():
    with pytest.raises(ValueError, match="source offset"):
        module.validate_offset_row("gim bap", {
            "source_start": 1, "source_end": 3, "source": "gim", "text": "김 bap",
            "replacement": "김", "parent_index": 0}, generated=True)


def test_failure_artifacts_are_rejected(tmp_path):
    module.require_no_failures(tmp_path)
    for name in ("failure.json", "failure-report.json", "resource-watchdog-failure.json"):
        path = tmp_path / name; path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match=name):
            module.require_no_failures(tmp_path)
        path.unlink()


def test_verifier_has_no_model_import_or_runner_aggregate_reuse():
    source = (module.ROOT / "scripts/verify_phonetic_word_union_dev.py").read_text()
    assert "import torch" not in source and "torch.load" not in source
    assert "evaluate_phonetic_word_union_dev" not in source
    assert "load_references" not in source


def test_import_succeeds_when_torch_and_model_modules_are_blocked():
    code = """
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.') or name.startswith('hanglish.models'):
        raise RuntimeError('blocked neural import: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
import scripts.verify_phonetic_word_union_dev
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=module.ROOT,
        text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_parser_requires_run():
    assert str(module.parser().parse_args(["--run", "x"]).run) == "x"
