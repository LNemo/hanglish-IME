import ast
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from scripts import verify_current_phonetic_choice_dev_screen as verifier


def segment(source, text, kind="converted"):
    return [{"input_start": 0, "input_end": len(source), "source": source,
             "text": text, "kind": kind}]


def candidate(source, text, score=1.0):
    return {"text": text, "score": score, "segments": segment(source, text)}


def branch(source, first, rest=(), ranked=True, latent=()):
    values = [candidate(source, first)] + [candidate(source, text, 0.5) for text in rest]
    return {"result": first, "candidates": values,
            "sentence_ranking_applied": ranked, "latent_top5": list(latent)}


def prediction(identifier, source, current, base, deployed, proposed, pool):
    current_branch = branch(source, current)
    current_branch["latent_top5"] = None
    return {"id": identifier, "request": {"input": source, "context": ""},
            "branches": {"current_production": current_branch,
                         "base_full": branch(source, base, latent=[base]),
                         "deployed_full": branch(source, deployed, latent=[deployed]),
                         "proposed": branch(source, proposed, latent=[proposed])},
            "score_rows": [{"text": text} for text in pool]}


def label(identifier, source, expected):
    return {"id": identifier, "input": source, "context": "", "split": "dev",
            "expected": expected}


def test_verifier_does_not_import_producer_or_ml_runtime():
    tree = ast.parse(Path(verifier.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert "scripts.evaluate_current_phonetic_choice_dev" not in imported
    assert not any(name.startswith(("torch", "transformers")) for name in imported)


def test_score_matrix_is_numpy_projection_and_rejects_bad_shapes():
    model = {"schema": "phonetic-score-scale-model-v1", "arm": "learned_scaled",
             "parameters": [2.0, 1.0, 2.0, 3.0, 4.0],
             "scale": [1.0, 2.0, 3.0, 4.0]}
    result = verifier.score_matrix([0.5, -1.0], [[1, 2, 3, 4], [0, 2, 0, 4]],
                                   {"proposed": model})
    np.testing.assert_array_equal(result["base_full"], [0.5, -1.0])
    np.testing.assert_allclose(result["proposed"], [11.0, 4.0])
    with pytest.raises(ValueError, match="shape"):
        verifier.score_matrix([1.0], [[1, 2, 3]], {"proposed": model})
    with pytest.raises(ValueError, match="model"):
        verifier.score_matrix([1.0], [[1, 2, 3, 4]],
                              {"proposed": {**model, "scale": [1, 0, 1, 1]}})


def test_segments_require_exact_contiguous_source_and_target_reconstruction():
    valid = [{"input_start": 0, "input_end": 2, "source": "ab", "text": "아",
              "kind": "converted"},
             {"input_start": 2, "input_end": 3, "source": "!", "text": "!",
              "kind": "literal"}]
    verifier.validate_segments("ab!", "아!", valid)
    for mutation in (
        [{**valid[0], "input_end": 1}, valid[1]],
        [valid[0], {**valid[1], "source": "?"}],
        [valid[0], {**valid[1], "text": "?"}],
    ):
        with pytest.raises(ValueError, match="candidate"):
            verifier.validate_segments("ab!", "아!", mutation)


def test_public_render_sorts_ties_and_reserves_raw_slot_without_changing_latent():
    source = "raw"
    paths = [(f"t{i}", float(9 - i), segment(source, f"t{i}")) for i in range(6)]
    paths += [("aa", 10.0, segment(source, "aa")), ("zz", 10.0, segment(source, "zz"))]
    public, latent = verifier.render_public(source, paths)
    assert latent == ["aa", "zz", "t0", "t1", "t2"]
    assert [row["text"] for row in public] == ["aa", "zz", "t0", "t1", source]
    assert public[-1] == verifier.raw_candidate(source)


def test_unranked_row_uses_empty_latent_and_exact_actual_current_branch():
    source = "123"
    raw = verifier.raw_candidate(source)
    response = {"version": verifier.FULL_VERSION, "input": source, "context": "",
                "external_context_used": False, "result": source, "candidates": [raw],
                "sentence_ranking_applied": False,
                "phonetic_choice": {"parent_version": verifier.FULL_PARENT_VERSION,
                                    "rows": []}}
    source_row = {"id": "x", "request": {"input": source, "context": ""},
                  "base_paths": [], "proposed": response, "union_reference": response}
    current_response = {**response, "version": verifier.CURRENT_VERSION}
    current_row = {"id": "x", "request": source_row["request"], "response": current_response}
    fallback = {"result": source, "candidates": [raw], "sentence_ranking_applied": False,
                "latent_top5": []}
    current = {**fallback, "latent_top5": None}
    frozen = {"id": "x", "request": source_row["request"],
              "kind": "cached-full-union-fixed-coefficient-projection",
              "branches": {"current_production": current, "base_full": fallback,
                           "deployed_full": fallback, "proposed": fallback}, "score_rows": []}
    dummy = {"schema": "phonetic-score-scale-model-v1", "arm": "learned_scaled",
             "parameters": [1, 0, 0, 0, 0], "scale": [1, 1, 1, 1]}
    assert verifier.replay_row(source_row, current_row, frozen, dummy, dummy) == 0
    frozen["branches"]["proposed"]["latent_top5"] = [source]
    with pytest.raises(ValueError, match="unranked"):
        verifier.replay_row(source_row, current_row, frozen, dummy, dummy)


def test_metrics_distinguish_public_latent_and_full_pool_coverage():
    first = prediction("a", "hello!", "오답!", "오답!", "오답!", "정답!",
                       ["정답!", "숨은정답!"])
    # Gold exists in the full scored pool but not in this branch's latent/public top five.
    first["branches"]["base_full"]["latent_top5"] = ["오답!"]
    first["branches"]["deployed_full"]["latent_top5"] = ["오답!"]
    result = verifier.evaluate_rows([first], [label("a", "hello!", ["정답!"])], expected_rows=1)
    assert result["branches"]["base_full"]["scored_pool_coverage"] == 1
    assert result["branches"]["base_full"]["latent_top5"] == 0
    assert result["branches"]["proposed"]["latent_top5"] == 1
    assert result["transitions"]["overall"]["corrections"] == 1
    assert result["quality_gate"] is True


def test_failed_regression_gate_remains_a_valid_metric_result():
    row = prediction("a", "test!", "정답 test!", "정답 test!", "정답 test!", "오답!",
                     ["정답 test!", "오답!"])
    result = verifier.evaluate_rows([row], [label("a", "test!", ["정답 test!"])], expected_rows=1)
    assert result["transitions"]["overall"]["regressions"] == 1
    assert result["transitions"]["overall"]["top5_losses"] == 1
    assert result["transitions"]["overall"]["ascii_signature_regressions"] == 1
    assert result["quality_gate"] is False
    assert result["actual_runtime_authorized"] is False


def resource(stage, operation, elapsed, operation_elapsed=0.01, rss=100, growth=0):
    return {"at": datetime.now(timezone.utc).isoformat(), "stage": stage,
            "operation": operation,
            "operation_elapsed_s": operation_elapsed if operation is not None else None,
            "elapsed_s": elapsed, "rss_bytes": rss,
            "swap_used_bytes": 1000 + growth, "swap_growth_bytes": growth}


def trace():
    rows, elapsed = [], 0.0
    for index, operation in enumerate(verifier.OPERATIONS):
        elapsed += .1; rows.append(resource("operation_start", operation, elapsed))
        elapsed += .1; rows.append(resource("operation_end", operation, elapsed))
        if index == len(verifier.OPERATIONS) - 2:
            elapsed += .1; rows.append(resource("after_report", None, elapsed))
    elapsed += .1; rows.append(resource("after_completion", None, elapsed))
    return rows


def test_resource_replay_enforces_sequence_and_all_limits():
    result = verifier.verify_resources(trace())
    assert result["operations"] == 6
    assert result["peak_rss_bytes"] == 100
    mutations = [("rss_bytes", verifier.LIMITS["rss_bytes"] + 1),
                 ("swap_growth_bytes", verifier.LIMITS["swap_growth_bytes"] + 1),
                 ("elapsed_s", verifier.LIMITS["whole_seconds"] + 1),
                 ("operation_elapsed_s", verifier.LIMITS["update_seconds"] + 1)]
    for key, value in mutations:
        rows = trace(); rows[0][key] = value
        if key == "swap_growth_bytes":
            rows[0]["swap_used_bytes"] = 1000 + value
        with pytest.raises(ValueError):
            verifier.verify_resources(rows)


def test_cli_requires_output_and_producer_sha():
    source = Path(verifier.__file__).read_text()
    assert 'parser.add_argument("--output", type=Path, required=True)' in source
    assert 'parser.add_argument("--producer-sha256", required=True)' in source
    assert verifier.PRODUCER_SHA256 == "6141379bc626b6c57c481243e7fd167f40f04658ac2c019dc9cd6dbbbd9cfb07"
    assert verifier.WRAPPER_SHA256 == "92acc04d8070ec32b190d93d16564a463774e463fa1f6f01c0c530ec906b8bdc"
