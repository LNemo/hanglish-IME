import ast
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts/verify_shared_edge_selected_test.py"
SPEC = importlib.util.spec_from_file_location("shared_edge_selected_test_verifier", VERIFIER_PATH)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def dump_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def segment(source, text, kind="morphology"):
    return {"input_start": 0, "input_end": len(source), "source": source,
            "text": text, "kind": kind}


def fixture(row_id="row-1"):
    source = "abc"
    first = {"candidate_id": "candidate-a", "path_id": "path-a", "text": "가",
             "origin": "test", "origins": ["test"], "segments": [segment(source, "가")],
             "score": 0.0}
    second = {"candidate_id": "candidate-b", "path_id": "path-b", "text": "나",
              "origin": "test", "origins": ["test"], "segments": [segment(source, "나")],
              "score": 0.0}
    proposal_segment = segment(source, "다", "neural")
    proposal = {"proposal_id": "proposal-c", "text": "다", "segments": [proposal_segment],
                "edge_ids": ["edge-c"], "seed_index": 0, "seed_text": "가",
                "changed_edges": 1, "score": None, "score_type": None,
                "origin": "source_edge_recombination", "selection_eligible": False}
    proposed = {"input": source, "context": "", "result": "가",
                "candidates": [first, second], "pipeline": {"pool": [first, second]}}
    runtime = {"id": row_id, "request": {"input": source, "context": ""},
               "elapsed_ms": 12.0, "proposed": proposed}
    proposal_row = {"id": row_id, "request": runtime["request"],
                    "schema_version": "source-edge-recombination-v1",
                    "edge_inventory": [{"edge_id": "edge-c",
                        "structural_segment": proposal_segment, "provenance": ["test"]}],
                    "proposals": [proposal]}
    return runtime, proposal_row


def prediction(runtime, proposal, supported=True):
    texts, representatives, groups, path_ids = verifier.HELPER.reconstruct_domain(runtime, proposal)
    source = runtime["request"]["input"]
    targets = verifier.SHARED.isolated_targets(groups)
    residuals = [verifier.SHARED.f32(.2)] * len(texts)
    base_scores = [verifier.SHARED.f32(value - .2) for value in [1., 3., 2.]]
    scores = [verifier.SHARED.f32(base + residual)
              for base, residual in zip(base_scores, residuals)]
    edges = sorted({verifier.SHARED.edge_id(span) for group in groups for path in group
                    for span in path["segments"]})
    audit = {"edge_ids": [list(edge) for edge in edges],
             "edge_scores": [.2] * len(edges), "residuals": residuals,
             "winning_path_ids": json.loads(json.dumps(path_ids))}
    selected = (verifier.HELPER.apply_policy(runtime["proposed"], representatives, texts,
                                             scores, source, verifier.SELECTED)
                if supported else runtime["proposed"])
    responses = {"captured_runtime": verifier.HELPER.compact_response(runtime["proposed"]),
                 verifier.SELECTED: verifier.HELPER.compact_response(selected)}
    return {"id": runtime["id"], "request": runtime["request"],
            "proposal_schema_version": proposal["schema_version"],
            "proposal_ids": ["proposal-c"], "candidate_texts": texts,
            "candidate_ids": [row.get("candidate_id") or row.get("proposal_id")
                              for row in representatives], "path_ids": path_ids,
            "isolated_target_texts": targets,
            "target_cache": {"requested": len(targets), "hits": 0,
                             "misses": len(targets)},
            "support": {"supported": supported,
                        "reason": None if supported else "full_candidate_exceeds_256_tokens"},
            "scores": scores if supported else [],
            "base_scores": base_scores if supported else [],
            "latency_ms": {"captured_runtime": 12.0, "source_edge_proposal": 1.0,
                           "parent_features": 2.0, "target_extraction": 1.0,
                           verifier.SELECTED: 3.0},
            "shared_edge_audit": audit if supported else None,
            "responses": responses,
            "protected_paths": verifier.HELPER.protected_paths(source, groups),
            "protected_responses": {arm: verifier.HELPER.protected_response(source, response)
                                    for arm, response in responses.items()}}


def test_verifier_has_no_neural_or_evaluator_static_imports_and_binds_helpers():
    tree = ast.parse(VERIFIER_PATH.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert not any(name == "torch" or name.startswith("transformers")
                   or name.startswith("hanglish") or name.startswith("scripts")
                   for name in imported)
    assert verifier.sha(verifier.DEV_HELPER) == verifier.DEV_HELPER_SHA256
    assert verifier.sha(verifier.GENERATOR) == verifier.GENERATOR_SHA256
    assert verifier.sha(verifier.SHARED_HELPER) == verifier.SHARED_HELPER_SHA256
    assert 'read(RUNTIME / "report.json")' not in VERIFIER_PATH.read_text()


def test_source_edge_proposal_replay_is_deterministic_and_source_only():
    runtime, _ = fixture()
    first = verifier.replay_proposal(runtime)
    second = verifier.replay_proposal(runtime)
    assert first == second
    assert first["id"] == runtime["id"] and first["request"] == runtime["request"]
    assert "expected" not in json.dumps(first)


def test_prediction_replay_checks_domain_paths_ranking_and_mutation(monkeypatch):
    runtime, proposal = fixture()
    monkeypatch.setattr(verifier, "replay_proposal", lambda row: proposal)
    saved = prediction(runtime, proposal)
    assert verifier.verify_prediction(saved, runtime, proposal, set()) is True
    assert [row["text"] for row in saved["responses"][verifier.SELECTED]["candidates"][:3]] == [
        "나", "다", "가"]
    saved["path_ids"][0][0] = "alignment-v1-tampered"
    with pytest.raises(ValueError, match="identity"):
        verifier.verify_prediction(saved, runtime, proposal, set())


@pytest.mark.parametrize("field,match", [
    ("edge", "edge IDs"), ("residual", "residual"), ("winner", "winning"),
    ("score", "FP32"), ("cache", "cache")])
def test_prediction_rejects_shared_edge_and_cache_tampering(monkeypatch, field, match):
    runtime, proposal = fixture()
    monkeypatch.setattr(verifier, "replay_proposal", lambda row: proposal)
    saved = prediction(runtime, proposal)
    if field == "edge": saved["shared_edge_audit"]["edge_ids"][0][2] = "조작"
    elif field == "residual": saved["shared_edge_audit"]["residuals"][0] += 1
    elif field == "winner": saved["shared_edge_audit"]["winning_path_ids"][0] = ["bad"]
    elif field == "score": saved["scores"][0] += 1
    else: saved["target_cache"]["misses"] -= 1
    with pytest.raises(ValueError, match=match):
        verifier.verify_prediction(saved, runtime, proposal, set())


def test_fallback_replay_requires_exact_captured_response(monkeypatch):
    runtime, proposal = fixture()
    monkeypatch.setattr(verifier, "replay_proposal", lambda row: proposal)
    saved = prediction(runtime, proposal, False)
    assert verifier.verify_prediction(saved, runtime, proposal, set()) is False
    saved["responses"][verifier.SELECTED]["result"] = "변조"
    with pytest.raises(ValueError, match="fallback"):
        verifier.verify_prediction(saved, runtime, proposal, set())


def test_metrics_delta_and_strict_gate_include_no_prior_losses(monkeypatch):
    monkeypatch.setattr(verifier, "ROWS", 2)
    monkeypatch.setattr(verifier, "BASELINE", {"top1": 1, "top5": 2})
    runtime1, proposal1 = fixture("one")
    runtime2, proposal2 = fixture("two")
    predictions = [prediction(runtime1, proposal1), prediction(runtime2, proposal2)]
    labels = [{"id": "one", "input": "abc", "expected": ["나"], "category": "hard"},
              {"id": "two", "input": "abc", "expected": ["가"], "category": "hard"}]
    metrics = {arm: verifier.summarize(predictions, labels, arm)
               for arm in ("captured_runtime", verifier.SELECTED)}
    decision = verifier.decide(metrics)
    assert decision["offline_1000_case_gate"] is False
    assert decision["versus_captured_runtime"]["top1_loss_ids"] == ["two"]
    assert metrics[verifier.SELECTED]["categories"]["hard"]["rows"] == 2


def resource_row(operation, stage, elapsed, baseline=2000):
    return {"at": "2026-09-21T00:00:00+00:00", "elapsed_s": elapsed,
            "rss_bytes": 1000, "swap_used_bytes": baseline, "swap_growth_bytes": 0,
            "operation_elapsed_s": elapsed if operation is not None else None,
            "stage": stage, "operation": operation}


def test_resource_sequence_includes_every_proposal_and_supported_neural_ops(tmp_path):
    runtime1, proposal1 = fixture("one")
    runtime2, proposal2 = fixture("two")
    predictions = [prediction(runtime1, proposal1), prediction(runtime2, proposal2, False)]
    names = verifier.expected_operations(predictions)
    trace, elapsed = [], 0.0
    for name in names:
        trace.append(resource_row(name, "operation_start", elapsed)); elapsed += .001
        trace.append(resource_row(name, "watchdog", elapsed)); elapsed += .001
        trace.append(resource_row(name, "operation_end", elapsed)); elapsed += .001
        trace.append(resource_row(None, "watchdog", elapsed)); elapsed += .001
    final = resource_row(None, "final", elapsed); trace.append(final); elapsed += .001
    trace.append(resource_row(None, "watchdog", elapsed)); elapsed += .001
    dump_jsonl(tmp_path / "resources.jsonl", trace)
    report = {"baseline_swap_bytes": 2000, "elapsed_s": elapsed,
              "final_resources": final}
    result = verifier.verify_resources(tmp_path, report, predictions)
    assert result["operation_pairs"] == 6 and result["supported_rows"] == 1
    trace[1]["operation"] = names[1]
    dump_jsonl(tmp_path / "resources.jsonl", trace)
    with pytest.raises(ValueError, match="transition"):
        verifier.verify_resources(tmp_path, report, predictions)


def test_protected_literal_mutation_is_detected_through_reused_independent_helper():
    source = "https://x.test `copy` reul"
    end = source.index(" reul")
    good = {"text": source[:end] + " 를", "segments": [
        {"input_start": 0, "input_end": end, "source": source[:end], "text": source[:end]},
        {"input_start": end, "input_end": len(source), "source": source[end:], "text": " 를"}]}
    assert verifier.HELPER.protected_response(source, {"candidates": [good]})["verified"] is True
    bad = json.loads(json.dumps(good))
    bad["segments"][0]["text"] = bad["segments"][0]["text"].replace("copy", "cope")
    bad["text"] = "".join(row["text"] for row in bad["segments"])
    assert verifier.HELPER.protected_response(source, {"candidates": [bad]})["verified"] is False
