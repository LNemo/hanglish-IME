from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_shared_edge_dev.py"
SPEC = importlib.util.spec_from_file_location("verify_shared_edge_dev", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(verifier)
HELPER = verifier._helper()


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))


def candidate(source="a", text="아"):
    return {"text": text, "score": .1, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source, "text": text}]}


def supported_fixture():
    source = "a"; value = candidate(source)
    original = {"result": value["text"], "candidates": [deepcopy(value)],
                "pipeline": {"pool": [deepcopy(value)]}}
    runtime = {"id": "row", "request": {"input": source, "context": ""},
               "elapsed_ms": 1.5, "proposed": original}
    proposal = {"id": "row", "request": deepcopy(runtime["request"]),
                "edge_inventory": [], "proposals": []}
    texts, reps, groups, path_ids = HELPER.reconstruct_domain(runtime, proposal)
    residual = verifier.f32(.2)
    scores = {"parent": [.1], "source_local_control": [.15],
              "shared_edge": [verifier.f32(.1 + residual)]}
    responses = {"captured_runtime": HELPER.compact_response(original)}
    hashes = {"captured_runtime": HELPER.canonical_hash(original)}
    for arm in verifier.MODELS:
        full = HELPER.apply_policy(original, reps, texts, scores[arm], source, arm)
        responses[arm] = HELPER.compact_response(full); hashes[arm] = HELPER.canonical_hash(full)
    audit = {"edge_ids": [[0, 1, "아"]], "edge_scores": [.2],
             "residuals": [residual], "winning_path_ids": path_ids}
    saved = {"id": "row", "request": deepcopy(runtime["request"]),
        "candidate_texts": texts, "path_ids": path_ids, "isolated_target_texts": ["아"],
        "target_cache": {"requested": 1, "hits": 0, "misses": 1},
        "support": {"supported": True, "reason": None}, "scores": scores,
        "latency_ms": {"captured_runtime": 1.5, "parent_features": .1,
            "target_extraction": .1, "source_local_control": .1, "shared_edge": .1},
        "shared_edge_audit": audit, "responses": responses, "full_response_sha256": hashes,
        "protected_paths": HELPER.protected_paths(source, groups),
        "protected_responses": {arm: HELPER.protected_response(source, response)
                                for arm, response in responses.items()}}
    historical = {"id": "row", "request": deepcopy(runtime["request"]),
        "candidate_texts": texts, "path_ids": path_ids,
        "support": {"supported": True, "reason": None},
        "scores": {"parent": [.1], "source_local": [.15]},
        "responses": {"source_local": deepcopy(responses["source_local_control"])},
        "full_response_sha256": {"source_local": hashes["source_local_control"]}}
    return saved, runtime, proposal, historical


def fallback_fixture():
    source = "raw"; raw = HELPER.raw_candidate(source)
    original = {"result": source, "candidates": [raw], "pipeline": {"pool": [raw]}}
    runtime = {"id": "fallback", "request": {"input": source, "context": ""},
               "elapsed_ms": 2., "proposed": original}
    proposal = {"id": "fallback", "request": deepcopy(runtime["request"]),
                "edge_inventory": [], "proposals": []}
    responses = {arm: HELPER.compact_response(original) for arm in verifier.ARMS}
    hashes = {arm: HELPER.canonical_hash(original) for arm in verifier.ARMS}
    saved = {"id": "fallback", "request": deepcopy(runtime["request"]),
        "candidate_texts": [], "path_ids": [], "isolated_target_texts": [],
        "target_cache": {"requested": 0, "hits": 0, "misses": 0},
        "support": {"supported": False, "reason": "empty_nonraw_pool"}, "scores": {},
        "latency_ms": {"captured_runtime": 2., "parent_features": 0.,
            "target_extraction": 0., "source_local_control": 0., "shared_edge": 0.},
        "shared_edge_audit": None, "responses": responses, "full_response_sha256": hashes,
        "protected_paths": HELPER.protected_paths(source, []),
        "protected_responses": {arm: HELPER.protected_response(source, response)
                                for arm, response in responses.items()}}
    historical = {"id": "fallback", "request": deepcopy(runtime["request"]),
        "candidate_texts": [], "path_ids": [],
        "support": {"supported": False, "reason": "empty_nonraw_pool"},
        "scores": {}, "responses": {"source_local": responses["source_local_control"]},
        "full_response_sha256": {"source_local": hashes["source_local_control"]}}
    return saved, runtime, proposal, historical


def test_supported_prediction_reconstructs_edges_paths_cache_and_policy():
    fixture = supported_fixture(); cached = set()
    assert verifier.verify_prediction(*fixture, HELPER, cached) is True
    assert cached == {"아"}


@pytest.mark.parametrize("mutation,match", [
    ("source", "source"), ("domain", "domain"), ("edge", "edge IDs"),
    ("residual", "residual"), ("winner", "winning"), ("score", "FP32"),
    ("rank", "ranked response"), ("control", "control"),
    ("cache", "cache"), ("protected", "protected")])
def test_prediction_rejects_tampered_evidence(mutation, match):
    saved, runtime, proposal, historical = supported_fixture()
    if mutation == "source": saved["request"]["input"] = "b"
    elif mutation == "domain": saved["candidate_texts"] = ["조작"]
    elif mutation == "edge": saved["shared_edge_audit"]["edge_ids"][0][2] = "조작"
    elif mutation == "residual": saved["shared_edge_audit"]["residuals"][0] += 1
    elif mutation == "winner": saved["shared_edge_audit"]["winning_path_ids"] = [["bad"]]
    elif mutation == "score": saved["scores"]["shared_edge"][0] += 1
    elif mutation == "rank": saved["responses"]["shared_edge"]["result"] = "조작"
    elif mutation == "control": saved["scores"]["source_local_control"][0] += 1
    elif mutation == "cache": saved["target_cache"]["misses"] = 0
    else: saved["protected_paths"]["verified"] = False
    with pytest.raises(ValueError, match=match):
        verifier.verify_prediction(saved, runtime, proposal, historical, HELPER, set())


def test_cache_hit_sequence_omits_second_target_extraction():
    first, *_ = supported_fixture(); second = deepcopy(first); second["id"] = "row2"
    second["target_cache"] = {"requested": 1, "hits": 1, "misses": 0}
    predictions = [first, second]
    assert verifier.expected_operations(predictions) == [
        "load_parent_and_residuals", "parent_features:row", "target_extraction:row",
        "source_local_control:row", "shared_edge:row", "parent_features:row2",
        "source_local_control:row2", "shared_edge:row2"]


def test_fallback_requires_exact_captured_response_and_does_not_mutate_cache():
    fixture = fallback_fixture(); cached = {"existing"}
    assert verifier.verify_prediction(*fixture, HELPER, cached) is False
    assert cached == {"existing"}
    changed = deepcopy(fixture[0]); changed["responses"]["shared_edge"]["result"] = "changed"
    with pytest.raises(ValueError, match="fallback changed"):
        verifier.verify_prediction(changed, *fixture[1:], HELPER, cached)


def outcome(row_id, top1, top5=True, protected=True):
    return {"id": row_id, "top1": top1, "top5": top5, "protected_exact": protected}


def metric(top1, top5, rows):
    return {"counts": {"top1": top1, "top5": top5}, "outcomes": rows}


def test_gate_is_strict_against_control_and_runtime_and_control_never_selects():
    base = [outcome("a", True), outcome("b", False), outcome("c", False)]
    shared = [outcome("a", True), outcome("b", True), outcome("c", False)]
    metrics = {"captured_runtime": metric(1, 3, base), "parent": metric(0, 3, base),
               "source_local_control": metric(1, 3, base), "shared_edge": metric(2, 3, shared)}
    decision = verifier.decide(metrics)
    assert decision["eligible_for_1000_case_regression"] is True
    assert decision["selected_for_1000_case_regression"] == "shared_edge"
    assert decision["historical_control_eligible"] is False
    metrics["shared_edge"]["counts"]["top1"] = 1
    assert verifier.decide(metrics)["selected_for_1000_case_regression"] is None


def resource(stage, operation, elapsed, operation_elapsed, swap_growth=0):
    return {"at": "time", "elapsed_s": elapsed, "rss_bytes": 100,
            "swap_used_bytes": 10 + swap_growth, "swap_growth_bytes": swap_growth,
            "operation_elapsed_s": operation_elapsed, "stage": stage, "operation": operation}


def test_resources_require_cache_dependent_operation_order_and_allow_postfinal_idle(tmp_path):
    prediction = supported_fixture()[0]; operations = verifier.expected_operations([prediction])
    rows, elapsed = [], 0.
    for operation in operations:
        elapsed += .1; rows.append(resource("operation_start", operation, elapsed, 0.))
        elapsed += .1; rows.append(resource("operation_end", operation, elapsed, .1))
    elapsed += .1; final = resource("final", None, elapsed, None, -2); rows.append(final)
    elapsed += .1; rows.append(resource("watchdog", None, elapsed, None, -1))
    path = tmp_path / "resources.jsonl"; write_jsonl(path, rows)
    report = {"baseline_swap_bytes": 10, "final_resources": final, "elapsed_s": elapsed + .1}
    result = verifier.verify_resources(report, [prediction], path)
    assert result["operation_pairs"] == len(operations)
    changed = deepcopy(rows); changed[2]["operation"] = "wrong"; write_jsonl(path, changed)
    with pytest.raises(ValueError, match="order"):
        verifier.verify_resources(report, [prediction], path)


def test_verifier_does_not_import_evaluator_torch_models_or_old_gold_report():
    source = SCRIPT.read_text()
    assert "import torch" not in source
    assert "from scripts.evaluate_shared_edge_residual" not in source
    assert "from hanglish" not in source
    assert 'CONTROL_DEV / "report.json"' not in source


def test_alternate_run_is_not_authorized(tmp_path):
    with pytest.raises(ValueError, match="alternate"):
        verifier.verify(tmp_path)
