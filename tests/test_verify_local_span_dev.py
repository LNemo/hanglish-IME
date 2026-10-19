from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_local_span_dev.py"
SPEC = importlib.util.spec_from_file_location("verify_local_span_dev", SCRIPT)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)
HELPER = verifier._helper()


def write_jsonl(path, values):
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))


def candidate(source="a", text="아"):
    return {"text": text, "score": .1, "segments": [{
        "input_start": 0, "input_end": len(source), "source": source, "text": text}]}


def supported_fixture(source="a"):
    value = candidate(source)
    original = {"result": value["text"], "candidates": [deepcopy(value)],
                "pipeline": {"pool": [deepcopy(value)]}}
    runtime = {"id": "row", "request": {"input": source, "context": ""},
               "elapsed_ms": 1.5, "proposed": original}
    proposal = {"id": "row", "request": deepcopy(runtime["request"]),
                "edge_inventory": [], "proposals": []}
    texts, reps, groups, path_ids = HELPER.reconstruct_domain(runtime, proposal)
    scores = {"parent": [.1], "source_local_control": [.2], "local_aux": [.3]}
    responses = {"captured_runtime": HELPER.compact_response(original)}
    hashes = {"captured_runtime": HELPER.canonical_hash(original)}
    for arm in verifier.MODELS:
        full = HELPER.apply_policy(original, reps, texts, scores[arm], source, arm)
        responses[arm] = HELPER.compact_response(full)
        hashes[arm] = HELPER.canonical_hash(full)
    saved = {"id": "row", "request": deepcopy(runtime["request"]),
             "candidate_texts": texts, "path_ids": path_ids,
             "support": {"supported": True, "reason": None}, "scores": scores,
             "latency_ms": {"captured_runtime": 1.5, "parent_features": .4,
                            "source_local_control": .2, "local_aux": .3},
             "responses": responses, "full_response_sha256": hashes,
             "protected_paths": HELPER.protected_paths(source, groups),
             "protected_responses": {arm: HELPER.protected_response(source, response)
                                     for arm, response in responses.items()}}
    historical = {"id": "row", "request": deepcopy(runtime["request"]),
                  "candidate_texts": texts, "path_ids": path_ids,
                  "support": deepcopy(saved["support"]),
                  "scores": {"parent": [.1], "source_local": [.2]},
                  "responses": {"source_local": deepcopy(responses["source_local_control"])},
                  "full_response_sha256": {
                      "source_local": hashes["source_local_control"]}}
    return saved, runtime, proposal, historical


def fallback_fixture():
    source = "raw"
    raw = HELPER.raw_candidate(source)
    original = {"result": source, "candidates": [raw], "pipeline": {"pool": [raw]}}
    runtime = {"id": "fallback", "request": {"input": source, "context": ""},
               "elapsed_ms": 2., "proposed": original}
    proposal = {"id": "fallback", "request": deepcopy(runtime["request"]),
                "edge_inventory": [], "proposals": []}
    responses = {arm: HELPER.compact_response(original) for arm in verifier.ARMS}
    hashes = {arm: HELPER.canonical_hash(original) for arm in verifier.ARMS}
    saved = {"id": "fallback", "request": deepcopy(runtime["request"]),
             "candidate_texts": [], "path_ids": [],
             "support": {"supported": False, "reason": "empty_nonraw_pool"},
             "scores": {}, "latency_ms": {"captured_runtime": 2., "parent_features": 0.,
                  "source_local_control": 0., "local_aux": 0.},
             "responses": responses, "full_response_sha256": hashes,
             "protected_paths": HELPER.protected_paths(source, []),
             "protected_responses": {arm: HELPER.protected_response(source, response)
                                     for arm, response in responses.items()}}
    historical = {"id": "fallback", "request": deepcopy(runtime["request"]),
        "candidate_texts": [], "path_ids": [], "support": deepcopy(saved["support"]),
        "scores": {}, "responses": {"source_local": responses["source_local_control"]},
        "full_response_sha256": {"source_local": hashes["source_local_control"]}}
    return saved, runtime, proposal, historical


def test_supported_prediction_replays_domain_scores_policy_and_control():
    saved, runtime, proposal, historical = supported_fixture()
    assert verifier.verify_prediction(saved, runtime, proposal, historical, HELPER) is True


@pytest.mark.parametrize("mutation,match", [
    ("source", "source"), ("domain", "domain"), ("score", "ranked response"),
    ("rank", "ranked response"), ("control", "control"), ("protected", "protected")])
def test_prediction_rejects_source_domain_scores_rank_control_and_protection(
        mutation, match):
    saved, runtime, proposal, historical = supported_fixture()
    if mutation == "source":
        saved["request"]["input"] = "b"
    elif mutation == "domain":
        saved["candidate_texts"] = ["조작"]
    elif mutation == "score":
        saved["scores"]["local_aux"][0] += 1
    elif mutation == "rank":
        saved["responses"]["local_aux"]["result"] = "조작"
    elif mutation == "control":
        saved["responses"]["source_local_control"]["result"] = "조작"
    else:
        saved["protected_paths"]["verified"] = False
    with pytest.raises(ValueError, match=match):
        verifier.verify_prediction(saved, runtime, proposal, historical, HELPER)


def test_fallback_requires_exact_captured_response_and_historical_control():
    saved, runtime, proposal, historical = fallback_fixture()
    assert verifier.verify_prediction(saved, runtime, proposal, historical, HELPER) is False
    changed = deepcopy(saved)
    changed["responses"]["local_aux"]["result"] = "changed"
    with pytest.raises(ValueError, match="fallback changed"):
        verifier.verify_prediction(changed, runtime, proposal, historical, HELPER)
    changed = deepcopy(saved)
    changed["responses"]["source_local_control"]["result"] = "changed"
    with pytest.raises(ValueError):
        verifier.verify_prediction(changed, runtime, proposal, historical, HELPER)


def outcome(row_id, top1, top5=True, protected=True):
    return {"id": row_id, "top1": top1, "top5": top5,
            "protected_exact": protected}


def metric(top1, top5, rows):
    return {"counts": {"top1": top1, "top5": top5}, "outcomes": rows}


def test_gate_is_strict_against_both_control_and_runtime_and_control_never_selects():
    ids = ["a", "b", "c"]
    runtime = [outcome("a", True), outcome("b", False), outcome("c", False)]
    control = [outcome("a", True), outcome("b", False), outcome("c", False)]
    aux = [outcome("a", True), outcome("b", True), outcome("c", False)]
    metrics = {"captured_runtime": metric(1, 3, runtime),
               "parent": metric(0, 3, runtime),
               "source_local_control": metric(1, 3, control),
               "local_aux": metric(2, 3, aux)}
    decision = verifier.decide(metrics)
    assert decision["eligible_for_1000_case_regression"] is True
    assert decision["selected_for_1000_case_regression"] == "local_aux"
    assert decision["historical_control_eligible"] is False
    metrics["local_aux"]["counts"]["top1"] = 1
    decision = verifier.decide(metrics)
    assert decision["eligible_for_1000_case_regression"] is False
    assert decision["selected_for_1000_case_regression"] is None


def resource(stage, operation, elapsed, operation_elapsed, swap_growth=0):
    return {"at": "time", "elapsed_s": elapsed, "rss_bytes": 100,
            "swap_used_bytes": 10 + swap_growth, "swap_growth_bytes": swap_growth,
            "operation_elapsed_s": operation_elapsed, "stage": stage,
            "operation": operation}


def test_resources_require_operation_order_and_terminal_final_sample(tmp_path):
    predictions = [{"id": "a", "support": {"supported": True}}]
    operations = verifier.expected_operations(predictions)
    rows, elapsed = [], 0.
    for operation in operations:
        elapsed += .1; rows.append(resource("operation_start", operation, elapsed, 0.))
        elapsed += .1; rows.append(resource("operation_end", operation, elapsed, .1))
    elapsed += .1; final = resource("final", None, elapsed, None, swap_growth=-2)
    rows.append(final)
    elapsed += .05; rows.append(resource("watchdog", None, elapsed, None, swap_growth=-1))
    path = tmp_path / "resources.jsonl"; write_jsonl(path, rows)
    report = {"baseline_swap_bytes": 10, "final_resources": final,
              "elapsed_s": elapsed + .1}
    result = verifier.verify_resources(report, predictions, path)
    assert result["operation_pairs"] == len(operations)
    assert result["peak_swap_growth_bytes"] == 0
    changed = deepcopy(rows); changed[-2]["stage"] = "watchdog"
    write_jsonl(path, changed)
    with pytest.raises(ValueError, match="final"):
        verifier.verify_resources(report, predictions, path)


def test_resource_tampered_order_and_bounds_fail(tmp_path):
    predictions = [{"id": "a", "support": {"supported": False}}]
    operations = verifier.expected_operations(predictions)
    rows = [resource("operation_start", operations[0], .1, 0.),
            resource("operation_end", operations[0], .2, .1),
            resource("final", None, .3, None)]
    path = tmp_path / "resources.jsonl"
    report = {"baseline_swap_bytes": 10, "final_resources": rows[-1], "elapsed_s": .4}
    changed = deepcopy(rows); changed[0]["operation"] = "wrong"; write_jsonl(path, changed)
    with pytest.raises(ValueError, match="order"):
        verifier.verify_resources(report, predictions, path)
    changed = deepcopy(rows); changed[1]["rss_bytes"] = verifier.EVAL_LIMITS["rss_bytes"] + 1
    write_jsonl(path, changed)
    with pytest.raises(ValueError, match="bound"):
        verifier.verify_resources(report, predictions, path)


def test_verifier_does_not_import_evaluator_torch_or_models_or_name_old_report():
    source = SCRIPT.read_text()
    assert "import torch" not in source
    assert "from scripts.evaluate_local_span_objective" not in source
    assert "from hanglish" not in source
    assert "CONTROL_DEV / \"report.json\"" not in source


def test_alternate_run_is_not_authorized(tmp_path):
    with pytest.raises(ValueError, match="alternate"):
        verifier.verify(tmp_path)
