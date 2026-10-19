import json
from pathlib import Path

import pytest

import scripts.evaluate_source_span_selected_test as selected
from scripts.evaluate_source_span_pair import collect_candidate_groups, validate_training_artifacts
from scripts.preflight_source_span_residual import sha


def segment(source, start, end, text):
    return {"input_start": start, "input_end": end, "source": source[start:end],
            "text": text, "kind": "mock"}


def pool_candidate(source, identifier, parts):
    segments = [segment(source, start, end, text) for start, end, text in parts]
    return {"candidate_id": identifier, "path_id": identifier,
            "text": "".join(row["text"] for row in segments),
            "origin": "mock", "origins": ["mock"], "segments": segments}


def source_row():
    source = "abc"
    pool = [
        pool_candidate(source, "path-a", [(0, 1, "가"), (1, 3, "bc")]),
        pool_candidate(source, "path-b", [(0, 1, "a"), (1, 2, "나"), (2, 3, "c")]),
        pool_candidate(source, "path-c", [(0, 2, "ab"), (2, 3, "다")]),
    ]
    public = [{"text": row["text"], "score": 0.0,
               "segments": row["segments"]} for row in pool]
    response = {"input": source, "context": "", "result": public[0]["text"],
                "candidates": public, "pipeline": {"pool": pool}}
    return {"id": "row-1", "request": {"input": source, "context": ""},
            "elapsed_ms": 2.0, "proposed": response}


def test_fresh_proposal_and_prediction_capture_all_domain_identities():
    runtime = source_row()
    proposal = selected.make_proposal_row(runtime)
    assert proposal["id"] == runtime["id"]
    assert proposal["schema_version"] == "source-edge-recombination-v1"
    assert proposal["diagnostics"]["score_fields_consumed"] is False
    texts, representatives, groups = collect_candidate_groups(runtime, proposal)
    latency = {"captured_runtime": 2.0, "source_edge_proposal": 1.0,
               "parent_features": 3.0, selected.SELECTED_ARM: 4.0}
    prediction = selected.build_prediction(runtime, proposal, texts, representatives,
        groups, [float(index) for index in range(len(texts))], None, latency)
    assert prediction["candidate_texts"] == texts
    assert len(prediction["candidate_ids"]) == len(texts)
    assert all(prediction["path_ids"])
    assert prediction["proposal_ids"] == [row["proposal_id"]
                                           for row in proposal["proposals"]]
    assert prediction["responses"][selected.SELECTED_ARM]["candidates"][-1]["text"] == "abc"


def test_unsupported_row_preserves_exact_captured_response():
    runtime = source_row()
    proposal = selected.make_proposal_row(runtime)
    texts, representatives, groups = collect_candidate_groups(runtime, proposal)
    latency = {"captured_runtime": 2.0, "source_edge_proposal": 1.0,
               "parent_features": 0.0, selected.SELECTED_ARM: 0.0}
    prediction = selected.build_prediction(runtime, proposal, texts, representatives,
        groups, None, "source_exceeds_2048_bytes", latency)
    assert prediction["support"] == {"supported": False,
                                     "reason": "source_exceeds_2048_bytes"}
    assert prediction["responses"][selected.SELECTED_ARM] \
        == prediction["responses"]["captured_runtime"]


def metric(outcomes, top1, top5, protected=1000):
    return {"counts": {"rows": 1000, "errors": 0, "top1": top1, "top5": top5,
                       "protected_exact": protected}, "outcomes": outcomes}


def test_fixed_test_gate_requires_strict_gain_without_any_losses():
    baseline = []
    improved = []
    for index in range(1000):
        before_top1 = index < 953
        before_top5 = index < 972
        baseline.append({"id": str(index), "top1": before_top1,
                         "top5": before_top5, "protected_exact": True})
        improved.append({"id": str(index), "top1": before_top1 or index == 953,
                         "top5": before_top5, "protected_exact": True})
    metrics = {"captured_runtime": metric(baseline, 953, 972),
               selected.SELECTED_ARM: metric(improved, 954, 972)}
    decision = selected.decide(metrics)
    assert decision["offline_1000_case_gate"] is True
    assert decision["production_qualified"] is False

    improved[0] = {**improved[0], "top1": False}
    metrics[selected.SELECTED_ARM] = metric(improved, 953, 972)
    decision = selected.decide(metrics)
    assert decision["offline_1000_case_gate"] is False
    assert decision["versus_captured_runtime"]["top1_loss_ids"] == ["0"]


def test_response_protection_metric_is_independent_of_unused_pool_paths(monkeypatch):
    monkeypatch.setattr(selected, "EXPECTED_ROWS", 1)
    row = {"id": "one", "request": {"input": "abc", "context": ""},
           "candidate_texts": ["가"], "support": {"supported": True},
           "responses": {selected.SELECTED_ARM: {
               "candidates": [{"text": "가"}]}},
           "protected_paths": {"verified": False},
           "protected_responses": {selected.SELECTED_ARM: {"verified": True}},
           "latency_ms": {"source_edge_proposal": 1.0, "parent_features": 2.0,
                          selected.SELECTED_ARM: 3.0}}
    result = selected.summarize([row], [{"id": "one", "input": "abc",
        "expected": ["가"], "category": "mock"}], selected.SELECTED_ARM)
    assert result["counts"]["protected_exact"] == 1


def test_selected_checkpoint_is_bound_to_dev_evaluated_training(tmp_path, monkeypatch):
    training_report = tmp_path / "report.json"
    residual = tmp_path / "residual.safetensors"
    training_report.write_text('{"run": 1}')
    residual.write_bytes(b"selected residual")
    monkeypatch.setattr(selected, "SELECTED_RESIDUAL_SHA256", sha(residual))
    auth = {"report_path": training_report,
            "checkpoints": {selected.SELECTED_ARM: {"residual_path": residual}}}
    dev_report = {"training_report_sha256": sha(training_report)}
    assert selected.validate_selected_training_binding(dev_report, auth)[
        "residual_sha256"] == sha(residual)
    training_report.write_text('{"run": 2}')
    with pytest.raises(ValueError, match="not bound"):
        selected.validate_selected_training_binding(dev_report, auth)


def test_reused_dev_dependency_must_match_frozen_dev_pin(tmp_path):
    dependency = tmp_path / "policy.py"
    dependency.write_text("frozen = True\n")
    report = {"pins": {str(dependency): sha(dependency)}}
    assert selected.validate_dev_reliance_pins(report, [dependency]) \
        == {str(dependency): sha(dependency)}
    dependency.write_text("frozen = False\n")
    with pytest.raises(ValueError, match="dependency differs"):
        selected.validate_dev_reliance_pins(report, [dependency])


def test_frozen_authorities_validate_before_model_or_target_load():
    dev = selected.validate_dev_selection(selected.DEFAULT_DEV)
    training = validate_training_artifacts(selected.DEFAULT_TRAINING)
    runtime = selected.validate_runtime_artifacts(selected.DEFAULT_RUNTIME)
    binding = selected.validate_selected_training_binding(dev["report"], training)
    assert binding["residual_sha256"] == selected.SELECTED_RESIDUAL_SHA256
    assert runtime["target_path"].name == "test.jsonl"
    assert runtime["target_sha256"] == selected.TEST_SHA256


def test_runtime_auth_hashes_but_never_parses_historical_labeled_report(monkeypatch):
    report_path = selected.DEFAULT_RUNTIME / "report.json"
    original = Path.read_text

    def guarded_read_text(path, *args, **kwargs):
        if path == report_path:
            raise AssertionError("historical TEST report must not be parsed")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    assert selected.validate_runtime_artifacts(selected.DEFAULT_RUNTIME)[
        "target_sha256"] == selected.TEST_SHA256


def test_runtime_document_contract_rejects_unverified_proof():
    directory = selected.DEFAULT_RUNTIME
    proof = json.loads((directory / "independent-verification.json").read_text())
    closure = json.loads((directory / "capture.complete.json").read_text())
    pins = json.loads((directory / "pins.json").read_text())
    proof["verified"] = False
    with pytest.raises(ValueError, match="proof differs"):
        selected.validate_runtime_documents(proof, closure, pins)
