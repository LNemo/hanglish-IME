import json
from pathlib import Path

import pytest

from scripts import evaluate_extended_runtime as runtime


def candidate(source, text, score=1.0):
    return {"text": text, "score": score,
            "segments": [{"input_start": 0, "input_end": len(source),
                          "source": source, "text": text, "kind": "neural"}]}


def response(source, texts, pool=None):
    candidates = [candidate(source, text, index) for index, text in enumerate(texts)]
    candidates[-1]["score"] = None; candidates[-1]["segments"][0]["kind"] = "raw"
    pool = pool or candidates
    return {"input": source, "context": "", "external_context_used": False,
            "result": texts[0], "candidates": candidates, "engine_ms": 1.0,
            "pipeline": {"source_only_generation": True, "source_truncated": False,
                         "pool": pool, "response_candidates": [
                             {"index": index, "candidate_id": f"c{index}",
                              "candidate": value}
                             for index, value in enumerate(candidates)]},
            "extended_sentence_ranking": {"applied": True,
                "identity": {"runtime_version": "v1", "parent_version": "p1"}}}


def test_validate_reference_report_is_fail_closed():
    expected = runtime.REFERENCE["dev"]
    valid = {"completed": True, "split": "dev", "input_contract": "roman_sentence_only",
             "gold_not_passed_to_predictor": True, "capture_sha256": expected["capture"],
             "proposed": {"counts": {"rows": 316, "top1": 262,
                                      "top5": 282, "errors": 0}},
             "pins": {"a": "b"}}
    runtime.validate_reference_report(valid, "dev")
    for mutation in (
        {"completed": False}, {"split": "test"}, {"capture_sha256": "forged"},
        {"gold_not_passed_to_predictor": False}, {"pins": {}},
        {"proposed": {"counts": {"rows": 315, "top1": 262,
                                  "top5": 282, "errors": 0}}},
    ):
        changed = {**valid, **mutation}
        with pytest.raises(ValueError):
            runtime.validate_reference_report(changed, "dev")


def test_source_reference_rejects_labels_and_duplicate_ids():
    row = {"id": "r1", "request": {"input": "imnida test", "context": ""},
           "proposed": {"input": "imnida test", "context": "",
                        "candidates": [{"text": "입니다 테스트"}]}}
    seen = set()
    assert runtime.validate_source_reference(row, seen)[:2] == ("r1", "imnida test")
    with pytest.raises(ValueError):
        runtime.validate_source_reference(row, seen)
    with pytest.raises(ValueError):
        runtime.validate_source_reference({**row, "id": "r2",
                                           "request": {**row["request"], "expected": ["입니다"]}}, set())


def test_runtime_response_requires_typed_pipeline_and_final_raw():
    source = "imnida test"
    value = response(source, ["입니다 테스트", source])
    assert runtime.validate_runtime_response(source, value)["applied"] is True
    bad = json.loads(json.dumps(value)); bad.pop("pipeline")
    with pytest.raises(ValueError):
        runtime.validate_runtime_response(source, bad)


def test_nonapplied_parent_may_keep_scored_original_first():
    source = "test API2"
    raw_first = candidate(source, source, score=4.0)
    converted = candidate(source, "테스트 API2", score=3.0)
    value = {"input": source, "context": "", "external_context_used": False,
             "result": source, "candidates": [raw_first, converted],
             "pipeline": {"source_only_generation": True, "source_truncated": False,
                          "pool": [raw_first, converted], "response_candidates": [
                              {"index": 0, "candidate_id": "raw", "candidate": raw_first},
                              {"index": 1, "candidate_id": "converted", "candidate": converted}]},
             "extended_sentence_ranking": {"applied": False}}
    assert runtime.validate_runtime_response(source, value)["applied"] is False
    value["extended_sentence_ranking"]["applied"] = True
    with pytest.raises(ValueError):
        runtime.validate_runtime_response(source, value)
    bad = json.loads(json.dumps(value)); bad["candidates"].reverse()
    with pytest.raises(ValueError):
        runtime.validate_runtime_response(source, bad)


def test_extended_identity_binds_loaded_models_and_sources(monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", Path("/repo"))
    monkeypatch.setattr(runtime, "EXTENDED_MODULE", Path("/repo/hanglish/core/extended.py"))
    monkeypatch.setattr(runtime, "SEPARATED_MODULE", Path("/repo/hanglish/core/separated.py"))
    models = {"/repo/checkpoint/model": "m", "/repo/hanglish/core/extended.py": "e",
              "/repo/hanglish/core/separated.py": "s"}
    pins = dict(models)
    extended = {"schema_version": "extended-separated-ranking-v1", "identity": {
        "experimental": True, "policy": "expanded_pool_256",
        "shared_original_model_and_tokenizer": True, "tuned_max_length": 96,
        "runtime_max_length": 256, "max_position_embeddings": 512,
        "model_artifacts": {"checkpoint/model": "m",
                            "hanglish/core/extended.py": "e",
                            "hanglish/core/separated.py": "s"},
        "artifacts": {"hanglish/core/extended.py": "e",
                      "hanglish/core/separated.py": "s"}}}
    assert runtime.validate_extended_identity(extended, models, pins)["policy"] == "expanded_pool_256"
    extended["identity"]["runtime_max_length"] = 96
    with pytest.raises(ValueError):
        runtime.validate_extended_identity(extended, models, pins)


def test_extended_identity_allows_lazy_wrapper_only_before_application(monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", Path("/repo"))
    monkeypatch.setattr(runtime, "EXTENDED_MODULE", Path("/repo/extended.py"))
    monkeypatch.setattr(runtime, "SEPARATED_MODULE", Path("/repo/separated.py"))
    models = {"/repo/model": "m"}
    pins = {"/repo/extended.py": "e", "/repo/separated.py": "s"}
    identity = {"experimental": True, "policy": "expanded_pool_256",
                "shared_original_model_and_tokenizer": False,
                "tuned_max_length": 96, "runtime_max_length": 256,
                "max_position_embeddings": 512,
                "model_artifacts": {"model": "m"},
                "artifacts": {"extended.py": "e", "separated.py": "s"}}
    preserved = {"schema_version": "extended-separated-ranking-v1",
                 "applied": False, "reason": "parent_sentence_ranking_already_applied",
                 "identity": identity}
    assert runtime.validate_extended_identity(preserved, models, pins) is identity
    applied = {**preserved, "applied": True,
               "reason": "scored_full_nonraw_pool_at_256"}
    with pytest.raises(ValueError):
        runtime.validate_extended_identity(applied, models, pins)


def test_offline_comparison_is_exact_including_candidate_score():
    source = "imnida test"; value = response(source, ["입니다 테스트", source])
    row = {"id": "r1", "request": {"input": source, "context": ""}, "proposed": value}
    offline = {"id": "r1", "request": row["request"],
               "policies": {"expanded_pool_256": {"response": value}}}
    runtime.compare_offline(row, offline)
    changed = json.loads(json.dumps(offline))
    changed["policies"]["expanded_pool_256"]["response"]["candidates"][0]["score"] += .001
    with pytest.raises(ValueError):
        runtime.compare_offline(row, changed)


def test_metrics_bind_ids_sources_pool_and_literal_regressions():
    source = "API2 testreul sihaeng"
    parent = response(source, ["API2 테스트를 실행", source])
    proposed = response(source, ["에이피아이 테스트를 시행", source],
                        pool=[{"text": "API2 테스트를 시행"}])
    rows = [runtime.compact_record({"id": "r1", "request": {"input": source, "context": ""},
                                    "parent": parent, "proposed": proposed})]
    labels = [{"id": "r1", "input": source, "expected": ["API2 테스트를 시행"],
               "category": "literal"}]
    metrics = runtime.compare_metrics(rows, labels)
    assert metrics["full_pool_coverage"] == {"proposed": 1}
    assert metrics["literal_signature_regressions"] == ["r1"]
    with pytest.raises(ValueError):
        runtime.compare_metrics(rows, [{**labels[0], "id": "other"}])
    with pytest.raises(ValueError):
        runtime.compare_metrics(rows, [{**labels[0], "input": "other source"}])


def test_quality_gate_needs_gain_without_any_loss():
    source = "imnida test"
    parent = response(source, ["임니다 테스트", source])
    proposed = response(source, ["입니다 테스트", source])
    rows = [runtime.compact_record({"id": "r1", "request": {"input": source, "context": ""},
                                    "parent": parent, "proposed": proposed})]
    labels = [{"id": "r1", "input": source, "expected": ["입니다 테스트"]}]
    metrics = runtime.compare_metrics(rows, labels)
    assert metrics["corrections"] == ["r1"] and metrics["quality_gate"] is True


def test_verify_pins_respects_only_explicit_deferred_path(tmp_path):
    first = tmp_path / "first"; second = tmp_path / "second"
    first.write_text("a"); second.write_text("b")
    pins = {str(first): runtime.sha256_file(first), str(second): "wrong"}
    runtime.verify_pins(pins, defer=second)
    with pytest.raises(ValueError):
        runtime.verify_pins(pins, defer=first)


def test_qualification_rejects_unverified_or_code_drift(tmp_path, monkeypatch):
    run = tmp_path / "run"; run.mkdir()
    capture = run / "capture.jsonl"; capture.write_text("{}\n")
    closure = run / "capture.complete.json"; closure.write_text("{}\n")
    current = {runtime.path_key(Path(runtime.__file__)): "eval",
               runtime.path_key(runtime.EXTENDED_MODULE): "adapter",
               runtime.path_key(runtime.SEPARATED_MODULE): "separated",
               runtime.path_key(runtime.PLAN): "plan",
               runtime.path_key(runtime.VERIFIER): "verifier", "/model": "model"}
    current.update({runtime.path_key(path): f"helper-{index}"
                    for index, path in enumerate(runtime.SHARED_HELPERS)})
    report = {"schema": runtime.SCHEMA, "completed": True, "split": "dev",
              "quality_gate": True, "offline_exact_parity_rows": 316,
              "regressions": [], "top5_losses": [], "literal_signature_regressions": [],
              "capture_sha256": runtime.sha256_file(capture),
              "closure_sha256": runtime.sha256_file(closure),
              "pins": dict(current), "model_pins": {"/model": "model"}}
    report_path = run / "report.json"; report_path.write_text(json.dumps(report))
    proof = tmp_path / "proof.json"
    identity = {"report_sha256": runtime.sha256_file(report_path),
                "capture_sha256": report["capture_sha256"],
                "closure_sha256": report["closure_sha256"],
                "evaluator_sha256": "eval", "module_sha256": "adapter",
                "separated_sha256": "separated", "plan_sha256": "plan",
                "verifier_sha256": "verifier", "model_pins": {"/model": "model"}}
    proof.write_text(json.dumps({"schema": "extended-separated-runtime-independent-verification-v1",
        "verified": True, "split": "dev", "identity": identity,
        "contracts": {"model_loaded": False, "model_inference": False,
                      "capture_closed_before_target_join": True},
        "decision": {"quality_gate": True, "regressions": [], "top5_losses": [],
                     "literal_signature_regressions": [], "offline_exact_parity_rows": 316}}))
    monkeypatch.setattr(runtime, "sha256_file",
                        lambda path: {report_path: identity["report_sha256"],
                                      capture: identity["capture_sha256"],
                                      closure: identity["closure_sha256"],
                                      proof: "proof",
                                      runtime.VERIFIER: "verifier"}.get(Path(path), "unused"))
    assert runtime.qualification_identity(run, proof, current)["proof_sha256"] == "proof"
    report["pins"][runtime.path_key(runtime.EXTENDED_MODULE)] = "drift"
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        runtime.qualification_identity(run, proof, current)


def test_self_test_has_no_model_dependency():
    assert runtime.self_test() == {"schema": runtime.SCHEMA, "self_test": True}
