import ast
import copy
import json
from pathlib import Path

import pytest

from scripts import verify_document_phonetic_shards_evaluation_v2 as verifier


def response(*texts, ranked=None, latency=None):
    value = {"result": texts[0],
             "candidates": [{"text": text} for text in texts]}
    if ranked is not None:
        value["sentence_ranking_applied"] = ranked
    if latency is not None:
        value["engine_ms"] = latency
    return value


def record(identifier, source, baseline, proposed, *, pool=(), ranked=True,
           latency=1.0):
    return {
        "id": identifier, "request": {"input": source, "context": ""},
        "baseline": response(*baseline),
        "proposed": response(*proposed, ranked=ranked, latency=latency),
        "base_paths": [[text, 0.0] for text in pool],
    }


def test_v2_imports_no_evaluator_model_or_tensor_module():
    tree = ast.parse(Path(verifier.__file__).read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = ("evaluate_document_phonetic_shards_v2",
                 "production_phonetic_choice_converter", "torch")
    assert not any(name == item or name.endswith("." + item)
                   for name in imports for item in forbidden)


def test_all_frozen_v2_source_identities_match_current_files():
    assert verifier.pure.sha(verifier.EVALUATOR) == verifier.EVALUATOR_SHA256
    assert verifier.pure.sha(verifier.METRIC_HELPER) == verifier.METRIC_HELPER_SHA256
    assert verifier.pure.sha(verifier.CAPTURE_PRODUCER) \
        == verifier.CAPTURE_PRODUCER_SHA256
    assert verifier.pure.sha(verifier.RUNTIME_VERIFIER) \
        == verifier.RUNTIME_VERIFIER_SHA256
    assert verifier.pure.sha(verifier.AUDIT_VERIFIERS[1]) \
        == verifier.AUDIT_VERIFIER_SHA256[1]


def test_report_scope_requires_mixed_audits_and_frozen_v1_runtime():
    report = {
        "schema": verifier.REPORT_SCHEMA, "completed": True,
        "rows": 1046, "groups": 821,
        "producer_sha256": verifier.EVALUATOR_SHA256,
        "reference_sha256": verifier.RAW_SHA256,
        "selection_sha256": verifier.SELECTION_SHA256,
        "audit_verifier_sha256_by_shard": {
            "0": verifier.AUDIT_VERIFIER_SHA256[0],
            "1": verifier.AUDIT_VERIFIER_SHA256[1]},
        "runtime_verifier_sha256": verifier.RUNTIME_VERIFIER_SHA256,
        "labels_opened_after_both_verified_captures": True,
        "exact_original_reference_strings": True,
        "production_changed": False,
        "baseline_is_reconstructed_public_output": True,
        "baseline_latency_measured": False, "fully_blind_generalization": False,
    }
    verifier.validate_report_scope(report)
    for field in ("audit_verifier_sha256_by_shard", "runtime_verifier_sha256"):
        damaged = copy.deepcopy(report)
        damaged[field] = {}
        with pytest.raises(ValueError, match="identity/scope"):
            verifier.validate_report_scope(damaged)


def test_proof0_v1_proof1_v2_but_both_runtime_pins_v1():
    report = {"runtime_verifier_sha256": verifier.RUNTIME_VERIFIER_SHA256}
    pins = {
        str(verifier.RUNTIME_VERIFIER.resolve()): verifier.RUNTIME_VERIFIER_SHA256,
        str(verifier.CAPTURE_PRODUCER.resolve()): verifier.CAPTURE_PRODUCER_SHA256,
    }
    shard_report = {"pins": pins}
    common = {
        "schema": verifier.SHARD_PROOF_SCHEMA, "verified": True,
        "producer_sha256": verifier.CAPTURE_PRODUCER_SHA256,
        "partition_hashes": verifier.PARTITION_HASHES,
        "performance_labels_opened": False, "accuracy_measured": False,
    }
    proof0 = {**common, "shard": 0, "rows": 520,
              "verifier_sha256": verifier.AUDIT_VERIFIER_SHA256[0]}
    verifier.validate_audit_identity(0, proof0, shard_report, report)
    proof1 = {**common, "shard": 1, "rows": 526,
              "verifier_sha256": verifier.AUDIT_VERIFIER_SHA256[1],
              "runtime_verifier_path": str(verifier.RUNTIME_VERIFIER.resolve()),
              "runtime_verifier_sha256": verifier.RUNTIME_VERIFIER_SHA256}
    verifier.validate_audit_identity(1, proof1, shard_report, report)

    wrong = dict(proof1, verifier_sha256=verifier.AUDIT_VERIFIER_SHA256[0])
    with pytest.raises(ValueError, match="mixed shard audit"):
        verifier.validate_audit_identity(1, wrong, shard_report, report)
    wrong = dict(proof1, runtime_verifier_sha256=verifier.AUDIT_VERIFIER_SHA256[1])
    with pytest.raises(ValueError, match="distinction"):
        verifier.validate_audit_identity(1, wrong, shard_report, report)


def test_metrics_remain_exact_and_independent_of_audit_version():
    rows = [
        record("keep", "yuji", ["유지"], ["유지"], pool=["유지"], latency=9),
        record("fix", "gochida", ["오답", "고치다"], ["고치다"],
               pool=["고치다"], latency=1),
        record("missing", "source", ["오답"], ["오답"], pool=["오답"], latency=2),
        record("unranked", "source2", ["오답"], ["오답"],
               pool=[], ranked=False, latency=3),
    ]
    labels = {"keep": "유지", "fix": "고치다",
              "missing": "정답", "unranked": "정답2"}
    metrics = verifier.aggregate(rows, labels)
    assert metrics["baseline"] == {"rows": 4, "top1": 1, "top5": 2}
    assert metrics["proposed"] == {"rows": 4, "top1": 2, "top5": 2}
    assert metrics["corrections"] == 1 and metrics["quality_gate"] is True
    assert metrics["stage_counts"] == {
        "ranked_rows": 3, "ranked_pool_gold": 2,
        "ranked_candidate_missing": 1, "ranked_selection_failures": 0,
        "unranked_rows": 1, "unranked_first_failures": 1,
    }
    assert metrics["actual_calibrated_latency"] == {
        "p50_ms": 2, "p95_ms": 9, "max_ms": 9}


def test_exact_text_ascii_literal_and_top5_regressions_still_close_gate():
    rows = [
        record("fix", "gochida", ["오답"], ["고치다"], pool=["고치다"]),
        record("loss", "jeongdap", ["오답", "정답 "], ["오답"],
               pool=["정답 ", "오답"]),
        record("copy", "testreul A-1!", ["test를 A-1!"], ["테스트"],
               pool=["test를 A-1!", "테스트"]),
    ]
    metrics = verifier.aggregate(
        rows, {"fix": "고치다", "loss": "정답 ", "copy": "test를 A-1!"})
    assert metrics["corrections"] == 1
    assert metrics["top5_losses"] == 2
    assert metrics["english_ascii_signature"]["signature_regressions"] == 1
    assert metrics["english_ascii_signature"]["literal_signature_regressions"] == 1
    assert metrics["quality_gate"] is False
    exact = verifier.aggregate(
        [record("a", "source", ["정답"], ["정답"], pool=["정답"])],
        {"a": "정답 "})
    assert exact["proposed"]["top1"] == 0


def test_verify_authenticates_before_any_reference_read_and_emits_aggregates_only(
        tmp_path, monkeypatch):
    calls = []
    records = [record("a", "source", ["오답"], ["가"], pool=["가"])]
    metrics = verifier.aggregate(records, {"a": "가"})
    report = {"version": "v", "weights": [1.0], **metrics}

    def authenticate(path):
        calls.append("authenticate")
        return report, records, {"a": "source"}, {"mixed": "proofs"}

    def selection(path, rows):
        assert calls == ["authenticate"]
        calls.append("selection")
        return [{"id": "a", "group_id": "g"}]

    def raw(path):
        assert calls == ["authenticate", "selection"]
        calls.append("raw")
        return iter([{"id": "a", "group_id": "g", "text": "가"}])

    def labels(selected, references, requests, **kwargs):
        assert calls == ["authenticate", "selection", "raw"]
        calls.append("labels")
        return {"a": "가"}

    monkeypatch.setattr(verifier, "authenticate_report", authenticate)
    monkeypatch.setattr(verifier.pure, "read_exact_jsonl", selection)
    monkeypatch.setattr(verifier.pure, "jsonl", raw)
    monkeypatch.setattr(verifier, "join_labels", labels)
    monkeypatch.setattr(verifier.pure, "sha", lambda path: "verifier-v2")
    output = tmp_path / "proof.json"
    proof = verifier.verify(tmp_path / "report.json", output)
    assert calls == ["authenticate", "selection", "raw", "labels"]
    assert "records" not in proof and "failures" not in proof
    assert json.loads(output.read_text())["metrics"] == metrics
