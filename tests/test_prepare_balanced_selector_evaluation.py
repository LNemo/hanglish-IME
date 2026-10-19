import hashlib
import json
from types import SimpleNamespace

import pytest

import scripts.prepare_balanced_selector_evaluation as module


def args(tmp_path):
    return SimpleNamespace(
        selector_report=tmp_path / "selector/report.json",
        selector_report_sha256="s" * 64,
        evaluation_partition=tmp_path / "eval.json",
        evaluation_partition_sha256=module.EVALUATION_PARTITION_SHA256,
        runtime_evidence=tmp_path / "runtime.json",
        runtime_evidence_sha256=module.RUNTIME_EVIDENCE_SHA256,
        output=tmp_path / "output",
    )


def test_selector_failure_prevents_raw_source_and_converter_access(tmp_path, monkeypatch):
    values = args(tmp_path)
    values.evaluation_partition.write_text("{}")
    raw = values.evaluation_partition.read_bytes()
    monkeypatch.setattr(module, "EVALUATION_PARTITION_SHA256",
                        hashlib.sha256(raw).hexdigest())
    values.evaluation_partition_sha256 = module.EVALUATION_PARTITION_SHA256
    monkeypatch.setattr(module, "read_evaluation_partition", lambda *args: (raw, {}))
    runtime = {"passed": True, "integrated_runtime_version": module.RUNTIME_VERSION,
               "accepted_weight_hashes": {str(i): "a" * 64 for i in range(16)},
               "integrated_runtime_source_hashes": {}, "runtime_source_hashes_after": {}}
    calls = []

    def fail_selector(*unused):
        calls.append("selector")
        raise ValueError("selector gate")

    with pytest.raises(ValueError, match="selector gate"):
        module.prepare(
            values, runtime_loader=lambda *unused: (b"runtime", runtime),
            selector_loader=fail_selector,
            source_loader=lambda *unused: (_ for _ in ()).throw(
                AssertionError("raw source read")),
            converter_factory=lambda **unused: (_ for _ in ()).throw(
                AssertionError("model created")),
        )
    assert calls == ["selector"]


def test_schedule_serialization_rejects_changed_or_missing_structure(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    report_path = tmp_path / "report.json"
    (checkpoint / "metadata.json").write_text("{}")
    (tmp_path / "schedule.jsonl").write_text("{}\n")
    report = {"experiment_schema": "wrong"}
    raw = json.dumps(report).encode()
    report_path.write_bytes(raw)
    monkeypatch.setattr(module, "SELECTOR_REPORT_SHA256",
                        hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="completed frozen"):
        module.load_balanced_selector_artifact(
            report_path, hashlib.sha256(raw).hexdigest(),
            {"integrated_runtime_version": module.RUNTIME_VERSION,
             "accepted_weight_hashes": {}, "integrated_runtime_source_hashes": {}}
        )


def test_frozen_actual_selector_artifact_passes_model_free_preflight():
    runtime_path = module.ROOT / ".omx/artifacts/numeric-structure-integration-v1/report.json"
    runtime_raw = runtime_path.read_bytes()
    assert hashlib.sha256(runtime_raw).hexdigest() == module.RUNTIME_EVIDENCE_SHA256
    runtime = json.loads(runtime_raw)
    report_path = module.ROOT / ".omx/artifacts/balanced-final-choice-selector-v2/report.json"
    artifact = module.load_balanced_selector_artifact(
        report_path, module.SELECTOR_REPORT_SHA256, runtime,
    )
    assert artifact["report"]["checkpoint_sha256"] == module.SELECTOR_CHECKPOINT_SHA256
    assert len(artifact["report"]["schedule_ids"]) == module.EXPECTED_BATCHES


def test_rehashed_unapproved_selector_report_is_rejected_before_artifact_read(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text("{}")
    with pytest.raises(ValueError, match="frozen successful"):
        module.load_balanced_selector_artifact(
            report_path, hashlib.sha256(report_path.read_bytes()).hexdigest(), {},
        )


def test_source_selection_reproduces_exact_order_and_constructor_fields(tmp_path, monkeypatch):
    rows = [
        {"id": "a", "group_id": "g", "text": "가", "left_context": "앞", "split": "train"},
        {"id": "b", "group_id": "g", "text": "나", "left_context": "", "split": "train"},
    ]
    raw = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    old = {"natural": {}}
    old_raw = json.dumps(old).encode()
    plan_raw = b"plan"
    raw_path, partition_path, plan_path = tmp_path / "raw", tmp_path / "partition", tmp_path / "plan"
    raw_path.write_bytes(raw); partition_path.write_bytes(old_raw); plan_path.write_bytes(plan_raw)
    monkeypatch.setattr(module, "RAW", raw_path)
    monkeypatch.setattr(module, "PARTITION_V1", partition_path)
    monkeypatch.setattr(module, "PLAN", plan_path)
    monkeypatch.setattr(module, "RAW_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(module, "PARTITION_V1_SHA256", hashlib.sha256(old_raw).hexdigest())
    monkeypatch.setattr(module, "PLAN_SHA256", hashlib.sha256(plan_raw).hexdigest())
    monkeypatch.setattr(module, "select_ids", lambda metadata, natural: {
        "eval_ids": ["b", "a"], "eval_group_ids": ["g"],
        "remaining_ids": [], "remaining_group_ids": []})
    sources, _ = module.read_evaluation_sources({
        "eval_ids": ["b", "a"], "eval_group_ids": ["g"],
        "remaining_ids": [], "remaining_group_ids": []})
    assert sources == [
        {"id": "b", "group_id": "g", "origin": "natural_eval_v2",
         "input": "na", "context": "", "expected": ["나"]},
        {"id": "a", "group_id": "g", "origin": "natural_eval_v2",
         "input": "ga", "context": "앞", "expected": ["가"]},
    ]


class LabelGuard(dict):
    def __getitem__(self, key):
        if key == "expected":
            raise AssertionError("label read before parent conversion")
        return super().__getitem__(key)


def test_capture_seam_receives_source_without_runner_reading_label():
    source = LabelGuard(id="id", input="ga", context="", expected=["가"])
    observed = []

    def capture(converter, value, **kwargs):
        observed.append((converter, value, kwargs))
        return {"id": value["id"], "input": value["input"]}

    result = capture("converter", source, expected_runtime_version="runtime",
                     expected_model_artifacts={})
    assert result == {"id": "id", "input": "ga"}
    assert observed[0][1] is source


def test_runtime_hash_mismatch_fails_before_selector(tmp_path):
    values = args(tmp_path)
    values.runtime_evidence_sha256 = "0" * 64
    with pytest.raises(ValueError, match="accepted numeric"):
        module.prepare(values, runtime_loader=lambda *args: (_ for _ in ()).throw(
            AssertionError("runtime loader called")))
