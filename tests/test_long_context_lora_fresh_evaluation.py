from copy import deepcopy
from types import SimpleNamespace
import json
import time

import pytest

from scripts import evaluate_long_context_lora_fresh as module
from scripts import long_context_lora_evaluation as selection


class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        return [11, 12] if tokenize else "prompt"

    def encode(self, value, **kwargs):
        return [11, 12] + ([] if value == "prompt" else [100 + int(value[-1])])


def captured(index, role):
    source = "imnida"
    def candidate(text):
        return {"text": text, "score": 1.0, "segments": [{"source": source,
            "text": text, "kind": "lexicon", "input_start": 0, "input_end": len(source)}]}
    natural = role == "fresh_natural"
    return {"role": role, "stratum": "natural" if natural else "preservation", "row": {
        "id": f"{role}-{index}", "group_id": str(min(index, 386)), "input": source,
        "context": "문맥", "result": "임니다", "expected": ["입니다" if natural else "임니다"],
        "candidates": [candidate("임니다"), candidate("입니다")],
        "raw_bypass_response": {"result": source, "candidates": [candidate(source)]}}}


@pytest.fixture
def cohort(monkeypatch):
    monkeypatch.setattr(selection, "system_message", lambda: {"role": "system", "content": "select"})
    items = [captured(i, "fresh_natural") for i in range(512)]
    items += [captured(i, "challenge_preservation") for i in range(376)]
    records = selection.prepare_diagnostic(items, Tokenizer())
    predictions = [selection.apply_logits(record, [float(text == "입니다") for _, text in record["choices"]]
                    if record["role"] == "fresh_natural" else None) for record in records]
    return items, records, predictions


def test_full_fresh_gate_requires_gain_ci_and_preservation(cohort):
    items, _, predictions = cohort
    result = module.score_fresh(items, predictions)
    assert result["gate"]["passed"]
    assert result["bootstrap"]["groups"] == 387
    assert result["bootstrap"]["corrections"] == 512
    assert result["suites"]["challenge_preservation"]["metrics"]["all"]["top1"] == 376


def test_one_preservation_regression_rejects_even_large_fresh_gain(cohort):
    items, records, predictions = cohort
    record = records[-1]
    predictions[-1] = selection.apply_logits(record, [float(text == "입니다") for _, text in record["choices"]])
    result = module.score_fresh(items, predictions)
    assert result["bootstrap"]["gain_and_interval_passed"]
    assert not result["gate"]["passed"]
    assert not result["gate"]["preservation_all_correct"]


def test_no_gain_cannot_pass_on_preservation_alone(cohort):
    items, records, _ = cohort
    predictions = [selection.apply_logits(record, None) for record in records]
    result = module.score_fresh(items, predictions)
    assert result["gate"]["preservation_all_correct"]
    assert not result["gate"]["passed"]


def test_dropped_rows_and_changed_objects_fail(cohort):
    items, _, predictions = cohort
    with pytest.raises(ValueError, match="denominator"):
        module.score_fresh(items[:-1], predictions[:-1])
    predictions = deepcopy(predictions)
    predictions[0]["candidates"][0]["score"] = 1000
    with pytest.raises(ValueError, match="invariant"):
        module.score_fresh(items, predictions)


def test_pin_conflicts_fail():
    with pytest.raises(ValueError, match="conflict"):
        module.merge_pins({"a": "old"}, {"a": "new"})


def test_training_rejection_precedes_any_fresh_capture_read(monkeypatch, tmp_path):
    from scripts import prepare_long_context_lora_fresh_capture as capture
    def rejected(*args):
        raise ValueError("TRAIN gate rejected")
    def forbidden(*args):
        raise AssertionError("fresh data read before accepted TRAIN gate")
    monkeypatch.setattr(capture, "validate_training_proof", rejected)
    monkeypatch.setattr(module, "load_capture", forbidden)
    args = SimpleNamespace(output=tmp_path / "out", training_proof=tmp_path / "proof.json",
        training_proof_sha256="0" * 64, capture=tmp_path / "capture", capture_manifest_sha256="1" * 64)
    with pytest.raises(ValueError, match="TRAIN gate rejected"):
        module.run(args)
    assert not args.output.exists()


def test_failed_cleanup_records_errors_without_replacing_original_failure(tmp_path):
    class Backend:
        def synchronize(self):
            raise RuntimeError("sync failed")
    def checkpoint(stage):
        assert stage == "failure_after_model_release"
        raise RuntimeError("resource still above limit")
    report = {"completed": False, "failure": "original scoring failure"}
    path = tmp_path / "failed.json"
    module.finalize_failure(report, path, time.monotonic(), Backend(), checkpoint)
    saved = json.loads(path.read_text())
    assert saved["failure"] == "original scoring failure"
    assert len(saved["cleanup_errors"]) == 2
    assert not saved.get("synchronized_after_failure", False)


def test_failure_observation_precedes_report_serialization(tmp_path):
    report = {"completed": False, "failure": "original"}
    path = tmp_path / "failed.json"
    class Backend:
        def synchronize(self):
            assert not path.exists()
    def checkpoint(stage):
        assert not path.exists()
        return {"elapsed_seconds": 1.0}
    module.finalize_failure(report, path, time.monotonic(), Backend(), checkpoint)
    assert json.loads(path.read_text())["synchronized_after_failure"]


@pytest.mark.parametrize("key,value", [
    ("training_proof_sha256", "wrong"), ("model_artifact_hashes", {"wrong": "model"}),
    ("deployed", True), ("fresh_labels_read", False), ("model_inference_run", False),
    ("preservation_reused_without_new_parent_prediction", False),
    ("full_candidate_objects_preserved", False), ("raw_bypass_preserved", False),
    ("source_hashes", {}), ("source_hashes", {"../outside": "0" * 64}),
    ("source_hashes", {"file": "bad-digest"}),
])
def test_contradictory_manifest_rejected_before_target_payload_read(monkeypatch, tmp_path, key, value):
    from scripts import prepare_long_context_lora_fresh_capture as capture
    runtime = tmp_path / "runtime.json"
    runtime.write_text(json.dumps({"accepted_weight_hashes": {"model": "accepted"}}))
    monkeypatch.setattr(capture, "RUNTIME_REPORT", runtime)
    monkeypatch.setattr(capture, "RUNTIME_REPORT_SHA256", module.sha(runtime))
    training_report = tmp_path / "training.json"
    training_report.write_text("{}")
    directory = tmp_path / "capture"
    directory.mkdir()
    manifest = {"schema": "long-context-lora-fresh-capture-v1", "completed": True,
        "terminal_status": "success", "runtime_version": "neural-1-867f13596ed5",
        "rows": 888, "natural_rows": 512, "natural_groups": 387, "preservation_rows": 376,
        "training_report_sha256": module.sha(training_report), "training_proof_sha256": "proof",
        "model_artifact_hashes": {"model": "accepted"}, "capture_profile": capture.PROFILE,
        "deployed": False, "model_inference_run": True, "fresh_labels_read": True,
        "preservation_labels_read": True, "preservation_reused_without_new_parent_prediction": True,
        "full_candidate_objects_preserved": True, "raw_bypass_preserved": True,
        "source_hashes": {}, "source_hashes_after": {},
        "file_hashes": {"items.jsonl": "missing", "selection.jsonl": "missing"}}
    manifest[key] = value
    if key == "source_hashes":
        manifest["source_hashes_after"] = value
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="contract or training identity|inventory"):
        module.load_capture(directory, module.sha(path), {"report_path": training_report}, "proof")
