import hashlib
import json
from pathlib import Path

import pytest

from scripts import prepare_long_context_lora_fresh_capture as capture


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_training_proof_gate_precedes_sources_and_requires_gain(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "ROOT", tmp_path)
    source = tmp_path / "source"
    source.write_text("fixed")
    report_path = tmp_path / "report.json"
    report = {"schema": "long-context-lora-train-v1", "completed": True,
              "terminal_status": "success", "training_gate_passed": True,
              "train_gate": {"passed": True, "required_net_gain": 5, "net_gain": 5},
              "source_hashes": {f"source-{i}": "0" * 64 for i in range(137)}}
    report["source_hashes_after"] = dict(report["source_hashes"])
    report_path.write_text(json.dumps(report))
    verifier = tmp_path / "verify.py"
    verifier.write_text("fixed verifier")
    proof_path = tmp_path / "proof.json"
    proof = {"passed": True, "training_gate_passed": True,
             "verifier_sha256": digest(verifier), "source_hashes_checked": 137,
             "training_records": 2771, "updates": 693, "final_update_microbatches": 3,
             "diagnostic_rows": 640, "reload_probes": 4,
             "model_loaded_by_verifier": False, "fresh_labels_read": False,
             "development_final_labels_read": False, "train_labels_read": True,
             "net_top1_gain": 5,
             "report_sha256": digest(report_path)}
    proof_path.write_text(json.dumps(proof))
    monkeypatch.setattr(capture, "TRAINING_REPORT", report_path)
    monkeypatch.setattr(capture, "TRAINING_VERIFIER", verifier)
    monkeypatch.setattr(capture, "TRAINING_VERIFIER_SHA256", digest(verifier))
    monkeypatch.setattr(capture, "verify_source_hashes", lambda values: None)
    result = capture.validate_training_proof(proof_path, digest(proof_path))
    assert result["report"] == report
    report["train_gate"]["net_gain"] = 4
    report_path.write_text(json.dumps(report))
    proof["report_sha256"] = digest(report_path)
    proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="TRAIN gate"):
        capture.validate_training_proof(proof_path, digest(proof_path))


def test_training_proof_rejected_before_source_verification(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "ROOT", tmp_path)
    proof = tmp_path / "proof.json"
    proof.write_text("{}")
    touched = []
    monkeypatch.setattr(capture, "verify_source_hashes", lambda values: touched.append(True))
    with pytest.raises(ValueError, match="proof"):
        capture.validate_training_proof(proof, "0" * 64)
    assert touched == []


def raw(identity, group, text="한글"):
    return {"id": identity, "group_id": group, "split": "train", "text": text,
            "left_context": "문맥"}


def test_fresh_source_construction_revalidates_whole_groups(monkeypatch):
    monkeypatch.setattr(capture, "FRESH_ROWS", 3)
    monkeypatch.setattr(capture, "FRESH_GROUPS", 2)
    selection = [{"id": "a", "group_id": "g1"}, {"id": "b", "group_id": "g1"},
                 {"id": "c", "group_id": "g2"}]
    rows = [raw("a", "g1"), raw("b", "g1", "문장"), raw("c", "g2")]
    result = capture.select_fresh_sources(selection, rows, lambda text: f"roman:{text}")
    assert result[0] == {"id": "a", "group_id": "g1", "origin": "fresh_natural",
                         "input": "roman:한글", "context": "문맥", "expected": ["한글"]}
    with pytest.raises(ValueError, match="complete groups"):
        capture.select_fresh_sources(selection, rows + [raw("d", "g1")], lambda text: text)


def test_wrappers_preserve_full_row_objects(monkeypatch):
    monkeypatch.setattr(capture, "FRESH_ROWS", 1)
    monkeypatch.setattr(capture, "PRESERVATION_ROWS", 1)
    fresh = {"id": "a", "group_id": "g", "candidates": [{"text": "A"}],
             "raw_bypass_response": {"result": "a"}}
    preservation = {"id": "b", "candidates": [{"text": "B"}],
                    "raw_bypass_response": {"result": "b"}}
    assert capture.wrap_items([fresh], [preservation]) == [
        {"role": "fresh_natural", "stratum": "natural", "row": fresh},
        {"role": "challenge_preservation", "stratum": "preservation",
         "row": preservation},
    ]


def test_resource_caps_and_existing_output(monkeypatch, tmp_path):
    monkeypatch.setattr(capture, "rss_bytes", lambda: capture.PROFILE["rss_limit_bytes"] + 1)
    with pytest.raises(capture.CaptureResourceError, match="3GiB") as raised:
        capture.require_resources(0, "fixture")
    assert raised.value.record["stage"] == "fixture"
    assert raised.value.record["rss_peak_bytes"] == capture.PROFILE["rss_limit_bytes"] + 1
    output = tmp_path / "existing"
    output.mkdir()
    args = type("Args", (), {"output": output, "training_proof": tmp_path / "none",
                             "training_proof_sha256": "0" * 64})()
    with pytest.raises(FileExistsError):
        capture.run(args)


def test_time_breach_retains_exact_resource_observation(monkeypatch):
    monkeypatch.setattr(capture, "rss_bytes", lambda: 123)
    monkeypatch.setattr(capture.time, "monotonic",
                        lambda: capture.PROFILE["time_limit_seconds"] + 0.5)
    with pytest.raises(capture.CaptureResourceError, match="1800") as raised:
        capture.require_resources(0, "timeout")
    assert raised.value.record == {"stage": "timeout", "elapsed_seconds": 1800.5,
                                   "rss_peak_bytes": 123}


def test_failure_report_write_error_does_not_replace_original(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(Path, "write_text",
                        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))

    def operation():
        try:
            raise ValueError("original capture failure")
        finally:
            capture.write_failed_report(tmp_path / "manifest.json", {"completed": False})

    with pytest.raises(ValueError, match="original capture failure"):
        operation()
    assert "disk full" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["fresh_labels_read", "preservation_labels_read"])
def test_target_access_is_recorded_before_reader_failure(flag):
    report = {flag: False}

    def fail():
        raise ValueError("synthetic target read failure")

    with pytest.raises(ValueError, match="synthetic target"):
        capture.access_target(report, flag, fail)
    assert report[flag] is True


def test_module_has_no_eager_model_or_torch_imports():
    source = Path(capture.__file__).read_text()
    prefix = source[:source.index("def run(")]
    assert "import torch" not in prefix
    assert "NeuralConverter" not in prefix
    run_source = source[source.index("def run("):]
    assert run_source.index("validate_training_proof(") < run_source.index(
        'access_target(report, "fresh_labels_read"')
    assert "load_selector" not in source
    assert "selector_gate" not in source
