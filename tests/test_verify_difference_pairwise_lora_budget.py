from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import struct

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/verify_difference_pairwise_lora_budget.py"
SPEC = importlib.util.spec_from_file_location("pairwise_budget_verifier", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def report():
    return MODULE.read_json(MODULE.REPORT)


def prepared():
    manifest, tokens = MODULE._validate_data()
    reps, schedule, stages = MODULE._expected_probe(tokens)
    return manifest, reps, schedule, stages


def test_real_artifacts_qualify_without_loading_mlx():
    payload = MODULE.verify_budget()
    assert payload["passed"] is True
    assert payload["launch_allowed"] is True
    assert payload["old_policy"] == {
        "limit_seconds": 7200.0,
        "estimate_seconds": MODULE.EXPECTED_ESTIMATE_SECONDS,
        "launch_allowed": False,
    }
    assert payload["new_policy"] == {
        "limit_seconds": 21600.0,
        "estimate_seconds": MODULE.EXPECTED_ESTIMATE_SECONDS,
        "launch_allowed": True,
    }
    assert payload["counts"] == {
        "qualified_pairs": 1928, "probe_units": 20, "probe_components": 76,
        "probe_updates": 5, "probe_inferences": 40,
        "resource_observations": 156, "adapter_tensors": 16,
        "adapter_scalars": 327680,
    }
    assert payload["source_inventory"]["count"] == 48
    assert len(payload["source_inventory"]["hashes"]) == 48
    assert payload["source_inventory"]["sha256"] == MODULE.digest_value(
        payload["source_inventory"]["hashes"])
    assert payload["training_started"] is False
    assert payload["training_quality_claimed"] is False
    assert payload["deployment_authorized"] is False
    assert payload["second_fit_authorized"] is False


def test_only_exact_reviewed_budget_change_qualifies():
    result = MODULE.qualify_budget(MODULE.EXPECTED_ESTIMATE_SECONDS)
    assert result["old_policy"]["launch_allowed"] is False
    assert result["new_policy"]["launch_allowed"] is True
    with pytest.raises(MODULE.VerificationError, match="only the reviewed"):
        MODULE.qualify_budget(MODULE.EXPECTED_ESTIMATE_SECONDS, new_limit=21601.0)
    with pytest.raises(MODULE.VerificationError, match="formula"):
        MODULE.qualify_budget(MODULE.EXPECTED_ESTIMATE_SECONDS - 1)


def test_safetensors_reconstructs_actual_mlx_dtype_hash_domain():
    raw_report = report()
    parsed = MODULE.read_safetensors(ROOT / raw_report["adapter"]["path"])
    assert parsed["hashes"] == raw_report["saved_trainable_tensor_hashes"]
    path = ROOT / raw_report["adapter"]["path"]
    raw = path.read_bytes()
    header_size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + header_size])
    body = raw[8 + header_size:]
    buggy = {}
    for name in MODULE.EXPECTED_NAMES:
        item = header[name]
        start, end = item["data_offsets"]
        domain = f"float32:{tuple(item['shape'])}:".encode()
        buggy[name] = MODULE.hashlib.sha256(domain + body[start:end]).hexdigest()
    assert all(buggy[name] != parsed["hashes"][name] for name in MODULE.EXPECTED_NAMES)


def test_adapter_body_corruption_rejects_reported_tensor_hashes(tmp_path):
    raw_report = report()
    resource = tmp_path / "resource"
    adapter_dir = resource / "adapter"
    adapter_dir.mkdir(parents=True)
    tensor_path = adapter_dir / "adapters.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    shutil.copyfile(ROOT / raw_report["adapter"]["path"], tensor_path)
    shutil.copyfile(ROOT / raw_report["adapter"]["config_path"], config_path)
    data = bytearray(tensor_path.read_bytes())
    data[-1] ^= 1
    tensor_path.write_bytes(data)
    altered = copy.deepcopy(raw_report)
    altered["adapter"] = {
        "path": str(tensor_path), "sha256": MODULE.file_digest(tensor_path),
        "config_path": str(config_path), "config_sha256": MODULE.file_digest(config_path),
    }
    with pytest.raises(MODULE.VerificationError, match="MLX-domain tensor hashes"):
        MODULE._validate_adapter(altered, resource)


def test_safetensors_shape_corruption_rejects(tmp_path):
    source = ROOT / report()["adapter"]["path"]
    raw = source.read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + size])
    name = MODULE.EXPECTED_NAMES[0]
    header[name]["shape"][0] += 1
    encoded = json.dumps(header, separators=(",", ":")).encode()
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + size:])
    with pytest.raises(MODULE.VerificationError, match="metadata differs"):
        MODULE.read_safetensors(broken)


def test_component_count_and_weight_corruption_rejects():
    _, reps, schedule, _ = prepared()
    altered = copy.deepcopy(report())
    altered["units"][0]["components"].pop()
    with pytest.raises(MODULE.VerificationError, match="component count"):
        MODULE._validate_measurements(altered, schedule, reps)
    altered = copy.deepcopy(report())
    altered["units"][0]["components"][0]["weight"] = .4
    with pytest.raises(MODULE.VerificationError, match="component measurement"):
        MODULE._validate_measurements(altered, schedule, reps)


def test_inclusive_update_interval_corruption_rejects():
    _, reps, schedule, _ = prepared()
    altered = copy.deepcopy(report())
    altered["updates"][0]["complete_update_seconds"] = 0.01
    with pytest.raises(MODULE.VerificationError, match="inclusive interval"):
        MODULE._validate_measurements(altered, schedule, reps)


def test_finalization_interval_corruption_rejects():
    _, reps, schedule, _ = prepared()
    altered = copy.deepcopy(report())
    altered["finalization_interval"]["end_elapsed"] = altered["elapsed_seconds"] + 1
    with pytest.raises(MODULE.VerificationError, match="finalization interval"):
        MODULE._validate_measurements(altered, schedule, reps)


def test_resource_observation_count_and_time_corruption_rejects():
    _, _, _, stages = prepared()
    altered = copy.deepcopy(report())
    altered["resource_observations"].pop()
    with pytest.raises(MODULE.VerificationError, match="count or order"):
        MODULE._validate_resources(altered, stages)
    altered = copy.deepcopy(report())
    altered["resource_observations"][1]["pre"]["elapsed_seconds"] = 0.0
    with pytest.raises(MODULE.VerificationError, match="moved backwards"):
        MODULE._validate_resources(altered, stages)


def test_source_mutation_is_rechecked(monkeypatch):
    target = "scripts/difference_pairwise_lora_model.py"
    original = MODULE.file_digest

    def changed(path):
        if Path(path).resolve() == (ROOT / target).resolve():
            return "0" * 64
        return original(path)

    monkeypatch.setattr(MODULE, "file_digest", changed)
    with pytest.raises(MODULE.VerificationError, match="source changed"):
        MODULE.verify_budget()


def test_failure_audit_payload_mutation_rejects_independently():
    raw_report = report()
    manifest, reps, schedule, stages = prepared()
    del manifest
    timing, _ = MODULE._validate_measurements(raw_report, schedule, reps)
    adapter = MODULE._validate_adapter(raw_report, MODULE.REPORT.parent)
    resource_summary, _ = MODULE._validate_resources(raw_report, stages)
    audit = MODULE.read_json(MODULE.AUDIT)
    audit["payload"]["launch_gate"]["training_started"] = True
    audit["digest"] = MODULE.digest_value(audit["payload"])
    with pytest.raises(MODULE.VerificationError, match="historical 7,200"):
        MODULE._validate_audit(audit, raw_report, adapter, timing, resource_summary)


def test_proof_is_canonical_hash_bound_and_exclusive(tmp_path):
    payload = MODULE.verify_budget()
    output = tmp_path / "proof.json"
    envelope = MODULE.write_proof(output, payload)
    assert output.read_bytes() == MODULE.canonical_bytes(envelope) + b"\n"
    assert envelope["digest"] == MODULE.digest_value(payload)
    with pytest.raises(MODULE.VerificationError, match="already exists"):
        MODULE.write_proof(output, payload)


def test_cli_rejects_alternate_report_before_writing_proof(tmp_path):
    output = tmp_path / "must-not-exist.json"
    code = MODULE.main([
        "--resource-report", str(tmp_path / "other.json"),
        "--failure-audit", str(MODULE.AUDIT),
        "--output", str(output),
    ])
    assert code == 1
    assert not output.exists()


def test_verifier_has_no_forbidden_validation_imports_or_model_dependency():
    source = SOURCE.read_text()
    assert "import mlx" not in source
    assert "verify_difference_pairwise_lora_resources import" not in source
    assert "difference-pairwise-lora-resource-v2-failure-audit.py import" not in source
    assert "probe_difference_pairwise_lora_resources import" not in source
