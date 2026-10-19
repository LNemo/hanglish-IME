from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import train_long_context_lora_v2 as trainer


HASH = "a" * 64


def write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return trainer.sha256_file(path)


def proof_fixture(tmp_path):
    report_path = tmp_path / "report.json"
    verifier_path = tmp_path / "verify.py"
    runner_path = tmp_path / "runner.py"
    verifier_path.write_text("# verifier\n")
    runner_path.write_text("# runner\n")
    projection = {
        "update_p95_seconds": 10.0,
        "probe_scoring_p95_seconds": 2.0,
        "formula": "1.5*(693*update_p95+1280*probe_scoring_p95)+300",
        "estimated_training_seconds": 14535.0,
        "within_21600_seconds": True,
    }
    report = {
        "schema": "long-context-lora-resource-checkpoint-v2",
        "completed": True, "resource_feasible": True, "terminal_status": "success",
        "real_fit_rows_read": False, "real_training_run": False,
        "runtime_estimate": projection,
    }
    report_sha = write_json(report_path, report)
    proof = {
        "passed": True, "report_sha256": report_sha,
        "verifier_sha256": trainer.sha256_file(verifier_path),
        "runner_sha256": trainer.sha256_file(runner_path),
        "helper_sha256": trainer.CHECKPOINT_HELPER_SHA256,
        "actual_fit_training_run": False, "source_files_checked": 137,
        "main_updates": 20, "restart_branches": 3,
        "restart_parity_exact": True,
        "original_global_swap_baseline_preserved": True,
        "runtime_projection": projection,
        "retained_payload_pairs_byte_verified": 8,
        "historical_pruned_payload_bytes_independently_verified": False,
    }
    proof_path = tmp_path / "proof.json"
    proof_sha = write_json(proof_path, proof)
    return proof_path, proof_sha, report_path, verifier_path, runner_path, proof


def validate_fixture(paths):
    proof_path, proof_sha, report, verifier, runner, _ = paths
    return trainer.validate_resource_proof(
        proof_path, proof_sha, report_path=report, verifier_path=verifier,
        runner_path=runner, helper_path=trainer.CHECKPOINT_HELPER)


def test_resource_proof_binds_explicit_hash_and_current_sources(tmp_path):
    paths = proof_fixture(tmp_path)
    result = validate_fixture(paths)
    assert result["proof"]["passed"] is True
    assert result["helper_sha256"] == trainer.CHECKPOINT_HELPER_SHA256
    with pytest.raises(trainer.TrainingV2Error, match="explicit pin"):
        trainer.validate_resource_proof(
            paths[0], "0" * 64, report_path=paths[2], verifier_path=paths[3],
            runner_path=paths[4], helper_path=trainer.CHECKPOINT_HELPER)


@pytest.mark.parametrize("field,value", [
    ("passed", False), ("actual_fit_training_run", True),
    ("main_updates", 19), ("restart_branches", 2),
    ("restart_parity_exact", False),
    ("original_global_swap_baseline_preserved", False),
    ("source_files_checked", True),
])
def test_resource_proof_gate_rejects_nonpassing_claims(tmp_path, field, value):
    paths = proof_fixture(tmp_path)
    proof = copy.deepcopy(paths[-1])
    proof[field] = value
    proof_sha = write_json(paths[0], proof)
    with pytest.raises(trainer.TrainingV2Error, match="did not pass"):
        trainer.validate_resource_proof(
            paths[0], proof_sha, report_path=paths[2], verifier_path=paths[3],
            runner_path=paths[4], helper_path=trainer.CHECKPOINT_HELPER)


def test_resource_proof_rejects_runtime_over_21600(tmp_path):
    paths = proof_fixture(tmp_path)
    proof = copy.deepcopy(paths[-1])
    proof["runtime_projection"]["estimated_training_seconds"] = 21600.01
    proof["runtime_projection"]["within_21600_seconds"] = False
    proof_sha = write_json(paths[0], proof)
    with pytest.raises(trainer.TrainingV2Error, match="did not pass"):
        trainer.validate_resource_proof(
            paths[0], proof_sha, report_path=paths[2], verifier_path=paths[3],
            runner_path=paths[4], helper_path=trainer.CHECKPOINT_HELPER)


@pytest.mark.parametrize("source_index", [2, 3, 4])
def test_resource_proof_rejects_bound_report_verifier_or_runner_drift(tmp_path, source_index):
    paths = proof_fixture(tmp_path)
    paths[source_index].write_text("changed after proof\n")
    with pytest.raises(trainer.TrainingV2Error, match="does not bind"):
        validate_fixture(paths)


def make_data():
    records, batches = [], []
    cursor = 0
    for update in range(1, 694):
        batch = []
        count = 3 if update == 693 else 4
        for _ in range(count):
            cursor += 1
            entry = {"role": "natural_fit", "id": f"row-{cursor:04d}",
                     "microbatch": cursor, "update": update}
            batch.append(entry)
            records.append({"role": entry["role"], "id": entry["id"]})
        batches.append(batch)
    return {"batches": batches, "records": records,
            "manifest": {"file_hashes": {"schedule.jsonl": HASH}}}


def fit_run():
    return {"schedule": {"sha256": HASH, "microbatches": 2771, "updates": 693,
                         "accumulation": 4, "final_update_microbatches": 3}}


def checkpoint_item(data, update):
    batch = data["batches"][update - 1]
    next_index = min(update * 4, 2771)
    return {"payload": {
        "update": update, "batch_records": copy.deepcopy(batch),
        "next_expected": None if next_index == 2771 else {
            "role": data["records"][next_index]["role"],
            "id": data["records"][next_index]["id"]}}}


def test_schedule_binding_checks_all_2771_rows_and_checkpoint_identity():
    data = make_data()
    snapshot = {"manifests": [checkpoint_item(data, 1), checkpoint_item(data, 341),
                              checkpoint_item(data, 693)]}
    trainer.validate_schedule_binding(data, fit_run(), snapshot)
    assert len(data["records"]) == 2771 and len(data["batches"][-1]) == 3


def test_schedule_binding_rejects_manifest_batch_or_next_identity():
    data = make_data()
    item = checkpoint_item(data, 341)
    item["payload"]["batch_records"][0]["id"] = "wrong"
    with pytest.raises(trainer.TrainingV2Error, match="batch identities"):
        trainer.validate_schedule_binding(data, fit_run(), {"manifests": [item]})
    item = checkpoint_item(data, 341)
    item["payload"]["next_expected"]["id"] = "wrong"
    with pytest.raises(trainer.TrainingV2Error, match="next identity"):
        trainer.validate_schedule_binding(data, fit_run(), {"manifests": [item]})


def test_schedule_binding_rejects_row_reordering_and_non_three_tail():
    data = make_data()
    data["records"][0], data["records"][1] = data["records"][1], data["records"][0]
    with pytest.raises(trainer.TrainingV2Error, match="row order"):
        trainer.validate_schedule_binding(data, fit_run())


def test_schedule_only_binds_checkpoint_before_fit_targets(tmp_path):
    data = make_data()
    schedule = [entry for batch in data["batches"] for entry in batch]
    schedule_path = tmp_path / "schedule.jsonl"
    schedule_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in schedule))
    manifest = {"file_hashes": {"schedule.jsonl": trainer.sha256_file(schedule_path)}}
    snapshot = {"manifests": [checkpoint_item(data, 341)]}
    loaded = trainer.load_and_validate_schedule_only(
        manifest, {"schedule": {**fit_run()["schedule"],
                                "sha256": manifest["file_hashes"]["schedule.jsonl"]}},
        snapshot, data_dir=tmp_path)
    assert len(loaded["records"]) == 2771
    snapshot["manifests"][0]["payload"]["batch_records"][0]["id"] = "wrong"
    with pytest.raises(trainer.TrainingV2Error, match="batch identities"):
        trainer.load_and_validate_schedule_only(
            manifest, {"schedule": {**fit_run()["schedule"],
                                    "sha256": manifest["file_hashes"]["schedule.jsonl"]}},
            snapshot, data_dir=tmp_path)
    data = make_data()
    data["batches"][-1].append({"role": "natural_fit", "id": "extra",
                                "microbatch": 2772, "update": 693})
    with pytest.raises(trainer.TrainingV2Error, match="final tail"):
        trainer.validate_schedule_binding(data, fit_run())


def test_run_request_fails_before_mutation():
    trainer.validate_run_request("fresh", Path("/definitely/missing/v2"), "segment-0001")
    with pytest.raises(trainer.TrainingV2Error, match="segment-0001"):
        trainer.validate_run_request("fresh", Path("/missing"), "segment-0002")
    with pytest.raises(trainer.TrainingV2Error, match="existing v2 root"):
        trainer.validate_run_request("resume", Path("/missing"), "segment-0002")


def test_invalid_proof_prevents_fit_loader_and_model_import(tmp_path, monkeypatch):
    touched = {"fit": False, "system": False}

    def fit_loader():
        touched["fit"] = True
        raise AssertionError("FIT must remain unread")

    def system_reader():
        touched["system"] = True
        raise AssertionError("resource launch must not begin")

    monkeypatch.setattr(trainer.v1, "load_and_validate_inputs", fit_loader)
    monkeypatch.setattr(trainer.resource_v1, "system_resources", system_reader)
    with pytest.raises(Exception):
        trainer.run(mode="fresh", output=tmp_path / "out",
                    resource_proof=tmp_path / "missing-proof.json",
                    resource_proof_sha256="0" * 64, segment_id="segment-0001")
    assert touched == {"fit": False, "system": False}
    assert not (tmp_path / "out").exists()


def test_prior_close_proof_is_required_and_bounded(tmp_path):
    event = {"kind": "segment_terminal", "segment_id": "segment-0001",
             "status": "resource_failure", "terminal_segment_seconds": 101.0}
    report = {"schema": "long-context-lora-train-v2-segment",
              "segment_id": "segment-0001", "terminal_status": "resource_failure",
              "completed": False,
              "close_proof": {"reserve_seconds": 1.0,
                              "actual_close_seconds": 0.25,
                              "completed_within_reserve": True,
                              "charged_terminal_seconds": 101.0}}
    write_json(tmp_path / "segments/segment-0001/report.json", report)
    trainer.validate_prior_close_proofs(tmp_path, [event], "segment-0002")
    report["close_proof"]["actual_close_seconds"] = 1.01
    write_json(tmp_path / "segments/segment-0001/report.json", report)
    with pytest.raises(trainer.TrainingV2Error, match="close proof"):
        trainer.validate_prior_close_proofs(tmp_path, [event], "segment-0002")


def test_final_adapter_export_is_exact_checkpoint_693_copy(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter.safetensors").write_bytes(b"adapter-693")
    (checkpoint / "adapter_config.json").write_text(json.dumps(trainer.ADAPTER_CONFIG))
    restored = type("Restored", (), {
        "path": checkpoint,
        "manifest": {"files": {
            "adapter.safetensors": {"sha256": trainer.sha256_file(
                checkpoint / "adapter.safetensors")},
            "adapter_config.json": {"sha256": trainer.sha256_file(
                checkpoint / "adapter_config.json")},
        }},
    })()
    attempt = tmp_path / "finalization-attempt-0001"
    attempt.mkdir()
    adapter_dir, hashes = trainer._export_final_adapter(restored, attempt)
    assert (adapter_dir / "adapters.safetensors").read_bytes() == b"adapter-693"
    assert hashes["adapter_sha256"] == restored.manifest["files"][
        "adapter.safetensors"]["sha256"]


def test_resume_payload_validation_precedes_fit_target_loader_in_source():
    source = Path(trainer.__file__).read_text()
    resume_validation = source.index("restored = store.load_eligible(mx)")
    fit_read = source.index("data = v1.load_and_validate_inputs()")
    model_load = source.index("model, tokenizer, digit_ids, diagnostic =")
    assert resume_validation < fit_read < model_load
    assert 'if train_gate["passed"] is not True:' in source


def test_source_has_no_network_or_inline_resume_path():
    source = Path(trainer.__file__).read_text()
    assert "requests" not in source and "urllib" not in source
    assert "while retry" not in source and "auto_resume\": True" not in source
    assert "fresh_data_read_for_evaluation\": False" in source
