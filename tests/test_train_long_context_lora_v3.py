from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import train_long_context_lora_v3 as trainer


HASH = "a" * 64


def write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return trainer.sha256_file(path)


def qualification_fixture(tmp_path, monkeypatch):
    qualifier = tmp_path / "qualify_resources.py"
    qualifier.write_text("# reviewed qualifier fixture\n")
    qualifier_sha = trainer.sha256_file(qualifier)
    monkeypatch.setattr(trainer, "QUALIFIER_SHA256", qualifier_sha)
    runtime = {
        "update_p95_seconds": 20.0,
        "probe_scoring_p95_seconds": 1.0,
        "formula": "1.5*(693*update_p95+1280*probe_scoring_p95)+300",
        "estimated_training_seconds": 23894.80004841287,
        "within_21600_seconds": False,
        "update_timing_scope": "gradient plus checkpoint transaction and retention",
        "checkpoint_frequency": "every optimizer update",
    }
    report_path = tmp_path / "report.json"
    write_json(report_path, {
        "schema": "long-context-lora-resource-checkpoint-v3",
        "completed": False, "resource_feasible": False, "terminal_status": "failed",
        "failure_stage": "runtime_projection", "real_fit_rows_read": False,
        "real_training_run": False, "runtime_estimate": runtime,
    })
    artifact_paths = {
        "report": report_path,
        "process_audit": trainer.RESOURCE_AUDIT,
        "failure_audit": trainer.ROOT / ".omx/artifacts/long-context-lora-work-v3/failure-audit-v2.json",
        "parent_verifier": trainer.ROOT / ".omx/artifacts/long-context-lora-work-v3/verify_checkpoint_probe.py",
        "checkpoint_helper_v2": trainer.FROZEN_HELPER,
        "checkpoint_helper_v3": trainer.CHECKPOINT_HELPER,
        ".omx/context/long-context-lora-budget-v4.md": trainer.CONTEXT,
        ".omx/plans/long-context-lora-budget-v4.md": trainer.PLAN,
        ".omx/plans/long-context-lora-budget-v4-test-spec.md": trainer.TEST_SPEC,
        ".omx/plans/long-context-lora-budget-v4-architect-review.md": trainer.ARCHITECT_REVIEW,
        ".omx/plans/long-context-lora-budget-v4-critic-review.md": trainer.CRITIC_REVIEW,
        ".omx/plans/long-context-lora-budget-v4-handoff.json": trainer.HANDOFF,
    }
    qualification = {
        "schema": "long-context-lora-resource-qualification-v4",
        "qualification_passed": True, "qualified_for_one_fresh_fit": True,
        "historical_probe_passed": False, "historical_failure_stage": "runtime_projection",
        "historical_process_returncode": 1, "historical_budget_seconds": 21600.0,
        "qualified_budget_seconds": 28800.0,
        "observed_estimated_training_seconds": 23894.80004841287,
        "runtime_estimate": runtime,
        "reason": "reviewed one-fit policy v4 accepts the pinned estimate under 28800 seconds while preserving the failed 21600-second v3 result",
        "physical_limits_unchanged_and_passed": True,
        "parity_and_restart_evidence_passed": True,
        "checkpoint_helper_exact_two_literal_transform": True,
        "source_files_verified": 113, "resource_observations_verified": 497,
        "durable_snapshots_verified": 6, "stores_verified": 5,
        "retained_payload_pairs_byte_verified": 10,
        "real_fit_rows_read": False, "labels_read": False, "mlx_imported": False,
        "qwen_model_loaded": False, "actual_training_run": False,
        "artifact_sha256": {name: trainer.sha256_file(path) for name, path in artifact_paths.items()},
        "qualifier_sha256": qualifier_sha,
    }
    path = tmp_path / "qualification.json"
    sha = write_json(path, qualification)
    return path, sha, qualifier, report_path, qualification


def validate_fixture(fixture):
    path, sha, qualifier, report, _ = fixture
    return trainer.validate_resource_qualification(
        path, sha, qualifier_path=qualifier, report_path=report,
        helper_path=trainer.CHECKPOINT_HELPER)


def test_resource_qualification_binds_explicit_hash_sources_and_policy(tmp_path, monkeypatch):
    fixture = qualification_fixture(tmp_path, monkeypatch)
    result = validate_fixture(fixture)
    assert result["qualification"]["qualified_budget_seconds"] == 28800.0
    with pytest.raises(trainer.TrainingV2Error, match="explicit pin"):
        trainer.validate_resource_qualification(
            fixture[0], "0" * 64, qualifier_path=fixture[2], report_path=fixture[3])


@pytest.mark.parametrize("field,value", [
    ("qualification_passed", False), ("qualified_for_one_fresh_fit", False),
    ("historical_probe_passed", True), ("historical_failure_stage", "other"),
    ("qualified_budget_seconds", 28801.0),
    ("checkpoint_helper_exact_two_literal_transform", False),
    ("source_files_verified", 112), ("mlx_imported", True),
])
def test_resource_qualification_rejects_policy_or_boundary_drift(
        tmp_path, monkeypatch, field, value):
    fixture = qualification_fixture(tmp_path, monkeypatch)
    qualification = copy.deepcopy(fixture[-1])
    qualification[field] = value
    sha = write_json(fixture[0], qualification)
    with pytest.raises(trainer.TrainingV2Error, match="did not pass"):
        trainer.validate_resource_qualification(
            fixture[0], sha, qualifier_path=fixture[2], report_path=fixture[3])


def test_helper_successor_validation_rejects_any_third_change(tmp_path):
    frozen = trainer.FROZEN_HELPER.read_text()
    good = tmp_path / "good.py"; good.write_text(frozen.replace("21600.0", "28800.0"))
    trainer.validate_helper_successor(trainer.FROZEN_HELPER, good)
    good.write_text(good.read_text() + "# drift\n")
    with pytest.raises(trainer.TrainingV2Error, match="two FIT budget literals"):
        trainer.validate_helper_successor(trainer.FROZEN_HELPER, good)


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


def test_schedule_binding_checks_checkpoint_token_digit_and_gold_identity():
    data = make_data()
    data["manifest"]["digit_continuation_ids_by_candidate_count"] = {"2": [16, 17]}
    for index, record in enumerate(data["records"]):
        record.update({"tokens": [index, index + 1], "choices": [{}, {}],
                       "allowed_gold_positions": [index % 2]})
    update = 341
    item = checkpoint_item(data, update)
    first = (update - 1) * 4
    item["payload"]["batch_records"] = [
        trainer.training_batch_record(entry, data["records"][first + offset],
                                      data["manifest"])
        for offset, entry in enumerate(data["batches"][update - 1])]
    trainer.validate_schedule_binding(data, fit_run(), {"manifests": [item]})
    item["payload"]["batch_records"][0]["allowed_gold_positions"] = [9]
    with pytest.raises(trainer.TrainingV2Error, match="token/digit/gold"):
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
    trainer.validate_run_request("fresh", Path("/definitely/missing/v3"), "segment-0001")
    with pytest.raises(trainer.TrainingV2Error, match="segment-0001"):
        trainer.validate_run_request("fresh", Path("/missing"), "segment-0002")
    with pytest.raises(trainer.TrainingV2Error, match="existing v3 root"):
        trainer.validate_run_request("resume", Path("/missing"), "segment-0002")


def test_invalid_qualification_prevents_fit_loader_and_model_import(tmp_path, monkeypatch):
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
                    resource_qualification=tmp_path / "missing-qualification.json",
                    resource_qualification_sha256="0" * 64, segment_id="segment-0001")
    assert touched == {"fit": False, "system": False}
    assert not (tmp_path / "out").exists()


def test_prior_close_proof_is_required_and_bounded(tmp_path):
    event = {"kind": "segment_terminal", "segment_id": "segment-0001",
             "status": "resource_failure", "terminal_segment_seconds": 101.0}
    from scripts.long_context_lora_checkpoint_v3 import canonical_digest
    digest = canonical_digest(event)
    close = {"reserve_seconds": 1.0, "prior_charged_seconds": 0.0,
             "current_active_observed_seconds": 100.0,
             "current_charged_seconds": 101.0,
             "cumulative_charged_seconds": 101.0,
             "terminal_event_digest": digest,
             "reservation_scope": "terminal event, locked report, unlock, and completion receipt"}
    report = {"schema": "long-context-lora-train-v3-segment",
              "segment_id": "segment-0001", "completed": False,
              "completion_pending": True, "ledger_terminal_status": "resource_failure",
              "terminal_outcome": "resource_failure", "close_reservation": close}
    report_path = tmp_path / "segments/segment-0001/report.json"
    report_sha = write_json(report_path, report)
    receipt = {"schema": "long-context-lora-train-v3-completion-receipt",
               "segment_id": "segment-0001", "report_sha256": report_sha,
               "terminal_event_digest": digest, "ledger_terminal_status": "resource_failure",
               "terminal_outcome": "resource_failure", "finalized": True,
               "training_succeeded": False, "reserve_seconds": 1.0,
               "observed_close_through_unlock_seconds": 0.25,
               "current_charged_seconds": 101.0, "prior_charged_seconds": 0.0,
               "cumulative_charged_seconds": 101.0}
    receipt_path = report_path.parent / "completion-receipt.json"
    write_json(receipt_path, receipt)
    trainer.validate_prior_close_proofs(tmp_path, [event], "segment-0002")
    receipt["observed_close_through_unlock_seconds"] = 1.01
    write_json(receipt_path, receipt)
    with pytest.raises(trainer.TrainingV2Error, match="close proof"):
        trainer.validate_prior_close_proofs(tmp_path, [event], "segment-0002")
    receipt_path.unlink()
    with pytest.raises(trainer.TrainingV2Error, match="required regular JSON"):
        trainer.validate_prior_close_proofs(tmp_path, [event], "segment-0002")


def test_prior_close_proof_rejects_forged_cumulative_accounting(tmp_path):
    event = {"kind": "segment_terminal", "segment_id": "segment-0001",
             "status": "resource_failure", "terminal_segment_seconds": 11.0}
    from scripts.long_context_lora_checkpoint_v3 import canonical_digest
    digest = canonical_digest(event)
    report = {
        "schema": "long-context-lora-train-v3-segment", "segment_id": "segment-0001",
        "completed": False, "completion_pending": True,
        "ledger_terminal_status": "resource_failure",
        "terminal_outcome": "resource_failure",
        "close_reservation": {
            "reserve_seconds": 1.0, "prior_charged_seconds": 7.0,
            "current_active_observed_seconds": 10.0,
            "current_charged_seconds": 11.0, "cumulative_charged_seconds": 18.0,
            "terminal_event_digest": digest,
            "reservation_scope": "terminal event, locked report, unlock, and completion receipt",
        },
    }
    report_path = tmp_path / "segments/segment-0001/report.json"
    report_sha = write_json(report_path, report)
    write_json(report_path.parent / "completion-receipt.json", {
        "schema": "long-context-lora-train-v3-completion-receipt",
        "segment_id": "segment-0001", "report_sha256": report_sha,
        "terminal_event_digest": digest, "ledger_terminal_status": "resource_failure",
        "terminal_outcome": "resource_failure", "finalized": True,
        "training_succeeded": False, "reserve_seconds": 1.0,
        "observed_close_through_unlock_seconds": 0.2,
        "current_charged_seconds": 11.0, "prior_charged_seconds": 7.0,
        "cumulative_charged_seconds": 18.0,
    })
    with pytest.raises(trainer.TrainingV2Error, match="close proof"):
        trainer.validate_prior_close_proofs(tmp_path, [event], "segment-0002")


def test_full_update_timer_includes_bookkeeping_and_settled_observation():
    report = {"updates": []}
    update = {"update": 12, "seconds": None,
              "helper_checkpoint_inclusive_seconds": 80.0}
    stages = []
    trainer.settle_update_record(
        report=report, update_record=update, update_index=12, cursor=48,
        update_started=10.0,
        checkpoint=lambda stage, started: stages.append((stage, started)),
        clock=lambda: 129.5)
    assert stages == [("update_12_settled", 10.0)]
    assert update["seconds"] == 119.5
    assert report["completed_microbatches"] == 48
    with pytest.raises(trainer.TrainingV2Error, match="120 seconds"):
        trainer.settle_update_record(
            report={"updates": []}, update_record={"seconds": None},
            update_index=13, cursor=52, update_started=10.0,
            checkpoint=lambda stage, started: None, clock=lambda: 130.0001)


def test_post_eligibility_settlement_failure_blocks_trainer_resume(tmp_path):
    event = {"kind": "segment_terminal", "segment_id": "segment-0001",
             "status": "failed", "terminal_segment_seconds": 11.0}
    from scripts.long_context_lora_checkpoint_v3 import canonical_digest
    digest = canonical_digest(event)
    report = {
        "schema": "long-context-lora-train-v3-segment", "segment_id": "segment-0001",
        "completed": False, "completion_pending": True,
        "ledger_terminal_status": "failed", "terminal_outcome": "failed",
        "close_reservation": {
            "reserve_seconds": 1.0, "prior_charged_seconds": 0.0,
            "current_active_observed_seconds": 10.0,
            "current_charged_seconds": 11.0, "cumulative_charged_seconds": 11.0,
            "terminal_event_digest": digest,
            "reservation_scope": "terminal event, locked report, unlock, and completion receipt",
        },
    }
    trainer.mark_post_eligibility_failure(
        report, trainer.PostEligibilityUpdateError(12, 44, RuntimeError("settled late")))
    assert report["resume_permitted"] is False
    assert report["completed_updates"] == 11
    assert report["completed_microbatches"] == 44
    assert report["rejected_eligible_update"] == 12
    report_path = tmp_path / "segments/segment-0001/report.json"
    report_sha = write_json(report_path, report)
    write_json(report_path.parent / "completion-receipt.json", {
        "schema": "long-context-lora-train-v3-completion-receipt",
        "segment_id": "segment-0001", "report_sha256": report_sha,
        "terminal_event_digest": digest, "ledger_terminal_status": "failed",
        "terminal_outcome": "failed", "finalized": True, "training_succeeded": False,
        "reserve_seconds": 1.0, "observed_close_through_unlock_seconds": 0.2,
        "current_charged_seconds": 11.0, "prior_charged_seconds": 0.0,
        "cumulative_charged_seconds": 11.0,
    })
    with pytest.raises(trainer.TrainingV2Error, match="close proof"):
        trainer.validate_prior_close_proofs(tmp_path, [event], "segment-0002")


def test_predicate_accounting_separates_logical_and_interrupted_work(tmp_path):
    prior_path = tmp_path / "segments/segment-0001/report.json"
    write_json(prior_path, {"predicate_reads_segment": {
        "predicate_host_reads": 100, "numeric_host_reads": 80}})
    events = [{"kind": "segment_terminal", "segment_id": "segment-0001"}]
    accounting = trainer.predicate_read_accounting(
        tmp_path, events, "segment-0002", {
            "predicate_host_reads": trainer.EXPECTED_PREDICATE_READS - 97,
            "numeric_host_reads": trainer.EXPECTED_NUMERIC_READS - 78})
    assert accounting == {
        "expected_logical": {
            "predicate_host_reads": 4157, "numeric_host_reads": 3464},
        "actual_cumulative": {
            "predicate_host_reads": 4160, "numeric_host_reads": 3466},
        "interrupted_extra": {
            "predicate_host_reads": 3, "numeric_host_reads": 2},
    }
    with pytest.raises(trainer.TrainingV2Error, match="incomplete"):
        trainer.predicate_read_accounting(
            tmp_path, [], "segment-0001", {
                "predicate_host_reads": 4156, "numeric_host_reads": 3464})


class FakeSegment:
    def __init__(self):
        self.finished = None

    def finish(self, status, seconds, **fields):
        self.finished = (status, seconds, fields)
        payload = {"kind": "segment_terminal", "segment_id": "segment-0002",
                   "status": status, "terminal_segment_seconds": seconds}
        return {"payload": payload, "digest": "d" * 64}


def test_cumulative_clock_charges_preflight_and_prior_segments(tmp_path):
    segment = FakeSegment()
    report_path = tmp_path / "report.json"
    report = {"schema": "long-context-lora-train-v3-segment",
              "segment_id": "segment-0002"}
    closing = trainer.finish_segment_locked(
        segment=segment, report=report, report_path=report_path,
        ledger_status="resource_failure", terminal_outcome="resource_failure",
        segment_started=10.0, prior_seconds=28700.0, clock=lambda: 100.0)
    assert segment.finished[1] == 91.0
    assert report["close_reservation"]["cumulative_charged_seconds"] == 28791.0
    assert closing["report_sha256"] == trainer.sha256_file(report_path)
    with pytest.raises(trainer.TrainingV2Error, match="whole logical-run"):
        trainer.finish_segment_locked(
            segment=FakeSegment(), report={"segment_id": "segment-0003"},
            report_path=tmp_path / "over.json", ledger_status="complete",
            terminal_outcome="success", segment_started=10.0,
            prior_seconds=28710.0, clock=lambda: 100.0)


@pytest.mark.parametrize("mode", ["fresh", "resume"])
def test_run_timer_starts_before_branch_resource_store_preflight(tmp_path, monkeypatch, mode):
    order = []
    monkeypatch.setattr(trainer, "validate_run_request", lambda *args: None)
    monkeypatch.setattr(trainer, "validate_helper_successor", lambda: None)
    monkeypatch.setattr(trainer, "validate_resource_qualification", lambda *args: {})
    monkeypatch.setattr(trainer, "capture_source_inventory", lambda proof: ({}, {}))
    monkeypatch.setattr(trainer, "sha256_file", lambda path: trainer.v1.DATA_MANIFEST_SHA256)
    monkeypatch.setattr(trainer, "_read_json", lambda path: {
        "file_hashes": {"schedule.jsonl": "a" * 64}})
    monkeypatch.setattr(trainer.time, "monotonic", lambda: order.append("clock") or 10.0)
    monkeypatch.setattr(trainer.resource_v1, "system_resources",
                        lambda: order.append("resources") or {
                            "swap_used_bytes": 0, "free_percent": 50.0})
    monkeypatch.setattr(trainer, "make_run_identity", lambda **kwargs: {})

    class StopHere(RuntimeError):
        pass

    if mode == "fresh":
        monkeypatch.setattr(trainer.CheckpointStore, "create",
                            lambda *args: order.append("store") or (_ for _ in ()).throw(
                                StopHere()))
    else:
        monkeypatch.setattr(trainer.CheckpointStore, "open",
                            lambda *args: order.append("store") or (_ for _ in ()).throw(
                                StopHere()))
    with pytest.raises(StopHere):
        trainer.run(mode=mode, output=tmp_path / "run",
                    resource_qualification=tmp_path / "qualification.json",
                    resource_qualification_sha256="b" * 64,
                    segment_id="segment-0001" if mode == "fresh" else "segment-0002")
    assert order[0] == "clock"
    if mode == "fresh":
        assert order[:3] == ["clock", "resources", "store"]
    else:
        assert order[:2] == ["clock", "store"]


def test_receipt_binds_locked_report_and_covers_unlock_and_final_fsync(tmp_path):
    segment = FakeSegment(); report_path = tmp_path / "report.json"
    report = {"schema": "long-context-lora-train-v3-segment",
              "segment_id": "segment-0001"}
    closing = trainer.finish_segment_locked(
        segment=segment, report=report, report_path=report_path,
        ledger_status="complete", terminal_outcome="success",
        segment_started=0.0, prior_seconds=0.0, clock=lambda: 10.0)
    times = iter((10.2, 10.3, 10.4))
    receipt = trainer.commit_segment_receipt(
        report_path=report_path, report=report, closing=closing,
        clock=lambda: next(times))
    assert receipt["training_succeeded"] is True
    assert receipt["report_sha256"] == trainer.sha256_file(report_path)
    assert json.loads((tmp_path / "completion-receipt.json").read_text()) == receipt


def test_slow_receipt_fsync_removes_receipt_and_preserves_false_report(tmp_path):
    segment = FakeSegment(); report_path = tmp_path / "report.json"
    report = {"schema": "long-context-lora-train-v3-segment",
              "segment_id": "segment-0001"}
    closing = trainer.finish_segment_locked(
        segment=segment, report=report, report_path=report_path,
        ledger_status="complete", terminal_outcome="success",
        segment_started=0.0, prior_seconds=0.0, clock=lambda: 10.0)
    times = iter((10.2, 10.3, 11.1))
    with pytest.raises(trainer.TrainingV2Error, match="close exceeded"):
        trainer.commit_segment_receipt(
            report_path=report_path, report=report, closing=closing,
            clock=lambda: next(times))
    assert json.loads(report_path.read_text())["completed"] is False
    assert not (tmp_path / "completion-receipt.json").exists()
    assert (tmp_path / "completion-receipt-failure.json").is_file()


def test_report_mutation_after_unlock_blocks_receipt(tmp_path):
    segment = FakeSegment(); report_path = tmp_path / "report.json"
    report = {"schema": "long-context-lora-train-v3-segment",
              "segment_id": "segment-0001"}
    closing = trainer.finish_segment_locked(
        segment=segment, report=report, report_path=report_path,
        ledger_status="failed", terminal_outcome="quality_rejected",
        segment_started=0.0, prior_seconds=0.0, clock=lambda: 10.0)
    report_path.write_text("{}")
    with pytest.raises(trainer.TrainingV2Error, match="changed before receipt"):
        trainer.commit_segment_receipt(
            report_path=report_path, report=report, closing=closing,
            clock=lambda: 10.1)


def test_quality_rejection_requires_metrics_in_immutable_attempt(tmp_path):
    report = {"quality_outcome": "quality_rejected",
              "diagnostic_metrics": {"fresh_parent": {}, "fresh_base": {},
                                     "final_adapter": {}},
              "train_gate": {"passed": False, "net_top1_gain": 4}}
    attempt = tmp_path / "manifest.json"
    write_json(attempt, {"evaluation_completed": True,
                         "quality_outcome": "quality_rejected",
                         "diagnostic_metrics": report["diagnostic_metrics"],
                         "train_gate": report["train_gate"]})
    with pytest.raises(trainer.TrainingV2Error, match="net-gain gate"):
        trainer.require_durable_train_gate(report, attempt)
    durable = json.loads(attempt.read_text())
    assert durable["diagnostic_metrics"] == report["diagnostic_metrics"]
    assert durable["quality_outcome"] == "quality_rejected"


class EventSegment:
    def __init__(self):
        self.events = []

    def append_event(self, kind, **fields):
        payload = {"sequence": 2, "prior_event_digest": "a" * 64,
                   "logical_run_id": "run", "segment_id": "segment-0001",
                   "kind": kind, **fields}
        self.events.append(payload)
        return {"payload": payload, "digest": "b" * 64}


def diagnostic_records():
    from scripts.long_context_lora_evaluation import canonical
    records = []
    for index in range(640):
        candidates = [{"text": f"source-{index}"}, {"text": f"other-{index}"}]
        raw = {"result": f"source-{index}", "candidates": candidates}
        records.append({"id": f"diag-{index:04d}", "role": "train_diagnostic",
                        "source": f"source-{index}", "context": "context",
                        "candidates": candidates, "choices": [(0, "source"), (1, "other")],
                        "raw_bypass_response": raw, "eligible": True,
                        "fallback_reason": None,
                        "_candidate_snapshot": canonical(candidates),
                        "_raw_snapshot": canonical(raw)})
    return records


def test_base_diagnostic_event_binds_manifest_and_all_640_replayed_rows(tmp_path):
    from scripts.long_context_lora_evaluation import apply_logits, canonical
    records = diagnostic_records()
    parent = [apply_logits(record, None) for record in records]
    base = [apply_logits(record, [1.0, 0.0]) for record in records]
    segment = EventSegment()
    binding = trainer._write_base_diagnostic(
        segment, tmp_path, records, parent, base, {"adapter": "c" * 64})
    loaded_parent, loaded_base, manifest = trainer._load_base_diagnostic(
        tmp_path, records, segment.events)
    assert canonical(loaded_parent) == canonical(parent)
    assert canonical(loaded_base) == canonical(base)
    assert manifest["diagnostic_identity_sha256"] == trainer.diagnostic_identity_sha256(records)
    assert binding["event_digest"] == "b" * 64
    paths = trainer._base_paths(tmp_path)
    paths["parent"].write_text(paths["parent"].read_text() + "\n")
    changed = json.loads(paths["manifest"].read_text())
    changed["parent_sha256"] = trainer.sha256_file(paths["parent"])
    write_json(paths["manifest"], changed)
    with pytest.raises(trainer.TrainingV2Error, match="provenance"):
        trainer._load_base_diagnostic(tmp_path, records, segment.events)


def test_prediction_replay_rejects_row_reorder_and_candidate_raw_drift():
    from scripts.long_context_lora_evaluation import apply_logits
    records = diagnostic_records()
    predictions = [apply_logits(record, None) for record in records]
    predictions[0], predictions[1] = predictions[1], predictions[0]
    with pytest.raises(trainer.TrainingV2Error, match="replay differs"):
        trainer.validate_prediction_replay(records, predictions, parent=True)
    predictions = [apply_logits(record, None) for record in records]
    predictions[0]["raw_sha256_after"] = "0" * 64
    with pytest.raises(trainer.TrainingV2Error, match="replay differs"):
        trainer.validate_prediction_replay(records, predictions, parent=True)


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
    durable_attempt = source.index('_atomic_json(attempt / "manifest.json"')
    quality_gate = source.index("require_durable_train_gate(report", durable_attempt)
    assert durable_attempt < quality_gate


def test_source_has_no_network_or_inline_resume_path():
    source = Path(trainer.__file__).read_text()
    assert "requests" not in source and "urllib" not in source
    assert "while retry" not in source and "auto_resume\": True" not in source
    assert "fresh_data_read_for_evaluation\": False" in source
