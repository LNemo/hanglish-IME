from pathlib import Path
import json

import pytest

from scripts import probe_long_context_lora_checkpoint_v3 as probe


def state(logits=None):
    if logits is None:
        logits = [
            {"prompt_sha256": str(index), "logits": [float(index), float(index + 1)]}
            for index in range(4)
        ]
    return {
        "adapter_tensor_hashes": {"adapter": "a"},
        "optimizer_tensor_hashes": {"step": "b"},
        "optimizer_step": 4,
        "cursor": {"completed_updates": 4, "completed_microbatches": 16,
                   "next_update": 5, "next_microbatch": 1},
        "microbatch_losses": [4.0, 3.0, 2.0, 1.0],
        "mean_loss": 2.5,
        "gradient_norm_before_clip": 0.75,
        "source_only_logits": logits,
    }


def test_recipe_is_exact_source_free_checkpoint_protocol():
    assert probe.MAIN_UPDATES == 20
    assert probe.PARITY_UPDATES == 4
    assert probe.CHECKPOINT_UPDATE == 2
    assert probe.REFERENCE_UPDATE == 4
    assert probe.INTERRUPT_AFTER_MICROBATCHES == (1, 2, 3)
    assert probe.ACCUMULATION == 4
    assert probe.SCORER_TIMINGS == 40
    assert probe.SYNTHETIC_LIMITS == {
        "rss_bytes": 6 * 1024**3,
        "mlx_peak_bytes": 6 * 1024**3,
        "cache_bytes": 256 * 1024**2,
        "whole_seconds": 900.0,
        "update_seconds": 120.0,
        "minimum_system_free_percent": 10.0,
        "maximum_swap_growth_bytes": 512 * 1024**2,
    }
    assert probe.RECIPE["lora"]["dropout"] == 0.0
    assert probe.RECIPE["post_initialization_random_consumers"] == []
    assert probe.OPTIMIZER_CONFIG == {
        "learning_rate": 1e-5, "betas": [0.9, 0.999], "eps": 1e-8,
        "bias_correction": True}


def test_descriptor_counts_only_logically_accepted_restart_rows():
    descriptor = probe.synthetic_descriptor()
    assert descriptor["source_free"] is True
    assert descriptor["fit_files_read"] is False
    assert descriptor["labels_read"] is False
    assert descriptor["v1_prediction_cache_read"] is False
    assert descriptor["main"] == {
        "updates": 20, "microbatches": 80, "checkpoint_every_update": True}
    assert descriptor["restart"] == {
        "branches": 3,
        "logical_updates_per_branch": 4,
        "logical_microbatches_per_branch": 16,
        "restore_checkpoint_update": 2,
        "interrupt_update": 3,
        "discarded_partial_microbatches": [1, 2, 3],
        "resumes_per_branch": 1,
    }
    assert descriptor["original_reference"] == {
        "updates": 4, "checkpoint_every_update": True,
        "direct_frozen_v2_calls": True}


def test_run_identity_uses_strict_synthetic_helper_contract():
    captured = {}

    def fake_make_run_identity(**kwargs):
        captured.update(kwargs)
        return kwargs

    baseline = {"swap_used_bytes": 123, "free_percent": 50.0}
    sources = {"source": "a" * 64}
    result = probe._run_identity(
        fake_make_run_identity, logical_run_id="fixture", baseline=baseline,
        source_hashes=sources, updates=4)
    assert result == captured
    assert captured["mode"] == "synthetic"
    assert captured["baseline_swap_used_bytes"] == 123
    assert captured["microbatches"] == 16
    assert captured["updates"] == 4
    assert captured["final_update_microbatches"] == 4
    assert captured["recipe_extra"] == {"synthetic_only": True}
    assert len(captured["schedule_sha256"]) == 64
    assert captured["source_hashes"] == sources


def test_schedule_identity_is_present_until_final_checkpoint_only():
    assert probe._schedule_identity(2, 4) == {
        "role": "synthetic", "id": "update-0003-microbatch-0001"}
    assert probe._schedule_identity(4, 4) is None


def test_resume_preflight_maps_system_reader_to_helper_contract():
    assert probe.resume_preflight_metrics({
        "free_percent": 37.0, "swap_used_bytes": 123}) == {
            "system_free_percent": 37.0, "swap_used_bytes": 123}
    with pytest.raises(ValueError, match="metrics differ"):
        probe.resume_preflight_metrics({
            "system_free_percent": 37.0, "swap_used_bytes": 123})


def test_terminal_close_reserve_charges_and_rejects_delayed_locked_work():
    assert probe.reserved_terminal_seconds(10.0, 12.5) == 3.5
    assert probe.validate_segment_close(20.0, clock=lambda: 21.0) == 1.0
    with pytest.raises(RuntimeError, match="exceeded close reserve"):
        probe.validate_segment_close(20.0, clock=lambda: 21.000001)
    terminal = {"payload": {"terminal_segment_seconds": 3.5}, "digest": "a" * 64}
    assert probe._closure_record("main", "segment-0001", terminal, 0.25) == {
        "run_relative_path": "main", "segment_id": "segment-0001",
        "reserve_seconds": 1.0, "actual_close_seconds": 0.25,
        "charged_terminal_seconds": 3.5, "terminal_event_digest": "a" * 64,
        "completed_within_reserve": True,
    }


def test_publish_batch_records_supply_helper_identity_fields():
    record = probe._batch_record(
        3, 2, {"prompt_sha256": "a" * 64, "prompt_token_count": 512})
    assert record == {
        "role": "synthetic", "id": "update-0003-microbatch-0002",
        "update": 3, "microbatch": 10, "position": 2,
        "prompt_sha256": "a" * 64, "prompt_tokens": 512,
    }
    with pytest.raises(ValueError, match="position differs"):
        probe._batch_record(
            3, 5, {"prompt_sha256": "a" * 64, "prompt_token_count": 512})


def test_runtime_projection_uses_all_checkpoint_inclusive_updates_and_fixed_formula():
    result = probe.checkpoint_runtime_estimate([2.0] * 20, [0.25] * 40)
    assert result["formula"] == "1.5*(693*update_p95+1280*probe_scoring_p95)+300"
    assert result["estimated_training_seconds"] == pytest.approx(
        1.5 * (693 * 2.0 + 1280 * 0.25) + 300)
    assert result["update_timing_scope"] == "gradient plus checkpoint transaction and retention"
    assert result["checkpoint_frequency"] == "every optimizer update"
    with pytest.raises(ValueError, match="exactly 20 updates and 40 scorer"):
        probe.checkpoint_runtime_estimate([2.0] * 19, [0.25] * 40)


def test_projection_is_durable_before_rejection(tmp_path):
    estimate = probe.checkpoint_runtime_estimate([30.0] * 20, [2.0] * 40)
    assert estimate["within_21600_seconds"] is False
    report = {"schema": "long-context-lora-resource-checkpoint-v3",
              "completed": False, "durable_phases": [],
              "runtime_estimate": estimate,
              "scorer_timings_seconds": [2.0] * 40,
              "predicate_reads": {"optimized_observed": {
                  "predicate_host_reads": 199, "numeric_host_reads": 167}}}
    payload = {key: report[key] for key in
               ("runtime_estimate", "scorer_timings_seconds", "predicate_reads")}
    probe.persist_phase(tmp_path, tmp_path / "report.json", report, "projection", payload)
    with pytest.raises(RuntimeError, match="exceeds 21600"):
        probe.require_runtime_projection(estimate)
    durable = json.loads((tmp_path / "report.json").read_text())
    assert durable["completed"] is False
    snapshot = json.loads((tmp_path / durable["durable_phases"][0]["path"]).read_text())
    assert snapshot["payload"] == payload


@pytest.mark.parametrize("phase,payload,gate", [
    ("main", {"fresh_checkpoint_reload": {"passed": False}},
     probe.require_main_phase),
    ("restart-2", {"parity": {"passed": False},
                   "trajectory_parity": {"passed": True}},
     probe.require_branch_phase),
])
def test_completed_phase_payload_is_durable_before_parity_rejection(
        tmp_path, phase, payload, gate):
    report = {"schema": "long-context-lora-resource-checkpoint-v3",
              "completed": False, "durable_phases": []}
    probe.persist_phase(tmp_path, tmp_path / "report.json", report, phase, payload)
    with pytest.raises(RuntimeError, match="parity failed"):
        gate(payload)
    durable = json.loads((tmp_path / "report.json").read_text())
    snapshot = json.loads((tmp_path / durable["durable_phases"][0]["path"]).read_text())
    assert snapshot["payload"] == payload
    assert durable["completed"] is False


@pytest.mark.parametrize("gate_name,message", [
    ("main_and_reference_parity", "original and grouped four-update states differ"),
    ("restart_2_parity", "checkpoint restart parity failed"),
    ("runtime_projection", "checkpoint-inclusive projected training runtime exceeds"),
])
def test_run_phase_orchestration_preserves_payload_before_gate_failure(
        tmp_path, gate_name, message):
    report_path = tmp_path / "report.json"
    report = {"schema": "long-context-lora-resource-checkpoint-v3",
              "completed": False, "durable_phases": []}
    payload = {"evidence": gate_name}

    def reject():
        raise RuntimeError(message)

    with pytest.raises(RuntimeError, match=message):
        probe.persist_observe_and_gate(
            tmp_path, report_path, report, "main", payload,
            persistence_gate="main_persistence", observation_stage="main_phase_persisted",
            validation_gate=gate_name, observe=lambda stage: None, validate=reject)
    durable = json.loads(report_path.read_text())
    snapshot = json.loads((tmp_path / durable["durable_phases"][0]["path"]).read_text())
    assert snapshot["payload"] == payload
    assert durable["completed"] is False
    assert report["active_gate"] == gate_name


def test_run_phase_orchestration_preserves_snapshot_on_resource_failure(tmp_path):
    report_path = tmp_path / "report.json"
    report = {"schema": "long-context-lora-resource-checkpoint-v3",
              "completed": False, "durable_phases": []}

    def resource_failure(stage):
        raise probe.ResourceObservationError("injected", {"stage": stage})

    with pytest.raises(probe.ResourceObservationError):
        probe.persist_observe_and_gate(
            tmp_path, report_path, report, "reference", {"states": [1, 2, 3, 4]},
            persistence_gate="reference_persistence",
            observation_stage="reference_phase_persisted",
            validation_gate="reference_persisted", observe=resource_failure,
            validate=lambda: None)
    durable = json.loads(report_path.read_text())
    assert durable["completed"] is False
    assert len(durable["durable_phases"]) == 1


def test_run_phase_orchestration_serialization_failure_preserves_prior_report(tmp_path):
    report_path = tmp_path / "report.json"
    probe._atomic_json(report_path, {"generation": 1})
    original = report_path.read_bytes()

    def serialization_failure(*args, **kwargs):
        raise TypeError("injected phase serialization")

    with pytest.raises(TypeError, match="injected phase serialization"):
        probe.persist_observe_and_gate(
            tmp_path, report_path, {"completed": False, "durable_phases": []},
            "projection", {"estimate": 1}, persistence_gate="projection_persistence",
            observation_stage="projection_phase_persisted",
            validation_gate="runtime_projection", observe=lambda stage: None,
            validate=lambda: None, persister=serialization_failure)
    assert report_path.read_bytes() == original


def test_final_success_report_requires_hash_bound_completion_receipt(tmp_path):
    report_path = tmp_path / "report.json"
    report = {"completed": True, "terminal_status": "success"}
    times = iter((100.0, 100.25, 100.5, 100.75))
    probe.commit_final_report(report_path, report, 0.0, clock=lambda: next(times))
    durable = json.loads(report_path.read_text())
    receipt = json.loads((tmp_path / "completion-receipt.json").read_text())
    assert durable["elapsed_seconds"] == 100.0
    assert durable["final_report_close"] == {
        "active_elapsed_observed_seconds": 100.0,
        "reserve_seconds": 1.0,
        "charged_elapsed_seconds": 900.0,
        "completion_receipt_required": True,
        "reservation_scope":
            "report rename/directory fsync and completion receipt atomic write",
    }
    assert receipt["report_sha256"] == probe._sha256_file(report_path)
    assert receipt["final_report_close"] == durable["final_report_close"]
    assert receipt["report_elapsed_through_fsync_seconds"] == 100.5


def test_final_success_report_without_close_reserve_has_no_completion_receipt(tmp_path):
    report_path = tmp_path / "report.json"
    report = {"completed": True, "terminal_status": "success"}
    times = iter((898.0, 899.25, 899.5))
    with pytest.raises(RuntimeError, match="before final report commit"):
        probe.commit_final_report(report_path, report, 0.0, clock=lambda: next(times))
    durable = json.loads(report_path.read_text())
    assert durable["completed"] is False
    assert durable["terminal_status"] == "failed"
    assert durable["failure_stage"] == "final_report_commit"
    assert not (tmp_path / "completion-receipt.json").exists()


def test_completion_receipt_failure_rewrites_report_as_failed(tmp_path):
    report_path = tmp_path / "report.json"
    report = {"completed": True, "terminal_status": "success"}
    times = iter((100.0, 100.25, 100.5, 100.75))

    def fail_receipt(path, value, **kwargs):
        if Path(path).name == "completion-receipt.json":
            raise OSError("injected receipt write failure")
        return probe._atomic_json(path, value, **kwargs)

    with pytest.raises(OSError, match="injected receipt"):
        probe.commit_final_report(
            report_path, report, 0.0, clock=lambda: next(times), writer=fail_receipt)
    durable = json.loads(report_path.read_text())
    assert durable["completed"] is False
    assert durable["resource_feasible"] is False
    assert durable["terminal_status"] == "failed"
    assert durable["failure_stage"] == "final_report_commit"
    assert durable["failure_type"] == "OSError"
    assert not (tmp_path / "completion-receipt.json").exists()


def test_exact_state_contract_and_tolerant_source_only_logits():
    reference = state()
    actual = state([
        {"prompt_sha256": str(index),
         "logits": [float(index) + 1e-7, float(index + 1) - 1e-7]}
        for index in range(4)
    ])
    result = probe.exact_state_match(reference, actual)
    assert result["passed"] is True
    assert result["mismatches"] == []
    assert len(result["logit_checks"]) == 4
    assert all(item["passed"] for item in result["logit_checks"])


@pytest.mark.parametrize("field,replacement", [
    ("adapter_tensor_hashes", {"adapter": "changed"}),
    ("optimizer_tensor_hashes", {"step": "changed"}),
    ("optimizer_step", 3),
    ("cursor", {"completed_updates": 3}),
    ("microbatch_losses", [1.0]),
    ("mean_loss", 1.0),
    ("gradient_norm_before_clip", 2.0),
])
def test_exact_state_contract_rejects_any_nonnumeric_state_drift(field, replacement):
    reference = state()
    actual = state()
    actual[field] = replacement
    result = probe.exact_state_match(reference, actual)
    assert result["passed"] is False
    assert field in result["mismatches"]


def test_exact_state_contract_rejects_logit_argmax_or_prompt_drift():
    reference = state()
    logits = [dict(item) for item in reference["source_only_logits"]]
    logits[0] = {"prompt_sha256": "different", "logits": [4.0, 0.0]}
    result = probe.exact_state_match(reference, state(logits))
    assert result["passed"] is False
    assert "source_only_logits" in result["mismatches"]


def test_exact_trajectory_rejects_any_per_update_loss_or_norm_drift():
    reference = [
        {"update": update,
         "cursor": {"completed_updates": update, "completed_microbatches": update * 4},
         "microbatch_losses": [float(update)] * 4,
         "mean_loss": float(update),
         "gradient_norm_before_clip": float(update) / 10}
        for update in range(1, 5)
    ]
    assert probe.exact_trajectory_match(reference, reference)["passed"] is True
    changed = [dict(row) for row in reference]
    changed[1]["gradient_norm_before_clip"] = 99.0
    result = probe.exact_trajectory_match(reference, changed)
    assert result["passed"] is False
    assert result["mismatched_updates"] == [2]


def test_existing_output_fails_before_provenance_or_mlx_import(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    called = False

    def forbidden():
        nonlocal called
        called = True
        raise AssertionError("provenance must not run")

    monkeypatch.setattr(probe, "capture_provenance", forbidden)
    with pytest.raises(FileExistsError):
        probe.run(output)
    assert called is False
    assert not (output / "report.json").exists()


def test_early_failure_writes_honest_immutable_report_without_importing_mlx(
        tmp_path, monkeypatch):
    output = tmp_path / "failed"
    monkeypatch.setattr(
        probe, "capture_provenance",
        lambda: (_ for _ in ()).throw(RuntimeError("pinned source differs")))
    with pytest.raises(RuntimeError, match="pinned source differs"):
        probe.run(output)
    report = json.loads((output / "report.json").read_text())
    assert report["completed"] is False
    assert report["resource_feasible"] is False
    assert report["terminal_status"] == "failed"
    assert report["failure_stage"] == "initialization"
    assert report["failure"] == "pinned source differs"
    assert report["real_fit_rows_read"] is False
    assert report["labels_read"] is False


def test_probe_source_has_no_fit_label_cache_or_network_execution_path():
    source = Path(probe.__file__).read_text()
    assert "torch" not in source
    assert "requests" not in source
    assert "urllib" not in source
    assert "fresh_selection" not in source
    assert "training_examples" not in source
    assert '"real_fit_rows_read": False' in source
    assert '"v1_prediction_cache_read": False' in source
    assert '"old_adapter_loaded": False' in source


def test_probe_records_helper_audit_snapshots_instead_of_summary_flags():
    source = Path(probe.__file__).read_text()
    assert source.count("audit_snapshot()") == 3
    assert source.count('"checkpoint_chain": audit') == 3
    assert "_checkpoint_summary" not in source


def test_exact_497_observation_grammar_and_eight_closures():
    stages = probe.expected_observation_stages()
    assert len(stages) == probe.EXPECTED_OBSERVATIONS == 497
    assert stages[0] == "tiny_math" and stages[-1] == "terminal"
    assert stages.count("reference_phase_persisted") == 1
    assert stages.count("main_phase_persisted") == 1
    assert stages[-2] == "projection_phase_persisted"
    assert probe.EXPECTED_CLOSURES == 8


def test_phase_snapshots_are_atomic_hash_chained_and_report_bound(tmp_path):
    report_path = tmp_path / "report.json"
    report = {"schema": "long-context-lora-resource-checkpoint-v3",
              "durable_phases": []}
    first_payload = {"updates": [1, 2, 3, 4]}
    first = probe.persist_phase(tmp_path, report_path, report, "reference", first_payload)
    second_payload = {"updates": list(range(1, 21))}
    second = probe.persist_phase(tmp_path, report_path, report, "main", second_payload)
    assert first["path"] == "phase-snapshots/0001-reference.json"
    assert second["path"] == "phase-snapshots/0002-main.json"
    first_snapshot = json.loads((tmp_path / first["path"]).read_text())
    second_snapshot = json.loads((tmp_path / second["path"]).read_text())
    assert first_snapshot["payload"] == first_payload
    assert first_snapshot["prior_snapshot_sha256"] is None
    assert second_snapshot["payload"] == second_payload
    assert second_snapshot["prior_snapshot_sha256"] == first["sha256"]
    assert json.loads(report_path.read_text())["durable_phases"] == [first, second]
    with pytest.raises(FileExistsError):
        probe._atomic_json(tmp_path / first["path"], first_snapshot, exclusive=True)


def test_atomic_report_failure_preserves_previous_durable_snapshot(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    probe._atomic_json(report_path, {"generation": 1})
    original = report_path.read_bytes()
    real_replace = probe.os.replace

    def fail_report_replace(source, destination):
        if Path(destination) == report_path:
            raise OSError("injected report replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(probe.os, "replace", fail_report_replace)
    with pytest.raises(OSError, match="injected"):
        probe._atomic_json(report_path, {"generation": 2})
    assert report_path.read_bytes() == original


def test_serialization_failure_leaves_prior_report_and_no_partial_temp(tmp_path, monkeypatch):
    report_path = tmp_path / "report.json"
    probe._atomic_json(report_path, {"generation": 1})
    original = report_path.read_bytes()
    monkeypatch.setattr(
        probe.json, "dump",
        lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("injected serialization")))
    with pytest.raises(TypeError, match="injected serialization"):
        probe._atomic_json(report_path, {"generation": object()})
    assert report_path.read_bytes() == original
    assert list(tmp_path.glob(".*.tmp")) == []


def test_reference_calls_frozen_v2_functions_directly_without_monkeypatch():
    source = Path(probe.__file__).read_text()
    for name in ("_new_model", "_microbatch", "_publish_update", "_state_record"):
        assert f"frozen_v2.{name}" in source
    assert "setattr(frozen_v2" not in source and "monkeypatch" not in source
