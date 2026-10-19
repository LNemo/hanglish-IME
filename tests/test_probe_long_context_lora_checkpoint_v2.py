from pathlib import Path
import json

import pytest

from scripts import probe_long_context_lora_checkpoint_v2 as probe


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
    assert source.count("audit_snapshot()") == 2
    assert source.count('"checkpoint_chain": audit') == 2
    assert "_checkpoint_summary" not in source
