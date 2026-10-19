import json
import math
from pathlib import Path
import time

import pytest

from scripts import train_pko_seq2seq as trainer


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def qualified_report(pins=None):
    after = "a" * 64
    return {
        "schema": trainer.PREFLIGHT_SCHEMA,
        "completed": True,
        "qualifies": True,
        "model": {
            "repo_id": trainer.preflight.MODEL_REPO,
            "revision": trainer.preflight.MODEL_REVISION,
            "manifest_sha256": trainer.preflight.MODEL_MANIFEST_SHA256,
            "license_note": "CC-BY-4.0 pending reconciliation",
        },
        "scope": {
            "train_rows": 7704, "dev_source_rows": 316,
            "probe_training_partition": "fit", "probe_fit_rows": 6916,
            "dev_file_parsed_including_label_fields": True,
            "dev_labels_file_read": True, "dev_targets_consumed": False,
            "dev_labels_accessed_or_retained": False,
            "dev_labels_not_passed_to_model": True,
            "dev_inference_performed": False, "test_opened": False,
            "production_changed": False,
        },
        "pins": pins or {"frozen": "1"},
        "tokenizer_audit": {"policy": {
            "official_normalizer": "NFKC", "adapted_normalizer": None,
            "adapted_split_special_tokens": True, "vocabulary_changed": False,
            "official_backend_sha256": "b" * 64,
            "adapted_backend_sha256": "c" * 64,
        }},
        "training": {
            "precision": "float32", "device": "cpu", "threads": 2,
            "batch_size": 1, "learning_rate": 1e-4, "weight_decay": .01,
            "clip_norm": 1.0, "parameters_changed": True,
            "after_parameter_sha256": after,
            "updates": [{"loss": 1.0, "gradient_norm": 2.0, "elapsed_s": 3.0}
                        for _ in range(16)],
        },
        "checkpoint": {
            "format": "safetensors", "restored_parameters_exact": True,
            "restored_parameter_sha256": after, "elapsed_s": 4.0,
            "output_parity": {"max_logit_delta": 1.9e-6,
                              "logits_bitwise_equal": False,
                              "logits_allclose": True, "logit_atol": 1e-5,
                              "logit_rtol": 1e-6, "argmax_same": True},
        },
        "schedule_estimate": {
            "fit_rows": 6916, "heldout_inner_validation_rows": 788,
            "pilot_budget_s": 3600.0, "bounded_pilot_updates": 100,
            "update_p95_s": 2.0, "estimated_dev_generation_s": 50.0,
            "measured_checkpoint_reload_s": 4.0,
            "estimated_bounded_pilot_s": 500.0,
        },
        "resources": {"final": {"rss_bytes": 2 * 1024**3,
                                  "swap_growth_bytes": 0,
                                  "elapsed_s": 500.0}},
    }


def write_qualification(directory, report, monkeypatch):
    directory.mkdir()
    report_path = directory / "report.json"
    write_json(report_path, report)
    report_sha = trainer.sha256_file(report_path)
    proof = {
        "verified": True,
        "identity": {"report_sha256": report_sha,
                     "frozen_source_sha256": trainer.PREFLIGHT_SOURCE_SHA256},
        "counts": {"train_rows": 7704, "fit_rows": 6916,
                   "inner_validation_rows": 788, "dev_source_rows": 316,
                   "selected_fit_rows": 8, "updates": 16,
                   "generated_train_rows": 8},
        "pins": report["pins"],
        "contracts": {"test_opened": False, "dev_inference_performed": False},
        "decision": "resource preflight qualifies the bounded 5204-microstep pilot",
    }
    proof_path = directory / "independent-verification.json"
    write_json(proof_path, proof)
    monkeypatch.setattr(trainer, "PREFLIGHT_REPORT_SHA256", report_sha)
    monkeypatch.setattr(trainer, "PREFLIGHT_PROOF_SHA256",
                        trainer.sha256_file(proof_path))
    return report_path


def test_qualification_replays_authorizing_contract(tmp_path, monkeypatch):
    pins = {"frozen": "1"}
    monkeypatch.setattr(trainer, "expected_preflight_pins", lambda model_dir: pins)
    report = qualified_report(pins)
    directory = tmp_path / "qualification"
    write_qualification(directory, report, monkeypatch)
    result = trainer.validate_qualification(directory, tmp_path / "model")
    assert result["report"] == report
    assert result["expected_pins"] == pins


@pytest.mark.parametrize("mutation", [
    lambda report: report.update(qualifies=False),
    lambda report: report["scope"].update(test_opened=True),
    lambda report: report["tokenizer_audit"]["policy"].update(adapted_normalizer="NFKC"),
    lambda report: report["checkpoint"]["output_parity"].update(argmax_same=False),
    lambda report: report["schedule_estimate"].update(bounded_pilot_updates=6917),
    lambda report: report["resources"]["final"].update(swap_growth_bytes=513 * 1024**2),
])
def test_qualification_rejects_tampered_evidence(tmp_path, monkeypatch, mutation):
    pins = {"frozen": "1"}
    monkeypatch.setattr(trainer, "expected_preflight_pins", lambda model_dir: pins)
    report = qualified_report(pins)
    mutation(report)
    directory = tmp_path / "qualification"
    path = write_qualification(directory, report, monkeypatch)
    with pytest.raises(ValueError):
        trainer.validate_qualification(path, tmp_path / "model")


def test_budget_is_fixed_before_learning_and_reserves_validation_and_dev():
    schedule = {"update_p95_s": 10.0, "estimated_dev_generation_s": 100.0,
                "measured_checkpoint_reload_s": 20.0,
                "bounded_pilot_updates": 400}
    result = trainer.choose_microsteps(
        schedule, elapsed_s=100.0, baseline_validation_s=200.0)
    expected_available = 3600 - 100 - 100 - 250 - 25 - 60
    assert result["budget_limited_microsteps"] == math.floor(expected_available / 12.5)
    assert result["microsteps"] == min(400, math.floor(expected_available / 12.5))
    assert result["fixed_before_learning"] is True
    assert result["estimated_total_s"] <= 3600


def test_budget_rejects_when_internal_validation_consumes_the_run():
    schedule = {"update_p95_s": 10.0, "estimated_dev_generation_s": 300.0,
                "measured_checkpoint_reload_s": 20.0,
                "bounded_pilot_updates": 400}
    with pytest.raises(trainer.ResourceExceeded):
        trainer.choose_microsteps(
            schedule, elapsed_s=1000.0, baseline_validation_s=2000.0)


def test_partition_replay_is_train_only_and_group_disjoint():
    fit, inner, proof = trainer.load_train_partition()
    assert (len(fit), len(inner)) == (6916, 788)
    assert proof["unique_ids"] == 7704
    assert proof["fit_inner_id_overlap"] == 0
    assert proof["fit_group_inner_group_overlap"] == 0


def test_shuffled_schedule_is_unique_deterministic_and_target_free_proof():
    fit = [{"id": f"row-{index}", "source": "x", "target": "정답"}
           for index in range(trainer.FIT_ROWS)]
    first, proof = trainer.shuffled_schedule(fit)
    second, second_proof = trainer.shuffled_schedule(list(reversed(fit)))
    # Input order is part of the frozen corpus identity; repeat the canonical call.
    repeated, repeated_proof = trainer.shuffled_schedule(fit)
    assert [row["id"] for row in first] == [row["id"] for row in repeated]
    assert proof == repeated_proof
    assert proof["unique_ids"] == trainer.FIT_ROWS
    assert proof["contains_targets"] is False
    assert [row["id"] for row in first] != [row["id"] for row in second]
    assert proof != second_proof


def test_gradient_accumulation_uses_actual_last_partial_divisor():
    rows = [{"id": str(index)} for index in range(19)]
    groups = trainer.accumulation_groups(rows, 19)
    assert [len(group) for group in groups] == [8, 8, 3]
    assert [trainer.loss_scale_for_group(group) for group in groups] == [1 / 8, 1 / 8, 1 / 3]
    assert math.isclose(sum(trainer.loss_scale_for_group(groups[-1]) for _ in groups[-1]), 1.0)


def test_output_contract_never_overwrites_terminal_report(tmp_path):
    output = tmp_path / "pilot"
    output.mkdir()
    write_json(output / "report.json", {"completed": True})
    with pytest.raises(FileExistsError):
        trainer.prepare_output(output, resume=False)
    write_json(output / "recovery.json", {"resumable": True})
    with pytest.raises(ValueError):
        trainer.prepare_output(output, resume=True)


def test_recovery_checkpoint_records_model_optimizer_rng_order_and_cursor(tmp_path):
    torch = pytest.importorskip("torch")

    class Saver:
        def save_pretrained(self, path, **kwargs):
            Path(path).mkdir(exist_ok=True)
            (Path(path) / "model.safetensors").write_bytes(b"safe")

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    state = {"optimizer_updates": 100, "next_group": 100, "microsteps": 800}
    identity = {"schedule_sha256": "a" * 64, "pins": {"input": "b" * 64}}
    pointer = trainer.save_recovery(
        tmp_path, Saver(), optimizer, state, identity, Saver())
    assert pointer["resumable"] is True
    assert {key: pointer["state"][key] for key in state} == state
    assert pointer["state"]["updates_log_bytes"] == 0
    assert pointer["identity"] == identity
    assert pointer["python_random_state"]
    assert pointer["artifacts"]["model_files"]
    assert len(pointer["artifacts"]["optimizer_sha256"]) == 64
    payload = torch.load(tmp_path / pointer["optimizer_path"], weights_only=True)
    assert "optimizer" in payload and "torch_rng_state" in payload


def test_watchdog_hard_exit_payload_preserves_cumulative_budget_cursor(tmp_path):
    watchdog = trainer.ResourceWatchdog(
        time.monotonic() - 12.0, 123456, tmp_path, interval_s=5.0)
    watchdog.failure = "ResourceExceeded: RSS bound exceeded"
    payload = watchdog.terminal_failure_payload(
        trainer.ResourceExceeded("RSS bound exceeded"))
    assert payload["cumulative_elapsed_s"] >= 12.0
    assert payload["original_baseline_swap_bytes"] == 123456
    assert payload["completed"] is False
    assert payload["resumable"] is False
