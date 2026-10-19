import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import capture_document_phonetic_pair as capture


def training_fixture(root: Path, arm: str, prepared: dict, prepared_sha: str,
                     initial: Path) -> tuple[dict, callable]:
    root.mkdir()
    base = capture.base
    checkpoint = root / "phonetic_transducer.pt"
    checkpoint.write_bytes((arm + " checkpoint").encode())
    updates = root / "updates.jsonl"
    updates.write_text("{}\n", encoding="utf-8")
    resource = {
        "elapsed_s": 50.0, "rss_bytes": 1000,
        "swap_used_bytes": 200, "swap_growth_bytes": 20,
    }
    initial_sha = base.sha256_file(initial)
    initial_tensor = "a" * 64
    final_tensor = ("b" if arm == "control" else "c") * 64
    report = {
        "schema": "observed-phonetic-pair-training-v1", "completed": True,
        "arm": arm, "recipe": prepared["recipe"],
        "model_config": prepared["model_config"],
        "prepared_sha256": prepared_sha,
        "probe_report_sha256": "d" * 64,
        "probe_completion_sha256": "e" * 64,
        "evaluation_inputs": prepared["evaluation_inputs"],
        "source_hashes": prepared["source_hashes"],
        "source_hashes_after": prepared["source_hashes"],
        "schedule": prepared["arms"][arm], "updates_completed": base.UPDATES,
        "parameter_finiteness_checks": 32,
        "checkpoint_selection": "final_fixed_update_only", "limits": base.LIMITS,
        "evaluation_targets_read": False, "production_changed": False,
        "mean_loss": 1.0, "final_loss": 0.5, "max_gradient_norm": 2.0,
        "final_resource": resource,
        "initial": {"checkpoint_sha256": initial_sha,
                    "tensor_sha256": initial_tensor, "path": str(initial)},
        "final": {"tensor_sha256": final_tensor,
                  "reload_tensor_sha256": final_tensor, "reload_exact": True,
                  "checkpoint_sha256": base.sha256_file(checkpoint)},
        "updates_sha256": base.sha256_file(updates),
    }

    def write(value=report):
        base.write_json(root / "report.json", value)
        base.write_json(root / "completion.json", {
            "schema": "observed-phonetic-pair-completion-v1", "completed": True,
            "report_sha256": base.sha256_file(root / "report.json"),
            "resource_after_report": resource,
        })
        base.write_json(root / "phonetic_transducer.json", {
            "format_version": 1,
            "checkpoint_sha256": value["final"]["checkpoint_sha256"],
            "config": prepared["model_config"], "arm": arm,
            "initial_checkpoint_sha256": value["initial"]["checkpoint_sha256"],
            "initial_tensor_sha256": value["initial"]["tensor_sha256"],
            "final_tensor_sha256": value["final"]["tensor_sha256"],
        })
        base.write_json(root / "independent-verification.json", {
            "schema": capture.TRAINING_PROOF_SCHEMA, "verified": True,
            "arm": arm,
            "report_sha256": base.sha256_file(root / "report.json"),
            "completion_sha256": base.sha256_file(root / "completion.json"),
            "checkpoint_sha256": value["final"]["checkpoint_sha256"],
            "checkpoint_metadata_sha256": base.sha256_file(
                root / "phonetic_transducer.json"),
            "updates_sha256": value["updates_sha256"],
            "prepared_sha256": prepared_sha,
            "probe_report_sha256": value["probe_report_sha256"],
            "probe_completion_sha256": value["probe_completion_sha256"],
            "verifier_sha256": base.sha256_file(capture.TRAINING_VERIFIER),
            "counts": {
                "pairs": 51361 if arm == "control" else 84345,
                "updates": base.UPDATES,
                "presentations": base.UPDATES * base.BATCH_SIZE,
                "parameter_finiteness_checks": 32,
            },
            "schedule": {key: item for key, item in value["schedule"].items()
                         if key not in {"probe_beam_inputs", "probe_update_positions"}},
            "training_aggregates": {
                "updates": base.UPDATES, "mean_loss": value["mean_loss"],
                "final_loss": value["final_loss"],
                "max_gradient_norm": value["max_gradient_norm"],
                "parameter_finiteness_checks": 32,
            },
            "resources": {
                "main": {"operations": base.UPDATES + 2,
                         "final": value["final_resource"]},
                "finalization": {"operations": 2, "final": resource},
            },
            "limits": base.LIMITS,
            "contracts_verified": {
                "prepared_v3_and_source_pins": True,
                "accepted_probe_and_shared_initial": True,
                "schedule_indices_costs_and_exposure": True,
                "finite_update_ledger": True,
                "finiteness_flags_exact": True,
                "final_changed_and_reload_recorded_exact": True,
                "checkpoint_file_and_metadata_hashes": True,
                "atomic_resources_and_finalization": True,
                "failure_artifact_absent": True,
                "evaluation_targets_read": False,
                "neural_or_checkpoint_load_performed": False,
            },
        })

    write()
    return report, write


def paired_fixture(tmp_path, monkeypatch):
    initial = tmp_path / "initial.pt"
    initial.write_bytes(b"shared initial")
    prepared_file = tmp_path / "prepared.json"
    prepared_file.write_text("{}\n", encoding="utf-8")
    prepared_sha = capture.base.sha256_file(prepared_file)
    prepared = {
        "recipe": {"frozen": True}, "model_config": capture.base.CONFIG,
        "evaluation_inputs": 10166, "source_hashes": {"fixture": "hash"},
        "arms": {"control": {"schedule": "control", "probe_beam_inputs": ["a"],
                             "probe_update_positions": [0]},
                 "augmented": {"schedule": "augmented", "probe_beam_inputs": ["a"],
                               "probe_update_positions": [0]}},
    }
    control = tmp_path / "control"
    augmented = tmp_path / "augmented"
    training_fixture(control, "control", prepared, prepared_sha, initial)
    training_fixture(augmented, "augmented", prepared, prepared_sha, initial)
    monkeypatch.setattr(capture.base, "verify_pins", lambda pins: None)
    return prepared, prepared_sha, control, augmented


def test_training_auth_requires_fixed_contract_and_proof(tmp_path, monkeypatch):
    prepared, prepared_sha, control, _ = paired_fixture(tmp_path, monkeypatch)
    report = capture.authenticate_training(control, "control", prepared, prepared_sha)
    assert report["parameter_finiteness_checks"] == 32

    bad = copy.deepcopy(report)
    bad["production_changed"] = True
    capture.base.write_json(control / "report.json", bad)
    with pytest.raises(ValueError, match="training contract"):
        capture.authenticate_training(control, "control", prepared, prepared_sha)


def test_training_auth_rejects_metadata_proof_and_checkpoint_tamper(tmp_path, monkeypatch):
    prepared, prepared_sha, control, _ = paired_fixture(tmp_path, monkeypatch)
    proof = capture.base.read_json(control / "independent-verification.json")
    proof["verified"] = False
    capture.base.write_json(control / "independent-verification.json", proof)
    with pytest.raises(ValueError, match="independent training proof"):
        capture.authenticate_training(control, "control", prepared, prepared_sha)

    _, write = training_fixture(tmp_path / "separate", "control", prepared,
                                prepared_sha, tmp_path / "initial.pt")
    metadata = capture.base.read_json(tmp_path / "separate/phonetic_transducer.json")
    metadata["arm"] = "augmented"
    capture.base.write_json(tmp_path / "separate/phonetic_transducer.json", metadata)
    with pytest.raises(ValueError, match="checkpoint metadata"):
        capture.authenticate_training(tmp_path / "separate", "control", prepared, prepared_sha)


@pytest.mark.parametrize("mutation,match", [
    (lambda proof: proof.update(arm="augmented"), "training proof differs"),
    (lambda proof: proof.update(verifier_sha256="0" * 64), "training proof differs"),
    (lambda proof: proof.pop("resources"), "resource proof differs"),
])
def test_training_auth_rejects_wrong_arm_stale_verifier_and_missing_resources(
        tmp_path, monkeypatch, mutation, match):
    prepared, prepared_sha, control, _ = paired_fixture(tmp_path, monkeypatch)
    proof = capture.base.read_json(control / "independent-verification.json")
    mutation(proof)
    capture.base.write_json(control / "independent-verification.json", proof)
    with pytest.raises(ValueError, match=match):
        capture.authenticate_training(control, "control", prepared, prepared_sha)


def test_original_swap_baseline_accepts_signed_growth_and_rejects_bad_values():
    assert capture.original_swap_baseline(
        {"swap_used_bytes": 120, "swap_growth_bytes": 20}) == 100
    assert capture.original_swap_baseline(
        {"swap_used_bytes": 90, "swap_growth_bytes": -10}) == 100
    for row in ({"swap_used_bytes": -1, "swap_growth_bytes": 0},
                {"swap_used_bytes": 1, "swap_growth_bytes": 2},
                {"swap_used_bytes": 1.0, "swap_growth_bytes": 0}):
        with pytest.raises(ValueError, match="swap"):
            capture.original_swap_baseline(row)


def test_authorization_is_exact_and_tamper_fails_before_capture(tmp_path, monkeypatch):
    prepared, prepared_sha, control, augmented = paired_fixture(tmp_path, monkeypatch)
    prepared_dir = tmp_path / "prepared-dir"
    prepared_dir.mkdir()
    (prepared_dir / "prepared.json").write_text("{}\n", encoding="utf-8")
    # Make the fixture SHA match the actual prepared path used by authorization.
    actual_sha = capture.base.sha256_file(prepared_dir / "prepared.json")
    for directory in (control, augmented):
        report = capture.base.read_json(directory / "report.json")
        report["prepared_sha256"] = actual_sha
        capture.base.write_json(directory / "report.json", report)
        completion = capture.base.read_json(directory / "completion.json")
        completion["report_sha256"] = capture.base.sha256_file(directory / "report.json")
        capture.base.write_json(directory / "completion.json", completion)
        proof = capture.base.read_json(directory / "independent-verification.json")
        proof["report_sha256"] = completion["report_sha256"]
        capture.base.write_json(directory / "independent-verification.json", proof)

    source = tmp_path / "inputs.jsonl"
    source.write_text('{"id":"x","input":"a"}\n', encoding="utf-8")
    monkeypatch.setattr(capture, "INPUTS", source)
    monkeypatch.setattr(capture, "INPUTS_SHA", capture.base.sha256_file(source))
    rows = [{"id": hashlib.sha256(b"a").hexdigest(), "input": "a"}]
    monkeypatch.setattr(capture, "load_inputs", lambda: rows)
    monkeypatch.setattr(capture.base, "load_prepared",
                        lambda directory: (prepared, {}, prepared["source_hashes"]))
    monkeypatch.setattr(capture.base, "path_key", lambda path: str(Path(path).resolve()))
    monkeypatch.setattr(capture, "_configure_grouped",
                        lambda: SimpleNamespace(balanced_groups=lambda values: [values]))

    authorization_path = tmp_path / "authorization.json"
    value = capture.write_authorization(
        "control", prepared_dir, control, augmented, authorization_path)
    assert value["authorized"] is True
    assert value["model_loaded"] is False
    assert value["parameter_finiteness_checks"] == 32
    assert capture.validate_authorization(
        authorization_path, "control", prepared_dir, control, augmented) == value

    changed = copy.deepcopy(value)
    changed["limits"]["whole_seconds"] += 1
    capture.base.write_json(authorization_path, changed)
    with pytest.raises(ValueError, match="authorization differs"):
        capture.validate_authorization(
            authorization_path, "control", prepared_dir, control, augmented)
    monkeypatch.setattr(
        capture.base, "torch_modules",
        lambda: (_ for _ in ()).throw(AssertionError("Torch must not load")))
    with pytest.raises(ValueError, match="authorization differs"):
        capture.run("control", prepared_dir, control, augmented,
                    authorization_path, tmp_path / "capture-output")


def test_source_requests_fail_closed_on_changed_bytes(tmp_path, monkeypatch):
    source = tmp_path / "inputs.jsonl"
    source.write_text('{}\n', encoding="utf-8")
    monkeypatch.setattr(capture, "INPUTS", source)
    with pytest.raises(ValueError, match="source requests differ"):
        capture.load_inputs()


def test_cli_requires_authorization_for_capture():
    args = capture.parser().parse_args([
        "authorize", "--arm", "control", "--prepared", "p",
        "--training", "t", "--other-training", "o", "--output", "a.json"])
    assert args.command == "authorize"
    with pytest.raises(SystemExit):
        capture.parser().parse_args([
            "capture", "--arm", "control", "--prepared", "p",
            "--training", "t", "--other-training", "o", "--output", "out"])
