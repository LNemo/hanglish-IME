from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from scripts import verify_difference_pairwise_lora_recovery as recovery


def resource_record():
    observed = {
        "rss_peak_bytes": 1024, "mlx_active_bytes": 0, "mlx_cache_bytes": 0,
        "mlx_peak_bytes": 0, "free_percent": 60.0, "swap_used_bytes": 0,
        "swap_growth_bytes": 0, "elapsed_seconds": 0.1,
    }
    return {"stage": "prelaunch_resource_check", "action": "none", "pre": observed,
            "post": None, "errors": [], "cleanup_seconds": 0.0,
            "checkpoint_seconds": 0.1}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(recovery.canonical(value) + b"\n")


def test_real_v1_failure_is_exact_complete_and_premodel():
    result = recovery.verify_v1_failure()
    assert result["terminal"] is True
    assert result["model_loaded"] is False
    assert result["optimizer_updates"] == 0
    assert result["adapter_created"] is False
    assert result["raw_files"] == recovery.V1_RAW_HASHES


@pytest.mark.parametrize("mutation", ["report_byte", "extra_directory"])
def test_full_disk_v1_copy_replays_and_one_byte_corruption_rejects(tmp_path, monkeypatch, mutation):
    original_root = recovery.ROOT
    paths = [recovery.PLAN, recovery.PLAN_REVIEW, recovery.FAILURE_AUDIT,
             recovery.BUDGET_PROOF]
    for source in paths:
        target = tmp_path / source.relative_to(original_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    copied_output = tmp_path / recovery.V1_OUTPUT.relative_to(original_root)
    copied_archive = tmp_path / recovery.ARCHIVE.parent.relative_to(original_root)
    shutil.copytree(recovery.V1_OUTPUT, copied_output)
    shutil.copytree(recovery.ARCHIVE.parent, copied_archive)
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    monkeypatch.setattr(recovery, "V1_OUTPUT", copied_output)
    monkeypatch.setattr(recovery, "ARCHIVE", copied_archive / "manifest.json")
    monkeypatch.setattr(recovery, "FAILURE_AUDIT", tmp_path / recovery.FAILURE_AUDIT.relative_to(original_root))
    monkeypatch.setattr(recovery, "BUDGET_PROOF", tmp_path / recovery.BUDGET_PROOF.relative_to(original_root))
    monkeypatch.setattr(recovery, "PLAN", tmp_path / recovery.PLAN.relative_to(original_root))
    monkeypatch.setattr(recovery, "PLAN_REVIEW", tmp_path / recovery.PLAN_REVIEW.relative_to(original_root))
    assert recovery.verify_v1_failure()["terminal"] is True
    if mutation == "extra_directory":
        (copied_output / "unexpected_checkpoint").mkdir()
    else:
        with (copied_output / "report.json").open("ab") as stream:
            stream.write(b" ")
    with pytest.raises(recovery.RecoveryError, match="raw evidence changed|exact five-file"):
        recovery.verify_v1_failure()


@pytest.mark.parametrize("probe_injected", [False, True])
def test_premodel_preflight_preserves_venv_invocation_and_resolved_binary(
        tmp_path, monkeypatch, probe_injected):
    root = tmp_path / "root"; root.mkdir()
    invocation = tmp_path / "venv/bin/python"; invocation.parent.mkdir(parents=True)
    resolved = tmp_path / "base/python"; resolved.parent.mkdir(parents=True)
    resolved.write_bytes(b"python-binary")
    invocation.symlink_to(resolved)
    monkeypatch.setattr(recovery, "ROOT", root)
    monkeypatch.setattr(recovery, "EXECUTION_SOURCES", set())
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        payload = (resource_record() if "check_pre_model_resources" in argv[-1] else
                   {"invocation_path": str(invocation), "resolved_path": str(resolved),
                    "packages": recovery.budget.VERSIONS})
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    output = root / "preflight.json"
    receipt = recovery.generate_premodel_preflight(
        output, str(invocation), runner=runner,
        **({"resource_probe": resource_record} if probe_injected else {}))
    identity = receipt["payload"]["interpreter"]
    assert identity["invocation_path"] == str(invocation)
    assert identity["resolved_path"] == str(resolved)
    assert identity["resolved_sha256"] == recovery.sha(resolved)
    assert receipt["payload"]["interpreter_after"] == identity
    assert [call[0] for call in calls] == [str(invocation)] * (2 if probe_injected else 3)
    assert recovery.validate_premodel_preflight(output)["interpreter"] == identity


@pytest.mark.parametrize("mutation", ["errors", "nonnumeric", "cap", "interpreter_after"])
def test_rehashed_corrupt_premodel_receipt_rejects(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    monkeypatch.setattr(recovery, "EXECUTION_SOURCES", set())
    binary = Path(os.path.realpath(os.sys.executable))
    identity = {"invocation_path": os.path.abspath(os.sys.executable),
                "resolved_path": str(binary), "resolved_sha256": recovery.sha(binary),
                "package_versions": recovery.budget.VERSIONS}
    payload = {"schema": recovery.PREFLIGHT_SCHEMA, "passed": True,
        "logical_fit_id": recovery.LOGICAL_FIT_ID, "sole_output": recovery.SOLE_OUTPUT,
        "interpreter": identity, "interpreter_after": deepcopy(identity),
        "resource_observation": resource_record(), "source_hashes_before": {},
        "source_hashes_after": {}, "tokenizer_loaded": False,
        "model_weights_loaded": False, "model_work_started": False, "claim_consumed": False}
    if mutation == "errors": payload["resource_observation"]["errors"] = ["bad"]
    elif mutation == "nonnumeric": payload["resource_observation"]["pre"]["rss_peak_bytes"] = "nan"
    elif mutation == "cap": payload["resource_observation"]["pre"]["free_percent"] = 1
    else: payload["interpreter_after"]["resolved_sha256"] = "0" * 64
    path = tmp_path / "preflight.json"
    write_json(path, {"schema": recovery.PREFLIGHT_SCHEMA, "payload": payload,
                      "digest": recovery.digest(payload)})
    with pytest.raises(recovery.RecoveryError):
        recovery.validate_premodel_preflight(path)


def valid_check_payload():
    commands = [{"argv": argv, "returncode": 0,
                 "stdout_sha256": hashlib.sha256(b"ok").hexdigest(),
                 "stderr_sha256": hashlib.sha256(b"").hexdigest()}
                for argv in recovery.default_check_commands()]
    return {"schema": recovery.CHECKS_SCHEMA, "passed": True, "commands": commands,
            "source_hashes_before": {}, "source_hashes_after": {},
            "model_weights_loaded": False, "model_work_started": False,
            "v2_claim_consumed": False}


def test_default_checks_are_model_free_and_cover_required_suites():
    commands = recovery.default_check_commands()
    recovery._validate_check_commands([
        {"argv": command, "returncode": 0, "stdout_sha256": "a" * 64,
         "stderr_sha256": "b" * 64} for command in commands])
    assert all("train_difference_pairwise_lora.py" not in command[:2] for command in commands)


@pytest.mark.parametrize("mutation", ["empty", "required", "failed", "fake_command"])
def test_rehashed_check_receipt_rejects_missing_command_provenance(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    monkeypatch.setattr(recovery, "EXECUTION_SOURCES", set())
    payload = valid_check_payload()
    if mutation == "empty": payload["commands"] = []
    elif mutation == "required": payload["commands"][0]["argv"].remove(
        "tests/test_pairwise_resource_binding.py")
    elif mutation == "fake_command":
        payload["commands"][0]["argv"] = ["/usr/bin/true", *payload["commands"][0]["argv"][4:]]
        payload["commands"][1]["argv"] = ["/usr/bin/true"]
    else: payload["commands"][1]["returncode"] = 1
    path = tmp_path / "checks.json"
    write_json(path, {"schema": recovery.CHECKS_SCHEMA, "payload": payload,
                      "digest": recovery.digest(payload)})
    with pytest.raises(recovery.RecoveryError):
        recovery.validate_check_receipt(path)


def test_run_checks_detects_source_mutation_and_never_writes_receipt(tmp_path, monkeypatch):
    source = tmp_path / "source.py"; source.write_text("before")
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    monkeypatch.setattr(recovery, "EXECUTION_SOURCES", {"source.py"})
    calls = [0]

    def runner(argv, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            source.write_text("after")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    commands = recovery.default_check_commands()
    output = tmp_path / "checks.json"
    with pytest.raises(recovery.RecoveryError, match="changed bound sources"):
        recovery.run_check_commands(output, commands, runner=runner)
    assert not output.exists()


def test_noncanonical_checks_rejected_before_any_command_runs(tmp_path):
    calls = []
    with pytest.raises(recovery.RecoveryError, match="canonical"):
        recovery.run_check_commands(tmp_path / "checks.json", [["/usr/bin/true"]],
                                    runner=lambda *args, **kwargs: calls.append(args))
    assert calls == []


def test_execution_review_must_bind_every_source_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    sources = {"a.py": "a" * 64, "b.py": "b" * 64}
    review = tmp_path / "review.md"
    review.write_text(f"Verdict: CLEAR\n{recovery.LOGICAL_FIT_ID}\n{recovery.SOLE_OUTPUT}\n{'a' * 64}\n")
    with pytest.raises(recovery.RecoveryError, match="every implementation source"):
        recovery.validate_execution_review(review, sources)
    review.write_text(review.read_text() + "b" * 64 + "\n")
    assert recovery.validate_execution_review(review, sources)["verdict"] == "CLEAR"


@pytest.mark.parametrize("extra", ["premodel_preflight", "implementation_checks",
    "execution_review", "v1_claim_released", "second_replacement_authorized"])
def test_prelaunch_rejects_duplicate_or_contradictory_recovery_fields(extra):
    payload = dict.fromkeys(recovery.PRELAUNCH_FIELDS)
    recovery.validate_prelaunch_shape(payload)
    payload[extra] = True
    with pytest.raises(recovery.RecoveryError, match="prelaunch fields"):
        recovery.validate_prelaunch_shape(payload)
