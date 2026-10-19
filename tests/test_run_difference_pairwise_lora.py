import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import run_difference_pairwise_lora as runner


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner.canonical(value) + b"\n")


def authorized_tree(tmp_path, monkeypatch):
    root = tmp_path
    output = root / "fit-output"
    source_hashes = {}
    for index in range(48):
        name = f"source/{index}.txt"
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(str(index))
        source_hashes[name] = runner.sha256_file(path)
    proof_path = root / "budget.json"
    proof_payload = {
        "stage": "budget_qualification", "passed": True, "launch_allowed": True,
        "logical_fit_id": "difference-pairwise-lora-fit-budget-v1",
        "sole_output": ".omx/artifacts/difference-pairwise-lora-train-v1",
        "old_policy": {"launch_allowed": False}, "new_policy": {"limit_seconds": 21600},
        "source_inventory": {"count": 48,
                             "sha256": hashlib.sha256(runner.canonical(source_hashes)).hexdigest(),
                             "hashes": source_hashes},
    }
    proof = {"schema": "difference-pairwise-lora-budget-proof-v1", "payload": proof_payload,
             "digest": hashlib.sha256(runner.canonical(proof_payload)).hexdigest()}
    write_json(proof_path, proof)
    recovery_path = root / "recovery.json"
    for name in ("preflight.json", "checks.json", "review.md"):
        path = root / name; path.write_text(name)
    recovery_payload = {
        "preflight": {"path": "preflight.json", "sha256": runner.sha256_file(root / "preflight.json"),
                      "interpreter": {"path": "/qualified/python", "sha256": "a" * 64,
                                      "package_versions": runner.VERSIONS}},
        "checks": {"path": "checks.json", "sha256": runner.sha256_file(root / "checks.json")},
        "execution_review": {"path": "review.md", "sha256": runner.sha256_file(root / "review.md")},
        "execution_sources": {},
    }
    write_json(recovery_path, {"synthetic": True})
    inventory = dict(source_hashes)
    for name in runner.EXECUTION_ADDITIONS:
        path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(name)
        inventory[name] = runner.sha256_file(path)
    inventory["budget.json"] = runner.sha256_file(proof_path)
    for name in ("recovery.json", "preflight.json", "checks.json", "review.md"):
        inventory[name] = runner.sha256_file(root / name)
    prelaunch = root / "prelaunch.json"
    payload = {
        "schema": runner.recovery.PRELAUNCH_SCHEMA,
        "passed": True,
        "launch_allowed": True,
        "logical_fit_id": runner.LOGICAL_FIT_ID,
        "sole_output": "fit-output",
        "package_versions": {"mlx": "0.31.2", "mlx-lm": "0.31.3",
                             "transformers": "5.12.1"},
        "budget_proof": {"path": "budget.json", "sha256": runner.sha256_file(proof_path)},
        "recovery_proof": {"path": "recovery.json", "sha256": runner.sha256_file(recovery_path)},
        "worker_python": recovery_payload["preflight"]["interpreter"],
        "inventory": inventory,
    }
    write_json(prelaunch, {"payload": payload,
                           "digest": hashlib.sha256(runner.canonical(payload)).hexdigest()})
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "SOLE_OUTPUT", "fit-output")
    monkeypatch.setattr(runner, "DEFAULT_OUTPUT", output)
    monkeypatch.setattr(runner, "DEFAULT_PRELAUNCH", prelaunch)
    monkeypatch.setattr(runner, "QUALIFIED_BUDGET", proof_path)
    monkeypatch.setattr(runner, "QUALIFIED_BUDGET_SHA256", runner.sha256_file(proof_path))
    monkeypatch.setattr(runner, "QUALIFIED_RECOVERY", recovery_path)
    monkeypatch.setattr(runner.recovery, "validate_recovery_proof", lambda path: recovery_payload)
    monkeypatch.setattr(runner, "validate_runtime", lambda python, expected=None: None)
    monkeypatch.setattr(runner, "validate_resource_runtime", lambda python: {})
    return output, prelaunch


def test_launch_claim_is_atomic_consumed_and_never_released(tmp_path, monkeypatch):
    output, prelaunch = authorized_tree(tmp_path, monkeypatch)
    first = runner.claim_launch(output, prelaunch, now=lambda: "now", claim_factory=lambda: "claim")
    assert first["consumed"] is True and first["released"] is False
    assert json.loads((output / "launch.json").read_bytes()) == first
    with pytest.raises(FileExistsError):
        runner.claim_launch(output, prelaunch, claim_factory=lambda: "second")
    assert json.loads((output / "launch.json").read_bytes())["worker_claim"] == "claim"


def test_staged_prelaunch_builder_binds_recovery_and_writes_no_claim(tmp_path, monkeypatch):
    output, prelaunch = authorized_tree(tmp_path, monkeypatch)
    prelaunch.unlink()
    receipt = runner.build_execution_prelaunch(prelaunch)
    payload = receipt["payload"]
    assert payload["schema"] == runner.recovery.PRELAUNCH_SCHEMA
    assert payload["logical_fit_id"] == runner.LOGICAL_FIT_ID
    assert payload["recovery_proof"]["sha256"] == runner.sha256_file(
        runner.QUALIFIED_RECOVERY)
    assert set(payload) == runner.recovery.PRELAUNCH_FIELDS
    runner._load_prelaunch(prelaunch, output)
    assert not output.exists()


def test_alternate_output_rejects_before_directory_or_worker(tmp_path, monkeypatch):
    output, prelaunch = authorized_tree(tmp_path, monkeypatch)
    alternate = tmp_path / "alternate"
    with pytest.raises(ValueError, match="fixed fit/output"):
        runner.claim_launch(alternate, prelaunch)
    assert not alternate.exists() and not output.exists()


def test_alternate_prelaunch_path_rejects_before_claim(tmp_path, monkeypatch):
    output, prelaunch = authorized_tree(tmp_path, monkeypatch)
    copy = tmp_path / "copy.json"; copy.write_bytes(prelaunch.read_bytes())
    with pytest.raises(ValueError, match="path differs"):
        runner.claim_launch(output, copy)
    assert not output.exists()


def test_incomplete_inventory_rejects_before_consuming_launch(tmp_path, monkeypatch):
    output, prelaunch = authorized_tree(tmp_path, monkeypatch)
    receipt = json.loads(prelaunch.read_bytes())
    receipt["payload"]["inventory"].pop("scripts/train_difference_pairwise_lora.py")
    receipt["digest"] = hashlib.sha256(runner.canonical(receipt["payload"])).hexdigest()
    write_json(prelaunch, receipt)
    with pytest.raises(ValueError, match="inventory is incomplete"):
        runner.claim_launch(output, prelaunch)
    assert not output.exists()


class SuccessfulProcess:
    pid = 73

    def __init__(self, command, cwd, output, code=0):
        self.command, self.cwd, self.code = command, cwd, code
        rejected = code == 2
        write_json(output / "report.json", {
            "schema": "difference-pairwise-lora-training-v2",
            "logical_fit_id": runner.LOGICAL_FIT_ID,
            "sole_output": runner.SOLE_OUTPUT,
            "completed": True, "execution_valid": True,
            "quality_passed": not rejected,
            "terminal_status": "quality_rejected" if rejected else "quality_passed",
        })

    def wait(self, timeout=None):
        return self.code


def test_supervisor_records_complete_external_success_after_worker_exit(tmp_path, monkeypatch):
    output, prelaunch = authorized_tree(tmp_path, monkeypatch)
    clock = iter([10.0, 10.5, 12.0])
    created = []

    def spawn(command, cwd):
        process = SuccessfulProcess(command, cwd, output)
        created.append(process)
        return process

    code = runner.run_supervised(output, prelaunch, popen_factory=spawn,
                                 monotonic=lambda: next(clock), wall_time=lambda: "now")
    receipt = json.loads((output / "external-execution.json").read_bytes())
    assert code == 0 and len(created) == 1
    assert receipt["authoritative_success"] is True
    assert receipt["worker_interval_valid"] is True
    assert receipt["deadline_terminated"] is False
    assert receipt["worker_exit_status"] == 0
    assert receipt["terminal_report_sha256"] == runner.sha256_file(output / "report.json")
    assert "--worker-claim" in created[0].command


def test_late_exit_overrides_internal_success(tmp_path, monkeypatch):
    output, prelaunch = authorized_tree(tmp_path, monkeypatch)
    clock = iter([0.0, 0.0, runner.FIT_LIMIT_SECONDS + 0.1])
    process = SuccessfulProcess([], "", output)
    code = runner.run_supervised(output, prelaunch,
                                 popen_factory=lambda command, cwd: process,
                                 monotonic=lambda: next(clock))
    receipt = json.loads((output / "external-execution.json").read_bytes())
    assert code == 1
    assert receipt["deadline_terminated"] is True
    assert receipt["authoritative_success"] is False


class BlockingProcess:
    pid = 99

    def __init__(self, output):
        self.output, self.killed, self.calls = output, False, 0
        write_json(output / "report.json", {
            "schema": "difference-pairwise-lora-training-v2",
            "logical_fit_id": runner.LOGICAL_FIT_ID, "sole_output": runner.SOLE_OUTPUT,
            "completed": True, "execution_valid": True, "quality_passed": True,
            "terminal_status": "quality_passed", "claimed_early": True})

    def wait(self, timeout=None):
        self.calls += 1
        if self.calls == 1:
            raise subprocess.TimeoutExpired("worker", timeout)
        return -9

    def kill(self):
        self.killed = True


class InterruptingProcess(BlockingProcess):
    def wait(self, timeout=None):
        self.calls += 1
        if self.calls == 1:
            raise KeyboardInterrupt()
        return -9


def test_blocking_worker_is_killed_and_early_report_cannot_pass(tmp_path, monkeypatch):
    output, prelaunch = authorized_tree(tmp_path, monkeypatch)
    process = BlockingProcess(output)
    clock = iter([0.0, 0.0, runner.FIT_LIMIT_SECONDS + 1.0])
    code = runner.run_supervised(output, prelaunch,
                                 popen_factory=lambda command, cwd: process,
                                 monotonic=lambda: next(clock))
    receipt = json.loads((output / "external-execution.json").read_bytes())
    assert code == 1 and process.killed is True
    assert receipt["deadline_terminated"] is True
    assert receipt["worker_exit_status"] == -9


def test_unexpected_wait_exception_still_kills_and_reaps_worker(tmp_path, monkeypatch):
    output, prelaunch = authorized_tree(tmp_path, monkeypatch)
    process = InterruptingProcess(output)
    clock = iter([0.0, 0.0, 1.0])
    code = runner.run_supervised(output, prelaunch,
                                 popen_factory=lambda command, cwd: process,
                                 monotonic=lambda: next(clock))
    receipt = json.loads((output / "external-execution.json").read_bytes())
    assert code == 1 and process.killed is True and process.calls == 2
    assert receipt["worker_exit_status"] == -9
    assert "KeyboardInterrupt" in receipt["spawn_error"]


def test_quality_rejection_is_distinct_terminal_exit(tmp_path, monkeypatch):
    output, prelaunch = authorized_tree(tmp_path, monkeypatch)
    clock = iter([0.0, 0.0, 1.0])
    code = runner.run_supervised(
        output, prelaunch,
        popen_factory=lambda command, cwd: SuccessfulProcess(command, cwd, output, code=2),
        monotonic=lambda: next(clock))
    receipt = json.loads((output / "external-execution.json").read_bytes())
    assert code == 2
    assert receipt["authoritative_success"] is False
    assert receipt["worker_interval_valid"] is True
    assert receipt["quality_rejected"] is True
    assert receipt["worker_exit_status"] == 2


def test_spawn_failure_preserves_claim_and_records_external_failure(tmp_path, monkeypatch):
    output, prelaunch = authorized_tree(tmp_path, monkeypatch)
    clock = iter([0.0, 0.5])

    def fail_spawn(command, cwd):
        raise OSError("cannot spawn")

    code = runner.run_supervised(output, prelaunch, popen_factory=fail_spawn,
                                 monotonic=lambda: next(clock))
    receipt = json.loads((output / "external-execution.json").read_bytes())
    assert code == 1 and (output / "launch.json").is_file()
    assert receipt["spawn_error"] == "OSError: cannot spawn"
    assert receipt["worker_interval_valid"] is False


def test_self_consistent_unqualified_budget_rejected_before_claim(tmp_path, monkeypatch):
    output, prelaunch = authorized_tree(tmp_path, monkeypatch)
    envelope = json.loads(prelaunch.read_bytes())
    payload = envelope['payload']
    bp = tmp_path / 'budget.json'
    proof = json.loads(bp.read_bytes())
    proof['payload']['extra_fabricated_success'] = True
    proof['digest'] = hashlib.sha256(runner.canonical(proof['payload'])).hexdigest()
    write_json(bp, proof)
    payload['budget_proof']['sha256'] = runner.sha256_file(bp)
    payload['inventory']['budget.json'] = runner.sha256_file(bp)
    envelope['digest'] = hashlib.sha256(runner.canonical(payload)).hexdigest()
    write_json(prelaunch, envelope)
    with pytest.raises(ValueError, match='budget proof pin'):
        runner.claim_launch(output, prelaunch)
    assert not output.exists()


def test_wrong_worker_environment_rejected_without_consuming_claim(tmp_path, monkeypatch):
    output, prelaunch = authorized_tree(tmp_path, monkeypatch)
    def reject(python, expected=None): raise ValueError('worker Python package versions differ')
    monkeypatch.setattr(runner, 'validate_runtime', reject)
    with pytest.raises(ValueError, match='Python package'):
        runner.run_supervised(output, prelaunch, python='/wrong/python')
    assert not output.exists()


def test_actual_mlx_python_distribution_versions_before_claim():
    invocation = Path(__file__).resolve().parents[1] / '.venv-mlx/bin/python'
    identity = runner.validate_runtime(str(invocation))
    assert identity['invocation_path'] == str(invocation)
    assert identity['resolved_path'] == str(invocation.resolve())
    assert identity['resolved_sha256'] == runner.sha256_file(invocation.resolve())


@pytest.mark.parametrize('shape', ['raw_payload', 'missing_digest'])
def test_prelaunch_envelope_required_before_consuming_claim(tmp_path, monkeypatch, shape):
    output, prelaunch = authorized_tree(tmp_path, monkeypatch)
    receipt = json.loads(prelaunch.read_bytes())
    write_json(prelaunch, receipt['payload'] if shape == 'raw_payload' else {'payload': receipt['payload']})
    with pytest.raises(ValueError, match='envelope is incomplete'):
        runner.claim_launch(output, prelaunch)
    assert not output.exists()


def test_host_resource_failure_does_not_consume_claim(tmp_path, monkeypatch):
    output, prelaunch = authorized_tree(tmp_path, monkeypatch)
    def reject(python): raise ValueError('worker resource preflight failed')
    monkeypatch.setattr(runner, 'validate_resource_runtime', reject)
    with pytest.raises(ValueError, match='resource preflight failed'):
        runner.run_supervised(output, prelaunch, python='unused')
    assert not output.exists()
