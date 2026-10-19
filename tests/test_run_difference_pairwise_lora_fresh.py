import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import run_difference_pairwise_lora_fresh as runner


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


class FakeWorker:
    LIMIT_SECONDS = runner.LIMIT_SECONDS

    def __init__(self, root, output, *, plan=None):
        self.ROOT = root
        self.OUTPUT = output
        self.calls = []
        self.plan = plan or {
            "schema": "synthetic-fresh-plan-v1",
            "worker_python": {"invocation_path": str(root / ".venv-mlx/bin/python")},
        }

    def prepare_launch(self):
        self.calls.append("prepare_launch")
        return self.plan

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def write_new(self, path, value):
        self.calls.append(("write_new", Path(path).name))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(canonical(value) + b"\n")


class FinishedProcess:
    pid = 71

    def __init__(self, worker, code, *, report=None):
        self.code = code
        self.wait_timeouts = []
        if report is not None:
            worker.write_new(worker.OUTPUT / "worker-report.json", report)

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        return self.code


def worker_report(code):
    completed, status, passed = {
        0: (True, "quality_passed", True),
        2: (True, "quality_rejected", False),
        1: (False, "failed", False),
    }[code]
    return {
        "schema": "difference-pairwise-lora-fresh-worker-v1",
        "completed": completed,
        "terminal_status": status,
        "quality_gate_passed": passed,
    }


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    root = tmp_path
    (root / ".venv-mlx/bin").mkdir(parents=True)
    (root / "scripts").mkdir()
    output = root / ".omx/artifacts/difference-pairwise-lora-fresh-v1"
    worker = FakeWorker(root, output)
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "WORKER_SCRIPT", root / "scripts/evaluate_difference_pairwise_lora_fresh.py")
    return worker


@pytest.mark.parametrize("code", [0, 2])
def test_valid_quality_outcomes_bind_launch_report_and_exact_worker_command(synthetic, code):
    clock = iter([10.0, 10.5, 12.0])
    spawned = []

    def spawn(command, cwd):
        process = FinishedProcess(synthetic, code, report=worker_report(code))
        spawned.append((command, cwd, process))
        return process

    result = runner.run_supervised(
        worker=synthetic, popen_factory=spawn, monotonic=lambda: next(clock),
        claim_factory=lambda: "claim-1")
    launch = json.loads((synthetic.OUTPUT / "launch.json").read_bytes())
    receipt = json.loads((synthetic.OUTPUT / "external-execution.json").read_bytes())
    command, cwd, process = spawned[0]
    assert result == code
    assert launch["worker_claim"] == "claim-1" and launch["consumed"] is True
    assert launch["released"] is False and launch["start_monotonic"] == 10.0
    assert command == [
        str(synthetic.ROOT / ".venv-mlx/bin/python"),
        str(synthetic.ROOT / "scripts/evaluate_difference_pairwise_lora_fresh.py"),
        "--worker-claim", "claim-1", "--start-monotonic", "10.0",
    ]
    assert cwd == str(synthetic.ROOT)
    assert process.wait_timeouts == [runner.LIMIT_SECONDS - 0.5]
    assert receipt["launch_sha256"] == synthetic.sha(synthetic.OUTPUT / "launch.json")
    assert receipt["worker_report_sha256"] == synthetic.sha(
        synthetic.OUTPUT / "worker-report.json")
    assert receipt["worker_interval_valid"] is True
    assert receipt["quality_gate_passed"] is (code == 0)
    assert receipt["quality_rejected"] is (code == 2)


class BlockingProcess:
    pid = 72

    def __init__(self, worker):
        self.worker = worker
        self.killed = False
        self.wait_calls = 0

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.wait_calls == 1:
            raise subprocess.TimeoutExpired("worker", timeout)
        return -9

    def kill(self):
        self.killed = True


def test_hung_worker_is_killed_reaped_and_cannot_pass(synthetic):
    process = BlockingProcess(synthetic)
    clock = iter([0.0, 0.1, runner.LIMIT_SECONDS + 0.1])
    result = runner.run_supervised(
        worker=synthetic, popen_factory=lambda command, cwd: process,
        monotonic=lambda: next(clock), claim_factory=lambda: "claim")
    receipt = json.loads((synthetic.OUTPUT / "external-execution.json").read_bytes())
    assert result == 1 and process.killed is True and process.wait_calls == 2
    assert receipt["worker_exit_status"] == -9
    assert receipt["deadline_terminated"] is True
    assert receipt["worker_interval_valid"] is False


@pytest.mark.parametrize("report", [None, {"bad": "shape"}, worker_report(0)])
def test_missing_malformed_or_exit_mismatched_report_fails(synthetic, report):
    result = runner.run_supervised(
        worker=synthetic,
        popen_factory=lambda command, cwd: FinishedProcess(synthetic, 1, report=report),
        monotonic=iter([0.0, 0.1, 0.2]).__next__, claim_factory=lambda: "claim")
    receipt = json.loads((synthetic.OUTPUT / "external-execution.json").read_bytes())
    assert result == 1
    assert receipt["report_matches_exit"] is False
    assert receipt["worker_interval_valid"] is False


def test_matching_failed_worker_is_valid_interval_but_supervisor_fails(synthetic):
    result = runner.run_supervised(
        worker=synthetic,
        popen_factory=lambda command, cwd: FinishedProcess(
            synthetic, 1, report=worker_report(1)),
        monotonic=iter([0.0, 0.1, 0.2]).__next__, claim_factory=lambda: "claim")
    receipt = json.loads((synthetic.OUTPUT / "external-execution.json").read_bytes())
    assert result == 1
    assert receipt["report_matches_exit"] is True
    assert receipt["worker_interval_valid"] is True
    assert receipt["quality_gate_passed"] is False


def test_spawn_failure_preserves_single_claim_and_records_error(synthetic):
    def fail(command, cwd):
        raise OSError("cannot spawn")

    result = runner.run_supervised(
        worker=synthetic, popen_factory=fail,
        monotonic=iter([0.0, 0.1, 0.2]).__next__, claim_factory=lambda: "claim")
    receipt = json.loads((synthetic.OUTPUT / "external-execution.json").read_bytes())
    assert result == 1 and (synthetic.OUTPUT / "launch.json").is_file()
    assert receipt["spawn_error"] == "OSError: cannot spawn"
    assert receipt["worker_pid"] is None
    assert receipt["worker_interval_valid"] is False


def test_existing_output_rejects_after_preflight_without_overwrite(synthetic):
    synthetic.OUTPUT.mkdir(parents=True)
    marker = synthetic.OUTPUT / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        runner.run_supervised(worker=synthetic)
    assert synthetic.calls == ["prepare_launch"]
    assert marker.read_text() == "keep"


def test_preflight_failure_consumes_no_claim_and_spawns_no_worker(synthetic):
    def reject():
        synthetic.calls.append("prepare_launch")
        raise ValueError("training proof is not accepted")

    synthetic.prepare_launch = reject
    spawned = []
    with pytest.raises(ValueError, match="training proof"):
        runner.run_supervised(
            worker=synthetic,
            popen_factory=lambda command, cwd: spawned.append(command))
    assert spawned == [] and not synthetic.OUTPUT.exists()


def test_wrong_or_resolved_interpreter_path_rejects_before_claim(synthetic):
    synthetic.plan["worker_python"]["invocation_path"] = "/usr/bin/python3"
    with pytest.raises(ValueError, match="MLX virtual environment"):
        runner.run_supervised(worker=synthetic)
    assert not synthetic.OUTPUT.exists()


def test_deadline_consumed_before_spawn_records_terminal_timeout(synthetic):
    spawned = []
    clock = iter([0.0, runner.LIMIT_SECONDS + 1.0, runner.LIMIT_SECONDS + 2.0])
    result = runner.run_supervised(
        worker=synthetic,
        popen_factory=lambda command, cwd: spawned.append(command),
        monotonic=lambda: next(clock), claim_factory=lambda: "claim")
    receipt = json.loads((synthetic.OUTPUT / "external-execution.json").read_bytes())
    assert result == 1 and spawned == []
    assert receipt["deadline_terminated"] is True
    assert receipt["worker_pid"] is None
