"""Single-use external supervisor for the fresh pairwise LoRA evaluation."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORKER_MODULE = "scripts.evaluate_difference_pairwise_lora_fresh"
WORKER_SCRIPT = ROOT / "scripts/evaluate_difference_pairwise_lora_fresh.py"
LIMIT_SECONDS = 21600.0


def _load_worker():
    return importlib.import_module(WORKER_MODULE)


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError as error:
        raise ValueError("fresh evaluation output is outside the repository") from error


def _validate_worker_contract(worker) -> tuple[Path, dict, str]:
    if Path(worker.ROOT).resolve() != ROOT.resolve():
        raise ValueError("worker repository root differs")
    output = Path(worker.OUTPUT)
    sole_output = _relative_to_root(output)
    if sole_output != ".omx/artifacts/difference-pairwise-lora-fresh-v1":
        raise ValueError("worker output differs from the fixed fresh evaluation output")
    if worker.LIMIT_SECONDS != LIMIT_SECONDS:
        raise ValueError("worker time limit differs")
    plan = worker.prepare_launch()
    if not isinstance(plan, dict):
        raise ValueError("worker launch plan is not an object")
    worker_python = plan.get("worker_python")
    invocation = worker_python.get("invocation_path") if isinstance(worker_python, dict) else None
    expected_invocation = ROOT / ".venv-mlx/bin/python"
    if (not isinstance(invocation, str)
            or Path(invocation).absolute() != expected_invocation.absolute()):
        raise ValueError("worker launch plan does not bind the MLX virtual environment")
    if output.exists():
        raise FileExistsError(output)
    return output, plan, invocation


def _read_worker_report(worker, output: Path) -> tuple[dict | None, str | None]:
    path = output / "worker-report.json"
    if not path.is_file():
        return None, None
    try:
        report_sha = worker.sha(path)
        report = json.loads(path.read_bytes())
    except (OSError, TypeError, ValueError):
        return None, None
    return (report if isinstance(report, dict) else None), report_sha


def _report_matches_exit(report: dict | None, exit_status: int | None) -> bool:
    if not isinstance(report, dict):
        return False
    if report.get("schema") != "difference-pairwise-lora-fresh-worker-v1":
        return False
    expected = {
        0: (True, "quality_passed", True),
        2: (True, "quality_rejected", False),
        1: (False, "failed", False),
    }.get(exit_status)
    observed = (report.get("completed"), report.get("terminal_status"),
                report.get("quality_gate_passed"))
    return expected is not None and observed == expected


def _kill_and_reap(process) -> tuple[int | None, str | None]:
    try:
        process.kill()
    except BaseException as error:
        kill_error = f"kill failed: {type(error).__name__}: {error}"
    else:
        kill_error = None
    try:
        exit_status = process.wait()
    except BaseException as error:
        reap_error = f"{type(error).__name__}: {error}"
        if kill_error:
            reap_error = f"{kill_error}; {reap_error}"
        return None, reap_error
    return exit_status, kill_error


def run_supervised(*, worker=None, popen_factory=subprocess.Popen,
                   monotonic=time.monotonic, claim_factory=None) -> int:
    """Run exactly one qualified worker and persist authoritative exit evidence."""
    worker = worker or _load_worker()
    output, plan, invocation = _validate_worker_contract(worker)

    started = monotonic()
    output.mkdir(parents=True, exist_ok=False)
    claim = (claim_factory or (lambda: str(uuid.uuid4())))()
    if not isinstance(claim, str) or not claim:
        raise ValueError("worker claim is empty")
    launch = {
        "schema": "difference-pairwise-lora-fresh-launch-v1",
        "plan": plan,
        "worker_claim": claim,
        "consumed": True,
        "released": False,
        "start_monotonic": started,
        "sole_output": _relative_to_root(output),
    }
    worker.write_new(output / "launch.json", launch)
    launch_sha = worker.sha(output / "launch.json")

    command = [
        invocation,
        str(WORKER_SCRIPT),
        "--worker-claim", claim,
        "--start-monotonic", repr(started),
    ]
    process = None
    exit_status = None
    deadline_terminated = False
    spawn_error = None
    reap_error = None
    try:
        remaining = LIMIT_SECONDS - (monotonic() - started)
        if remaining <= 0.0:
            deadline_terminated = True
        else:
            process = popen_factory(command, cwd=str(ROOT))
            try:
                exit_status = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                deadline_terminated = True
                exit_status, reap_error = _kill_and_reap(process)
    except BaseException as error:
        spawn_error = f"{type(error).__name__}: {error}"
        if process is not None:
            exit_status, reap_error = _kill_and_reap(process)

    ended = monotonic()
    elapsed = ended - started
    if elapsed > LIMIT_SECONDS:
        deadline_terminated = True

    report, report_sha = _read_worker_report(worker, output)
    report_matches_exit = _report_matches_exit(report, exit_status)
    worker_interval_valid = (
        process is not None
        and spawn_error is None
        and reap_error is None
        and not deadline_terminated
        and 0.0 <= elapsed <= LIMIT_SECONDS
        and report_matches_exit
    )
    quality_passed = worker_interval_valid and exit_status == 0
    quality_rejected = worker_interval_valid and exit_status == 2
    receipt = {
        "schema": "difference-pairwise-lora-fresh-external-execution-v1",
        "sole_output": _relative_to_root(output),
        "launch_sha256": launch_sha,
        "worker_report_sha256": report_sha,
        "worker_claim": claim,
        "worker_pid": getattr(process, "pid", None),
        "start_monotonic": started,
        "end_monotonic": ended,
        "elapsed_seconds": elapsed,
        "limit_seconds": LIMIT_SECONDS,
        "worker_exit_status": exit_status,
        "deadline_terminated": deadline_terminated,
        "spawn_error": spawn_error,
        "reap_error": reap_error,
        "report_matches_exit": report_matches_exit,
        "worker_interval_valid": worker_interval_valid,
        "quality_gate_passed": quality_passed,
        "quality_rejected": quality_rejected,
        "additional_model_work_authorized": False,
    }
    worker.write_new(output / "external-execution.json", receipt)
    if quality_passed:
        return 0
    if quality_rejected:
        return 2
    return 1


def main() -> int:
    try:
        return run_supervised()
    except BaseException as error:
        print(json.dumps({"error": f"{type(error).__name__}: {error}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
