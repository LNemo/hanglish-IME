"""Single-claim supervisor for the approved pairwise LoRA fit.

The supervisor owns the non-releasable launch claim and the authoritative
monotonic interval.  Model work happens only in the one child process.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_difference_pairwise_lora_recovery as recovery
LOGICAL_FIT_ID = recovery.LOGICAL_FIT_ID
SOLE_OUTPUT = recovery.SOLE_OUTPUT
FIT_LIMIT_SECONDS = 21600.0
DEFAULT_OUTPUT = ROOT / SOLE_OUTPUT
DEFAULT_PRELAUNCH = ROOT / recovery.PRELAUNCH_PATH
QUALIFIED_BUDGET = ROOT / ".omx/artifacts/difference-pairwise-lora-budget-v1-proof.json"
QUALIFIED_BUDGET_SHA256 = "e59fe3cf9293f2df7cfa19e7e9e198f4691489d41c11215efb19877141888d6a"
QUALIFIED_RECOVERY = ROOT / recovery.RECOVERY_PROOF_PATH
VERSIONS = {"mlx": "0.31.2", "mlx-lm": "0.31.3", "transformers": "5.12.1"}
EXECUTION_ADDITIONS = {
    "scripts/run_difference_pairwise_lora.py",
    "scripts/train_difference_pairwise_lora.py",
    "scripts/verify_difference_pairwise_lora_budget.py",
    "scripts/verify_difference_pairwise_lora_training.py",
    "scripts/difference_pairwise_lora_model.py",
    "scripts/difference_pairwise_lora_evaluation.py",
    "scripts/long_context_lora_model.py",
    "scripts/long_context_lora_resources.py",
    "scripts/probe_long_context_lora_resources.py",
    "scripts/final_choice_policy.py",
    "scripts/prepare_difference_pairwise_lora_data.py",
    "scripts/verify_difference_pairwise_lora_recovery.py",
    "tests/test_run_difference_pairwise_lora.py",
    "tests/test_pairwise_resource_binding.py",
    "tests/test_train_difference_pairwise_lora.py",
    "tests/test_verify_difference_pairwise_lora_training.py",
    "tests/test_verify_difference_pairwise_lora_recovery.py",
}


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_durable(path: Path, payload: dict) -> None:
    data = canonical(payload) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _load_prelaunch(path: Path, output: Path) -> tuple[dict, str, str]:
    if path.resolve() != DEFAULT_PRELAUNCH.resolve():
        raise ValueError("prelaunch receipt path differs")
    raw = path.read_bytes()
    receipt = json.loads(raw)
    payload = receipt.get("payload")
    if (set(receipt) != {"payload", "digest"} or not isinstance(payload, dict)):
        raise ValueError("prelaunch envelope is incomplete")
    if receipt["digest"] != hashlib.sha256(
            canonical(payload)).hexdigest():
        raise ValueError("prelaunch digest differs")
    recovery.validate_prelaunch_shape(payload)
    if (payload.get("schema") != recovery.PRELAUNCH_SCHEMA
            or payload.get("passed") is not True
            or payload.get("launch_allowed") is not True
            or payload.get("logical_fit_id") != LOGICAL_FIT_ID
            or payload.get("sole_output") != SOLE_OUTPUT
            or payload.get("package_versions") != VERSIONS
            or output.resolve() != DEFAULT_OUTPUT.resolve()):
        raise ValueError("prelaunch does not authorize the fixed fit/output")
    budget = payload.get("budget_proof")
    if not isinstance(budget, dict) or set(budget) != {"path", "sha256"}:
        raise ValueError("prelaunch budget proof pin is missing")
    budget_path = ROOT / budget["path"]
    if (budget_path.resolve() != QUALIFIED_BUDGET.resolve()
            or budget["sha256"] != QUALIFIED_BUDGET_SHA256
            or sha256_file(budget_path) != QUALIFIED_BUDGET_SHA256):
        raise ValueError("budget proof pin differs")
    proof = json.loads(budget_path.read_bytes())
    proof_payload = proof.get("payload", proof)
    if (proof.get("schema") != "difference-pairwise-lora-budget-proof-v1"
            or proof.get("digest") != hashlib.sha256(canonical(proof_payload)).hexdigest()
            or proof_payload.get("stage") != "budget_qualification"
            or proof_payload.get("passed") is not True
            or proof_payload.get("launch_allowed") is not True
            or proof_payload.get("logical_fit_id") != "difference-pairwise-lora-fit-budget-v1"
            or proof_payload.get("sole_output") != ".omx/artifacts/difference-pairwise-lora-train-v1"
            or proof_payload.get("old_policy", {}).get("launch_allowed") is not False
            or proof_payload.get("new_policy", {}).get("limit_seconds") != 21600):
        raise ValueError("budget proof does not authorize this fit")
    source_inventory = proof_payload.get("source_inventory")
    hashes = source_inventory.get("hashes") if isinstance(source_inventory, dict) else None
    if (not isinstance(hashes, dict) or len(hashes) != 48
            or source_inventory.get("count") != 48
            or source_inventory.get("sha256") != hashlib.sha256(canonical(hashes)).hexdigest()):
        raise ValueError("budget source inventory differs")
    recovery_pin = payload.get("recovery_proof")
    if (not isinstance(recovery_pin, dict) or set(recovery_pin) != {"path", "sha256"}
            or (ROOT / recovery_pin["path"]).resolve() != QUALIFIED_RECOVERY.resolve()
            or sha256_file(QUALIFIED_RECOVERY) != recovery_pin["sha256"]):
        raise ValueError("recovery proof pin differs")
    recovery_payload = recovery.validate_recovery_proof(QUALIFIED_RECOVERY)
    bound_artifacts = {
        recovery_pin["path"], recovery_payload["preflight"]["path"],
        recovery_payload["checks"]["path"], recovery_payload["execution_review"]["path"]}
    inventory = payload.get("inventory")
    expected = set(hashes) | EXECUTION_ADDITIONS | set(recovery_payload["execution_sources"]) | {
        budget["path"], *bound_artifacts}
    if not isinstance(inventory, dict) or set(inventory) != expected:
        raise ValueError("execution inventory is incomplete or has unbound entries")
    if any(inventory.get(name) != digest for name, digest in hashes.items()):
        raise ValueError("execution inventory differs from budget source pins")
    for name, digest in inventory.items():
        if not isinstance(name, str) or not isinstance(digest, str) or sha256_file(ROOT / name) != digest:
            raise ValueError(f"execution inventory differs: {name}")
    worker_python = payload.get("worker_python")
    if worker_python != recovery_payload["preflight"]["interpreter"]:
        raise ValueError("worker interpreter differs from qualified preflight")
    return payload, hashlib.sha256(raw).hexdigest(), budget["sha256"]


def build_execution_prelaunch(output=DEFAULT_PRELAUNCH) -> dict:
    """Build the staged v2 prelaunch after recovery proof and review exist."""
    if Path(output).resolve() != DEFAULT_PRELAUNCH.resolve():
        raise ValueError("prelaunch output path differs")
    qualified = recovery.validate_recovery_proof(QUALIFIED_RECOVERY)
    old_budget = json.loads(QUALIFIED_BUDGET.read_bytes())["payload"]
    recovery_path = str(QUALIFIED_RECOVERY.relative_to(ROOT))
    base = old_budget["source_inventory"]["hashes"]
    artifact_paths = {qualified["preflight"]["path"], qualified["checks"]["path"],
                      qualified["execution_review"]["path"], recovery_path,
                      str(QUALIFIED_BUDGET.relative_to(ROOT))}
    names = set(base) | EXECUTION_ADDITIONS | set(qualified["execution_sources"]) | artifact_paths
    inventory = {name: sha256_file(ROOT / name) for name in sorted(names)}
    payload = {"schema": recovery.PRELAUNCH_SCHEMA, "passed": True, "launch_allowed": True,
               "logical_fit_id": LOGICAL_FIT_ID, "sole_output": SOLE_OUTPUT,
               "package_versions": VERSIONS,
               "worker_python": qualified["preflight"]["interpreter"],
               "budget_proof": {"path": str(QUALIFIED_BUDGET.relative_to(ROOT)),
                                "sha256": sha256_file(QUALIFIED_BUDGET)},
               "recovery_proof": {"path": recovery_path,
                                  "sha256": sha256_file(QUALIFIED_RECOVERY)},
               "inventory": inventory}
    receipt = {"payload": payload, "digest": hashlib.sha256(canonical(payload)).hexdigest()}
    recovery.write_once(Path(output), receipt)
    return receipt


def claim_launch(output: Path, prelaunch: Path, *, now=None, claim_factory=None) -> dict:
    """Create the irreversible per-output launch receipt before model imports."""
    payload, prelaunch_sha, budget_sha = _load_prelaunch(prelaunch, output)
    output.mkdir(parents=True, exist_ok=True)
    claim = {
        "schema": "difference-pairwise-lora-launch-v2",
        "logical_fit_id": LOGICAL_FIT_ID,
        "sole_output": SOLE_OUTPUT,
        "prelaunch": {"path": str(prelaunch.resolve().relative_to(ROOT)),
                      "sha256": prelaunch_sha},
        "budget_proof_sha256": budget_sha,
        "recovery_proof_sha256": payload["recovery_proof"]["sha256"],
        "worker_claim": (claim_factory or (lambda: str(uuid.uuid4())))(),
        "created_at": (now or (lambda: datetime.now(timezone.utc).isoformat()))(),
        "consumed": True,
        "released": False,
    }
    if not isinstance(claim["worker_claim"], str) or not claim["worker_claim"]:
        raise ValueError("worker claim is empty")
    _write_new_durable(output / "launch.json", claim)
    return claim


def _write_external_receipt(output: Path, payload: dict) -> None:
    _write_new_durable(output / "external-execution.json", payload)


def validate_runtime(python: str, expected: dict | None = None) -> dict:
    invocation = Path(python).absolute()
    code = ("import importlib.metadata,json,os,sys; print(json.dumps({"
            "'invocation_path':os.path.abspath(sys.executable),'resolved_path':os.path.realpath(sys.executable),"
            "'packages':{name:importlib.metadata.version(name) "
            "for name in ['mlx','mlx-lm','transformers']}}))")
    result = subprocess.run([str(invocation), "-c", code], capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise ValueError("worker Python environment lacks the qualified packages")
    observed = json.loads(result.stdout)
    packages = observed.get("packages")
    if packages != VERSIONS:
        raise ValueError("worker Python package versions differ")
    resolved = Path(observed.get("resolved_path", ""))
    identity = {"invocation_path": observed.get("invocation_path"),
                "resolved_path": str(resolved), "resolved_sha256": sha256_file(resolved),
                "package_versions": packages}
    if expected is not None and identity != expected:
        raise ValueError("worker Python differs from qualified interpreter")
    return identity


def validate_resource_runtime(python: str) -> dict:
    code = ("import json; from scripts.train_difference_pairwise_lora import check_pre_model_resources; "
            "print(json.dumps(check_pre_model_resources()))")
    result = subprocess.run([python, "-c", code], cwd=str(ROOT), capture_output=True,
                            text=True, timeout=30)
    if result.returncode != 0:
        raise ValueError("worker resource preflight failed: " + result.stderr[-2000:])
    record = json.loads(result.stdout)
    if record.get("stage") != "prelaunch_resource_check" or record.get("errors") != []:
        raise ValueError("worker resource preflight evidence invalid")
    return record


def run_supervised(output: Path, prelaunch: Path, *, python: str = sys.executable,
                   popen_factory=subprocess.Popen, monotonic=time.monotonic,
                   wall_time=None) -> int:
    """Consume one claim, run one worker, and record authoritative exit evidence."""
    prelaunch_payload, _, _ = _load_prelaunch(prelaunch, output)
    validate_runtime(python, prelaunch_payload["worker_python"])
    validate_resource_runtime(python)
    launch = claim_launch(output, prelaunch, now=wall_time)
    started = monotonic()
    command = [python, str(ROOT / "scripts/train_difference_pairwise_lora.py"),
               "--output", str(output), "--prelaunch", str(prelaunch),
               "--launch", str(output / "launch.json"),
               "--fit-start-monotonic", repr(started),
               "--worker-claim", launch["worker_claim"]]
    process = None
    deadline_terminated = False
    spawn_error = None
    exit_status = None
    try:
        process = popen_factory(command, cwd=str(ROOT))
        try:
            exit_status = process.wait(timeout=max(0.0, FIT_LIMIT_SECONDS - (monotonic() - started)))
        except subprocess.TimeoutExpired:
            deadline_terminated = True
            process.kill()
            exit_status = process.wait()
    except BaseException as error:
        spawn_error = f"{type(error).__name__}: {error}"
        if process is not None:
            try:
                process.kill()
            except BaseException:
                pass
            try:
                exit_status = process.wait()
            except BaseException as reap_error:
                spawn_error += f"; reap failed: {type(reap_error).__name__}: {reap_error}"
    ended = monotonic()
    elapsed = ended - started
    if elapsed > FIT_LIMIT_SECONDS and process is not None and exit_status is None:
        deadline_terminated = True
        process.kill()
        exit_status = process.wait()
        ended, elapsed = monotonic(), monotonic() - started
    report_path = output / "report.json"
    report_sha = sha256_file(report_path) if report_path.is_file() else None
    report = None
    if report_sha is not None:
        try:
            report = json.loads(report_path.read_bytes())
        except (OSError, ValueError, TypeError):
            report = None
    valid_report = (isinstance(report, dict)
                    and report.get("schema") == "difference-pairwise-lora-training-v2"
                    and report.get("logical_fit_id") == LOGICAL_FIT_ID
                    and report.get("sole_output") == SOLE_OUTPUT
                    and report.get("completed") is True
                    and report.get("execution_valid") is True)
    worker_interval_valid = (exit_status in (0, 2) and not deadline_terminated
                             and elapsed <= FIT_LIMIT_SECONDS and valid_report)
    authoritative_success = (worker_interval_valid and exit_status == 0
                             and report.get("quality_passed") is True
                             and report.get("terminal_status") == "quality_passed")
    quality_rejected = (worker_interval_valid and exit_status == 2
                        and report.get("quality_passed") is False
                        and report.get("terminal_status") == "quality_rejected")
    receipt = {
        "schema": "difference-pairwise-lora-external-execution-v2",
        "logical_fit_id": LOGICAL_FIT_ID,
        "sole_output": SOLE_OUTPUT,
        "launch_sha256": sha256_file(output / "launch.json"),
        "prelaunch_sha256": launch["prelaunch"]["sha256"],
        "budget_proof_sha256": launch["budget_proof_sha256"],
        "recovery_proof_sha256": launch["recovery_proof_sha256"],
        "worker_claim": launch["worker_claim"],
        "worker_pid": getattr(process, "pid", None),
        "fit_start_monotonic": started,
        "fit_end_monotonic": ended,
        "fit_elapsed_seconds": elapsed,
        "fit_limit_seconds": FIT_LIMIT_SECONDS,
        "worker_exit_status": exit_status,
        "deadline_terminated": deadline_terminated or elapsed > FIT_LIMIT_SECONDS,
        "spawn_error": spawn_error,
        "terminal_report_sha256": report_sha,
        "worker_interval_valid": worker_interval_valid,
        "quality_rejected": quality_rejected,
        "authoritative_success": authoritative_success,
        "additional_model_work_authorized": False,
    }
    _write_external_receipt(output, receipt)
    if authoritative_success:
        return 0
    if quality_rejected:
        return 2
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prelaunch", type=Path, default=DEFAULT_PRELAUNCH)
    args = parser.parse_args()
    try:
        return run_supervised(args.output, args.prelaunch)
    except BaseException as error:
        print(json.dumps({"error": f"{type(error).__name__}: {error}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
