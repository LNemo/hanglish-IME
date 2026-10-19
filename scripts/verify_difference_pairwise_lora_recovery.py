"""Qualify one v2 replacement after the immutable v1 pre-model failure.

This module never imports model weights.  It independently replays the failed
execution and original budget, produces model-free preflight/check receipts,
and emits a recovery proof only after reviewed evidence is complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_difference_pairwise_lora_budget as budget


SCHEMA = "difference-pairwise-lora-recovery-proof-v1"
PREFLIGHT_SCHEMA = "difference-pairwise-lora-premodel-preflight-v2"
CHECKS_SCHEMA = "difference-pairwise-lora-recovery-checks-v1"
LOGICAL_FIT_ID = "difference-pairwise-lora-fit-budget-v2"
SOLE_OUTPUT = ".omx/artifacts/difference-pairwise-lora-train-v2"
PRELAUNCH_SCHEMA = "difference-pairwise-lora-execution-prelaunch-v2"
PRELAUNCH_FIELDS = frozenset({"schema", "passed", "launch_allowed", "logical_fit_id",
    "sole_output", "package_versions", "worker_python", "budget_proof",
    "recovery_proof", "inventory"})
PRELAUNCH_PATH = ".omx/artifacts/difference-pairwise-lora-execution-prelaunch-v2.json"
RECOVERY_PROOF_PATH = ".omx/artifacts/difference-pairwise-lora-recovery-v1-proof.json"
PREFLIGHT_PATH = ".omx/artifacts/difference-pairwise-lora-premodel-preflight-v2.json"
CHECKS_PATH = ".omx/artifacts/difference-pairwise-lora-recovery-checks-v1.json"
EXECUTION_REVIEW_PATH = ".omx/plans/pairwise-fit-execution-review-v2.md"
V1_OUTPUT = ROOT / ".omx/artifacts/difference-pairwise-lora-train-v1"
ARCHIVE = ROOT / ".omx/artifacts/difference-pairwise-lora-train-v1-failure-sources/manifest.json"
FAILURE_AUDIT = ROOT / ".omx/artifacts/pairwise-premodel-failure-audit-v1.md"
BUDGET_PROOF = ROOT / ".omx/artifacts/difference-pairwise-lora-budget-v1-proof.json"
PLAN = ROOT / ".omx/plans/pairwise-premodel-recovery-v1.md"
PLAN_REVIEW = ROOT / ".omx/plans/pairwise-premodel-recovery-review-v1.md"
FIXED_HASHES = {
    str(PLAN.relative_to(ROOT)): "7790fc0032db005c1c433d7745027ecff7f730f89354f202465a52ef9d9c40c5",
    str(PLAN_REVIEW.relative_to(ROOT)): "bff4c28df297988985aa22ef4e3dfd21b1b7e5b51aea1473caf4b03865669aa6",
    str(FAILURE_AUDIT.relative_to(ROOT)): "0e2e9f91d75f8bf41589ea02978c1980bc0e18eb87fad8f8989cc02071ce5f1c",
    str(ARCHIVE.relative_to(ROOT)): "dcdc3d92ab15b0defe692dca231c351122a8bf2ead106ccd7dc36bcb900ee325",
    str(BUDGET_PROOF.relative_to(ROOT)): "e59fe3cf9293f2df7cfa19e7e9e198f4691489d41c11215efb19877141888d6a",
}
V1_RAW_HASHES = {
    "launch.json": "393d271807799b15903b6452b84d985a04b4f890dd3840be1a622d9032196b16",
    "worker-start.json": "24c8bd6382a89a62c67c0b1a5fbe0ae23f229c5f56ec203eefd52d6206ac7fe4",
    "report.json": "cc67e3f13d2dcac6754a693a0235ef4b9a400b4df6ed9416e21f70989a3ee25b",
    "external-execution.json": "09895fa5879591b99e13d73152c4bba326f737a8472732ed3f5b4fca7b093117",
    "progress.jsonl": "a38eeb2c1ededf424cb77dd8508fd7f6bb68096923c3f093f573cc79b3581858",
}
EXECUTION_SOURCES = {
    "scripts/long_context_lora_resources.py",
    "scripts/probe_long_context_lora_resources.py",
    "scripts/run_difference_pairwise_lora.py",
    "scripts/train_difference_pairwise_lora.py",
    "scripts/verify_difference_pairwise_lora_training.py",
    "scripts/verify_difference_pairwise_lora_recovery.py",
    "tests/test_run_difference_pairwise_lora.py",
    "tests/test_pairwise_resource_binding.py",
    "tests/test_train_difference_pairwise_lora.py",
    "tests/test_verify_difference_pairwise_lora_training.py",
    "tests/test_verify_difference_pairwise_lora_recovery.py",
}


class RecoveryError(ValueError):
    pass


def canonical(value) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise RecoveryError("value is not canonical JSON") from error


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path: Path) -> str:
    value = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(chunk)
    except OSError as error:
        raise RecoveryError(f"required file unavailable: {path}") from error
    return value.hexdigest()


def need(condition, message):
    if not condition:
        raise RecoveryError(message)


def validate_prelaunch_shape(payload):
    need(isinstance(payload, dict) and set(payload) == PRELAUNCH_FIELDS,
         "prelaunch fields differ from the canonical recovery contract")


def read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise RecoveryError(f"invalid JSON: {path}") from error
    need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_once(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    parent = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def source_hashes(paths=None) -> dict[str, str]:
    if paths is None:
        paths = EXECUTION_SOURCES
    return {name: sha(ROOT / name) for name in sorted(paths)}


def verify_v1_failure() -> dict:
    for name, expected in FIXED_HASHES.items():
        need(sha(ROOT / name) == expected, f"fixed recovery evidence changed: {name}")
    entries = list(V1_OUTPUT.iterdir())
    actual_files = {path.name for path in entries}
    need(actual_files == set(V1_RAW_HASHES)
         and all(path.is_file() and not path.is_symlink() for path in entries),
         "v1 output is not the exact five-file pre-model failure")
    for name, expected in V1_RAW_HASHES.items():
        need(sha(V1_OUTPUT / name) == expected, f"v1 raw evidence changed: {name}")
    launch = read_json(V1_OUTPUT / "launch.json")
    worker = read_json(V1_OUTPUT / "worker-start.json")
    report = read_json(V1_OUTPUT / "report.json")
    external = read_json(V1_OUTPUT / "external-execution.json")
    progress = [json.loads(line) for line in (V1_OUTPUT / "progress.jsonl").read_bytes().splitlines()]
    claim = launch.get("worker_claim")
    need(launch.get("logical_fit_id") == "difference-pairwise-lora-fit-budget-v1"
         and launch.get("sole_output") == ".omx/artifacts/difference-pairwise-lora-train-v1"
         and launch.get("consumed") is True and launch.get("released") is False
         and isinstance(claim, str) and claim, "v1 launch claim differs")
    need(worker.get("worker_claim") == report.get("worker_claim") == external.get("worker_claim") == claim
         and worker.get("worker_pid") == external.get("worker_pid")
         and worker.get("launch_sha256") == V1_RAW_HASHES["launch.json"],
         "v1 receipts do not bind one worker")
    need(report.get("completed") is False and report.get("execution_valid") is False
         and report.get("failure_type") == "AttributeError"
         and report.get("failure") == "module 'scripts.probe_long_context_lora_resources' has no attribute 'current_rss_bytes'"
         and report.get("fresh_base_initialized") is False
         and report.get("real_training_run") is False
         and report.get("units") == [] and report.get("updates") == []
         and report.get("resource_observations") == [] and report.get("predictions") == {}
         and report.get("quality") is None and report.get("eligibility") == {
             "evaluation": False, "resume": False, "deploy": False},
         "v1 report is not the proven pre-model failure")
    need(external.get("worker_exit_status") == 1
         and external.get("worker_interval_valid") is False
         and external.get("authoritative_success") is False
         and external.get("additional_model_work_authorized") is False
         and external.get("terminal_report_sha256") == V1_RAW_HASHES["report.json"]
         and external.get("fit_elapsed_seconds") == 2.381744166000317,
         "v1 external terminal evidence differs")
    need(len(progress) == 1 and progress[0].get("event") == "terminal"
         and progress[0].get("exit_code") == 1, "v1 progress is not terminal")
    archive = read_json(ARCHIVE)
    need(archive.get("schema") == "pairwise-fit-premodel-failure-snapshot-v1"
         and archive.get("execution_artifacts") == V1_RAW_HASHES
         and archive.get("model_loaded") is False and archive.get("optimizer_updates") == 0
         and archive.get("claim_released") is False
         and archive.get("retry_authorized_by_original_plan") is False,
         "v1 source archive contract differs")
    archive_root = ARCHIVE.parent
    for name, expected in archive.get("files", {}).items():
        need(sha(archive_root / name) == expected, f"archived v1 source changed: {name}")
    return {"logical_fit_id": launch["logical_fit_id"], "sole_output": launch["sole_output"],
            "raw_files": dict(V1_RAW_HASHES), "archive_manifest_sha256": sha(ARCHIVE),
            "failure_type": report["failure_type"], "failure": report["failure"],
            "elapsed_seconds": external["fit_elapsed_seconds"], "model_loaded": False,
            "optimizer_updates": 0, "adapter_created": False, "terminal": True}


def verify_old_budget() -> dict:
    proof = read_json(BUDGET_PROOF)
    expected = budget.verify_budget()
    need(proof.get("schema") == budget.SCHEMA and proof.get("payload") == expected
         and proof.get("digest") == budget.digest_value(expected), "original budget replay differs")
    need(expected.get("logical_fit_id") == "difference-pairwise-lora-fit-budget-v1"
         and expected.get("second_fit_authorized") is False,
         "original budget cannot authorize a replacement")
    return {"path": str(BUDGET_PROOF.relative_to(ROOT)), "sha256": sha(BUDGET_PROOF),
            "estimate_seconds": expected["new_policy"]["estimate_seconds"],
            "limit_seconds": expected["new_policy"]["limit_seconds"],
            "replacement_authorized": False}


def _interpreter(python: str, runner=subprocess.run) -> dict:
    invocation = Path(python).absolute()
    need(invocation.is_file(), "worker interpreter invocation is unavailable")
    code = ("import importlib.metadata,json,os,sys; "
            "print(json.dumps({'invocation_path':os.path.abspath(sys.executable),"
            "'resolved_path':os.path.realpath(sys.executable),'packages':{n:importlib.metadata.version(n) "
            "for n in ['mlx','mlx-lm','transformers']}}))")
    result = runner([str(invocation), "-c", code], capture_output=True, text=True, timeout=30)
    need(result.returncode == 0, "worker interpreter inspection failed")
    value = json.loads(result.stdout)
    invocation_path = Path(value.get("invocation_path", ""))
    resolved_path = Path(value.get("resolved_path", ""))
    need(invocation_path == invocation and invocation_path.is_file()
         and resolved_path == Path(os.path.realpath(invocation)) and resolved_path.is_file()
         and value.get("packages") == budget.VERSIONS,
         "worker interpreter or packages differ")
    return {"invocation_path": str(invocation_path), "resolved_path": str(resolved_path),
            "resolved_sha256": sha(resolved_path), "package_versions": value["packages"]}


def _validate_resource_record(record: dict) -> dict:
    need(isinstance(record, dict) and set(record) == {
             "stage", "action", "pre", "post", "errors", "cleanup_seconds",
             "checkpoint_seconds"}
         and record.get("stage") == "prelaunch_resource_check"
         and record.get("errors") == [] and record.get("action") in {
             "none", "clear_unused_mlx_cache"}, "resource preflight evidence differs")
    names = ("rss_peak_bytes", "mlx_active_bytes", "mlx_cache_bytes", "mlx_peak_bytes",
             "free_percent", "swap_used_bytes", "swap_growth_bytes", "elapsed_seconds")
    need(isinstance(record.get("pre"), dict)
         and (record.get("post") is None if record["action"] == "none"
              else isinstance(record.get("post"), dict))
         and type(record.get("cleanup_seconds")) in (int, float)
         and math.isfinite(record["cleanup_seconds"]) and record["cleanup_seconds"] >= 0
         and type(record.get("checkpoint_seconds")) in (int, float)
         and math.isfinite(record["checkpoint_seconds"]) and record["checkpoint_seconds"] >= 0
         , "resource preflight metrics are invalid or exceed frozen caps")
    observations = [record["pre"]] + ([record["post"]] if record["post"] is not None else [])
    need(all(set(observed) == set(names)
             and all(type(observed.get(name)) in (int, float) and math.isfinite(observed[name])
                     for name in names)
             and observed["rss_peak_bytes"] <= budget.LIMITS["rss_bytes"]
             and observed["mlx_peak_bytes"] <= budget.LIMITS["mlx_peak_bytes"]
             and observed["free_percent"] >= budget.LIMITS["minimum_system_free_percent"]
             and observed["swap_growth_bytes"] <= budget.LIMITS["maximum_swap_growth_bytes"]
             and observed["elapsed_seconds"] <= budget.LIMITS["whole_seconds"]
             for observed in observations)
         and (record["action"] == "none") == (
             record["pre"]["mlx_cache_bytes"] <= budget.LIMITS["cache_bytes"])
         and observations[-1]["mlx_cache_bytes"] <= budget.LIMITS["cache_bytes"],
         "resource preflight metrics are invalid or exceed frozen caps")
    return observations[-1]


def _validate_check_commands(commands: list[dict]) -> None:
    need(isinstance(commands, list) and len(commands) >= 2
         and all(isinstance(row, dict) and set(row) == {
                     "argv", "returncode", "stdout_sha256", "stderr_sha256"}
                 and isinstance(row.get("argv"), list)
                 and row["argv"] and all(isinstance(value, str) and value for value in row["argv"])
                 and row.get("returncode") == 0
                 and all(isinstance(row.get(name), str) and len(row[name]) == 64
                         and set(row[name]) <= set("0123456789abcdef")
                         for name in ("stdout_sha256", "stderr_sha256")) for row in commands),
         "recovery check command evidence differs")
    need([row["argv"] for row in commands] == default_check_commands(),
         "canonical recovery test/compile commands differ")


def generate_premodel_preflight(output: Path, python: str, *, runner=subprocess.run,
                                resource_probe=None) -> dict:
    before = source_hashes()
    interpreter = _interpreter(python, runner=runner)
    if resource_probe is None:
        code = ("import json; from scripts.train_difference_pairwise_lora import "
                "check_pre_model_resources; print(json.dumps(check_pre_model_resources()))")
        result = runner([interpreter["invocation_path"], "-c", code], cwd=str(ROOT), capture_output=True,
                        text=True, timeout=30)
        need(result.returncode == 0, "model-free resource preflight failed")
        record = json.loads(result.stdout)
    else:
        record = resource_probe()
    interpreter_after = _interpreter(str(Path(python).absolute()), runner=runner)
    after = source_hashes()
    need(before == after and interpreter == interpreter_after,
         "execution sources or worker interpreter changed during preflight")
    _validate_resource_record(record)
    payload = {"schema": PREFLIGHT_SCHEMA, "passed": True,
               "logical_fit_id": LOGICAL_FIT_ID, "sole_output": SOLE_OUTPUT,
               "interpreter": interpreter, "interpreter_after": interpreter_after,
               "resource_observation": record,
               "source_hashes_before": before, "source_hashes_after": after,
               "tokenizer_loaded": False, "model_weights_loaded": False,
               "model_work_started": False, "claim_consumed": False}
    receipt = {"schema": PREFLIGHT_SCHEMA, "payload": payload, "digest": digest(payload)}
    write_once(output, receipt)
    return receipt


def validate_premodel_preflight(path: Path) -> dict:
    receipt = read_json(path); payload = receipt.get("payload")
    need(set(receipt) == {"schema", "payload", "digest"}
         and receipt.get("schema") == PREFLIGHT_SCHEMA and isinstance(payload, dict)
         and set(payload) == {"schema", "passed", "logical_fit_id", "sole_output",
             "interpreter", "interpreter_after", "resource_observation",
             "source_hashes_before", "source_hashes_after", "tokenizer_loaded",
             "model_weights_loaded", "model_work_started", "claim_consumed"}
         and receipt.get("digest") == digest(payload)
         and payload.get("schema") == PREFLIGHT_SCHEMA and payload.get("passed") is True
         and payload.get("logical_fit_id") == LOGICAL_FIT_ID
         and payload.get("sole_output") == SOLE_OUTPUT
         and payload.get("interpreter") == payload.get("interpreter_after")
         and payload.get("source_hashes_before") == payload.get("source_hashes_after") == source_hashes()
         and payload.get("tokenizer_loaded") is False
         and payload.get("model_weights_loaded") is False
         and payload.get("model_work_started") is False
         and payload.get("claim_consumed") is False, "pre-model preflight receipt differs")
    interpreter = payload.get("interpreter", {})
    invocation = Path(interpreter.get("invocation_path", ""))
    binary = Path(interpreter.get("resolved_path", ""))
    need(set(interpreter) == {"invocation_path", "resolved_path", "resolved_sha256",
                             "package_versions"}
         and invocation.is_absolute() and invocation.is_file()
         and Path(os.path.realpath(invocation)) == binary
         and binary.is_absolute() and binary.is_file()
         and sha(binary) == interpreter.get("resolved_sha256")
         and interpreter.get("package_versions") == budget.VERSIONS,
         "preflight interpreter binding differs")
    _validate_resource_record(payload.get("resource_observation"))
    return payload


def run_check_commands(output: Path, commands: list[list[str]], *, runner=subprocess.run) -> dict:
    need(commands == default_check_commands(), "canonical recovery test/compile commands differ")
    need(isinstance(commands, list) and len(commands) >= 2
         and all(isinstance(command, list) and command
                 and all(isinstance(value, str) and value for value in command) for command in commands),
         "at least two argv-form check commands are required")
    before = source_hashes(); results = []
    for command in commands:
        result = runner(command, cwd=str(ROOT), capture_output=True, text=True)
        results.append({"argv": command, "returncode": result.returncode,
                        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
                        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest()})
    after = source_hashes()
    need(before == after and all(row["returncode"] == 0 for row in results),
         "recovery checks failed or changed bound sources")
    _validate_check_commands(results)
    payload = {"schema": CHECKS_SCHEMA, "passed": True, "commands": results,
               "source_hashes_before": before, "source_hashes_after": after,
               "model_weights_loaded": False, "model_work_started": False,
               "v2_claim_consumed": False}
    receipt = {"schema": CHECKS_SCHEMA, "payload": payload, "digest": digest(payload)}
    write_once(output, receipt)
    return receipt


def default_check_commands() -> list[list[str]]:
    return [
        [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q",
         "tests/test_verify_difference_pairwise_lora_recovery.py",
         "tests/test_run_difference_pairwise_lora.py",
         "tests/test_train_difference_pairwise_lora.py",
         "tests/test_verify_difference_pairwise_lora_training.py",
         "tests/test_pairwise_resource_binding.py"],
        [str(ROOT / ".venv-mlx/bin/python"), "-m", "py_compile",
         "scripts/verify_difference_pairwise_lora_recovery.py",
         "scripts/run_difference_pairwise_lora.py",
         "scripts/train_difference_pairwise_lora.py",
         "scripts/verify_difference_pairwise_lora_training.py"],
    ]


def validate_check_receipt(path: Path) -> dict:
    receipt = read_json(path); payload = receipt.get("payload")
    need(set(receipt) == {"schema", "payload", "digest"}
         and receipt.get("schema") == CHECKS_SCHEMA and isinstance(payload, dict)
         and set(payload) == {"schema", "passed", "commands", "source_hashes_before",
             "source_hashes_after", "model_weights_loaded", "model_work_started",
             "v2_claim_consumed"}
         and payload.get("schema") == CHECKS_SCHEMA
         and receipt.get("digest") == digest(payload) and payload.get("passed") is True
         and payload.get("source_hashes_before") == payload.get("source_hashes_after") == source_hashes()
         and payload.get("model_weights_loaded") is False
         and payload.get("model_work_started") is False
         and payload.get("v2_claim_consumed") is False,
         "recovery check receipt differs")
    _validate_check_commands(payload.get("commands"))
    return payload


def validate_execution_review(path: Path, sources: dict) -> dict:
    try:
        text = Path(path).read_text()
    except OSError as error:
        raise RecoveryError("execution review unavailable") from error
    need("Verdict: CLEAR" in text and LOGICAL_FIT_ID in text and SOLE_OUTPUT in text
         and all(value in text for value in sources.values()),
         "independent execution review does not bind every implementation source")
    return {"path": str(Path(path).resolve().relative_to(ROOT)), "sha256": sha(path),
            "verdict": "CLEAR"}


def qualify_recovery(preflight=ROOT / PREFLIGHT_PATH, checks=ROOT / CHECKS_PATH,
                     execution_review=ROOT / EXECUTION_REVIEW_PATH) -> dict:
    sources = source_hashes()
    failure = verify_v1_failure()
    original_budget = verify_old_budget()
    preflight_payload = validate_premodel_preflight(preflight)
    validate_check_receipt(checks)
    review = validate_execution_review(execution_review, sources)
    need(source_hashes() == sources, "execution sources changed during recovery qualification")
    return {"stage": "premodel_recovery_qualification", "passed": True,
            "launch_allowed": True, "logical_fit_id": LOGICAL_FIT_ID,
            "sole_output": SOLE_OUTPUT, "replacement_count": 1,
            "second_replacement_authorized": False, "resume_authorized": False,
            "recipe_changed": False, "threshold_changed": False,
            "v1_failure": failure, "original_budget": original_budget,
            "plan": {"path": str(PLAN.relative_to(ROOT)), "sha256": sha(PLAN)},
            "plan_review": {"path": str(PLAN_REVIEW.relative_to(ROOT)),
                            "sha256": sha(PLAN_REVIEW), "verdict": "CLEAR"},
            "failure_audit": {"path": str(FAILURE_AUDIT.relative_to(ROOT)),
                              "sha256": sha(FAILURE_AUDIT)},
            "preflight": {"path": str(Path(preflight).resolve().relative_to(ROOT)),
                          "sha256": sha(preflight),
                          "interpreter": preflight_payload["interpreter"]},
            "checks": {"path": str(Path(checks).resolve().relative_to(ROOT)),
                       "sha256": sha(checks)}, "execution_review": review,
            "execution_sources": sources, "quality_claimed": False,
            "deployment_authorized": False, "model_weights_loaded": False,
            "v2_claim_consumed": False, "execution_prelaunch_required": True}


def write_recovery_proof(output: Path, **kwargs) -> dict:
    payload = qualify_recovery(**kwargs)
    receipt = {"schema": SCHEMA, "payload": payload, "digest": digest(payload)}
    write_once(output, receipt)
    return receipt


def validate_recovery_proof(path=ROOT / RECOVERY_PROOF_PATH) -> dict:
    receipt = read_json(path); expected = qualify_recovery()
    need(receipt.get("schema") == SCHEMA and receipt.get("payload") == expected
         and receipt.get("digest") == digest(expected), "recovery proof replay differs")
    return expected


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    preflight = sub.add_parser("preflight"); preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--python", required=True)
    checks = sub.add_parser("checks"); checks.add_argument("--output", type=Path, required=True)
    qualify = sub.add_parser("qualify"); qualify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.mode == "preflight":
            generate_premodel_preflight(args.output, args.python)
        elif args.mode == "checks":
            run_check_commands(args.output, default_check_commands())
        else:
            write_recovery_proof(args.output)
        print(json.dumps({"passed": True, "mode": args.mode}))
        return 0
    except (RecoveryError, OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({"passed": False, "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
