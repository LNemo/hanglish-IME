"""One-shot worker for the approved balanced pairwise KEEP/SWITCH LoRA fit."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import difference_pairwise_lora_evaluation as evaluation
from scripts import difference_pairwise_lora_model as pairwise
from scripts import long_context_lora_model as shared
from scripts import probe_long_context_lora_resources as system
from scripts import verify_difference_pairwise_lora_recovery as recovery
from scripts.long_context_lora_resources import ResourceObservationError, observe_resources


LOGICAL_FIT_ID = recovery.LOGICAL_FIT_ID
SOLE_OUTPUT = recovery.SOLE_OUTPUT
DATA = ROOT / ".omx/artifacts/difference-pairwise-lora-data-v1"
DATA_PROOF = ROOT / ".omx/artifacts/difference-pairwise-lora-data-v1-proof.json"
MODEL = ROOT / "data/external/qwen3-4b-mlx-4bit"
DEFAULT_PRELAUNCH = ROOT / recovery.PRELAUNCH_PATH
QUALIFIED_BUDGET = ROOT / ".omx/artifacts/difference-pairwise-lora-budget-v1-proof.json"
QUALIFIED_BUDGET_SHA256 = "e59fe3cf9293f2df7cfa19e7e9e198f4691489d41c11215efb19877141888d6a"
MANIFEST_SHA = "32e83b145e71de39912e122f9925c9cf362875a5206e49dddcb7611a464be132"
DATA_PROOF_SHA = "03943e073ec333b4e8a8e8386ace042c0c3dda6370cdc5b284961262bd2dea72"
VERSIONS = {"mlx": "0.31.2", "mlx-lm": "0.31.3", "transformers": "5.12.1"}
LIMITS = {"rss_bytes": 6 * 1024**3, "mlx_peak_bytes": 6 * 1024**3,
          "cache_bytes": 256 * 1024**2, "whole_seconds": 21600.0,
          "update_seconds": 120.0, "minimum_system_free_percent": 10.0,
          "maximum_swap_growth_bytes": 512 * 1024**2}
EXPECTED = {"units": 59, "components": 232, "updates": 15,
            "mechanism_rows": 175, "E_mechanism": 481,
            "train640_rows": 640, "E_train640": 1556}
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
    return evaluation.canonical(value)


def file_sha(path: Path) -> str:
    return system.sha256_file(Path(path))


def key(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def _durable_replace(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(canonical(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    with Path(path).open("rb") as stream:
        os.fsync(stream.fileno())
    descriptor = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def claim_worker_start(output: Path, launch_sha256: str, worker_claim: str,
                       *, pid: int | None = None) -> dict:
    receipt = {"schema": "difference-pairwise-lora-worker-start-v2",
               "logical_fit_id": LOGICAL_FIT_ID, "sole_output": SOLE_OUTPUT,
               "launch_sha256": launch_sha256, "worker_claim": worker_claim,
               "worker_pid": os.getpid() if pid is None else pid,
               "consumed": True, "released": False}
    path = output / "worker-start.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(receipt) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    directory = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return receipt


class ProgressLog:
    def __init__(self, path: Path):
        self.path = path

    def append(self, event: dict) -> None:
        with self.path.open("ab") as stream:
            stream.write(canonical(event) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())


def observe_training_resources(*, stage, mx, started, baseline, monotonic=time.monotonic):
    """Use the same host readers in pre-model checks and the actual fit."""
    return observe_resources(stage=stage, mx=mx, started=started,
                             baseline=baseline, limits=LIMITS,
                             system_reader=system.system_resources,
                             rss_reader=system.rss_bytes, monotonic=monotonic)


def check_pre_model_resources():
    """Exercise actual host/MLX observation without loading tokenizer or weights."""
    import mlx.core as mx
    started = time.monotonic()
    baseline = system.system_resources()
    _, record = observe_training_resources(stage="prelaunch_resource_check", mx=mx,
                                          started=started, baseline=baseline)
    return record


def validate_prelaunch(prelaunch_path: Path, launch_path: Path, output: Path,
                       worker_claim: str) -> tuple[dict, dict]:
    if (prelaunch_path.resolve() != DEFAULT_PRELAUNCH.resolve()
            or launch_path.resolve() != (output / "launch.json").resolve()):
        raise ValueError("fixed receipt path differs")
    receipt = json.loads(prelaunch_path.read_bytes())
    payload = receipt.get("payload")
    if (set(receipt) != {"payload", "digest"} or not isinstance(payload, dict)):
        raise ValueError("prelaunch envelope is incomplete")
    if receipt["digest"] != hashlib.sha256(canonical(payload)).hexdigest():
        raise ValueError("prelaunch digest differs")
    recovery.validate_prelaunch_shape(payload)
    if (payload.get("schema") != recovery.PRELAUNCH_SCHEMA
            or payload.get("passed") is not True or payload.get("launch_allowed") is not True
            or payload.get("logical_fit_id") != LOGICAL_FIT_ID
            or payload.get("sole_output") != SOLE_OUTPUT
            or payload.get("package_versions") != VERSIONS
            or output.resolve() != (ROOT / SOLE_OUTPUT).resolve()):
        raise ValueError("prelaunch authorization differs")
    launch = json.loads(launch_path.read_bytes())
    if (launch.get("schema") != "difference-pairwise-lora-launch-v2"
            or launch.get("logical_fit_id") != LOGICAL_FIT_ID
            or launch.get("sole_output") != SOLE_OUTPUT
            or launch.get("worker_claim") != worker_claim
            or launch.get("consumed") is not True or launch.get("released") is not False
            or launch.get("prelaunch", {}).get("sha256") != file_sha(prelaunch_path)):
        raise ValueError("launch claim differs")
    inventory = payload.get("inventory")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("execution inventory is missing")
    for name, digest in inventory.items():
        if (not isinstance(name, str) or not isinstance(digest, str)
                or file_sha(ROOT / name) != digest):
            raise ValueError(f"execution inventory differs: {name}")
    budget = payload.get("budget_proof")
    if (not isinstance(budget, dict) or set(budget) != {"path", "sha256"}
            or (ROOT / budget["path"]).resolve() != QUALIFIED_BUDGET.resolve()
            or budget["sha256"] != QUALIFIED_BUDGET_SHA256
            or file_sha(ROOT / budget["path"]) != QUALIFIED_BUDGET_SHA256
            or launch.get("budget_proof_sha256") != budget["sha256"]):
        raise ValueError("budget proof differs")
    proof = json.loads((ROOT / budget["path"]).read_bytes())
    proof_payload = proof.get("payload", proof)
    if (proof.get("digest") is not None
            and proof["digest"] != hashlib.sha256(canonical(proof_payload)).hexdigest()):
        raise ValueError("budget proof digest differs")
    if (proof.get("schema") != "difference-pairwise-lora-budget-proof-v1"
            or proof_payload.get("stage") != "budget_qualification"
            or proof_payload.get("passed") is not True
            or proof_payload.get("launch_allowed") is not True
            or proof_payload.get("logical_fit_id") != "difference-pairwise-lora-fit-budget-v1"
            or proof_payload.get("sole_output") != ".omx/artifacts/difference-pairwise-lora-train-v1"
            or proof_payload.get("new_policy", {}).get("limit_seconds") != 21600
            or proof_payload.get("old_policy", {}).get("launch_allowed") is not False):
        raise ValueError("budget proof does not authorize this fit")
    source_inventory = proof_payload.get("source_inventory")
    hashes = source_inventory.get("hashes") if isinstance(source_inventory, dict) else None
    if (not isinstance(hashes, dict) or len(hashes) != 48
            or source_inventory.get("count") != 48
            or source_inventory.get("sha256") != hashlib.sha256(canonical(hashes)).hexdigest()):
        raise ValueError("budget source inventory differs")
    recovery_pin = payload.get("recovery_proof")
    if (not isinstance(recovery_pin, dict) or set(recovery_pin) != {"path", "sha256"}
            or recovery_pin.get("path") != recovery.RECOVERY_PROOF_PATH
            or file_sha(ROOT / recovery_pin["path"]) != recovery_pin["sha256"]
            or launch.get("recovery_proof_sha256") != recovery_pin["sha256"]):
        raise ValueError("recovery proof differs")
    recovery_payload = recovery.validate_recovery_proof(ROOT / recovery_pin["path"])
    bound_artifacts = {recovery_pin["path"], recovery_payload["preflight"]["path"],
                       recovery_payload["checks"]["path"],
                       recovery_payload["execution_review"]["path"]}
    required_inventory = (set(hashes) | EXECUTION_ADDITIONS
                          | set(recovery_payload["execution_sources"])
                          | {budget["path"], *bound_artifacts})
    if set(inventory) != required_inventory:
        raise ValueError("execution inventory is incomplete or has unbound entries")
    if any(inventory.get(name) != digest for name, digest in hashes.items()):
        raise ValueError("execution inventory differs from budget source pins")
    worker_python = payload.get("worker_python")
    invocation_binary = Path(os.path.abspath(sys.executable))
    current_binary = Path(os.path.realpath(sys.executable))
    current_identity = {"invocation_path": str(invocation_binary),
                        "resolved_path": str(current_binary),
                        "resolved_sha256": file_sha(current_binary),
                        "package_versions": VERSIONS}
    if worker_python != recovery_payload["preflight"]["interpreter"] or worker_python != current_identity:
        raise ValueError("worker Python differs from qualified preflight")
    return payload, launch


def validate_and_load_data() -> dict:
    if file_sha(DATA / "manifest.json") != MANIFEST_SHA or file_sha(DATA_PROOF) != DATA_PROOF_SHA:
        raise ValueError("qualified data/proof pin differs")
    manifest = json.loads((DATA / "manifest.json").read_bytes())
    proof = json.loads(DATA_PROOF.read_bytes())
    if (proof.get("digest") != hashlib.sha256(canonical(proof.get("payload"))).hexdigest()
            or proof.get("payload", {}).get("passed") is not True):
        raise ValueError("qualified data proof differs")
    for name, digest in manifest.get("file_hashes", {}).items():
        if file_sha(DATA / name) != digest:
            raise ValueError(f"qualified data changed: {name}")
    counts = manifest.get("counts", {})
    required = {"repair_units": 59, "component_references": 232, "updates": 15,
                "mechanism_distinct_rows": 175, "E_mechanism": 481,
                "train640_rows": 640, "E_train640": 1556}
    if any(counts.get(name) != value for name, value in required.items()):
        raise ValueError("qualified counts differ")
    pairs = read_jsonl(DATA / "pairs.jsonl")
    tokens = read_jsonl(DATA / "tokens.jsonl")
    units = read_jsonl(DATA / "units.jsonl")
    schedule = read_jsonl(DATA / "schedule.jsonl")
    train640 = read_jsonl(DATA / "train640.jsonl")
    pair_by_id = {item["pair_id"]: item for item in pairs}
    token_by_id = {item["pair_id"]: item for item in tokens}
    if len(pair_by_id) != len(pairs) or set(pair_by_id) != set(token_by_id):
        raise ValueError("pair/token identity differs")
    validate_schedule(units, schedule, pair_by_id, token_by_id)
    return {"manifest": manifest, "pairs": pairs, "pair_by_id": pair_by_id,
            "token_by_id": token_by_id, "units": units, "schedule": schedule,
            "train640": train640}


def validate_schedule(units, schedule, pair_by_id, token_by_id) -> None:
    if len(units) != 59 or len(schedule) != 59:
        raise ValueError("schedule must contain 59 units")
    by_unit = {item["unit_id"]: item for item in units}
    if len(by_unit) != 59 or [item.get("unit_order") for item in schedule] != list(range(1, 60)):
        raise ValueError("unit order differs")
    expected_updates = [*sum(([update] * 4 for update in range(1, 15)), []), *([15] * 3)]
    if [item.get("update") for item in schedule] != expected_updates:
        raise ValueError("update grouping differs")
    seen = []
    for position, scheduled in enumerate(schedule, 1):
        unit = by_unit.get(scheduled.get("unit_id"))
        if unit is None or unit.get("order") != position:
            raise ValueError("scheduled unit binding differs")
        refs = unit.get("references")
        expected_weights = pairwise.unit_weights(item.get("target") for item in refs)
        if tuple(item.get("weight") for item in refs) != expected_weights:
            raise ValueError("balanced unit weights differ")
        for ref in refs:
            pair_id = ref.get("pair_id")
            if pair_id not in pair_by_id or pair_id not in token_by_id:
                raise ValueError("unit pair is unqualified")
            seen.append(pair_id)
    if len(seen) != 232:
        raise ValueError("component accounting differs")


def _source_rows(data: dict) -> tuple[list[dict], list[dict]]:
    manifest = data["manifest"]
    role_rows = {}
    for role, source in manifest["fit_sources"].items():
        rows = read_jsonl(ROOT / source["cases"]["path"])
        for row in rows:
            role_rows[(role, row["id"])] = row
    mechanism_keys = sorted({(pair["role"], pair["row_id"])
                             for pair in data["pairs"] if "mechanism" in pair["scopes"]})
    mechanism = [{"role": role, "row": role_rows[(role, row_id)], "stratum": "mechanism"}
                 for role, row_id in mechanism_keys]
    diagnostic = read_jsonl(ROOT / manifest["train640_source"]["path"])
    diag_rows = {(item["role"], item["row"]["id"]): item["row"] for item in diagnostic}
    train = [{"role": item["role"], "row": diag_rows[(item["role"], item["row_id"])],
              "stratum": item["stratum"]} for item in data["train640"]]
    if len(mechanism) != 175 or len(train) != 640:
        raise ValueError("evaluation row accounting differs")
    return mechanism, train


def _pairs_for_row(data: dict, role: str, row_id: str, scope: str) -> list[dict]:
    return sorted((pair for pair in data["pairs"]
                   if pair["role"] == role and pair["row_id"] == row_id and scope in pair["scopes"]),
                  key=lambda pair: pair["alternative_source_index"])


def _score_scope(mx, model, data, items, scope, deadline_check,
                 checkpoint, progress, monotonic=time.monotonic) -> tuple[list, list]:
    predictions, evidence = [], []
    for ordinal, item in enumerate(items, 1):
        role, row = item["role"], item["row"]
        pairs = _pairs_for_row(data, role, row["id"], scope)
        prepared = evaluation.prepare_selection(role, row, pairs)
        logits = {}
        for pair in pairs:
            deadline_check()
            inference_started = monotonic()
            scores = shared.score_choices(mx, model, data["token_by_id"][pair["pair_id"]]["tokens"], (16, 17))
            mx.eval(scores)
            values = scores.tolist()
            if len(values) != 2 or not all(math.isfinite(value) for value in values):
                raise FloatingPointError("pair inference is nonfinite")
            logits[pair["pair_id"]] = values
            checkpoint(f"{scope}_pair_after_{pair['pair_id']}")
            evidence.append({"scope": scope, "role": role, "row_id": row["id"],
                             "alternative_source_index": pair["alternative_source_index"],
                             "pair_id": pair["pair_id"], "logits": values,
                             "seconds": monotonic() - inference_started})
        prediction = evaluation.resolve_selection(prepared, logits)
        if not prediction["complete"]:
            raise RuntimeError("qualified evaluation prediction is incomplete")
        predictions.append(prediction)
        progress.append({"event": "evaluation_row", "scope": scope, "ordinal": ordinal,
                         "role": role, "row_id": row["id"], "pair_count": len(pairs)})
    return predictions, evidence


def execute_training_schedule(data: dict, *, materialize, accumulate, apply_update,
                              checkpoint, progress, monotonic=time.monotonic,
                              units_out=None, updates_out=None) -> tuple[list, list]:
    """Traverse the frozen 59-unit schedule once using injectable numeric operations."""
    units_out = [] if units_out is None else units_out
    updates_out = [] if updates_out is None else updates_out
    schedule_by_unit = {item["unit_id"]: item for item in data["units"]}
    accumulated = None
    update_units: list[str] = []
    update_started = 0.0
    for scheduled in data["schedule"]:
        unit = schedule_by_unit[scheduled["unit_id"]]
        update = scheduled["update"]
        if scheduled["position_in_update"] == 1:
            update_started, accumulated, update_units = monotonic(), None, []
        components = [{"pair_id": ref["pair_id"], "target": ref["target"],
                       "tokens": data["token_by_id"][ref["pair_id"]]["tokens"]}
                      for ref in unit["references"]]

        def component_observer(record, uid=unit["unit_id"]):
            checkpoint(f"component_{uid}_{record['component_index'] + 1}")

        checkpoint(f"unit_{unit['order']}_before")
        result = materialize(components, component_observer)
        records = result.get("components") if isinstance(result, dict) else None
        if (not isinstance(result, dict)
                or type(result.get("loss")) not in (int, float)
                or not math.isfinite(result["loss"])
                or type(result.get("unit_gradient_seconds")) not in (int, float)
                or not math.isfinite(result["unit_gradient_seconds"])
                or result["unit_gradient_seconds"] < 0
                or not isinstance(records, list) or len(records) != len(components)):
            raise FloatingPointError("materialized unit evidence is malformed or nonfinite")
        expected_weights = pairwise.unit_weights(item["target"] for item in components)
        for record, component, weight in zip(records, components, expected_weights):
            if (record.get("target") != component["target"] or record.get("weight") != weight
                    or type(record.get("loss")) not in (int, float)
                    or not math.isfinite(record["loss"])
                    or type(record.get("gradient_seconds")) not in (int, float)
                    or not math.isfinite(record["gradient_seconds"])
                    or record["gradient_seconds"] < 0):
                raise FloatingPointError("component evidence is malformed or nonfinite")
        accumulated = accumulate(accumulated, result["gradient"])
        update_units.append(unit["unit_id"])
        unit_record = {"order": unit["order"], "unit_id": unit["unit_id"], "update": update,
                       "loss": result["loss"], "unit_gradient_seconds": result["unit_gradient_seconds"],
                       "components": [{**record, "pair_id": ref["pair_id"]}
                                      for record, ref in zip(result["components"], unit["references"])]}
        units_out.append(unit_record)
        progress.append({"event": "unit_complete", **unit_record})
        checkpoint(f"unit_{unit['order']}_after")
        is_end = scheduled["position_in_update"] == 4 or scheduled["unit_order"] == 59
        if is_end:
            optimizer_started = monotonic()
            norm = apply_update(accumulated, len(update_units))
            if type(norm) not in (int, float) or not math.isfinite(norm):
                raise FloatingPointError("gradient norm is nonfinite")
            optimizer_seconds = monotonic() - optimizer_started
            checkpoint(f"update_{update}_after_optimizer")
            complete_seconds = monotonic() - update_started
            if complete_seconds > LIMITS["update_seconds"]:
                raise TimeoutError("complete update exceeds 120 seconds")
            update_record = {"update": update, "unit_ids": list(update_units),
                             "actual_unit_count": len(update_units),
                             "gradient_norm_before_clip": norm,
                             "optimizer_seconds": optimizer_seconds,
                             "complete_update_seconds": complete_seconds}
            updates_out.append(update_record)
            progress.append({"event": "update_complete", **update_record})
    if len(units_out) != 59 or len(updates_out) != 15:
        raise RuntimeError("training accounting is incomplete")
    return units_out, updates_out


def run(output: Path, prelaunch: Path, launch: Path, fit_started: float,
        worker_claim: str, *, monotonic=time.monotonic) -> tuple[dict, int]:
    if (output.resolve() != (ROOT / SOLE_OUTPUT).resolve()
            or launch.resolve() != (output / "launch.json").resolve()):
        raise ValueError("fixed worker output or launch path differs")
    launch_identity = json.loads(launch.read_bytes())
    if (launch_identity.get("logical_fit_id") != LOGICAL_FIT_ID
            or launch_identity.get("schema") != "difference-pairwise-lora-launch-v2"
            or launch_identity.get("sole_output") != SOLE_OUTPUT
            or launch_identity.get("worker_claim") != worker_claim
            or launch_identity.get("consumed") is not True
            or launch_identity.get("released") is not False
            or not isinstance(launch_identity.get("recovery_proof_sha256"), str)):
        raise ValueError("worker does not own this launch")
    worker_start = None
    claim_error = None
    try:
        worker_start = claim_worker_start(output, file_sha(launch), worker_claim)
    except FileExistsError:
        # Another worker owns the output; never replace its report or progress.
        raise
    except OSError as error:
        claim_error = error
    report = {"schema": "difference-pairwise-lora-training-v2",
              "logical_fit_id": LOGICAL_FIT_ID, "sole_output": SOLE_OUTPUT,
              "completed": False, "execution_valid": False, "quality_passed": False,
              "terminal_status": "execution_failure", "real_training_run": False,
              "fresh_base_initialized": False, "old_adapter_loaded": False,
              "disposable_adapter_loaded": False, "one_epoch": True,
              "eligibility": {"evaluation": False, "resume": False, "deploy": False},
              "worker_claim": worker_claim, "fit_start_monotonic": fit_started,
              "worker_start": worker_start,
              "counts_expected": EXPECTED,
              "recipe": {"seed": shared.SEED, "lora_layers": shared.LORA_LAYERS,
                         "lora_config": shared.LORA_CONFIG,
                         "learning_rate": shared.LEARNING_RATE,
                         "adam_betas": list(shared.ADAM_BETAS), "adam_eps": shared.ADAM_EPS,
                         "bias_correction": True, "weight_decay": 0.0,
                         "accumulation_units": 4, "final_update_units": 3,
                         "clip_norm": shared.CLIP_NORM, "epochs": 1},
              "limits": LIMITS, "units": [], "updates": [], "resource_observations": [],
              "predictions": {}, "quality": None}
    progress = ProgressLog(output / "progress.jsonl")
    mx = model = tokenizer = optimizer = value_and_grad = raw_value_and_grad = None
    initial = accumulated = None
    baseline = None
    exit_code = 1

    def expired() -> bool:
        return monotonic() - fit_started > LIMITS["whole_seconds"]

    def deadline_check():
        if expired():
            raise TimeoutError("whole fit deadline exceeded")

    def checkpoint(stage: str):
        deadline_check()
        _, record = observe_training_resources(stage=stage, mx=mx, started=fit_started,
                                               baseline=baseline, monotonic=monotonic)
        report["resource_observations"].append(record)
        deadline_check()

    try:
        if claim_error is not None:
            raise claim_error
        prelaunch_payload, launch_payload = validate_prelaunch(
            prelaunch, launch, output, worker_claim)
        report["recovery_proof"] = prelaunch_payload["recovery_proof"]
        data = validate_and_load_data()
        report["prelaunch_sha256"] = file_sha(prelaunch)
        report["launch_sha256"] = file_sha(launch)
        report["budget_proof"] = prelaunch_payload["budget_proof"]
        report["data"] = {"manifest_sha256": MANIFEST_SHA, "proof_sha256": DATA_PROOF_SHA}
        report["source_hashes_before"] = dict(prelaunch_payload["inventory"])
        if {name: importlib.metadata.version(name) for name in VERSIONS} != VERSIONS:
            raise RuntimeError("package versions differ")
        report["package_versions"] = dict(VERSIONS)
        deadline_check()
        import mlx.core as mlx_core
        from mlx_lm import load
        from mlx_lm.tuner.utils import load_adapters
        mx = mlx_core
        mx.set_memory_limit(LIMITS["mlx_peak_bytes"])
        mx.set_cache_limit(LIMITS["cache_bytes"])
        baseline = system.system_resources()
        report["baseline"] = baseline
        checkpoint("before_model_load")
        model, tokenizer = load(str(MODEL), tokenizer_config={"trust_remote_code": False})
        checkpoint("model_loaded")
        initial = shared.configure_fresh_lora(mx, model)
        model.train()
        report["fresh_base_initialized"] = True
        report["initial_trainable_tensor_hashes"] = shared.tensor_hashes(initial)
        optimizer = shared.create_optimizer()
        raw_value_and_grad = pairwise.make_component_value_and_grad(mx, model)

        def value_and_grad(current_model, tokens, target):
            deadline_check()
            return raw_value_and_grad(current_model, tokens, target)

        def apply_update(gradient, count):
            deadline_check()
            return pairwise.apply_unit_update(mx, model, optimizer, gradient, count)
        checkpoint("fresh_adapter_initialized")
        report["real_training_run"] = True
        report["units"], report["updates"] = execute_training_schedule(
            data,
            materialize=lambda components, observer: pairwise.materialize_unit(
                mx, model, components, value_and_grad, observe=observer),
            accumulate=lambda left, right: shared.add_materialized_gradients(mx, left, right),
            apply_update=apply_update,
            checkpoint=checkpoint, progress=progress, monotonic=monotonic,
            units_out=report["units"], updates_out=report["updates"])
        report["final_trainable_tensor_hashes"] = shared.tensor_hashes(
            shared.require_trainable_contract(model))
        if report["final_trainable_tensor_hashes"] == report["initial_trainable_tensor_hashes"]:
            raise RuntimeError("adapter did not change")
        model.eval()
        mechanism_items, train_items = _source_rows(data)
        closed_labels = []
        for item in (*mechanism_items, *train_items):
            row = item["row"]
            closed_labels.append((row, row.pop("expected")))
        try:
            mechanism_predictions, mechanism_evidence = _score_scope(
                mx, model, data, mechanism_items, "mechanism", deadline_check,
                checkpoint, progress, monotonic)
            train_predictions, train_evidence = _score_scope(
                mx, model, data, train_items, "train640", deadline_check,
                checkpoint, progress, monotonic)
        finally:
            for row, labels in closed_labels:
                row["expected"] = labels
        if len(mechanism_evidence) != 481 or len(train_evidence) != 1556:
            raise RuntimeError("pair inference accounting differs")
        mechanism_quality = evaluation.score_predictions(
            mechanism_items, mechanism_predictions, expected_rows=175, minimum_net_gain=-175)
        train_quality = evaluation.score_predictions(
            train_items, train_predictions, expected_rows=640, minimum_net_gain=5)
        report["predictions"] = {"mechanism": mechanism_predictions,
                                 "train640": train_predictions}
        report["pair_inference"] = {"mechanism": mechanism_evidence,
                                    "train640": train_evidence}
        report["quality"] = {"mechanism": mechanism_quality, "train640": train_quality}
        deadline_check()
        adapter_dir = output / "adapter"
        adapter_dir.mkdir()
        adapter_path = adapter_dir / "adapters.safetensors"
        report["saved_trainable_tensor_hashes"] = shared.save_adapter(mx, model, adapter_path)
        _fsync_file(adapter_path)
        config_path = adapter_dir / "adapter_config.json"
        config = {"fine_tune_type": "lora", "num_layers": shared.LORA_LAYERS,
                  "lora_parameters": shared.LORA_CONFIG}
        _durable_replace(config_path, config)
        report["adapter"] = {"path": key(adapter_path), "sha256": file_sha(adapter_path),
                             "config_path": key(config_path), "config_sha256": file_sha(config_path)}
        representative_ids = [row["pair_id"] for row in sorted(
            data["token_by_id"].values(), key=lambda row: (row["prompt_tokens"], row["pair_id"]))]
        representative_ids = [representative_ids[index] for index in
                              (0, (len(representative_ids) - 1) // 2,
                               math.ceil(.95 * len(representative_ids)) - 1,
                               len(representative_ids) - 1)]
        before_reload = {}
        for pair_id in representative_ids:
            deadline_check()
            values = shared.score_choices(mx, model, data["token_by_id"][pair_id]["tokens"], (16, 17))
            mx.eval(values)
            before_reload[pair_id] = values.tolist()
            checkpoint(f"save_probe_{pair_id}")
        model = tokenizer = optimizer = value_and_grad = raw_value_and_grad = None
        initial = accumulated = None
        gc.collect(); mx.clear_cache(); checkpoint("first_model_released")
        deadline_check()
        model, tokenizer = load(str(MODEL), tokenizer_config={"trust_remote_code": False})
        model.freeze(); load_adapters(model, str(adapter_dir)); model.eval()
        report["reloaded_trainable_tensor_hashes"] = shared.tensor_hashes(
            shared.require_trainable_contract(model))
        if not (report["final_trainable_tensor_hashes"]
                == report["saved_trainable_tensor_hashes"]
                == report["reloaded_trainable_tensor_hashes"]):
            raise RuntimeError("saved/reloaded tensor hashes differ")
        reload_rows = []
        for pair_id in representative_ids:
            values = shared.score_choices(mx, model, data["token_by_id"][pair_id]["tokens"], (16, 17))
            mx.eval(values); actual = values.tolist(); expected = before_reload[pair_id]
            close = all(abs(a - b) <= 1e-5 + 1e-5 * abs(b) for a, b in zip(actual, expected))
            same_argmax = max(range(2), key=lambda i: actual[i]) == max(range(2), key=lambda i: expected[i])
            if not close or not same_argmax:
                raise RuntimeError("reload logits differ")
            reload_rows.append({"pair_id": pair_id, "expected_logits": expected,
                                "actual_logits": actual, "close": close,
                                "same_argmax": same_argmax})
            checkpoint(f"reload_probe_{pair_id}")
        report["reload"] = reload_rows
        model = tokenizer = None; gc.collect(); mx.clear_cache(); checkpoint("final_model_released")
        validate_prelaunch(prelaunch, launch, output, worker_claim)
        report["source_hashes_after"] = dict(prelaunch_payload["inventory"])
        report["completed"] = True
        report["execution_valid"] = True
        report["quality_passed"] = train_quality["gate"]["passed"] is True
        report["terminal_status"] = "quality_passed" if report["quality_passed"] else "quality_rejected"
        report["independent_training_verification_required"] = True
        report["external_execution_receipt_required"] = True
        report["eligibility"] = {"evaluation": False, "resume": False, "deploy": False}
        exit_code = 0 if report["quality_passed"] else 2
    except BaseException as error:
        if isinstance(error, ResourceObservationError):
            report["resource_observations"].append(error.record)
        report["failure_type"] = type(error).__name__
        report["failure"] = str(error)
    finally:
        model = tokenizer = optimizer = value_and_grad = raw_value_and_grad = None
        initial = accumulated = None
        gc.collect()
        if mx is not None:
            try:
                mx.synchronize(); mx.clear_cache()
            except BaseException as error:
                report["cleanup_error"] = f"{type(error).__name__}: {error}"
                report.update({"completed": False, "execution_valid": False,
                               "quality_passed": False, "terminal_status": "execution_failure",
                               "eligibility": {"evaluation": False, "resume": False, "deploy": False}})
                exit_code = 1
        if report.get("completed") is True:
            try:
                checkpoint("terminal_cleanup")
            except BaseException as error:
                if isinstance(error, ResourceObservationError):
                    report["resource_observations"].append(error.record)
                report["terminal_cleanup_error"] = f"{type(error).__name__}: {error}"
                report.update({"completed": False, "execution_valid": False,
                               "quality_passed": False, "terminal_status": "execution_failure",
                               "eligibility": {"evaluation": False, "resume": False, "deploy": False}})
                exit_code = 1
        if report.get("completed") is True:
            try:
                validate_prelaunch(prelaunch, launch, output, worker_claim)
            except BaseException as error:
                report["final_inventory_error"] = f"{type(error).__name__}: {error}"
                report.update({"completed": False, "execution_valid": False,
                               "quality_passed": False, "terminal_status": "execution_failure",
                               "eligibility": {"evaluation": False, "resume": False, "deploy": False}})
                exit_code = 1
        report["worker_fit_elapsed_seconds"] = monotonic() - fit_started
        if report["worker_fit_elapsed_seconds"] > LIMITS["whole_seconds"]:
            report.update({"completed": False, "execution_valid": False,
                           "quality_passed": False, "terminal_status": "deadline_failure",
                           "failure": "whole fit deadline exceeded",
                           "eligibility": {"evaluation": False, "resume": False, "deploy": False}})
            exit_code = 1
        progress.append({"event": "terminal", "terminal_status": report["terminal_status"],
                         "exit_code": exit_code, "elapsed_seconds": report["worker_fit_elapsed_seconds"]})
        _durable_replace(output / "report.json", report)
    return report, exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prelaunch", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--fit-start-monotonic", type=float, required=True)
    parser.add_argument("--worker-claim", required=True)
    args = parser.parse_args()
    _, code = run(args.output, args.prelaunch, args.launch,
                  args.fit_start_monotonic, args.worker_claim)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
