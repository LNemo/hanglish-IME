"""Independently verify the synthetic resource probe for pairwise LoRA."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import statistics
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "difference-pairwise-lora-proof-v1"
RESOURCE_SCHEMA = "difference-pairwise-lora-resource-v1"
DATA = ROOT / ".omx/artifacts/difference-pairwise-lora-data-v1"
DATA_MANIFEST = DATA / "manifest.json"
DATA_PROOF = ROOT / ".omx/artifacts/difference-pairwise-lora-data-v1-proof.json"
DATA_MANIFEST_SHA256 = "32e83b145e71de39912e122f9925c9cf362875a5206e49dddcb7611a464be132"
DATA_PROOF_SHA256 = "03943e073ec333b4e8a8e8386ace042c0c3dda6370cdc5b284961262bd2dea72"
DATA_VERIFIER_SHA256 = "d9c507e78679db9b3de900ad3f13a22a1f85912d646eb8f8a8217d3894629217"
VERSIONS = {"mlx": "0.31.2", "mlx-lm": "0.31.3", "transformers": "5.12.1"}
LIMITS = {"rss_bytes": 6 * 1024**3, "mlx_peak_bytes": 6 * 1024**3,
          "cache_bytes": 256 * 1024**2, "whole_seconds": 900.0,
          "update_seconds": 120.0, "minimum_system_free_percent": 10.0,
          "maximum_swap_growth_bytes": 512 * 1024**2}
RECIPE = {"seed": 20260915, "lora_layers": 4,
          "lora_config": {"rank": 8, "scale": 20.0, "dropout": 0.0,
                          "keys": ["self_attn.q_proj", "self_attn.v_proj"]},
          "learning_rate": 1e-5, "adam_betas": [0.9, 0.999], "adam_eps": 1e-8,
          "bias_correction": True, "weight_decay": 0.0,
          "accumulation_units": 4, "clip_norm": 1.0}
EXPECTED_NAMES = sorted(
    f"model.layers.{layer}.self_attn.{projection}.{suffix}"
    for layer in range(32, 36) for projection in ("q_proj", "v_proj")
    for suffix in ("lora_a", "lora_b"))
EXPECTED_SHAPES = {
    f"model.layers.{layer}.self_attn.{projection}.{suffix}": shape
    for layer in range(32, 36)
    for projection, suffix, shape in (
        ("q_proj", "lora_a", [2560, 8]), ("q_proj", "lora_b", [8, 4096]),
        ("v_proj", "lora_a", [2560, 8]), ("v_proj", "lora_b", [8, 1024]))}


class VerificationError(ValueError):
    pass


def need(condition, message):
    if not condition:
        raise VerificationError(message)


def canonical_bytes(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise VerificationError("value is not canonical-JSON encodable") from exc


def digest_value(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise VerificationError(f"required file is unavailable: {path}") from exc
    return digest.hexdigest()


def path_key(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path):
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSON: {path}") from exc
    need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def read_jsonl(path):
    try:
        rows = [json.loads(line) for line in Path(path).read_bytes().splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid JSONL: {path}") from exc
    need(all(isinstance(row, dict) for row in rows), f"JSONL objects required: {path}")
    return rows


def finite(value, *, nonnegative=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (not nonnegative or value >= 0))


def p95(values):
    need(values and all(finite(value, nonnegative=True) for value in values),
         "timing series must be finite and nonnegative")
    ordered = sorted(values)
    return ordered[math.ceil(.95 * len(ordered)) - 1]


def _qualified_data():
    need(file_digest(DATA_MANIFEST) == DATA_MANIFEST_SHA256
         and file_digest(DATA_PROOF) == DATA_PROOF_SHA256,
         "qualified data manifest/proof pin differs")
    manifest, proof = read_json(DATA_MANIFEST), read_json(DATA_PROOF)
    payload = proof.get("payload")
    need(proof.get("schema") == SCHEMA and isinstance(payload, dict)
         and proof.get("digest") == digest_value(payload)
         and payload.get("passed") is True and payload.get("stage") == "data"
         and payload.get("data_manifest_sha256") == DATA_MANIFEST_SHA256
         and payload.get("verifier_source") == {
             "path": "scripts/verify_difference_pairwise_lora.py",
             "sha256": DATA_VERIFIER_SHA256}
         and file_digest(ROOT / payload["verifier_source"]["path"]) == DATA_VERIFIER_SHA256,
         "qualified independent data proof differs")
    need(manifest.get("completed") is True and manifest.get("counts") == payload.get("counts")
         and manifest.get("file_hashes") == payload.get("file_hashes"),
         "qualified data manifest/proof content differs")
    for name, expected in manifest["file_hashes"].items():
        need(file_digest(DATA / name) == expected, f"qualified data artifact changed: {name}")
    for name, expected in manifest.get("source_hashes_after", {}).items():
        need(file_digest(ROOT / name) == expected, f"qualified data source changed: {name}")
    return manifest, read_jsonl(DATA / "tokens.jsonl")


def expected_representatives_and_schedule(token_rows):
    need(len(token_rows) == 1928 and len({row.get("pair_id") for row in token_rows}) == 1928,
         "qualified token identities differ")
    for row in token_rows:
        need(type(row.get("prompt_tokens")) is int
             and row["prompt_tokens"] == len(row.get("tokens", []))
             and 0 < row["prompt_tokens"] <= 1024
             and digest_value(row["tokens"]) == row.get("tokens_sha256")
             and row.get("digit_continuation_ids") == {"1": 16, "2": 17},
             "qualified token row differs")
    ordered = sorted(token_rows, key=lambda row: (row["prompt_tokens"], row["pair_id"]))
    ranks = (0, (len(ordered) - 1) // 2, math.ceil(.95 * len(ordered)) - 1,
             len(ordered) - 1)
    representatives = []
    for quantile, rank in zip(("min", "median", "p95", "max"), ranks):
        row = ordered[rank]
        representatives.append({"quantile": quantile, "pair_id": row["pair_id"],
            "prompt_tokens": row["prompt_tokens"], "tokens_sha256": row["tokens_sha256"]})
    schedule, cursor = [], 0
    for index in range(20):
        controls = 2 if index % 5 == 4 else 3
        targets = ["SWITCH"] + ["KEEP"] * controls
        weights = [0.5] + [0.5 / controls] * controls
        components = []
        for target, weight in zip(targets, weights):
            row = representatives[cursor % 4]
            components.append({"pair_id": row["pair_id"], "target": target, "weight": weight,
                "prompt_tokens": row["prompt_tokens"], "tokens_sha256": row["tokens_sha256"]})
            cursor += 1
        schedule.append({"unit": index + 1, "update": index // 4 + 1,
                         "components": components})
    return representatives, schedule


def expected_observation_stages(schedule, representatives):
    stages = ["before_model_load", "model_loaded", "fresh_adapter_initialized"]
    for update in range(1, 6):
        for unit in schedule[(update - 1) * 4:update * 4]:
            stages.extend(f"unit_{unit['unit']}_component_{index}"
                          for index in range(len(unit["components"])))
            stages.append(f"unit_{unit['unit']}_complete")
        stages.append(f"update_{update}_complete")
    stages.extend(f"equivalence_{row['quantile']}" for row in representatives)
    stages.extend(f"inference_{row['quantile']}_{repeat}"
                  for row in representatives for repeat in range(1, 11))
    stages.extend(["adapter_saved", "first_model_released"])
    stages.extend(f"reload_{row['quantile']}" for row in representatives)
    stages.extend(["final_model_released", "terminal"])
    return stages


def verify_observations(records, stages, baseline):
    need(isinstance(baseline, dict) and set(baseline) == {"free_percent", "swap_used_bytes"}
         and all(finite(value, nonnegative=True) for value in baseline.values()),
         "resource baseline differs")
    need(isinstance(records, list) and [row.get("stage") for row in records] == stages,
         "resource observation stage sequence differs")
    settled = []
    for row in records:
        need(row.get("action") in ("none", "clear_unused_mlx_cache")
             and row.get("errors") == []
             and finite(row.get("cleanup_seconds"), nonnegative=True)
             and finite(row.get("checkpoint_seconds"), nonnegative=True),
             "resource observation wrapper differs")
        pre, post = row.get("pre"), row.get("post")
        need(isinstance(pre, dict)
             and ((row["action"] == "none" and post is None)
                  or (row["action"] == "clear_unused_mlx_cache" and isinstance(post, dict))),
             "resource cache observation differs")
        for observation in (pre,) + ((post,) if post is not None else ()):
            need(set(observation) == {"rss_peak_bytes", "mlx_active_bytes", "mlx_cache_bytes",
                "mlx_peak_bytes", "free_percent", "swap_used_bytes", "swap_growth_bytes",
                "elapsed_seconds"}
                and all(finite(value, nonnegative=name not in ("swap_growth_bytes",))
                        for name, value in observation.items()),
                 "resource metric is missing or nonfinite")
            need(math.isclose(observation["swap_growth_bytes"],
                              observation["swap_used_bytes"] - baseline["swap_used_bytes"],
                              rel_tol=0, abs_tol=1e-6),
                 "swap growth does not match the baseline")
        settled.append(post if post is not None else pre)
    summary = {"rss_peak_bytes": max(row["rss_peak_bytes"] for row in settled),
               "mlx_peak_bytes": max(row["mlx_peak_bytes"] for row in settled),
               "settled_cache_bytes": max(row["mlx_cache_bytes"] for row in settled),
               "minimum_free_percent": min(row["free_percent"] for row in settled),
               "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in settled)}
    feasible = (summary["rss_peak_bytes"] <= LIMITS["rss_bytes"]
        and summary["mlx_peak_bytes"] <= LIMITS["mlx_peak_bytes"]
        and summary["settled_cache_bytes"] <= LIMITS["cache_bytes"]
        and summary["minimum_free_percent"] >= LIMITS["minimum_system_free_percent"]
        and summary["maximum_swap_growth_bytes"] <= LIMITS["maximum_swap_growth_bytes"])
    return summary, feasible


def _argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def verify_logits(rows, representatives, *, reload):
    need(isinstance(rows, list) and len(rows) == 4, "logit parity rows differ")
    for row, shape in zip(rows, representatives):
        need(row.get("pair_id") == shape["pair_id"], "logit parity pair order differs")
        left = row.get("actual_logits" if reload else "selected_logits")
        right = row.get("expected_logits" if reload else "ordinary_logits")
        need(isinstance(left, list) and isinstance(right, list) and len(left) == len(right) == 2
             and all(finite(value) for value in left + right) and _argmax(left) == _argmax(right),
             "digit logits or argmax differ")
        for actual, expected in zip(left, right):
            tolerance = 1e-5 + 1e-5 * abs(expected) if reload else .125
            need(abs(actual - expected) <= tolerance, "digit logits exceed parity tolerance")


def verify_measurements(report, schedule, representatives):
    units, updates, inference = report.get("units"), report.get("updates"), report.get("inference")
    need(isinstance(units, list) and len(units) == 20, "exactly 20 measured units are required")
    component_times, unit_times = [], []
    for observed, expected in zip(units, schedule):
        need(observed.get("unit") == expected["unit"] and observed.get("update") == expected["update"]
             and finite(observed.get("loss"))
             and finite(observed.get("unit_gradient_seconds"), nonnegative=True),
             "measured unit identity, loss, or duration differs")
        components = observed.get("components")
        need(isinstance(components, list) and len(components) == len(expected["components"]),
             "measured component count differs")
        weighted_losses = []
        for index, (component, planned) in enumerate(zip(components, expected["components"])):
            need(component.get("component_index") == index
                 and component.get("target") == planned["target"]
                 and component.get("weight") == planned["weight"]
                 and finite(component.get("loss"))
                 and finite(component.get("gradient_seconds"), nonnegative=True),
                 "measured component evidence differs")
            weighted_losses.append(component["weight"] * component["loss"])
            component_times.append(component["gradient_seconds"])
        need(math.isclose(observed["loss"], math.fsum(weighted_losses),
                          rel_tol=1e-12, abs_tol=1e-12),
             "weighted unit loss differs")
        need(observed["unit_gradient_seconds"] + 1e-9 >= math.fsum(
                 component["gradient_seconds"] for component in components),
             "unit duration is shorter than its component gradients")
        unit_times.append(observed["unit_gradient_seconds"])
    need(isinstance(updates, list) and len(updates) == 5, "exactly five updates are required")
    optimizer_times, complete_times = [], []
    for index, row in enumerate(updates, 1):
        need(row.get("update") == index and row.get("unit_ids") == list(range(4*index-3, 4*index+1))
             and finite(row.get("gradient_norm_before_clip"), nonnegative=True)
             and finite(row.get("optimizer_seconds"), nonnegative=True)
             and finite(row.get("complete_update_seconds"), nonnegative=True)
             and row["complete_update_seconds"] <= LIMITS["update_seconds"],
             "optimizer update evidence differs")
        optimizer_times.append(row["optimizer_seconds"])
        complete_times.append(row["complete_update_seconds"])
        measured_units = units[(index - 1) * 4:index * 4]
        need(row["complete_update_seconds"] + 1e-9 >= math.fsum(
                 item["unit_gradient_seconds"] for item in measured_units)
                 + row["optimizer_seconds"],
             "complete update duration is shorter than unit and optimizer work")
    need(isinstance(inference, list) and len(inference) == 40, "exactly 40 inferences are required")
    inference_times = []
    for row, (shape, repetition) in zip(inference,
            ((shape, repeat) for shape in representatives for repeat in range(1, 11))):
        logits = row.get("logits")
        need(row.get("pair_id") == shape["pair_id"] and row.get("repetition") == repetition
             and finite(row.get("seconds"), nonnegative=True)
             and isinstance(logits, list) and len(logits) == 2
             and all(finite(value) for value in logits), "pair inference evidence differs")
        inference_times.append(row["seconds"])
    interval = report.get("finalization_interval")
    need(isinstance(interval, dict) and set(interval) == {"start_elapsed", "end_elapsed"}
         and all(finite(value, nonnegative=True) for value in interval.values())
         and interval["end_elapsed"] >= interval["start_elapsed"],
         "finalization interval differs")
    timing = {"component_gradient_p95": p95(component_times),
              "unit_gradient_p95": p95(unit_times),
              "optimizer_update_p95": p95(optimizer_times),
              "pair_inference_p95": p95(inference_times),
              "save_reload_finalization_seconds":
                  interval["end_elapsed"] - interval["start_elapsed"]}
    declared_timing = report.get("timing_summary")
    need(isinstance(declared_timing, dict) and set(declared_timing) == set(timing)
         and all(finite(declared_timing[name], nonnegative=True)
                 and math.isclose(declared_timing[name], value, rel_tol=1e-12, abs_tol=1e-9)
                 for name, value in timing.items()), "timing summary differs")
    return timing, complete_times


def read_safetensors(path):
    try:
        raw = Path(path).read_bytes()
        need(len(raw) >= 8, "safetensors header is truncated")
        header_length = struct.unpack("<Q", raw[:8])[0]
        need(0 < header_length <= len(raw) - 8,
             "safetensors header length differs")
        header = json.loads(raw[8:8 + header_length])
    except (OSError, json.JSONDecodeError, struct.error) as exc:
        raise VerificationError("safetensors header is malformed") from exc
    need(isinstance(header, dict), "safetensors header must be an object")
    body = raw[8 + header_length:]
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    need(sorted(tensors) == EXPECTED_NAMES, "saved safetensors names differ")
    hashes, scalar_count, occupied = {}, 0, []
    for name in EXPECTED_NAMES:
        item = tensors[name]
        need(isinstance(item, dict) and item.get("dtype") == "F32"
             and item.get("shape") == EXPECTED_SHAPES[name]
             and isinstance(item.get("data_offsets"), list)
             and len(item["data_offsets"]) == 2
             and all(type(value) is int for value in item["data_offsets"]),
             f"saved safetensors metadata differs: {name}")
        start, end = item["data_offsets"]
        count = math.prod(item["shape"])
        need(0 <= start < end <= len(body) and end - start == count * 4,
             f"saved safetensors byte span differs: {name}")
        occupied.append((start, end)); scalar_count += count
        shape = tuple(item["shape"])
        hashes[name] = hashlib.sha256(
            f"float32:{shape}:".encode() + body[start:end]).hexdigest()
    occupied = sorted(occupied)
    need(occupied[0][0] == 0
         and all(left[1] == right[0] for left, right in zip(occupied, occupied[1:]))
         and occupied[-1][1] == len(body),
         "saved safetensors body has gaps, overlap, or trailing bytes")
    need(scalar_count == 327680, "saved safetensors scalar count differs")
    return hashes


def verify_trainable_and_adapter(report, directory):
    trainable = report.get("trainable")
    need(trainable == {"names": EXPECTED_NAMES, "scalar_count": 327680},
         "trainable tensor tree differs")
    maps = [report.get(name) for name in ("initial_trainable_tensor_hashes",
        "final_trainable_tensor_hashes", "saved_trainable_tensor_hashes",
        "reloaded_trainable_tensor_hashes")]
    for item in maps:
        need(isinstance(item, dict) and sorted(item) == EXPECTED_NAMES
             and all(re.fullmatch(r"[0-9a-f]{64}", value or "") for value in item.values()),
             "trainable tensor hashes differ")
    need(maps[1] == maps[2] == maps[3] and maps[0] != maps[1],
         "adapter did not change or save/reload hashes differ")
    adapter = report.get("adapter")
    need(isinstance(adapter, dict), "adapter evidence is missing")
    for path_field, hash_field in (("path", "sha256"), ("config_path", "config_sha256")):
        path = ROOT / adapter.get(path_field, "")
        need(path.resolve().is_relative_to(directory.resolve())
             and file_digest(path) == adapter.get(hash_field),
             f"adapter artifact differs: {path_field}")
    config = read_json(ROOT / adapter["config_path"])
    need(config == {"fine_tune_type": "lora", "num_layers": 4,
                    "lora_parameters": RECIPE["lora_config"]},
         "adapter configuration differs")
    need(read_safetensors(ROOT / adapter["path"]) == maps[2],
         "saved safetensors bytes differ from reported tensor hashes")


def verify_numerics(value):
    need(value == {"component_losses_finite": 76, "component_gradients_finite": 76,
                   "unit_losses_finite": 20, "unit_gradients_finite": 20,
                   "optimizer_states_finite": 5},
         "finite numerical-check accounting differs")


def verify_resource(resource):
    resource = Path(resource).resolve()
    report_path = resource / "report.json" if resource.is_dir() else resource
    directory = report_path.parent
    report_sha = file_digest(report_path)
    report = read_json(report_path)
    need(report.get("schema") == RESOURCE_SCHEMA
         and report.get("completed") is True
         and report.get("terminal_status") in ("success", "rejected")
         and report.get("synthetic_only") is True
         and report.get("real_training_run") is False
         and report.get("old_adapter_loaded") is False,
         "completed synthetic resource report differs")
    manifest, token_rows = _qualified_data()
    need(report.get("data") == {
        "manifest": {"path": path_key(DATA_MANIFEST), "sha256": DATA_MANIFEST_SHA256},
        "proof": {"path": path_key(DATA_PROOF), "sha256": DATA_PROOF_SHA256}},
        "resource report data lineage differs")
    need(report.get("versions") == VERSIONS and report.get("limits") == LIMITS
         and report.get("recipe") == RECIPE, "resource versions, limits, or recipe differ")
    before = report.get("source_hashes_before")
    need(isinstance(before, dict) and before == report.get("source_hashes_after"),
         "resource source snapshots differ")
    for name, expected in before.items():
        need(file_digest(ROOT / name) == expected, f"resource source changed: {name}")
    required_sources = set(manifest["source_hashes_after"])
    required_sources.update(path_key(DATA / name) for name in manifest["file_hashes"])
    required_sources.update({path_key(DATA_MANIFEST), path_key(DATA_PROOF),
        "scripts/probe_difference_pairwise_lora_resources.py",
        "scripts/verify_difference_pairwise_lora_resources.py",
        "scripts/difference_pairwise_lora_model.py", "scripts/long_context_lora_model.py",
        "scripts/long_context_lora_resources.py", "scripts/probe_long_context_lora_resources.py"})
    need(set(before) == required_sources, "resource source inventory differs")
    representatives, schedule = expected_representatives_and_schedule(token_rows)
    need(report.get("representatives") == representatives and report.get("schedule") == schedule,
         "representative prompt shapes or synthetic schedule differ")
    timing, complete_times = verify_measurements(report, schedule, representatives)
    verify_numerics(report.get("numerics"))
    verify_logits(report.get("equivalence"), representatives, reload=False)
    verify_logits(report.get("reload"), representatives, reload=True)
    verify_trainable_and_adapter(report, directory)
    stages = expected_observation_stages(schedule, representatives)
    resource_summary, hardware_feasible = verify_observations(
        report.get("resource_observations"), stages, report.get("baseline"))
    elapsed = report.get("elapsed_seconds")
    interval = report["finalization_interval"]
    need(finite(elapsed, nonnegative=True) and interval["end_elapsed"] <= elapsed + 1e-9,
         "finalization interval exceeds whole probe elapsed time")
    hardware_feasible = hardware_feasible and finite(elapsed, nonnegative=True) \
        and elapsed <= LIMITS["whole_seconds"] and max(complete_times) <= LIMITS["update_seconds"]
    counts = manifest["counts"]
    estimate = 1.5 * (counts["repair_units"] * timing["unit_gradient_p95"]
        + counts["updates"] * timing["optimizer_update_p95"]
        + (counts["E_mechanism"] + counts["E_train640"]) * timing["pair_inference_p95"]
        + timing["save_reload_finalization_seconds"]) + 300
    expected_estimate = {"seconds": estimate, "within_7200": estimate <= 7200}
    declared_estimate = report.get("runtime_estimate")
    need(isinstance(declared_estimate, dict)
         and declared_estimate.get("within_7200") is expected_estimate["within_7200"]
         and finite(declared_estimate.get("seconds"), nonnegative=True)
         and math.isclose(declared_estimate["seconds"], estimate, rel_tol=1e-12, abs_tol=1e-8),
         "7,200-second launch estimate differs")
    need(report.get("resource_feasible") is hardware_feasible
         and report.get("terminal_status") == ("success" if hardware_feasible else "rejected"),
         "resource feasibility declaration differs")
    source_after = {name: file_digest(ROOT / name) for name in sorted(before)}
    need(source_after == before and file_digest(report_path) == report_sha,
         "resource inputs or report changed during verification")
    return {"stage": "resource", "passed": hardware_feasible,
        "launch_allowed": hardware_feasible and expected_estimate["within_7200"],
        "resource_report_sha256": report_sha,
        "data_manifest_sha256": DATA_MANIFEST_SHA256, "data_proof_sha256": DATA_PROOF_SHA256,
        "source_hashes_sha256": digest_value(before), "representatives": representatives,
        "counts": {"units": 20, "components": sum(len(x["components"]) for x in schedule),
                   "updates": 5, "inferences": 40, "resource_observations": len(stages)},
        "timing_summary": {**timing, "complete_update_p95": p95(complete_times)},
        "runtime_estimate": expected_estimate,
        "resource_summary": resource_summary,
        "verifier_source": {"path": path_key(Path(__file__)),
                            "sha256": file_digest(Path(__file__))},
        "model_weights_loaded_for_verification": False,
        "fresh_dev_final_targets_read": False}


def write_proof(path, payload):
    path = Path(path)
    need(not path.exists(), "proof output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"schema": SCHEMA, "payload": payload, "digest": digest_value(payload)}
    path.write_bytes(canonical_bytes(envelope) + b"\n")
    return envelope


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = verify_resource(args.resource)
        if not payload["passed"]:
            print(json.dumps({"schema": SCHEMA, "passed": False,
                              "reason": "resource_gate_rejected"}, sort_keys=True), file=sys.stderr)
            return 2
        write_proof(args.output, payload)
    except VerificationError as exc:
        print(json.dumps({"schema": SCHEMA, "passed": False, "error": str(exc)},
                         ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"schema": SCHEMA, "passed": True, "output": str(args.output)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
