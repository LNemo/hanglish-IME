"""Independently qualify the measured pairwise LoRA probe under a six-hour cap.

This verifier intentionally does not import the probe, the rejected resource
verifier, or the forensic-audit implementation.  It reads their immutable
artifacts as untrusted bytes and reconstructs the relevant evidence itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "difference-pairwise-lora-budget-proof-v1"
RESOURCE_SCHEMA = "difference-pairwise-lora-resource-v1"
REPORT = ROOT / ".omx/artifacts/difference-pairwise-lora-resource-v2/report.json"
AUDIT = ROOT / ".omx/artifacts/difference-pairwise-lora-resource-v2-failure-audit.json"
DATA = ROOT / ".omx/artifacts/difference-pairwise-lora-data-v1"
MANIFEST = DATA / "manifest.json"
DATA_PROOF = ROOT / ".omx/artifacts/difference-pairwise-lora-data-v1-proof.json"
REPORT_SHA256 = "718f68d0863d02de56d127c9a9b590d6910ca28da053bdcf209087cb07489b70"
AUDIT_SHA256 = "03edebb221bea975fe6cab375ea9d1767374d2a231b0b94ba459ea3ac10a24f3"
MANIFEST_SHA256 = "32e83b145e71de39912e122f9925c9cf362875a5206e49dddcb7611a464be132"
DATA_PROOF_SHA256 = "03943e073ec333b4e8a8e8386ace042c0c3dda6370cdc5b284961262bd2dea72"
OLD_LIMIT_SECONDS = 7200.0
NEW_LIMIT_SECONDS = 21600.0
EXPECTED_ESTIMATE_SECONDS = 21351.99752716787
LOGICAL_FIT_ID = "difference-pairwise-lora-fit-budget-v1"
SOLE_OUTPUT = ".omx/artifacts/difference-pairwise-lora-train-v1"
PLANNING_FILES = {
    ".omx/context/pairwise-fit-budget-20260916.md": "9e4632a78b757335cfbbaef47f9f7a3c7dfa2555915feb7e32352fbf0d9d320f",
    ".omx/plans/pairwise-fit-budget-v1.md": "22be5ac70a08223fcc3ad7c4dac72c6852e0db7797694c0be756b34eaed5e20d",
    ".omx/plans/pairwise-fit-budget-test-spec-v1.md": "a9ca4df3175f076e78f8f800f7f5e0054588a718d4e4f0ec13730dce5e2ea389",
    ".omx/plans/pairwise-fit-budget-handoff-v1.json": "cec25fb88a5005a7cf7b9518ffde2dddc39f2494ac4ad282be0b4932c07ff1ce",
    ".omx/plans/pairwise-fit-budget-architect-review-v2.md": "042add6d4c539226e46a4f798f47c0ed65551584b5d4f975ca7fcd1f6c06a2f8",
    ".omx/plans/pairwise-fit-budget-critic-review-v1.md": "97bc201312259a5c9a2cc4530eb88e6d5dee24b174f98b4d8f8b99f494646927",
    ".omx/plans/lora-successor-handoff-v1.json": "785ef5ce678c684563a22f4c21c9dfcf9350a791256e0fe51de8b19e0da71092",
    ".omx/plans/lora-successor-compact-kind-handoff-v1.json": "90f3b6382af349bf61b96862b658bb81bb87caaf7acb613db400c0737291f4c4",
}
LIMITS = {
    "rss_bytes": 6 * 1024**3, "mlx_peak_bytes": 6 * 1024**3,
    "cache_bytes": 256 * 1024**2, "whole_seconds": 900.0,
    "update_seconds": 120.0, "minimum_system_free_percent": 10.0,
    "maximum_swap_growth_bytes": 512 * 1024**2,
}
RECIPE = {
    "seed": 20260915, "lora_layers": 4,
    "lora_config": {"rank": 8, "scale": 20.0, "dropout": 0.0,
                    "keys": ["self_attn.q_proj", "self_attn.v_proj"]},
    "learning_rate": 1e-5, "adam_betas": [0.9, 0.999], "adam_eps": 1e-8,
    "bias_correction": True, "weight_decay": 0.0,
    "accumulation_units": 4, "clip_norm": 1.0,
}
VERSIONS = {"mlx": "0.31.2", "mlx-lm": "0.31.3", "transformers": "5.12.1"}
EXPECTED_NAMES = sorted(
    f"model.layers.{layer}.self_attn.{projection}.{suffix}"
    for layer in range(32, 36) for projection in ("q_proj", "v_proj")
    for suffix in ("lora_a", "lora_b")
)
EXPECTED_SHAPES = {
    f"model.layers.{layer}.self_attn.{projection}.{suffix}": shape
    for layer in range(32, 36)
    for projection, suffix, shape in (
        ("q_proj", "lora_a", [2560, 8]), ("q_proj", "lora_b", [8, 4096]),
        ("v_proj", "lora_a", [2560, 8]), ("v_proj", "lora_b", [8, 1024]),
    )
}


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
        raise VerificationError("value is not canonical JSON") from exc


def digest_value(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise VerificationError(f"required file unavailable: {path}") from exc
    return digest.hexdigest()


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
    return type(value) in (int, float) and math.isfinite(value) and (
        not nonnegative or value >= 0)


def p95(values):
    need(values and all(finite(value, nonnegative=True) for value in values),
         "timing series contains invalid values")
    values = sorted(values)
    return values[math.ceil(0.95 * len(values)) - 1]


def qualify_budget(estimate, *, old_limit=OLD_LIMIT_SECONDS,
                   new_limit=NEW_LIMIT_SECONDS):
    need(finite(estimate, nonnegative=True)
         and math.isclose(estimate, EXPECTED_ESTIMATE_SECONDS,
                          rel_tol=1e-12, abs_tol=1e-8),
         "runtime formula result differs")
    need(old_limit == OLD_LIMIT_SECONDS and new_limit == NEW_LIMIT_SECONDS,
         "only the reviewed 7,200 to 21,600 policy change is allowed")
    need(estimate > old_limit and estimate <= new_limit,
         "old-failed/new-passing budget verdict differs")
    return {
        "old_policy": {"limit_seconds": old_limit,
                       "estimate_seconds": estimate, "launch_allowed": False},
        "new_policy": {"limit_seconds": new_limit,
                       "estimate_seconds": estimate, "launch_allowed": True},
    }


def _validate_planning():
    for name, expected in PLANNING_FILES.items():
        need(file_digest(ROOT / name) == expected, f"approved planning input changed: {name}")
    handoff = read_json(ROOT / ".omx/plans/pairwise-fit-budget-handoff-v1.json")
    need(handoff.get("terminal_planning_state") == "complete"
         and handoff.get("review_cycle") == 2
         and handoff.get("ralplan_consensus_gate", {}).get("complete") is True
         and handoff.get("ralplan_execution_handoff", {}).get("authorized") is True
         and handoff.get("logical_fit_id") == LOGICAL_FIT_ID
         and handoff.get("fit_output") == SOLE_OUTPUT
         and handoff.get("fit_limit_seconds") == 21600
         and handoff.get("model_launch_ready") is False,
         "approved handoff contract differs")


def _validate_data():
    need(file_digest(MANIFEST) == MANIFEST_SHA256
         and file_digest(DATA_PROOF) == DATA_PROOF_SHA256,
         "qualified data pins differ")
    manifest = read_json(MANIFEST)
    proof = read_json(DATA_PROOF)
    payload = proof.get("payload")
    need(isinstance(payload, dict) and proof.get("digest") == digest_value(payload)
         and payload.get("passed") is True and payload.get("stage") == "data"
         and payload.get("data_manifest_sha256") == MANIFEST_SHA256,
         "qualified data proof differs")
    need(manifest.get("completed") is True
         and manifest.get("counts") == payload.get("counts")
         and manifest.get("file_hashes") == payload.get("file_hashes"),
         "qualified data manifest differs")
    for name, expected in manifest["file_hashes"].items():
        need(file_digest(DATA / name) == expected, f"qualified data changed: {name}")
    for name, expected in manifest["source_hashes_after"].items():
        need(file_digest(ROOT / name) == expected, f"qualified data source changed: {name}")
    tokens = read_jsonl(DATA / "tokens.jsonl")
    need(len(tokens) == 1928 and len({row.get("pair_id") for row in tokens}) == 1928,
         "qualified token count or identity differs")
    for row in tokens:
        need(type(row.get("prompt_tokens")) is int
             and row["prompt_tokens"] == len(row.get("tokens", []))
             and 0 < row["prompt_tokens"] <= 1024
             and digest_value(row["tokens"]) == row.get("tokens_sha256")
             and row.get("digit_continuation_ids") == {"1": 16, "2": 17},
             "qualified token row differs")
    return manifest, tokens


def _expected_probe(tokens):
    ordered = sorted(tokens, key=lambda row: (row["prompt_tokens"], row["pair_id"]))
    ranks = (0, (len(ordered) - 1) // 2, math.ceil(.95 * len(ordered)) - 1,
             len(ordered) - 1)
    reps = []
    for quantile, rank in zip(("min", "median", "p95", "max"), ranks):
        row = ordered[rank]
        reps.append({"quantile": quantile, "pair_id": row["pair_id"],
                     "prompt_tokens": row["prompt_tokens"],
                     "tokens_sha256": row["tokens_sha256"]})
    schedule, cursor = [], 0
    for index in range(20):
        controls = 2 if index % 5 == 4 else 3
        targets = ["SWITCH"] + ["KEEP"] * controls
        weights = [.5] + [.5 / controls] * controls
        components = []
        for target, weight in zip(targets, weights):
            row = reps[cursor % 4]
            components.append({"pair_id": row["pair_id"], "target": target,
                               "weight": weight, "prompt_tokens": row["prompt_tokens"],
                               "tokens_sha256": row["tokens_sha256"]})
            cursor += 1
        schedule.append({"unit": index + 1, "update": index // 4 + 1,
                         "components": components})
    stages = ["before_model_load", "model_loaded", "fresh_adapter_initialized"]
    for update in range(1, 6):
        for unit in schedule[(update - 1) * 4:update * 4]:
            stages += [f"unit_{unit['unit']}_component_{i}"
                       for i in range(len(unit["components"]))]
            stages.append(f"unit_{unit['unit']}_complete")
        stages.append(f"update_{update}_complete")
    stages += [f"equivalence_{row['quantile']}" for row in reps]
    stages += [f"inference_{row['quantile']}_{repeat}"
               for row in reps for repeat in range(1, 11)]
    stages += ["adapter_saved", "first_model_released"]
    stages += [f"reload_{row['quantile']}" for row in reps]
    stages += ["final_model_released", "terminal"]
    return reps, schedule, stages


def _validate_measurements(report, schedule, reps):
    units = report.get("units")
    need(isinstance(units, list) and len(units) == 20, "unit count differs")
    component_times, unit_times = [], []
    for actual, expected in zip(units, schedule):
        need(actual.get("unit") == expected["unit"]
             and actual.get("update") == expected["update"]
             and finite(actual.get("loss"))
             and finite(actual.get("unit_gradient_seconds"), nonnegative=True),
             "unit identity or measurement differs")
        components = actual.get("components")
        need(isinstance(components, list)
             and len(components) == len(expected["components"]),
             "component count differs")
        weighted = []
        for number, (component, planned) in enumerate(zip(components, expected["components"])):
            need(component.get("component_index") == number
                 and component.get("target") == planned["target"]
                 and component.get("weight") == planned["weight"]
                 and finite(component.get("loss"))
                 and finite(component.get("gradient_seconds"), nonnegative=True),
                 "component measurement differs")
            weighted.append(component["weight"] * component["loss"])
            component_times.append(component["gradient_seconds"])
        need(math.isclose(actual["loss"], math.fsum(weighted), rel_tol=1e-12, abs_tol=1e-12)
             and actual["unit_gradient_seconds"] + 1e-9 >= math.fsum(
                 row["gradient_seconds"] for row in components),
             "unit loss or inclusive interval differs")
        unit_times.append(actual["unit_gradient_seconds"])
    need(sum(len(row["components"]) for row in units) == 76, "component total differs")
    updates = report.get("updates")
    need(isinstance(updates, list) and len(updates) == 5, "update count differs")
    optimizer_times, complete_times = [], []
    for index, row in enumerate(updates, 1):
        related = units[(index - 1) * 4:index * 4]
        need(row.get("update") == index
             and row.get("unit_ids") == list(range(index * 4 - 3, index * 4 + 1))
             and finite(row.get("gradient_norm_before_clip"), nonnegative=True)
             and finite(row.get("optimizer_seconds"), nonnegative=True)
             and finite(row.get("complete_update_seconds"), nonnegative=True)
             and row["complete_update_seconds"] <= LIMITS["update_seconds"]
             and row["complete_update_seconds"] + 1e-9 >=
                 math.fsum(item["unit_gradient_seconds"] for item in related)
                 + row["optimizer_seconds"],
             "update measurement or inclusive interval differs")
        optimizer_times.append(row["optimizer_seconds"])
        complete_times.append(row["complete_update_seconds"])
    inference = report.get("inference")
    need(isinstance(inference, list) and len(inference) == 40, "inference count differs")
    inference_times = []
    expected_inference = ((rep, repeat) for rep in reps for repeat in range(1, 11))
    for row, (rep, repeat) in zip(inference, expected_inference):
        logits = row.get("logits")
        need(row.get("pair_id") == rep["pair_id"] and row.get("repetition") == repeat
             and finite(row.get("seconds"), nonnegative=True)
             and isinstance(logits, list) and len(logits) == 2
             and all(finite(value) for value in logits), "inference evidence differs")
        inference_times.append(row["seconds"])
    interval = report.get("finalization_interval")
    need(isinstance(interval, dict) and set(interval) == {"start_elapsed", "end_elapsed"}
         and all(finite(value, nonnegative=True) for value in interval.values())
         and interval["start_elapsed"] <= interval["end_elapsed"]
         and interval["end_elapsed"] <= report.get("elapsed_seconds", -1) + 1e-9,
         "finalization interval differs")
    timing = {
        "component_gradient_p95": p95(component_times),
        "unit_gradient_p95": p95(unit_times),
        "optimizer_update_p95": p95(optimizer_times),
        "pair_inference_p95": p95(inference_times),
        "save_reload_finalization_seconds": interval["end_elapsed"] - interval["start_elapsed"],
    }
    declared = report.get("timing_summary")
    need(isinstance(declared, dict) and set(declared) == set(timing)
         and all(math.isclose(declared[name], value, rel_tol=1e-12, abs_tol=1e-9)
                 for name, value in timing.items()), "timing summary differs")
    return timing, complete_times


def _validate_logits(rows, reps, *, reload):
    need(isinstance(rows, list) and len(rows) == 4, "parity row count differs")
    for row, rep in zip(rows, reps):
        left = row.get("actual_logits" if reload else "selected_logits")
        right = row.get("expected_logits" if reload else "ordinary_logits")
        need(row.get("pair_id") == rep["pair_id"]
             and isinstance(left, list) and isinstance(right, list)
             and len(left) == len(right) == 2
             and all(finite(value) for value in left + right)
             and max(range(2), key=left.__getitem__) == max(range(2), key=right.__getitem__),
             "logit parity differs")
        for actual, expected in zip(left, right):
            tolerance = 1e-5 + 1e-5 * abs(expected) if reload else .125
            need(abs(actual - expected) <= tolerance, "logit parity tolerance exceeded")


def read_safetensors(path):
    try:
        raw = Path(path).read_bytes()
        need(len(raw) >= 8, "safetensors header truncated")
        header_size = struct.unpack("<Q", raw[:8])[0]
        need(0 < header_size <= len(raw) - 8, "safetensors header size differs")
        header_bytes = raw[8:8 + header_size]
        header = json.loads(header_bytes)
    except (OSError, struct.error, json.JSONDecodeError) as exc:
        raise VerificationError("malformed safetensors file") from exc
    need(isinstance(header, dict), "safetensors header must be an object")
    body = raw[8 + header_size:]
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    need(sorted(tensors) == EXPECTED_NAMES, "safetensors tensor names differ")
    hashes, spans, scalars = {}, [], 0
    for name in EXPECTED_NAMES:
        item = tensors[name]
        need(isinstance(item, dict) and item.get("dtype") == "F32"
             and item.get("shape") == EXPECTED_SHAPES[name]
             and isinstance(item.get("data_offsets"), list)
             and len(item["data_offsets"]) == 2
             and all(type(value) is int for value in item["data_offsets"]),
             f"safetensors metadata differs: {name}")
        start, end = item["data_offsets"]
        count = math.prod(item["shape"])
        need(0 <= start < end <= len(body) and end - start == count * 4,
             f"safetensors body span differs: {name}")
        domain = f"mlx.core.float32:{tuple(item['shape'])}:".encode()
        hashes[name] = hashlib.sha256(domain + body[start:end]).hexdigest()
        spans.append((start, end)); scalars += count
    spans.sort()
    need(spans[0][0] == 0
         and all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
         and spans[-1][1] == len(body) and scalars == 327680,
         "safetensors body coverage or scalar count differs")
    return {"hashes": hashes, "tensor_count": 16, "scalar_count": scalars,
            "header_bytes": header_size, "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
            "body_bytes": len(body), "body_sha256": hashlib.sha256(body).hexdigest(),
            "bytes": len(raw), "file_sha256": hashlib.sha256(raw).hexdigest()}


def _validate_adapter(report, resource_dir):
    trainable = report.get("trainable")
    need(trainable == {"names": EXPECTED_NAMES, "scalar_count": 327680},
         "trainable tensor inventory differs")
    maps = [report.get(name) for name in (
        "initial_trainable_tensor_hashes", "final_trainable_tensor_hashes",
        "saved_trainable_tensor_hashes", "reloaded_trainable_tensor_hashes")]
    need(all(isinstance(item, dict) and sorted(item) == EXPECTED_NAMES
             and all(isinstance(value, str) and len(value) == 64 for value in item.values())
             for item in maps), "trainable tensor hashes differ")
    need(maps[0] != maps[1] and maps[1] == maps[2] == maps[3],
         "adapter change/save/reload identity differs")
    adapter = report.get("adapter")
    need(isinstance(adapter, dict), "adapter metadata missing")
    adapter_path = ROOT / adapter.get("path", "")
    config_path = ROOT / adapter.get("config_path", "")
    need(adapter_path.resolve().is_relative_to(resource_dir.resolve())
         and config_path.resolve().is_relative_to(resource_dir.resolve())
         and file_digest(adapter_path) == adapter.get("sha256")
         and file_digest(config_path) == adapter.get("config_sha256"),
         "adapter file lineage differs")
    need(read_json(config_path) == {"fine_tune_type": "lora", "num_layers": 4,
                                    "lora_parameters": RECIPE["lora_config"]},
         "adapter configuration differs")
    parsed = read_safetensors(adapter_path)
    need(parsed["hashes"] == maps[2], "MLX-domain tensor hashes differ")
    return parsed


def _validate_resources(report, stages):
    baseline = report.get("baseline")
    records = report.get("resource_observations")
    need(isinstance(baseline, dict) and set(baseline) == {"free_percent", "swap_used_bytes"}
         and all(finite(value, nonnegative=True) for value in baseline.values()),
         "resource baseline differs")
    need(isinstance(records, list) and len(records) == 156
         and [row.get("stage") for row in records] == stages,
         "resource observation count or order differs")
    settled = []
    expected_fields = {"rss_peak_bytes", "mlx_active_bytes", "mlx_cache_bytes",
                       "mlx_peak_bytes", "free_percent", "swap_used_bytes",
                       "swap_growth_bytes", "elapsed_seconds"}
    elapsed_observations = []
    for row in records:
        need(row.get("action") in ("none", "clear_unused_mlx_cache")
             and row.get("errors") == []
             and finite(row.get("cleanup_seconds"), nonnegative=True)
             and finite(row.get("checkpoint_seconds"), nonnegative=True),
             "resource wrapper differs")
        pre, post = row.get("pre"), row.get("post")
        need(isinstance(pre, dict)
             and ((row["action"] == "none" and post is None)
                  or (row["action"] == "clear_unused_mlx_cache" and isinstance(post, dict))),
             "resource cache action differs")
        for observation in [pre] + ([post] if post is not None else []):
            need(set(observation) == expected_fields
                 and all(finite(value, nonnegative=name != "swap_growth_bytes")
                         for name, value in observation.items())
                 and math.isclose(observation["swap_growth_bytes"],
                                  observation["swap_used_bytes"] - baseline["swap_used_bytes"],
                                  rel_tol=0, abs_tol=1e-6),
                 "resource observation differs")
            elapsed_observations.append(observation["elapsed_seconds"])
        settled.append(post if post is not None else pre)
    need(all(left <= right + 1e-9
             for left, right in zip(elapsed_observations, elapsed_observations[1:])),
         "resource observation time moved backwards")
    summary = {
        "rss_peak_bytes": max(row["rss_peak_bytes"] for row in settled),
        "mlx_peak_bytes": max(row["mlx_peak_bytes"] for row in settled),
        "settled_cache_bytes": max(row["mlx_cache_bytes"] for row in settled),
        "minimum_free_percent": min(row["free_percent"] for row in settled),
        "maximum_swap_growth_bytes": max(row["swap_growth_bytes"] for row in settled),
    }
    feasible = (summary["rss_peak_bytes"] <= LIMITS["rss_bytes"]
                and summary["mlx_peak_bytes"] <= LIMITS["mlx_peak_bytes"]
                and summary["settled_cache_bytes"] <= LIMITS["cache_bytes"]
                and summary["minimum_free_percent"] >= LIMITS["minimum_system_free_percent"]
                and summary["maximum_swap_growth_bytes"] <= LIMITS["maximum_swap_growth_bytes"])
    return summary, feasible


def _validate_audit(audit, report, adapter, timing, resource_summary):
    payload = audit.get("payload")
    need(audit.get("schema") == "difference-pairwise-lora-resource-v2-failure-audit-v1"
         and isinstance(payload, dict) and audit.get("digest") == digest_value(payload),
         "forensic audit envelope differs")
    need(payload.get("schema") == audit["schema"]
         and payload.get("audit_passed") is True
         and payload.get("terminal_conclusion") ==
             "hardware_valid_launch_rejected_by_fixed_runtime_estimate"
         and payload.get("resource_report") == {"path": str(REPORT.relative_to(ROOT)),
                                                  "sha256": REPORT_SHA256}
         and payload.get("qualified_data", {}).get("source_snapshot_count") == 48
         and payload.get("qualified_data", {}).get("source_snapshots_unchanged") is True,
         "forensic audit conclusion differs")
    need(payload.get("audit_source") == {
             "path": ".omx/artifacts/difference-pairwise-lora-resource-v2-failure-audit.py",
             "sha256": "05ec5a96d532ab420112f37d411b6f8bd27745aae15e8d77ba6e9ea42d2a18de"}
         and payload.get("original_failed_verifier") == {
             "failure": "saved safetensors bytes differ from reported tensor hashes",
             "path": "scripts/verify_difference_pairwise_lora_resources.py",
             "preserved_unchanged": True,
             "sha256": "009e2fc2fd449d62b9cf410cd4b7bcacf0e2e771eb89ae910d958c4a538c822e"}
         and payload.get("mlx_imported_for_audit") is False
         and payload.get("model_weights_loaded_for_audit") is False
         and payload.get("fresh_dev_final_targets_read") is False,
         "forensic implementation boundary differs")
    launch = payload.get("launch_gate")
    need(launch == {"fixed_estimate_seconds": EXPECTED_ESTIMATE_SECONDS,
                    "launch_allowed": False, "limit_seconds": 7200,
                    "quality_claim_made": False, "quality_evaluation_allowed": False,
                    "training_started": False, "within_7200": False},
         "historical 7,200-second rejection differs")
    recorded_adapter = payload.get("adapter")
    need(recorded_adapter.get("reported_hash_domain") == "mlx.core.float32"
         and recorded_adapter.get("reported_hash_matches") == 16
         and recorded_adapter.get("recomputed_mlx_tensor_hashes") == adapter["hashes"]
         and recorded_adapter.get("reported_tensor_hashes") == report["saved_trainable_tensor_hashes"]
         and all(recorded_adapter.get(name) == adapter[name] for name in (
             "tensor_count", "scalar_count", "header_bytes", "header_sha256",
             "body_bytes", "body_sha256", "bytes", "file_sha256")),
         "forensic adapter reconstruction differs")
    contract = payload.get("probe_contract")
    need(contract.get("synthetic_units") == 20
         and contract.get("components") == 76
         and contract.get("optimizer_updates") == 5
         and contract.get("inferences") == 40
         and contract.get("resource_observations") == 156
         and contract.get("resource_summary") == resource_summary
         and contract.get("timing_summary") == timing
         and math.isclose(contract.get("complete_update_p95", -1),
                          p95([row["complete_update_seconds"] for row in report["updates"]]),
                          rel_tol=1e-12, abs_tol=1e-9)
         and contract.get("numerics") == report.get("numerics"),
         "forensic probe summary differs")


def verify_budget(report_path=REPORT, audit_path=AUDIT):
    report_path, audit_path = Path(report_path).resolve(), Path(audit_path).resolve()
    need(report_path == REPORT.resolve() and audit_path == AUDIT.resolve(),
         "only the immutable v2 report and audit are accepted")
    need(file_digest(report_path) == REPORT_SHA256, "resource report bytes changed")
    need(file_digest(audit_path) == AUDIT_SHA256, "failure audit bytes changed")
    _validate_planning()
    manifest, tokens = _validate_data()
    report, audit = read_json(report_path), read_json(audit_path)
    need(report.get("schema") == RESOURCE_SCHEMA and report.get("completed") is True
         and report.get("resource_feasible") is True
         and report.get("terminal_status") == "success"
         and report.get("synthetic_only") is True
         and report.get("real_training_run") is False
         and report.get("old_adapter_loaded") is False
         and report.get("limits") == LIMITS and report.get("recipe") == RECIPE
         and report.get("versions") == VERSIONS,
         "resource probe contract differs")
    need(report.get("data") == {
        "manifest": {"path": str(MANIFEST.relative_to(ROOT)), "sha256": MANIFEST_SHA256},
        "proof": {"path": str(DATA_PROOF.relative_to(ROOT)), "sha256": DATA_PROOF_SHA256}},
        "resource data lineage differs")
    sources = report.get("source_hashes_before")
    need(isinstance(sources, dict) and len(sources) == 48
         and sources == report.get("source_hashes_after"),
         "48-source inventory differs")
    for name, expected in sources.items():
        need(file_digest(ROOT / name) == expected, f"resource source changed: {name}")
    reps, schedule, stages = _expected_probe(tokens)
    need(report.get("representatives") == reps and report.get("schedule") == schedule,
         "representative or schedule evidence differs")
    timing, complete_times = _validate_measurements(report, schedule, reps)
    need(report.get("numerics") == {"component_losses_finite": 76,
         "component_gradients_finite": 76, "unit_losses_finite": 20,
         "unit_gradients_finite": 20, "optimizer_states_finite": 5},
         "numerical accounting differs")
    _validate_logits(report.get("equivalence"), reps, reload=False)
    _validate_logits(report.get("reload"), reps, reload=True)
    adapter = _validate_adapter(report, report_path.parent)
    resource_summary, feasible = _validate_resources(report, stages)
    need(feasible and report["elapsed_seconds"] <= LIMITS["whole_seconds"]
         and max(complete_times) <= LIMITS["update_seconds"],
         "physical resource limits failed")
    counts = manifest["counts"]
    need(counts.get("required_unique_pairs") == 1928
         and counts.get("repair_units") == 59 and counts.get("component_references") == 232
         and counts.get("updates") == 15 and counts.get("final_update_units") == 3,
         "training schedule counts differ")
    estimate = 1.5 * (counts["repair_units"] * timing["unit_gradient_p95"]
        + counts["updates"] * timing["optimizer_update_p95"]
        + (counts["E_mechanism"] + counts["E_train640"]) * timing["pair_inference_p95"]
        + timing["save_reload_finalization_seconds"]) + 300
    policies = qualify_budget(estimate)
    need(report.get("runtime_estimate") == {"seconds": estimate, "within_7200": False},
         "runtime estimate or policy verdict differs")
    _validate_audit(audit, report, adapter, timing, resource_summary)
    _validate_planning()
    need(file_digest(report_path) == REPORT_SHA256 and file_digest(audit_path) == AUDIT_SHA256,
         "inputs changed during qualification")
    return {
        "stage": "budget_qualification", "passed": True, "launch_allowed": True,
        "logical_fit_id": LOGICAL_FIT_ID, "sole_output": SOLE_OUTPUT,
        **policies,
        "resource_report": {"path": str(REPORT.relative_to(ROOT)), "sha256": REPORT_SHA256},
        "failure_audit": {"path": str(AUDIT.relative_to(ROOT)), "sha256": AUDIT_SHA256},
        "qualified_data": {"manifest_sha256": MANIFEST_SHA256,
                           "proof_sha256": DATA_PROOF_SHA256},
        "counts": {"qualified_pairs": 1928, "probe_units": 20,
                   "probe_components": 76, "probe_updates": 5,
                   "probe_inferences": 40, "resource_observations": 156,
                   "adapter_tensors": 16, "adapter_scalars": 327680},
        "limits": LIMITS, "recipe": RECIPE, "timing_summary": timing,
        "resource_summary": resource_summary,
        "source_inventory": {"count": 48, "sha256": digest_value(sources),
                             "hashes": sources},
        "planning_files": PLANNING_FILES,
        "verifier_source": {"path": str(Path(__file__).resolve().relative_to(ROOT)),
                            "sha256": file_digest(Path(__file__))},
        "model_or_mlx_loaded_for_verification": False,
        "training_started": False, "training_quality_claimed": False,
        "deployment_authorized": False, "second_fit_authorized": False,
        "execution_prelaunch_required": True,
    }


def write_proof(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = {"schema": SCHEMA, "payload": payload, "digest": digest_value(payload)}
    data = canonical_bytes(envelope) + b"\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise VerificationError("proof output already exists") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise
    return envelope


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-report", type=Path, default=REPORT)
    parser.add_argument("--failure-audit", type=Path, default=AUDIT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = verify_budget(args.resource_report, args.failure_audit)
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
