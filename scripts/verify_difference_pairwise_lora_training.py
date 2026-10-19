"""Independently verify the one-shot pairwise fit without loading a model.

Predictions are replayed from raw logits and frozen candidate bytes. The trainer
and its evaluation implementation are intentionally not imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_difference_pairwise_lora_budget as budget
from scripts import verify_difference_pairwise_lora_recovery as recovery
from scripts.final_choice_policy import finite_candidate_indices, proposal_allowed

SCHEMA = "difference-pairwise-lora-training-proof-v2"
EVALUATION_SCHEMA = "difference-pairwise-lora-evaluation-v1"
LOGICAL_FIT_ID = recovery.LOGICAL_FIT_ID
SOLE_OUTPUT = recovery.SOLE_OUTPUT
OUTPUT = ROOT / SOLE_OUTPUT
PRELAUNCH = ROOT / recovery.PRELAUNCH_PATH
BUDGET_PROOF = ROOT / ".omx/artifacts/difference-pairwise-lora-budget-v1-proof.json"
RECOVERY_PROOF = ROOT / recovery.RECOVERY_PROOF_PATH
ADDITIONS = {
    "scripts/run_difference_pairwise_lora.py", "scripts/train_difference_pairwise_lora.py",
    "scripts/verify_difference_pairwise_lora_budget.py",
    "scripts/verify_difference_pairwise_lora_training.py",
    "scripts/difference_pairwise_lora_model.py", "scripts/difference_pairwise_lora_evaluation.py",
    "scripts/long_context_lora_model.py", "scripts/long_context_lora_resources.py",
    "scripts/probe_long_context_lora_resources.py", "scripts/final_choice_policy.py",
    "scripts/prepare_difference_pairwise_lora_data.py",
    "scripts/verify_difference_pairwise_lora_recovery.py",
    "tests/test_run_difference_pairwise_lora.py", "tests/test_train_difference_pairwise_lora.py",
    "tests/test_pairwise_resource_binding.py",
    "tests/test_verify_difference_pairwise_lora_training.py",
    "tests/test_verify_difference_pairwise_lora_recovery.py",
}
COUNTS = {"units": 59, "components": 232, "updates": 15, "mechanism_rows": 175,
          "E_mechanism": 481, "train640_rows": 640, "E_train640": 1556}
LIMITS = {**budget.LIMITS, "whole_seconds": 21600.0}
RECIPE = {**budget.RECIPE, "final_update_units": 3, "epochs": 1}
need = budget.need
canonical = budget.canonical_bytes
sha = budget.file_digest
read_json = budget.read_json
read_jsonl = budget.read_jsonl
finite = budget.finite


def bound_file(relative, expected):
    path = ROOT / relative
    need(path.resolve().is_relative_to(ROOT.resolve()) and sha(path) == expected,
         f"bound file differs: {relative}")


def validate_prelaunch():
    proof = read_json(BUDGET_PROOF)
    expected = budget.verify_budget()
    need(proof.get("schema") == budget.SCHEMA and proof.get("payload") == expected
         and proof.get("digest") == budget.digest_value(expected), "budget proof replay differs")
    receipt = read_json(PRELAUNCH)
    payload = receipt.get("payload")
    need(isinstance(payload, dict) and receipt.get("digest") == budget.digest_value(payload),
         "execution prelaunch envelope differs")
    recovery.validate_prelaunch_shape(payload)
    recovery_payload = recovery.validate_recovery_proof(RECOVERY_PROOF)
    need(payload.get("schema") == recovery.PRELAUNCH_SCHEMA
         and payload.get("passed") is True and payload.get("launch_allowed") is True
         and payload.get("logical_fit_id") == LOGICAL_FIT_ID
         and payload.get("sole_output") == SOLE_OUTPUT
         and payload.get("package_versions") == budget.VERSIONS,
         "execution prelaunch identity differs")
    bp = {"path": str(BUDGET_PROOF.relative_to(ROOT)), "sha256": sha(BUDGET_PROOF)}
    need(payload.get("budget_proof") == bp, "execution budget binding differs")
    rp = {"path": recovery.RECOVERY_PROOF_PATH, "sha256": sha(RECOVERY_PROOF)}
    need(payload.get("recovery_proof") == rp
         and payload.get("worker_python") == recovery_payload["preflight"]["interpreter"],
         "execution recovery or interpreter binding differs")
    sources = expected["source_inventory"]["hashes"]
    inventory = payload.get("inventory")
    artifacts = {rp["path"], recovery_payload["preflight"]["path"],
                 recovery_payload["checks"]["path"], recovery_payload["execution_review"]["path"]}
    need(isinstance(inventory, dict) and set(inventory) == (
        set(sources) | ADDITIONS | set(recovery_payload["execution_sources"])
        | {bp["path"], *artifacts}),
         "execution inventory is incomplete")
    need(all(inventory.get(name) == value for name, value in sources.items()),
         "original source inventory differs")
    for name, value in inventory.items():
        bound_file(name, value)
    return payload


def load_data():
    manifest, tokens = budget._validate_data()
    pairs = read_jsonl(budget.DATA / "pairs.jsonl")
    units = read_jsonl(budget.DATA / "units.jsonl")
    schedule = read_jsonl(budget.DATA / "schedule.jsonl")
    selected = read_jsonl(budget.DATA / "train640.jsonl")
    source_rows = {}
    for role, source in manifest["fit_sources"].items():
        for row in read_jsonl(ROOT / source["cases"]["path"]):
            source_rows[(role, row["id"])] = row
    mechanisms = sorted({(p["role"], p["row_id"]) for p in pairs if "mechanism" in p["scopes"]})
    mechanism = [{"role": role, "row": source_rows[(role, row_id)], "stratum": "mechanism"}
                 for role, row_id in mechanisms]
    diagnostic = {(x["role"], x["row"]["id"]): x["row"]
                  for x in read_jsonl(ROOT / manifest["train640_source"]["path"])}
    train = [{"role": x["role"], "row": diagnostic[(x["role"], x["row_id"])],
              "stratum": x["stratum"]} for x in selected]
    need(len(pairs) == 1928 and len(mechanism) == 175 and len(train) == 640,
         "qualified row accounting differs")
    return {"pairs": pairs, "tokens": tokens, "units": units, "schedule": schedule,
            "items": {"mechanism": mechanism, "train640": train}}


def verify_units(report, data):
    units, updates = report.get("units"), report.get("updates")
    need(isinstance(units, list) and len(units) == 59
         and isinstance(updates, list) and len(updates) == 15, "training accounting incomplete")
    planned = {row["unit_id"]: row for row in data["units"]}
    for actual, scheduled in zip(units, data["schedule"]):
        expected = planned[scheduled["unit_id"]]
        need(actual.get("order") == expected["order"]
             and actual.get("unit_id") == expected["unit_id"]
             and actual.get("update") == scheduled["update"]
             and finite(actual.get("loss"), nonnegative=True)
             and finite(actual.get("unit_gradient_seconds"), nonnegative=True),
             "training unit identity or numeric evidence differs")
        components, refs = actual.get("components"), expected["references"]
        need(isinstance(components, list) and len(components) == len(refs), "component count differs")
        for index, (component, ref) in enumerate(zip(components, refs)):
            need(component.get("component_index") == index
                 and all(component.get(k) == ref[k] for k in ("target", "weight", "pair_id"))
                 and finite(component.get("loss"), nonnegative=True)
                 and finite(component.get("gradient_seconds"), nonnegative=True),
                 "component schedule or numeric evidence differs")
        need(math.isclose(actual["loss"], math.fsum(c["weight"] * c["loss"] for c in components),
                          rel_tol=1e-12, abs_tol=1e-12)
             and actual["unit_gradient_seconds"] + 1e-8 >= sum(c["gradient_seconds"] for c in components),
             "unit loss or inclusive interval differs")
    for index, update in enumerate(updates, 1):
        related = [row for row in units if row["update"] == index]
        count = 3 if index == 15 else 4
        need(update.get("update") == index and len(related) == count
             and update.get("unit_ids") == [row["unit_id"] for row in related]
             and update.get("actual_unit_count") == count
             and all(finite(update.get(k), nonnegative=True) for k in
                     ("gradient_norm_before_clip", "optimizer_seconds", "complete_update_seconds"))
             and update["complete_update_seconds"] <= 120
             and update["complete_update_seconds"] + 1e-8 >=
                 sum(row["unit_gradient_seconds"] for row in related) + update["optimizer_seconds"],
             "optimizer grouping, norm, or inclusive interval differs")
    need(sum(len(row["components"]) for row in units) == 232, "component total differs")


def permitted_indices(row):
    candidates = row["candidates"]
    indices = finite_candidate_indices(candidates)
    if not indices or indices[0] != 0:
        return []
    return [i for i in indices if i and proposal_allowed(row["input"], candidates[0], candidates[i])]


def replay_prediction(item, prediction, pairs, inference):
    row, role = item["row"], item["role"]
    candidates, raw = row["candidates"], row["raw_bypass_response"]
    permitted = permitted_indices(row)
    need([p["alternative_source_index"] for p in pairs] == permitted,
         "source-permitted pair inventory differs")
    need(len(inference) == len(pairs), "inference pair accounting differs")
    scores = []
    for pair, record in zip(pairs, inference):
        index = pair["alternative_source_index"]
        identity = {"role": role, "row_id": row["id"], "current_source_index": 0,
                    "alternative_source_index": index}
        need(pair["pair_id"] == budget.digest_value(identity)
             and pair["candidate_objects_sha256"] == budget.digest_value(candidates)
             and pair["raw_response_sha256"] == budget.digest_value(raw)
             and pair["source"] == row["input"]
             and record.get("pair_id") == pair["pair_id"]
             and record.get("role") == role and record.get("row_id") == row["id"]
             and record.get("alternative_source_index") == index
             and finite(record.get("seconds"), nonnegative=True), "pair inference binding differs")
        logits = record.get("logits")
        need(isinstance(logits, list) and len(logits) == 2 and all(finite(x) for x in logits),
             "pair logits malformed or nonfinite")
        keep, switch = map(float, logits)
        need(math.isfinite(switch - keep), "pair margin nonfinite")
        scores.append({"pair_id": pair["pair_id"], "alternative_source_index": index,
                       "keep_logit": keep, "switch_logit": switch, "margin": switch - keep})
    positive = sorted((x for x in scores if x["margin"] > 0),
                      key=lambda x: (-x["margin"], x["alternative_source_index"]))
    chosen = positive[0]["alternative_source_index"] if positive else 0
    reason = "switch_margin" if positive else ("all_keep" if scores else "no_permitted_non_parent_alternative")
    ordered = [candidates[chosen]] + [x for i, x in enumerate(candidates) if i != chosen]
    expected = {"schema": EVALUATION_SCHEMA, "role": role, "id": row["id"],
                "source": row["input"], "context": row["context"], "raw_mode": False,
                "complete": True, "reason": reason, "failures": [], "chosen_index": chosen,
                "proposed_index": chosen if positive else None, "rejected_switch_indices": [],
                "expected_pair_count": len(pairs), "scored_pair_count": len(pairs),
                "pair_scores": scores, "result": ordered[0]["text"], "candidates": ordered,
                "raw_bypass_response": raw, "candidate_sha256_before": budget.digest_value(candidates),
                "raw_sha256_before": budget.digest_value(raw)}
    need(canonical(prediction) == canonical(expected), "prediction does not replay or candidate/raw bytes changed")
    return expected


def recompute_metrics(items, predictions, required_rows, minimum_gain):
    names = ("rows", "before_top1", "top1", "before_top5", "top5", "corrections", "regressions",
             "actions", "fallbacks", "candidate_present", "candidate_absent", "recoverable_errors")
    metrics, strata, outcomes = {}, {}, []
    for item, prediction in zip(items, predictions):
        row, role, stratum = item["row"], item["role"], item["stratum"]
        labels = row.get("expected")
        need(isinstance(labels, list) and labels and all(isinstance(x, str) and x for x in labels),
             "accepted answers malformed")
        need(row.get("result") == row["candidates"][0]["text"], "baseline differs from candidate zero")
        expected = set(labels)
        before, after = row["result"] in expected, prediction["result"] in expected
        before5 = any(x["text"] in expected for x in row["candidates"][:5])
        after5 = any(x["text"] in expected for x in prediction["candidates"][:5])
        present = any(x["text"] in expected for x in row["candidates"])
        recoverable = not before and any(row["candidates"][i]["text"] in expected for i in permitted_indices(row))
        chosen = prediction["chosen_index"]
        increments = dict(zip(names, (1, int(before), int(after), int(before5), int(after5),
            int(after and not before), int(before and not after), int(chosen != 0), int(chosen == 0),
            int(present), int(not present), int(recoverable))))
        for collection, key in ((metrics, "all"), (metrics, role), (strata, stratum)):
            counts = collection.setdefault(key, dict.fromkeys(names, 0))
            for name, value in increments.items():
                counts[name] += value
        outcomes.append({"role": role, "id": row["id"], "stratum": stratum,
            "before_top1": before, "top1": after, "before_top5": before5, "top5": after5,
            "corrected": after and not before, "regressed": before and not after,
            "candidate_present": present, "recoverable_error": recoverable,
            "chosen_index": chosen, "reason": prediction["reason"]})
    need(len(outcomes) == required_rows, "metric row count differs")
    total = metrics["all"]
    net = total["top1"] - total["before_top1"]
    losses = [x["id"] for x in outcomes if x["before_top5"] and not x["top5"]]
    gate = {"passed": net >= minimum_gain and not losses, "expected_rows": required_rows,
            "minimum_net_top1_gain": minimum_gain, "all_rows_accounted": True,
            "net_top1_gain": net, "corrections": total["corrections"],
            "regressions": total["regressions"], "candidate_absent": total["candidate_absent"],
            "recoverable_errors": total["recoverable_errors"], "zero_top5_losses": not losses,
            "top5_loss_ids": losses, "candidate_objects_preserved": True,
            "raw_responses_preserved": True, "protected_policy_replayed": True}
    return {"schema": EVALUATION_SCHEMA, "metrics": metrics, "strata": strata,
            "outcomes": outcomes, "gate": gate}


def verify_evaluations(report, data):
    qualities = {}
    for scope, row_count, pair_count, gain in (("mechanism", 175, 481, -175), ("train640", 640, 1556, 5)):
        items = data["items"][scope]
        predictions = report.get("predictions", {}).get(scope)
        inference = report.get("pair_inference", {}).get(scope)
        need(isinstance(predictions, list) and len(predictions) == row_count
             and isinstance(inference, list) and len(inference) == pair_count,
             "prediction/inference accounting incomplete")
        cursor, replayed = 0, []
        for item, prediction in zip(items, predictions):
            pairs = sorted((p for p in data["pairs"] if p["role"] == item["role"]
                and p["row_id"] == item["row"]["id"] and scope in p["scopes"]),
                key=lambda p: p["alternative_source_index"])
            records = inference[cursor:cursor + len(pairs)]
            need(all(r.get("scope") == scope for r in records), "inference scope differs")
            replayed.append(replay_prediction(item, prediction, pairs, records))
            cursor += len(pairs)
        need(cursor == pair_count, "inference cursor differs")
        qualities[scope] = recompute_metrics(items, replayed, row_count, gain)
    need(report.get("quality") == qualities, "reported quality differs from independent replay")
    total = qualities["train640"]["metrics"]["all"]
    need(total["before_top1"] == 447 and total["candidate_present"] == 474
         and total["candidate_absent"] == 166 and total["recoverable_errors"] == 20,
         "frozen TRAIN diagnostic baseline differs")
    return qualities


def expected_resource_stages(data):
    stages = ["before_model_load", "model_loaded", "fresh_adapter_initialized"]
    units = {x["unit_id"]: x for x in data["units"]}
    for scheduled in data["schedule"]:
        unit = units[scheduled["unit_id"]]
        stages.append(f"unit_{unit['order']}_before")
        stages.extend(f"component_{unit['unit_id']}_{i + 1}" for i in range(len(unit["references"])))
        stages.append(f"unit_{unit['order']}_after")
        if scheduled["position_in_update"] == 4 or scheduled["unit_order"] == 59:
            stages.append(f"update_{scheduled['update']}_after_optimizer")
    for scope in ("mechanism", "train640"):
        for item in data["items"][scope]:
            pairs = sorted((p for p in data["pairs"] if p["role"] == item["role"]
                and p["row_id"] == item["row"]["id"] and scope in p["scopes"]),
                key=lambda p: p["alternative_source_index"])
            stages.extend(f"{scope}_pair_after_{p['pair_id']}" for p in pairs)
    tokens = sorted(data["tokens"], key=lambda x: (x["prompt_tokens"], x["pair_id"]))
    representatives = [tokens[i]["pair_id"] for i in
                       (0, (len(tokens) - 1) // 2, math.ceil(.95 * len(tokens)) - 1, len(tokens) - 1)]
    stages.extend(f"save_probe_{p}" for p in representatives)
    stages.append("first_model_released")
    stages.extend(f"reload_probe_{p}" for p in representatives)
    stages.extend(("final_model_released", "terminal_cleanup"))
    return stages, representatives


def verify_resources(report, expected_stages):
    records = report.get("resource_observations")
    baseline = report.get("baseline")
    need(isinstance(records, list) and [r.get("stage") for r in records] == expected_stages,
         "resource observation stages incomplete or reordered")
    need(isinstance(baseline, dict) and set(baseline) == {"free_percent", "swap_used_bytes"}
         and all(finite(x, nonnegative=True) for x in baseline.values()), "resource baseline differs")
    fields = {"rss_peak_bytes", "mlx_active_bytes", "mlx_cache_bytes", "mlx_peak_bytes",
              "free_percent", "swap_used_bytes", "swap_growth_bytes", "elapsed_seconds"}
    elapsed = 0.0
    settled = []
    for record in records:
        need(record.get("errors") == [] and record.get("action") in ("none", "clear_unused_mlx_cache")
             and finite(record.get("cleanup_seconds"), nonnegative=True)
             and finite(record.get("checkpoint_seconds"), nonnegative=True), "resource wrapper invalid")
        pre, post = record.get("pre"), record.get("post")
        need(isinstance(pre, dict) and (post is None if record["action"] == "none" else isinstance(post, dict)),
             "resource cleanup evidence invalid")
        for observed in [pre] + ([post] if post is not None else []):
            need(set(observed) == fields and all(finite(v, nonnegative=k != "swap_growth_bytes")
                 for k, v in observed.items()), "resource observation invalid")
            need(observed["elapsed_seconds"] >= elapsed and observed["elapsed_seconds"] <= 21600
                 and observed["rss_peak_bytes"] <= LIMITS["rss_bytes"]
                 and observed["mlx_peak_bytes"] <= LIMITS["mlx_peak_bytes"]
                 and 10 <= observed["free_percent"] <= 100
                 and observed["swap_growth_bytes"] <= LIMITS["maximum_swap_growth_bytes"]
                 and observed["swap_growth_bytes"] == observed["swap_used_bytes"] - baseline["swap_used_bytes"],
                 "resource cap or interval breached")
            elapsed = observed["elapsed_seconds"]
        final = post if post is not None else pre
        need(final["mlx_cache_bytes"] <= LIMITS["cache_bytes"], "settled cache exceeds limit")
        settled.append(final)
    need(elapsed <= report["worker_fit_elapsed_seconds"], "terminal resource exceeds worker interval")
    return {"observations": len(records),
            "rss_peak_bytes": max(x["rss_peak_bytes"] for x in settled),
            "mlx_peak_bytes": max(x["mlx_peak_bytes"] for x in settled),
            "maximum_swap_growth_bytes": max(x["swap_growth_bytes"] for x in settled)}


def verify_adapter(report, output, representatives):
    adapter = report.get("adapter")
    need(isinstance(adapter, dict), "adapter missing")
    ap = output / "adapter/adapters.safetensors"
    cp = output / "adapter/adapter_config.json"
    need(adapter == {"path": str(ap.relative_to(ROOT)), "sha256": sha(ap),
                     "config_path": str(cp.relative_to(ROOT)), "config_sha256": sha(cp)},
         "adapter artifact binding differs")
    need(read_json(cp) == {"fine_tune_type": "lora", "num_layers": 4,
                           "lora_parameters": budget.RECIPE["lora_config"]}, "adapter config differs")
    parsed = budget.read_safetensors(ap)
    raw = ap.read_bytes()
    header_size = struct.unpack("<Q", raw[:8])[0]
    need(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', raw[8 + header_size:])),
         "adapter contains nonfinite tensor values")
    maps = [report.get(k) for k in ("initial_trainable_tensor_hashes", "final_trainable_tensor_hashes",
                                   "saved_trainable_tensor_hashes", "reloaded_trainable_tensor_hashes")]
    need(all(isinstance(m, dict) and sorted(m) == budget.EXPECTED_NAMES
             and all(isinstance(h, str) and len(h) == 64 and set(h) <= set('0123456789abcdef')
                     for h in m.values()) for m in maps)
         and maps[0] != maps[1] and maps[1] == maps[2] == maps[3] == parsed["hashes"],
         "initial/final/saved/reloaded tensor identity differs")
    reloads = report.get("reload")
    need(isinstance(reloads, list) and [x.get("pair_id") for x in reloads] == representatives,
         "reload representative accounting differs")
    for row in reloads:
        left, right = row.get("actual_logits"), row.get("expected_logits")
        need(isinstance(left, list) and isinstance(right, list) and len(left) == len(right) == 2
             and all(finite(x) for x in left + right)
             and all(abs(a - b) <= 1e-5 + 1e-5 * abs(b) for a, b in zip(left, right))
             and max(range(2), key=left.__getitem__) == max(range(2), key=right.__getitem__)
             and row.get("close") is True and row.get("same_argmax") is True, "reload logits differ")
    return {"sha256": parsed["file_sha256"], "tensors": 16, "scalars": 327680}


def verify_progress(report, data, progress):
    expected = []
    for update in report["updates"]:
        for unit in report["units"]:
            if unit["update"] == update["update"]:
                expected.append({"event": "unit_complete", **unit})
        expected.append({"event": "update_complete", **update})
    for scope in ("mechanism", "train640"):
        for index, item in enumerate(data["items"][scope], 1):
            pair_count = sum(p["role"] == item["role"] and p["row_id"] == item["row"]["id"]
                             and scope in p["scopes"] for p in data["pairs"])
            expected.append({"event": "evaluation_row", "scope": scope, "ordinal": index,
                             "role": item["role"], "row_id": item["row"]["id"], "pair_count": pair_count})
    expected.append({"event": "terminal", "terminal_status": report["terminal_status"],
                     "exit_code": 0 if report["quality_passed"] else 2,
                     "elapsed_seconds": report["worker_fit_elapsed_seconds"]})
    need(progress == expected, "durable progress does not match complete execution")


def verify_training(output=OUTPUT):
    output = Path(output).resolve()
    need(output == OUTPUT.resolve(), "only the fixed one-fit output is accepted")
    # Establish complete terminal evidence before opening the FIT label-bearing sources.
    required = ("report.json", "external-execution.json", "launch.json", "worker-start.json", "progress.jsonl")
    need(all((output / name).is_file() for name in required), "terminal execution evidence missing")
    artifact_hashes = {name: sha(output / name) for name in required}
    report, external, launch, worker = [read_json(output / name) for name in required[:4]]
    need(report.get("schema") == "difference-pairwise-lora-training-v2"
         and report.get("completed") is True and report.get("execution_valid") is True
         and report.get("terminal_status") in ("quality_passed", "quality_rejected"),
         "training execution is incomplete or invalid")
    prelaunch = validate_prelaunch()
    pre_sha, launch_sha = sha(PRELAUNCH), sha(output / "launch.json")
    for record in (report, external, launch):
        need(record.get("logical_fit_id") == LOGICAL_FIT_ID
             and record.get("sole_output") == SOLE_OUTPUT, "fit identity differs")
    need(launch.get("schema") == "difference-pairwise-lora-launch-v2"
         and launch.get("consumed") is True and launch.get("released") is False
         and launch.get("prelaunch") == {"path": str(PRELAUNCH.relative_to(ROOT)), "sha256": pre_sha}
         and launch.get("budget_proof_sha256") == sha(BUDGET_PROOF)
         and launch.get("recovery_proof_sha256") == sha(RECOVERY_PROOF)
         and isinstance(launch.get("worker_claim"), str) and launch["worker_claim"], "launch claim differs")
    need(report.get("worker_claim") == external.get("worker_claim") == launch["worker_claim"],
         "worker identity differs")
    # Worker-start schema is shared with the worker's exclusive process claim.
    need(worker.get("schema") == "difference-pairwise-lora-worker-start-v2"
         and worker.get("worker_claim") == launch["worker_claim"]
         and worker.get("launch_sha256") == launch_sha
         and worker.get("logical_fit_id") == LOGICAL_FIT_ID
         and worker.get("sole_output") == SOLE_OUTPUT
         and worker.get("consumed") is True and worker.get("released") is False
         and type(worker.get("worker_pid")) is int and worker["worker_pid"] > 0
         and worker.get("worker_pid") == external.get("worker_pid"), "exclusive worker claim differs")
    need(report.get("worker_start") == worker, "reported worker claim differs from exclusive receipt")
    need(external.get("schema") == "difference-pairwise-lora-external-execution-v2"
         and external.get("prelaunch_sha256") == report.get("prelaunch_sha256") == pre_sha
         and external.get("launch_sha256") == report.get("launch_sha256") == launch_sha
         and external.get("budget_proof_sha256") == sha(BUDGET_PROOF)
         and external.get("recovery_proof_sha256") == sha(RECOVERY_PROOF)
         and external.get("terminal_report_sha256") == sha(output / "report.json")
         and external.get("worker_interval_valid") is True
         and external.get("deadline_terminated") is False and external.get("spawn_error") is None
         and external.get("additional_model_work_authorized") is False
         and external.get("fit_limit_seconds") == 21600, "external execution receipt invalid")
    start, end, elapsed = [external.get(k) for k in ("fit_start_monotonic", "fit_end_monotonic", "fit_elapsed_seconds")]
    need(all(finite(x, nonnegative=True) for x in (start, end, elapsed))
         and 0 <= end - start <= 21600 and math.isclose(end - start, elapsed, rel_tol=0, abs_tol=1e-8)
         and report.get("fit_start_monotonic") == start
         and finite(report.get("worker_fit_elapsed_seconds"), nonnegative=True)
         and report["worker_fit_elapsed_seconds"] <= elapsed, "authoritative worker interval invalid")
    need(report.get("real_training_run") is True and report.get("fresh_base_initialized") is True
         and report.get("old_adapter_loaded") is False and report.get("disposable_adapter_loaded") is False
         and report.get("one_epoch") is True
         and report.get("eligibility") == {"evaluation": False, "resume": False, "deploy": False}
         and report.get("limits") == LIMITS and report.get("recipe") == RECIPE
         and report.get("counts_expected") == COUNTS and report.get("package_versions") == budget.VERSIONS
         and report.get("data") == {"manifest_sha256": budget.MANIFEST_SHA256, "proof_sha256": budget.DATA_PROOF_SHA256}
         and report.get("budget_proof") == prelaunch["budget_proof"]
         and report.get("recovery_proof") == prelaunch["recovery_proof"]
         and report.get("source_hashes_before") == report.get("source_hashes_after") == prelaunch["inventory"]
         and report.get("independent_training_verification_required") is True
         and report.get("external_execution_receipt_required") is True, "training recipe or provenance differs")
    need(not any(k in report for k in ("failure", "cleanup_error", "terminal_cleanup_error", "final_inventory_error")),
         "execution contains a failure record")
    data = load_data()
    verify_units(report, data)
    stages, representatives = expected_resource_stages(data)
    resources = verify_resources(report, stages)
    adapter = verify_adapter(report, output, representatives)
    quality = verify_evaluations(report, data)
    verify_progress(report, data, read_jsonl(output / "progress.jsonl"))
    passed = quality["train640"]["gate"]["passed"]
    need(report.get("quality_passed") is passed
         and report.get("terminal_status") == ("quality_passed" if passed else "quality_rejected")
         and external.get("worker_exit_status") == (0 if passed else 2)
         and external.get("authoritative_success") is passed
         and external.get("quality_rejected") is (not passed), "quality/execution verdict mismatch")
    for name, value in prelaunch["inventory"].items():
        bound_file(name, value)
    need(artifact_hashes == {name: sha(output / name) for name in required},
         "execution artifacts changed during verification")
    return {"stage": "training", "verification_passed": True, "execution_valid": True,
            "quality_gate_passed": passed, "downstream_evaluation_allowed": passed,
            "promotion_allowed": False, "resume_allowed": False, "deployed": False,
            "logical_fit_id": LOGICAL_FIT_ID, "sole_output": SOLE_OUTPUT,
            "prelaunch_sha256": pre_sha,
            "artifacts": artifact_hashes,
            "adapter": adapter, "resources": resources, "quality": quality,
            "fit_elapsed_seconds": elapsed, "source_hashes": prelaunch["inventory"],
            "verifier_sha256": sha(Path(__file__))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='immutable proof output')
    args = parser.parse_args(argv)
    try:
        payload = verify_training()
        envelope = {"schema": SCHEMA, "payload": payload, "digest": budget.digest_value(payload)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('xb') as stream:
            stream.write(canonical(envelope) + b'\n')
        print(json.dumps({"verification_passed": True, "quality_gate_passed": payload['quality_gate_passed']}))
        return 0 if payload['quality_gate_passed'] else 2
    except (budget.VerificationError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"verification_passed": False, "error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
