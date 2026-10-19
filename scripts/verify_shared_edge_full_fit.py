#!/usr/bin/env python3
"""Independent verifier for the frozen shared-edge full-FIT diagnostic."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".omx/artifacts/shared-edge-full-fit-audit-v1"
TRAIN = ROOT / ".omx/artifacts/shared-edge-residual-training-v1"
CENSUS = ROOT / ".omx/artifacts/recombination-fresh-census-v1"
DATA = ROOT / ".omx/artifacts/source-correspondence-fit-v2"
INVENTORY = ROOT / ".omx/artifacts/shared-edge-fit-inventory-v1"
PRODUCER = ROOT / "scripts/audit_shared_edge_full_fit.py"
PRODUCER_SHA256 = "365700e180e0bcdbbda3d78003822bd696321dda99c22b0f0d4ae040fa6c059e"
HELPER_PATH = ROOT / ".omx/artifacts/verify-source-span-pair-v1.py"
HELPER_SHA256 = "4fbf04d3e340d0565367741f89597c5787cd1383bf283f5ad56919918eeba16a"
SCHEMA = "shared-edge-full-fit-independent-verification-v1"
RUN_SCHEMA = "shared-edge-full-fit-audit-v1"
ROWS = 1483
CANDIDATES = 202603
LIMITS = {"whole_seconds": 1800.0, "update_seconds": 60.0,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
FROZEN = {
    TRAIN / "report.json": "481ff33a1ae6e9f0e8f3d5c24566f5e076a42d29dd02145e1ad964fd6a06e5da",
    TRAIN / "independent-verification.json": "e4d48a8be5373198e83a11bc88793f6262520f6a2b3b075da2d3a1d8e28cf0e1",
    CENSUS / "capture.jsonl": "847a8d6603dbd66ed87d2f9863b4e20581989887da76679b1a5586bb6d552154",
    CENSUS / "capture.complete.json": "934b832bccf7b89c6f5a9700d2b3468dcf886caeb92be5a5a9ee0f88f5a06dcb",
    CENSUS / "independent-verification.json": "a744c8324fca8c2a1ae170fb9e51aa0ce06f9e0a6310446aecbe93b11e5deb29",
}
OUTPUTS = {"pins.json", "selection.json", "targets.json", "tokens.json",
           "target-features.safetensors", "target-features.json", "resources.jsonl",
           "predictions.jsonl"}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try: value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSONL at {path}:{number}") from error
            need(isinstance(value, dict), f"non-object JSONL row at {path}:{number}")
            result.append(value)
    return result


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def helper():
    need(sha(HELPER_PATH) == HELPER_SHA256, "frozen pure FIT helper changed")
    spec = importlib.util.spec_from_file_location("pure_fit_helper", HELPER_PATH)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def reconstruct():
    training_selection = read(TRAIN / "selection.json")
    need(isinstance(training_selection, list) and len(training_selection) == ROWS,
         "training selection differs")
    selected = {row["id"]: row for row in training_selection}
    need(len(selected) == ROWS, "training selection IDs duplicate")
    records = {row["id"]: row for row in jsonl(DATA / "fit-paths.jsonl")
               if row.get("id") in selected}
    captures = {row["id"]: row for row in jsonl(CENSUS / "capture.jsonl")
                if row.get("split") == "fit" and row.get("id") in selected}
    old_rows = {row["id"]: row for row in jsonl(INVENTORY / "rows.jsonl")}
    need(set(records) == set(captures) == set(old_rows) == set(selected),
         "full FIT source denominator differs")
    projected, inventories, targets = [], {}, set()
    for chosen in training_selection:
        identifier = chosen["id"]; record = records[identifier]; capture = captures[identifier]
        need(record["input"] == chosen["input"] == capture["input"]
             and record["component_id"] == chosen["component_id"] == capture["component_id"]
             and "expected" not in capture, "full FIT source identity differs")
        score_texts, parent_scores = capture.get("score_texts"), capture.get("scores")
        need(isinstance(score_texts, list) and isinstance(parent_scores, list)
             and len(score_texts) == len(parent_scores) == len(set(score_texts))
             and all(finite(value) for value in parent_scores), "census parent score domain differs")
        by_text = defaultdict(list)
        for path in record["paths"]:
            if path["text"] != chosen["input"]:
                by_text[path["text"]].append(path)
        candidates = sorted(by_text); need(set(candidates) == set(score_texts),
                                            "census/full structural domain differs")
        score_map = dict(zip(score_texts, parent_scores)); index = {text: i for i, text in enumerate(candidates)}
        positive_texts = {chosen["candidates"][i] for i in chosen["positive_indices"]}
        need(positive_texts and positive_texts.issubset(chosen["candidates"])
             and set(chosen["candidates"]).issubset(by_text), "selected/positive domain differs")
        edges = record["edges"]
        canonical = sorted({(edges[i]["input_start"], edges[i]["input_end"], edges[i]["text"])
            for paths in by_text.values() for path in paths for i in path["edge_indices"]})
        edge_index = {edge: i for i, edge in enumerate(canonical)}
        groups = []
        for text in candidates:
            paths = []
            for path in sorted(by_text[text], key=lambda item: item["path_id"]):
                keys = [(edges[i]["input_start"], edges[i]["input_end"], edges[i]["text"])
                        for i in path["edge_indices"]]
                paths.append({"path_id": path["path_id"],
                    "edge_indices": [edge_index[key] for key in keys],
                    "weight_numerators": [key[1] - key[0] for key in keys],
                    "weight_denominator": len(record["input"])})
                need(sum(paths[-1]["weight_numerators"]) == len(record["input"]),
                     "full path source weight mass differs")
                targets.update(key[2] for key in keys)
            groups.append(paths)
        old = old_rows[identifier]
        for old_index, full_index in enumerate(index[text] for text in chosen["candidates"]):
            need([path["path_id"] for path in groups[full_index]]
                 == [path["path_id"] for path in old["paths"][old_index]],
                 "selected candidate path domain expanded")
        row = {"id": identifier, "input": chosen["input"],
            "component_id": chosen["component_id"], "candidates": candidates,
            "positive_indices": [index[text] for text in sorted(positive_texts)],
            "selected_indices": [index[text] for text in chosen["candidates"]],
            "base_scores": [score_map[text] for text in candidates],
            "candidate_path_counts": [len(group) for group in groups],
            "selected_path_expansion_count": 0}
        projected.append(row); inventories[identifier] = {"edges": canonical, "groups": groups}
    need(sum(len(row["candidates"]) for row in projected) == CANDIDATES,
         "full FIT candidate count differs")
    return projected, inventories, sorted(targets)


def ulp_sum_tolerance(values):
    magnitude = max(1., sum(abs(value) for value in values))
    ulp = 2. ** (math.frexp(magnitude)[1] - 24)
    levels = math.ceil(math.log2(len(values))) if len(values) > 1 else 0
    return 4 * (levels + 1) * ulp


def row_counts(row, scores, known_targets, inventory):
    positive, selected = set(row["positive_indices"]), set(row["selected_indices"])
    order = lambda indices: sorted(indices, key=lambda i: (-scores[i], row["candidates"][i]))
    full, restricted = order(range(len(scores))), order(selected)
    full_top1, restricted_top1 = int(full[0] in positive), int(restricted[0] in positive)
    full_top5 = int(bool(set(full[:4]) & positive)); restricted_top5 = int(bool(set(restricted[:4]) & positive))
    need(full_top1 <= restricted_top1 and full_top5 <= restricted_top5,
         "full expansion gained despite retained positives")
    winning_targets = {inventory["edges"][edge][2]
                       for path in inventory["groups"][full[0]] for edge in path["edge_indices"]}
    best_positive = max(scores[i] for i in positive)
    excluded = [i for i in full if i not in selected]
    return {"restricted_top1": restricted_top1, "full_top1": full_top1,
        "restricted_top5": restricted_top5, "full_top5": full_top5,
        "top1_expansion_loss": restricted_top1 - full_top1,
        "top5_expansion_loss": restricted_top5 - full_top5,
        "winner_outside_selected": int(full[0] not in selected),
        "winner_has_new_target_type": int(bool(winning_targets - known_targets)),
        "excluded_above_best_positive": sum(scores[i] > best_positive for i in excluded)}


def audit_prediction(saved, row, inventory, historical, known_targets):
    need(set(saved) == {"id", "scores", "residuals", "edge_ids", "edge_scores",
                        "winning_path_ids", "counts", "historical_selected_score_max_abs_drift",
                        "historical_selected_base_max_abs_drift",
                        "historical_selected_residual_max_abs_drift"}
         and saved["id"] == row["id"], "prediction fields/identity differ")
    edges = [list(edge) for edge in inventory["edges"]]; edge_scores = saved["edge_scores"]
    need(saved["edge_ids"] == edges and len(edge_scores) == len(edges)
         and all(finite(value) for value in edge_scores), "full edge domain/scores differ")
    residuals, winners = [], []
    for paths in inventory["groups"]:
        path_values = []
        for path in paths:
            weights = [f32(value / path["weight_denominator"]) for value in path["weight_numerators"]]
            contributions = [f32(f32(edge_scores[index]) * weight)
                             for index, weight in zip(path["edge_indices"], weights)]
            value = f32(sum(contributions))
            path_values.append((path["path_id"], value, contributions))
        maximum = max(value for _, value, _ in path_values)
        recorded = saved["residuals"][len(residuals)]
        winner_parts = next(parts for _, value, parts in path_values if value == maximum)
        need(finite(recorded) and abs(recorded - sum(winner_parts)) <= ulp_sum_tolerance(winner_parts),
             "full path residual differs")
        residuals.append(recorded)
        winners.append([path_id for path_id, value, _ in path_values if value == maximum])
    need(saved["winning_path_ids"] == winners
         and saved["scores"] == [f32(base + residual)
             for base, residual in zip(row["base_scores"], residuals)],
         "full path winners or FP32 scores differ")
    counts = row_counts(row, saved["scores"], known_targets, inventory)
    chosen = row["selected_indices"]
    counts["historical_selected_correct"] = historical["correct"]
    counts["historical_correctness_disagreement"] = int(
        historical["correct"] != counts["restricted_top1"])
    score_drift = max(abs(saved["scores"][index] - historical["scores"][j])
                      for j, index in enumerate(chosen))
    base_drift = max(abs(row["base_scores"][index] - historical["base_scores"][j])
                     for j, index in enumerate(chosen))
    residual_drift = max(abs(residuals[index] - historical["residuals"][j])
                         for j, index in enumerate(chosen))
    need(saved["counts"] == counts
         and saved["historical_selected_score_max_abs_drift"] == score_drift
         and saved["historical_selected_base_max_abs_drift"] == base_drift
         and saved["historical_selected_residual_max_abs_drift"] == residual_drift,
         "FIT counts or historical drift differs")
    return counts, score_drift, base_drift, residual_drift


def verify_cache(run, targets, selection, report, pure):
    tokens = read(run / "tokens.json")
    need(set(tokens) == {"reserved_tokens", "target_counts", "candidate_counts",
                         "max_target_tokens", "max_candidate_tokens"}, "token census fields differ")
    reserved = tokens["reserved_tokens"]; target_counts = tokens["target_counts"]
    candidate_counts = tokens["candidate_counts"]
    need(type(reserved) is int and isinstance(target_counts, list) and len(target_counts) == len(targets)
         and isinstance(candidate_counts, list) and len(candidate_counts) == ROWS
         and all(len(values) == len(row["candidates"]) for values, row in zip(candidate_counts, selection))
         and all(type(value) is int and reserved <= value <= 256
                 for value in target_counts for _ in (0,))
         and all(type(value) is int and reserved <= value <= 256
                 for values in candidate_counts for value in values)
         and tokens["max_target_tokens"] == max(target_counts)
         and tokens["max_candidate_tokens"] == max(max(values) for values in candidate_counts),
         "token census lengths differ")
    metadata = read(run / "target-features.json"); old_targets = read(TRAIN / "targets.json")
    try:
        from safetensors import safe_open
        import numpy as np
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("safetensors and NumPy are required") from error
    def tensors(path):
        with safe_open(path, framework="numpy") as handle:
            need(set(handle.keys()) == {"vectors", "has_tokens"}, "cache tensor keys differ")
            return handle.get_tensor("vectors"), handle.get_tensor("has_tokens")
    vectors, flags = tensors(run / "target-features.safetensors")
    old_vectors, old_flags = tensors(TRAIN / "target-features.safetensors")
    need(vectors.shape == (len(targets), 256) and str(vectors.dtype) == "float32"
         and flags.shape == (len(targets),) and str(flags.dtype) == "bool"
         and np.isfinite(vectors).all() and np.count_nonzero(vectors[~flags]) == 0,
         "merged target cache shape/dtype/value differs")
    expected_flags = np.asarray([value > reserved for value in target_counts], dtype=np.bool_)
    need(np.array_equal(flags, expected_flags), "merged target cache token flags differ")
    old_index = {text: i for i, text in enumerate(old_targets)}; target_index = {text: i for i, text in enumerate(targets)}
    reused = [text for text in targets if text in old_index]
    need(np.array_equal(vectors[[target_index[t] for t in reused]],
                        old_vectors[[old_index[t] for t in reused]])
         and np.array_equal(flags[[target_index[t] for t in reused]],
                            old_flags[[old_index[t] for t in reused]]),
         "known target cache tensors were not reused exactly")
    hashes = {"vectors": pure.tensor_digest(vectors), "has_tokens": pure.tensor_digest(flags)}
    old_hashes = {"vectors": pure.tensor_digest(old_vectors), "has_tokens": pure.tensor_digest(old_flags)}
    expected_metadata = {"schema": RUN_SCHEMA,
        "old_target_file_sha256": sha(TRAIN / "targets.json"),
        "old_tensor_file_sha256": sha(TRAIN / "target-features.safetensors"),
        "old_tensor_hashes": old_hashes, "tensor_hashes": hashes,
        "tensor_file_sha256": sha(run / "target-features.safetensors"),
        "tokens_sha256": sha(run / "tokens.json"), "reused_targets": len(reused),
        "new_targets": len(targets) - len(reused), "known_reuse_exact": True,
        "chunk_size": 256, "tensor_bytes": vectors.nbytes + flags.nbytes}
    need(metadata == expected_metadata and report["target_cache_tensor_hashes"] == hashes
         and report["old_target_cache_unchanged"] is True,
         "target cache metadata/report differs")
    return {"targets": len(targets), "reused": len(reused),
            "new": len(targets) - len(reused), "tensor_hashes": hashes}


def expected_operations(selection, new_targets):
    names = ["load_frozen_model_and_cache", "token_census"]
    names.extend(f"new_targets:{start}" for start in range(0, new_targets, 256))
    names.append("close_target_cache")
    names.extend("full_fit:" + row["id"] for row in selection)
    return names


def verify_resources(report, path, operations):
    trace = jsonl(path); baseline = report.get("baseline_swap_bytes")
    need(trace and type(baseline) is int, "resource trace/baseline differs")
    next_index = 0; current = None; previous = -1.; final_seen = False; final_row = None
    peaks = {"rss": 0, "swap": 0, "operation": 0.}
    for row in trace:
        need(set(row) == {"at", "elapsed_s", "rss_bytes", "swap_used_bytes", "swap_growth_bytes",
                          "operation_elapsed_s", "stage", "operation"}
             and row["stage"] in {"operation_start", "operation_end", "watchdog", "final"}
             and finite(row["elapsed_s"]) and row["elapsed_s"] >= previous
             and type(row["rss_bytes"]) is int and 0 <= row["rss_bytes"] <= LIMITS["rss_bytes"]
             and type(row["swap_used_bytes"]) is int and type(row["swap_growth_bytes"]) is int
             and row["swap_used_bytes"] - baseline == row["swap_growth_bytes"]
             and row["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"],
             "resource observation/bound differs")
        previous = row["elapsed_s"]
        if row["operation"] is None:
            need(row["stage"] in {"watchdog", "final"} and row["operation_elapsed_s"] is None,
                 "idle resource attribution differs")
        else:
            need(finite(row["operation_elapsed_s"])
                 and 0 <= row["operation_elapsed_s"] <= LIMITS["update_seconds"],
                 "operation resource bound differs")
        if row["stage"] == "final":
            need(not final_seen and current is None and next_index == len(operations),
                 "final resource order differs")
            final_seen = True; final_row = row
        elif row["stage"] == "operation_start":
            need(not final_seen and current is None and next_index < len(operations)
                 and row["operation"] == operations[next_index], "operation start order differs")
            current = row["operation"]; next_index += 1
        elif row["stage"] == "operation_end":
            need(not final_seen and current == row["operation"], "operation pairing differs")
            current = None
        else:
            need(row["operation"] == current, "atomic watchdog attribution differs")
        peaks["rss"] = max(peaks["rss"], row["rss_bytes"])
        peaks["swap"] = max(peaks["swap"], row["swap_growth_bytes"])
        if row["operation_elapsed_s"] is not None:
            peaks["operation"] = max(peaks["operation"], row["operation_elapsed_s"])
    need(final_seen and current is None and next_index == len(operations)
         and report.get("final_resources") == final_row and finite(report.get("elapsed_s"))
         and previous <= report["elapsed_s"] <= LIMITS["whole_seconds"],
         "resource final/inventory differs")
    return {"observations": len(trace), "operation_pairs": len(operations),
            "peak_rss_bytes": peaks["rss"], "peak_swap_growth_bytes": peaks["swap"],
            "peak_operation_seconds": peaks["operation"],
            "atomic_operation_attribution_verified": True}


def verify(run=RUN):
    run = Path(run).resolve(); need(run == RUN.resolve(), "alternate full-FIT run is not authorized")
    need(sha(PRODUCER) == PRODUCER_SHA256, "frozen full-FIT producer changed")
    need(all(sha(path) == digest for path, digest in FROZEN.items()), "frozen diagnostic input changed")
    report = read(run / "report.json")
    need(report.get("schema") == RUN_SCHEMA and report.get("completed") is True
         and report.get("rows") == ROWS and report.get("candidates") == CANDIDATES
         and report.get("limits") == LIMITS, "completed full-FIT report differs")
    pure = helper(); selection, inventories, targets = reconstruct()
    need(read(run / "selection.json") == selection and read(run / "targets.json") == targets,
         "saved full-FIT selection/targets differ")
    pins = read(run / "pins.json"); need(pins == report.get("pins"), "full-FIT pins differ")
    auth = []
    for raw, digest in pins.items():
        path = Path(raw)
        if path.suffix != ".json" or not path.is_file():
            continue
        value = read(path)
        if isinstance(value, dict) and value.get("schema") == RUN_SCHEMA + "-authorization":
            auth.append((path, digest))
    need(len(auth) == 1, "full-FIT authorization pin differs")
    auth_path, auth_sha = auth[0]; base_pins = {raw: digest for raw, digest in pins.items()
                                               if raw != str(auth_path.resolve())}
    need(read(auth_path) == {"schema": RUN_SCHEMA + "-authorization", "rows": ROWS,
        "candidates": CANDIDATES, "targets": len(targets), "limits": LIMITS, "pins": base_pins}
         and auth_sha == sha(auth_path)
         and all(Path(raw).is_file() and sha(raw) == digest for raw, digest in pins.items()),
         "consumed authorization or pin changed")
    training = read(TRAIN / "report.json"); proof = read(TRAIN / "independent-verification.json")
    census_proof = read(CENSUS / "independent-verification.json")
    closure = read(CENSUS / "capture.complete.json")
    need(training.get("completed") is True and proof.get("verified") is True
         and proof.get("report_sha256") == FROZEN[TRAIN / "report.json"]
         and census_proof.get("verified") is True
         and census_proof.get("capture_sha256") == FROZEN[CENSUS / "capture.jsonl"]
         and census_proof.get("contracts", {}).get("source_scores_checked_before_labels") is True
         and closure == {"capture_sha256": FROZEN[CENSUS / "capture.jsonl"],
            "input_truncation": False, "rows": 2048,
            "schema": "recombination-fresh-training-parent-census-v1-source-closure",
            "score_context": "", "source_only": True, "targets_joined": False},
         "training/census authorization differs")
    census_parent = census_proof.get("model_pins", {})
    need(isinstance(census_parent, dict) and census_parent
         and all(training.get("pins", {}).get(raw) == digest
                 and Path(raw).is_file() and sha(raw) == digest
                 for raw, digest in census_parent.items()),
         "census/training parent model pins differ")
    cache = verify_cache(run, targets, selection, report, pure)
    predictions = jsonl(run / "predictions.jsonl"); historical = jsonl(TRAIN / "after.jsonl")
    historical_by_id = {row["id"]: row for row in historical}
    need(len(predictions) == len(historical_by_id) == ROWS, "prediction/historical denominator differs")
    known = set(read(TRAIN / "targets.json")); totals = Counter(); max_drift = [0., 0., 0.]
    for saved, row in zip(predictions, selection):
        need(saved["id"] == row["id"] and row["id"] in historical_by_id,
             "prediction order/identity differs")
        counts, score, base, residual = audit_prediction(
            saved, row, inventories[row["id"]], historical_by_id[row["id"]], known)
        totals.update(counts); max_drift = [max(old, value) for old, value in
                                           zip(max_drift, (score, base, residual))]
    need(report.get("counts") == dict(totals)
         and report.get("historical_selected_score_max_abs_drift") == max_drift[0]
         and report.get("historical_selected_base_max_abs_drift") == max_drift[1]
         and report.get("historical_selected_residual_max_abs_drift") == max_drift[2]
         and report.get("full_path_count") == sum(sum(row["candidate_path_counts"]) for row in selection)
         and report.get("selected_path_expansion_count") == 0
         and report.get("reused_targets") == cache["reused"]
         and report.get("new_targets") == cache["new"]
         and report.get("targets") == len(targets), "full-FIT aggregate report differs")
    need(report.get("model_before") == report.get("model_after") == training.get("final_hashes")
         and report.get("parent_before") == report.get("parent_after") == training.get("parent_after"),
         "frozen model/parent hashes differ")
    need(report.get("scope") == {"fit_labels_only": True, "optimizer_updates": 0,
        "dev_targets_opened": False, "test_targets_opened": False,
        "inner_hold_labels_opened": False, "production_changed": False,
        "full_and_restricted_use_same_forward": True,
        "source_only_mixed_census_filtered": True}, "full-FIT scope differs")
    resources = verify_resources(report, run / "resources.jsonl",
                                 expected_operations(selection, cache["new"]))
    outputs = {name: sha(run / name) for name in OUTPUTS}
    need(report.get("outputs") == outputs and not (run / "failure.json").exists()
         and {path.name for path in run.iterdir() if path.is_file()}
             <= OUTPUTS | {"report.json", "independent-verification.json"},
         "full-FIT output inventory differs")
    return {"schema": SCHEMA, "verified": True, "report_sha256": sha(run / "report.json"),
        "producer_sha256": sha(PRODUCER), "verifier_sha256": sha(Path(__file__)),
        "counts": {"rows": ROWS, "candidates": CANDIDATES, "targets": len(targets),
                   "paths": report["full_path_count"], "reused_targets": cache["reused"],
                   "new_targets": cache["new"]}, "metrics": dict(totals),
        "drift": {"selected_score_max_abs": max_drift[0],
                  "selected_base_max_abs": max_drift[1],
                  "selected_residual_max_abs": max_drift[2]},
        "resources": resources,
        "contracts": {"authorized_fit_only_compact_projection_reconstructed": True,
            "closed_source_only_census_and_parent_pins_authenticated": True,
            "full_nonraw_domain_and_all_selected_positives_retained": True,
            "selected_path_domain_unchanged": True,
            "known_target_cache_tensors_reused_exactly": True,
            "new_target_texts_only_encoded": True,
            "all_edges_paths_weights_residuals_winners_and_fp32_scores_replayed": True,
            "same_forward_restricted_and_full_metrics_replayed": True,
            "full_correct_never_exceeded_restricted_correct": True,
            "optimizer_updates": 0, "dev_opened": False, "test_opened": False,
            "inner_hold_labels_opened": False, "production_changed": False,
            "torch_imported": False, "model_instantiated": False,
            "neural_forward_rerun": False, "generalization_claimed": False},
        "limitations": ["This diagnostic uses authorized FIT labels and cannot establish generalization.",
            "Neural forwards are authenticated from saved evidence rather than independently rerun.",
            "Historical selected-score parity is descriptive because cached parent batching may drift."]}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path); args = parser.parse_args()
    result = verify(args.run); output = args.output or args.run / "independent-verification.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"verified": True, "counts": result["counts"],
                      "metrics": result["metrics"], "drift": result["drift"],
                      "resources": result["resources"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
