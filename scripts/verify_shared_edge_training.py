#!/usr/bin/env python3
"""Independent offline verifier for shared-edge FIT training.

No trainer, new model, Torch, tokenizer, or neural forward is imported.  The
authorized FIT edge inventory is reconstructed from frozen compact source paths,
and safetensors are read through NumPy only.
"""
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
RUN = ROOT / ".omx/artifacts/shared-edge-residual-training-v1"
OLD = ROOT / ".omx/artifacts/source-span-paired-training-v1"
INVENTORY = ROOT / ".omx/artifacts/shared-edge-fit-inventory-v1"
DATA = ROOT / ".omx/artifacts/source-correspondence-fit-v2"
SELECTION = ROOT / ".omx/artifacts/recombination-paired-training-v1/selection.json"
PARENT = ROOT / "checkpoints/sentence_ranker"
PLAN = ROOT / ".omx/plans/shared-edge-residual-v1.md"
AUTHORIZATION = ROOT / ".omx/artifacts/shared-edge-residual-prelaunch-v2.json"
HELPER_PATH = ROOT / ".omx/artifacts/verify-source-span-pair-v1.py"
HELPER_SHA256 = "4fbf04d3e340d0565367741f89597c5787cd1383bf283f5ad56919918eeba16a"
PRODUCER = ROOT / "scripts/train_shared_edge_residual.py"
PRODUCER_SHA256 = "6ad03f4096b57525547d4809d193329560924e117814e3baa742e99529704a0a"
MODEL_SHA256 = "12c647835c2e268f0c5dc92b96edaf1e528ec3ab1d2c33e848f2ab3f8f716ed1"
SCHEMA = "shared-edge-residual-training-independent-verification-v1"
RUN_SCHEMA = "shared-edge-residual-training-v1"
ROWS = 1483
TARGETS = 16814
EDGES = 47282
CANDIDATES = 11793
EPOCHS = 2
UPDATES = ROWS * EPOCHS
TARGET_CHUNK = 256
TARGET_BATCH = 8
CACHE_BYTES = 256 * 1024**2
RELOAD_INDICES = sorted(set([0, ROWS - 1] + list(range(0, ROWS, 100))))
GRADIENT_GROUPS = ("source_embedding.", "source_encoder.",
                   "segment_head.0.", "segment_head.2.")
RECIPE = {"seed": 20260921, "epochs": 2, "learning_rate": .001,
          "weight_decay": .01, "clip_norm": 1., "threads": 2,
          "dtype": "float32", "optimizer": "AdamW", "accumulation": 1}
LIMITS = {"whole_seconds": 1800., "update_seconds": 60.,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
FIXED = {
    DATA / "independent-verification.json":
        "449a8f286a60cc4a7c542612054c4ca762378caedfb0cbc9ca350801030f2de2",
    DATA / "fit-paths.jsonl":
        "f1cd66da23cacc6025e8aa2203944f664f0d74d916a0bde135cac65ef56afd8c",
    SELECTION:
        "cbac8994a33d2a8b4d0f0c4596a1a07e206d6b75562c3416ff49877864829fd8",
}
OUTPUTS = {"pins.json", "selection.json", "targets.json", "resources.jsonl",
           "initial.safetensors", "tokens.json", "target-features.safetensors",
           "target-features.json", "before.jsonl", "updates.jsonl", "after.jsonl",
           "residual.safetensors", "reload.json", "checkpoint.json"}


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
    return json.loads(Path(path).read_text())


def jsonl(path):
    values = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSONL at {path}:{number}") from error
    return values


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def fp32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def digest(value):
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def helper_module():
    need(sha(HELPER_PATH) == HELPER_SHA256, "frozen pure training helper changed")
    spec = importlib.util.spec_from_file_location("frozen_source_span_training_verifier", HELPER_PATH)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    return helper


def expand_path(record, compact):
    source, edges = record.get("input"), record.get("edges")
    need(isinstance(source, str) and source and isinstance(edges, list) and edges,
         "compact FIT source/edges malformed")
    need(isinstance(compact, dict) and set(compact) == {"path_id", "text", "edge_indices"}
         and isinstance(compact["path_id"], str) and isinstance(compact["text"], str)
         and isinstance(compact["edge_indices"], list) and compact["edge_indices"],
         "compact FIT path malformed")
    spans, cursor = [], 0
    for index in compact["edge_indices"]:
        need(type(index) is int and 0 <= index < len(edges), "compact edge index differs")
        edge = edges[index]
        need(isinstance(edge, dict) and set(edge) == {"input_start", "input_end", "text"},
             "compact edge fields differ")
        start, end, text = edge["input_start"], edge["input_end"], edge["text"]
        need(type(start) is int and type(end) is int and isinstance(text, str)
             and start == cursor and start < end <= len(source), "compact path coverage differs")
        spans.append({"input_start": start, "input_end": end, "text": text})
        cursor = end
    text = "".join(span["text"] for span in spans)
    path_id = "alignment-v1-" + digest(
        [source, [[span["input_start"], span["input_end"], span["text"]] for span in spans]])
    need(cursor == len(source) and text == compact["text"] and path_id == compact["path_id"],
         "compact path identity differs")
    return {"path_id": path_id, "text": text, "spans": spans}


def reconstruct_inventory(selection_rows, fit_records):
    """Build the exact inventory without importing its producer."""
    need(len(selection_rows) == len(fit_records), "FIT selection/path denominator differs")
    targets_set, prepared = set(), []
    for selected, record in zip(selection_rows, fit_records):
        need(record.get("id") == selected.get("id")
             and record.get("input") == selected.get("input")
             and record.get("component_id") == selected.get("component_id"),
             "FIT row identity/order differs")
        arm = selected.get("arms", {}).get("b_union", {})
        candidates, positives = arm.get("candidates"), arm.get("positive_indices")
        need(isinstance(candidates, list) and 2 <= len(candidates) <= 8
             and len(set(candidates)) == len(candidates)
             and isinstance(positives, list) and positives
             and len(set(positives)) == len(positives)
             and all(type(index) is int and 0 <= index < len(candidates) for index in positives)
             and len(positives) < len(candidates), "FIT candidate/target domain differs")
        groups = defaultdict(list)
        seen_paths = set()
        for compact in record.get("paths", []):
            path = expand_path(record, compact)
            need(path["path_id"] not in seen_paths, "duplicate FIT structural path")
            seen_paths.add(path["path_id"])
            if path["text"] in candidates:
                groups[path["text"]].append(path)
                targets_set.update(span["text"] for span in path["spans"])
        need(set(groups) == set(candidates), "selected FIT candidate path missing")
        prepared.append((selected, [[*sorted(groups[text], key=lambda item: item["path_id"])]
                                    for text in candidates]))
    targets = sorted(targets_set); target_indices = {text: index for index, text in enumerate(targets)}
    rows, counts = [], Counter()
    for selected, groups in prepared:
        source = selected["input"]
        keys = sorted({(span["input_start"], span["input_end"], span["text"])
                       for group in groups for path in group for span in path["spans"]})
        indices = {key: index for index, key in enumerate(keys)}
        edges = [{"input_start": start, "input_end": end, "text": text,
                  "target_index": target_indices[text]} for start, end, text in keys]
        path_groups = []
        for group in groups:
            values = []
            for path in group:
                edge_indices = [indices[(span["input_start"], span["input_end"], span["text"])]
                                for span in path["spans"]]
                numerators = [span["input_end"] - span["input_start"] for span in path["spans"]]
                need(sum(numerators) == len(source), "source length mass differs")
                values.append({"path_id": path["path_id"], "edge_indices": edge_indices,
                    "weight_numerators": numerators, "weight_denominator": len(source)})
            path_groups.append(values)
        row = {"id": selected["id"], "input": source,
               "component_id": selected["component_id"], "candidates": list(selected["arms"]["b_union"]["candidates"]),
               "positive_indices": list(selected["arms"]["b_union"]["positive_indices"]),
               "edges": edges, "paths": path_groups}
        rows.append(row); counts["rows"] += 1; counts["edges"] += len(edges)
        counts["candidates"] += len(row["candidates"])
        counts["paths"] += sum(len(group) for group in path_groups)
        for start, end, text in keys:
            copied = source[start:end] == text
            counts["copy_edges" if copied else "convert_edges"] += 1
            counts["unresolved_ascii_edges"] += int(not copied and any(
                "a" <= char <= "z" or "A" <= char <= "Z" or "0" <= char <= "9"
                for char in text))
    counts["unique_target_texts"] = len(targets)
    counts["max_target_characters"] = max(map(len, targets))
    counts["max_edges_per_row"] = max(len(row["edges"]) for row in rows)
    return targets, rows, dict(sorted(counts.items()))


def authorized_inventory():
    need(all(sha(path) == expected for path, expected in FIXED.items()),
         "frozen FIT authorization changed")
    proof = read(DATA / "independent-verification.json")
    need(proof.get("verified") is True and proof.get("counts", {}).get("rows") == 1642
         and proof.get("capture_sha256") == FIXED[DATA / "fit-paths.jsonl"],
         "FIT path proof differs")
    selected = read(SELECTION).get("rows")
    need(isinstance(selected, list) and len(selected) == ROWS
         and len({row.get("id") for row in selected}) == ROWS, "FIT selection differs")
    wanted, records = {row["id"] for row in selected}, {}
    for record in jsonl(DATA / "fit-paths.jsonl"):
        if record.get("id") in wanted:
            need(record["id"] not in records, "duplicate FIT path row")
            records[record["id"]] = record
    need(set(records) == wanted, "selected FIT path row missing")
    targets, rows, counts = reconstruct_inventory(selected, [records[row["id"]] for row in selected])
    report = read(INVENTORY / "report.json")
    need(report.get("schema") == "shared-edge-fit-inventory-v1"
         and report.get("scope") == "authorized_fixed_fit_only"
         and report.get("targets_sha256") == sha(INVENTORY / "targets.json")
         and report.get("rows_sha256") == sha(INVENTORY / "rows.jsonl")
         and all(sha(path) == digest for path, digest in report.get("pins", {}).items())
         and report.get("counts") == counts
         and targets == read(INVENTORY / "targets.json")
         and rows == jsonl(INVENTORY / "rows.jsonl"), "saved FIT inventory differs")
    need(counts["rows"] == ROWS and counts["unique_target_texts"] == TARGETS
         and counts["edges"] == EDGES and counts["candidates"] == CANDIDATES,
         "FIT inventory fixed counts differ")
    selection = [{"id": row["id"], "input": row["input"],
                  "component_id": row["component_id"], "candidates": row["candidates"],
                  "positive_indices": row["positive_indices"]} for row in rows]
    return targets, rows, selection, counts, report


def expected_pins(old_report):
    files = [PLAN, PRODUCER, ROOT / "hanglish/models/shared_edge_residual.py",
             ROOT / "scripts/prepare_shared_edge_fit.py", INVENTORY / "report.json",
             INVENTORY / "targets.json", INVENTORY / "rows.jsonl", OLD / "report.json",
             OLD / "independent-verification.json", OLD / "initial.safetensors",
             OLD / "feature-manifest.json", OLD / "before.jsonl",
             OLD / "source_local/after.jsonl"]
    base = dict(old_report["pins"]); base.update({str(path): sha(path) for path in files})
    authorization = read(AUTHORIZATION)
    need(authorization == {"schema": "shared-edge-residual-prelaunch-authorization-v1",
                           "rows": ROWS, "targets": TARGETS, "recipe": RECIPE,
                           "pins": base}, "prelaunch authorization differs")
    return {**base, str(AUTHORIZATION.resolve()): sha(AUTHORIZATION)}


def verify_target_cache(run, targets, selection, report, helper):
    tokens = read(run / "tokens.json")
    need(set(tokens) == {"reserved_tokens", "target_counts", "candidate_counts",
                         "max_target_tokens", "max_candidate_tokens"},
         "token census fields differ")
    reserved, target_counts, candidate_counts = (tokens["reserved_tokens"],
                                                  tokens["target_counts"],
                                                  tokens["candidate_counts"])
    need(type(reserved) is int and 0 <= reserved <= 256
         and isinstance(target_counts, list) and len(target_counts) == len(targets)
         and all(type(value) is int and reserved <= value <= 256 for value in target_counts)
         and isinstance(candidate_counts, list) and len(candidate_counts) == len(selection)
         and all(isinstance(values, list) and len(values) == len(row["candidates"])
                 and all(type(value) is int and reserved <= value <= 256 for value in values)
                 for values, row in zip(candidate_counts, selection))
         and tokens["max_target_tokens"] == max(target_counts)
         and tokens["max_candidate_tokens"] == max(max(values) for values in candidate_counts),
         "token length metadata differs")
    metadata = read(run / "target-features.json")
    expected_batches = sum(math.ceil(min(TARGET_CHUNK, len(targets) - start) / TARGET_BATCH)
                           for start in range(0, len(targets), TARGET_CHUNK))
    need(set(metadata) == {"schema", "targets_sha256", "tokens_sha256", "tensor_hashes",
                           "tensor_file_sha256", "parent_forward_batches", "chunk_size", "batch_size"}
         and metadata["schema"] == RUN_SCHEMA
         and metadata["targets_sha256"] == sha(run / "targets.json")
         and metadata["tokens_sha256"] == sha(run / "tokens.json")
         and metadata["tensor_file_sha256"] == sha(run / "target-features.safetensors")
         and metadata["parent_forward_batches"] == expected_batches
         and metadata["chunk_size"] == TARGET_CHUNK and metadata["batch_size"] == TARGET_BATCH,
         "target feature metadata differs")
    try:
        from safetensors import safe_open
        import numpy as np
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("safetensors and NumPy are required") from error
    with safe_open(run / "target-features.safetensors", framework="numpy") as tensors:
        need(set(tensors.keys()) == {"vectors", "has_tokens"}, "target cache tensors differ")
        vectors, flags = tensors.get_tensor("vectors"), tensors.get_tensor("has_tokens")
        need(tuple(vectors.shape) == (TARGETS, 256) and str(vectors.dtype) == "float32"
             and tuple(flags.shape) == (TARGETS,) and str(flags.dtype) == "bool"
             and vectors.flags["C_CONTIGUOUS"] and flags.flags["C_CONTIGUOUS"]
             and np.isfinite(vectors).all(), "target cache shape/dtype/value differs")
        expected_flags = np.asarray([value > reserved for value in target_counts], dtype=np.bool_)
        need(np.array_equal(flags, expected_flags)
             and np.count_nonzero(vectors[~flags]) == 0, "target token flags/zero vectors differ")
        hashes = {"vectors": helper.tensor_digest(vectors), "has_tokens": helper.tensor_digest(flags)}
    need(metadata["tensor_hashes"] == hashes
         and report.get("target_cache_tensor_hashes") == hashes
         and report.get("target_parent_forward_batches") == expected_batches
         and report.get("target_cache_bytes") == (run / "target-features.safetensors").stat().st_size
         and report["target_cache_bytes"] <= CACHE_BYTES, "target cache hash/size report differs")
    return {"targets": len(targets), "vector_shape": [TARGETS, 256],
            "has_tokens": int(sum(bool(value) for value in flags)),
            "parent_forward_batches": expected_batches,
            "bytes": report["target_cache_bytes"], "tensor_hashes": hashes}


def ulp_sum_tolerance(contributions):
    need(contributions and all(finite(value) for value in contributions),
         "contribution tolerance domain differs")
    magnitude = max(1., sum(abs(value) for value in contributions))
    ulp = 2. ** (math.frexp(magnitude)[1] - 24)
    levels = math.ceil(math.log2(len(contributions))) if len(contributions) > 1 else 0
    return 4 * (levels + 1) * ulp


def audit_after_row(saved, inventory, base, control, helper):
    expected_fields = {"id", "correct", "loss", "scores", "base_scores", "residuals",
                       "edge_ids", "edge_scores", "path_audit", "winning_path_ids"}
    need(set(saved) == expected_fields and saved["id"] == inventory["id"],
         "after row fields/identity differ")
    edges = [[edge["input_start"], edge["input_end"], edge["text"]]
             for edge in inventory["edges"]]
    edge_scores = saved["edge_scores"]
    need(saved["edge_ids"] == edges and isinstance(edge_scores, list)
         and len(edge_scores) == len(edges) and all(finite(value) for value in edge_scores),
         "edge domain/scores differ")
    audits, candidate_residuals, winners = saved["path_audit"], [], []
    need(isinstance(audits, list) and len(audits) == len(inventory["paths"]),
         "candidate path audit inventory differs")
    contribution_by_edge = {}
    for expected_paths, actual_paths in zip(inventory["paths"], audits):
        need(len(expected_paths) == len(actual_paths), "distinct path count differs")
        path_scores, path_ids = [], []
        for expected, actual in zip(expected_paths, actual_paths):
            need(set(actual) == {"path_id", "edge_ids", "weights", "contributions", "residual"}
                 and actual["path_id"] == expected["path_id"]
                 and actual["edge_ids"] == expected["edge_indices"], "path identity differs")
            weights = [fp32(value / expected["weight_denominator"])
                       for value in expected["weight_numerators"]]
            need(actual["weights"] == weights and abs(sum(weights) - 1.) <= 2e-6,
                 "path source weights differ")
            contributions = [fp32(fp32(edge_scores[index]) * fp32(weight))
                             for index, weight in zip(expected["edge_indices"], weights)]
            need(actual["contributions"] == contributions and finite(actual["residual"])
                 and abs(actual["residual"] - sum(contributions))
                     <= ulp_sum_tolerance(contributions), "path weighted contribution differs")
            for edge_index, contribution in zip(expected["edge_indices"], contributions):
                if edge_index in contribution_by_edge:
                    need(contribution_by_edge[edge_index] == contribution,
                         "shared edge contribution changed across paths")
                else:
                    contribution_by_edge[edge_index] = contribution
            path_scores.append(actual["residual"]); path_ids.append(actual["path_id"])
        maximum = max(path_scores)
        candidate_residuals.append(maximum)
        winners.append([path_id for path_id, value in zip(path_ids, path_scores)
                        if value == maximum])
    need(len(contribution_by_edge) == len(edges), "an edge was not covered by path audit")
    need(saved["winning_path_ids"] == winners and saved["residuals"] == candidate_residuals,
         "path maximum or winning IDs differ")
    need(saved["base_scores"] == base["scores"]
         and len(saved["scores"]) == len(saved["base_scores"]) == len(saved["residuals"])
         and all(score == fp32(fp32(parent) + fp32(residual))
                 for score, parent, residual in zip(
                     saved["scores"], saved["base_scores"], saved["residuals"])),
         "final score is not FP32 base plus residual")
    loss = helper.listwise_loss(saved["scores"], inventory["positive_indices"])
    hit = int(helper.argmax(saved["scores"]) in inventory["positive_indices"])
    need(abs(loss - saved["loss"]) <= helper.loss_tolerance(saved["scores"])
         and saved["correct"] == hit, "after CE loss or argmax differs")
    return {"correct": hit, "gain": int(hit and not control["correct"]),
            "loss": int(control["correct"] and not hit),
            "edge_scores": edge_scores, "scores": saved["scores"]}


def verify_before(run, selection, inventory, old_before):
    rows = jsonl(run / "before.jsonl")
    need(len(rows) == ROWS, "initial parity row count differs")
    for selected, projected, historical, saved in zip(selection, inventory, old_before, rows):
        need(set(saved) == {"id", "scores", "shared_edges", "all_path_weights_verified",
                            "all_shared_contributions_equal"}
             and saved["id"] == selected["id"] and saved["scores"] == historical["scores"]
             and saved["shared_edges"] == len(projected["edges"])
             and saved["all_path_weights_verified"] is True
             and saved["all_shared_contributions_equal"] is True,
             "initial parent/shared-edge parity differs")
    return {"rows": len(rows), "all_parent_scores_exact": True}


def verify_updates(run, row_count=ROWS, epochs=EPOCHS):
    rows = jsonl(run / "updates.jsonl")
    need(len(rows) == row_count * epochs, "shared-edge update count differs")
    maxima = dict.fromkeys(GRADIENT_GROUPS, 0.); totals = [0.] * epochs
    selection = read(run / "selection.json")
    need(len(selection) == row_count, "update selection denominator differs")
    for index, saved in enumerate(rows):
        epoch = index // row_count
        need(set(saved) == {"update", "epoch", "id", "loss", "gradient_norm", "gradient_groups"}
             and saved["update"] == index + 1 and saved["epoch"] == epoch
             and saved["id"] == selection[index % row_count]["id"]
             and finite(saved["loss"]) and saved["loss"] >= -1e-5
             and finite(saved["gradient_norm"]) and saved["gradient_norm"] >= 0,
             "shared-edge update order/loss differs")
        groups = saved["gradient_groups"]
        need(isinstance(groups, dict) and set(groups) == set(GRADIENT_GROUPS)
             and all(finite(value) and value >= 0 for value in groups.values()),
             "shared-edge gradient evidence differs")
        if index > 0:
            for name, value in groups.items(): maxima[name] = max(maxima[name], value)
        totals[epoch] += saved["loss"]
    need(all(value > 0 for value in maxima.values()), "upstream gradient stayed zero")
    return {"updates": len(rows), "gradient_maxima": maxima,
            "epoch_mean_loss": [value / row_count for value in totals]}


def expected_operations(selection=None, targets=TARGETS, updates=UPDATES):
    names = ["load_parent_and_initial", "token_census"]
    names.extend(f"isolated_targets:{start}" for start in range(0, targets, TARGET_CHUNK))
    names.append("close_target_cache")
    selection = read(RUN / "selection.json") if selection is None else selection
    names.extend("initial:" + row["id"] for row in selection)
    names.extend(f"shared_edge:update:{index}" for index in range(1, updates + 1))
    names.extend("fit:" + row["id"] for row in selection)
    names.append("save_reload")
    return names


def verify_resources(report, path, selection=None, targets=TARGETS, updates=UPDATES):
    trace = jsonl(path); baseline = report.get("baseline_swap_bytes")
    need(trace and type(baseline) is int and baseline >= 0, "resource trace/baseline differs")
    expected = expected_operations(selection, targets, updates)
    next_index, current, previous = 0, None, -1.
    peak_rss = peak_swap = peak_operation = 0.
    for row in trace:
        need(set(row) == {"at", "elapsed_s", "rss_bytes", "swap_used_bytes",
                          "swap_growth_bytes", "operation_elapsed_s", "stage", "operation"}
             and row["stage"] in {"operation_start", "operation_end", "watchdog"}
             and finite(row["elapsed_s"]) and row["elapsed_s"] >= previous
             and type(row["rss_bytes"]) is int and row["rss_bytes"] >= 0
             and type(row["swap_used_bytes"]) is int and row["swap_used_bytes"] >= 0
             and type(row["swap_growth_bytes"]) is int
             and row["swap_used_bytes"] - baseline == row["swap_growth_bytes"],
             "resource observation differs")
        previous = row["elapsed_s"]
        need(row["elapsed_s"] <= LIMITS["whole_seconds"]
             and row["rss_bytes"] <= LIMITS["rss_bytes"]
             and row["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"],
             "resource hard limit exceeded")
        if row["operation"] is None:
            need(row["stage"] == "watchdog" and row["operation_elapsed_s"] is None,
                 "idle watchdog attribution differs")
        else:
            need(finite(row["operation_elapsed_s"])
                 and 0 <= row["operation_elapsed_s"] <= LIMITS["update_seconds"],
                 "resource operation time exceeded")
        if row["stage"] == "operation_start":
            need(current is None and next_index < len(expected)
                 and row["operation"] == expected[next_index], "resource start order differs")
            current = row["operation"]; next_index += 1
        elif row["stage"] == "operation_end":
            need(current is not None and row["operation"] == current,
                 "resource operation pairing differs")
            current = None
        else:
            need(row["operation"] == current, "atomic watchdog attribution differs")
        peak_rss = max(peak_rss, row["rss_bytes"]); peak_swap = max(peak_swap, row["swap_growth_bytes"])
        if row["operation_elapsed_s"] is not None:
            peak_operation = max(peak_operation, row["operation_elapsed_s"])
    need(current is None and next_index == len(expected), "resource operation inventory differs")
    final = report.get("final_resources", {})
    need(final.get("stage") == "final" and final.get("operation") is None
         and final.get("operation_elapsed_s") is None and finite(final.get("elapsed_s"))
         and previous <= final["elapsed_s"] <= LIMITS["whole_seconds"]
         and type(final.get("rss_bytes")) is int and final["rss_bytes"] <= LIMITS["rss_bytes"]
         and type(final.get("swap_used_bytes")) is int
         and type(final.get("swap_growth_bytes")) is int
         and final["swap_used_bytes"] - baseline == final["swap_growth_bytes"]
         and final["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"]
         and finite(report.get("elapsed_s"))
         and final["elapsed_s"] <= report["elapsed_s"] <= LIMITS["whole_seconds"],
         "final resource observation differs")
    return {"observations": len(trace), "operation_pairs": len(expected),
            "peak_rss_bytes": max(peak_rss, final["rss_bytes"]),
            "peak_swap_growth_bytes": max(peak_swap, final["swap_growth_bytes"]),
            "peak_operation_seconds": peak_operation,
            "atomic_operation_attribution_verified": True}


def parent_tensor_hashes(helper):
    try:
        from safetensors import safe_open
        import numpy as np
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("safetensors and NumPy are required") from error
    with safe_open(PARENT / "model.safetensors", framework="numpy") as tensors:
        result = {}
        for name in tensors.keys():
            value = tensors.get_tensor(name); need(np.isfinite(value).all(), "parent tensor nonfinite")
            result[name] = helper.tensor_digest(value)
    return result


def verify(run=RUN):
    run = Path(run).resolve(); need(run == RUN.resolve(), "alternate shared-edge run unauthorized")
    need(sha(PRODUCER) == PRODUCER_SHA256
         and sha(ROOT / "hanglish/models/shared_edge_residual.py") == MODEL_SHA256,
         "frozen shared-edge producer/model changed")
    helper = helper_module()
    targets, inventory, selection, counts, inventory_report = authorized_inventory()
    report = read(run / "report.json")
    need(report.get("schema") == RUN_SCHEMA and report.get("completed") is True
         and report.get("rows") == ROWS and report.get("updates") == UPDATES
         and report.get("recipe") == RECIPE and report.get("limits") == LIMITS,
         "completed shared-edge report differs")
    contracts = {"fit_only": True, "hold_opened": False, "dev_opened": False,
        "test_opened": False, "production_changed": False,
        "candidate_hidden_states_used_in_residual": False,
        "reloaded_target_cache_used_for_inference": True,
        "all_fit_initial_parent_scores_exact": True,
        "all_fit_shared_edge_contributions_equal": True,
        "source_codepoint_weight_mass_verified": True}
    need(report.get("contracts") == contracts, "shared-edge scope contract differs")
    old_report, old_proof = read(OLD / "report.json"), read(OLD / "independent-verification.json")
    need(old_report.get("completed") is True and old_report.get("rows") == ROWS
         and old_report.get("recipe") == RECIPE and old_proof.get("verified") is True
         and old_proof.get("run_report_sha256") == sha(OLD / "report.json"),
         "historical control proof differs")
    need(all(sha(path) == digest for path, digest in old_report["pins"].items())
         and all(sha(OLD / name) == digest for name, digest in old_report["outputs"].items()),
         "historical control artifact changed")
    pins = read(run / "pins.json"); expected = expected_pins(old_report)
    need(pins == expected == report.get("pins")
         and all(sha(path) == digest for path, digest in pins.items()), "run pin inventory differs")
    need(read(run / "selection.json") == selection and read(run / "targets.json") == targets,
         "saved selection/targets differ")
    need(set(report.get("outputs", {})) == OUTPUTS, "reported output inventory differs")
    actual = {path.name for path in run.iterdir() if path.is_file()
              and path.name not in {"report.json", "independent-verification.json"}}
    need(actual == OUTPUTS and all(sha(run / name) == digest
                                   for name, digest in report["outputs"].items()),
         "closed output hash/inventory differs")
    target_cache = verify_target_cache(run, targets, selection, report, helper)
    old_before, control = jsonl(OLD / "before.jsonl"), jsonl(OLD / "source_local/after.jsonl")
    before = verify_before(run, selection, inventory, old_before)
    updates = verify_updates(run)
    after_rows = jsonl(run / "after.jsonl")
    need(len(after_rows) == ROWS, "after FIT row count differs")
    audited = [audit_after_row(saved, projected, base, historical, helper)
               for saved, projected, base, historical in zip(
                   after_rows, inventory, old_before, control)]
    fit_correct = sum(value["correct"] for value in audited)
    gains, losses = sum(value["gain"] for value in audited), sum(value["loss"] for value in audited)
    need(report.get("fit_correct") == fit_correct
         and report.get("control_fit_correct") == sum(row["correct"] for row in control)
         and report.get("fit_gains") == gains and report.get("fit_losses") == losses
         and report.get("upstream_gradient_maxima") == updates["gradient_maxima"]
         and len(report.get("epoch_mean_loss", [])) == EPOCHS
         and all(abs(left - right) <= 2e-6 for left, right in zip(
             report["epoch_mean_loss"], updates["epoch_mean_loss"])),
         "reported FIT metrics/update aggregates differ")
    initial_hashes = helper.validate_tensor_file(run / "initial.safetensors")
    final_hashes = helper.validate_tensor_file(run / "residual.safetensors")
    old_initial = helper.validate_tensor_file(OLD / "initial.safetensors")
    parent_hashes = parent_tensor_hashes(helper)
    need(initial_hashes == old_initial == report.get("initial_hashes")
         and final_hashes == report.get("final_hashes")
         and report.get("parent_before") == report.get("parent_after") == parent_hashes,
         "initial/final/parent tensor hashes differ")
    architecture = {key: value for key, value in helper.ARCHITECTURE.items()
                    if key != "span_reduction"}
    architecture.update({"schema": "shared-source-edge-residual-v1",
        "path_reduction": "distinct_path_weighted_sum_max",
        "edge_reduction": "source_codepoint_fraction_sum",
        "target_context": "isolated_exact_output_text"})
    checkpoint = read(run / "checkpoint.json")
    need(report.get("architecture") == architecture and checkpoint == {
        "schema": RUN_SCHEMA, "arm": "shared_edge", "architecture": architecture,
        "recipe": RECIPE, "pins": pins,
        "residual_sha256": sha(run / "residual.safetensors"),
        "initial_sha256": sha(run / "initial.safetensors"),
        "selection_sha256": sha(run / "selection.json")}, "checkpoint identity differs")
    reload = read(run / "reload.json")
    need(isinstance(reload, list) and len(reload) == len(RELOAD_INDICES)
         and report.get("reload_rows") == len(RELOAD_INDICES), "reload count differs")
    for index, saved in zip(RELOAD_INDICES, reload):
        need(set(saved) == {"id", "scores", "reloaded", "edge_scores", "reloaded_edge_scores"}
             and saved["id"] == selection[index]["id"]
             and saved["scores"] == saved["reloaded"] == audited[index]["scores"]
             and saved["edge_scores"] == saved["reloaded_edge_scores"]
                 == audited[index]["edge_scores"], "reload scores/edge scores differ")
    resources = verify_resources(report, run / "resources.jsonl", selection)
    need(not (run / "failure.json").exists(), "training failure artifact exists")
    return {"schema": SCHEMA, "verified": True,
        "report_sha256": sha(run / "report.json"), "producer_sha256": sha(PRODUCER),
        "verifier_sha256": sha(Path(__file__)),
        "counts": {"rows": ROWS, "targets": TARGETS, "edges": EDGES,
                   "candidate_paths": CANDIDATES, "updates": UPDATES,
                   "reload_rows": len(RELOAD_INDICES)},
        "inventory_counts": counts, "target_cache": target_cache, "before": before,
        "updates": updates, "after": {"fit_correct": fit_correct,
            "control_fit_correct": sum(row["correct"] for row in control),
            "gains": gains, "losses": losses},
        "tensors": {"trainable_parameters": 71073,
            "residual_tensors_per_checkpoint": len(helper.RESIDUAL_SHAPES),
            "parent_tensors": len(parent_hashes), "initial_matches_historical": True,
            "parent_unchanged": True}, "resources": resources,
        "inventory_report_sha256": sha(INVENTORY / "report.json"),
        "contracts": {"authorized_fit_inventory_independently_reconstructed": True,
            "all_target_cache_shapes_flags_metadata_and_hashes_verified": True,
            "historical_initial_and_parent_frozen": True,
            "exact_CE_only_update_order_and_losses_verified": True,
            "all_final_edges_paths_weights_contributions_and_winners_replayed": True,
            "shared_edge_contributions_equal_across_paths": True,
            "fp32_base_plus_residual_and_fit_metrics_replayed": True,
            "reload_scores_and_edge_scores_verified": True,
            "closed_output_and_pin_hashes_verified": True,
            "fit_only": True, "hold_opened": False, "dev_opened": False,
            "test_opened": False, "production_changed": False,
            "torch_imported": False, "tokenizer_instantiated": False,
            "neural_model_instantiated": False, "neural_forward_rerun": False,
            "checkpoint_tensors_read_with_safetensors_numpy": True},
        "limitations": [
            "The verifier does not rerun the tokenizer, frozen parent, source encoder, or residual head.",
            "Saved token lengths and isolated target vectors are authenticated and structurally checked, not independently recomputed.",
            "Producer source paths are alignment hypotheses rather than gold segmentation.",
            "FIT accuracy and structural sharing do not establish unseen DEV or 1,000-case quality."],
    }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path); args = parser.parse_args()
    result = verify(args.run); output = args.output or args.run / "independent-verification.json"
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"verified": True, "counts": result["counts"],
                      "after": result["after"], "resources": result["resources"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
