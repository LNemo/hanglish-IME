#!/usr/bin/env python3
"""Independent offline verifier for local-span auxiliary FIT training.

The verifier never imports torch, a tokenizer, the trainer, or project model
code.  Neural forwards are not repeated.  It authenticates frozen inputs,
replays all scalar calculations and order contracts, and reads safetensors
through the NumPy backend only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".omx/artifacts/local-span-objective-training-v1"
OLD = ROOT / ".omx/artifacts/source-span-paired-training-v1"
PREFERENCES = ROOT / ".omx/artifacts/conditional-span-preferences-fit-v1"
PARENT = ROOT / "checkpoints/sentence_ranker"
PLAN = ROOT / ".omx/plans/local-span-objective-v1.md"
HELPER_PATH = ROOT / ".omx/artifacts/verify-source-span-pair-v1.py"
HELPER_SHA256 = "4fbf04d3e340d0565367741f89597c5787cd1383bf283f5ad56919918eeba16a"
RUN_SCHEMA = "local-span-objective-training-v1"
SCHEMA = "local-span-objective-training-independent-verification-v1"
ROWS = 1483
EPOCHS = 2
UPDATES = ROWS * EPOCHS
PREFERENCES_COUNT = 6889
RELOAD_INDICES = sorted(set([0, ROWS - 1] + list(range(0, ROWS, 100))))
GRADIENT_GROUPS = ("source_embedding.", "source_encoder.",
                   "segment_head.0.", "segment_head.2.")
RECIPE = {"seed": 20260921, "epochs": 2, "learning_rate": .001,
          "weight_decay": .01, "clip_norm": 1., "threads": 2,
          "dtype": "float32", "optimizer": "AdamW", "accumulation": 1,
          "local_loss_weight": .25}
LIMITS = {"whole_seconds": 1800., "update_seconds": 60.,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
OUTPUTS = {"pins.json", "selection.json", "resources.jsonl",
           "control-parity.jsonl", "initial.safetensors", "updates.jsonl",
           "after.jsonl", "residual.safetensors", "reload.json", "checkpoint.json"}


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
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSONL at {path}:{line_number}") from error
    return values


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def fp32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def softplus(value):
    need(finite(value), "local margin is nonfinite")
    return max(value, 0.) + math.log1p(math.exp(-abs(value)))


def mean_softplus_negative(margins):
    need(isinstance(margins, list) and all(finite(value) for value in margins),
         "local margins malformed")
    if not margins:
        return 0.
    return sum(softplus(-value) for value in margins) / len(margins)


def softplus_loss_tolerance(margins, expected):
    """Bound FP32 elementwise softplus plus tree-reduction rounding error."""
    need(isinstance(margins, list) and finite(expected),
         "softplus tolerance domain differs")
    if not margins:
        need(expected == 0., "empty local loss differs")
        return 0.
    terms = [softplus(-value) for value in margins]
    magnitude = max(1., abs(expected), *(abs(value) for value in terms))
    ulp = 2. ** (math.frexp(magnitude)[1] - 24)
    reduction_levels = math.ceil(math.log2(len(terms))) if len(terms) > 1 else 0
    # Four ULPs cover the FP32 softplus kernel; each tree level and final
    # division receive another four ULPs at the largest participating scale.
    return 4 * (reduction_levels + 2) * ulp


def _load_helper():
    need(sha(HELPER_PATH) == HELPER_SHA256, "frozen pure verifier helper changed")
    spec = importlib.util.spec_from_file_location("frozen_source_span_verifier", HELPER_PATH)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def expected_pins(old_report):
    files = [PLAN, ROOT / "scripts/train_local_span_objective.py",
             ROOT / "hanglish/models/local_span_objective.py",
             ROOT / "hanglish/models/span_supervision.py",
             PREFERENCES / "report.json", PREFERENCES / "independent-verification.json",
             PREFERENCES / "preferences.jsonl", OLD / "report.json",
             OLD / "independent-verification.json", OLD / "initial.safetensors",
             OLD / "feature-manifest.json", OLD / "source_local/residual.safetensors",
             OLD / "source_local/after.jsonl"]
    result = dict(old_report["pins"])
    result.update({str(path): sha(path) for path in files})
    return result


def verify_inputs(helper):
    old_report = read(OLD / "report.json")
    old_proof = read(OLD / "independent-verification.json")
    need(old_report.get("completed") is True
         and old_report.get("schema") == "source-span-paired-training-v1"
         and old_report.get("rows") == ROWS
         and old_proof.get("verified") is True
         and old_proof.get("run_report_sha256") == sha(OLD / "report.json"),
         "historical control authentication differs")
    need(all(old_report["recipe"].get(key) == value
             for key, value in RECIPE.items() if key != "local_loss_weight"),
         "historical control recipe differs")
    need(all(sha(path) == digest for path, digest in old_report["pins"].items())
         and all(sha(OLD / name) == digest
                 for name, digest in old_report["outputs"].items()),
         "historical control pin or output changed")
    preference_report = read(PREFERENCES / "report.json")
    preference_proof = read(PREFERENCES / "independent-verification.json")
    need(preference_proof.get("schema")
             == "conditional-span-preference-independent-verification-v1"
         and preference_proof.get("verified") is True
         and preference_proof.get("report_sha256") == sha(PREFERENCES / "report.json")
         and preference_report.get("schema") == "conditional-span-preference-v1"
         and preference_report.get("scope") == "authorized_fit_selected_candidates_only"
         and preference_proof.get("preferences_sha256")
             == preference_report.get("preferences_sha256")
         and preference_proof.get("selection_sha256") == sha(helper.SELECTION)
         and preference_proof.get("fit_paths_sha256") == sha(helper.DATA / "fit-paths.jsonl")
         and preference_proof.get("counts") == preference_report.get("counts")
         and preference_report.get("counts", {}).get("rows") == ROWS
         and preference_report.get("counts", {}).get("distinct_preferences")
             == PREFERENCES_COUNT
         and preference_report.get("contextual_pairs_with_both_preferences") == 0
         and preference_proof.get("contracts") == {
             "accepted_labels_reconstructed_from_frozen_selection": True,
             "all_selected_candidate_paths_expanded_from_compact_edges": True,
             "common_source_boundaries_recomputed": True,
             "exact_preference_pair_set_and_witnesses_recomputed": True,
             "models_or_torch_loaded": False,
             "multi_change_outside_segmentation_whole_source_and_ambiguous_excluded": True,
             "row_and_report_aggregates_recomputed": True,
             "test_hold_and_canonical_labels_accessed": False,
         }
         and preference_report.get("preferences_sha256")
             == sha(PREFERENCES / "preferences.jsonl"),
         "local preference proof differs")
    need(all(sha(path) == digest for path, digest in preference_report["pins"].items()),
         "local preference input changed")
    selection = helper.reconstruct_selection()
    labels = jsonl(PREFERENCES / "preferences.jsonl")
    need(len(selection) == len(labels) == ROWS, "FIT/local row count differs")
    total_preferences = 0
    for row, label in zip(selection, labels):
        need(all(label.get(key) == row[key] for key in row),
             "FIT/local row order or domain differs")
        preferences = label.get("preferences")
        need(isinstance(preferences, list), "local preferences malformed")
        for preference in preferences:
            witnesses = preference.get("witnesses")
            need(isinstance(witnesses, list) and len(witnesses) == 1,
                 "local loss cannot be reconstructed from mean margin")
        total_preferences += len(preferences)
    need(total_preferences == PREFERENCES_COUNT,
         "local preference denominator differs")
    old_after = jsonl(OLD / "source_local/after.jsonl")
    old_before = jsonl(OLD / "before.jsonl")
    need(len(old_after) == len(old_before) == ROWS
         and [value.get("id") for value in old_after]
             == [row["id"] for row in selection]
         and [value.get("id") for value in old_before]
             == [row["id"] for row in selection],
         "historical control score order differs")
    return selection, labels, old_before, old_after, old_report, preference_proof


def verify_control(rows, labels, old_after, path):
    records = jsonl(path)
    need(len(records) == len(rows), "historical control replay row count differs")
    preference_count = positive = 0
    loss_total = 0.
    for row, label, historical, saved in zip(rows, labels, old_after, records):
        need(set(saved) == {"id", "scores", "local_loss", "local_margins"}
             and saved["id"] == row["id"]
             and saved["scores"] == historical["scores"],
             "historical control replay scores differ")
        margins = saved["local_margins"]
        need(isinstance(margins, list)
             and len(margins) == len(label["preferences"])
             and all(finite(value) for value in margins)
             and finite(saved["local_loss"]), "control local margin/loss differs")
        expected_loss = mean_softplus_negative(margins)
        need(abs(expected_loss - saved["local_loss"])
             <= softplus_loss_tolerance(margins, expected_loss),
             "control local softplus loss differs")
        preference_count += len(margins)
        positive += sum(value > 0 for value in margins)
        loss_total += expected_loss
    return {"rows": len(records), "preferences": preference_count,
            "positive_margins": positive, "mean_local_loss": loss_total / len(records)}


def verify_updates(rows, path):
    records = jsonl(path)
    need(len(records) == UPDATES, "local training update count differs")
    maxima = dict.fromkeys(GRADIENT_GROUPS, 0.)
    epoch_sums = [{"total": 0., "candidate": 0., "local": 0.}
                  for _ in range(EPOCHS)]
    for index, saved in enumerate(records):
        epoch = index // len(rows)
        need(set(saved) == {"update", "epoch", "id", "losses",
                            "gradient_norm", "gradient_groups"}
             and saved["update"] == index + 1 and saved["epoch"] == epoch
             and saved["id"] == rows[index % len(rows)]["id"],
             "local training update order differs")
        losses = saved["losses"]
        need(isinstance(losses, dict) and set(losses) == {"total", "candidate", "local"}
             and all(finite(value) for value in losses.values())
             and losses["candidate"] >= -1e-5 and losses["local"] >= 0.,
             "local training loss malformed")
        combined = fp32(fp32(losses["candidate"])
                        + fp32(RECIPE["local_loss_weight"] * fp32(losses["local"])))
        need(losses["total"] == combined, "combined local/candidate loss differs")
        need(finite(saved["gradient_norm"]) and saved["gradient_norm"] >= 0,
             "gradient norm is nonfinite")
        groups = saved["gradient_groups"]
        need(isinstance(groups, dict) and set(groups) == set(GRADIENT_GROUPS)
             and all(finite(value) and value >= 0 for value in groups.values()),
             "gradient group evidence differs")
        if index > 0:
            for name, value in groups.items():
                maxima[name] = max(maxima[name], value)
        for key in epoch_sums[epoch]:
            epoch_sums[epoch][key] += losses[key]
    need(all(value > 0 for value in maxima.values()),
         "upstream gradient group stayed zero")
    means = [{key: value / len(rows) for key, value in totals.items()}
             for totals in epoch_sums]
    return {"updates": len(records), "gradient_maxima": maxima,
            "epoch_mean_losses": means}


def verify_after(helper, rows, labels, old_before, old_after, path):
    records = jsonl(path)
    need(len(records) == len(rows), "trained FIT row count differs")
    correct = gains = losses = positive = margin_count = 0
    local_total = 0.
    for row, label, before, control, saved in zip(
            rows, labels, old_before, old_after, records):
        need(set(saved) == {"id", "correct", "scores", "base_scores", "residuals",
                            "local_loss", "local_margins"}
             and saved["id"] == row["id"], "trained FIT row identity differs")
        scores, base, residuals = saved["scores"], saved["base_scores"], saved["residuals"]
        need(isinstance(scores, list) and isinstance(base, list)
             and isinstance(residuals, list)
             and len(scores) == len(base) == len(residuals) == len(row["candidates"])
             and all(finite(value) for value in scores + base + residuals)
             and base == before["scores"]
             and all(score == fp32(fp32(parent) + fp32(residual))
                     for score, parent, residual in zip(scores, base, residuals)),
             "trained score is not frozen base plus residual")
        hit = int(helper.argmax(scores) in row["positive_indices"])
        need(saved["correct"] == hit, "trained FIT argmax differs")
        control_hit = control["correct"]
        correct += hit
        gains += int(hit and not control_hit)
        losses += int(control_hit and not hit)
        margins = saved["local_margins"]
        need(isinstance(margins, list)
             and len(margins) == len(label["preferences"])
             and all(finite(value) for value in margins)
             and finite(saved["local_loss"]), "trained local margin/loss differs")
        expected_loss = mean_softplus_negative(margins)
        need(abs(expected_loss - saved["local_loss"])
             <= softplus_loss_tolerance(margins, expected_loss),
             "trained local softplus loss differs")
        positive += sum(value > 0 for value in margins)
        margin_count += len(margins)
        local_total += expected_loss
    return {"rows": len(records), "correct": correct, "gains": gains, "losses": losses,
            "local_positive_margins": positive, "local_margin_count": margin_count,
            "mean_local_loss": local_total / len(records), "records": records}


def expected_operations(rows):
    names = ["load_parent_control"]
    names.extend("control_parity:" + row["id"] for row in rows)
    names.append("initialize_training")
    names.extend(f"local_aux:update:{index}" for index in range(1, UPDATES + 1))
    names.extend("fit:" + row["id"] for row in rows)
    names.append("save_reload")
    return names


def validate_resource_row(saved, baseline, previous_elapsed):
    need(set(saved) == {"at", "elapsed_s", "rss_bytes", "swap_used_bytes",
                        "swap_growth_bytes", "operation_elapsed_s", "stage", "operation"}
         and saved["stage"] in {"operation_start", "operation_end", "watchdog"}
         and finite(saved["elapsed_s"]) and saved["elapsed_s"] >= previous_elapsed
         and type(saved["rss_bytes"]) is int and saved["rss_bytes"] >= 0
         and type(saved["swap_used_bytes"]) is int and saved["swap_used_bytes"] >= 0
         and type(saved["swap_growth_bytes"]) is int
         and saved["swap_used_bytes"] - baseline == saved["swap_growth_bytes"],
         "resource observation differs")
    need(saved["elapsed_s"] <= LIMITS["whole_seconds"]
         and saved["rss_bytes"] <= LIMITS["rss_bytes"]
         and saved["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"],
         "resource hard limit exceeded")
    if saved["operation"] is None:
        need(saved["stage"] == "watchdog" and saved["operation_elapsed_s"] is None,
             "idle resource observation differs")
    else:
        need(finite(saved["operation_elapsed_s"])
             and 0 <= saved["operation_elapsed_s"] <= LIMITS["update_seconds"],
             "resource operation timing differs")


def verify_resources(rows, report, path):
    records = jsonl(path)
    need(records, "resource trace is empty")
    baseline = report.get("baseline_swap_bytes")
    need(type(baseline) is int and baseline >= 0, "swap baseline differs")
    expected = expected_operations(rows)
    current = None
    next_operation = 0
    previous = -1.
    peak_rss = peak_swap = peak_operation = 0
    for saved in records:
        validate_resource_row(saved, baseline, previous)
        previous = saved["elapsed_s"]
        if saved["stage"] == "operation_start":
            need(current is None and next_operation < len(expected)
                 and saved["operation"] == expected[next_operation],
                 "resource operation start order differs")
            current = saved["operation"]
            next_operation += 1
        elif saved["stage"] == "operation_end":
            need(current is not None and saved["operation"] == current,
                 "resource operation end pairing differs")
            current = None
        else:
            need(saved["operation"] == current,
                 "atomic watchdog operation attribution differs")
        peak_rss = max(peak_rss, saved["rss_bytes"])
        peak_swap = max(peak_swap, saved["swap_growth_bytes"])
        if saved["operation_elapsed_s"] is not None:
            peak_operation = max(peak_operation, saved["operation_elapsed_s"])
    need(current is None and next_operation == len(expected),
         "resource operation inventory differs")
    final = report.get("final_resources")
    need(isinstance(final, dict) and final.get("stage") == "final"
         and final.get("operation") is None and final.get("operation_elapsed_s") is None
         and finite(final.get("elapsed_s")) and previous <= final["elapsed_s"] <= LIMITS["whole_seconds"]
         and type(final.get("rss_bytes")) is int and final["rss_bytes"] <= LIMITS["rss_bytes"]
         and type(final.get("swap_used_bytes")) is int
         and type(final.get("swap_growth_bytes")) is int
         and final["swap_used_bytes"] - baseline == final["swap_growth_bytes"]
         and final["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"]
         and finite(report.get("elapsed_s"))
         and final["elapsed_s"] <= report["elapsed_s"] <= LIMITS["whole_seconds"],
         "final resource observation differs")
    return {"observations": len(records), "operation_pairs": len(expected),
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
        hashes = {}
        for name in tensors.keys():
            value = tensors.get_tensor(name)
            need(np.isfinite(value).all(), "parent tensor is nonfinite")
            hashes[name] = helper.tensor_digest(value)
    return hashes


def verify(run=RUN):
    run = Path(run).resolve()
    need(run == RUN.resolve(), "alternate local training run is not authorized")
    helper = _load_helper()
    rows, labels, old_before, old_after, old_report, preference_proof = verify_inputs(helper)
    report = read(run / "report.json")
    need(report.get("schema") == RUN_SCHEMA and report.get("completed") is True
         and report.get("rows") == ROWS and report.get("updates") == UPDATES
         and report.get("recipe") == RECIPE and report.get("limits") == LIMITS,
         "completed local training report differs")
    need(report.get("contracts") == {"fit_only": True, "dev_opened": False,
         "test_opened": False, "hold_opened": False, "production_changed": False},
         "training access/scope contract differs")
    need(set(report.get("outputs", {})) == OUTPUTS,
         "reported output inventory differs")
    actual = {path.name for path in run.iterdir() if path.is_file()
              and path.name not in {"report.json", "independent-verification.json"}}
    need(actual == OUTPUTS, "unreported training output differs")
    for name, digest in report["outputs"].items():
        need(sha(run / name) == digest, "reported output hash differs: " + name)

    need(read(run / "selection.json") == rows, "saved FIT selection differs")
    pins = read(run / "pins.json")
    expected = expected_pins(old_report)
    need(pins == expected == report.get("pins"), "training pin inventory differs")
    need(all(sha(path) == digest for path, digest in pins.items()),
         "pinned training input changed")

    control = verify_control(rows, labels, old_after, run / "control-parity.jsonl")
    updates = verify_updates(rows, run / "updates.jsonl")
    after = verify_after(helper, rows, labels, old_before, old_after, run / "after.jsonl")
    need(report.get("historical_control_all_row_parity") is True
         and report.get("trained_global_all_row_parity") is True
         and report.get("control_fit_correct") == sum(value["correct"] for value in old_after)
         and report.get("fit_correct") == after["correct"]
         and report.get("fit_gains") == after["gains"]
         and report.get("fit_losses") == after["losses"]
         and report.get("local_positive_margins") == after["local_positive_margins"]
         and report.get("local_margin_count") == after["local_margin_count"]
         and report.get("upstream_gradient_maxima") == updates["gradient_maxima"]
         and len(report.get("epoch_mean_losses", [])) == EPOCHS
         and all(abs(saved[key] - calculated[key]) <= 2e-6
                 for saved, calculated in zip(report["epoch_mean_losses"],
                                              updates["epoch_mean_losses"])
                 for key in ("total", "candidate", "local")),
         "reported training metrics differ")

    initial_hashes = helper.validate_tensor_file(run / "initial.safetensors")
    final_hashes = helper.validate_tensor_file(run / "residual.safetensors")
    old_initial_hashes = helper.validate_tensor_file(OLD / "initial.safetensors")
    need(initial_hashes == old_initial_hashes == report.get("initial_hashes")
         and final_hashes == report.get("final_hashes"),
         "initial/final residual tensor hashes differ")
    parent_hashes = parent_tensor_hashes(helper)
    need(report.get("parent_before") == report.get("parent_after") == parent_hashes,
         "frozen parent tensor hashes differ")

    checkpoint = read(run / "checkpoint.json")
    need(checkpoint == {"schema": RUN_SCHEMA, "arm": "local_aux",
         "architecture": helper.ARCHITECTURE, "recipe": RECIPE, "pins": pins,
         "residual_sha256": sha(run / "residual.safetensors"),
         "initial_sha256": sha(run / "initial.safetensors"),
         "selection_sha256": sha(run / "selection.json")},
         "checkpoint identity differs")
    reload_rows = read(run / "reload.json")
    need(isinstance(reload_rows, list) and len(reload_rows) == len(RELOAD_INDICES)
         and report.get("reload_rows") == len(RELOAD_INDICES),
         "reload sample count differs")
    for index, saved in zip(RELOAD_INDICES, reload_rows):
        need(set(saved) == {"id", "scores", "reloaded"}
             and saved["id"] == rows[index]["id"]
             and saved["scores"] == saved["reloaded"] == after["records"][index]["scores"],
             "save/reload score parity differs")
    resources = verify_resources(rows, report, run / "resources.jsonl")

    after_public = dict(after)
    after_public.pop("records")
    return {
        "schema": SCHEMA, "verified": True,
        "report_sha256": sha(run / "report.json"),
        "producer_sha256": sha(ROOT / "scripts/train_local_span_objective.py"),
        "verifier_sha256": sha(Path(__file__)),
        "counts": {"selected_fit_rows": ROWS, "preferences": PREFERENCES_COUNT,
                   "updates": UPDATES, "reload_rows": len(RELOAD_INDICES)},
        "control": control, "updates": updates, "after": after_public,
        "tensors": {"trainable_parameters": 71073,
                    "residual_tensors_per_checkpoint": len(helper.RESIDUAL_SHAPES),
                    "parent_tensors": len(parent_hashes),
                    "initial_matches_historical_control": True,
                    "parent_unchanged": True},
        "resources": resources,
        "preference_proof_sha256": sha(PREFERENCES / "independent-verification.json"),
        "preference_report_sha256": preference_proof["report_sha256"],
        "contracts": {
            "full_fixed_fit_selection_reconstructed": True,
            "local_dataset_independent_proof_authenticated": True,
            "one_witness_per_preference_asserted": True,
            "historical_control_all_scores_exact": True,
            "score_fp32_base_plus_residual_replayed": True,
            "local_softplus_losses_replayed_from_saved_margins": True,
            "exact_two_epoch_update_order_replayed": True,
            "fit_argmax_gains_losses_replayed": True,
            "fit_only": True, "hold_opened": False, "dev_opened": False,
            "test_opened": False, "production_changed": False,
            "torch_imported": False, "tokenizer_instantiated": False,
            "neural_model_instantiated": False, "neural_inference_rerun": False,
            "checkpoint_tensors_read_with_safetensors_numpy": True,
        },
        "limitations": [
            "The verifier does not rerun parent or residual neural forwards or tokenization.",
            "Local softplus losses are recomputed from saved neural margins; tensor-to-margin computation is authenticated through pinned code and artifacts, not independently inferred.",
            "This FIT-only run does not establish unseen quality, candidate generation coverage, DEV quality, or the 1,000-case regression.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.run)
    output = args.output or args.run / "independent-verification.json"
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        stream.write("\n")
    print(json.dumps({"verified": True, "counts": result["counts"],
                      "after": result["after"], "resources": result["resources"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
