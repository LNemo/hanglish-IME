"""Four-fold TRAIN diagnostic: candidate scores plus an explicit NONE class."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch

from scripts.listwise_semantic_contract import decision, listwise_none_loss
from scripts.probe_role_free_semantic_head import read_targets
from scripts.probe_symmetric_pair_selector import canonical, rss_bytes, sha
from scripts.probe_listwise_semantic_resources import (
    RECIPE, configure_model, encode_row, load_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".omx/artifacts/listwise-semantic-selector-v1"
RESOURCE = ROOT / ".omx/artifacts/listwise-semantic-resource-v1/report.json"
RESOURCE_SHA = "f455629f7d4aecc75c1c2a74419199321e6d77ec6e7c85b82ba5ad10c3e7a7d5"
INPUT_ROWS_SHA = "5379897cd818ce8dce391ab3b8a22d27ba05ff72ad666a67d584ef2a46cd4edc"
PLAN = ROOT / ".omx/plans/listwise-semantic-selector-v1.md"


def write_json(path, value):
    Path(path).write_bytes(canonical(value) + b"\n")


def verify_pins(pins):
    for name, digest in pins.items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"source changed: {name}")


def resources(started):
    elapsed, memory = time.monotonic() - started, rss_bytes()
    if elapsed > RECIPE["max_seconds"] or memory > RECIPE["max_rss_bytes"]:
        raise RuntimeError("listwise semantic training resource cap exceeded")
    return {"elapsed_seconds": elapsed, "peak_rss_bytes": memory}


def fold_indices(rows, fold):
    train = [row["index"] for row in rows if row["eligible"] and row["fold"] != fold]
    held = [row["index"] for row in rows if row["eligible"] and row["fold"] == fold]
    if not train or not held or set(train) & set(held):
        raise ValueError("nonempty disjoint fold required")
    for key in ("component", "group_id"):
        if {rows[index][key] for index in train} & {rows[index][key] for index in held}:
            raise ValueError("training and held groups overlap")
    return train, held


def accepted_indices(row, expected):
    accepted = set(expected)
    return [index for index, candidate in enumerate(row["candidates"])
            if candidate["text"] in accepted]


def target_state(row, positives):
    if not positives:
        return "NONE"
    if row["current_index"] not in positives:
        return "ALT_ONLY"
    return "KEEP_ONLY" if len(positives) == 1 else "MULTI_WITH_KEEP"


def parameter_digest(model, *, trainable):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad == trainable:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_resource_report(path=RESOURCE, expected_sha=RESOURCE_SHA):
    if sha(path) != expected_sha:
        raise ValueError("frozen resource report hash differs")
    report = json.loads(Path(path).read_bytes())
    if (report.get("schema") != "listwise-semantic-resource-v1"
            or report.get("completed") is not True or report.get("passed") is not True
            or report.get("recipe") != RECIPE
            or report.get("input_manifest_sha256") != sha(ROOT / ".omx/artifacts/listwise-semantic-input-v1/manifest.json")
            or report.get("input_rows_sha256") != INPUT_ROWS_SHA
            or report.get("accuracy_measured") is not False
            or report.get("checkpoint_saved") is not False
            or report.get("experiment_training_run") is not False
            or report.get("probe_weights_discarded") is not True
            or report.get("finite_loss_and_gradients") is not True
            or report.get("dropout_disabled") is not True
            or report.get("alternating_dummy_supervision") != ["candidate_0", "NONE"]):
        raise ValueError("resource report contract differs")
    for name, maximum in (("projected_total_seconds", RECIPE["max_seconds"]),
                          ("peak_rss_bytes", RECIPE["max_rss_bytes"])):
        value = report.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError("resource report exceeds bounds")
    verify_pins(report["source_hashes"])
    return report


def restore_fresh_checkpoint(checkpoint, factory=None):
    """Reconstruct a delta over a fresh upstream model, never the trained object."""
    factory = configure_model if factory is None else factory
    model, tokenizer, optimizer, _, _ = factory()
    del optimizer
    if parameter_digest(model, trainable=False) != checkpoint["frozen_parameters_sha256"]:
        raise ValueError("fresh upstream frozen parameters differ")
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(checkpoint["state_dict"]) != trainable_names:
        raise ValueError("checkpoint trainable parameter closure differs")
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if (incompatible.unexpected_keys
            or set(incompatible.missing_keys) != set(model.state_dict()) - trainable_names):
        raise ValueError("partial checkpoint key closure differs")
    model.eval()
    return model, tokenizer


def score_row(model, tokenizer, row):
    encoded = encode_row(tokenizer, row)
    with torch.no_grad():
        scores = model(**encoded).logits.reshape(-1).tolist()
    if len(scores) != len(row["candidates"]):
        raise ValueError("missing candidate score")
    return scores


def prediction(row, scores):
    selected = decision(scores, row["current_index"])
    return {
        "row_index": row["index"], "id": row["id"],
        "source_role": row["source_role"], "fold": row["fold"],
        "component": row["component"], "source_indices": row["source_indices"],
        "scores": scores, "none_score": 0.0, **selected,
        "current_text": row["current_result"],
        "chosen_text": row["candidates"][selected["selected_index"]]["text"],
    }


def metrics(rows, predictions, targets):
    if (len(rows) != len(predictions) or set(targets) != set(range(len(rows)))
            or [row["index"] for row in rows] != list(range(len(rows)))):
        raise ValueError("complete denominator and targets required")
    counts = Counter()
    states, roles, folds = {}, {}, {}
    fixes, losses = [], []
    for row, pred in zip(rows, predictions):
        index = row["index"]
        if pred["row_index"] != index or pred["id"] != row["id"]:
            raise ValueError("prediction row order differs")
        expected = targets[index]
        positives = accepted_indices(row, expected)
        state = target_state(row, positives)
        before, after = row["current_result"] in expected, pred["chosen_text"] in expected
        switched = pred["chosen_text"] != row["current_result"]
        none = pred["reason"] == "none"
        values = {"rows": 1, "before_top1": int(before), "top1": int(after),
                  "corrections": int(after and not before),
                  "regressions": int(before and not after),
                  "switches": int(switched), "correct_switches": int(switched and after),
                  "abstentions": int(pred["abstained"]),
                  "none_predictions": int(none),
                  "none_correct": int(none and not positives),
                  "candidate_set_covered": int(bool(positives))}
        counts.update(values)
        for mapping, key in ((states, state), (roles, row["source_role"]), (folds, str(row["fold"]))):
            mapping.setdefault(key, Counter()).update(values)
        if after and not before:
            fixes.append(row["id"])
        if before and not after:
            losses.append(row["id"])
    return {
        "counts": dict(counts), "states": states, "roles": roles, "folds": folds,
        "correction_ids": fixes, "regression_ids": losses,
        "switch_precision": counts["correct_switches"] / counts["switches"] if counts["switches"] else None,
        "mechanism_gate_passed": roles.get("natural_fit", {}).get("corrections", 0) >= 1
                                 and counts["regressions"] == 0,
    }


def train(output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    started = time.monotonic()
    manifest, rows, pins = load_inputs()
    resource = validate_resource_report()
    pins.update(resource["source_hashes"])
    for path in (RESOURCE, PLAN, Path(__file__).resolve()):
        pins[str(path.relative_to(ROOT))] = sha(path)
    verify_pins(pins)
    output.mkdir(parents=True)
    write_json(output / "configuration.json", {"recipe": RECIPE, "source_hashes": pins,
                                               "input_manifest": manifest})

    def log(event, **values):
        value = {"event": event, **resources(started), **values}
        with (output / "progress.jsonl").open("ab") as stream:
            stream.write(canonical(value) + b"\n")
        print(json.dumps(value, ensure_ascii=False), flush=True)

    all_predictions = [None] * len(rows)
    records = []
    for fold in range(RECIPE["folds"]):
        train_indices, held_indices = fold_indices(rows, fold)
        # Only this fold's training targets are opened before optimization.
        targets = read_targets(train_indices, rows)
        positives = {index: accepted_indices(rows[index], targets[index]) for index in train_indices}
        training_states = Counter(target_state(rows[index], positives[index]) for index in train_indices)
        if not training_states["ALT_ONLY"] or not training_states["NONE"]:
            raise ValueError("fold lacks correction or NONE supervision")
        model, tokenizer, optimizer, layer_info, loading = configure_model()
        log("fold_start", fold=fold, train_rows=len(train_indices), held_rows=len(held_indices),
            training_states=dict(training_states), layer_info=layer_info)
        frozen_before = parameter_digest(model, trainable=False)
        mutable_before = parameter_digest(model, trainable=True)
        order = list(train_indices)
        random.Random(RECIPE["seed"] + fold).shuffle(order)
        losses = []
        # Dropout stays disabled, while the selected encoder layer and head have gradients.
        model.eval()
        for step, index in enumerate(order):
            row = rows[index]
            encoded = encode_row(tokenizer, row)
            optimizer.zero_grad(set_to_none=True)
            scores = model(**encoded).logits.reshape(-1)
            loss = listwise_none_loss(scores, [len(row["candidates"])], [positives[index]])
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("nonfinite listwise gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            resources(started)
            if (step + 1) % 128 == 0:
                log("training", fold=fold, updates=step + 1, training_rows=len(order),
                    mean_recent_loss=sum(losses[-128:]) / len(losses[-128:]))
        if frozen_before != parameter_digest(model, trainable=False):
            raise ValueError("frozen encoder parameters changed")
        mutable_after = parameter_digest(model, trainable=True)
        if mutable_after == mutable_before:
            raise ValueError("trainable parameters did not change")
        trainable = {name: parameter.detach().cpu().clone()
                     for name, parameter in model.named_parameters() if parameter.requires_grad}
        checkpoint = output / f"fold-{fold}.pt"
        torch.save({"fold": fold, "state_dict": trainable,
                    "train_source_indices": train_indices, "held_source_indices": held_indices,
                    "training_order": order, "frozen_parameters_sha256": frozen_before}, checkpoint)
        before_save_load = score_row(model, tokenizer, rows[held_indices[0]])
        # Release the trained model before constructing the independent fresh base.
        del model, tokenizer, optimizer, trainable, scores, loss, encoded
        import gc
        gc.collect()
        restored = torch.load(checkpoint, weights_only=True, map_location="cpu")
        model, tokenizer = restore_fresh_checkpoint(restored)
        resources(started)
        if before_save_load != score_row(model, tokenizer, rows[held_indices[0]]):
            raise ValueError("saved checkpoint scores differ")
        for position, index in enumerate(held_indices):
            all_predictions[index] = prediction(rows[index], score_row(model, tokenizer, rows[index]))
            resources(started)
            if (position + 1) % 128 == 0:
                log("held_prediction", fold=fold, completed=position + 1, rows=len(held_indices))
        record = {"fold": fold, "train_source_indices": train_indices,
                  "held_source_indices": held_indices, "training_order": order,
                  "training_states": dict(training_states), "optimizer_updates": len(order),
                  "mean_training_loss": sum(losses) / len(losses),
                  "initial_uniform_mean_loss": sum(math.log(len(rows[index]["candidates"]) + 1)
                        - (math.log(len(positives[index])) if positives[index] else 0)
                        for index in train_indices) / len(train_indices),
                  "trainable_before_sha256": mutable_before, "trainable_after_sha256": mutable_after,
                  "frozen_parameters_sha256": frozen_before, "layer_info": layer_info,
                  "checkpoint_sha256": sha(checkpoint), "loading_info": loading}
        records.append(record)
        write_json(output / "folds.json", records)
        log("fold_complete", fold=fold, optimizer_updates=len(order))
        del model, tokenizer, restored
        gc.collect()

    for row in rows:
        if not row["eligible"]:
            all_predictions[row["index"]] = {
                "row_index": row["index"], "id": row["id"], "source_role": row["source_role"],
                "fold": row["fold"], "component": row["component"], "source_indices": row["source_indices"],
                "scores": [], "none_score": 0.0, "selected_index": row["current_index"],
                "abstained": True, "reason": "length_bypass", "current_text": row["current_result"],
                "chosen_text": row["current_result"],
            }
    if any(pred is None for pred in all_predictions):
        raise ValueError("OOF predictions do not cover every source row")
    path = output / "source-predictions.jsonl"
    with path.open("xb") as stream:
        for pred in all_predictions:
            stream.write(canonical(pred) + b"\n")
    # Aggregate target access happens only after every held prediction is closed.
    result = metrics(rows, all_predictions, read_targets(range(len(rows)), rows))
    verify_pins(pins)
    result.update({"completed": True, "recipe": RECIPE, **resources(started),
                   "source_hashes": pins, "predictions_sha256": sha(path),
                   "folds_sha256": sha(output / "folds.json"), "decision_only": True,
                   "generalization_claim_allowed": False, "deployed": False,
                   "production_changed": False, "promotion_allowed": False})
    write_json(output / "report.json", result)
    print(json.dumps({"completed": True, **result["counts"],
                      "mechanism_gate_passed": result["mechanism_gate_passed"]}), flush=True)
    return result


if __name__ == "__main__":
    train()
