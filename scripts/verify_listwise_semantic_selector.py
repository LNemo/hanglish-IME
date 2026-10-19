"""Independent verifier for the completed listwise semantic selector v1."""
from __future__ import annotations

from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import sys
import time

import torch

from scripts.probe_listwise_semantic_resources import RECIPE, configure_model, encode_row
from scripts.verify_role_free_semantic_head import independent_assign_folds


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / ".omx/artifacts/listwise-semantic-input-v1"
RUN = ROOT / ".omx/artifacts/listwise-semantic-selector-v1"
SUPERVISOR = ROOT / ".omx/artifacts/listwise-semantic-selector-v1-supervisor.json"
PREFLIGHT = ROOT / ".omx/artifacts/listwise-semantic-selector-v1-preflight.json"
OUTPUT = ROOT / ".omx/artifacts/listwise-semantic-selector-v1-independent.json"
NATURAL_CASES = ROOT / "data/natural_context/final-choice-v1/natural_fit/cases.jsonl"
CHALLENGE_CASES = ROOT / "data/natural_context/final-choice-v1/challenge_fit/cases.jsonl"

INPUT_MANIFEST_SHA = "7610422a1b44fe852748c5ef6d87ca5aeb065e7ecc8c0cdf0d71166caf19fdf4"
INPUT_ROWS_SHA = "5379897cd818ce8dce391ab3b8a22d27ba05ff72ad666a67d584ef2a46cd4edc"
RESOURCE_SHA = "f455629f7d4aecc75c1c2a74419199321e6d77ec6e7c85b82ba5ad10c3e7a7d5"
TRAINER_SHA = "dc83a33884db7db5e7ac65611ec60f134502d81d3cec90c65aa0c307dda3c3b2"
PLAN_SHA = "0d6b08917d3c7ed42da621ca4f1da556dc7d4b37a65dc90e617af5423a162e5b"
PREFLIGHT_SHA = "602ad277e1fd1af394bf296ca9a3d490864715b71b634aa04966aaebe7f77350"
PRIOR_VERIFIER_SHA = "b071fdebfee9a02977a9f0d44a37493a7142a3930bbb65d7b7241c8340ffc482"
ROWS = 3492
CANDIDATES = 11169
EXPECTED_STATES = {
    "NONE": 721, "KEEP_ONLY": 2532,
    "ALT_ONLY": 59, "MULTI_WITH_KEEP": 180,
}
MAX_SECONDS = 2700
MAX_SUPERVISOR_SECONDS = 2800
MAX_VERIFY_SECONDS = 900
MAX_RSS = 3 * 1024**3


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines() if line]


def verifier_resources(started):
    elapsed = time.monotonic() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss if sys.platform == "darwin" else rss * 1024
    require(elapsed <= MAX_VERIFY_SECONDS and rss <= MAX_RSS,
            "independent verifier resource cap exceeded")
    return elapsed, rss


def verify_pins(pins):
    require(isinstance(pins, dict) and pins, "source pins missing")
    for name, digest in pins.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        require(path.is_file() and sha(path) == digest, f"source pin differs: {name}")


def parameter_digest(model, *, trainable):
    value = hashlib.sha256()
    found = False
    for name, parameter in model.named_parameters():
        if parameter.requires_grad == trainable:
            found = True
            value.update(name.encode())
            value.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    require(found, "parameter digest selection is empty")
    return value.hexdigest()


def fold_indices(rows, fold):
    train = [row["index"] for row in rows if row["eligible"] and row["fold"] != fold]
    held = [row["index"] for row in rows if row["eligible"] and row["fold"] == fold]
    require(train and held and not set(train) & set(held), "invalid fold membership")
    for key in ("component", "group_id"):
        require(not ({rows[index][key] for index in train}
                     & {rows[index][key] for index in held}),
                f"train/held {key} overlap")
    return train, held


def validate_fold_record(fold, record, checkpoint, rows):
    train, held = fold_indices(rows, fold)
    require(record.get("fold") == checkpoint.get("fold") == fold,
            "fold identity differs")
    require(record.get("train_source_indices") == checkpoint.get("train_source_indices") == train,
            "train source membership differs")
    require(record.get("held_source_indices") == checkpoint.get("held_source_indices") == held,
            "held source membership differs")
    order = record.get("training_order")
    expected_order = list(train)
    random.Random(RECIPE["seed"] + fold).shuffle(expected_order)
    require(order == checkpoint.get("training_order")
            and order == expected_order
            and len(order) == len(train) == record.get("optimizer_updates")
            and len(set(order)) == len(order) and set(order) == set(train),
            "each train row must occur exactly once")
    return train, held


def restore_checkpoint(checkpoint, record, factory=configure_model):
    torch.set_num_threads(RECIPE["threads"])
    model, tokenizer, optimizer, layer_info, loading = factory()
    del optimizer
    frozen_before = parameter_digest(model, trainable=False)
    mutable_before = parameter_digest(model, trainable=True)
    require(frozen_before == checkpoint.get("frozen_parameters_sha256")
            == record.get("frozen_parameters_sha256"),
            "fresh frozen-base digest differs")
    require(mutable_before == record.get("trainable_before_sha256"),
            "fresh trainable initialization differs")
    trainable_names = {name for name, parameter in model.named_parameters()
                       if parameter.requires_grad}
    state = checkpoint.get("state_dict")
    require(isinstance(state, dict) and set(state) == trainable_names,
            "trainable delta key closure differs")
    for name, tensor in state.items():
        parameter = dict(model.named_parameters())[name]
        require(isinstance(tensor, torch.Tensor) and tensor.shape == parameter.shape
                and tensor.dtype == parameter.dtype and bool(torch.isfinite(tensor).all()),
                "trainable delta tensor differs")
    incompatible = model.load_state_dict(state, strict=False)
    require(not incompatible.unexpected_keys
            and set(incompatible.missing_keys) == set(model.state_dict()) - trainable_names,
            "partial delta load closure differs")
    require(parameter_digest(model, trainable=False) == frozen_before,
            "delta changed frozen parameters")
    mutable_after = parameter_digest(model, trainable=True)
    require(mutable_after == record.get("trainable_after_sha256")
            and mutable_after != mutable_before,
            "trainable delta digest differs")
    require(record.get("layer_info") == layer_info
            and record.get("loading_info") == loading,
            "fresh model configuration differs")
    model.eval()
    return model, tokenizer


def choose(scores, current_index):
    require(isinstance(scores, list) and scores
            and all(type(score) in (int, float) and math.isfinite(score)
                    for score in scores), "finite candidate scores required")
    require(type(current_index) is int and 0 <= current_index < len(scores),
            "current candidate index differs")
    maximum = max(scores)
    if maximum <= 0.0:
        return {"selected_index": current_index, "abstained": True, "reason": "none"}
    winners = [index for index, score in enumerate(scores) if score == maximum]
    if len(winners) != 1:
        return {"selected_index": current_index, "abstained": True, "reason": "tie"}
    return {"selected_index": winners[0], "abstained": False, "reason": "candidate"}


def score_row(model, tokenizer, row):
    encoded = encode_row(tokenizer, row)
    with torch.no_grad():
        scores = model(**encoded).logits.reshape(-1).tolist()
    require(len(scores) == len(row["candidates"])
            and all(math.isfinite(score) for score in scores),
            "complete finite candidate scores required")
    return scores


def replay_prediction(row, scores):
    selected = choose(scores, row["current_index"])
    return {
        "row_index": row["index"], "id": row["id"],
        "source_role": row["source_role"], "fold": row["fold"],
        "component": row["component"], "source_indices": row["source_indices"],
        "scores": scores, "none_score": 0.0, **selected,
        "current_text": row["current_result"],
        "chosen_text": row["candidates"][selected["selected_index"]]["text"],
    }


def read_targets(rows):
    targets = {}
    for role, path, offset, count in (
        ("natural_fit", NATURAL_CASES, 0, 2048),
        ("challenge_fit", CHALLENGE_CASES, 2048, 1444),
    ):
        with path.open() as stream:
            for local_index, line in enumerate(stream):
                if local_index >= count:
                    break
                source = json.loads(line)
                index = offset + local_index
                require(rows[index]["source_role"] == role
                        and rows[index]["id"] == source["id"],
                        "target source order differs")
                targets[index] = source["expected"]
    require(set(targets) == set(range(len(rows))), "complete TRAIN targets required")
    return targets


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


def compute_metrics(rows, predictions, targets):
    require(len(rows) == len(predictions) and set(targets) == set(range(len(rows))),
            "complete denominator and targets required")
    counts = Counter()
    states, roles, folds = {}, {}, {}
    correction_ids, regression_ids = [], []
    state_rows = Counter()
    for row, prediction in zip(rows, predictions):
        index = row["index"]
        require(prediction["row_index"] == index and prediction["id"] == row["id"],
                "prediction order differs")
        positives = accepted_indices(row, targets[index])
        state = target_state(row, positives)
        state_rows[state] += 1
        before = row["current_result"] in targets[index]
        after = prediction["chosen_text"] in targets[index]
        switched = prediction["chosen_text"] != row["current_result"]
        none = prediction["reason"] == "none"
        values = {"rows": 1, "before_top1": int(before), "top1": int(after),
                  "corrections": int(after and not before),
                  "regressions": int(before and not after), "switches": int(switched),
                  "correct_switches": int(switched and after),
                  "abstentions": int(prediction["abstained"]),
                  "none_predictions": int(none),
                  "none_correct": int(none and not positives),
                  "candidate_set_covered": int(bool(positives))}
        counts.update(values)
        for mapping, key in ((states, state), (roles, row["source_role"]),
                             (folds, str(row["fold"]))):
            mapping.setdefault(key, Counter()).update(values)
        if after and not before:
            correction_ids.append(row["id"])
        if before and not after:
            regression_ids.append(row["id"])
    result = {
        "counts": dict(counts),
        "states": {key: dict(value) for key, value in states.items()},
        "roles": {key: dict(value) for key, value in roles.items()},
        "folds": {key: dict(value) for key, value in folds.items()},
        "correction_ids": correction_ids, "regression_ids": regression_ids,
        "switch_precision": (counts["correct_switches"] / counts["switches"]
                             if counts["switches"] else None),
        "mechanism_gate_passed": (
            roles.get("natural_fit", {}).get("corrections", 0) >= 1
            and counts["regressions"] == 0),
        "target_state_rows": dict(state_rows),
        "repair_recall": (states.get("ALT_ONLY", {}).get("top1", 0)
                          / state_rows["ALT_ONLY"] if state_rows["ALT_ONLY"] else None),
    }
    return result


def validate_training_evidence(rows, fold_records, targets):
    """Recompute label-dependent fold summaries after predictions are closed."""
    for fold, record in enumerate(fold_records):
        train, _ = fold_indices(rows, fold)
        states = Counter()
        uniform_losses = []
        for index in train:
            positives = accepted_indices(rows[index], targets[index])
            states[target_state(rows[index], positives)] += 1
            uniform_losses.append(
                math.log(len(rows[index]["candidates"]) + 1)
                - (math.log(len(positives)) if positives else 0.0))
        require(record.get("training_states") == dict(states),
                "fold training-state evidence differs")
        require(math.isclose(record.get("initial_uniform_mean_loss", math.nan),
                             sum(uniform_losses) / len(uniform_losses),
                             rel_tol=0.0, abs_tol=1e-12),
                "fold initial-loss evidence differs")
        require(type(record.get("mean_training_loss")) in (int, float)
                and math.isfinite(record["mean_training_loss"])
                and record["mean_training_loss"] >= 0.0,
                "fold training-loss evidence differs")


def verify(run=RUN, supervisor_path=SUPERVISOR, output=OUTPUT,
           factory=configure_model):
    run, supervisor_path, output = Path(run), Path(supervisor_path), Path(output)
    started = time.monotonic()
    peak_rss = 0

    def progress(event, **values):
        nonlocal peak_rss
        elapsed, rss = verifier_resources(started)
        peak_rss = max(peak_rss, rss)
        payload = {"event": event, "elapsed_seconds": elapsed,
                   "peak_rss_bytes": peak_rss, **values}
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    require(not output.exists(), "independent output already exists")
    supervisor = json.loads(supervisor_path.read_bytes())
    require(supervisor.get("exit_code") == 0 and supervisor.get("timed_out") is False,
            "successful supervisor receipt required")
    require(type(supervisor.get("elapsed_seconds")) in (int, float)
            and supervisor["elapsed_seconds"] <= MAX_SUPERVISOR_SECONDS,
            "external supervisor time cap exceeded")
    report_path = run / "report.json"
    require(sha(report_path) == supervisor.get("report_sha256"),
            "supervisor report hash differs")
    report = json.loads(report_path.read_bytes())
    require(report.get("completed") is True, "completed training report required")
    require(report.get("decision_only") is True
            and report.get("generalization_claim_allowed") is False
            and report.get("deployed") is False
            and report.get("production_changed") is False
            and report.get("promotion_allowed") is False,
            "diagnostic scope flags differ")
    require(report.get("recipe") == RECIPE
            and report.get("elapsed_seconds") <= MAX_SECONDS
            and report.get("peak_rss_bytes") <= MAX_RSS,
            "recipe or resource cap differs")

    require(sha(PREFLIGHT) == PREFLIGHT_SHA, "frozen preflight required")
    preflight = json.loads(PREFLIGHT.read_bytes())
    preflight_pins = preflight.get("source_hashes", {})
    require(preflight.get("schema") == "listwise-semantic-selector-preflight-v1"
            and preflight.get("passed") is True
            and preflight.get("rows") == ROWS
            and preflight.get("candidates") == CANDIDATES
            and preflight.get("recipe") == RECIPE
            and preflight_pins.get("scripts/train_listwise_semantic_selector.py") == TRAINER_SHA
            and preflight_pins.get(".omx/plans/listwise-semantic-selector-v1.md") == PLAN_SHA
            and preflight_pins.get(".omx/artifacts/listwise-semantic-resource-v1/report.json")
            == RESOURCE_SHA
            and preflight.get("input_manifest_sha256") == INPUT_MANIFEST_SHA
            and preflight.get("input_rows_sha256") == INPUT_ROWS_SHA,
            "preflight execution lineage differs")
    verify_pins(preflight_pins)
    require(supervisor.get("preflight_sha256") == PREFLIGHT_SHA,
            "supervisor preflight binding differs")
    configuration = json.loads((run / "configuration.json").read_bytes())
    require(configuration.get("recipe") == RECIPE, "configuration recipe differs")
    pins = configuration.get("source_hashes")
    require(pins == report.get("source_hashes"),
            "configuration/report source pins differ")
    require(pins.get("scripts/train_listwise_semantic_selector.py") == TRAINER_SHA
            and pins.get(".omx/plans/listwise-semantic-selector-v1.md") == PLAN_SHA
            and pins.get(".omx/artifacts/listwise-semantic-input-v1/manifest.json")
            == INPUT_MANIFEST_SHA
            and pins.get(".omx/artifacts/listwise-semantic-input-v1/rows.jsonl")
            == INPUT_ROWS_SHA
            and pins.get(".omx/artifacts/listwise-semantic-resource-v1/report.json")
            == RESOURCE_SHA,
            "fixed source lineage differs")
    verify_pins(pins)
    require(sha(ROOT / "scripts/verify_role_free_semantic_head.py") == PRIOR_VERIFIER_SHA,
            "fold replay dependency changed")

    manifest = json.loads((INPUT / "manifest.json").read_bytes())
    require(sha(INPUT / "manifest.json") == INPUT_MANIFEST_SHA
            and sha(INPUT / "rows.jsonl") == INPUT_ROWS_SHA
            and manifest.get("rows") == ROWS and manifest.get("candidates") == CANDIDATES,
            "fixed listwise input differs")
    rows = read_jsonl(INPUT / "rows.jsonl")
    require(len(rows) == ROWS
            and [row.get("index") for row in rows] == list(range(ROWS))
            and all(row.get("eligible") is True for row in rows),
            "complete eligible denominator required")
    independent_assign_folds(rows)
    planned_folds = [{"fold": fold, "train_rows": len(fold_indices(rows, fold)[0]),
                      "held_rows": len(fold_indices(rows, fold)[1])}
                     for fold in range(RECIPE["folds"])]
    require(preflight.get("folds") == planned_folds
            and preflight.get("optimizer_updates_planned")
            == sum(item["train_rows"] for item in planned_folds),
            "preflight fold plan differs")

    folds_path = run / "folds.json"
    predictions_path = run / "source-predictions.jsonl"
    require(sha(folds_path) == report.get("folds_sha256")
            and sha(predictions_path) == report.get("predictions_sha256"),
            "reported artifact hash differs")
    fold_records = json.loads(folds_path.read_bytes())
    require(isinstance(fold_records, list) and len(fold_records) == 4,
            "four completed fold records required")
    stored_predictions = read_jsonl(predictions_path)
    require(len(stored_predictions) == ROWS, "closed all-row predictions required")

    replayed = [None] * ROWS
    for fold in range(4):
        progress("fold_start", fold=fold)
        record = fold_records[fold]
        checkpoint_path = run / f"fold-{fold}.pt"
        require(sha(checkpoint_path) == record.get("checkpoint_sha256"),
                "checkpoint hash differs")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        _, held = validate_fold_record(fold, record, checkpoint, rows)
        model, tokenizer = restore_checkpoint(checkpoint, record, factory)
        for position, index in enumerate(held):
            replayed[index] = replay_prediction(rows[index], score_row(
                model, tokenizer, rows[index]))
            elapsed, rss = verifier_resources(started)
            peak_rss = max(peak_rss, rss)
            if (position + 1) % 128 == 0:
                progress("held_replay", fold=fold, completed=position + 1,
                         held_rows=len(held))
        del model, tokenizer, checkpoint
        gc.collect()
    require(all(prediction is not None for prediction in replayed),
            "OOF replay does not cover every row")
    require(stored_predictions == replayed,
            "independent held score/action replay differs")

    # Aggregate TRAIN targets are opened only after stored and replayed predictions close.
    targets = read_targets(rows)
    validate_training_evidence(rows, fold_records, targets)
    metrics = compute_metrics(rows, replayed, targets)
    require(metrics["target_state_rows"] == EXPECTED_STATES,
            "fixed target-state inventory differs")
    for key in ("counts", "states", "roles", "folds", "correction_ids",
                "regression_ids", "switch_precision", "mechanism_gate_passed"):
        require(report.get(key) == metrics[key], f"reported metrics differ: {key}")
    require(report.get("mechanism_gate_passed") == (
        metrics["roles"].get("natural_fit", {}).get("corrections", 0) >= 1
        and metrics["counts"]["regressions"] == 0),
        "fixed gate calculation differs")
    verify_pins(pins)
    progress("verification_complete", rows=ROWS, folds=4)
    elapsed, rss = verifier_resources(started)
    peak_rss = max(peak_rss, rss)

    result = {
        "schema": "listwise-semantic-selector-v1-independent-verification",
        "verified": True, "rows": ROWS, "candidates": CANDIDATES,
        "folds_replayed": 4, "scores_and_actions_replayed": True,
        "targets_opened_after_prediction_replay": True,
        "target_state_rows": metrics["target_state_rows"],
        "counts": metrics["counts"], "state_metrics": metrics["states"],
        "roles": metrics["roles"],
        "fold_metrics": metrics["folds"], "repair_recall": metrics["repair_recall"],
        "switch_precision": metrics["switch_precision"],
        "mechanism_gate_passed": metrics["mechanism_gate_passed"],
        "quality_pass_required_for_verification": False,
        "verifier_elapsed_seconds": elapsed,
        "verifier_peak_rss_bytes": peak_rss,
        "verifier_max_seconds": MAX_VERIFY_SECONDS,
        "verifier_max_rss_bytes": MAX_RSS,
        "generalization_claim_allowed": False, "deployed": False,
        "report_sha256": sha(report_path),
        "predictions_sha256": sha(predictions_path),
        "folds_sha256": sha(folds_path), "source_hashes": pins,
        "verifier_sha256": sha(Path(__file__)),
    }
    output.write_bytes(canonical(result) + b"\n")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False))
