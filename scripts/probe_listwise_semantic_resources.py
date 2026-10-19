"""Resource-only qualification for an explicit-NONE listwise scorer."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace
import time
import gc

import torch

from scripts.listwise_semantic_contract import listwise_none_loss
from scripts.probe_roberta_ranker_resources import (
    load_cross_encoder, sha256, validate_source,
)
from scripts.train_natural_sentence_ranker import rss_bytes
from scripts.train_sentence_ranker import _optimizer, _set_trainable_layers


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / ".omx/artifacts/listwise-semantic-input-v1"
OUTPUT = ROOT / ".omx/artifacts/listwise-semantic-resource-v1"
INPUT_MANIFEST_SHA = "7610422a1b44fe852748c5ef6d87ca5aeb065e7ecc8c0cdf0d71166caf19fdf4"
INPUT_ROWS_SHA = "5379897cd818ce8dce391ab3b8a22d27ba05ff72ad666a67d584ef2a46cd4edc"
EXPECTED_ROWS = 3492
EXPECTED_CANDIDATES = 11169
PROBE_ROWS = 16
MAX_PROBE_SECONDS = 240

RECIPE = {
    "seed": 23, "folds": 4, "epochs": 1, "top_layers": 1,
    "head_lr": 5e-5, "encoder_lr": 1e-5, "weight_decay": 0.01,
    "max_tokens": 512, "threads": 2, "group_batch": 1,
    "max_seconds": 2700, "max_rss_bytes": 3 * 1024**3,
    "dropout_disabled": True,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_pins(pins):
    for name, expected in pins.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        require(path.is_file() and sha256(path) == expected,
                f"resource source changed: {name}")


def load_inputs():
    require(sha256(INPUT / "manifest.json") == INPUT_MANIFEST_SHA,
            "listwise input manifest changed")
    manifest = json.loads((INPUT / "manifest.json").read_bytes())
    require(manifest.get("schema") == "listwise-semantic-training-input-v1"
            and manifest.get("completed") is True
            and manifest.get("rows") == EXPECTED_ROWS
            and manifest.get("eligible_rows") == EXPECTED_ROWS
            and manifest.get("candidates") == EXPECTED_CANDIDATES
            and manifest.get("eligible_candidates") == EXPECTED_CANDIDATES
            and manifest.get("candidate_count_histogram")
            == {"1": 328, "2": 637, "3": 541, "4": 1986}
            and manifest.get("maximum_tokens", 513) <= RECIPE["max_tokens"]
            and manifest.get("labels_or_expected_saved") is False
            and manifest.get("candidate_selection_uses_labels") is False,
            "listwise input contract differs")
    require(sha256(INPUT / "rows.jsonl") == INPUT_ROWS_SHA
            and manifest.get("rows_sha256") == INPUT_ROWS_SHA,
            "listwise input rows changed")
    pins = dict(manifest.get("source_hashes", {}))
    pins.update({
        str((INPUT / "manifest.json").relative_to(ROOT)): INPUT_MANIFEST_SHA,
        str((INPUT / "rows.jsonl").relative_to(ROOT)): INPUT_ROWS_SHA,
    })
    verify_pins(pins)
    rows = [json.loads(line) for line in (INPUT / "rows.jsonl").read_bytes().splitlines()
            if line]
    validate_rows(rows, expected_rows=EXPECTED_ROWS)
    return manifest, rows, pins


def validate_rows(rows, *, expected_rows):
    require(len(rows) == expected_rows, "complete listwise rows required")
    for index, row in enumerate(rows):
        count = len(row.get("candidates", []))
        lengths = row.get("token_lengths")
        require(row.get("index") == index and row.get("eligible") is True,
                "eligible ordered row required")
        require(1 <= count <= 4 and len(row.get("neural_views", [])) == count
                and isinstance(lengths, list) and len(lengths) == count
                and all(type(length) is int and 0 < length <= RECIPE["max_tokens"]
                        for length in lengths),
                "bounded complete candidate group required")
        require(row.get("current_index") == 0
                and len(row.get("source_indices", [])) == count,
                "current/source candidate mapping differs")
    return True


def workload(row):
    return len(row["candidates"]) * max(row["token_lengths"]) ** 2


def quantile_sample(rows, count=PROBE_ROWS):
    eligible = [row for row in rows if row.get("eligible")]
    require(eligible and type(count) is int and count > 1, "eligible quantile rows required")
    ordered = sorted(eligible, key=lambda row: (workload(row), row["index"]))
    positions = sorted(set(round(i * (len(ordered) - 1) / (count - 1))
                           for i in range(count)))
    require(len(positions) == count, "insufficient rows for resource quantiles")
    return [ordered[position] for position in positions]


def configure_model():
    torch.set_num_threads(RECIPE["threads"])
    torch.manual_seed(RECIPE["seed"])
    model, tokenizer, loading = load_cross_encoder()
    layer_info = _set_trainable_layers(model, RECIPE["top_layers"])
    output = getattr(getattr(model, "classifier", None), "out_proj", None)
    require(output is not None and output.bias is not None,
            "RoBERTa scalar classifier output is required")
    with torch.no_grad():
        output.weight.zero_()
        output.bias.zero_()
    require(int(output.weight.count_nonzero()) == 0
            and int(output.bias.count_nonzero()) == 0,
            "classifier output must initialize to NONE-equivalent zero")
    model.cpu()
    model.eval()
    args = SimpleNamespace(
        head_lr=RECIPE["head_lr"], encoder_lr=RECIPE["encoder_lr"],
        weight_decay=RECIPE["weight_decay"],
    )
    optimizer = _optimizer(model, args, False)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups
                     for parameter in group["params"]}
    trainable_ids = {id(parameter) for parameter in model.parameters()
                     if parameter.requires_grad}
    require(trainable_ids and trainable_ids == optimizer_ids,
            "trainable layer/optimizer configuration differs")
    return model, tokenizer, optimizer, layer_info, loading


def encode_row(tokenizer, row):
    encoded = tokenizer(
        row["neural_views"], truncation=False, padding=True,
        add_special_tokens=True, return_tensors="pt",
    )
    require("attention_mask" in encoded, "tokenizer attention mask required")
    lengths = encoded["attention_mask"].sum(dim=1).tolist()
    require(lengths == row["token_lengths"] and max(lengths) <= RECIPE["max_tokens"],
            "runtime tokens differ from prepared full group")
    require(encoded["input_ids"].shape[0] == len(row["candidates"]) <= 4,
            "complete single-row candidate group required")
    return encoded


def projected_seconds(mean_train_step, mean_inference_step, eligible_rows):
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0
                for value in (mean_train_step, mean_inference_step))
            and type(eligible_rows) is int and eligible_rows > 0,
            "finite projection inputs required")
    train_steps = (RECIPE["folds"] - 1) * RECIPE["epochs"] * eligible_rows
    inference_steps = eligible_rows
    return 1.5 * (mean_train_step * train_steps
                  + mean_inference_step * inference_steps) + 120.0


def record_probe_measurements(report, train_seconds, inference_seconds, losses,
                              peak_rss_bytes, elapsed_seconds, eligible_rows):
    mean_train = statistics.fmean(train_seconds)
    mean_inference = statistics.fmean(inference_seconds)
    projected = projected_seconds(mean_train, mean_inference, eligible_rows)
    report.update({
        "train_step_seconds": list(train_seconds),
        "inference_step_seconds": list(inference_seconds),
        "mean_train_step_seconds": mean_train,
        "mean_inference_step_seconds": mean_inference,
        "projection_train_steps": 3 * eligible_rows,
        "projection_inference_steps": eligible_rows,
        "projected_total_seconds": projected,
        "probe_elapsed_seconds": elapsed_seconds,
        "peak_rss_bytes": peak_rss_bytes,
        "losses": list(losses),
        "finite_loss_and_gradients": True,
    })
    require(projected <= RECIPE["max_seconds"],
            "projected listwise experiment exceeds fixed time budget")
    return projected


def _check_resources(started):
    peak = rss_bytes()
    if peak > RECIPE["max_rss_bytes"] or time.monotonic() - started > MAX_PROBE_SECONDS:
        raise RuntimeError("listwise resource probe cap exceeded")
    return peak


def run(output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    report_path = output / "report.json"
    started = time.monotonic()
    report = {
        "schema": "listwise-semantic-resource-v1", "completed": False,
        "passed": False, "accuracy_measured": False, "checkpoint_saved": False,
        "probe_weights_discarded": False, "experiment_training_run": False,
        "recipe": RECIPE, "input_manifest_sha256": INPUT_MANIFEST_SHA,
        "input_rows_sha256": INPUT_ROWS_SHA,
    }
    model = optimizer = None
    try:
        manifest, rows, pins = load_inputs()
        dependencies = (
            Path(__file__), ROOT / "scripts/listwise_semantic_contract.py",
            ROOT / "scripts/probe_roberta_ranker_resources.py",
            ROOT / "scripts/train_natural_sentence_ranker.py",
            ROOT / "scripts/train_sentence_ranker.py",
        )
        pins.update({str(path.relative_to(ROOT)): sha256(path) for path in dependencies})
        source_before = validate_source()
        pins.update({f"data/external/klue-roberta-small/{name}": digest
                     for name, digest in source_before.items()})
        verify_pins(pins)
        sample = quantile_sample(rows)
        report.update({
            "rows": manifest["rows"], "eligible_rows": manifest["eligible_rows"],
            "candidates": manifest["candidates"],
            "sample_row_indices": [row["index"] for row in sample],
            "sample_workloads": [workload(row) for row in sample],
            "source_hashes": dict(sorted(pins.items())), "model_loaded": False,
        })
        model, tokenizer, optimizer, layer_info, loading = configure_model()
        report["model_loaded"] = True
        peak = _check_resources(started)
        train_seconds, inference_seconds, losses = [], [], []
        for position, row in enumerate(sample):
            tick = time.perf_counter()
            encoded = encode_row(tokenizer, row)
            optimizer.zero_grad(set_to_none=True)
            scores = model(**encoded).logits.reshape(-1)
            positives = [[0]] if position % 2 == 0 else [[]]
            loss = listwise_none_loss(scores, [len(row["candidates"])], positives)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                  error_if_nonfinite=True)
            require(bool(torch.isfinite(norm)), "non-finite resource gradients")
            optimizer.step()
            train_seconds.append(time.perf_counter() - tick)
            losses.append(float(loss.detach()))
            peak = max(peak, _check_resources(started))
        model.eval()
        with torch.inference_mode():
            for row in sample:
                tick = time.perf_counter()
                encoded = encode_row(tokenizer, row)
                scores = model(**encoded).logits.reshape(-1)
                require(bool(torch.isfinite(scores).all()), "non-finite inference scores")
                inference_seconds.append(time.perf_counter() - tick)
                peak = max(peak, _check_resources(started))
        projected = record_probe_measurements(
            report, train_seconds, inference_seconds, losses, peak,
            time.monotonic() - started, manifest["eligible_rows"])
        require(validate_source() == source_before, "RoBERTa source changed during probe")
        verify_pins(pins)
        dropout_disabled = not model.training
        del optimizer, model
        optimizer = model = None
        gc.collect()
        report.update({
            "completed": True, "passed": True, "probe_weights_discarded": True,
            "experiment_training_run": False, "resource_probe_updates": len(sample),
            "sample_candidate_counts": [len(row["candidates"]) for row in sample],
            "sample_max_token_lengths": [max(row["token_lengths"]) for row in sample],
            "alternating_dummy_supervision": ["candidate_0", "NONE"],
            "dropout_disabled": dropout_disabled,
            "layer_info": layer_info, "loading_info": loading,
        })
    except Exception as error:
        loaded = model is not None
        if optimizer is not None:
            del optimizer
        if model is not None:
            del model
        optimizer = model = None
        gc.collect()
        if loaded:
            report["probe_weights_discarded"] = True
        report["failure"] = f"{type(error).__name__}: {error}"
        report.setdefault("probe_elapsed_seconds", time.monotonic() - started)
        report["peak_rss_bytes"] = max(report.get("peak_rss_bytes", 0), rss_bytes())
    report_path.write_text(json.dumps(
        report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps({key: report.get(key) for key in (
        "completed", "passed", "projected_total_seconds", "peak_rss_bytes", "failure")},
        ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    outcome = run()
    raise SystemExit(0 if outcome["passed"] else 1)
