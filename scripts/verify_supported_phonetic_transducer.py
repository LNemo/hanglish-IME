"""Independent accepted-set metrics and checkpoint replay for supported training."""
from __future__ import annotations

from collections import defaultdict
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.phonology import romanize_word

DATA = ROOT / ".omx/artifacts/phonetic-supported-data-v1"
MANIFEST_SHA = "9a7d7a7f2b0fab35ba31c8aae20b28d5af3a5bbfd0d56cb67afb4c0e45a16bba"
PAIR_SHA = "cf3e43f75420c17fbe8e1ec118a29a0f6a71c0e55cedbdbae5d7ef3668c84a84"
HELD_COUNTS = {"lexical_holdout": 2552, "morph_combination_holdout": 763}
EXPECTED_RECIPE = {
    "seed": 20260914, "epochs": 6, "batch_size": 128, "optimizer": "AdamW",
    "learning_rate": .002, "weight_decay": 1e-5, "label_smoothing": .02,
    "gradient_clip_norm": 1., "selection_metric": "morph_combination_holdout_token_cross_entropy",
    "beam_size": 8, "cpu_threads": 2, "interop_threads": 1, "time_limit_seconds": 1800,
    "rss_limit_bytes": 2 * 1024**3, "probe_update_batches": 16, "probe_beam_inputs": 256,
    "projection_multiplier": 1.3, "projection_margin_seconds": 60,
}
EXPECTED_CONFIG = {"embedding_dim": 80, "encoder_hidden": 160, "decoder_hidden": 320,
                   "dropout": .1, "max_roman_length": 48}
# The completed v1 producer expanded checkpoint metadata after its report
# schema key. Preserve that artifact; recognize only its exact closed bytes.
LEGACY_REPORT_SHA = "7e001e0c405fa5afb2714c0db1c769acb9427c736b48061f292f4ccfebd57389"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_training_header(report, report_sha):
    legacy = (report_sha == LEGACY_REPORT_SHA
              and report.get("schema") == "supported-phonetic-transducer-checkpoint-v1")
    require((report.get("schema") == "supported-phonetic-transducer-training-v1" or legacy)
            and report.get("completed") is True and report.get("training_run") is True,
            "training incomplete or report schema unrecognized")
    return legacy


def checked_bytes(path, digest):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, f"hash differs: {path}")
    return raw


def verify_pins(pins):
    require(isinstance(pins, dict) and pins, "source pins missing")
    for name, digest in pins.items():
        path = Path(name)
        checked_bytes(path if path.is_absolute() else ROOT / path, digest)


def load_held_records():
    manifest = json.loads(checked_bytes(DATA / "manifest.json", MANIFEST_SHA))
    require(manifest["pairs_jsonl_sha256"] == PAIR_SHA, "pair identity differs")
    require(manifest["completed"] is True, "data preparation incomplete")
    verify_pins(manifest["source_hashes"])
    rows = [json.loads(line) for line in checked_bytes(DATA / "pairs.jsonl", PAIR_SHA).splitlines()]
    require([row["index"] for row in rows] == list(range(54676)), "pair indices differ")
    held = [row for row in rows if row["split"] in HELD_COUNTS]
    for split, count in HELD_COUNTS.items():
        require(sum(row["split"] == split for row in held) == count, "held counts differ")
    train_romans = {row["roman"] for row in rows if row["split"] == "train"}
    require(not train_romans.intersection(row["roman"] for row in held), "held Roman in train")
    return held


def validate_predictions(records, predictions):
    groups = defaultdict(list)
    for row in records:
        groups[(row["split"], row["roman"])].append(row)
    indexed = {}
    for prediction in predictions:
        key = (prediction["split"], prediction["roman"])
        require(key in groups and key not in indexed, "extra or duplicate prediction")
        require(prediction["pair_indices"] == sorted(row["index"] for row in groups[key]),
                "prediction target membership differs")
        outputs = prediction["outputs"]
        require(isinstance(outputs, list) and len(outputs) <= 8, "beam exceeds eight")
        seen = set()
        previous = float("inf")
        for output in outputs:
            text, score = output["text"], output["logprob"]
            require(isinstance(text, str) and text and all("가" <= c <= "힣" for c in text),
                    "output is not whole Hangul")
            require(text not in seen and romanize_word(text) == key[1], "invalid or duplicate spelling")
            require(type(score) in (int, float) and math.isfinite(score) and score <= previous,
                    "beam scores malformed or unsorted")
            previous = score
            seen.add(text)
        indexed[key] = prediction
    require(set(indexed) == set(groups), "held predictions incomplete")
    return indexed


def subset_metrics(records, indexed):
    accepted = defaultdict(set)
    for row in records:
        accepted[(row["split"], row["roman"])].add(row["surface"])
    counts = {"inputs": len(accepted), "variants": sum(map(len, accepted.values())), "empty": 0}
    for k in (1, 4, 8):
        counts[f"accepted_top{k}"] = 0
        counts[f"variants_top{k}"] = 0
    for key, answers in accepted.items():
        texts = [o["text"] for o in indexed[key]["outputs"]]
        counts["empty"] += not texts
        for k in (1, 4, 8):
            found = answers.intersection(texts[:k])
            counts[f"accepted_top{k}"] += bool(found)
            counts[f"variants_top{k}"] += len(found)
    return counts


def recompute_metrics(records, predictions):
    indexed = validate_predictions(records, predictions)
    metrics = {}
    for split in sorted({row["split"] for row in records}):
        subset = [row for row in records if row["split"] == split]
        strata = {"all": subset,
                  "known_root": [row for row in subset if row["known_family_root"]],
                  "root_unknown": [row for row in subset if not row["known_family_root"]]}
        strata.update({
            "family_has_train": [row for row in subset if row["family_has_train"]],
            "family_without_train": [row for row in subset if not row["family_has_train"]],
        })
        metrics[split] = {name: subset_metrics(values, indexed) for name, values in strata.items()}
    return metrics


def replay_predictions(model, predictions, guard=lambda: None):
    """Regenerate every held input; compare spellings and scores, not just totals."""
    for prediction in predictions:
        guard()
        outputs = model.generate(prediction["roman"], beam_size=8)
        saved = prediction["outputs"]
        require([text for text, _ in outputs] == [o["text"] for o in saved],
                f"checkpoint replay spellings differ: {prediction['roman']}")
        require(all(math.isclose(score, o["logprob"], rel_tol=0, abs_tol=1e-6)
                    for (_, score), o in zip(outputs, saved)), "checkpoint replay scores differ")
    guard()
    return len(predictions)


def compare_reported_metrics(independent, reported):
    names = {"all": "overall", "known_root": "root_known"}
    require(set(independent) == set(reported), "reported split metrics differ")
    for split, strata in independent.items():
        for name, counts in strata.items():
            claimed = reported[split][names.get(name, name)]
            for key in ("inputs", "variants"):
                require(claimed[key] == counts[key], "metric denominator differs")
            require(claimed["empty_outputs"] == counts["empty"], "empty-output count differs")
            for k in (1, 4, 8):
                for count_key, denominator, rate_key in (
                    (f"accepted_top{k}", "inputs", f"top{k}_input_accuracy"),
                    (f"variants_top{k}", "variants", f"top{k}_variant_recall"),
                ):
                    rate = counts[count_key] / counts[denominator] if counts[denominator] else 0.
                    require(math.isclose(rate, claimed[rate_key], rel_tol=0, abs_tol=1e-12),
                            f"reported rate differs: {split}/{name}/{rate_key}")


def validate_probe(probe, contract):
    require(probe["schema"] == "supported-phonetic-transducer-probe-v1"
            and probe["completed"] is True and probe["accepted"] is True
            and probe["contract"] == contract, "probe contract differs")
    require(probe["source_hashes"] == probe["source_hashes_after"] == contract["source_hashes"],
            "probe source drift")
    for key, expected in {"prototype_discarded": True, "training_run": False,
                          "held_evaluation_run": False, "morph_validation_forward_run": True,
                          "lexical_validation_run": False}.items():
        require(probe[key] is expected, f"probe phase differs: {key}")
    require(probe["model_updates_run"] == 16 and probe["beam_inputs_run"] == 256,
            "probe workload incomplete")
    timing, projection = probe["timing"], probe["projection"]
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in timing.values()),
            "probe timing malformed")
    require(projection["train_update_batches"] == 2412
            and projection["morph_selection_full_passes"] == 6
            and projection["held_unique_beam_inputs"] == 3292
            and projection["held_beam_evaluation_passes"] == 1
            and projection["multiplier"] == 1.3 and projection["margin_seconds"] == 60,
            "probe projection workload differs")
    observations = probe["update_observations"]
    require(len(observations) == 16, "probe update measurements incomplete")
    require(all(type(o["padded_cost"]) is int and o["padded_cost"] > 0
                and math.isfinite(o["seconds"]) and o["seconds"] >= 0 for o in observations),
            "probe update cost malformed")
    total_cost = sum(o["padded_cost"] for o in observations)
    update_seconds = sum(o["seconds"] for o in observations)
    require(total_cost == projection["observed_update_padded_cost"]
            and math.isclose(update_seconds, timing["updates_seconds"], abs_tol=1e-8),
            "probe observed update sum differs")
    full_cost = projection["full_training_padded_cost"]
    require(type(full_cost) is int and full_cost >= total_cost, "probe full workload invalid")
    expected = timing["probe_elapsed_seconds"] + 60 + 1.3 * (
        update_seconds / total_cost * full_cost + timing["morph_validation_seconds"] * 6
        + timing["beams_seconds"] / 256 * 3292)
    require(math.isclose(expected, timing["projected_full_train_and_all_held_beam_seconds"], abs_tol=1e-8)
            and expected <= 1800 and 0 < probe["peak_rss_bytes"] <= 2 * 1024**3,
            "probe resource gate failed")


def verify_run(run_dir, report_sha, probe_path, output):
    started = time.monotonic()
    run_dir, output = Path(run_dir), Path(output)
    require(not output.exists(), "verification output already exists")
    report_raw = checked_bytes(run_dir / "report.json", report_sha)
    report = json.loads(report_raw)
    legacy_schema = validate_training_header(report, report_sha)
    require(report["from_scratch"] is True and report["lexical_used_for_training"] is False
            and report["lexical_used_for_selection"] is False, "held use differs")
    require(report["source_hashes"] == report["source_hashes_after"], "training source drift")
    require(report["contract"]["recipe"] == EXPECTED_RECIPE
            and report["contract"]["model_config"] == EXPECTED_CONFIG
            and report["contract"]["source_hashes"] == report["source_hashes"], "frozen recipe differs")
    verify_pins(report["source_hashes"])
    probe_raw = checked_bytes(probe_path, report["probe_report_sha256"])
    probe = json.loads(probe_raw)
    validate_probe(probe, report["contract"])
    require(report["probe"] == {"report_sha256": report["probe_report_sha256"],
                                "accepted": True, "contract": probe["contract"]},
            "nested probe binding differs")
    require(report["predictions_sha256"] == report["held_predictions_sha256"],
            "prediction digest binding differs")
    require(report["updates"] == 2412, "training update count differs")
    history = report["epoch_losses"]
    require([row["epoch"] for row in history] == list(range(1, 7)), "epochs incomplete")
    require(all(math.isfinite(row[k]) for row in history
                for k in ("train_token_ce", "morph_token_ce")), "loss nonfinite")
    best = min(history, key=lambda row: row["morph_token_ce"])
    require(report["best_epoch"] == best["epoch"]
            and report["best_morph_token_ce"] == best["morph_token_ce"], "checkpoint choice differs")
    require(0 <= report["elapsed_seconds"] <= 1800
            and 0 < report["peak_rss_bytes"] <= 2 * 1024**3, "training resource cap exceeded")
    checkpoint = run_dir / "phonetic_transducer.pt"
    checked_bytes(checkpoint, report["checkpoint_sha256"])
    metadata_raw = checked_bytes(run_dir / "phonetic_transducer.json", report["metadata_sha256"])
    metadata = json.loads(metadata_raw)
    for key in ("checkpoint_sha256", "contract", "best_epoch", "metrics", "held_predictions_sha256"):
        require(metadata[key] == report[key], f"checkpoint metadata differs: {key}")
    predictions_raw = checked_bytes(run_dir / "held_predictions.jsonl", report["predictions_sha256"])
    predictions = [json.loads(line) for line in predictions_raw.splitlines()]
    held = load_held_records()
    metrics = recompute_metrics(held, predictions)
    require(len(predictions) == 3292, "unique held input count differs")
    compare_reported_metrics(metrics, report["metrics"])

    def guard():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = peak if sys.platform == "darwin" else peak * 1024
        require(time.monotonic() - started <= 600 and peak <= 2 * 1024**3,
                "independent replay resource cap exceeded")
        return peak

    import torch
    from hanglish.models.phonetic_transducer import PhoneticTransducer
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = PhoneticTransducer.load(checkpoint, device="cpu")
    checked_bytes(checkpoint, report["checkpoint_sha256"])
    checked_bytes(run_dir / "phonetic_transducer.json", report["metadata_sha256"])
    require(asdict(model.config) == EXPECTED_CONFIG, "restored model config differs")
    replayed = replay_predictions(model, predictions, guard)
    verify_pins(report["source_hashes"])
    checked_bytes(run_dir / "report.json", report_sha)
    checked_bytes(checkpoint, report["checkpoint_sha256"])
    checked_bytes(run_dir / "held_predictions.jsonl", report["predictions_sha256"])
    require((run_dir / "phonetic_transducer.json").read_bytes() == metadata_raw, "metadata changed")
    result = {
        "schema": "supported-phonetic-transducer-independent-v1", "completed": True,
        "report_sha256": report_sha, "checkpoint_sha256": report["checkpoint_sha256"],
        "legacy_report_schema_collision": legacy_schema,
        "predictions_sha256": report["predictions_sha256"], "fresh_replayed_inputs": replayed,
        "metrics": metrics, "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": guard(), "production_changed": False,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": ["Morphology is development data used for checkpoint selection.",
                        "Root-unknown families do not prove linguistic family separation.",
                        "Checkpoint replay verifies execution, not sentence accuracy."],
    }
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-sha", required=True)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_run(args.run_dir, args.report_sha, args.probe_report, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
