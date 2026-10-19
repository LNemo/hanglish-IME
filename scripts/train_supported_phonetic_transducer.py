#!/usr/bin/env python3
"""Probe and train a from-scratch phonetic transducer on reviewed TRAIN-only data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader

from hanglish.models.phonetic_transducer import (
    OUTPUT_TO_ID, PAD, PhoneticTransducer, TransducerConfig, _Network,
    checkpoint_payload,
)
from hanglish.core.phonology import romanize_word
from scripts.train_phonetic_transducer import PairDataset, collate, loss_on


DATA = ROOT / ".omx/artifacts/phonetic-supported-data-v1"
MANIFEST_SHA256 = "9a7d7a7f2b0fab35ba31c8aae20b28d5af3a5bbfd0d56cb67afb4c0e45a16bba"
PAIRS_SHA256 = "cf3e43f75420c17fbe8e1ec118a29a0f6a71c0e55cedbdbae5d7ef3668c84a84"
FAMILIES_SHA256 = "c34c3eddc80efebc5dec22988353f3addbf204d60c3038a4d42b9cfec4f063c9"
PAIR_SET_SHA256 = "5764355e6bfebc1451a961fe6a6d3998902394888e0fec35e0bd23f160828ccf"
EXPECTED_COUNTS = {"train": 51361, "morph_combination_holdout": 763,
                   "lexical_holdout": 2552}
SEED = 20260914
RECIPE = {
    "seed": SEED,
    "epochs": 6,
    "batch_size": 128,
    "optimizer": "AdamW",
    "learning_rate": 0.002,
    "weight_decay": 1e-5,
    "label_smoothing": 0.02,
    "gradient_clip_norm": 1.0,
    "selection_metric": "morph_combination_holdout_token_cross_entropy",
    "beam_size": 8,
    "cpu_threads": 2,
    "interop_threads": 1,
    "time_limit_seconds": 1800,
    "rss_limit_bytes": 2 * 1024**3,
    "probe_update_batches": 16,
    "probe_beam_inputs": 256,
    "projection_multiplier": 1.3,
    "projection_margin_seconds": 60,
}
RUNTIME_SOURCES = (
    ".omx/plans/supported-phonetic-training-v1.md",
    "hanglish/models/phonetic_transducer.py",
    "scripts/train_phonetic_transducer.py",
    "scripts/train_supported_phonetic_transducer.py",
)
INTERPRETATION_LIMITS = (
    "The existing production checkpoint was trained on pairs overlapping these holdouts; a direct comparison is not an independent generalization estimate.",
    "Dictionary and frequency attestation does not prove contextual grammaticality.",
    "Forward-valid means equality with the current deterministic romanize_word implementation; it does not establish linguistic uniqueness or contextual correctness.",
    "Training completion alone does not authorize production adoption.",
)


class ResourceGateError(RuntimeError):
    pass


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_key(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def quantile_indices(size: int, count: int) -> tuple[int, ...]:
    if size <= 0 or count <= 0:
        return ()
    count = min(size, count)
    if count == 1:
        return (size - 1,)
    result = tuple(index * (size - 1) // (count - 1) for index in range(count))
    if len(set(result)) != count:
        raise ValueError("quantile selection is not unique")
    return result


def pair_set_sha(records) -> str:
    return hashlib.sha256("\n".join(
        f"{row['roman']}\t{row['surface']}" for row in records
    ).encode()).hexdigest()


def parse_jsonl(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.splitlines() if line]


def verify_pins(pins: dict[str, str]) -> None:
    if not isinstance(pins, dict) or not pins:
        raise ValueError("source pins are missing")
    for name, expected in pins.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source pin differs: {name}")


def load_supported_data(directory: Path = DATA) -> dict:
    """Read each frozen payload once, hash those bytes, then parse and validate."""
    directory = Path(directory)
    manifest_raw = (directory / "manifest.json").read_bytes()
    pairs_raw = (directory / "pairs.jsonl").read_bytes()
    families_raw = (directory / "families.jsonl").read_bytes()
    if sha256_bytes(manifest_raw) != MANIFEST_SHA256:
        raise ValueError("supported manifest SHA-256 differs")
    if sha256_bytes(pairs_raw) != PAIRS_SHA256:
        raise ValueError("supported pairs SHA-256 differs")
    if sha256_bytes(families_raw) != FAMILIES_SHA256:
        raise ValueError("supported families SHA-256 differs")
    manifest = json.loads(manifest_raw)
    pairs, families = parse_jsonl(pairs_raw), parse_jsonl(families_raw)
    if (manifest.get("schema") != "phonetic-supported-data-v1"
            or manifest.get("completed") is not True
            or manifest.get("pairs") != sum(EXPECTED_COUNTS.values())
            or manifest.get("split_counts") != EXPECTED_COUNTS
            or manifest.get("pair_set_sha256") != PAIR_SET_SHA256
            or manifest.get("pairs_jsonl_sha256") != PAIRS_SHA256
            or manifest.get("families_jsonl_sha256") != FAMILIES_SHA256
            or manifest.get("model_loaded") is not False
            or manifest.get("training_run") is not False):
        raise ValueError("supported manifest contract differs")
    if len(pairs) != sum(EXPECTED_COUNTS.values()) or pair_set_sha(pairs) != PAIR_SET_SHA256:
        raise ValueError("supported pair records differ")
    allowed = set(EXPECTED_COUNTS)
    counts = defaultdict(int)
    roman_splits = defaultdict(set)
    for index, row in enumerate(pairs):
        if (row.get("index") != index or row.get("split") not in allowed
                or not isinstance(row.get("roman"), str) or not row["roman"].isascii()
                or not row["roman"].islower() or not row["roman"].isalpha()
                or not isinstance(row.get("surface"), str) or not row["surface"]
                or any(not "가" <= char <= "힣" for char in row["surface"])
                or not isinstance(row.get("source_flags"), dict)
                or not isinstance(row.get("family_roots"), list)
                or not isinstance(row.get("morphology_signatures"), list)
                or not isinstance(row.get("known_family_root"), bool)
                or not isinstance(row.get("family_has_train"), bool)):
            raise ValueError("supported pair row is malformed")
        counts[row["split"]] += 1
        roman_splits[row["roman"]].add(row["split"])
    if dict(counts) != EXPECTED_COUNTS or any(len(value) != 1 for value in roman_splits.values()):
        raise ValueError("supported pair split closure differs")
    covered = []
    for value in families:
        indices = value.get("pair_indices")
        if (not isinstance(indices, list) or not all(type(index) is int for index in indices)
                or value.get("splits") != sorted(set(pairs[index]["split"] for index in indices))
                or value.get("has_train") != ("train" in value["splits"])
                or value.get("has_held") != any(role != "train" for role in value["splits"])):
            raise ValueError("supported family row is malformed")
        covered.extend(indices)
        if any(pairs[index]["family_sha256"] != value.get("family_sha256") for index in indices):
            raise ValueError("supported family membership differs")
    if sorted(covered) != list(range(len(pairs))) or len(covered) != len(set(covered)):
        raise ValueError("supported family rows do not partition pairs")
    inherited = manifest.get("source_hashes")
    verify_pins(inherited)
    pins = dict(inherited)
    pins.update({
        path_key(directory / "manifest.json"): MANIFEST_SHA256,
        path_key(directory / "pairs.jsonl"): PAIRS_SHA256,
        path_key(directory / "families.jsonl"): FAMILIES_SHA256,
        **{name: sha256_file(ROOT / name) for name in RUNTIME_SOURCES},
    })
    verify_pins(pins)
    return {"manifest": manifest, "pairs": pairs, "families": families,
            "pins": dict(sorted(pins.items()))}


def configure_torch() -> None:
    torch.set_num_threads(RECIPE["cpu_threads"])
    try:
        torch.set_num_interop_threads(RECIPE["interop_threads"])
    except RuntimeError:
        if torch.get_num_interop_threads() != RECIPE["interop_threads"]:
            raise
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def criterion():
    return nn.CrossEntropyLoss(
        ignore_index=OUTPUT_TO_ID[PAD], label_smoothing=RECIPE["label_smoothing"]
    )


def records_as_pairs(records):
    return [(row["roman"], row["surface"]) for row in records]


def padded_batch_cost(records):
    count = len(records)
    source_width = max(len(row["roman"]) + 2 for row in records)
    target_width = max(3 * len(row["surface"]) + 1 for row in records)
    return count * (source_width + 2 * target_width)


def training_schedule(train_records):
    """Return the exact fixed-seed batches used by both probe and training."""
    generator = torch.Generator().manual_seed(SEED)
    schedule = []
    for epoch in range(1, RECIPE["epochs"] + 1):
        permutation = torch.randperm(len(train_records), generator=generator).tolist()
        for batch_index, offset in enumerate(range(0, len(permutation), RECIPE["batch_size"])):
            records = [train_records[index]
                       for index in permutation[offset:offset + RECIPE["batch_size"]]]
            schedule.append({
                "epoch": epoch, "batch": batch_index, "records": records,
                "padded_cost": padded_batch_cost(records),
                "min_roman_length": min(len(row["roman"]) for row in records),
                "max_roman_length": max(len(row["roman"]) for row in records),
            })
    expected = math.ceil(len(train_records) / RECIPE["batch_size"]) * RECIPE["epochs"]
    if len(schedule) != expected:
        raise ValueError("fixed training schedule is incomplete")
    return schedule


def selected_update_batches(train_records, count=RECIPE["probe_update_batches"]):
    schedule = training_schedule(train_records)
    ordered = sorted(schedule, key=lambda item: (
        item["padded_cost"], item["max_roman_length"], item["epoch"], item["batch"]
    ))
    selected = {id(ordered[index]): ordered[index]
                for index in quantile_indices(len(ordered), count)}
    shortest = min(len(row["roman"]) for row in train_records)
    longest = max(len(row["roman"]) for row in train_records)
    mandatory = [ordered[0], ordered[-1]]
    mandatory.append(next(item for item in schedule
                          if any(len(row["roman"]) == shortest for row in item["records"])))
    mandatory.append(next(item for item in schedule
                          if any(len(row["roman"]) == longest for row in item["records"])))
    for item in mandatory:
        selected[id(item)] = item
    ranked = sorted(selected.values(), key=lambda item: (
        item["padded_cost"], item["max_roman_length"], item["epoch"], item["batch"]
    ))
    while len(ranked) > count:
        removable_values = [item for item in ranked[1:-1] if item not in mandatory]
        if not removable_values:
            raise ValueError("requested probe count cannot cover mandatory batch extremes")
        removable = removable_values[0]
        ranked.remove(removable)
    if len(ranked) != count:
        raise ValueError("update probe batch selection differs")
    batches = [collate([PairDataset(records_as_pairs(item["records"]))[index]
                        for index in range(len(item["records"]))]) for item in ranked]
    metadata = [{key: value for key, value in item.items() if key != "records"}
                | {"pair_indices": [row["index"] for row in item["records"]]}
                for item in ranked]
    return batches, metadata, schedule


def selected_beam_inputs(train_records, count=RECIPE["probe_beam_inputs"]):
    unique = sorted({row["roman"] for row in train_records}, key=lambda value: (len(value), value))
    return [unique[index] for index in quantile_indices(len(unique), count)]


def update_batches(network, batches, optimizer, loss_function, guard=None) -> list[float]:
    network.train()
    losses = []
    for source, target_input, target_output in batches:
        optimizer.zero_grad(set_to_none=True)
        logits = network(source, target_input)
        loss = loss_function(logits.reshape(-1, logits.shape[-1]), target_output.reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss is not finite")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(network.parameters(), RECIPE["gradient_clip_norm"],
                                       error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if not math.isfinite(float(norm)):
            raise FloatingPointError("gradient norm is not finite")
        if guard is not None:
            guard()
    return losses


def check_resources(started, clock=time.monotonic, rss_reader=rss_bytes):
    elapsed, peak = clock() - started, rss_reader()
    if elapsed > RECIPE["time_limit_seconds"]:
        raise ResourceGateError("execution exceeded 1800 seconds")
    if peak > RECIPE["rss_limit_bytes"]:
        raise ResourceGateError("execution exceeded 2 GiB RSS")
    return elapsed, peak


def current_contract(pins, config):
    return {"source_hashes": pins, "recipe": RECIPE, "model_config": asdict(config)}


def schedule_identity(schedule):
    return [{"epoch": item["epoch"], "batch": item["batch"],
             "padded_cost": item["padded_cost"],
             "pair_indices": [row["index"] for row in item["records"]]}
            for item in schedule]


def validate_probe_report(report, pins, config, schedule=None, expected_selection=None,
                          expected_beams=None):
    timing = report.get("timing", {})
    projection = report.get("projection", {})
    expected_updates = math.ceil(EXPECTED_COUNTS["train"] / RECIPE["batch_size"]) * RECIPE["epochs"]
    expected_held = 759 + 2533
    finite_timing = all(
        type(timing.get(name)) in (int, float) and math.isfinite(timing[name])
        and timing[name] >= 0
        for name in ("updates_seconds", "morph_validation_seconds", "beams_seconds",
                     "probe_elapsed_seconds", "projected_full_train_and_all_held_beam_seconds")
    )
    observations = report.get("update_observations")
    valid_observations = (
        isinstance(observations, list)
        and len(observations) == RECIPE["probe_update_batches"]
        and all(type(item.get("seconds")) in (int, float)
                and math.isfinite(item["seconds"]) and item["seconds"] >= 0
                and type(item.get("padded_cost")) is int and item["padded_cost"] > 0
                and isinstance(item.get("pair_indices"), list) and item["pair_indices"]
                for item in observations)
    )
    observed_cost = (sum(item["padded_cost"] for item in observations)
                     if valid_observations else None)
    observed_seconds = (sum(item["seconds"] for item in observations)
                        if valid_observations else None)
    full_cost = projection.get("full_training_padded_cost")
    expected_projection = None
    if (finite_timing and valid_observations and type(full_cost) is int and full_cost > 0):
        expected_projection = (
            timing["probe_elapsed_seconds"] + RECIPE["projection_margin_seconds"]
            + RECIPE["projection_multiplier"] * (
                observed_seconds / observed_cost * full_cost
                + timing["morph_validation_seconds"] * RECIPE["epochs"]
                + timing["beams_seconds"] / RECIPE["probe_beam_inputs"] * expected_held
            )
        )
    schedule_matches = True
    if schedule is not None:
        identity = schedule_identity(schedule)
        schedule_matches = (
            full_cost == sum(item["padded_cost"] for item in schedule)
            and projection.get("training_schedule_sha256") == sha256_bytes(canonical(identity))
        )
    selection_matches = True
    if expected_selection is not None:
        observed_selection = [
            {key: value for key, value in item.items() if key != "seconds"}
            for item in observations
        ] if valid_observations else None
        selection_matches = (
            observed_selection == expected_selection
            and report.get("update_selection_sha256")
            == sha256_bytes(canonical(expected_selection))
        )
    beam_selection_matches = (
        expected_beams is None
        or report.get("beam_selection_sha256") == sha256_bytes(canonical(expected_beams))
    )
    if (report.get("schema") != "supported-phonetic-transducer-probe-v1"
            or report.get("completed") is not True or report.get("accepted") is not True
            or report.get("contract") != current_contract(pins, config)
            or report.get("model_updates_run") != RECIPE["probe_update_batches"]
            or report.get("beam_inputs_run") != RECIPE["probe_beam_inputs"]
            or report.get("prototype_discarded") is not True
            or report.get("training_run") is not False
            or report.get("held_evaluation_run") is not False
            or report.get("morph_validation_forward_run") is not True
            or report.get("lexical_validation_run") is not False
            or report.get("source_hashes") != pins or report.get("source_hashes_after") != pins
            or report.get("source_hashes_unchanged") is not True
            or report.get("update_selection_includes_min_and_max_cost") is not True
            or report.get("update_selection_includes_shortest_and_longest_roman") is not True
            or report.get("beam_selection_includes_shortest_and_longest") is not True
            or type(report.get("mean_update_loss")) not in (int, float)
            or not math.isfinite(report["mean_update_loss"])
            or type(report.get("morph_validation_token_ce")) not in (int, float)
            or not math.isfinite(report["morph_validation_token_ce"])
            or not valid_observations or not schedule_matches
            or not selection_matches or not beam_selection_matches
            or report.get("resource") != {
                "time_limit_seconds": RECIPE["time_limit_seconds"],
                "rss_limit_bytes": RECIPE["rss_limit_bytes"],
                "elapsed_seconds": timing.get("probe_elapsed_seconds"),
                "peak_rss_bytes": report.get("peak_rss_bytes"),
            }
            or not finite_timing
            or not math.isclose(timing.get("updates_seconds", float("nan")),
                                observed_seconds if observed_seconds is not None else float("nan"),
                                rel_tol=1e-9, abs_tol=1e-9)
            or not math.isclose(
                timing.get("projected_full_train_and_all_held_beam_seconds", float("nan")),
                expected_projection if expected_projection is not None else float("nan"),
                rel_tol=1e-9, abs_tol=1e-9)
            or timing.get("projected_full_train_and_all_held_beam_seconds")
            > RECIPE["time_limit_seconds"]
            or type(report.get("peak_rss_bytes")) is not int
            or not 0 <= report["peak_rss_bytes"] <= RECIPE["rss_limit_bytes"]
            or projection.get("train_update_batches") != expected_updates
            or projection.get("observed_update_padded_cost") != observed_cost
            or not math.isclose(
                projection.get("update_seconds_per_padded_unit", float("nan")),
                observed_seconds / observed_cost, rel_tol=1e-9, abs_tol=1e-12)
            or projection.get("morph_selection_full_passes") != RECIPE["epochs"]
            or projection.get("held_unique_beam_inputs") != expected_held
            or projection.get("held_beam_evaluation_passes") != 1
            or projection.get("multiplier") != RECIPE["projection_multiplier"]
            or projection.get("margin_seconds") != RECIPE["projection_margin_seconds"]):
        raise ValueError("accepted probe does not match current code/data/config")


def capture_held_predictions(model, records, guard=None):
    groups = defaultdict(list)
    for row in records:
        groups[row["roman"]].append(row)
    predictions = []
    for roman in sorted(groups):
        rows = groups[roman]
        outputs = []
        previous = float("inf")
        seen = set()
        for text, score in model.generate(roman, beam_size=8):
            if (not isinstance(text, str) or not text or text in seen
                    or any(not "가" <= char <= "힣" for char in text)
                    or romanize_word(text) != roman or len(outputs) >= 8
                    or type(score) not in (int, float) or not math.isfinite(score)
                    or score > previous):
                raise ValueError("held beam output is malformed")
            outputs.append({"text": text, "logprob": float(score)})
            seen.add(text)
            previous = score
        splits = {row["split"] for row in rows}
        if len(splits) != 1:
            raise ValueError("held Roman input crosses splits")
        predictions.append({"split": splits.pop(), "roman": roman,
                            "pair_indices": sorted(row["index"] for row in rows),
                            "outputs": outputs})
        if guard is not None:
            guard()
    return predictions


def metrics_from_predictions(predictions, records):
    records_by_index = {row["index"]: row for row in records}

    def selected_variants(predicate):
        selected = []
        for prediction in predictions:
            variants = [records_by_index[index] for index in prediction["pair_indices"]
                        if predicate(records_by_index[index])]
            if variants:
                selected.append((prediction, variants))
        return selected

    def score(selected):
        variant_count = sum(len(variants) for _, variants in selected)
        empty = sum(not prediction["outputs"] for prediction, _ in selected)
        result = {"inputs": len(selected), "variants": variant_count,
                  "empty_outputs": empty,
                  "empty_output_rate": empty / len(selected) if selected else 0.0}
        for cutoff in (1, 4, 8):
            input_hits = variant_hits = 0
            for prediction, variants in selected:
                outputs = {value["text"] for value in prediction["outputs"][:cutoff]}
                hits = sum(value["surface"] in outputs for value in variants)
                input_hits += hits > 0
                variant_hits += hits
            result[f"top{cutoff}_input_accuracy"] = input_hits / len(selected) if selected else 0.0
            result[f"top{cutoff}_variant_recall"] = variant_hits / variant_count if variant_count else 0.0
        return result

    return {
        "overall": score(selected_variants(lambda _: True)),
        "root_known": score(selected_variants(lambda row: row["known_family_root"])),
        "root_unknown": score(selected_variants(lambda row: not row["known_family_root"])),
        "family_has_train": score(selected_variants(lambda row: row["family_has_train"])),
        "family_without_train": score(selected_variants(lambda row: not row["family_has_train"])),
    }


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(value) + b"\n")
    os.replace(temporary, path)


def write_jsonl(path: Path, values) -> str:
    raw = b"".join(canonical(value) + b"\n" for value in values)
    path.write_bytes(raw)
    return sha256_bytes(raw)


def run_probe(output: Path, *, data_dir: Path = DATA, network_factory=_Network,
              clock=time.monotonic, rss_reader=rss_bytes):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = clock()
    try:
        configure_torch()
        data = load_supported_data(data_dir)
        config = TransducerConfig()
        train_records = [row for row in data["pairs"] if row["split"] == "train"]
        batches, selected, schedule = selected_update_batches(train_records)
        if len(batches) != RECIPE["probe_update_batches"]:
            raise ValueError("probe update batch count differs")
        network = network_factory(config).cpu()
        check_resources(started, clock, rss_reader)
        optimizer = torch.optim.AdamW(network.parameters(), lr=RECIPE["learning_rate"],
                                      weight_decay=RECIPE["weight_decay"])
        losses, update_observations = [], []
        for batch, metadata in zip(batches, selected):
            batch_started = clock()
            batch_losses = update_batches(
                network, [batch], optimizer, criterion(),
                guard=lambda: check_resources(started, clock, rss_reader),
            )
            duration = clock() - batch_started
            losses.extend(batch_losses)
            update_observations.append({**metadata, "seconds": duration})
        update_seconds = sum(item["seconds"] for item in update_observations)
        check_resources(started, clock, rss_reader)
        morph_records = [row for row in data["pairs"]
                         if row["split"] == "morph_combination_holdout"]
        morph_loader = DataLoader(PairDataset(records_as_pairs(morph_records)),
                                  batch_size=RECIPE["batch_size"], shuffle=False,
                                  collate_fn=collate)
        validation_started = clock()
        validation_loss = loss_on(network, morph_loader, criterion(), "cpu")
        validation_seconds = clock() - validation_started
        check_resources(started, clock, rss_reader)
        model = PhoneticTransducer(network, config, "cpu")
        beam_inputs = selected_beam_inputs(train_records)
        beam_started = clock()
        for roman in beam_inputs:
            model.generate(roman, beam_size=RECIPE["beam_size"])
            check_resources(started, clock, rss_reader)
        beam_seconds = clock() - beam_started
        elapsed, peak = check_resources(started, clock, rss_reader)
        held_inputs = len({row["roman"] for row in data["pairs"] if row["split"] != "train"})
        train_batches = len(schedule)
        observed_cost = sum(item["padded_cost"] for item in update_observations)
        full_cost = sum(item["padded_cost"] for item in schedule)
        projected_work = (
            update_seconds / observed_cost * full_cost
            + validation_seconds * RECIPE["epochs"]
            + beam_seconds / len(beam_inputs) * held_inputs
        )
        projected = (elapsed + RECIPE["projection_margin_seconds"]
                     + RECIPE["projection_multiplier"] * projected_work)
        accepted = projected <= RECIPE["time_limit_seconds"] and peak <= RECIPE["rss_limit_bytes"]
        report = {
            "schema": "supported-phonetic-transducer-probe-v1", "completed": True,
            "accepted": accepted, "contract": current_contract(data["pins"], config),
            "source_hashes": data["pins"], "source_hashes_after": data["pins"],
            "source_hashes_unchanged": True,
            "model_updates_run": len(batches), "beam_inputs_run": len(beam_inputs),
            "update_selection_sha256": sha256_bytes(canonical(selected)),
            "update_observations": update_observations,
            "update_selection_includes_min_and_max_cost": (
                min(row["padded_cost"] for row in selected)
                == min(row["padded_cost"] for row in schedule)
                and max(row["padded_cost"] for row in selected)
                == max(row["padded_cost"] for row in schedule)
            ),
            "update_selection_includes_shortest_and_longest_roman": (
                min(row["min_roman_length"] for row in selected)
                == min(len(row["roman"]) for row in train_records)
                and max(row["max_roman_length"] for row in selected)
                == max(len(row["roman"]) for row in train_records)
            ),
            "beam_selection_sha256": sha256_bytes(canonical(beam_inputs)),
            "beam_selection_includes_shortest_and_longest": True,
            "mean_update_loss": sum(losses) / len(losses),
            "morph_validation_token_ce": validation_loss,
            "timing": {"updates_seconds": update_seconds,
                       "morph_validation_seconds": validation_seconds,
                       "beams_seconds": beam_seconds,
                       "probe_elapsed_seconds": elapsed,
                       "projected_full_train_and_all_held_beam_seconds": projected},
            "projection": {
                "train_update_batches": train_batches,
                "full_training_padded_cost": full_cost,
                "training_schedule_sha256": sha256_bytes(canonical(schedule_identity(schedule))),
                "observed_update_padded_cost": observed_cost,
                "update_seconds_per_padded_unit": update_seconds / observed_cost,
                "morph_selection_full_passes": RECIPE["epochs"],
                "held_unique_beam_inputs": held_inputs,
                "held_beam_evaluation_passes": 1,
                "multiplier": RECIPE["projection_multiplier"],
                "margin_seconds": RECIPE["projection_margin_seconds"],
            },
            "peak_rss_bytes": peak, "prototype_discarded": True,
            "training_run": False, "held_evaluation_run": False,
            "morph_validation_forward_run": True,
            "lexical_validation_run": False,
            "resource": {"time_limit_seconds": RECIPE["time_limit_seconds"],
                         "rss_limit_bytes": RECIPE["rss_limit_bytes"],
                         "elapsed_seconds": elapsed, "peak_rss_bytes": peak},
        }
        verify_pins(data["pins"])
        report["source_hashes_after"] = data["pins"]
        del model, network, optimizer
        write_json(output / "report.json", report)
        write_json(output / "status.json", {"state": "complete", "accepted": accepted})
        return report
    except Exception as error:
        write_json(output / "failure.json", {
            "schema": "supported-phonetic-transducer-failure-v1", "mode": "probe",
            "error_type": type(error).__name__, "error": str(error),
            "elapsed_seconds": clock() - started, "peak_rss_bytes": rss_reader(),
        })
        raise


def train_epoch(network, loader, optimizer, loss_function, started,
                clock=time.monotonic, rss_reader=rss_bytes):
    network.train()
    total, tokens = 0.0, 0
    for batch in loader:
        losses = update_batches(
            network, [batch], optimizer, loss_function,
            guard=lambda: check_resources(started, clock, rss_reader),
        )
        count = int(batch[2].ne(OUTPUT_TO_ID[PAD]).sum())
        total += losses[0] * count
        tokens += count
        check_resources(started, clock, rss_reader)
    return total / max(tokens, 1)


def run_train(output: Path, probe_report: Path, *, data_dir: Path = DATA,
              network_factory=_Network, clock=time.monotonic, rss_reader=rss_bytes):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = clock()
    staged = output / "phonetic_transducer.staged.pt"
    try:
        write_json(output / "status.json", {"state": "running", "phase": "initializing"})
        configure_torch()
        data = load_supported_data(data_dir)
        config = TransducerConfig()
        probe_raw = Path(probe_report).read_bytes()
        probe = json.loads(probe_raw)
        train_records = [row for row in data["pairs"] if row["split"] == "train"]
        morph_records = [row for row in data["pairs"]
                         if row["split"] == "morph_combination_holdout"]
        lexical_records = [row for row in data["pairs"] if row["split"] == "lexical_holdout"]
        _, expected_selection, schedule = selected_update_batches(train_records)
        expected_beams = selected_beam_inputs(train_records)
        validate_probe_report(
            probe, data["pins"], config, schedule=schedule,
            expected_selection=expected_selection, expected_beams=expected_beams,
        )
        network = network_factory(config).cpu()
        optimizer = torch.optim.AdamW(network.parameters(), lr=RECIPE["learning_rate"],
                                      weight_decay=RECIPE["weight_decay"])
        loss_function = criterion()
        morph_loader = DataLoader(PairDataset(records_as_pairs(morph_records)),
                                  batch_size=RECIPE["batch_size"], shuffle=False,
                                  collate_fn=collate)
        best_loss, best_epoch, best_state, history = float("inf"), None, None, []
        for epoch in range(1, RECIPE["epochs"] + 1):
            epoch_batches = []
            for item in schedule:
                if item["epoch"] != epoch:
                    continue
                dataset = PairDataset(records_as_pairs(item["records"]))
                epoch_batches.append(collate([dataset[index] for index in range(len(dataset))]))
            train_loss = train_epoch(network, epoch_batches, optimizer, loss_function,
                                     started, clock, rss_reader)
            morph_loss = loss_on(network, morph_loader, loss_function, "cpu")
            if not math.isfinite(morph_loss):
                raise FloatingPointError("morph selection loss is not finite")
            history.append({"epoch": epoch, "train_token_ce": train_loss,
                            "morph_token_ce": morph_loss})
            if morph_loss < best_loss:
                best_loss, best_epoch = morph_loss, epoch
                best_state = {name: value.detach().cpu().clone()
                              for name, value in network.state_dict().items()}
            elapsed, peak = check_resources(started, clock, rss_reader)
            write_json(output / "status.json", {
                "state": "running", "phase": "training", "epoch": epoch,
                "elapsed_seconds": elapsed, "peak_rss_bytes": peak,
            })
        if best_state is None:
            raise RuntimeError("no best checkpoint was selected")
        network.load_state_dict(best_state)
        verify_pins(data["pins"])
        torch.save(checkpoint_payload(network, config), staged)
        checkpoint_hash = sha256_file(staged)
        metadata = {
            "format_version": 1, "schema": "supported-phonetic-transducer-checkpoint-v1",
            "checkpoint_sha256": checkpoint_hash, "contract": current_contract(data["pins"], config),
            "probe_report_sha256": sha256_bytes(probe_raw), "best_epoch": best_epoch,
            "best_morph_token_ce": best_loss, "epoch_history": history,
            "parameter_count": sum(value.numel() for value in network.parameters()),
            "from_scratch": True, "lexical_used_for_training": False,
            "lexical_used_for_selection": False,
        }
        write_json(output / "phonetic_transducer.json", metadata)
        os.replace(staged, output / "phonetic_transducer.pt")
        del network, optimizer
        restored = PhoneticTransducer.load(output / "phonetic_transducer.pt", device="cpu")
        guard = lambda: check_resources(started, clock, rss_reader)
        morph_predictions = capture_held_predictions(restored, morph_records, guard=guard)
        lexical_predictions = capture_held_predictions(restored, lexical_records, guard=guard)
        predictions = morph_predictions + lexical_predictions
        del restored
        predictions_sha = write_jsonl(output / "held_predictions.jsonl", predictions)
        metrics = {
            "morph_combination_holdout": metrics_from_predictions(morph_predictions, morph_records),
            "lexical_holdout": metrics_from_predictions(lexical_predictions, lexical_records),
        }
        elapsed, peak = check_resources(started, clock, rss_reader)
        verify_pins(data["pins"])
        metadata.update({"metrics": metrics, "held_predictions_sha256": predictions_sha,
                         "held_prediction_inputs": len(predictions),
                         "elapsed_seconds": elapsed, "peak_rss_bytes": peak,
                         "predictions_from_restored_checkpoint": True})
        write_json(output / "phonetic_transducer.json", metadata)
        PhoneticTransducer.load(output / "phonetic_transducer.pt", device="cpu")
        report = {
            "schema": "supported-phonetic-transducer-training-v1", "completed": True,
            **metadata, "checkpoint": "phonetic_transducer.pt", "training_run": True,
            "all_held_beam8_evaluated": True,
            "source_hashes": data["pins"], "source_hashes_after": data["pins"],
            "source_hashes_unchanged": True,
            "predictions_sha256": predictions_sha, "epoch_losses": history,
            "updates": len(schedule),
            "resource": {"elapsed_seconds": elapsed, "peak_rss_bytes": peak,
                         "time_limit_seconds": RECIPE["time_limit_seconds"],
                         "rss_limit_bytes": RECIPE["rss_limit_bytes"]},
            "probe": {"report_sha256": sha256_bytes(probe_raw), "accepted": True,
                      "contract": probe["contract"]},
            "interpretation_limits": list(INTERPRETATION_LIMITS),
        }
        report["metadata_sha256"] = sha256_file(output / "phonetic_transducer.json")
        write_json(output / "report.json", report)
        check_resources(started, clock, rss_reader)
        write_json(output / "status.json", {"state": "complete", "phase": "complete",
                                             "checkpoint_sha256": checkpoint_hash})
        return report
    except Exception as error:
        staged.unlink(missing_ok=True)
        (output / "phonetic_transducer.pt").unlink(missing_ok=True)
        (output / "phonetic_transducer.json").unlink(missing_ok=True)
        (output / "held_predictions.jsonl").unlink(missing_ok=True)
        (output / "report.json").unlink(missing_ok=True)
        write_json(output / "failure.json", {
            "schema": "supported-phonetic-transducer-failure-v1", "mode": "train",
            "error_type": type(error).__name__, "error": str(error),
            "elapsed_seconds": clock() - started, "peak_rss_bytes": rss_reader(),
        })
        raise


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="mode", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--probe-report", type=Path, required=True)
    args = parser.parse_args()
    result = (run_probe(args.output) if args.mode == "probe"
              else run_train(args.output, args.probe_report))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
