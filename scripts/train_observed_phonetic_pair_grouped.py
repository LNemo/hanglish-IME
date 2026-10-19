#!/usr/bin/env python3
"""Grouped execution wrapper for the immutable observed-phonetic pair runner."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.models.batched_phonetic_decode import generate_batched
from scripts import train_observed_phonetic_pair as base


PLAN = ROOT / ".omx/plans/observed-phonetic-pair-execution-v3.md"
BASE_SHA256 = "7073c0276f463cb89a7ef49f148ae0567197deadc900ca822dcb29575989ca87"
BATCHED_SHA256 = "09029b8f2e33dbf0292217cc26ce49305fe40c924791e12ce668ae8e37c45bdd"
PLAN_SHA256 = "827248ca6c4703b0d2de78b427edd638279bb0405d59d78a73be006372496483"
PROBE_INPUTS = 128
MAX_GROUP_SIZE = 32
_BASE_SELECT_BEAM_INPUTS = base.select_beam_inputs


def balanced_groups(values: list, maximum: int = MAX_GROUP_SIZE) -> list[list]:
    """Split ordered values into contiguous balanced groups bounded by maximum."""
    base.require(type(maximum) is int and maximum > 0, "group maximum must be positive")
    if not values:
        return []
    count = math.ceil(len(values) / maximum)
    small, extra = divmod(len(values), count)
    sizes = [small + (index < extra) for index in range(count)]
    groups, offset = [], 0
    for size in sizes:
        groups.append(values[offset:offset + size])
        offset += size
    base.require(offset == len(values) and max(sizes) <= maximum
                 and max(sizes) - min(sizes) <= 1,
                 "balanced group construction differs")
    return groups


def validate_outputs(source: str, outputs) -> list[dict]:
    from hanglish.core.phonology import romanize_word

    base.require(isinstance(outputs, (list, tuple)) and len(outputs) <= 8,
                 "beam output count differs")
    result, seen, previous = [], set(), float("inf")
    for value in outputs:
        base.require(isinstance(value, (list, tuple)) and len(value) == 2,
                     "beam output item malformed")
        text, score = value
        base.require(isinstance(text, str) and text
                     and all("가" <= char <= "힣" for char in text)
                     and text not in seen and romanize_word(text) == source,
                     "beam output is not unique forward-valid Hangul")
        base.require(type(score) in (int, float) and math.isfinite(score)
                     and score <= previous, "beam score malformed or unsorted")
        result.append({"text": text, "score": float(score)})
        seen.add(text); previous = score
    return result


def capture_group(model, group: list[dict], watchdog, journal: Path,
                  operation: str, *, clock=time.monotonic,
                  generator=generate_batched) -> dict:
    """Capture one independent-input group in one bounded durable operation.

    Output rows are ``{id,input,outputs,latency_s}``.  ``id`` defaults to the
    SHA-256 of the input when the source row omits it.  Group ``seconds`` covers
    every generation, validation, JSON serialization, flush, and fsync.
    """
    base.require(isinstance(group, list) and group, "capture group is empty")
    base.require(len(group) <= MAX_GROUP_SIZE, "capture group exceeds maximum")
    normalized, seen = [], set()
    for row in group:
        base.require(isinstance(row, dict), "capture source row malformed")
        source = row.get("input")
        base.require(isinstance(source, str) and source and source not in seen,
                     "capture input invalid or duplicated")
        expected_identity = hashlib.sha256(source.encode()).hexdigest()
        identity = row.get("id", expected_identity)
        base.require(identity == expected_identity, "capture identity differs from input")
        normalized.append({"id": identity, "input": source})
        seen.add(source)
    started = clock()
    captured = []
    with watchdog.operation(operation, update=True):
        for row in normalized:
            input_started = clock()
            outputs = validate_outputs(row["input"], generator(model, row["input"], 8))
            captured.append({**row, "outputs": outputs,
                             "latency_s": clock() - input_started})
        with Path(journal).open("a", encoding="utf-8") as stream:
            for row in captured:
                stream.write(base.canonical(row).decode() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    seconds = clock() - started
    return {
        "inputs": [row["input"] for row in normalized],
        "work_units": sum(base.beam_work_units(len(row["input"])) for row in normalized),
        "seconds": seconds, "count": len(normalized),
    }


def select_probe_inputs(records: list[dict], count: int = PROBE_INPUTS,
                        evaluation_max_length: int | None = None) -> list[dict]:
    return _BASE_SELECT_BEAM_INPUTS(
        records, count=count, evaluation_max_length=evaluation_max_length)


def grouped_projection(update_seconds: float, observed_cost: int, full_cost: int,
                       groups: list[dict], histogram: dict[str, int]) -> dict:
    base.require(type(update_seconds) in (int, float) and math.isfinite(update_seconds)
                 and update_seconds >= 0 and observed_cost > 0 and full_cost > 0,
                 "grouped training projection input invalid")
    base.require(isinstance(groups, list) and groups, "group observations missing")
    base.require(all(type(row.get("seconds")) in (int, float)
                     and math.isfinite(row["seconds"]) and row["seconds"] >= 0
                     and type(row.get("work_units")) is int and row["work_units"] > 0
                     and row.get("count") == len(row.get("inputs", []))
                     for row in groups), "group observation malformed")
    training = update_seconds / observed_cost * full_cost
    maximum_rate = max(row["seconds"] / row["work_units"] for row in groups)
    evaluation_work = sum(base.beam_work_units(int(length)) * amount
                          for length, amount in histogram.items())
    evaluation = maximum_rate * evaluation_work
    total = (base.RECIPE["projection_margin_seconds"]
             + base.RECIPE["projection_multiplier"] * (training + evaluation))
    return {
        "training_seconds": training,
        "maximum_observed_group_seconds_per_work_unit": maximum_rate,
        "evaluation_work_units": evaluation_work,
        "evaluation_seconds": evaluation,
        "projected_total_seconds": total,
    }


def _expected_groups(prepared: dict, arm: str) -> list[list[dict]]:
    return balanced_groups(prepared["arms"][arm]["probe_beam_inputs"])


def validate_initial_arguments(arm: str, initial: Path | None,
                               shared_initial_report: Path | None) -> None:
    base.require(initial is not None,
                 "grouped probe requires the frozen v1 control initial")
    if arm == "augmented":
        base.require(shared_initial_report is not None,
                     "augmented probe requires grouped control report")
    else:
        base.require(shared_initial_report is None,
                     "control probe cannot consume grouped control report")


def validate_grouped_probe(report: dict, completion: dict, arm: str, prepared: dict,
                           prepared_sha: str, pins: dict[str, str],
                           report_sha: str | None = None) -> None:
    positions = prepared["arms"][arm]["probe_update_positions"]
    expected_beams = prepared["arms"][arm]["probe_beam_inputs"]
    expected_groups = _expected_groups(prepared, arm)
    updates = report.get("observations")
    groups = report.get("group_observations")
    updates_valid = (
        isinstance(updates, list) and len(updates) == base.PROBE_UPDATES
        and [row.get("schedule_position") for row in updates] == positions
        and all(row.get("ordinal") == index + 1
                and type(row.get("padded_cost")) is int and row["padded_cost"] > 0
                and type(row.get("seconds")) in (int, float)
                and math.isfinite(row["seconds"]) and row["seconds"] >= 0
                for index, row in enumerate(updates)))
    groups_valid = (
        isinstance(groups, list) and len(groups) == len(expected_groups)
        and all(row.get("inputs") == [value["input"] for value in expected]
                and row.get("count") == len(expected)
                and row.get("work_units") == sum(
                    base.beam_work_units(len(value["input"])) for value in expected)
                and type(row.get("seconds")) in (int, float)
                and math.isfinite(row["seconds"]) and row["seconds"] >= 0
                for row, expected in zip(groups, expected_groups)))
    recomputed = None
    if updates_valid and groups_valid:
        recomputed = grouped_projection(
            sum(row["seconds"] for row in updates),
            sum(row["padded_cost"] for row in updates),
            prepared["arms"][arm]["total_padded_cost"], groups,
            prepared["evaluation"]["input_length_histogram"])
    prediction_path = Path(report.get("probe_predictions_path", ""))
    predictions_valid = False
    if prediction_path.is_file() and base.sha256_file(prediction_path) == report.get(
            "probe_predictions_sha256"):
        rows = base.read_jsonl(prediction_path)
        predictions_valid = (
            len(rows) == len(expected_beams)
            and [row.get("input") for row in rows]
                == [row["input"] for row in expected_beams]
            and all(row.get("id") == source.get(
                "id", hashlib.sha256(source["input"].encode()).hexdigest())
                    and type(row.get("latency_s")) in (int, float)
                    and math.isfinite(row["latency_s"]) and row["latency_s"] >= 0
                    and validate_outputs(row["input"], [
                        (output.get("text"), output.get("score"))
                        for output in row.get("outputs", [])]) == row.get("outputs")
                    for row, source in zip(rows, expected_beams)))
    initial = report.get("initial")
    initial_path = Path(initial.get("path", "")) if isinstance(initial, dict) else Path()
    initial_valid = (
        isinstance(initial, dict)
        and len(initial.get("checkpoint_sha256", "")) == 64
        and len(initial.get("tensor_sha256", "")) == 64
        and type(initial.get("parameter_count")) is int and initial["parameter_count"] > 0
        and initial.get("created_by_this_probe") is False
        and initial_path.is_file()
        and base.sha256_file(initial_path) == initial["checkpoint_sha256"]
        and ((arm == "control" and initial.get("control_probe_report_sha256") is None)
             or (arm == "augmented"
                 and len(initial.get("control_probe_report_sha256", "")) == 64)))
    base.require(
        report.get("schema") == "observed-phonetic-pair-grouped-probe-v1"
        and report.get("completed") is True and report.get("accepted") is True
        and report.get("arm") == arm and report.get("prepared_sha256") == prepared_sha
        and report.get("evaluation_inputs") == prepared["evaluation_inputs"]
        and report.get("recipe") == base.RECIPE and report.get("limits") == base.LIMITS
        and report.get("source_hashes") == pins == report.get("source_hashes_after")
        and report.get("model_updates_run") == base.PROBE_UPDATES
        and report.get("train_beam_inputs_run") == PROBE_INPUTS
        and report.get("beam_inputs_run") == len(expected_beams)
        and report.get("update_selection") == positions
        and report.get("group_selection_sha256")
            == hashlib.sha256(base.canonical(expected_groups)).hexdigest()
        and report.get("group_sizes") == [len(group) for group in expected_groups]
        and updates_valid and groups_valid and predictions_valid
        and report.get("projection") == recomputed
        and recomputed["projected_total_seconds"] <= base.LIMITS["whole_seconds"]
        and report.get("coverage") == {
            "maximum_padded_cost": True, "longest_train_source": True,
            "evaluation_max_length": True, "longest_target_batch_included": True}
        and base.resource_within_limits(report.get("final_resource"))
        and report.get("prototype_discarded") is True
        and report.get("training_run") is False
        and report.get("evaluation_targets_read") is False
        and initial_valid,
        "accepted grouped probe contract differs")
    base.require(
        completion.get("schema") == "observed-phonetic-pair-completion-v1"
        and completion.get("completed") is True
        and (report_sha is None or completion.get("report_sha256") == report_sha)
        and base.resource_within_limits(completion.get("resource_after_report")),
        "grouped probe completion contract differs")


def run_grouped_probe(arm: str, prepared_dir: Path, output: Path,
                      initial: Path | None = None,
                      shared_initial_report: Path | None = None) -> dict:
    started = time.monotonic()
    base.require(arm in base.ARMS, "unknown arm")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    prepared, all_records, pins = base.load_prepared(prepared_dir)
    prepared_sha = base.sha256_file(Path(prepared_dir) / "prepared.json")
    records = all_records[arm]
    schedule = base.cyclic_schedule(len(records))
    positions = prepared["arms"][arm]["probe_update_positions"]
    beams = prepared["arms"][arm]["probe_beam_inputs"]
    groups = balanced_groups(beams)
    output.mkdir(parents=True)
    modules = base.torch_modules(); torch = modules["torch"]
    baseline_swap = modules["system_swap_bytes"]()
    base.configure_torch(torch)
    validate_initial_arguments(arm, initial, shared_initial_report)
    initial_path = Path(initial)
    shared_report_sha = None
    network, _, initial_sha, initial_tensor = base.load_initial(initial_path, modules)
    parameters = sum(value.numel() for value in network.parameters())
    del network
    if shared_initial_report is not None:
        shared = base.read_json(shared_initial_report)
        shared_completion = base.read_json(Path(shared_initial_report).with_name("completion.json"))
        validate_grouped_probe(
            shared, shared_completion, "control", prepared, prepared_sha, pins,
            base.sha256_file(shared_initial_report))
        base.require(shared["initial"]["checkpoint_sha256"] == initial_sha
                     and shared["initial"]["tensor_sha256"] == initial_tensor,
                     "control shared initial differs")
        shared_report_sha = base.sha256_file(shared_initial_report)
    network, config, _, _ = base.load_initial(initial_path, modules, initial_sha, initial_tensor)
    optimizer = torch.optim.AdamW(network.parameters(), lr=base.RECIPE["learning_rate"],
                                  weight_decay=base.RECIPE["weight_decay"])
    observations, losses, gradients = [], [], []
    update_path = output / "updates.jsonl"
    predictions_path = output / "probe-predictions.jsonl"
    with modules["AtomicResourceWatchdog"](
            started, baseline_swap, output / "resources.jsonl", limits=base.LIMITS) as watchdog:
        for ordinal, position in enumerate(positions, 1):
            before = time.monotonic()
            with watchdog.operation(f"probe_update_{ordinal}", update=True):
                loss, gradient = base.one_update(
                    network, optimizer,
                    base.make_batch(records, schedule[position], modules), modules,
                    check_parameters=True)
                base.append_jsonl(update_path, {
                    "ordinal": ordinal, "schedule_position": position,
                    "loss": loss, "gradient_norm": gradient,
                    "padded_cost": base.batch_cost(records, schedule[position]),
                    "pair_indices": schedule[position],
                    "compute_seconds": time.monotonic() - before,
                })
            losses.append(loss); gradients.append(gradient)
            observations.append({
                "ordinal": ordinal, "schedule_position": position,
                "padded_cost": base.batch_cost(records, schedule[position]),
                "seconds": time.monotonic() - before,
            })
        model = modules["PhoneticTransducer"](network, config, "cpu")
        group_observations = [
            capture_group(model, group, watchdog, predictions_path,
                          f"probe_beam_group_{index + 1}")
            for index, group in enumerate(groups)
        ]
        base.verify_pins(pins)
        final_resource = watchdog.sample_and_append("final_before_report")
    projection = grouped_projection(
        sum(row["seconds"] for row in observations),
        sum(row["padded_cost"] for row in observations),
        prepared["arms"][arm]["total_padded_cost"], group_observations,
        prepared["evaluation"]["input_length_histogram"])
    report = {
        "schema": "observed-phonetic-pair-grouped-probe-v1", "completed": True,
        "accepted": (projection["projected_total_seconds"] <= base.LIMITS["whole_seconds"]
                     and base.resource_within_limits(final_resource)),
        "arm": arm, "prepared_sha256": prepared_sha,
        "evaluation_inputs": prepared["evaluation_inputs"],
        "recipe": base.RECIPE, "limits": base.LIMITS,
        "source_hashes": pins, "source_hashes_after": pins,
        "initial": {"path": str(initial_path.resolve()),
                    "checkpoint_sha256": initial_sha, "tensor_sha256": initial_tensor,
                    "parameter_count": parameters, "created_by_this_probe": False,
                    "control_probe_report_sha256": shared_report_sha},
        "model_updates_run": len(observations), "beam_inputs_run": len(beams),
        "train_beam_inputs_run": sum(row["kind"] == "train_length_stratified"
                                     for row in beams),
        "synthetic_beam_inputs_run": sum(row["kind"].startswith("synthetic")
                                         for row in beams),
        "update_selection": positions, "observations": observations,
        "group_selection_sha256": hashlib.sha256(base.canonical(groups)).hexdigest(),
        "group_sizes": [len(group) for group in groups],
        "group_observations": group_observations,
        "probe_predictions_path": str(predictions_path.resolve()),
        "probe_predictions_sha256": base.sha256_file(predictions_path),
        "mean_loss": sum(losses) / len(losses), "max_gradient_norm": max(gradients),
        "coverage": {
            "maximum_padded_cost": max(row["padded_cost"] for row in observations)
                == prepared["arms"][arm]["max_batch_cost"],
            "longest_train_source": max(len(row["input"]) for row in beams
                                        if row["kind"] == "train_length_stratified")
                == max(len(row["roman"]) for row in records),
            "evaluation_max_length": max(len(row["input"]) for row in beams)
                >= prepared["evaluation"]["maximum_input_length"],
            "longest_target_batch_included": max(
                max(len(records[i]["surface"]) for i in schedule[position])
                for position in positions) == max(len(row["surface"]) for row in records),
        },
        "projection": projection, "final_resource": final_resource,
        "prototype_discarded": True, "training_run": False,
        "evaluation_targets_read": False,
    }
    base.finalize_report(output, report, modules, started, baseline_swap)
    return report


def configure() -> None:
    base.require(base.sha256_file(ROOT / "scripts/train_observed_phonetic_pair.py")
                 == BASE_SHA256, "frozen base runner differs")
    base.require(base.sha256_file(ROOT / "hanglish/models/batched_phonetic_decode.py")
                 == BATCHED_SHA256, "frozen batched decoder differs")
    base.require(PLAN.is_file() and base.sha256_file(PLAN) == PLAN_SHA256,
                 "grouped execution plan differs")
    base.PROBE_BEAMS = PROBE_INPUTS
    base.RECIPE = dict(
        base.RECIPE,
        probe_beam_inputs=PROBE_INPUTS,
        decoding_execution="batched-active-beams-v1",
        inference_operation="balanced-contiguous-groups-v1",
        maximum_group_size=MAX_GROUP_SIZE,
    )
    for path in (Path(__file__), ROOT / "hanglish/models/batched_phonetic_decode.py", PLAN):
        base.FIXED_HASHES[path] = base.sha256_file(path)
    original_modules = base.torch_modules

    def modules():
        result = original_modules()

        class BatchedPhoneticTransducer(result["PhoneticTransducer"]):
            def generate(self, roman, beam_size=4):
                return generate_batched(self, roman, beam_size)

        result["PhoneticTransducer"] = BatchedPhoneticTransducer
        return result

    base.select_beam_inputs = select_probe_inputs
    base.torch_modules = modules
    base.validate_probe = validate_grouped_probe
    base.run_probe = run_grouped_probe


if __name__ == "__main__":
    configure()
    base.main()
