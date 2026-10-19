#!/usr/bin/env python3
"""Prepare, qualify, and train fixed-compute observed-phonetic paired arms.

``prepare`` is deliberately model-free.  ``probe`` and ``train`` are the only
commands that import Torch or construct the transducer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / ".omx/artifacts/phonetic-supported-data-v1"
DELTA = ROOT / ".omx/artifacts/observed-phonetic-delta-v1"
PLAN = ROOT / ".omx/plans/observed-phonetic-pair-v1.md"
EVALUATION_MANIFEST = ROOT / ".omx/artifacts/document-phonetic-inputs-v1/manifest.json"
ARMS = ("control", "augmented")
SEED = 20260921
UPDATES = 3954
BATCH_SIZE = 128
PRESENTATIONS = UPDATES * BATCH_SIZE
PROBE_UPDATES = 16
PROBE_BEAMS = 32
LIMITS = {
    "whole_seconds": 1800.0, "update_seconds": 60.0,
    "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2,
}
RECIPE = {
    "seed": SEED, "updates": UPDATES, "batch_size": BATCH_SIZE,
    "presentations": PRESENTATIONS, "optimizer": "AdamW",
    "learning_rate": 0.002, "weight_decay": 1e-5,
    "label_smoothing": 0.02, "gradient_clip_norm": 1.0,
    "beam_size": 8, "cpu_threads": 2, "interop_threads": 1,
    "deterministic_algorithms": True, "dtype": "float32",
    "projection_multiplier": 1.3, "projection_margin_seconds": 60.0,
    "probe_updates": PROBE_UPDATES, "probe_beam_inputs": PROBE_BEAMS,
}
CONFIG = {
    "embedding_dim": 80, "encoder_hidden": 160, "decoder_hidden": 320,
    "dropout": 0.1, "max_roman_length": 48,
}
FIXED_HASHES = {
    BASE / "manifest.json": "9a7d7a7f2b0fab35ba31c8aae20b28d5af3a5bbfd0d56cb67afb4c0e45a16bba",
    BASE / "pairs.jsonl": "cf3e43f75420c17fbe8e1ec118a29a0f6a71c0e55cedbdbae5d7ef3668c84a84",
    DELTA / "manifest.json": "f90e21a92bb1b3ebed4ff59aeaab69d5ed54f24c59aafbc14cdd6e67df5991ef",
    DELTA / "delta.jsonl": "bc299116e618ef37c044d8ac0f3061a55e46efdc1b4f1d0c8d9dc1b8ab655e1a",
    DELTA / "quarantine.jsonl": "aebea22744a9e1567fa55648f8c7baa1ac00034eaf6bd72a7602cefa46fdea07",
    DELTA / "independent-verification.json": "f49017edd17e89adcf4d2a3fc8319241b325eead388fe9b1c1cc6217a8643c17",
    PLAN: "779fa7685401e24461545a96a01b90eb0f184af1590e9e8dd0ab4e00a9170a94",
    EVALUATION_MANIFEST: "da5ed2f782ca5272627dc01b90312cc5e61fba3272aba8cd4a0515784bc6fd87",
    ROOT / "hanglish/models/phonetic_transducer.py": "dcdfdf4758e922eec35d67f8926e1d5597950a4751c214fe917d651dbd0e2b05",
    ROOT / "scripts/train_phonetic_transducer.py": "6ad878e7ee9f1ec7f85b71ebbcde86fb2def8035ae287bd03505b51fb1c8772a",
    ROOT / "scripts/source_span_resources.py": "4bdf5f598677d1d5c94bcca39545015e7ae12370b0a91482aca1ba42e11a75bd",
    ROOT / "scripts/preflight_pko_seq2seq.py": "89bc4bd2a9875ace9eb010ca443168df4db84ed34772342dd55923e2ab7518f2",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_key(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    result = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line:
            value = json.loads(line)
            require(isinstance(value, dict), f"invalid JSONL row {number}: {path}")
            result.append(value)
    return result


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(value) + b"\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical(value).decode() + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_pins(pins: dict[str, str]) -> None:
    require(isinstance(pins, dict) and pins, "source pins missing")
    for name, wanted in pins.items():
        path = ROOT / name
        require(path.is_file() and sha256_file(path) == wanted,
                f"source pin differs: {name}")


def pair_cost(record: dict) -> int:
    return len(record["roman"]) + 2 + 2 * (3 * len(record["surface"]) + 1)


def batch_cost(records: list[dict], indices: list[int]) -> int:
    selected = [records[index] for index in indices]
    return len(indices) * (
        max(len(row["roman"]) + 2 for row in selected)
        + 2 * max(3 * len(row["surface"]) + 1 for row in selected)
    )


def cyclic_schedule(size: int, *, seed: int = SEED,
                    updates: int = UPDATES, batch_size: int = BATCH_SIZE) -> list[list[int]]:
    require(type(size) is int and size > 0, "schedule size must be positive")
    require(type(updates) is int and updates > 0, "updates must be positive")
    require(type(batch_size) is int and batch_size > 0, "batch size must be positive")
    rng = random.Random(seed)
    result, pending = [], []
    while len(result) < updates:
        permutation = list(range(size))
        rng.shuffle(permutation)
        pending.extend(permutation)
        while len(pending) >= batch_size and len(result) < updates:
            result.append(pending[:batch_size])
            del pending[:batch_size]
    require(len(result) == updates and all(len(x) == batch_size for x in result),
            "cyclic schedule is incomplete")
    return result


def schedule_summary(records: list[dict], schedule: list[list[int]]) -> dict:
    exposure = Counter(index for batch in schedule for index in batch)
    require(set(exposure) == set(range(len(records))), "schedule omits a pair")
    costs = [batch_cost(records, batch) for batch in schedule]
    return {
        "pairs": len(records), "updates": len(schedule),
        "presentations": sum(map(len, schedule)),
        "schedule_sha256": hashlib.sha256(canonical(schedule)).hexdigest(),
        "exposure_histogram": {str(key): value for key, value in sorted(
            Counter(exposure.values()).items())},
        "min_exposure": min(exposure.values()), "max_exposure": max(exposure.values()),
        "total_padded_cost": sum(costs), "min_batch_cost": min(costs),
        "max_batch_cost": max(costs),
    }


def arm_contract(records: list[dict]) -> dict:
    schedule = cyclic_schedule(len(records))
    result = schedule_summary(records, schedule)
    result["probe_update_positions"] = select_probe_batches(records, schedule)
    result["probe_beam_inputs"] = select_beam_inputs(
        records, evaluation_max_length=load_evaluation_contract()["maximum_input_length"])
    return result


def quantile_indices(size: int, count: int) -> list[int]:
    require(size >= count > 1, "quantile request invalid")
    return [index * (size - 1) // (count - 1) for index in range(count)]


def select_probe_batches(records: list[dict], schedule: list[list[int]],
                         count: int = PROBE_UPDATES) -> list[int]:
    metadata = [{
        "position": position, "cost": batch_cost(records, batch),
        "max_source": max(len(records[i]["roman"]) for i in batch),
        "max_target": max(len(records[i]["surface"]) for i in batch),
    } for position, batch in enumerate(schedule)]
    by_cost = sorted(metadata, key=lambda row: (row["cost"], row["position"]))
    chosen = {by_cost[index]["position"] for index in quantile_indices(len(by_cost), count)}
    mandatory = {
        max(metadata, key=lambda row: (row["cost"], -row["position"]))["position"],
        max(metadata, key=lambda row: (row["max_source"], -row["position"]))["position"],
        max(metadata, key=lambda row: (row["max_target"], -row["position"]))["position"],
    }
    chosen |= mandatory
    while len(chosen) > count:
        removable = sorted(chosen - mandatory,
                           key=lambda pos: (metadata[pos]["cost"], pos))
        require(bool(removable), "probe batch mandatory coverage exceeds count")
        chosen.remove(removable[0])
    require(len(chosen) == count, "probe batch selection differs")
    return sorted(chosen, key=lambda pos: (metadata[pos]["cost"], pos))


def select_beam_inputs(records: list[dict], count: int = PROBE_BEAMS,
                       evaluation_max_length: int | None = None) -> list[dict]:
    unique = sorted({row["roman"] for row in records}, key=lambda value: (len(value), value))
    selected = [{"input": unique[index], "kind": "train_length_stratified"}
                for index in quantile_indices(len(unique), count)]
    if evaluation_max_length is not None and evaluation_max_length > len(unique[-1]):
        selected.append({"input": "a" * evaluation_max_length,
                         "kind": "synthetic_evaluation_max_length_resource_only"})
    return selected


def projected_seconds(update_seconds: float, observed_cost: int, full_cost: int,
                      beam_observations: list[dict],
                      evaluation_histogram: dict[str, int]) -> dict:
    values = (update_seconds, *(row["seconds"] for row in beam_observations))
    require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in values),
            "probe timing invalid")
    evaluation_inputs = sum(evaluation_histogram.values())
    require(observed_cost > 0 and full_cost > 0 and evaluation_inputs > 0
            and beam_observations,
            "projection denominator invalid")
    train = update_seconds / observed_cost * full_cost
    max_input_seconds = max(row["seconds"] for row in beam_observations)
    max_work_rate = max(row["seconds"] / beam_work_units(row["length"])
                        for row in beam_observations)
    per_input = max_input_seconds * evaluation_inputs
    evaluation_work_units = sum(
        beam_work_units(int(length)) * count
        for length, count in evaluation_histogram.items())
    evaluation = max_work_rate * evaluation_work_units
    total = RECIPE["projection_margin_seconds"] + RECIPE["projection_multiplier"] * (
        train + evaluation)
    return {"training_seconds": train,
            "evaluation_global_worst_per_input_diagnostic_seconds": per_input,
            "maximum_observed_seconds_per_work_unit": max_work_rate,
            "evaluation_work_units": evaluation_work_units,
            "evaluation_work_bound_seconds": evaluation,
            "evaluation_seconds": evaluation, "projected_total_seconds": total}


def beam_work_units(length: int) -> int:
    require(type(length) is int and length > 0, "beam input length invalid")
    return length + 2 + min(3 * length + 4, 73)


def load_evaluation_contract() -> dict:
    manifest = read_json(EVALUATION_MANIFEST)
    histogram = manifest.get("input_length_histogram")
    require(manifest.get("schema") == "document-phonetic-inputs-v1"
            and manifest.get("completed") is True
            and manifest.get("model_activity") is False
            and manifest.get("performance_labels_joined") is False
            and manifest.get("counts", {}).get("unique_inputs") == 10166
            and isinstance(histogram, dict)
            and all(str(key).isdigit() and type(value) is int and value > 0
                    for key, value in histogram.items())
            and sum(histogram.values()) == 10166
            and min(map(int, histogram)) == 1 and max(map(int, histogram)) == 43,
            "evaluation input manifest differs")
    return {"manifest_sha256": FIXED_HASHES[EVALUATION_MANIFEST],
            "unique_inputs": 10166, "input_length_histogram": histogram,
            "maximum_input_length": 43}


def load_records() -> tuple[dict[str, list[dict]], dict[str, str]]:
    for path, wanted in FIXED_HASHES.items():
        require(path.is_file() and sha256_file(path) == wanted, f"fixed source differs: {path}")
    base_manifest, delta_manifest = read_json(BASE / "manifest.json"), read_json(DELTA / "manifest.json")
    proof = read_json(DELTA / "independent-verification.json")
    evaluation = load_evaluation_contract()
    require(base_manifest.get("split_counts", {}).get("train") == 51361,
            "control count differs")
    require(delta_manifest.get("counts", {}).get("accepted_pairs") == 32984,
            "delta count differs")
    require(proof.get("schema") == "observed-phonetic-delta-independent-verification-v1"
            and proof.get("verified") is True
            and proof.get("manifest_sha256") == FIXED_HASHES[DELTA / "manifest.json"]
            and proof.get("delta_sha256") == FIXED_HASHES[DELTA / "delta.jsonl"],
            "delta independent proof differs")
    control = [{"pair_id": f"base:{row['index']}", "roman": row["roman"],
                "surface": row["surface"]}
               for row in read_jsonl(BASE / "pairs.jsonl") if row["split"] == "train"]
    delta = [{"pair_id": f"delta:{index}", "roman": row["roman"],
              "surface": row["surface"]}
             for index, row in enumerate(read_jsonl(DELTA / "delta.jsonl"))]
    require(len(control) == 51361 and len(delta) == 32984, "pair count differs")
    require(len({(x["roman"], x["surface"]) for x in control + delta}) == 84345,
            "control and delta pairs overlap")
    pins = {path_key(path): wanted for path, wanted in FIXED_HASHES.items()}
    require(isinstance(base_manifest.get("source_hashes"), dict)
            and isinstance(delta_manifest.get("source_hashes"), dict),
            "upstream provenance pins missing")
    runner = Path(__file__).resolve()
    pins[path_key(runner)] = sha256_file(runner)
    verify_pins(pins)
    return ({"control": control, "augmented": control + delta},
            dict(sorted(pins.items())), evaluation)


def prepare(output: Path, evaluation_inputs: int) -> dict:
    output = Path(output)
    require(type(evaluation_inputs) is int and evaluation_inputs > 0,
            "evaluation input count must be positive")
    if output.exists():
        raise FileExistsError(output)
    records, pins, evaluation = load_records()
    require(evaluation_inputs == evaluation["unique_inputs"],
            "evaluation input count does not match frozen manifest")
    summaries = {}
    for arm in ARMS:
        schedule = cyclic_schedule(len(records[arm]))
        summaries[arm] = arm_contract(records[arm])
    output.mkdir(parents=True)
    contract = {
        "schema": "observed-phonetic-pair-prepared-v1", "completed": True,
        "evaluation_inputs": evaluation_inputs, "recipe": RECIPE,
        "evaluation": evaluation, "model_config": CONFIG,
        "arms": summaries, "source_hashes": pins,
        "contracts": {"model_loaded": False, "model_inference": False,
                      "training_run": False, "document_reserve_targets_read": False,
                      "fixed_compute_paired_arms": True},
    }
    write_json(output / "prepared.json", contract)
    verify_pins(pins)
    return contract


def load_prepared(directory: Path) -> tuple[dict, dict[str, list[dict]], dict[str, str]]:
    path = Path(directory) / "prepared.json"
    report = read_json(path)
    records, pins, evaluation = load_records()
    require(report.get("schema") == "observed-phonetic-pair-prepared-v1"
            and report.get("completed") is True and report.get("recipe") == RECIPE
            and report.get("model_config") == CONFIG and report.get("source_hashes") == pins
            and report.get("evaluation") == evaluation
            and report.get("contracts", {}).get("model_loaded") is False,
            "prepared contract differs")
    for arm in ARMS:
        require(report.get("arms", {}).get(arm)
                == arm_contract(records[arm]),
                f"prepared {arm} schedule differs")
    require(type(report.get("evaluation_inputs")) is int and report["evaluation_inputs"] > 0,
            "prepared evaluation count differs")
    return report, records, pins


def torch_modules():
    import torch
    from torch import nn
    from hanglish.models.phonetic_transducer import (
        OUTPUT_TO_ID, PAD, PhoneticTransducer, TransducerConfig, _Network,
        checkpoint_payload,
    )
    from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes
    from scripts.train_phonetic_transducer import PairDataset, collate
    return locals()


def configure_torch(torch) -> None:
    torch.set_num_threads(RECIPE["cpu_threads"])
    try:
        torch.set_num_interop_threads(RECIPE["interop_threads"])
    except RuntimeError:
        require(torch.get_num_interop_threads() == RECIPE["interop_threads"],
                "Torch inter-op threads differ")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def tensor_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(canonical(list(value.shape)) + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def save_initial(path: Path, modules) -> tuple[str, str, int]:
    torch, config_type, network_type = modules["torch"], modules["TransducerConfig"], modules["_Network"]
    config = config_type(**CONFIG)
    network = network_type(config).cpu()
    torch.save(modules["checkpoint_payload"](network, config), path)
    return sha256_file(path), tensor_hash(network.state_dict()), sum(x.numel() for x in network.parameters())


def load_initial(path: Path, modules, expected_sha: str | None = None,
                 expected_tensor: str | None = None):
    torch = modules["torch"]
    require(path.is_file(), "initial checkpoint missing")
    actual_sha = sha256_file(path)
    if expected_sha is not None:
        require(actual_sha == expected_sha, "initial checkpoint hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload.get("config") == CONFIG, "initial model config differs")
    config = modules["TransducerConfig"](**CONFIG)
    network = modules["_Network"](config).cpu()
    network.load_state_dict(payload["state_dict"], strict=True)
    actual_tensor = tensor_hash(network.state_dict())
    if expected_tensor is not None:
        require(actual_tensor == expected_tensor, "initial tensor hash differs")
    return network, config, actual_sha, actual_tensor


def make_batch(records, indices, modules):
    dataset = modules["PairDataset"]([(records[i]["roman"], records[i]["surface"])
                                      for i in indices])
    return modules["collate"]([dataset[i] for i in range(len(dataset))])


def parameters_finite(network, torch) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in network.parameters())


def one_update(network, optimizer, batch, modules, *, check_parameters: bool) -> tuple[float, float]:
    torch, nn = modules["torch"], modules["nn"]
    source, target_input, target_output = batch
    optimizer.zero_grad(set_to_none=True)
    logits = network(source, target_input)
    criterion = nn.CrossEntropyLoss(ignore_index=modules["OUTPUT_TO_ID"][modules["PAD"]],
                                    label_smoothing=RECIPE["label_smoothing"])
    loss = criterion(logits.reshape(-1, logits.shape[-1]), target_output.reshape(-1))
    require(bool(torch.isfinite(loss)), "non-finite training loss")
    loss.backward()
    norm = nn.utils.clip_grad_norm_(network.parameters(), RECIPE["gradient_clip_norm"],
                                   error_if_nonfinite=True)
    optimizer.step()
    require(math.isfinite(float(norm)), "non-finite gradient norm")
    if check_parameters:
        require(parameters_finite(network, torch), "non-finite trained parameter")
    return float(loss.detach()), float(norm)


def resource_within_limits(row: dict) -> bool:
    return (isinstance(row, dict)
            and type(row.get("elapsed_s")) in (int, float)
            and math.isfinite(row["elapsed_s"])
            and 0 <= row["elapsed_s"] <= LIMITS["whole_seconds"]
            and type(row.get("rss_bytes")) is int
            and 0 <= row["rss_bytes"] <= LIMITS["rss_bytes"]
            and type(row.get("swap_growth_bytes")) is int
            and row["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"])


def finalize_report(output: Path, report: dict, modules, started: float,
                    baseline_swap: int) -> dict:
    report_path = output / "report.json"
    completion_path = output / "completion.json"
    with modules["AtomicResourceWatchdog"](
            started, baseline_swap, output / "resources-finalize.jsonl", limits=LIMITS) as finalizer:
        with finalizer.operation("write_report", update=True):
            write_json(report_path, report)
        after_report = finalizer.sample_and_append("after_report")
        require(resource_within_limits(after_report), "post-report resources exceed limits")
        completion = {
            "schema": "observed-phonetic-pair-completion-v1", "completed": True,
            "report_sha256": sha256_file(report_path),
            "resource_after_report": after_report,
        }
        with finalizer.operation("write_completion", update=True):
            write_json(completion_path, completion)
        finalizer.sample_and_append("after_completion")
    return completion


def validate_probe(report: dict, completion: dict, arm: str, prepared: dict,
                   prepared_sha: str, pins: dict[str, str],
                   report_sha: str | None = None) -> None:
    observations = report.get("observations")
    beams = report.get("beam_observations")
    expected_positions = prepared["arms"][arm]["probe_update_positions"]
    expected_beams = prepared["arms"][arm]["probe_beam_inputs"]
    observations_valid = (
        isinstance(observations, list) and len(observations) == PROBE_UPDATES
        and [row.get("schedule_position") for row in observations] == expected_positions
        and all(row.get("ordinal") == index + 1
                and type(row.get("padded_cost")) is int and row["padded_cost"] > 0
                and type(row.get("seconds")) in (int, float)
                and math.isfinite(row["seconds"]) and row["seconds"] >= 0
                for index, row in enumerate(observations)))
    beams_valid = (
        isinstance(beams, list) and len(beams) == len(expected_beams)
        and [{key: row.get(key) for key in ("input", "kind")} for row in beams]
            == expected_beams
        and all(row.get("length") == len(row.get("input", ""))
                and type(row.get("seconds")) in (int, float)
                and math.isfinite(row["seconds"]) and row["seconds"] >= 0
                for row in beams))
    recomputed = None
    if observations_valid and beams_valid:
        recomputed = projected_seconds(
            sum(row["seconds"] for row in observations),
            sum(row["padded_cost"] for row in observations),
            prepared["arms"][arm]["total_padded_cost"], beams,
            prepared["evaluation"]["input_length_histogram"])
    require(report.get("schema") == "observed-phonetic-pair-probe-v1"
            and report.get("completed") is True and report.get("accepted") is True
            and report.get("arm") == arm
            and report.get("prepared_sha256") == prepared_sha
            and report.get("evaluation_inputs") == prepared["evaluation_inputs"]
            and report.get("recipe") == RECIPE and report.get("limits") == LIMITS
            and report.get("source_hashes") == pins
            and report.get("source_hashes_after") == pins
            and report.get("prototype_discarded") is True
            and report.get("model_updates_run") == PROBE_UPDATES
            and report.get("train_beam_inputs_run") == PROBE_BEAMS
            and report.get("beam_inputs_run") in (PROBE_BEAMS, PROBE_BEAMS + 1)
            and report.get("synthetic_beam_inputs_run")
                == report.get("beam_inputs_run") - PROBE_BEAMS
            and observations_valid and beams_valid
            and report.get("update_selection") == expected_positions
            and report.get("update_selection_sha256")
                == hashlib.sha256(canonical(expected_positions)).hexdigest()
            and report.get("beam_selection_sha256")
                == hashlib.sha256(canonical(expected_beams)).hexdigest()
            and report.get("coverage") == {
                "maximum_padded_cost": True, "longest_train_source": True,
                "evaluation_max_length": True, "longest_target_batch_included": True}
            and report.get("projection") == recomputed
            and recomputed["projected_total_seconds"] <= LIMITS["whole_seconds"]
            and resource_within_limits(report.get("final_resource"))
            and isinstance(report.get("initial"), dict)
            and len(report["initial"].get("checkpoint_sha256", "")) == 64
            and len(report["initial"].get("tensor_sha256", "")) == 64,
            "accepted probe contract differs")
    require(completion.get("schema") == "observed-phonetic-pair-completion-v1"
            and completion.get("completed") is True
            and (report_sha is None or completion.get("report_sha256") == report_sha)
            and resource_within_limits(completion.get("resource_after_report")),
            "probe completion contract differs")


def run_probe(arm: str, prepared_dir: Path, output: Path,
              initial: Path | None = None,
              shared_initial_report: Path | None = None) -> dict:
    started = time.monotonic()
    require(arm in ARMS, "unknown arm")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    prepared, all_records, pins = load_prepared(prepared_dir)
    prepared_path = Path(prepared_dir) / "prepared.json"
    prepared_sha = sha256_file(prepared_path)
    records = all_records[arm]
    schedule = cyclic_schedule(len(records))
    selected = prepared["arms"][arm]["probe_update_positions"]
    beams = prepared["arms"][arm]["probe_beam_inputs"]
    output.mkdir(parents=True)
    modules = torch_modules()
    torch = modules["torch"]
    baseline_swap = modules["system_swap_bytes"]()
    configure_torch(torch)
    if arm == "augmented":
        require(initial is not None and shared_initial_report is not None,
                "augmented probe requires control shared initial and report")
    if arm == "control":
        require(shared_initial_report is None,
                "control probe cannot consume a shared initial report")
    initial_path = Path(initial) if initial is not None else output / "initial.pt"
    shared_report_sha = None
    if initial is None:
        initial_sha, initial_tensor, parameters = save_initial(initial_path, modules)
    else:
        network, _, initial_sha, initial_tensor = load_initial(initial_path, modules)
        parameters = sum(x.numel() for x in network.parameters())
        del network
    if shared_initial_report is not None:
        shared = read_json(shared_initial_report)
        shared_completion_path = Path(shared_initial_report).with_name("completion.json")
        shared_completion = read_json(shared_completion_path)
        validate_probe(shared, shared_completion, "control", prepared, prepared_sha, pins,
                       sha256_file(shared_initial_report))
        require(shared.get("initial", {}).get("checkpoint_sha256") == initial_sha
                and shared.get("initial", {}).get("tensor_sha256") == initial_tensor,
                "control shared initial report differs")
        shared_report_sha = sha256_file(shared_initial_report)
    network, config, _, _ = load_initial(initial_path, modules, initial_sha, initial_tensor)
    optimizer = torch.optim.AdamW(network.parameters(), lr=RECIPE["learning_rate"],
                                  weight_decay=RECIPE["weight_decay"])
    observations, losses, gradients = [], [], []
    probe_updates_path = output / "updates.jsonl"
    with modules["AtomicResourceWatchdog"](
            started, baseline_swap, output / "resources.jsonl", limits=LIMITS) as watchdog:
        for ordinal, position in enumerate(selected, 1):
            op = f"probe_update_{ordinal}"
            before = time.monotonic()
            with watchdog.operation(op, update=True):
                loss, gradient = one_update(
                    network, optimizer, make_batch(records, schedule[position], modules), modules,
                    check_parameters=True)
                append_jsonl(probe_updates_path, {
                    "ordinal": ordinal, "schedule_position": position,
                    "loss": loss, "gradient_norm": gradient,
                    "padded_cost": batch_cost(records, schedule[position]),
                    "pair_indices": schedule[position],
                    "compute_seconds": time.monotonic() - before,
                })
            seconds = time.monotonic() - before
            losses.append(loss); gradients.append(gradient)
            observations.append({"ordinal": ordinal, "schedule_position": position,
                                 "padded_cost": batch_cost(records, schedule[position]),
                                 "seconds": seconds})
        model = modules["PhoneticTransducer"](network, config, "cpu")
        beam_observations = []
        for ordinal, beam in enumerate(beams, 1):
            before = time.monotonic()
            with watchdog.operation(f"probe_beam8_{ordinal}", update=True):
                model.generate(beam["input"], beam_size=8)
            beam_observations.append({**beam, "length": len(beam["input"]),
                                      "seconds": time.monotonic() - before})
        verify_pins(pins)
        final_resource = watchdog.sample_and_append("final_before_report")
    observed_cost = sum(row["padded_cost"] for row in observations)
    update_seconds = sum(row["seconds"] for row in observations)
    projection = projected_seconds(
        update_seconds, observed_cost, prepared["arms"][arm]["total_padded_cost"],
        beam_observations, prepared["evaluation"]["input_length_histogram"])
    accepted = (projection["projected_total_seconds"] <= LIMITS["whole_seconds"]
                and resource_within_limits(final_resource))
    report = {
        "schema": "observed-phonetic-pair-probe-v1", "completed": True,
        "accepted": accepted, "arm": arm, "prepared_sha256": prepared_sha,
        "evaluation_inputs": prepared["evaluation_inputs"],
        "recipe": RECIPE,
        "source_hashes": pins, "source_hashes_after": pins,
        "initial": {"path": str(initial_path.resolve()),
                    "checkpoint_sha256": initial_sha, "tensor_sha256": initial_tensor,
                    "parameter_count": parameters, "created_by_this_probe": initial is None,
                    "control_probe_report_sha256": shared_report_sha},
        "model_updates_run": len(observations), "beam_inputs_run": len(beams),
        "train_beam_inputs_run": sum(row["kind"] == "train_length_stratified"
                                     for row in beams),
        "synthetic_beam_inputs_run": sum(row["kind"].startswith("synthetic")
                                         for row in beams),
        "update_selection": selected,
        "update_selection_sha256": hashlib.sha256(canonical(selected)).hexdigest(),
        "beam_selection_sha256": hashlib.sha256(canonical(beams)).hexdigest(),
        "coverage": {
            "maximum_padded_cost": max(row["padded_cost"] for row in observations)
                == prepared["arms"][arm]["max_batch_cost"],
            "longest_train_source": max(len(row["input"]) for row in beams
                                        if row["kind"] == "train_length_stratified")
                == max(len(x["roman"]) for x in records),
            "evaluation_max_length": max(len(row["input"]) for row in beams)
                >= prepared["evaluation"]["maximum_input_length"],
            "longest_target_batch_included": max(
                max(len(records[i]["surface"]) for i in schedule[pos]) for pos in selected)
                == max(len(x["surface"]) for x in records),
        },
        "observations": observations, "mean_loss": sum(losses) / len(losses),
        "max_gradient_norm": max(gradients),
        "beam_observations": beam_observations,
        "projection": projection, "limits": LIMITS,
        "final_resource": final_resource, "prototype_discarded": True,
        "training_run": False, "evaluation_targets_read": False,
    }
    finalize_report(output, report, modules, started, baseline_swap)
    return report


def run_train(arm: str, prepared_dir: Path, probe_dir: Path,
              output: Path, initial: Path | None = None) -> dict:
    started = time.monotonic()
    require(arm in ARMS, "unknown arm")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    prepared, all_records, pins = load_prepared(prepared_dir)
    prepared_sha = sha256_file(Path(prepared_dir) / "prepared.json")
    probe_path = Path(probe_dir) / "report.json"
    probe_completion_path = Path(probe_dir) / "completion.json"
    probe = read_json(probe_path)
    probe_completion = read_json(probe_completion_path)
    validate_probe(probe, probe_completion, arm, prepared, prepared_sha, pins,
                   sha256_file(probe_path))
    initial_path = Path(initial) if initial is not None else Path(probe["initial"]["path"])
    require(sha256_file(initial_path) == probe["initial"]["checkpoint_sha256"],
            "shared initial checkpoint differs")
    records = all_records[arm]
    schedule = cyclic_schedule(len(records))
    require(arm_contract(records) == prepared["arms"][arm],
            "training schedule differs from prepared contract")
    output.mkdir(parents=True)
    modules = torch_modules(); torch = modules["torch"]
    baseline_swap = modules["system_swap_bytes"]()
    configure_torch(torch)
    network, config, initial_sha, initial_tensor = load_initial(
        initial_path, modules, probe["initial"]["checkpoint_sha256"],
        probe["initial"]["tensor_sha256"])
    optimizer = torch.optim.AdamW(network.parameters(), lr=RECIPE["learning_rate"],
                                  weight_decay=RECIPE["weight_decay"])
    update_path = output / "updates.jsonl"
    losses, gradients = [], []
    with modules["AtomicResourceWatchdog"](
            started, baseline_swap, output / "resources.jsonl", limits=LIMITS) as watchdog:
        for update, indices in enumerate(schedule, 1):
            before = time.monotonic()
            with watchdog.operation(f"optimizer_update_{update}", update=True):
                loss, gradient = one_update(
                    network, optimizer, make_batch(records, indices, modules), modules,
                    check_parameters=(update == 1 or update % 128 == 0 or update == UPDATES))
                row = {"update": update, "loss": loss, "gradient_norm": gradient,
                       "compute_seconds": time.monotonic() - before,
                       "padded_cost": batch_cost(records, indices), "pair_indices": indices,
                       "parameter_finiteness_checked": (
                           update == 1 or update % 128 == 0 or update == UPDATES)}
                append_jsonl(update_path, row)
            losses.append(loss); gradients.append(gradient)
        final_path = output / "phonetic_transducer.pt"
        with watchdog.operation("save_final_checkpoint", update=True):
            torch.save(modules["checkpoint_payload"](network, config), final_path)
        final_tensor = tensor_hash(network.state_dict())
        final_sha = sha256_file(final_path)
        del network, optimizer
        with watchdog.operation("reload_final_checkpoint", update=True):
            restored_payload = torch.load(final_path, map_location="cpu", weights_only=True)
            restored = modules["_Network"](config).cpu()
            restored.load_state_dict(restored_payload["state_dict"], strict=True)
        reload_tensor = tensor_hash(restored.state_dict())
        require(reload_tensor == final_tensor, "final checkpoint reload differs")
        require(final_tensor != initial_tensor, "training did not change model parameters")
        require(parameters_finite(restored, torch), "reloaded final parameter is non-finite")
        metadata = {
            "format_version": 1, "checkpoint_sha256": final_sha,
            "config": CONFIG, "arm": arm, "initial_checkpoint_sha256": initial_sha,
            "initial_tensor_sha256": initial_tensor, "final_tensor_sha256": final_tensor,
        }
        write_json(output / "phonetic_transducer.json", metadata)
        verify_pins(pins)
        final_resource = watchdog.sample_and_append("final_before_report")
    report = {
        "schema": "observed-phonetic-pair-training-v1", "completed": True,
        "arm": arm, "prepared_sha256": prepared_sha,
        "probe_report_sha256": sha256_file(probe_path),
        "probe_completion_sha256": sha256_file(probe_completion_path),
        "evaluation_inputs": prepared["evaluation_inputs"],
        "source_hashes": pins, "source_hashes_after": pins,
        "recipe": RECIPE, "model_config": CONFIG,
        "schedule": prepared["arms"][arm], "updates_completed": len(losses),
        "mean_loss": sum(losses) / len(losses), "final_loss": losses[-1],
        "max_gradient_norm": max(gradients),
        "parameter_finiteness_checks": sum(
            update == 1 or update % 128 == 0 or update == UPDATES
            for update in range(1, UPDATES + 1)),
        "initial": {"checkpoint_sha256": initial_sha, "tensor_sha256": initial_tensor,
                    "path": str(initial_path.resolve())},
        "final": {"checkpoint_sha256": final_sha, "tensor_sha256": final_tensor,
                  "reload_tensor_sha256": reload_tensor, "reload_exact": True},
        "updates_sha256": sha256_file(update_path), "limits": LIMITS,
        "final_resource": final_resource, "evaluation_targets_read": False,
        "checkpoint_selection": "final_fixed_update_only", "production_changed": False,
    }
    finalize_report(output, report, modules, started, baseline_swap)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)
    prepare_cmd = commands.add_parser("prepare")
    prepare_cmd.add_argument("--output", type=Path, required=True)
    prepare_cmd.add_argument("--evaluation-inputs", type=int, required=True)
    probe_cmd = commands.add_parser("probe")
    probe_cmd.add_argument("--arm", choices=ARMS, required=True)
    probe_cmd.add_argument("--prepared", type=Path, required=True)
    probe_cmd.add_argument("--output", type=Path, required=True)
    probe_cmd.add_argument("--initial", type=Path)
    probe_cmd.add_argument("--shared-initial-report", type=Path)
    train_cmd = commands.add_parser("train")
    train_cmd.add_argument("--arm", choices=ARMS, required=True)
    train_cmd.add_argument("--prepared", type=Path, required=True)
    train_cmd.add_argument("--probe", type=Path, required=True)
    train_cmd.add_argument("--output", type=Path, required=True)
    train_cmd.add_argument("--initial", type=Path)
    return value


def main() -> None:
    args = parser().parse_args()
    existed = args.output.exists()
    try:
        if args.command == "prepare":
            result = prepare(args.output, args.evaluation_inputs)
        elif args.command == "probe":
            result = run_probe(args.arm, args.prepared, args.output, args.initial,
                               args.shared_initial_report)
        else:
            result = run_train(args.arm, args.prepared, args.probe, args.output, args.initial)
    except BaseException as error:
        if not existed and args.output.is_dir():
            write_json(args.output / "failure-report.json", {
                "schema": "observed-phonetic-pair-failure-v1", "completed": False,
                "command": args.command, "error_type": type(error).__name__,
                "error": str(error),
            })
        raise
    print(json.dumps({"schema": result["schema"], "completed": result["completed"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
