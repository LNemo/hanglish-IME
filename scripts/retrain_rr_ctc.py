#!/usr/bin/env python3
"""Refresh RR sentence inputs and train isolated word/growth CTC candidates."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BASE_CORPUS = ROOT / "data/challenge/sentence-v3"
ARMS = {
    "word": {
        "extra": ROOT / "data/challenge/sentence-train-extra-v2",
        "rows": 6680,
        "changed": 132,
    },
    "growth": {
        "extra": ROOT / "data/challenge/sentence-train-extra-v3",
        "rows": 7704,
        "changed": 163,
    },
}
HELD_SPLITS = (
    (ROOT / "data/challenge/sentence-v3", "dev"),
    (ROOT / "data/challenge/sentence-v3", "test"),
    (ROOT / "data/challenge/v2.2", "dev"),
    (ROOT / "data/challenge/v2.2", "test"),
)
OLD_PHONOLOGY = (
    ROOT / ".omx/artifacts/rr-retraining-20261020/baseline-source/"
           "hanglish/core/phonology.py")
SEED = 20260921
BATCH_SIZE = 16
EPOCHS = 8
CHECKPOINT_INTERVAL = 50
LIMITS = {
    "whole_seconds": 7200.0,
    "update_seconds": 60.0,
    "rss_bytes": 3 * 1024**3,
    "swap_growth_bytes": 1536 * 1024**2,
}
RECIPE = {
    "seed": SEED,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": .01,
    "gradient_clip_norm": 1.0,
    "cpu_threads": 1,
    "interop_threads": 1,
    "deterministic_algorithms": True,
    "checkpoint_interval": CHECKPOINT_INTERVAL,
}
SOURCE_FILES = (
    "hanglish/core/phonology.py",
    "hanglish/core/predicate_boundaries.py",
    "hanglish/core/data/predicate_boundary_inventory.json",
    "hanglish/models/sentence_ctc.py",
    "hanglish/models/sentence_transducer.py",
    "scripts/sentence_data.py",
    "scripts/train_sentence_ctc.py",
    "scripts/train_sentence_transducer.py",
    "scripts/sentence_transducer_recovery.py",
    "scripts/source_span_resources.py",
    "scripts/preflight_pko_seq2seq.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, raw: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value) -> str:
    raw = canonical(value) + b"\n"
    atomic_write(path, raw)
    return hashlib.sha256(raw).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> str:
    raw = b"".join(canonical(row) + b"\n" for row in rows)
    atomic_write(path, raw)
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line]


def verify_hashes(pins: dict[str, str]) -> None:
    for name, wanted in pins.items():
        require(sha256_file(ROOT / name) == wanted, f"source changed: {name}")


def _load_old_romanizer(path: Path = OLD_PHONOLOGY):
    require(path.is_file(), "audited pre-change phonology snapshot is missing")
    spec = importlib.util.spec_from_file_location(
        "hanglish.core._rr_ctc_old_phonology", path)
    require(spec is not None and spec.loader is not None, "old phonology cannot load")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module.romanize_word


def source_pins() -> dict[str, str]:
    paths = [ROOT / name for name in SOURCE_FILES]
    paths.extend((Path(__file__), OLD_PHONOLOGY))
    for arm in ARMS.values():
        for corpus in (BASE_CORPUS, arm["extra"]):
            paths.extend((corpus / "train.jsonl", corpus / "manifest.json"))
    for corpus, split in HELD_SPLITS:
        paths.extend((corpus / f"{split}.jsonl", corpus / "manifest.json"))
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(set(paths))}


def held_inputs() -> tuple[set[str], dict[str, int]]:
    """Extract held source strings; expected fields never leave this boundary."""
    result, counts = set(), {}
    for corpus, split in HELD_SPLITS:
        name = f"{corpus.name}/{split}"
        rows = [json.loads(line) for line in (corpus / f"{split}.jsonl").read_text(
            encoding="utf-8").splitlines() if line]
        values = [row.get("input") for row in rows]
        require(all(isinstance(value, str) for value in values),
                f"invalid held source input: {name}")
        counts[name] = len(values)
        result.update(values)
    return result, counts


def refresh_rows(rows: list[dict], old_romanize, new_romanize,
                 *, expected_changed: int | None = None) -> tuple[list[dict], list[dict]]:
    """Replace only stale generated Roman inputs; preserve all other fields."""
    refreshed, changes = [], []
    seen_ids, seen_inputs = set(), set()
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("id"), str),
                "invalid training row")
        require(isinstance(row.get("input"), str)
                and isinstance(row.get("expected"), list)
                and len(row["expected"]) == 1
                and isinstance(row["expected"][0], str),
                f"invalid training pair: {row['id']}")
        target = row["expected"][0]
        old_input, new_input = old_romanize(target), new_romanize(target)
        updated = dict(row)
        if old_input != new_input:
            require(row["input"] == old_input,
                    f"changed RR row was not generated by frozen old rules: {row['id']}")
            updated["input"] = new_input
            changes.append({"id": row["id"], "old_input": old_input,
                            "new_input": new_input, "target": target})
        require(updated["id"] not in seen_ids, f"duplicate id: {updated['id']}")
        require(updated["input"] not in seen_inputs,
                f"refreshed input collision: {updated['input']}")
        seen_ids.add(updated["id"]); seen_inputs.add(updated["input"])
        # A refresh is allowed to alter exactly one field.
        require({key: value for key, value in updated.items() if key != "input"}
                == {key: value for key, value in row.items() if key != "input"},
                f"non-input training field changed: {row['id']}")
        refreshed.append(updated)
    if expected_changed is not None:
        require(len(changes) == expected_changed,
                f"RR change count differs: {len(changes)} != {expected_changed}")
    return refreshed, changes


def prepare(output: Path) -> dict:
    from hanglish.core.phonology import romanize_word
    from hanglish.models.sentence_ctc import training_batch
    from scripts.sentence_data import read_split

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    pins = source_pins()
    old_romanize = _load_old_romanizer()
    held, held_counts = held_inputs()
    output.mkdir(parents=True)
    arm_reports = {}
    for name, spec in ARMS.items():
        original = read_split(BASE_CORPUS, "train") + read_split(spec["extra"], "train")
        require(len(original) == spec["rows"], f"{name} row count differs")
        rows, changes = refresh_rows(
            original, old_romanize, romanize_word, expected_changed=spec["changed"])
        old_overlap = sorted({row["input"] for row in original} & held)
        new_overlap = sorted({row["input"] for row in rows} & held)
        require(not old_overlap and not new_overlap,
                f"{name} TRAIN input overlaps a held source")
        for row in rows:
            training_batch([(row["input"], row["expected"][0])])
        arm_dir = output / name
        arm_dir.mkdir()
        train_sha = write_jsonl(arm_dir / "train.jsonl", rows)
        changes_sha = write_jsonl(arm_dir / "changes.jsonl", changes)
        arm_reports[name] = {
            "rows": len(rows), "changed_rows": len(changes),
            "train_sha256": train_sha, "changes_sha256": changes_sha,
            "row_ids_sha256": hashlib.sha256(canonical([row["id"] for row in rows])).hexdigest(),
            "targets_sha256": hashlib.sha256(canonical([row["expected"] for row in rows])).hexdigest(),
            "frame_feasibility_checked": True,
            "held_input_overlap_before": len(old_overlap),
            "held_input_overlap_after": len(new_overlap),
        }
    manifest = {
        "schema": "rr-ctc-retrain-prepared-v1", "completed": True,
        "recipe": RECIPE, "source_hashes": pins, "arms": arm_reports,
        "held_source_counts": held_counts,
        "contracts": {"training_targets_unchanged": True,
                      "only_roman_input_refreshed": True,
                      "held_targets_used_for_training": False,
                      "held_inputs_used_for_overlap_check_only": True,
                      "production_weights_changed": False},
    }
    manifest_sha = write_json(output / "prepared.json", manifest)
    write_json(output / "completion.json", {"completed": True,
               "prepared_sha256": manifest_sha})
    return manifest


def load_prepared(directory: Path, arm: str) -> tuple[dict, list[dict], str]:
    require(arm in ARMS, "unknown CTC arm")
    directory = Path(directory)
    manifest_path = directory / "prepared.json"
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == "rr-ctc-retrain-prepared-v1"
            and manifest.get("completed") is True
            and manifest.get("recipe") == RECIPE,
            "prepared CTC contract differs")
    verify_hashes(manifest.get("source_hashes", {}))
    info = manifest.get("arms", {}).get(arm, {})
    rows = read_jsonl(directory / arm / "train.jsonl")
    require(len(rows) == info.get("rows") == ARMS[arm]["rows"]
            and sha256_file(directory / arm / "train.jsonl") == info.get("train_sha256")
            and sha256_file(directory / arm / "changes.jsonl") == info.get("changes_sha256")
            and info.get("changed_rows") == ARMS[arm]["changed"],
            "prepared CTC arm differs")
    return manifest, rows, sha256_file(manifest_path)


def configure_torch(torch) -> None:
    torch.set_num_threads(RECIPE["cpu_threads"])
    try:
        torch.set_num_interop_threads(RECIPE["interop_threads"])
    except RuntimeError:
        require(torch.get_num_interop_threads() == RECIPE["interop_threads"],
                "Torch inter-op thread count differs")
    random.seed(SEED); torch.manual_seed(SEED)
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


def schedule(rows: list[dict]) -> list[list[list[dict]]]:
    from scripts.train_sentence_transducer import batches
    return [batches(rows, epoch) for epoch in range(EPOCHS)]


def _length(row: dict) -> int:
    from hanglish.models.sentence_transducer import encode_source, encode_target
    return max(len(encode_source(row["input"])), len(encode_target(row["expected"][0])) + 1)


def probe_groups(rows: list[dict]) -> list[list[dict]]:
    groups = [group for epoch in schedule(rows) for group in epoch]
    ordered = sorted(groups, key=lambda group: max(_length(row) for row in group))
    positions = sorted(set(round((len(ordered) - 1) * fraction)
                           for fraction in (0, .25, .5, .75, 1)))
    return [ordered[position] for position in positions]


def project_training_seconds(observations: list[dict], rows: list[dict]) -> float:
    require(observations, "CTC probe observations are missing")
    timing = []
    maximum = 0.
    for row in sorted(observations, key=lambda item: item["actual_max_tokens"]):
        require(type(row.get("actual_max_tokens")) is int
                and type(row.get("seconds")) in (int, float)
                and math.isfinite(row["seconds"]) and row["seconds"] > 0,
                "invalid CTC probe observation")
        maximum = max(maximum, float(row["seconds"]))
        timing.append((row["actual_max_tokens"], maximum))
    total = 0.
    for epoch in schedule(rows):
        for group in epoch:
            length = max(_length(row) for row in group)
            matches = [seconds for boundary, seconds in timing if length <= boundary]
            require(matches, "probe did not measure the maximum actual training batch")
            total += matches[0]
    return total * 1.25 + 60.


def runtime_modules():
    import torch
    from hanglish.models.sentence_ctc import SentenceCTC, SentenceCTCConfig
    from scripts.preflight_pko_seq2seq import system_swap_bytes
    from scripts.sentence_transducer_recovery import save_checkpoint, restore_checkpoint
    from scripts.source_span_resources import AtomicResourceWatchdog
    from scripts.train_sentence_ctc import optimizer_step
    return locals()


def _limits(max_seconds: float) -> dict:
    require(type(max_seconds) in (int, float) and math.isfinite(max_seconds)
            and 0 < max_seconds <= LIMITS["whole_seconds"], "invalid runtime limit")
    return {**LIMITS, "whole_seconds": float(max_seconds)}


def run_probe(prepared_dir: Path, arm: str, output: Path,
              max_seconds: float = LIMITS["whole_seconds"]) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    _, rows, prepared_sha = load_prepared(prepared_dir, arm)
    output.mkdir(parents=True)
    modules = runtime_modules(); torch = modules["torch"]
    configure_torch(torch)
    model = modules["SentenceCTC"](modules["SentenceCTCConfig"]())
    initial_hash = tensor_hash(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE["learning_rate"],
                                  weight_decay=RECIPE["weight_decay"])
    limits = _limits(max_seconds); started = time.monotonic()
    observations = []
    groups = probe_groups(rows)
    with modules["AtomicResourceWatchdog"](
            started, modules["system_swap_bytes"](), output / "resources.jsonl",
            limits=limits, interval_s=1.) as watchdog:
        for index, group in enumerate(groups, 1):
            before = time.monotonic()
            with watchdog.operation(f"probe_update_{index}", update=True):
                step = modules["optimizer_step"](model, optimizer, group)
            observations.append({"ordinal": index,
                                 "actual_max_tokens": max(_length(row) for row in group),
                                 "seconds": time.monotonic() - before, **step})
        final_resource = watchdog.sample_and_append("final")
    total_updates = sum(len(epoch) for epoch in schedule(rows))
    projected = project_training_seconds(observations, rows)
    accepted = (projected <= limits["whole_seconds"]
                and final_resource["rss_bytes"] <= limits["rss_bytes"]
                and final_resource["swap_growth_bytes"] <= limits["swap_growth_bytes"])
    report = {"schema": "rr-ctc-retrain-probe-v1", "completed": True,
              "accepted": accepted, "arm": arm, "prepared_sha256": prepared_sha,
              "recipe": RECIPE, "limits": limits, "initial_tensor_sha256": initial_hash,
              "observations": observations, "total_training_updates": total_updates,
              "projected_total_seconds": projected, "final_resource": final_resource,
              "prototype_discarded": True, "production_changed": False}
    report_sha = write_json(output / "report.json", report)
    write_json(output / "completion.json", {"completed": True,
               "accepted": accepted, "report_sha256": report_sha})
    return report


def validate_probe(probe_dir: Path, arm: str, prepared_sha: str,
                   limits: dict) -> dict:
    report_path = Path(probe_dir) / "report.json"
    report = read_json(report_path)
    completion = read_json(Path(probe_dir) / "completion.json")
    require(report.get("schema") == "rr-ctc-retrain-probe-v1"
            and report.get("completed") is True and report.get("accepted") is True
            and report.get("arm") == arm
            and report.get("prepared_sha256") == prepared_sha
            and report.get("recipe") == RECIPE and report.get("limits") == limits
            and len(report.get("observations", [])) >= 2
            and report.get("prototype_discarded") is True,
            "matching accepted CTC probe is required")
    require(completion.get("completed") is True
            and completion.get("accepted") is True
            and completion.get("report_sha256") == sha256_file(report_path),
            "CTC probe completion differs")
    return report


def recovery_identity(arm: str, prepared_sha: str, rows: list[dict],
                      initial_hash: str, probe_sha: str, config: dict) -> dict:
    ids = [[[row["id"] for row in group] for group in epoch] for epoch in schedule(rows)]
    return {"schema": "rr-ctc-retrain-identity-v1", "arm": arm,
            "prepared_sha256": prepared_sha, "probe_sha256": probe_sha,
            "recipe": RECIPE, "config": config,
            "initial_tensor_sha256": initial_hash,
            "schedule_sha256": hashlib.sha256(canonical(ids)).hexdigest()}


def _validate_cursor(state: dict, schedules: list[list[list[dict]]]) -> None:
    epoch, batch = state["next_epoch"], state["next_batch"]
    require(type(epoch) is int and 0 <= epoch <= EPOCHS
            and type(batch) is int and batch >= 0, "invalid recovery cursor")
    require((epoch == EPOCHS and batch == 0)
            or (epoch < EPOCHS and batch < len(schedules[epoch])),
            "recovery cursor exceeds schedule")
    expected = sum(len(groups) for groups in schedules[:epoch]) + batch
    require(state["updates"] == expected, "recovery update count differs")


def run_train(prepared_dir: Path, probe_dir: Path, arm: str, output: Path,
              *, resume: bool = False,
              max_seconds: float = LIMITS["whole_seconds"]) -> dict:
    output = Path(output)
    manifest, rows, prepared_sha = load_prepared(prepared_dir, arm)
    limits = _limits(max_seconds)
    probe = validate_probe(probe_dir, arm, prepared_sha, limits)
    modules = runtime_modules(); torch = modules["torch"]
    configure_torch(torch)
    config = modules["SentenceCTCConfig"]()
    model = modules["SentenceCTC"](config)
    initial_hash = tensor_hash(model.state_dict())
    require(initial_hash == probe["initial_tensor_sha256"],
            "probe and training initialization differ")
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE["learning_rate"],
                                  weight_decay=RECIPE["weight_decay"])
    identity = recovery_identity(arm, prepared_sha, rows, initial_hash,
                                 sha256_file(Path(probe_dir) / "report.json"), asdict(config))
    state = {"updates": 0, "next_epoch": 0, "next_batch": 0,
             "epochs_report": [], "epoch_loss_sum": 0., "epoch_token_sum": 0,
             "training_elapsed_s": 0., "peak_rss_bytes": 0,
             "discarded_tail_updates": 0}
    if resume:
        require(output.is_dir(), "resume output is missing")
        state = modules["restore_checkpoint"](output, model, optimizer, identity)
    else:
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
    schedules = schedule(rows)
    _validate_cursor(state, schedules)
    prior_elapsed = state["training_elapsed_s"]
    require(prior_elapsed <= limits["whole_seconds"]
            and state["peak_rss_bytes"] <= limits["rss_bytes"],
            "recovered CTC training already exceeds its resource limit")
    started = time.monotonic()
    with modules["AtomicResourceWatchdog"](
            started - prior_elapsed, modules["system_swap_bytes"](),
            output / "resources.jsonl",
            limits=limits, interval_s=1.) as watchdog, \
            (output / "updates.jsonl").open("a" if resume else "x") as stream:
        for epoch in range(state["next_epoch"], EPOCHS):
            first = state["next_batch"] if epoch == state["next_epoch"] else 0
            for batch_index in range(first, len(schedules[epoch])):
                group = schedules[epoch][batch_index]
                before = time.monotonic()
                with watchdog.operation(f"optimizer_update_{state['updates'] + 1}", update=True):
                    step = modules["optimizer_step"](model, optimizer, group)
                state["updates"] += 1
                state["training_elapsed_s"] = prior_elapsed + time.monotonic() - started
                observation = watchdog.sample_and_append("update_complete")
                state["peak_rss_bytes"] = max(state["peak_rss_bytes"], observation["rss_bytes"])
                record = {"update": state["updates"], "epoch": epoch + 1,
                          "batch": batch_index + 1, "ids": [row["id"] for row in group],
                          "seconds": time.monotonic() - before,
                          "training_elapsed_s": state["training_elapsed_s"], **step}
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                state["epoch_loss_sum"] += step["loss"] * step["nonpadding_target_tokens"]
                state["epoch_token_sum"] += step["nonpadding_target_tokens"]
                if batch_index + 1 == len(schedules[epoch]):
                    state["epochs_report"].append({
                        "epoch": epoch + 1,
                        "target_normalized_ctc_loss": (
                            state["epoch_loss_sum"] / state["epoch_token_sum"]),
                        "tokens": state["epoch_token_sum"]})
                    state.update(next_epoch=epoch + 1, next_batch=0,
                                 epoch_loss_sum=0., epoch_token_sum=0)
                else:
                    state.update(next_epoch=epoch, next_batch=batch_index + 1)
                if state["updates"] % CHECKPOINT_INTERVAL == 0 or state["next_batch"] == 0:
                    modules["save_checkpoint"](
                        output, model, optimizer, identity, state, stream)
                if state["updates"] % CHECKPOINT_INTERVAL == 0:
                    print(json.dumps({"phase": "ctc_training", "arm": arm,
                                      "update": state["updates"],
                                      "elapsed_s": state["training_elapsed_s"]}), flush=True)
        final_resource = watchdog.sample_and_append("final")
    expected_updates = sum(len(groups) for groups in schedules)
    require(state["updates"] == expected_updates and state["next_epoch"] == EPOCHS,
            "CTC training schedule did not complete exactly once")
    verify_hashes(manifest["source_hashes"])
    model_path = output / "model.pt"
    model.save(model_path, {"candidate_only": True, "production_changed": False,
                            "arm": arm, "prepared_sha256": prepared_sha,
                            "probe_sha256": identity["probe_sha256"], "recipe": RECIPE})
    final_hash = tensor_hash(model.state_dict())
    restored = modules["SentenceCTC"].load(model_path)
    require(tensor_hash(restored.state_dict()) == final_hash,
            "saved CTC tensor differs after reload")
    report = {"schema": "rr-ctc-retrain-training-v1", "completed": True,
              "candidate_only": True, "production_changed": False, "arm": arm,
              "prepared_sha256": prepared_sha, "probe_sha256": identity["probe_sha256"],
              "recipe": RECIPE, "limits": limits, "config": asdict(config),
              "initial_tensor_sha256": initial_hash, "final_tensor_sha256": final_hash,
              "checkpoint_sha256": sha256_file(model_path),
              "metadata_sha256": sha256_file(model_path.with_suffix(".json")),
              "updates_completed": state["updates"], "expected_updates": expected_updates,
              "epochs": state["epochs_report"],
              "training_elapsed_s": state["training_elapsed_s"],
              "final_resource": final_resource, "test_split_read": False,
              "evaluation_performed": False, "resumed": resume,
              "discarded_tail_updates": state.get("discarded_tail_updates", 0)}
    report_sha = write_json(output / "report.json", report)
    write_json(output / "completion.json", {"completed": True,
               "report_sha256": report_sha,
               "checkpoint_sha256": report["checkpoint_sha256"]})
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--output", type=Path, required=True)
    for name in ("probe", "train"):
        command = commands.add_parser(name)
        command.add_argument("--prepared", type=Path, required=True)
        command.add_argument("--arm", choices=sorted(ARMS), required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--max-seconds", type=float, default=LIMITS["whole_seconds"])
        if name == "train":
            command.add_argument("--probe", type=Path, required=True)
            command.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output)
    elif args.command == "probe":
        result = run_probe(args.prepared, args.arm, args.output, args.max_seconds)
    else:
        result = run_train(args.prepared, args.probe, args.arm, args.output,
                           resume=args.resume, max_seconds=args.max_seconds)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
