"""Bounded TRAIN-only matched-source versus deranged-source capacity control.

Both serial arms begin from the same verified stage-A model, AdamW state, and
random state.  The only treatment difference is whether each target is paired
with its own source or the predeclared length-quartile donor source.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
from datetime import datetime, timezone


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import continue_pko_fit as stage
from scripts import evaluate_pko_generation as generation
from scripts import preflight_pko_seq2seq as preflight
from scripts import probe_pko_source_conditioning as diagnostic
from scripts import train_pko_seq2seq as trainer


SCHEMA = "pko-capacity-control-v1-20260921"
PARENT = ROOT / ".omx/artifacts/pko-unseen-fit-v1"
INNER_DIAGNOSTIC = ROOT / ".omx/artifacts/pko-source-conditioning-v1"
DEFAULT_SELECTION = ROOT / ".omx/artifacts/pko-capacity-selection-v1.json"
SELECTION_PROOF = ROOT / ".omx/artifacts/pko-capacity-selection-verification-v1.json"
PLAN = ROOT / ".omx/plans/pko-capacity-control-v1.md"
ROWS = 64
CYCLES = 16
CHECKPOINT_CYCLES = {8, 16}
ACCUMULATION = 8
UPDATES_PER_CYCLE = 8
UPDATES_PER_ARM = 128
MICROSTEPS_PER_ARM = 1_024
CONDITIONING_CYCLES = {4, 8, 16}
LIMITS = {"rss_bytes": 6 * 1024**3, "swap_growth_bytes": 512 * 1024**2,
          "whole_seconds": 1200.0, "update_seconds": 120.0}
EXPECTED_PARENT_PARAMETER_SHA256 = (
    "ab5ffa8b45993465f9dcc03bdc45c2bdf4624c1cf174d8a4f9eb9cc849f64886")
EXPECTED_CATEGORY_COUNTS = {
    "liaison": 8, "nasal_liquid_assimilation": 8, "tense_consonants": 8,
    "vowel_sequences": 8, "copula_assimilation": 4, "palatalization": 4,
    "english_particle": 8, "protected_token": 8,
    "natural_sentence_context": 4, "natural_sentence_oov_growth": 2,
    "natural_sentence_oov_growth_v3": 2,
}

STATIC_PINS = {
    Path(stage.__file__): "7eeeb7843d9128e622223cbae5165e770f7e8fb0a4ff23b1ba8fb220721abdd1",
    Path(trainer.__file__): "7cb34f7cde2ce8a493e2f0bf75793329e63096f8baea9ac42fc7f02a55663113",
    Path(preflight.__file__): "89bc4bd2a9875ace9eb010ca443168df4db84ed34772342dd55923e2ab7518f2",
    Path(diagnostic.__file__): "4d27624169c56396d62da9087f6414990a80c3557196ca5bc39ee24a826c0bbb",
    Path(generation.__file__): "e545cd6e193c905c5ff1db985653783f84839644e572f1e592bf70808e90e640",
    PARENT / "report.json": "64a8ddb010d34aa98ac15e409b337986801bc211e1bccf2c70dcfd131a0c1cf8",
    PARENT / "independent-verification.json":
        "9aae0ef2dc54d5a08228b16d384157fdf3a7bf024a38cf46b677bd273e7aac8c",
    PARENT / "checkpoint.json":
        "d5fdbeacad5876b45087eb81262c9b9ec297d3e54777da681451b4dd0c1bf550",
    PARENT / "checkpoint-optimizer-0.pt":
        "98a254c0ec66369754d2ea421bde4627e60c0963cc1b0c3408ec2fdbe433a672",
    PARENT / "checkpoint-model-0/config.json":
        "a7e58e092c4bd9b6fa63bdac420e47f56bc8628971b91c07a1d20d211f9bb072",
    PARENT / "checkpoint-model-0/generation_config.json":
        "8145d7eecabff8e16a9876617a6d52728e9b8fbe24c426e6bf9ebbd6bfb87737",
    PARENT / "checkpoint-model-0/model.safetensors":
        "17c5204cf6d3331dd1f503bd3e83c3431e51c95be75b1ef83ec528157872ed58",
    PARENT / "checkpoint-model-0/special_tokens_map.json":
        "9dd2846a446460500ae8204f70afef81847b1adb817cfbd42b2cc4c8d98e682d",
    PARENT / "checkpoint-model-0/tokenizer.json":
        "6445b45a7e5e70cc8a6fa0f217075c02bf83dc6c3045628e2f3beb08ab9ea924",
    PARENT / "checkpoint-model-0/tokenizer_config.json":
        "610822b34a6106447a845f5bafd36c6e45c784cccf45f44669e82dc0c3659175",
    INNER_DIAGNOSTIC / "selection.json":
        "406d517f623a9ec1cb266e967f825c4df4bb261f75ae47adf3a533bd0a3627f6",
    Path(__file__).resolve().with_name("prepare_pko_capacity.py"):
        "dce22f5a9a8370fd58b18f2ff6ee078fec2de1ffa72e513ae753d7266ed273f4",
    SELECTION_PROOF:
        "181606c370271a071ada54a41677ce5b8bc2196fd2ae42adebb1dda0184b9105",
}
SELECTION_SHA256 = "d66200d6d1deda42ff3f357cf68ff8b8817a1af7f8c4ab47d783625ca599a927"
PLAN_SHA256 = "7b0ee31d84c61d10fbd29fd0ff07754feea6bc766c14397b3a8825e92d03bbc6"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    return preflight.sha256_file(Path(path))


def write_json(path: Path, value: dict) -> None:
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict, lock=None) -> None:
    manager = lock if lock is not None else contextlib.nullcontext()
    with manager:
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush(); os.fsync(stream.fileno())


def _tuple_state(value):
    if isinstance(value, list):
        return tuple(_tuple_state(item) for item in value)
    return value


def runtime_initial_state(model, optimizer) -> dict:
    """Compact runtime proof that both arms restored the same actual state."""
    import torch
    steps = []
    for state in optimizer.state.values():
        step = state.get("step")
        if step is not None:
            steps.append(float(step.item() if torch.is_tensor(step) else step))
    python_state = json.dumps(stage._json_state(random.getstate()),
                              separators=(",", ":")).encode()
    return {
        "parameter_sha256": preflight.parameter_fingerprint(model),
        "optimizer_state_entries": len(optimizer.state),
        "optimizer_parameter_groups": len(optimizer.param_groups),
        "optimizer_step_min": min(steps) if steps else None,
        "optimizer_step_max": max(steps) if steps else None,
        "optimizer_unique_steps": sorted(set(steps)),
        "torch_rng_sha256": hashlib.sha256(
            torch.get_rng_state().cpu().numpy().tobytes()).hexdigest(),
        "python_rng_sha256": hashlib.sha256(python_state).hexdigest(),
    }


def validate_selection(payload: dict, fit_by_id: dict[str, dict]) -> list[dict]:
    rows = payload.get("rows")
    require(isinstance(rows, list) and len(rows) == ROWS, "selection must contain 64 rows")
    required = {"id", "source", "target", "group", "category", "donor_id",
                "shuffled_source"}
    require(all(set(row) >= required for row in rows), "selection row fields differ")
    ids = [row["id"] for row in rows]
    donors = [row["donor_id"] for row in rows]
    require(len(set(ids)) == ROWS and len(set(donors)) == ROWS,
            "selection IDs and donors must be unique")
    require(set(ids) == set(donors), "donors must be a permutation of selected FIT rows")
    require(all(row["donor_id"] != row["id"] for row in rows),
            "deranged source must not use its own row")
    require(all(row["id"] in fit_by_id and row["donor_id"] in fit_by_id for row in rows),
            "selection contains a non-FIT row")
    require(all(row["source"] == fit_by_id[row["id"]]["source"]
                and row["target"] == fit_by_id[row["id"]]["target"]
                and row["shuffled_source"] == fit_by_id[row["donor_id"]]["source"]
                for row in rows), "selection text differs from frozen TRAIN")
    by_id = {row["id"]: row for row in rows}
    require(len({row["group"] for row in rows}) == ROWS,
            "selection groups must be unique")
    require(all(row.get("length_bin") in {0, 1, 2, 3}
                and by_id[row["donor_id"]].get("length_bin") == row["length_bin"]
                for row in rows), "donors must stay inside length quartiles")
    counts = {key: 0 for key in EXPECTED_CATEGORY_COUNTS}
    for row in rows:
        require(row["category"] in counts, "unknown selection category")
        counts[row["category"]] += 1
    require(counts == EXPECTED_CATEGORY_COUNTS, "selection category quotas differ")
    return rows


def build_schedule(rows: list[dict], arm: str) -> list[list[dict]]:
    require(arm in {"matched", "deranged"}, "unknown arm")
    scheduled = []
    source_field = "source" if arm == "matched" else "shuffled_source"
    for cycle in range(1, CYCLES + 1):
        cycle_rows = [{"id": row["id"], "source": row[source_field],
                       "target": row["target"], "cycle": cycle,
                       "source_row_id": (row["id"] if arm == "matched"
                                         else row["donor_id"])} for row in rows]
        scheduled.extend(stage.groups(cycle_rows, ACCUMULATION))
    require(len(scheduled) == UPDATES_PER_ARM
            and sum(map(len, scheduled)) == MICROSTEPS_PER_ARM
            and all(len(group) == ACCUMULATION for group in scheduled),
            "capacity schedule dimensions differ")
    return scheduled


def schedule_payload(rows: list[dict]) -> dict:
    matched = build_schedule(rows, "matched")
    deranged = build_schedule(rows, "deranged")
    return {
        "schema": SCHEMA, "captured_before_learning": True,
        "cycles": CYCLES, "rows_per_cycle": ROWS,
        "microsteps_per_arm": MICROSTEPS_PER_ARM,
        "updates_per_arm": UPDATES_PER_ARM, "accumulation": ACCUMULATION,
        "target_order": [row["id"] for row in rows],
        "matched_source_order": [row["id"] for row in rows],
        "deranged_source_order": [row["donor_id"] for row in rows],
        "arms_have_identical_target_order": [
            row["id"] for group in matched for row in group] == [
                row["id"] for group in deranged for row in group],
        "arms": {
            arm: [{"optimizer_update": index,
                   "cycle": (index - 1) // UPDATES_PER_CYCLE + 1,
                   "target_ids": [row["id"] for row in group],
                   "source_row_ids": [row["source_row_id"] for row in group]}
                  for index, group in enumerate(groups, start=1)]
            for arm, groups in (("matched", matched), ("deranged", deranged))
        },
    }


def capacity_decision(baseline: dict, matched: dict, deranged: dict) -> dict:
    """Apply the fixed bounded reconstruction and input-alignment contrast gates."""
    matched_fit = matched["generation"]["fit"]
    deranged_fit = deranged["generation"]["fit"]
    base_fit = baseline["generation"]["fit"]
    matched_gain = matched_fit["whole_exact"] - base_fit["whole_exact"]
    deranged_gain = deranged_fit["whole_exact"] - base_fit["whole_exact"]
    base_condition = baseline["conditioning"]["fit"]
    matched_condition = matched["conditioning"]["cycle_16"]["fit"]
    deranged_condition = deranged["conditioning"]["cycle_16"]["fit"]
    matched_nll = matched_condition["normal_token_nll"]
    deranged_nll = deranged_condition["normal_token_nll"]
    reconstruction_checks = {
        "matched_fit_normal_nll_decreases_at_least_50pct":
            matched_nll <= .5 * base_condition["normal_token_nll"],
        "matched_fit_normal_source_wins_at_least_56_of_64":
            matched_condition["normal_lower_rows"] >= 56,
        "matched_fit_median_margin_reaches_predeclared_floor":
            matched_condition["median_row_delta"]
            >= max(2 * base_condition["median_row_delta"], .1),
        "matched_fit_exact_at_least_32_of_64": matched_fit["whole_exact"] >= 32,
        "matched_fit_eos_at_least_60_of_64": matched_fit["eos_complete"] >= 60,
    }
    contrast_checks = {
        "matched_fit_normal_nll_at_least_0_1_below_deranged":
            matched_nll + .1 <= deranged_nll,
        "matched_fit_exact_above_deranged":
            matched_fit["whole_exact"] > deranged_fit["whole_exact"],
    }
    transfer = {
        "baseline_inner_exact": baseline["generation"]["inner"]["whole_exact"],
        "matched_inner_exact": matched["generation"]["inner"]["whole_exact"],
        "deranged_inner_exact": deranged["generation"]["inner"]["whole_exact"],
        "matched_inner_exact_change": matched["generation"]["inner"]["whole_exact"]
            - baseline["generation"]["inner"]["whole_exact"],
        "matched_inner_normal_nll_change":
            matched["conditioning"]["cycle_16"]["inner"]["normal_token_nll"]
            - baseline["conditioning"]["inner"]["normal_token_nll"],
    }
    return {"small_sample_reconstruction_gate": all(reconstruction_checks.values()),
            "input_alignment_contrast_gate": all(contrast_checks.values()),
            "supports_bounded_alignment_learnability":
                all(reconstruction_checks.values()) and all(contrast_checks.values()),
            "reconstruction_checks": reconstruction_checks,
            "contrast_checks": contrast_checks, "transfer_observation": transfer,
            "matched_fit_exact_gain": matched_gain,
            "deranged_fit_exact_gain": deranged_gain,
            "matched_minus_deranged_fit_exact":
                matched_fit["whole_exact"] - deranged_fit["whole_exact"],
            "matched_minus_deranged_fit_normal_nll": matched_nll - deranged_nll,
            "interpretation_limit": (
                "Does not rule out target memorization or the incorrect-mapping "
                "confound; inner64 is a separate TRAIN-only transfer observation."),
            "runtime_promotion": False}


class CapacityWatchdog(preflight.ResourceWatchdog):
    def _monitor(self):
        while not self.stop_event.wait(self.interval_s):
            try:
                stage.append_jsonl(self.trace, self._sample("watchdog"), self.lock)
            except BaseException as error:
                self.failure = f"{type(error).__name__}: {error}"
                write_json(self.trace.parent / "report.json", {
                    "schema": SCHEMA, "completed": False,
                    "error_type": type(error).__name__, "error": str(error),
                    "watchdog_failure": self.failure, "dev_opened": False,
                    "test_opened": False, "production_changed": False,
                    "runtime_promotion": False})
                os.kill(os.getpid(), signal.SIGUSR1)
                if not self.stop_event.wait(5.0):  # pragma: no cover
                    os._exit(124)
                return


def prepare(selection_path: Path = DEFAULT_SELECTION) -> dict:
    require(SELECTION_SHA256 != "PENDING" and PLAN_SHA256 != "PENDING",
            "capacity selection and plan are not frozen")
    pins = dict(STATIC_PINS)
    pins[Path(selection_path)] = SELECTION_SHA256
    pins[PLAN] = PLAN_SHA256
    for path, digest in pins.items():
        require(path.is_file() and sha256_file(path) == digest,
                f"frozen capacity input differs: {path}")
    for path, digest in preflight.EXPECTED_FILES.items():
        if path != preflight.BASE / "dev.jsonl":
            require(path.is_file() and sha256_file(path) == digest,
                    f"pinned TRAIN-only input differs: {path}")
            pins[path] = digest

    parent_report = json.loads((PARENT / "report.json").read_text())
    parent_proof = json.loads((PARENT / "independent-verification.json").read_text())
    checkpoint = json.loads((PARENT / "checkpoint.json").read_text())
    selection_proof = json.loads(SELECTION_PROOF.read_text())
    require(selection_proof.get("verified") is True
            and selection_proof.get("artifact", {}).get("sha256") == SELECTION_SHA256
            and selection_proof.get("source_sha256")
            == STATIC_PINS[Path(__file__).resolve().with_name("prepare_pko_capacity.py")],
            "selection proof does not bind the frozen artifact and helper")
    require(parent_report.get("completed") is True
            and parent_proof.get("verified") is True
            and parent_proof.get("identity", {}).get("report_sha256")
            == STATIC_PINS[PARENT / "report.json"], "parent proof is incomplete")
    require(parent_report.get("model", {}).get("parameter_sha256")
            == EXPECTED_PARENT_PARAMETER_SHA256
            == parent_report.get("model", {}).get("restored_parameter_sha256"),
            "parent parameter identity differs")
    require(checkpoint.get("final") is True and checkpoint.get("slot") == 0
            and checkpoint.get("model_dir") == "checkpoint-model-0"
            and checkpoint.get("optimizer_path") == "checkpoint-optimizer-0.pt"
            and checkpoint.get("optimizer_sha256")
            == STATIC_PINS[PARENT / "checkpoint-optimizer-0.pt"],
            "parent final checkpoint differs")
    observed_model_files = {p.name: sha256_file(p) for p in sorted(
        (PARENT / checkpoint["model_dir"]).iterdir()) if p.is_file()}
    require(observed_model_files == checkpoint.get("model_files"),
            "parent checkpoint model files differ")

    fit, inner, partition_proof = trainer.load_train_partition()
    fit_by_id = {row["id"]: row for row in fit}
    inner_by_id = {row["id"]: row for row in inner}
    selection = json.loads(Path(selection_path).read_text())
    require(selection.get("completed") is True
            and selection.get("fit_only") is True
            and selection.get("model_loaded") is False
            and selection.get("dev_opened") is False
            and selection.get("test_opened") is False
            and selection.get("rows_count") == ROWS
            and selection.get("unique_groups") == ROWS,
            "selection authorization metadata differs")
    rows = validate_selection(selection, fit_by_id)
    inner_selection = json.loads((INNER_DIAGNOSTIC / "selection.json").read_text())
    inner_pairs = inner_selection.get("pairs")
    require(isinstance(inner_pairs, list) and len(inner_pairs) == ROWS
            and all(row["id"] in inner_by_id for row in inner_pairs),
            "frozen inner64 selection differs")
    return {
        "parent_report": parent_report, "checkpoint": checkpoint,
        "fit_rows": rows, "fit_targets": {row["id"]: row["target"] for row in rows},
        "inner_rows": inner_pairs,
        "inner_targets": {row["id"]: inner_by_id[row["id"]]["target"]
                          for row in inner_pairs},
        "schedules": {arm: build_schedule(rows, arm)
                      for arm in ("matched", "deranged")},
        "schedule": schedule_payload(rows), "partition_proof": partition_proof,
        "pins": {preflight.path_key(path): digest for path, digest in sorted(
            pins.items(), key=lambda item: str(item[0]))},
    }


def prepare_output(output: Path) -> Path:
    output = Path(output); output.mkdir(parents=True, exist_ok=False); return output


def load_initial(authorization: dict):
    import torch
    from transformers import T5ForConditionalGeneration, T5TokenizerFast

    checkpoint = authorization["checkpoint"]
    model_dir = PARENT / checkpoint["model_dir"]
    tokenizer = T5TokenizerFast.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False)
    preflight.adapt_lossless_tokenizer(tokenizer)
    model = T5ForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        weights_only=True, torch_dtype="float32")
    require(preflight.parameter_fingerprint(model) == EXPECTED_PARENT_PARAMETER_SHA256,
            "loaded parent parameters differ")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    payload = torch.load(PARENT / checkpoint["optimizer_path"], weights_only=True)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng_state"])
    random.setstate(_tuple_state(checkpoint["python_random_state"]))
    return model, optimizer, tokenizer


def conditioning_capture(model, tokenizer, rows, targets, watchdog, path, stage_name):
    return stage.conditioning_capture(
        model, tokenizer, rows, targets, watchdog, path, stage=stage_name)


def generation_capture(model, tokenizer, rows, targets, watchdog, path, stage_name):
    captures, raw_sha = stage.generation_capture(
        model, tokenizer, rows, watchdog, path, stage=stage_name)
    return stage.score_generation(captures, targets), raw_sha


def save_arm_checkpoint(output: Path, arm: str, cycle: int, model, optimizer,
                        tokenizer, identity: dict, state: dict) -> dict:
    import torch
    arm_dir = output / arm
    arm_dir.mkdir(exist_ok=True)
    model_dir = arm_dir / f"checkpoint-cycle-{cycle}"
    model_dir.mkdir(exist_ok=True)
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    files = {p.name: sha256_file(p) for p in sorted(model_dir.iterdir()) if p.is_file()}
    optimizer_path = arm_dir / f"optimizer-cycle-{cycle}.pt"
    temporary = optimizer_path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        torch.save({"optimizer": optimizer.state_dict(),
                    "torch_rng_state": torch.get_rng_state()}, stream)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, optimizer_path)
    torch_rng_sha = hashlib.sha256(
        torch.get_rng_state().cpu().numpy().tobytes()).hexdigest()
    pointer = {"schema": SCHEMA, "arm": arm, "cycle": cycle,
               "model_dir": str(model_dir.relative_to(output)),
               "model_files": files,
               "optimizer_path": str(optimizer_path.relative_to(output)),
               "optimizer_sha256": sha256_file(optimizer_path),
               "torch_rng_sha256": torch_rng_sha,
               "parameter_sha256": preflight.parameter_fingerprint(model),
               "python_random_state": stage._json_state(random.getstate()),
               "state": dict(state), "identity": identity,
               "saved_at_utc": datetime.now(timezone.utc).isoformat()}
    write_json(arm_dir / f"checkpoint-cycle-{cycle}.json", pointer)
    return pointer


def run_arm(arm: str, authorization: dict, output: Path, watchdog, identity: dict) -> dict:
    require(arm in {"matched", "deranged"}, "unknown arm")
    with watchdog.operation(f"{arm}:load_initial", update=True):
        model, optimizer, tokenizer = load_initial(authorization)
    initial_runtime = runtime_initial_state(model, optimizer)
    initial_parameter_sha = initial_runtime["parameter_sha256"]
    initial_tokenizer_sha = preflight.tokenizer_fingerprint(tokenizer)
    state = {"cycles": 0, "microsteps": 0, "optimizer_updates": 0,
             "target_tokens": 0, "loss_sum": 0.0}
    conditioning = {}
    checkpoints = {}
    updates_path = output / f"{arm}-updates.jsonl"
    training_started = time.monotonic()
    for update_index, group in enumerate(authorization["schedules"][arm], start=1):
        with watchdog.operation(f"{arm}:optimizer_update_{update_index}", update=True):
            update = stage.optimizer_update(model, optimizer, tokenizer, group)
        cycle = (update_index - 1) // UPDATES_PER_CYCLE + 1
        state["cycles"] = cycle; state["microsteps"] += len(group)
        state["optimizer_updates"] = update_index
        state["target_tokens"] += update["target_tokens"]
        state["loss_sum"] += update["token_weighted_loss"] * update["target_tokens"]
        update.update({"arm": arm, "cycle": cycle,
                       "optimizer_update": update_index,
                       "source_row_ids": [row["source_row_id"] for row in group]})
        append_jsonl(updates_path, update)
        if update_index % UPDATES_PER_CYCLE == 0 and cycle in CONDITIONING_CYCLES:
            cycle_result = {}
            for name in ("fit", "inner"):
                rows = authorization[f"{name}_rows"]
                targets = authorization[f"{name}_targets"]
                summary, _ = conditioning_capture(
                    model, tokenizer, rows, targets, watchdog,
                    output / f"{arm}-conditioning-{name}-cycle-{cycle}.jsonl",
                    f"{arm}:cycle_{cycle}:{name}")
                cycle_result[name] = summary
                cycle_result[name]["capture_sha256"] = sha256_file(
                    output / f"{arm}-conditioning-{name}-cycle-{cycle}.jsonl")
            conditioning[f"cycle_{cycle}"] = cycle_result
        if update_index % UPDATES_PER_CYCLE == 0 and cycle in CHECKPOINT_CYCLES:
            with watchdog.operation(f"{arm}:checkpoint_cycle_{cycle}", update=True):
                checkpoints[f"cycle_{cycle}"] = save_arm_checkpoint(
                    output, arm, cycle, model, optimizer, tokenizer, identity, state)
    training_elapsed = time.monotonic() - training_started
    require(state["cycles"] == CYCLES and state["microsteps"] == MICROSTEPS_PER_ARM
            and state["optimizer_updates"] == UPDATES_PER_ARM,
            f"{arm} did not finish its fixed schedule")

    generation_scores = {}; generation_shas = {}; generation_closure_shas = {}
    for name in ("fit", "inner"):
        score, capture_sha = generation_capture(
            model, tokenizer, authorization[f"{name}_rows"],
            authorization[f"{name}_targets"], watchdog,
            output / f"{arm}-generation-{name}.jsonl", f"{arm}:final:{name}")
        generation_scores[name] = score; generation_shas[name] = capture_sha
        generation_closure_shas[name] = sha256_file(
            output / f"{arm}-generation-{name}.complete.json")
    final_parameter_sha = preflight.parameter_fingerprint(model)
    final_pointer = checkpoints["cycle_16"]
    require(final_pointer["parameter_sha256"] == final_parameter_sha,
            f"{arm} final checkpoint parameters differ")
    del model, optimizer, tokenizer
    gc.collect()

    from transformers import T5ForConditionalGeneration, T5TokenizerFast
    with watchdog.operation(f"{arm}:reload_final", update=True):
        model_dir = output / final_pointer["model_dir"]
        restored = T5ForConditionalGeneration.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False,
            weights_only=True, torch_dtype="float32")
        restored_tokenizer = T5TokenizerFast.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False)
        preflight.adapt_lossless_tokenizer(restored_tokenizer)
        final_optimizer_payload = __import__("torch").load(
            output / final_pointer["optimizer_path"], weights_only=True)
        restored_optimizer = __import__("torch").optim.AdamW(
            restored.parameters(), lr=1e-4, weight_decay=.01)
        restored_optimizer.load_state_dict(final_optimizer_payload["optimizer"])
    restored_sha = preflight.parameter_fingerprint(restored)
    require(restored_sha == final_parameter_sha, f"{arm} reload parameters differ")
    restored_tokenizer_sha = preflight.tokenizer_fingerprint(restored_tokenizer)
    require(restored_tokenizer_sha == initial_tokenizer_sha,
            f"{arm} reloaded tokenizer policy differs")
    restored_rng_sha = hashlib.sha256(
        final_optimizer_payload["torch_rng_state"].cpu().numpy().tobytes()).hexdigest()
    require(restored_rng_sha == final_pointer["torch_rng_sha256"],
            f"{arm} checkpoint RNG differs")
    for probe in (authorization["fit_rows"][0]["source"],
                  authorization["fit_rows"][0]["target"], "㈜3.3㎡14㎝",
                  "literal<extra_id_0>tag", "<pad>", "한글"):
        ids = preflight._token_ids(restored_tokenizer, probe)
        require(restored_tokenizer.decode(ids, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False) == probe,
                f"{arm} reloaded tokenizer is not lossless")
    del restored, restored_tokenizer, restored_optimizer, final_optimizer_payload
    gc.collect()
    return {
        "initial_parameter_sha256": initial_parameter_sha,
        "initial_tokenizer_sha256": initial_tokenizer_sha,
        "initial_runtime_state": initial_runtime,
        "final_parameter_sha256": final_parameter_sha,
        "restored_parameter_sha256": restored_sha,
        "reload_parameters_exact": True,
        "tokenizer_policy_reload_exact": True,
        "optimizer_reload_valid": True,
        "rng_reload_exact": True,
        "initial_optimizer_sha256": authorization["checkpoint"]["optimizer_sha256"],
        "initial_rng_source": "frozen parent checkpoint optimizer payload and pointer",
        "training": {"cycles": CYCLES, "microsteps": state["microsteps"],
                     "optimizer_updates": state["optimizer_updates"],
                     "accumulation": ACCUMULATION, "target_tokens": state["target_tokens"],
                     "token_weighted_loss": state["loss_sum"] / state["target_tokens"],
                     "elapsed_s": training_elapsed,
                     "updates_sha256": sha256_file(updates_path)},
        "conditioning": conditioning,
        "generation": generation_scores,
        "generation_capture_shas": generation_shas,
        "generation_closure_marker_shas": generation_closure_shas,
        "checkpoints": {key: {**pointer,
            "pointer_sha256": sha256_file(output / arm / f"checkpoint-cycle-{pointer['cycle']}.json")}
            for key, pointer in checkpoints.items()},
    }


def run(selection_path: Path, output: Path) -> dict:
    import torch

    started = time.monotonic(); baseline_swap = preflight.system_swap_bytes()
    authorization = prepare(selection_path)
    output = prepare_output(output)
    lock = (output / ".run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close(); raise RuntimeError("capacity output is already owned")
    source_sha = sha256_file(Path(__file__))
    source_snapshot = output / "capacity-source.py"
    shutil.copyfile(Path(__file__), source_snapshot)
    shutil.copyfile(selection_path, output / "selection.json")
    write_json(output / "learning-schedule.json", authorization["schedule"])
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    identity = {"seed": trainer.SEED, "rows": ROWS, "cycles": CYCLES,
                "microsteps_per_arm": MICROSTEPS_PER_ARM,
                "updates_per_arm": UPDATES_PER_ARM,
                "parent_parameter_sha256": EXPECTED_PARENT_PARAMETER_SHA256,
                "pins": {**authorization["pins"],
                         preflight.path_key(Path(__file__)): source_sha}}
    try:
        with CapacityWatchdog(started, baseline_swap, output / "resources.jsonl",
                              limits=LIMITS, interval_s=5.0) as watchdog:
            with watchdog.operation("baseline:load_initial", update=True):
                model, optimizer, tokenizer = load_initial(authorization)
            del optimizer
            baseline_conditioning = {}
            baseline_generation = {}
            baseline_capture_shas = {"conditioning": {}, "generation": {}}
            for name in ("fit", "inner"):
                summary, _ = conditioning_capture(
                    model, tokenizer, authorization[f"{name}_rows"],
                    authorization[f"{name}_targets"], watchdog,
                    output / f"baseline-conditioning-{name}.jsonl",
                    f"baseline:{name}")
                baseline_conditioning[name] = summary
                baseline_capture_shas["conditioning"][name] = sha256_file(
                    output / f"baseline-conditioning-{name}.jsonl")
                score, capture_sha = generation_capture(
                    model, tokenizer, authorization[f"{name}_rows"],
                    authorization[f"{name}_targets"], watchdog,
                    output / f"baseline-generation-{name}.jsonl",
                    f"baseline:{name}")
                baseline_generation[name] = score
                baseline_capture_shas["generation"][name] = capture_sha
                baseline_capture_shas.setdefault("generation_closure", {})[name] = sha256_file(
                    output / f"baseline-generation-{name}.complete.json")
            del model, tokenizer
            gc.collect()
            baseline = {"conditioning": baseline_conditioning,
                        "generation": baseline_generation,
                        "capture_shas": baseline_capture_shas}

            arms = {}
            for arm in ("matched", "deranged"):
                arms[arm] = run_arm(arm, authorization, output, watchdog, identity)
            decision = capacity_decision(baseline, arms["matched"], arms["deranged"])
            require(arms["matched"]["initial_parameter_sha256"]
                    == arms["deranged"]["initial_parameter_sha256"]
                    == EXPECTED_PARENT_PARAMETER_SHA256,
                    "arms did not start from identical parameters")
            require(arms["matched"]["initial_runtime_state"]
                    == arms["deranged"]["initial_runtime_state"],
                    "arms did not start from identical optimizer and RNG state")
            require(arms["matched"]["initial_tokenizer_sha256"]
                    == arms["deranged"]["initial_tokenizer_sha256"],
                    "arms did not start from identical tokenizer policy")
            require(sha256_file(Path(__file__)) == source_sha,
                    "capacity source changed during execution")
            for raw_path, digest in identity["pins"].items():
                path = ROOT / raw_path if not Path(raw_path).is_absolute() else Path(raw_path)
                require(path.is_file() and sha256_file(path) == digest,
                        f"capacity input changed during execution: {raw_path}")
            observation = watchdog._sample("final")
        report = {
            "schema": SCHEMA, "completed": True,
            "artifact_status": "train_only_capacity_control",
            "scope": {"fit_rows": ROWS, "inner_rows": ROWS,
                      "arms": ["matched", "deranged"], "cycles_per_arm": CYCLES,
                      "dev_opened": False, "test_opened": False,
                      "generation_source_only": True,
                      "targets_joined_after_each_complete_generation_capture": True},
            "identity": identity,
            "schedule": {**authorization["schedule"],
                         "sha256": sha256_file(output / "learning-schedule.json")},
            "selection_capture": {"path": "selection.json",
                                  "sha256": sha256_file(output / "selection.json"),
                                  "captured_before_learning": True},
            "baseline": baseline, "arms": arms, "decision": decision,
            "interpretation_limit": (
                "This bounded control does not rule out target memorization or the "
                "incorrect-mapping confound; generalization is only observed on the "
                "separate TRAIN-inner64 transfer diagnostic."),
            "promoted": False, "runtime_promotion": False,
            "production_changed": False,
            "resources": {"limits": LIMITS, "baseline_swap_bytes": baseline_swap,
                          "final": observation,
                          "trace_sha256": sha256_file(output / "resources.jsonl")},
            "source_snapshot": {"path": source_snapshot.name,
                                "sha256": sha256_file(source_snapshot)},
            "elapsed_s": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(output / "report.json", report)
        return report
    except BaseException as error:
        if not (output / "report.json").exists():
            write_json(output / "report.json", {
                "schema": SCHEMA, "completed": False,
                "error_type": type(error).__name__, "error": str(error),
                "checkpoint_available": any(output.glob("*/checkpoint-cycle-*.json")),
                "restart_supported": False, "dev_opened": False,
                "test_opened": False, "production_changed": False,
                "runtime_promotion": False,
                "finished_at_utc": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        lock.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        authorization = prepare(args.selection)
        print(json.dumps({"prepared": True, "fit_rows": len(authorization["fit_rows"]),
                          "inner_rows": len(authorization["inner_rows"]),
                          "cycles": CYCLES, "microsteps_per_arm": MICROSTEPS_PER_ARM,
                          "updates_per_arm": UPDATES_PER_ARM,
                          "model_loaded": False, "dev_opened": False,
                          "test_opened": False}, sort_keys=True))
        return
    if args.output is None:
        parser.error("--output is required unless --self-test is used")
    report = run(args.selection, args.output)
    print(json.dumps({"completed": report["completed"],
                      "decision": report["decision"],
                      "elapsed_s": report["elapsed_s"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
