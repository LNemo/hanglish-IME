"""Train the FIT-only balanced direct final-choice selector experiment."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hanglish.core.neural_converter import file_sha256
from hanglish.models.sentence_ranker import SentenceRanker, encode_sentence_pairs
from scripts.train_final_choice_selector import (
    PARENT,
    PARENT_WEIGHT,
    RECIPE as BASE_RECIPE,
    build_example,
    configure_trainable_model,
    evaluate_examples,
    load_fit_role,
    merge_hashes,
    provenance_hashes as base_provenance_hashes,
    require_finite_parameters,
    score_parity_delta,
    selector_loss,
    sha256_bytes,
    validate_fit_pair,
    verify_hashes,
)
from scripts.train_natural_sentence_ranker import rss_bytes
from scripts.train_sentence_ranker import _optimizer


OUTPUT = ROOT / ".omx/artifacts/balanced-final-choice-selector-v2"
PLAN_SHA256 = "004990ee8f30c049e03b2bbd38e1b0bdd71fcac3ee4a7ca72c7d6821560adbbc"
EVALUATION_SCHEMA = "balanced-final-choice-fresh-evaluation-v2"
EVALUATION_SALT = "balanced-final-choice-v2-2026-09-15"
EVALUATION_TARGET = 1024
EVALUATION_PARTITION_SHA256 = "57170914d4f14d5cd725ea0fd52a96b68b0859ba83041b93f1ecc09daa7cc8ca"
FIT_MANIFEST_SHA256 = {
    "natural_fit": "fc22b1635cc496ed70aa757750944726d583d640f2d9f3a598e380f3e262a1c9",
    "challenge_fit": "2b8a6349e0ad091e797d86694c2a5165b81fa906a91c3d540ece9042b1b66194",
}
PARTITION_V1_SHA256 = "e997f06558bdb5633446171d35912140fb34b55066c13fa13b668939d6347846"
RAW_TRAIN_SHA256 = "904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b"
PARTITION_BUILDER_SHA256 = "983353a6b8a95aaa5693f506a992e2680f21c31892e0ee8c539a014084ad9335"
BASE_HELPER_SHA256 = "e66d7fdef1461b39a2874654aca6be12487b8dc241405ab9694efbf934f92cc8"
FROZEN_HELPERS = {
    "scripts/train_final_choice_selector.py": BASE_HELPER_SHA256,
    "scripts/final_choice_capture.py":
        "80bad648b1c9d46a9483fafabd61f3b82736853735189697b3d12c6c500926fa",
    "scripts/final_choice_policy.py":
        "3acb1480fce1c490892e003a9677b0ad5354d07e971dd3ca73446c08d69cf717",
    "scripts/prepare_final_choice_data.py":
        "41da4d13418480de47034ba414d7c9f49c6b1473b833f487ece13e87f0eff402",
    "scripts/train_sentence_ranker.py":
        "dfc1009b3877ae9f1889d3cdaf7a00531508a73a37bdb8490226ff33d6eb7fce",
    "scripts/train_natural_sentence_ranker.py":
        "06f0299190b4831c54852fa3cb73a6fe4bcf697f4e6ffe3326d7ef745e3c4a61",
    "scripts/partition_balanced_selector_holdout.py": PARTITION_BUILDER_SHA256,
    "hanglish/models/sentence_ranker.py":
        "72ed3e7223aac215d692f96890ab642cf9c1da82d25c4485c5b901ff832bd1ca",
}
EXPECTED_PARENT_CORRECT = 2415
EXPECTED_PARENT_ERROR = 58
EXPECTED_BATCHES = 805
RECIPE = {
    **BASE_RECIPE,
    "name": "balanced-direct-final-choice-selector-fit-v2",
    "parent_error_weight": 1.,
    "other_weight": 1.,
    "preserve_per_batch": 3,
    "error_per_batch": 1,
    "expected_parent_correct": EXPECTED_PARENT_CORRECT,
    "expected_parent_error": EXPECTED_PARENT_ERROR,
    "expected_optimizer_steps": EXPECTED_BATCHES,
    "schedule": "shuffle every parent-correct once; cycle one reshuffled parent-error per batch",
    "separate_learned_gate": False,
}


def read_plan(path, explicit_sha):
    if (not isinstance(explicit_sha, str) or len(explicit_sha) != 64
            or set(explicit_sha) - set("0123456789abcdef")):
        raise ValueError("an explicit balanced-selector plan SHA-256 is required")
    if explicit_sha != PLAN_SHA256:
        raise ValueError("balanced-selector plan is not the frozen v2 plan")
    raw = path.read_bytes()
    if sha256_bytes(raw) != explicit_sha:
        raise ValueError("balanced-selector plan hash differs")
    return raw


def read_evaluation_partition(path, explicit_sha, natural_fit_rows):
    if (not isinstance(explicit_sha, str) or len(explicit_sha) != 64
            or set(explicit_sha) - set("0123456789abcdef")):
        raise ValueError("an explicit EVAL partition SHA-256 is required")
    if explicit_sha != EVALUATION_PARTITION_SHA256:
        raise ValueError("fresh EVAL partition is not the frozen v2 selection")
    raw = path.read_bytes()
    if sha256_bytes(raw) != explicit_sha:
        raise ValueError("fresh EVAL partition hash differs")
    manifest = json.loads(raw)
    eval_ids, eval_groups = manifest.get("eval_ids"), manifest.get("eval_group_ids")
    remaining_ids = manifest.get("remaining_ids")
    remaining_groups = manifest.get("remaining_group_ids")
    expected_sources = {
        ".omx/artifacts/final-choice-partition-v1/manifest.json": PARTITION_V1_SHA256,
        "data/natural_context/klue-v1.1-pilot-v2/train.jsonl": RAW_TRAIN_SHA256,
        ".omx/artifacts/balanced-final-choice-plan-v2.md": PLAN_SHA256,
    }
    if (manifest.get("schema") != EVALUATION_SCHEMA
            or manifest.get("salt") != EVALUATION_SALT
            or manifest.get("target_rows") != EVALUATION_TARGET
            or manifest.get("selector_plan_sha256") != PLAN_SHA256
            or manifest.get("builder_sha256") != PARTITION_BUILDER_SHA256
            or manifest.get("source_hashes") != expected_sources
            or type(manifest.get("rows")) is not int
            or manifest["rows"] < EVALUATION_TARGET
            or type(manifest.get("groups")) is not int or manifest["groups"] <= 0
            or not isinstance(eval_ids, list) or len(eval_ids) != manifest["rows"]
            or not isinstance(eval_groups, list) or len(eval_groups) != manifest["groups"]
            or not isinstance(remaining_ids, list) or not isinstance(remaining_groups, list)
            or any(not isinstance(value, str) or not value
                   for values in (eval_ids, eval_groups, remaining_ids, remaining_groups)
                   for value in values)
            or len(eval_ids) != len(set(eval_ids))
            or len(eval_groups) != len(set(eval_groups))
            or len(remaining_ids) != len(set(remaining_ids))
            or len(remaining_groups) != len(set(remaining_groups))
            or set(eval_ids) & set(remaining_ids)
            or set(eval_groups) & set(remaining_groups)
            or manifest.get("selection_uses_target_text") is not False
            or manifest.get("model_inference_run") is not False
            or manifest.get("training_run") is not False
            or manifest.get("old_splits_modified") is not False):
        raise ValueError("fresh EVAL partition metadata is malformed")
    if manifest.get("upstream_pretraining_overlap") != "unknown":
        raise ValueError("fresh EVAL partition overlap claim differs")
    fit_ids = {row["id"] for row in natural_fit_rows}
    fit_groups = {row["group_id"] for row in natural_fit_rows}
    if fit_ids & set(eval_ids) or fit_groups & set(eval_groups):
        raise ValueError("fresh EVAL partition overlaps natural FIT")
    return raw, manifest


def balanced_schedule(examples, seed=RECIPE["seed"]):
    """Use every preserve example once and cycle one error into each batch."""
    preserve = [replace(example, loss_weight=1.)
                for example in examples if example.parent_correct]
    errors = [replace(example, loss_weight=1.)
              for example in examples if not example.parent_correct]
    if not preserve:
        raise ValueError("balanced selector requires parent-correct FIT examples")
    if not errors:
        raise ValueError("balanced selector requires parent-error FIT examples")
    identities = [(example.role, example.case_id) for example in examples]
    if len(identities) != len(set(identities)):
        raise ValueError("balanced selector FIT example identities are duplicated")
    preserve_rng = random.Random(seed)
    repair_rng = random.Random(seed)
    preserve_rng.shuffle(preserve)
    batches = []
    error_cycle, error_cursor = [], 0
    for start in range(0, len(preserve), RECIPE["preserve_per_batch"]):
        if error_cursor == len(error_cycle):
            error_cycle = list(errors)
            repair_rng.shuffle(error_cycle)
            error_cursor = 0
        batch = [error_cycle[error_cursor],
                 *preserve[start:start + RECIPE["preserve_per_batch"]]]
        error_cursor += 1
        batches.append(tuple(batch))
    flattened_preserve = [example for batch in batches for example in batch
                          if example.parent_correct]
    if (Counter((example.role, example.case_id) for example in flattened_preserve)
            != Counter((example.role, example.case_id) for example in preserve)
            or any(sum(not example.parent_correct for example in batch) != 1
                   for batch in batches)
            or any(len(batch) != RECIPE["group_batch"]
                   for batch in batches[:-1])
            or not 2 <= len(batches[-1]) <= RECIPE["group_batch"]):
        raise RuntimeError("balanced selector schedule invariants failed")
    return batches


def schedule_record(batches):
    records = [{"step": index + 1,
                "examples": [{"role": example.role, "id": example.case_id,
                              "parent_correct": example.parent_correct}
                             for example in batch]}
               for index, batch in enumerate(batches)]
    raw = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
                   for row in records)
    error_counts = Counter(
        f"{example.role}:{example.case_id}" for batch in batches for example in batch
        if not example.parent_correct
    )
    return records, raw, dict(sorted(error_counts.items()))


def validate_expected_schedule(batches):
    preserve = [example for batch in batches for example in batch if example.parent_correct]
    errors = [example for batch in batches for example in batch if not example.parent_correct]
    if (len(preserve) != EXPECTED_PARENT_CORRECT
            or len({(example.role, example.case_id) for example in preserve})
               != EXPECTED_PARENT_CORRECT
            or len({(example.role, example.case_id) for example in errors})
               != EXPECTED_PARENT_ERROR
            or len(batches) != EXPECTED_BATCHES):
        raise ValueError("balanced selector FIT counts differ from the frozen schedule")


def verify_frozen_helpers():
    for relative, expected in FROZEN_HELPERS.items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"frozen v1 selector helper differs: {relative}")


def balanced_provenance_hashes(natural, challenge, plan_path, plan_sha,
                               evaluation_path, evaluation_sha, runner_sha):
    # The base helper expects its own source hash, not this runner's hash.
    hashes = base_provenance_hashes(natural, challenge, BASE_HELPER_SHA256)
    merge_hashes(hashes, FROZEN_HELPERS, "frozen v1 helper")
    try:
        plan_key = str(plan_path.resolve().relative_to(ROOT))
    except ValueError:
        plan_key = str(plan_path.resolve())
    merge_hashes(hashes, {
        plan_key: plan_sha,
        str(evaluation_path.resolve().relative_to(ROOT)): evaluation_sha,
        "scripts/train_balanced_final_choice_selector.py": runner_sha,
    }, "balanced selector source")
    return dict(sorted(hashes.items()))


def require_resources(started, stage):
    if (rss_bytes() > RECIPE["rss_limit_bytes"]
            or time.monotonic() - started > RECIPE["training_seconds_limit"]):
        raise RuntimeError(f"balanced selector resource bound exceeded during {stage}")


def train(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = Path(__file__).read_bytes()
    source_sha = sha256_bytes(source)
    plan_raw = read_plan(args.plan, args.plan_sha256)
    if ({"natural_fit": args.natural_manifest_sha256,
         "challenge_fit": args.challenge_manifest_sha256} != FIT_MANIFEST_SHA256):
        raise ValueError("balanced selector requires the frozen v1 FIT captures")
    verify_frozen_helpers()
    natural = load_fit_role(args.natural_fit, args.natural_manifest_sha256, "natural_fit")
    evaluation_raw, evaluation = read_evaluation_partition(
        args.evaluation_partition, args.evaluation_partition_sha256, natural["rows"]
    )
    challenge = load_fit_role(args.challenge_fit, args.challenge_manifest_sha256,
                              "challenge_fit")
    validate_fit_pair(natural, challenge)
    before = balanced_provenance_hashes(
        natural, challenge, args.plan, sha256_bytes(plan_raw),
        args.evaluation_partition, sha256_bytes(evaluation_raw), source_sha
    )
    verify_hashes(before)
    if file_sha256(PARENT / "model.safetensors") != PARENT_WEIGHT:
        raise ValueError("balanced selector parent checkpoint differs")
    require_resources(started, "input validation")

    examples, eligibility = [], {}
    for role, capture in (("natural_fit", natural), ("challenge_fit", challenge)):
        role_examples = [replace(example, loss_weight=1.) for row in capture["rows"]
                         if (example := build_example(row, role)) is not None]
        examples.extend(role_examples)
        eligibility[role] = {
            "rows": len(capture["rows"]), "eligible_with_positive": len(role_examples),
            "missing_eligible_positive": sum(
                row["selector_eligible"] and not row["accepted_positive_indices"]
                for row in capture["rows"]),
            "ineligible": sum(not row["selector_eligible"] for row in capture["rows"]),
            "parent_correct": sum(example.parent_correct for example in role_examples),
            "parent_error": sum(not example.parent_correct for example in role_examples),
        }
    batches = balanced_schedule(examples)
    validate_expected_schedule(batches)
    records, schedule_raw, error_repeats = schedule_record(batches)

    torch.set_num_threads(RECIPE["threads"])
    os.environ["OMP_NUM_THREADS"] = str(RECIPE["threads"])
    random.seed(RECIPE["seed"])
    torch.manual_seed(RECIPE["seed"])
    ranker = SentenceRanker.load(PARENT)
    if ranker.max_length != RECIPE["max_length"]:
        raise ValueError("balanced selector parent length differs")
    ranker.batch_size = RECIPE["inference_batch_size"]
    parent = evaluate_examples(ranker, examples)
    require_resources(started, "parent FIT evaluation")

    args.output.mkdir(parents=True)
    schedule_path = args.output / "schedule.jsonl"
    schedule_path.write_bytes(schedule_raw)
    layers = configure_trainable_model(ranker.model)
    optimizer = _optimizer(ranker.model, SimpleNamespace(**RECIPE), head_only=False)
    losses = []
    ranker.model.train()
    training_started = time.monotonic()
    for batch in batches:
        candidates = [candidate for example in batch for candidate in example.candidates]
        contexts = [example.context for example in batch for _ in example.candidates]
        encoded = encode_sentence_pairs(
            ranker.tokenizer, candidates, contexts, RECIPE["max_length"]
        )
        optimizer.zero_grad(set_to_none=True)
        scores = ranker.model(**encoded).logits.reshape(-1)
        loss = selector_loss(scores, batch, RECIPE)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite balanced selector loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            ranker.model.parameters(), RECIPE["gradient_clip_norm"]
        )
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("nonfinite balanced selector gradients")
        optimizer.step()
        require_finite_parameters(ranker.model)
        losses.append(float(loss.detach()))
        require_resources(started, "training")
    training_seconds = time.monotonic() - training_started
    ranker.model.eval()

    bundle = args.output / "checkpoint"
    ranker.model.save_pretrained(bundle, safe_serialization=True)
    ranker.tokenizer.save_pretrained(bundle)
    artifacts = {path.name: file_sha256(path) for path in bundle.iterdir() if path.is_file()}
    parent_metadata = ranker.metadata
    metadata = {key: parent_metadata[key] for key in
                ("base_model", "base_revision", "base_source", "license",
                 "candidate_generator") if key in parent_metadata}
    metadata.update(
        format_version=1, created_at=datetime.now(timezone.utc).isoformat(),
        artifact_hashes=artifacts, max_length=RECIPE["max_length"],
        inference_batch_size=RECIPE["inference_batch_size"],
        parent_checkpoint_sha256=PARENT_WEIGHT, parent_metadata=parent_metadata,
        parent_runtime_version=natural["manifest"]["runtime_version"],
        fit_captures={role: {"path": str(capture["directory"]),
                             "manifest_sha256": sha256_bytes(capture["manifest_raw"])}
                      for role, capture in (("natural_fit", natural),
                                            ("challenge_fit", challenge))},
        balanced_plan={"path": str(args.plan), "sha256": sha256_bytes(plan_raw)},
        fresh_evaluation_partition={"path": str(args.evaluation_partition),
                                    "sha256": sha256_bytes(evaluation_raw),
                                    "rows": evaluation["rows"],
                                    "groups": evaluation["groups"]},
        frozen_source_hashes=before, training=RECIPE,
        training_eligibility=eligibility, trainable_layers=layers,
        schedule_sha256=sha256_bytes(schedule_raw),
        balanced_direct_selector=True, compatible_with_v1_calibrator=False,
        experiment_schema="balanced-final-choice-direct-selector-v2",
        separate_learned_gate=False, calibration_data_read=False,
        holdout_data_read=False, preservation_data_read=False,
        development_data_read=False, final_data_read=False, deployed=False,
        library_versions={name: package_version(name) for name in
                          ("torch", "transformers", "tokenizers", "safetensors")},
    )
    (bundle / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    loaded = SentenceRanker.load(bundle)
    parity_examples = examples[:4]
    memory = [ranker.score_pairs([example.context] * len(example.candidates),
                                 list(example.candidates)) for example in parity_examples]
    saved = [loaded.score_pairs([example.context] * len(example.candidates),
                                list(example.candidates)) for example in parity_examples]
    parity_delta = score_parity_delta(memory, saved, parity_examples)
    require_finite_parameters(loaded.model)
    final = evaluate_examples(loaded, examples)
    require_resources(started, "saved FIT evaluation")

    load_fit_role(args.natural_fit, args.natural_manifest_sha256, "natural_fit")
    load_fit_role(args.challenge_fit, args.challenge_manifest_sha256, "challenge_fit")
    after = balanced_provenance_hashes(
        natural, challenge, args.plan, sha256_bytes(plan_raw),
        args.evaluation_partition, sha256_bytes(evaluation_raw), source_sha
    )
    verify_hashes(after)
    if (after != before or file_sha256(PARENT / "model.safetensors") != PARENT_WEIGHT
            or file_sha256(Path(__file__)) != source_sha
            or file_sha256(schedule_path) != sha256_bytes(schedule_raw)):
        raise RuntimeError("balanced selector sources, parent, or schedule changed")
    source_path = args.output / "trainer-source.py"
    source_path.write_bytes(source)
    if file_sha256(source_path) != source_sha:
        raise RuntimeError("balanced selector source snapshot differs")
    require_resources(started, "post-validation")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "balanced FIT-only direct final-choice selector; no learned gate or deployment decision",
        "experiment_schema": "balanced-final-choice-direct-selector-v2",
        "recipe": RECIPE, "runtime_version": natural["manifest"]["runtime_version"],
        "plan_path": str(args.plan), "plan_sha256": sha256_bytes(plan_raw),
        "fresh_evaluation_partition": {
            "path": str(args.evaluation_partition),
            "sha256": sha256_bytes(evaluation_raw),
            "rows": evaluation["rows"], "groups": evaluation["groups"]},
        "fit_manifest_sha256": {
            "natural_fit": sha256_bytes(natural["manifest_raw"]),
            "challenge_fit": sha256_bytes(challenge["manifest_raw"])},
        "source_hashes": before, "source_hashes_after": after,
        "parent_checkpoint_sha256": PARENT_WEIGHT,
        "checkpoint_sha256": artifacts["model.safetensors"],
        "eligibility": eligibility, "parent_fit": parent, "trained_fit": final,
        "schedule_ids": records, "schedule_sha256": sha256_bytes(schedule_raw),
        "schedule_rows": sum(len(batch) for batch in batches),
        "parent_correct_once": EXPECTED_PARENT_CORRECT,
        "parent_error_unique": EXPECTED_PARENT_ERROR,
        "parent_error_repeats": error_repeats,
        "parent_error_draws": len(batches),
        "parent_error_repeat_min": min(error_repeats.values()),
        "parent_error_repeat_max": max(error_repeats.values()),
        "optimizer_steps": len(losses), "mean_loss": sum(losses) / len(losses),
        "training_seconds": training_seconds,
        "save_load_score_max_delta": parity_delta,
        "peak_rss_bytes": rss_bytes(), "elapsed_seconds": time.monotonic() - started,
        "balanced_direct_selector": True, "compatible_with_v1_calibrator": False,
        "separate_learned_gate": False, "calibration_data_read": False,
        "fresh_evaluation_cases_read": False,
        "holdout_data_read": False, "preservation_data_read": False,
        "development_data_read": False, "final_data_read": False,
        "eligible_for_deployment": False, "deployed": False,
        "next_gate": "Evaluate this frozen direct selector on source-preserving TRAIN holdouts.",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"optimizer_steps": len(losses),
                      "schedule_sha256": report["schedule_sha256"],
                      "parent_fit_correct": parent["correct"],
                      "trained_fit_correct": final["correct"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--evaluation-partition", type=Path, required=True)
    parser.add_argument("--evaluation-partition-sha256", required=True)
    parser.add_argument("--natural-fit", type=Path, required=True)
    parser.add_argument("--natural-manifest-sha256", required=True)
    parser.add_argument("--challenge-fit", type=Path, required=True)
    parser.add_argument("--challenge-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    train(parser.parse_args())
