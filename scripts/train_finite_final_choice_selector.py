"""Train the finite-candidate RoBERTa final-choice selector v3 on FIT only."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hanglish.core.neural_converter import file_sha256
from hanglish.models.sentence_ranker import (
    REQUIRED_ARTIFACTS, SentenceRanker, encode_sentence_pairs,
)
from scripts.finite_final_choice_training import (
    EXPECTED, balanced_schedule, build_all_fit,
)
from scripts.train_final_choice_selector import (
    configure_trainable_model, load_fit_role,
    require_finite_parameters, score_parity_delta, selector_loss,
    validate_fit_pair, validate_scores,
)
from scripts.train_sentence_ranker import _optimizer


PLAN_SHA256 = "a9728be30841efa13501c1b04e64adc717d140c9f11e4caac32655f1615d2958"
HELPER_SHA256 = "657d8439343e242028f82eb594d5f8e31a108f6dfbeff74293f5bb5e8c82621e"
POLICY_SHA256 = "80b1baff0c65981809a03ee3158ec6402d1a627803a95e70e390a25f07a17b05"
PREFLIGHT_SHA256 = "020b234ae34b7abb6e34dd6900a16710a256554a810ead7900c0b4172a217538"
SCHEDULE_SHA256 = "ab3237cb483ed8d96145728203a6aa5189d3b47fc52183f586084d11c3079651"
LINEAGE_SHA256 = "16b045d2ac8176336b09df9a4026da04fb1adc5d89157b8180f7e4cbf29e4bec"
EVALUATION_PARTITION_SHA256 = "8f699052b3e8d6e07dcaf59dc8696cc13a8ffb15b4e1bb5df62571e906ae3e61"
EVALUATION_BUILDER_SHA256 = "7c1851c7dcf9f65783f80e5925075b078ba45d50d6b59d81fd31f44772bced2b"
WARM_METADATA_SHA256 = "29ad7b22c2d01e0caba34b3489aa909778359ea7025926ef0935e88bd0caed9e"
WARM_WEIGHT_SHA256 = "7e687917ba496d2cec61907c64bafdf88857d471fac2f7f43bfab86c687c6f0a"
WARM_REPORT_SHA256 = "0c1400b5af87a172c5e1825ef82dfb1c589b8fde4263874149cd3bdd06e1bb91"
RUNTIME_VERSION = "neural-1-867f13596ed5"
FIT_MANIFESTS = {
    "natural_fit": "fc22b1635cc496ed70aa757750944726d583d640f2d9f3a598e380f3e262a1c9",
    "challenge_fit": "2b8a6349e0ad091e797d86694c2a5165b81fa906a91c3d540ece9042b1b66194",
}
WARM = ROOT / ".omx/artifacts/roberta-context-ranker-v1/checkpoint"
OUTPUT = ROOT / ".omx/artifacts/finite-final-choice-selector-v3"
RECIPE = {
    "name": "finite-candidate-roberta-final-choice-fit-v3",
    "seed": 23, "epochs": 1, "top_layers": 2,
    "encoder_lr": 1e-5, "head_lr": 5e-5, "weight_decay": .01,
    "gradient_clip_norm": 1., "max_length": 96,
    "inference_batch_size": 8, "group_batch": 4,
    "distillation_weight": 1., "temperature": 2.,
    "all_case_weight": 1., "preserve_per_batch": 3,
    "repair_per_batch": 1, "threads": 2,
    "rss_limit_bytes": 3 * 1024**3, "whole_run_seconds_limit": 1800,
    "expected_examples": 2791, "expected_parent_correct": 2712,
    "expected_parent_error": 79, "expected_optimizer_steps": 904,
}


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path, expected_sha, label):
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha:
        raise ValueError(f"{label} hash differs")
    return raw, json.loads(raw)


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def require_resources(started, stage):
    if (rss_bytes() > RECIPE["rss_limit_bytes"]
            or time.monotonic() - started > RECIPE["whole_run_seconds_limit"]):
        raise RuntimeError(f"finite selector resource bound exceeded during {stage}")


def progress(event, started, **values):
    print(json.dumps({"event": event,
                      "elapsed_seconds": round(time.monotonic() - started, 2),
                      **values}, ensure_ascii=False), flush=True)


def validate_preflight(path, explicit_sha):
    if explicit_sha != PREFLIGHT_SHA256:
        raise ValueError("finite selector requires the frozen FIT preflight")
    raw, report = read_json(path, explicit_sha, "FIT preflight")
    if (report.get("schema") != "finite-final-choice-fit-preflight-v3"
            or report.get("passed") is not True
            or report.get("counts") != {
                "examples": 2791, "parent_correct": 2712, "parent_error": 79,
                "roles": {"challenge_fit": 1444, "natural_fit": 1347},
                "excluded": {"candidate_budget": 13, "no_eligible_positive": 688}}
            or report.get("optimizer_steps_planned") != 904
            or report.get("repair_repeat_histogram") != {"11": 44, "12": 35}
            or report.get("schedule_sha256") != SCHEDULE_SHA256
            or report.get("source_hashes") != report.get("source_hashes_after")
            or report.get("model_loaded") is not False
            or report.get("optimization_run") is not False
            or report.get("evaluation_cases_read") is not False
            or report.get("development_cases_read") is not False
            or report.get("final_cases_read") is not False
            or report.get("deployed") is not False):
        raise ValueError("finite selector FIT preflight contract differs")
    return raw, report


def validate_lineage(path, explicit_sha):
    if explicit_sha != LINEAGE_SHA256:
        raise ValueError("finite selector requires the frozen warm lineage report")
    raw, report = read_json(path, explicit_sha, "warm lineage")
    if (report.get("schema") != "finite-final-choice-warm-optimizer-lineage-v3"
            or report.get("passed") is not True
            or report.get("warm_optimizer_reserve_overlap_ids") != 0
            or report.get("warm_optimizer_reserve_overlap_groups") != 0
            or report.get("fit_reserve_overlap_ids") != 0
            or report.get("fit_reserve_overlap_groups") != 0
            or report.get("input_hashes_before") != report.get("input_hashes_after")
            or report.get("evaluation_cases_read") is not False
            or report.get("development_cases_read") is not False
            or report.get("final_cases_read") is not False
            or report.get("model_loaded") is not False):
        raise ValueError("warm optimizer lineage contract differs")
    return raw, report


def validate_evaluation_partition(path, explicit_sha, natural_rows):
    if (not isinstance(explicit_sha, str) or len(explicit_sha) != 64
            or set(explicit_sha) - set("0123456789abcdef")):
        raise ValueError("explicit finite v3 evaluation partition SHA is required")
    if explicit_sha != EVALUATION_PARTITION_SHA256:
        raise ValueError("finite selector requires the frozen v3 evaluation partition")
    raw, manifest = read_json(path, explicit_sha, "finite evaluation partition")
    ids, groups = manifest.get("eval_ids"), manifest.get("eval_group_ids")
    remaining_ids = manifest.get("remaining_ids")
    remaining_groups = manifest.get("remaining_group_ids")
    sources = manifest.get("source_hashes")
    if (manifest.get("schema") != "finite-final-choice-fresh-evaluation-v3"
            or manifest.get("salt") != "finite-final-choice-v3-2026-09-15"
            or manifest.get("target_rows") != 1024
            or manifest.get("selector_plan_sha256") != PLAN_SHA256
            or type(manifest.get("rows")) is not int or manifest["rows"] < 1024
            or manifest.get("rows") != 1024 or manifest.get("groups") != 785
            or not isinstance(ids, list) or len(ids) != manifest["rows"]
            or not isinstance(groups, list) or len(groups) != manifest["groups"]
            or not isinstance(remaining_ids, list) or not isinstance(remaining_groups, list)
            or any(not isinstance(value, str) or not value
                   for values in (ids, groups, remaining_ids, remaining_groups)
                   for value in values)
            or any(len(values) != len(set(values))
                   for values in (ids, groups, remaining_ids, remaining_groups))
            or set(ids) & set(remaining_ids) or set(groups) & set(remaining_groups)
            or not isinstance(sources, dict) or sources.get(
                ".omx/artifacts/finite-final-choice-plan-v3.md") != PLAN_SHA256
            or sources.get(
                ".omx/artifacts/finite-final-choice-warm-lineage-v3/report.json")
               != LINEAGE_SHA256
            or manifest.get("builder_sha256") != EVALUATION_BUILDER_SHA256
            or manifest.get("warm_lineage_sha256") != LINEAGE_SHA256
            or manifest.get("selection_uses_target_text") is not False
            or manifest.get("model_inference_run") is not False
            or manifest.get("training_run") is not False
            or manifest.get("old_splits_modified") is not False
            or manifest.get("upstream_pretraining_overlap") != "unknown"):
        raise ValueError("finite v3 evaluation partition contract differs")
    fit_ids = {row["id"] for row in natural_rows}
    fit_groups = {row["group_id"] for row in natural_rows}
    if fit_ids & set(ids) or fit_groups & set(groups):
        raise ValueError("finite v3 evaluation partition overlaps natural FIT")
    return raw, manifest


def validate_warm_checkpoint(path=WARM):
    if file_sha256(path.parent / "report.json") != WARM_REPORT_SHA256:
        raise ValueError("warm historical training report hash differs")
    metadata_raw, metadata = read_json(
        path / "metadata.json", WARM_METADATA_SHA256, "warm checkpoint metadata"
    )
    artifacts = metadata.get("artifact_hashes")
    if (metadata.get("format_version") != 1 or metadata.get("max_length") != 96
            or not isinstance(artifacts, dict) or set(artifacts) != REQUIRED_ARTIFACTS
            or artifacts.get("model.safetensors") != WARM_WEIGHT_SHA256):
        raise ValueError("warm checkpoint metadata contract differs")
    for name, digest in artifacts.items():
        if file_sha256(path / name) != digest:
            raise ValueError(f"warm checkpoint artifact differs: {name}")
    return metadata_raw, metadata, artifacts


def schedule_records(batches):
    records = [{"step": step, "examples": [
        {"role": example.role, "id": example.case_id,
         "parent_correct": example.parent_correct} for example in batch
    ]} for step, batch in enumerate(batches, 1)]
    raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode()
                   for row in records)
    repeats = Counter(f"{batch[0].role}:{batch[0].case_id}" for batch in batches)
    return records, raw, dict(sorted(repeats.items()))


def verify_hashes(hashes):
    for name, digest in hashes.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if file_sha256(path) != digest:
            raise ValueError(f"finite selector input changed: {name}")


def provenance(preflight, preflight_path, preflight_raw, plan_path, plan_raw,
               partition_path, partition_raw, partition, lineage_path,
               lineage_raw, lineage, runner_sha):
    hashes = dict(preflight["source_hashes"])
    for label, values in (("partition", partition["source_hashes"]),
                          ("lineage", lineage["input_hashes_before"])):
        for name, digest in values.items():
            if name in hashes and hashes[name] != digest:
                raise ValueError(f"finite selector {label} source conflict: {name}")
            hashes[name] = digest
    additions = {
        str(preflight_path.resolve().relative_to(ROOT)): sha256_bytes(preflight_raw),
        str(plan_path.resolve().relative_to(ROOT)): sha256_bytes(plan_raw),
        str(partition_path.resolve().relative_to(ROOT)): sha256_bytes(partition_raw),
        str(lineage_path.resolve().relative_to(ROOT)): sha256_bytes(lineage_raw),
        str((WARM.parent / "report.json").relative_to(ROOT)): WARM_REPORT_SHA256,
        "scripts/partition_finite_selector_holdout.py": partition["builder_sha256"],
        "scripts/finite_final_choice_training.py": HELPER_SHA256,
        "scripts/finite_final_choice_policy.py": POLICY_SHA256,
        "scripts/train_finite_final_choice_selector.py": runner_sha,
    }
    for name, digest in additions.items():
        if name in hashes and hashes[name] != digest:
            raise ValueError(f"finite selector provenance conflict: {name}")
        hashes[name] = digest
    return dict(sorted(hashes.items()))


@torch.inference_mode()
def evaluate_fit(ranker, examples, group_batch=8):
    ranker.model.eval()
    outcomes = []
    for start in range(0, len(examples), group_batch):
        batch = examples[start:start + group_batch]
        candidates = [candidate for example in batch for candidate in example.candidates]
        contexts = [example.context for example in batch for _ in example.candidates]
        scores = ranker.score_pairs(contexts, candidates)
        validate_scores(scores, len(candidates))
        cursor = 0
        for example in batch:
            group = scores[cursor:cursor + len(example.candidates)]
            cursor += len(example.candidates)
            chosen = min(range(len(group)), key=lambda index: (-group[index], index))
            outcomes.append({
                "id": example.case_id, "role": example.role,
                "parent_correct": example.parent_correct,
                "chosen_finite_index": chosen,
                "chosen_source_index": example.finite_source_indices[chosen],
                "correct": chosen in example.positives,
            })
        if cursor != len(scores):
            raise RuntimeError("batched FIT scores do not match example boundaries")
    return {"rows": len(outcomes), "correct": sum(row["correct"] for row in outcomes),
            "outcomes": outcomes}


def captured_production_summary(natural, challenge, examples):
    rows = [*natural["rows"], *challenge["rows"]]
    eligible = {(example.role, example.case_id) for example in examples}
    return {
        "all_rows": len(rows),
        "all_top1": sum(row["result"] in row["expected"] for row in rows),
        "eligible_rows": len(examples),
        "eligible_top1": sum(
            row["result"] in row["expected"]
            for role, capture in (("natural_fit", natural), ("challenge_fit", challenge))
            for row in capture["rows"] if (role, row["id"]) in eligible),
        "source": "captured production parent responses; not warm RoBERTa scores",
    }


def train(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    runner_source = Path(__file__).read_bytes()
    if args.plan_sha256 != PLAN_SHA256:
        raise ValueError("finite selector requires the frozen v3 plan")
    plan_raw = args.plan.read_bytes()
    if sha256_bytes(plan_raw) != PLAN_SHA256:
        raise ValueError("finite selector plan hash differs")
    preflight_raw, preflight = validate_preflight(
        args.preflight_report, args.preflight_report_sha256
    )
    lineage_raw, lineage = validate_lineage(
        args.lineage_report, args.lineage_report_sha256
    )
    if ({"natural_fit": args.natural_manifest_sha256,
         "challenge_fit": args.challenge_manifest_sha256} != FIT_MANIFESTS):
        raise ValueError("finite selector requires the frozen FIT captures")
    natural = load_fit_role(args.natural_fit, args.natural_manifest_sha256, "natural_fit")
    challenge = load_fit_role(args.challenge_fit, args.challenge_manifest_sha256,
                              "challenge_fit")
    validate_fit_pair(natural, challenge)
    partition_raw, partition = validate_evaluation_partition(
        args.evaluation_partition, args.evaluation_partition_sha256, natural["rows"]
    )
    if args.warm_checkpoint.resolve() != WARM.resolve():
        raise ValueError("finite selector requires the canonical warm checkpoint path")
    _, warm_metadata, warm_artifacts = validate_warm_checkpoint(args.warm_checkpoint)
    before = provenance(preflight, args.preflight_report, preflight_raw,
                        args.plan, plan_raw, args.evaluation_partition,
                        partition_raw, partition, args.lineage_report, lineage_raw,
                        lineage,
                        sha256_bytes(runner_source))
    verify_hashes(before)
    require_resources(started, "pre-model validation")
    progress("input_validation_complete", started)

    torch.set_num_threads(RECIPE["threads"])
    os.environ["OMP_NUM_THREADS"] = str(RECIPE["threads"])
    random.seed(RECIPE["seed"])
    torch.manual_seed(RECIPE["seed"])
    ranker = SentenceRanker.load(args.warm_checkpoint)
    if (ranker.max_length != 96 or ranker.batch_size != 8
            or ranker.metadata != warm_metadata):
        raise ValueError("loaded warm RoBERTa checkpoint differs")
    examples, eligibility = build_all_fit(natural["rows"], challenge["rows"], ranker)
    batches = balanced_schedule(examples)
    records, schedule_raw, repeats = schedule_records(batches)
    if (len(batches) != EXPECTED["optimizer_steps"]
            or sha256_bytes(schedule_raw) != SCHEDULE_SHA256
            or Counter(repeats.values()) != Counter({12: 35, 11: 44})):
        raise ValueError("finite selector schedule differs from frozen preflight")
    captured_parent = captured_production_summary(natural, challenge, examples)
    progress("warm_fit_start", started, examples=len(examples))
    warm_fit = evaluate_fit(ranker, examples)
    require_resources(started, "warm FIT evaluation")
    progress("warm_fit_complete", started, correct=warm_fit["correct"])

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
        encoded = encode_sentence_pairs(ranker.tokenizer, candidates, contexts, 96)
        optimizer.zero_grad(set_to_none=True)
        scores = ranker.model(**encoded).logits.reshape(-1)
        loss = selector_loss(scores, batch, RECIPE)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite finite-selector loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ranker.model.parameters(), 1.)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("nonfinite finite-selector gradients")
        optimizer.step()
        require_finite_parameters(ranker.model)
        losses.append(float(loss.detach()))
        require_resources(started, "training")
        if len(losses) % 64 == 0:
            progress("training_step", started, step=len(losses), loss=losses[-1])
    training_seconds = time.monotonic() - training_started
    if len(losses) != 904 or any(not math.isfinite(value) for value in losses):
        raise RuntimeError("finite selector optimizer steps or losses differ")
    ranker.model.eval()

    bundle = args.output / "checkpoint"
    ranker.model.save_pretrained(bundle, safe_serialization=True)
    ranker.tokenizer.save_pretrained(bundle)
    artifacts = {path.name: file_sha256(path) for path in bundle.iterdir() if path.is_file()}
    if set(artifacts) != REQUIRED_ARTIFACTS:
        raise RuntimeError("finite selector checkpoint artifact set differs")
    metadata = {key: warm_metadata[key] for key in
                ("base_model", "base_revision", "base_source", "license")
                if key in warm_metadata}
    metadata.update(
        format_version=1, created_at=datetime.now(timezone.utc).isoformat(),
        artifact_hashes=artifacts, max_length=96, inference_batch_size=8,
        experiment_schema="finite-candidate-roberta-final-choice-selector-v3",
        parent_checkpoint_sha256=WARM_WEIGHT_SHA256,
        parent_checkpoint_metadata_sha256=WARM_METADATA_SHA256,
        parent_metadata=warm_metadata, parent_runtime_version=RUNTIME_VERSION,
        fit_captures={role: {"path": str(capture["directory"]),
                             "manifest_sha256": FIT_MANIFESTS[role]}
                      for role, capture in (("natural_fit", natural),
                                            ("challenge_fit", challenge))},
        finite_plan={"path": str(args.plan), "sha256": PLAN_SHA256},
        fresh_evaluation_partition={"path": str(args.evaluation_partition),
                                    "sha256": sha256_bytes(partition_raw),
                                    "rows": partition["rows"],
                                    "groups": partition["groups"]},
        warm_lineage_sha256=LINEAGE_SHA256,
        preflight_sha256=PREFLIGHT_SHA256, schedule_sha256=SCHEDULE_SHA256,
        frozen_source_hashes=before, training=RECIPE,
        training_eligibility=eligibility, trainable_layers=layers,
        finite_candidate_view=True, all_case_weight=1.,
        evaluation_cases_read=False, development_data_read=False,
        final_data_read=False, eligible_for_deployment=False, deployed=False,
        library_versions={name: package_version(name) for name in
                          ("torch", "transformers", "tokenizers", "safetensors")},
    )
    (bundle / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    progress("reload_and_saved_fit_start", started)
    saved = SentenceRanker.load(bundle)
    parity_examples = examples[:4]
    memory_scores = [ranker.score_pairs([example.context] * len(example.candidates),
                                        list(example.candidates))
                     for example in parity_examples]
    saved_scores = [saved.score_pairs([example.context] * len(example.candidates),
                                      list(example.candidates))
                    for example in parity_examples]
    parity_delta = score_parity_delta(memory_scores, saved_scores, parity_examples)
    require_finite_parameters(saved.model)
    trained_fit = evaluate_fit(saved, examples)
    require_resources(started, "saved FIT evaluation")

    load_fit_role(args.natural_fit, args.natural_manifest_sha256, "natural_fit")
    load_fit_role(args.challenge_fit, args.challenge_manifest_sha256, "challenge_fit")
    validate_warm_checkpoint(args.warm_checkpoint)
    after = provenance(preflight, args.preflight_report, preflight_raw,
                       args.plan, plan_raw, args.evaluation_partition,
                       partition_raw, partition, args.lineage_report, lineage_raw,
                       lineage,
                       sha256_bytes(runner_source))
    verify_hashes(after)
    if (after != before or file_sha256(schedule_path) != SCHEDULE_SHA256
            or Path(__file__).read_bytes() != runner_source):
        raise RuntimeError("finite selector sources or schedule changed")
    (args.output / "trainer-source.py").write_bytes(runner_source)
    require_resources(started, "terminal report")
    report = {
        "schema": "finite-candidate-roberta-final-choice-selector-v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "FIT-only finite-candidate RoBERTa selector; no deployment decision",
        "recipe": RECIPE, "runtime_version": RUNTIME_VERSION,
        "plan_sha256": PLAN_SHA256,
        "fresh_evaluation_partition": {"path": str(args.evaluation_partition),
            "sha256": sha256_bytes(partition_raw), "rows": partition["rows"],
            "groups": partition["groups"]},
        "fit_manifest_sha256": FIT_MANIFESTS,
        "preflight_sha256": sha256_bytes(preflight_raw),
        "warm_lineage_sha256": LINEAGE_SHA256,
        "warm_checkpoint_metadata_sha256": WARM_METADATA_SHA256,
        "warm_checkpoint_artifact_hashes": warm_artifacts,
        "source_hashes": before, "source_hashes_after": after,
        "checkpoint_sha256": artifacts["model.safetensors"],
        "eligibility": eligibility,
        "captured_production_fit": captured_parent,
        "warm_roberta_fit": warm_fit, "trained_roberta_fit": trained_fit,
        "schedule_ids": records, "schedule_sha256": SCHEDULE_SHA256,
        "repair_repeats": repeats, "optimizer_steps": len(losses),
        "mean_loss": sum(losses) / len(losses),
        "training_seconds": training_seconds,
        "save_load_score_max_delta": parity_delta,
        "peak_rss_bytes": rss_bytes(), "elapsed_seconds": time.monotonic() - started,
        "finite_candidate_view": True, "model_quality_gate_passed": False,
        "evaluation_cases_read": False, "development_data_read": False,
        "final_data_read": False, "eligible_for_deployment": False, "deployed": False,
        "next_gate": "Capture and evaluate the frozen fresh v3 partition without tuning.",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"optimizer_steps": len(losses),
                      "warm_fit": warm_fit["correct"],
                      "trained_fit": trained_fit["correct"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--preflight-report-sha256", required=True)
    parser.add_argument("--lineage-report", type=Path, required=True)
    parser.add_argument("--lineage-report-sha256", required=True)
    parser.add_argument("--evaluation-partition", type=Path, required=True)
    parser.add_argument("--evaluation-partition-sha256", required=True)
    parser.add_argument("--natural-fit", type=Path, required=True)
    parser.add_argument("--natural-manifest-sha256", required=True)
    parser.add_argument("--challenge-fit", type=Path, required=True)
    parser.add_argument("--challenge-manifest-sha256", required=True)
    parser.add_argument("--warm-checkpoint", type=Path, default=WARM)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    train(parser.parse_args())
