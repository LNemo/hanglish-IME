"""Bounded continuation of the current word ranker on actual runtime groups."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as package_version
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hanglish.core.neural_converter import file_sha256
from hanglish.models.semantic_ranker import SemanticRanker, encode_pairs
from scripts.prepare_current_context_data import (
    OUTPUT as PREPARED,
    RUNTIME_VERSION,
    source_hashes,
    validate_data,
)
from scripts.train_natural_sentence_ranker import balanced_schedule, rss_bytes
from scripts.train_natural_word_ranker import evaluate_groups, select_groups
from scripts.train_preserved_context_ranker import (
    PreservedExample,
    full_teacher_correct,
    preserved_example,
    ranking_loss,
)
from scripts.train_sentence_ranker import _optimizer, _set_trainable_layers


PARENT = ROOT / "checkpoints/semantic_ranker"
PARENT_WEIGHT = "54ca7d24b2bf17d1ba8f5181476ce91032acc94149a2baf401fe47e859f2907c"
OUTPUT = ROOT / ".omx/artifacts/current-word-ranker-v1"
RECIPE = {
    "name": "current-runtime-word-top2-weighted-correction-v1",
    "seed": 20260915,
    "epochs": 1,
    "top_layers": 2,
    "head_lr": 5e-5,
    "encoder_lr": 1e-5,
    "weight_decay": .01,
    "group_batch": 4,
    "candidate_limit": 8,
    "source_group_limit": 16384,
    "selection_salt": "current-runtime-word-natural-groups-v1",
    "natural_retention_ratio": "1:1",
    "base_weight": .15,
    "distillation_weight": 1.,
    "temperature": 2.,
    "teacher_wrong_natural_weight": 3.,
    "other_group_weight": 1.,
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "training_seconds_limit": 1800,
}


@dataclass(frozen=True)
class WeightedExample:
    example: PreservedExample
    loss_weight: float
    teacher_correct: bool
    role: str

    @property
    def context(self):
        return self.example.context

    @property
    def candidates(self):
        return self.example.candidates


def build_weighted_example(group, role, recipe=RECIPE):
    """Build one aligned group, judging its teacher before candidate capping."""
    if role not in ("natural_train", "retention_train"):
        raise ValueError("weighted examples can only use fixed training roles")
    if group.get("label_aligned") is not True:
        return None
    teacher_correct = full_teacher_correct(group, recipe["base_weight"])
    origin = "retention" if role == "retention_train" else "natural"
    example = preserved_example(origin, group, recipe)
    if example is None:
        return None
    weight = (recipe["teacher_wrong_natural_weight"]
              if role == "natural_train" and not teacher_correct
              else recipe["other_group_weight"])
    return WeightedExample(example, weight, teacher_correct, role)


def build_examples(data, recipe=RECIPE):
    """Select natural groups source-only; retain every retention group."""
    result = {}
    eligibility = {}
    for role in ("natural_train", "retention_train"):
        selected = (select_groups(data[role], recipe) if role == "natural_train"
                    else [(row["origin"], group)
                          for row in data[role] for group in row["groups"]])
        examples = [example for _, group in selected
                    if (example := build_weighted_example(group, role, recipe)) is not None]
        if any(len(item.candidates) > recipe["candidate_limit"] for item in examples):
            raise ValueError("positive candidate set exceeds fixed candidate bound")
        result[role] = examples
        eligibility[role] = {
            "source_rows": len(data[role]),
            "total_groups": sum(len(row["groups"]) for row in data[role]),
            "selected_before_labels": len(selected),
            "eligible_groups": len(examples),
            "excluded_unaligned": sum(group.get("label_aligned") is not True
                                      for _, group in selected),
            "teacher_correct_groups": sum(item.teacher_correct for item in examples),
            "teacher_wrong_groups": sum(not item.teacher_correct for item in examples),
            "correction_weighted_groups": sum(item.loss_weight > 1 for item in examples),
            "distilled_groups": sum(item.example.distill_teacher for item in examples),
        }
    return result, eligibility


def weighted_ranking_loss(scores, examples, recipe=RECIPE):
    cursor = 0
    losses = []
    weights = []
    for item in examples:
        end = cursor + len(item.candidates)
        group = scores[cursor:end]
        cursor = end
        losses.append(item.loss_weight * ranking_loss(group, [item.example], recipe))
        weights.append(item.loss_weight)
    if cursor != len(scores) or not losses:
        raise ValueError("weighted word groups must consume every logit")
    return torch.stack(losses).sum() / scores.new_tensor(weights).sum()


def training_hashes():
    hashes = source_hashes()
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/train_preserved_context_ranker.py",
        ROOT / "scripts/train_natural_word_ranker.py",
        ROOT / "scripts/train_natural_sentence_ranker.py",
        ROOT / "scripts/train_sentence_ranker.py",
        ROOT / "hanglish/models/semantic_ranker.py",
    )
    for path in paths:
        hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    for path in PARENT.iterdir():
        if path.is_file():
            hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    return dict(sorted(hashes.items()))


def train(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, data = validate_data(args.prepared, args.manifest_sha256)
    if (manifest.get("runtime_version") != RUNTIME_VERSION
            or file_sha256(PARENT / "model.safetensors") != PARENT_WEIGHT):
        raise ValueError("current runtime data or semantic parent differs")
    before = training_hashes()
    torch.set_num_threads(RECIPE["threads"])
    random.seed(RECIPE["seed"])
    torch.manual_seed(RECIPE["seed"])
    operation_started = time.monotonic()

    def require_resources(stage):
        if (rss_bytes() > RECIPE["rss_limit_bytes"]
                or time.monotonic() - operation_started > RECIPE["training_seconds_limit"]):
            raise RuntimeError(f"current word resource bound exceeded during {stage}")

    ranker = SemanticRanker.load(PARENT)
    parent = evaluate_groups(ranker, data["natural_dev"], check_teacher=True)
    if parent["cases"] != len(data["natural_dev"]):
        raise ValueError("development evaluation did not cover every row")
    require_resources("parent development evaluation")

    groups, eligibility = build_examples(data)
    schedule = balanced_schedule(
        groups["natural_train"], groups["retention_train"], RECIPE["seed"]
    )
    args.output.mkdir(parents=True)

    def log(event, **values):
        message = json.dumps({"event": event, **values}, ensure_ascii=False)
        with (args.output / "training-events.jsonl").open("a") as stream:
            stream.write(message + "\n")
        print(message, flush=True)

    layers = _set_trainable_layers(ranker.model, RECIPE["top_layers"])
    optimizer = _optimizer(ranker.model, SimpleNamespace(**RECIPE), head_only=False)
    log("training_start", recipe=RECIPE, eligibility=eligibility,
        scheduled_groups=len(schedule), parent_correct=parent["correct"], layers=layers)
    losses = []
    started = time.monotonic()
    ranker.model.train()
    for start in range(0, len(schedule), RECIPE["group_batch"]):
        batch = schedule[start:start + RECIPE["group_batch"]]
        candidates = [candidate for item in batch for candidate in item.candidates]
        contexts = [item.context for item in batch for _ in item.candidates]
        encoded = encode_pairs(ranker.tokenizer, contexts, candidates, ranker.max_length)
        optimizer.zero_grad(set_to_none=True)
        scores = ranker.model(**encoded).logits.reshape(-1)
        loss = weighted_ranking_loss(scores, batch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite current word training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ranker.model.parameters(), 1.)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("nonfinite current word gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
        elapsed = time.monotonic() - started
        require_resources("training")
        if len(losses) % 64 == 0:
            log("training_step", step=len(losses), loss=losses[-1],
                elapsed_seconds=round(elapsed, 2), peak_rss_bytes=rss_bytes())
    seconds = time.monotonic() - started
    ranker.model.eval()

    bundle = args.output / "checkpoint"
    ranker.model.save_pretrained(bundle, safe_serialization=True)
    ranker.tokenizer.save_pretrained(bundle)
    artifacts = {path.name: file_sha256(path) for path in bundle.iterdir() if path.is_file()}
    metadata = {key: ranker.metadata[key] for key in
                ("base_model", "base_revision", "base_model_path", "base_source",
                 "license", "candidate_generator") if key in ranker.metadata}
    metadata.update(
        format_version=1,
        created_at=datetime.now(timezone.utc).isoformat(),
        artifact_hashes=artifacts,
        max_length=ranker.max_length,
        inference_batch_size=ranker.batch_size,
        parent_checkpoint_sha256=PARENT_WEIGHT,
        parent_metadata=ranker.metadata,
        parent_runtime_version=RUNTIME_VERSION,
        current_preparation={"path": str(args.prepared),
                             "manifest_sha256": args.manifest_sha256},
        frozen_source_hashes=before,
        training=RECIPE,
        training_eligibility=eligibility,
        trainable_layers=layers,
        test_data_read=False,
        library_versions={name: package_version(name) for name in
                          ("torch", "transformers", "tokenizers", "safetensors")},
    )
    (bundle / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    parity_contexts = ["주변 [MASK] 문맥"] * 2
    parity_candidates = ["검증", "검정"]
    memory_scores = ranker.score_pairs(parity_contexts, parity_candidates)
    loaded = SemanticRanker.load(bundle)
    loaded_scores = loaded.score_pairs(parity_contexts, parity_candidates)
    delta = max(abs(left - right) for left, right in zip(memory_scores, loaded_scores))
    if delta > 1e-4:
        raise RuntimeError("saved current word model differs from memory")
    require_resources("save-load parity")

    final = evaluate_groups(loaded, data["natural_dev"])
    if final["cases"] != len(data["natural_dev"]):
        raise ValueError("final development evaluation did not cover every row")
    require_resources("final development evaluation")
    validate_data(args.prepared, args.manifest_sha256)
    if (before != training_hashes()
            or file_sha256(PARENT / "model.safetensors") != PARENT_WEIGHT):
        raise RuntimeError("current word training sources changed")
    old = {row["id"]: row for row in parent["outcomes"]}
    corrections = [row["id"] for row in final["outcomes"]
                   if row["correct"] and not old[row["id"]]["correct"]]
    regressions = [row["id"] for row in final["outcomes"]
                   if not row["correct"] and old[row["id"]]["correct"]]
    report = {
        "recipe": RECIPE,
        "runtime_version": RUNTIME_VERSION,
        "preparation_manifest_sha256": args.manifest_sha256,
        "source_hashes": before,
        "parent_checkpoint_sha256": PARENT_WEIGHT,
        "parent": parent,
        "final": final,
        "eligibility": eligibility,
        "corrections": corrections,
        "regressions": regressions,
        "eligible_for_actual_hybrid_dev": final["correct"] > parent["correct"] and not regressions,
        "checkpoint_sha256": artifacts["model.safetensors"],
        "save_load_score_max_delta": delta,
        "optimizer_steps": len(losses),
        "mean_loss": sum(losses) / len(losses),
        "training_seconds": seconds,
        "elapsed_seconds": time.monotonic() - operation_started,
        "peak_rss_bytes": rss_bytes(),
        "device": "cpu",
        "development_rows_read": len(data["natural_dev"]),
        "development_sampled": False,
        "test_data_read": False,
        "deployed": False,
        "next_gate": "Fresh actual hybrid dev368 before final regression or deployment.",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "trainer-source.py").write_bytes(Path(__file__).read_bytes())
    log("complete", parent_correct=parent["correct"], final_correct=final["correct"],
        corrections=corrections, regressions=regressions,
        training_seconds=seconds, peak_rss_bytes=rss_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    train(parser.parse_args())
