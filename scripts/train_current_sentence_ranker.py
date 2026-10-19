"""Bounded continuation of the current sentence ranker on actual runtime pools."""
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
from hanglish.models.sentence_ranker import SentenceRanker, encode_sentence_pairs
from scripts.prepare_current_context_data import (
    OUTPUT as PREPARED,
    RUNTIME_VERSION,
    source_hashes,
    validate_data,
)
from scripts.train_natural_sentence_ranker import balanced_schedule, rss_bytes
from scripts.train_sentence_ranker import _optimizer, _set_trainable_layers
from scripts.train_union_sentence_ranker import (
    UnionExample,
    build_example,
    evaluate_pool,
    ranking_loss,
    teacher_choice,
)


PARENT = ROOT / "checkpoints/sentence_ranker"
PARENT_WEIGHT = "a68f3bd26733e9bc897a182f78b1decafefa3582fa82b6b2cfcaa34672412ffa"
OUTPUT = ROOT / ".omx/artifacts/current-sentence-ranker-v1"
RECIPE = {
    "name": "current-runtime-sentence-top2-weighted-correction-v1",
    "seed": 20260915,
    "epochs": 1,
    "top_layers": 2,
    "head_lr": 5e-5,
    "encoder_lr": 1e-5,
    "weight_decay": .01,
    "group_batch": 4,
    "candidate_limit": 8,
    "natural_retention_ratio": "1:1",
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
    example: UnionExample
    loss_weight: float
    teacher_correct: bool
    role: str

    @property
    def context(self):
        return self.example.context

    @property
    def candidates(self):
        return self.example.candidates


def build_weighted_example(row, role, limit=8):
    if role not in ("natural_train", "retention_train"):
        raise ValueError("weighted examples can only use fixed training roles")
    retention = role == "retention_train"
    example = build_example(row, retention=retention, limit=limit)
    if example is None:
        return None
    teacher_correct = teacher_choice(row) in row["positive_indices"]
    weight = (RECIPE["teacher_wrong_natural_weight"]
              if not retention and not teacher_correct
              else RECIPE["other_group_weight"])
    return WeightedExample(example, weight, teacher_correct, role)


def weighted_ranking_loss(scores, examples, recipe=RECIPE):
    cursor = 0
    weighted = []
    weights = []
    for item in examples:
        end = cursor + len(item.candidates)
        group = scores[cursor:end]
        cursor = end
        weighted.append(item.loss_weight * ranking_loss(group, [item.example], recipe))
        weights.append(item.loss_weight)
    if cursor != len(scores) or not weighted:
        raise ValueError("weighted sentence groups must consume every logit")
    return torch.stack(weighted).sum() / scores.new_tensor(weights).sum()


def training_hashes():
    hashes = source_hashes()
    for path in (
        Path(__file__).resolve(),
        ROOT / "scripts/train_union_sentence_ranker.py",
        ROOT / "scripts/train_natural_sentence_ranker.py",
        ROOT / "scripts/train_sentence_ranker.py",
    ):
        hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    return dict(sorted(hashes.items()))


def train(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, data = validate_data(args.prepared, args.manifest_sha256)
    if (manifest.get("runtime_version") != RUNTIME_VERSION
            or file_sha256(PARENT / "model.safetensors") != PARENT_WEIGHT):
        raise ValueError("current runtime data or sentence parent differs")
    before = training_hashes()
    torch.set_num_threads(RECIPE["threads"])
    random.seed(RECIPE["seed"])
    torch.manual_seed(RECIPE["seed"])
    operation_started = time.monotonic()

    def require_resources(stage):
        if (rss_bytes() > RECIPE["rss_limit_bytes"]
                or time.monotonic() - operation_started > RECIPE["training_seconds_limit"]):
            raise RuntimeError(f"current sentence resource bound exceeded during {stage}")

    ranker = SentenceRanker.load(PARENT)
    parent = evaluate_pool(data["natural_dev"], ranker)
    require_resources("parent development evaluation")

    # Reproduce captured teacher logits before using the data for training.
    for row in [row for row in data["natural_dev"] if row["sentence_eligible"]][:4]:
        actual = ranker.score_pairs([row["context"]] * len(row["candidates"]), row["candidates"])
        if (len(actual) != len(row["teacher_scores"])
                or max(abs(left - right) for left, right in zip(actual, row["teacher_scores"])) > 1e-4):
            raise ValueError("captured current sentence teacher scores do not reproduce")
    require_resources("teacher reproduction")

    groups = {}
    eligibility = {}
    for role in ("natural_train", "retention_train"):
        examples = [example for row in data[role]
                    if (example := build_weighted_example(
                        row, role, RECIPE["candidate_limit"]
                    )) is not None]
        groups[role] = examples
        eligibility[role] = {
            "rows": len(data[role]),
            "eligible": len(examples),
            "teacher_correct": sum(example.teacher_correct for example in examples),
            "teacher_wrong": sum(not example.teacher_correct for example in examples),
            "correction_weighted": sum(example.loss_weight > 1 for example in examples),
            "distilled": sum(example.example.distill_teacher for example in examples),
        }
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
        encoded = encode_sentence_pairs(ranker.tokenizer, candidates, contexts, ranker.max_length)
        optimizer.zero_grad(set_to_none=True)
        scores = ranker.model(**encoded).logits.reshape(-1)
        loss = weighted_ranking_loss(scores, batch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite current sentence training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ranker.model.parameters(), 1.)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("nonfinite current sentence gradients")
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
                ("base_model", "base_revision", "base_source", "license", "candidate_generator")
                if key in ranker.metadata}
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
    loaded = SentenceRanker.load(bundle)
    parity_candidates = ["검증 문장입니다", "검정 문장입니다"]
    memory_scores = ranker.score_pairs([""] * 2, parity_candidates)
    loaded_scores = loaded.score_pairs([""] * 2, parity_candidates)
    delta = max(abs(left - right) for left, right in zip(memory_scores, loaded_scores))
    if delta > 1e-4:
        raise RuntimeError("saved current sentence model differs from memory")
    require_resources("save-load parity")

    final = evaluate_pool(data["natural_dev"], loaded)
    require_resources("final development evaluation")
    validate_data(args.prepared, args.manifest_sha256)
    if (before != training_hashes()
            or file_sha256(PARENT / "model.safetensors") != PARENT_WEIGHT):
        raise RuntimeError("current sentence training sources changed")
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
        "eligible_for_runtime_dev": final["correct"] > parent["correct"] and not regressions,
        "checkpoint_sha256": artifacts["model.safetensors"],
        "save_load_score_max_delta": delta,
        "optimizer_steps": len(losses),
        "mean_loss": sum(losses) / len(losses),
        "training_seconds": seconds,
        "elapsed_seconds": time.monotonic() - operation_started,
        "peak_rss_bytes": rss_bytes(),
        "device": "cpu",
        "development_rows_read": len(data["natural_dev"]),
        "test_data_read": False,
        "deployed": False,
        "next_gate": "Fresh actual runtime dev128/140/60/40 before final regression or deployment.",
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
