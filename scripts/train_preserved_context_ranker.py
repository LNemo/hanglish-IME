"""One bounded continuation of the deployed semantic context ranker."""
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

from hanglish.models.semantic_ranker import SemanticRanker, encode_pairs
from scripts.prepare_context_training import OUTPUT, source_hashes, validate_context_data
from scripts.train_natural_sentence_ranker import balanced_schedule, rss_bytes
from scripts.train_natural_word_ranker import (
    bounded_example,
    evaluate_groups,
    select_groups,
)
from scripts.train_sentence_ranker import _optimizer, _set_trainable_layers


PARENT = ROOT / "checkpoints/semantic_ranker"
PARENT_WEIGHT = "54ca7d24b2bf17d1ba8f5181476ce91032acc94149a2baf401fe47e859f2907c"
CONTEXT_MANIFEST = "7c1c03205dbc9c8983c2b3dee37e4a6bcabb04d45a265e9e1699460595275b10"
DEFAULT_OUTPUT = ROOT / ".omx/artifacts/preserved-context-ranker-v1"
RECIPE = {
    "name": "preserved-semantic-top2-kd-v1",
    "seed": 20260914,
    "epochs": 1,
    "top_layers": 2,
    "head_lr": 5e-5,
    "encoder_lr": 1e-5,
    "weight_decay": .01,
    "group_batch": 4,
    "candidate_limit": 8,
    "source_group_limit": 8192,
    "selection_salt": "preserved-context-natural-groups-v1",
    "natural_retention_ratio": "1:1",
    "base_weight": .15,
    "distillation_weight": 1.,
    "temperature": 2.,
    "expected_natural_eligible": 5316,
    "expected_natural_teacher_correct": 5143,
    "expected_retention_eligible": 3995,
    "rss_limit_bytes": 3 * 1024**3,
    "training_seconds_limit": 1800,
}


@dataclass(frozen=True)
class PreservedExample:
    context: str
    candidates: tuple[str, ...]
    positives: tuple[int, ...]
    base_scores: tuple[float, ...]
    teacher_scores: tuple[float, ...]
    case_id: str
    origin: str
    distill_teacher: bool


def full_teacher_correct(group, base_weight=RECIPE["base_weight"]):
    """Judge the captured teacher on the full group, before candidate capping."""
    positives = group["positive_indices"]
    if not positives or len(group["candidates"]) < 2:
        return False
    chosen = min(range(len(group["candidates"])), key=lambda index: (
        -(group["teacher_scores"][index] + base_weight * group["base_scores"][index]),
        group["candidates"][index],
    ))
    return chosen in positives


def preserved_example(origin, group, recipe=RECIPE):
    teacher_correct = full_teacher_correct(group, recipe["base_weight"])
    base = bounded_example(origin, group, recipe)
    if base is None:
        return None
    return PreservedExample(
        base.context, base.candidates, base.positives, base.base_scores,
        base.teacher_scores, base.case_id, base.origin,
        origin == "retention" or teacher_correct,
    )


def ranking_loss(scores, examples, recipe=RECIPE):
    losses = []
    cursor = 0
    for example in examples:
        count = len(example.candidates)
        bias = scores.new_tensor(example.base_scores) * recipe["base_weight"]
        group = scores[cursor:cursor + count] + bias
        cursor += count
        positive = torch.tensor(example.positives, device=group.device)
        loss = torch.logsumexp(group, 0) - torch.logsumexp(group[positive], 0)
        if example.distill_teacher:
            teacher = group.new_tensor(example.teacher_scores) + bias
            temperature = recipe["temperature"]
            divergence = torch.nn.functional.kl_div(
                torch.log_softmax(group / temperature, dim=0),
                torch.softmax(teacher / temperature, dim=0),
                reduction="sum",
            )
            loss = loss + recipe["distillation_weight"] * temperature**2 * divergence
        losses.append(loss)
    if cursor != len(scores):
        raise ValueError("candidate groups do not consume all logits")
    return torch.stack(losses).mean()


def training_hashes():
    hashes = source_hashes()
    paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/prepare_context_training.py",
        ROOT / "scripts/train_natural_word_ranker.py",
        ROOT / "scripts/train_natural_sentence_ranker.py",
        ROOT / "scripts/train_sentence_ranker.py",
        ROOT / "hanglish/models/semantic_ranker.py",
    ]
    paths.extend(path for path in PARENT.iterdir() if path.is_file())
    for path in sorted(set(paths)):
        hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    return dict(sorted(hashes.items()))


def file_sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_examples(data):
    result = {}
    eligibility = {}
    for role in ("natural_train", "retention_train"):
        selected = (select_groups(data[role], RECIPE) if role == "natural_train" else
                    [(row["origin"], group) for row in data[role] for group in row["groups"]])
        eligible = [(group, example) for origin, group in selected
                    if (example := preserved_example(origin, group)) is not None]
        examples = [example for _, example in eligible]
        if any(len(example.candidates) > RECIPE["candidate_limit"] for example in examples):
            raise ValueError("positive candidate set exceeds fixed candidate bound")
        teacher_correct = sum(full_teacher_correct(group) for group, _ in eligible)
        result[role] = examples
        eligibility[role] = {
            "source_rows": len(data[role]),
            "selected_before_labels": len(selected),
            "eligible_groups": len(examples),
            "teacher_correct_groups": teacher_correct,
            "distilled_groups": sum(example.distill_teacher for example in examples),
        }
    if eligibility["natural_train"]["eligible_groups"] != RECIPE["expected_natural_eligible"]:
        raise ValueError("natural eligible group count differs from fixed trial")
    if eligibility["natural_train"]["teacher_correct_groups"] != RECIPE["expected_natural_teacher_correct"]:
        raise ValueError("natural teacher-correct count differs from fixed trial")
    if eligibility["retention_train"]["eligible_groups"] != RECIPE["expected_retention_eligible"]:
        raise ValueError("retention eligible group count differs from fixed trial")
    return result, eligibility


def train(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.manifest_sha256 != CONTEXT_MANIFEST:
        raise ValueError("training requires the fixed joint context manifest")
    manifest, data = validate_context_data(args.prepared, args.manifest_sha256)
    if file_sha256(PARENT / "model.safetensors") != PARENT_WEIGHT:
        raise ValueError("semantic parent differs from deployed checkpoint")
    before = training_hashes()
    torch.set_num_threads(2)
    random.seed(RECIPE["seed"])
    torch.manual_seed(RECIPE["seed"])
    ranker = SemanticRanker.load(PARENT)
    initial = evaluate_groups(ranker, data["natural_dev"], check_teacher=True)
    examples, eligibility = build_examples(data)
    schedule = balanced_schedule(
        examples["natural_train"], examples["retention_train"], RECIPE["seed"]
    )
    if len(schedule) != 2 * max(map(len, examples.values())):
        raise ValueError("training schedule is not balanced 1:1")

    args.output.mkdir(parents=True)
    log_path = args.output / "training-events.jsonl"

    def log(event, **values):
        message = json.dumps({"event": event, **values}, ensure_ascii=False)
        with log_path.open("a") as stream:
            stream.write(message + "\n")
        print(message, flush=True)

    layers = _set_trainable_layers(ranker.model, RECIPE["top_layers"])
    optimizer = _optimizer(ranker.model, SimpleNamespace(**RECIPE), head_only=False)
    log("training_start", recipe=RECIPE, eligibility=eligibility,
        scheduled_groups=len(schedule), layer_info=layers,
        initial_correct=initial["correct"])
    started = time.monotonic()
    losses = []
    ranker.model.train()
    for start in range(0, len(schedule), RECIPE["group_batch"]):
        batch = schedule[start:start + RECIPE["group_batch"]]
        contexts = [example.context for example in batch for _ in example.candidates]
        candidates = [candidate for example in batch for candidate in example.candidates]
        encoded = encode_pairs(ranker.tokenizer, contexts, candidates, ranker.max_length)
        optimizer.zero_grad(set_to_none=True)
        scores = ranker.model(**encoded).logits.reshape(-1)
        loss = ranking_loss(scores, batch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite preserved-context loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ranker.model.parameters(), 1.)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("nonfinite preserved-context gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
        elapsed = time.monotonic() - started
        if rss_bytes() > RECIPE["rss_limit_bytes"] or elapsed > RECIPE["training_seconds_limit"]:
            raise RuntimeError("preserved-context training resource bound exceeded")
        if len(losses) % 64 == 0:
            log("training_step", step=len(losses), loss=losses[-1],
                elapsed_seconds=round(elapsed, 2), peak_rss_bytes=rss_bytes())
    seconds = time.monotonic() - started
    ranker.model.eval()

    bundle = args.output / "checkpoint"
    ranker.model.save_pretrained(bundle, safe_serialization=True)
    ranker.tokenizer.save_pretrained(bundle)
    hashes = {path.name: file_sha256(path) for path in bundle.iterdir() if path.is_file()}
    metadata = {
        key: ranker.metadata[key] for key in
        ("base_model", "base_revision", "base_model_path", "base_source",
         "license", "candidate_generator")
        if key in ranker.metadata
    }
    metadata.update({
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_hashes": hashes,
        "max_length": ranker.max_length,
        "inference_batch_size": ranker.batch_size,
        "parent_checkpoint_sha256": PARENT_WEIGHT,
        "parent_metadata_sha256": before["checkpoints/semantic_ranker/metadata.json"],
        "parent_metadata": ranker.metadata,
        "parent_runtime_version": manifest["runtime_version"],
        "library_versions": {
            name: package_version(name)
            for name in ("torch", "transformers", "tokenizers", "safetensors")
        },
        "context_preparation": {
            "path": str(args.prepared),
            "manifest_sha256": args.manifest_sha256,
        },
        "frozen_source_hashes": before,
        "training": RECIPE,
        "training_eligibility": eligibility,
        "trainable_layers": layers,
        "test_data_read": False,
    })
    (bundle / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )

    probe_contexts = ["주변 [MASK] 문맥"] * 2
    probe_candidates = ["검증", "검정"]
    in_memory = ranker.score_pairs(probe_contexts, probe_candidates)
    loaded = SemanticRanker.load(bundle)
    roundtrip = loaded.score_pairs(probe_contexts, probe_candidates)
    delta = max(abs(left - right) for left, right in zip(in_memory, roundtrip))
    if delta > 1e-4:
        raise RuntimeError("safe-loaded model differs from in-memory model")
    final = evaluate_groups(loaded, data["natural_dev"])
    validate_context_data(args.prepared, args.manifest_sha256)
    if before != training_hashes():
        raise RuntimeError("preserved-context training inputs changed")
    old = {row["id"]: row for row in initial["outcomes"]}
    report = {
        "recipe": RECIPE,
        "initial": initial,
        "final": final,
        "eligibility": eligibility,
        "corrections": [row["id"] for row in final["outcomes"]
                        if row["correct"] and not old[row["id"]]["correct"]],
        "regressions": [row["id"] for row in final["outcomes"]
                        if not row["correct"] and old[row["id"]]["correct"]],
        "device": "cpu",
        "optimizer_steps": len(losses),
        "mean_loss": sum(losses) / len(losses),
        "training_seconds": seconds,
        "peak_conservative_combined_bytes": rss_bytes(),
        "checkpoint_sha256": hashes["model.safetensors"],
        "save_load_score_max_delta": delta,
        "test_data_read": False,
        "deployed": False,
        "next_gate": "Actual hybrid natural128 and original dev140/60/40 before any final regression.",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "trainer-source.py").write_bytes(Path(__file__).read_bytes())
    log("complete", initial=initial["correct"], final=final["correct"],
        training_seconds=seconds, peak_rss_bytes=rss_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=OUTPUT)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    train(parser.parse_args())
