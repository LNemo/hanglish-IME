"""Bounded sentence continuation on source-only, parent-preserving proposal pools."""
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
from scripts.prepare_union_sentence_data import OUTPUT, source_hashes, validate_union_data
from scripts.train_natural_sentence_ranker import balanced_schedule, rss_bytes
from scripts.train_sentence_ranker import _optimizer, _set_trainable_layers

PARENT = ROOT / "checkpoints/sentence_ranker"
PARENT_WEIGHT = "a68f3bd26733e9bc897a182f78b1decafefa3582fa82b6b2cfcaa34672412ffa"
DEFAULT_OUTPUT = ROOT / ".omx/artifacts/union-sentence-ranker-v1"
RECIPE = {
    "name": "union-sentence-top2-kd-v1", "seed": 20260914, "epochs": 1,
    "top_layers": 2, "head_lr": 5e-5, "encoder_lr": 1e-5,
    "weight_decay": .01, "group_batch": 4, "candidate_limit": 8,
    "natural_retention_ratio": "1:1", "distillation_weight": 1.,
    "temperature": 2., "threads": 2, "rss_limit_bytes": 3 * 1024**3,
    "training_seconds_limit": 1800,
}


@dataclass(frozen=True)
class UnionExample:
    context: str
    candidates: tuple[str, ...]
    positives: tuple[int, ...]
    teacher_scores: tuple[float, ...]
    case_id: str
    distill_teacher: bool


def teacher_choice(row, parent_only=False):
    if not row["sentence_eligible"]:
        return 0
    count = len(row["parent_candidates"] if parent_only else row["candidates"])
    return min(range(count), key=lambda i: (-row["teacher_scores"][i], row["candidates"][i]))


def build_example(row, *, retention=False, limit=8):
    if not row["sentence_eligible"] or not row["positive_indices"] or len(row["candidates"]) < 2:
        return None
    # Teacher correctness is determined on the entire runtime pool before capping.
    distill = retention or teacher_choice(row) in row["positive_indices"]
    selected = list(row["positive_indices"])
    if len(selected) > limit:
        raise ValueError("positive set exceeds training candidate bound")
    negatives = sorted((i for i in range(len(row["candidates"])) if i not in selected),
                       key=lambda i: (-row["teacher_scores"][i], row["candidates"][i]))
    selected.extend(negatives[:limit - len(selected)])
    return UnionExample(row["context"], tuple(row["candidates"][i] for i in selected),
                        tuple(range(len(row["positive_indices"]))),
                        tuple(row["teacher_scores"][i] for i in selected), row["id"], distill)


def ranking_loss(scores, examples, recipe=RECIPE):
    cursor, losses = 0, []
    for example in examples:
        group = scores[cursor:cursor + len(example.candidates)]
        cursor += len(example.candidates)
        positives = torch.tensor(example.positives, device=group.device)
        loss = torch.logsumexp(group, 0) - torch.logsumexp(group[positives], 0)
        if example.distill_teacher:
            temperature = recipe["temperature"]
            teacher = group.new_tensor(example.teacher_scores)
            divergence = torch.nn.functional.kl_div(
                torch.log_softmax(group / temperature, dim=0),
                torch.softmax(teacher / temperature, dim=0), reduction="sum")
            loss = loss + recipe["distillation_weight"] * temperature**2 * divergence
        losses.append(loss)
    if cursor != len(scores):
        raise ValueError("sentence groups do not consume every logit")
    return torch.stack(losses).mean()


def evaluate_pool(rows, ranker=None, *, parent_only=False):
    """Evaluate every source row, including unchanged standalone/fallback rows."""
    outcomes = []
    for row in rows:
        candidates = row["parent_candidates"] if parent_only else row["candidates"]
        if row["sentence_eligible"]:
            scores = (ranker.score_pairs([row["context"]] * len(candidates), candidates)
                      if ranker is not None else row["teacher_scores"][:len(candidates)])
            if len(scores) != len(candidates):
                raise ValueError("sentence evaluation score count differs")
            choice = min(range(len(candidates)), key=lambda i: (-scores[i], candidates[i]))
        else:
            choice = 0
        outcomes.append({"id": row["id"], "result": candidates[choice],
                         "correct": candidates[choice] in row["expected"],
                         "covered": any(c in row["expected"] for c in candidates),
                         "sentence_eligible": row["sentence_eligible"]})
    return {"rows": len(outcomes), "correct": sum(r["correct"] for r in outcomes),
            "covered": sum(r["covered"] for r in outcomes),
            "fallbacks": sum(not r["sentence_eligible"] for r in outcomes), "outcomes": outcomes}


def training_hashes():
    hashes = source_hashes()
    for path in (Path(__file__).resolve(), ROOT / "scripts/train_natural_sentence_ranker.py",
                 ROOT / "scripts/train_sentence_ranker.py"):
        hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    return dict(sorted(hashes.items()))


def train(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, data = validate_union_data(args.prepared, args.manifest_sha256)
    if file_sha256(PARENT / "model.safetensors") != PARENT_WEIGHT:
        raise ValueError("sentence parent differs from current deployed model")
    before = training_hashes()
    torch.set_num_threads(RECIPE["threads"])
    random.seed(RECIPE["seed"])
    torch.manual_seed(RECIPE["seed"])
    ranker = SentenceRanker.load(PARENT)
    parent = evaluate_pool(data["natural_dev"], parent_only=True)
    initial = evaluate_pool(data["natural_dev"])
    # Verify cached teacher scores against fresh scores on a bounded source-ordered sample.
    for row in [r for r in data["natural_dev"] if r["sentence_eligible"]][:4]:
        actual = ranker.score_pairs([row["context"]] * len(row["candidates"]), row["candidates"])
        if max(abs(a - b) for a, b in zip(actual, row["teacher_scores"])) > 1e-4:
            raise ValueError("captured sentence teacher scores do not reproduce")
    groups, eligibility = {}, {}
    for role in ("natural_train", "retention_train"):
        examples = [example for row in data[role]
                    if (example := build_example(row, retention=role == "retention_train",
                                                 limit=RECIPE["candidate_limit"])) is not None]
        groups[role] = examples
        eligibility[role] = {"rows": len(data[role]), "eligible": len(examples),
                             "distilled": sum(e.distill_teacher for e in examples)}
    schedule = balanced_schedule(groups["natural_train"], groups["retention_train"], RECIPE["seed"])
    args.output.mkdir(parents=True)

    def log(event, **values):
        message = json.dumps({"event": event, **values}, ensure_ascii=False)
        with (args.output / "training-events.jsonl").open("a") as stream:
            stream.write(message + "\n")
        print(message, flush=True)

    layers = _set_trainable_layers(ranker.model, RECIPE["top_layers"])
    optimizer = _optimizer(ranker.model, SimpleNamespace(**RECIPE), head_only=False)
    log("training_start", recipe=RECIPE, eligibility=eligibility, scheduled_groups=len(schedule),
        parent_correct=parent["correct"], initial_union_correct=initial["correct"], layers=layers)
    losses, started = [], time.monotonic()
    ranker.model.train()
    for start in range(0, len(schedule), RECIPE["group_batch"]):
        batch = schedule[start:start + RECIPE["group_batch"]]
        candidates = [c for example in batch for c in example.candidates]
        contexts = [example.context for example in batch for _ in example.candidates]
        encoded = encode_sentence_pairs(ranker.tokenizer, candidates, contexts, ranker.max_length)
        optimizer.zero_grad(set_to_none=True)
        loss = ranking_loss(ranker.model(**encoded).logits.reshape(-1), batch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite sentence training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(ranker.model.parameters(), 1.)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError("nonfinite sentence training gradients")
        optimizer.step()
        losses.append(float(loss.detach()))
        elapsed = time.monotonic() - started
        if rss_bytes() > RECIPE["rss_limit_bytes"] or elapsed > RECIPE["training_seconds_limit"]:
            raise RuntimeError("sentence training resource bound exceeded")
        if len(losses) % 64 == 0:
            log("training_step", step=len(losses), loss=losses[-1], elapsed_seconds=round(elapsed, 2),
                peak_rss_bytes=rss_bytes())
    seconds = time.monotonic() - started
    ranker.model.eval()
    bundle = args.output / "checkpoint"
    ranker.model.save_pretrained(bundle, safe_serialization=True)
    ranker.tokenizer.save_pretrained(bundle)
    artifacts = {p.name: file_sha256(p) for p in bundle.iterdir() if p.is_file()}
    metadata = {key: ranker.metadata[key] for key in
                ("base_model", "base_revision", "base_source", "license", "candidate_generator")
                if key in ranker.metadata}
    metadata.update(format_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                    artifact_hashes=artifacts, max_length=ranker.max_length,
                    inference_batch_size=ranker.batch_size, parent_checkpoint_sha256=PARENT_WEIGHT,
                    parent_metadata=ranker.metadata, parent_runtime_version=manifest["parent_runtime_version"],
                    union_preparation={"path": str(args.prepared), "manifest_sha256": args.manifest_sha256},
                    frozen_source_hashes=before, training=RECIPE, training_eligibility=eligibility,
                    trainable_layers=layers, test_data_read=False,
                    library_versions={name: package_version(name) for name in
                                      ("torch", "transformers", "tokenizers", "safetensors")})
    (bundle / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    loaded = SentenceRanker.load(bundle)
    candidates = ["검증 문장입니다", "검정 문장입니다"]
    memory = ranker.score_pairs([""] * 2, candidates)
    saved = loaded.score_pairs([""] * 2, candidates)
    delta = max(abs(a - b) for a, b in zip(memory, saved))
    if delta > 1e-4:
        raise RuntimeError("saved sentence model differs from in-memory scores")
    final = evaluate_pool(data["natural_dev"], loaded)
    validate_union_data(args.prepared, args.manifest_sha256)
    if before != training_hashes():
        raise RuntimeError("sentence training sources changed")
    old = {row["id"]: row for row in parent["outcomes"]}
    regressions = [r["id"] for r in final["outcomes"] if not r["correct"] and old[r["id"]]["correct"]]
    report = {"recipe": RECIPE, "parent": parent, "initial_union": initial, "final": final,
              "eligibility": eligibility, "corrections": [r["id"] for r in final["outcomes"]
                 if r["correct"] and not old[r["id"]]["correct"]], "regressions": regressions,
              "eligible_for_runtime_dev": final["correct"] > parent["correct"] and not regressions,
              "checkpoint_sha256": artifacts["model.safetensors"], "save_load_score_max_delta": delta,
              "optimizer_steps": len(losses), "mean_loss": sum(losses) / len(losses),
              "training_seconds": seconds, "peak_rss_bytes": rss_bytes(), "device": "cpu",
              "test_data_read": False, "deployed": False,
              "next_gate": "Fresh full hybrid dev128/140/60/40 before final regression or deployment."}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.output / "trainer-source.py").write_bytes(Path(__file__).read_bytes())
    log("complete", parent_correct=parent["correct"], initial_union_correct=initial["correct"],
        final_correct=final["correct"], regressions=regressions, training_seconds=seconds,
        peak_rss_bytes=rss_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=OUTPUT)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    train(parser.parse_args())
