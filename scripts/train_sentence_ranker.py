#!/usr/bin/env python3
"""Train a bounded full-sentence reranker from frozen train/dev data only."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import resource
import shutil
import tempfile
import time

import torch

from hanglish.core.lattice import CandidateGenerator, WordOption
from hanglish.models.sentence_ranker import SentenceRanker, encode_sentence_pairs
from hanglish.models.training_guard import validate_training_augmentation, validate_training_splits
from scripts.train_context_ranker import _options_for_source, align_words
from scripts.train_semantic_ranker import PINNED_MODEL, PINNED_REVISION, validate_base_snapshot, checkpoint_source_files


ROOT = Path(__file__).resolve().parents[1]
FREEZE_FILES = (
    "hanglish/core/lattice.py", "hanglish/core/lexical_data.py",
    "hanglish/core/phonology.py", "hanglish/core/neural_converter.py",
    "hanglish/models/semantic_ranker.py", "hanglish/models/sentence_ranker.py",
    "hanglish/models/phonetic_transducer.py", "scripts/train_context_ranker.py",
    "scripts/train_sentence_ranker.py", "scripts/train_semantic_ranker.py",
    "data/challenge/train.jsonl",
    "data/challenge/dev.jsonl", "data/challenge/manifest.json",
    "data/challenge/train_lexicon.json", "data/morphology_lexicon.json",
    "data/workbench_lexicon.json",
    "data/external/mecab-ko-dic/NNBC.csv",
)


@dataclass(frozen=True)
class SentenceExample:
    context: str
    candidates: tuple[str, ...]
    positives: tuple[int, ...]
    case_id: str
    category: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_fresh_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def frozen_source_hashes() -> dict[str, str]:
    return {relative: _sha256(ROOT / relative) for relative in FREEZE_FILES}


def _surface_spans(surface: str, golds: list[str]) -> list[tuple[int, int]]:
    """Locate aligned gold groups while tolerating whitespace merged by alignment."""
    compact = []
    offsets = []
    for index, char in enumerate(surface):
        if not char.isspace():
            compact.append(char)
            offsets.append(index)
    compact_surface = "".join(compact)
    cursor = 0
    spans = []
    for gold in golds:
        needle = "".join(char for char in gold if not char.isspace())
        position = compact_surface.find(needle, cursor)
        if position < 0:
            raise ValueError(f"aligned target {gold!r} is absent from {surface!r}")
        spans.append((offsets[position], offsets[position + len(needle) - 1] + 1))
        cursor = position + len(needle)
    return spans


def _sentence_variants(source: str, expected: str, generator: CandidateGenerator,
                       beam_size: int, inject_gold: bool) -> tuple[dict[str, float], bool]:
    aligned = align_words(source, expected)
    if not aligned:
        return {}, False
    golds = [gold for _, gold in aligned]
    spans = _surface_spans(expected, golds)
    beams = [("", 0.0)]
    cursor = 0
    naturally_covered = True
    for (roman, gold), (start, end) in zip(aligned, spans):
        literal = expected[cursor:start]
        options = list(_options_for_source(roman, generator))
        has_gold = any(option.text == gold for option in options)
        naturally_covered &= has_gold
        if inject_gold and not has_gold:
            options.append(WordOption(gold, -20.0, "injected-train-gold"))
        expanded = [
            (prefix + literal + option.text, score + option.score)
            for prefix, score in beams for option in options[:12]
        ]
        unique: dict[str, float] = {}
        for text, score in expanded:
            unique[text] = max(score, unique.get(text, float("-inf")))
        beams = sorted(unique.items(), key=lambda item: (-item[1], item[0]))[:beam_size]
        cursor = end
    suffix = expected[cursor:]
    variants = {text + suffix: score for text, score in beams}
    if inject_gold:
        variants.setdefault(expected, min(variants.values(), default=-20.0) - 1.0)
    return variants, naturally_covered


def build_sentence_examples(path: Path, generator: CandidateGenerator, *,
                            candidate_limit: int, inject_gold: bool) -> tuple[list[SentenceExample], dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    examples = []
    naturally_covered = bounded_covered = rankable = 0
    for row in rows:
        variants: dict[str, float] = {}
        row_natural = False
        for expected in row["expected"]:
            current, covered = _sentence_variants(
                row["input"], expected, generator, candidate_limit * 8, inject_gold)
            row_natural |= covered
            for sentence, score in current.items():
                variants[sentence] = max(score, variants.get(sentence, float("-inf")))
        ordered = sorted(variants, key=lambda value: (-variants[value], value))
        if inject_gold:
            positives = [value for value in row["expected"] if value in variants]
            selected = list(dict.fromkeys(positives + ordered))[:candidate_limit]
        else:
            selected = ordered[:candidate_limit]
        positive_indices = tuple(
            index for index, value in enumerate(selected) if value in row["expected"])
        naturally_covered += row_natural
        bounded_covered += bool(positive_indices)
        rankable += len(selected) > 1
        examples.append(SentenceExample(
            row["context"], tuple(selected), positive_indices, row["id"], row["category"]))
    count = len(rows)
    return examples, {
        "rows": count,
        "oracle_segment_covered_rows": naturally_covered,
        "oracle_segment_coverage": naturally_covered / count if count else 0.0,
        "gold_aligned_bounded_candidate_covered_rows": bounded_covered,
        "gold_aligned_bounded_candidate_coverage": bounded_covered / count if count else 0.0,
        "rankable_rows": rankable,
        "candidate_limit": candidate_limit,
        "train_gold_injection": inject_gold,
        "generation_mode": "gold_aligned_oracle_analysis",
        "runtime_candidate_coverage": None,
    }


def build_runtime_sentence_examples(path: Path, converter) -> tuple[list[SentenceExample], dict]:
    """Build DEV proposals from runtime inputs before consulting expected labels."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    proposed = []
    for row in rows:
        result = converter.sentence_proposals(row["input"], row["context"])
        proposed.append(tuple(path[0] for path in result["paths"][:48]))
    examples = []
    covered = rankable = 0
    for row, candidates in zip(rows, proposed):
        positives = tuple(index for index, value in enumerate(candidates)
                          if value in row["expected"])
        covered += bool(positives)
        rankable += len(candidates) > 1
        examples.append(SentenceExample(
            row["context"], candidates, positives, row["id"], row["category"]))
    count = len(rows)
    return examples, {
        "rows": count, "runtime_candidate_covered_rows": covered,
        "runtime_candidate_coverage": covered / count if count else 0.0,
        "rankable_rows": rankable, "candidate_limit": 48,
        "generation_mode": "runtime_source_only_static_and_semantic_union",
        "expected_used_after_proposal_generation": True,
    }


def runtime_proposal_artifact_hashes() -> dict[str, str]:
    paths = checkpoint_source_files(ROOT / "checkpoints/semantic_ranker")
    paths.extend((ROOT / relative for relative in (
        "checkpoints/phonetic_transducer.pt", "checkpoints/phonetic_transducer.json")))
    return {str(path.relative_to(ROOT)): _sha256(path)
            for path in sorted(paths) if path.is_file()}


def runtime_proposal_fingerprint(examples: list[SentenceExample]) -> str:
    payload = [(example.case_id, list(example.candidates)) for example in examples]
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def balanced_order(examples: list[SentenceExample], per_category: int, seed: int) -> list[int]:
    by_category: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_category[example.category].append(index)
    rng = random.Random(seed)
    order = []
    for category in sorted(by_category):
        indices = by_category[category]
        if per_category <= 0:
            order.extend(indices)
        elif len(indices) >= per_category:
            order.extend(rng.sample(indices, per_category))
        else:
            order.extend(rng.choice(indices) for _ in range(per_category))
    rng.shuffle(order)
    return order


def selection_key(metrics: dict) -> tuple[float, float]:
    """Prefer sentence accuracy, then lower covered-row NLL, including exact zero."""
    nll = metrics.get("covered_mean_nll")
    return metrics["sentence_accuracy"], -(nll if nll is not None else float("inf"))


def _set_trainable_layers(model, top_layers: int) -> dict:
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    electra = getattr(model.base_model, "electra", model.base_model)
    layers = getattr(getattr(electra, "encoder", None), "layer", None)
    if layers is None:
        raise RuntimeError("unsupported ELECTRA encoder structure")
    for layer in layers[-top_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for name, parameter in model.named_parameters():
        if not name.startswith(model.base_model_prefix + "."):
            parameter.requires_grad = True
    return {
        "encoder_layers": len(layers), "unfrozen_encoder_layers": min(top_layers, len(layers)),
        "trainable_parameters": sum(x.numel() for x in model.parameters() if x.requires_grad),
        "total_parameters": sum(x.numel() for x in model.parameters()),
    }


def _loss(model, tokenizer, examples: list[SentenceExample], args) -> torch.Tensor:
    sentences, contexts, groups = [], [], []
    for example in examples:
        start = len(sentences)
        sentences.extend(example.candidates)
        contexts.extend([example.context] * len(example.candidates))
        groups.append((start, len(sentences), example.positives))
    encoded = encode_sentence_pairs(tokenizer, sentences, contexts, args.max_length)
    encoded = {name: value.to(args.device) for name, value in encoded.items()}
    scores = model(**encoded).logits.reshape(-1)
    losses = []
    for start, end, positives in groups:
        if not positives:
            continue
        group = scores[start:end]
        positive = torch.tensor(positives, device=group.device)
        losses.append(torch.logsumexp(group, 0) - torch.logsumexp(group[positive], 0))
    return torch.stack(losses).mean()


@torch.inference_mode()
def evaluate(model, tokenizer, examples: list[SentenceExample], args) -> dict:
    model.eval()
    correct = covered = 0
    losses = []
    categories: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for example in examples:
        hit = False
        if example.candidates:
            encoded = encode_sentence_pairs(
                tokenizer, list(example.candidates), [example.context] * len(example.candidates),
                args.max_length)
            encoded = {name: value.to(args.device) for name, value in encoded.items()}
            scores = model(**encoded).logits.reshape(-1)
            if example.positives:
                covered += 1
                positive = torch.tensor(example.positives, device=scores.device)
                losses.append(float(torch.logsumexp(scores, 0) - torch.logsumexp(scores[positive], 0)))
                hit = int(scores.argmax()) in example.positives
        correct += int(hit)
        categories[example.category][0] += int(hit)
        categories[example.category][1] += 1
    total = len(examples)
    return {
        "sentence_accuracy": correct / total if total else 0.0,
        "correct": correct, "rows": total, "covered_rows": covered,
        "covered_mean_nll": sum(losses) / len(losses) if losses else None,
        "categories": {key: {"correct": value[0], "total": value[1],
                             "accuracy": value[0] / value[1]}
                       for key, value in sorted(categories.items())},
    }


def _optimizer(model, args, head_only: bool):
    encoder, head = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (encoder if name.startswith(model.base_model_prefix + ".") else head).append(parameter)
    if head_only:
        return torch.optim.AdamW(head, lr=args.head_lr, weight_decay=args.weight_decay)
    return torch.optim.AdamW([
        {"params": encoder, "lr": args.encoder_lr},
        {"params": head, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)


def _run_epochs(model, tokenizer, train_examples, dev_examples, args, epochs, *, head_only, offset):
    optimizer = _optimizer(model, args, head_only)
    history = []
    best_key = None
    best_state = None
    for epoch in range(epochs):
        order = balanced_order(train_examples, args.examples_per_category, args.seed + offset + epoch)
        model.train()
        losses = []
        for start in range(0, len(order), args.group_batch):
            batch = [train_examples[index] for index in order[start:start + args.group_batch]]
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, tokenizer, batch, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(model, tokenizer, dev_examples, args)
        mean_loss = sum(losses) / len(losses)
        history.append({"mean_loss": mean_loss, "dev": metrics})
        key = selection_key(metrics)
        if best_key is None or key > best_key:
            best_key = key
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return history, best_state


def resource_smoke(model, tokenizer, args) -> dict:
    sentences = ["문장을 자연스럽게 변환한다."] * args.inference_batch_size
    contexts = ["중립적인 동작을 설명한다."] * args.inference_batch_size
    encoded = encode_sentence_pairs(tokenizer, sentences, contexts, args.max_length)
    encoded = {name: value.to(args.device) for name, value in encoded.items()}
    samples = []
    model.eval()
    with torch.inference_mode():
        model(**encoded)
        for _ in range(10):
            start = time.perf_counter(); model(**encoded)
            samples.append((time.perf_counter() - start) * 1000)
    return {"batch_size": args.inference_batch_size,
            "median_ms": sorted(samples)[len(samples) // 2], "max_ms": max(samples)}


def train(args) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, __version__ as transformers_version
    from tokenizers import __version__ as tokenizers_version
    from safetensors import __version__ as safetensors_version

    torch.set_num_threads(args.threads)
    random.seed(args.seed); torch.manual_seed(args.seed)
    augmentation_args = (args.augmentation, args.augmentation_manifest,
                         args.augmentation_attestation)
    if any(value is not None for value in augmentation_args) and not all(
            value is not None for value in augmentation_args):
        raise ValueError("augmentation, manifest, and attestation must be provided together")
    if args.output is None:
        name = "sentence_ranker_augmented" if args.augmentation else "sentence_ranker"
        args.output = ROOT / "checkpoints" / name
    if args.augmentation and args.output.resolve() == (ROOT / "checkpoints/sentence_ranker").resolve():
        raise ValueError("augmented training must use a side-by-side output checkpoint")
    require_fresh_output(args.output)
    guard = validate_training_splits(args.train, args.dev, args.manifest)
    augmentation_guard = None
    if args.augmentation is not None:
        augmentation_guard = validate_training_augmentation(
            args.augmentation, args.augmentation_manifest, args.augmentation_attestation,
            args.train, args.dev, args.manifest, guard)
    start_hashes = frozen_source_hashes()
    if augmentation_guard is not None:
        start_hashes[str(args.augmentation)] = augmentation_guard["augmentation_sha256"]
        start_hashes[str(args.augmentation_manifest)] = augmentation_guard["manifest_sha256"]
        start_hashes[str(args.augmentation_attestation)] = augmentation_guard[
            "held_out_context_attestation_sha256"]
    generator = CandidateGenerator()
    train_examples, train_coverage = build_sentence_examples(
        args.train, generator, candidate_limit=args.candidates, inject_gold=True)
    if args.augmentation is not None:
        augmented_examples, augmented_coverage = build_sentence_examples(
            args.augmentation, generator, candidate_limit=args.candidates, inject_gold=True)
        train_examples.extend(augmented_examples)
        train_coverage = {"base": train_coverage, "augmentation": augmented_coverage,
                          "combined_rank_examples": len(train_examples)}
    oracle_dev_examples, oracle_dev_coverage = build_sentence_examples(
        args.dev, generator, candidate_limit=args.candidates, inject_gold=False)
    from hanglish.core.neural_converter import NeuralConverter
    proposal_artifacts_start = runtime_proposal_artifact_hashes()
    proposal_converter = NeuralConverter(ranker_kind="semantic")
    dev_examples, dev_coverage = build_runtime_sentence_examples(args.dev, proposal_converter)
    proposal_fingerprint = runtime_proposal_fingerprint(dev_examples)
    if not train_examples or not dev_examples:
        raise RuntimeError("sentence-ranker needs train and dev examples")

    base_source = validate_base_snapshot(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True,
                                               trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=False,
        weights_only=True, num_labels=1).to(args.device)
    layer_info = _set_trainable_layers(model, args.top_layers)
    smoke = resource_smoke(model, tokenizer, args)
    initial = evaluate(model, tokenizer, dev_examples, args)
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    started = time.monotonic()

    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    warmup, warmup_state = _run_epochs(model, tokenizer, train_examples, dev_examples, args,
                                       args.warmup_epochs, head_only=True, offset=0)
    layer_info = _set_trainable_layers(model, args.top_layers)
    tuned, best_state = _run_epochs(model, tokenizer, train_examples, dev_examples, args,
                                    args.epochs, head_only=False, offset=100)
    all_choices = [(initial, initial_state, "initial")]
    if warmup_state is not None:
        best_warmup = max((item["dev"] for item in warmup),
                          key=selection_key)
        all_choices.append((best_warmup, warmup_state, "warmup"))
    if best_state is not None:
        best_tuned = max((item["dev"] for item in tuned),
                         key=selection_key)
        all_choices.append((best_tuned, best_state, "tuned"))
    selected_metrics, selected_state, selected_kind = max(
        all_choices, key=lambda item: selection_key(item[0]))
    model.load_state_dict(selected_state)
    final_dev = evaluate(model, tokenizer, dev_examples, args)
    oracle_dev_diagnostic = evaluate(model, tokenizer, oracle_dev_examples, args)
    end_hashes = frozen_source_hashes()
    if augmentation_guard is not None:
        end_hashes[str(args.augmentation)] = _sha256(args.augmentation)
        end_hashes[str(args.augmentation_manifest)] = _sha256(args.augmentation_manifest)
        end_hashes[str(args.augmentation_attestation)] = _sha256(args.augmentation_attestation)
    if start_hashes != end_hashes:
        raise RuntimeError("candidate or corpus sources changed during sentence-ranker training")
    proposal_artifacts_end = runtime_proposal_artifact_hashes()
    if proposal_artifacts_start != proposal_artifacts_end:
        raise RuntimeError("runtime proposal model artifacts changed during sentence-ranker training")

    temporary = Path(tempfile.mkdtemp(prefix="sentence-ranker-", dir=args.output.parent))
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        artifacts = {path.name: _sha256(path) for path in temporary.iterdir() if path.is_file()}
        metadata = {
            "format_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "base_model": PINNED_MODEL, "base_revision": PINNED_REVISION,
            "base_model_path": str(args.base_model), "base_source": base_source,
            "versions": {"torch": torch.__version__, "transformers": transformers_version,
                         "tokenizers": tokenizers_version, "safetensors": safetensors_version},
            "seed": args.seed, "training_data_guard": guard,
            "augmentation_data_guard": augmentation_guard,
            "test_data_read": guard["held_out_test_read"], "source_hashes": guard["source_hashes"],
            "frozen_source_hashes_start": start_hashes, "frozen_source_hashes_end": end_hashes,
            "code_hashes": {
                "hanglish/models/sentence_ranker.py": _sha256(ROOT / "hanglish/models/sentence_ranker.py"),
                "scripts/train_sentence_ranker.py": _sha256(Path(__file__).resolve()),
            },
            "candidate_generator": {
                "external_version": generator.external_version,
                "external_word_count": generator.external_word_count,
                "external_fingerprint": generator.external_fingerprint,
                "dictionary_fingerprint": generator.dictionary_fingerprint,
                "dictionary_rejected_rows": generator.dictionary_rejected_rows,
            },
            "max_length": args.max_length, "inference_batch_size": args.inference_batch_size,
            "training": {"warmup_epochs": args.warmup_epochs, "epochs": args.epochs,
                         "examples_per_category_per_epoch": args.examples_per_category,
                         "group_batch": args.group_batch, "candidates": args.candidates,
                         "head_lr": args.head_lr, "encoder_lr": args.encoder_lr,
                         "weight_decay": args.weight_decay, **layer_info,
                         "warmup_history": warmup, "tune_history": tuned},
            "coverage": {"train_gold_aligned": train_coverage, "dev_runtime": dev_coverage,
                         "dev_gold_aligned_oracle_diagnostic": oracle_dev_coverage},
            "runtime_proposal_artifact_hashes": proposal_artifacts_start,
            "runtime_dev_proposal_fingerprint": proposal_fingerprint,
            "ablations": {"initial_model": initial},
            "selection": {"selected": selected_kind, "metric_order": ["sentence_accuracy", "covered_mean_nll"]},
            "dev_metrics": final_dev,
            "dev_gold_aligned_oracle_diagnostic": oracle_dev_diagnostic,
            "resource_smoke_before_training": smoke,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
            "artifact_hashes": artifacts,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staged = SentenceRanker.load(temporary)
        if len(staged.score_pairs(["문맥"] * 2, ["첫 문장", "둘째 문장"])) != 2:
            raise RuntimeError("staged sentence ranker failed its inference smoke test")
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite existing checkpoint: {args.output}")
        temporary.rename(args.output)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, default=ROOT / "data/external/koelectra-small-v3")
    parser.add_argument("--train", type=Path, default=ROOT / "data/challenge/train.jsonl")
    parser.add_argument("--dev", type=Path, default=ROOT / "data/challenge/dev.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/challenge/manifest.json")
    parser.add_argument("--augmentation", type=Path,
                        help="validated train-only context augmentation JSONL")
    parser.add_argument("--augmentation-manifest", type=Path,
                        help="manifest binding augmentation to the frozen train split")
    parser.add_argument("--augmentation-attestation", type=Path,
                        help="independent zero-overlap attestation for frozen held-out contexts")
    parser.add_argument("--output", type=Path,
                        help="checkpoint directory; augmented default is sentence_ranker_augmented")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--candidates", type=int, choices=(4, 8), default=8)
    parser.add_argument("--group-batch", type=int, default=4)
    parser.add_argument("--inference-batch-size", type=int, choices=(4, 8), default=8)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--top-layers", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--examples-per-category", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), ensure_ascii=False, indent=2))
