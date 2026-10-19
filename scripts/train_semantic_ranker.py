#!/usr/bin/env python3
"""Fine-tune a pinned local Korean ELECTRA snapshot as a listwise ranker."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from hanglish.core.lattice import CandidateGenerator
from hanglish.models.semantic_ranker import SemanticRanker, encode_pairs
from hanglish.models.training_guard import validate_training_augmentation, validate_training_splits
from scripts.train_context_ranker import RankingExample, build_examples


ROOT = Path(__file__).resolve().parents[1]
PINNED_MODEL = "monologg/koelectra-small-v3-discriminator"
PINNED_REVISION = "7488f8db0f208beff4a1f3f9bb3ed04650a89ed7"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_source_files(directory: Path, *, pretrained=False) -> list[Path]:
    names = ({"config.json", "vocab.txt", "tokenizer_config.json", "pytorch_model.bin",
              "SOURCE.json", "LICENSE"} if pretrained else
             {"model.safetensors", "config.json", "tokenizer.json", "vocab.txt",
              "tokenizer_config.json", "special_tokens_map.json", "metadata.json"})
    entries = list(directory.iterdir())
    unexpected = sorted(path.name for path in entries if path.name not in names)
    if unexpected:
        raise ValueError(f"checkpoint source has unverified entries: {unexpected}")
    return sorted(path for path in entries if path.is_file())


def require_fresh_output(path: Path) -> None:
    """Reserve no files, but fail before training if publication would overwrite."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def training_source_hashes(args) -> dict[str, str]:
    paths = [
        args.train, args.dev, args.manifest, ROOT / "data/challenge/train_lexicon.json",
        ROOT / "data/morphology_lexicon.json", ROOT / "data/workbench_lexicon.json",
        ROOT / "data/common_english.txt", ROOT / "data/it_terms.txt",
        ROOT / "hanglish/core/lattice.py", ROOT / "hanglish/core/lexical_data.py",
        ROOT / "hanglish/core/phonology.py", ROOT / "hanglish/models/training_guard.py",
        ROOT / "hanglish/models/semantic_ranker.py", ROOT / "scripts/train_context_ranker.py",
        Path(__file__).resolve(),
    ]
    for value in (args.augmentation, args.augmentation_manifest, args.augmentation_attestation):
        if value is not None:
            paths.append(value)
    parent = args.resume if args.resume is not None else args.base_model
    paths.extend(checkpoint_source_files(Path(parent), pretrained=args.resume is None))
    return {str(Path(path).resolve()): _sha256(Path(path)) for path in paths}


def validate_base_snapshot(path: Path) -> dict:
    source = json.loads((path / "SOURCE.json").read_text(encoding="utf-8"))
    if source.get("repository") != PINNED_MODEL or source.get("revision") != PINNED_REVISION:
        raise ValueError("base model is not the pinned KoELECTRA revision")
    for relative, expected in source.get("files", {}).items():
        artifact = path / relative
        if not artifact.is_file() or _sha256(artifact) != expected:
            raise ValueError(f"base model checksum mismatch: {relative}")
    return source


def parent_provenance(path: Path, metadata: dict) -> dict:
    return {
        "path": str(path),
        "metadata_sha256": _sha256(path / "metadata.json"),
        "artifact_hashes": dict(metadata.get("artifact_hashes", {})),
        "source_hashes": dict(metadata.get("source_hashes", {})),
        "augmentation_data_guard": metadata.get("augmentation_data_guard"),
        "dev_metrics": metadata.get("dev_metrics"),
    }


def validate_parent_training_provenance(metadata: dict, guard: dict,
                                        augmentation_guard: dict | None = None) -> dict:
    """Bind resumed weights to the pinned base and currently approved train/dev."""
    if (metadata.get("base_model") != PINNED_MODEL
            or metadata.get("base_revision") != PINNED_REVISION):
        raise ValueError("resume checkpoint uses an unapproved base model revision")
    base_source = metadata.get("base_source")
    if (not isinstance(base_source, dict)
            or base_source.get("repository") != PINNED_MODEL
            or base_source.get("revision") != PINNED_REVISION):
        raise ValueError("resume checkpoint has invalid pinned base provenance")
    if metadata.get("test_data_read") is not False:
        raise ValueError("resume checkpoint does not attest test_data_read false")
    def by_role(source_hashes):
        if not isinstance(source_hashes, dict):
            return None
        resolved = {}
        for name, value in source_hashes.items():
            basename = Path(name).name
            role = basename.removesuffix(".jsonl") if basename in {"train.jsonl", "dev.jsonl"} else name
            if role not in {"train", "dev"} or role in resolved:
                return None
            resolved[role] = value
        return resolved

    parent_sources = metadata.get("source_hashes")
    approved_sources = by_role(guard["source_hashes"])
    if by_role(parent_sources) != approved_sources:
        raise ValueError("resume checkpoint source hashes differ from approved train/dev")
    if guard["held_out_test_sha256"] in parent_sources.values():
        raise ValueError("resume checkpoint source roles contain the held-out test hash")
    approved_lexicon = guard["candidate_lexicon"]["sha256"]
    parent_lexicon = metadata.get("candidate_generator", {}).get("data_hashes", {}).get(
        "data/challenge/train_lexicon.json")
    if parent_lexicon != approved_lexicon:
        raise ValueError("resume checkpoint candidate lexicon differs from approved corpus")
    nested = metadata.get("training_data_guard")
    if nested is not None:
        if (not isinstance(nested, dict) or nested.get("held_out_test_read") is not False
                or nested.get("corpus_version") != guard["corpus_version"]
                or by_role(nested.get("source_hashes")) != approved_sources):
            raise ValueError("resume checkpoint training guard differs from approved corpus")
        if "candidate_lexicon" in nested and nested["candidate_lexicon"].get("sha256") != approved_lexicon:
            raise ValueError("resume checkpoint nested candidate lexicon differs from approved corpus")
    parent_augmentation = metadata.get("augmentation_data_guard")
    if parent_augmentation is not None and parent_augmentation != augmentation_guard:
        raise ValueError("resume checkpoint has unapproved augmentation lineage")
    return {
        "validated": True,
        "compatibility": "guarded" if nested is not None else "safe-legacy-v2.1",
        "corpus_version": guard["corpus_version"],
        "source_hashes": dict(parent_sources),
        "held_out_test_read": False,
        "augmentation_data_guard": parent_augmentation,
        "candidate_lexicon_sha256": approved_lexicon,
    }


def _log(args, event: str, **values) -> None:
    if args.log_file is None:
        return
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, **values}, ensure_ascii=False) + "\n")


def choose_options(example: RankingExample, limit: int) -> tuple[list[str], tuple[int, ...]]:
    """Keep every positive and the strongest static hard negatives."""
    selected = list(example.positives)
    selected.extend(
        index for index in sorted(range(len(example.options)),
                                  key=lambda i: -example.options[i].score)
        if index not in selected
    )
    selected = selected[:max(limit, len(example.positives))]
    positives = tuple(index for index, original in enumerate(selected)
                      if original in example.positives)
    return [example.options[index].text for index in selected], positives


def _set_trainable_layers(model, top_layers: int) -> dict:
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    encoder = getattr(getattr(model.base_model, "encoder", None), "layer", None)
    if encoder is None:
        electra = getattr(model.base_model, "electra", model.base_model)
        encoder = getattr(getattr(electra, "encoder", None), "layer", None)
    if encoder is None:
        raise RuntimeError("unsupported ELECTRA encoder structure")
    for layer in encoder[-top_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    for name, parameter in model.named_parameters():
        if not name.startswith(model.base_model_prefix + "."):
            parameter.requires_grad = True
    return {
        "encoder_layers": len(encoder),
        "unfrozen_encoder_layers": min(top_layers, len(encoder)),
        "trainable_parameters": sum(x.numel() for x in model.parameters() if x.requires_grad),
        "total_parameters": sum(x.numel() for x in model.parameters()),
    }


def _batch_loss(model, tokenizer, examples: list[RankingExample], args) -> torch.Tensor:
    contexts: list[str] = []
    candidates_flat: list[str] = []
    groups: list[tuple[int, int, tuple[int, ...]]] = []
    for example in examples:
        candidates, positives = choose_options(example, args.candidates)
        start = len(candidates_flat)
        candidates_flat.extend(candidates)
        contexts.extend([example.context] * len(candidates))
        groups.append((start, len(candidates_flat), positives))
    encoded = encode_pairs(tokenizer, contexts, candidates_flat, args.max_length)
    encoded = {name: value.to(args.device) for name, value in encoded.items()}
    scores = model(**encoded).logits.reshape(-1)
    losses = []
    for start, end, positives in groups:
        group = scores[start:end]
        positive = torch.tensor(positives, device=group.device)
        losses.append(torch.logsumexp(group, 0) - torch.logsumexp(group[positive], 0))
    return torch.stack(losses).mean()


@torch.inference_mode()
def evaluate(model, tokenizer, examples: list[RankingExample], args) -> dict:
    model.eval()
    total = correct = 0
    losses = []
    categories: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for start in range(0, len(examples), args.eval_group_batch):
        batch = examples[start:start + args.eval_group_batch]
        for example in batch:
            candidates, positives = choose_options(example, args.candidates)
            encoded = encode_pairs(
                tokenizer, [example.context] * len(candidates), candidates, args.max_length)
            encoded = {name: value.to(args.device) for name, value in encoded.items()}
            scores = model(**encoded).logits.reshape(-1)
            positive = torch.tensor(positives, device=scores.device)
            losses.append(float(torch.logsumexp(scores, 0) - torch.logsumexp(scores[positive], 0)))
            hit = int(int(scores.argmax()) in positives)
            correct += hit
            total += 1
            categories[example.category][0] += hit
            categories[example.category][1] += 1
    return {
        "oracle_rank_accuracy": correct / total if total else 0.0,
        "decisions": total,
        "mean_listwise_nll": sum(losses) / len(losses) if losses else None,
        "categories": {key: {"correct": value[0], "total": value[1],
                             "accuracy": value[0] / value[1]}
                       for key, value in sorted(categories.items())},
    }


def static_accuracy(examples: list[RankingExample]) -> float:
    correct = 0
    for example in examples:
        # Measure the runtime's full static group, without the training-only
        # positive-preserving top-k selection.
        winner = max(range(len(example.options)), key=lambda i: example.options[i].score)
        correct += int(winner in example.positives)
    return correct / len(examples) if examples else 0.0


def selection_key(metrics: dict) -> tuple[float, float]:
    """Prefer accuracy, then lower listwise NLL without mishandling exact zero."""
    nll = metrics.get("mean_listwise_nll")
    return metrics["oracle_rank_accuracy"], -(nll if nll is not None else float("inf"))


def resource_smoke(model, tokenizer, args) -> dict:
    model.eval()
    contexts = ["물건의 가격과 쓸모를 평가한다. [MASK]"] * args.inference_batch_size
    candidates = (["가치", "같이", "값", "가격"] * args.inference_batch_size)[:args.inference_batch_size]
    encoded = encode_pairs(tokenizer, contexts, candidates, args.max_length)
    encoded = {name: value.to(args.device) for name, value in encoded.items()}
    with torch.inference_mode():
        model(**encoded)
    samples = []
    for _ in range(10):
        begin = time.perf_counter()
        with torch.inference_mode():
            model(**encoded)
        samples.append((time.perf_counter() - begin) * 1000)
    return {"batch_size": args.inference_batch_size,
            "median_ms": sorted(samples)[len(samples) // 2],
            "max_ms": max(samples)}


def _optimizer(model, args, head_only: bool):
    if head_only:
        parameters = [x for name, x in model.named_parameters()
                      if x.requires_grad and not name.startswith(model.base_model_prefix + ".")]
        return torch.optim.AdamW(parameters, lr=args.head_lr, weight_decay=args.weight_decay)
    encoder, head = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (encoder if name.startswith(model.base_model_prefix + ".") else head).append(parameter)
    return torch.optim.AdamW([
        {"params": encoder, "lr": args.encoder_lr},
        {"params": head, "lr": args.head_lr},
    ], weight_decay=args.weight_decay)


def balanced_order(examples: list[RankingExample], per_category: int, seed: int) -> list[int]:
    """Sample every train-only category equally, repeating rare groups."""
    by_category: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_category[example.category].append(index)
    randomizer = random.Random(seed)
    order = []
    for category in sorted(by_category):
        indices = by_category[category]
        if per_category <= 0:
            order.extend(indices)
        elif len(indices) >= per_category:
            order.extend(randomizer.sample(indices, per_category))
        else:
            order.extend(randomizer.choice(indices) for _ in range(per_category))
    randomizer.shuffle(order)
    return order


def _run_epochs(model, tokenizer, examples, dev_examples, args, epochs, *, head_only, seed_offset):
    optimizer = _optimizer(model, args, head_only)
    history = []
    best_key = None
    best_state = None
    for epoch in range(epochs):
        order = balanced_order(
            examples, args.examples_per_category, args.seed + seed_offset + epoch)
        model.train()
        losses = []
        for start in range(0, len(order), args.group_batch):
            batch = [examples[index] for index in order[start:start + args.group_batch]]
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(model, tokenizer, batch, args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        dev = evaluate(model, tokenizer, dev_examples, args)
        history.append({"mean_loss": sum(losses) / len(losses), "dev": dev})
        _log(args, "epoch", phase="head" if head_only else "encoder",
             epoch=epoch + 1, mean_loss=history[-1]["mean_loss"], dev=dev)
        key = selection_key(dev)
        if best_key is None or key > best_key:
            best_key = key
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
    return history, best_state


def train(args) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, __version__ as transformers_version
    from tokenizers import __version__ as tokenizers_version
    from safetensors import __version__ as safetensors_version

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    augmentation_args = (args.augmentation, args.augmentation_manifest,
                         args.augmentation_attestation)
    if any(value is not None for value in augmentation_args) and not all(
            value is not None for value in augmentation_args):
        raise ValueError("augmentation, manifest, and attestation must be provided together")
    if args.output is None:
        name = "semantic_ranker_augmented" if args.augmentation else "semantic_ranker"
        args.output = ROOT / "checkpoints" / name
    if args.augmentation and args.output.resolve() == (ROOT / "checkpoints/semantic_ranker").resolve():
        raise ValueError("augmented training must use a side-by-side output checkpoint")
    require_fresh_output(args.output)
    guard = validate_training_splits(args.train, args.dev, args.manifest)
    augmentation_guard = None
    if args.augmentation is not None:
        augmentation_guard = validate_training_augmentation(
            args.augmentation, args.augmentation_manifest, args.augmentation_attestation,
            args.train, args.dev, args.manifest, guard)
    source_hashes_start = training_source_hashes(args)
    generator = CandidateGenerator()
    train_examples, train_coverage = build_examples(args.train, generator)
    if args.augmentation is not None:
        augmented_examples, augmented_coverage = build_examples(args.augmentation, generator)
        train_examples.extend(augmented_examples)
        train_coverage = {"base": train_coverage, "augmentation": augmented_coverage,
                          "combined_rank_examples": len(train_examples)}
    dev_examples, dev_coverage = build_examples(args.dev, generator)
    if args.max_train_examples:
        train_examples = train_examples[:args.max_train_examples]
    parent = None
    if args.resume is not None:
        resumed = SemanticRanker.load(args.resume)
        resume_validation = validate_parent_training_provenance(
            resumed.metadata, guard, augmentation_guard)
        tokenizer = resumed.tokenizer
        model = resumed.model.to(args.device)
        base_source = resumed.metadata["base_source"]
        parent = parent_provenance(args.resume, resumed.metadata)
        parent["training_provenance_validation"] = resume_validation
    else:
        base_source = validate_base_snapshot(args.base_model)
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, local_files_only=True, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model, local_files_only=True, trust_remote_code=False,
            weights_only=True, num_labels=1).to(args.device)
    layer_info = _set_trainable_layers(model, args.top_layers)
    smoke_before = resource_smoke(model, tokenizer, args)
    untrained_dev = evaluate(model, tokenizer, dev_examples, args)
    static_dev = static_accuracy(dev_examples)
    initial_state = {name: value.detach().cpu().clone()
                     for name, value in model.state_dict().items()}
    _log(args, "start", resume=str(args.resume) if args.resume else None,
         train_examples=len(train_examples), initial_dev=untrained_dev,
         source_hashes=guard["source_hashes"])
    start = time.monotonic()

    # The randomly initialized task head first learns a stable score scale while
    # the pretrained Korean representation remains untouched.
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    warmup_history, _ = _run_epochs(
        model, tokenizer, train_examples, dev_examples, args,
        args.warmup_epochs, head_only=True, seed_offset=0)
    layer_info = _set_trainable_layers(model, args.top_layers)
    tune_history, best_state = _run_epochs(
        model, tokenizer, train_examples, dev_examples, args,
        args.epochs, head_only=False, seed_offset=100)
    best_tuned = max(
        (entry["dev"] for entry in tune_history),
        key=selection_key,
        default={"oracle_rank_accuracy": -1.0, "mean_listwise_nll": None})
    best_tuned_accuracy = best_tuned["oracle_rank_accuracy"]
    initial_key = selection_key(untrained_dev)
    tuned_key = selection_key(best_tuned)
    retained_parent = initial_key >= tuned_key
    if retained_parent:
        model.load_state_dict(initial_state)
    elif best_state is not None:
        model.load_state_dict(best_state)
    dev_metrics = evaluate(model, tokenizer, dev_examples, args)
    elapsed = time.monotonic() - start
    if augmentation_guard is not None:
        if (_sha256(args.augmentation) != augmentation_guard["augmentation_sha256"]
                or _sha256(args.augmentation_manifest) != augmentation_guard["manifest_sha256"]
                or _sha256(args.augmentation_attestation)
                != augmentation_guard["held_out_context_attestation_sha256"]):
            raise RuntimeError("augmentation sources changed during semantic-ranker training")
    source_hashes_end = training_source_hashes(args)
    if source_hashes_start != source_hashes_end:
        raise RuntimeError("semantic-ranker training sources changed during training")

    temporary = Path(tempfile.mkdtemp(prefix="semantic-ranker-", dir=args.output.parent))
    try:
        model.save_pretrained(temporary, safe_serialization=True)
        tokenizer.save_pretrained(temporary)
        artifact_hashes = {
            path.name: _sha256(path) for path in temporary.iterdir()
            if path.is_file()
        }
        metadata = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base_model": PINNED_MODEL,
            "base_revision": PINNED_REVISION,
            "base_model_path": str(args.base_model),
            "base_source": base_source,
            "parent_checkpoint": parent,
            "versions": {"torch": torch.__version__, "transformers": transformers_version,
                         "tokenizers": tokenizers_version, "safetensors": safetensors_version},
            "seed": args.seed,
            "training_data_guard": guard,
            "augmentation_data_guard": augmentation_guard,
            "test_data_read": guard["held_out_test_read"],
            "source_hashes": guard["source_hashes"],
            "frozen_source_hashes_start": source_hashes_start,
            "frozen_source_hashes_end": source_hashes_end,
            "code_hashes": {
                "hanglish/models/semantic_ranker.py": _sha256(ROOT / "hanglish/models/semantic_ranker.py"),
                "scripts/train_semantic_ranker.py": _sha256(Path(__file__).resolve()),
                "scripts/train_context_ranker.py": _sha256(ROOT / "scripts/train_context_ranker.py"),
                "hanglish/core/lattice.py": _sha256(ROOT / "hanglish/core/lattice.py"),
                "hanglish/core/lexical_data.py": _sha256(ROOT / "hanglish/core/lexical_data.py"),
                "hanglish/core/phonology.py": _sha256(ROOT / "hanglish/core/phonology.py"),
            },
            "candidate_generator": {
                "external_version": generator.external_version,
                "external_word_count": generator.external_word_count,
                "external_fingerprint": generator.external_fingerprint,
                "dictionary_fingerprint": generator.dictionary_fingerprint,
                "dictionary_rejected_rows": generator.dictionary_rejected_rows,
                "data_hashes": {
                    "data/challenge/train_lexicon.json": _sha256(ROOT / "data/challenge/train_lexicon.json"),
                    "data/morphology_lexicon.json": _sha256(ROOT / "data/morphology_lexicon.json"),
                    "data/workbench_lexicon.json": _sha256(ROOT / "data/workbench_lexicon.json"),
                },
            },
            "max_length": args.max_length,
            "inference_batch_size": args.inference_batch_size,
            "training": {"warmup_epochs": args.warmup_epochs, "epochs": args.epochs,
                         "examples_used": len(train_examples),
                         "examples_per_category_per_epoch": args.examples_per_category,
                         "examples_sampled_per_epoch": len(balanced_order(
                             train_examples, args.examples_per_category, args.seed)),
                         "group_batch": args.group_batch, "candidates": args.candidates,
                         "head_lr": args.head_lr, "encoder_lr": args.encoder_lr,
                         "weight_decay": args.weight_decay, **layer_info,
                         "warmup_history": warmup_history, "tune_history": tune_history},
            "coverage": {"train": train_coverage, "dev": dev_coverage},
            "ablations": {"initial_model": untrained_dev,
                          "initial_model_kind": "resumed_checkpoint" if parent else "pretrained_encoder_random_head",
                          "static_option_accuracy": static_dev},
            "dev_metrics": dev_metrics,
            "selection": {
                "initial_dev_accuracy": untrained_dev["oracle_rank_accuracy"],
                "initial_dev_mean_listwise_nll": untrained_dev["mean_listwise_nll"],
                "best_tuned_dev_accuracy": best_tuned_accuracy,
                "best_tuned_dev_mean_listwise_nll": best_tuned["mean_listwise_nll"],
                "metric_order": ["oracle_rank_accuracy", "mean_listwise_nll"],
                "retained_parent_weights": retained_parent,
            },
            "resource_smoke_before_training": smoke_before,
            "elapsed_seconds": elapsed,
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
            "artifact_hashes": artifact_hashes,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staged = SemanticRanker.load(temporary)
        if len(staged.score_pairs(["[MASK]"] * 2, ["같이", "가치"])) != 2:
            raise RuntimeError("staged semantic ranker failed its inference smoke test")
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite existing checkpoint: {args.output}")
        temporary.rename(args.output)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    _log(args, "complete", output=str(args.output), dev=dev_metrics,
         retained_parent_weights=retained_parent)
    return metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path,
                        default=ROOT / "data/external/koelectra-small-v3")
    parser.add_argument("--resume", type=Path,
                        help="verified local SemanticRanker checkpoint to continue")
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
                        help="checkpoint directory; augmented default is semantic_ranker_augmented")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--candidates", type=int, choices=(4, 8), default=4)
    parser.add_argument("--group-batch", type=int, default=4)
    parser.add_argument("--eval-group-batch", type=int, default=4)
    parser.add_argument("--inference-batch-size", type=int, choices=(4, 8), default=8)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--top-layers", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--max-train-examples", type=int, default=0,
                        help="bounded resource smoke; zero uses every training example")
    parser.add_argument("--examples-per-category", type=int, default=300,
                        help="balanced train-only samples per category and epoch; zero uses all")
    parser.add_argument("--log-file", type=Path,
                        help="append machine-readable progress JSONL")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), ensure_ascii=False, indent=2))
