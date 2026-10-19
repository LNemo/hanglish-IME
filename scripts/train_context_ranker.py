#!/usr/bin/env python3
"""Train the context candidate ranker without reading the challenge test split."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import resource
import tempfile
import time

import torch

from hanglish.core.lattice import CandidateGenerator, WordOption
from hanglish.core.phonology import romanize_word
from hanglish.models.context_ranker import ContextRanker, _RankerNet
from hanglish.models.training_guard import validate_training_splits


ROOT = Path(__file__).resolve().parents[1]
LATIN_WORD = re.compile(r"^[A-Za-z]+(?:-[A-Za-z]+)*$")


@dataclass(frozen=True)
class RankingExample:
    context: str
    options: tuple[WordOption, ...]
    positives: tuple[int, ...]
    case_id: str
    category: str


def _compact_roman(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", text).lower()


def _segment_cost(source: list[str], target: list[str]) -> int:
    left = _compact_roman("".join(source))
    right = _compact_roman(romanize_word("".join(target)))
    if left == right:
        return 0
    # English/protected tokens keep their spelling while the attached Korean
    # suffix is romanized; this still gives a useful alignment preference.
    return abs(len(left) - len(right)) + sum(a != b for a, b in zip(left, right)) + 3


def align_words(source: str, target: str) -> list[tuple[str, str]]:
    """Align whitespace groups with a bounded dynamic program.

    The grouping handles examples such as ``i beon-e`` -> ``이번에`` and
    ``boksa haetda`` -> ``복사했다`` without relying on evaluation labels at
    inference time.
    """
    left, right = source.split(), target.split()
    n, m = len(left), len(right)
    best: dict[tuple[int, int], tuple[int, list[tuple[str, str]]]] = {(0, 0): (0, [])}
    for i in range(n + 1):
        for j in range(m + 1):
            current = best.get((i, j))
            if current is None:
                continue
            for take_left in range(1, min(3, n - i) + 1):
                for take_right in range(1, min(3, m - j) + 1):
                    source_group = left[i:i + take_left]
                    target_group = right[j:j + take_right]
                    # Prefer the smallest exact groups.  A nonzero merge cost
                    # avoids collapsing an entire sentence when both sides have
                    # the same concatenated romanization.
                    merge_cost = take_left + take_right - 2
                    cost = current[0] + _segment_cost(source_group, target_group) + merge_cost
                    key = (i + take_left, j + take_right)
                    candidate = current[1] + [("".join(source_group), "".join(target_group))]
                    if key not in best or cost < best[key][0]:
                        best[key] = (cost, candidate)
    return best.get((n, m), (0, []))[1]


def _generator_input(source: str) -> str:
    return source.replace("-", "") if LATIN_WORD.fullmatch(source) else source


def _options_for_source(source: str, generator: CandidateGenerator) -> tuple[WordOption, ...]:
    """Try spelling-preserving and Korean separator interpretations."""
    variants = (source, _generator_input(source))
    found: dict[str, WordOption] = {}
    for variant in variants:
        for option in generator.options(variant):
            previous = found.get(option.text)
            if previous is None or option.score > previous.score:
                found[option.text] = option
    return tuple(sorted(found.values(), key=lambda option: (-option.score, option.text)))[:24]


def build_examples(path: Path, generator: CandidateGenerator) -> tuple[list[RankingExample], dict]:
    examples: list[RankingExample] = []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    covered_cases: set[str] = set()
    fully_covered_cases: set[str] = set()
    candidate_decisions = 0
    aligned_segments = 0
    covered_segments = 0
    for row in rows:
        # Multiple expected values express acceptable ambiguity, not one hidden
        # gold choice.  Align every alternative and mark every present answer.
        aligned_by_source: dict[str, set[str]] = {}
        for expected in row["expected"]:
            for source, gold in align_words(row["input"], expected):
                aligned_by_source.setdefault(source, set()).add(gold)
        converted_gold = row["expected"][0]
        row_fully_covered = True
        for source, golds in aligned_by_source.items():
            aligned_segments += 1
            options = _options_for_source(source, generator)
            positives = tuple(index for index, option in enumerate(options) if option.text in golds)
            if not positives:
                row_fully_covered = False
                continue
            covered_segments += 1
            if len(options) < 2:
                continue
            mask_context = converted_gold
            for gold in sorted(golds, key=len, reverse=True):
                if gold in mask_context:
                    mask_context = mask_context.replace(gold, "[MASK]", 1)
                    break
            context = f"{row.get('context', '')} {mask_context}".strip()
            examples.append(RankingExample(context, options, positives, row["id"], row["category"]))
            covered_cases.add(row["id"])
            candidate_decisions += 1
        if row_fully_covered:
            fully_covered_cases.add(row["id"])
    return examples, {
        "rows": len(rows),
        "cases_with_rankable_decision": len(covered_cases),
        "fully_candidate_covered_cases": len(fully_covered_cases),
        "full_case_candidate_coverage": len(fully_covered_cases) / len(rows) if rows else 0.0,
        "aligned_segments": aligned_segments,
        "candidate_covered_segments": covered_segments,
        "segment_candidate_coverage": covered_segments / aligned_segments if aligned_segments else 0.0,
        "oracle_ranking_decisions": candidate_decisions,
    }


def _accuracy(model: _RankerNet | None, examples: list[RankingExample], static: bool = False) -> float:
    if not examples:
        return 0.0
    correct = 0
    if model is not None:
        model.eval()
    with torch.inference_mode():
        for example in examples:
            if static:
                scores = [option.score for option in example.options]
            else:
                assert model is not None
                scores = model([example.context] * len(example.options),
                               [option.text for option in example.options]).tolist()
            correct += int(max(range(len(scores)), key=scores.__getitem__) in example.positives)
    return correct / len(examples)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _temporary_path(directory: Path, suffix: str) -> Path:
    handle, name = tempfile.mkstemp(prefix=".context-ranker-", suffix=suffix, dir=directory)
    os.close(handle)
    return Path(name)


def train(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    guard = validate_training_splits(args.train, args.dev, args.manifest)
    generator = CandidateGenerator()
    train_examples, train_coverage = build_examples(args.train, generator)
    dev_examples, dev_coverage = build_examples(args.dev, generator)
    if not train_examples or not dev_examples:
        raise RuntimeError("training and development data need covered candidate decisions")

    config = {"buckets": args.buckets, "embedding_dim": args.embedding_dim,
              "hidden_dim": args.hidden_dim, "dropout": args.dropout}
    model = _RankerNet(**config).to(device)
    untrained_accuracy = _accuracy(model, dev_examples)
    static_accuracy = _accuracy(None, dev_examples, static=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    start = time.monotonic()
    history = []
    best_accuracy = -1.0
    best_state = None
    stale = 0
    order = list(range(len(train_examples)))
    for epoch in range(1, args.epochs + 1):
        random.Random(args.seed + epoch).shuffle(order)
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, index in enumerate(order, 1):
            example = train_examples[index]
            scores = model([example.context] * len(example.options),
                           [option.text for option in example.options])
            positive = torch.tensor(example.positives, device=device)
            loss = torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[positive], dim=0)
            (loss / args.batch_size).backward()
            total_loss += float(loss.detach())
            if step % args.batch_size == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        dev_accuracy = _accuracy(model, dev_examples)
        history.append({"epoch": epoch, "mean_loss": total_loss / len(order),
                        "oracle_dev_rank_accuracy": dev_accuracy})
        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    elapsed = time.monotonic() - start
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    metadata = {
        "format_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "training_data_guard": guard,
        "source_hashes": guard["source_hashes"],
        "code_hashes": {
            "hanglish/models/context_ranker.py": _sha256(ROOT / "hanglish/models/context_ranker.py"),
            "scripts/train_context_ranker.py": _sha256(Path(__file__).resolve()),
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
                "data/common_english.txt": _sha256(ROOT / "data/common_english.txt"),
                "data/it_terms.txt": _sha256(ROOT / "data/it_terms.txt"),
            },
        },
        "test_data_read": guard["held_out_test_read"],
        "model_config": config,
        "parameter_count": parameter_count,
        "hyperparameters": {"epochs_requested": args.epochs, "epochs_completed": len(history),
                            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
                            "weight_decay": args.weight_decay, "patience": args.patience,
                            "device": str(device), "threads": args.threads},
        "coverage": {"train": train_coverage, "dev": dev_coverage},
        "ablation": {"scope": "oracle ranking on dev segments where a gold candidate exists",
                     "dev_untrained_rank_accuracy": untrained_accuracy,
                     "dev_static_option_accuracy": static_accuracy,
                     "dev_trained_rank_accuracy": best_accuracy},
        "history": history,
        "elapsed_seconds": elapsed,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_temp = _temporary_path(args.output.parent, ".pt")
    metadata_temp = _temporary_path(args.output.parent, ".json")
    try:
        torch.save({"format_version": 1, "model_config": config, "state_dict": best_state}, checkpoint_temp)
        metadata["checkpoint_sha256"] = _sha256(checkpoint_temp)
        metadata_temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(checkpoint_temp, args.output)
        os.replace(metadata_temp, args.output.with_suffix(".json"))
    finally:
        checkpoint_temp.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)
    # Prove the public loader can consume exactly the artifact just written.
    loaded = ContextRanker.load(args.output)
    if len(loaded.score_pairs(["[MASK]"] * 2, ["낫", "낮"])) != 2:
        raise RuntimeError("checkpoint smoke test failed")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data/challenge/train.jsonl")
    parser.add_argument("--dev", type=Path, default=ROOT / "data/challenge/dev.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/challenge/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/context_ranker.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--buckets", type=int, default=8192)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(train(parse_args()), ensure_ascii=False, indent=2))
