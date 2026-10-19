#!/usr/bin/env python3
"""Train the bounded Roman-to-jamo candidate generator without eval data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hanglish.core.lattice import CandidateGenerator
from hanglish.core.phonology import romanize_word
from hanglish.models.phonetic_transducer import (
    BOS,
    EOS,
    INPUT_TO_ID,
    OUTPUT_TO_ID,
    PAD,
    _Network,
    checkpoint_payload,
    hangul_to_jamo,
    PhoneticTransducer,
    TransducerConfig,
)


SEED = 20260914
SOURCE_FILES = (
    "data/morphology_lexicon.json",
    "data/workbench_lexicon.json",
    "data/challenge/train_lexicon.json",
    "data/challenge/manifest.json",
    "hanglish/core/lattice.py",
    "hanglish/core/lexical_data.py",
    "hanglish/core/phonology.py",
    "hanglish/models/phonetic_transducer.py",
    "scripts/train_phonetic_transducer.py",
)


def stable_bucket(value: str, modulo: int = 20) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % modulo


def training_pairs() -> tuple[list[tuple[str, str]], set[str], dict]:
    generator = CandidateGenerator()
    words = {
        option.text
        for options in generator.index.values()
        for option in options.values()
        if option.text and all("가" <= char <= "힣" for char in option.text)
    }
    pairs = sorted((romanize_word(word), word) for word in words)
    pairs = [(roman, word) for roman, word in pairs if roman.isascii() and roman.isalpha()]
    external_words = []
    frequency_fingerprint = None
    if generator.external_version is not None:
        from wordfreq import get_frequency_dict, top_n_list
        frequencies = get_frequency_dict("ko")
        external_words = [
            word for word in top_n_list("ko", 50000)
            if all("가" <= char <= "힣" for char in word) and len(word) <= 12
        ]
        external = {word: frequencies[word] for word in external_words}
        frequency_fingerprint = hashlib.sha256(
            (generator.external_version + json.dumps(
                external, ensure_ascii=False, sort_keys=True
            )).encode()
        ).hexdigest()
    provenance = {
        "external_package": "wordfreq" if generator.external_version else None,
        "external_version": generator.external_version,
        "external_word_count": generator.external_word_count,
        "external_frequency_fingerprint": frequency_fingerprint,
        "external_words_sha256": hashlib.sha256("\n".join(external_words).encode()).hexdigest(),
        "runtime_dictionary_fingerprint": getattr(generator, "dictionary_fingerprint", None),
        "runtime_dictionary_rejected_rows": getattr(generator, "dictionary_rejected_rows", None),
        "combined_external_fingerprint": getattr(generator, "external_fingerprint", None),
        "training_index_keys": len(generator.index),
        "runtime_dictionary_index_keys": len(getattr(generator, "dictionary_index", {})),
        "runtime_dictionary_in_training_pairs": False,
        "pairs_sha256": hashlib.sha256(
            "\n".join(f"{roman}\t{word}" for roman, word in pairs).encode()
        ).hexdigest(),
    }
    return pairs, generator.nouns, provenance


def lexical_family(word: str, nouns: set[str]) -> str:
    matches = (noun for noun in nouns if word.startswith(noun))
    return max(matches, key=len, default=word)


def split_pairs(pairs: list[tuple[str, str]], nouns: set[str]):
    lexical_dev, remainder = [], []
    for pair in pairs:
        destination = lexical_dev if stable_bucket(lexical_family(pair[1], nouns)) == 0 else remainder
        destination.append(pair)
    compositional_dev, train = [], []
    for pair in remainder:
        (compositional_dev if stable_bucket(pair[1]) == 1 else train).append(pair)
    return train, compositional_dev, lexical_dev


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        roman, word = self.pairs[index]
        source = [INPUT_TO_ID[BOS], *(INPUT_TO_ID[c] for c in roman), INPUT_TO_ID[EOS]]
        target = [OUTPUT_TO_ID[token] for token in hangul_to_jamo(word)]
        return source, [OUTPUT_TO_ID[BOS], *target], [*target, OUTPUT_TO_ID[EOS]]


def collate(items):
    source_length = max(len(item[0]) for item in items)
    target_length = max(len(item[1]) for item in items)
    sources, inputs, outputs = [], [], []
    for source, target_input, target_output in items:
        sources.append(source + [INPUT_TO_ID[PAD]] * (source_length - len(source)))
        inputs.append(target_input + [OUTPUT_TO_ID[PAD]] * (target_length - len(target_input)))
        outputs.append(target_output + [OUTPUT_TO_ID[PAD]] * (target_length - len(target_output)))
    return tuple(torch.tensor(values, dtype=torch.long) for values in (sources, inputs, outputs))


def loss_on(network, loader, criterion, device):
    network.eval()
    total, tokens = 0.0, 0
    with torch.inference_mode():
        for source, target_input, target_output in loader:
            source, target_input, target_output = (x.to(device) for x in (source, target_input, target_output))
            logits = network(source, target_input)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target_output.reshape(-1))
            count = target_output.ne(OUTPUT_TO_ID[PAD]).sum().item()
            total += loss.item() * count
            tokens += count
    return total / max(tokens, 1)


def exact_metrics(model, pairs, beam_size=4):
    top1 = topk = generated = 0
    started = time.perf_counter()
    for roman, expected in pairs:
        candidates = [word for word, _ in model.generate(roman, beam_size)]
        generated += bool(candidates)
        top1 += bool(candidates and candidates[0] == expected)
        topk += expected in candidates
    count = len(pairs)
    return {
        "words": count,
        "top1_exact": top1 / count if count else 0.0,
        "top4_exact": topk / count if count else 0.0,
        "valid_candidate_rate": generated / count if count else 0.0,
        "seconds": round(time.perf_counter() - started, 3),
    }


def file_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def peak_rss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024


def write_status(path: Path, **values) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def require_unchanged_sources(start: dict[str, str], end: dict[str, str]) -> None:
    if start != end:
        raise RuntimeError("training sources changed while the model was being built")


def run_with_failure_status(status_path: Path, operation) -> None:
    started = time.perf_counter()
    try:
        operation()
    except Exception as error:
        write_status(
            status_path,
            state="failed",
            phase="failed",
            error_type=type(error).__name__,
            error=str(error),
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )
        raise


def train(args, status_path: Path) -> None:
    write_status(status_path, state="running", phase="initializing")
    source_sha256_start = file_hashes()

    staged_checkpoint = args.output.with_name(f"{args.output.stem}.staged{args.output.suffix}")
    staged_metadata = staged_checkpoint.with_suffix(".json")
    staged_checkpoint.unlink(missing_ok=True)
    staged_metadata.unlink(missing_ok=True)

    try:
        _train(args, status_path, source_sha256_start, staged_checkpoint, staged_metadata)
    except Exception:
        staged_checkpoint.unlink(missing_ok=True)
        staged_metadata.unlink(missing_ok=True)
        raise


def _train(args, status_path, source_sha256_start, staged_checkpoint, staged_metadata):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    memory_limit_mb = 2048
    resource_limit_applied = False
    if platform.system() == "Darwin" and hasattr(resource, "RLIMIT_RSS"):
        _, hard_limit = resource.getrlimit(resource.RLIMIT_RSS)
        try:
            resource.setrlimit(resource.RLIMIT_RSS, (memory_limit_mb * 1024**2, hard_limit))
            resource_limit_applied = True
        except ValueError:
            # Some macOS/Python builds expose RLIM_INFINITY as an integer that
            # cannot be passed back.  The per-step RSS guard below remains the
            # enforced process-side cap and this fact is recorded in metadata.
            pass
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    pairs, nouns, data_provenance = training_pairs()
    train, compositional_dev, lexical_dev = split_pairs(pairs, nouns)
    assert not ({word for _, word in train} & {word for _, word in compositional_dev + lexical_dev})
    assert not (
        {lexical_family(word, nouns) for _, word in train}
        & {lexical_family(word, nouns) for _, word in lexical_dev}
    )
    config = TransducerConfig()
    network = _Network(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.002, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=OUTPUT_TO_ID[PAD], label_smoothing=0.02)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(PairDataset(train), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, generator=generator)
    comp_loader = DataLoader(PairDataset(compositional_dev), batch_size=args.batch_size,
                             collate_fn=collate)

    started = time.perf_counter()
    steps = 0
    pilot = None
    epoch_losses = []
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        network.train()
        total, tokens = 0.0, 0
        for source, target_input, target_output in train_loader:
            source, target_input, target_output = (x.to(device) for x in (source, target_input, target_output))
            optimizer.zero_grad(set_to_none=True)
            logits = network(source, target_input)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), target_output.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            if peak_rss_mb() > memory_limit_mb:
                raise MemoryError(f"training exceeded the {memory_limit_mb} MB memory cap")
            count = target_output.ne(OUTPUT_TO_ID[PAD]).sum().item()
            total += loss.item() * count
            tokens += count
            steps += 1
            if pilot is None and steps >= args.pilot_steps:
                elapsed = time.perf_counter() - started
                pilot = {
                    "steps": steps,
                    "seconds": round(elapsed, 3),
                    "steps_per_second": round(steps / elapsed, 3),
                    "projected_training_seconds": round(
                        len(train_loader) * args.epochs / (steps / elapsed), 1
                    ),
                    "peak_rss_mb": round(peak_rss_mb(), 1),
                }
                print("pilot", json.dumps(pilot, ensure_ascii=False))
                write_status(status_path, state="running", phase="training", pilot=pilot)
        train_loss = total / max(tokens, 1)
        dev_loss = loss_on(network, comp_loader, criterion, device)
        epoch_losses.append({"epoch": epoch, "train": train_loss, "compositional_dev": dev_loss})
        print(f"epoch={epoch} train={train_loss:.4f} dev={dev_loss:.4f}")
        write_status(status_path, state="running", phase="training", epoch=epoch,
                     epochs=args.epochs, train_loss=train_loss, dev_loss=dev_loss,
                     peak_rss_mb=round(peak_rss_mb(), 1), pilot=pilot)
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()}

    assert best_state is not None
    network.load_state_dict(best_state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(network.cpu(), config), staged_checkpoint)
    model = PhoneticTransducer(network.cpu(), config)
    write_status(status_path, state="running", phase="evaluating", epoch=args.epochs,
                 epochs=args.epochs, peak_rss_mb=round(peak_rss_mb(), 1), pilot=pilot)
    metadata = {
        "format_version": 1,
        "seed": SEED,
        "device": str(device),
        "torch_version": torch.__version__,
        "threads": {"intraop": 2, "interop": 1},
        "memory_limit_mb": memory_limit_mb,
        "os_resource_limit_applied": resource_limit_applied,
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "parameter_count": parameter_count,
        "pairs": {"total": len(pairs), "train": len(train),
                  "compositional_dev": len(compositional_dev), "lexical_dev": len(lexical_dev)},
        "data_provenance": data_provenance,
        "split_policy": {
            "lexical_dev": "sha256(longest known noun stem or whole word) modulo 20 == 0",
            "compositional_dev": "remaining whole-word sha256 modulo 20 == 1",
            "note": "compositional dev may share a noun stem with train; lexical dev holds its noun family out",
        },
        "pilot": pilot,
        "epoch_losses": epoch_losses,
        "best_epoch": min(epoch_losses, key=lambda item: item["compositional_dev"])["epoch"],
        "metrics": {
            "compositional_dev": exact_metrics(model, compositional_dev),
            "lexical_dev": exact_metrics(model, lexical_dev),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "checkpoint_sha256": hashlib.sha256(staged_checkpoint.read_bytes()).hexdigest(),
        "source_sha256_start": source_sha256_start,
        "source_sha256_end": file_hashes(),
        "data_policy": "CandidateGenerator.index only; challenge dev/test JSONL are not read",
    }
    require_unchanged_sources(metadata["source_sha256_start"], metadata["source_sha256_end"])
    staged_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    PhoneticTransducer.load(staged_checkpoint)

    metadata_path = args.output.with_suffix(".json")
    # Publishing metadata first creates at most a brief fail-closed checksum
    # mismatch with the old checkpoint; a loader can never accept a wrong pair.
    os.replace(staged_metadata, metadata_path)
    os.replace(staged_checkpoint, args.output)
    PhoneticTransducer.load(args.output)
    print(json.dumps(metadata["metrics"], ensure_ascii=False, indent=2))
    write_status(status_path, state="complete", phase="complete",
                 checkpoint=str(args.output), checkpoint_sha256=metadata["checkpoint_sha256"],
                 metrics=metadata["metrics"], peak_rss_mb=metadata["peak_rss_mb"],
                 elapsed_seconds=metadata["elapsed_seconds"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pilot-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints" / "phonetic_transducer.pt")
    parser.add_argument("--status", type=Path)
    args = parser.parse_args()
    status_path = args.status or args.output.with_suffix(".status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    run_with_failure_status(status_path, lambda: train(args, status_path))


if __name__ == "__main__":
    main()
