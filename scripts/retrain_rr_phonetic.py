#!/usr/bin/env python3
"""Rebuild the frozen RR training corpus and train a non-production candidate.

The runner has three deliberately separate phases. ``prepare`` reconstructs the
historical augmented TRAIN corpus from its pinned TRAIN sources, recomputes only
those inputs with the current romanizer, and freezes the result. ``probe`` runs
16 representative optimizer updates from the historical initialization and
qualifies the resource budget. ``train`` consumes only an accepted, hash-bound
probe and never writes the production checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import types


ROOT = Path(__file__).resolve().parents[1]
BASE_MANIFEST = ROOT / ".omx/artifacts/phonetic-supported-data-v1/manifest.json"
DELTA_MANIFEST = ROOT / ".omx/artifacts/observed-phonetic-delta-v1/manifest.json"
OLD_PREPARED = ROOT / ".omx/artifacts/observed-phonetic-pair-prepared-v3/prepared.json"
OLD_TRAINING = ROOT / ".omx/artifacts/observed-phonetic-augmented-training-v1/report.json"
OLD_INITIAL = ROOT / ".omx/artifacts/observed-phonetic-control-probe-v1/initial.pt"

BASE_ROWS_SHA256 = "cf3e43f75420c17fbe8e1ec118a29a0f6a71c0e55cedbdbae5d7ef3668c84a84"
DELTA_ROWS_SHA256 = "bc299116e618ef37c044d8ac0f3061a55e46efdc1b4f1d0c8d9dc1b8ab655e1a"
INITIAL_TENSOR_SHA256 = "e5be11cadf430c22927377056d57facd3b7eed631d64314de6f1ce2671c3ae37"
INITIAL_CHECKPOINT_SHA256 = "f8b7afe557ef256307373e784b470a3bcc4cedbceb399a10dc0eb019dc3974ea"
HISTORICAL_SOURCE_COMMIT = "f34769ca1981ed83148cad5dfcb5cc444e3521bc"
BASE_PHONOLOGY_SNAPSHOT = (
    ROOT / ".omx/artifacts/phonology-v1-source-snapshot/hanglish/core/phonology.py")
BASE_PHONOLOGY_SHA256 = "9e1f26cc153571c5a102577530a772f1288fc24294f573d3a470af45166e8b49"
DELTA_PHONOLOGY_SHA256 = "e593f06409646b74d8aa8ca62663dcdd1b62759254ae1d1d23fc948fed54b865"

SEED = 20260921
UPDATES = 3954
BATCH_SIZE = 128
PROBE_UPDATES = 16
LIMITS = {
    "whole_seconds": 1800.0,
    "update_seconds": 60.0,
    "rss_bytes": 1 * 1024**3,
    "swap_growth_bytes": 1536 * 1024**2,
}
RECIPE = {
    "seed": SEED,
    "updates": UPDATES,
    "batch_size": BATCH_SIZE,
    "presentations": UPDATES * BATCH_SIZE,
    "optimizer": "AdamW",
    "learning_rate": 0.002,
    "weight_decay": 1e-5,
    "label_smoothing": 0.02,
    "gradient_clip_norm": 1.0,
    "cpu_threads": 1,
    "interop_threads": 1,
    "deterministic_algorithms": True,
    "dtype": "float32",
    "probe_updates": PROBE_UPDATES,
    "projection_multiplier": 1.3,
    "projection_margin_seconds": 60.0,
}
OPERATIONAL_POLICY = {
    "schema": "rr-phonetic-resource-policy-v1",
    "reason": (
        "Use one CPU thread to reduce contention; bound process RSS at 1 GiB. "
        "Allow 1.5 GiB system-wide swap growth because idle observations varied "
        "by about 448 MiB and the prior run reached about 700 MiB while process "
        "RSS remained stable near 540 MiB."),
    "model_architecture_changed": False,
    "optimizer_batch_or_update_count_changed": False,
}
CONFIG = {
    "embedding_dim": 80,
    "encoder_hidden": 160,
    "decoder_hidden": 320,
    "dropout": 0.1,
    "max_roman_length": 48,
}
CURRENT_ROMANIZER_DEPENDENCIES = (
    "hanglish/core/phonology.py",
    "hanglish/core/predicate_boundaries.py",
    "hanglish/core/data/predicate_boundary_inventory.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def runtime_limits(max_seconds: float = LIMITS["whole_seconds"]) -> dict:
    require(type(max_seconds) in (int, float) and math.isfinite(max_seconds)
            and max_seconds > 0, "maximum runtime must be a positive finite number")
    return {**LIMITS, "whole_seconds": float(max_seconds)}


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_sha256(rows: list[dict]) -> str:
    return hashlib.sha256(b"".join(canonical(row) + b"\n" for row in rows)).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    result = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        value = json.loads(line)
        require(isinstance(value, dict), f"invalid JSONL row {number}: {path}")
        result.append(value)
    return result


def atomic_write(path: Path, raw: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value) -> None:
    atomic_write(path, canonical(value) + b"\n")


def write_jsonl(path: Path, rows: list[dict]) -> str:
    raw = b"".join(canonical(row) + b"\n" for row in rows)
    atomic_write(path, raw)
    return hashlib.sha256(raw).hexdigest()


def verify_hashes(root: Path, pins: dict[str, str]) -> None:
    for name, wanted in pins.items():
        if name == "git-blob:historical-phonology":
            raw = subprocess.run(
                ["git", "show", f"{HISTORICAL_SOURCE_COMMIT}:hanglish/core/phonology.py"],
                cwd=root, check=True, capture_output=True).stdout
            require(hashlib.sha256(raw).hexdigest() == wanted,
                    "frozen source differs: git-blob:historical-phonology")
            continue
        path = Path(name)
        path = path if path.is_absolute() else Path(root) / path
        require(path.is_file() and sha256_file(path) == wanted,
                f"frozen source differs: {name}")


def build_refreshed_records(base: list[dict], delta: list[dict], romanizer) -> dict:
    """Refresh TRAIN inputs without reading held-out target surfaces."""
    base_train = [row for row in base if row.get("split") == "train"]
    require(len({(row.get("roman"), row.get("surface")) for row in base_train})
            == len(base_train), "historical base TRAIN contains duplicate pairs")
    held = sorted({(row.get("split"), row.get("roman")) for row in base
                   if row.get("split") != "train"})
    require(all(split in {"lexical_holdout", "morph_combination_holdout"}
                        and isinstance(roman, str) and roman
                        for split, roman in held), "held input inventory differs")
    held_inputs = {roman for _, roman in held}

    sources = []
    for index, row in enumerate(base):
        if row.get("split") == "train":
            sources.append(("base", index, row))
    for index, row in enumerate(delta):
        require(row.get("split") == "train", "delta contains a non-TRAIN row")
        sources.append(("delta", index, row))

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    differences, excluded = [], []
    for source, index, row in sources:
        old_roman, surface = row.get("roman"), row.get("surface")
        require(isinstance(old_roman, str) and old_roman, "TRAIN old roman missing")
        require(isinstance(surface, str) and surface, "TRAIN surface missing")
        roman = romanizer(surface)
        require(isinstance(roman, str) and roman.isascii() and roman.isalpha(),
                f"current romanizer produced unsupported input: {surface}")
        require(len(roman) <= CONFIG["max_roman_length"],
                f"current input exceeds model limit: {surface}")
        origin = {"source": source, "index": index, "old_roman": old_roman}
        if old_roman != roman:
            differences.append({"source": source, "index": index, "surface": surface,
                                "old_roman": old_roman, "new_roman": roman})
        if roman in held_inputs:
            excluded.append({"reason": "historical_held_input_collision",
                             "roman": roman, "surface": surface, **origin})
            continue
        grouped[(roman, surface)].append(origin)

    records = []
    duplicate_sources = 0
    for pair_index, ((roman, surface), origins) in enumerate(sorted(grouped.items())):
        duplicate_sources += len(origins) - 1
        records.append({
            "index": pair_index,
            "pair_id": hashlib.sha256(f"{roman}\t{surface}".encode()).hexdigest(),
            "roman": roman,
            "surface": surface,
            "split": "train",
            "origins": sorted(origins, key=lambda row: (row["source"], row["index"])),
        })
    require(not ({row["roman"] for row in records} & held_inputs),
            "refreshed TRAIN input crosses historical holdout")
    return {
        "records": records,
        "differences": sorted(differences, key=lambda row: (
            row["new_roman"], row["surface"], row["source"], row["index"])),
        "excluded": sorted(excluded, key=lambda row: (
            row["roman"], row["surface"], row["source"], row["index"])),
        "held_inputs": [{"split": split, "roman": roman} for split, roman in held],
        "counts": {
            "historical_train_sources": len(sources),
            "refreshed_train_pairs": len(records),
            "roman_changed_sources": len(differences),
            "held_input_collision_sources_excluded": len(excluded),
            "duplicate_sources_deduplicated": duplicate_sources,
            "historical_held_inputs": len(held_inputs),
        },
    }


def reconstruct_historical() -> tuple[list[dict], list[dict], dict[str, str]]:
    """Recreate the missing JSONL files from their still-frozen TRAIN sources."""
    delta_phonology = subprocess.run(
        ["git", "show", f"{HISTORICAL_SOURCE_COMMIT}:hanglish/core/phonology.py"],
        cwd=ROOT, check=True, capture_output=True).stdout
    require(hashlib.sha256(delta_phonology).hexdigest() == DELTA_PHONOLOGY_SHA256,
            "historical delta romanizer source blob differs")
    require(BASE_PHONOLOGY_SNAPSHOT.is_file()
            and sha256_file(BASE_PHONOLOGY_SNAPSHOT) == BASE_PHONOLOGY_SHA256,
            "historical base romanizer snapshot differs")
    # The old JSONL files were intentionally pruned. Load the authenticated
    # historical romanizer while rebuilding them, then authenticate the entire
    # reconstructed serialization against the hashes recorded at training time.
    def install_phonology(raw: bytes, label: str) -> None:
        module = types.ModuleType("hanglish.core.phonology")
        module.__file__ = label
        module.__package__ = "hanglish.core"
        sys.modules["hanglish.core.phonology"] = module
        exec(compile(raw, label, "exec"), module.__dict__)

    install_phonology(BASE_PHONOLOGY_SNAPSHOT.read_bytes(), str(BASE_PHONOLOGY_SNAPSHOT))
    from scripts import prepare_phonetic_supported_training as supported
    from scripts import audit_phonetic_training_families as family
    from scripts import audit_phonetic_pair_sources as quality
    from hanglish.core.lattice import CandidateGenerator

    quality_raw = supported.QUALITY_REPORT.read_bytes()
    require(hashlib.sha256(quality_raw).hexdigest() == supported.QUALITY_REPORT_SHA256,
            "reviewed source-quality report hash differs")
    quality_report = json.loads(quality_raw)
    require(quality_report.get("schema") == "phonetic-training-source-quality-v1"
            and quality_report["attested_only_alternative"]["pairs"]
                == supported.EXPECTED_PAIRS
            and quality_report["attested_only_alternative"]["pairs_sha256"]
                == supported.EXPECTED_PAIR_SHA256,
            "reviewed source-quality recipe differs")
    generator = CandidateGenerator()
    analyses = family.read_inflect()
    _, train_words = family.load_train_words()
    original, _, anchored_new, _, _ = family.build_corpus(generator, analyses, train_words)
    source_sets, _, frequency_fingerprint = quality.source_sets(
        generator, analyses, train_words)
    supported.verify_external_provenance(quality_report["external"], {
        "wordfreq_version": generator.external_version,
        "wordfreq_frequency_fingerprint": frequency_fingerprint,
        "runtime_external_fingerprint": generator.external_fingerprint,
        "mecab_dictionary_fingerprint": generator.dictionary_fingerprint,
    })
    independent = (source_sets["external_frequency"] | source_sets["mecab_direct"]
                   | source_sets["inflect"])
    supported_original = original & (
        source_sets["train_observed"] | source_sets["explicit_authored"] | independent)
    observed_pairs = {pair for word in train_words
                      if (pair := family.eligible_pair(word)) is not None}
    pairs = supported_original | observed_pairs | anchored_new
    require(len(pairs) == supported.EXPECTED_PAIRS
            and supported.pair_digest(pairs) == supported.EXPECTED_PAIR_SHA256,
            "reconstructed supported pair set differs")
    flags = {pair: supported.source_flags(
        pair, original, anchored_new, source_sets, independent) for pair in pairs}
    aliases = family.productive_aliases(generator, pairs)
    base, _ = supported.build_records(pairs, analyses, aliases, flags)
    require(Counter(row["split"] for row in base) == supported.EXPECTED_SPLIT,
            "reconstructed historical split differs")
    require(rows_sha256(base) == BASE_ROWS_SHA256,
            "reconstructed historical base rows differ")

    # Delta preparation happened after the audited romanizer revision e593f0.
    # Reload only its preparation modules under that exact source blob.
    import importlib
    install_phonology(
        delta_phonology,
        f"git:{HISTORICAL_SOURCE_COMMIT}:hanglish/core/phonology.py")
    family = importlib.reload(family)
    from scripts import prepare_attested_inflection_delta as inflection_delta
    importlib.reload(inflection_delta)
    from scripts import prepare_observed_phonetic_delta as observed
    observed = importlib.reload(observed)
    source_rows = {str(path.relative_to(ROOT)): observed.read_jsonl(path)
                   for path in observed.TRAIN_FILES}
    # The historical reserve was used only to prove source separation. It is
    # intentionally not read here; no evaluation target participates in this run.
    delta, _, _, _, _ = observed.prepare(
        source_rows, base, family.read_inflect(), [], [])
    require(rows_sha256(delta) == DELTA_ROWS_SHA256,
            "reconstructed historical delta rows differ")
    used_names = set(quality_report.get("source_hashes", {}))
    used_names.update({
        str(supported.QUALITY_REPORT.relative_to(ROOT)),
        "scripts/prepare_phonetic_supported_training.py",
        "scripts/prepare_observed_phonetic_delta.py",
        "scripts/prepare_attested_inflection_delta.py",
        "scripts/audit_phonetic_pair_sources.py",
        "scripts/audit_phonetic_training_families.py",
    })
    pins = {name: sha256_file(ROOT / name) for name in sorted(used_names)
            if (ROOT / name).is_file() and name != "hanglish/core/phonology.py"}
    for path in (*observed.TRAIN_FILES, family.MECAB / "Inflect.csv"):
        pins[str(path.relative_to(ROOT))] = sha256_file(path)
    for path in (BASE_MANIFEST, DELTA_MANIFEST, OLD_PREPARED, OLD_TRAINING,
                 ROOT / "hanglish/core/phonology.py",
                 ROOT / "hanglish/core/predicate_boundaries.py",
                 ROOT / "hanglish/core/data/predicate_boundary_inventory.json",
                 ROOT / "hanglish/models/phonetic_transducer.py",
                 ROOT / "scripts/train_phonetic_transducer.py",
                 ROOT / "scripts/train_observed_phonetic_pair.py",
                 ROOT / "scripts/source_span_resources.py",
                 Path(__file__).resolve()):
        require(path.is_file(), f"missing frozen provenance file: {path}")
        pins[str(path.relative_to(ROOT))] = sha256_file(path)
    pins[str(BASE_PHONOLOGY_SNAPSHOT.relative_to(ROOT))] = BASE_PHONOLOGY_SHA256
    pins["git-blob:historical-phonology"] = DELTA_PHONOLOGY_SHA256
    return base, delta, dict(sorted(pins.items()))


def recorded_pin_status(manifest_path: Path) -> dict:
    recorded = read_json(manifest_path).get("source_hashes", {})
    rows = []
    for name, wanted in sorted(recorded.items()):
        path = ROOT / name
        actual = sha256_file(path) if path.is_file() else None
        rows.append({"path": name, "recorded_sha256": wanted,
                     "current_sha256": actual,
                     "status": "match" if actual == wanted else
                               "missing" if actual is None else "changed"})
    return dict(Counter(row["status"] for row in rows)) | {"files": rows}


def current_romanizer():
    """Load the working-tree romanizer separately from the historical module."""
    import importlib.util
    path = ROOT / "hanglish/core/phonology.py"
    spec = importlib.util.spec_from_file_location("_rr_current_phonology", path)
    require(spec is not None and spec.loader is not None, "cannot load current romanizer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.romanize_word


def cyclic_schedule(size: int, *, updates: int = UPDATES,
                    batch_size: int = BATCH_SIZE) -> list[list[int]]:
    require(type(size) is int and size > 0, "schedule size must be positive")
    rng, pending, result = random.Random(SEED), [], []
    while len(result) < updates:
        values = list(range(size)); rng.shuffle(values); pending.extend(values)
        while len(pending) >= batch_size and len(result) < updates:
            result.append(pending[:batch_size]); del pending[:batch_size]
    return result


def batch_cost(records: list[dict], indices: list[int]) -> int:
    selected = [records[index] for index in indices]
    return len(indices) * (max(len(row["roman"]) + 2 for row in selected)
                           + 2 * max(3 * len(row["surface"]) + 1 for row in selected))


def probe_positions(records: list[dict], schedule: list[list[int]]) -> list[int]:
    ordered = sorted(range(len(schedule)), key=lambda index: (batch_cost(
        records, schedule[index]), index))
    positions = {ordered[i * (len(ordered) - 1) // (PROBE_UPDATES - 1)]
                 for i in range(PROBE_UPDATES)}
    require(len(positions) == PROBE_UPDATES, "probe position selection collapsed")
    return sorted(positions, key=lambda index: (batch_cost(records, schedule[index]), index))


def prepare(output: Path, rule_audit: Path | None = None) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    base, delta, pins = reconstruct_historical()
    if rule_audit is not None:
        rule_audit = Path(rule_audit).resolve()
        require(rule_audit.is_file(), "rule audit missing")
        pins[str(rule_audit.relative_to(ROOT))] = sha256_file(rule_audit)
    refreshed = build_refreshed_records(base, delta, current_romanizer())
    schedule = cyclic_schedule(len(refreshed["records"]))
    schedule_summary = {
        "updates": len(schedule), "presentations": sum(map(len, schedule)),
        "schedule_sha256": hashlib.sha256(canonical(schedule)).hexdigest(),
        "total_padded_cost": sum(batch_cost(refreshed["records"], row) for row in schedule),
        "probe_positions": probe_positions(refreshed["records"], schedule),
    }
    output.mkdir(parents=True)
    try:
        files = {
            "train.jsonl": write_jsonl(output / "train.jsonl", refreshed["records"]),
            "roman-differences.jsonl": write_jsonl(
                output / "roman-differences.jsonl", refreshed["differences"]),
            "excluded.jsonl": write_jsonl(output / "excluded.jsonl", refreshed["excluded"]),
            "held-inputs.jsonl": write_jsonl(
                output / "held-inputs.jsonl", refreshed["held_inputs"]),
        }
        manifest = {
            "schema": "rr-phonetic-retrain-prepared-v1", "completed": True,
            "counts": refreshed["counts"], "files": files,
            "historical": {
                "base_rows_sha256": BASE_ROWS_SHA256,
                "delta_rows_sha256": DELTA_ROWS_SHA256,
                "old_augmented_pairs": 84345,
                "initial_checkpoint_sha256": INITIAL_CHECKPOINT_SHA256,
                "initial_tensor_sha256": INITIAL_TENSOR_SHA256,
                "recorded_source_pin_status": {
                    "base": recorded_pin_status(BASE_MANIFEST),
                    "delta": recorded_pin_status(DELTA_MANIFEST),
                },
            },
            "recipe": RECIPE, "model_config": CONFIG, "schedule": schedule_summary,
            "operational_policy": OPERATIONAL_POLICY,
            "source_hashes": pins,
            "contracts": {
                "historical_train_surfaces_only": True,
                "historical_holdout_inputs_only": True,
                "fresh_evaluation_targets_read": False,
                "document_reserve_targets_read": False,
                "historical_held_target_access": (
                    "reconstruction-only preservation closure; omitted from prepared output"),
                "evaluation_targets_used_for_training": False,
                "production_checkpoint_changed": False,
                "training_run": False,
            },
        }
        write_json(output / "prepared.json", manifest)
        verify_hashes(ROOT, pins)
        return manifest
    except BaseException:
        write_json(output / "failure-report.json", {
            "schema": "rr-phonetic-retrain-failure-v1", "completed": False,
            "phase": "prepare",
        })
        raise


def load_prepared(directory: Path) -> tuple[dict, list[dict], str]:
    directory = Path(directory)
    manifest_path = directory / "prepared.json"
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == "rr-phonetic-retrain-prepared-v1"
            and manifest.get("completed") is True
            and manifest.get("recipe") == RECIPE
            and manifest.get("model_config") == CONFIG
            and manifest.get("operational_policy") == OPERATIONAL_POLICY
            and manifest.get("contracts", {}).get("fresh_evaluation_targets_read") is False
            and manifest.get("contracts", {}).get("evaluation_targets_used_for_training") is False,
            "prepared contract differs")
    verify_hashes(ROOT, manifest.get("source_hashes", {}))
    for name, wanted in manifest.get("files", {}).items():
        require(sha256_file(directory / name) == wanted, f"prepared file differs: {name}")
    records = read_jsonl(directory / "train.jsonl")
    schedule = cyclic_schedule(len(records))
    require(manifest.get("schedule", {}).get("schedule_sha256")
            == hashlib.sha256(canonical(schedule)).hexdigest(), "prepared schedule differs")
    return manifest, records, sha256_file(manifest_path)


def torch_modules():
    import torch
    from torch import nn
    from hanglish.models.phonetic_transducer import (
        OUTPUT_TO_ID, PAD, TransducerConfig, _Network, checkpoint_payload,
    )
    from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes
    from scripts.train_phonetic_transducer import PairDataset, collate
    return locals()


def configure_torch(torch) -> None:
    torch.set_num_threads(RECIPE["cpu_threads"])
    try:
        torch.set_num_interop_threads(RECIPE["interop_threads"])
    except RuntimeError:
        require(torch.get_num_interop_threads() == RECIPE["interop_threads"],
                "Torch inter-op threads differ")
    random.seed(SEED); torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)


def tensor_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(canonical(list(value.shape)) + b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def initial_network(modules, initial: Path | None = None):
    torch = modules["torch"]
    config = modules["TransducerConfig"](**CONFIG)
    if initial is not None:
        initial = Path(initial)
        require(initial.is_file(), "explicit initial checkpoint missing")
        require(sha256_file(initial) == INITIAL_CHECKPOINT_SHA256,
                "historical initial checkpoint bytes differ")
        payload = torch.load(initial, map_location="cpu", weights_only=True)
        source = "historical_checkpoint"
    elif OLD_INITIAL.is_file():
        require(sha256_file(OLD_INITIAL) == INITIAL_CHECKPOINT_SHA256,
                "default historical initial checkpoint bytes differ")
        payload = torch.load(OLD_INITIAL, map_location="cpu", weights_only=True)
        source = "historical_checkpoint"
    else:
        # The archived binary may be pruned. The historical seed and exact
        # tensor hash provide an equivalent initialization without inventing it.
        network = modules["_Network"](config).cpu()
        require(tensor_hash(network.state_dict()) == INITIAL_TENSOR_SHA256,
                "seed-recreated historical initialization differs")
        return network, config, "recreated_from_historical_seed"
    require(payload.get("config") == CONFIG, "initial model config differs")
    network = modules["_Network"](config).cpu()
    network.load_state_dict(payload["state_dict"], strict=True)
    require(tensor_hash(network.state_dict()) == INITIAL_TENSOR_SHA256,
            "historical initial tensor differs")
    return network, config, source


def make_batch(records: list[dict], indices: list[int], modules):
    dataset = modules["PairDataset"]([(records[i]["roman"], records[i]["surface"])
                                      for i in indices])
    return modules["collate"]([dataset[i] for i in range(len(dataset))])


def one_update(network, optimizer, batch, modules) -> tuple[float, float]:
    torch, nn = modules["torch"], modules["nn"]
    source, target_input, target_output = batch
    optimizer.zero_grad(set_to_none=True)
    logits = network(source, target_input)
    criterion = nn.CrossEntropyLoss(
        ignore_index=modules["OUTPUT_TO_ID"][modules["PAD"]],
        label_smoothing=RECIPE["label_smoothing"])
    loss = criterion(logits.reshape(-1, logits.shape[-1]), target_output.reshape(-1))
    require(bool(torch.isfinite(loss)), "non-finite training loss")
    loss.backward()
    norm = nn.utils.clip_grad_norm_(network.parameters(), RECIPE["gradient_clip_norm"],
                                   error_if_nonfinite=True)
    optimizer.step()
    require(math.isfinite(float(norm)), "non-finite gradient norm")
    return float(loss.detach()), float(norm)


def run_probe(prepared_dir: Path, output: Path, initial: Path | None = None,
              max_seconds: float = LIMITS["whole_seconds"]) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    prepared, records, prepared_sha = load_prepared(prepared_dir)
    limits = runtime_limits(max_seconds)
    schedule = cyclic_schedule(len(records)); selected = probe_positions(records, schedule)
    output.mkdir(parents=True)
    write_json(output / "status.json", {
        "schema": "rr-phonetic-retrain-status-v1", "state": "running", "phase": "probe"})
    modules = torch_modules(); torch = modules["torch"]
    configure_torch(torch)
    network, _, initial_source = initial_network(modules, initial)
    optimizer = torch.optim.AdamW(network.parameters(), lr=RECIPE["learning_rate"],
                                  weight_decay=RECIPE["weight_decay"])
    started = time.monotonic(); baseline_swap = modules["system_swap_bytes"]()
    rows = []
    with modules["AtomicResourceWatchdog"](
            started, baseline_swap, output / "resources.jsonl", limits=limits) as watchdog:
        for ordinal, position in enumerate(selected, 1):
            before = time.monotonic()
            with watchdog.operation(f"probe_update_{ordinal}", update=True):
                loss, gradient = one_update(
                    network, optimizer, make_batch(records, schedule[position], modules), modules)
            rows.append({"ordinal": ordinal, "schedule_position": position,
                         "padded_cost": batch_cost(records, schedule[position]),
                         "seconds": time.monotonic() - before, "loss": loss,
                         "gradient_norm": gradient})
        final_resource = watchdog.sample_and_append("final")
    observed_cost = sum(row["padded_cost"] for row in rows)
    training_seconds = (sum(row["seconds"] for row in rows) / observed_cost
                        * prepared["schedule"]["total_padded_cost"])
    projected = (RECIPE["projection_margin_seconds"]
                 + RECIPE["projection_multiplier"] * training_seconds)
    accepted = (projected <= limits["whole_seconds"]
                and final_resource["rss_bytes"] <= limits["rss_bytes"]
                and final_resource["swap_growth_bytes"] <= limits["swap_growth_bytes"])
    report = {
        "schema": "rr-phonetic-retrain-probe-v1", "completed": True,
        "accepted": accepted, "prepared_sha256": prepared_sha,
        "recipe": RECIPE, "limits": limits,
        "operational_policy": OPERATIONAL_POLICY, "initial_source": initial_source,
        "initial_tensor_sha256": INITIAL_TENSOR_SHA256,
        "observations": rows, "projected_training_seconds": training_seconds,
        "projected_total_seconds": projected, "final_resource": final_resource,
        "prototype_discarded": True, "training_run": False,
    }
    write_json(output / "report.json", report)
    write_json(output / "completion.json", {
        "schema": "rr-phonetic-retrain-completion-v1", "completed": True,
        "report_sha256": sha256_file(output / "report.json"),
    })
    write_json(output / "status.json", {
        "schema": "rr-phonetic-retrain-status-v1", "state": "complete",
        "phase": "probe", "accepted": accepted,
        "report_sha256": sha256_file(output / "report.json"),
    })
    return report


def validate_probe(probe_dir: Path, prepared_sha: str,
                   limits: dict | None = None) -> dict:
    limits = runtime_limits() if limits is None else limits
    report_path = Path(probe_dir) / "report.json"
    report = read_json(report_path)
    completion = read_json(Path(probe_dir) / "completion.json")
    require(report.get("schema") == "rr-phonetic-retrain-probe-v1"
            and report.get("completed") is True and report.get("accepted") is True
            and report.get("prepared_sha256") == prepared_sha
            and report.get("recipe") == RECIPE
            and report.get("limits") == limits
            and report.get("operational_policy") == OPERATIONAL_POLICY
            and report.get("initial_tensor_sha256") == INITIAL_TENSOR_SHA256
            and len(report.get("observations", [])) == PROBE_UPDATES,
            "accepted probe contract differs")
    require(completion.get("completed") is True
            and completion.get("report_sha256") == sha256_file(report_path),
            "probe completion differs")
    return report


def run_train(prepared_dir: Path, probe_dir: Path, output: Path,
              initial: Path | None = None,
              max_seconds: float = LIMITS["whole_seconds"]) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    prepared, records, prepared_sha = load_prepared(prepared_dir)
    limits = runtime_limits(max_seconds)
    probe = validate_probe(probe_dir, prepared_sha, limits)
    schedule = cyclic_schedule(len(records))
    output.mkdir(parents=True)
    write_json(output / "status.json", {
        "schema": "rr-phonetic-retrain-status-v1", "state": "running", "phase": "train"})
    modules = torch_modules(); torch = modules["torch"]
    configure_torch(torch)
    network, config, initial_source = initial_network(modules, initial)
    optimizer = torch.optim.AdamW(network.parameters(), lr=RECIPE["learning_rate"],
                                  weight_decay=RECIPE["weight_decay"])
    started = time.monotonic(); baseline_swap = modules["system_swap_bytes"]()
    losses, gradients, ledger = [], [], []
    with modules["AtomicResourceWatchdog"](
            started, baseline_swap, output / "resources.jsonl", limits=limits) as watchdog:
        for update, indices in enumerate(schedule, 1):
            before = time.monotonic()
            with watchdog.operation(f"optimizer_update_{update}", update=True):
                loss, gradient = one_update(
                    network, optimizer, make_batch(records, indices, modules), modules)
            losses.append(loss); gradients.append(gradient)
            ledger.append({"update": update, "loss": loss, "gradient_norm": gradient,
                           "seconds": time.monotonic() - before,
                           "padded_cost": batch_cost(records, indices),
                           "pair_indices": indices})
        ledger_sha = write_jsonl(output / "updates.jsonl", ledger)
        staged = output / "phonetic_transducer.staged.pt"
        torch.save(modules["checkpoint_payload"](network, config), staged)
        final_tensor = tensor_hash(network.state_dict())
        final = output / "phonetic_transducer.pt"
        os.replace(staged, final)
        final_sha = sha256_file(final)
        payload = torch.load(final, map_location="cpu", weights_only=True)
        restored = modules["_Network"](config).cpu()
        restored.load_state_dict(payload["state_dict"], strict=True)
        require(tensor_hash(restored.state_dict()) == final_tensor,
                "trained checkpoint reload differs")
        final_resource = watchdog.sample_and_append("final")
    metadata = {
        "format_version": 1, "candidate_only": True,
        "checkpoint_sha256": final_sha, "tensor_sha256": final_tensor,
        "config": CONFIG, "prepared_sha256": prepared_sha,
        "initial_tensor_sha256": INITIAL_TENSOR_SHA256,
        "initial_source": initial_source,
    }
    write_json(output / "phonetic_transducer.json", metadata)
    report = {
        "schema": "rr-phonetic-retrain-training-v1", "completed": True,
        "candidate_only": True, "production_changed": False,
        "prepared_sha256": prepared_sha,
        "probe_report_sha256": sha256_file(Path(probe_dir) / "report.json"),
        "probe_projected_total_seconds": probe["projected_total_seconds"],
        "recipe": RECIPE, "limits": limits,
        "operational_policy": OPERATIONAL_POLICY, "initial_source": initial_source,
        "initial_tensor_sha256": INITIAL_TENSOR_SHA256,
        "updates_completed": len(losses), "updates_sha256": ledger_sha,
        "mean_loss": sum(losses) / len(losses), "final_loss": losses[-1],
        "max_gradient_norm": max(gradients),
        "final_checkpoint_sha256": final_sha, "final_tensor_sha256": final_tensor,
        "final_resource": final_resource, "evaluation_targets_read": False,
    }
    write_json(output / "report.json", report)
    write_json(output / "completion.json", {
        "schema": "rr-phonetic-retrain-completion-v1", "completed": True,
        "report_sha256": sha256_file(output / "report.json"),
        "checkpoint_sha256": final_sha,
    })
    write_json(output / "status.json", {
        "schema": "rr-phonetic-retrain-status-v1", "state": "complete",
        "phase": "train", "report_sha256": sha256_file(output / "report.json"),
        "checkpoint_sha256": final_sha,
    })
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)
    prepare_cmd = commands.add_parser("prepare")
    prepare_cmd.add_argument("--output", type=Path, required=True)
    prepare_cmd.add_argument("--rule-audit", type=Path)
    probe_cmd = commands.add_parser("probe")
    probe_cmd.add_argument("--prepared", type=Path, required=True)
    probe_cmd.add_argument("--output", type=Path, required=True)
    probe_cmd.add_argument("--initial", type=Path)
    probe_cmd.add_argument("--max-seconds", type=float, default=LIMITS["whole_seconds"])
    train_cmd = commands.add_parser("train")
    train_cmd.add_argument("--prepared", type=Path, required=True)
    train_cmd.add_argument("--probe", type=Path, required=True)
    train_cmd.add_argument("--output", type=Path, required=True)
    train_cmd.add_argument("--initial", type=Path)
    train_cmd.add_argument("--max-seconds", type=float, default=LIMITS["whole_seconds"])
    return value


def main() -> None:
    args = parser().parse_args()
    existed = args.output.exists()
    try:
        if args.command == "prepare":
            result = prepare(args.output, args.rule_audit)
        elif args.command == "probe":
            result = run_probe(args.prepared, args.output, args.initial, args.max_seconds)
        else:
            result = run_train(
                args.prepared, args.probe, args.output, args.initial, args.max_seconds)
    except BaseException as error:
        if not existed and args.output.is_dir():
            failure = {
                "schema": "rr-phonetic-retrain-failure-v1", "completed": False,
                "phase": args.command, "error_type": type(error).__name__,
                "error": str(error),
            }
            write_json(args.output / "failure-report.json", failure)
            write_json(args.output / "status.json", {
                "schema": "rr-phonetic-retrain-status-v1", "state": "failed",
                **{key: failure[key] for key in ("phase", "error_type", "error")},
            })
        raise
    print(json.dumps({"schema": result["schema"], "completed": result["completed"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
