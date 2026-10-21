"""Fresh, bounded word-ranker training under the current phonology/runtime."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version as package_version
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.core.lexical_data import FILES as LEXICAL_FILES
from hanglish.core.phonology import romanize_word
from hanglish.models.semantic_ranker import SemanticRanker
from scripts.prepare_natural_word_ranker import (
    Recorder, add_targets, capture_groups, validate_case,
)
from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes
from scripts.train_natural_sentence_ranker import balanced_schedule
from scripts.train_natural_word_ranker import (
    RECIPE, bounded_example, select_groups, word_loss,
)
from scripts.train_sentence_ranker import _optimizer, _set_trainable_layers


PARENT = ROOT / "checkpoints/semantic_ranker"
PARENT_WEIGHT = "54ca7d24b2bf17d1ba8f5181476ce91032acc94149a2baf401fe47e859f2907c"
DATA = ROOT / "data/natural_context/word-ranker-pilot-v1"
TRAIN_FILES = {
    "natural_train": DATA / "natural_train.jsonl",
    "retention_train": DATA / "retention_train.jsonl",
}
DEV_FILES = {
    "natural_dev": DATA / "natural_dev.jsonl",
    "original_dev": ROOT / "data/challenge/dev.jsonl",
    "selection_dev": ROOT / "data/challenge/selection-dev-v1.1/cases.jsonl",
    "reasoning_dev": ROOT / "data/challenge/reasoning-dev-v1.1/cases.jsonl",
}
OUTPUT = ROOT / ".omx/artifacts/context-ranking-improvement-20261021/training"
LIMITS = {
    "whole_seconds": 5400.0,
    "update_seconds": 60.0,
    "rss_bytes": 3 * 1024**3,
    "swap_growth_bytes": 1536 * 1024**2,
}
EXPECTED_ROWS = {"natural_train": 512, "retention_train": 512}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write_json(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def append_update(path, event, **values):
    record = {"event": event, **values}
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _paths_hash(paths):
    return {str(path.relative_to(ROOT)): file_sha256(path)
            for path in sorted(paths) if path.is_file()}


def frozen_pins():
    input_paths = [*TRAIN_FILES.values(), *DEV_FILES.values()]
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/prepare_natural_word_ranker.py",
        ROOT / "scripts/train_natural_word_ranker.py",
        ROOT / "scripts/train_natural_sentence_ranker.py",
        ROOT / "scripts/train_sentence_ranker.py",
        ROOT / "scripts/source_span_resources.py",
        ROOT / "scripts/preflight_pko_seq2seq.py",
        ROOT / "hanglish/core/phonology.py",
        ROOT / "hanglish/core/lattice.py",
        ROOT / "hanglish/core/workbench.py",
        ROOT / "hanglish/core/productive_endings.py",
        ROOT / "hanglish/core/predicate_boundaries.py",
        ROOT / "hanglish/core/data/predicate_boundary_inventory.json",
        ROOT / "hanglish/core/lexical_data.py",
        ROOT / "hanglish/core/numeric_structure.py",
        ROOT / "hanglish/core/neural_converter.py",
        ROOT / "hanglish/models/semantic_ranker.py",
        ROOT / "hanglish/models/context_ranker.py",
        ROOT / "hanglish/models/phonetic_transducer.py",
        ROOT / "hanglish/models/sentence_ranker.py",
        ROOT / "data/morphology_lexicon.json",
        ROOT / "data/workbench_lexicon.json",
        ROOT / "data/common_english.txt",
        ROOT / "data/it_terms.txt",
        ROOT / "data/challenge/train_lexicon.json",
    ]
    model_paths = list(PARENT.iterdir())
    dictionary = ROOT / "data/external/mecab-ko-dic"
    source_paths.extend(dictionary / f"{name}.csv" for name in LEXICAL_FILES)
    source_paths.append(dictionary / "SOURCE.json")
    source_paths.append(dictionary / "NNBC.csv")
    return {
        "inputs": _paths_hash(input_paths),
        "sources": _paths_hash(source_paths),
        "parent_model": _paths_hash(model_paths),
    }


def _content_key(row, input_text=None):
    return (input_text if input_text is not None else row.get("input", ""),
            row.get("context", ""), tuple(row.get("expected", ())))


def build_dev_guard():
    rows = {role: read_jsonl(path) for role, path in DEV_FILES.items()}
    all_rows = [row for values in rows.values() for row in values]
    return {
        "ids": {row.get("id") for row in all_rows if row.get("id")},
        "group_ids": {value for row in all_rows
                      for value in (row.get("group_id"), row.get("family")) if value},
        "content": {_content_key(row) for row in all_rows},
        "natural_expected": {text for row in rows["natural_dev"]
                             for text in row.get("expected", ())},
        "counts": {role: len(values) for role, values in rows.items()},
    }


def refresh_source(row, role, dev_guard):
    required = ("id", "origin", "split", "group_id", "context", "expected")
    if any(key not in row for key in required) or not isinstance(row["expected"], list) \
            or not row["expected"] or any(not isinstance(x, str) or not x
                                           for x in row["expected"]):
        raise ValueError(f"malformed {role} source row")
    expected_origin = "natural" if role == "natural_train" else "retention"
    if row["split"] != "train" or row["origin"] != expected_origin:
        raise ValueError(f"{role} source provenance differs")
    canonical = romanize_word(row["expected"][0])
    alternates = [text for text in row["expected"]
                  if romanize_word(text) != canonical]
    if alternates:
        return None, {"id": row["id"], "reason": "alternate_forward_mismatch",
                      "canonical_input": canonical, "excluded_expected": alternates}
    source = {key: row[key] for key in required}
    source["input"] = canonical
    identity_values = {source["id"], source["group_id"]}
    if identity_values & (dev_guard["ids"] | dev_guard["group_ids"]):
        return None, {"id": source["id"], "reason": "development_identity_overlap"}
    if _content_key(source, canonical) in dev_guard["content"]:
        return None, {"id": source["id"], "reason": "development_content_overlap"}
    natural_overlap = [target for target in dev_guard["natural_expected"]
                       if target and (target in source["context"]
                                      or any(target in gold
                                             for gold in source["expected"]))]
    if natural_overlap:
        return None, {"id": source["id"], "reason": "natural_dev_target_overlap",
                      "targets": sorted(natural_overlap)}
    return source, None


def capture_case(converter, recorder, source):
    recorder.records = []
    _, pieces, _, _, _ = converter._prepare_candidates(
        source["input"], source["context"], learned=True, raw=False)
    # Runtime candidates and their scores are frozen before labels are consulted.
    groups = capture_groups(pieces, recorder.records, source["context"])
    groups = add_targets(source, groups)
    case = {key: source[key] for key in
            ("id", "origin", "split", "group_id", "input", "context", "expected")}
    case["groups"] = groups
    validate_case(case, source)
    return case


def prepare_training_data(converter, output, dev_guard, update_path, guard):
    recorder = Recorder(converter.ranker)
    converter.ranker = recorder
    prepared = {}
    exclusions = {}
    output.mkdir()
    for role, source_path in TRAIN_FILES.items():
        rows = read_jsonl(source_path)
        if len(rows) != EXPECTED_ROWS[role]:
            raise ValueError(f"unexpected {role} row count")
        prepared[role], exclusions[role] = [], []
        path = output / f"{role}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for index, row in enumerate(rows):
                with guard.operation(f"prepare:{role}:{index}", update=True):
                    source, excluded = refresh_source(row, role, dev_guard)
                    if excluded is not None:
                        exclusions[role].append(excluded)
                    else:
                        case = capture_case(converter, recorder, source)
                        prepared[role].append(case)
                        stream.write(json.dumps(
                            case, ensure_ascii=False, sort_keys=True,
                            allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                append_update(update_path, "preparation_case", role=role,
                              source_index=index, id=row["id"],
                              prepared=excluded is None,
                              exclusion_reason=(None if excluded is None
                                                else excluded["reason"]))
                if (index + 1) % 64 == 0:
                    append_update(update_path, "preparation_progress", role=role,
                                  completed=index + 1, total=len(rows))
            stream.flush()
            os.fsync(stream.fileno())
        append_update(update_path, "preparation_role_complete", role=role,
                      source_rows=len(rows), prepared_rows=len(prepared[role]),
                      exclusions=len(exclusions[role]),
                      groups=sum(len(row["groups"]) for row in prepared[role]),
                      ambiguous_groups=sum(len(group["candidates"]) > 1
                                           for row in prepared[role]
                                           for group in row["groups"]),
                      covered_ambiguous_groups=sum(
                          len(group["candidates"]) > 1
                          and bool(group["positive_indices"])
                          for row in prepared[role] for group in row["groups"]),
                      sha256=file_sha256(path))
    return prepared, exclusions


def build_training_examples(data):
    examples, eligibility = {}, {}
    for role in TRAIN_FILES:
        selected = select_groups(data[role], RECIPE)
        current = [example for origin, group in selected
                   if (example := bounded_example(origin, group, RECIPE)) is not None]
        examples[role] = current
        eligibility[role] = {
            "source_rows": len(data[role]),
            "total_groups": sum(len(row["groups"]) for row in data[role]),
            "selected_before_labels": len(selected),
            "eligible_groups": len(current),
        }
    return examples, eligibility


def runtime_identity(converter):
    return {
        "version": converter.version,
        "model_artifacts": dict(sorted(converter.model_artifacts.items())),
        "external_lexicon_fingerprint": converter.generator.external_fingerprint,
        "external_lexicon_version": converter.generator.external_version,
        "dictionary_fingerprint": converter.generator.dictionary_fingerprint,
    }


def verify_runtime_identity(converter, expected):
    if runtime_identity(converter) != expected:
        return False
    return all((ROOT / relative).is_file()
               and file_sha256(ROOT / relative) == digest
               for relative, digest in expected["model_artifacts"].items())


def schedule_document(schedule):
    groups = []
    candidate_offset = 0
    for index, example in enumerate(schedule):
        end = candidate_offset + len(example.candidates)
        groups.append({
            "index": index,
            "id": example.case_id,
            "origin": example.origin,
            "candidate_start": candidate_offset,
            "candidate_end": end,
        })
        candidate_offset = end
    return {"recipe": RECIPE, "groups": groups,
            "scheduled_groups": len(groups), "scheduled_candidates": candidate_offset}


def train(args):
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    if RECIPE["epochs"] != 1 or RECIPE["source_group_limit"] != 2048:
        raise ValueError("guarded-context training requires the fixed one-epoch recipe")
    if file_sha256(PARENT / "model.safetensors") != PARENT_WEIGHT:
        raise ValueError("semantic parent differs from the fixed baseline")
    pins = frozen_pins()
    dev_guard = build_dev_guard()
    output.mkdir(parents=True)
    updates = output / "updates.jsonl"
    write_json(output / "pins.json", pins)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    random.seed(RECIPE["seed"])
    torch.manual_seed(RECIPE["seed"])
    started = time.monotonic()

    with AtomicResourceWatchdog(started, system_swap_bytes(),
                                output / "resources.jsonl", limits=LIMITS,
                                interval_s=.5) as guard:
        with guard.operation("load_current_runtime", update=True):
            converter = NeuralConverter(ranker_kind="semantic")
            converter.load_models()
            parent_ranker = converter.ranker
            runtime = runtime_identity(converter)
        write_json(output / "runtime-identity.json", runtime)
        data, exclusions = prepare_training_data(
            converter, output / "prepared", dev_guard, updates, guard)
        examples, eligibility = build_training_examples(data)
        schedule = balanced_schedule(
            examples["natural_train"], examples["retention_train"], RECIPE["seed"])
        schedule_data = schedule_document(schedule)
        write_json(output / "schedule.json", schedule_data)
        schedule_sha256 = file_sha256(output / "schedule.json")
        append_update(updates, "schedule_complete", eligibility=eligibility,
                      scheduled_groups=len(schedule), epochs=1,
                      development_rows_trained=0,
                      schedule_sha256=schedule_sha256)

        ranker = parent_ranker
        layers = _set_trainable_layers(ranker.model, RECIPE["top_layers"])
        optimizer = _optimizer(ranker.model, SimpleNamespace(**RECIPE), head_only=False)
        ranker.model.train()
        losses = []
        completed_groups = 0
        for start in range(0, len(schedule), RECIPE["group_batch"]):
            batch = schedule[start:start + RECIPE["group_batch"]]
            step = len(losses) + 1
            end = start + len(batch)
            candidate_start = schedule_data["groups"][start]["candidate_start"]
            candidate_end = schedule_data["groups"][end - 1]["candidate_end"]
            with guard.operation(f"train_batch:{step}", update=True):
                optimizer.zero_grad(set_to_none=True)
                loss = word_loss(
                    ranker.model, ranker.tokenizer, batch, ranker.max_length, RECIPE)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite guarded-context training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(ranker.model.parameters(), 1.)
                if not bool(torch.isfinite(norm)):
                    raise RuntimeError("nonfinite guarded-context gradients")
                optimizer.step()
                losses.append(float(loss.detach()))
                append_update(updates, "training_update", step=step,
                              total_steps=(len(schedule) + RECIPE["group_batch"] - 1)
                              // RECIPE["group_batch"], loss=losses[-1],
                              group_start=start, group_end=end,
                              candidate_start=candidate_start,
                              candidate_end=candidate_end,
                              groups=len(batch))
            if start != completed_groups:
                raise RuntimeError("training schedule coverage is not contiguous")
            completed_groups = end
        if completed_groups != len(schedule):
            raise RuntimeError("training schedule was not fully consumed")
        ranker.model.eval()

        with guard.operation("save_and_reload", update=True):
            bundle = output / "checkpoint"
            ranker.model.save_pretrained(bundle, safe_serialization=True)
            ranker.tokenizer.save_pretrained(bundle)
            artifacts = {path.name: file_sha256(path)
                         for path in bundle.iterdir() if path.is_file()}
            metadata = {key: ranker.metadata[key] for key in
                        ("base_model", "base_revision", "base_model_path",
                         "base_source", "license", "candidate_generator")
                        if key in ranker.metadata}
            metadata.update(
                format_version=1,
                created_at=datetime.now(timezone.utc).isoformat(),
                artifact_hashes=artifacts,
                max_length=ranker.max_length,
                inference_batch_size=ranker.batch_size,
                parent_checkpoint_sha256=PARENT_WEIGHT,
                parent_metadata=ranker.metadata,
                frozen_pins=pins,
                training=RECIPE,
                training_eligibility=eligibility,
                trainable_layers=layers,
                preparation={
                    "input_recomputed_with": "romanize_word(expected[0])",
                    "historical_groups_ignored": True,
                    "features_captured_before_labels": True,
                    "excluded_rows": exclusions,
                    "development_exclusion_counts": dev_guard["counts"],
                },
                development_rows_trained=0,
                threshold_tuning=False,
                test_data_read=False,
                library_versions={name: package_version(name) for name in
                                  ("torch", "transformers", "tokenizers", "safetensors")},
            )
            (bundle / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True,
                           indent=2, allow_nan=False) + "\n")
            loaded = SemanticRanker.load(bundle)
            contexts = ["주변 [MASK] 문맥"] * 2
            candidates = ["검증", "검정"]
            memory_scores = ranker.score_pairs(contexts, candidates)
            loaded_scores = loaded.score_pairs(contexts, candidates)
            if (len(memory_scores) != len(candidates) or len(loaded_scores) != len(candidates)
                    or not all(math.isfinite(value) for value in memory_scores + loaded_scores)):
                raise RuntimeError('saved model returned invalid verification scores')
            delta = max(abs(a - b) for a, b in zip(memory_scores, loaded_scores))
            if delta > 1e-4:
                raise RuntimeError("saved model differs from in-memory inference")
            append_update(updates, "artifact_complete", artifacts=artifacts,
                          save_load_score_max_delta=delta)

        if (frozen_pins() != pins or not verify_runtime_identity(converter, runtime)
                or file_sha256(output / "schedule.json") != schedule_sha256):
            raise RuntimeError("training inputs, sources, or parent model changed")
        report = {
            "completed": True,
            "recipe": RECIPE,
            "limits": LIMITS,
            "pins": pins,
            "runtime_identity": runtime,
            "eligibility": eligibility,
            "exclusions": exclusions,
            "scheduled_groups": len(schedule),
            "schedule_sha256": schedule_sha256,
            "optimizer_steps": len(losses),
            "mean_loss": sum(losses) / len(losses),
            "checkpoint": str(bundle),
            "checkpoint_sha256": artifacts["model.safetensors"],
            "save_load_score_max_delta": delta,
            "development_rows_trained": 0,
            "development_exclusion_counts": dev_guard["counts"],
            "test_data_read": False,
            "threshold_tuning": False,
            "deployed": False,
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(output / "report.json", report)
        (output / "trainer-source.py").write_bytes(Path(__file__).read_bytes())
        append_update(updates, "complete", optimizer_steps=len(losses),
                      checkpoint_sha256=artifacts["model.safetensors"])
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    train(parser.parse_args())
