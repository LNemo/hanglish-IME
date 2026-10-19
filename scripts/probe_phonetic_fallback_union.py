"""Train-only actual-runtime probe for bounded phonetic fallback preservation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.core.phonology import romanize_word
from scripts.phonetic_fallback_union import preserved_options
from scripts.prepare_context_training import (
    OUTPUT as JOINT,
    RUNTIME_VERSION,
    source_hashes as context_source_hashes,
    validate_context_data,
)
from scripts.prepare_union_sentence_data import load_probe
from scripts.train_natural_sentence_ranker import rss_bytes


JOINT_MANIFEST = "7c1c03205dbc9c8983c2b3dee37e4a6bcabb04d45a265e9e1699460595275b10"
PARENT_PROBE = ROOT / ".omx/artifacts/context-proposal-union-v1"
PARENT_REPORT_SHA256 = "eb2f39be4efa0f11adf1d9d2d41117237738a285bd4973c2170004aedb558e67"
OUTPUT = ROOT / ".omx/artifacts/phonetic-fallback-union-v1"
PROFILE = {
    "name": "actual-runtime-phonetic-fallback-k4-v1",
    "natural_train_rows": 512,
    "fallback_limit": 4,
    "word_cache_size": 1024,
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 1800,
    "progress_interval": 64,
    "selection": "All fixed natural_train rows; no target-dependent selection",
}


class SentencePoolRecorder:
    """Record the pool used by the existing sentence ranker without rescoring."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.candidates = None

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def score_pairs(self, contexts, candidates):
        self.candidates = list(candidates)
        return self.delegate.score_pairs(contexts, candidates)


def source_hashes():
    hashes = context_source_hashes()
    paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/phonetic_fallback_union.py",
        ROOT / "scripts/prepare_union_sentence_data.py",
        ROOT / "scripts/probe_context_proposal_union.py",
        PARENT_PROBE / "report.json",
        PARENT_PROBE / "cases.jsonl",
        JOINT / "manifest.json",
        JOINT / "natural_train.jsonl",
        JOINT / "natural_dev.jsonl",
        JOINT / "retention_train.jsonl",
    ]
    for path in paths:
        hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    return dict(sorted(hashes.items()))


def validate_option_union(source, baseline, union, limit=4):
    """Require an exact baseline prefix plus bounded forward-valid additions."""
    if tuple(union[:len(baseline)]) != tuple(baseline):
        raise ValueError("fallback union changed baseline word options")
    additions = tuple(union[len(baseline):])
    if len(additions) > limit or len({option.text for option in union}) != len(union):
        raise ValueError("fallback union violates its candidate bound or uniqueness")
    if any(option.kind != "syllable" or romanize_word(option.text) != source
           for option in additions):
        raise ValueError("fallback union contains an invalid mechanical addition")
    return additions


def install_fallback_union(converter, limit=4):
    """Install the experiment wrapper and return its source-only audit records."""
    audits = {}

    def options(source, learned):
        baseline = converter._word_options(source, learned)
        delegate = SimpleNamespace(
            _word_options=lambda _source, _learned: baseline,
            generator=converter.generator,
        )
        union = preserved_options(delegate, source, learned, limit=limit)
        additions = validate_option_union(source, baseline, union, limit)
        audits[(source, learned)] = {
            "baseline_count": len(baseline),
            "union_count": len(union),
            "additions": [option.text for option in additions],
        }
        return union

    converter.word_options = lru_cache(maxsize=PROFILE["word_cache_size"])(options)
    return audits


def summarize(cases):
    keys = ("before_top1", "new_top1", "before_top5", "new_top5",
            "before_pool_covered", "new_pool_covered")
    metrics = {key: sum(case[key] is True for case in cases) for key in keys}
    metrics["rows"] = len(cases)
    metrics["sentence_scored_rows"] = sum(
        case["new_pool_covered"] is not None for case in cases
    )
    return metrics


def p95(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    _, joint = validate_context_data(JOINT, JOINT_MANIFEST)
    parent_report, parent_cases = load_probe(PARENT_REPORT_SHA256, joint)
    rows = joint["natural_train"]
    if (len(rows) != PROFILE["natural_train_rows"]
            or len(parent_cases) != PROFILE["natural_train_rows"]):
        raise ValueError("fixed natural training selection is incomplete")
    before = source_hashes()
    torch.set_num_threads(PROFILE["threads"])
    converter = NeuralConverter(ranker_kind="hybrid")
    converter.load_models()
    if converter.version != RUNTIME_VERSION:
        raise ValueError("runtime differs from the pinned parent")
    recorder = SentencePoolRecorder(converter.sentence_ranker)
    converter.sentence_ranker = recorder
    option_audits = install_fallback_union(converter, PROFILE["fallback_limit"])
    args.output.mkdir(parents=True)
    cases_path = args.output / "cases.jsonl"
    started = time.monotonic()
    cases = []

    with cases_path.open("x") as stream:
        for index, (row, trusted) in enumerate(zip(rows, parent_cases)):
            if any(trusted.get(key) != row[key] for key in ("id", "input", "context")):
                raise ValueError("trusted parent case differs from fixed source ordering")
            before_top5 = trusted.get("parent_top5")
            if (not isinstance(before_top5, list) or not 1 <= len(before_top5) <= 5
                    or any(not isinstance(candidate, str) for candidate in before_top5)):
                raise ValueError("trusted parent top-five candidates are incomplete")
            recorder.candidates = None
            converted = converter.convert(row["input"], row["context"])
            actual = [candidate["text"] for candidate in converted["candidates"]]
            pool = recorder.candidates
            if converted["sentence_ranking_applied"] != (pool is not None):
                raise ValueError("recorded sentence pool differs from runtime flag")

            # Candidate generation and actual ranking are complete before labels are read.
            expected = row["expected"]
            before_pool = trusted["parent_candidates"]
            before_top1 = before_top5[0] in expected
            before_covered = any(candidate in expected for candidate in before_pool)
            if (before_top1 != trusted["parent_correct"]
                    or before_covered != trusted["parent_covered"]):
                raise ValueError("trusted parent metrics differ from its candidates")
            before_scored = trusted["sentence_eligible"]
            case = {
                "id": row["id"],
                "input": row["input"],
                "context": row["context"],
                "expected": expected,
                "before_top5_candidates": before_top5,
                "new_top5_candidates": actual,
                "before_top1": before_top1,
                "new_top1": actual[0] in expected,
                "before_top5": any(candidate in expected for candidate in before_top5),
                "new_top5": any(candidate in expected for candidate in actual),
                "before_pool_size": len(before_pool) if before_scored else None,
                "new_pool_size": None if pool is None else len(pool),
                "before_pool_covered": before_covered if before_scored else None,
                "new_pool_covered": None if pool is None else any(
                    candidate in expected for candidate in pool
                ),
                "before_sentence_ranking_applied": before_scored,
                "new_flags": {
                    key: converted[key] for key in (
                        "context_used", "external_context_used", "ranking_applied",
                        "sentence_ranking_applied", "warnings",
                    )
                },
                "engine_ms": converted["engine_ms"],
            }
            cases.append(case)
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
            elapsed = time.monotonic() - started
            if rss_bytes() > PROFILE["rss_limit_bytes"] or elapsed > PROFILE["time_limit_seconds"]:
                raise RuntimeError("phonetic fallback probe resource bound exceeded")
            if (index + 1) % PROFILE["progress_interval"] == 0:
                print(json.dumps({"event": "phonetic_fallback_probe", "completed": index + 1,
                                  "elapsed_seconds": round(elapsed, 2)}), flush=True)

    validate_context_data(JOINT, JOINT_MANIFEST)
    load_probe(PARENT_REPORT_SHA256, joint)
    after = source_hashes()
    if before != after:
        raise ValueError("probe sources changed during inference")
    metrics = summarize(cases)
    corrections = [case["id"] for case in cases
                   if case["new_top1"] and not case["before_top1"]]
    regressions = [case["id"] for case in cases
                  if case["before_top1"] and not case["new_top1"]]
    top5_gains = [case["id"] for case in cases
                 if case["new_top5"] and not case["before_top5"]]
    changed = [audit for audit in option_audits.values() if audit["additions"]]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": PROFILE,
        "parent_version": RUNTIME_VERSION,
        "joint_manifest_sha256": JOINT_MANIFEST,
        "parent_probe_report_sha256": PARENT_REPORT_SHA256,
        "parent_probe_cases_sha256": parent_report["cases_sha256"],
        "source_hashes": before,
        "metrics": metrics,
        "corrections": corrections,
        "regressions": regressions,
        "top5_gains": top5_gains,
        "word_option_audit": {
            "unique_calls": len(option_audits),
            "changed_calls": len(changed),
            "added_options": sum(len(audit["additions"]) for audit in changed),
            "max_additions": max((len(audit["additions"]) for audit in changed), default=0),
            "max_total_options": max(
                (audit["union_count"] for audit in option_audits.values()), default=0
            ),
            "all_baseline_options_preserved": True,
            "all_additions_forward_valid": True,
        },
        "median_sentence_pool_size": statistics.median(
            case["new_pool_size"] for case in cases if case["new_pool_size"] is not None
        ),
        "latency_ms": {
            "median": statistics.median(case["engine_ms"] for case in cases),
            "p95": p95([case["engine_ms"] for case in cases]),
        },
        "timing_policy": (
            "Descriptive changed-path latency only; the trusted parent capture has no "
            "comparable runtime timing, so no speedup claim is made."
        ),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
        "development_inference_run": False,
        "test_data_read": False,
        "deployed": False,
        "cases_sha256": file_sha256(cases_path),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: report[key] for key in
                      ("metrics", "corrections", "regressions", "top5_gains",
                       "word_option_audit", "elapsed_seconds", "peak_rss_bytes")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
