"""Gate a current word-ranker continuation on fresh actual hybrid development."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.models.semantic_ranker import SemanticRanker
from scripts.evaluate_roberta_context_ranker import DEV_COUNTS, DEV_SUITES
from scripts.prepare_current_context_data import (
    OUTPUT as PREPARED,
    RUNTIME_VERSION,
    source_hashes as preparation_source_hashes,
    validate_data,
)
from scripts.train_current_word_ranker import (
    OUTPUT as TRAINING_OUTPUT,
    PARENT_WEIGHT,
    training_hashes,
)
from scripts.train_natural_sentence_ranker import rss_bytes


PREPARATION_MANIFEST = "b280bd12081c90fdd6bf53a3356b0a5f0ff0ccf83b905ca26b027c9d09027989"
PROFILE = {
    "name": "current-word-actual-hybrid-development-v1",
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 1800,
    "progress_interval": 32,
    "suite_counts": DEV_COUNTS,
}


class SentencePoolRecorder:
    """Observe the actual sentence candidate pool without another inference."""

    def __init__(self, delegate):
        self.delegate = delegate
        self.candidates = None
        self.scores = None

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def score_pairs(self, contexts, candidates):
        if self.candidates is not None:
            raise ValueError("unexpected repeated sentence scoring call")
        self.candidates = list(candidates)
        scores = list(self.delegate.score_pairs(contexts, candidates))
        if len(scores) != len(candidates) or any(
                isinstance(score, bool) or not math.isfinite(float(score)) for score in scores):
            raise ValueError("invalid recorded sentence scores")
        self.scores = list(map(float, scores))
        return scores


def checkpoint_hashes(checkpoint):
    return {path.name: file_sha256(path) for path in sorted(checkpoint.iterdir())
            if path.is_file()}


def evaluation_hashes():
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/evaluate_roberta_context_ranker.py",
        ROOT / "scripts/prepare_current_context_data.py",
        ROOT / "scripts/train_current_word_ranker.py",
        ROOT / "scripts/train_natural_sentence_ranker.py",
        ROOT / "hanglish/core/neural_converter.py",
        ROOT / "hanglish/models/semantic_ranker.py",
    )
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def suite_hashes():
    """Pin each external development corpus and its identity for the whole run."""
    paths = [path for _, relative, identity in DEV_SUITES
             for path in (ROOT / relative, ROOT / '.omx/artifacts' / identity)]
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def validate_training_gate(training, metadata, hashes, explicit_report_sha,
                           actual_report_sha, current_training_hashes, *, diagnostic_only=False):
    preparation = metadata.get("current_preparation", {})
    regressions = training.get("regressions")
    if not isinstance(regressions, list) or any(not isinstance(value, str) for value in regressions):
        raise ValueError("complete training regression inventory is required")
    expected_gate = (training.get("final", {}).get("correct", -1)
                     > training.get("parent", {}).get("correct", -1) and not regressions)
    if training.get("eligible_for_actual_hybrid_dev") is not expected_gate:
        raise ValueError("training gate does not match its observed group outcomes")
    if not diagnostic_only and not expected_gate:
        raise ValueError("current word training gate did not pass")
    if (explicit_report_sha != actual_report_sha
            or training.get("test_data_read") is not False
            or training.get("deployed") is not False
            or training.get("development_rows_read") != DEV_COUNTS["natural_dev"]
            or training.get("development_sampled") is not False
            or training.get("runtime_version") != RUNTIME_VERSION
            or training.get("preparation_manifest_sha256") != PREPARATION_MANIFEST
            or training.get("parent_checkpoint_sha256") != PARENT_WEIGHT
            or training.get("checkpoint_sha256") != hashes.get("model.safetensors")
            or training.get("source_hashes") != current_training_hashes
            or training.get("final", {}).get("cases") != DEV_COUNTS["natural_dev"]
            or training.get("parent", {}).get("cases") != DEV_COUNTS["natural_dev"]
            or training.get("final", {}).get("correct", -1)
               <= training.get("parent", {}).get("correct", -1)
            or metadata.get("parent_runtime_version") != RUNTIME_VERSION
            or metadata.get("parent_checkpoint_sha256") != PARENT_WEIGHT
            or metadata.get("frozen_source_hashes") != current_training_hashes
            or metadata.get("test_data_read") is not False
            or metadata.get("training") != training.get("recipe")
            or preparation.get("manifest_sha256") != PREPARATION_MANIFEST):
        raise ValueError("complete current word training gain and provenance are required")
    artifact_hashes = metadata.get("artifact_hashes")
    actual_artifacts = {name: digest for name, digest in hashes.items()
                        if name != "metadata.json"}
    if artifact_hashes != actual_artifacts:
        raise ValueError("current word checkpoint artifact hashes differ")


def evaluate_case(converter, parent_ranker, child_ranker, recorder, case):
    """Run both actual hybrid paths before consulting expected labels."""
    try:
        converter.ranker = parent_ranker
        recorder.candidates = None
        recorder.scores = None
        parent = converter.convert(case["input"], case["context"])
        parent_pool = recorder.candidates
        parent_pool_scores = recorder.scores
        converter.ranker = child_ranker
        recorder.candidates = None
        recorder.scores = None
        child = converter.convert(case["input"], case["context"])
        child_pool = recorder.candidates
        child_pool_scores = recorder.scores
    finally:
        converter.ranker = parent_ranker
    before_actual = [candidate["text"] for candidate in parent["candidates"]]
    actual = [candidate["text"] for candidate in child["candidates"]]
    if (not before_actual or not actual or len(before_actual) > 5 or len(actual) > 5
            or parent.get("sentence_ranking_applied") != (parent_pool is not None)
            or child.get("sentence_ranking_applied") != (child_pool is not None)):
        raise ValueError("actual runtime returned an invalid top-five or sentence flag")

    # Labels are used only after both source-only conversions have completed.
    expected = case["expected"]
    return {
        "id": case["id"],
        "input": case["input"],
        "context": case["context"],
        "expected": expected,
        "before_actual": before_actual,
        "actual": actual,
        "before_candidates": parent["candidates"],
        "candidates": child["candidates"],
        "before_sentence_pool": parent_pool,
        "sentence_pool": child_pool,
        "before_sentence_scores": parent_pool_scores,
        "sentence_scores": child_pool_scores,
        "before_top1": before_actual[0] in expected,
        "top1": actual[0] in expected,
        "before_top5": any(candidate in expected for candidate in before_actual),
        "top5": any(candidate in expected for candidate in actual),
        "before_sentence_pool_covered": (None if parent_pool is None else
                                          any(candidate in expected for candidate in parent_pool)),
        "sentence_pool_covered": (None if child_pool is None else
                                   any(candidate in expected for candidate in child_pool)),
        "before_sentence_pool_size": None if parent_pool is None else len(parent_pool),
        "sentence_pool_size": None if child_pool is None else len(child_pool),
        "before_sentence_ranking_applied": parent_pool is not None,
        "sentence_ranking_applied": child_pool is not None,
        "before_engine_ms": parent["engine_ms"],
        "engine_ms": child["engine_ms"],
    }


def summarize(cases):
    return {
        "rows": len(cases),
        "before_top1": sum(case["before_top1"] for case in cases),
        "top1": sum(case["top1"] for case in cases),
        "before_top5": sum(case["before_top5"] for case in cases),
        "top5": sum(case["top5"] for case in cases),
        "before_sentence_scored_rows": sum(
            case["before_sentence_pool_covered"] is not None for case in cases
        ),
        "sentence_scored_rows": sum(
            case["sentence_pool_covered"] is not None for case in cases
        ),
        "before_sentence_pool_covered": sum(
            case["before_sentence_pool_covered"] is True for case in cases
        ),
        "sentence_pool_covered": sum(
            case["sentence_pool_covered"] is True for case in cases
        ),
        "corrections": [case["id"] for case in cases
                        if case["top1"] and not case["before_top1"]],
        "regressions": [case["id"] for case in cases
                        if case["before_top1"] and not case["top1"]],
        "top5_gains": [case["id"] for case in cases
                       if case["top5"] and not case["before_top5"]],
        "top5_regressions": [case["id"] for case in cases
                             if case["before_top5"] and not case["top5"]],
        "sentence_pool_coverage_gains": [case["id"] for case in cases
                                         if case["sentence_pool_covered"] is True
                                         and case["before_sentence_pool_covered"] is not True],
        "sentence_pool_coverage_losses": [case["id"] for case in cases
                                          if case["before_sentence_pool_covered"] is True
                                          and case["sentence_pool_covered"] is not True],
    }


def development_passed(suites):
    if set(suites) != set(DEV_COUNTS):
        return False
    for name, count in DEV_COUNTS.items():
        suite = suites[name]
        if (suite.get("rows") != count or suite.get("regressions")
                or suite.get("top5_regressions")
                or suite.get("sentence_pool_coverage_losses")):
            return False
    natural = suites["natural_dev"]
    return natural["top1"] > natural["before_top1"]


def result_gate(suites, *, diagnostic_only):
    return bool(development_passed(suites) and not diagnostic_only)


def load_fixed_suites(natural_rows):
    suites = [("natural_dev", natural_rows, None)]
    sources = {"natural_dev": {"manifest_sha256": PREPARATION_MANIFEST}}
    for name, relative, identity_relative in DEV_SUITES:
        path = ROOT / relative
        identity_path = ROOT / ".omx/artifacts" / identity_relative
        identity = json.loads(identity_path.read_text())
        expected_hash = identity.get("sha256") or identity["corpus_identity"]["sha256"]
        if file_sha256(path) != expected_hash:
            raise ValueError("development suite differs from frozen identity")
        cases = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(cases) != DEV_COUNTS[name]:
            raise ValueError("development suite count differs from fixed identity")
        suites.append((name, cases, path))
        sources[name] = {
            "sha256": expected_hash,
            "identity_report_sha256": file_sha256(identity_path),
        }
    return suites, sources


def evaluate(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = args.checkpoint.resolve()
    training_path = checkpoint.parent / "report.json"
    actual_report_sha = file_sha256(training_path)
    training = json.loads(training_path.read_text())
    child_ranker = SemanticRanker.load(checkpoint)
    hashes = checkpoint_hashes(checkpoint)
    current_training_hashes = training_hashes()
    validate_training_gate(training, child_ranker.metadata, hashes,
                           args.training_report_sha256, actual_report_sha,
                           current_training_hashes, diagnostic_only=args.diagnostic_only)
    provenance = child_ranker.metadata["current_preparation"]
    if Path(provenance["path"]).resolve() != PREPARED.resolve():
        raise ValueError("word checkpoint refers to a different prepared directory")
    manifest, data = validate_data(PREPARED, PREPARATION_MANIFEST)
    if manifest.get("runtime_version") != RUNTIME_VERSION:
        raise ValueError("current preparation runtime differs")
    before = {
        "evaluation_hashes": evaluation_hashes(),
        "training_hashes": current_training_hashes,
        "preparation_source_hashes": preparation_source_hashes(),
        "checkpoint_hashes": hashes,
        "training_report_sha256": actual_report_sha,
        "preparation_manifest_sha256": file_sha256(PREPARED / "manifest.json"),
        "development_suite_hashes": suite_hashes(),
    }
    torch.set_num_threads(PROFILE["threads"])
    started = time.monotonic()

    def require_resources(stage):
        if (rss_bytes() > PROFILE["rss_limit_bytes"]
                or time.monotonic() - started > PROFILE["time_limit_seconds"]):
            raise RuntimeError(f"current word evaluation resource bound exceeded during {stage}")

    converter = NeuralConverter(ranker_kind="hybrid")
    converter.load_models()
    if converter.version != RUNTIME_VERSION:
        raise ValueError("actual hybrid parent runtime differs")
    parent_ranker = converter.ranker
    recorder = SentencePoolRecorder(converter.sentence_ranker)
    converter.sentence_ranker = recorder
    suites, sources = load_fixed_suites(data["natural_dev"])
    results = {}
    outcomes = []
    completed = 0
    for name, cases, source_path in suites:
        current = []
        for case in cases:
            row = evaluate_case(converter, parent_ranker, child_ranker, recorder, case)
            row["suite"] = name
            current.append(row)
            completed += 1
            require_resources("development inference")
            if completed % PROFILE["progress_interval"] == 0:
                print(json.dumps({"suite": name, "completed": completed,
                                  "elapsed_seconds": round(time.monotonic() - started, 2)}),
                      flush=True)
        results[name] = summarize(current)
        outcomes.extend(current)
        if source_path is not None and file_sha256(source_path) != sources[name]["sha256"]:
            raise ValueError("development suite changed during inference")
        print(json.dumps({"suite": name, **results[name]}, ensure_ascii=False), flush=True)
    if converter.ranker is not parent_ranker:
        raise RuntimeError("actual hybrid parent ranker was not restored")
    validate_data(PREPARED, PREPARATION_MANIFEST)
    after = {
        "evaluation_hashes": evaluation_hashes(),
        "training_hashes": training_hashes(),
        "preparation_source_hashes": preparation_source_hashes(),
        "checkpoint_hashes": checkpoint_hashes(checkpoint),
        "training_report_sha256": file_sha256(training_path),
        "preparation_manifest_sha256": file_sha256(PREPARED / "manifest.json"),
        "development_suite_hashes": suite_hashes(),
    }
    if before != after:
        raise ValueError("evaluation source, model, or preparation changed")
    passed = result_gate(results, diagnostic_only=args.diagnostic_only)
    report = {
        "passed": bool(passed),
        "eligible_for_further_evaluation": bool(passed),
        "scope": ("diagnostic only; failed training group gate is not superseded"
                  if args.diagnostic_only else "fresh actual hybrid development only"),
        "diagnostic_only": args.diagnostic_only,
        "training_group_gate_passed": training['eligible_for_actual_hybrid_dev'],
        "training_group_regressions": training['regressions'],
        "observed_hybrid_criteria_met": development_passed(results),
        "parent_version": RUNTIME_VERSION,
        "experimental_version": (RUNTIME_VERSION + "-word-"
                                 + hashes["model.safetensors"][:12]),
        "profile": PROFILE,
        "checkpoint": str(checkpoint),
        "checkpoint_hashes": hashes,
        "training_report_sha256": actual_report_sha,
        "preparation_manifest_sha256": PREPARATION_MANIFEST,
        "source_state": before,
        "sources": sources,
        "suites": results,
        "outcomes": outcomes,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
        "timing_policy": (
            "Parent runs first with shared generator caches; timings are descriptive, "
            "not a controlled speed comparison."
        ),
        "test_data_read": False,
        "deployed": False,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": bool(passed), "suites": results}, ensure_ascii=False),
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=TRAINING_OUTPUT / "checkpoint")
    parser.add_argument("--training-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostic-only", action="store_true",
                        help="Inspect a net-positive group model without granting final/deployment eligibility")
    evaluate(parser.parse_args())
