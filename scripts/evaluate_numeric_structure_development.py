"""Development-only paired evaluation of the numeric sentence-pool overlay.

The production converter and the temporary child share the same frozen models.
Only the child conversion runs under ``numeric_beam_overlay``. Development
labels are consulted after both source-only conversions and parent parity with
the accepted acronym experiment have been established.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

import hanglish.core.neural_converter as neural_module
from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.core.phonology import romanize_word
from scripts.evaluate_current_word_ranker import DEV_COUNTS, development_passed, summarize
from scripts.evaluate_roberta_context_ranker import DEV_SUITES
from scripts.numeric_structure_candidates import NumericSpanGenerator
from scripts.probe_numeric_structure_runtime import (
    ORIGINAL_BEAM_PATHS,
    PROFILE as TRAIN_PROFILE,
    SentencePoolRecorder,
    metrics as train_metrics,
    numeric_beam_overlay,
    pool_prefix_proof,
    validate_candidate_objects,
)
from scripts.train_natural_sentence_ranker import rss_bytes


OUTPUT = ROOT / ".omx/artifacts/numeric-structure-development-v1"
TRAIN_OUTPUT = ROOT / ".omx/artifacts/numeric-structure-runtime-v1"
ACRONYM_DEVELOPMENT = ROOT / ".omx/artifacts/acronym-suffix-development-v1"
INTEGRATION = ROOT / ".omx/artifacts/acronym-suffix-integration-v1/report.json"
NATURAL_DEV = ROOT / "data/natural_context/current-context-v1/natural_dev.jsonl"
NATURAL_IDENTITY = ROOT / "data/natural_context/current-context-v1/manifest.json"

TRAIN_REPORT_SHA256 = "ecbe82f14f96446b5ae8e660244393ce9c3293cfd4d6cb1db388fbe9ae58fe88"
TRAIN_CASES_SHA256 = "50550395a81a82635b4f0cf0ff5d1840e28f221d86426716cd6ff70e0bc50557"
TRAIN_SOURCE_SHA256 = "105b628043c2f8bff839135eee9b4b7e68323c5ee6f59e38dad0bfad3b93aa20"
ACRONYM_REPORT_SHA256 = "2b485a219af24e918052563198138abc6b293773063407f7609075500da21954"
ACRONYM_CASES_SHA256 = "053390be56f8a47408401f1b1dc7dfa693515b6a80fc2622a822d7d31961a343"
INTEGRATION_SHA256 = "3148b4eb9a156457ea95a9b2afec1072d3db03273baacbb5159524d0ea6cd134"
RUNTIME_VERSION = "neural-1-5bcbab126cae"
PROFILE = {
    "name": "numeric-structure-actual-hybrid-development-v1",
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 1800,
    "progress_interval": 32,
    "suite_counts": DEV_COUNTS,
    "score_tolerance": 1e-4,
    "maximum_original_pool": TRAIN_PROFILE["maximum_original_pool"],
    "maximum_additions": TRAIN_PROFILE["maximum_additions"],
    "maximum_augmented_pool": TRAIN_PROFILE["maximum_augmented_pool"],
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def parse_jsonl(value):
    return [json.loads(line) for line in value.decode("utf-8").splitlines() if line]


def read_pinned(path, expected, *, jsonl=False):
    """Hash and parse one exact byte snapshot, avoiding a hash/read race."""
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected:
        raise ValueError(f"pinned artifact hash differs: {path}")
    return raw, (parse_jsonl(raw) if jsonl else json.loads(raw))


def candidate_objects_equal(left, right, tolerance=PROFILE["score_tolerance"]):
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return False
    for actual, expected in zip(left, right):
        if actual.get("text") != expected.get("text") or actual.get("segments") != expected.get("segments"):
            return False
        actual_score, expected_score = actual.get("score"), expected.get("score")
        if (actual_score is None) != (expected_score is None):
            return False
        if actual_score is not None and (
                type(actual_score) not in (int, float)
                or type(expected_score) not in (int, float)
                or not math.isfinite(actual_score) or not math.isfinite(expected_score)
                or abs(actual_score - expected_score) > tolerance):
            return False
    return True


def score_arrays_equal(left, right, tolerance=PROFILE["score_tolerance"]):
    if left is None or right is None:
        return left is right
    return (len(left) == len(right)
            and all(type(a) in (int, float) and type(b) in (int, float)
                    and math.isfinite(a) and math.isfinite(b)
                    and abs(a - b) <= tolerance for a, b in zip(left, right)))


def validate_segments(response, source):
    validate_candidate_objects(response, source)
    for candidate in response["candidates"]:
        for segment in candidate["segments"]:
            if segment.get("kind") == "raw" and segment.get("text") != segment.get("source"):
                raise ValueError("actual candidate changed a raw segment")
            if (segment.get("kind") == "numeric"
                    and romanize_word(segment.get("text", "")) != segment.get("source")):
                raise ValueError("actual numeric segment is not forward valid")


def validate_train_gate(report, cases, explicit_sha, actual_sha):
    metrics = report.get("metrics", {})
    adapter = report.get("adapter", {})
    if (explicit_sha != TRAIN_REPORT_SHA256 or actual_sha != explicit_sha
            or report.get("passed") is not True
            or report.get("eligible_for_development") is not True
            or report.get("scope") != "fresh actual hybrid natural TRAIN first512 numeric overlay probe"
            or report.get("runtime_version") != RUNTIME_VERSION
            or report.get("probe_source_sha256") != TRAIN_SOURCE_SHA256
            or report.get("cases_sha256") != TRAIN_CASES_SHA256
            or report.get("profile") != TRAIN_PROFILE
            or report.get("source_hashes") != report.get("source_hashes_after")
            or report.get("labels_used_for_candidate_production") is not False
            or report.get("development_data_read") is not False
            or report.get("test_data_read") is not False
            or report.get("new_weights_created") is not False
            or report.get("deployed") is not False
            or metrics.get("regressions") != []
            or metrics.get("top5_regressions") != []
            or metrics.get("pool_regressions") != []
            or metrics.get("child_top1", -1) <= metrics.get("parent_top1", -1)
            or metrics.get("child_top5", -1) < metrics.get("parent_top5", -1)
            or metrics.get("child_pool_or_fallback_top5_covered", -1)
               < metrics.get("parent_pool_or_fallback_top5_covered", -1)
            or adapter.get("baseline_prefix_and_scores_preserved") is not True
            or adapter.get("only_first_all_converted_assignment") is not True
            or adapter.get("word_options_identity_preserved") is not True
            or adapter.get("hook_restored_after_every_child") is not True):
        raise ValueError("complete regression-free numeric TRAIN gate is required")
    if (len(cases) != TRAIN_PROFILE["rows"]
            or len({case.get("id") for case in cases}) != len(cases)
            or train_metrics(cases) != metrics):
        raise ValueError("numeric TRAIN cases do not reproduce the accepted report")


def validate_acronym_reference(report, cases):
    if (report.get("passed") is not True
            or report.get("eligible_for_further_evaluation") is not True
            or report.get("scope") != "fresh actual hybrid development only"
            or report.get("cases_sha256") != ACRONYM_CASES_SHA256
            or report.get("test_data_read") is not False
            or report.get("deployed") is not False
            or len(cases) != sum(DEV_COUNTS.values())
            or report.get("outcomes") != cases):
        raise ValueError("accepted acronym development reference is incomplete")
    grouped = {name: [] for name in DEV_COUNTS}
    for case in cases:
        if case.get("suite") not in grouped:
            raise ValueError("accepted acronym case has an unknown suite")
        grouped[case["suite"]].append(case)
    if any(len(grouped[name]) != count for name, count in DEV_COUNTS.items()):
        raise ValueError("accepted acronym suite counts differ")
    if {name: summarize(values) for name, values in grouped.items()} != report.get("suites"):
        raise ValueError("accepted acronym suite metrics differ from its cases")
    return grouped


def validate_lineage(train_report, integration):
    source_hashes = train_report.get("source_hashes", {})
    weights = integration.get("accepted_weight_hashes", {})
    runtime_sources = integration.get("integrated_runtime_source_hashes", {})
    if (file_sha256(INTEGRATION) != INTEGRATION_SHA256
            or integration.get("passed") is not True
            or integration.get("integrated_runtime_version") != RUNTIME_VERSION
            or len(weights) != 16
            or any(source_hashes.get(path) != digest for path, digest in weights.items())
            or any(source_hashes.get(path) != digest for path, digest in runtime_sources.items())):
        raise ValueError("numeric TRAIN runtime or its sixteen model artifacts differ")
    for path, digest in source_hashes.items():
        if file_sha256(ROOT / path) != digest:
            raise ValueError(f"accepted numeric TRAIN source changed: {path}")
    return source_hashes


def read_development_sources(acronym_report, acronym_groups):
    """Read each development corpus and identity once, parsing those bytes."""
    paths = {"natural_dev": (NATURAL_DEV, NATURAL_IDENTITY)}
    paths.update({name: (ROOT / relative, ROOT / ".omx/artifacts" / identity)
                  for name, relative, identity in DEV_SUITES})
    state = acronym_report["source_state"]
    source_hashes = dict(state["source_hashes"])
    for relative, digest in state["development_suite_hashes"].items():
        if relative in source_hashes and source_hashes[relative] != digest:
            raise ValueError("accepted development hash maps conflict")
        source_hashes[relative] = digest
    suites, snapshots = {}, {}
    for name, (corpus_path, identity_path) in paths.items():
        corpus_raw = corpus_path.read_bytes()
        identity_raw = identity_path.read_bytes()
        corpus_relative = str(corpus_path.relative_to(ROOT))
        identity_relative = str(identity_path.relative_to(ROOT))
        expected_corpus = source_hashes.get(corpus_relative)
        expected_identity = source_hashes.get(identity_relative)
        if (sha256_bytes(corpus_raw) != expected_corpus
                or sha256_bytes(identity_raw) != expected_identity):
            raise ValueError("development corpus or identity differs from accepted acronym evaluation")
        rows = parse_jsonl(corpus_raw)
        if len(rows) != DEV_COUNTS[name] or len(acronym_groups[name]) != len(rows):
            raise ValueError("development suite row count differs")
        for row, reference in zip(rows, acronym_groups[name]):
            if any(row.get(key) != reference.get(key) for key in ("id", "input", "context")):
                raise ValueError("development source identity or order differs")
        suites[name] = rows
        snapshots[corpus_relative] = expected_corpus
        snapshots[identity_relative] = expected_identity
    return suites, snapshots


def validate_parent_parity(parent, pool, scores, reference):
    if (not candidate_objects_equal(parent.get("candidates"), reference.get("candidates"))
            or pool != reference.get("sentence_pool")
            or not score_arrays_equal(scores, reference.get("sentence_scores"))
            or parent.get("sentence_ranking_applied") != (pool is not None)):
        raise ValueError("fresh production parent differs from accepted acronym development output")


def response_flags(response):
    return {key: response.get(key) for key in (
        "version", "ranker", "context_used", "ranking_applied",
        "sentence_ranking_applied", "warnings",
    )}


def evaluate_case(converter, recorder, generator, source, reference,
                  overlay=numeric_beam_overlay):
    """Run and validate both source-only paths before reading labels."""
    original_options = converter.word_options
    recorder.reset()
    parent = converter.convert(source["input"], source["context"])
    validate_segments(parent, source["input"])
    parent_pool, parent_scores = recorder.candidates, recorder.scores
    validate_parent_parity(parent, parent_pool, parent_scores, reference)
    recorder.reset()
    try:
        with overlay(converter, generator) as adapter:
            child = converter.convert(source["input"], source["context"])
    finally:
        if converter.word_options is not original_options:
            raise ValueError("numeric development adapter changed word options")
    validate_segments(child, source["input"])
    child_pool, child_scores = recorder.candidates, recorder.scores
    if neural_module.beam_paths is not ORIGINAL_BEAM_PATHS:
        raise ValueError("numeric development adapter did not restore beam_paths")
    prefix = pool_prefix_proof(parent_pool, parent_scores, child_pool, child_scores)
    if not prefix:
        raise ValueError("numeric overlay changed the original pool or score prefix")
    if adapter.get("applied"):
        if (adapter.get("calls") != 2
                or parent_pool != [path[0] for path in adapter.get("original_pool", [])]
                or child_pool != [path[0] for path in adapter.get("augmented_pool", [])]
                or len(adapter.get("extras", [])) > PROFILE["maximum_additions"]
                or not all(converter.sentence_ranker.can_score(text) for text in parent_pool)
                or not all(converter.sentence_ranker.can_score(path[0])
                           for path in adapter.get("extras", []))):
            raise ValueError("applied numeric overlay does not match the scored runtime pool")
    elif (not candidate_objects_equal(parent["candidates"], child["candidates"])
          or parent_pool != child_pool or not score_arrays_equal(parent_scores, child_scores)
          or parent.get("sentence_ranking_applied") != child.get("sentence_ranking_applied")):
        raise ValueError("non-applied numeric adapter changed fallback behavior")

    # Label access begins only after paired inference, lineage, and pool checks.
    expected = source["expected"]
    if expected != reference.get("expected"):
        raise ValueError("development labels differ from accepted acronym reference")
    parent_texts = [item["text"] for item in parent["candidates"]]
    child_texts = [item["text"] for item in child["candidates"]]
    pool_additions = [] if child_pool is None else child_pool[len(parent_pool):]
    return {
        "id": source["id"], "input": source["input"], "context": source["context"],
        "expected": expected, "before_candidates": parent["candidates"],
        "candidates": child["candidates"], "before_actual": parent_texts,
        "actual": child_texts, "before_sentence_pool": parent_pool,
        "sentence_pool": child_pool, "before_sentence_scores": parent_scores,
        "sentence_scores": child_scores,
        "before_sentence_pool_size": None if parent_pool is None else len(parent_pool),
        "sentence_pool_size": None if child_pool is None else len(child_pool),
        "before_sentence_ranking_applied": parent_pool is not None,
        "sentence_ranking_applied": child_pool is not None,
        "before_flags": response_flags(parent), "flags": response_flags(child),
        "before_engine_ms": parent["engine_ms"], "engine_ms": child["engine_ms"],
        "adapter": adapter, "overlay_applied": bool(adapter.get("applied")),
        "pool_additions": pool_additions,
        "visible_candidates_changed": not candidate_objects_equal(
            parent["candidates"], child["candidates"]
        ),
        "before_top1": parent_texts[0] in expected,
        "top1": child_texts[0] in expected,
        "before_top5": any(text in expected for text in parent_texts),
        "top5": any(text in expected for text in child_texts),
        "before_sentence_pool_covered": (None if parent_pool is None else
                                          any(text in expected for text in parent_pool)),
        "sentence_pool_covered": (None if child_pool is None else
                                   any(text in expected for text in child_pool)),
        "baseline_pool_and_score_prefix_preserved": prefix,
    }


def development_gate(suites):
    return bool(development_passed(suites)
                and suites["natural_dev"]["top1"]
                >= suites["natural_dev"]["before_top1"] + 1)


def require_resources(started, stage):
    if (rss_bytes() > PROFILE["rss_limit_bytes"]
            or time.monotonic() - started > PROFILE["time_limit_seconds"]):
        raise RuntimeError(f"numeric development resource bound exceeded during {stage}")


def run(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = Path(__file__).read_bytes()
    source_sha = sha256_bytes(source)
    train_report_raw, train_report = read_pinned(
        TRAIN_OUTPUT / "report.json", TRAIN_REPORT_SHA256
    )
    _, train_cases = read_pinned(TRAIN_OUTPUT / "cases.jsonl", TRAIN_CASES_SHA256, jsonl=True)
    train_source_raw = (TRAIN_OUTPUT / "source.py").read_bytes()
    if sha256_bytes(train_source_raw) != TRAIN_SOURCE_SHA256:
        raise ValueError("accepted numeric TRAIN adapter source differs")
    validate_train_gate(train_report, train_cases, args.training_report_sha256,
                        sha256_bytes(train_report_raw))
    integration_raw, integration = read_pinned(INTEGRATION, INTEGRATION_SHA256)
    accepted_sources = validate_lineage(train_report, integration)
    _, acronym_report = read_pinned(
        ACRONYM_DEVELOPMENT / "report.json", ACRONYM_REPORT_SHA256
    )
    _, acronym_cases = read_pinned(
        ACRONYM_DEVELOPMENT / "cases.jsonl", ACRONYM_CASES_SHA256, jsonl=True
    )
    acronym_groups = validate_acronym_reference(acronym_report, acronym_cases)
    suites, development_hashes = read_development_sources(acronym_report, acronym_groups)

    fixed_hashes = dict(accepted_sources)
    fixed_hashes.update(development_hashes)
    fixed_hashes.update({
        ".omx/artifacts/numeric-structure-runtime-v1/report.json": TRAIN_REPORT_SHA256,
        ".omx/artifacts/numeric-structure-runtime-v1/cases.jsonl": TRAIN_CASES_SHA256,
        ".omx/artifacts/numeric-structure-runtime-v1/source.py": TRAIN_SOURCE_SHA256,
        ".omx/artifacts/acronym-suffix-development-v1/report.json": ACRONYM_REPORT_SHA256,
        ".omx/artifacts/acronym-suffix-development-v1/cases.jsonl": ACRONYM_CASES_SHA256,
        ".omx/artifacts/acronym-suffix-integration-v1/report.json": sha256_bytes(integration_raw),
        "scripts/evaluate_numeric_structure_development.py": source_sha,
    })
    for relative, expected in fixed_hashes.items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"numeric development input changed before inference: {relative}")

    torch.set_num_threads(PROFILE["threads"])
    os.environ["OMP_NUM_THREADS"] = str(PROFILE["threads"])
    converter = NeuralConverter(ranker_kind="hybrid")
    converter.load_models()
    if converter.version != RUNTIME_VERSION:
        raise ValueError("production runtime version differs")
    original_options = converter.word_options
    recorder = SentencePoolRecorder(converter.sentence_ranker)
    converter.sentence_ranker = recorder
    generator = NumericSpanGenerator(converter.numeric_units)
    results, all_rows = {}, []
    completed = 0
    for name in DEV_COUNTS:
        rows = []
        for source_row, reference in zip(suites[name], acronym_groups[name]):
            row = evaluate_case(converter, recorder, generator, source_row, reference)
            row["suite"] = name
            rows.append(row)
            all_rows.append(row)
            completed += 1
            require_resources(started, "paired inference")
            if completed % PROFILE["progress_interval"] == 0:
                print(json.dumps({"event": "numeric_development", "rows": completed,
                                  "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
        results[name] = summarize(rows)

    if converter.word_options is not original_options or neural_module.beam_paths is not ORIGINAL_BEAM_PATHS:
        raise ValueError("numeric development left a runtime hook installed")
    after = {relative: file_sha256(ROOT / relative) for relative in fixed_hashes}
    if after != fixed_hashes:
        raise ValueError("numeric development source, model, corpus, or artifact changed")
    require_resources(started, "post-validation")
    passed = development_gate(results)
    args.output.mkdir(parents=True)
    cases_path = args.output / "cases.jsonl"
    with cases_path.open("x") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    source_path = args.output / "source.py"
    source_path.write_bytes(source)
    if (file_sha256(Path(__file__)) != source_sha
            or file_sha256(source_path) != source_sha):
        raise ValueError("numeric development evaluator source changed")
    require_resources(started, "final report")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed, "eligible_for_further_evaluation": passed,
        "scope": "fresh actual hybrid development only for numeric sentence-pool overlay",
        "profile": PROFILE, "runtime_version": RUNTIME_VERSION,
        "training_report_sha256": args.training_report_sha256,
        "training_cases_sha256": TRAIN_CASES_SHA256,
        "training_adapter_source_sha256": TRAIN_SOURCE_SHA256,
        "accepted_acronym_development_report_sha256": ACRONYM_REPORT_SHA256,
        "accepted_acronym_development_cases_sha256": ACRONYM_CASES_SHA256,
        "source_hashes": fixed_hashes, "source_hashes_after": after,
        "model_artifact_hash_count": len(integration["accepted_weight_hashes"]),
        "suites": results,
        "overlay": {
            "applied_rows": sum(row["overlay_applied"] for row in all_rows),
            "visible_changed_rows": sum(row["visible_candidates_changed"] for row in all_rows),
            "pool_additions": sum(len(row["pool_additions"]) for row in all_rows),
            "baseline_pool_and_score_prefix_preserved": all(
                row["baseline_pool_and_score_prefix_preserved"] for row in all_rows
            ),
            "word_options_identity_preserved": True,
            "beam_hook_restored": True,
        },
        "cases_sha256": file_sha256(cases_path),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
        "labels_used_for_candidate_production": False,
        "final_data_read": False, "new_weights_created": False, "deployed": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"passed": passed, "suites": results,
                      "overlay": report["overlay"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-report-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
