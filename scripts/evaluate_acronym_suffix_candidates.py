"""Fresh actual-hybrid development gate for acronym suffix candidates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hanglish.core.neural_converter import NeuralConverter, file_sha256
from scripts.evaluate_current_word_ranker import (
    DEV_COUNTS,
    SentencePoolRecorder,
    development_passed,
    load_fixed_suites,
    suite_hashes,
    summarize,
)
from scripts.evaluate_roberta_context_ranker import DEV_SUITES
from scripts.prepare_current_context_data import (
    OUTPUT as PREPARED,
    RUNTIME_VERSION,
    source_hashes as preparation_source_hashes,
    validate_data,
)
from scripts.probe_acronym_suffix_candidates import (
    OUTPUT as TRAIN_PROBE,
    POLICY as TRAIN_POLICY,
    install,
    metrics as train_metrics,
    source_hashes as train_source_hashes,
    validate_audits,
)
from scripts.train_natural_sentence_ranker import rss_bytes


TRAIN_PROBE_SHA256 = "7869047ba2f7cbaf54756d8c30067554c0bfb46d8459590e955cc28e4da740a1"
PREPARATION_MANIFEST = "b280bd12081c90fdd6bf53a3356b0a5f0ff0ccf83b905ca26b027c9d09027989"
OUTPUT = ROOT / ".omx/artifacts/acronym-suffix-development-v1"
PROFILE = {
    "name": "acronym-suffix-actual-hybrid-development-v1",
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 1800,
    "progress_interval": 32,
    "suite_counts": DEV_COUNTS,
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def validate_suite_hashes(hashes):
    required = {str(path.relative_to(ROOT)) for _, relative, identity in
                DEV_SUITES
                for path in (ROOT / relative, ROOT / ".omx/artifacts" / identity)}
    if set(hashes) != required or any(
            not isinstance(value, str) or len(value) != 64 for value in hashes.values()):
        raise ValueError("complete development corpus and identity hashes are required")


def policy_fingerprint():
    payload = json.dumps(TRAIN_POLICY, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def require_resources(started, stage):
    if (rss_bytes() > PROFILE["rss_limit_bytes"]
            or time.monotonic() - started > PROFILE["time_limit_seconds"]):
        raise RuntimeError(f"acronym suffix development resource bound exceeded during {stage}")


def finalize_development_gate(results, before, after, started):
    """Refuse eligibility when provenance or whole-run resources fail."""
    if before != after:
        raise ValueError("acronym evaluation source, model, probe, or corpus changed")
    require_resources(started, "post-validation")
    return development_passed(results)


def validate_train_gate(report, explicit_sha, actual_sha, cases, selection,
                        option_rows, fixed):
    metrics = report.get("metrics", {})
    audit = report.get("candidate_audit", {})
    if (explicit_sha != TRAIN_PROBE_SHA256 or actual_sha != explicit_sha
            or report.get("scope") != "Actual child conversion on fixed natural TRAIN first512 only"
            or report.get("policy") != TRAIN_POLICY
            or report.get("parent_version") != RUNTIME_VERSION
            or report.get("preparation_manifest_sha256") != PREPARATION_MANIFEST
            or report.get("source_hashes") != train_source_hashes()
            or report.get("natural_dev_inference_run") is not False
            or report.get("development_inference_run") is not False
            or report.get("test_data_read") is not False
            or report.get("deployed") is not False
            or file_sha256(TRAIN_PROBE / "selection.json") != report.get("selection_sha256")
            or file_sha256(TRAIN_PROBE / "cases.jsonl") != report.get("cases_sha256")
            or file_sha256(TRAIN_PROBE / "option-audit.jsonl") != report.get("option_audit_sha256")
            or metrics.get("rows") != TRAIN_POLICY["rows"]
            or metrics.get("new_top1", -1) <= metrics.get("before_top1", -1)
            or metrics.get("regressions") != []
            or metrics.get("top5_regressions") != []
            or metrics.get("pool_coverage_losses") != []
            or metrics.get("new_top5", -1) < metrics.get("before_top5", -1)
            or metrics.get("new_pool_covered", -1) < metrics.get("before_pool_covered", -1)
            or audit.get("baseline_prefix_preserved") is not True
            or audit.get("scores_unmodified_from_suffix_options") is not True
            or audit.get("all_additions_forward_valid") is not True
            or type(audit.get("max_added")) is not int
            or not 0 <= audit["max_added"] <= TRAIN_POLICY["additional_candidates"]):
        raise ValueError("complete regression-free acronym TRAIN gain is required")
    if train_metrics(cases) != metrics:
        raise ValueError("acronym TRAIN case metrics differ from report")
    if not (len(cases) == len(selection) == len(fixed) == TRAIN_POLICY["rows"]):
        raise ValueError("acronym TRAIN source selection is incomplete")
    keys = ("id", "input", "context", "expected")
    for case, selected, source in zip(cases, selection, fixed):
        if (any(case.get(key) != source.get(key) for key in keys)
                or any(selected.get(key) != source.get(key) for key in keys)):
            raise ValueError("acronym TRAIN source identity or order differs")
    keyed = {(row.get("source"), row.get("learned")): row for row in option_rows}
    if len(keyed) != len(option_rows):
        raise ValueError("acronym TRAIN option audit has duplicate keys")
    validate_audits(keyed)


def evaluate_case(converter, parent_options, child_options, recorder, case):
    """Run both source-only option policies before reading expected labels."""
    try:
        converter.word_options = parent_options
        recorder.candidates = recorder.scores = None
        parent = converter.convert(case["input"], case["context"])
        parent_pool, parent_scores = recorder.candidates, recorder.scores
        converter.word_options = child_options
        recorder.candidates = recorder.scores = None
        child = converter.convert(case["input"], case["context"])
        child_pool, child_scores = recorder.candidates, recorder.scores
    finally:
        converter.word_options = parent_options
    parent_texts = [candidate["text"] for candidate in parent["candidates"]]
    child_texts = [candidate["text"] for candidate in child["candidates"]]
    if (not parent_texts or not child_texts or len(parent_texts) > 5 or len(child_texts) > 5
            or parent.get("sentence_ranking_applied") != (parent_pool is not None)
            or child.get("sentence_ranking_applied") != (child_pool is not None)
            or (parent_pool is None) != (parent_scores is None)
            or (child_pool is None) != (child_scores is None)
            or (parent_pool is not None and len(parent_pool) != len(parent_scores))
            or (child_pool is not None and len(child_pool) != len(child_scores))
            or any(not math.isfinite(score) for scores in (parent_scores, child_scores)
                   if scores is not None for score in scores)):
        raise ValueError("actual acronym evaluation returned malformed candidates or scores")
    expected = case["expected"]
    return {
        "id": case["id"], "input": case["input"], "context": case["context"],
        "expected": expected,
        "before_candidates": parent["candidates"], "candidates": child["candidates"],
        "before_actual": parent_texts, "actual": child_texts,
        "before_sentence_pool": parent_pool, "sentence_pool": child_pool,
        "before_sentence_scores": parent_scores, "sentence_scores": child_scores,
        "before_top1": parent_texts[0] in expected, "top1": child_texts[0] in expected,
        "before_top5": any(text in expected for text in parent_texts),
        "top5": any(text in expected for text in child_texts),
        "before_sentence_pool_covered": (None if parent_pool is None else
                                          any(text in expected for text in parent_pool)),
        "sentence_pool_covered": (None if child_pool is None else
                                   any(text in expected for text in child_pool)),
        "before_sentence_pool_size": None if parent_pool is None else len(parent_pool),
        "sentence_pool_size": None if child_pool is None else len(child_pool),
        "before_engine_ms": parent["engine_ms"], "engine_ms": child["engine_ms"],
    }


def source_hashes():
    hashes = train_source_hashes()
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/evaluate_current_word_ranker.py",
        ROOT / "scripts/evaluate_roberta_context_ranker.py",
        ROOT / "scripts/probe_acronym_suffix_candidates.py",
        ROOT / "scripts/prepare_current_context_data.py",
        ROOT / "scripts/train_natural_sentence_ranker.py",
        TRAIN_PROBE / "report.json", TRAIN_PROBE / "selection.json",
        TRAIN_PROBE / "cases.jsonl", TRAIN_PROBE / "option-audit.jsonl",
    )
    for path in paths:
        hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    return dict(sorted(hashes.items()))


def run(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, data = validate_data(PREPARED, PREPARATION_MANIFEST)
    if manifest.get("runtime_version") != RUNTIME_VERSION:
        raise ValueError("current preparation runtime differs")
    probe_path = TRAIN_PROBE / "report.json"
    report = json.loads(probe_path.read_text())
    train_cases = read_jsonl(TRAIN_PROBE / "cases.jsonl")
    selection = json.loads((TRAIN_PROBE / "selection.json").read_text())
    option_rows = read_jsonl(TRAIN_PROBE / "option-audit.jsonl")
    validate_train_gate(report, args.training_probe_sha256, file_sha256(probe_path),
                        train_cases, selection, option_rows,
                        data["natural_train"][:TRAIN_POLICY["rows"]])
    development_hashes = suite_hashes()
    validate_suite_hashes(development_hashes)
    before = {"source_hashes": source_hashes(),
              "preparation_source_hashes": preparation_source_hashes(),
              "development_suite_hashes": development_hashes}
    torch.set_num_threads(PROFILE["threads"])
    converter = NeuralConverter(ranker_kind="hybrid")
    converter.load_models()
    if converter.version != RUNTIME_VERSION:
        raise ValueError("actual hybrid parent runtime differs")
    parent_options = converter.word_options
    child_audit = install(converter)
    child_options = converter.word_options
    recorder = SentencePoolRecorder(converter.sentence_ranker)
    converter.sentence_ranker = recorder
    suites, sources = load_fixed_suites(data["natural_dev"])
    args.output.mkdir(parents=True)
    cases_path = args.output / "cases.jsonl"
    outcomes = []
    results = {}

    with cases_path.open("x") as stream:
        completed = 0
        for name, cases, source_path in suites:
            current = []
            for case in cases:
                row = evaluate_case(converter, parent_options, child_options, recorder, case)
                row["suite"] = name
                current.append(row)
                outcomes.append(row)
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                completed += 1
                require_resources(started, "development inference")
                if completed % PROFILE["progress_interval"] == 0:
                    print(json.dumps({"suite": name, "completed": completed,
                                      "elapsed_seconds": round(time.monotonic() - started, 2)}),
                          flush=True)
            results[name] = summarize(current)
            if source_path is not None and file_sha256(source_path) != sources[name]["sha256"]:
                raise ValueError("development corpus changed during inference")
            print(json.dumps({"suite": name, **results[name]}, ensure_ascii=False), flush=True)
    if converter.word_options is not parent_options:
        raise RuntimeError("prototype did not restore the parent word options")
    validate_audits(child_audit)
    child_audit_path = args.output / "option-audit.jsonl"
    with child_audit_path.open("x") as stream:
        for key in sorted(child_audit):
            stream.write(json.dumps(child_audit[key], ensure_ascii=False, sort_keys=True) + "\n")
    validate_data(PREPARED, PREPARATION_MANIFEST)
    validate_train_gate(json.loads(probe_path.read_text()), args.training_probe_sha256,
                        file_sha256(probe_path), read_jsonl(TRAIN_PROBE / "cases.jsonl"),
                        json.loads((TRAIN_PROBE / "selection.json").read_text()),
                        read_jsonl(TRAIN_PROBE / "option-audit.jsonl"),
                        data["natural_train"][:TRAIN_POLICY["rows"]])
    after = {"source_hashes": source_hashes(),
             "preparation_source_hashes": preparation_source_hashes(),
             "development_suite_hashes": suite_hashes()}
    validate_suite_hashes(after["development_suite_hashes"])
    passed = finalize_development_gate(results, before, after, started)
    changed = [row for row in child_audit.values() if row["additions"]]
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": bool(passed), "eligible_for_further_evaluation": bool(passed),
        "scope": "fresh actual hybrid development only",
        "parent_version": RUNTIME_VERSION,
        "experimental_version": (RUNTIME_VERSION + "-acronym-suffix-"
                                 + policy_fingerprint()[:12]),
        "experimental_policy_sha256": policy_fingerprint(),
        "profile": PROFILE, "training_probe_sha256": args.training_probe_sha256,
        "preparation_manifest_sha256": PREPARATION_MANIFEST,
        "source_state": before, "sources": sources, "suites": results,
        "outcomes": outcomes,
        "child_option_audit": {
            "sha256": file_sha256(child_audit_path), "unique_calls": len(child_audit),
            "changed_calls": len(changed),
            "added_candidates": sum(len(row["additions"]) for row in changed),
            "max_added": max((len(row["additions"]) for row in changed), default=0),
            "baseline_prefix_preserved": True,
            "scores_unmodified_from_suffix_options": True,
            "all_additions_forward_valid": True,
        },
        "cases_sha256": file_sha256(cases_path),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
        "timing_policy": "Parent runs first with shared caches; descriptive only.",
        "test_data_read": False, "deployed": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"passed": bool(passed), "suites": results,
                      "child_option_audit": output["child_option_audit"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-probe-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
