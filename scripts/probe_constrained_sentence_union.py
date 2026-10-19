"""Train-only sentence reranking over recorded constrained-runtime pools."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hanglish.core.neural_converter import NeuralConverter, file_sha256
from scripts.prepare_context_training import load_sources, mixed_split_guard
from scripts.prepare_natural_ranker import source_fields
from scripts.probe_constrained_runtime import (
    OUTPUT as RECORDED,
    POLICY as RECORDED_POLICY,
    VERSION as PARENT_VERSION,
    hashes as recorded_source_hashes,
    metrics as recorded_metrics,
)
from scripts.train_natural_sentence_ranker import rss_bytes


RECORDED_REPORT_SHA256 = "42d08b9c850a6d34d5f55661c350b130029cd0a82639ab63898b399531f576ae"
OUTPUT = ROOT / ".omx/artifacts/constrained-sentence-union-v1"
PROFILE = {
    "name": "recorded-constrained-sentence-union-train-v1",
    "rows": 512,
    "max_union_pool": 96,
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 900,
    "progress_interval": 64,
    "selection": "All fixed natural train rows in recorded source order",
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def source_hashes():
    hashes = recorded_source_hashes()
    for path in (
        Path(__file__).resolve(),
        RECORDED / "report.json",
        RECORDED / "cases.jsonl",
        RECORDED / "selection.json",
    ):
        hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    return dict(sorted(hashes.items()))


def visible_candidates(ranked, original):
    candidates = list(ranked[:5])
    if original not in candidates:
        candidates = candidates[:4] + [original]
    return candidates


def union_pool(before_pool, new_pool, limit=96):
    """Preserve recorded order and deduplicate without consulting targets."""
    if before_pool is None:
        return None
    values = list(dict.fromkeys([*before_pool, *(new_pool or [])]))
    if values[:len(before_pool)] != before_pool:
        raise ValueError("sentence union changed the parent pool prefix")
    if len(values) > limit or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("sentence union violates its candidate bound")
    return values


def validate_recording(report, cases, selection, fixed_selection):
    if (file_sha256(RECORDED / "report.json") != RECORDED_REPORT_SHA256
            or report.get("source_hashes") != recorded_source_hashes()
            or report.get("policy") != RECORDED_POLICY
            or report.get("parent_version") != PARENT_VERSION
            or report.get("test_data_read") is not False
            or report.get("development_inference_run") is not False
            or report.get("deployed") is not False
            or file_sha256(RECORDED / "cases.jsonl") != report.get("cases_sha256")
            or file_sha256(RECORDED / "selection.json") != report.get("selection_sha256")
            or report.get("audit", {}).get("max_added", 99) > RECORDED_POLICY["additional_candidates"]
            or RECORDED_POLICY.get("preserve_parent_prefix") is not True):
        raise ValueError("constrained runtime recording is not the pinned train-only source")
    if not (len(cases) == len(selection) == len(fixed_selection) == PROFILE["rows"]):
        raise ValueError("constrained runtime recording is incomplete")
    for case, selected, fixed in zip(cases, selection, fixed_selection):
        keys = ("id", "input", "context", "expected")
        if (any(case.get(key) != selected.get(key) for key in keys)
                or any(selected.get(key) != fixed.get(key) for key in keys)):
            raise ValueError("recorded cases differ from the fixed train selection")
        for side in ("before", "new"):
            candidates = case.get(f"{side}_candidates")
            pool = case.get(f"{side}_pool")
            if (not isinstance(candidates, list) or not candidates
                    or any(not isinstance(value, str) for value in candidates)
                    or (pool is not None and (not isinstance(pool, list)
                        or len(pool) > 48 or len(set(pool)) != len(pool)))):
                raise ValueError("recorded candidate or sentence pool is invalid")
    if recorded_metrics(cases) != report.get("metrics"):
        raise ValueError("recorded metrics do not reproduce from cases")


def evaluate_case(case, sentence_ranker):
    """Build and score the source-only union before reading expected labels."""
    pool = union_pool(case["before_pool"], case["new_pool"], PROFILE["max_union_pool"])
    if pool is None:
        actual = list(case["before_candidates"])
        sentence_ranking_applied = False
        scores = []
    else:
        scores = [float(value) for value in sentence_ranker.score_pairs(
            [case["context"]] * len(pool), pool
        )]
        if len(scores) != len(pool) or any(not math.isfinite(value) for value in scores):
            raise ValueError("sentence ranker returned an invalid score count")
        ranked = [candidate for candidate, _ in sorted(
            zip(pool, scores), key=lambda item: (-item[1], item[0])
        )]
        actual = visible_candidates(ranked, case["input"])
        sentence_ranking_applied = True

    # Pool construction and sentence selection are target-independent.
    expected = case["expected"]
    before = case["before_candidates"]
    return {
        "id": case["id"],
        "input": case["input"],
        "context": case["context"],
        "expected": expected,
        "before_candidates": before,
        "actual": actual,
        "union_pool": pool,
        "sentence_scores": scores,
        "before_pool_size": None if case["before_pool"] is None else len(case["before_pool"]),
        "new_pool_size": None if case["new_pool"] is None else len(case["new_pool"]),
        "union_pool_size": None if pool is None else len(pool),
        "added_pool_candidates": 0 if pool is None else len(pool) - len(case["before_pool"]),
        "before_pool_covered": None if case["before_pool"] is None else any(
            candidate in expected for candidate in case["before_pool"]
        ),
        "new_pool_covered": None if case["new_pool"] is None else any(
            candidate in expected for candidate in case["new_pool"]
        ),
        "union_pool_covered": None if pool is None else any(
            candidate in expected for candidate in pool
        ),
        "before_top1": before[0] in expected,
        "top1": actual[0] in expected,
        "before_top5": any(candidate in expected for candidate in before),
        "top5": any(candidate in expected for candidate in actual),
        "before_first": before[0],
        "first": actual[0],
        "first_choice_changed": before[0] != actual[0],
        "sentence_ranking_applied": sentence_ranking_applied,
    }


def summarize(cases):
    return {
        "rows": len(cases),
        "before_top1": sum(case["before_top1"] for case in cases),
        "top1": sum(case["top1"] for case in cases),
        "before_top5": sum(case["before_top5"] for case in cases),
        "top5": sum(case["top5"] for case in cases),
        "before_pool_covered": sum(case["before_pool_covered"] is True for case in cases),
        "new_pool_covered": sum(case["new_pool_covered"] is True for case in cases),
        "union_pool_covered": sum(case["union_pool_covered"] is True for case in cases),
        "sentence_scored_rows": sum(case["union_pool_covered"] is not None for case in cases),
        "first_choice_changes": sum(case["first_choice_changed"] for case in cases),
        "corrections": [case["id"] for case in cases
                        if case["top1"] and not case["before_top1"]],
        "regressions": [case["id"] for case in cases
                        if case["before_top1"] and not case["top1"]],
    }


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    report_path = RECORDED / "report.json"
    report = json.loads(report_path.read_text())
    cases = read_jsonl(RECORDED / "cases.jsonl")
    selection = json.loads((RECORDED / "selection.json").read_text())
    raw, challenge_guard = load_sources()
    fixed = [source_fields(row, "natural_train") for row in raw["natural_train"]]
    split_guard = mixed_split_guard({
        role: [source_fields(row, role) for row in rows] for role, rows in raw.items()
    })
    validate_recording(report, cases, selection, fixed)
    before = source_hashes()
    torch.set_num_threads(PROFILE["threads"])
    converter = NeuralConverter(ranker_kind="hybrid")
    converter.load_models()
    if converter.version != PARENT_VERSION:
        raise ValueError("current runtime differs from the recorded parent")
    sentence_ranker = converter.sentence_ranker
    args.output.mkdir(parents=True)
    selection_path = args.output / "selection.json"
    selection_path.write_text(json.dumps(selection, ensure_ascii=False, sort_keys=True) + "\n")
    output_selection_sha256 = file_sha256(selection_path)
    cases_path = args.output / "cases.jsonl"
    started = time.monotonic()
    outcomes = []
    with cases_path.open("x") as stream:
        for index, case in enumerate(cases):
            outcome = evaluate_case(case, sentence_ranker)
            outcomes.append(outcome)
            stream.write(json.dumps(outcome, ensure_ascii=False, sort_keys=True) + "\n")
            elapsed = time.monotonic() - started
            if rss_bytes() > PROFILE["rss_limit_bytes"] or elapsed > PROFILE["time_limit_seconds"]:
                raise RuntimeError("constrained sentence probe resource bound exceeded")
            if (index + 1) % PROFILE["progress_interval"] == 0:
                print(json.dumps({"event": "constrained_sentence_union",
                                  "completed": index + 1,
                                  "elapsed_seconds": round(elapsed, 2)}), flush=True)
    validate_recording(report, cases, selection, fixed)
    after = source_hashes()
    if before != after or file_sha256(selection_path) != output_selection_sha256:
        raise ValueError("sentence union sources changed during evaluation")
    metrics = summarize(outcomes)
    pool_sizes = [case["union_pool_size"] for case in outcomes
                  if case["union_pool_size"] is not None]
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Recorded source-only train512 sentence-pool experiment; this does not "
            "implement or claim a new convert runtime."
        ),
        "profile": PROFILE,
        "policy": {
            "parent_pool_prefix_preserved": True,
            "deduplicate_by_candidate_text": True,
            "recorded_before_pool_then_recorded_new_pool": True,
            "fallback_when_parent_sentence_pool_absent": "keep recorded parent output",
            "targets_used_for_candidate_addition_or_selection": False,
        },
        "parent_version": PARENT_VERSION,
        "recorded_report_sha256": RECORDED_REPORT_SHA256,
        "recorded_cases_sha256": report["cases_sha256"],
        "recorded_selection_sha256": report["selection_sha256"],
        "source_hashes": before,
        "challenge_guard": challenge_guard,
        "split_guard": split_guard,
        "selection_sha256": output_selection_sha256,
        "cases_sha256": file_sha256(cases_path),
        "metrics": metrics,
        "pool_sizes": {
            "maximum": max(pool_sizes, default=0),
            "median": statistics.median(pool_sizes) if pool_sizes else None,
        },
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
        "test_data_read": False,
        "development_inference_run": False,
        "deployed": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"metrics": metrics, "pool_sizes": output["pool_sizes"],
                      "elapsed_seconds": output["elapsed_seconds"],
                      "peak_rss_bytes": output["peak_rss_bytes"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
