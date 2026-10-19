"""Train-only probe of sentence-ranker context truncation side on a fixed prefix."""
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

from hanglish.core.neural_converter import file_sha256
from hanglish.models.sentence_ranker import SentenceRanker, encode_sentence_pairs
from scripts.prepare_current_context_data import (
    OUTPUT as PREPARED,
    RUNTIME_VERSION,
    load_sources,
    source_hashes as preparation_source_hashes,
    validate_case,
)
from scripts.train_natural_sentence_ranker import rss_bytes


PARENT = ROOT / "checkpoints/sentence_ranker"
PREFIX_PARITY = ROOT / ".omx/artifacts/current-context-training-work-v1/train-prefix-parity.json"
OUTPUT = ROOT / ".omx/artifacts/sentence-context-side-v1"
PROFILE = {
    "name": "sentence-context-truncation-side-train-prefix-v1",
    "rows": 512,
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 900,
    "progress_interval": 32,
    "selection": "First 512 fixed natural TRAIN rows in source order",
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical_rows_sha256(rows):
    """Match the accepted parity audit's sorted-key JSON array digest."""
    return sha256_bytes(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode())


def read_prefix(path, rows=PROFILE["rows"]):
    """Read exactly a complete JSONL prefix while the source may be appended."""
    lines = []
    with path.open("rb") as stream:
        for _ in range(rows):
            line = stream.readline()
            if not line.endswith(b"\n"):
                raise ValueError("current training prefix is incomplete")
            lines.append(line)
    raw = b"".join(lines)
    parsed = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if len(parsed) != rows:
        raise ValueError("current training prefix has the wrong row count")
    return raw, parsed


def visible_candidates(ranked, original):
    values = list(ranked[:5])
    return values if original in values else values[:4] + [original]


def ranked_candidates(candidates, scores):
    if (len(candidates) != len(scores)
            or any(not math.isfinite(float(score)) for score in scores)):
        raise ValueError("sentence score vector is invalid")
    return [candidate for candidate, _ in sorted(
        zip(candidates, scores), key=lambda item: (-item[1], item[0])
    )]


def validate_pair_encoding(ranker, candidate, context, side):
    """Prove only sequence B truncates and the requested edge of B survives."""
    tokenizer = ranker.tokenizer
    if side not in ("left", "right") or tokenizer.truncation_side != side:
        raise ValueError("tokenizer truncation side differs from probe policy")
    encoded = encode_sentence_pairs(tokenizer, [candidate], [context], ranker.max_length)
    sequence_ids = encoded.sequence_ids(0)
    ids = encoded["input_ids"][0].tolist()
    first = [token for token, sequence in zip(ids, sequence_ids) if sequence == 0]
    second = [token for token, sequence in zip(ids, sequence_ids) if sequence == 1]
    normalized_candidate = " ".join(candidate.strip().split())
    expected_first = tokenizer(
        normalized_candidate, add_special_tokens=False, truncation=False
    )["input_ids"]
    if expected_first and isinstance(expected_first[0], list):
        expected_first = expected_first[0]
    if first != expected_first:
        raise ValueError("candidate sequence A was truncated or changed")
    normalized_context = " ".join(context.strip().split())
    expected_second = tokenizer(
        normalized_context, add_special_tokens=False, truncation=False
    )["input_ids"]
    if expected_second and isinstance(expected_second[0], list):
        expected_second = expected_second[0]
    truncated = len(second) < len(expected_second)
    retained = expected_second[-len(second):] if side == "left" and second else expected_second[:len(second)]
    if second != retained:
        raise ValueError("sequence B did not preserve the requested context edge")
    return {"truncated": truncated, "candidate_tokens": len(first),
            "context_tokens": len(expected_second), "retained_context_tokens": len(second)}


def score_side(ranker, row, side):
    previous = ranker.tokenizer.truncation_side
    try:
        ranker.tokenizer.truncation_side = side
        audits = [validate_pair_encoding(ranker, candidate, row["context"], side)
                  for candidate in row["candidates"]]
        scores = [float(value) for value in ranker.score_pairs(
            [row["context"]] * len(row["candidates"]), row["candidates"]
        )]
    finally:
        ranker.tokenizer.truncation_side = previous
    return scores, audits


def evaluate_case(row, right_scores=None, left_scores=None):
    """Apply the existing visible top-five policy without changing the pool."""
    if not row["sentence_eligible"]:
        right = left = list(row["actual_candidates"])
        if right_scores not in (None, []) or left_scores not in (None, []):
            raise ValueError("fallback row must not be rescored")
        right_scores = left_scores = []
    else:
        if right_scores is None or left_scores is None:
            raise ValueError("eligible row requires both score vectors")
        right = visible_candidates(
            ranked_candidates(row["candidates"], right_scores), row["input"]
        )
        left = visible_candidates(
            ranked_candidates(row["candidates"], left_scores), row["input"]
        )
        if right != row["actual_candidates"]:
            raise ValueError("fresh right-side scoring differs from recorded runtime")
    expected = row["expected"]
    return {
        "id": row["id"],
        "input": row["input"],
        "context": row["context"],
        "expected": expected,
        "candidates": row["candidates"],
        "sentence_eligible": row["sentence_eligible"],
        "right_scores": right_scores,
        "left_scores": left_scores,
        "right_actual": right,
        "left_actual": left,
        "pool_covered": (None if not row["sentence_eligible"] else
                         any(candidate in expected for candidate in row["candidates"])),
        "right_top1": right[0] in expected,
        "left_top1": left[0] in expected,
        "right_top5": any(candidate in expected for candidate in right),
        "left_top5": any(candidate in expected for candidate in left),
        "right_first": right[0],
        "left_first": left[0],
        "first_choice_changed": right[0] != left[0],
    }


def summarize(cases):
    return {
        "rows": len(cases),
        "sentence_scored_rows": sum(case["sentence_eligible"] for case in cases),
        "fallback_rows": sum(not case["sentence_eligible"] for case in cases),
        "pool_covered": sum(case["pool_covered"] is True for case in cases),
        "right_top1": sum(case["right_top1"] for case in cases),
        "left_top1": sum(case["left_top1"] for case in cases),
        "right_top5": sum(case["right_top5"] for case in cases),
        "left_top5": sum(case["left_top5"] for case in cases),
        "first_choice_changes": sum(case["first_choice_changed"] for case in cases),
        "corrections": [case["id"] for case in cases
                        if case["left_top1"] and not case["right_top1"]],
        "regressions": [case["id"] for case in cases
                        if case["right_top1"] and not case["left_top1"]],
    }


def source_hashes():
    hashes = preparation_source_hashes()
    paths = [Path(__file__).resolve(), PREFIX_PARITY,
             ROOT / "hanglish/models/sentence_ranker.py"]
    paths.extend(path for path in PARENT.iterdir() if path.is_file())
    for path in paths:
        hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    return dict(sorted(hashes.items()))


def validate_selection(rows, fixed):
    if len(rows) != PROFILE["rows"] or len(fixed) < PROFILE["rows"]:
        raise ValueError("fixed first-512 selection is incomplete")
    for row, source in zip(rows, fixed[:PROFILE["rows"]]):
        validate_case(row, source)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    input_path = args.prepared / "natural_train.jsonl"
    prefix, rows = read_prefix(input_path)
    fixed, challenge_guard, split_guard = load_sources()
    validate_selection(rows, fixed["natural_train"])
    parity = json.loads(PREFIX_PARITY.read_text())
    prefix_sha256 = sha256_bytes(prefix)
    parity_rows_sha256 = canonical_rows_sha256(rows)
    if (parity.get("rows") != PROFILE["rows"]
            or parity.get("top5_exact_parity") != PROFILE["rows"]
            or parity.get("current_version") != RUNTIME_VERSION
            or parity.get("current_prefix_sha256") != parity_rows_sha256):
        raise ValueError("current prefix differs from the accepted train parity audit")
    before = source_hashes()
    torch.set_num_threads(PROFILE["threads"])
    started = time.monotonic()

    def require_resources(stage):
        if (rss_bytes() > PROFILE["rss_limit_bytes"]
                or time.monotonic() - started > PROFILE["time_limit_seconds"]):
            raise RuntimeError(f"sentence context-side resource bound exceeded during {stage}")

    ranker = SentenceRanker.load(PARENT)
    if ranker.tokenizer.truncation_side != "right":
        raise ValueError("canonical sentence tokenizer is not right-truncating")
    args.output.mkdir(parents=True)
    snapshot_path = args.output / "natural-train-prefix512.jsonl"
    snapshot_path.write_bytes(prefix)
    cases_path = args.output / "cases.jsonl"
    outcomes = []
    audit = {side: {"pairs": 0, "truncated_context_pairs": 0,
                    "candidate_sequence_preserved": True,
                    "requested_context_edge_preserved": True}
             for side in ("right", "left")}
    max_teacher_delta = 0.
    with cases_path.open("x") as stream:
        for index, row in enumerate(rows):
            if row["sentence_eligible"]:
                right_scores, right_audits = score_side(ranker, row, "right")
                left_scores, left_audits = score_side(ranker, row, "left")
                delta = max(abs(left - right) for left, right in
                            zip(right_scores, row["teacher_scores"]))
                max_teacher_delta = max(max_teacher_delta, delta)
                if delta > 1e-4:
                    raise ValueError("fresh right-side scores differ from captured teacher")
                for side, values in (("right", right_audits), ("left", left_audits)):
                    audit[side]["pairs"] += len(values)
                    audit[side]["truncated_context_pairs"] += sum(
                        value["truncated"] for value in values
                    )
            else:
                right_scores = left_scores = []
            outcome = evaluate_case(row, right_scores, left_scores)
            outcomes.append(outcome)
            stream.write(json.dumps(outcome, ensure_ascii=False, sort_keys=True) + "\n")
            require_resources("evaluation")
            if (index + 1) % PROFILE["progress_interval"] == 0:
                print(json.dumps({"event": "sentence_context_side",
                                  "completed": index + 1,
                                  "elapsed_seconds": round(time.monotonic() - started, 2)}),
                      flush=True)
    if ranker.tokenizer.truncation_side != "right":
        raise RuntimeError("probe did not restore the tokenizer truncation side")
    current_prefix, current_rows = read_prefix(input_path)
    validate_selection(current_rows, fixed["natural_train"])
    after = source_hashes()
    if (current_prefix != prefix or before != after
            or snapshot_path.read_bytes() != prefix):
        raise RuntimeError("fixed prefix or probe inputs changed during evaluation")
    metrics = summarize(outcomes)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Train-only first512 diagnostic of the same recorded sentence pools. "
            "It changes only context sequence-B truncation and makes no latency comparison."
        ),
        "profile": PROFILE,
        "policy": {
            "candidate_pool_unchanged": True,
            "candidate_sequence_a_preserved": True,
            "truncation": "only_second",
            "baseline_context_side": "right",
            "experimental_context_side": "left",
            "left_policy": "preserve the end of context sequence B",
            "fallback_policy": "keep recorded actual candidates",
            "targets_used_for_candidates_or_selection": False,
            "latency_comparison_claimed": False,
        },
        "runtime_version": RUNTIME_VERSION,
        "source_hashes": before,
        "challenge_guard": challenge_guard,
        "split_guard": split_guard,
        "prefix_parity_sha256": file_sha256(PREFIX_PARITY),
        "canonical_prefix_rows_sha256": parity_rows_sha256,
        "input_prefix_sha256": prefix_sha256,
        "input_prefix_bytes": len(prefix),
        "snapshot_sha256": file_sha256(snapshot_path),
        "snapshot_bytes": snapshot_path.stat().st_size,
        "cases_sha256": file_sha256(cases_path),
        "max_right_teacher_score_delta": max_teacher_delta,
        "encoding_audit": audit,
        "metrics": metrics,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
        "development_data_read": False,
        "test_data_read": False,
        "deployed": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"metrics": metrics,
                      "max_right_teacher_score_delta": max_teacher_delta,
                      "encoding_audit": audit,
                      "elapsed_seconds": report["elapsed_seconds"],
                      "peak_rss_bytes": report["peak_rss_bytes"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, default=PREPARED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
