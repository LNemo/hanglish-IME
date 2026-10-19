#!/usr/bin/env python3
"""Fresh sentence-ranker scoring over the production/supported candidate union."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.neural_converter import file_sha256
from hanglish.models.sentence_ranker import REQUIRED_ARTIFACTS, SentenceRanker
from scripts.probe_supported_phonetic_runtime import preserved_segments


RUNTIME = ROOT / ".omx/artifacts/supported-phonetic-runtime-v1"
SOURCE = ROOT / ".omx/artifacts/semantic-pair-training-input-v2"
TARGETS = ROOT / "data/natural_context/final-choice-v1/natural_fit"
PROOF = ROOT / ".omx/artifacts/supported-phonetic-training-v1-independent-v2.json"
RUNTIME_REPORT_SHA = "dc2571818a08f796f74d7f3c972cf05f6a21da1b1c32700ee4b897b430698e5d"
RUNTIME_CAPTURE_SHA = "e5be9c3e175f32bb79099708a9528febd72f7d8b446704ceed10e8f2eafa0ad8"
SOURCE_ROWS_SHA = "6a1f7f61a7f7ffeb9209772559412931137d71ef0173d7429264d9cb53aed396"
SOURCE_MANIFEST_SHA = "3ab561d1a767e7a65b4706c2cd0f668366693536849e17f2f3e620fbcf5550a6"
TARGET_CASES_SHA = "789e45a3b6028e7416ca3b82739c906c558ec414320796c53b0b76ed251b9a18"
TARGET_MANIFEST_SHA = "fc22b1635cc496ed70aa757750944726d583d640f2d9f3a598e380f3e262a1c9"
PROOF_SHA = "c0c2cdb8c5a19abd4b53585536f8eac52e11df7a401ed545841461014a2f802b"
ROWS = 512
TIME_LIMIT = 600
RSS_LIMIT = 2 * 1024**3
SENTENCE_DIR = ROOT / "checkpoints/sentence_ranker"
RUNTIME_FILES = (
    "hanglish/models/sentence_ranker.py",
    "scripts/probe_supported_phonetic_runtime.py",
    "scripts/probe_supported_sentence_union.py",
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def key(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def checked(path: Path, expected: str) -> bytes:
    raw = Path(path).read_bytes()
    if sha(raw) != expected:
        raise ValueError(f"pinned sentence-union source differs: {path}")
    return raw


def jsonl(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.splitlines() if line]


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def guard(started, clock=time.monotonic, rss_reader=rss_bytes):
    elapsed, peak = clock() - started, rss_reader()
    if elapsed > TIME_LIMIT:
        raise RuntimeError("supported sentence union exceeded 600 seconds")
    if peak > RSS_LIMIT:
        raise MemoryError("supported sentence union exceeded 2 GiB RSS")
    return elapsed, peak


def finite_candidates(candidates):
    values = [candidate for candidate in candidates
              if type(candidate.get("score")) in (int, float)
              and math.isfinite(candidate["score"])]
    if len(values) != 4:
        raise ValueError("recorded runtime must expose exactly four finite candidates")
    return values


def sentence_artifacts(artifacts):
    return {name: digest for name, digest in artifacts.items()
            if name.startswith("checkpoints/sentence_ranker/")}


def expected_sentence_artifacts():
    names = set(REQUIRED_ARTIFACTS) | {"metadata.json"}
    return {f"checkpoints/sentence_ranker/{name}": file_sha256(SENTENCE_DIR / name)
            for name in sorted(names)}


def candidate_union(old_candidates, new_candidates):
    old = finite_candidates(old_candidates)
    new = finite_candidates(new_candidates)
    union = []
    seen = set()
    for candidate in [*old, *new]:
        if candidate["text"] not in seen:
            union.append(deepcopy(candidate))
            seen.add(candidate["text"])
    if not 4 <= len(union) <= 8 or [item["text"] for item in union[:4]] != [
            item["text"] for item in old]:
        raise ValueError("sentence union did not preserve the complete old pool prefix")
    if not all(preserved_segments(candidate) for candidate in union):
        raise ValueError("sentence union contains a changed protected/English/raw segment")
    return union


def visible_candidates(ranked, raw_candidate):
    if raw_candidate.get("score") is not None or raw_candidate["text"] == "":
        raise ValueError("recorded raw candidate is malformed")
    return [*ranked[:4], deepcopy(raw_candidate)]


def score_case(row, ranker, expected_artifacts):
    old = row["baseline_candidates"]
    new = row["new_candidates"]
    if not all(preserved_segments(candidate) for candidate in [*old, *new]):
        raise ValueError("recorded candidate changed a protected/English/raw segment")
    new_flags = row["new_response_flags"]
    applicable = (
        new_flags.get("sentence_ranking_applied") is True
        and sentence_artifacts(new_flags.get("model_artifacts", {})) == expected_artifacts
        and all(ranker.can_score(candidate["text"])
                for candidate in [*finite_candidates(old), *finite_candidates(new)])
    )
    if not applicable:
        return {
            "applicable": False, "union_pool": None, "union_candidates": deepcopy(old),
            "union_result": row["baseline_result"], "sentence_scores": [],
            "score_ms": 0.0,
        }
    pool = candidate_union(old, new)
    started = time.perf_counter()
    scores = list(ranker.score_pairs([row["context"]] * len(pool),
                                     [candidate["text"] for candidate in pool]))
    score_ms = (time.perf_counter() - started) * 1000
    if (len(scores) != len(pool)
            or any(type(score) not in (int, float) or not math.isfinite(score)
                   for score in scores)):
        raise ValueError("sentence ranker returned malformed union scores")
    rescored = []
    for candidate, score in zip(pool, scores):
        value = deepcopy(candidate)
        value["score"] = float(score)
        rescored.append(value)
    ranked = sorted(rescored, key=lambda candidate: (-candidate["score"], candidate["text"]))
    if {candidate["text"] for candidate in pool} != {candidate["text"] for candidate in ranked}:
        raise ValueError("sentence ranking deleted a union-pool candidate")
    displayed = visible_candidates(ranked, old[-1])
    return {
        "applicable": True, "union_pool": rescored, "union_candidates": displayed,
        "union_result": displayed[0]["text"], "sentence_scores": [float(v) for v in scores],
        "score_ms": score_ms,
    }


def load_sources():
    source_manifest = json.loads(checked(SOURCE / "manifest.json", SOURCE_MANIFEST_SHA))
    source_rows = [row for row in jsonl(checked(SOURCE / "rows.jsonl", SOURCE_ROWS_SHA))
                   if row.get("source_role") == "natural_fit"][:ROWS]
    captures = jsonl(checked(RUNTIME / "capture.jsonl", RUNTIME_CAPTURE_SHA))
    proof = json.loads(checked(PROOF, PROOF_SHA))
    if (source_manifest.get("rows_sha256") != SOURCE_ROWS_SHA
            or source_manifest.get("labels_or_expected_saved") is not False
            or len(source_rows) != ROWS or len(captures) != ROWS
            or proof.get("completed") is not True
            or proof.get("fresh_replayed_inputs") != 3292):
        raise ValueError("sentence-union source contract differs")
    for index, (source, capture) in enumerate(zip(source_rows, captures)):
        identity = (index, source["id"], source["group_id"], source["input"], source["context"])
        if identity != (capture["index"], capture["id"], capture["group_id"],
                        capture["input"], capture["context"]):
            raise ValueError("runtime capture differs from source-only row order")
        if (capture["baseline_result"] != source["current_result"]
                or capture["baseline_candidates"][:-1]
                != [item["object"] for item in source["finite_candidates"]]):
            raise ValueError("runtime baseline differs from source-only finite candidates")
    return source_manifest, source_rows, captures, proof


def source_pins(source_manifest, expected_artifacts, *, include_labels=False):
    direct = {
        key(SOURCE / "manifest.json"): SOURCE_MANIFEST_SHA,
        key(SOURCE / "rows.jsonl"): SOURCE_ROWS_SHA,
        key(RUNTIME / "capture.jsonl"): RUNTIME_CAPTURE_SHA,
        key(PROOF): PROOF_SHA,
        **expected_artifacts,
        **{name: file_sha256(ROOT / name) for name in RUNTIME_FILES},
    }
    if not include_labels:
        return dict(sorted(direct.items()))
    additions = {
        key(RUNTIME / "report.json"): RUNTIME_REPORT_SHA,
        key(TARGETS / "cases.jsonl"): TARGET_CASES_SHA,
        key(TARGETS / "manifest.json"): TARGET_MANIFEST_SHA,
        **source_manifest.get("source_hashes", {}),
    }
    for name, digest in additions.items():
        if name in direct and direct[name] != digest:
            raise ValueError(f"conflicting sentence-union pin: {name}")
        direct[name] = digest
    return dict(sorted(direct.items()))


def verify_pins(pins):
    for name, digest in pins.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"sentence-union source changed: {name}")


def capture_scores(captures, ranker, artifacts, path, *, resource_guard=lambda: None):
    rows = []
    with Path(path).open("x") as stream:
        for index, capture in enumerate(captures):
            scored = score_case(capture, ranker, artifacts)
            row = {
                "index": index, "id": capture["id"], "group_id": capture["group_id"],
                "input": capture["input"], "context": capture["context"],
                "baseline_result": capture["baseline_result"],
                "baseline_candidates": deepcopy(capture["baseline_candidates"]),
                "new_result": capture["new_result"],
                "new_candidates": deepcopy(capture["new_candidates"]),
                **scored,
            }
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) + "\n")
            stream.flush()
            resource_guard()
            if (index + 1) % 64 == 0:
                print(json.dumps({"event": "supported_sentence_union",
                                  "rows": index + 1}), flush=True)
    return rows


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))]


def summarize(captures, targets, runtime_report, expected_artifacts):
    outcomes = []
    for capture, target in zip(captures, targets):
        identity = (capture["id"], capture["group_id"], capture["input"], capture["context"])
        if identity != (target["id"], target["group_id"], target["input"], target["context"]):
            raise ValueError("late target join identity differs")
        flags = target["response_flags"]
        baseline_applicable = (
            flags.get("sentence_ranking_applied") is True
            and sentence_artifacts(flags.get("model_artifacts", {})) == expected_artifacts
        )
        if capture["applicable"] and not baseline_applicable:
            raise ValueError("scored union lacks a frozen baseline sentence-ranking flag")
        if (target["result"] != capture["baseline_result"]
                or target["candidates"] != capture["baseline_candidates"]):
            raise ValueError("frozen baseline candidates differ")
        expected = target["expected"]
        old_pool = finite_candidates(capture["baseline_candidates"])
        new_pool = finite_candidates(capture["new_candidates"])
        union_pool = capture["union_pool"] or old_pool
        outcome = {
            "id": capture["id"], "expected": expected,
            "baseline_result": capture["baseline_result"], "new_result": capture["new_result"],
            "union_result": capture["union_result"], "applicable": capture["applicable"],
            "baseline_top1": capture["baseline_result"] in expected,
            "new_top1": capture["new_result"] in expected,
            "union_top1": capture["union_result"] in expected,
            "baseline_top5": any(c["text"] in expected for c in capture["baseline_candidates"]),
            "new_top5": any(c["text"] in expected for c in capture["new_candidates"]),
            "union_top5": any(c["text"] in expected for c in capture["union_candidates"]),
            "old_pool_covered": any(c["text"] in expected for c in old_pool),
            "new_pool_covered": any(c["text"] in expected for c in new_pool),
            "union_pool_covered": any(c["text"] in expected for c in union_pool),
        }
        outcomes.append(outcome)
    totals = {name: sum(row[name] for row in outcomes) for name in (
        "baseline_top1", "new_top1", "union_top1", "baseline_top5", "new_top5",
        "union_top5", "old_pool_covered", "new_pool_covered", "union_pool_covered",
    )}
    if (totals["baseline_top1"], totals["baseline_top5"], totals["new_top1"],
            totals["new_top5"]) != (318, 339, 341, 361):
        raise ValueError("pinned baseline/new runtime metrics differ")
    if runtime_report.get("counts") != {
            "rows": 512, "before_top1": 318, "top1": 341,
            "before_top5": 339, "top5": 361,
            "before_beam_coverage": 315, "beam_coverage": 336}:
        raise ValueError("supported runtime report metrics differ")
    totals["rows"] = len(outcomes)
    totals["applicable_rows"] = sum(row["applicable"] for row in outcomes)
    return {
        "counts": totals,
        "correction_ids": [r["id"] for r in outcomes
                           if not r["baseline_top1"] and r["union_top1"]],
        "regression_ids": [r["id"] for r in outcomes
                           if r["baseline_top1"] and not r["union_top1"]],
        "top5_gain_ids": [r["id"] for r in outcomes
                          if not r["baseline_top5"] and r["union_top5"]],
        "top5_loss_ids": [r["id"] for r in outcomes
                          if r["baseline_top5"] and not r["union_top5"]],
        "outcomes": outcomes,
    }


def run(output: Path, *, ranker_factory=SentenceRanker.load,
        clock=time.monotonic, rss_reader=rss_bytes):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = clock()
    inference_started = False
    try:
        source_manifest, _, runtime_rows, proof = load_sources()
        artifacts = expected_sentence_artifacts()
        if sentence_artifacts(runtime_rows[0]["new_response_flags"]["model_artifacts"]) \
                != artifacts:
            raise ValueError("local sentence ranker differs from recorded runtime artifact")
        pre_pins = source_pins(source_manifest, artifacts)
        verify_pins(pre_pins)
        import torch
        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        ranker = ranker_factory(SENTENCE_DIR)
        guard(started, clock, rss_reader)
        inference_started = True
        capture_path = output / "capture.jsonl"
        captures = capture_scores(
            runtime_rows, ranker, artifacts, capture_path,
            resource_guard=lambda: guard(started, clock, rss_reader),
        )
        capture_sha = file_sha256(capture_path)
        # All expected-answer parsing follows closed source-only score capture.
        runtime_report = json.loads(checked(RUNTIME / "report.json", RUNTIME_REPORT_SHA))
        target_manifest = json.loads(checked(TARGETS / "manifest.json", TARGET_MANIFEST_SHA))
        targets = jsonl(checked(TARGETS / "cases.jsonl", TARGET_CASES_SHA))[:ROWS]
        if (runtime_report.get("completed") is not True
                or runtime_report.get("capture_sha256") != RUNTIME_CAPTURE_SHA
                or runtime_report.get("independent_training_verification", {}).get("sha256")
                != PROOF_SHA
                or sentence_artifacts(runtime_report.get("model_artifacts", {})) != artifacts
                or target_manifest.get("rows") < ROWS or len(targets) != ROWS):
            raise ValueError("late label sources differ")
        if sentence_artifacts(target_manifest.get("model_artifact_hashes", {})) != artifacts:
            raise ValueError("production and supported runtimes used different sentence rankers")
        pins = source_pins(source_manifest, artifacts, include_labels=True)
        verify_pins(pins)
        summary = summarize(captures, targets, runtime_report, artifacts)
        verify_pins(pins)
        elapsed, peak = guard(started, clock, rss_reader)
        times = [row["score_ms"] for row in captures if row["applicable"]]
        outcomes_path = output / "outcomes.jsonl"
        outcomes_path.write_bytes(b"".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")).encode() + b"\n"
            for row in summary["outcomes"]
        ))
        report = {
            "schema": "supported-sentence-union-probe-v1", "completed": True,
            "scope": "natural FIT TRAIN first512 diagnostic; no new candidates or training",
            "counts": summary["counts"], "correction_ids": summary["correction_ids"],
            "regression_ids": summary["regression_ids"],
            "top5_gain_ids": summary["top5_gain_ids"],
            "top5_loss_ids": summary["top5_loss_ids"],
            "capture_sha256": capture_sha, "outcomes_sha256": file_sha256(outcomes_path),
            "runtime_report_sha256": RUNTIME_REPORT_SHA,
            "runtime_capture_sha256": RUNTIME_CAPTURE_SHA,
            "independent_training_proof_sha256": PROOF_SHA,
            "independent_fresh_replayed_inputs": proof["fresh_replayed_inputs"],
            "sentence_ranker_artifacts": artifacts,
            "source_hashes": pins, "source_hashes_after": pins,
            "latency_ms": {"p50": statistics.median(times) if times else 0.0,
                           "p95": percentile(times, .95)},
            "old_pool_preserved_before_selection": True,
            "protected_english_raw_segments_preserved": True,
            "eligibility_policy": (
                "Before label access: supported-runtime sentence_ranking_applied, identical "
                "sentence-ranker artifacts, and every union candidate can_score. The source-only "
                "cache has no baseline sentence_ranking_applied field; every scored row is "
                "validated against that frozen baseline flag after capture closes."
            ),
            "baseline_sentence_flag_available_before_capture": False,
            "scored_rows_baseline_flag_validated_after_capture": True,
            "labels_parsed_after_closed_capture": True,
            "actual_sentence_ranker_inference_run": True,
            "new_transducer_inference_run": False, "training_run": False,
            "development_data_read": False, "reserve_data_read": False,
            "final_data_read": False, "production_changed": False,
            "elapsed_seconds": elapsed, "peak_rss_bytes": peak,
        }
        report_path = output / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False,
                                          sort_keys=True, indent=2) + "\n")
        try:
            guard(started, clock, rss_reader)
        except Exception:
            report_path.unlink(missing_ok=True)
            raise
        (output / "status.json").write_text(json.dumps({
            "state": "complete", "report_sha256": file_sha256(report_path),
        }, sort_keys=True) + "\n")
        return report
    except Exception as error:
        (output / "report.json").unlink(missing_ok=True)
        (output / "status.json").unlink(missing_ok=True)
        (output / "failure.json").write_text(json.dumps({
            "schema": "supported-sentence-union-probe-failure-v1",
            "error_type": type(error).__name__, "error": str(error),
            "actual_sentence_ranker_inference_run": inference_started,
            "elapsed_seconds": clock() - started, "peak_rss_bytes": rss_reader(),
        }, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, sort_keys=True))
