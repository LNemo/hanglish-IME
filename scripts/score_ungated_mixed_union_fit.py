#!/usr/bin/env python3
"""Score the closed repaired FIT pool with verified mixed-boundary additions."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.score_ungated_joint_fit import (
    CHECKPOINT, model_pins, need, read, sha, verify_pins, write,
)

BASELINE = ROOT / ".omx/artifacts/ungated-joint-sentence-score-fit-v1"
MIXED = ROOT / ".omx/artifacts/ungated-mixed-boundary-fit-v1"
PLAN = ROOT / ".omx/plans/ungated-mixed-union-fit-v1.md"
ROWS = 13
TOLERANCE = 1e-5
LIMITS = {"whole_seconds": 600.0, "row_seconds": 60.0,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2,
          "output_bytes": 128 * 1024**2}


def authenticate_sources():
    base_report, base_close, base_proof = (
        read(BASELINE / "report.json"), read(BASELINE / "capture.complete.json"),
        read(BASELINE / "independent-verification.json"))
    need(base_report.get("schema") == "ungated-joint-sentence-scoring-v1"
         and base_report.get("completed") is True and base_report.get("rows") == ROWS
         and base_report.get("labels_opened") is False
         and base_report.get("generation_inference") is False
         and base_close.get("completed") is True and base_close.get("rows") == ROWS
         and base_close.get("report_sha256") == sha(BASELINE / "report.json")
         and base_close.get("capture_sha256") == sha(BASELINE / "capture.jsonl")
         and base_proof.get("verified") is True and base_proof.get("rows") == ROWS
         and base_proof.get("score_report_sha256") == sha(BASELINE / "report.json")
         and base_proof.get("score_capture_sha256") == sha(BASELINE / "capture.jsonl"),
         "closed verified joint score capture required")
    mixed_report, mixed_close, mixed_proof = (
        read(MIXED / "report.json"), read(MIXED / "capture.complete.json"),
        read(MIXED / "independent-verification.json"))
    need(mixed_report.get("schema") == "ungated-mixed-boundary-fit-capture-v1"
         and mixed_report.get("completed") is True
         and mixed_report.get("counts", {}).get("rows") == 66
         and mixed_report.get("counts", {}).get("proposals") == 293
         and mixed_report.get("labels_opened") is False
         and mixed_report.get("sentence_scoring_run") is False
         and mixed_close.get("completed") is True and mixed_close.get("rows") == 66
         and mixed_close.get("report_sha256") == sha(MIXED / "report.json")
         and mixed_close.get("capture_sha256") == sha(MIXED / "capture.jsonl")
         and mixed_proof.get("verified") is True and mixed_proof.get("rows") == 66
         and mixed_proof.get("report_sha256") == sha(MIXED / "report.json")
         and mixed_proof.get("capture_sha256") == sha(MIXED / "capture.jsonl"),
         "closed verified mixed-boundary capture required")
    verify_pins(base_report["pins"]); verify_pins(mixed_report["pins"])
    pins = {**base_report["pins"], **mixed_report["pins"], **model_pins()}
    for path in (BASELINE / "report.json", BASELINE / "capture.complete.json",
                 BASELINE / "independent-verification.json", BASELINE / "capture.jsonl",
                 MIXED / "report.json", MIXED / "capture.complete.json",
                 MIXED / "independent-verification.json", MIXED / "capture.jsonl",
                 PLAN, Path(__file__).resolve()):
        pins[str(path.resolve())] = sha(path)
    verify_pins(pins)
    return pins


def load_mixed_rows():
    rows = {}
    with (MIXED / "capture.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            need(row["id"] not in rows, "duplicate mixed source ID")
            rows[row["id"]] = row
    need(len(rows) == 66, "mixed source row count differs")
    return rows


def _candidate(row, provenance=None):
    need(isinstance(row, dict) and isinstance(row.get("text"), str) and row["text"],
         "candidate text differs")
    if provenance is None:
        support = row.get("support")
        provenances = deepcopy(row.get("provenance"))
        need(support in {"supported", "unsupported"}
             and isinstance(provenances, list) and provenances,
             "baseline candidate differs")
    else:
        token = row.get("token_support")
        need(isinstance(token, dict) and type(token.get("supported")) is bool,
             "mixed token support differs")
        support = "supported" if token["supported"] else "unsupported"
        provenances = [provenance]
    return {"text": row["text"], "support": support, "provenance": provenances}


def project_row(baseline, mixed=None):
    need(baseline.get("request", {}).get("context") == ""
         and set(baseline.get("arms", {})) == {"original", "roman_repaired"},
         "baseline row contract differs")
    source = baseline["request"]["input"]
    fallback = baseline.get("fallback_public")
    need(isinstance(fallback, list) and len(fallback) == 5
         and fallback[-1].get("text") == source
         and isinstance(fallback[-1].get("segments"), list)
         and bool(fallback[-1]["segments"])
         and all(s.get("kind") == "raw" for s in fallback[-1]["segments"]),
         "raw reserve differs")
    base = [_candidate(row) for row in baseline["arms"]["roman_repaired"]["pool"]]
    need(len(base) == len({row["text"] for row in base}), "baseline pool duplicates")
    union = deepcopy(base); by_text = {row["text"]: row for row in union}
    if mixed is not None:
        need(mixed.get("id") == baseline.get("id")
             and mixed.get("request") == baseline.get("request"),
             "mixed source identity differs")
        for raw in mixed.get("proposals", []):
            candidate = _candidate(raw, "mixed_boundary")
            previous = by_text.get(candidate["text"])
            if previous is None:
                by_text[candidate["text"]] = candidate; union.append(candidate)
            else:
                need(previous["support"] == candidate["support"],
                     "duplicate support differs")
                if "mixed_boundary" not in previous["provenance"]:
                    previous["provenance"].append("mixed_boundary")
    return {"id": baseline["id"], "request": deepcopy(baseline["request"]),
            "fallback_public": deepcopy(fallback), "raw_reserve": deepcopy(fallback[-1]),
            "baseline": base, "mixed_union": union,
            "old_scores": {row["text"]: float(row["score"])
                           for row in baseline.get("supported_scores", [])},
            "old_supported_order": [row["text"]
                                    for row in baseline.get("supported_scores", [])],
            "expected_baseline_order": deepcopy(
                baseline["arms"]["roman_repaired"]["supported_order"]),
            "expected_baseline_public_texts": [row["text"] for row in
                baseline["arms"]["roman_repaired"]["public"]],
            "mixed_source_present": mixed is not None}


def _arm(pool, scores, fallback, raw):
    supported = sorted((row["text"] for row in pool if row["support"] == "supported"),
                       key=lambda text: (-scores[text], text))
    unsupported = [row["text"] for row in pool if row["support"] == "unsupported"]
    if supported:
        public = [{"text": text, "sentence_score": scores[text]}
                  for text in supported[:4]] + [deepcopy(raw)]
        fallback_used = False
    else:
        public = deepcopy(fallback); fallback_used = True
    return {"pool": deepcopy(pool), "supported_order": supported,
            "unsupported_texts": unsupported, "public": public,
            "fallback_used": fallback_used}


def score_row(projected, ranker):
    status, order = {}, []
    for row in projected["mixed_union"]:
        text = row["text"]
        if text not in status: status[text] = row["support"]; order.append(text)
        else: need(status[text] == row["support"], "union support differs")
    for text in order:
        actual = "supported" if ranker.can_score(text) else "unsupported"
        need(actual == status[text], "tokenizer support differs")
    # The frozen source capture scored the union of both old arms in this exact
    # order. Preserve that batching/padding context; score only truly additive
    # mixed candidates in a separate call.
    old_order = projected["old_supported_order"]
    need(len(old_order) == len(set(old_order)) == len(projected["old_scores"]),
         "old supported score order differs")
    old_values = ranker.score_pairs([""] * len(old_order), old_order) if old_order else []
    additive = [text for text in order
                if status[text] == "supported" and text not in projected["old_scores"]]
    additive_values = ranker.score_pairs([""] * len(additive), additive) if additive else []
    values = old_values + additive_values
    scored_order = old_order + additive
    need(len(values) == len(scored_order)
         and all(not isinstance(v, bool) and isinstance(v, (int, float))
                 and math.isfinite(v) for v in values), "fresh score differs")
    scores = dict(zip(scored_order, map(float, values)))
    deltas = []
    for row in projected["baseline"]:
        if row["support"] == "supported":
            need(row["text"] in projected["old_scores"], "old baseline score missing")
            delta = abs(scores[row["text"]] - projected["old_scores"][row["text"]])
            need(delta <= TOLERANCE, "fresh baseline score exceeds tolerance")
            deltas.append(delta)
    arms = {name: _arm(projected[name], scores, projected["fallback_public"],
                       projected["raw_reserve"])
            for name in ("baseline", "mixed_union")}
    need(arms["baseline"]["supported_order"] == projected["expected_baseline_order"],
         "fresh baseline ordering differs")
    need([row["text"] for row in arms["baseline"]["public"]]
         == projected["expected_baseline_public_texts"],
         "fresh baseline public differs")
    return {"id": projected["id"], "request": projected["request"],
            "fallback_public": projected["fallback_public"],
            "supported_scores": [{"text": text, "score": scores[text]}
                                 for text in scored_order],
            "old_supported_score_count": len(old_order),
            "additive_supported_score_count": len(additive),
            "unsupported_texts": [text for text in order if status[text] == "unsupported"],
            "arms": arms, "baseline_max_score_delta": max(deltas, default=0.0),
            "mixed_source_present": projected["mixed_source_present"]}


def capture(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    from scripts.train_character_edge_full_fit import ResourceGuard
    guard = ResourceGuard(output)
    def observe(stage):
        value = guard.observe(stage)
        need(value["seconds"] <= LIMITS["whole_seconds"]
             and value["rss_bytes"] <= LIMITS["rss_bytes"]
             and value["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"],
             "resource limit exceeded")
        need(sum(p.stat().st_size for p in output.iterdir() if p.is_file())
             <= LIMITS["output_bytes"], "output limit exceeded")
        return value
    try:
        pins = authenticate_sources(); mixed = load_mixed_rows(); observe("authenticated")
        import torch
        from hanglish.models.sentence_ranker import SentenceRanker
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        ranker = SentenceRanker.load(CHECKPOINT)
        need(ranker.max_length == 96 and ranker.batch_size == 8, "ranker settings differ")
        counts = {name: Counter(rows=0, pool=0, supported=0, unsupported=0,
                                fallback_rows=0) for name in ("baseline", "mixed_union")}
        totals = Counter(rows=0, mixed_source_rows=0, unique_supported_scored=0,
                         unique_unsupported_unscored=0)
        max_delta = 0.0
        with (BASELINE / "capture.jsonl").open(encoding="utf-8") as source, \
                (output / "capture.jsonl").open("x", encoding="utf-8") as target:
            for index, line in enumerate(source):
                need(index < ROWS, "extra baseline row")
                baseline = json.loads(line); projected = project_row(baseline, mixed.get(baseline["id"]))
                started = time.monotonic(); result = score_row(projected, ranker)
                need(time.monotonic() - started <= LIMITS["row_seconds"], "row time exceeded")
                target.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"); target.flush()
                totals.update(rows=1, mixed_source_rows=int(result["mixed_source_present"]),
                              unique_supported_scored=len(result["supported_scores"]),
                              unique_unsupported_unscored=len(result["unsupported_texts"]))
                max_delta = max(max_delta, result["baseline_max_score_delta"])
                for name, arm in result["arms"].items():
                    counts[name].update(rows=1, pool=len(arm["pool"]),
                                        supported=len(arm["supported_order"]),
                                        unsupported=len(arm["unsupported_texts"]),
                                        fallback_rows=int(arm["fallback_used"]))
                observe("row-" + result["id"])
        need(totals["rows"] == ROWS, "baseline row count differs")
        verify_pins(pins); final = observe("complete")
        report = {"schema": "ungated-mixed-union-sentence-scoring-v1",
                  "completed": True, "rows": ROWS,
                  "counts": {"total": dict(totals),
                             "arms": {k: dict(v) for k, v in counts.items()}},
                  "baseline_max_score_delta": max_delta, "score_tolerance": TOLERANCE,
                  "pins": pins, "limits": LIMITS, "resources": final,
                  "max_rss_bytes": guard.peak_rss,
                  "max_swap_growth_bytes": guard.peak_swap_growth,
                  "hashes": {"capture.jsonl": sha(output / "capture.jsonl"),
                             "resources.jsonl": sha(output / "resources.jsonl")},
                  "settings": {"max_length": 96, "batch_size": 8, "threads": 1,
                               "interop_threads": 1, "external_context": ""},
                  "labels_opened": False, "generation_inference": False,
                  "sentence_inference": True, "training_run": False,
                  "dev_opened": False, "test_opened": False,
                  "production_changed": False}
        write(output / "report.json", report)
        write(output / "capture.complete.json", {"completed": True, "rows": ROWS,
              "capture_sha256": report["hashes"]["capture.jsonl"],
              "resources_sha256": report["hashes"]["resources.jsonl"],
              "report_sha256": sha(output / "report.json"), "labels_opened": False})
    except Exception as error:
        write(output / "failure.json", {"type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    capture(parser.parse_args().output)
