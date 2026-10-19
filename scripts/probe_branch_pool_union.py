#!/usr/bin/env python3
"""Compare complete branch-pool union against the frozen late ensemble."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.branch_pool_converter import BranchPoolConverter
from hanglish.core.ensemble_converter import EnsembleConverter
from scripts.probe_proposal_union import SOURCE, SOURCE_SHA, select_sources
from scripts.evaluate_ensemble_regression import (
    CHECKPOINT, VERIFICATION, VERIFICATION_SHA, RUNTIME_FILES, artifact_file_pins,
    checked_bytes, file_sha256, merge_pins, percentile, protected_segments_preserved,
    rss_bytes, validate_independent_training, validate_response, verify_pins,
)


class RecordingRanker:
    """Observe original scoring inputs/outputs without changing inference."""
    def __init__(self, ranker):
        self.ranker = ranker
        self.calls = []

    def can_score(self, text):
        return self.ranker.can_score(text)

    def score_pairs(self, contexts, candidates):
        scores = self.ranker.score_pairs(contexts, candidates)
        self.calls.append({"contexts": list(contexts), "candidates": list(candidates),
                           "scores": list(scores)})
        return scores


def check_pools(record):
    original = record["ensemble_calls"]
    shared = record["branch_pool_calls"]
    trace = record["trace"]
    if len(original) != 3 or len(shared) != 1 or trace["fallback_reason"] is not None:
        raise ValueError("pilot expected both branches to use sentence scoring")
    if (original[0]["candidates"] != trace["baseline_pool"]
            or original[1]["candidates"] != trace["supported_pool"]):
        raise ValueError("extracted branch pool differs from original model inputs")
    expected = list(dict.fromkeys(original[0]["candidates"] + original[1]["candidates"]))
    if expected != trace["union_pool"] or expected != shared[0]["candidates"]:
        raise ValueError("complete branch union lost/reordered candidates")
    if any(ctx != record["context"] for call in [*original, *shared] for ctx in call["contexts"]):
        raise ValueError("scoring context differs")


def compare_display(old, new):
    old_candidates, new_candidates = old["candidates"], new["candidates"]
    texts_equal = [c["text"] for c in old_candidates] == [c["text"] for c in new_candidates]
    segments_equal = texts_equal and all(a["segments"] == b["segments"]
                                        for a, b in zip(old_candidates, new_candidates))
    old_by_text = {c["text"]: c for c in old_candidates}
    differences = [abs(c["score"] - old_by_text[c["text"]]["score"])
                   for c in new_candidates if c["text"] in old_by_text
                   and c["score"] is not None and old_by_text[c["text"]]["score"] is not None]
    return {"texts_equal": texts_equal, "segments_equal": segments_equal,
            "full_candidates_equal": old_candidates == new_candidates,
            "max_common_rounded_score_difference": max(differences, default=0.)}


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def guard():
        if time.monotonic() - started > 900 or rss_bytes() > 3 * 1024**3:
            raise RuntimeError("pilot exceeded 900s/3GiB bound")
    try:
        proof = validate_independent_training(CHECKPOINT, VERIFICATION, VERIFICATION_SHA)
        report = proof["training_report"]
        rows = select_sources()
        names = set(RUNTIME_FILES) | {"hanglish/core/branch_pool_converter.py",
            "scripts/probe_branch_pool_union.py", "scripts/probe_proposal_union.py",
            "scripts/probe_supported_phonetic_runtime.py"}
        pins = merge_pins(proof["pins"], {str(SOURCE.relative_to(ROOT)): SOURCE_SHA},
                          {name: file_sha256(ROOT / name) for name in names})
        verify_pins(pins)
        (output / "inputs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        ensemble = EnsembleConverter(CHECKPOINT, checkpoint_sha256=report["checkpoint_sha256"],
                                    metadata_sha256=report["metadata_sha256"], experimental=True)
        branch_pool = BranchPoolConverter(ensemble, experimental=True)
        branch_pool.load_models()
        recorder = RecordingRanker(ensemble.baseline.sentence_ranker)
        ensemble.baseline.sentence_ranker = recorder
        ensemble.supported.sentence_ranker = recorder
        pins = merge_pins(pins, artifact_file_pins(ensemble))
        verify_pins(pins)
        smoke = []
        for text in ("imnida", "testreul sihaeng", "https://example.com testreul sihaeng"):
            response = branch_pool.convert(text)
            validate_response(response, text, "")
            assert protected_segments_preserved(response)
            raw = branch_pool.convert(text, raw=True)
            validate_response(raw, text, "", raw=True)
            smoke.append({"input": text, "response": response, "raw": raw})
        (output / "smoke.json").write_text(json.dumps(smoke, ensure_ascii=False, indent=2))
        captures = []
        with (output / "capture.jsonl").open("x") as stream:
            for i, row in enumerate(rows):
                record = dict(row)
                record["arm_order"] = ["ensemble", "branch_pool"] if i % 2 == 0 else ["branch_pool", "ensemble"]
                for arm in record["arm_order"]:
                    for branch in (ensemble.baseline, ensemble.supported):
                        branch.word_options.cache_clear()
                    recorder.calls.clear()
                    converter = ensemble if arm == "ensemble" else branch_pool
                    record[arm] = deepcopy(converter.convert(row["input"], row["context"]))
                    record[arm + "_calls"] = deepcopy(recorder.calls)
                    if arm == "ensemble":
                        record["baseline"] = deepcopy(ensemble.last_baseline_response)
                        record["supported"] = deepcopy(ensemble.last_supported_response)
                    else:
                        record["trace"] = deepcopy(branch_pool.last_trace)
                    guard()
                check_pools(record)
                record["display_comparison"] = compare_display(record["ensemble"], record["branch_pool"])
                for arm in ("baseline", "supported", "ensemble", "branch_pool"):
                    validate_response(record[arm], row["input"], row["context"])
                    assert protected_segments_preserved(record[arm])
                record["raw"] = branch_pool.convert(row["input"], row["context"], raw=True)
                validate_response(record["raw"], row["input"], row["context"], raw=True)
                captures.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                if (i + 1) % 8 == 0:
                    print(json.dumps({"captured": i + 1, "elapsed_s": time.monotonic() - started}), flush=True)
        capture_sha = file_sha256(output / "capture.jsonl")
        verify_pins(pins)
        targets = {r["id"]: r for r in (json.loads(line) for line in checked_bytes(SOURCE, SOURCE_SHA).splitlines())}
        outcomes = []
        for row in captures:
            target = targets[row["id"]]
            assert row["baseline"]["candidates"] == target["child_candidates"]
            outcome = {"id": row["id"]}
            for arm in ("baseline", "supported", "ensemble", "branch_pool"):
                outcome[arm] = {"top1": row[arm]["result"] in target["expected"],
                                "top5": any(c["text"] in target["expected"] for c in row[arm]["candidates"])}
            outcomes.append(outcome)
        metrics = {arm: {"rows": 64, "top1": sum(r[arm]["top1"] for r in outcomes),
                         "top5": sum(r[arm]["top5"] for r in outcomes)}
                   for arm in ("baseline", "supported", "ensemble", "branch_pool")}
        timings = {arm: {"p50_ms": statistics.median(r[arm]["engine_ms"] for r in captures),
                         "p95_ms": percentile([r[arm]["engine_ms"] for r in captures], .95)}
                   for arm in ("ensemble", "branch_pool")}
        timings["paired_median_ratio"] = statistics.median(r["branch_pool"]["engine_ms"] / r["ensemble"]["engine_ms"] for r in captures)
        timings["phase_p50_ms"] = {phase: statistics.median(r["trace"][phase] for r in captures)
                                   for phase in ("baseline_prepare_ms", "supported_prepare_ms", "sentence_score_ms")}
        parity = {key: sum(r["display_comparison"][key] for r in captures)
                  for key in ("texts_equal", "segments_equal", "full_candidates_equal")}
        parity["maximum_common_score_drift"] = max(r["display_comparison"]["max_common_rounded_score_difference"] for r in captures)
        final = {"schema": "branch-pool-union-pilot-v1", "completed": True,
                 "production_changed": False, "new_generalization": False,
                 "pins": pins, "capture_sha256": capture_sha,
                 "inputs_sha256": file_sha256(output / "inputs.jsonl"),
                 "metrics": metrics, "timings": timings, "display_parity": parity,
                 "all_64_original_branch_pools_preserved": True,
                 "outcomes": outcomes, "elapsed_s": time.monotonic() - started,
                 "peak_rss_bytes": rss_bytes(),
                 "timing_scope": "alternating order; cleared word caches; shared model/lexical caches; recording wrapper in both arms; not web"}
        guard()
        (output / "report.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: final[k] for k in ("completed", "metrics", "timings", "display_parity", "elapsed_s", "peak_rss_bytes")}), flush=True)
    except BaseException as exc:
        (output / "failure.json").write_text(json.dumps({"completed": False, "error": repr(exc), "elapsed_s": time.monotonic() - started}, indent=2))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
