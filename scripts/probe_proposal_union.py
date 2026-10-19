#!/usr/bin/env python3
"""Source-selected TRAIN pilot of early versus late candidate union."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.ensemble_converter import EnsembleConverter
from hanglish.core.proposal_union_converter import ProposalUnionConverter
from scripts.evaluate_ensemble_regression import (
    CHECKPOINT, VERIFICATION, VERIFICATION_SHA, RUNTIME_FILES, artifact_file_pins,
    checked_bytes, file_sha256, merge_pins, percentile, protected_segments_preserved,
    rss_bytes, validate_independent_training, validate_response, verify_pins,
)

SOURCE = ROOT / ".omx/artifacts/numeric-structure-runtime-v1/cases.jsonl"
SOURCE_SHA = "50550395a81a82635b4f0cf0ff5d1840e28f221d86426716cd6ff70e0bc50557"
ARMS = ("baseline", "supported", "ensemble", "proposal_union")


def select_sources(path=SOURCE, digest=SOURCE_SHA):
    rows = [json.loads(line) for line in checked_bytes(path, digest).splitlines() if line]
    projected = [{key: row[key] for key in ("id", "input", "context")} for row in rows]
    if len(projected) != 512 or len({row["id"] for row in projected}) != 512:
        raise ValueError("expected 512 unique TRAIN inputs")
    return sorted(projected, key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())[:64]


def summarize(captures, targets):
    results = {}
    outcomes = []
    for row in captures:
        gold = set(targets[row["id"]]["expected"])
        outcome = {"id": row["id"]}
        for arm in ARMS:
            candidate_texts = [c["text"] for c in row[arm]["candidates"]]
            outcome[arm] = {"top1": row[arm]["result"] in gold,
                            "top5": bool(gold.intersection(candidate_texts))}
        outcome["sentence_pool_gold"] = bool(gold.intersection(row["trace"]["sentence_pool"]))
        outcomes.append(outcome)
    for arm in ARMS:
        results[arm] = {"rows": len(captures),
                        "top1": sum(row[arm]["top1"] for row in outcomes),
                        "top5": sum(row[arm]["top5"] for row in outcomes)}
    changes = {}
    for parent in ("baseline", "ensemble"):
        changes[parent] = {metric: {
            "fix_ids": [r["id"] for r in outcomes if not r[parent][metric] and r["proposal_union"][metric]],
            "loss_ids": [r["id"] for r in outcomes if r[parent][metric] and not r["proposal_union"][metric]],
        } for metric in ("top1", "top5")}
    timings = {arm: {"p50_ms": statistics.median(row[arm]["engine_ms"] for row in captures),
                     "p95_ms": percentile([row[arm]["engine_ms"] for row in captures], .95)}
               for arm in ("ensemble", "proposal_union")}
    timings["paired_median_ratio"] = statistics.median(
        row["proposal_union"]["engine_ms"] / row["ensemble"]["engine_ms"] for row in captures)
    return {"metrics": results, "changes": changes, "timings": timings, "outcomes": outcomes}


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def guard():
        if time.monotonic() - started > 900 or rss_bytes() > 3 * 1024**3:
            raise RuntimeError("pilot exceeded 900s/3GiB resource bound")

    try:
        proof = validate_independent_training(CHECKPOINT, VERIFICATION, VERIFICATION_SHA)
        training = proof["training_report"]
        rows = select_sources()
        source_names = set(RUNTIME_FILES) | {
            "hanglish/core/proposal_union_converter.py", "scripts/probe_proposal_union.py",
            "scripts/probe_supported_phonetic_runtime.py"}
        pins = merge_pins(proof["pins"], {str(SOURCE.relative_to(ROOT)): SOURCE_SHA},
                          {name: file_sha256(ROOT / name) for name in source_names})
        verify_pins(pins)
        (output / "inputs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        ensemble = EnsembleConverter(CHECKPOINT, checkpoint_sha256=training["checkpoint_sha256"],
                                    metadata_sha256=training["metadata_sha256"], experimental=True)
        early = ProposalUnionConverter(ensemble, experimental=True)
        early.load_models()
        pins = merge_pins(pins, artifact_file_pins(ensemble))
        verify_pins(pins)
        smoke = []
        for text in ("imnida", "testreul sihaeng", "https://example.com testreul sihaeng"):
            response = early.convert(text)
            validate_response(response, text, "")
            assert protected_segments_preserved(response)
            raw = early.convert(text, raw=True)
            validate_response(raw, text, "", raw=True)
            smoke.append({"input": text, "response": response, "raw": raw})
        (output / "smoke.json").write_text(json.dumps(smoke, ensure_ascii=False, indent=2))
        guard()
        captures = []
        with (output / "capture.jsonl").open("x") as stream:
            for index, row in enumerate(rows):
                record = dict(row)
                order = ("ensemble", "proposal_union") if index % 2 == 0 else ("proposal_union", "ensemble")
                record["arm_order"] = order
                for arm in order:
                    for branch in (ensemble.baseline, ensemble.supported, early):
                        branch.word_options.cache_clear()
                    converter = ensemble if arm == "ensemble" else early
                    record[arm] = deepcopy(converter.convert(row["input"], row["context"]))
                    if arm == "ensemble":
                        record["baseline"] = deepcopy(ensemble.last_baseline_response)
                        record["supported"] = deepcopy(ensemble.last_supported_response)
                    else:
                        record["trace"] = deepcopy(early.last_trace)
                    guard()
                for arm in ARMS:
                    validate_response(record[arm], row["input"], row["context"])
                    if not protected_segments_preserved(record[arm]):
                        raise ValueError("protected segment changed")
                raw = early.convert(row["input"], row["context"], raw=True)
                validate_response(raw, row["input"], row["context"], raw=True)
                record["raw"] = raw
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                captures.append(record)
                if (index + 1) % 8 == 0:
                    print(json.dumps({"captured": index + 1, "elapsed_s": time.monotonic() - started}), flush=True)
        capture_sha = file_sha256(output / "capture.jsonl")
        verify_pins(pins)
        # Targets are joined only after the complete prediction capture is closed.
        targets = {r["id"]: r for r in (json.loads(line) for line in checked_bytes(SOURCE, SOURCE_SHA).splitlines())}
        for row in captures:
            if row["baseline"]["candidates"] != targets[row["id"]]["child_candidates"]:
                raise ValueError("historical baseline full candidates changed")
        report = {"schema": "proposal-union-pilot-v1", "completed": True,
                  "production_changed": False, "new_generalization": False,
                  "selection": "first 64 SHA256(id) from exposed natural TRAIN512",
                  "pins": pins, "capture_sha256": capture_sha,
                  "inputs_sha256": file_sha256(output / "inputs.jsonl"),
                  "elapsed_s": time.monotonic() - started, "peak_rss_bytes": rss_bytes(),
                  "timing_scope": "shared model/lexical caches; word caches cleared each arm; alternating arm order; no web timing",
                  **summarize(captures, targets)}
        guard()
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in ("completed", "metrics", "timings", "elapsed_s", "peak_rss_bytes")}), flush=True)
    except BaseException as exc:
        (output / "failure.json").write_text(json.dumps({"completed": False, "error": repr(exc),
            "elapsed_s": time.monotonic() - started}, indent=2))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
