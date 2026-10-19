#!/usr/bin/env python3
"""Post-hoc cached DEV ablation of calibration on the production pool only.

This is exploratory factor isolation after the union DEV result.  It does not
rerun either model, tune weights, search filters, inspect individual errors, or
provide deployment evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".omx/artifacts/phonetic-choice-runtime-dev-v1"
WEIGHTS = ROOT / ".omx/artifacts/phonetic-choice-calibration-training-v1/weights.json"
DEV = ROOT / "data/challenge/sentence-v3/dev.jsonl"
SCHEMA = "phonetic-choice-production-pool-ablation-v1"
EXPECTED = {
    SOURCE / "report.json": "a0072c79116da9acdbff83b373dccbb8c3ef8c26ac5164f575490941808d4473",
    SOURCE / "capture.jsonl": "e0447efa305e1d4f61052866b1ee121a62b400f3adf163dfdeac1b0bb4734bf5",
    SOURCE / "capture.complete.json": "e2f88f3383baf9c8f8cfebf643e02ecdf3e4652b4fc0d3b495004ebebdb6c023",
    SOURCE / "completion.json": "f62ac3ee9b2b8afed211d0c2ccf418af841ce3c041f5eb38f40d578462759bae",
    SOURCE / "independent-verification.json": "f2e2eca118ef05c1397304d89ccea99e1d9ef28841fe109fcd04446d0928e6c4",
    WEIGHTS: "930e3ef942061e6d459d7527c0e073fc6ad5626d59f5170e9b11df8499b39f4b",
}
ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
LITERAL_HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]\Z")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSONL at {path}:{number}") from error
            need(isinstance(value, dict), f"non-object JSONL row at {path}:{number}")
            yield value


def write_json(path, value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def render_public(source, paths):
    need(paths, "ranked production pool is empty")
    candidates = [{"text": text, "score": round(score, 4), "segments": segments}
                  for text, score, segments in paths[:5]]
    if not any(candidate["text"] == source for candidate in candidates):
        candidates = candidates[:4] + [{"text": source, "score": None, "segments": [{
            "input_start": 0, "input_end": len(source), "source": source,
            "text": source, "kind": "raw"}]}]
    return candidates


def derive_row(row, weights):
    source = row["request"]["input"]
    ranked = row["proposed"].get("sentence_ranking_applied") is True
    parent_pool = row["parent_pool"]
    base_paths = row["base_paths"]
    feature_rows = row["proposed"]["phonetic_choice"]["rows"]
    need(row["request"] == {"input": source, "context": ""}
         and isinstance(parent_pool, list) and len(parent_pool) == len(set(parent_pool)),
         "source or production pool identity differs")
    if not ranked:
        need(not parent_pool and not base_paths and not feature_rows,
             "unranked row unexpectedly has a sentence pool")
        return {"id": row["id"], "request": row["request"], "ranked": False,
                "parent": row["parent"], "derived": row["parent"],
                "production_paths": [], "calibrated_paths": []}
    need(len(base_paths) == len(feature_rows), "feature/base path denominator differs")
    parent_set = set(parent_pool)
    selected = [(path, feature) for path, feature in zip(base_paths, feature_rows)
                if path[0] in parent_set]
    production_paths = [path for path, _ in selected]
    need(len(production_paths) == len(parent_pool)
         and len({path[0] for path in production_paths}) == len(parent_pool)
         and set(path[0] for path in production_paths) == parent_set,
         "production pool membership differs")
    need(production_paths == sorted(production_paths, key=lambda path: (-path[1], path[0])),
         "production base order differs")
    need(render_public(source, production_paths) == row["parent"]["candidates"]
         and row["parent"]["result"] == row["parent"]["candidates"][0]["text"],
         "unweighted production public response differs")
    calibrated = []
    for path, feature in selected:
        text, score, segments = path
        values = feature.get("features")
        need(feature.get("text") == text and feature.get("base_score") == score
             and feature.get("segments") == segments
             and isinstance(values, list) and len(values) == len(weights)
             and all(finite(value) for value in values), "saved feature correspondence differs")
        residual = sum(weight * value for weight, value in zip(weights, values))
        adjusted = float(score) + residual
        need(finite(adjusted), "derived production score is nonfinite")
        calibrated.append((text, adjusted, segments))
    calibrated.sort(key=lambda path: (-path[1], path[0]))
    candidates = render_public(source, calibrated)
    # Candidate paths and segments are inherited byte-for-byte; only ordering
    # and numeric scores change under the already-frozen weights.
    derived = {"result": candidates[0]["text"], "candidates": candidates}
    return {"id": row["id"], "request": row["request"], "ranked": True,
            "parent": row["parent"], "derived": derived,
            "production_paths": production_paths, "calibrated_paths": calibrated}


def literal_signature(text):
    return "".join(character for character in text
                   if not ("a" <= character <= "z")
                   and LITERAL_HANGUL.fullmatch(character) is None)


def summarize(rows, labels, branch):
    totals, categories, outcomes = Counter(), defaultdict(Counter), []
    for row, label in zip(rows, labels):
        texts = [candidate["text"] for candidate in row[branch]["candidates"][:5]]
        expected = set(label["expected"])
        top1 = bool(texts) and texts[0] in expected
        top5 = bool(expected & set(texts))
        values = {"rows": 1, "top1": int(top1), "top5": int(top5), "errors": 0}
        totals.update(values); categories[label.get("category", "uncategorized")].update(values)
        outcomes.append((top1, top5))
    return {"counts": dict(totals), "categories": {k: dict(v) for k, v in categories.items()},
            "passed": totals["top1"] == len(rows)}, outcomes


def metrics(rows, labels):
    need(len(rows) == len(labels) == 316
         and [row["id"] for row in rows] == [label.get("id") for label in labels],
         "prediction/label identity differs")
    parent, before = summarize(rows, labels, "parent")
    derived, after = summarize(rows, labels, "derived")
    changes, english = Counter(), Counter()
    for row, label, old, new in zip(rows, labels, before, after):
        need(row["request"] == {"input": label.get("input"), "context": ""}
             and label.get("context", "") == "" and label.get("split") == "dev",
             "late DEV label source differs")
        changes["corrections"] += not old[0] and new[0]
        changes["regressions"] += old[0] and not new[0]
        changes["top5_losses"] += old[1] and not new[1]
        source_signature = literal_signature(row["request"]["input"])
        english["literal_signature_regressions"] += (
            literal_signature(row["parent"]["result"]) == source_signature
            and literal_signature(row["derived"]["result"]) != source_signature)
        signatures = {tuple(ASCII_WORD.findall(text)) for text in label["expected"]}
        if any(signatures):
            english["rows_with_reference_ascii_words"] += 1
            old_match = tuple(ASCII_WORD.findall(row["parent"]["result"])) in signatures
            new_match = tuple(ASCII_WORD.findall(row["derived"]["result"])) in signatures
            english["parent_exact_signature"] += old_match
            english["derived_exact_signature"] += new_match
            english["signature_regressions"] += old_match and not new_match
    gate = bool(changes["corrections"] and not changes["regressions"]
                and not changes["top5_losses"] and not english["signature_regressions"]
                and not english["literal_signature_regressions"])
    return {"parent": parent, "derived": derived,
            "corrections": changes["corrections"], "regressions": changes["regressions"],
            "top5_losses": changes["top5_losses"], "strict_gate": gate,
            "english_ascii_signature": dict(english),
            "english_scope": "Exact case-sensitive ASCII word sequence on references containing ASCII words; not all English-copy behavior."}


def run(output):
    started = time.monotonic(); output = Path(output); output.mkdir(parents=True, exist_ok=False)
    declaration = {
        "schema": SCHEMA + "-analysis-plan", "declared_before_analysis": True,
        "post_hoc_exploratory_dev_analysis": True,
        "purpose": "isolate frozen calibration on the exact original production pool",
        "weights_fixed_without_search": True, "filters_or_error_specific_rules_searched": False,
        "models_loaded_or_run": False, "test_or_document_reserve_accessed": False,
        "deployment_or_fresh_validation_evidence": False,
        "limitation": "Cached likelihood batches include union-only pairs; production-only runtime and latency are not measured.",
    }
    write_json(output / "analysis-plan.json", declaration)
    need(all(path.is_file() and sha(path) == digest for path, digest in EXPECTED.items()),
         "frozen runtime/calibration input changed")
    report, proof = read(SOURCE / "report.json"), read(SOURCE / "independent-verification.json")
    completion, closure = read(SOURCE / "completion.json"), read(SOURCE / "capture.complete.json")
    need(report.get("completed") is True and proof.get("verified") is True
         and proof.get("report_sha256") == sha(SOURCE / "report.json")
         and completion.get("completed") is True
         and closure == {"schema": "phonetic-choice-runtime-dev-v1-closure", "rows": 316,
                         "capture_sha256": sha(SOURCE / "capture.jsonl"), "labels_opened": False},
         "frozen runtime completion/proof differs")
    need(report.get("pins", {}).get(str(DEV.resolve())) == sha(DEV), "DEV source pin differs")
    weights_record = read(WEIGHTS); weights = weights_record.get("weights")
    need(isinstance(weights, list) and len(weights) == 4 and all(finite(v) for v in weights),
         "frozen calibration weights differ")
    derived_rows = []
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for row in jsonl(SOURCE / "capture.jsonl"):
            derived = derive_row(row, weights)
            stream.write(json.dumps(derived, ensure_ascii=False, allow_nan=False,
                                    separators=(",", ":")) + "\n")
            derived_rows.append(derived)
        stream.flush(); os.fsync(stream.fileno())
    need(len(derived_rows) == 316 and sum(not row["ranked"] for row in derived_rows) == 4,
         "derived prediction denominator differs")
    prediction_sha = sha(output / "predictions.jsonl")
    write_json(output / "predictions.complete.json", {
        "schema": SCHEMA + "-closure", "rows": 316,
        "predictions_sha256": prediction_sha, "labels_opened": False})
    # Canonical DEV labels open only after every derived prediction is durable
    # and the source-only closure is hashed.
    labels = list(jsonl(DEV))
    result = metrics(derived_rows, labels)
    need(result["parent"]["counts"] == {"rows": 316, "top1": 262, "top5": 282, "errors": 0},
         "production reference metric differs")
    pins = {str(path.resolve()): digest for path, digest in EXPECTED.items()}
    pins[str(Path(__file__).resolve())] = sha(Path(__file__))
    pins[str(DEV.resolve())] = sha(DEV)
    output_report = {
        "schema": SCHEMA, "completed": True, "split": "dev", **result,
        "rows": 316, "ranked_rows": 312, "unranked_rows": 4,
        "weights": weights, "pins": dict(sorted(pins.items())),
        "predictions_sha256": prediction_sha,
        "closure_sha256": sha(output / "predictions.complete.json"),
        "analysis_plan_sha256": sha(output / "analysis-plan.json"),
        "source_runtime_report_sha256": sha(SOURCE / "report.json"),
        "source_runtime_proof_sha256": sha(SOURCE / "independent-verification.json"),
        "post_hoc_exploratory": True, "production_changed": False,
        "test_or_document_reserve_accessed": False, "models_loaded_or_run": False,
        "labels_opened_after_prediction_closure": True,
        "interpretation": "Separates calibration from candidate-generation addition on cached DEV evidence.",
        "limitations": [
            "This analysis was declared after observing the union DEV result and is not fresh validation.",
            "Cached likelihood scoring included union-only pairs; production-only runtime and latency are not measured.",
            "The fixed weights were trained on FIT candidate pools and are applied unchanged; no weight, filter, or error-specific search was performed.",
            "No TEST, document reserve, web, or production claim follows from this ablation.",
        ],
        "elapsed_s": time.monotonic() - started,
    }
    write_json(output / "report.json", output_report)
    write_json(output / "completion.json", {"completed": True,
        "report_sha256": sha(output / "report.json"),
        "predictions_sha256": prediction_sha,
        "closure_sha256": sha(output / "predictions.complete.json")})
    return output_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, sort_keys=True))
