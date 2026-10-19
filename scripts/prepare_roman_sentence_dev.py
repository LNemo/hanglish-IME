"""Prepare fixed dev pools from the completed source-only sentence stage audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.sentence_data import DEFAULT_CORPUS, read_split


AUDIT = ROOT / ".omx/artifacts/sentence-only-stage-audit-v1"
BASELINE = ROOT / ".omx/artifacts/sentence-only-dev-baseline-v1"
OUTPUT = ROOT / ".omx/artifacts/roman-sentence-dev-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def _validate_candidate_pool(record):
    candidates = record.get("final_pre_display_pool")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("audit row has no pre-display candidates")
    texts = []
    scores = []
    for candidate in candidates:
        if (not isinstance(candidate, dict)
                or not isinstance(candidate.get("text"), str)
                or not candidate["text"]
                or isinstance(candidate.get("score"), bool)
                or not isinstance(candidate.get("score"), (int, float))
                or not math.isfinite(candidate["score"])):
            raise ValueError("audit row has a missing or nonfinite candidate score")
        texts.append(candidate["text"])
        scores.append(candidate["score"])
    if len(set(texts)) != len(texts):
        raise ValueError("audit pre-display pool contains duplicate candidates")
    displayed = record.get("displayed_candidates")
    if (not isinstance(displayed, list) or not displayed
            or any(not isinstance(item, dict)
                   or not isinstance(item.get("text"), str)
                   or not item["text"] for item in displayed)):
        raise ValueError("audit row has invalid displayed candidates")
    rankers = record.get("ranker_called")
    if (not isinstance(rankers, dict)
            or not isinstance(rankers.get("sentence"), bool)):
        raise ValueError("audit row has invalid sentence-ranker evidence")
    return texts, scores, [item["text"] for item in displayed]


def validate_inputs(audit, baseline, corpus):
    """Validate all source-only evidence before any expected label is loaded."""
    audit = Path(audit)
    baseline = Path(baseline)
    corpus = Path(corpus)
    audit_report_path = audit / "report.json"
    audit_capture_path = audit / "capture.jsonl"
    baseline_report_path = baseline / "report.json"
    baseline_capture_path = baseline / "capture.jsonl"
    audit_report = json.loads(audit_report_path.read_text())
    baseline_report = json.loads(baseline_report_path.read_text())
    data_path = corpus / "dev.jsonl"
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    if (audit_report.get("completed") is not True
            or audit_report.get("schema") != "sentence-only-stage-audit-v1"
            or audit_report.get("input_contract") != "roman_sentence_only"
            or audit_report.get("external_context_allowed") is not False
            or audit_report.get("labels_joined_after_all_captures") is not True):
        raise ValueError("sentence-only stage audit is incomplete or invalid")
    if audit_report.get("capture_sha256") != sha256(audit_capture_path):
        raise ValueError("sentence-only stage audit capture hash differs from report")
    if (audit_report.get("source_sha256") != sha256(data_path)
            or audit_report.get("manifest_sha256") != sha256(manifest_path)
            or audit_report.get("source_sha256") != baseline_report.get("source_sha256")
            or audit_report.get("manifest_sha256") != baseline_report.get("manifest_sha256")):
        raise ValueError("audit, baseline, and sentence-v3 source hashes differ")
    if (baseline_report.get("capture_sha256") != sha256(baseline_capture_path)
            or audit_report.get("baseline_capture_sha256")
            != baseline_report.get("capture_sha256")):
        raise ValueError("frozen baseline capture hash differs from audit evidence")
    model_artifacts = audit_report.get("model_artifacts")
    if not isinstance(model_artifacts, dict) or model_artifacts != baseline_report.get("model_artifacts"):
        raise ValueError("audit model artifacts differ from frozen baseline")
    for relative, expected in model_artifacts.items():
        path = Path(relative)
        path = path if path.is_absolute() else ROOT / path
        if sha256(path) != expected:
            raise ValueError("current model artifact differs from audit and baseline")

    captures = read_jsonl(audit_capture_path)
    baselines = read_jsonl(baseline_capture_path)
    expected_rows = audit_report.get("counts", {}).get("rows")
    if (not isinstance(expected_rows, int) or expected_rows <= 0
            or expected_rows != manifest.get("counts", {}).get("dev")
            or len(captures) != expected_rows or len(baselines) != expected_rows):
        raise ValueError("audit/baseline capture count differs from declared denominator")
    prepared_source = []
    seen = set()
    for capture, frozen in zip(captures, baselines):
        request = frozen.get("request", {})
        response = frozen.get("response", {})
        case_id = capture.get("id")
        if (not isinstance(case_id, str) or case_id in seen
                or case_id != frozen.get("id")):
            raise ValueError("audit and baseline IDs/order differ")
        seen.add(case_id)
        if (capture.get("source_only") is not True
                or capture.get("target_fields_read") is not False
                or capture.get("source") != request.get("input")
                or request.get("context") != ""
                or request.get("engine") != "neural"
                or request.get("mode") != "batch"
                or request.get("raw") is not False):
            raise ValueError("audit source differs from frozen source-only request")
        if (capture.get("version") != response.get("version")
                or capture.get("displayed_candidates") != response.get("candidates")
                or capture.get("result") != response.get("result")):
            raise ValueError("audit runtime output differs from frozen baseline")
        texts, scores, displayed = _validate_candidate_pool(capture)
        prepared_source.append({
            "id": case_id, "input": capture["source"],
            "candidates": texts, "teacher_scores": scores,
            "result": capture["result"],
            "sentence_ranked": capture["ranker_called"]["sentence"],
            "displayed": displayed,
        })
    return prepared_source, {
        "audit_report_sha256": sha256(audit_report_path),
        "audit_capture_sha256": sha256(audit_capture_path),
        "baseline_report_sha256": sha256(baseline_report_path),
        "baseline_capture_sha256": sha256(baseline_capture_path),
        "source_sha256": sha256(data_path),
        "manifest_sha256": sha256(manifest_path),
    }


def run(output=OUTPUT, audit=AUDIT, baseline=BASELINE, corpus=DEFAULT_CORPUS):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    source_rows, input_hashes = validate_inputs(audit, baseline, corpus)

    # Expected sentences enter only after every captured prediction and score
    # has passed source, hash, model, and frozen-runtime parity validation.
    labels = read_split(corpus, "dev")
    if len(labels) != len(source_rows):
        raise ValueError("development labels differ from the captured denominator")
    prepared = []
    for source, label in zip(source_rows, labels):
        if source["id"] != label.get("id"):
            raise ValueError("development label IDs/order differ from captures")
        if source["input"] != label.get("input"):
            raise ValueError("development source differs from captured input")
        prepared.append({
            **source,
            "expected": label["expected"],
            "category": label["category"],
        })

    output.mkdir(parents=True)
    dev_path = output / "dev.jsonl"
    dev_path.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in prepared
    ))
    dev_digest = sha256(dev_path)
    report = {
        "completed": True,
        "schema": "roman-sentence-fixed-dev-v1",
        "input_contract": "roman_sentence_and_generated_candidate",
        "split": "dev",
        "rows": len(prepared),
        "labels_joined_after_capture_validation": True,
        "dev_sha256": dev_digest,
        "sha256": {"dev.jsonl": dev_digest},
        "input_hashes": input_hashes,
        "inference_run": False,
        "test_accessed": False,
        "production_changed": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    report = run(args.output, args.audit, args.baseline, args.corpus)
    print(json.dumps({"completed": report["completed"], "rows": report["rows"],
                      "dev_sha256": report["dev_sha256"]}))


if __name__ == "__main__":
    main()
