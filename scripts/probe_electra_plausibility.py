"""Capture frozen ELECTRA RTD features, then cross-fit a TRAIN-only score blend."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE = ROOT / ".omx/artifacts/semantic-pair-training-input-v2"
ROWS_SHA = "6a1f7f61a7f7ffeb9209772559412931137d71ef0173d7429264d9cb53aed396"
MANIFEST_SHA = "3ab561d1a767e7a65b4706c2cd0f668366693536849e17f2f3e620fbcf5550a6"
LIMIT_SECONDS = 900
LIMIT_RSS = 2 * 1024**3


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"changed source: {path}")
    return raw


def rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def guard(started):
    if time.monotonic() - started > LIMIT_SECONDS or rss_bytes() > LIMIT_RSS:
        raise RuntimeError("ELECTRA diagnostic exceeded its time or memory limit")


def load_sources():
    manifest = json.loads(checked(SOURCE / "manifest.json", MANIFEST_SHA))
    rows = [json.loads(line) for line in checked(SOURCE / "rows.jsonl", ROWS_SHA).splitlines()]
    if (manifest.get("labels_or_expected_saved") is not False
            or Counter(row["source_role"] for row in rows)
            != {"natural_fit": 2048, "challenge_fit": 1444}):
        raise ValueError("unexpected source-only TRAIN partition")
    if any("expected" in row for row in rows):
        raise ValueError("source-only rows contain targets")
    return manifest, rows


def join_targets(captures, manifest):
    targets = {}
    for role in ("natural_fit", "challenge_fit"):
        relative = f"data/natural_context/final-choice-v1/{role}/cases.jsonl"
        for line in checked(ROOT / relative, manifest["source_hashes"][relative]).splitlines():
            row = json.loads(line)
            targets[(role, row["id"])] = row
    if len(targets) != len(captures):
        raise ValueError("target denominator differs")
    joined = []
    for row in captures:
        target = targets[(row["source_role"], row["id"])]
        if any(row[key] != target[key] for key in ("input", "context", "group_id")):
            raise ValueError("target identity differs")
        joined.append({**row, "expected": target["expected"]})
    return joined


def feature_metrics(rows, feature):
    roles = {}
    for role in ("natural_fit", "challenge_fit"):
        subset = [row for row in rows if row["source_role"] == role]
        counts = Counter(rows=len(subset))
        corrections, regressions = [], []
        for row in subset:
            scores = row["plausibility"]
            choice = (min(range(len(scores)), key=lambda i: (
                -scores[i][feature], row["candidate_texts"][i])) if scores is not None else None)
            result = row["current_result"] if choice is None else row["candidate_texts"][choice]
            before, after = row["current_result"] in row["expected"], result in row["expected"]
            counts.update(baseline_correct=int(before), correct=int(after),
                          scored=int(scores is not None))
            if not before and after:
                corrections.append(row["id"])
            if before and not after:
                regressions.append(row["id"])
        roles[role] = {**counts, "correction_ids": corrections, "regression_ids": regressions}
    return roles


def verify_pins(pins):
    for path, expected in pins.items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"source changed during execution: {path}")


def run(output):
    from hanglish.models.electra_plausibility import ElectraPlausibility
    from scripts.fit_electra_score_calibration import evaluate_calibration
    import torch

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        manifest, sources = load_sources()
        pins = {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), ROOT / "hanglish/models/electra_plausibility.py",
            ROOT / "scripts/fit_electra_score_calibration.py",
            SOURCE / "rows.jsonl", SOURCE / "manifest.json")}
        model = ElectraPlausibility.load()
        guard(started)
        pins.update(model.source_hashes)
        verify_pins(pins)
        guard(started)
        initialization_seconds = time.monotonic() - started
        captures = []
        with (output / "capture.jsonl").open("x") as stream:
            for index, source in enumerate(sources):
                objects = [item["object"] for item in source["finite_candidates"]]
                texts = [item["text"] for item in objects]
                tick = time.perf_counter()
                scores = model.score_candidates(source["context"], texts)
                if scores is not None and len(scores) != len(texts):
                    raise ValueError("feature and candidate count mismatch")
                row = {key: source[key] for key in (
                    "id", "group_id", "component", "fold", "source_role", "input",
                    "context", "current_result")}
                row.update(index=index, candidate_texts=texts,
                           baseline_scores=[item["score"] for item in objects],
                           plausibility=scores, score_ms=(time.perf_counter() - tick) * 1000)
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                        allow_nan=False) + "\n")
                stream.flush()
                captures.append(row)
                guard(started)
                if (index + 1) % 128 == 0:
                    print(f"ELECTRA feature capture {index + 1}/{len(sources)}", flush=True)
        capture_sha = digest(output / "capture.jsonl")
        verify_pins(pins)
        # No expected-answer file is opened until all input-only features are closed.
        rows = join_targets(captures, manifest)
        calibration = evaluate_calibration(rows)
        guard(started)
        report = {
            "schema": "electra-plausibility-train-probe-v1", "completed": True,
            "scope": "natural_fit2048 + challenge_fit1444; cross-fitted diagnostic, not generalization",
            "rows": len(rows), "scored_rows": sum(row["plausibility"] is not None for row in rows),
            "capture_sha256": capture_sha,
            "standalone_mean": feature_metrics(rows, "mean_original_logprob"),
            "standalone_changed": feature_metrics(rows, "changed_original_logprob"),
            "calibration": calibration, "source_hashes": pins,
            "label_source_hashes": {path: value for path, value in manifest["source_hashes"].items()
                                    if path.endswith("/cases.jsonl")},
            "labels_joined_after_closed_capture": True, "neural_weights_trained": False,
            "linear_calibration_trained_out_of_fold": True,
            "development_or_final_data_read": False, "production_changed": False,
            "initialization_seconds": initialization_seconds,
            "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": rss_bytes(),
            "score_ms_p50": statistics.median(row["score_ms"] for row in rows),
            "score_is_normalized_language_likelihood": False,
        }
        verify_pins(pins)
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False,
                                                       indent=2, allow_nan=False) + "\n")
        guard(started)
        return report
    except Exception as error:
        (output / "report.json").unlink(missing_ok=True)
        (output / "failure.json").write_text(json.dumps({
            "error_type": type(error).__name__, "error": str(error),
            "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": rss_bytes(),
        }, ensure_ascii=False, indent=2) + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps({key: result[key] for key in (
        "completed", "rows", "scored_rows", "elapsed_seconds", "peak_rss_bytes")}), flush=True)
