#!/usr/bin/env python3
"""Evaluate the experimental ensemble on exposed v2.2 regression suites."""
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

from hanglish.core.neural_converter import PROTECTED, file_sha256  # noqa: E402
from scripts.probe_supported_phonetic_runtime import (  # noqa: E402
    validate_independent_training,
)


CORPUS = ROOT / "data/challenge/v2.2"
MANIFEST_SHA = "76e5b3aa11ff047cbbb1a7aa1cb6067d7caf9f47cb72c928abf61b3bdcb0471e"
SUITES = {
    "dev": {"file": "dev.jsonl", "sha256": "bc130d28ca56a15f0e1531d94af21c0b29c1981d8d961348699c5c95984f4f38", "rows": 140},
    "test": {"file": "test.jsonl", "sha256": "3d2fe696cb62e3e895017f817e8f5ccca8a651f17a325725ef624b6d3335c720", "rows": 1000},
}
HISTORICAL_BASELINE = {
    "dev": {"rows": 140, "top1": 140, "top5": 140},
    "test": {"rows": 1000, "top1": 988, "top5": 1000},
}
CHECKPOINT = ROOT / ".omx/artifacts/supported-phonetic-training-v1/phonetic_transducer.pt"
VERIFICATION = ROOT / ".omx/artifacts/supported-phonetic-training-v1-independent-v2.json"
VERIFICATION_SHA = "c0c2cdb8c5a19abd4b53585536f8eac52e11df7a401ed545841461014a2f802b"
TIME_LIMIT = 1200
RSS_LIMIT = 3 * 1024**3
RUNTIME_FILES = (
    "hanglish/core/lattice.py", "hanglish/core/lexical_data.py",
    "hanglish/core/neural_converter.py", "hanglish/core/numeric_structure.py",
    "hanglish/core/phonology.py", "hanglish/core/ensemble_converter.py",
    "hanglish/models/phonetic_transducer.py", "hanglish/models/semantic_ranker.py",
    "hanglish/models/sentence_ranker.py", "scripts/evaluate_ensemble_regression.py",
)


def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def path_key(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def checked_bytes(path, expected):
    raw = Path(path).read_bytes()
    if sha256_bytes(raw) != expected:
        raise ValueError(f"pinned corpus differs: {path}")
    return raw


def parse_jsonl(raw):
    return [json.loads(line) for line in raw.splitlines() if line]


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def check_resources(started, clock=time.monotonic, rss_reader=rss_bytes):
    elapsed, peak = clock() - started, rss_reader()
    if elapsed > TIME_LIMIT:
        raise RuntimeError("ensemble regression exceeded 1200 seconds")
    if peak > RSS_LIMIT:
        raise MemoryError("ensemble regression exceeded 3 GiB RSS")
    return elapsed, peak


def validate_manifest(corpus_dir=CORPUS, manifest_sha=MANIFEST_SHA, suites=SUITES):
    corpus_dir = Path(corpus_dir)
    raw = checked_bytes(corpus_dir / "manifest.json", manifest_sha)
    manifest = json.loads(raw)
    if (manifest.get("schema_version") != 2 or manifest.get("corpus_version") != "2.2"
            or {name: manifest.get("counts", {}).get(name) for name in suites}
            != {name: spec["rows"] for name, spec in suites.items()}):
        raise ValueError("v2.2 challenge manifest contract differs")
    for spec in suites.values():
        if manifest.get("sha256", {}).get(spec["file"]) != spec["sha256"]:
            raise ValueError("challenge suite digest differs from manifest")
    return manifest, raw


def load_source_rows(corpus_dir=CORPUS, manifest_sha=MANIFEST_SHA, suites=SUITES):
    """Project source metadata only; input JSON parsing necessarily sees all bytes."""
    manifest, manifest_raw = validate_manifest(corpus_dir, manifest_sha, suites)
    projected, pins = [], {path_key(Path(corpus_dir) / "manifest.json"): sha256_bytes(manifest_raw)}
    for suite, spec in suites.items():
        path = Path(corpus_dir) / spec["file"]
        raw = checked_bytes(path, spec["sha256"])
        rows = parse_jsonl(raw)
        if len(rows) != spec["rows"]:
            raise ValueError("challenge suite row count differs")
        expected_categories = manifest.get("category_counts", {}).get(suite)
        category_counts = {}
        for row in rows:
            if (row.get("split") != suite or not all(
                    isinstance(row.get(key), str) for key in ("id", "input", "context", "category"))):
                raise ValueError("challenge row is malformed")
            category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
            projected.append({key: row[key] for key in ("id", "input", "context", "category")}
                             | {"suite": suite})
        if category_counts != expected_categories:
            raise ValueError("challenge category counts differ")
        pins[path_key(path)] = spec["sha256"]
        del rows
    if len({row["id"] for row in projected}) != len(projected):
        raise ValueError("challenge IDs are not globally unique")
    return projected, dict(sorted(pins.items()))


def load_targets(corpus_dir=CORPUS, suites=SUITES):
    targets = {}
    for suite, spec in suites.items():
        raw = checked_bytes(Path(corpus_dir) / spec["file"], spec["sha256"])
        for row in parse_jsonl(raw):
            if (row.get("split") != suite or not isinstance(row.get("id"), str)
                    or not isinstance(row.get("expected"), list) or not row["expected"]
                    or any(not isinstance(value, str) or not value for value in row["expected"])):
                raise ValueError("challenge target row is malformed")
            targets[(suite, row["id"])] = list(row["expected"])
    if len(targets) != sum(spec["rows"] for spec in suites.values()):
        raise ValueError("target join denominator differs")
    return targets


def validate_response(response, source, context, *, raw=False):
    if (not isinstance(response, dict) or response.get("input") != source
            or response.get("context") != context or response.get("result") is None
            or type(response.get("engine_ms")) not in (int, float)
            or not math.isfinite(response["engine_ms"]) or response["engine_ms"] < 0
            or response.get("raw") is not raw):
        raise ValueError("converter response metadata differs")
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 5:
        raise ValueError("converter candidate count differs")
    texts = []
    for candidate in candidates:
        if not isinstance(candidate.get("text"), str) or not isinstance(candidate.get("segments"), list):
            raise ValueError("candidate object is malformed")
        rebuilt, cursor = "", 0
        for segment in candidate["segments"]:
            start, end = segment.get("input_start"), segment.get("input_end")
            if (type(start) is not int or type(end) is not int or start != cursor
                    or not 0 <= start <= end <= len(source)
                    or (start == end and source)
                    or segment.get("source") != source[start:end]
                    or not isinstance(segment.get("text"), str)
                    or not isinstance(segment.get("kind"), str)):
                raise ValueError("candidate segment is malformed")
            rebuilt += segment["text"]
            cursor = end
        score = candidate.get("score")
        if (cursor != len(source) or rebuilt != candidate["text"]
                or (score is not None and (type(score) not in (int, float)
                                           or not math.isfinite(score)))):
            raise ValueError("candidate segments do not reconstruct text")
        if score is None and (candidate["text"] != source
                              or any(segment["kind"] != "raw"
                                     or segment["text"] != segment["source"]
                                     for segment in candidate["segments"])):
            raise ValueError("None score is reserved for an original raw candidate")
        texts.append(candidate["text"])
    if len(texts) != len(set(texts)) or response["result"] != texts[0]:
        raise ValueError("candidate ordering or uniqueness differs")
    if raw and (len(candidates) != 1 or texts[0] != source
                or candidates[0]["segments"] != [{
                    "input_start": 0, "input_end": len(source), "source": source,
                    "text": source, "kind": "raw",
                }]):
        raise ValueError("raw conversion did not preserve the source")
    return response


def protected_segments_preserved(response):
    for candidate in response["candidates"]:
        for segment in candidate["segments"]:
            if (segment.get("kind") in {"raw", "english"}
                    or PROTECTED.fullmatch(segment["source"])) \
                    and segment["text"] != segment["source"]:
                return False
    return True


def percentile(values, quantile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil(quantile * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def capture_rows(converter, rows, capture_path, *, guard=lambda: None, timer=time.perf_counter):
    captured = []
    with Path(capture_path).open("x") as stream:
        for index, row in enumerate(rows):
            started = timer()
            ensemble = deepcopy(converter.convert(row["input"], row["context"]))
            whole_ms = (timer() - started) * 1000
            baseline = deepcopy(converter.last_baseline_response)
            supported = deepcopy(converter.last_supported_response)
            for response in (baseline, supported, ensemble):
                validate_response(response, row["input"], row["context"])
                if not protected_segments_preserved(response):
                    raise ValueError("protected/English/raw segment changed")
            raw = deepcopy(converter.convert(row["input"], row["context"], raw=True))
            validate_response(raw, row["input"], row["context"], raw=True)
            if converter.last_supported_response is not None:
                raise ValueError("raw conversion unexpectedly invoked supported model")
            record = {**row, "index": index, "baseline": baseline, "supported": supported,
                      "ensemble": ensemble, "raw": raw, "whole_ensemble_ms": whole_ms}
            captured.append(record)
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")) + "\n")
            stream.flush()
            guard()
            if (index + 1) % 64 == 0 or index + 1 == len(rows):
                print(f"ensemble regression captured {index + 1}/{len(rows)}", flush=True)
    return captured


def arm_metrics(rows, arm):
    result = {"rows": len(rows), "top1": 0, "top5": 0}
    for row in rows:
        expected, response = row["expected"], row[arm]
        texts = [candidate["text"] for candidate in response["candidates"]]
        result["top1"] += response["result"] in expected
        result["top5"] += any(text in expected for text in texts)
    return result


def score_captures(captures, targets):
    rows = []
    for captured in captures:
        key = (captured["suite"], captured["id"])
        if key not in targets:
            raise ValueError("capture has no matching target")
        rows.append({**captured, "expected": targets[key]})
    if len(rows) != len(targets):
        raise ValueError("capture and target sets differ")
    suites = {}
    for suite in sorted({row["suite"] for row in rows}):
        suite_rows = [row for row in rows if row["suite"] == suite]
        arms = {arm: arm_metrics(suite_rows, arm) for arm in ("baseline", "supported", "ensemble")}
        categories = {}
        for category in sorted({row["category"] for row in suite_rows}):
            subset = [row for row in suite_rows if row["category"] == category]
            categories[category] = {arm: arm_metrics(subset, arm)
                                    for arm in ("baseline", "supported", "ensemble")}
        baseline = {row["id"]: row["baseline"] for row in suite_rows}
        changes = {}
        for arm in ("supported", "ensemble"):
            correction, regression, top5_gain, top5_regression = [], [], [], []
            for row in suite_rows:
                expected = row["expected"]
                before = baseline[row["id"]]
                after = row[arm]
                before_top1, after_top1 = before["result"] in expected, after["result"] in expected
                before_top5 = any(c["text"] in expected for c in before["candidates"])
                after_top5 = any(c["text"] in expected for c in after["candidates"])
                if not before_top1 and after_top1: correction.append(row["id"])
                if before_top1 and not after_top1: regression.append(row["id"])
                if not before_top5 and after_top5: top5_gain.append(row["id"])
                if before_top5 and not after_top5: top5_regression.append(row["id"])
            changes[arm] = {"correction_ids": correction, "regression_ids": regression,
                            "top5_gain_ids": top5_gain,
                            "top5_regression_ids": top5_regression}
        suites[suite] = {"arms": arms, "categories": categories, "versus_baseline": changes,
                         "historical_reference": HISTORICAL_BASELINE.get(suite),
                         "baseline_matches_historical_reference":
                             arms["baseline"] == HISTORICAL_BASELINE.get(suite)}
    outcomes = []
    for row in rows:
        outcome = {"suite": row["suite"], "id": row["id"], "category": row["category"]}
        for arm in ("baseline", "supported", "ensemble"):
            texts = [candidate["text"] for candidate in row[arm]["candidates"]]
            outcome[f"{arm}_result"] = row[arm]["result"]
            outcome[f"{arm}_top1"] = row[arm]["result"] in row["expected"]
            outcome[f"{arm}_top5"] = any(text in row["expected"] for text in texts)
        outcomes.append(outcome)
    return suites, outcomes


def runtime_identities(captures):
    identities = {}
    for arm in ("baseline", "supported", "ensemble"):
        observed = {(row[arm].get("version"), json.dumps(row[arm].get("model_artifacts", {}),
                                                         sort_keys=True)) for row in captures}
        if len(observed) != 1:
            raise ValueError(f"{arm} runtime identity changed during capture")
        version, artifacts = observed.pop()
        identities[arm] = {"version": version, "model_artifacts": json.loads(artifacts)}
    return identities


def timing_summary(captures):
    fields = {"baseline": [], "supported": [], "ensemble_response": [], "whole_ensemble": []}
    for row in captures:
        fields["baseline"].append(row["baseline"]["engine_ms"])
        fields["supported"].append(row["supported"]["engine_ms"])
        fields["ensemble_response"].append(row["ensemble"]["engine_ms"])
        fields["whole_ensemble"].append(row["whole_ensemble_ms"])
    return {name: {"first_ms": values[0], "warm_p50_ms": statistics.median(values[1:]),
                   "warm_p95_ms": percentile(values[1:], .95)}
            for name, values in fields.items()}


def verify_pins(pins):
    for name, expected in pins.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"evaluation source changed: {name}")


def merge_pins(*mappings):
    result = {}
    for mapping in mappings:
        for name, digest in mapping.items():
            if name in result and result[name] != digest:
                raise ValueError(f"conflicting evaluation pin: {name}")
            result[name] = digest
    return dict(sorted(result.items()))


def artifact_file_pins(converter):
    pins = {}
    for branch in (converter.baseline, converter.supported):
        for name, digest in branch.model_artifacts.items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            if not path.is_file():
                continue
            key = path_key(path)
            if key in pins and pins[key] != digest:
                raise ValueError(f"branch model artifact conflict: {key}")
            if file_sha256(path) != digest:
                raise ValueError(f"branch model artifact differs: {key}")
            pins[key] = digest
    return dict(sorted(pins.items()))


def default_converter_factory(**kwargs):
    from hanglish.core.ensemble_converter import EnsembleConverter
    return EnsembleConverter(**kwargs)


def run(output, checkpoint=CHECKPOINT, verification=VERIFICATION,
        verification_sha=VERIFICATION_SHA, *, corpus_dir=CORPUS,
        manifest_sha=MANIFEST_SHA, suites=SUITES, converter_factory=default_converter_factory,
        proof_validator=validate_independent_training, target_loader=load_targets,
        clock=time.monotonic, timer=time.perf_counter, rss_reader=rss_bytes):
    output, checkpoint, verification = Path(output), Path(checkpoint), Path(verification)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started, inference_started = clock(), False
    try:
        proof = proof_validator(checkpoint, verification, verification_sha)
        report = proof["training_report"]
        rows, corpus_pins = load_source_rows(corpus_dir, manifest_sha, suites)
        code_pins = {name: file_sha256(ROOT / name) for name in RUNTIME_FILES}
        pins = merge_pins(proof.get("pins", {}), corpus_pins, code_pins)
        verify_pins(pins)
        import torch
        torch.set_num_threads(2)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            if torch.get_num_interop_threads() != 1:
                raise
        initialization_started = timer()
        converter = converter_factory(
            checkpoint=checkpoint, checkpoint_sha256=report["checkpoint_sha256"],
            metadata_sha256=report["metadata_sha256"], experimental=True,
            context_ranking=True, external_lexicon=True, ranker_kind="auto",
        )
        converter.load_models()
        initialization_ms = (timer() - initialization_started) * 1000
        model_files_before = artifact_file_pins(converter)
        pins = merge_pins(pins, model_files_before)
        verify_pins(pins)
        check_resources(started, clock, rss_reader)
        inference_started = True
        capture_path = output / "capture.jsonl"
        captures = capture_rows(
            converter, rows, capture_path,
            guard=lambda: check_resources(started, clock, rss_reader), timer=timer,
        )
        capture_sha = file_sha256(capture_path)
        # Files were parsed for source projection, but expected values are only
        # joined and scored after the complete capture has been closed and hashed.
        targets = target_loader(corpus_dir, suites)
        metrics, outcomes = score_captures(captures, targets)
        del targets
        model_files_after = artifact_file_pins(converter)
        if model_files_after != model_files_before:
            raise ValueError("branch model artifacts changed during evaluation")
        verify_pins(pins)
        elapsed, peak = check_resources(started, clock, rss_reader)
        report_value = {
            "schema": "ensemble-regression-evaluation-v1", "completed": True,
            "scope": "historical exposed v2.2 dev140 and test1000 regression; not fresh holdout",
            "rows": len(captures), "capture_sha256": capture_sha,
            "suites": metrics,
            "timings": {"initialization_ms": initialization_ms,
                        **timing_summary(captures)},
            "source_hashes": pins, "source_hashes_after": pins,
            "model_file_hashes_before": model_files_before,
            "model_file_hashes_after": model_files_after,
            "runtime_identities": runtime_identities(captures),
            "model_identity": {"checkpoint_sha256": report["checkpoint_sha256"],
                               "metadata_sha256": report["metadata_sha256"],
                               "independent_verification_sha256": verification_sha},
            "labels": {"json_parsing_read_expected_bytes_during_source_projection": True,
                       "labels_passed_to_converter": False,
                       "expected_join_and_scoring_after_capture_closed": True},
            "raw_preserved_all": all(row["raw"]["result"] == row["input"] for row in captures),
            "protected_segments_preserved_all": all(
                protected_segments_preserved(row[arm]) for row in captures
                for arm in ("baseline", "supported", "ensemble")),
            "training_run": False, "production_changed": False,
            "development_data_read": True, "final_data_read": True,
            "generalization_claim_allowed": False,
            "elapsed_seconds": elapsed, "peak_rss_bytes": peak,
            "limits": {"seconds": TIME_LIMIT, "rss_bytes": RSS_LIMIT},
            "limitations": ["Both suites are historical exposed regression data.",
                            "Latency arms share one ensemble call and are diagnostic only.",
                            "Quality differences are reported without an automatic deployment gate."],
        }
        outcomes_path = output / "outcomes.jsonl"
        outcomes_path.write_bytes(b"".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")).encode() + b"\n" for row in outcomes
        ))
        report_value["outcomes_sha256"] = file_sha256(outcomes_path)
        report_path = output / "report.json"
        report_path.write_text(json.dumps(report_value, ensure_ascii=False, sort_keys=True,
                                          indent=2, allow_nan=False) + "\n")
        check_resources(started, clock, rss_reader)
        (output / "status.json").write_text(json.dumps(
            {"state": "complete", "report_sha256": file_sha256(report_path)},
            sort_keys=True) + "\n")
        return report_value
    except Exception as error:
        (output / "report.json").unlink(missing_ok=True)
        (output / "status.json").unlink(missing_ok=True)
        (output / "failure.json").write_text(json.dumps({
            "schema": "ensemble-regression-evaluation-failure-v1",
            "error_type": type(error).__name__, "error": str(error),
            "actual_model_inference_run": inference_started,
            "elapsed_seconds": clock() - started, "peak_rss_bytes": rss_reader(),
        }, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--verification", type=Path, default=VERIFICATION)
    parser.add_argument("--verification-sha", default=VERIFICATION_SHA)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    args = parser.parse_args(argv)
    value = run(args.output, args.checkpoint, args.verification, args.verification_sha,
                corpus_dir=args.corpus)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
