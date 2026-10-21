#!/usr/bin/env python3
"""Compare the current and RR-retrained phonetic models in the full engine.

Each arm runs in its own spawned process so its model memory is released before
the next arm starts.  Expected answers are joined only after both source-only
captures have closed.  The exposed DEV and TEST suites are regression screens,
not blind estimates of generalization.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.source_span_resources import ResourceExceeded  # noqa: E402
BASELINE_CHECKPOINT = (
    ROOT / ".omx/artifacts/observed-phonetic-augmented-training-v1/phonetic_transducer.pt"
)
SCORE_SCALE_CHECKPOINT = (
    ROOT / ".omx/artifacts/phonetic-score-scale-training-v1/model.json"
)
CORPUS = ROOT / "data/challenge/v2.2"
DIAGNOSTICS = ROOT / "tests/fixtures/rr_official_examples.json"
LIMITS = {
    "whole_seconds": 3600.0,
    "update_seconds": 60.0,
    "rss_bytes": 3 * 1024**3,
    "swap_growth_bytes": 1536 * 1024**2,
}
RESOURCE_RATIONALE = (
    "The 3 GiB process RSS bound and two Torch CPU threads remain unchanged.",
    "The 1.5 GiB system swap-growth allowance covers observed host-wide background variation; "
    "it is not a larger process-memory allowance.",
    "The shared one-hour wall bound accommodates two sequential full-engine arms under CPU "
    "contention without changing cases, checkpoints, scoring, or model selection.",
)
EXPECTED_SUITES = {
    "dev": {"file": "dev.jsonl", "rows": 140},
    "test": {"file": "test.jsonl", "rows": 1000},
}
RAW_DIAGNOSTICS = (
    "test", "OpenAI", "API-123", "https://example.com", "user@example.com",
)
HISTORICAL_RULE_DIAGNOSTICS = (
    ("imnida", "입니다"),
    ("ilgo", "읽고"),
    ("gulgi", "굵기"),
    ("bap", "밟"),
    ("jitbap", "짓밟"),
    ("neolpyeo", "넓혀"),
    ("neolpyeot", "넓혔"),
    ("balgi", "밝기"),
)
RUNTIME_SOURCES = (
    "hanglish/core/lattice.py",
    "hanglish/core/lexical_data.py",
    "hanglish/core/neural_converter.py",
    "hanglish/core/numeric_structure.py",
    "hanglish/core/phonology.py",
    "hanglish/core/predicate_boundaries.py",
    "hanglish/core/data/predicate_boundary_inventory.json",
    "hanglish/core/ctc_word_converter.py",
    "hanglish/core/ctc_union_converter.py",
    "hanglish/core/phonetic_choice_converter.py",
    "hanglish/core/production_phonetic_choice_converter.py",
    "hanglish/core/score_scale_converter.py",
    "hanglish/models/phonetic_transducer.py",
    "hanglish/models/semantic_ranker.py",
    "hanglish/models/sentence_ranker.py",
    "hanglish/models/phonetic_choice_features.py",
    "hanglish/models/phonetic_likelihood.py",
    "hanglish/models/batched_phonetic_decode.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value) -> None:
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     indent=2, allow_nan=False) + "\n")


def append_jsonl(stream, value) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def load_cases(split: str) -> tuple[list[dict], dict]:
    """Load cases once, then project a label-free input document for inference."""
    if split in EXPECTED_SUITES:
        spec = EXPECTED_SUITES[split]
        path = CORPUS / spec["file"]
        cases = _load_jsonl(path)
        if len(cases) != spec["rows"] or any(
                row.get("split") != split
                or not isinstance(row.get("id"), str)
                or not isinstance(row.get("input"), str)
                or not isinstance(row.get("context", ""), str)
                or not isinstance(row.get("expected"), list)
                or not row["expected"] for row in cases):
            raise ValueError(f"malformed {split} regression corpus")
        manifest = json.loads((CORPUS / "manifest.json").read_text())
        if (manifest.get("counts", {}).get(split) != len(cases)
                or manifest.get("sha256", {}).get(spec["file"]) != sha256(path)):
            raise ValueError(f"{split} regression manifest differs")
        return cases, {
            "source": str(path), "source_sha256": sha256(path),
            "manifest": str(CORPUS / "manifest.json"),
            "manifest_sha256": sha256(CORPUS / "manifest.json"),
        }
    if split != "diagnostics":
        raise ValueError(f"unsupported split: {split}")
    document = json.loads(DIAGNOSTICS.read_text())
    fixtures = [row for row in document.get("fixtures", [])
                if row.get("mode") == "rr_normal"]
    cases = [{
        "id": f"rr-normal-{index:03d}", "input": row["rr"].lower(),
        "context": "", "expected": [row["hangul"]], "category": "official_rr_normal",
        "rule": row.get("rule"), "raw": False,
    } for index, row in enumerate(fixtures, 1)]
    cases.extend({
        "id": f"historical-rule-{index:02d}", "input": roman, "context": "",
        "expected": [hangul], "category": "historical_rule_diagnostic",
        "rule": "known_historical_training_rule_case", "raw": False,
    } for index, (roman, hangul) in enumerate(HISTORICAL_RULE_DIAGNOSTICS, 1))
    cases.extend({
        "id": f"raw-ascii-{index:02d}", "input": source, "context": "",
        "expected": [source], "category": "raw_ascii_protection",
        "rule": "raw_mode_exact_copy", "raw": True,
    } for index, source in enumerate(RAW_DIAGNOSTICS, 1))
    if not fixtures or len({row["id"] for row in cases}) != len(cases):
        raise ValueError("RR diagnostic fixtures are missing or duplicate")
    return cases, {
        "source": str(DIAGNOSTICS), "source_sha256": sha256(DIAGNOSTICS),
        "fixture_schema": document.get("schema"),
    }


def project_inputs(cases: list[dict]) -> list[dict]:
    return [{
        "id": row["id"], "input": row["input"],
        "context": row.get("context", ""), "raw": bool(row.get("raw", False)),
    } for row in cases]


def _checkpoint_files(checkpoint: Path) -> dict[str, str]:
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise ValueError(f"phonetic checkpoint missing: {checkpoint}")
    paths = [checkpoint]
    metadata = checkpoint.with_suffix(".json")
    if metadata.is_file():
        paths.append(metadata)
    return {str(path): sha256(path) for path in paths}


def production_converter_kwargs(checkpoint: Path) -> dict:
    """Mirror Workbench's production constructor arguments exactly."""
    return {
        "score_scale_checkpoint": SCORE_SCALE_CHECKPOINT,
        "phonetic_checkpoint": Path(checkpoint),
    }


def make_score_scale_converter(checkpoint: Path, converter_class):
    # Deliberately omit external_lexicon: Workbench relies on the production
    # NeuralConverter default (True), and evaluation must exercise that path.
    return converter_class(**production_converter_kwargs(checkpoint))


def _source_pins() -> dict[str, str]:
    paths = [Path(__file__), *(ROOT / name for name in RUNTIME_SOURCES)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("runtime source missing: " + ", ".join(missing))
    return {str(path.resolve()): sha256(path) for path in paths}


def _validate_response(response: dict, item: dict) -> None:
    if (not isinstance(response, dict) or response.get("input") != item["input"]
            or response.get("context") != item["context"]
            or response.get("result") is None
            or type(response.get("engine_ms")) not in (int, float)
            or not math.isfinite(response["engine_ms"])
            or response["engine_ms"] < 0):
        raise ValueError("runtime response metadata differs")
    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 5:
        raise ValueError("runtime candidate count differs")
    texts = [candidate.get("text") for candidate in candidates]
    if any(not isinstance(text, str) for text in texts) or response["result"] != texts[0]:
        raise ValueError("runtime candidate ordering differs")
    if item["raw"] and (response.get("raw") is not True or texts != [item["input"]]):
        raise ValueError("raw ASCII diagnostic was not preserved")


def capture_request(converter, item: dict, guard, *, timer=time.perf_counter) -> dict:
    """Capture one request, while preserving watchdog hard-stop semantics."""
    started = timer()
    record = {"id": item["id"], "request": item}
    try:
        with guard.operation(f"predict:{item['id']}", update=True):
            response = converter.convert(
                item["input"], context=item["context"], engine="neural",
                mode="batch", raw=item["raw"])
            _validate_response(response, item)
        record["response"] = {
            "result": response["result"],
            "candidates": response["candidates"],
            "version": response.get("version"),
            "engine_ms": response["engine_ms"],
            "raw": response.get("raw", False),
        }
    except ResourceExceeded:
        # Resource limits are arm-fatal.  Recording one timed-out request and
        # continuing would let a partial arm look like a completed comparison.
        raise
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record["elapsed_ms"] = (timer() - started) * 1000
    return record


def _arm_capture(arm: str, checkpoint: str, input_path: str, arm_dir: str,
                 global_started: float, baseline_swap: int) -> None:
    """Child-process entry point.  It receives no expected answers."""
    arm_path = Path(arm_dir)
    arm_path.mkdir(parents=False, exist_ok=False)
    try:
        import torch
        from hanglish.core.score_scale_converter import ScoreScaleConverter
        from scripts.source_span_resources import AtomicResourceWatchdog

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        inputs = json.loads(Path(input_path).read_text())
        input_sha = sha256(Path(input_path))
        source_pins = _source_pins()
        checkpoint_pins = _checkpoint_files(Path(checkpoint))
        with AtomicResourceWatchdog(
                global_started, baseline_swap, arm_path / "resources.jsonl",
                limits=LIMITS, interval_s=0.5,
                failure_report=arm_path / "resource-failure.json") as guard:
            with guard.operation("load_models", update=True):
                converter = make_score_scale_converter(
                    Path(checkpoint), ScoreScaleConverter)
                converter.load_models()
                converter.load_phonetic_model()
                converter.load_word_model()
                converter.load_growth_model()
                model_artifacts = dict(converter.model_artifacts)
                version = converter.version
            captures = []
            with (arm_path / "capture.jsonl").open("x") as stream:
                for index, item in enumerate(inputs):
                    record = capture_request(converter, item, guard)
                    captures.append(record)
                    append_jsonl(stream, record)
                    if (index + 1) % 25 == 0 or index + 1 == len(inputs):
                        print(json.dumps({"arm": arm, "captured": index + 1,
                                          "total": len(inputs)}), flush=True)
            with guard.operation("verify_identity", update=True):
                if sha256(Path(input_path)) != input_sha:
                    raise ValueError("label-free input projection changed during inference")
                for path, digest in {**source_pins, **checkpoint_pins}.items():
                    if sha256(Path(path)) != digest:
                        raise ValueError(f"pinned source/model changed: {path}")
                write_json(arm_path / "identity.json", {
                    "arm": arm, "rows": len(captures), "input_sha256": input_sha,
                    "source_hashes": source_pins, "checkpoint_files": checkpoint_pins,
                    "runtime_version": version, "model_artifacts": model_artifacts,
                    "capture_sha256": sha256(arm_path / "capture.jsonl"),
                })
            guard.sample_and_append("complete")
    except BaseException as error:
        write_json(arm_path / "failure.json", {
            "completed": False, "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(),
        })
        raise


def _run_arm(arm: str, checkpoint: Path, input_path: Path, output: Path,
             global_started: float, baseline_swap: int) -> None:
    remaining = LIMITS["whole_seconds"] - (time.monotonic() - global_started)
    if remaining <= 0:
        raise RuntimeError(
            f"evaluation exceeded {LIMITS['whole_seconds']:g} seconds before arm launch")
    process = multiprocessing.get_context("spawn").Process(
        target=_arm_capture,
        args=(arm, str(checkpoint.resolve()), str(input_path.resolve()),
              str((output / arm).resolve()), global_started, baseline_swap),
    )
    process.start()
    process.join(remaining)
    if process.is_alive():
        process.terminate()
        process.join(15)
        raise RuntimeError(
            f"{arm} evaluation exceeded the shared {LIMITS['whole_seconds']:g} second limit")
    if process.exitcode != 0:
        failure = output / arm / "failure.json"
        detail = json.loads(failure.read_text()).get("error") if failure.is_file() else None
        raise RuntimeError(f"{arm} evaluation failed" + (f": {detail}" if detail else ""))


def topk_outcome(capture: dict, expected: list[str]) -> dict:
    error = capture.get("error")
    candidates = capture.get("response", {}).get("candidates", [])
    texts = [candidate.get("text") for candidate in candidates]
    return {
        "top1": not error and bool(texts) and texts[0] in expected,
        "top5": not error and any(text in expected for text in texts[:5]),
        "error": error,
        "actual": texts[:5],
    }


def verify_capture_identity(baseline_identity: dict, proposed_identity: dict,
                            baseline: list[dict], proposed: list[dict], *,
                            baseline_capture_sha256: str,
                            proposed_capture_sha256: str) -> None:
    if baseline_identity.get("input_sha256") != proposed_identity.get("input_sha256"):
        raise ValueError("baseline/proposed input hash mismatch")
    if baseline_identity.get("source_hashes") != proposed_identity.get("source_hashes"):
        raise ValueError("baseline/proposed runtime source hash mismatch")
    for name, identity, rows, digest in (
        ("baseline", baseline_identity, baseline, baseline_capture_sha256),
        ("proposed", proposed_identity, proposed, proposed_capture_sha256),
    ):
        if identity.get("rows") != len(rows):
            raise ValueError(f"{name} capture row count differs from identity")
        if identity.get("capture_sha256") != digest:
            raise ValueError(f"{name} capture hash differs from identity")
    if len(baseline) != len(proposed):
        raise ValueError("baseline/proposed row count mismatch")
    for before, after in zip(baseline, proposed):
        if before.get("id") != after.get("id") or before.get("request") != after.get("request"):
            raise ValueError("baseline/proposed cases or rules differ")


def compare_captures(baseline: list[dict], proposed: list[dict],
                     cases: list[dict]) -> dict:
    if [row.get("id") for row in baseline] != [row.get("id") for row in cases]:
        raise ValueError("capture/label identity mismatch")
    if [row.get("id") for row in proposed] != [row.get("id") for row in cases]:
        raise ValueError("proposed/label identity mismatch")
    totals = {arm: Counter() for arm in ("baseline", "proposed")}
    categories = {arm: defaultdict(Counter) for arm in totals}
    changes = {name: [] for name in (
        "corrections", "regressions", "top5_gains", "top5_regressions")}
    outcomes = []
    for case, before, after in zip(cases, baseline, proposed):
        old = topk_outcome(before, case["expected"])
        new = topk_outcome(after, case["expected"])
        category = case.get("category", "uncategorized")
        for arm, outcome in (("baseline", old), ("proposed", new)):
            row = {"rows": 1, "top1": int(outcome["top1"]),
                   "top5": int(outcome["top5"]),
                   "errors": int(outcome["error"] is not None)}
            totals[arm].update(row)
            categories[arm][category].update(row)
        for name, condition in (
            ("corrections", not old["top1"] and new["top1"]),
            ("regressions", old["top1"] and not new["top1"]),
            ("top5_gains", not old["top5"] and new["top5"]),
            ("top5_regressions", old["top5"] and not new["top5"]),
        ):
            if condition:
                changes[name].append(case["id"])
        outcomes.append({
            "id": case["id"], "category": category, "input": case["input"],
            "expected": case["expected"], "baseline": old, "proposed": new,
        })
    metrics = {
        arm: {"counts": dict(totals[arm]),
              "categories": {key: dict(value)
                             for key, value in categories[arm].items()}}
        for arm in totals
    }
    return {**metrics, **changes, "outcomes": outcomes}


def _latency(captures: list[dict]) -> dict:
    values = sorted(row["elapsed_ms"] for row in captures)
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("invalid latency capture")
    pick = lambda q: values[max(0, math.ceil(q * len(values)) - 1)]
    return {"p50_ms": pick(0.5), "p95_ms": pick(0.95), "max_ms": pick(1.0)}


def _non_phonetic_artifacts(identity: dict) -> dict:
    checkpoint_names = {Path(path).name
                        for path in identity.get("checkpoint_files", {})}
    return {name: digest for name, digest in identity.get("model_artifacts", {}).items()
            if Path(name).name not in checkpoint_names}


def run(checkpoint: Path, output: Path, split: str) -> dict:
    from scripts.source_span_resources import system_swap_bytes

    started = time.monotonic()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        cases, source_identity = load_cases(split)
        inputs = project_inputs(cases)
        input_path = output / "inputs.json"
        write_json(input_path, inputs)
        input_hash = sha256(input_path)
        baseline_files = _checkpoint_files(BASELINE_CHECKPOINT)
        proposed_files = _checkpoint_files(Path(checkpoint))
        source_pins = _source_pins()
        write_json(output / "frozen-input.json", {
            "split": split, "rows": len(inputs), "input_sha256": input_hash,
            "source_identity": source_identity, "runtime_source_hashes": source_pins,
            "baseline_checkpoint_files": baseline_files,
            "proposed_checkpoint_files": proposed_files,
            "rules_hash": canonical_hash([row.get("rule") for row in cases]),
        })
        baseline_swap = system_swap_bytes()
        _run_arm("baseline", BASELINE_CHECKPOINT, input_path, output, started, baseline_swap)
        _run_arm("proposed", Path(checkpoint), input_path, output, started, baseline_swap)

        if sha256(input_path) != input_hash:
            raise ValueError("label-free input projection changed")
        for path, digest in {**source_pins, **baseline_files, **proposed_files}.items():
            if sha256(Path(path)) != digest:
                raise ValueError(f"pinned source/model changed: {path}")
        baseline_identity = json.loads((output / "baseline/identity.json").read_text())
        proposed_identity = json.loads((output / "proposed/identity.json").read_text())
        baseline_capture = output / "baseline/capture.jsonl"
        proposed_capture = output / "proposed/capture.jsonl"
        baseline = _load_jsonl(baseline_capture)
        proposed = _load_jsonl(proposed_capture)
        verify_capture_identity(
            baseline_identity, proposed_identity, baseline, proposed,
            baseline_capture_sha256=sha256(baseline_capture),
            proposed_capture_sha256=sha256(proposed_capture))
        if _non_phonetic_artifacts(baseline_identity) != _non_phonetic_artifacts(proposed_identity):
            raise ValueError("non-phonetic model weights differ between comparison arms")
        # Expected labels are used only here, after both inference processes exited.
        metrics = compare_captures(baseline, proposed, cases)
        outcomes = metrics.pop("outcomes")
        with (output / "outcomes.jsonl").open("x") as stream:
            for row in outcomes:
                append_jsonl(stream, row)
        raw_ids = {row["id"] for row in cases if row.get("raw")}
        raw_protected = all(
            outcome["proposed"]["top1"] and outcome["baseline"]["top1"]
            for outcome in outcomes if outcome["id"] in raw_ids)
        errors = (metrics["baseline"]["counts"].get("errors", 0)
                  + metrics["proposed"]["counts"].get("errors", 0))
        report = {
            "schema": "rr-retraining-full-engine-comparison-v1",
            "completed": True, "split": split, "rows": len(cases),
            "input_contract": "source_only_until_both_captures_close",
            "labels_passed_to_engine": False,
            "expected_joined_after_both_arms": True,
            "separate_process_per_arm": True,
            "baseline": metrics["baseline"], "proposed": metrics["proposed"],
            "corrections": metrics["corrections"],
            "regressions": metrics["regressions"],
            "top5_gains": metrics["top5_gains"],
            "top5_regressions": metrics["top5_regressions"],
            "raw_ascii_protected_all": raw_protected,
            "capture_errors": errors,
            "latency": {"baseline": _latency(baseline), "proposed": _latency(proposed)},
            "identities": {"baseline": baseline_identity,
                           "proposed": proposed_identity},
            "frozen_input_sha256": input_hash,
            "frozen_input_record_sha256": sha256(output / "frozen-input.json"),
            "outcomes_sha256": sha256(output / "outcomes.jsonl"),
            "limits": LIMITS,
            "resource_limit_rationale": list(RESOURCE_RATIONALE),
            "automatic_promotion": False,
            "quality_summary_only": True,
            "generalization_claim_allowed": False,
            "limitations": ([
                "DEV and TEST are known regression suites, not blind generalization data.",
                "No TEST result may be used to tune or select the retrained checkpoint.",
            ] if split in EXPECTED_SUITES else [
                "Official RR diagnostics are known examples and may be ambiguous in reverse.",
                "Historical rule diagnostics are known, overlap training history, and may be ambiguous in reverse.",
                "Raw ASCII checks exercise explicit raw-mode preservation only.",
            ]),
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(output / "report.json", report)
        write_json(output / "status.json", {
            "state": "complete", "report_sha256": sha256(output / "report.json"),
            "capture_errors": errors,
        })
        return report
    except BaseException as error:
        write_json(output / "failure.json", {
            "completed": False, "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - started,
        })
        raise


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test", "diagnostics"), required=True)
    args = parser.parse_args(argv)
    report = run(args.checkpoint, args.output, args.split)
    print(json.dumps({
        "completed": report["completed"], "split": report["split"],
        "baseline": report["baseline"]["counts"],
        "proposed": report["proposed"]["counts"],
        "corrections": len(report["corrections"]),
        "regressions": len(report["regressions"]),
        "capture_errors": report["capture_errors"],
    }, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
