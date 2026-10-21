"""Paired full-runtime evaluation for productive terminal ``-네`` forms."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import evaluate_guarded_context as guarded
from scripts import evaluate_rr_retraining as rr


LIMITS = dict(whole_seconds=5400.0, update_seconds=60.0,
              rss_bytes=3 * 1024**3, swap_growth_bytes=1536 * 1024**2)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project_inputs(cases):
    return [{key: row[key] for key in ("id", "input", "context", "raw")}
            for row in cases]


def load_cases():
    development, pins = guarded.development_cases()
    cases = [dict(row, category=row.get("category", row["suite"]))
             for row in development]
    test, test_identity = rr.load_cases("test")
    cases.extend({**row, "id": "test:" + row["id"], "suite": "test",
                  "category": row.get("category", "test"), "raw": False}
                 for row in test)
    diagnostics, diagnostic_identity = rr.load_cases("diagnostics")
    cases.extend({**row, "id": "diagnostics:" + row["id"],
                  "suite": "diagnostics", "category": row["category"]}
                 for row in diagnostics)
    pins.update({
        test_identity["source"]: test_identity["source_sha256"],
        test_identity["manifest"]: test_identity["manifest_sha256"],
        diagnostic_identity["source"]: diagnostic_identity["source_sha256"],
    })
    if len(cases) != 368 + 1000 + 151 or len({row["id"] for row in cases}) != len(cases):
        raise ValueError("terminal-ne evaluation case identity differs")
    return cases, pins


def clear_runtime_caches(converter):
    for cached in (converter.word_options, converter.generator.options,
                   converter.generator.stem_options):
        cached.cache_clear()


def capture(converter, inputs, path, guard, arm):
    rows = []
    with Path(path).open("x", encoding="utf-8") as stream:
        for index, item in enumerate(inputs):
            row = rr.capture_request(converter, item, guard)
            rows.append(row)
            rr.append_jsonl(stream, row)
            if (index + 1) % 50 == 0 or index + 1 == len(inputs):
                print(json.dumps({"arm": arm, "completed": index + 1,
                                  "total": len(inputs)}), flush=True)
    return rows


def capture_arms(converter, cases, output, guard, *, capture_fn=capture):
    inputs = project_inputs(cases)
    endings = converter.generator.productive_endings
    enabled = endings.ne
    if not enabled:
        raise ValueError("terminal-ne productive index is empty")
    try:
        endings.ne = {}
        clear_runtime_caches(converter)
        baseline = capture_fn(
            converter, inputs, output / "baseline.jsonl", guard, "baseline")
        endings.ne = enabled
        clear_runtime_caches(converter)
        proposed = capture_fn(
            converter, inputs, output / "proposed.jsonl", guard, "proposed")
        return baseline, proposed
    finally:
        endings.ne = enabled
        clear_runtime_caches(converter)


def grouped_latency(captures, cases, field):
    grouped = {}
    for capture_row, case in zip(captures, cases):
        key = case[field]
        grouped.setdefault(key, []).append(capture_row)
    return {key: rr._latency(rows) for key, rows in sorted(grouped.items())}


def grouped_quality(captures, cases, field):
    grouped = {}
    for capture_row, case in zip(captures, cases):
        outcome = rr.topk_outcome(capture_row, case["expected"])
        counts = grouped.setdefault(case[field], {
            "rows": 0, "top1": 0, "top5": 0, "errors": 0})
        counts["rows"] += 1
        counts["top1"] += int(outcome["top1"])
        counts["top5"] += int(outcome["top5"])
        counts["errors"] += int(outcome["error"] is not None)
    return dict(sorted(grouped.items()))


def additional_source_pins():
    from hanglish.core.lexical_data import FILES

    dictionary = ROOT / "data/external/mecab-ko-dic"
    paths = [
        Path(__file__), Path(guarded.__file__),
        ROOT / "hanglish/core/productive_endings.py",
        ROOT / "hanglish/core/workbench.py",
        ROOT / "hanglish/core/data/predicate_boundary_inventory.json",
        ROOT / "scripts/preflight_pko_seq2seq.py",
        ROOT / "scripts/source_span_resources.py",
        ROOT / "data/morphology_lexicon.json",
        ROOT / "data/workbench_lexicon.json",
        ROOT / "data/common_english.txt",
        ROOT / "data/it_terms.txt",
        ROOT / "data/challenge/train_lexicon.json",
        dictionary / "SOURCE.json",
        dictionary / "NNBC.csv",
        *(dictionary / f"{name}.csv" for name in FILES),
    ]
    return {str(path): digest(path) for path in paths}


def adoption_gate(metrics, cases):
    errors = (metrics.get("baseline", {}).get("counts", {}).get("errors") == 0
              and metrics.get("proposed", {}).get("counts", {}).get("errors") == 0)
    suites = {row["id"]: row["suite"] for row in cases}
    diagnostic_gains = [case_id for case_id in metrics.get("top5_gains", ())
                        if suites.get(case_id) == "diagnostics"]
    adopted = bool(errors and not metrics.get("regressions")
                   and not metrics.get("top5_regressions") and diagnostic_gains)
    return {"adopted": adopted, "diagnostic_top5_gains": diagnostic_gains,
            "requirements": {
                "baseline_errors_zero": metrics.get("baseline", {}).get(
                    "counts", {}).get("errors") == 0,
                "proposed_errors_zero": metrics.get("proposed", {}).get(
                    "counts", {}).get("errors") == 0,
                "top1_regressions_zero": not metrics.get("regressions"),
                "top5_regressions_zero": not metrics.get("top5_regressions"),
                "diagnostic_top5_gain_present": bool(diagnostic_gains),
            }}


def verify_numeric_captures(rows, expected):
    if len(rows) != expected:
        raise ValueError("capture row count differs")
    if any(not isinstance(row.get("elapsed_ms"), (int, float))
           or not math.isfinite(row["elapsed_ms"]) or row["elapsed_ms"] < 0
           for row in rows):
        raise ValueError("capture latency differs")


def run(output):
    import torch
    from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    cases, source_pins = load_cases()
    source_pins.update(rr._source_pins())
    source_pins.update(additional_source_pins())
    inputs = project_inputs(cases)
    rr.write_json(output / "inputs.json", inputs)
    input_sha256 = digest(output / "inputs.json")
    rr.write_json(output / "frozen.json", {
        "source_pins": source_pins,
        "input_sha256": input_sha256,
        "rows": len(inputs),
        "comparison": "same_runtime_terminal_ne_disabled_vs_enabled",
        "labels_passed_to_engine": False,
        "training": False,
        "threshold_tuning": False,
    })

    with AtomicResourceWatchdog(
            started, system_swap_bytes(), output / "resources.jsonl",
            limits=LIMITS, interval_s=.5) as guard:
        with guard.operation("load_models", update=True):
            converter = guarded.make_converter()
            artifacts = dict(converter.model_artifacts)
            version = converter.version
        baseline, proposed = capture_arms(converter, cases, output, guard)
        verify_numeric_captures(baseline, len(cases))
        verify_numeric_captures(proposed, len(cases))
        if (converter.version != version or converter.model_artifacts != artifacts):
            raise ValueError("runtime identity changed between feature arms")
        if digest(output / "inputs.json") != input_sha256:
            raise ValueError("source-only input snapshot changed")
        for path, expected in {**source_pins, **artifacts}.items():
            if digest(ROOT / path if not Path(path).is_absolute() else path) != expected:
                raise ValueError("pinned source/model/data changed: " + path)

        # Gold labels are joined only after both source-only captures close.
        comparison = rr.compare_captures(baseline, proposed, cases)
        outcomes = comparison.pop("outcomes")
        rr.write_json(output / "outcomes.json", outcomes)
        gate = adoption_gate(comparison, cases)
        report = {
            "completed": True,
            "rows": len(cases),
            "suite_rows": {suite: sum(row["suite"] == suite for row in cases)
                           for suite in sorted({row["suite"] for row in cases})},
            "comparison": comparison,
            "gate": gate,
            "quality": {
                "by_suite": {
                    "baseline": grouped_quality(baseline, cases, "suite"),
                    "proposed": grouped_quality(proposed, cases, "suite")},
                "by_category": {
                    "baseline": grouped_quality(baseline, cases, "category"),
                    "proposed": grouped_quality(proposed, cases, "category")},
            },
            "latency": {
                "overall": {"baseline": rr._latency(baseline),
                            "proposed": rr._latency(proposed)},
                "by_suite": {"baseline": grouped_latency(baseline, cases, "suite"),
                             "proposed": grouped_latency(proposed, cases, "suite")},
                "by_category": {
                    "baseline": grouped_latency(baseline, cases, "category"),
                    "proposed": grouped_latency(proposed, cases, "category")},
            },
            "runtime_identity": {"version": version, "model_artifacts": artifacts},
            "arms": {
                "baseline": {"feature": "terminal_ne", "enabled": False},
                "proposed": {"feature": "terminal_ne", "enabled": True},
            },
            "baseline_contract": (
                "Production runtime with only the terminal-ne index disabled; "
                "code, version metadata, and model artifact identity are shared."),
            "labels_passed_to_engine": False,
            "expected_joined_after_both_captures": True,
            "same_converter_sequential_arms": True,
            "training": False,
            "threshold_tuning": False,
            "eligible_for_adoption": gate["adopted"],
            "installed_models_changed": False,
            "generalization_claim_allowed": False,
            "source_pins": source_pins,
            "input_sha256": input_sha256,
            "baseline_capture_sha256": digest(output / "baseline.jsonl"),
            "proposed_capture_sha256": digest(output / "proposed.jsonl"),
            "outcomes_sha256": digest(output / "outcomes.json"),
            "limits": LIMITS,
            "elapsed_seconds": time.monotonic() - started,
        }
        rr.write_json(output / "report.json", report)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
