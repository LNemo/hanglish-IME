"""Gated final regression for the numeric sentence-pool overlay."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

import hanglish.core.neural_converter as neural_module
from hanglish.core.neural_converter import NeuralConverter, file_sha256
from scripts.evaluate_current_word_ranker import DEV_COUNTS, summarize
from scripts.evaluate_numeric_structure_development import (
    ACRONYM_CASES_SHA256,
    ACRONYM_REPORT_SHA256,
    INTEGRATION,
    INTEGRATION_SHA256,
    PROFILE as DEVELOPMENT_PROFILE,
    RUNTIME_VERSION,
    TRAIN_CASES_SHA256,
    TRAIN_OUTPUT,
    TRAIN_REPORT_SHA256,
    TRAIN_SOURCE_SHA256,
    development_gate,
    evaluate_case,
    read_pinned,
    sha256_bytes,
    validate_lineage,
    validate_train_gate,
)
from scripts.numeric_structure_candidates import NumericSpanGenerator
from scripts.probe_numeric_structure_runtime import ORIGINAL_BEAM_PATHS, SentencePoolRecorder
from scripts.train_natural_sentence_ranker import rss_bytes


OUTPUT = ROOT / ".omx/artifacts/numeric-structure-final-v1"
DEVELOPMENT = ROOT / ".omx/artifacts/numeric-structure-development-v1"
DEVELOPMENT_REPORT_SHA256 = "750e777f784e3bbdd2e79246b8abc68532e4840dc18e6c9f035da3e62a9b14e0"
DEVELOPMENT_CASES_SHA256 = "2baaccfbca4d7f4370a9c0b61c548fa9d4e33c50440c38c6dabdcd67d2ca920c"
DEVELOPMENT_SOURCE_SHA256 = "b1d79401fcbff77fde670f46c63877cc66dd64196e9cfab44387df248e588f33"
ACRONYM_FINAL = ROOT / ".omx/artifacts/acronym-suffix-final-v1"
ACRONYM_FINAL_REPORT_SHA256 = "918a2f19ecd0ccb5d4003d2a1a75f88ec577a6ca9e7bcc2cf4b85620d7cfb855"
FINAL_SOURCE = ROOT / "data/challenge/v2.2/test.jsonl"
FINAL_SOURCE_SHA256 = "3d2fe696cb62e3e895017f817e8f5ccca8a651f17a325725ef624b6d3335c720"
FINAL_IDENTITY = ROOT / ".omx/artifacts/challenge-v22-test-natural-v3-deployed.json"
FINAL_IDENTITY_SHA256 = "caa3a5783de91e86493c1b354abbbf0e0003b611b794d9ccd97114639b7bc264"
PROFILE = {
    "name": "numeric-structure-gated-final-v1",
    "rows": 1000,
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 1800,
    "progress_interval": 32,
    "score_tolerance": DEVELOPMENT_PROFILE["score_tolerance"],
}


def validate_development_report(report, cases, explicit_sha, actual_sha,
                                actual_cases_sha):
    if (explicit_sha != DEVELOPMENT_REPORT_SHA256
            or actual_sha != explicit_sha
            or report.get("passed") is not True
            or report.get("eligible_for_further_evaluation") is not True
            or report.get("scope") !=
            "fresh actual hybrid development only for numeric sentence-pool overlay"
            or report.get("profile") != DEVELOPMENT_PROFILE
            or report.get("runtime_version") != RUNTIME_VERSION
            or report.get("training_report_sha256") != TRAIN_REPORT_SHA256
            or report.get("training_cases_sha256") != TRAIN_CASES_SHA256
            or report.get("training_adapter_source_sha256") != TRAIN_SOURCE_SHA256
            or report.get("accepted_acronym_development_report_sha256") !=
            ACRONYM_REPORT_SHA256
            or report.get("accepted_acronym_development_cases_sha256") !=
            ACRONYM_CASES_SHA256
            or report.get("source_hashes") != report.get("source_hashes_after")
            or report.get("source_hashes", {}).get(
                "scripts/evaluate_numeric_structure_development.py"
            ) != DEVELOPMENT_SOURCE_SHA256
            or report.get("model_artifact_hash_count") != 16
            or actual_cases_sha != DEVELOPMENT_CASES_SHA256
            or report.get("cases_sha256") != actual_cases_sha
            or report.get("labels_used_for_candidate_production") is not False
            or report.get("final_data_read") is not False
            or report.get("new_weights_created") is not False
            or report.get("deployed") is not False
            or report.get("overlay", {}).get(
                "baseline_pool_and_score_prefix_preserved"
            ) is not True
            or report.get("overlay", {}).get("word_options_identity_preserved") is not True
            or report.get("overlay", {}).get("beam_hook_restored") is not True):
        raise ValueError("complete approved numeric development report is required")
    if (len(cases) != sum(DEV_COUNTS.values())
            or len({row.get("id") for row in cases}) != len(cases)):
        raise ValueError("numeric development cases are incomplete or duplicated")
    rebuilt = {}
    for name, count in DEV_COUNTS.items():
        rows = [row for row in cases if row.get("suite") == name]
        if len(rows) != count:
            raise ValueError("numeric development suite count differs")
        rebuilt[name] = summarize(rows)
    if rebuilt != report.get("suites") or not development_gate(rebuilt):
        raise ValueError("numeric development metrics or strict gate differ")


def validate_development_lineage(report, train_report, train_cases, integration,
                                 explicit_sha, actual_sha):
    validate_train_gate(train_report, train_cases, TRAIN_REPORT_SHA256,
                        TRAIN_REPORT_SHA256)
    accepted_sources = validate_lineage(train_report, integration)
    if (explicit_sha != actual_sha
            or report.get("source_hashes") != report.get("source_hashes_after")):
        raise ValueError("numeric development source state differs")
    observed = report["source_hashes"]
    if any(observed.get(path) != digest for path, digest in accepted_sources.items()):
        raise ValueError("numeric development dropped accepted TRAIN lineage")
    if len(integration.get("accepted_weight_hashes", {})) != 16:
        raise ValueError("numeric development requires exactly sixteen model artifacts")
    for relative, expected in observed.items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"numeric development input changed: {relative}")
    return dict(observed)


def read_final_inputs(authorized, *, final_source=FINAL_SOURCE,
                      final_identity=FINAL_IDENTITY,
                      acronym_report_path=None, acronym_cases_path=None):
    """Read final bytes only after the complete development gate succeeds."""
    if authorized is not True:
        raise PermissionError("approved numeric development gate is required before final read")
    acronym_report_path = acronym_report_path or ACRONYM_FINAL / "report.json"
    acronym_cases_path = acronym_cases_path or ACRONYM_FINAL / "cases.jsonl"
    final_raw = final_source.read_bytes()
    identity_raw = final_identity.read_bytes()
    report_raw = acronym_report_path.read_bytes()
    cases_raw = acronym_cases_path.read_bytes()
    if (sha256_bytes(final_raw) != FINAL_SOURCE_SHA256
            or sha256_bytes(identity_raw) != FINAL_IDENTITY_SHA256
            or sha256_bytes(report_raw) != ACRONYM_FINAL_REPORT_SHA256):
        raise ValueError("final source, identity, or acronym reference differs")
    identity = json.loads(identity_raw)
    report = json.loads(report_raw)
    cases = [json.loads(line) for line in cases_raw.splitlines() if line]
    sources = [json.loads(line) for line in final_raw.splitlines() if line]
    identity_source_sha = identity.get("sha256") or identity.get(
        "corpus_identity", {}
    ).get("sha256")
    if (identity_source_sha != FINAL_SOURCE_SHA256
            or report.get("cases_sha256") != sha256_bytes(cases_raw)
            or len(sources) != PROFILE["rows"] or len(cases) != PROFILE["rows"]):
        raise ValueError("final source identity or acronym cases are incomplete")
    return {
        "source_bytes": final_raw, "identity_bytes": identity_raw,
        "acronym_report_bytes": report_raw, "acronym_cases_bytes": cases_raw,
        "sources": sources, "acronym_report": report, "acronym_cases": cases,
    }


def validate_acronym_final(report, cases):
    if (report.get("passed") is not True or report.get("goal_met") is not False
            or report.get("scope") != "gated exposed final regression"
            or report.get("final_source_sha256") != FINAL_SOURCE_SHA256
            or report.get("final_identity_sha256") != FINAL_IDENTITY_SHA256
            or report.get("cases_sha256") is None
            or report.get("deployed") is not False
            or report.get("outcomes") != cases
            or len({row.get("id") for row in cases}) != PROFILE["rows"]):
        raise ValueError("accepted acronym final reference is incomplete")
    metrics = summarize(cases)
    if (metrics != report.get("metrics")
            or metrics.get("rows") != PROFILE["rows"]
            or metrics.get("before_top1") != 988 or metrics.get("top1") != 988
            or metrics.get("before_top5") != PROFILE["rows"]
            or metrics.get("top5") != PROFILE["rows"]
            or metrics.get("regressions") != []
            or metrics.get("top5_regressions") != []
            or metrics.get("sentence_pool_coverage_losses") != []):
        raise ValueError("accepted acronym final metrics differ")


def align_final_sources(sources, references):
    for source, reference in zip(sources, references):
        if any(source.get(key) != reference.get(key) for key in ("id", "input", "context")):
            raise ValueError("final source identity or order differs from acronym reference")


def final_gate(metrics):
    return bool(
        metrics.get("rows") == PROFILE["rows"]
        and metrics.get("before_top1") == 988
        and metrics.get("before_top5") == PROFILE["rows"]
        and metrics.get("top1", -1) >= 988
        and metrics.get("top5") == PROFILE["rows"]
        and metrics.get("regressions") == []
        and metrics.get("top5_regressions") == []
        and metrics.get("sentence_pool_coverage_losses") == []
    )


def require_resources(started, stage):
    if (rss_bytes() > PROFILE["rss_limit_bytes"]
            or time.monotonic() - started > PROFILE["time_limit_seconds"]):
        raise RuntimeError(f"numeric final resource bound exceeded during {stage}")


def run(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_bytes = Path(__file__).read_bytes()
    source_sha = sha256_bytes(source_bytes)

    # Every development and TRAIN gate precedes all reads of final content.
    development_raw, development = read_pinned(
        DEVELOPMENT / "report.json", DEVELOPMENT_REPORT_SHA256
    )
    development_cases_raw, development_cases = read_pinned(
        DEVELOPMENT / "cases.jsonl", DEVELOPMENT_CASES_SHA256, jsonl=True
    )
    development_sha = sha256_bytes(development_raw)
    validate_development_report(
        development, development_cases, args.development_report_sha256,
        development_sha, sha256_bytes(development_cases_raw),
    )
    _, train_report = read_pinned(TRAIN_OUTPUT / "report.json", TRAIN_REPORT_SHA256)
    _, train_cases = read_pinned(
        TRAIN_OUTPUT / "cases.jsonl", TRAIN_CASES_SHA256, jsonl=True
    )
    _, integration = read_pinned(INTEGRATION, INTEGRATION_SHA256)
    fixed_hashes = validate_development_lineage(
        development, train_report, train_cases, integration,
        args.development_report_sha256, development_sha,
    )
    if file_sha256(Path(__file__)) != source_sha:
        raise ValueError("numeric final evaluator changed before final authorization")
    final = read_final_inputs(True)
    validate_acronym_final(final["acronym_report"], final["acronym_cases"])
    align_final_sources(final["sources"], final["acronym_cases"])

    fixed_hashes.update({
        ".omx/artifacts/numeric-structure-development-v1/report.json": development_sha,
        ".omx/artifacts/numeric-structure-development-v1/cases.jsonl":
        sha256_bytes(development_cases_raw),
        ".omx/artifacts/acronym-suffix-final-v1/report.json":
        ACRONYM_FINAL_REPORT_SHA256,
        ".omx/artifacts/acronym-suffix-final-v1/cases.jsonl":
        sha256_bytes(final["acronym_cases_bytes"]),
        "data/challenge/v2.2/test.jsonl": FINAL_SOURCE_SHA256,
        ".omx/artifacts/challenge-v22-test-natural-v3-deployed.json":
        FINAL_IDENTITY_SHA256,
        "scripts/evaluate_numeric_structure_final.py": source_sha,
    })
    for relative, expected in fixed_hashes.items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"numeric final input changed before inference: {relative}")

    torch.set_num_threads(PROFILE["threads"])
    os.environ["OMP_NUM_THREADS"] = str(PROFILE["threads"])
    converter = NeuralConverter(ranker_kind="hybrid")
    converter.load_models()
    if converter.version != RUNTIME_VERSION:
        raise ValueError("production runtime version differs")
    original_options = converter.word_options
    recorder = SentencePoolRecorder(converter.sentence_ranker)
    converter.sentence_ranker = recorder
    generator = NumericSpanGenerator(converter.numeric_units)
    outcomes = []
    for index, (source, reference) in enumerate(zip(
            final["sources"], final["acronym_cases"])):
        row = evaluate_case(converter, recorder, generator, source, reference)
        row["suite"] = "final_v22"
        outcomes.append(row)
        require_resources(started, "paired final inference")
        if (index + 1) % PROFILE["progress_interval"] == 0:
            print(json.dumps({"event": "numeric_final", "rows": index + 1,
                              "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
    if converter.word_options is not original_options or neural_module.beam_paths is not ORIGINAL_BEAM_PATHS:
        raise ValueError("numeric final left a runtime hook installed")
    metrics = summarize(outcomes)
    passed = final_gate(metrics)
    after = {relative: file_sha256(ROOT / relative) for relative in fixed_hashes}
    if after != fixed_hashes:
        raise ValueError("numeric final source, model, artifact, or corpus changed")
    require_resources(started, "post-validation")

    args.output.mkdir(parents=True)
    final_snapshot = args.output / "final-source.jsonl"
    identity_snapshot = args.output / "final-identity.json"
    final_snapshot.write_bytes(final["source_bytes"])
    identity_snapshot.write_bytes(final["identity_bytes"])
    cases_path = args.output / "cases.jsonl"
    with cases_path.open("x") as stream:
        for row in outcomes:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    source_snapshot = args.output / "source.py"
    source_snapshot.write_bytes(source_bytes)
    if (file_sha256(final_snapshot) != FINAL_SOURCE_SHA256
            or file_sha256(identity_snapshot) != FINAL_IDENTITY_SHA256
            or file_sha256(source_snapshot) != source_sha
            or file_sha256(Path(__file__)) != source_sha):
        raise ValueError("numeric final source snapshots differ")
    require_resources(started, "final report")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "goal_met": bool(passed and metrics["top1"] == PROFILE["rows"]),
        "scope": "gated exposed final regression for numeric sentence-pool overlay",
        "profile": PROFILE, "runtime_version": RUNTIME_VERSION,
        "development_report_sha256": development_sha,
        "training_report_sha256": TRAIN_REPORT_SHA256,
        "accepted_acronym_final_report_sha256": ACRONYM_FINAL_REPORT_SHA256,
        "accepted_acronym_final_cases_sha256":
        sha256_bytes(final["acronym_cases_bytes"]),
        "source_hashes": fixed_hashes, "source_hashes_after": after,
        "final_source_sha256": FINAL_SOURCE_SHA256,
        "final_identity_sha256": FINAL_IDENTITY_SHA256,
        "source_snapshot_sha256": file_sha256(final_snapshot),
        "identity_snapshot_sha256": file_sha256(identity_snapshot),
        "cases_sha256": file_sha256(cases_path),
        "metrics": metrics,
        "outcomes": outcomes,
        "overlay": {
            "applied_rows": sum(row["overlay_applied"] for row in outcomes),
            "visible_changed_rows": sum(row["visible_candidates_changed"] for row in outcomes),
            "pool_additions": sum(len(row["pool_additions"]) for row in outcomes),
            "baseline_pool_and_score_prefix_preserved": all(
                row["baseline_pool_and_score_prefix_preserved"] for row in outcomes
            ),
            "word_options_identity_preserved": True, "beam_hook_restored": True,
        },
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
        "labels_used_for_candidate_production": False,
        "final_data_read": True,
        "new_weights_created": False, "deployed": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"passed": passed, "goal_met": report["goal_met"],
                      "metrics": metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-report-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
