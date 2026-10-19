"""Gated final regression for the source-only acronym suffix prototype."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hanglish.core.neural_converter import NeuralConverter, file_sha256
from scripts.evaluate_acronym_suffix_candidates import (
    OUTPUT as DEVELOPMENT_OUTPUT,
    PREPARATION_MANIFEST,
    PROFILE as DEVELOPMENT_PROFILE,
    TRAIN_PROBE,
    TRAIN_PROBE_SHA256,
    TRAIN_POLICY,
    SentencePoolRecorder,
    development_passed,
    evaluate_case,
    load_fixed_suites,
    read_jsonl,
    policy_fingerprint,
    source_hashes as development_source_hashes,
    suite_hashes,
    summarize,
    validate_suite_hashes,
    validate_train_gate,
)
from scripts.prepare_current_context_data import (
    OUTPUT as PREPARED,
    RUNTIME_VERSION,
    source_hashes as preparation_source_hashes,
    validate_data,
)
from scripts.probe_acronym_suffix_candidates import install, validate_audits
from scripts.train_natural_sentence_ranker import rss_bytes


FINAL_SOURCE = ROOT / "data/challenge/v2.2/test.jsonl"
FINAL_SOURCE_SHA256 = "3d2fe696cb62e3e895017f817e8f5ccca8a651f17a325725ef624b6d3335c720"
FINAL_IDENTITY = ROOT / ".omx/artifacts/challenge-v22-test-natural-v3-deployed.json"
FINAL_IDENTITY_SHA256 = "caa3a5783de91e86493c1b354abbbf0e0003b611b794d9ccd97114639b7bc264"
DEVELOPMENT_REPORT_SHA256 = "2b485a219af24e918052563198138abc6b293773063407f7609075500da21954"
OUTPUT = ROOT / ".omx/artifacts/acronym-suffix-final-v1"
PROFILE = {
    "name": "acronym-suffix-gated-final-v1",
    "rows": 1000,
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 1800,
    "progress_interval": 32,
}


def expected_development_state():
    hashes = suite_hashes()
    validate_suite_hashes(hashes)
    return {
        "source_hashes": development_source_hashes(),
        "preparation_source_hashes": preparation_source_hashes(),
        "development_suite_hashes": hashes,
    }


def validate_development_report(report, explicit_sha, actual_sha, state,
                                cases_sha256, option_audit_sha256):
    if (explicit_sha != DEVELOPMENT_REPORT_SHA256 or actual_sha != explicit_sha
            or report.get("passed") is not True
            or report.get("eligible_for_further_evaluation") is not True
            or report.get("scope") != "fresh actual hybrid development only"
            or report.get("parent_version") != RUNTIME_VERSION
            or report.get("training_probe_sha256") != TRAIN_PROBE_SHA256
            or report.get("preparation_manifest_sha256") != PREPARATION_MANIFEST
            or report.get("experimental_policy_sha256") != policy_fingerprint()
            or report.get("source_state") != state
            or report.get("cases_sha256") != cases_sha256
            or report.get("child_option_audit", {}).get("sha256") != option_audit_sha256
            or report.get("test_data_read") is not False
            or report.get("deployed") is not False):
        raise ValueError("complete approved acronym development report is required")
    outcomes = report.get("outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != sum(
            DEVELOPMENT_PROFILE["suite_counts"].values()):
        raise ValueError("acronym development outcomes are incomplete")
    rebuilt = {}
    for name, count in DEVELOPMENT_PROFILE["suite_counts"].items():
        rows = [row for row in outcomes if row.get("suite") == name]
        if len(rows) != count:
            raise ValueError("acronym development suite identities are incomplete")
        rebuilt[name] = summarize(rows)
    if rebuilt != report.get("suites") or not development_passed(rebuilt):
        raise ValueError("acronym development metrics or regression gate differ")


def validate_development_option_audit(report, rows):
    keyed = {(row.get("source"), row.get("learned")): row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError("development option audit contains duplicate keys")
    validate_audits(keyed)
    changed = [row for row in rows if row.get("additions")]
    expected = {
        "sha256": file_sha256(DEVELOPMENT_OUTPUT / "option-audit.jsonl"),
        "unique_calls": len(rows),
        "changed_calls": len(changed),
        "added_candidates": sum(len(row["additions"]) for row in changed),
        "max_added": max((len(row["additions"]) for row in changed), default=0),
        "baseline_prefix_preserved": True,
        "scores_unmodified_from_suffix_options": True,
        "all_additions_forward_valid": True,
    }
    if report.get("child_option_audit") != expected:
        raise ValueError("development option audit summary differs")


def validate_training_probe(data):
    report_path = TRAIN_PROBE / "report.json"
    actual_sha = file_sha256(report_path)
    report = json.loads(report_path.read_text())
    cases = read_jsonl(TRAIN_PROBE / "cases.jsonl")
    selection = json.loads((TRAIN_PROBE / "selection.json").read_text())
    audits = read_jsonl(TRAIN_PROBE / "option-audit.jsonl")
    validate_train_gate(report, TRAIN_PROBE_SHA256, actual_sha, cases, selection,
                        audits, data["natural_train"][:TRAIN_POLICY["rows"]])


def read_final_source(authorized, source=FINAL_SOURCE, identity_path=FINAL_IDENTITY):
    """Open final case content only after the development authorization succeeds."""
    if authorized is not True:
        raise PermissionError("approved development gate is required before final read")
    raw_identity = identity_path.read_bytes()
    if hashlib.sha256(raw_identity).hexdigest() != FINAL_IDENTITY_SHA256:
        raise ValueError("final identity report differs from its explicit pin")
    identity = json.loads(raw_identity)
    identity_source_sha = identity.get("sha256") or identity.get("corpus_identity", {}).get("sha256")
    raw = source.read_bytes()
    if (identity_source_sha != FINAL_SOURCE_SHA256
            or hashlib.sha256(raw).hexdigest() != FINAL_SOURCE_SHA256):
        raise ValueError("final source differs from its fixed identity")
    rows = [json.loads(line) for line in raw.splitlines() if line]
    if len(rows) != PROFILE["rows"]:
        raise ValueError("final source does not contain exactly 1000 rows")
    return raw, rows, raw_identity


def final_passed(metrics):
    return bool(
        metrics.get("rows") == PROFILE["rows"]
        and metrics.get("before_top5") == PROFILE["rows"]
        and metrics.get("top5") == PROFILE["rows"]
        and not metrics.get("regressions")
        and not metrics.get("top5_regressions")
        and not metrics.get("sentence_pool_coverage_losses")
    )


def require_resources(started, stage):
    if (rss_bytes() > PROFILE["rss_limit_bytes"]
            or time.monotonic() - started > PROFILE["time_limit_seconds"]):
        raise RuntimeError(f"acronym suffix final resource bound exceeded during {stage}")


def finalize_gate(metrics, before, after, started):
    if before != after:
        raise ValueError("final evaluation source, model, probe, or gate changed")
    require_resources(started, "post-validation")
    return final_passed(metrics)


def source_hashes(development_report):
    hashes = development_source_hashes()
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/evaluate_acronym_suffix_candidates.py",
        development_report,
        DEVELOPMENT_OUTPUT / "cases.jsonl",
        DEVELOPMENT_OUTPUT / "option-audit.jsonl",
        DEVELOPMENT_OUTPUT / "source.py",
        TRAIN_PROBE / "report.json", TRAIN_PROBE / "selection.json",
        TRAIN_PROBE / "cases.jsonl", TRAIN_PROBE / "option-audit.jsonl",
        PREPARED / "manifest.json",
        FINAL_SOURCE, FINAL_IDENTITY,
    )
    for path in paths:
        hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    return dict(sorted(hashes.items()))


def run(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    development_path = DEVELOPMENT_OUTPUT / "report.json"
    development_sha = file_sha256(development_path)
    development = json.loads(development_path.read_text())
    state = expected_development_state()
    validate_development_report(
        development, args.development_report_sha256, development_sha, state,
        file_sha256(DEVELOPMENT_OUTPUT / "cases.jsonl"),
        file_sha256(DEVELOPMENT_OUTPUT / "option-audit.jsonl"),
    )
    validate_development_option_audit(
        development, read_jsonl(DEVELOPMENT_OUTPUT / "option-audit.jsonl")
    )
    manifest, data = validate_data(PREPARED, PREPARATION_MANIFEST)
    if manifest.get("runtime_version") != RUNTIME_VERSION:
        raise ValueError("current preparation runtime differs")
    _, expected_development_sources = load_fixed_suites(data["natural_dev"])
    if development.get("sources") != expected_development_sources:
        raise ValueError("development report source identities differ")
    validate_training_probe(data)
    # No final content is opened before every development and TRAIN gate above.
    raw_final, cases, raw_identity = read_final_source(True)
    before = {
        "source_hashes": source_hashes(development_path),
        "development_state": state,
        "development_report_sha256": development_sha,
    }
    torch.set_num_threads(PROFILE["threads"])
    converter = NeuralConverter(ranker_kind="hybrid")
    converter.load_models()
    if converter.version != RUNTIME_VERSION:
        raise ValueError("actual hybrid parent runtime differs")
    parent_options = converter.word_options
    child_audit = install(converter)
    child_options = converter.word_options
    recorder = SentencePoolRecorder(converter.sentence_ranker)
    converter.sentence_ranker = recorder
    args.output.mkdir(parents=True)
    source_snapshot = args.output / "final-source.jsonl"
    identity_snapshot = args.output / "final-identity.json"
    source_snapshot.write_bytes(raw_final)
    identity_snapshot.write_bytes(raw_identity)
    cases_path = args.output / "cases.jsonl"
    outcomes = []
    with cases_path.open("x") as stream:
        for index, case in enumerate(cases):
            row = evaluate_case(converter, parent_options, child_options, recorder, case)
            row["suite"] = "final_v22"
            outcomes.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            require_resources(started, "final inference")
            if (index + 1) % PROFILE["progress_interval"] == 0:
                print(json.dumps({"suite": "final_v22", "completed": index + 1,
                                  "elapsed_seconds": round(time.monotonic() - started, 2)}),
                      flush=True)
    if converter.word_options is not parent_options:
        raise RuntimeError("prototype did not restore parent word options")
    validate_audits(child_audit)
    audit_path = args.output / "option-audit.jsonl"
    with audit_path.open("x") as stream:
        for key in sorted(child_audit):
            stream.write(json.dumps(child_audit[key], ensure_ascii=False, sort_keys=True) + "\n")
    validate_data(PREPARED, PREPARATION_MANIFEST)
    validate_training_probe(data)
    validate_development_report(
        json.loads(development_path.read_text()), args.development_report_sha256,
        file_sha256(development_path), expected_development_state(),
        file_sha256(DEVELOPMENT_OUTPUT / "cases.jsonl"),
        file_sha256(DEVELOPMENT_OUTPUT / "option-audit.jsonl"),
    )
    current_final = FINAL_SOURCE.read_bytes()
    current_identity = FINAL_IDENTITY.read_bytes()
    snapshot_final = source_snapshot.read_bytes()
    snapshot_identity = identity_snapshot.read_bytes()
    if (hashlib.sha256(current_final).hexdigest() != FINAL_SOURCE_SHA256
            or hashlib.sha256(current_identity).hexdigest() != FINAL_IDENTITY_SHA256
            or hashlib.sha256(snapshot_final).hexdigest() != FINAL_SOURCE_SHA256
            or hashlib.sha256(snapshot_identity).hexdigest() != FINAL_IDENTITY_SHA256
            or snapshot_final != raw_final or snapshot_identity != raw_identity):
        raise ValueError("final source or saved snapshots changed")
    after = {
        "source_hashes": source_hashes(development_path),
        "development_state": expected_development_state(),
        "development_report_sha256": file_sha256(development_path),
    }
    metrics = summarize(outcomes)
    passed = finalize_gate(metrics, before, after, started)
    changed = [row for row in child_audit.values() if row["additions"]]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": bool(passed),
        "goal_met": bool(passed and metrics["top1"] == PROFILE["rows"]),
        "scope": "gated exposed final regression",
        "parent_version": RUNTIME_VERSION,
        "development_report_sha256": development_sha,
        "training_probe_sha256": TRAIN_PROBE_SHA256,
        "preparation_manifest_sha256": PREPARATION_MANIFEST,
        "source_state": before,
        "final_source_sha256": FINAL_SOURCE_SHA256,
        "final_identity_sha256": FINAL_IDENTITY_SHA256,
        "source_snapshot_sha256": file_sha256(source_snapshot),
        "identity_snapshot_sha256": file_sha256(identity_snapshot),
        "cases_sha256": file_sha256(cases_path),
        "metrics": metrics,
        "outcomes": outcomes,
        "child_option_audit": {
            "sha256": file_sha256(audit_path), "unique_calls": len(child_audit),
            "changed_calls": len(changed),
            "added_candidates": sum(len(row["additions"]) for row in changed),
            "max_added": max((len(row["additions"]) for row in changed), default=0),
        },
        "profile": PROFILE,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
        "deployed": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"passed": bool(passed), "goal_met": report["goal_met"],
                      "metrics": metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-report-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
