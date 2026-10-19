"""Post-integration parity for the accepted numeric sentence-pool overlay."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hanglish.core.neural_converter import NeuralConverter, file_sha256
from scripts.evaluate_current_word_ranker import DEV_COUNTS, summarize
from scripts.evaluate_numeric_structure_development import (
    PROFILE as DEVELOPMENT_PROFILE,
    development_gate,
    train_metrics,
    validate_train_gate,
)
from scripts.evaluate_numeric_structure_final import (
    PROFILE as FINAL_PROFILE,
    final_gate,
    validate_development_report,
)
from scripts.probe_numeric_structure_runtime import SentencePoolRecorder
from scripts.train_natural_sentence_ranker import rss_bytes


OUTPUT = ROOT / ".omx/artifacts/numeric-structure-integration-v1"
ACCEPTANCE = OUTPUT / "accepted-experiment.json"
ACCEPTANCE_SHA256 = "4cd1ae7375c1458bf4758712acd81ccb81c8c1026d1eed27f1eb26e42beededb"
PARENT_SOURCE = OUTPUT / "parent-neural-converter.py"
PARENT_SOURCE_SHA256 = "acc7f5f4c43d1100d706593867bb99345d279054e7c3e85a90cb6c37a04cca56"
REPORTS = {
    "train": (ROOT / ".omx/artifacts/numeric-structure-runtime-v1/report.json",
              "ecbe82f14f96446b5ae8e660244393ce9c3293cfd4d6cb1db388fbe9ae58fe88",
              ROOT / ".omx/artifacts/numeric-structure-runtime-v1/cases.jsonl",
              "50550395a81a82635b4f0cf0ff5d1840e28f221d86426716cd6ff70e0bc50557"),
    "development": (ROOT / ".omx/artifacts/numeric-structure-development-v1/report.json",
                    "750e777f784e3bbdd2e79246b8abc68532e4840dc18e6c9f035da3e62a9b14e0",
                    ROOT / ".omx/artifacts/numeric-structure-development-v1/cases.jsonl",
                    "2baaccfbca4d7f4370a9c0b61c548fa9d4e33c50440c38c6dabdcd67d2ca920c"),
    "final": (ROOT / ".omx/artifacts/numeric-structure-final-v1/report.json",
              "c8521df6b1dfdc5b64c0b0ea9d70f08359a7b5e18faa83be380d16290dcb554b",
              ROOT / ".omx/artifacts/numeric-structure-final-v1/cases.jsonl",
              "acf3ad84599e34485127b7dea4281af38b5349f0e2c5ad634283ff91a1520529"),
}
PARENT_VERSION = "neural-1-5bcbab126cae"
DECLARED_CHANGED_SOURCE = "hanglish/core/neural_converter.py"
DECLARED_NEW_SOURCE = "hanglish/core/numeric_structure.py"
VERIFIER_SOURCE = "scripts/verify_numeric_structure_integration.py"
PROFILE = {
    "name": "numeric-structure-post-integration-parity-v1",
    "train_rows": 512,
    "development_rows": sum(DEV_COUNTS.values()),
    "final_rows": FINAL_PROFILE["rows"],
    "total_rows": 512 + sum(DEV_COUNTS.values()) + FINAL_PROFILE["rows"],
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 1800,
    "progress_interval": 32,
    "score_tolerance": 1e-4,
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read_artifact(path, expected_sha, *, jsonl=False):
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha:
        raise ValueError(f"accepted numeric artifact hash differs: {path}")
    parsed = ([json.loads(line) for line in raw.splitlines() if line]
              if jsonl else json.loads(raw))
    return raw, parsed


def load_accepted(explicit_hashes):
    if explicit_hashes.get("acceptance") != ACCEPTANCE_SHA256:
        raise ValueError("explicit numeric acceptance SHA differs")
    acceptance_raw, acceptance = read_artifact(ACCEPTANCE, ACCEPTANCE_SHA256)
    if file_sha256(PARENT_SOURCE) != PARENT_SOURCE_SHA256:
        raise ValueError("numeric parent source snapshot differs")
    reports, cases, snapshots = {}, {}, {
        str(ACCEPTANCE.relative_to(ROOT)): sha256_bytes(acceptance_raw),
        str(PARENT_SOURCE.relative_to(ROOT)): PARENT_SOURCE_SHA256,
    }
    for name, (report_path, report_sha, cases_path, cases_sha) in REPORTS.items():
        if explicit_hashes.get(name) != report_sha:
            raise ValueError(f"explicit accepted {name} report SHA differs")
        _, reports[name] = read_artifact(report_path, report_sha)
        _, cases[name] = read_artifact(cases_path, cases_sha, jsonl=True)
        snapshots[str(report_path.relative_to(ROOT))] = report_sha
        snapshots[str(cases_path.relative_to(ROOT))] = cases_sha
    validate_acceptance(acceptance, reports)
    validate_reports(reports, cases)
    return acceptance, reports, cases, dict(sorted(snapshots.items()))


def validate_acceptance(accepted, reports):
    for name, (_, report_sha, _, cases_sha) in REPORTS.items():
        values = accepted.get("reports", {}).get(name, {})
        if values.get("sha256") != report_sha or values.get("cases_sha256") != cases_sha:
            raise ValueError("acceptance snapshot report lineage differs")
    if (accepted.get("parent_runtime_version") != PARENT_VERSION
            or accepted.get("parent_source_sha256") != PARENT_SOURCE_SHA256):
        raise ValueError("acceptance snapshot parent identity differs")
    weights = accepted.get("model_artifacts")
    if (not isinstance(weights, dict) or len(weights) != 16
            or accepted.get("score_tolerance") != PROFILE["score_tolerance"]
            or not isinstance(accepted.get("policy"), str)
            or any(report.get("source_hashes", {}).get(path) != digest
                   for report in reports.values() for path, digest in weights.items())):
        raise ValueError("acceptance snapshot requires sixteen model artifacts")


def validate_reports(reports, cases):
    train, development, final = reports["train"], reports["development"], reports["final"]
    validate_train_gate(train, cases["train"], REPORTS["train"][1], REPORTS["train"][1])
    validate_development_report(
        development, cases["development"], REPORTS["development"][1],
        REPORTS["development"][1], REPORTS["development"][3],
    )
    if (final.get("passed") is not True
            or final.get("scope") !=
            "gated exposed final regression for numeric sentence-pool overlay"
            or final.get("runtime_version") != PARENT_VERSION
            or final.get("development_report_sha256") != REPORTS["development"][1]
            or final.get("training_report_sha256") != REPORTS["train"][1]
            or final.get("cases_sha256") != REPORTS["final"][3]
            or final.get("source_hashes") != final.get("source_hashes_after")
            or final.get("labels_used_for_candidate_production") is not False
            or final.get("new_weights_created") is not False
            or final.get("deployed") is not False
            or final.get("outcomes") != cases["final"]
            or len(cases["final"]) != FINAL_PROFILE["rows"]
            or len({row.get("id") for row in cases["final"]}) != len(cases["final"])):
        raise ValueError("accepted numeric final report is incomplete")
    final_summary = summarize(cases["final"])
    if final_summary != final.get("metrics") or not final_gate(final_summary):
        raise ValueError("accepted numeric final metrics do not reproduce")
    if train_metrics(cases["train"]) != train.get("metrics"):
        raise ValueError("accepted numeric TRAIN metrics do not reproduce")
    development_groups = {
        name: [row for row in cases["development"] if row.get("suite") == name]
        for name in DEV_COUNTS
    }
    rebuilt = {name: summarize(rows) for name, rows in development_groups.items()}
    if rebuilt != development.get("suites") or not development_gate(rebuilt):
        raise ValueError("accepted numeric development metrics do not reproduce")


def candidate_objects_equal(actual, expected, tolerance=PROFILE["score_tolerance"]):
    if not isinstance(actual, list) or not isinstance(expected, list) or len(actual) != len(expected):
        return False
    for left, right in zip(actual, expected):
        if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
            return False
        for key in left:
            if key != "score":
                if left[key] != right[key]:
                    return False
                continue
            if left[key] is None or right[key] is None:
                if left[key] is not right[key]:
                    return False
            elif (type(left[key]) not in (int, float)
                  or type(right[key]) not in (int, float)
                  or not math.isfinite(left[key]) or not math.isfinite(right[key])
                  or abs(left[key] - right[key]) > tolerance):
                return False
    return True


def score_lists_equal(actual, expected, tolerance=PROFILE["score_tolerance"]):
    if actual is None or expected is None:
        return actual is expected
    return (len(actual) == len(expected)
            and all(type(left) in (int, float) and type(right) in (int, float)
                    and math.isfinite(left) and math.isfinite(right)
                    and abs(left - right) <= tolerance
                    for left, right in zip(actual, expected)))


def reference_fields(reference, role):
    if role == "train":
        return (reference["child_candidates"], reference["child_sentence_pool"],
                reference["child_sentence_scores"], reference["parent_top1"],
                reference["child_top1"], reference["parent_top5"],
                reference["child_top5"], reference["parent_pool_covered"],
                reference["child_pool_covered"])
    return (reference["candidates"], reference["sentence_pool"],
            reference["sentence_scores"], reference["before_top1"],
            reference["top1"], reference["before_top5"], reference["top5"],
            reference["before_sentence_pool_covered"],
            reference["sentence_pool_covered"])


def verify_case(converter, recorder, reference, role, expected_version, expected_weights):
    expected_candidates, expected_pool, expected_scores, before_top1, reference_top1, \
        before_top5, reference_top5, before_pool, reference_pool_covered = reference_fields(
            reference, role
        )
    recorder.reset()
    converted = converter.convert(reference["input"], reference["context"])
    pool, scores = recorder.candidates, recorder.scores
    if (converted.get("version") != expected_version
            or converted.get("model_artifacts") != expected_weights
            or not candidate_objects_equal(converted.get("candidates"), expected_candidates)
            or pool != expected_pool or not score_lists_equal(scores, expected_scores)
            or converted.get("sentence_ranking_applied") != (pool is not None)):
        raise ValueError(f"integrated numeric runtime differs at {role}:{reference.get('id')}")
    # Labels are consulted only after complete object, pool, score, and version parity.
    expected = reference["expected"]
    texts = [candidate["text"] for candidate in converted["candidates"]]
    top1 = texts[0] in expected
    top5 = any(text in expected for text in texts)
    pool_covered = None if pool is None else any(text in expected for text in pool)
    if (top1 is not reference_top1 or top5 is not reference_top5
            or pool_covered is not reference_pool_covered):
        raise ValueError("integrated numeric labels do not reproduce accepted outcomes")
    suite = "train" if role == "train" else reference.get("suite")
    if not isinstance(suite, str):
        raise ValueError("integrated numeric reference suite is missing")
    return {
        "id": reference["id"], "suite": suite,
        "candidate_objects_equal": True, "sentence_pool_equal": True,
        "sentence_scores_within_tolerance": True, "version_equal": True,
        "before_top1": before_top1, "top1": top1,
        "before_top5": before_top5, "top5": top5,
        "before_sentence_pool_covered": before_pool,
        "sentence_pool_covered": pool_covered,
    }


def validate_source_transition(accepted_sources, weights, verifier_sha):
    python_sources = {name: digest for name, digest in accepted_sources.items()
                      if name.endswith(".py")}
    if python_sources.get(DECLARED_CHANGED_SOURCE) != PARENT_SOURCE_SHA256:
        raise ValueError("accepted parent core hash differs from its snapshot")
    for name, expected in python_sources.items():
        if name == DECLARED_CHANGED_SOURCE:
            continue
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"undeclared accepted source changed: {name}")
    if not (ROOT / DECLARED_NEW_SOURCE).is_file():
        raise ValueError("integrated numeric runtime source is missing")
    current = {name: file_sha256(ROOT / name) for name in python_sources}
    current[DECLARED_NEW_SOURCE] = file_sha256(ROOT / DECLARED_NEW_SOURCE)
    current[VERIFIER_SOURCE] = verifier_sha
    if current[DECLARED_CHANGED_SOURCE] == PARENT_SOURCE_SHA256:
        raise ValueError("numeric runtime integration did not change parent core")
    actual_weights = {name: file_sha256(ROOT / name) for name in weights}
    if actual_weights != weights:
        raise ValueError("integrated numeric runtime model artifacts changed")
    return dict(sorted(current.items())), dict(sorted(actual_weights.items()))


def prototype_hashes(snapshots):
    return {name: file_sha256(ROOT / name) for name in snapshots}


def require_resources(started, stage):
    if (rss_bytes() > PROFILE["rss_limit_bytes"]
            or time.monotonic() - started > PROFILE["time_limit_seconds"]):
        raise RuntimeError(f"numeric integration resource bound exceeded during {stage}")


def run(args):
    started = time.monotonic()
    owned = tuple(args.output / name for name in ("report.json", "cases.jsonl", "source.py"))
    if any(path.exists() for path in owned):
        raise FileExistsError("numeric integration verifier output already exists")
    source = Path(__file__).read_bytes()
    source_sha = sha256_bytes(source)
    explicit = {"acceptance": args.acceptance_sha256, "train": args.train_report_sha256,
                "development": args.development_report_sha256,
                "final": args.final_report_sha256}
    acceptance, reports, cases, snapshots = load_accepted(explicit)
    weights = acceptance["model_artifacts"]
    accepted_sources = reports["final"]["source_hashes"]
    runtime_before, weights_before = validate_source_transition(
        accepted_sources, weights, source_sha
    )
    artifacts_before = prototype_hashes(snapshots)
    references = [
        *(('train', row) for row in cases["train"]),
        *(('development', row) for row in cases["development"]),
        *(('final', row) for row in cases["final"]),
    ]
    if len(references) != PROFILE["total_rows"]:
        raise ValueError("accepted numeric references do not contain 1880 rows")

    torch.set_num_threads(PROFILE["threads"])
    os.environ["OMP_NUM_THREADS"] = str(PROFILE["threads"])
    converter = NeuralConverter(ranker_kind="hybrid")
    converter.load_models()
    integrated_version = converter.version
    if converter.model_artifacts != weights or integrated_version == PARENT_VERSION:
        raise ValueError("integrated runtime version or model artifacts differ")
    recorder = SentencePoolRecorder(converter.sentence_ranker)
    converter.sentence_ranker = recorder
    outcomes = []
    for index, (role, reference) in enumerate(references):
        outcome = verify_case(
            converter, recorder, reference, role, integrated_version, weights
        )
        outcomes.append(outcome)
        require_resources(started, "runtime parity")
        if (index + 1) % PROFILE["progress_interval"] == 0:
            print(json.dumps({"event": "numeric_integration", "rows": index + 1,
                              "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)

    artifacts_after = prototype_hashes(snapshots)
    runtime_after, weights_after = validate_source_transition(
        accepted_sources, weights, source_sha
    )
    if (artifacts_after != artifacts_before or runtime_after != runtime_before
            or weights_after != weights_before or converter.version != integrated_version
            or converter.model_artifacts != weights):
        raise ValueError("numeric integration sources, artifacts, weights, or version changed")
    require_resources(started, "post-validation")
    actual_suites = {
        "train": summarize([row for row in outcomes if row["suite"] == "train"]),
        **{name: summarize([row for row in outcomes if row["suite"] == name])
           for name in DEV_COUNTS},
        "final_v22": summarize([row for row in outcomes if row["suite"] == "final_v22"]),
    }
    reference_suites = {
        "train": summarize([{
            "before_top1": row["parent_top1"], "top1": row["child_top1"],
            "before_top5": row["parent_top5"], "top5": row["child_top5"],
            "before_sentence_pool_covered": row["parent_pool_covered"],
            "sentence_pool_covered": row["child_pool_covered"], "id": row["id"],
        } for row in cases["train"]]),
        **reports["development"]["suites"],
        "final_v22": reports["final"]["metrics"],
    }
    if actual_suites != reference_suites:
        raise ValueError("integrated numeric runtime suite metrics differ")

    parity_flags = {
        "all_candidate_objects_equal": all(
            row["candidate_objects_equal"] for row in outcomes
        ),
        "all_sentence_pools_equal": all(row["sentence_pool_equal"] for row in outcomes),
        "all_sentence_scores_within_tolerance": all(
            row["sentence_scores_within_tolerance"] for row in outcomes
        ),
        "all_versions_equal": all(row["version_equal"] for row in outcomes),
    }
    passed = (len(outcomes) == PROFILE["total_rows"]
              and actual_suites == reference_suites
              and all(parity_flags.values()))
    if not passed:
        raise ValueError("integrated numeric runtime parity is incomplete")

    args.output.mkdir(parents=True, exist_ok=True)
    cases_path = args.output / "cases.jsonl"
    with cases_path.open("x") as stream:
        for row in outcomes:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    source_path = args.output / "source.py"
    source_path.write_bytes(source)
    if file_sha256(source_path) != source_sha or file_sha256(Path(__file__)) != source_sha:
        raise ValueError("numeric integration verifier source changed")
    require_resources(started, "final report")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "scope": "post-integration actual runtime parity",
        "integrated_runtime_version": integrated_version,
        "integrated_runtime_source_hashes": runtime_before,
        "runtime_source_hashes_after": runtime_after,
        "accepted_train_report_sha256": REPORTS["train"][1],
        "accepted_development_report_sha256": REPORTS["development"][1],
        "accepted_final_report_sha256": REPORTS["final"][1],
        "acceptance_snapshot_sha256": ACCEPTANCE_SHA256,
        "accepted_weight_hashes": weights_before,
        "prototype_artifact_hashes": artifacts_before,
        "profile": PROFILE, "rows": len(outcomes),
        "suite_rows": {name: sum(row["suite"] == name for row in outcomes)
                       for name in actual_suites},
        **parity_flags,
        "actual_suites": actual_suites, "reference_suites": reference_suites,
        "cases_sha256": file_sha256(cases_path),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(), "deployed": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"passed": report["passed"], "rows": report["rows"],
                      "suite_rows": report["suite_rows"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acceptance-sha256", required=True)
    parser.add_argument("--train-report-sha256", required=True)
    parser.add_argument("--development-report-sha256", required=True)
    parser.add_argument("--final-report-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
