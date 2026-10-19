"""Post-integration runtime parity against accepted acronym prototype artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hanglish.core.neural_converter import NeuralConverter, artifact_version, file_sha256
from scripts.evaluate_acronym_suffix_candidates import (
    OUTPUT as DEVELOPMENT_OUTPUT,
    PREPARATION_MANIFEST,
    TRAIN_PROBE_SHA256,
    SentencePoolRecorder,
    development_passed,
    summarize,
)
from scripts.evaluate_acronym_suffix_final import OUTPUT as FINAL_OUTPUT, PROFILE as FINAL_PROFILE
from scripts.evaluate_roberta_context_ranker import DEV_COUNTS
from scripts.train_natural_sentence_ranker import rss_bytes


OUTPUT = ROOT / ".omx/artifacts/acronym-suffix-integration-v1"
ACCEPTANCE = OUTPUT / "accepted-experiment.json"
ACCEPTANCE_SHA256 = "f5095076f25f59de845e878b1315e884a59063235807b76981c454feed674aba"
PARENT_SOURCE = OUTPUT / "parent-neural-converter.py"
PARENT_SOURCE_SHA256 = "c637758dbbd670db7bff78596359caea2c9082fdc093667b13d2e3fcf85df8e9"
PROFILE = {
    "name": "acronym-suffix-post-integration-parity-v1",
    "development_rows": sum(DEV_COUNTS.values()),
    "final_rows": FINAL_PROFILE["rows"],
    "total_rows": sum(DEV_COUNTS.values()) + FINAL_PROFILE["rows"],
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 1800,
    "progress_interval": 32,
    "score_tolerance": 1e-5,
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def rebuilt_suites(outcomes):
    names = sorted({row.get("suite") for row in outcomes})
    return {name: summarize([row for row in outcomes if row.get("suite") == name])
            for name in names}


def validate_development(report, explicit_sha, actual_sha, cases_sha, audit_sha):
    outcomes = report.get("outcomes")
    if (explicit_sha != actual_sha or report.get("passed") is not True
            or report.get("eligible_for_further_evaluation") is not True
            or report.get("scope") != "fresh actual hybrid development only"
            or report.get("training_probe_sha256") != TRAIN_PROBE_SHA256
            or report.get("preparation_manifest_sha256") != PREPARATION_MANIFEST
            or report.get("cases_sha256") != cases_sha
            or report.get("child_option_audit", {}).get("sha256") != audit_sha
            or report.get("test_data_read") is not False
            or report.get("deployed") is not False
            or not isinstance(outcomes, list)
            or len(outcomes) != PROFILE["development_rows"]):
        raise ValueError("accepted complete acronym development artifact is required")
    suites = rebuilt_suites(outcomes)
    if (suites != report.get("suites") or set(suites) != set(DEV_COUNTS)
            or any(suites[name]["rows"] != count for name, count in DEV_COUNTS.items())
            or not development_passed(suites)):
        raise ValueError("acronym development metrics do not reproduce")


def validate_final(report, explicit_sha, actual_sha, development_sha,
                   cases_sha, audit_sha, source_snapshot_sha, identity_snapshot_sha):
    outcomes = report.get("outcomes")
    if (explicit_sha != actual_sha or report.get("passed") is not True
            or report.get("scope") != "gated exposed final regression"
            or report.get("development_report_sha256") != development_sha
            or report.get("training_probe_sha256") != TRAIN_PROBE_SHA256
            or report.get("preparation_manifest_sha256") != PREPARATION_MANIFEST
            or report.get("cases_sha256") != cases_sha
            or report.get("child_option_audit", {}).get("sha256") != audit_sha
            or report.get("source_snapshot_sha256") != source_snapshot_sha
            or report.get("identity_snapshot_sha256") != identity_snapshot_sha
            or report.get("deployed") is not False
            or not isinstance(outcomes, list) or len(outcomes) != PROFILE["final_rows"]):
        raise ValueError("accepted complete acronym final artifact is required")
    rebuilt = summarize(outcomes)
    if rebuilt != report.get("metrics") or rebuilt.get("rows") != PROFILE["final_rows"]:
        raise ValueError("acronym final metrics do not reproduce")
    if (rebuilt.get("before_top5") != PROFILE["final_rows"]
            or rebuilt.get("top5") != PROFILE["final_rows"]
            or rebuilt.get("regressions") or rebuilt.get("top5_regressions")
            or rebuilt.get("sentence_pool_coverage_losses")):
        raise ValueError("accepted final artifact has a regression or coverage loss")


def validate_acceptance_snapshot(explicit_development_sha, explicit_final_sha,
                                 development_report, final_report):
    if (file_sha256(ACCEPTANCE) != ACCEPTANCE_SHA256
            or file_sha256(PARENT_SOURCE) != PARENT_SOURCE_SHA256):
        raise ValueError("accepted integration snapshot or parent source changed")
    accepted = json.loads(ACCEPTANCE.read_text())
    if (accepted.get("development_sha256") != explicit_development_sha
            or accepted.get("final_sha256") != explicit_final_sha
            or accepted.get("train_sha256") != TRAIN_PROBE_SHA256
            or accepted.get("parent_source_sha256") != PARENT_SOURCE_SHA256
            or accepted.get("model_weights_changed") is not False
            or accepted.get("parent_version") != development_report.get("parent_version")
            or accepted.get("final_metrics") != final_report.get("metrics")):
        raise ValueError("explicit reports differ from accepted integration experiment")
    return accepted


def candidate_objects_equal(actual, expected, tolerance=PROFILE["score_tolerance"]):
    if not isinstance(actual, list) or not isinstance(expected, list) or len(actual) != len(expected):
        return False
    for left, right in zip(actual, expected):
        if not isinstance(left, dict) or not isinstance(right, dict) or set(left) != set(right):
            return False
        for key in left:
            if key == "score":
                if left[key] is None or right[key] is None:
                    if left[key] is not right[key]:
                        return False
                    continue
                if (type(left[key]) not in (int, float) or type(right[key]) not in (int, float)
                        or not math.isfinite(left[key]) or not math.isfinite(right[key])
                        or abs(left[key] - right[key]) > tolerance):
                    return False
            elif left[key] != right[key]:
                return False
    return True


def score_lists_equal(actual, expected, tolerance=PROFILE["score_tolerance"]):
    if actual is None or expected is None:
        return actual is expected
    return (len(actual) == len(expected)
            and all(math.isfinite(float(left)) and math.isfinite(float(right))
                    and abs(float(left) - float(right)) <= tolerance
                    for left, right in zip(actual, expected)))


def verify_case(converter, recorder, reference):
    recorder.candidates = recorder.scores = None
    converted = converter.convert(reference["input"], reference["context"])
    pool, scores = recorder.candidates, recorder.scores
    # The expected spelling is intentionally not needed for parity selection.
    if (not candidate_objects_equal(converted.get("candidates"), reference.get("candidates"))
            or pool != reference.get("sentence_pool")
            or not score_lists_equal(scores, reference.get("sentence_scores"))
            or converted.get("sentence_ranking_applied") != (pool is not None)):
        raise ValueError(f"integrated runtime differs from prototype case {reference.get('id')}")
    expected = reference["expected"]
    texts = [candidate["text"] for candidate in converted["candidates"]]
    return {"id": reference["id"], "suite": reference["suite"],
            "candidate_objects_equal": True, "sentence_pool_equal": True,
            "sentence_scores_within_tolerance": True,
            "before_top1": reference["before_top1"],
            "top1": texts[0] in expected,
            "before_top5": reference["before_top5"],
            "top5": any(text in expected for text in texts),
            "before_sentence_pool_covered": reference["before_sentence_pool_covered"],
            "sentence_pool_covered": (None if pool is None else
                                       any(text in expected for text in pool))}


def prototype_hashes():
    paths = [path for directory in (DEVELOPMENT_OUTPUT, FINAL_OUTPUT)
             for path in directory.iterdir() if path.is_file()]
    paths.extend((ACCEPTANCE, PARENT_SOURCE))
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in sorted(paths)}


def accepted_weight_hashes(development_report):
    hashes = development_report.get("source_state", {}).get("source_hashes", {})
    weights = {name: digest for name, digest in hashes.items()
               if name.startswith("checkpoints/")}
    if not weights or not any(name.endswith("model.safetensors") for name in weights):
        raise ValueError("accepted development artifact lacks model weight identities")
    return dict(sorted(weights.items()))


def current_weight_hashes(expected):
    result = {}
    for name in expected:
        path = ROOT / name
        if not path.is_file():
            raise ValueError("an accepted model artifact is missing")
        result[name] = file_sha256(path)
    return dict(sorted(result.items()))


def current_runtime_source_hashes(accepted_source_hashes):
    names = {name for name in accepted_source_hashes if name.endswith(".py")}
    names.add(str(Path(__file__).resolve().relative_to(ROOT)))
    result = {}
    for name in sorted(names):
        path = ROOT / name
        if not path.is_file():
            raise ValueError("an accepted runtime source file is missing")
        result[name] = file_sha256(path)
    return result


def validate_runtime_identity(initial_hashes, accepted_source_hashes,
                              initial_version, converter):
    current_hashes = current_runtime_source_hashes(accepted_source_hashes)
    current_version = artifact_version(converter.generator.external_fingerprint)
    if (current_hashes != initial_hashes or current_version != initial_version
            or converter.version != initial_version):
        raise ValueError("integrated runtime source or artifact version changed")
    return current_hashes


def require_resources(started, stage):
    if (rss_bytes() > PROFILE["rss_limit_bytes"]
            or time.monotonic() - started > PROFILE["time_limit_seconds"]):
        raise RuntimeError(f"acronym integration resource bound exceeded during {stage}")


def run(args):
    started = time.monotonic()
    owned_outputs = tuple(args.output / name for name in
                          ("cases.jsonl", "report.json", "source.py"))
    if any(path.exists() for path in owned_outputs):
        raise FileExistsError("integration verifier output already exists")
    development_path = DEVELOPMENT_OUTPUT / "report.json"
    final_path = FINAL_OUTPUT / "report.json"
    development_sha = file_sha256(development_path)
    final_sha = file_sha256(final_path)
    development = json.loads(development_path.read_text())
    final = json.loads(final_path.read_text())
    validate_development(
        development, args.development_report_sha256, development_sha,
        file_sha256(DEVELOPMENT_OUTPUT / "cases.jsonl"),
        file_sha256(DEVELOPMENT_OUTPUT / "option-audit.jsonl"),
    )
    validate_final(
        final, args.final_report_sha256, final_sha, development_sha,
        file_sha256(FINAL_OUTPUT / "cases.jsonl"),
        file_sha256(FINAL_OUTPUT / "option-audit.jsonl"),
        file_sha256(FINAL_OUTPUT / "final-source.jsonl"),
        file_sha256(FINAL_OUTPUT / "final-identity.json"),
    )
    accepted = validate_acceptance_snapshot(
        args.development_report_sha256, args.final_report_sha256, development, final
    )
    before = prototype_hashes()
    accepted_weights = accepted_weight_hashes(development)
    if current_weight_hashes(accepted_weights) != accepted_weights:
        raise ValueError("integrated runtime model weights differ from accepted prototype")
    accepted_sources = development["source_state"]["source_hashes"]
    runtime_sources_before = current_runtime_source_hashes(accepted_sources)
    references = [*development["outcomes"], *final["outcomes"]]
    if len(references) != PROFILE["total_rows"]:
        raise ValueError("accepted prototype does not cover all 1368 rows")
    torch.set_num_threads(PROFILE["threads"])
    converter = NeuralConverter(ranker_kind="hybrid")
    converter.load_models()
    integrated_version = converter.version
    recorder = SentencePoolRecorder(converter.sentence_ranker)
    converter.sentence_ranker = recorder
    args.output.mkdir(parents=True, exist_ok=True)
    cases_path = args.output / "cases.jsonl"
    outcomes = []
    with cases_path.open("x") as stream:
        for index, reference in enumerate(references):
            outcome = verify_case(converter, recorder, reference)
            outcomes.append(outcome)
            stream.write(json.dumps(outcome, ensure_ascii=False, sort_keys=True) + "\n")
            require_resources(started, "runtime parity")
            if (index + 1) % PROFILE["progress_interval"] == 0:
                print(json.dumps({"completed": index + 1,
                                  "elapsed_seconds": round(time.monotonic() - started, 2)}),
                      flush=True)
    after = prototype_hashes()
    if before != after or current_weight_hashes(accepted_weights) != accepted_weights:
        raise ValueError("prototype artifacts or model weights changed during parity")
    runtime_sources_after = validate_runtime_identity(
        runtime_sources_before, accepted_sources, integrated_version, converter
    )
    require_resources(started, "post-validation")
    actual_suites = rebuilt_suites(outcomes)
    reference_suites = {**development["suites"], "final_v22": final["metrics"]}
    if actual_suites != reference_suites:
        raise ValueError("integrated runtime suite metrics differ from accepted prototype")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": len(outcomes) == PROFILE["total_rows"],
        "scope": "post-integration actual runtime parity",
        "integrated_runtime_version": converter.version,
        "integrated_runtime_source_hashes": runtime_sources_before,
        "runtime_source_hashes_after": runtime_sources_after,
        "accepted_development_report_sha256": development_sha,
        "accepted_final_report_sha256": final_sha,
        "acceptance_snapshot_sha256": ACCEPTANCE_SHA256,
        "accepted_weight_hashes": accepted_weights,
        "prototype_artifact_hashes": before,
        "profile": PROFILE,
        "rows": len(outcomes),
        "suite_rows": {name: sum(row["suite"] == name for row in outcomes)
                       for name in (*DEV_COUNTS, "final_v22")},
        "all_candidate_objects_equal": all(row["candidate_objects_equal"] for row in outcomes),
        "all_sentence_pools_equal": all(row["sentence_pool_equal"] for row in outcomes),
        "all_sentence_scores_within_tolerance": all(
            row["sentence_scores_within_tolerance"] for row in outcomes
        ),
        "actual_suites": actual_suites,
        "reference_suites": reference_suites,
        "cases_sha256": file_sha256(cases_path),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
        "deployed": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"passed": report["passed"], "rows": report["rows"],
                      "suite_rows": report["suite_rows"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-report-sha256", required=True)
    parser.add_argument("--final-report-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
