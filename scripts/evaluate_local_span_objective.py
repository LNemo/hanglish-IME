"""Closed DEV evaluation for the fixed local-span auxiliary objective.

Candidate construction, support decisions, and public response policy are the
same as the frozen source-span DEV experiment.  Predictions are durably closed
before DEV targets are opened.  This is an exposed-DEV selection gate only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_source_span_pair import (
    DEV_SHA256,
    append_jsonl,
    canonical_hash,
    collect_candidate_groups,
    iter_jsonl,
    paired_delta,
    protected_fixtures,
    response_protects_literals,
    support_reason,
    validate_training_artifacts as validate_control_artifacts,
    verify_paths_protect_literals,
)
from scripts.preflight_source_span_residual import PARENT, PARENT_HASHES, require, sha, write
from scripts.score_source_edge_recombination import (
    DEFAULT_INPUT,
    DEFAULT_PROOF,
    DEV,
    EXPECTED_ROWS,
    RUNTIME,
    apply_policy,
    compact_public,
    validate_inputs,
)
from scripts.train_local_span_objective import LIMITS as TRAIN_LIMITS
from scripts.train_local_span_objective import RECIPE, SCHEMA as TRAIN_SCHEMA


SCHEMA = "local-span-objective-dev-evaluation-v1"
PLAN = ROOT / ".omx/plans/local-span-objective-v1.md"
DEFAULT_CONTROL = ROOT / ".omx/artifacts/source-span-paired-training-v1"
DEFAULT_TRAINING = ROOT / ".omx/artifacts/local-span-objective-training-v1"
CONTROL_DEV = ROOT / ".omx/artifacts/source-span-paired-dev-v1"
CONTROL_PREDICTIONS_SHA256 = "152cbab38ea99cd5f5e83967b2d528be004d772364db8eaaa16e292ac220e5a9"
CONTROL_CLOSURE_SHA256 = "523bdf989b0575fc439ffcd7d6818bf759c00a96d6536cb85dacc94293983be8"
ARM_NAMES = ("captured_runtime", "parent", "source_local_control", "local_aux")
MODEL_ARMS = ("source_local_control", "local_aux")
EVAL_LIMITS = {"whole_seconds": 1800.0, "update_seconds": 60.0,
               "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
PROOF_SCHEMA = "local-span-objective-training-independent-verification-v1"


def _finite_nonnegative(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _finite_number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_control_dev_capture(directory: Path = CONTROL_DEV) -> dict:
    """Authenticate the historical source-only DEV capture without its report."""
    directory = Path(directory)
    predictions = directory / "predictions.jsonl"
    closure_path = directory / "predictions.complete.json"
    require(predictions.is_file() and closure_path.is_file()
            and sha(predictions) == CONTROL_PREDICTIONS_SHA256
            and sha(closure_path) == CONTROL_CLOSURE_SHA256,
            "historical source-only DEV capture identity differs")
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    require(closure == {
        "candidate_policy": "recombined_pool_256", "common_support_rule": True,
        "input_truncation": False, "predictions_sha256": CONTROL_PREDICTIONS_SHA256,
        "rows": EXPECTED_ROWS,
        "schema": "source-span-paired-dev-evaluation-v1-source-only-closure",
        "source_only": True, "targets_joined": False},
        "historical source-only DEV closure differs")
    return {"predictions": predictions, "closure": closure_path,
            "pins": {str(predictions): CONTROL_PREDICTIONS_SHA256,
                     str(closure_path): CONTROL_CLOSURE_SHA256}}


def verify_control_replay(prediction: dict, historical: dict) -> None:
    """Require exact source/domain/path/support and old source-local score replay."""
    require(prediction["id"] == historical.get("id")
            and prediction["request"] == historical.get("request")
            and prediction["candidate_texts"] == historical.get("candidate_texts")
            and prediction["path_ids"] == historical.get("path_ids")
            and prediction["support"] == historical.get("support")
            and prediction["scores"].get("parent") == historical.get("scores", {}).get("parent")
            and prediction["scores"].get("source_local_control")
                == historical.get("scores", {}).get("source_local")
            and prediction["responses"]["source_local_control"]
                == historical.get("responses", {}).get("source_local")
            and prediction["full_response_sha256"]["source_local_control"]
                == historical.get("full_response_sha256", {}).get("source_local"),
            "historical source-local DEV replay differs")


def build_prediction(runtime_row: dict, texts: list[str], representatives: list[dict],
                     groups: list[list], scores: dict[str, list[float]] | None,
                     latencies_ms: dict[str, float], reason: str | None) -> dict:
    """Apply one shared support/fallback decision to all evaluated branches."""
    source = runtime_row["request"]["input"]
    original = runtime_row["proposed"]
    require(set(latencies_ms) == {"captured_runtime", "parent_features", *MODEL_ARMS}
            and all(_finite_nonnegative(value) for value in latencies_ms.values()),
            "latency record differs")
    responses = {"captured_runtime": deepcopy(original)}
    hashes = {"captured_runtime": canonical_hash(original)}
    score_rows = {}
    if reason is None:
        require(scores is not None and set(scores) == {"parent", *MODEL_ARMS},
                "complete supported score map is absent")
        for arm in ("parent", *MODEL_ARMS):
            values = scores[arm]
            require(len(values) == len(texts)
                    and all(type(value) in (int, float) and math.isfinite(value)
                            for value in values), "model returned malformed scores")
            score_rows[arm] = list(map(float, values))
            policy_model_id = "source_local" if arm == "source_local_control" else arm
            applied = apply_policy(original, representatives,
                dict(zip(texts, score_rows[arm])), supported=True, source=source,
                model_id=policy_model_id)
            require(applied["applied"] is True, "supported policy was not applied")
            responses[arm] = applied["response"]
            hashes[arm] = canonical_hash(applied["response"])
    else:
        require(scores is None, "fallback row may not contain model scores")
        for arm in ("parent", *MODEL_ARMS):
            responses[arm] = deepcopy(original)
            hashes[arm] = canonical_hash(responses[arm])
            require(responses[arm] == original, "fallback changed captured response")
    return {
        "id": runtime_row["id"],
        "request": deepcopy(runtime_row["request"]),
        "candidate_texts": list(texts),
        "path_ids": [[path.path_id for path in group] for group in groups],
        "support": {"supported": reason is None, "reason": reason},
        "scores": score_rows,
        "latency_ms": dict(latencies_ms),
        "responses": {arm: compact_public(response) for arm, response in responses.items()},
        "full_response_sha256": hashes,
        "protected_paths": verify_paths_protect_literals(source, groups),
        "protected_responses": {
            arm: response_protects_literals(source, response)
            for arm, response in responses.items()
        },
    }


def _public_counts(response: dict, expected: list[str]):
    texts = [candidate.get("text") for candidate in response.get("candidates", [])[:5]
             if isinstance(candidate, dict)]
    return texts, bool(texts and texts[0] in expected), any(text in expected for text in texts)


def summarize(predictions: list[dict], labels: list[dict], arm: str) -> dict:
    require(arm in ARM_NAMES and len(predictions) == len(labels) == EXPECTED_ROWS
            and [row["id"] for row in predictions] == [row.get("id") for row in labels],
            "DEV metric identity differs")
    counts = Counter(); categories = defaultdict(Counter); outcomes = []; latency = []
    for row, label in zip(predictions, labels):
        require(row["request"] == {"input": label.get("input"), "context": ""},
                "late DEV source identity differs")
        expected = label.get("expected")
        require(isinstance(expected, list) and expected, "DEV expected values differ")
        texts, top1, top5 = _public_counts(row["responses"][arm], expected)
        covered = bool(set(row["candidate_texts"] + [label["input"]]).intersection(expected))
        protected = row["protected_responses"][arm]["verified"]
        supported = row["support"]["supported"]
        values = {"rows": 1, "top1": int(top1), "top5": int(top5),
                  "full_pool_covered": int(covered), "supported": int(supported),
                  "fallback": int(not supported), "protected_exact": int(protected)}
        counts.update(values); categories[label.get("category", "uncategorized")].update(values)
        latency_key = ("captured_runtime" if arm == "captured_runtime"
                       else "parent_features" if arm == "parent" else arm)
        measured = row["latency_ms"][latency_key]
        if arm in MODEL_ARMS:
            measured += row["latency_ms"]["parent_features"]
        latency.append(measured)
        outcomes.append({"id": row["id"], "actual": texts, "expected": expected,
                         "top1": top1, "top5": top5, "full_pool_covered": covered,
                         "supported": supported, "protected_exact": protected})
    latency.sort()
    return {
        "counts": dict(counts),
        "categories": {key: dict(value) for key, value in sorted(categories.items())},
        "latency": {"measured_rows": len(latency), "total_ms": sum(latency),
                    "mean_ms": sum(latency) / len(latency), "max_ms": max(latency),
                    "p95_ms": latency[min(len(latency) - 1,
                                           math.ceil(.95 * len(latency)) - 1)]},
        "outcomes": outcomes,
    }


def decide(metrics: dict[str, dict]) -> dict:
    """The new arm alone may advance; the historical control is descriptive."""
    require(set(metrics) == set(ARM_NAMES), "evaluation metric arms differ")
    baselines = ("source_local_control", "captured_runtime")
    comparisons = {baseline: paired_delta(metrics[baseline], metrics["local_aux"])
                   for baseline in baselines}
    new_counts = metrics["local_aux"]["counts"]
    eligible = all(
        new_counts["top1"] > metrics[baseline]["counts"]["top1"]
        and new_counts["top5"] >= metrics[baseline]["counts"]["top5"]
        and not comparisons[baseline]["protected_regression_ids"]
        for baseline in baselines)
    return {
        "evaluated_arm": "local_aux",
        "historical_control_eligible": False,
        "versus": comparisons,
        "eligible_for_1000_case_regression": eligible,
        "selected_for_1000_case_regression": "local_aux" if eligible else None,
        "production_qualified": False,
        "production_qualification_reason":
            "DEV eligibility still requires the fixed 1,000-case and runtime gates",
    }


def validate_training_artifacts(training: Path, control: Path = DEFAULT_CONTROL) -> dict:
    """Authenticate the historical control and the one fixed new-arm output."""
    control_auth = validate_control_artifacts(control)
    training = Path(training)
    report_path = training / "report.json"
    proof_path = training / "independent-verification.json"
    require(report_path.is_file() and proof_path.is_file(),
            "local objective report or independent proof is absent")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    report_sha = sha(report_path)
    proof_contracts = proof.get("contracts", {})
    expected_proof_contracts = {
        "fit_only": True, "hold_opened": False, "dev_opened": False,
        "test_opened": False, "production_changed": False,
        "historical_control_all_scores_exact": True,
        "score_fp32_base_plus_residual_replayed": True,
        "local_softplus_losses_replayed_from_saved_margins": True}
    require(proof.get("schema") == PROOF_SCHEMA and proof.get("verified") is True
            and proof.get("report_sha256") == report_sha
            and proof.get("producer_sha256")
                == sha(ROOT / "scripts/train_local_span_objective.py")
            and proof.get("counts") == {
                "selected_fit_rows": 1483, "preferences": 6889,
                "updates": 2966, "reload_rows": 16}
            and all(proof_contracts.get(key) == value
                    for key, value in expected_proof_contracts.items())
            and proof.get("tensors", {}).get("parent_unchanged") is True
            and proof.get("resources", {}).get("atomic_operation_attribution_verified") is True,
            "local objective independent proof differs")
    require(report.get("schema") == TRAIN_SCHEMA and report.get("completed") is True
            and report.get("rows") == 1483 and report.get("updates") == 2966
            and report.get("recipe") == RECIPE and report.get("limits") == TRAIN_LIMITS
            and report.get("contracts") == {
                "fit_only": True, "dev_opened": False, "test_opened": False,
                "hold_opened": False, "production_changed": False}
            and report.get("historical_control_all_row_parity") is True
            and report.get("trained_global_all_row_parity") is True
            and report.get("parent_before") == report.get("parent_after"),
            "local objective training contract differs")
    pins = report.get("pins")
    require(isinstance(pins, dict) and pins, "local objective pins are absent")
    for raw, digest in pins.items():
        path = Path(raw)
        require(path.is_file() and sha(path) == digest,
                "local objective input pin differs: " + raw)
    for name, digest in PARENT_HASHES.items():
        require(pins.get(str(PARENT / name)) == digest and sha(PARENT / name) == digest,
                "frozen parent identity differs: " + name)
    checkpoint_path = training / "checkpoint.json"
    residual_path = training / "residual.safetensors"
    initial_path = training / "initial.safetensors"
    selection_path = training / "selection.json"
    require(all(path.is_file() for path in
                (checkpoint_path, residual_path, initial_path, selection_path)),
            "local objective checkpoint bundle is incomplete")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    control_checkpoint = control_auth["checkpoints"]["source_local"]["metadata"]
    require(checkpoint.get("schema") == TRAIN_SCHEMA
            and checkpoint.get("arm") == "local_aux"
            and checkpoint.get("recipe") == RECIPE
            and checkpoint.get("pins") == pins
            and checkpoint.get("residual_sha256") == sha(residual_path)
            and checkpoint.get("initial_sha256") == sha(initial_path)
            and checkpoint.get("selection_sha256") == sha(selection_path)
            and checkpoint.get("architecture") == control_auth["architecture"]
            and sha(initial_path) == control_checkpoint["initial_sha256"]
            and report.get("initial_hashes")
                == control_auth["report"]["arms"]["source_local"]["initial_hashes"]
            and report.get("reload_rows") == 16,
            "local objective checkpoint/control identity differs")
    gradients = report.get("upstream_gradient_maxima")
    expected_gradients = {"source_embedding.", "source_encoder.",
                          "segment_head.0.", "segment_head.2."}
    require(isinstance(gradients, dict) and set(gradients) == expected_gradients
            and all(type(value) in (int, float) and math.isfinite(value) and value > 0
                    for value in gradients.values()), "local objective gradients differ")
    epoch_losses = report.get("epoch_mean_losses")
    require(isinstance(epoch_losses, list) and len(epoch_losses) == RECIPE["epochs"]
            and all(isinstance(row, dict) and set(row) == {"total", "candidate", "local"}
                    and all(type(value) in (int, float) and math.isfinite(value)
                            for value in row.values()) for row in epoch_losses),
            "local objective epoch losses differ")
    required_outputs = {"pins.json", "selection.json", "initial.safetensors",
                        "control-parity.jsonl", "updates.jsonl", "after.jsonl",
                        "residual.safetensors", "reload.json", "checkpoint.json",
                        "resources.jsonl"}
    outputs = report.get("outputs")
    require(isinstance(outputs, dict) and required_outputs <= set(outputs),
            "local objective output manifest is incomplete")
    for relative, digest in outputs.items():
        path = training / relative
        require(path.is_file() and sha(path) == digest,
                "local objective output differs: " + relative)
    artifact_pins = {str(report_path): report_sha, str(proof_path): sha(proof_path)}
    artifact_pins.update({str(training / relative): digest
                          for relative, digest in outputs.items()})
    return {"report": report, "report_path": report_path, "proof": proof,
            "proof_path": proof_path, "pins": pins, "artifact_pins": artifact_pins,
            "architecture": checkpoint["architecture"],
            "control": control_auth["checkpoints"]["source_local"],
            "local_aux": {"metadata": checkpoint, "metadata_path": checkpoint_path,
                          "residual_path": residual_path},
            "control_auth": control_auth}


def run(training: Path, output: Path, control: Path = DEFAULT_CONTROL) -> dict:
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        training_auth = validate_training_artifacts(training, control)
        source_auth = validate_inputs(DEFAULT_INPUT, DEFAULT_PROOF)
        control_dev_auth = validate_control_dev_capture()
        from safetensors.torch import load_file
        import torch
        from hanglish.models.sentence_ranker import SentenceRanker
        from hanglish.models.source_span_residual import SourceSpanResidualRanker
        from hanglish.models.span_token_bridge import encode_aligned_candidates
        from scripts.preflight_pko_seq2seq import system_swap_bytes
        from scripts.source_span_resources import AtomicResourceWatchdog

        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        pins = dict(training_auth["pins"])
        pins.update(training_auth["artifact_pins"]); pins.update(source_auth["pins"])
        pins.update(training_auth["control_auth"]["artifact_pins"])
        pins.update(control_dev_auth["pins"])
        tracked = [Path(__file__).resolve(), PLAN,
                   ROOT / "scripts/evaluate_source_span_pair.py",
                   ROOT / "scripts/score_source_edge_recombination.py",
                   ROOT / "scripts/train_local_span_objective.py",
                   ROOT / "scripts/train_source_span_pair.py",
                   ROOT / "scripts/source_span_resources.py",
                   ROOT / "hanglish/models/source_span_residual.py",
                   ROOT / "hanglish/models/span_token_bridge.py",
                   ROOT / "hanglish/core/source_correspondence.py"]
        pins.update({str(path): sha(path) for path in tracked})
        write(output / "pins.json", dict(sorted(pins.items())))
        baseline_swap = system_swap_bytes()
        watchdog = AtomicResourceWatchdog(started, baseline_swap,
            output / "resources.jsonl", interval_s=2.0, limits=EVAL_LIMITS)
        predictions = []
        with watchdog:
            with watchdog.operation("load_parent_and_residuals", update=True):
                parent = SentenceRanker.load(PARENT)
                models = {}
                for arm, artifact in (("source_local_control", training_auth["control"]),
                                      ("local_aux", training_auth["local_aux"])):
                    model = SourceSpanResidualRanker(parent)
                    require(model.architecture == training_auth["architecture"],
                            "evaluation architecture differs from checkpoint")
                    model.load_residual_state_dict(load_file(artifact["residual_path"]))
                    model.eval(); models[arm] = model
            capture_path = output / "predictions.jsonl"
            with capture_path.open("x", encoding="utf-8") as stream:
                runtime_rows = iter_jsonl(RUNTIME / "capture.jsonl")
                proposal_rows = iter_jsonl(DEFAULT_INPUT / "capture.jsonl")
                historical_rows = iter_jsonl(control_dev_auth["predictions"])
                for runtime_row, proposal_row, historical_row in zip_longest(
                        runtime_rows, proposal_rows, historical_rows):
                    require(runtime_row is not None and proposal_row is not None
                            and historical_row is not None,
                            "runtime/proposal/control denominator differs")
                    require(runtime_row.get("id") == proposal_row.get("id")
                            == historical_row.get("id")
                            and runtime_row.get("request") == proposal_row.get("request")
                            == historical_row.get("request"),
                            "runtime/proposal/control source identity differs")
                    texts, representatives, groups = collect_candidate_groups(runtime_row, proposal_row)
                    source = runtime_row["request"]["input"]
                    reason = support_reason(source, texts, lambda values: (
                        encode_aligned_candidates(parent.tokenizer, values, 256), True)[1])
                    captured_ms = runtime_row.get("elapsed_ms")
                    require(_finite_nonnegative(captured_ms), "captured runtime latency differs")
                    latency = {"captured_runtime": float(captured_ms), "parent_features": 0.0,
                               "source_local_control": 0.0, "local_aux": 0.0}
                    scores = None
                    if reason is None:
                        before = time.monotonic()
                        with watchdog.operation("parent_features:" + runtime_row["id"], update=True):
                            feature = models["source_local_control"].extract_features(texts)
                        latency["parent_features"] = (time.monotonic() - before) * 1000
                        scores = {"parent": feature.logits.tolist()}
                        for arm in MODEL_ARMS:
                            before = time.monotonic()
                            with watchdog.operation(arm + ":" + runtime_row["id"], update=True), torch.no_grad():
                                values = models[arm](source, groups, features=feature)["scores"]
                            latency[arm] = (time.monotonic() - before) * 1000
                            scores[arm] = values.tolist()
                    prediction = build_prediction(runtime_row, texts, representatives, groups,
                                                  scores, latency, reason)
                    verify_control_replay(prediction, historical_row)
                    predictions.append(prediction); append_jsonl(stream, prediction); stream.flush()
                    if len(predictions) % 25 == 0:
                        print(json.dumps({"captured": len(predictions), "rows": EXPECTED_ROWS,
                                          "elapsed_s": time.monotonic() - started}), flush=True)
                os.fsync(stream.fileno())
            require(len(predictions) == EXPECTED_ROWS, "DEV prediction denominator differs")
            capture_sha = sha(capture_path)
            closure = {"schema": SCHEMA + "-source-only-closure", "rows": EXPECTED_ROWS,
                       "predictions_sha256": capture_sha, "source_only": True,
                       "targets_joined": False, "candidate_policy": "recombined_pool_256",
                       "common_support_rule": True, "input_truncation": False}
            write(output / "predictions.complete.json", closure)
            closure_sha = sha(output / "predictions.complete.json")
            require(sha(DEV) == DEV_SHA256, "DEV targets changed before late join")
            labels = list(iter_jsonl(DEV))
            metrics = {arm: summarize(predictions, labels, arm) for arm in ARM_NAMES}
            require(metrics["captured_runtime"]["counts"]["top1"] == 264
                    and metrics["captured_runtime"]["counts"]["top5"] == 284,
                    "captured extended-runtime baseline differs")
            decision = decide(metrics)
            protected_rows = [row for row in predictions
                              if protected_fixtures(row["request"]["input"])]
            protected_summary = {
                "rows": len(protected_rows),
                "fixture_counts": dict(Counter(fixture["kind"] for row in protected_rows
                    for fixture in protected_fixtures(row["request"]["input"]))),
                "all_observed_candidate_paths_verified": all(
                    row["protected_paths"]["verified"] for row in protected_rows),
                "production_runtime_preservation_verified": False,
            }
            protected_summary["response_failures"] = {
                arm: [row["id"] for row in protected_rows
                      if not row["protected_responses"][arm]["verified"]]
                for arm in ARM_NAMES}
            final_resources = watchdog.sample_and_append("final")
            require(final_resources.get("operation") is None
                    and final_resources.get("operation_elapsed_s") is None
                    and _finite_nonnegative(final_resources.get("elapsed_s"))
                    and _finite_nonnegative(final_resources.get("rss_bytes"))
                    and _finite_number(final_resources.get("swap_growth_bytes"))
                    and final_resources["elapsed_s"] <= EVAL_LIMITS["whole_seconds"]
                    and final_resources["rss_bytes"] <= EVAL_LIMITS["rss_bytes"]
                    and final_resources["swap_growth_bytes"]
                        <= EVAL_LIMITS["swap_growth_bytes"],
                    "final evaluation resource observation differs")
        for raw, digest in pins.items():
            require(Path(raw).is_file() and sha(Path(raw)) == digest,
                    "evaluation pin changed: " + raw)
        elapsed = time.monotonic() - started
        require(elapsed <= EVAL_LIMITS["whole_seconds"],
                "whole-run evaluation time limit exceeded")
        report = {
            "schema": SCHEMA, "completed": True, "split": "dev", "rows": EXPECTED_ROWS,
            "scope": {"source_only_predictions_closed_before_target_join": True,
                      "dev_is_exposed_development": True, "test_opened": False,
                      "production_changed": False, "input_truncation": False},
            "metrics": metrics, "decision": decision,
            "protected_literal_contract": protected_summary,
            "capture_sha256": capture_sha, "closure_sha256": closure_sha,
            "training_report_sha256": sha(training_auth["report_path"]),
            "control_training_report_sha256": sha(training_auth["control_auth"]["report_path"]),
            "dev_labels_sha256": DEV_SHA256, "pins": dict(sorted(pins.items())),
            "limits": EVAL_LIMITS, "baseline_swap_bytes": baseline_swap,
            "final_resources": final_resources, "elapsed_s": elapsed,
            "limitations": [
                "DEV metrics do not establish blind generalization",
                "protected literals are verified on offline candidate paths",
                "captured and rescored latency are not an apples-to-apples benchmark",
                "the historical control is descriptive and is never selectable in this experiment",
            ],
        }
        write(output / "report.json", report)
        print(json.dumps({"completed": True,
            "metrics": {arm: value["counts"] for arm, value in metrics.items()},
            "decision": decision}, ensure_ascii=False), flush=True)
        return report
    except BaseException as error:
        write(output / "failure.json", {"schema": SCHEMA, "completed": False,
              "error_type": type(error).__name__, "error": str(error),
              "elapsed_s": time.monotonic() - started})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.training, arguments.output, arguments.control)
