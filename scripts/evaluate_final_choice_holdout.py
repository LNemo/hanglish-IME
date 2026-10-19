"""Evaluate the frozen final-choice selector and gate on TRAIN holdouts only."""
from __future__ import annotations

import argparse
from copy import deepcopy
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

from hanglish.core.neural_converter import file_sha256
from hanglish.models.sentence_ranker import SentenceRanker
from scripts.calibrate_final_choice_selector import PROFILE as CALIBRATION_PROFILE, load_selector
from scripts.final_choice_capture import _validate_segments, validate_response
from scripts.final_choice_gate import (
    MIN_CLASS_GROUPS,
    loads_gate,
    predict_accept,
    predict_probability,
)
from scripts.final_choice_policy import (
    decision_features,
    finite_candidate_indices,
    reorder_candidates,
)
from scripts.prepare_final_choice_data import PARTITION_SHA256
from scripts.train_natural_sentence_ranker import rss_bytes


PLAN = ROOT / ".omx/artifacts/final-choice-learning-plan-v1.md"
PLAN_SHA256 = "9adcb7c6d0fbbe596e44cf734d5b3d7b269c7709ca084e82fd2e36783dc71c49"
CALIBRATOR_SHA256 = "4765212aa1e5f633a461185fd930b16f8f4b7552cd258b5c1703d725393e471b"
TRAINER_SHA256 = "e66d7fdef1461b39a2874654aca6be12487b8dc241405ab9694efbf934f92cc8"
GATE_SHA256 = "b86e85916f49d9c8a23124ca617780c98784fada24e44eb2f74fbc3a0eaf0da7"
POLICY_SHA256 = "3acb1480fce1c490892e003a9677b0ad5354d07e971dd3ca73446c08d69cf717"
OUTPUT = ROOT / ".omx/artifacts/final-choice-holdout-v1"
ROLE_SPECS = {
    "natural_holdout": (512, "natural"),
    "challenge_preservation": (376, "challenge"),
}
PROFILE = {
    "name": "final-choice-holdout-and-preservation-v1",
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 1800,
    "progress_interval": 32,
    "natural_minimum_net_top1_gain": 2,
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _valid_hashes(values):
    return (isinstance(values, dict) and bool(values)
            and all(isinstance(name, str) and name
                    and isinstance(digest, str) and len(digest) == 64
                    and not set(digest) - set("0123456789abcdef")
                    for name, digest in values.items()))


def _read_json(path, expected_sha):
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha:
        raise ValueError(f"final-choice artifact hash differs: {path}")
    return raw, json.loads(raw)


def _read_jsonl(path, expected_sha):
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha:
        raise ValueError(f"final-choice artifact hash differs: {path}")
    return raw, [json.loads(line) for line in raw.splitlines() if line]


def _runtime_key(value):
    path = Path(value)
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _validate_row(row, selected, role, runtime_version, model_artifacts):
    keys = ("id", "group_id", "origin", "input", "context", "expected")
    if ({key: row.get(key) for key in keys}
            != {key: selected.get(key) for key in keys}):
        raise ValueError("evaluation row differs from its frozen selection")
    _, origin = ROLE_SPECS[role]
    candidates = row.get("candidates")
    if (row.get("origin") != origin or not isinstance(candidates, list)
            or not 1 <= len(candidates) <= 5
            or row.get("result") != candidates[0].get("text")
            or row.get("runtime_version") != runtime_version
            or not isinstance(row.get("expected"), list) or not row["expected"]
            or any(not isinstance(value, str) or not value for value in row["expected"])
            or type(row.get("engine_ms")) not in (int, float)
            or not math.isfinite(row["engine_ms"]) or row["engine_ms"] < 0
            or type(row.get("raw_engine_ms")) not in (int, float)
            or not math.isfinite(row["raw_engine_ms"]) or row["raw_engine_ms"] < 0):
        raise ValueError("evaluation capture row is malformed")
    texts = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
            raise ValueError("evaluation candidate is malformed")
        score = candidate.get("score")
        if (score is not None and (type(score) not in (int, float)
                or not math.isfinite(score) or round(score, 4) != score)):
            raise ValueError("evaluation returned score is malformed")
        _validate_segments(candidate, row["input"])
        if score is None and candidate != {
                "text": row["input"], "score": None,
                "segments": [{"input_start": 0, "input_end": len(row["input"]),
                              "source": row["input"], "text": row["input"],
                              "kind": "raw"}]}:
            raise ValueError("evaluation None score is not the complete raw candidate")
        texts.append(candidate["text"])
    if len(texts) != len(set(texts)):
        raise ValueError("evaluation candidates are duplicated")
    finite = finite_candidate_indices(candidates)
    positives = [index for index in finite if texts[index] in row["expected"]]
    can_score = row.get("candidate_can_score")
    if (not isinstance(can_score, list) or len(can_score) != len(candidates)
            or any(type(value) is not bool for value in can_score)):
        raise ValueError("evaluation candidate budget flags are malformed")
    reasons = []
    if not all(can_score):
        reasons.append("candidate_exceeds_sentence_model_budget")
    if len(finite) < 2 or 0 not in finite:
        reasons.append("insufficient_finite_candidates")
    if (row.get("finite_indices") != finite
            or row.get("accepted_positive_indices") != positives
            or row.get("label_aligned") is not bool(positives)
            or row.get("all_candidates_can_score") is not all(can_score)
            or row.get("selector_eligible") is not (not reasons)
            or row.get("ineligible_reasons") != reasons):
        raise ValueError("evaluation capture eligibility differs")
    flags = row.get("response_flags", {})
    if (flags.get("version") != runtime_version
            or flags.get("model_artifacts") != model_artifacts
            or flags.get("raw") is not False
            or row.get("model_artifact_flags") != {
                "expected_count": len(model_artifacts), "response_matches_expected": True}):
        raise ValueError("evaluation capture runtime identity differs")
    validate_response(row.get("raw_bypass_response"), row["input"], row["context"],
                      runtime_version, model_artifacts, raw=True)


def load_evaluation_capture(directory, explicit_manifest_sha, role):
    """Load only one of the two frozen evaluation roles."""
    if role not in ROLE_SPECS:
        raise ValueError("only holdout and preservation roles are allowed")
    manifest_raw, manifest = _read_json(directory / "manifest.json", explicit_manifest_sha)
    cases_raw, rows = _read_jsonl(directory / "cases.jsonl", manifest.get("cases_sha256"))
    selection_raw, selection = _read_jsonl(
        directory / "selection.jsonl", manifest.get("selection_sha256")
    )
    count, _ = ROLE_SPECS[role]
    ordered = manifest.get("partition_ordered_ids")
    if (manifest.get("completed") is not True or manifest.get("role") != role
            or manifest.get("scope") != "fresh actual parent TRAIN capture for final-choice learning"
            or manifest.get("partition_manifest_sha256") != PARTITION_SHA256
            or manifest.get("expected_rows") != count or manifest.get("rows") != count
            or len(rows) != count or len(selection) != count
            or not isinstance(ordered, list) or len(ordered) != count
            or [row.get("id") for row in rows] != ordered
            or [row.get("id") for row in selection] != ordered
            or len(set(ordered)) != count
            or manifest.get("selection_sha256_before_inference")
               != manifest.get("selection_sha256")
            or manifest.get("source_hashes") != manifest.get("source_hashes_after")
            or manifest.get("training_run") is not False
            or manifest.get("new_selector_inference_run") is not False
            or manifest.get("development_data_read") is not False
            or manifest.get("final_data_read") is not False
            or manifest.get("deployed") is not False):
        raise ValueError("completed exact evaluation capture is required")
    source_hashes = manifest.get("source_hashes")
    runtime_sources = manifest.get("runtime_source_hashes")
    models = manifest.get("model_artifact_hashes")
    inputs = manifest.get("train_inputs")
    if (not _valid_hashes(source_hashes) or not _valid_hashes(runtime_sources)
            or not _valid_hashes(models) or len(models) != 16
            or any(source_hashes.get(name) != digest
                   for name, digest in {**runtime_sources, **models}.items())
            or not isinstance(inputs, dict) or not inputs
            or any(not isinstance(value, dict)
                   or source_hashes.get(value.get("path")) != value.get("sha256")
                   for value in inputs.values())
            or source_hashes.get(_runtime_key(manifest.get("runtime_evidence_path", "")))
               != manifest.get("runtime_evidence_sha256")):
        raise ValueError("evaluation capture provenance is malformed")
    for row, selected in zip(rows, selection):
        _validate_row(row, selected, role, manifest.get("runtime_version"), models)
    diagnostics = {
        "top1": sum(row["result"] in row["expected"] for row in rows),
        "top5": sum(any(candidate["text"] in row["expected"]
                        for candidate in row["candidates"]) for row in rows),
        "label_aligned": sum(row["label_aligned"] for row in rows),
        "selector_eligible": sum(row["selector_eligible"] for row in rows),
        "all_candidates_can_score": sum(row["all_candidates_can_score"] for row in rows),
    }
    if manifest.get("diagnostics") != diagnostics:
        raise ValueError("evaluation capture diagnostics do not reproduce")
    if (manifest.get("baseline_accuracy_is_diagnostic_only") is not True
            or manifest.get("split_or_threshold_selected_from_capture_outcomes") is not False):
        raise ValueError("evaluation capture outcome policy differs")
    return {"directory": directory, "manifest_raw": manifest_raw,
            "cases_raw": cases_raw, "selection_raw": selection_raw,
            "manifest": manifest, "rows": rows}


def load_calibration_result(report_path, explicit_sha, selector_path, selector_sha,
                            runtime_version):
    raw, report = _read_json(report_path, explicit_sha)
    gate_path = report_path.parent / "gate.json"
    gate_raw = gate_path.read_bytes()
    if (report.get("status") != "completed" or report.get("passed") is not True
            or report.get("eligible_for_holdout_evaluation") is not True
            or report.get("scope") !=
            "natural CAL-only final-choice accept/stay gate calibration"
            or report.get("profile") != CALIBRATION_PROFILE
            or report.get("runtime_version") != runtime_version
            or report.get("selector_report_sha256") != selector_sha
            or report.get("source_hashes") != report.get("source_hashes_after")
            or report.get("gate_prediction_save_load_parity", {}).get("exact") is not True
            or report["gate_prediction_save_load_parity"].get("calibration_rows") != 1024
            or report["gate_prediction_save_load_parity"].get("gate_sha256")
               != sha256_bytes(gate_raw)
            or report.get("rows") != 1024
            or report.get("resplit_or_fit_fallback_used") is not False
            or report.get("fit_data_read") is not False
            or report.get("holdout_data_read") is not False
            or report.get("preservation_data_read") is not False
            or report.get("development_data_read") is not False
            or report.get("final_data_read") is not False
            or report.get("eligible_for_deployment") is not False
            or report.get("deployed") is not False):
        raise ValueError("successful frozen CAL gate report is required")
    gate = loads_gate(gate_raw.decode("utf-8"))
    counts = report.get("class_counts", {})
    if (counts.get("rows") != gate["counts"]["rows"]
            or counts.get("accept_rows") != gate["counts"]["accept_rows"]
            or counts.get("stay_rows") != gate["counts"]["stay_rows"]
            or counts.get("accept_groups") != gate["counts"]["accept_groups"]
            or counts.get("stay_groups") != gate["counts"]["stay_groups"]
            or min(counts.get("accept_groups", 0), counts.get("stay_groups", 0))
               < MIN_CLASS_GROUPS):
        raise ValueError("CAL gate class counts differ")
    audit_raw, audit = _read_jsonl(
        report_path.parent / "scoring-cases.jsonl", report.get("scoring_cases_sha256")
    )
    if len(audit) != 1024 or len({row.get("id") for row in audit}) != 1024:
        raise ValueError("CAL scoring audit is incomplete")
    feature_rows = 0
    for row in audit:
        if row.get("reason") == "eligible_disagreement":
            feature_rows += 1
            probability = row.get("gate_probability")
            if (type(probability) not in (int, float) or not math.isfinite(probability)
                    or probability != predict_probability(gate, row.get("features"))
                    or row.get("gate_accept") is not predict_accept(gate, row.get("features"))):
                raise ValueError("CAL stored gate prediction differs")
        elif row.get("gate_probability") is not None or row.get("gate_accept") is not None:
            raise ValueError("CAL nonapplicable row contains a gate prediction")
    parity = report["gate_prediction_save_load_parity"]
    if (parity.get("feature_rows") != feature_rows
            or parity.get("nonapplicable_rows") != len(audit) - feature_rows):
        raise ValueError("CAL prediction coverage differs")
    selector = load_selector(selector_path, selector_sha, {"runtime_version": runtime_version})
    if report.get("selector_checkpoint_sha256") != selector["report"]["checkpoint_sha256"]:
        raise ValueError("CAL selector checkpoint differs")
    for name, digest in report["source_hashes"].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if file_sha256(path) != digest:
            raise ValueError(f"CAL source changed: {name}")
    return {"report_raw": raw, "report": report, "gate_raw": gate_raw,
            "gate": gate, "audit_raw": audit_raw, "selector": selector}


def verify_raw_policy(row):
    raw = row["raw_bypass_response"]
    candidates = raw["candidates"]
    before = deepcopy(candidates)
    decision = decision_features(row["input"], row["context"], candidates, [], raw=True)
    if decision.get("reason") != "raw_mode" or decision.get("proposal_index") != 0:
        raise RuntimeError("raw selector policy did not retain the parent")
    # Raw actions bypass the learned gate and retain every existing object.
    reordered = reorder_candidates(candidates, decision["proposal_index"])
    if (candidates != before or reordered != before
            or len(reordered) != len(candidates)
            or any(left is not right for left, right in zip(reordered, candidates))
            or reordered[0]["text"] != row["input"] or raw["result"] != row["input"]):
        raise RuntimeError("raw selector policy changed the original response")
    return {"preserved": True, "reason": decision["reason"],
            "chosen_index": 0, "gate_applied": False,
            "object_identity_and_order_preserved": True}


def evaluate_case(ranker, gate, row, role):
    """Score and decide before consulting fixed TRAIN holdout labels."""
    candidates = row["candidates"]
    parent_objects = list(candidates)
    protected_before = {
        id(candidate): deepcopy([segment for segment in candidate["segments"]
                                 if segment["kind"] in {"raw", "mixed", "numeric"}
                                 or any(not char.islower() for char in segment["source"])])
        for candidate in parent_objects
    }
    if row["selector_eligible"] and row["all_candidates_can_score"]:
        current_budget = [bool(ranker.can_score(candidate["text"])) for candidate in candidates]
        if current_budget != row["candidate_can_score"] or not all(current_budget):
            raise ValueError("selector budget differs from evaluation capture")
        scores = ranker.score_pairs(
            [row["context"]] * len(candidates), [candidate["text"] for candidate in candidates]
        )
        if (len(scores) != len(candidates)
                or any(type(score) not in (int, float) or not math.isfinite(score)
                       for score in scores)):
            raise ValueError("selector scores must cover the complete candidate list")
        decision = decision_features(row["input"], row["context"], candidates, scores)
    else:
        scores = []
        decision = {"reason": "capture_ineligible", "proposal_index": 0}
    gate_applied = decision["reason"] == "eligible_disagreement"
    accepted = gate_applied and predict_accept(gate, decision["features"])
    chosen = decision["proposal_index"] if accepted else 0
    reordered = reorder_candidates(candidates, chosen)
    if ({id(item) for item in reordered} != {id(item) for item in parent_objects}
            or reordered[0] is not parent_objects[chosen]
            or reordered[1:] != parent_objects[:chosen] + parent_objects[chosen + 1:]):
        raise RuntimeError("selector did not preserve candidate object identity and order")
    for candidate in reordered:
        protected_after = [segment for segment in candidate["segments"]
                           if segment["kind"] in {"raw", "mixed", "numeric"}
                           or any(not char.islower() for char in segment["source"])]
        if protected_after != protected_before[id(candidate)]:
            raise RuntimeError("protected segment content changed")
    raw_policy = verify_raw_policy(row)
    # Labels are used only after full scoring, policy, gate, and reorder operations.
    expected = row["expected"]
    before_top1 = candidates[0]["text"] in expected
    top1 = reordered[0]["text"] in expected
    before_top5 = any(candidate["text"] in expected for candidate in candidates)
    top5 = any(candidate["text"] in expected for candidate in reordered)
    if before_top5 != top5 or {candidate["text"] for candidate in candidates} != {
            candidate["text"] for candidate in reordered}:
        raise RuntimeError("selector changed candidate coverage or set")
    return {"id": row["id"], "group_id": row["group_id"], "suite": role,
            "reason": decision["reason"], "proposal_index": decision["proposal_index"],
            "chosen_index": chosen, "gate_applied": gate_applied,
            "gate_accepted": accepted, "selector_scores": list(scores),
            "before_candidates": deepcopy(candidates),
            "candidates": deepcopy(reordered),
            "raw_bypass_response": deepcopy(row["raw_bypass_response"]),
            "before_top1": before_top1, "top1": top1,
            "before_top5": before_top5, "top5": top5,
            "candidate_count": len(candidates), "candidate_set_preserved": True,
            "relative_order_preserved": True, "object_identity_preserved": True,
            "raw_bypass_preserved": raw_policy["preserved"], "raw_policy": raw_policy,
            "protected_segments_preserved": True,
            "missing_candidate_coverage": not before_top5}


def summarize(rows):
    return {"rows": len(rows),
            "before_top1": sum(row["before_top1"] for row in rows),
            "top1": sum(row["top1"] for row in rows),
            "before_top5": sum(row["before_top5"] for row in rows),
            "top5": sum(row["top5"] for row in rows),
            "corrections": [row["id"] for row in rows
                            if row["top1"] and not row["before_top1"]],
            "regressions": [row["id"] for row in rows
                            if row["before_top1"] and not row["top1"]],
            "top5_losses": [row["id"] for row in rows
                            if row["before_top5"] and not row["top5"]],
            "eligible_disagreements": sum(row["gate_applied"] for row in rows),
            "accepted_actions": sum(row["gate_accepted"] for row in rows),
            "missing_candidate_coverage": sum(row["missing_candidate_coverage"] for row in rows)}


def holdout_gate(suites):
    natural = suites["natural_holdout"]
    preservation = suites["challenge_preservation"]
    no_losses = all(not values["regressions"] and not values["top5_losses"]
                    for values in suites.values())
    actions = sum(values["accepted_actions"] for values in suites.values())
    passed = (natural["top1"] - natural["before_top1"]
              >= PROFILE["natural_minimum_net_top1_gain"] and no_losses and actions > 0)
    return {"passed": passed, "inconclusive": actions == 0,
            "eligible_for_development": passed,
            "zero_row_regressions": no_losses,
            "natural_net_top1_gain": natural["top1"] - natural["before_top1"],
            "accepted_actions": actions,
            "preservation_top1_delta": preservation["top1"] - preservation["before_top1"]}


def merge_hashes(target, additions, label):
    for name, digest in additions.items():
        if name in target and target[name] != digest:
            raise ValueError(f"conflicting {label} hash: {name}")
        target[name] = digest


def provenance_hashes(captures, calibration, evaluator_sha):
    hashes = {}
    for capture in captures.values():
        merge_hashes(hashes, capture["manifest"]["source_hashes"], "capture source")
        relative = str(capture["directory"].relative_to(ROOT))
        hashes[f"{relative}/manifest.json"] = sha256_bytes(capture["manifest_raw"])
        hashes[f"{relative}/cases.jsonl"] = sha256_bytes(capture["cases_raw"])
        hashes[f"{relative}/selection.jsonl"] = sha256_bytes(capture["selection_raw"])
    additions = {
        str(PLAN.relative_to(ROOT)): PLAN_SHA256,
        "scripts/calibrate_final_choice_selector.py": CALIBRATOR_SHA256,
        "scripts/train_final_choice_selector.py": TRAINER_SHA256,
        "scripts/final_choice_gate.py": GATE_SHA256,
        "scripts/final_choice_policy.py": POLICY_SHA256,
        "scripts/evaluate_final_choice_holdout.py": evaluator_sha,
    }
    merge_hashes(hashes, additions, "evaluation dependency")
    return dict(sorted(hashes.items()))


def verify_hashes(hashes):
    for name, expected in hashes.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if file_sha256(path) != expected:
            raise ValueError(f"holdout evaluation input changed: {name}")


def require_resources(started, stage):
    if (rss_bytes() > PROFILE["rss_limit_bytes"]
            or time.monotonic() - started > PROFILE["time_limit_seconds"]):
        raise RuntimeError(f"final-choice holdout resource bound exceeded during {stage}")


def run(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = Path(__file__).read_bytes()
    source_sha = sha256_bytes(source)
    fixed = {PLAN: PLAN_SHA256,
             ROOT / "scripts/calibrate_final_choice_selector.py": CALIBRATOR_SHA256,
             ROOT / "scripts/train_final_choice_selector.py": TRAINER_SHA256,
             ROOT / "scripts/final_choice_gate.py": GATE_SHA256,
             ROOT / "scripts/final_choice_policy.py": POLICY_SHA256}
    if any(file_sha256(path) != digest for path, digest in fixed.items()):
        raise ValueError("frozen final-choice implementation differs")
    captures = {
        "natural_holdout": load_evaluation_capture(
            args.natural_holdout, args.natural_manifest_sha256, "natural_holdout"),
        "challenge_preservation": load_evaluation_capture(
            args.challenge_preservation, args.challenge_manifest_sha256,
            "challenge_preservation"),
    }
    left, right = (captures[name]["manifest"] for name in captures)
    if (left["runtime_version"] != right["runtime_version"]
            or left["runtime_source_hashes"] != right["runtime_source_hashes"]
            or left["model_artifact_hashes"] != right["model_artifact_hashes"]
            or left["source_hashes"] != right["source_hashes"]
            or left["train_inputs"] != right["train_inputs"]):
        raise ValueError("holdout and preservation capture lineage differs")
    calibration = load_calibration_result(
        args.calibration_report, args.calibration_report_sha256,
        args.selector_report, args.selector_report_sha256, left["runtime_version"]
    )
    before = provenance_hashes(captures, calibration, source_sha)
    merge_hashes(before, calibration["report"]["source_hashes"], "CAL source")
    calibration_relative = str(args.calibration_report.relative_to(ROOT))
    selector_relative = str(args.selector_report.relative_to(ROOT))
    before[calibration_relative] = sha256_bytes(calibration["report_raw"])
    before[str((args.calibration_report.parent / "gate.json").relative_to(ROOT))] = sha256_bytes(
        calibration["gate_raw"])
    before[str((args.calibration_report.parent / "scoring-cases.jsonl").relative_to(ROOT))] = (
        sha256_bytes(calibration["audit_raw"]))
    before[selector_relative] = sha256_bytes(calibration["selector"]["report_raw"])
    before = dict(sorted(before.items()))
    verify_hashes(before)
    require_resources(started, "pre-inference validation")

    torch.set_num_threads(PROFILE["threads"])
    os.environ["OMP_NUM_THREADS"] = str(PROFILE["threads"])
    ranker = SentenceRanker.load(calibration["selector"]["checkpoint"])
    args.output.mkdir(parents=True)
    audit_path = args.output / "cases.jsonl"
    outcomes = []
    with audit_path.open("x") as stream:
        for role, capture in captures.items():
            for row in capture["rows"]:
                outcome = evaluate_case(ranker, calibration["gate"], row, role)
                outcomes.append(outcome)
                stream.write(json.dumps(outcome, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                require_resources(started, "holdout scoring")
                if len(outcomes) % PROFILE["progress_interval"] == 0:
                    print(json.dumps({"event": "final_choice_holdout",
                                      "rows": len(outcomes)}), flush=True)
    suites = {role: summarize([row for row in outcomes if row["suite"] == role])
              for role in ROLE_SPECS}
    gate_result = holdout_gate(suites)

    load_evaluation_capture(args.natural_holdout, args.natural_manifest_sha256,
                            "natural_holdout")
    load_evaluation_capture(args.challenge_preservation,
                            args.challenge_manifest_sha256,
                            "challenge_preservation")
    after = provenance_hashes(captures, calibration, source_sha)
    merge_hashes(after, calibration["report"]["source_hashes"], "CAL source")
    after[calibration_relative] = file_sha256(args.calibration_report)
    after[str((args.calibration_report.parent / "gate.json").relative_to(ROOT))] = file_sha256(
        args.calibration_report.parent / "gate.json")
    after[str((args.calibration_report.parent / "scoring-cases.jsonl").relative_to(ROOT))] = (
        file_sha256(args.calibration_report.parent / "scoring-cases.jsonl"))
    after[selector_relative] = file_sha256(args.selector_report)
    after = dict(sorted(after.items()))
    verify_hashes(after)
    if after != before or file_sha256(Path(__file__)) != source_sha:
        raise RuntimeError("holdout sources changed")
    source_path = args.output / "source.py"
    source_path.write_bytes(source)
    if file_sha256(source_path) != source_sha:
        raise RuntimeError("holdout evaluator source snapshot differs")
    require_resources(started, "post-validation")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": gate_result["passed"], "inconclusive": gate_result["inconclusive"],
        "eligible_for_development": gate_result["eligible_for_development"],
        "eligible_for_deployment": False,
        "scope": "frozen TRAIN holdout and challenge preservation evaluation only",
        "profile": PROFILE, "gate": gate_result, "suites": suites,
        "rows": len(outcomes), "cases_sha256": file_sha256(audit_path),
        "calibration_report_sha256": sha256_bytes(calibration["report_raw"]),
        "selector_report_sha256": sha256_bytes(calibration["selector"]["report_raw"]),
        "capture_manifest_sha256": {
            role: sha256_bytes(capture["manifest_raw"]) for role, capture in captures.items()},
        "runtime_version": left["runtime_version"],
        "runtime_model_artifact_hashes": left["model_artifact_hashes"],
        "selector_checkpoint_sha256": calibration["selector"]["report"]["checkpoint_sha256"],
        "source_hashes": before, "source_hashes_after": after,
        "candidate_objects_only_reordered": True,
        "expected_used_for_candidate_production": False,
        "missing_candidate_coverage_retained": True,
        "fit_data_read": False, "calibration_optimized": False,
        "development_data_read": False, "final_data_read": False,
        "deployed": False, "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"passed": report["passed"], "gate": gate_result,
                      "suites": suites}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-report", type=Path, required=True)
    parser.add_argument("--calibration-report-sha256", required=True)
    parser.add_argument("--selector-report", type=Path, required=True)
    parser.add_argument("--selector-report-sha256", required=True)
    parser.add_argument("--natural-holdout", type=Path, required=True)
    parser.add_argument("--natural-manifest-sha256", required=True)
    parser.add_argument("--challenge-preservation", type=Path, required=True)
    parser.add_argument("--challenge-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
