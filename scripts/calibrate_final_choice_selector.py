"""Fit the final-choice accept/stay gate from the fixed natural CAL capture."""
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

from hanglish.core.neural_converter import file_sha256
from hanglish.models.sentence_ranker import SentenceRanker
from scripts.final_choice_capture import _validate_segments, validate_response
from scripts.final_choice_gate import (
    CALIBRATION_ROLE,
    GateLabelsInsufficient,
    MIN_CLASS_GROUPS,
    dumps_gate,
    fit_gate,
    loads_gate,
    predict_accept,
    predict_probability,
)
from scripts.final_choice_policy import decision_features, finite_candidate_indices
from scripts.prepare_final_choice_data import PARTITION_SHA256
from scripts.train_final_choice_selector import PARENT_WEIGHT, RECIPE as SELECTOR_RECIPE
from scripts.train_natural_sentence_ranker import rss_bytes


PLAN = ROOT / ".omx/artifacts/final-choice-learning-plan-v1.md"
PLAN_SHA256 = "9adcb7c6d0fbbe596e44cf734d5b3d7b269c7709ca084e82fd2e36783dc71c49"
GATE_SOURCE_SHA256 = "b86e85916f49d9c8a23124ca617780c98784fada24e44eb2f74fbc3a0eaf0da7"
OUTPUT = ROOT / ".omx/artifacts/final-choice-calibration-v1"
CALIBRATION_ROWS = 1024
PROFILE = {
    "name": "final-choice-natural-calibration-v1",
    "role": CALIBRATION_ROLE,
    "rows": CALIBRATION_ROWS,
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 1800,
    "progress_interval": 32,
    "score_tolerance": 1e-4,
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


def _runtime_evidence_key(path):
    value = Path(path)
    if not value.is_absolute():
        return str(value)
    try:
        return str(value.resolve().relative_to(ROOT))
    except ValueError:
        return str(value.resolve())


def _validate_calibration_row(row, selected, runtime_version, model_artifacts):
    identity_keys = ("id", "group_id", "origin", "input", "context", "expected")
    if ({key: row.get(key) for key in identity_keys}
            != {key: selected.get(key) for key in identity_keys}):
        raise ValueError("CAL row differs from its frozen selection")
    candidates = row.get("candidates")
    if (row.get("origin") != "natural" or not isinstance(candidates, list)
            or not 1 <= len(candidates) <= 5
            or row.get("result") != candidates[0].get("text")
            or row.get("runtime_version") != runtime_version
            or not isinstance(row.get("expected"), list) or not row["expected"]
            or any(not isinstance(value, str) or not value for value in row["expected"])
            or type(row.get("engine_ms")) not in (int, float)
            or not math.isfinite(row["engine_ms"]) or row["engine_ms"] < 0
            or type(row.get("raw_engine_ms")) not in (int, float)
            or not math.isfinite(row["raw_engine_ms"]) or row["raw_engine_ms"] < 0):
        raise ValueError("CAL row identity or candidates are malformed")
    texts = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("text"), str):
            raise ValueError("CAL candidate object is malformed")
        score = candidate.get("score")
        if (score is not None and (type(score) not in (int, float)
                or not math.isfinite(score) or round(score, 4) != score)):
            raise ValueError("CAL returned candidate score is malformed")
        _validate_segments(candidate, row["input"])
        if score is None and candidate != {
                "text": row["input"], "score": None,
                "segments": [{"input_start": 0, "input_end": len(row["input"]),
                              "source": row["input"], "text": row["input"],
                              "kind": "raw"}]}:
            raise ValueError("CAL None score is not the complete raw candidate")
        texts.append(candidate["text"])
    if len(texts) != len(set(texts)):
        raise ValueError("CAL candidate texts are duplicated")
    finite = finite_candidate_indices(candidates)
    positives = [index for index in finite if texts[index] in row["expected"]]
    can_score = row.get("candidate_can_score")
    if (not isinstance(can_score, list) or len(can_score) != len(candidates)
            or any(type(value) is not bool for value in can_score)):
        raise ValueError("CAL candidate budget flags are malformed")
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
        raise ValueError("CAL eligibility or positive indices differ")
    flags = row.get("response_flags", {})
    if (flags.get("version") != runtime_version
            or flags.get("model_artifacts") != model_artifacts
            or flags.get("raw") is not False
            or row.get("model_artifact_flags") != {
                "expected_count": len(model_artifacts), "response_matches_expected": True}):
        raise ValueError("CAL runtime identity differs")
    validate_response(
        row.get("raw_bypass_response"), row["input"], row["context"],
        runtime_version, model_artifacts, raw=True,
    )


def load_calibration_capture(directory, explicit_manifest_sha):
    """Load exactly the completed natural_calibration role; no other split is accepted."""
    manifest_raw, manifest = _read_json(directory / "manifest.json", explicit_manifest_sha)
    cases_raw, rows = _read_jsonl(directory / "cases.jsonl", manifest.get("cases_sha256"))
    selection_raw, selection = _read_jsonl(
        directory / "selection.jsonl", manifest.get("selection_sha256")
    )
    ordered_ids = manifest.get("partition_ordered_ids")
    if (manifest.get("completed") is not True or manifest.get("role") != CALIBRATION_ROLE
            or manifest.get("scope") != "fresh actual parent TRAIN capture for final-choice learning"
            or manifest.get("partition_manifest_sha256") != PARTITION_SHA256
            or manifest.get("expected_rows") != CALIBRATION_ROWS
            or manifest.get("rows") != CALIBRATION_ROWS
            or len(rows) != CALIBRATION_ROWS or len(selection) != CALIBRATION_ROWS
            or not isinstance(ordered_ids, list) or len(ordered_ids) != CALIBRATION_ROWS
            or [row.get("id") for row in rows] != ordered_ids
            or [row.get("id") for row in selection] != ordered_ids
            or len(set(ordered_ids)) != CALIBRATION_ROWS
            or manifest.get("selection_sha256_before_inference")
               != manifest.get("selection_sha256")
            or manifest.get("source_hashes") != manifest.get("source_hashes_after")
            or manifest.get("training_run") is not False
            or manifest.get("new_selector_inference_run") is not False
            or manifest.get("development_data_read") is not False
            or manifest.get("final_data_read") is not False
            or manifest.get("deployed") is not False):
        raise ValueError("completed exact natural_calibration capture is required")
    source_hashes = manifest.get("source_hashes")
    runtime_sources = manifest.get("runtime_source_hashes")
    model_artifacts = manifest.get("model_artifact_hashes")
    train_inputs = manifest.get("train_inputs")
    runtime_version = manifest.get("runtime_version")
    if (not isinstance(runtime_version, str) or not runtime_version
            or not _valid_hashes(source_hashes) or not _valid_hashes(runtime_sources)
            or not _valid_hashes(model_artifacts) or len(model_artifacts) != 16
            or any(source_hashes.get(name) != digest
                   for name, digest in {**runtime_sources, **model_artifacts}.items())
            or not isinstance(train_inputs, dict) or not train_inputs
            or any(not isinstance(value, dict)
                   or source_hashes.get(value.get("path")) != value.get("sha256")
                   for value in train_inputs.values())
            or source_hashes.get(_runtime_evidence_key(manifest.get("runtime_evidence_path", "")))
               != manifest.get("runtime_evidence_sha256")):
        raise ValueError("CAL capture provenance is malformed")
    for row, selected in zip(rows, selection):
        _validate_calibration_row(row, selected, runtime_version, model_artifacts)
    diagnostics = {
        "top1": sum(row["result"] in row["expected"] for row in rows),
        "top5": sum(any(candidate["text"] in row["expected"]
                        for candidate in row["candidates"]) for row in rows),
        "label_aligned": sum(row["label_aligned"] for row in rows),
        "selector_eligible": sum(row["selector_eligible"] for row in rows),
        "all_candidates_can_score": sum(row["all_candidates_can_score"] for row in rows),
    }
    if (manifest.get("diagnostics") != diagnostics
            or manifest.get("baseline_accuracy_is_diagnostic_only") is not True
            or manifest.get("split_or_threshold_selected_from_capture_outcomes") is not False):
        raise ValueError("CAL diagnostics do not reproduce from cases")
    return {"directory": directory, "manifest_raw": manifest_raw,
            "cases_raw": cases_raw, "selection_raw": selection_raw,
            "manifest": manifest, "rows": rows}


def load_selector(report_path, explicit_sha, calibration_manifest):
    report_raw, report = _read_json(report_path, explicit_sha)
    checkpoint = report_path.parent / "checkpoint"
    metadata_raw = (checkpoint / "metadata.json").read_bytes()
    metadata = json.loads(metadata_raw)
    if (report.get("scope") !=
            "FIT-only final-candidate selector training; no deployment decision"
            or report.get("selector_only") is not True
            or report.get("recipe") != SELECTOR_RECIPE
            or report.get("partition_manifest_sha256") != PARTITION_SHA256
            or report.get("parent_checkpoint_sha256") != PARENT_WEIGHT
            or report.get("checkpoint_sha256")
               != file_sha256(checkpoint / "model.safetensors")
            or report.get("source_hashes") != report.get("source_hashes_after")
            or set(report.get("fit_manifest_sha256", {}))
               != {"natural_fit", "challenge_fit"}
            or any(not isinstance(value, str) or len(value) != 64
                   for value in report.get("fit_manifest_sha256", {}).values())
            or type(report.get("optimizer_steps")) is not int
            or report["optimizer_steps"] <= 0
            or type(report.get("save_load_score_max_delta")) not in (int, float)
            or not math.isfinite(report["save_load_score_max_delta"])
            or report["save_load_score_max_delta"] > PROFILE["score_tolerance"]
            or report.get("runtime_version") != calibration_manifest["runtime_version"]
            or report.get("calibration_data_read") is not False
            or report.get("holdout_data_read") is not False
            or report.get("preservation_data_read") is not False
            or report.get("development_data_read") is not False
            or report.get("final_data_read") is not False
            or report.get("eligible_for_deployment") is not False
            or report.get("deployed") is not False):
        raise ValueError("completed FIT-only selector report is required")
    if (metadata.get("format_version") != 1 or metadata.get("selector_only") is not True
            or metadata.get("training") != SELECTOR_RECIPE
            or metadata.get("parent_checkpoint_sha256") != PARENT_WEIGHT
            or metadata.get("parent_runtime_version") != report["runtime_version"]
            or metadata.get("frozen_source_hashes") != report["source_hashes"]
            or {role: value.get("manifest_sha256")
                for role, value in metadata.get("fit_captures", {}).items()}
               != report["fit_manifest_sha256"]
            or metadata.get("max_length") != SELECTOR_RECIPE["max_length"]
            or metadata.get("inference_batch_size")
               != SELECTOR_RECIPE["inference_batch_size"]
            or metadata.get("calibration_data_read") is not False
            or metadata.get("holdout_data_read") is not False
            or metadata.get("preservation_data_read") is not False
            or metadata.get("development_data_read") is not False
            or metadata.get("final_data_read") is not False
            or metadata.get("deployed") is not False):
        raise ValueError("selector checkpoint metadata differs from its report")
    artifact_hashes = metadata.get("artifact_hashes")
    if not _valid_hashes(artifact_hashes):
        raise ValueError("selector checkpoint artifact hashes are malformed")
    for name, digest in artifact_hashes.items():
        if file_sha256(checkpoint / name) != digest:
            raise ValueError("selector checkpoint artifact differs")
    for name, digest in report["source_hashes"].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if file_sha256(path) != digest:
            raise ValueError(f"selector training source changed: {name}")
    return {"report_raw": report_raw, "report": report, "checkpoint": checkpoint,
            "metadata_raw": metadata_raw, "metadata": metadata}


def score_calibration_case(ranker, row):
    """Score source-only candidates and call the frozen decision policy before labels."""
    if not row["selector_eligible"] or not row["all_candidates_can_score"]:
        return {"id": row["id"], "group_id": row["group_id"],
                "role": CALIBRATION_ROLE, "reason": "capture_ineligible",
                "selector_scores": [], "proposal_index": 0,
                "parent_accepted": None, "proposal_accepted": None}
    candidates = row["candidates"]
    can_score = [bool(ranker.can_score(candidate["text"])) for candidate in candidates]
    if can_score != row["candidate_can_score"] or not all(can_score):
        raise ValueError("selector candidate budget differs from CAL capture")
    scores = ranker.score_pairs(
        [row["context"]] * len(candidates), [candidate["text"] for candidate in candidates]
    )
    if (len(scores) != len(candidates)
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in scores)):
        raise ValueError("selector must score the complete CAL candidate list")
    decision = decision_features(row["input"], row["context"], candidates, scores)
    result = {"id": row["id"], "group_id": row["group_id"],
              "role": CALIBRATION_ROLE, "reason": decision["reason"],
              "selector_scores": [float(value) for value in scores],
              "proposal_index": decision["proposal_index"],
              "features": decision.get("features"),
              "finite_indices": decision.get("finite_indices")}
    # Calibration labels are accessed only after complete scoring and policy evaluation.
    expected = row["expected"]
    proposed = decision["proposal_index"]
    result["parent_accepted"] = candidates[0]["text"] in expected
    result["proposal_accepted"] = candidates[proposed]["text"] in expected
    return result


def calibration_rows(audit):
    return [{key: row[key] for key in
             ("group_id", "role", "reason", "features",
              "parent_accepted", "proposal_accepted")}
            for row in audit if row["reason"] == "eligible_disagreement"]


def class_counts(rows):
    accept = [row for row in rows
              if not row["parent_accepted"] and row["proposal_accepted"]]
    stay = [row for row in rows
            if row["parent_accepted"] or not row["proposal_accepted"]]
    return {"rows": len(rows), "accept_rows": len(accept), "stay_rows": len(stay),
            "accept_groups": len({row["group_id"] for row in accept}),
            "stay_groups": len({row["group_id"] for row in stay})}


def fit_and_persist_gate(output, fit_rows, audit):
    """Persist a gate only when both frozen class group minima are satisfied."""
    counts = class_counts(fit_rows)
    try:
        if min(counts["accept_groups"], counts["stay_groups"]) < MIN_CLASS_GROUPS:
            raise GateLabelsInsufficient(
                "gate labels insufficient: accept and stay each require at least "
                f"{MIN_CLASS_GROUPS} distinct natural_calibration groups"
            )
        gate = fit_gate(fit_rows)
    except GateLabelsInsufficient as error:
        return None, "labels_insufficient", str(error), None, counts
    gate_path = output / "gate.json"
    serialized = dumps_gate(gate)
    with gate_path.open("x", encoding="utf-8") as stream:
        stream.write(serialized + "\n")
    loaded = loads_gate(gate_path.read_text(encoding="utf-8"))
    memory = [(predict_probability(gate, row["features"]),
               predict_accept(gate, row["features"])) for row in fit_rows]
    saved = [(predict_probability(loaded, row["features"]),
              predict_accept(loaded, row["features"])) for row in fit_rows]
    if memory != saved:
        raise RuntimeError("saved final-choice gate predictions differ")
    prediction_by_id = {row["id"]: prediction for row, prediction in zip(
        (row for row in audit if row["reason"] == "eligible_disagreement"), saved
    )}
    for row in audit:
        prediction = prediction_by_id.get(row["id"])
        row["gate_probability"] = None if prediction is None else prediction[0]
        row["gate_accept"] = None if prediction is None else prediction[1]
    parity = {"calibration_rows": len(audit), "feature_rows": len(fit_rows),
              "nonapplicable_rows": len(audit) - len(fit_rows), "exact": True,
              "gate_sha256": file_sha256(gate_path)}
    return gate, "completed", None, parity, counts


def merge_hashes(target, additions, label):
    for name, digest in additions.items():
        if name in target and target[name] != digest:
            raise ValueError(f"conflicting {label} hash: {name}")
        target[name] = digest


def provenance_hashes(capture, selector, runner_sha):
    hashes = dict(capture["manifest"]["source_hashes"])
    capture_relative = str(capture["directory"].relative_to(ROOT))
    hashes[f"{capture_relative}/manifest.json"] = sha256_bytes(capture["manifest_raw"])
    hashes[f"{capture_relative}/cases.jsonl"] = sha256_bytes(capture["cases_raw"])
    hashes[f"{capture_relative}/selection.jsonl"] = sha256_bytes(capture["selection_raw"])
    report_path = selector["checkpoint"].parent / "report.json"
    selector_relative = str(selector["checkpoint"].parent.relative_to(ROOT))
    additions = {
        str(report_path.relative_to(ROOT)): sha256_bytes(selector["report_raw"]),
        f"{selector_relative}/checkpoint/metadata.json": sha256_bytes(selector["metadata_raw"]),
        str(PLAN.relative_to(ROOT)): PLAN_SHA256,
        "scripts/final_choice_gate.py": GATE_SOURCE_SHA256,
        "scripts/final_choice_policy.py": file_sha256(ROOT / "scripts/final_choice_policy.py"),
        "scripts/final_choice_capture.py": file_sha256(ROOT / "scripts/final_choice_capture.py"),
        "scripts/prepare_final_choice_data.py": file_sha256(ROOT / "scripts/prepare_final_choice_data.py"),
        "scripts/train_final_choice_selector.py": file_sha256(
            ROOT / "scripts/train_final_choice_selector.py"),
        "scripts/calibrate_final_choice_selector.py": runner_sha,
    }
    for name, digest in selector["metadata"]["artifact_hashes"].items():
        additions[f"{selector_relative}/checkpoint/{name}"] = digest
    merge_hashes(hashes, additions, "calibration dependency")
    return dict(sorted(hashes.items()))


def verify_hashes(hashes):
    observed = {}
    for name, expected in hashes.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        observed[name] = file_sha256(path)
        if observed[name] != expected:
            raise ValueError(f"CAL input changed: {name}")
    return observed


def require_resources(started, stage):
    if (rss_bytes() > PROFILE["rss_limit_bytes"]
            or time.monotonic() - started > PROFILE["time_limit_seconds"]):
        raise RuntimeError(f"final-choice calibration resource bound exceeded during {stage}")


def run(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = Path(__file__).read_bytes()
    source_sha = sha256_bytes(source)
    if file_sha256(PLAN) != PLAN_SHA256 or file_sha256(
            ROOT / "scripts/final_choice_gate.py") != GATE_SOURCE_SHA256:
        raise ValueError("frozen final-choice plan or gate differs")
    capture = load_calibration_capture(args.calibration, args.calibration_manifest_sha256)
    selector = load_selector(args.selector_report, args.selector_report_sha256,
                             capture["manifest"])
    before = provenance_hashes(capture, selector, source_sha)
    verify_hashes(before)
    require_resources(started, "pre-inference validation")

    torch.set_num_threads(PROFILE["threads"])
    os.environ["OMP_NUM_THREADS"] = str(PROFILE["threads"])
    ranker = SentenceRanker.load(selector["checkpoint"])
    if (ranker.max_length != SELECTOR_RECIPE["max_length"]
            or ranker.batch_size != SELECTOR_RECIPE["inference_batch_size"]):
        raise ValueError("loaded selector inference bounds differ")
    audit = []
    for index, row in enumerate(capture["rows"]):
        audit.append(score_calibration_case(ranker, row))
        require_resources(started, "CAL scoring")
        if (index + 1) % PROFILE["progress_interval"] == 0:
            print(json.dumps({"event": "final_choice_calibration", "rows": index + 1,
                              "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
    fit_rows = calibration_rows(audit)
    args.output.mkdir(parents=True)
    gate, status, reason, parity, counts = fit_and_persist_gate(
        args.output, fit_rows, audit
    )
    audit_path = args.output / "scoring-cases.jsonl"
    with audit_path.open("x") as stream:
        for row in audit:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    load_calibration_capture(args.calibration, args.calibration_manifest_sha256)
    load_selector(args.selector_report, args.selector_report_sha256, capture["manifest"])
    after = provenance_hashes(capture, selector, source_sha)
    verify_hashes(after)
    if after != before or file_sha256(Path(__file__)) != source_sha:
        raise RuntimeError("CAL sources changed")
    source_path = args.output / "source.py"
    source_path.write_bytes(source)
    if file_sha256(source_path) != source_sha:
        raise RuntimeError("CAL source snapshot differs")
    require_resources(started, "post-validation")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status, "passed": gate is not None,
        "eligible_for_holdout_evaluation": gate is not None,
        "scope": "natural CAL-only final-choice accept/stay gate calibration",
        "reason": reason, "profile": PROFILE, "class_counts": counts,
        "calibration_manifest_sha256": sha256_bytes(capture["manifest_raw"]),
        "selector_report_sha256": sha256_bytes(selector["report_raw"]),
        "selector_checkpoint_sha256": selector["report"]["checkpoint_sha256"],
        "selector_checkpoint_artifact_hashes": selector["metadata"]["artifact_hashes"],
        "runtime_version": capture["manifest"]["runtime_version"],
        "runtime_model_artifact_hashes": capture["manifest"]["model_artifact_hashes"],
        "source_hashes": before, "source_hashes_after": after,
        "scoring_cases_sha256": file_sha256(audit_path),
        "gate_prediction_save_load_parity": parity,
        "rows": len(audit), "eligible_disagreements": len(fit_rows),
        "labels_used_after_selector_scoring_and_policy": True,
        "resplit_or_fit_fallback_used": False,
        "fit_data_read": False, "holdout_data_read": False,
        "preservation_data_read": False, "development_data_read": False,
        "final_data_read": False, "eligible_for_deployment": False,
        "deployed": False, "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"status": status, "class_counts": counts,
                      "eligible_for_holdout_evaluation": gate is not None}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-report", type=Path, required=True)
    parser.add_argument("--selector-report-sha256", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--calibration-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
