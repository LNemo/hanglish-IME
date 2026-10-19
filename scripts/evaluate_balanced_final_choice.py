"""Evaluate the frozen balanced selector on fresh EVAL-v2 and preservation."""
from __future__ import annotations

import argparse
from copy import deepcopy
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

from hanglish.models.sentence_ranker import SentenceRanker
from scripts.balanced_final_choice_policy import select_candidates
from scripts.evaluate_final_choice_holdout import (
    _validate_row, load_evaluation_capture, merge_hashes, verify_hashes,
)
from scripts.prepare_balanced_selector_evaluation import (
    PROFILE as CAPTURE_PROFILE, RUNTIME_VERSION, load_balanced_selector_artifact,
    read_evaluation_sources,
)
from scripts.prepare_final_choice_data import read_runtime_evidence
from scripts.train_balanced_final_choice_selector import (
    PLAN_SHA256, EVALUATION_PARTITION_SHA256, read_evaluation_partition,
)
from scripts.train_natural_sentence_ranker import rss_bytes

SELECTOR_REPORT_SHA = "f04c61ffc7fc03c5f2215f5a82b6b248c1d236dd969a8cae46863dfccf4005af"
SELECTOR_WEIGHT_SHA = "bd76d6b2aefd9826a7a34ac06366cfd0fa47a7bff78e3968f0a507bb02731028"
RUNTIME_EVIDENCE_SHA = "77c04334af817e15bc4c5ccc9d27b1d6ab9f9f6a759b4207e752762ae49d3aab"
PRESERVATION_SHA = "c4f587d42ac0838ce7eb2f31881bc182d8f1e0615bab47965f35da032cbaa133"
FROZEN_SOURCES = {
    "scripts/balanced_final_choice_policy.py": "0a0afb0c0f73023f62d300917b05987585e62afeb7dc33a16ae563a8dd518d6e",
    "scripts/final_choice_policy.py": "3acb1480fce1c490892e003a9677b0ad5354d07e971dd3ca73446c08d69cf717",
    "scripts/evaluate_final_choice_holdout.py": "77b6743184cdc7a4ea23cee4a18e1c86f6899267954d9d34892a502163d13a33",
    "scripts/train_balanced_final_choice_selector.py": "808fc72567a21a62cb8ead9ef43ecbe1cbe8b377beeaf6bc40d2f73950ce9c9b",
}
PROFILE = {"threads": 2, "rss_limit_bytes": 3 * 1024**3,
           "time_limit_seconds": 1800, "minimum_natural_gain": 2}
OUTPUT = ROOT / ".omx/artifacts/balanced-final-choice-evaluation-v2"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path, expected):
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError(f"evaluation artifact changed: {path}")
    return raw, json.loads(raw)


def validate_fresh_row(row, selected, runtime):
    if row.get("origin") != "natural_eval_v2" or selected.get("origin") != "natural_eval_v2":
        raise ValueError("fresh evaluation row origin differs")
    # Preserve the v2 provenance in stored records. Adapt only local copies for
    # the frozen validator, whose natural-origin name belongs to its v1 schema.
    _validate_row(dict(row, origin="natural"), dict(selected, origin="natural"),
                  "natural_holdout", RUNTIME_VERSION, runtime["accepted_weight_hashes"])


def load_fresh_capture(directory, expected_sha, partition, runtime):
    raw, manifest = read_json(directory / "manifest.json", expected_sha)
    cases_raw = (directory / "cases.jsonl").read_bytes()
    selection_raw = (directory / "selection.jsonl").read_bytes()
    if (digest(cases_raw) != manifest.get("cases_sha256")
            or digest(selection_raw) != manifest.get("selection_sha256")):
        raise ValueError("fresh capture data hash differs")
    rows = [json.loads(line) for line in cases_raw.splitlines() if line]
    selection = [json.loads(line) for line in selection_raw.splitlines() if line]
    ids = partition["eval_ids"]
    if (manifest.get("scope") != "fresh actual parent capture for balanced final-choice EVAL-v2"
            or manifest.get("role") != "natural_evaluation_v2"
            or manifest.get("completed") is not True
            or manifest.get("rows") != len(ids) or manifest.get("expected_rows") != len(ids)
            or len(rows) != len(ids) or len(selection) != len(ids)
            or manifest.get("partition_ordered_ids") != ids
            or manifest.get("partition_ordered_group_ids") != partition["eval_group_ids"]
            or manifest.get("evaluation_partition_sha256") != EVALUATION_PARTITION_SHA256
            or [r.get("id") for r in rows] != ids
            or [r.get("id") for r in selection] != ids
            or manifest.get("profile") != CAPTURE_PROFILE
            or manifest.get("runtime_version") != RUNTIME_VERSION
            or manifest.get("runtime_source_hashes") != runtime["integrated_runtime_source_hashes"]
            or manifest.get("model_artifact_hashes") != runtime["accepted_weight_hashes"]
            or manifest.get("runtime_evidence_sha256") != RUNTIME_EVIDENCE_SHA
            or manifest.get("selector_report_sha256") != SELECTOR_REPORT_SHA
            or manifest.get("selector_checkpoint_sha256") != SELECTOR_WEIGHT_SHA
            or manifest.get("source_hashes") != manifest.get("source_hashes_after")
            or not manifest.get("source_hashes")
            or manifest.get("selection_sha256_before_inference") != digest(selection_raw)
            or manifest.get("selector_artifact_validated_without_inference") is not True
            or manifest.get("baseline_accuracy_is_diagnostic_only") is not True
            or any(manifest.get(key) is not False for key in (
                "selector_inference_run", "training_run", "new_selector_inference_run",
                "development_data_read", "final_data_read", "deployed",
                "split_or_threshold_selected_from_capture_outcomes"))):
        raise ValueError("fresh capture identity or scope differs")
    verify_hashes(manifest["source_hashes"])
    expected_sources, _ = read_evaluation_sources(partition)
    if selection != expected_sources:
        raise ValueError("fresh selection differs from frozen source rows")
    for row, selected in zip(rows, selection):
        validate_fresh_row(row, selected, runtime)
    diagnostics = {
        "top1": sum(r["result"] in r["expected"] for r in rows),
        "top5": sum(any(c["text"] in r["expected"] for c in r["candidates"]) for r in rows),
        "label_aligned": sum(r["label_aligned"] for r in rows),
        "selector_eligible": sum(r["selector_eligible"] for r in rows),
        "all_candidates_can_score": sum(r["all_candidates_can_score"] for r in rows),
    }
    if diagnostics != manifest.get("diagnostics"):
        raise ValueError("fresh capture diagnostics do not reproduce")
    return {"directory": directory, "manifest": manifest, "manifest_raw": raw,
            "cases_raw": cases_raw, "selection_raw": selection_raw, "rows": rows}


def evaluate_case(ranker, row, role):
    candidates = row["candidates"]
    before = deepcopy(candidates)
    actual_budget = [ranker.can_score(c["text"]) for c in candidates]
    if actual_budget != row["candidate_can_score"]:
        raise ValueError("selector candidate budget differs from capture")
    selected = select_candidates(ranker, row["input"], row["context"], candidates)
    chosen = selected["chosen_index"]
    ordered = selected["candidates"]
    expected_order = [candidates[chosen], *candidates[:chosen], *candidates[chosen + 1:]]
    if (candidates != before or len(ordered) != len(candidates)
            or any(a is not b for a, b in zip(ordered, expected_order))):
        raise RuntimeError("direct selector changed candidate objects or content")
    raw = row["raw_bypass_response"]
    raw_before = deepcopy(raw)
    raw_selected = select_candidates(None, row["input"], row["context"], raw["candidates"], raw=True)
    if (raw != raw_before or raw_selected["chosen_index"] != 0
            or raw_selected["decision"]["reason"] != "raw_mode"
            or raw_selected["candidates"] is not raw["candidates"]
            or raw_selected["result"] != raw["result"] or raw["result"] != row["input"]):
        raise RuntimeError("direct selector changed raw bypass")
    # Labels are only used after both normal and raw selection are complete.
    expected = row["expected"]
    return {"id": row["id"], "group_id": row["group_id"], "suite": role,
            "chosen_index": chosen, "reason": selected["decision"]["reason"],
            "selector_scored": selected["selector_scored"],
            "selector_scores": selected["selector_scores"],
            "before_candidates": before, "candidates": deepcopy(ordered),
            "raw_bypass_response": raw_before, "raw_bypass_preserved": True,
            "before_top1": before[0]["text"] in expected,
            "top1": ordered[0]["text"] in expected,
            "before_top5": any(c["text"] in expected for c in before),
            "top5": any(c["text"] in expected for c in ordered),
            "candidate_objects_and_content_preserved": True,
            "accepted_action": chosen != 0}


def summarize(rows):
    return {"rows": len(rows), "before_top1": sum(r["before_top1"] for r in rows),
            "top1": sum(r["top1"] for r in rows),
            "before_top5": sum(r["before_top5"] for r in rows),
            "top5": sum(r["top5"] for r in rows),
            "corrections": [r["id"] for r in rows if r["top1"] and not r["before_top1"]],
            "regressions": [r["id"] for r in rows if r["before_top1"] and not r["top1"]],
            "top5_losses": [r["id"] for r in rows if r["before_top5"] and not r["top5"]],
            "accepted_actions": sum(r["accepted_action"] for r in rows),
            "scored_rows": sum(r["selector_scored"] for r in rows)}


def evaluation_gate(suites):
    natural = suites["natural_evaluation_v2"]
    gain = natural["top1"] - natural["before_top1"]
    losses = any(s["regressions"] or s["top5_losses"] for s in suites.values())
    actions = sum(s["accepted_actions"] for s in suites.values())
    passed = gain >= PROFILE["minimum_natural_gain"] and not losses and actions > 0
    return {"passed": passed, "eligible_for_development": passed,
            "natural_net_top1_gain": gain, "zero_row_regressions": not losses,
            "accepted_actions": actions, "inconclusive": actions == 0}


def require_resources(started):
    if rss_bytes() > PROFILE["rss_limit_bytes"] or time.monotonic() - started > PROFILE["time_limit_seconds"]:
        raise RuntimeError("balanced evaluation resource bound exceeded")


def run(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_hashes(FROZEN_SOURCES)
    source = Path(__file__).read_bytes()
    if args.selector_report_sha256 != SELECTOR_REPORT_SHA:
        raise ValueError("only the frozen balanced v2 checkpoint may be evaluated")
    if args.runtime_evidence_sha256 != RUNTIME_EVIDENCE_SHA:
        raise ValueError("accepted runtime evidence differs")
    runtime_raw, runtime = read_runtime_evidence(args.runtime_evidence, args.runtime_evidence_sha256)
    selector = load_balanced_selector_artifact(args.selector_report, args.selector_report_sha256, runtime)
    if selector["report"]["checkpoint_sha256"] != SELECTOR_WEIGHT_SHA:
        raise ValueError("balanced selector checkpoint differs")
    partition_raw, partition = read_evaluation_partition(
        args.evaluation_partition, EVALUATION_PARTITION_SHA256, [])
    captures = {
        "natural_evaluation_v2": load_fresh_capture(args.natural_evaluation,
            args.natural_manifest_sha256, partition, runtime),
        "challenge_preservation": load_evaluation_capture(args.challenge_preservation,
            PRESERVATION_SHA, "challenge_preservation"),
    }
    before = dict(FROZEN_SOURCES)
    merge_hashes(before, selector["report"]["source_hashes"], "training")
    for cap in captures.values():
        if (cap["manifest"]["runtime_version"] != RUNTIME_VERSION
                or cap["manifest"]["model_artifact_hashes"] != runtime["accepted_weight_hashes"]):
            raise ValueError("evaluation capture runtime differs")
        merge_hashes(before, cap["manifest"]["source_hashes"], "capture")
        for name in ("manifest.json", "cases.jsonl", "selection.jsonl"):
            path = cap["directory"] / name
            before[str(path.relative_to(ROOT))] = digest(path.read_bytes())
    paths = [args.selector_report, args.selector_report.parent / "schedule.jsonl",
             selector["checkpoint"] / "metadata.json", args.evaluation_partition,
             args.runtime_evidence, Path(__file__), ROOT / "scripts/prepare_balanced_selector_evaluation.py"]
    paths += [selector["checkpoint"] / name for name in selector["metadata"]["artifact_hashes"]]
    for path in paths:
        name, value = str(path.relative_to(ROOT)), digest(path.read_bytes())
        merge_hashes(before, {name: value}, "evaluation dependency")
    verify_hashes(before)
    require_resources(started)
    torch.set_num_threads(PROFILE["threads"])
    os.environ["OMP_NUM_THREADS"] = str(PROFILE["threads"])
    ranker = SentenceRanker.load(selector["checkpoint"])
    args.output.mkdir(parents=True)
    cases_path = args.output / "cases.jsonl"
    outcomes = []
    with cases_path.open("x") as stream:
        for role, cap in captures.items():
            for row in cap["rows"]:
                outcome = evaluate_case(ranker, row, role)
                outcomes.append(outcome)
                stream.write(json.dumps(outcome, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                require_resources(started)
                if len(outcomes) % 32 == 0:
                    print(json.dumps({"event": "balanced_selector_evaluation", "rows": len(outcomes)}), flush=True)
    suites = {role: summarize([r for r in outcomes if r["suite"] == role]) for role in captures}
    gate = evaluation_gate(suites)
    verify_hashes(before)
    if digest(Path(__file__).read_bytes()) != digest(source):
        raise RuntimeError("evaluation source changed")
    (args.output / "source.py").write_bytes(source)
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "frozen balanced selector fresh EVAL-v2 and challenge regression only",
        "profile": PROFILE, "passed": gate["passed"], "gate": gate, "suites": suites,
        "eligible_for_development": gate["passed"], "eligible_for_deployment": False,
        "rows": len(outcomes), "cases_sha256": digest(cases_path.read_bytes()),
        "source_hashes": before, "source_hashes_after": dict(before),
        "selector_report_sha256": SELECTOR_REPORT_SHA, "selector_checkpoint_sha256": SELECTOR_WEIGHT_SHA,
        "plan_sha256": PLAN_SHA256, "evaluation_partition_sha256": EVALUATION_PARTITION_SHA256,
        "capture_manifest_sha256": {role: digest(cap["manifest_raw"]) for role, cap in captures.items()},
        "runtime_version": RUNTIME_VERSION, "model_artifact_hashes": runtime["accepted_weight_hashes"],
        "candidate_objects_only_reordered": True, "raw_bypass_preserved": True,
        "separate_learned_gate": False, "training_run": False,
        "development_data_read": False, "final_data_read": False, "deployed": False,
        "preservation_is_new_independent_evaluation": False,
        "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": rss_bytes()}
    require_resources(started)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"gate": gate, "suites": suites}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("selector-report", "runtime-evidence", "evaluation-partition",
                 "natural-evaluation", "challenge-preservation"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("selector-report-sha256", "runtime-evidence-sha256", "natural-manifest-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
