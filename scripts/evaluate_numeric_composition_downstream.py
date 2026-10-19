"""Capture and evaluate the reviewed numeric-composition downstream cohort.

The pipeline writes all source-only predictions before reopening target records.
It is an isolated experiment and never mutates the production runtime.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCHEMA = "numeric-composition-downstream-v1"
RUNTIME_VERSION = "neural-1-867f13596ed5"
FRESH_ROWS = 512
FRESH_GROUPS = 387
PRESERVATION_ROWS = 376
TOTAL_ROWS = 888
PROFILE = {"cpu_threads": 2, "rss_limit_bytes": 3 * 1024**3,
           "time_limit_seconds": 1800.0, "progress_interval": 32}
FIT_PROOF = ROOT / ".omx/artifacts/numeric-composition-fit-v1-proof-v2.json"
FIT_PROOF_SHA256 = "c0b5c6f59b35b28122f165ed1327f25c07f8cca0ef26197bf473aed52ff3d933"
FIT_VERIFIER_SHA256 = "a21a05b03b549d2aced7e7212ffd3863a414848d4cc2a977ac7787448c9a672d"
FIT_REPORT_SHA256 = "ad51fc2c014fd076301be65a522d73b662cf8a781e20ef08c3e749688753fd9f"
FIT_CAPTURES_SHA256 = "eb489a8c3ba932bfde13afad8824ae8401d55f988f6b0dda00d7d560743bacdc"
FIT_SOURCES_SHA256 = "a28176b7e67a7c8370fef9994877220c4a76dc963d726e4f1fea8803fe9ed209"
DATA = ROOT / ".omx/artifacts/long-context-lora-data-v1-review2"
DATA_MANIFEST_SHA256 = "cb4558249aa9acefb246abca9275ba8020f39cee27eb9d4143dfb94f266ecd06"
FRESH_SELECTION_SHA256 = "fb1257833c99acc406a831336f535d9ef3a786fc38bb08a84902598e5fe2287d"
FRESH_RAW = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
FRESH_RAW_SHA256 = "904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b"
PRESERVATION = ROOT / "data/natural_context/final-choice-v1/challenge_preservation"
PRESERVATION_MANIFEST_SHA256 = "c4f587d42ac0838ce7eb2f31881bc182d8f1e0615bab47965f35da032cbaa133"
PRESERVATION_LOADER_SHA256 = "77b6743184cdc7a4ea23cee4a18e1c86f6899267954d9d34892a502163d13a33"
RUNTIME_REPORT = ROOT / ".omx/artifacts/numeric-structure-integration-v1/report.json"
RUNTIME_REPORT_SHA256 = "77c04334af817e15bc4c5ccc9d27b1d6ab9f9f6a759b4207e752762ae49d3aab"
HANDOFF = ROOT / ".omx/plans/numeric-composition-downstream-handoff-v1.json"
HANDOFF_SHA256 = "bff235188624cab16ad7f103014c20d8bbc35e83e834da648508a28d44818fee"
FLAG_FIELDS = ("engine", "mode", "raw", "version", "ranker", "model_artifacts",
               "context_used", "external_context_used", "ranking_applied",
               "sentence_ranking_applied", "warnings")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest_value(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def path_key(path):
    return str(Path(path).resolve().relative_to(ROOT))


def write_jsonl(path, rows):
    raw = b"".join(canonical(row) + b"\n" for row in rows)
    Path(path).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_bytes().splitlines()]


def verify_hashes(values):
    if (not isinstance(values, dict) or not values
            or any(not isinstance(name, str) or not name
                   or not isinstance(value, str) or len(value) != 64
                   for name, value in values.items())):
        raise ValueError("source hash inventory is malformed")
    for name, expected in values.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if file_digest(path) != expected:
            raise ValueError(f"source changed: {name}")


def merge_hashes(target, paths):
    for path in paths:
        name, value = path_key(path), file_digest(path)
        if name in target and target[name] != value:
            raise ValueError(f"source hash conflict: {name}")
        target[name] = value


def merge_inventory(target, values):
    for name, value in values.items():
        if name in target and target[name] != value:
            raise ValueError(f"source hash conflict: {name}")
        target[name] = value


def validate_static_entry():
    """Validate reviewed authority and source-only entry before target access."""
    if file_digest(HANDOFF) != HANDOFF_SHA256:
        raise ValueError("numeric downstream handoff changed")
    handoff = json.loads(HANDOFF.read_bytes())
    if (handoff.get("schema") != "numeric-composition-downstream-handoff-v1"
            or handoff.get("terminal_planning_state") != "complete"
            or handoff.get("review") != "CLEAR"
            or handoff.get("model_concurrency_allowed") is not False
            or handoff.get("promotion_allowed") is not False):
        raise ValueError("numeric downstream handoff is not executable")
    for name, expected in handoff.get("files", {}).items():
        if file_digest(ROOT / name) != expected:
            raise ValueError(f"downstream authority changed: {name}")
    if file_digest(FIT_PROOF) != FIT_PROOF_SHA256:
        raise ValueError("FIT proof-v2 changed")
    envelope = json.loads(FIT_PROOF.read_bytes())
    payload = envelope.get("payload")
    if (envelope.get("schema") != "numeric-composition-fit-proof-v2"
            or not isinstance(payload, dict)
            or envelope.get("digest") != digest_value(payload)
            or payload.get("verification_passed") is not True
            or payload.get("quality_gate_passed") is not True
            or payload.get("downstream_evaluation_allowed") is not True
            or payload.get("promotion_allowed") is not False
            or payload.get("deployed") is not False
            or payload.get("verifier") != {"path": "scripts/verify_numeric_composition_fit.py",
                                            "sha256": FIT_VERIFIER_SHA256}
            or payload.get("report", {}).get("sha256") != FIT_REPORT_SHA256
            or payload.get("captures", {}).get("sha256") != FIT_CAPTURES_SHA256
            or payload.get("sources", {}).get("sha256") != FIT_SOURCES_SHA256):
        raise ValueError("FIT proof-v2 semantics or lineage differ")
    fit_report = json.loads((ROOT / payload["report"]["path"]).read_bytes())
    if (file_digest(ROOT / payload["report"]["path"]) != FIT_REPORT_SHA256
            or file_digest(ROOT / payload["captures"]["path"]) != FIT_CAPTURES_SHA256
            or file_digest(ROOT / payload["sources"]["path"]) != FIT_SOURCES_SHA256
            or file_digest(ROOT / payload["verifier"]["path"]) != FIT_VERIFIER_SHA256
            or digest_value(fit_report.get("source_hashes_before"))
               != payload.get("source_hashes_sha256")):
        raise ValueError("FIT proof-v2 bound files differ")
    verify_hashes(fit_report["source_hashes_before"])

    manifest_path = DATA / "manifest.json"
    selection_path = DATA / "fresh_selection.jsonl"
    if (file_digest(manifest_path) != DATA_MANIFEST_SHA256
            or file_digest(selection_path) != FRESH_SELECTION_SHA256):
        raise ValueError("frozen fresh partition changed")
    manifest = json.loads(manifest_path.read_bytes())
    fresh = manifest.get("fresh", {})
    limitations = manifest.get("historical_qwen_limitations")
    expected_limitations = {
        "group_collision_check": "unavailable: historical messages have no group IDs",
        "id_collision_check": "unavailable: historical messages have no IDs",
        "semantic_family_independence": "unestablished",
        "upstream_pretraining_overlap": "unknown",
    }
    if (manifest.get("file_hashes", {}).get("fresh_selection.jsonl") != FRESH_SELECTION_SHA256
            or fresh.get("rows") != FRESH_ROWS or fresh.get("groups") != FRESH_GROUPS
            or fresh.get("whole_group_selection") is not True
            or fresh.get("target_fields_copied") is not False
            or fresh.get("selection_fields") != ["id", "group_id"]
            or len(fresh.get("group_ids", [])) != FRESH_GROUPS
            or limitations != expected_limitations):
        raise ValueError("frozen fresh partition/collision contract differs")
    selection = read_jsonl(selection_path)
    if (len(selection) != FRESH_ROWS or any(set(row) != {"id", "group_id"} for row in selection)
            or len({row["id"] for row in selection}) != FRESH_ROWS
            or len({row["group_id"] for row in selection}) != FRESH_GROUPS):
        raise ValueError("frozen fresh selection identities differ")
    if file_digest(PRESERVATION / "manifest.json") != PRESERVATION_MANIFEST_SHA256:
        raise ValueError("preservation manifest changed")
    if file_digest(ROOT / "scripts/evaluate_final_choice_holdout.py") != PRESERVATION_LOADER_SHA256:
        raise ValueError("preservation loader changed")
    from scripts.prepare_final_choice_data import read_runtime_evidence
    from scripts import prepare_long_context_lora_data as preparation
    _, runtime = read_runtime_evidence(RUNTIME_REPORT, RUNTIME_REPORT_SHA256)
    if runtime.get("integrated_runtime_version") != RUNTIME_VERSION:
        raise ValueError("production runtime version differs")
    source_hashes = dict(manifest.get("source_hashes", {}))
    merge_inventory(source_hashes, fit_report["source_hashes_before"])
    merge_inventory(source_hashes, runtime.get("integrated_runtime_source_hashes", {}))
    merge_inventory(source_hashes, runtime.get("accepted_weight_hashes", {}))
    fixed_paths = [FIT_PROOF, HANDOFF, manifest_path, selection_path, RUNTIME_REPORT,
        ROOT / "scripts/numeric_composition_runtime.py",
        ROOT / "hanglish/core/numeric_composition.py",
        ROOT / "scripts/prepare_long_context_lora_data.py",
        ROOT / "scripts/prepare_long_context_lora_fresh_capture.py",
        ROOT / "scripts/evaluate_final_choice_holdout.py",
        ROOT / "scripts/final_choice_capture.py",
        ROOT / "scripts/audit_final_choice_candidate_stages.py",
        ROOT / "scripts/long_context_lora_fresh_metrics.py",
        ROOT / "scripts/prepare_final_choice_data.py",
        ROOT / "hanglish/core/neural_converter.py",
        ROOT / "hanglish/core/phonology.py", preparation.PARTITION, Path(__file__)]
    fixed_paths.extend(ROOT / name for name in handoff["files"])
    merge_hashes(source_hashes, fixed_paths)
    verify_hashes(source_hashes)
    return {"manifest": manifest, "selection": selection, "runtime": runtime,
            "source_hashes": dict(sorted(source_hashes.items())),
            "historical_qwen_limitations": limitations}


def load_and_split_targets(entry, access):
    """Access target-bearing files once, then split inference and target records."""
    access["fresh_target_bytes_accessed"] = True
    fresh_raw = FRESH_RAW.read_bytes()
    if hashlib.sha256(fresh_raw).hexdigest() != FRESH_RAW_SHA256:
        raise ValueError("fresh Korean source changed")
    from scripts import prepare_long_context_lora_data as preparation
    raw_rows = [json.loads(line) for line in fresh_raw.splitlines() if line]
    _, partition = preparation.read_pinned_json(
        preparation.PARTITION, preparation.PARTITION_SHA256)
    collision_ids, collision_groups, collision_pairs, collision_hashes = (
        preparation.collision_inventory())
    groups, fresh_sources, excluded, admissible_groups, admissible_rows = (
        preparation.select_fresh(raw_rows, partition, collision_ids,
                                 collision_groups, collision_pairs))
    fresh_manifest = entry["manifest"]["fresh"]
    selected = [{"id": row["id"], "group_id": row["group_id"]}
                for row in fresh_sources]
    if (selected != entry["selection"] or groups != fresh_manifest["group_ids"]
            or excluded != fresh_manifest["excluded_groups"]
            or admissible_groups != fresh_manifest["admissible_groups"]
            or admissible_rows != fresh_manifest["admissible_rows"]):
        raise ValueError("fresh reserve/collision replay differs")
    from hanglish.core.phonology import romanize_word
    from scripts.prepare_long_context_lora_fresh_capture import select_fresh_sources
    joined_fresh = select_fresh_sources(entry["selection"], raw_rows, romanize_word)
    replayed_sources = [{name: row[name] for name in
                         ("id", "group_id", "input", "context")}
                        for row in joined_fresh]
    if replayed_sources != fresh_sources:
        raise ValueError("fresh selected input/context replay differs")
    access["preservation_target_bytes_accessed"] = True
    from scripts.evaluate_final_choice_holdout import load_evaluation_capture
    preservation = load_evaluation_capture(
        PRESERVATION, PRESERVATION_MANIFEST_SHA256, "challenge_preservation")
    if (len(preservation["rows"]) != PRESERVATION_ROWS
            or preservation["manifest"].get("runtime_version") != RUNTIME_VERSION
            or preservation["manifest"].get("model_artifact_hashes")
               != entry["runtime"]["accepted_weight_hashes"]):
        raise ValueError("frozen preservation cohort differs")
    rows = [("fresh_natural", row, None) for row in joined_fresh]
    rows += [("challenge_preservation", row, row) for row in preservation["rows"]]
    sources, targets, baselines = [], [], []
    for role, row, baseline in rows:
        source = {name: row[name] for name in ("id", "group_id", "origin", "input", "context")}
        target = {"id": row["id"], "group_id": row["group_id"], "role": role,
                  "expected": deepcopy(row["expected"])}
        sources.append(source); targets.append(target)
        if baseline is not None:
            baselines.append({"id": row["id"], "result": row["result"],
                "candidates": deepcopy(row["candidates"]),
                "response_flags": deepcopy(row["response_flags"]),
                "runtime_version": row["runtime_version"],
                "raw_bypass_response": _without_timing(row["raw_bypass_response"])})
    if (len(sources) != TOTAL_ROWS or len(targets) != TOTAL_ROWS
            or len(baselines) != PRESERVATION_ROWS
            or len({row["id"] for row in sources}) != TOTAL_ROWS):
        raise ValueError("downstream source/target split accounting differs")
    target_paths = [FRESH_RAW, PRESERVATION / "manifest.json",
                    PRESERVATION / "cases.jsonl", PRESERVATION / "selection.jsonl",
                    ROOT / "scripts/evaluate_final_choice_holdout.py",
                    ROOT / "scripts/prepare_long_context_lora_fresh_capture.py",
                    ROOT / "hanglish/core/phonology.py"]
    target_hashes = {}
    merge_hashes(target_hashes, target_paths)
    merge_inventory(target_hashes, collision_hashes)
    merge_inventory(target_hashes, preservation["manifest"].get("source_hashes", {}))
    verify_hashes(target_hashes)
    return sources, targets, baselines, dict(sorted(target_hashes.items()))


def _flags(response):
    return {name: deepcopy(response.get(name)) for name in FLAG_FIELDS}


def load_models(expected_artifacts):
    import torch
    from scripts.audit_final_choice_candidate_stages import StageConverter, SentenceRecorder
    from scripts.numeric_composition_runtime import NumericCompositionRuntime

    torch.set_num_threads(PROFILE["cpu_threads"])
    os.environ["OMP_NUM_THREADS"] = str(PROFILE["cpu_threads"])
    parent = StageConverter(ranker_kind="hybrid", external_lexicon=True)
    parent.load_models()
    if parent.version != RUNTIME_VERSION or parent.model_artifacts != expected_artifacts:
        raise ValueError("loaded production model identity differs")
    recorder = SentenceRecorder(parent.sentence_ranker)
    parent.sentence_ranker = recorder
    child = NumericCompositionRuntime.__new__(NumericCompositionRuntime)
    child.__dict__ = parent.__dict__.copy()
    return {"parent": parent, "child": child, "recorder": recorder}


def _without_timing(response):
    return {name: deepcopy(value) for name, value in response.items() if name != "engine_ms"}


def capture_source(bundle, source, baseline, expected_artifacts):
    from scripts import audit_final_choice_candidate_stages as prior
    from scripts.final_choice_capture import validate_response

    parent, child, recorder = bundle["parent"], bundle["child"], bundle["recorder"]
    text, context = source["input"], source["context"]
    recorder.reset()
    before = parent.convert(text, context)
    learned, static = prior.beam_paths(parent.audit_pieces), prior.beam_paths(parent.audit_static_pieces)
    pool_probe = len(learned) == 1 or (len(learned) > 1 and len(text.split()) > 1)
    base = dict((path[0], path) for path in learned)
    if pool_probe:
        for path in static:
            base.setdefault(path[0], path)
    parent_base = [{"text": path[0], "score": path[1], "segments": deepcopy(path[2])}
                   for path in base.values()]
    if before["sentence_ranking_applied"]:
        _, _, full, _, _ = prior.reconstruct_pool(
            parent, parent.audit_pieces, parent.audit_static_pieces, sentence_possible=True)
        parent_pool = [{"text": path[0], "score": path[1], "segments": deepcopy(path[2])}
                       for path in full]
        if recorder.candidates != [path["text"] for path in parent_pool]:
            raise ValueError("production sentence pool reconstruction differs")
    else:
        parent_pool = [{"text": path[0], "score": path[1], "segments": deepcopy(path[2])}
                       for path in learned]
    recorder.reset()
    after = child.convert(text, context)
    trace = deepcopy(child.numeric_composition_trace)
    score_inputs = deepcopy(recorder.candidates)
    parent_raw = parent.convert(text, context, raw=True)
    child_raw = child.convert(text, context, raw=True)
    for response, raw in ((before, False), (after, False), (parent_raw, True), (child_raw, True)):
        validate_response(response, text, context, RUNTIME_VERSION, expected_artifacts, raw=raw)
    if _without_timing(parent_raw) != _without_timing(child_raw):
        raise ValueError("numeric runtime changed raw response")
    if trace["base"] != parent_base:
        raise ValueError("numeric runtime changed original base pool")
    if before["sentence_ranking_applied"]:
        preserved = trace["base"] + trace["accepted_old_extras"]
        if preserved != parent_pool or trace["scoring_pool"][:len(preserved)] != preserved:
            raise ValueError("numeric runtime changed scored old pool")
    if after["sentence_ranking_applied"]:
        if score_inputs != [path["text"] for path in trace["scoring_pool"]]:
            raise ValueError("numeric runtime scorer trace differs")
    elif score_inputs is not None:
        raise ValueError("unexpected numeric runtime scorer call")
    if baseline is not None:
        if (baseline["result"] != before["result"]
                or baseline["candidates"] != before["candidates"]
                or baseline["response_flags"] != _flags(before)
                or baseline["runtime_version"] != before["version"]
                or baseline["raw_bypass_response"] != _without_timing(parent_raw)):
            raise ValueError("preservation production baseline changed")
    return {"id": source["id"], "group_id": source["group_id"], "origin": source["origin"],
            "input": text, "context": context, "parent": _without_timing(before),
            "child": _without_timing(after), "parent_raw": _without_timing(parent_raw),
            "child_raw": _without_timing(child_raw), "parent_pool": parent_pool,
            "parent_base": parent_base, "trace": trace,
            "parent_sentence_scored": before["sentence_ranking_applied"],
            "child_sentence_scored": after["sentence_ranking_applied"],
            "child_score_inputs": score_inputs, "parent_ms": before["engine_ms"],
            "child_ms": after["engine_ms"]}


def release_models(bundle):
    bundle.clear()
    gc.collect()


def resources(started, stage):
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = int(usage if sys.platform == "darwin" else usage * 1024)
    record = {"stage": stage, "elapsed_seconds": time.monotonic() - started,
              "rss_peak_bytes": rss}
    if rss > PROFILE["rss_limit_bytes"] or record["elapsed_seconds"] > PROFILE["time_limit_seconds"]:
        raise RuntimeError("numeric downstream resource bound exceeded")
    return record


def validate_capture_replay(source, capture):
    """Replay source-only preservation claims before any metric is accepted."""
    canonical(capture)
    if any(capture.get(name) != source[name]
           for name in ("id", "group_id", "origin", "input", "context")):
        raise ValueError("closed capture source identity differs")
    if capture.get("parent_raw") != capture.get("child_raw"):
        raise ValueError("closed capture raw response differs")
    trace = capture.get("trace")
    if not isinstance(trace, dict) or trace.get("base") != capture.get("parent_base"):
        raise ValueError("closed capture original base pool differs")
    if capture.get("parent_sentence_scored"):
        preserved = trace.get("base", []) + trace.get("accepted_old_extras", [])
        if (preserved != capture.get("parent_pool")
                or trace.get("scoring_pool", [])[:len(preserved)] != preserved):
            raise ValueError("closed capture original full pool differs")
    if capture.get("child_sentence_scored"):
        if capture.get("child_score_inputs") != [
                row["text"] for row in trace.get("scoring_pool", [])]:
            raise ValueError("closed capture scorer inputs differ")
    elif capture.get("child_score_inputs") is not None:
        raise ValueError("closed capture has an unexpected scorer call")


def score_closed_captures(sources, targets, captures):
    from scripts.long_context_lora_fresh_metrics import paired_document_bootstrap

    if len(sources) != len(targets) or len(targets) != len(captures) or not targets:
        raise ValueError("closed downstream rows are incomplete")
    roles = {
        "fresh_natural": {"rows": 0, "before_top1": 0, "top1": 0, "before_top5": 0,
                          "top5": 0, "before_pool": 0, "pool": 0, "corrections": [],
                          "regressions": [], "top5_gains": [], "top5_losses": []},
        "challenge_preservation": {"rows": 0, "before_top1": 0, "top1": 0,
                          "before_top5": 0, "top5": 0, "before_pool": 0, "pool": 0,
                          "corrections": [], "regressions": [], "top5_gains": [],
                          "top5_losses": []},
    }
    outcomes = []
    ids = set()
    for source, target, capture in zip(sources, targets, captures):
        validate_capture_replay(source, capture)
        identity = source["id"]
        if (identity in ids or target.get("id") != identity or capture.get("id") != identity
                or target.get("group_id") != source["group_id"]
                or capture.get("group_id") != source["group_id"]):
            raise ValueError("target join identity/order differs")
        ids.add(identity)
        role = target.get("role")
        if role not in roles:
            raise ValueError("target role differs")
        expected = target.get("expected")
        if not isinstance(expected, list) or not expected or any(not isinstance(x, str) for x in expected):
            raise ValueError("target answers are malformed")
        parent, child, trace = capture["parent"], capture["child"], capture["trace"]
        before = parent["result"] in expected; after = child["result"] in expected
        before5 = any(row["text"] in expected for row in parent["candidates"])
        after5 = any(row["text"] in expected for row in child["candidates"])
        child_pool = trace["scoring_pool"] if capture["child_sentence_scored"] else child["candidates"]
        values = roles[role]; values["rows"] += 1
        for name, condition in (("before_top1", before), ("top1", after),
                ("before_top5", before5), ("top5", after5),
                ("before_pool", any(row["text"] in expected for row in capture["parent_pool"])),
                ("pool", any(row["text"] in expected for row in child_pool))):
            values[name] += int(condition)
        for condition, name in ((after and not before, "corrections"),
                (before and not after, "regressions"), (after5 and not before5, "top5_gains"),
                (before5 and not after5, "top5_losses")):
            if condition:
                values[name].append(identity)
        if role == "fresh_natural":
            outcomes.append({"id": identity, "group_id": source["group_id"],
                             "before_correct": before, "after_correct": after})
    if roles["fresh_natural"]["rows"] != FRESH_ROWS \
            or roles["challenge_preservation"]["rows"] != PRESERVATION_ROWS:
        raise ValueError("closed downstream rows are incomplete: role denominator differs")
    bootstrap = paired_document_bootstrap(outcomes)
    fresh, preservation = roles["fresh_natural"], roles["challenge_preservation"]
    invariants = not fresh["top5_losses"] and not preservation["top5_losses"]
    protected = preservation["before_top1"] == preservation["top1"] == PRESERVATION_ROWS
    passed = (bootstrap["rows"] == FRESH_ROWS and bootstrap["groups"] == FRESH_GROUPS
              and bootstrap["gain_and_interval_passed"] and invariants and protected)
    return {"roles": roles, "bootstrap": bootstrap,
            "gate": {"passed": passed, "invariants_preserved": invariants,
                     "raw_preserved": True, "original_pool_preserved": True,
                     "scorer_trace_preserved": True,
                     "preservation_all_correct": protected,
                     "downstream_evaluation_allowed": passed, "promotion_allowed": False}}


def execute_pipeline(output, entry, target_loader, model_loader, capture_one,
                     model_releaser=release_models, resource_observer=resources):
    """Execute the fixed closure/release/target-join order; dependencies aid tests."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    started = time.monotonic()
    report = {"schema": SCHEMA, "completed": False, "terminal_status": "running",
              "deployed": False, "promotion_allowed": False, "profile": PROFILE,
              "target_access": {"fresh_target_bytes_accessed": False,
                                "preservation_target_bytes_accessed": False},
              "captures_closed": False, "models_released_before_target_join": False,
              "targets_joined": False, "resource_observations": []}
    bundle = None
    try:
        sources, targets, baselines, target_hashes = target_loader(entry, report["target_access"])
        if (len(sources) != TOTAL_ROWS or len(targets) != TOTAL_ROWS
                or len(baselines) != PRESERVATION_ROWS
                or any(set(row) != {"id", "group_id", "origin", "input", "context"}
                       for row in sources)
                or any(not all(isinstance(row.get(name), str)
                               and (name == "context" or bool(row[name]))
                               for name in row) for row in sources)):
            raise ValueError("source-only downstream row accounting differs")
        expected_roles = (["fresh_natural"] * FRESH_ROWS
                          + ["challenge_preservation"] * PRESERVATION_ROWS)
        if (any(set(target) != {"id", "group_id", "role", "expected"}
                or target["id"] != source["id"]
                or target["group_id"] != source["group_id"]
                or target["role"] != role
                or not isinstance(target["expected"], list)
                or not target["expected"]
                or any(not isinstance(value, str) or not value
                       for value in target["expected"])
                for source, target, role in zip(sources, targets, expected_roles))
                or len({row["group_id"] for row in sources[:FRESH_ROWS]}) != FRESH_GROUPS
                or [row.get("id") for row in baselines]
                   != [row["id"] for row in sources[FRESH_ROWS:]]):
            raise ValueError("downstream target/baseline accounting differs")
        source_hashes = dict(entry["source_hashes"])
        merge_inventory(source_hashes, target_hashes)
        verify_hashes(source_hashes)
        report["source_hashes_before"] = dict(sorted(source_hashes.items()))
        report["historical_qwen_limitations"] = deepcopy(
            entry.get("historical_qwen_limitations"))
        report["files"] = {
            "sources.jsonl": write_jsonl(output / "sources.jsonl", sources),
            "targets.jsonl": write_jsonl(output / "targets.jsonl", targets),
            "preservation_baselines.jsonl": write_jsonl(
                output / "preservation_baselines.jsonl", baselines),
        }
        baseline_by_id = {row["id"]: row for row in baselines}
        del targets, baselines
        model_load_started = time.monotonic()
        bundle = model_loader(entry["runtime"]["accepted_weight_hashes"])
        report["model_load_seconds"] = time.monotonic() - model_load_started
        report["model_loaded"] = True
        captures = []
        report["captured_rows"] = 0
        with (output / "captures.jsonl").open("xb") as stream:
            for index, source in enumerate(sources, 1):
                capture = capture_one(bundle, source, baseline_by_id.get(source["id"]),
                                      entry["runtime"]["accepted_weight_hashes"])
                stream.write(canonical(capture) + b"\n"); stream.flush()
                captures.append(capture)
                report["captured_rows"] = index
                report["resource_observations"].append(
                    resource_observer(started, f"capture_{index}"))
        report["captures_closed"] = True
        report["files"]["captures.jsonl"] = file_digest(output / "captures.jsonl")
        model_releaser(bundle)
        report["model_loaded"] = False
        report["models_released_before_target_join"] = True
        report["terminal_resources"] = resource_observer(started, "after_model_release")
        report["resource_observations"].append(report["terminal_resources"])
        verify_hashes(source_hashes)
        reopened_targets = read_jsonl(output / "targets.jsonl")
        report["targets_joined"] = True
        metrics = score_closed_captures(sources, reopened_targets, captures)
        verify_hashes(source_hashes)
        report.update({"completed": True, "terminal_status": "success", "rows": len(sources),
            "fresh_rows": FRESH_ROWS, "fresh_groups": FRESH_GROUPS,
            "preservation_rows": PRESERVATION_ROWS, "runtime_version": RUNTIME_VERSION,
            "model_artifacts": entry["runtime"]["accepted_weight_hashes"],
            "source_hashes_after": dict(sorted(source_hashes.items())), "metrics": metrics,
            "quality_gate_passed": metrics["gate"]["passed"],
            "downstream_evaluation_allowed": metrics["gate"]["passed"],
            "promotion_allowed": False, "deployed": False,
            "latency": {"first_parent_ms": captures[0]["parent_ms"],
                "first_child_ms": captures[0]["child_ms"],
                "warm_parent_p95_ms": sorted(row["parent_ms"] for row in captures[1:])[
                    math.ceil(.95 * (len(captures) - 1)) - 1],
                "warm_child_p95_ms": sorted(row["child_ms"] for row in captures[1:])[
                    math.ceil(.95 * (len(captures) - 1)) - 1],
                "scope": "engine calls after model load; not browser or cold model loading"},
            "elapsed_seconds": time.monotonic() - started,
            "created_at": datetime.now(timezone.utc).isoformat()})
    except BaseException as error:
        if bundle is not None and report.get("model_loaded"):
            try:
                model_releaser(bundle)
                report["model_loaded"] = False
                report["models_released_on_failure"] = True
            except BaseException as release_error:
                report["models_released_on_failure"] = False
                report["release_failure_type"] = type(release_error).__name__
                report["release_failure"] = str(release_error)
        partial = output / "captures.jsonl"
        if partial.exists():
            report["partial_captures_sha256"] = file_digest(partial)
        report.update({"completed": False, "terminal_status": "failed",
                       "failure_type": type(error).__name__, "failure": str(error),
                       "elapsed_seconds": time.monotonic() - started,
                       "created_at": datetime.now(timezone.utc).isoformat()})
    (output / "report.json").write_bytes(canonical(report) + b"\n")
    return report


def run(output):
    entry = validate_static_entry()
    return execute_pipeline(output, entry, load_and_split_targets, load_models, capture_source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / ".omx/artifacts/numeric-composition-downstream-v1")
    result = run(parser.parse_args().output)
    print(json.dumps({"completed": result["completed"],
                      "quality_gate_passed": result.get("quality_gate_passed"),
                      "downstream_evaluation_allowed": result.get("downstream_evaluation_allowed"),
                      "promotion_allowed": False}, sort_keys=True))
    raise SystemExit(0 if result["completed"] else 1)
