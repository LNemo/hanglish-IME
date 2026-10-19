"""Actual-hybrid TRAIN probe for the approved numeric sentence-pool overlay.

The temporary ``beam_paths`` hook is installed only around one child
``convert`` call in this standalone script. It is never installed by runtime or
web-server code. Candidate construction is source-only; TRAIN labels are read
after the paired parent and child inference calls.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
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

import torch

import hanglish.core.neural_converter as neural_module
from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.core.phonology import romanize_word
from scripts.numeric_structure_candidates import NumericSpanGenerator
from scripts.probe_numeric_structure_candidates import numeric_assignments


OUTPUT = ROOT / ".omx/artifacts/numeric-structure-runtime-v1"
APPROVED = ROOT / ".omx/artifacts/numeric-structure-candidates-v1"
INTEGRATION = ROOT / ".omx/artifacts/acronym-suffix-integration-v1/report.json"
APPROVED_REPORT_SHA256 = "6c99bc7f07d5eaa8735fbb436771d973b4e754523b261f4a85b2b0f949123c81"
APPROVED_CASES_SHA256 = "1648e0c714ffe9e2c482eee6d5d3690b87a3ccf24d5deed1dd4c44503e9e5475"
APPROVED_SOURCE_SHA256 = "e45aeb2421ae3b93d92bc16c4bb045ac967358f2ac0d4a33b320f8d8ebe3dfbe"
NUMERIC_HELPER_SHA256 = "0bec99b1f2514f07d42604eadad05f9b61b80a274c9bcac7b40250bf6f55e4cc"
INTEGRATION_REPORT_SHA256 = "3148b4eb9a156457ea95a9b2afec1072d3db03273baacbb5159524d0ea6cd134"
DECISION_SHA256 = "e5e70c58cdb74e332ffecfe1b15a95ba3fca959813f5b8f9fca6d57c3323a13a"
RUNTIME_VERSION = "neural-1-5bcbab126cae"
PROFILE = {
    "name": "numeric-structure-actual-hybrid-train-v1",
    "rows": 512,
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 1800,
    "progress_interval": 32,
    "maximum_original_pool": 48,
    "maximum_additions": 48,
    "maximum_augmented_pool": 96,
    "sentence_score_prefix_tolerance": 1e-4,
    "assignment_policy": "first row-wide all-best numeric assignment",
}


def read_jsonl_bytes(value):
    return [json.loads(line) for line in value.decode().splitlines() if line]


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read_approved(explicit_sha):
    if explicit_sha != APPROVED_REPORT_SHA256:
        raise ValueError("explicit approved numeric report SHA differs")
    paths = {
        "report": (APPROVED / "report.json", APPROVED_REPORT_SHA256),
        "cases": (APPROVED / "cases.jsonl", APPROVED_CASES_SHA256),
        "source": (APPROVED / "source.py", APPROVED_SOURCE_SHA256),
        "decision": (APPROVED / "decision.json", DECISION_SHA256),
        "integration": (INTEGRATION, INTEGRATION_REPORT_SHA256),
    }
    raw = {}
    for name, (path, expected) in paths.items():
        value = path.read_bytes()
        if sha256_bytes(value) != expected:
            raise ValueError(f"approved artifact hash differs: {name}")
        raw[name] = value
    report = json.loads(raw["report"])
    cases = read_jsonl_bytes(raw["cases"])
    decision = json.loads(raw["decision"])
    integration = json.loads(raw["integration"])
    validate_approved(report, cases, integration, decision)
    return report, cases, integration, paths


def validate_approved(report, cases, integration, decision=None):
    profile = report.get("profile", {})
    if (report.get("scope") != "Frozen natural TRAIN first512 cached candidate-structure evaluation"
            or report.get("rows") != PROFILE["rows"]
            or report.get("cases_sha256") != APPROVED_CASES_SHA256
            or report.get("probe_source_sha256") != APPROVED_SOURCE_SHA256
            or report.get("frozen_acronym_child_coverage") != 319
            or report.get("overlay_coverage") != 339
            or report.get("baseline_prefix_preserved") is not True
            or report.get("only_selected_raw_pieces_replaced") is not True
            or report.get("labels_used_for_candidate_production") is not False
            or report.get("new_model_inference_calls") != 0
            or report.get("natural_dev_read") is not False
            or report.get("test_data_read") is not False
            or report.get("deployed") is not False
            or profile.get("primary_overlay_additions_per_row") != PROFILE["maximum_additions"]
            or profile.get("primary_overlay_total_paths") != PROFILE["maximum_augmented_pool"]
            or report.get("source_hashes", {}).get("scripts/probe_numeric_structure_candidates.py") != APPROVED_SOURCE_SHA256
            or report.get("source_hashes", {}).get("scripts/numeric_structure_candidates.py") != NUMERIC_HELPER_SHA256
            or integration.get("passed") is not True
            or integration.get("integrated_runtime_version") != RUNTIME_VERSION):
        raise ValueError("approved numeric candidate or runtime lineage is incomplete")
    if decision is not None and (
            decision.get("candidate_stage_passed") is not True
            or decision.get("report_sha256") != APPROVED_REPORT_SHA256
            or decision.get("overlay_before") != 319
            or decision.get("overlay_after") != 339
            or decision.get("development_authorized_by_this_artifact") is not False
            or decision.get("deployed") is not False):
        raise ValueError("approved numeric candidate decision is incomplete")
    if len(cases) != PROFILE["rows"] or len({row.get("id") for row in cases}) != len(cases):
        raise ValueError("approved numeric TRAIN cases are incomplete or duplicated")
    for row in cases:
        if (not isinstance(row.get("input"), str) or not isinstance(row.get("context"), str)
                or not isinstance(row.get("baseline_visible_candidates"), list)
                or not isinstance(row.get("overlay_paths"), list)
                or row["overlay_paths"][:len(row.get("baseline_paths", []))] != row.get("baseline_paths")
                or len(row.get("overlay_additions", [])) > PROFILE["maximum_additions"]):
            raise ValueError("approved numeric TRAIN case is malformed")


def serialize_pieces(pieces):
    return [[source, offset, [
        {"text": option.text, "score": option.score, "kind": option.kind}
        for option in options
    ]] for source, offset, options in pieces]


def stable_union(learned_paths, static_paths):
    unique = {}
    for path in (*learned_paths, *static_paths):
        unique.setdefault(path[0], path)
    return list(unique.values())


def validate_complete_segments(pieces, path):
    text, score, segments = path
    if not math.isfinite(score) or not segments:
        raise ValueError("numeric augmented path shape differs")
    source_text = "".join(source for source, _, _ in pieces)
    cursor = 0
    for source, offset, _ in pieces:
        if offset != cursor:
            raise ValueError("runtime pieces do not cover contiguous input")
        cursor += len(source)
    cursor = 0
    for segment in segments:
        if (segment["input_start"] != cursor or segment["input_end"] <= cursor
                or segment["source"] != source_text[
                    segment["input_start"]:segment["input_end"]]):
            raise ValueError("numeric augmented segments do not preserve source offsets")
        if segment["kind"] == "raw" and segment["text"] != segment["source"]:
            raise ValueError("numeric augmented path changed a raw literal")
        if (segment["kind"] == "numeric"
                and romanize_word(segment["text"]) != segment["source"]):
            raise ValueError("numeric augmented segment is not forward valid")
        cursor = segment["input_end"]
    if cursor != len(source_text) or text != "".join(
            segment["text"] for segment in segments):
        raise ValueError("numeric augmented path output differs from its segments")


def augment_path(path, pieces, assignment):
    text, score, segments = path
    if len(segments) != len(pieces):
        raise ValueError("runtime beam path no longer has one segment per piece")
    replacements = {item["piece_index"]: item for item in assignment["replacements"]
                    if item["converted"]}
    output_segments = []
    for index, segment in enumerate(segments):
        replacement = replacements.get(index)
        if replacement is None:
            output_segments.append(dict(segment))
            continue
        if (segment["source"] != replacement["source"]
                or segment["text"] != replacement["source"]
                or segment["kind"] != "raw"):
            raise ValueError("numeric supplement selected a changed or non-raw path segment")
        output_segments.extend(dict(item) for item in replacement["segments"])
    result = ("".join(item["text"] for item in output_segments), score, output_segments)
    validate_complete_segments(pieces, result)
    return result


def build_extras(learned_pieces, static_pieces, original_pool, generator):
    assignments, parser_timings, _ = numeric_assignments(
        {"learned_pieces": serialize_pieces(learned_pieces),
         "static_pieces": serialize_pieces(static_pieces)}, generator
    )
    if not assignments:
        return [], None, parser_timings
    assignment = assignments[0]
    if assignment.get("conversion_count") != len(assignment.get("replacements", ())):
        raise ValueError("primary numeric assignment did not convert every selected piece")
    seen = {path[0] for path in original_pool}
    extras = []
    for path in original_pool:
        candidate = augment_path(path, learned_pieces, assignment)
        if candidate[0] in seen:
            continue
        seen.add(candidate[0])
        extras.append(candidate)
        if len(extras) == PROFILE["maximum_additions"]:
            break
    return extras, assignment, parser_timings


@contextmanager
def numeric_beam_overlay(converter, generator):
    """Temporarily append scoreable numeric paths on the static beam call."""
    if neural_module.beam_paths is not ORIGINAL_BEAM_PATHS:
        raise RuntimeError("beam_paths changed before numeric adapter installation")
    original = ORIGINAL_BEAM_PATHS
    state = {"calls": 0, "applied": False, "reason": "not_reached",
             "original_pool": None, "augmented_pool": None, "extras": [],
             "assignment": None, "parser_timings_ms": []}
    learned_pieces = learned_paths = None

    def hooked(pieces):
        nonlocal learned_pieces, learned_paths
        state["calls"] += 1
        if state["calls"] > 2:
            raise RuntimeError("numeric beam adapter observed more than two beam calls")
        paths = original(pieces)
        if state["calls"] == 1:
            learned_pieces, learned_paths = pieces, paths
            return paths
        static_pieces, static_paths = pieces, paths
        original_pool = stable_union(learned_paths, static_paths)
        if len(original_pool) > PROFILE["maximum_original_pool"]:
            raise ValueError("runtime original sentence pool exceeds its bound")
        state["original_pool"] = original_pool
        if not all(converter.sentence_ranker.can_score(path[0]) for path in original_pool):
            state["reason"] = "original_pool_unscoreable"
            return paths
        extras, assignment, timings = build_extras(
            learned_pieces, static_pieces, original_pool, generator
        )
        state["assignment"], state["parser_timings_ms"] = assignment, timings
        if not extras:
            state["reason"] = "no_source_derived_extras"
            return paths
        if not all(converter.sentence_ranker.can_score(path[0]) for path in extras):
            state["reason"] = "numeric_extra_unscoreable"
            return paths
        state["applied"], state["reason"], state["extras"] = True, "applied", extras
        state["augmented_pool"] = original_pool + extras
        if len(state["augmented_pool"]) > PROFILE["maximum_augmented_pool"]:
            raise ValueError("runtime augmented sentence pool exceeds its bound")
        return paths + extras

    neural_module.beam_paths = hooked
    try:
        yield state
    finally:
        neural_module.beam_paths = original


class SentencePoolRecorder:
    def __init__(self, delegate):
        self.delegate = delegate
        self.candidates = None
        self.scores = None

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def reset(self):
        self.candidates = self.scores = None

    def score_pairs(self, contexts, candidates):
        self.candidates = list(candidates)
        self.scores = list(self.delegate.score_pairs(contexts, candidates))
        return self.scores


def validate_candidate_objects(response, source):
    candidates = response.get("candidates")
    if (not isinstance(candidates, list) or not 1 <= len(candidates) <= 5
            or response.get("result") != candidates[0].get("text")
            or response.get("version") != RUNTIME_VERSION):
        raise ValueError("actual runtime candidate response is malformed")
    for candidate in candidates:
        score = candidate.get("score")
        segments = candidate.get("segments")
        if (not isinstance(candidate.get("text"), str) or not candidate["text"]
                or (score is not None and (type(score) not in (int, float) or not math.isfinite(score)))
                or not isinstance(segments, list) or not segments):
            raise ValueError("actual runtime candidate object is malformed")
        cursor = 0
        for segment in segments:
            if (segment.get("input_start") != cursor
                    or type(segment.get("input_end")) is not int
                    or segment["input_end"] <= cursor
                    or source[cursor:segment["input_end"]] != segment.get("source")):
                raise ValueError("actual runtime candidate segments differ from source")
            cursor = segment["input_end"]
        if cursor != len(source) or candidate["text"] != "".join(
                segment.get("text", "") for segment in segments):
            raise ValueError("actual runtime candidate segment output differs")


def evaluate_source(converter, recorder, generator, source):
    """Run both source policies before the caller accesses TRAIN labels."""
    recorder.reset()
    parent = converter.convert(source["input"], source["context"])
    validate_candidate_objects(parent, source["input"])
    parent_pool, parent_scores = recorder.candidates, recorder.scores
    parent_texts = [candidate["text"] for candidate in parent["candidates"]]
    if (parent_texts != source["baseline_visible_candidates"]
            or (parent_pool is not None) != source["sentence_eligible"]
            or (parent_pool is not None and parent_pool != source["baseline_paths"])):
        raise ValueError("fresh production parent differs from frozen acronym TRAIN child")
    recorder.reset()
    with numeric_beam_overlay(converter, generator) as adapter:
        child = converter.convert(source["input"], source["context"])
    validate_candidate_objects(child, source["input"])
    child_pool, child_scores = recorder.candidates, recorder.scores
    if neural_module.beam_paths is not ORIGINAL_BEAM_PATHS:
        raise RuntimeError("numeric adapter did not restore beam_paths")
    return parent, parent_pool, parent_scores, child, child_pool, child_scores, adapter


ORIGINAL_BEAM_PATHS = neural_module.beam_paths


def metrics(rows):
    result = {"rows": len(rows)}
    for side in ("parent", "child"):
        result[f"{side}_top1"] = sum(row[f"{side}_top1"] for row in rows)
        result[f"{side}_top5"] = sum(row[f"{side}_top5"] for row in rows)
        result[f"{side}_pool_covered"] = sum(row[f"{side}_pool_covered"] is True for row in rows)
        result[f"{side}_pool_or_fallback_top5_covered"] = sum(
            row[f"{side}_top5"] if row[f"{side}_pool_covered"] is None
            else row[f"{side}_pool_covered"] for row in rows
        )
    result["corrections"] = [row["id"] for row in rows if row["child_top1"] and not row["parent_top1"]]
    result["regressions"] = [row["id"] for row in rows if row["parent_top1"] and not row["child_top1"]]
    result["top5_regressions"] = [row["id"] for row in rows if row["parent_top5"] and not row["child_top5"]]
    result["pool_regressions"] = [row["id"] for row in rows
                                  if row["parent_pool_covered"] is True and row["child_pool_covered"] is not True]
    return result


def pool_prefix_proof(parent_pool, parent_scores, child_pool, child_scores,
                      tolerance=PROFILE["sentence_score_prefix_tolerance"]):
    if parent_pool is None:
        return child_pool is None and parent_scores is None and child_scores is None
    if (parent_scores is None or child_pool is None or child_scores is None
            or len(parent_scores) != len(parent_pool)
            or len(child_scores) != len(child_pool)
            or child_pool[:len(parent_pool)] != parent_pool
            or len(parent_scores) > len(child_scores)):
        return False
    return all(abs(left - right) <= tolerance
               for left, right in zip(parent_scores, child_scores[:len(parent_scores)]))


def runtime_passed(summary):
    return bool(
        summary.get("rows") == PROFILE["rows"]
        and summary.get("parent_pool_or_fallback_top5_covered") == 319
        and summary.get("child_pool_or_fallback_top5_covered") == 339
        and summary.get("child_top1", -1) >= summary.get("parent_top1", 0)
        and summary.get("child_top5", -1) >= summary.get("parent_top5", 0)
        and summary.get("regressions") == []
        and summary.get("top5_regressions") == []
        and summary.get("pool_regressions") == []
    )


def current_hashes(integration):
    expected_sources = integration["integrated_runtime_source_hashes"]
    result = {path: file_sha256(ROOT / path) for path in expected_sources}
    for relative in (
        "scripts/probe_numeric_structure_runtime.py",
        "scripts/numeric_structure_candidates.py",
        "scripts/probe_numeric_structure_candidates.py",
        ".omx/artifacts/numeric-structure-candidates-v1/report.json",
        ".omx/artifacts/numeric-structure-candidates-v1/cases.jsonl",
        ".omx/artifacts/numeric-structure-candidates-v1/source.py",
        ".omx/artifacts/numeric-structure-candidates-v1/decision.json",
        ".omx/artifacts/acronym-suffix-integration-v1/report.json",
    ):
        result[relative] = file_sha256(ROOT / relative)
    for relative in integration["accepted_weight_hashes"]:
        result[relative] = file_sha256(ROOT / relative)
    return dict(sorted(result.items()))


def validate_current_hashes(integration, observed):
    for relative, expected in integration["integrated_runtime_source_hashes"].items():
        if observed.get(relative) != expected:
            raise ValueError(f"integrated runtime source changed: {relative}")
    for relative, expected in integration["accepted_weight_hashes"].items():
        if observed.get(relative) != expected:
            raise ValueError(f"accepted model artifact changed: {relative}")
    if (observed.get("scripts/numeric_structure_candidates.py") != NUMERIC_HELPER_SHA256
            or observed.get("scripts/probe_numeric_structure_candidates.py") != APPROVED_SOURCE_SHA256):
        raise ValueError("approved numeric source implementation changed")


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def require_resources(started):
    if (rss_bytes() > PROFILE["rss_limit_bytes"]
            or time.monotonic() - started > PROFILE["time_limit_seconds"]):
        raise RuntimeError("numeric structure runtime TRAIN probe resource bound exceeded")


def run(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    report, sources, integration, approved_paths = read_approved(args.candidate_probe_sha256)
    probe_source = Path(__file__).read_bytes()
    probe_source_sha256 = sha256_bytes(probe_source)
    before = current_hashes(integration)
    validate_current_hashes(integration, before)
    torch.set_num_threads(PROFILE["threads"])
    os.environ["OMP_NUM_THREADS"] = str(PROFILE["threads"])
    converter = NeuralConverter(ranker_kind="hybrid")
    converter.load_models()
    if converter.version != RUNTIME_VERSION:
        raise ValueError("actual hybrid runtime differs from accepted integration")
    parent_word_options = converter.word_options
    recorder = SentencePoolRecorder(converter.sentence_ranker)
    converter.sentence_ranker = recorder
    generator = NumericSpanGenerator(converter.numeric_units)
    inferred = []
    for index, source in enumerate(sources):
        values = evaluate_source(converter, recorder, generator, source)
        inferred.append((source, values))
        require_resources(started)
        if (index + 1) % PROFILE["progress_interval"] == 0:
            print(json.dumps({"event": "numeric_runtime_probe", "rows": index + 1,
                              "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)

    # Only now align the already-produced source candidates to TRAIN labels.
    rows = []
    for source, values in inferred:
        parent, parent_pool, parent_scores, child, child_pool, child_scores, adapter = values
        expected = source.get("expected")
        if not isinstance(expected, list) or not expected:
            raise ValueError("approved TRAIN labels are malformed")
        parent_texts = [item["text"] for item in parent["candidates"]]
        child_texts = [item["text"] for item in child["candidates"]]
        if parent_texts != source["baseline_visible_candidates"]:
            raise ValueError("fresh production parent differs from frozen acronym TRAIN child")
        if ((parent_pool is None) != (parent_scores is None)
                or (child_pool is None) != (child_scores is None)
                or any(not math.isfinite(score) for scores in (parent_scores, child_scores)
                       if scores is not None for score in scores)
                or converter.word_options is not parent_word_options):
            raise ValueError("paired actual runtime output or word options differ")
        prefix_proven = pool_prefix_proof(
            parent_pool, parent_scores, child_pool, child_scores
        )
        if not prefix_proven:
            raise ValueError("numeric overlay changed baseline pool order or sentence scores")
        if adapter["applied"]:
            if (parent_pool != [path[0] for path in adapter["original_pool"]]
                    or child_pool != [path[0] for path in adapter["augmented_pool"]]
                    or child_pool != source["overlay_paths"]):
                raise ValueError("actual numeric overlay differs from approved source-only pool")
        elif child_pool != parent_pool:
            raise ValueError("non-applied numeric adapter changed the sentence pool")
        rows.append({
            "id": source["id"], "input": source["input"], "context": source["context"],
            "expected": expected, "parent_candidates": parent["candidates"],
            "child_candidates": child["candidates"], "parent_sentence_pool": parent_pool,
            "child_sentence_pool": child_pool, "parent_sentence_scores": parent_scores,
            "child_sentence_scores": child_scores, "sentence_ranking_applied": child["sentence_ranking_applied"],
            "baseline_pool_and_score_prefix_preserved": prefix_proven,
            "parent_flags": {key: parent[key] for key in (
                "version", "ranker", "context_used", "ranking_applied",
                "sentence_ranking_applied", "warnings")},
            "child_flags": {key: child[key] for key in (
                "version", "ranker", "context_used", "ranking_applied",
                "sentence_ranking_applied", "warnings")},
            "adapter": adapter, "parent_top1": parent_texts[0] in expected,
            "child_top1": child_texts[0] in expected,
            "parent_top5": any(text in expected for text in parent_texts),
            "child_top5": any(text in expected for text in child_texts),
            "parent_pool_covered": None if parent_pool is None else any(text in expected for text in parent_pool),
            "child_pool_covered": None if child_pool is None else any(text in expected for text in child_pool),
            "parent_engine_ms": parent["engine_ms"], "child_engine_ms": child["engine_ms"],
        })

    args.output.mkdir(parents=True)
    cases_path = args.output / "cases.jsonl"
    with cases_path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    after = current_hashes(integration)
    validate_current_hashes(integration, after)
    for _, (path, expected) in approved_paths.items():
        if file_sha256(path) != expected:
            raise ValueError("approved artifact changed during actual inference")
    if before != after or converter.word_options is not parent_word_options or neural_module.beam_paths is not ORIGINAL_BEAM_PATHS:
        raise ValueError("runtime source, weights, word options, or beam hook changed")
    require_resources(started)
    summary = metrics(rows)
    if (summary["parent_pool_or_fallback_top5_covered"] != 319
            or summary["child_pool_or_fallback_top5_covered"] != 339):
        raise ValueError("actual runtime candidate coverage differs from approved source-only result")
    passed = runtime_passed(summary)
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed, "eligible_for_development": passed,
        "scope": "fresh actual hybrid natural TRAIN first512 numeric overlay probe",
        "profile": PROFILE, "runtime_version": RUNTIME_VERSION,
        "candidate_probe_sha256": args.candidate_probe_sha256,
        "candidate_probe_cases_sha256": APPROVED_CASES_SHA256,
        "probe_source_sha256": probe_source_sha256,
        "source_hashes": before, "source_hashes_after": after,
        "metrics": summary,
        "adapter": {
            "applied_rows": sum(row["adapter"]["applied"] for row in rows),
            "maximum_calls": max(row["adapter"]["calls"] for row in rows),
            "maximum_additions": max((len(row["adapter"]["extras"]) for row in rows), default=0),
            "baseline_prefix_and_scores_preserved": all(
                row["baseline_pool_and_score_prefix_preserved"] for row in rows
            ),
            "only_first_all_converted_assignment": True,
            "word_options_identity_preserved": True,
            "hook_restored_after_every_child": True,
        },
        "cases_sha256": file_sha256(cases_path),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
        "labels_used_for_candidate_production": False,
        "development_data_read": False, "test_data_read": False,
        "new_weights_created": False, "deployed": False,
    }
    (args.output / "report.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    source_path = args.output / "source.py"
    source_path.write_bytes(probe_source)
    if file_sha256(Path(__file__)) != probe_source_sha256 or file_sha256(source_path) != probe_source_sha256:
        raise ValueError("runtime probe source or snapshot changed")
    require_resources(started)
    print(json.dumps({"metrics": summary, "adapter": output["adapter"],
                      "elapsed_seconds": output["elapsed_seconds"],
                      "peak_rss_bytes": output["peak_rss_bytes"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-probe-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
