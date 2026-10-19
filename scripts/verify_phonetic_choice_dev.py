#!/usr/bin/env python3
"""Pure independent verifier for the closed phonetic-choice DEV run.

This verifier never imports Torch, the evaluator, the converter, or the feature
projector.  It authenticates recorded source-only evidence, independently
replays the four runtime features and public-candidate policy, and opens DEV
labels only after the capture closure and terminal run artifacts are verified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".omx/artifacts/phonetic-choice-runtime-dev-v1"
FROZEN_UNION = ROOT / ".omx/artifacts/phonetic-word-union-dev-v1"
CALIBRATION = ROOT / ".omx/artifacts/phonetic-choice-calibration-training-v1"
DEV = ROOT / "data/challenge/sentence-v3/dev.jsonl"
PRODUCER = ROOT / "scripts/evaluate_phonetic_choice_dev.py"
ADAPTER = ROOT / "hanglish/core/phonetic_choice_converter.py"
PRODUCER_SHA256 = "cba2df7578603597a4b1020e80360f1185746d9d074801772824a5e58c37c52c"
ADAPTER_SHA256 = "c6f6d355e387bcecdf9c77c53a524e23f26bd8866bc6d550cafc29b96e8da087"
RUN_SCHEMA = "phonetic-choice-runtime-dev-v1"
SCHEMA = "phonetic-choice-runtime-dev-independent-verification-v1"
ROWS = 316
LIMITS = {"whole_seconds": 1800.0, "update_seconds": 60.0,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
FROZEN_HASHES = {
    FROZEN_UNION / "report.json": "a87575bc2080fbe649029e59ce55c38131096a806b0b99203d40b0bfceeec368",
    FROZEN_UNION / "capture.jsonl": "e8d45820b7edb691bf18f011f63fe0b4a0577a387c4e59d0fdf584e36bcb6131",
    FROZEN_UNION / "independent-verification.json": "82e99e5799ca005d17442b9cb90e62c8e98e846d369b27f648fd0184fae6df94",
    CALIBRATION / "weights.json": "930e3ef942061e6d459d7527c0e073fc6ad5626d59f5170e9b11df8499b39f4b",
    CALIBRATION / "report.json": "b692262a2fcd8c525d543ca58aa86218948a5a04f5ab60b2afb4ebcb0c7143a6",
    CALIBRATION / "independent-verification.json": "7181d0ca4e783c0cf40e10ed28f26d4e2401e49829f5aade13f2123d5da06a83",
}
FEATURE_NAMES = (
    "length_weighted_mean_jamo_eos_log_probability",
    "supported_source_fraction",
    "unsupported_conversion_source_fraction",
    "forward_mismatch_supported_source_fraction",
)
LOWER_ASCII = re.compile(r"[a-z]+\Z")
HANGUL = re.compile(r"[가-힣]+\Z")
ASCII_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
LITERAL_HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]\Z")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    need(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSONL at {path}:{number}") from error
            need(isinstance(value, dict), f"non-object JSONL row at {path}:{number}")
            yield value


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def path_key(path):
    return str(Path(path).resolve())


def pair_id(source, target):
    payload = json.dumps(["phonetic-likelihood-fit-v1", source, target],
                         ensure_ascii=False, separators=(",", ":")).encode()
    return "phonetic-pair-" + hashlib.sha256(payload).hexdigest()


def expected_status(source, target):
    if not source or not target:
        return "empty"
    if source == target:
        return "copy"
    if LOWER_ASCII.fullmatch(source) is None:
        return "unsupported_ascii"
    if HANGUL.fullmatch(target) is None:
        return "unsupported_shape"
    if len(source) > 48 or len(target) > 24:
        return "length"
    return "supported"


def romanize_word(text):
    # Import only the deterministic pure romanizer.  This module has no model
    # or Torch dependency and is hash-bound by the run pins.
    from hanglish.core.phonology import romanize_word as implementation
    return implementation(text)


def verify_candidate(source, candidate):
    need(isinstance(candidate, dict) and isinstance(candidate.get("text"), str)
         and (finite(candidate.get("score")) or candidate.get("score") is None),
         "candidate schema differs")
    cursor, pieces = 0, []
    segments = candidate.get("segments")
    need(isinstance(segments, list) and segments, "candidate segments missing")
    for segment in segments:
        end = segment.get("input_end")
        need(segment.get("input_start") == cursor and type(end) is int
             and cursor < end <= len(source) and segment.get("source") == source[cursor:end]
             and isinstance(segment.get("text"), str), "candidate source coverage differs")
        cursor = end; pieces.append(segment["text"])
    need(cursor == len(source) and "".join(pieces) == candidate["text"],
         "candidate target reconstruction differs")
    if candidate.get("score") is None:
        need(candidate == {"text": source, "score": None, "segments": [{
            "input_start": 0, "input_end": len(source), "source": source,
            "text": source, "kind": "raw"}]}, "canonical raw candidate differs")


def render_public(source, paths):
    need(paths, "cannot render empty path pool")
    candidates = [{"text": text, "score": round(score, 4), "segments": segments}
                  for text, score, segments in paths[:5]]
    if not any(row["text"] == source for row in candidates):
        candidates = candidates[:4] + [{"text": source, "score": None, "segments": [{
            "input_start": 0, "input_end": len(source), "source": source,
            "text": source, "kind": "raw"}]}]
    return candidates


def verify_choice_row(source, base_path, row, scores):
    text, base_score, segments = base_path
    need(isinstance(row, dict) and row.get("text") == text
         and row.get("base_score") == base_score and row.get("segments") == segments,
         "choice row/base path identity differs")
    edges = row.get("edges")
    need(isinstance(edges, list) and edges, "choice edges missing")
    cursor, pieces = 0, []
    features = [0.0] * 4
    for edge, segment in zip(edges, segments):
        need(isinstance(edge, dict) and edge.get("input_start") == cursor
             and edge.get("input_start") == segment.get("input_start")
             and edge.get("input_end") == segment.get("input_end")
             and edge.get("source") == segment.get("source")
             and edge.get("target") == segment.get("text"), "choice edge/segment differs")
        end = edge["input_end"]; status = expected_status(edge["source"], edge["target"])
        need(edge.get("status") == status and 0 <= cursor < end <= len(source),
             "choice edge status or offset differs")
        weight = (end - cursor) / len(source)
        if status == "supported":
            identifier = pair_id(edge["source"], edge["target"])
            need(edge.get("pair_id") == identifier
                 and edge.get("romanize_match") is (romanize_word(edge["target"]) == edge["source"]),
                 "supported edge identity differs")
            score = scores.get(identifier)
            need(isinstance(score, dict) and score.get("source") == edge["source"]
                 and score.get("target") == edge["target"], "supported score join differs")
            features[0] += weight * score["mean_log_probability"]
            features[1] += weight
            if edge["romanize_match"] is False:
                features[3] += weight
        elif status != "copy":
            features[2] += weight
        cursor = end; pieces.append(edge["target"])
    need(len(edges) == len(segments) and cursor == len(source) and "".join(pieces) == text,
         "choice path coverage differs")
    need(row.get("features") == features, "independently recomputed feature vector differs")
    return features


def verify_score_records(records):
    need(isinstance(records, list), "score records missing")
    result = {}
    for record in records:
        need(isinstance(record, dict) and set(record) == {
            "pair_id", "source", "target", "log_probability", "token_count",
            "mean_log_probability"}, "score record schema differs")
        identifier = pair_id(record["source"], record["target"])
        total, mean, tokens = (record["log_probability"],
                               record["mean_log_probability"], record["token_count"])
        need(record["pair_id"] == identifier and identifier not in result
             and expected_status(record["source"], record["target"]) == "supported"
             and finite(total) and total <= 0 and finite(mean) and mean <= 0
             and type(tokens) is int and tokens == 3 * len(record["target"]) + 1
             and mean == total / tokens, "score record value differs")
        result[identifier] = record
    need(list(result) == sorted(result), "score records are not pair-ID ordered")
    return result


def verify_response(source, frozen, response, base_paths, weights):
    need(response.get("input") == source and response.get("context") == ""
         and response.get("external_context_used") is False, "source-only response differs")
    candidates = response.get("candidates")
    need(isinstance(candidates, list) and 1 <= len(candidates) <= 5
         and response.get("result") == candidates[0].get("text"), "public response differs")
    for candidate in candidates:
        verify_candidate(source, candidate)
    need(any(row["text"] == source for row in candidates), "public response omitted source")
    if source not in [row["text"] for row in candidates[:4]]:
        need(candidates[-1]["text"] == source, "raw source slot differs")
    before = frozen["proposed"]
    for name in ("ctc_word", "ctc_union"):
        left, right = dict(before[name]), dict(response[name])
        left.pop("generation_ms", None); right.pop("generation_ms", None)
        need(left == right, f"{name} source generation changed")
    old_trace, trace = before["phonetic_word_union"], response["phonetic_word_union"]
    for name, value in old_trace.items():
        if name not in {"generation_ms", "public", "selected_extras", "missing_extras"}:
            need(trace.get(name) == value, f"union trace changed: {name}")
    choice = response.get("phonetic_choice")
    need(isinstance(choice, dict) and choice.get("experimental") is True
         and choice.get("effective_weights") == list(weights), "choice identity differs")
    if response.get("sentence_ranking_applied") is not True:
        need(base_paths == [] and choice.get("rows") == []
             and candidates == before["candidates"], "unranked calibration differs")
        return {"paths": 0, "pairs": 0, "supported_edges": 0}
    need(isinstance(base_paths, list) and base_paths
         and len({path[0] for path in base_paths}) == len(base_paths)
         and base_paths == sorted(base_paths, key=lambda path: (-path[1], path[0]))
         and [path[0] for path in base_paths] == old_trace["union"], "base pool differs")
    need(render_public(source, base_paths) == before["candidates"], "zero public parity differs")
    scores = verify_score_records(choice.get("score_records"))
    rows = choice.get("rows")
    need(isinstance(rows, list) and len(rows) == len(base_paths), "choice row denominator differs")
    calibrated, used = [], set()
    status_counts = Counter()
    for path, row in zip(base_paths, rows):
        features = verify_choice_row(source, path, row, scores)
        for edge in row["edges"]:
            status_counts[edge["status"]] += 1
            if edge["status"] == "supported":
                used.add(edge["pair_id"])
        residual = sum(weight * feature for weight, feature in zip(weights, features))
        adjusted = float(path[1]) + residual
        need(row.get("residual") == residual and row.get("adjusted_score") == adjusted,
             "residual or adjusted score differs")
        calibrated.append((path[0], adjusted, path[2]))
    need(used == set(scores) and choice.get("supported_pair_count") == len(scores)
         and choice.get("edge_status_counts") == dict(sorted(status_counts.items()))
         and choice.get("zero_weight_parity") is True,
         "choice score/edge census or zero shadow differs")
    calibrated.sort(key=lambda path: (-path[1], path[0]))
    need(render_public(source, calibrated) == candidates
         and choice.get("public") == [row["text"] for row in candidates],
         "calibrated public assembly differs")
    return {"paths": len(base_paths), "pairs": len(scores),
            "supported_edges": status_counts["supported"]}


def probe_groups(requests):
    ordered = sorted(requests, key=lambda row: (len(row[1]), row[0]))
    small, extra = divmod(len(ordered), 16); groups = []; offset = 0
    for index in range(16):
        count = small + int(index < extra); group = ordered[offset:offset + count]
        groups.append({"ids": [row[0] for row in group], "probe_id": group[-1][0],
                       "count": count, "maximum_source_length": len(group[-1][1])})
        offset += count
    return groups


def literal_signature(text):
    return "".join(char for char in text if not ("a" <= char <= "z")
                   and LITERAL_HANGUL.fullmatch(char) is None)


def branch_metrics(rows, labels, branch):
    totals, categories, outcomes = Counter(), defaultdict(Counter), []
    for row, label in zip(rows, labels):
        texts = [candidate["text"] for candidate in row[branch]["candidates"][:5]]
        expected = set(label["expected"]); top1 = bool(texts) and texts[0] in expected
        top5 = bool(expected & set(texts)); counts = {
            "rows": 1, "top1": int(top1), "top5": int(top5), "errors": 0}
        totals.update(counts); categories[label.get("category", "uncategorized")].update(counts)
        outcomes.append((top1, top5))
    return {"counts": dict(totals), "categories": {k: dict(v) for k, v in categories.items()},
            "passed": totals["top1"] == len(rows)}, outcomes


def aggregate(rows, labels, baseline):
    parent, before = branch_metrics(rows, labels, baseline)
    proposed, after = branch_metrics(rows, labels, "proposed")
    changes, stages, english = Counter(), Counter(), Counter()
    for row, label, old, new in zip(rows, labels, before, after):
        need(row["id"] == label.get("id") and row["request"] == {
            "input": label.get("input"), "context": ""}
             and label.get("context", "") == "" and label.get("split") == "dev",
             "late label identity differs")
        changes["corrections"] += not old[0] and new[0]
        changes["regressions"] += old[0] and not new[0]
        changes["top5_losses"] += old[1] and not new[1]
        gold = set(label["expected"]); trace = row["proposed"]["phonetic_word_union"]
        pool = set(row["parent_pool"]); stage = {
            "parent_pool": pool,
            "with_generated": pool | {item["text"] for item in trace["generated_extras"]},
            "scoreable_union": set(trace["union"]),
            "public_top5": set(trace["public"][:5]),
            "selected": {row["proposed"]["result"]},
        }
        if row["proposed"].get("sentence_ranking_applied"):
            stages["ranked_rows"] += 1
            for name, values in stage.items(): stages[name + "_correct"] += bool(gold & values)
            stages["marginal_generated_gold_rows"] += not bool(gold & pool) and bool(gold & stage["with_generated"])
            stages["generated_to_scoreable_gold_losses"] += bool(gold & stage["with_generated"]) and not bool(gold & stage["scoreable_union"])
            stages["union_to_public_gold_losses"] += bool(gold & stage["scoreable_union"]) and not bool(gold & stage["public_top5"])
        stages["rows"] += 1; stages["generated_extras"] += len(trace["generated_extras"])
        stages["scoreable_extras"] += len(trace["scoreable_extras"])
        stages["generation_budget_exhausted_rows"] += trace["generation_budget_exhausted"]
        stages["skipped_known_english_spans"] += len(trace.get("skipped_known_english", []))
        source_literal = literal_signature(row["request"]["input"])
        old_result, result = row[baseline]["result"], row["proposed"]["result"]
        english["literal_signature_regressions"] += (
            literal_signature(old_result) == source_literal and literal_signature(result) != source_literal)
        signatures = {tuple(ASCII_WORD.findall(text)) for text in gold}
        if any(signatures):
            english["rows_with_reference_ascii_words"] += 1
            old_match = tuple(ASCII_WORD.findall(old_result)) in signatures
            new_match = tuple(ASCII_WORD.findall(result)) in signatures
            english["parent_exact_signature"] += old_match
            english["proposed_exact_signature"] += new_match
            english["signature_regressions"] += old_match and not new_match
    improvement = bool(changes["corrections"] and not changes["regressions"] and not changes["top5_losses"])
    quality = bool(improvement and not english["signature_regressions"]
                   and not english["literal_signature_regressions"])
    return {"parent": parent, "proposed": proposed,
            "corrections": changes["corrections"], "regressions": changes["regressions"],
            "top5_losses": changes["top5_losses"], "improvement_gate": improvement,
            "quality_gate": quality, "stage_counts": dict(stages),
            "english_ascii_signature": dict(english),
            "english_scope": "Exact case-sensitive ASCII word sequence on references containing ASCII words; not all English-copy behavior."}


def latency(rows, branch, field="engine_ms"):
    values = sorted(row[branch][field] for row in rows)
    need(values and all(finite(value) and value >= 0 for value in values), "latency differs")
    return {name: values[max(0, math.ceil(q * len(values)) - 1)]
            for name, q in (("p50_ms", .5), ("p95_ms", .95), ("max_ms", 1.0))}


def verify_resources(rows, operations):
    need(rows, "resource trace empty")
    current = None; starts = []; ends = []; after_report = after_completion = None
    baseline = None; previous = -1.0
    for row in rows:
        need(isinstance(row, dict) and finite(row.get("elapsed_s"))
             and previous <= row["elapsed_s"] <= LIMITS["whole_seconds"]
             and type(row.get("rss_bytes")) is int and 0 <= row["rss_bytes"] <= LIMITS["rss_bytes"]
             and type(row.get("swap_used_bytes")) is int and row["swap_used_bytes"] >= 0
             and type(row.get("swap_growth_bytes")) is int
             and row["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"], "resource row differs")
        datetime.fromisoformat(row["at"]); previous = row["elapsed_s"]
        observed_baseline = row["swap_used_bytes"] - row["swap_growth_bytes"]
        need(observed_baseline >= 0, "swap baseline invalid")
        baseline = observed_baseline if baseline is None else baseline
        need(observed_baseline == baseline, "swap baseline changed")
        stage = row["stage"]
        if stage == "operation_start":
            need(current is None and isinstance(row["operation"], str), "operation start differs")
            current = row["operation"]; starts.append(current)
        elif stage == "operation_end":
            need(current == row["operation"] and finite(row["operation_elapsed_s"])
                 and 0 <= row["operation_elapsed_s"] <= LIMITS["update_seconds"], "operation end differs")
            ends.append(current); current = None
        elif stage == "watchdog":
            need(row["operation"] in (None, current), "watchdog attribution differs")
        elif stage == "after_report":
            need(after_report is None and current is None and starts == operations[:-1], "after-report resource differs")
            after_report = row
        elif stage == "after_completion":
            need(after_report is not None and after_completion is None and current is None
                 and starts == operations, "after-completion resource differs")
            after_completion = row
        else:
            raise ValueError("unexpected resource stage: " + str(stage))
    need(starts == operations == ends and after_report and after_completion,
         "resource operation sequence differs")
    return {"rows": len(rows), "operations": len(starts), "final": after_report,
            "after_completion": after_completion, "baseline_swap_bytes": baseline,
            "peak_rss_bytes": max(row["rss_bytes"] for row in rows),
            "peak_swap_growth_bytes": max(row["swap_growth_bytes"] for row in rows)}


def verify(directory):
    directory = Path(directory); proof_path = directory / "independent-verification.json"
    need(directory.is_dir() and not proof_path.exists(), "run/proof state invalid")
    for name in ("failure.json", "failure-report.json", "resource-watchdog-failure.json"):
        need(not (directory / name).exists(), "failure artifact exists: " + name)
    required = ("report.json", "completion.json", "capture.jsonl", "capture.complete.json",
                "probe.jsonl", "probe-schedule.json", "qualification.json", "pins.json",
                "resources.jsonl")
    need(all((directory / name).is_file() for name in required), "terminal run files missing")
    need(sha(PRODUCER) == PRODUCER_SHA256 and sha(ADAPTER) == ADAPTER_SHA256,
         "producer or adapter changed")
    need(all(path.is_file() and sha(path) == digest for path, digest in FROZEN_HASHES.items()),
         "frozen parent/calibration artifact changed")
    report, completion = read(directory / "report.json"), read(directory / "completion.json")
    closure, pins = read(directory / "capture.complete.json"), read(directory / "pins.json")
    capture_sha = sha(directory / "capture.jsonl")
    need(closure == {"schema": RUN_SCHEMA + "-closure", "rows": ROWS,
                     "capture_sha256": capture_sha, "labels_opened": False},
         "source-only closure differs")
    need(completion.get("completed") is True
         and completion.get("report_sha256") == sha(directory / "report.json"),
         "terminal completion differs")
    need(report.get("completed") is True and report.get("capture_sha256") == capture_sha
         and report.get("closure_sha256") == sha(directory / "capture.complete.json")
         and report.get("pins") == pins, "terminal report/closure differs")
    need(pins.get(path_key(PRODUCER)) == PRODUCER_SHA256
         and pins.get(path_key(ADAPTER)) == ADAPTER_SHA256
         and pins.get(path_key(DEV)) == sha(DEV), "critical pins differ")
    for raw, digest in pins.items():
        need("/test.jsonl" not in raw and "document-reserve" not in raw,
             "forbidden TEST/reserve pin")
        path = Path(raw) if Path(raw).is_absolute() else ROOT / raw
        if path.resolve() != DEV.resolve():
            need(path.is_file() and sha(path) == digest, "pin changed: " + raw)

    frozen_rows = list(jsonl(FROZEN_UNION / "capture.jsonl"))
    need(len(frozen_rows) == ROWS, "frozen source count differs")
    requests = [(row["id"], row["request"]["input"]) for row in frozen_rows]
    need(len({identifier for identifier, _ in requests}) == ROWS
         and all(row["request"].get("context") == "" for row in frozen_rows),
         "frozen source identity differs")
    weights_record = read(CALIBRATION / "weights.json")
    weights = weights_record.get("weights")
    need(weights_record.get("feature_names") == list(FEATURE_NAMES)
         and isinstance(weights, list) and len(weights) == 4
         and all(finite(value) for value in weights), "calibration weights differ")
    schedule = read(directory / "probe-schedule.json")
    expected_groups = probe_groups([(identifier, source, None) for identifier, source in requests])
    need(schedule == {"groups": expected_groups,
                      "selection": "16 source-length balanced groups; longest source per group"},
         "probe schedule differs")

    frozen_by_id = {row["id"]: row for row in frozen_rows}
    probes = list(jsonl(directory / "probe.jsonl"))
    need(len(probes) == 16, "probe row count differs")
    for group, row in zip(expected_groups, probes):
        source = frozen_by_id[group["probe_id"]]["request"]["input"]
        need(row.get("id") == group["probe_id"]
             and row.get("request") == {"input": source, "context": ""}
             and row.get("exact_full_path_parity") is True
             and row.get("parent", {}).get("result") == row.get("proposed", {}).get("result")
             and row.get("parent", {}).get("candidates") == row.get("proposed", {}).get("candidates"),
             "fresh zero probe public parity differs")
        verify_response(source, frozen_by_id[row["id"]], row["proposed"],
                        row["base_paths"], [0.0] * 4)

    qualification = read(directory / "qualification.json")
    qprobes = qualification.get("probes")
    need(qualification.get("qualified") is True and isinstance(qprobes, list)
         and len(qprobes) == 16 and report.get("zero_probe_timing") == qprobes,
         "qualification probe identity differs")
    weighted = 0.0
    for group, timing in zip(expected_groups, qprobes):
        need(timing.get("id") == group["probe_id"] and timing.get("group_size") == group["count"]
             and finite(timing.get("zero_seconds")) and timing["zero_seconds"] >= 0
             and finite(timing.get("union_engine_ms")) and timing["union_engine_ms"] >= 0
             and finite(timing.get("zero_engine_ms")) and timing["zero_engine_ms"] >= 0,
             "qualification timing differs")
        weighted += timing["zero_seconds"] * group["count"]
    projected = qualification.get("projected_seconds")
    implied_elapsed = (projected - 60.0) / 1.3 - weighted
    need(finite(projected) and projected <= LIMITS["whole_seconds"]
         and implied_elapsed >= 0 and qualification.get("projection")
             == "1.3*(elapsed+sum(zero_seconds*group_size))+60",
         "qualification projection differs")

    rows = []; census = Counter()
    for expected, row in zip(frozen_rows, jsonl(directory / "capture.jsonl")):
        need(set(row) == {"id", "request", "parent", "union_reference", "parent_pool",
                          "proposed", "base_paths"}
             and row["id"] == expected["id"] and row["request"] == expected["request"]
             and row["parent"] == expected["parent"]
             and row["union_reference"] == expected["proposed"]
             and row["parent_pool"] == expected["parent_pool"], "capture/frozen identity differs")
        values = verify_response(row["request"]["input"], expected, row["proposed"],
                                 row["base_paths"], weights)
        census.update(values); rows.append(row)
    need(len(rows) == ROWS, "capture row count differs")

    # Only after the complete source-only capture, closure, terminal report,
    # completion, and per-row replay above do canonical DEV labels open.
    labels = list(jsonl(DEV))
    need(len(labels) == ROWS, "DEV label count differs")
    production = aggregate(rows, labels, "parent")
    union = aggregate(rows, labels, "union_reference"); union.pop("stage_counts")
    metrics = {"versus_production": production, "versus_union": union,
               "quality_gate": production["quality_gate"] and union["quality_gate"]}
    need(production["parent"]["counts"] == {"rows": 316, "top1": 262, "top5": 282, "errors": 0}
         and union["parent"]["counts"] == {"rows": 316, "top1": 273, "top5": 291, "errors": 0},
         "frozen reference metrics differ")
    for key, value in metrics.items():
        need(report.get(key) == value, "reported metric differs: " + key)
    learned_counts = report["versus_union"]["proposed"]["counts"]
    threshold_gate = bool(metrics["quality_gate"]
                          and learned_counts["top1"] > 273
                          and learned_counts["top5"] >= 291)
    need(report.get("quality_gate") is metrics["quality_gate"]
         and metrics["quality_gate"] is threshold_gate,
         "predeclared quality gate differs")
    expected_latency = {
        "actual_calibrated": latency(rows, "proposed"),
        "historical_production": latency(rows, "parent"),
        "historical_union": latency(rows, "union_reference"),
        "likelihood": latency([{"choice": row["proposed"]["phonetic_choice"]} for row in rows],
                              "choice", "likelihood_ms"),
        "calibration_without_likelihood": latency(
            [{"choice": row["proposed"]["phonetic_choice"]} for row in rows],
            "choice", "calibration_ms"),
    }
    need(report.get("latency") == expected_latency
         and report.get("timing_scope")
            == "actual calibrated runtime; full historical reference times are not paired"
         and report.get("actual_zero_probe_rows") == 16
         and report.get("full_zero_shadow_parity_rows") == ROWS
         and report.get("weights") == weights and report.get("limits") == LIMITS
         and report.get("production_changed") is False and report.get("test_accessed") is False
         and report.get("labels_opened_after_capture") is True
         and report.get("gold_not_passed_to_predictor") is True,
         "report identity/timing contract differs")

    operations = ["authenticate_inputs", "load_models"]
    operations += [name + ":" + group["probe_id"] for group in expected_groups
                   for name in ("probe_union", "probe_zero")]
    operations += ["convert:" + identifier for identifier, _ in requests]
    operations += ["close_capture", "late_label_join", "write_report", "write_completion"]
    need(len(operations) == 354, "expected operation count differs")
    resources = verify_resources(list(jsonl(directory / "resources.jsonl")), operations)
    need(completion == {"completed": True, "report_sha256": sha(directory / "report.json"),
                        "resource_after_report": resources["final"]},
         "completion resource linkage differs")
    need(report.get("probe_sha256") == sha(directory / "probe.jsonl")
         and report.get("qualification_sha256") == sha(directory / "qualification.json"),
         "probe/qualification hash differs")
    proof = {
        "schema": SCHEMA, "verified": True,
        "report_sha256": sha(directory / "report.json"),
        "completion_sha256": sha(directory / "completion.json"),
        "capture_sha256": capture_sha,
        "closure_sha256": sha(directory / "capture.complete.json"),
        "probe_sha256": sha(directory / "probe.jsonl"),
        "qualification_sha256": sha(directory / "qualification.json"),
        "producer_sha256": PRODUCER_SHA256, "adapter_sha256": ADAPTER_SHA256,
        "verifier_sha256": sha(Path(__file__)),
        "counts": {"rows": ROWS, "probe_rows": 16, "paths": census["paths"],
                   "unique_pair_scores_per_request_sum": census["pairs"],
                   "supported_edge_occurrences": census["supported_edges"],
                   "operations": resources["operations"]},
        "metrics": metrics, "decision": {"quality_gate": threshold_gate,
            "learned_top1": learned_counts["top1"], "learned_top5": learned_counts["top5"],
            "required_strictly_above_union_top1": 273, "required_minimum_top5": 291},
        "latency": expected_latency,
        "resources": resources, "limits": LIMITS,
        "contracts_verified": {
            "source_only_capture_closed_before_dev_labels": True,
            "frozen_production_and_union_public_references": True,
            "zero_weight_public_and_full_pool_order_parity_all_rows": True,
            "fresh_zero_probe_public_and_recorded_full_path_assertions": True,
            "independent_edge_status_pair_join_feature_and_residual_replay": True,
            "independent_public_raw_slot_and_scored_source_policy": True,
            "independent_metrics_against_both_references": True,
            "pins_completion_and_atomic_resources": True,
            "test_or_reserve_accessed": False,
            "model_loaded_or_inference_run_by_verifier": False,
        },
        "limitations": [
            "Recorded teacher-forced likelihood values are validated but neural likelihoods are not rerun.",
            "Probe records do not contain the reference engine's full paths; exact_full_path_parity is a producer assertion. The verifier independently replays the zero adapter pool, features, public output, and frozen public reference.",
            "Historical production and union latency values are not paired with the fresh calibrated execution.",
            "DEV evidence is not TEST, document-reserve, web, or production approval.",
        ],
    }
    proof_path.write_bytes(canonical(proof))
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    print(json.dumps(verify(args.run), ensure_ascii=False, sort_keys=True))
