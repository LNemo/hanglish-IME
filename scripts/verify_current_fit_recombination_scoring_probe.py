#!/usr/bin/env python3
"""Independently verify the closed source-only recombination scoring probe."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / ".omx/scripts/probe_current_fit_recombination_scoring_v1.py"
CORE = ROOT / "scripts/current_recombination_scoring.py"
INTAKE = ROOT / ".omx/artifacts/current-fit-recombination-coverage-intake-v1.json"
PREPARED = ROOT / ".omx/artifacts/current-phonetic-train-inputs-v1/requests.json"
SOURCE_DIRS = tuple(ROOT / f".omx/artifacts/current-phonetic-train-shard-{i}-v1"
                    for i in range(6))
INTAKE_SHA256 = "64456f9616f3615d24b82c9163edd092fc0a93fe503c64d5b59784115176dbec"
PREPARED_SHA256 = "97b416a805b9da8dc15d4ff9178b3a1db2a98f57f961a0e369c51bc771e1762d"
LIMITS = {"whole_seconds": 900.0, "update_seconds": 30.0,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
NAMES = ("report.json", "capture.jsonl", "capture.complete.json", "completion.json",
         "independent-verification.json", "pins.json", "resources.jsonl")
ASCII = re.compile(r"[a-z]+\Z")
HANGUL = re.compile(r"[가-힣]+\Z")


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    need(isinstance(value, dict), "expected JSON object")
    return value


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            need(line.strip(), f"blank JSONL line {number}")
            value = json.loads(line)
            need(isinstance(value, dict), "JSONL row is not an object")
            yield value


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_segments(source, text, segments):
    need(isinstance(segments, list) and segments, "segments missing")
    cursor, pieces = 0, []
    for segment in segments:
        start, end = segment.get("input_start"), segment.get("input_end")
        need(type(start) is int and type(end) is int and start == cursor
             and start < end <= len(source) and segment.get("source") == source[start:end]
             and isinstance(segment.get("text"), str) and isinstance(segment.get("kind"), str),
             "segment source coverage differs")
        cursor = end; pieces.append(segment["text"])
    need(cursor == len(source) and "".join(pieces) == text, "segment reconstruction differs")


def expected_status(source, target):
    if not source or not target: return "empty"
    if source == target: return "copy"
    if ASCII.fullmatch(source) is None: return "unsupported_ascii"
    if HANGUL.fullmatch(target) is None: return "unsupported_shape"
    if len(source) > 48 or len(target) > 24: return "length"
    return "supported"


def score_records(records):
    need(isinstance(records, list), "score records missing")
    result = {}
    for row in records:
        need(isinstance(row, dict) and set(row) == {"pair_id", "source", "target",
             "log_probability", "token_count", "mean_log_probability"},
             "score record schema differs")
        pair = row["pair_id"]
        need(isinstance(pair, str) and pair not in result and finite(row["log_probability"])
             and row["log_probability"] <= 0 and finite(row["mean_log_probability"])
             and row["mean_log_probability"] <= 0 and type(row["token_count"]) is int
             and row["token_count"] == 3 * len(row["target"]) + 1
             and row["mean_log_probability"] == row["log_probability"] / row["token_count"],
             "score record value differs")
        result[pair] = row
    return result


def features(source, text, segments, edges, scores):
    validate_segments(source, text, segments)
    need(isinstance(edges, list) and len(edges) == len(segments), "edge coverage differs")
    values = [0.0] * 4; cursor = 0
    for segment, edge in zip(segments, edges):
        need(isinstance(edge, dict) and edge.get("input_start") == cursor
             and edge.get("input_start") == segment["input_start"]
             and edge.get("input_end") == segment["input_end"]
             and edge.get("source") == segment["source"] and edge.get("target") == segment["text"]
             and edge.get("status") == expected_status(edge["source"], edge["target"]),
             "captured edge differs")
        cursor = edge["input_end"]; weight = (edge["input_end"] - edge["input_start"]) / len(source)
        if edge["status"] == "supported":
            score = scores.get(edge.get("pair_id"))
            need(score is not None and score["source"] == edge["source"]
                 and score["target"] == edge["target"] and type(edge.get("romanize_match")) is bool,
                 "supported edge score identity differs")
            values[0] += weight * score["mean_log_probability"]
            values[1] += weight
            values[3] += weight * (edge["romanize_match"] is False)
        elif edge["status"] != "copy": values[2] += weight
    return values


def verify_scoring_row(original, census, captured, *, original_parity_required=True):
    need(captured.get("id") == original.get("id") == census.get("id")
         and captured.get("component_id") == original.get("component_id") == census.get("component_id")
         and captured.get("request") == original.get("request") == census.get("request"),
         "scoring source identity differs")
    source = original["request"]["input"]
    need(captured.get("source_shard") == census["source_reference"]["source_shard"]
         and captured.get("source_row_index") == census["source_reference"]["source_row_index"],
         "source location differs")
    choice = original["response"]["phonetic_choice"]
    scores = score_records(choice.get("score_records"))
    paths, rows = original.get("base_paths"), choice.get("rows")
    scoring = captured.get("scoring")
    need(isinstance(scoring, dict) and scoring.get("schema") == "current-recombination-scoring-row-v1"
         and scoring.get("id") == captured["id"] and scoring.get("request") == captured["request"]
         and scoring.get("source_reference") == census.get("source_reference"),
         "scoring row contract differs")
    originals = scoring.get("original")
    ranked = original["response"].get("sentence_ranking_applied")
    need(type(ranked) is bool and scoring.get("ranked") is ranked
         and census["source_reference"].get("ranked") is ranked
         and bool(paths) is ranked
         and scoring.get("fallback_public") == (None if ranked else original["response"]["candidates"]),
         "ranked state or fallback public candidates differ")
    need(isinstance(paths, list) and isinstance(rows, list) and isinstance(originals, list)
         and len(paths) == len(rows) == len(originals), "original score denominator differs")
    locations = {}
    for index, (path, row, result) in enumerate(zip(paths, rows, originals)):
        vector = features(source, path[0], path[2], row.get("edges"), scores)
        need(row.get("text") == path[0] and row.get("base_score") == path[1]
             and row.get("segments") == path[2] and row.get("features") == vector
             and result == {"text": path[0], "sentence_score": float(path[1]),
                            "features": vector, "segments": path[2], "status": "preserved",
                            "score_source": "captured_original"},
             "original cached score or feature differs")
        for part, edge in enumerate(row["edges"]): locations[(index, part)] = edge
    inventory, exact = {}, {}
    for item in census.get("used_edge_inventory", []):
        edge_id = item.get("edge_id"); segment = item.get("structural_segment")
        chosen = None
        for evidence in item.get("provenance", []):
            key = (evidence.get("pool_index"), evidence.get("segment_index"))
            edge = locations.get(key)
            need(edge is not None and paths[key[0]][2][key[1]]["kind"] == evidence.get("kind")
                 and (edge["input_start"], edge["input_end"], edge["target"])
                     == (segment["input_start"], segment["input_end"], segment["text"]),
                 "novel edge provenance differs")
            need(chosen is None or chosen == edge, "heterogeneous provenance score edge differs")
            chosen = edge
        need(chosen is not None and edge_id not in inventory, "used edge inventory differs")
        inventory[edge_id] = segment; exact[edge_id] = chosen
    proposals, novel = census.get("proposals"), scoring.get("novel")
    need(isinstance(proposals, list) and isinstance(novel, list) and len(proposals) == len(novel),
         "novel denominator differs")
    for proposal, result in zip(proposals, novel):
        edge_ids = proposal.get("edge_ids"); edges = [exact.get(item) for item in edge_ids]
        need(None not in edges and proposal.get("segments") == [inventory[item] for item in edge_ids],
             "novel edge list differs")
        vector = features(source, proposal["text"], proposal["segments"], edges, scores)
        need(result.get("proposal_id") == proposal.get("proposal_id")
             and result.get("text") == proposal.get("text")
             and result.get("segments") == proposal.get("segments")
             and result.get("edge_ids") == edge_ids and result.get("features") == vector,
             "novel candidate identity or feature differs")
        status = result.get("status")
        need((status == "scored" and finite(result.get("sentence_score")) and result.get("reason") is None)
             or (status == "unsupported" and result.get("sentence_score") is None
                 and result.get("reason") == "sentence_ranker_can_score_false"),
             "novel score/status differs")
    parity = scoring.get("original_score_parity")
    need(scoring.get("score_texts") == [row["text"] for row in novel if row["status"] == "scored"],
         "scored text order differs")
    if not original_parity_required:
        need(parity is None, "full capture must not claim an unperformed parity replay")
        return {"original": len(originals), "novel": len(novel),
                "scored_novel": sum(row["status"] == "scored" for row in novel),
                "unsupported_novel": sum(row["status"] == "unsupported" for row in novel)}
    need(isinstance(parity, dict) and parity == {"checked": True, "tolerance": 1e-5,
                    "max_absolute_difference": parity.get("max_absolute_difference"),
                    "rows": len(originals)}
         and finite(parity["max_absolute_difference"])
         and 0 <= parity["max_absolute_difference"] <= 1e-5,
         "recorded original parity summary differs")
    return {"original": len(originals), "novel": len(novel),
            "scored_novel": sum(row["status"] == "scored" for row in novel),
            "unsupported_novel": sum(row["status"] == "unsupported" for row in novel)}


def verify_resources(rows, operations):
    need(rows, "resource trace empty")
    current = None; starts = []; ends = []; previous = -1.; baseline = None
    after_rehash = after_completion = None
    bounded = {op for op in operations if op.startswith("score:")}
    bounded |= {"write_closure", "write_report", "write_completion"}
    for row in rows:
        need(finite(row.get("elapsed_s")) and previous <= row["elapsed_s"] <= 900
             and type(row.get("rss_bytes")) is int and 0 <= row["rss_bytes"] <= LIMITS["rss_bytes"]
             and type(row.get("swap_used_bytes")) is int and type(row.get("swap_growth_bytes")) is int
             and row["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"], "resource row differs")
        datetime.fromisoformat(row["at"]); previous = row["elapsed_s"]
        op, elapsed = row.get("operation"), row.get("operation_elapsed_s")
        # update=False retains the operation name but has no per-update clock.
        # Initialization and rehashing still obey the whole-run deadline.
        if op in bounded:
            need(finite(elapsed) and elapsed >= 0, "bounded operation duration missing")
            need(elapsed <= 30, "bounded operation exceeded 30 seconds")
        else:
            need(elapsed is None, "unbounded operation has an update clock")
        observed = row["swap_used_bytes"] - row["swap_growth_bytes"]
        baseline = observed if baseline is None else baseline
        need(observed == baseline and observed >= 0, "swap baseline differs")
        stage = row.get("stage")
        if stage == "operation_start": need(current is None, "nested operation"); current=op; starts.append(op)
        elif stage == "operation_end": need(current == op, "operation end differs"); current=None; ends.append(op)
        elif stage == "watchdog": need(op == current, "watchdog attribution differs")
        elif stage == "after_final_rehash": need(current is None and starts == operations[:-1], "rehash sample differs"); after_rehash=row
        elif stage == "after_completion": need(after_rehash is not None and starts == operations, "completion sample differs"); after_completion=row
        else: raise ValueError("unexpected resource stage")
    need(starts == operations == ends and after_rehash and after_completion, "resource sequence differs")
    return {"rows": len(rows), "operations": len(starts), "after_final_rehash": after_rehash,
            "after_completion": after_completion, "peak_rss_bytes": max(row["rss_bytes"] for row in rows),
            "peak_swap_growth_bytes": max(row["swap_growth_bytes"] for row in rows)}


def probe_groups(rows):
    ordered = sorted(rows, key=lambda row: (len(row["input"]), row["id"]))
    small, extra = divmod(len(ordered), 16); result=[]; offset=0
    for index in range(16):
        count = small + int(index < extra); group=ordered[offset:offset+count]
        result.append({"ids": [row["id"] for row in group], "probe_id": group[-1]["id"],
                       "count": count, "maximum_source_length": len(group[-1]["input"])})
        offset += count
    return result


def verify_projection(groups, shards, timings, initialization, recorded, qualified):
    need(isinstance(timings,dict) and set(timings)=={group["probe_id"] for group in groups}
         and finite(initialization) and initialization>=0
         and all(finite(value) and value>=0 for value in timings.values()),"timings differ")
    by_id={identifier:timings[group["probe_id"]] for group in groups for identifier in group["ids"]}
    expected=[1.3*(initialization+sum(by_id[item] for item in shard))+60 for shard in shards]
    decision=all(value<=900 for value in expected)
    need(recorded==expected and qualified is decision,"schedule projection differs")
    return expected


def authenticate_selected():
    need(sha(INTAKE) == INTAKE_SHA256 and sha(PREPARED) == PREPARED_SHA256,
         "frozen intake/prepared requests differ")
    manifest, prepared = read(INTAKE), read(PREPARED)
    rows, groups, shards = prepared.get("rows"), prepared.get("groups"), prepared.get("shards")
    need(isinstance(rows, list) and len(rows) == 1642 and groups == probe_groups(rows)
         and isinstance(shards, list) and len(shards) == 6, "prepared source schedule differs")
    wanted = [group["probe_id"] for group in groups]; selected = set(wanted); found={}
    entries = manifest.get("shards")
    need(manifest.get("schema") == "current-fit-recombination-coverage-intake-v1"
         and isinstance(entries, list) and len(entries) == 6, "census intake differs")
    by_id = {row["id"]: row for row in rows}
    for shard, entry in enumerate(entries):
        directory = Path(entry.get("directory", "")); hashes=entry.get("hashes")
        need(directory.is_absolute() and isinstance(hashes, dict) and set(hashes) == set(NAMES),
             "census shard inventory differs")
        for name,digest in hashes.items(): need(sha(directory/name)==digest,"census shard hash differs")
        report,proof,closure=(read(directory/name) for name in
                              ("report.json","independent-verification.json","capture.complete.json"))
        need(report.get("completed") is True and report.get("mode")=="capture"
             and report.get("shard")==shard and report.get("ids")==shards[shard]
             and proof.get("verified") is True and proof.get("report_sha256")==hashes["report.json"]
             and closure.get("capture_sha256")==hashes["capture.jsonl"],"closed census differs")
        originals=jsonl(SOURCE_DIRS[shard]/"capture.jsonl"); census_rows=jsonl(directory/"capture.jsonl")
        for position,(original,census) in enumerate(zip_longest(originals,census_rows)):
            need(original is not None and census is not None and position < len(shards[shard])
                 and original.get("id")==census.get("id")==shards[shard][position],
                 "source/census order differs")
            if census["id"] in selected:
                need(census["id"] not in found and by_id[census["id"]]["component_id"]==census["component_id"],
                     "selected source identity differs")
                found[census["id"]]=(original,census)
    need(set(found)==selected,"selected rows missing")
    return prepared,wanted,found


def verify(output, runner_sha256, core_sha256):
    output=Path(output); need(output.is_absolute(),"output must be absolute")
    output=output.resolve(); proof_path=output/"independent-verification.json"
    need(output.is_dir() and not proof_path.exists(),"run/proof state invalid")
    need(not any((output/name).exists() for name in ("failure.json","resource-watchdog-failure.json")),
         "failure artifact exists")
    report,closure,completion,pins=(read(output/name) for name in
        ("report.json","capture.complete.json","completion.json","pins.json"))
    need(re.fullmatch(r"[0-9a-f]{64}",runner_sha256) and re.fullmatch(r"[0-9a-f]{64}",core_sha256)
         and sha(CORE)==core_sha256,"core identity differs")
    launch_path=output.with_suffix(".launch.json"); launch=read(launch_path)
    command=launch.get("command"); need(isinstance(command,list) and len(command)==6,"launch command differs")
    runner_path=Path(command[1])
    need(command==[str(ROOT/".venv/bin/python"),str(runner_path),"--core-sha256",core_sha256,
                   "--runner-sha256",runner_sha256]
         and runner_path.is_absolute() and sha(runner_path)==runner_sha256
         and launch.get("runner_sha256")==runner_sha256
         and launch.get("core_sha256")==core_sha256 and launch.get("output")==str(output),
         "launch identity differs")
    try: os.kill(launch["pid"],0)
    except ProcessLookupError: pass
    else: raise ValueError("runner PID is still present")
    capture_sha=sha(output/"capture.jsonl")
    need(report.get("schema")=="current-fit-recombination-scoring-probe-v1"
         and report.get("completed") is True and report.get("mode")=="probe"
         and report.get("rows")==16 and report.get("runner_sha256")==runner_sha256
         and report.get("core_sha256")==core_sha256 and report.get("capture_sha256")==capture_sha
         and report.get("closure_sha256")==sha(output/"capture.complete.json")
         and report.get("limits")==LIMITS and report.get("original_score_parity_tolerance")==1e-5
         and report.get("model_metadata")=={"max_length":96,"batch_size":8,"device":"cpu",
                                             "threads":2,"interop_threads":1}
         and report.get("source_only") is True and report.get("fit_labels_opened") is False
         and report.get("dev_opened") is False and report.get("test_opened") is False
         and report.get("model_inference") is True and report.get("phonetic_model_inference") is False
         and report.get("neural_training") is False and report.get("production_changed") is False,
         "terminal report differs")
    need(closure=={"schema":"current-fit-recombination-scoring-closure-v1","mode":"probe",
                  "completed":True,"rows":16,"ids":report["ids"],"capture_sha256":capture_sha,
                  "runner_sha256":runner_sha256,"core_sha256":core_sha256,"labels_opened":False,
                  "model_inference":True,"phonetic_model_inference":False},"closure differs")
    need(completion.get("completed") is True and completion.get("qualified") is report.get("qualified")
         and completion.get("report_sha256")==sha(output/"report.json"),"completion differs")
    need(report.get("pins")==pins and pins.get(str(runner_path.resolve()))==runner_sha256
         and pins.get(str(CORE.resolve()))==core_sha256 and pins.get(str(INTAKE.resolve()))==INTAKE_SHA256,
         "required pins differ")
    for path,digest in pins.items(): need(Path(path).is_file() and sha(path)==digest,"runtime pin differs")
    replay = report.get("original_batch_replay")
    if replay is not None:
        directory = Path(replay.get("directory", "")); hashes = replay.get("hashes")
        need(directory.is_absolute() and isinstance(hashes, dict)
             and set(hashes) == {"report.json", "capture.jsonl", "completion.json",
                                "independent-verification.json", "pins.json", "resources.jsonl"},
             "batch replay inventory differs")
        for name,digest in hashes.items():
            need(pins.get(str(directory/name)) == digest and sha(directory/name) == digest,
                 "batch replay pin differs")
        batch_proof, batch_report = read(directory/"independent-verification.json"), read(directory/"report.json")
        need(batch_proof.get("verified") is True
             and batch_proof.get("report_sha256") == hashes["report.json"]
             and batch_report.get("capture_sha256") == hashes["capture.jsonl"]
             and batch_report.get("ids") == report.get("ids"), "batch replay proof differs")
        helper = ROOT/"scripts/sentence_batch_replay.py"
        need(pins.get(str(helper)) == replay.get("helper_sha256") == sha(helper),
             "batch replay helper differs")
    prepared,wanted,sources=authenticate_selected()
    need(report.get("ids")==wanted,"probe ID order differs")
    observed=[]; counts={}
    for identifier,captured in zip_longest(wanted,jsonl(output/"capture.jsonl")):
        need(identifier is not None and captured is not None and captured.get("id")==identifier,
             "scoring capture order differs")
        original,census=sources[identifier]
        counts[identifier]=verify_scoring_row(original,census,captured); observed.append(identifier)
    need(report.get("row_counts")==counts,"row count summary differs")
    projections=verify_projection(prepared["groups"],prepared["shards"],
        report.get("timing_seconds"),report.get("initialization_seconds"),
        report.get("projected_shard_seconds"),report.get("qualified"))
    qualified=report["qualified"]
    operations=["authenticate_sources","load_frozen_sentence_ranker"]+["score:"+i for i in wanted]
    operations += ["write_closure","write_report","final_rehash","write_completion"]
    resources=verify_resources(list(jsonl(output/"resources.jsonl")),operations)
    need(completion.get("resource_after_final_rehash")==resources["after_final_rehash"],
         "completion resource binding differs")
    proof={"schema":"current-fit-recombination-scoring-probe-independent-verification-v1",
           "verified":True,"qualified":qualified,"rows":16,"report_sha256":sha(output/"report.json"),
           "capture_sha256":capture_sha,"closure_sha256":sha(output/"capture.complete.json"),
           "launch_sha256":sha(launch_path),"runner_sha256":runner_sha256,"core_sha256":core_sha256,
           "verifier_sha256":sha(__file__),"resources":resources,
           "scope":"Cached original scores, independently recomputed four phonetic features, source/census coverage, recorded score status/parity summaries, projections, pins, closure, and resources.",
           "limitations":"Does not replay neural sentence-ranker logits or tokenizer can_score decisions; original parity is validated as recorded evidence only.",
           "labels_opened":False,"model_loaded":False,"model_inference_replayed":False}
    with proof_path.open("x") as stream:
        json.dump(proof,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);stream.write("\n")
    return proof


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--runner-sha256",required=True)
    parser.add_argument("--core-sha256",required=True)
    args=parser.parse_args(); result=verify(args.output,args.runner_sha256,args.core_sha256)
    print(json.dumps({key:result[key] for key in ("verified","qualified","rows")},ensure_ascii=False))
