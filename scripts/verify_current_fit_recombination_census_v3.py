#!/usr/bin/env python3
"""Independently verify a source-only FIT edge-recombination census (v3).

This verifier authenticates captured pools and proposal provenance.  It does
not rerun the bounded graph search and therefore makes no optimality claim.
"""
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
PRODUCER = ROOT / "scripts/current_fit_recombination_census.py"
PREPARED = ROOT / ".omx/artifacts/current-phonetic-train-inputs-v1"
SOURCE_DIRS = tuple(ROOT / f".omx/artifacts/current-phonetic-train-shard-{i}-v1"
                    for i in range(6))
ROWS = 1642
SHARDS = 6
LIMITS = {"whole_seconds": 900.0, "update_seconds": 30.0,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
DISK_LIMITS = {"per_shard_output_bytes": 2 * 1024**3,
               "all_shards_output_bytes": 8 * 1024**3,
               "projection_margin": 1.3}
EXPECTED_LIMITS = {"max_proposals": 96, "max_pops_per_seed": 20_000,
                   "max_frontier_per_seed": 4_000, "max_seeds": 4}
GENERATOR = ROOT / "hanglish/core/source_edge_recombination.py"
RESOURCE_HELPER = ROOT / "scripts/source_span_resources.py"
PREFLIGHT = ROOT / "scripts/preflight_pko_seq2seq.py"
PLAN = ROOT / ".omx/plans/current-fit-recombination-census-v1.md"
STATIC_HASHES = {
    GENERATOR: "c4e7d7c59c3edb8623b28dc933f429b220b4d17ff58416d47449eeaa4d78d119",
    RESOURCE_HELPER: "4bdf5f598677d1d5c94bcca39545015e7ae12370b0a91482aca1ba42e11a75bd",
    PREFLIGHT: "89bc4bd2a9875ace9eb010ca443168df4db84ed34772342dd55923e2ab7518f2",
    PLAN: "700494b1e96b1ac22c576c2ed2cfe906814c0c31a5166dbbcd9c7e3a69fc84de",
}
PREPARED_HASHES = {
    "manifest.json": "c135c4c7f35c763257bca3b4369ab15165b186b20692bf5d8c1d53a6ec337685",
    "requests.json": "97b416a805b9da8dc15d4ff9178b3a1db2a98f57f961a0e369c51bc771e1762d",
    "independent-verification.json": "4b34c0622f33afc6ec02c16415119159857f4a8947143412e84290a2379efc14",
}
SOURCE_HASHES = (
    {"report.json": "c1ba55856d91986cbe364d146982382141f9c32a176942c33c5f3205f69fbbe5", "independent-verification.json": "54f0d8268ee1d4f2b992ee1919bbfb8c449f324df377f2e5afbc079bfba5f93f", "capture.jsonl": "0053b278869aec714bd48a2d5e5adc504292c75a78cfd5fa2ba8205ef2e24d4d", "capture.complete.json": "9689f30218d3c2d5a2f08e65c4caefd8fe52e5968b4bd75beab239fd8e27fce9", "completion.json": "6a5820a744b660943561498f244d859d491f1ff1cdb181dc9c807d067af92e8c"},
    {"report.json": "4c4645a179d442b6279dbe24b6a3444cbd056260e714fddfeb7e18a71cd540f4", "independent-verification.json": "357d7efac62bb50a53320e3ac1dd60444a9b367835f49669606065c8d8a82f77", "capture.jsonl": "909accff83c0cc783049a5c75ed40e64e895e6b159082f60b7bb1264701adb29", "capture.complete.json": "5f1744b71388378a2832f4badca1041ae7f4616ace7d6639df082a0768e7cb13", "completion.json": "ad60bf3d9c2d360efc020a930440ca0f222065fdc58582586b3bdaadb73eafa3"},
    {"report.json": "e186b383c3a004758cd17936387cd4c6d200453a88ae02d8a3551be4f030f2a2", "independent-verification.json": "1a7ce752e5cc246e4b6b31927fe7e36ea1c3844862950e2adbb2067dbf1ee852", "capture.jsonl": "de7d0da7228c3ad44f28d80bb69e64bfe0f73d0c9b55d9fa749ac4f498580882", "capture.complete.json": "9b103b00b55b4321edfb196743aaea7ff40ee27fd9f7c3ea06b2aacc72ecaea6", "completion.json": "c0681d816889155b044293a71a95c2fb91b0accff8e28cae58191a55a31278e7"},
    {"report.json": "0a13ff490331bb7ce696890d8de669afd594046f685b4212a03ac5a65bd07e7c", "independent-verification.json": "f67a0dabb0c528cbc8ec92de37127fe3c1da10f2ae7b2e7535c8a9fc8fbd46a9", "capture.jsonl": "317346be3b2efc2c04a2e849d49adf752ea6dd775ef751c41cd79a7243a54019", "capture.complete.json": "f5a5927d8a193fc81b4446d47cf5f91185a679714a66927cc94451a29daa4ea2", "completion.json": "26f137cd6da9aeb1309a2d358a0b9a2bf8ebb9c40a08789845c07da62e3a2151"},
    {"report.json": "afe2ac68783c27dd47cfb71fad576dc49233d5e9316d8bbf95472a9079cb8b3b", "independent-verification.json": "8c59e3d4b98f12fdc3e01e9f1f83e1b9a536ad7a14f9abce242e71064974b592", "capture.jsonl": "9ced02dfa254bd29f3094be38a96307dc79a7285edd18bb7b56a838064f9aaaf", "capture.complete.json": "e9fc8285731aecd80c23ee12e31df46101b24f33b2e14af0506bb5842ad194a9", "completion.json": "e6dcb11cb55bdbdff7a001b34a7ca230ccfa9db7bf717406f3f12915a6037c98"},
    {"report.json": "e8adee9ab3163b503ce18308caa32e4a9d6dedf172cd74e4414fb04f70d5a757", "independent-verification.json": "ddb82affedd08e79a9c800e5e4ad529aba6b3bd32fdea32d486c41067b3e56c4", "capture.jsonl": "8f1a260362660346fc8b809b0e2369db6b7ec3b34c05235b13ddc9e66c9771ac", "capture.complete.json": "6bfcda7470e9ff75b484a115c9f54aad125bb3868d75c0a103a589cd563309f3", "completion.json": "ad7900e8bed80e383161b3fec0551c2e29ecae90d804be17ab116181ccdbf4ec"},
)
SOURCE_LAUNCH_HASHES = (
    "c37672099fe5d7f4b967dd0ffee858c7a6106ab7a8b10a3da4c7dd25662785d2",
    "27d16ac3e8d0cf4b820c152532b4333c9b244255d05ef8ff9241f713b77a9439",
    "67b55c3c1c995ada10066fef7cd92aaf4c795ac173c85273bce28bd6d8149e22",
    "17a88a1ec480ff615ce8b7c20f5435fe222fd67797adb6e6588ee31f3f1c6cd6",
    "9e802608864e79ffa36fd5934ccb7e267b8d7c72e2054a590e628e8adcb6f91a",
    "f670f5ab44949d4e769f42e9fdd83362bd3e1b1142463a87f616477ff5d528cc",
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def stable_id(prefix, value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return prefix + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    need(isinstance(value, dict), "expected JSON object: " + str(path))
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
            need(isinstance(value, dict), "non-object JSONL row")
            yield value


def jsonl_with_sizes(path):
    with Path(path).open("rb") as stream:
        for number, line in enumerate(stream, 1):
            need(line.strip(), f"blank JSONL line at {path}:{number}")
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"malformed JSONL at {path}:{number}") from error
            need(isinstance(value, dict), "non-object JSONL row")
            yield value, len(line)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_segments(source, text, segments):
    need(isinstance(segments, list) and segments, "segments missing")
    cursor, pieces = 0, []
    for segment in segments:
        start, end = segment.get("input_start"), segment.get("input_end")
        need(type(start) is int and type(end) is int and start == cursor
             and start < end <= len(source)
             and segment.get("source") == source[start:end]
             and isinstance(segment.get("text"), str)
             and isinstance(segment.get("kind"), str), "segment coverage differs")
        cursor = end; pieces.append(segment["text"])
    need(cursor == len(source) and "".join(pieces) == text,
         "segment reconstruction differs")


def edge_inventory(identifier, source, base_paths):
    """Rebuild the helper's deduplicated edge graph from captured paths."""
    edges, by_key, occurrences = [], {}, 0
    for pool_index, path in enumerate(base_paths):
        need(isinstance(path, list) and len(path) == 3 and isinstance(path[0], str)
             and finite(path[1]) and isinstance(path[2], list), "captured path differs")
        validate_segments(source, path[0], path[2])
        candidate_id = path_id = f"current-phonetic-fit:{identifier}:{pool_index}"
        for segment_index, segment in enumerate(path[2]):
            occurrences += 1
            key = (segment["input_start"], segment["input_end"], segment["text"])
            provenance = {"candidate_id": candidate_id, "path_id": path_id,
                          "pool_index": pool_index, "segment_index": segment_index,
                          "origin": "captured_full_union",
                          "origins": ["captured_full_union"], "kind": segment["kind"]}
            if key not in by_key:
                copied = segment["source"] == segment["text"]
                edge = {"edge_id": stable_id("edge-v1-", {
                            "source": source, "input_start": key[0],
                            "input_end": key[1], "text": key[2]}),
                        "action": "copy" if copied else "convert",
                        "copy_exact": True if copied else None,
                        "structural_segment": segment, "provenance": []}
                by_key[key] = edge; edges.append(edge)
            by_key[key]["provenance"].append(provenance)
    return edges, occurrences


def verify_proposal(source, proposal, inventory, original_texts):
    need(isinstance(proposal, dict) and proposal.get("origin") == "source_edge_recombination"
         and proposal.get("selection_eligible") is False
         and proposal.get("score") is None and proposal.get("score_type") is None,
         "proposal policy differs")
    text, segments, edge_ids = proposal.get("text"), proposal.get("segments"), proposal.get("edge_ids")
    need(isinstance(text, str) and text not in original_texts
         and isinstance(edge_ids, list) and len(edge_ids) == len(segments),
         "proposal novelty or edge coverage differs")
    validate_segments(source, text, segments)
    by_id = {edge["edge_id"]: edge for edge in inventory}
    copied = 0
    for segment, edge_id in zip(segments, edge_ids):
        edge = by_id.get(edge_id)
        need(edge is not None and edge["structural_segment"] == segment
             , "proposal uses an unauthenticated edge")
        if edge["action"] == "copy":
            need(edge["copy_exact"] is True and segment["source"] == segment["text"],
                 "copy edge differs")
            copied += 1
    expected_id = stable_id("proposal-v1-", {"source": source, "edge_ids": edge_ids})
    need(proposal.get("proposal_id") == expected_id
         and type(proposal.get("seed_index")) is int
         and isinstance(proposal.get("seed_text"), str)
         and type(proposal.get("changed_edges")) is int
         and proposal["changed_edges"] >= 0, "proposal identity differs")
    return copied


def verify_used_inventory(used, full_inventory, referenced):
    by_id = {edge["edge_id"]: edge for edge in full_inventory}
    need(isinstance(used, list) and len({edge.get("edge_id") for edge in used}) == len(used)
         and {edge.get("edge_id") for edge in used} == set(referenced)
         and all(by_id.get(edge.get("edge_id")) == edge for edge in used),
         "used edge inventory differs")
    return used


def verify_row(record, source_row, source_shard, source_index, capture_sha):
    need(record.get("schema") == "current-fit-recombination-census-row-v1"
         and record.get("id") == source_row.get("id")
         and record.get("component_id") == source_row.get("component_id")
         and record.get("request") == source_row.get("request"), "census row identity differs")
    source = source_row["request"]["input"]
    paths = source_row.get("base_paths")
    response = source_row.get("response", {})
    public = response.get("candidates")
    ranked = response.get("sentence_ranking_applied")
    need(isinstance(paths, list) and isinstance(public, list) and public
         and type(ranked) is bool and response.get("input") == source
         and response.get("context") == "" and response.get("external_context_used") is False
         and ((ranked and bool(paths)) or (not ranked and not paths)),
         "captured source pool differs")
    for candidate in public:
        need(isinstance(candidate, dict) and isinstance(candidate.get("text"), str),
             "captured public candidate differs")
        validate_segments(source, candidate["text"], candidate.get("segments"))
    reference = record.get("source_reference")
    need(reference == {"source_shard": source_shard, "source_row_index": source_index,
                       "source_capture_sha256": capture_sha,
                       "base_paths_sha256": canonical_sha(paths),
                       "public_candidates_sha256": canonical_sha(public),
                       "pool_rows": len(paths), "ranked": ranked},
         "source reference differs")
    status, proposals = record.get("status"), record.get("proposals")
    need(status in ("generated", "no_pool") and isinstance(proposals, list),
         "row status differs")
    if not paths:
        need(status == "no_pool" and proposals == [] and record.get("fallback_public") == public,
             "no-pool row differs")
    else:
        need(status == "generated" and ranked is True and record.get("fallback_public") is None,
             "ranked row status differs")
    inventory, occurrences = edge_inventory(record["id"], source, paths)
    used = record.get("used_edge_inventory")
    referenced = [edge_id for proposal in proposals for edge_id in proposal.get("edge_ids", [])]
    verify_used_inventory(used, inventory, referenced)
    diagnostics = record.get("diagnostics")
    need(isinstance(diagnostics, dict) and diagnostics.get("limits") == EXPECTED_LIMITS
         and diagnostics.get("source_length") == len(source)
         and diagnostics.get("pool_rows") == len(paths)
         and diagnostics.get("edge_occurrences") == occurrences
         and diagnostics.get("edge_count") == len(inventory)
         and diagnostics.get("proposal_count") == len(proposals)
         and diagnostics.get("score_fields_consumed") is False
         and len(proposals) <= EXPECTED_LIMITS["max_proposals"],
         "proposal diagnostics differ")
    seeds = diagnostics.get("seeds")
    need(isinstance(seeds, list) and diagnostics.get("seed_count") == len(seeds)
         and len(seeds) <= EXPECTED_LIMITS["max_seeds"]
         and diagnostics.get("complete_paths_rejected_as_existing_or_duplicate")
             == sum(seed.get("complete_paths_examined", -1) for seed in seeds) - len(proposals),
         "seed/duplicate diagnostics differ")
    allowed_reasons = {"no_complete_path", "exhausted", "max_pops", "max_frontier",
                       "global_proposal_limit"}
    seed_candidates = [row for row in public if row["text"] != source][:EXPECTED_LIMITS["max_seeds"]]
    need(len(seeds) == len(seed_candidates)
         and diagnostics.get("global_limit_reached") is (len(proposals) == EXPECTED_LIMITS["max_proposals"]),
         "seed/global limit census differs")
    accepted_by_seed = {index: 0 for index in range(len(seeds))}
    for proposal in proposals:
        index = proposal.get("seed_index")
        need(type(index) is int and index in accepted_by_seed, "proposal seed index differs")
        accepted_by_seed[index] += 1
        seed_keys = {(part["input_start"], part["input_end"], part["text"])
                     for part in seed_candidates[index]["segments"]}
        changed = sum((part["input_start"], part["input_end"], part["text"]) not in seed_keys
                      for part in proposal["segments"])
        need(proposal.get("seed_text") == seed_candidates[index]["text"]
             and proposal.get("changed_edges") == changed,
             "proposal seed/change accounting differs")
    for index, (seed, candidate) in enumerate(zip(seeds, seed_candidates)):
        need(seed.get("termination_reason") in allowed_reasons
             and all(type(seed.get(key)) is int and seed[key] >= 0 for key in
                     ("seed_index", "base_candidate_index", "pops", "max_frontier",
                      "complete_paths_examined", "accepted_proposals"))
             and seed.get("seed_index") == index
             and seed.get("base_candidate_index") == public.index(candidate)
             and seed.get("seed_text") == candidate["text"]
             and seed.get("pops") <= EXPECTED_LIMITS["max_pops_per_seed"]
             and seed.get("max_frontier") <= EXPECTED_LIMITS["max_frontier_per_seed"]
             and seed.get("accepted_proposals") == accepted_by_seed[index],
             "seed diagnostic differs")
    need(sum(seed["accepted_proposals"] for seed in seeds) == len(proposals),
         "accepted proposal census differs")
    if not paths:
        need(diagnostics.get("pool_rows") == 0 and diagnostics.get("edge_count") == 0
             and diagnostics.get("edge_occurrences") == 0
             and all(seed["termination_reason"] == "no_complete_path"
                     and seed["pops"] == 0 and seed["max_frontier"] == 0
                     and seed["complete_paths_examined"] == 0
                     and seed["accepted_proposals"] == 0
                     for seed in seeds),
             "no-pool diagnostics differ")
        return {"proposals": 0, "copy_edges": 0, "pool_rows": 0}
    original = {source, *(path[0] for path in paths), *(row["text"] for row in public)}
    seen, copies = set(), 0
    for proposal in proposals:
        need(proposal.get("text") not in seen, "duplicate proposal text")
        seen.add(proposal["text"])
        copies += verify_proposal(source, proposal, used, original)
    return {"proposals": len(proposals), "copy_edges": copies, "pool_rows": len(paths)}


def verify_schedule(prepared, report):
    groups, shards = prepared.get("groups"), prepared.get("shards")
    need(isinstance(groups, list) and len(groups) == 16
         and isinstance(shards, list) and len(shards) == SHARDS,
         "prepared schedule differs")
    rows = prepared.get("rows")
    by_id = {row["id"]: row for row in rows}
    ordered = sorted(rows, key=lambda row: (len(row["input"]), row["id"]))
    small, extra = divmod(len(ordered), 16)
    expected, offset = [], 0
    for index in range(16):
        count = small + int(index < extra)
        quantile = ordered[offset:offset + count]
        expected.append({"ids": [row["id"] for row in quantile],
                         "probe_id": quantile[-1]["id"], "count": count,
                         "maximum_source_length": len(quantile[-1]["input"])})
        offset += count
    need(groups == expected, "16-quantile probe schedule differs")
    timing = report.get("timing_seconds")
    projected = report.get("projected_shard_seconds")
    initialization = report.get("initialization_seconds")
    need(isinstance(timing, dict) and set(timing) == {row["probe_id"] for row in groups}
         and finite(initialization) and initialization >= 0
         and isinstance(projected, list) and len(projected) == SHARDS,
         "probe timing contract differs")
    per_id = {identifier: timing[group["probe_id"]]
              for group in groups for identifier in group["ids"]}
    expected_projection = [1.3 * (initialization + sum(per_id[item] for item in shard)) + 60.0
                           for shard in shards]
    need(all(finite(value) and value >= 0 for value in timing.values())
         and projected == expected_projection
         and report.get("qualified") is (all(value <= 900.0 for value in projected)),
         "six-shard projection differs")
    need(set(by_id) == {item for shard in shards for item in shard}, "shard coverage differs")
    components = {}
    for index, shard in enumerate(shards):
        for identifier in shard:
            component = by_id[identifier]["component_id"]
            need(component not in components or components[component] == index,
                 "component split across shards")
            components[component] = index
    return expected_projection


def verify_storage_schedule(prepared, report, serialized_sizes):
    groups, shards = prepared["groups"], prepared["shards"]
    need(set(serialized_sizes) == {group["probe_id"] for group in groups}
         and all(type(value) is int and value > 0 for value in serialized_sizes.values()),
         "probe serialized sizes differ")
    per_id = {identifier: serialized_sizes[group["probe_id"]]
              for group in groups for identifier in group["ids"]}
    expected = [math.ceil(DISK_LIMITS["projection_margin"]
                          * sum(per_id[item] for item in shard)) for shard in shards]
    need(report.get("projected_shard_output_bytes") == expected
         and report.get("projected_total_output_bytes") == sum(expected)
         and all(value <= DISK_LIMITS["per_shard_output_bytes"] for value in expected)
         and sum(expected) <= DISK_LIMITS["all_shards_output_bytes"],
         "projected storage budget differs")
    return expected


def verify_resources(rows, operations):
    need(rows, "resource trace empty")
    starts, ends, current = [], [], None
    after_report = after_completion = None
    baseline, previous = None, -1.0
    for row in rows:
        need(finite(row.get("elapsed_s")) and previous <= row["elapsed_s"] <= LIMITS["whole_seconds"]
             and type(row.get("rss_bytes")) is int and 0 <= row["rss_bytes"] <= LIMITS["rss_bytes"]
             and type(row.get("swap_used_bytes")) is int and row["swap_used_bytes"] >= 0
             and type(row.get("swap_growth_bytes")) is int
             and row["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"], "resource row differs")
        datetime.fromisoformat(row["at"]); previous = row["elapsed_s"]
        operation, elapsed = row.get("operation"), row.get("operation_elapsed_s")
        need((operation is None) is (elapsed is None), "operation attribution differs")
        if elapsed is not None:
            need(finite(elapsed) and 0 <= elapsed <= LIMITS["update_seconds"], "operation limit differs")
        observed = row["swap_used_bytes"] - row["swap_growth_bytes"]
        need(observed >= 0, "swap baseline invalid")
        baseline = observed if baseline is None else baseline
        need(observed == baseline, "swap baseline changed")
        stage = row.get("stage")
        if stage == "operation_start":
            need(current is None and isinstance(operation, str), "operation start differs")
            current = operation; starts.append(operation)
        elif stage == "operation_end":
            need(current == operation, "operation end differs")
            current = None; ends.append(operation)
        elif stage == "watchdog": need(operation == current, "watchdog attribution differs")
        elif stage == "after_report":
            need(current is None and after_report is None and starts == operations[:-1], "after-report differs")
            after_report = row
        elif stage == "after_completion":
            need(current is None and after_report is not None and starts == operations, "after-completion differs")
            after_completion = row
        else: raise ValueError("unexpected resource stage: " + str(stage))
    need(starts == operations == ends and after_report and after_completion,
         "resource operation sequence differs")
    return {"rows": len(rows), "operations": len(starts), "final": after_report,
            "after_completion": after_completion, "baseline_swap_bytes": baseline,
            "peak_rss_bytes": max(row["rss_bytes"] for row in rows),
            "peak_swap_growth_bytes": max(row["swap_growth_bytes"] for row in rows)}


def authenticate_inputs():
    pins = {}
    for name, digest in PREPARED_HASHES.items():
        path = PREPARED / name
        need(path.is_file() and sha256_file(path) == digest, "prepared input differs: " + name)
        pins[str(path.resolve())] = digest
    for path, digest in STATIC_HASHES.items():
        need(path.is_file() and sha256_file(path) == digest, "fixed helper differs: " + str(path))
        pins[str(path.resolve())] = digest
    prepared = read_json(PREPARED / "requests.json")
    manifest = read_json(PREPARED / "manifest.json")
    proof = read_json(PREPARED / "independent-verification.json")
    rows, groups, shards = prepared.get("rows"), prepared.get("groups"), prepared.get("shards")
    need(manifest.get("rows") == ROWS and manifest.get("labels_opened") is False
         and proof.get("verified") is True and proof.get("counts", {}).get("rows") == ROWS
         and proof.get("manifest_sha256") == PREPARED_HASHES["manifest.json"]
         and proof.get("requests_sha256") == PREPARED_HASHES["requests.json"]
         and isinstance(rows, list) and len(rows) == ROWS
         and len({row.get("id") for row in rows}) == ROWS
         and all(set(row) == {"id", "input", "component_id"} for row in rows)
         and isinstance(groups, list) and len(groups) == 16
         and isinstance(shards, list) and len(shards) == SHARDS
         and sorted(item for shard in shards for item in shard) == sorted(row["id"] for row in rows),
         "prepared FIT domain differs")
    for index, (directory, hashes, launch_hash) in enumerate(
            zip(SOURCE_DIRS, SOURCE_HASHES, SOURCE_LAUNCH_HASHES)):
        for name, digest in hashes.items():
            path = directory / name
            need(path.is_file() and sha256_file(path) == digest,
                 f"source shard {index} {name} differs")
            pins[str(path.resolve())] = digest
        launch_path = directory.with_suffix(".launch.json")
        need(launch_path.is_file() and sha256_file(launch_path) == launch_hash,
             f"source shard {index} launch differs")
        pins[str(launch_path.resolve())] = launch_hash
        report = read_json(directory / "report.json")
        source_proof = read_json(directory / "independent-verification.json")
        closure = read_json(directory / "capture.complete.json")
        completion = read_json(directory / "completion.json")
        need(report.get("completed") is True and report.get("mode") == "capture"
             and report.get("shard") == index and report.get("ids") == shards[index]
             and report.get("capture_sha256") == hashes["capture.jsonl"]
             and source_proof.get("verified") is True
             and source_proof.get("mode") == "capture" and source_proof.get("shard") == index
             and source_proof.get("report_sha256") == hashes["report.json"]
             and source_proof.get("capture_sha256") == hashes["capture.jsonl"]
             and closure.get("rows") == len(shards[index])
             and closure.get("capture_sha256") == hashes["capture.jsonl"]
             and closure.get("labels_opened") is False
             and completion.get("completed") is True
             and completion.get("report_sha256") == hashes["report.json"],
             f"source shard {index} proof/closure differs")
    return prepared, {row["id"]: row for row in rows}, pins


def _authenticate_probe(directory, binding=None):
    directory = Path(directory).resolve()
    required = ("report.json", "capture.jsonl", "capture.complete.json",
                "completion.json", "independent-verification.json")
    need(all((directory / name).is_file() for name in required), "probe terminal files missing")
    report = read_json(directory / "report.json")
    proof = read_json(directory / "independent-verification.json")
    closure = read_json(directory / "capture.complete.json")
    completion = read_json(directory / "completion.json")
    hashes = {name: sha256_file(directory / name) for name in required}
    need(report.get("schema") == "current-fit-recombination-census-v1"
         and report.get("completed") is True and report.get("mode") == "probe"
         and report.get("shard") is None and report.get("rows") == 16
         and report.get("qualified") is True
         and proof.get("verified") is True and proof.get("mode") == "probe"
         and proof.get("qualified") is True
         and proof.get("report_sha256") == hashes["report.json"]
         and proof.get("capture_sha256") == hashes["capture.jsonl"]
         and closure.get("capture_sha256") == hashes["capture.jsonl"]
         and closure.get("labels_opened") is False
         and completion.get("completed") is True
         and completion.get("report_sha256") == hashes["report.json"],
         "qualified independent probe differs")
    if binding is not None:
        need(binding == {"directory": str(directory),
                         "report_sha256": hashes["report.json"],
                         "capture_sha256": hashes["capture.jsonl"],
                         "closure_sha256": hashes["capture.complete.json"],
                         "completion_sha256": hashes["completion.json"],
                         "proof_sha256": hashes["independent-verification.json"]},
             "capture/probe binding differs")
    return report, hashes


def _counts(records):
    keys = ("rows", "ranked_rows", "unranked_rows", "no_pool_rows",
            "proposal_count", "global_limit_rows", "multiple_segment_proposals",
            "complete_paths_rejected_as_existing_or_duplicate", "edge_count",
            "edge_occurrences", "seed_count", "exhausted_seeds", "max_pops_seeds",
            "max_frontier_seeds", "no_complete_path_seeds",
            "global_proposal_limit_seeds")
    counts = {key: 0 for key in keys}; reasons = {}
    for row in records:
        _update_counts(counts, reasons, row)
    return counts, dict(sorted(reasons.items()))


def _update_counts(counts, reasons, row):
    diag = row["diagnostics"]
    counts["rows"] += 1
    counts["ranked_rows"] += row["source_reference"]["ranked"]
    counts["unranked_rows"] += not row["source_reference"]["ranked"]
    counts["no_pool_rows"] += row["status"] == "no_pool"
    counts["proposal_count"] += len(row["proposals"])
    counts["global_limit_rows"] += diag["global_limit_reached"]
    counts["multiple_segment_proposals"] += sum(len(p["segments"]) > 1 for p in row["proposals"])
    for key in ("complete_paths_rejected_as_existing_or_duplicate", "edge_count",
                "edge_occurrences", "seed_count"):
        counts[key] += diag[key]
    for seed in diag["seeds"]:
        reason = seed["termination_reason"]
        reasons[reason] = reasons.get(reason, 0) + 1
        counts[reason + "_seeds"] += 1


def _selected_source_rows(mode, shard, prepared, row_map):
    wanted = ([group["probe_id"] for group in prepared["groups"]]
              if mode == "probe" else prepared["shards"][shard])
    if mode == "capture":
        def stream():
            iterator = jsonl(SOURCE_DIRS[shard] / "capture.jsonl")
            for index, identifier in enumerate(wanted):
                original = next(iterator, None)
                need(original is not None and original.get("id") == identifier,
                     "capture source order differs")
                yield original, row_map[identifier], shard, index, SOURCE_HASHES[shard]["capture.jsonl"]
            need(next(iterator, None) is None, "capture source has extra rows")
        return wanted, stream()
    found = {}
    selected = set(wanted)
    for source_shard, directory in enumerate(SOURCE_DIRS):
        for index, original in enumerate(jsonl(directory / "capture.jsonl")):
            identifier = original.get("id")
            if identifier in selected:
                need(identifier not in found, "duplicate probe source row")
                found[identifier] = (original, row_map[identifier], source_shard, index,
                                     SOURCE_HASHES[source_shard]["capture.jsonl"])
    need(set(found) == selected, "probe source rows missing")
    return wanted, (found[identifier] for identifier in wanted)


def _expected_launch(output, report):
    command = [str(ROOT / ".venv/bin/python"), "-m",
               "scripts.current_fit_recombination_census",
               "--mode", report["mode"], "--output", str(output)]
    if report["mode"] == "capture":
        binding = report["probe_binding"]
        command += ["--shard", str(report["shard"]), "--probe", binding["directory"],
                    "--probe-report-sha256", binding["report_sha256"]]
    return command


def verify(output, producer_sha256):
    raw_output = Path(output)
    need(raw_output.is_absolute(), "output must be absolute")
    output = raw_output.resolve()
    need(re.fullmatch(r"[0-9a-f]{64}", producer_sha256) is not None
         and sha256_file(PRODUCER) == producer_sha256, "producer identity differs")
    proof_path = output / "independent-verification.json"
    need(output.is_dir() and not proof_path.exists(), "run/proof state invalid")
    need(not any((output / name).exists() for name in
                 ("failure.json", "failure-report.json", "resource-watchdog-failure.json")),
         "failure artifact exists")
    required = ("report.json", "capture.jsonl", "capture.complete.json",
                "completion.json", "pins.json", "resources.jsonl")
    need(all((output / name).is_file() for name in required), "terminal files missing")
    report, closure, completion, pins = (read_json(output / name) for name in
                                         ("report.json", "capture.complete.json",
                                          "completion.json", "pins.json"))
    mode, shard = report.get("mode"), report.get("shard")
    need(mode in ("probe", "capture")
         and ((mode == "probe" and shard is None) or
              (mode == "capture" and type(shard) is int and 0 <= shard < SHARDS)),
         "mode/shard differs")
    capture_sha = sha256_file(output / "capture.jsonl")
    report_sha = sha256_file(output / "report.json")
    need(report.get("schema") == "current-fit-recombination-census-v1"
         and report.get("completed") is True and report.get("qualified") is True
         and report.get("limits") == LIMITS and report.get("disk_limits") == DISK_LIMITS
         and report.get("producer_sha256") == producer_sha256
         and report.get("generator_sha256") == STATIC_HASHES[GENERATOR]
         and report.get("generator_schema") == "source-edge-recombination-v1"
         and report.get("generator_limits") == EXPECTED_LIMITS
         and report.get("pins") == pins and report.get("capture_sha256") == capture_sha
         and report.get("closure_sha256") == sha256_file(output / "capture.complete.json")
         and all(report.get(key) is False for key in
                 ("labels_opened", "fit_feature_files_opened", "dev_opened", "test_opened",
                  "model_loaded", "model_inference", "neural_training", "production_changed"))
         and report.get("source_only") is True and report.get("train_only") is True,
         "terminal report contract differs")
    need(closure == {"schema": "current-fit-recombination-census-closure-v1",
                     "completed": True, "mode": mode, "shard": shard,
                     "rows": report.get("rows"), "capture_sha256": capture_sha,
                     "labels_opened": False, "model_loaded": False, "model_inference": False},
         "capture closure differs")
    need(completion.get("completed") is True and completion.get("report_sha256") == report_sha,
         "completion differs")
    launch_path = output.with_suffix(".launch.json")
    launch = read_json(launch_path)
    try:
        os.kill(launch.get("pid"), 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("producer PID is still present")
    need(launch.get("command") == _expected_launch(output, report)
         and launch.get("output") == str(output)
         and launch.get("runner_sha256") == producer_sha256,
         "launch identity differs")

    prepared, row_map, expected_pins = authenticate_inputs()
    expected_pins[str(PRODUCER.resolve())] = producer_sha256
    probe_report = probe_hashes = None
    if mode == "capture":
        binding = report.get("probe_binding")
        need(isinstance(binding, dict), "probe binding missing")
        probe_report, probe_hashes = _authenticate_probe(binding.get("directory"), binding)
        for name, digest in probe_hashes.items():
            expected_pins[str((Path(binding["directory"]) / name).resolve())] = digest
    else:
        need(report.get("probe_binding") is None, "probe run has a probe binding")
    need(all(pins.get(name) == digest for name, digest in expected_pins.items()),
         "required source/helper pin missing")
    for name, digest in pins.items():
        path = Path(name)
        need(path.is_file() and sha256_file(path) == digest, "runtime pin differs: " + name)

    wanted, selected = _selected_source_rows(mode, shard, prepared, row_map)
    count_keys = ("rows", "ranked_rows", "unranked_rows", "no_pool_rows",
                  "proposal_count", "global_limit_rows", "multiple_segment_proposals",
                  "complete_paths_rejected_as_existing_or_duplicate", "edge_count",
                  "edge_occurrences", "seed_count", "exhausted_seeds", "max_pops_seeds",
                  "max_frontier_seeds", "no_complete_path_seeds",
                  "global_proposal_limit_seeds")
    counts, reasons, observed_ids, serialized = ({key: 0 for key in count_keys}, {}, [], {})
    capture_rows = jsonl_with_sizes(output / "capture.jsonl")
    for expected_source, captured in zip_longest(selected, capture_rows):
        need(expected_source is not None and captured is not None, "census denominator differs")
        original, expected, source_shard, source_index, source_capture_sha = expected_source
        record, byte_count = captured
        verify_row(record, original, source_shard, source_index, source_capture_sha)
        need(record["id"] == expected["id"], "prepared/census ID differs")
        observed_ids.append(record["id"])
        serialized[record["id"]] = byte_count
        _update_counts(counts, reasons, record)
        del record
    reasons = dict(sorted(reasons.items()))
    need(observed_ids == wanted
         and report.get("ids") == wanted and report.get("rows") == len(wanted),
         "census row order differs")
    need(report.get("counts") == counts
         and report.get("seed_termination_reasons") == reasons,
         "census aggregate differs")
    need(report.get("serialized_output_bytes") == serialized
         and report.get("capture_output_bytes") == sum(serialized.values()) == (output / "capture.jsonl").stat().st_size
         and report["capture_output_bytes"] <= DISK_LIMITS["per_shard_output_bytes"],
         "serialized output census differs")
    timings = report.get("timing_seconds")
    need(isinstance(timings, dict) and set(timings) == set(wanted)
         and all(finite(value) and value >= 0 for value in timings.values()),
         "per-row timings differ")
    if mode == "probe":
        verify_schedule(prepared, report)
        verify_storage_schedule(prepared, report, serialized)
    else:
        need(report.get("projected_shard_seconds") == probe_report["projected_shard_seconds"]
             and report.get("projected_shard_output_bytes") == probe_report["projected_shard_output_bytes"]
             and report.get("projected_total_output_bytes") == probe_report["projected_total_output_bytes"],
             "capture projection binding differs")
    operations = ["authenticate"] + ["propose:" + identifier for identifier in wanted]
    operations += ["write_report", "write_completion"]
    resources = verify_resources(list(jsonl(output / "resources.jsonl")), operations)
    need(completion.get("resource_after_report") == resources["final"],
         "completion resource binding differs")
    proof = {"schema": "current-fit-recombination-census-independent-verification-v1",
             "verified": True, "mode": mode, "shard": shard, "rows": len(wanted),
             "qualified": True, "report_sha256": report_sha,
             "capture_sha256": capture_sha, "closure_sha256": sha256_file(output / "capture.complete.json"),
             "launch_sha256": sha256_file(launch_path), "producer_sha256": producer_sha256,
             "verifier_sha256": sha256_file(__file__), "counts": counts,
             "projected_shard_seconds": report["projected_shard_seconds"],
             "projected_shard_output_bytes": report["projected_shard_output_bytes"],
             "projected_total_output_bytes": report["projected_total_output_bytes"],
             "resources": resources, "source_only": True, "labels_opened": False,
             "model_loaded": False, "model_inference": False, "training": False,
             "scope": "Authenticated source edges, row order, components, contiguous reconstruction, proposal novelty, duplicate exclusion, copy edges, unique used-edge provenance, closures, time/disk projections, and resource limits.",
             "limitations": "The bounded search is not rerun; this proof validates recorded provenance and contracts, not proposal optimality or graph-unreachability for capped misses."}
    with proof_path.open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--producer-sha256", required=True)
    args = parser.parse_args()
    result = verify(args.output, args.producer_sha256)
    print(json.dumps({key: result[key] for key in
                      ("verified", "mode", "shard", "rows", "qualified")},
                     ensure_ascii=False))
