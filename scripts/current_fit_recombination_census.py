#!/usr/bin/env python3
"""Capture a source-only recombination census over the frozen 1,642 FIT rows."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time

from hanglish.core.source_edge_recombination import (
    MAX_FRONTIER_PER_SEED,
    MAX_POPS_PER_SEED,
    MAX_PROPOSALS,
    MAX_SEEDS,
    SCHEMA_VERSION as GENERATOR_SCHEMA,
    propose,
)
from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / ".omx/artifacts/current-phonetic-train-inputs-v1"
SHARDS = tuple(ROOT / f".omx/artifacts/current-phonetic-train-shard-{index}-v1"
               for index in range(6))
GENERATOR = ROOT / "hanglish/core/source_edge_recombination.py"
RESOURCE_HELPER = ROOT / "scripts/source_span_resources.py"
PREFLIGHT = ROOT / "scripts/preflight_pko_seq2seq.py"
PLAN = ROOT / ".omx/plans/current-fit-recombination-census-v1.md"

ROWS = 1642
PROBES = 16
SHARD_COUNT = 6
LIMITS = {"whole_seconds": 900.0, "update_seconds": 30.0,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
DISK_LIMITS = {"per_shard_output_bytes": 2 * 1024**3,
               "all_shards_output_bytes": 8 * 1024**3,
               "projection_margin": 1.3}
SCHEMA = "current-fit-recombination-census-v1"
ROW_SCHEMA = "current-fit-recombination-census-row-v1"
CLOSURE_SCHEMA = "current-fit-recombination-census-closure-v1"
PREPARED_HASHES = {
    "requests.json": "97b416a805b9da8dc15d4ff9178b3a1db2a98f57f961a0e369c51bc771e1762d",
    "manifest.json": "c135c4c7f35c763257bca3b4369ab15165b186b20692bf5d8c1d53a6ec337685",
    "independent-verification.json": "4b34c0622f33afc6ec02c16415119159857f4a8947143412e84290a2379efc14",
}
STATIC_HASHES = {
    GENERATOR: "c4e7d7c59c3edb8623b28dc933f429b220b4d17ff58416d47449eeaa4d78d119",
    RESOURCE_HELPER: "4bdf5f598677d1d5c94bcca39545015e7ae12370b0a91482aca1ba42e11a75bd",
    PREFLIGHT: "89bc4bd2a9875ace9eb010ca443168df4db84ed34772342dd55923e2ab7518f2",
    PLAN: "700494b1e96b1ac22c576c2ed2cfe906814c0c31a5166dbbcd9c7e3a69fc84de",
}
SHARD_HASHES = (
    {"capture.jsonl": "0053b278869aec714bd48a2d5e5adc504292c75a78cfd5fa2ba8205ef2e24d4d",
     "report.json": "c1ba55856d91986cbe364d146982382141f9c32a176942c33c5f3205f69fbbe5",
     "independent-verification.json": "54f0d8268ee1d4f2b992ee1919bbfb8c449f324df377f2e5afbc079bfba5f93f",
     "capture.complete.json": "9689f30218d3c2d5a2f08e65c4caefd8fe52e5968b4bd75beab239fd8e27fce9",
     "completion.json": "6a5820a744b660943561498f244d859d491f1ff1cdb181dc9c807d067af92e8c",
     "launch.json": "c37672099fe5d7f4b967dd0ffee858c7a6106ab7a8b10a3da4c7dd25662785d2"},
    {"capture.jsonl": "909accff83c0cc783049a5c75ed40e64e895e6b159082f60b7bb1264701adb29",
     "report.json": "4c4645a179d442b6279dbe24b6a3444cbd056260e714fddfeb7e18a71cd540f4",
     "independent-verification.json": "357d7efac62bb50a53320e3ac1dd60444a9b367835f49669606065c8d8a82f77",
     "capture.complete.json": "5f1744b71388378a2832f4badca1041ae7f4616ace7d6639df082a0768e7cb13",
     "completion.json": "ad60bf3d9c2d360efc020a930440ca0f222065fdc58582586b3bdaadb73eafa3",
     "launch.json": "27d16ac3e8d0cf4b820c152532b4333c9b244255d05ef8ff9241f713b77a9439"},
    {"capture.jsonl": "de7d0da7228c3ad44f28d80bb69e64bfe0f73d0c9b55d9fa749ac4f498580882",
     "report.json": "e186b383c3a004758cd17936387cd4c6d200453a88ae02d8a3551be4f030f2a2",
     "independent-verification.json": "1a7ce752e5cc246e4b6b31927fe7e36ea1c3844862950e2adbb2067dbf1ee852",
     "capture.complete.json": "9b103b00b55b4321edfb196743aaea7ff40ee27fd9f7c3ea06b2aacc72ecaea6",
     "completion.json": "c0681d816889155b044293a71a95c2fb91b0accff8e28cae58191a55a31278e7",
     "launch.json": "67b55c3c1c995ada10066fef7cd92aaf4c795ac173c85273bce28bd6d8149e22"},
    {"capture.jsonl": "317346be3b2efc2c04a2e849d49adf752ea6dd775ef751c41cd79a7243a54019",
     "report.json": "0a13ff490331bb7ce696890d8de669afd594046f685b4212a03ac5a65bd07e7c",
     "independent-verification.json": "f67a0dabb0c528cbc8ec92de37127fe3c1da10f2ae7b2e7535c8a9fc8fbd46a9",
     "capture.complete.json": "f5a5927d8a193fc81b4446d47cf5f91185a679714a66927cc94451a29daa4ea2",
     "completion.json": "26f137cd6da9aeb1309a2d358a0b9a2bf8ebb9c40a08789845c07da62e3a2151",
     "launch.json": "17a88a1ec480ff615ce8b7c20f5435fe222fd67797adb6e6588ee31f3f1c6cd6"},
    {"capture.jsonl": "9ced02dfa254bd29f3094be38a96307dc79a7285edd18bb7b56a838064f9aaaf",
     "report.json": "afe2ac68783c27dd47cfb71fad576dc49233d5e9316d8bbf95472a9079cb8b3b",
     "independent-verification.json": "8c59e3d4b98f12fdc3e01e9f1f83e1b9a536ad7a14f9abce242e71064974b592",
     "capture.complete.json": "e9fc8285731aecd80c23ee12e31df46101b24f33b2e14af0506bb5842ad194a9",
     "completion.json": "e6dcb11cb55bdbdff7a001b34a7ca230ccfa9db7bf717406f3f12915a6037c98",
     "launch.json": "9e802608864e79ffa36fd5934ccb7e267b8d7c72e2054a590e628e8adcb6f91a"},
    {"capture.jsonl": "8f1a260362660346fc8b809b0e2369db6b7ec3b34c05235b13ddc9e66c9771ac",
     "report.json": "e8adee9ab3163b503ce18308caa32e4a9d6dedf172cd74e4414fb04f70d5a757",
     "independent-verification.json": "ddb82affedd08e79a9c800e5e4ad529aba6b3bd32fdea32d486c41067b3e56c4",
     "capture.complete.json": "6bfcda7470e9ff75b484a115c9f54aad125bb3868d75c0a103a589cd563309f3",
     "completion.json": "ad7900e8bed80e383161b3fec0551c2e29ecae90d804be17ab116181ccdbf4ec",
     "launch.json": "f670f5ab44949d4e769f42e9fdd83362bd3e1b1142463a87f616477ff5d528cc"},
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "expected JSON object: " + str(path))
    return value


def iter_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSONL at {path}:{number}") from error
            require(isinstance(value, dict), "JSONL row is not an object")
            yield value


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def append_jsonl(stream, value, *, current_bytes=0, limit_bytes=None):
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                          allow_nan=False) + "\n")
    size = len(encoded.encode("utf-8"))
    require(limit_bytes is None or current_bytes + size <= limit_bytes,
            "capture output would exceed the hard 2 GiB bound")
    stream.write(encoded)
    stream.flush(); os.fsync(stream.fileno())
    return size


def probe_groups(rows):
    ordered = sorted(rows, key=lambda row: (len(row["input"]), row["id"]))
    small, extra = divmod(len(ordered), PROBES)
    result, offset = [], 0
    for index in range(PROBES):
        count = small + int(index < extra)
        group = ordered[offset:offset + count]
        require(group, "empty probe quantile")
        result.append({"ids": [row["id"] for row in group],
                       "probe_id": group[-1]["id"], "count": count,
                       "maximum_source_length": len(group[-1]["input"])})
        offset += count
    require(offset == len(rows), "probe quantile coverage differs")
    return result


def projected_shard_seconds(groups, shards, timings, initialization):
    require(isinstance(initialization, (int, float)) and initialization >= 0
            and set(timings) == {group["probe_id"] for group in groups}
            and all(isinstance(value, (int, float)) and value >= 0
                    for value in timings.values()), "probe timings differ")
    estimates = {identifier: timings[group["probe_id"]]
                 for group in groups for identifier in group["ids"]}
    return [1.3 * (initialization + sum(estimates[identifier]
                                       for identifier in shard)) + 60.0
            for shard in shards]


def projected_shard_output_bytes(groups, shards, observed_bytes):
    require(set(observed_bytes) == {group["probe_id"] for group in groups}
            and all(type(value) is int and value > 0
                    for value in observed_bytes.values()),
            "probe serialized output sizes differ")
    estimates = {identifier: observed_bytes[group["probe_id"]]
                 for group in groups for identifier in group["ids"]}
    return [math.ceil(DISK_LIMITS["projection_margin"]
                      * sum(estimates[identifier] for identifier in shard))
            for shard in shards]


def _validate_segments(source, text, segments, label):
    require(isinstance(segments, list) and segments, label + " segments missing")
    cursor, rendered = 0, []
    for segment in segments:
        start, end = segment.get("input_start"), segment.get("input_end")
        require(type(start) is int and type(end) is int and start == cursor
                and start < end <= len(source)
                and segment.get("source") == source[start:end]
                and isinstance(segment.get("text"), str)
                and isinstance(segment.get("kind"), str),
                label + " source edge differs")
        cursor = end; rendered.append(segment["text"])
    require(cursor == len(source) and "".join(rendered) == text,
            label + " reconstruction differs")


def _pool(identifier, source, paths):
    pool = []
    for index, path in enumerate(paths):
        require(isinstance(path, list) and len(path) == 3
                and isinstance(path[0], str)
                and type(path[1]) in (int, float) and math.isfinite(path[1]),
                "captured full path differs")
        _validate_segments(source, path[0], path[2], "captured full path")
        identity = f"current-phonetic-fit:{identifier}:{index}"
        pool.append({"candidate_id": identity, "path_id": identity,
                     "pool_index": index, "text": path[0],
                     "segments": path[2], "origin": "captured_full_union",
                     "origins": ["captured_full_union"], "score": path[1]})
    return pool


def _validate_public(source, candidates):
    require(isinstance(candidates, list) and candidates, "captured public fallback missing")
    for index, candidate in enumerate(candidates):
        require(isinstance(candidate, dict) and isinstance(candidate.get("text"), str),
                "captured public candidate differs")
        _validate_segments(source, candidate["text"], candidate.get("segments"),
                           f"captured public candidate {index}")


def validate_generation(source, pool, generated):
    require(generated.get("schema_version") == GENERATOR_SCHEMA
            and isinstance(generated.get("proposals"), list)
            and isinstance(generated.get("edge_inventory"), list)
            and isinstance(generated.get("diagnostics"), dict),
            "generator result contract differs")
    proposals = generated["proposals"]
    require(len(proposals) <= MAX_PROPOSALS
            and len({row.get("text") for row in proposals}) == len(proposals),
            "proposal count or uniqueness differs")
    expected = {}
    for pool_index, row in enumerate(pool):
        for segment_index, segment in enumerate(row["segments"]):
            key = (segment["input_start"], segment["input_end"], segment["text"])
            expected.setdefault(key, []).append({
                "candidate_id": row["candidate_id"], "path_id": row["path_id"],
                "pool_index": pool_index, "segment_index": segment_index,
                "origin": row["origin"], "origins": row["origins"],
                "kind": segment["kind"]})
    inventory = {}
    for edge in generated["edge_inventory"]:
        segment = edge.get("structural_segment")
        require(isinstance(segment, dict), "edge inventory segment missing")
        key = (segment.get("input_start"), segment.get("input_end"), segment.get("text"))
        require(key in expected and edge.get("provenance") == expected[key]
                and isinstance(edge.get("edge_id"), str)
                and edge["edge_id"] not in inventory,
                "foreign or misattributed source edge")
        inventory[edge["edge_id"]] = edge
    result, used_edge_ids = [], set()
    for proposal in proposals:
        _validate_segments(source, proposal.get("text"), proposal.get("segments"),
                           "recombined proposal")
        keys = [(part["input_start"], part["input_end"], part["text"])
                for part in proposal["segments"]]
        edge_ids = proposal.get("edge_ids")
        require(isinstance(edge_ids, list) and len(edge_ids) == len(keys)
                and all(edge_id in inventory for edge_id in edge_ids)
                and all((edge["structural_segment"]["input_start"],
                         edge["structural_segment"]["input_end"],
                         edge["structural_segment"]["text"]) == key
                        for edge, key in zip((inventory[item] for item in edge_ids), keys)),
                "proposal uses a foreign source edge")
        result.append(deepcopy(proposal))
        used_edge_ids.update(edge_ids)
    diagnostics = deepcopy(generated["diagnostics"])
    require(diagnostics.get("limits") == {
                "max_proposals": MAX_PROPOSALS,
                "max_pops_per_seed": MAX_POPS_PER_SEED,
                "max_frontier_per_seed": MAX_FRONTIER_PER_SEED,
                "max_seeds": MAX_SEEDS}
            and diagnostics.get("proposal_count") == len(proposals),
            "generator diagnostics differ")
    diagnostics["complete_paths_rejected_as_existing_or_duplicate"] = (
        sum(seed["complete_paths_examined"] for seed in diagnostics["seeds"])
        - len(proposals))
    require(diagnostics["complete_paths_rejected_as_existing_or_duplicate"] >= 0,
            "proposal duplicate accounting differs")
    used_inventory = [deepcopy(edge) for edge in generated["edge_inventory"]
                      if edge["edge_id"] in used_edge_ids]
    require({edge["edge_id"] for edge in used_inventory} == used_edge_ids,
            "used edge inventory coverage differs")
    return result, diagnostics, used_inventory


def prepare_record(original, expected, source_shard, source_row_index, capture_sha,
                   generator=propose):
    identifier, source = original.get("id"), original.get("request", {}).get("input")
    require(identifier == expected.get("id")
            and original.get("component_id") == expected.get("component_id")
            and original.get("request") == {"input": expected.get("input"), "context": ""}
            and source == expected.get("input"), "captured/prepared source identity differs")
    response, paths = original.get("response"), original.get("base_paths")
    require(isinstance(response, dict) and isinstance(paths, list)
            and response.get("input") == source and response.get("context") == ""
            and response.get("external_context_used") is False
            and type(response.get("sentence_ranking_applied")) is bool,
            "captured source response differs")
    candidates = response.get("candidates")
    _validate_public(source, candidates)
    ranked = response["sentence_ranking_applied"]
    require((ranked and bool(paths)) or (not ranked and not paths),
            "captured ranked/full-pool relation differs")
    pool = _pool(identifier, source, paths)
    generated = generator(source, pool, candidates)
    proposals, diagnostics, used_inventory = validate_generation(source, pool, generated)
    status = "generated" if ranked else "no_pool"
    require(ranked or (not proposals and diagnostics.get("pool_rows") == 0),
            "unranked row generated proposals")
    return {
        "schema": ROW_SCHEMA, "id": identifier,
        "component_id": expected["component_id"],
        "request": {"input": source, "context": ""},
        "source_reference": {
            "source_shard": source_shard, "source_row_index": source_row_index,
            "source_capture_sha256": capture_sha,
            "base_paths_sha256": canonical_sha256(paths),
            "public_candidates_sha256": canonical_sha256(candidates),
            "pool_rows": len(paths), "ranked": ranked,
        },
        "status": status, "proposals": proposals,
        "used_edge_inventory": used_inventory, "diagnostics": diagnostics,
        "fallback_public": deepcopy(candidates) if not ranked else None,
    }


def _source_paths(shard):
    directory = SHARDS[shard]
    return {name: (directory.with_suffix(".launch.json") if name == "launch.json"
                   else directory / name) for name in SHARD_HASHES[shard]}


def authenticate_inputs():
    pins = {}
    for name, digest in PREPARED_HASHES.items():
        path = PREPARED / name
        require(path.is_file() and sha256_file(path) == digest,
                "prepared FIT source differs: " + name)
        pins[str(path.resolve())] = digest
    for path, digest in STATIC_HASHES.items():
        require(path.is_file() and sha256_file(path) == digest,
                "fixed helper/plan differs: " + str(path))
        pins[str(path.resolve())] = digest
    requests = read_json(PREPARED / "requests.json")
    manifest = read_json(PREPARED / "manifest.json")
    proof = read_json(PREPARED / "independent-verification.json")
    rows, groups, shards = requests.get("rows"), requests.get("groups"), requests.get("shards")
    require(manifest.get("rows") == ROWS and manifest.get("labels_opened") is False
            and proof.get("verified") is True
            and proof.get("counts", {}).get("rows") == ROWS
            and proof.get("manifest_sha256") == PREPARED_HASHES["manifest.json"]
            and proof.get("requests_sha256") == PREPARED_HASHES["requests.json"]
            and isinstance(rows, list) and len(rows) == ROWS
            and probe_groups(rows) == groups and len(groups) == PROBES
            and isinstance(shards, list) and len(shards) == SHARD_COUNT
            and sorted(identifier for values in shards for identifier in values)
                == sorted(row["id"] for row in rows),
            "prepared FIT domain/groups/shards differ")
    row_map = {row.get("id"): row for row in rows}
    require(len(row_map) == ROWS
            and all(set(row) == {"id", "input", "component_id"}
                    for row in rows), "prepared FIT source schema differs")
    for shard, expected_hashes in enumerate(SHARD_HASHES):
        paths = _source_paths(shard)
        for name, digest in expected_hashes.items():
            require(paths[name].is_file() and sha256_file(paths[name]) == digest,
                    f"source shard {shard} {name} differs")
            pins[str(paths[name].resolve())] = digest
        report = read_json(paths["report.json"])
        shard_proof = read_json(paths["independent-verification.json"])
        closure = read_json(paths["capture.complete.json"])
        completion = read_json(paths["completion.json"])
        require(report.get("completed") is True and report.get("mode") == "capture"
                and report.get("shard") == shard and report.get("ids") == shards[shard]
                and report.get("capture_sha256") == expected_hashes["capture.jsonl"]
                and shard_proof.get("verified") is True
                and shard_proof.get("mode") == "capture"
                and shard_proof.get("shard") == shard
                and shard_proof.get("capture_sha256") == expected_hashes["capture.jsonl"]
                and shard_proof.get("report_sha256") == expected_hashes["report.json"]
                and closure.get("labels_opened") is False
                and closure.get("capture_sha256") == expected_hashes["capture.jsonl"]
                and completion.get("completed") is True
                and completion.get("report_sha256") == expected_hashes["report.json"],
                f"verified source shard {shard} closure differs")
    pins[str(Path(__file__).resolve())] = sha256_file(Path(__file__))
    return rows, row_map, groups, shards, dict(sorted(pins.items()))


def authenticate_probe(directory, expected_report_sha):
    directory = Path(directory).resolve()
    require(directory.is_absolute() and isinstance(expected_report_sha, str)
            and len(expected_report_sha) == 64
            and sha256_file(directory / "report.json") == expected_report_sha,
            "probe report SHA differs")
    report = read_json(directory / "report.json")
    proof = read_json(directory / "independent-verification.json")
    closure = read_json(directory / "capture.complete.json")
    completion = read_json(directory / "completion.json")
    require(not (directory / "failure.json").exists()
            and not (directory / "resource-watchdog-failure.json").exists()
            and report.get("schema") == SCHEMA and report.get("completed") is True
            and report.get("mode") == "probe" and report.get("shard") is None
            and report.get("rows") == PROBES and report.get("qualified") is True
            and report.get("producer_sha256") == sha256_file(Path(__file__))
            and report.get("generator_sha256") == STATIC_HASHES[GENERATOR]
            and isinstance(report.get("projected_shard_seconds"), list)
            and len(report["projected_shard_seconds"]) == SHARD_COUNT
            and all(value <= LIMITS["whole_seconds"]
                    for value in report["projected_shard_seconds"])
            and isinstance(report.get("projected_shard_output_bytes"), list)
            and len(report["projected_shard_output_bytes"]) == SHARD_COUNT
            and all(value <= DISK_LIMITS["per_shard_output_bytes"]
                    for value in report["projected_shard_output_bytes"])
            and report.get("projected_total_output_bytes")
                == sum(report["projected_shard_output_bytes"])
            and report["projected_total_output_bytes"]
                <= DISK_LIMITS["all_shards_output_bytes"]
            and closure.get("capture_sha256") == sha256_file(directory / "capture.jsonl")
            and closure.get("rows") == PROBES and closure.get("labels_opened") is False
            and report.get("capture_sha256") == closure.get("capture_sha256")
            and report.get("closure_sha256") == sha256_file(directory / "capture.complete.json")
            and proof.get("verified") is True and proof.get("mode") == "probe"
            and proof.get("qualified") is True
            and proof.get("report_sha256") == expected_report_sha
            and proof.get("capture_sha256") == report.get("capture_sha256")
            and completion.get("completed") is True
            and completion.get("report_sha256") == expected_report_sha,
            "qualified probe artifact differs")
    pins = {str((directory / name).resolve()): sha256_file(directory / name)
            for name in ("report.json", "capture.jsonl", "capture.complete.json",
                         "completion.json", "independent-verification.json")}
    return report, pins


def selected_source_rows(mode, shard, groups, shards, row_map):
    wanted = ([group["probe_id"] for group in groups]
              if mode == "probe" else shards[shard])
    if mode == "probe":
        positions = {identifier: index for index, identifier in enumerate(wanted)}
        found = {}
        for source_shard in range(SHARD_COUNT):
            for source_row_index, original in enumerate(
                    iter_jsonl(SHARDS[source_shard] / "capture.jsonl")):
                identifier = original.get("id")
                if identifier in positions:
                    require(identifier not in found, "duplicate selected source row")
                    found[identifier] = (original, row_map[identifier], source_shard,
                                         source_row_index,
                                         SHARD_HASHES[source_shard]["capture.jsonl"])
        require(set(found) == set(wanted), "selected probe source rows missing")
        return wanted, (found[identifier] for identifier in wanted)

    def stream_shard():
        iterator = iter_jsonl(SHARDS[shard] / "capture.jsonl")
        for source_row_index, identifier in enumerate(wanted):
            original = next(iterator, None)
            require(original is not None and original.get("id") == identifier,
                    "capture source shard order differs")
            yield (original, row_map[identifier], shard, source_row_index,
                   SHARD_HASHES[shard]["capture.jsonl"])
        require(next(iterator, None) is None, "capture source shard has extra rows")

    return wanted, stream_shard()


def _empty_counts():
    return Counter({"rows": 0, "ranked_rows": 0, "unranked_rows": 0,
                    "no_pool_rows": 0, "proposal_count": 0,
                    "global_limit_rows": 0, "multiple_segment_proposals": 0,
                    "complete_paths_rejected_as_existing_or_duplicate": 0,
                    "edge_count": 0, "edge_occurrences": 0, "seed_count": 0,
                    "exhausted_seeds": 0, "max_pops_seeds": 0,
                    "max_frontier_seeds": 0, "no_complete_path_seeds": 0,
                    "global_proposal_limit_seeds": 0})


def _add_counts(counts, row):
    diagnostics = row["diagnostics"]
    counts["rows"] += 1
    counts["ranked_rows"] += row["source_reference"]["ranked"]
    counts["unranked_rows"] += not row["source_reference"]["ranked"]
    counts["no_pool_rows"] += row["status"] == "no_pool"
    counts["proposal_count"] += len(row["proposals"])
    counts["global_limit_rows"] += diagnostics["global_limit_reached"]
    counts["multiple_segment_proposals"] += sum(
        len(proposal["segments"]) > 1 for proposal in row["proposals"])
    for key in ("complete_paths_rejected_as_existing_or_duplicate", "edge_count",
                "edge_occurrences", "seed_count"):
        counts[key] += diagnostics[key]
    terminations = Counter(seed["termination_reason"] for seed in diagnostics["seeds"])
    for reason in ("exhausted", "max_pops", "max_frontier", "no_complete_path",
                   "global_proposal_limit"):
        counts[reason + "_seeds"] += terminations[reason]


def run(output, mode, shard=None, probe=None, probe_report_sha256=None):
    require(mode in ("probe", "capture")
            and ((mode == "probe" and shard is None and probe is None
                  and probe_report_sha256 is None)
                 or (mode == "capture" and type(shard) is int and 0 <= shard < 6
                     and probe is not None and probe_report_sha256 is not None)),
            "mode/shard/probe arguments differ")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        with AtomicResourceWatchdog(started, system_swap_bytes(),
                                    output / "resources.jsonl", limits=LIMITS,
                                    interval_s=0.5) as guard:
            with guard.operation("authenticate", update=True):
                rows, row_map, groups, shards, pins = authenticate_inputs()
                probe_report = None
                if mode == "capture":
                    probe_report, probe_pins = authenticate_probe(
                        Path(probe), probe_report_sha256)
                    pins.update(probe_pins)
                write_json(output / "pins.json", pins)
                ids, selected = selected_source_rows(mode, shard, groups, shards, row_map)
            initialization = time.monotonic() - started
            counts, reasons, timings = _empty_counts(), Counter(), {}
            output_bytes, total_output_bytes = {}, 0
            with (output / "capture.jsonl").open("x", encoding="utf-8") as stream:
                for index, values in enumerate(selected):
                    original, expected, source_shard, source_row_index, capture_sha = values
                    identifier = expected["id"]
                    tick = time.monotonic()
                    with guard.operation("propose:" + identifier, update=True):
                        record = prepare_record(original, expected, source_shard,
                                                source_row_index, capture_sha)
                        written = append_jsonl(
                            stream, record, current_bytes=total_output_bytes,
                            limit_bytes=DISK_LIMITS["per_shard_output_bytes"])
                        total_output_bytes += written
                    timings[identifier] = time.monotonic() - tick
                    output_bytes[identifier] = written
                    _add_counts(counts, record)
                    reasons.update(seed["termination_reason"]
                                   for seed in record["diagnostics"]["seeds"])
                    if mode == "probe" or (index + 1) % 25 == 0:
                        print(json.dumps({"captured": index + 1, "total": len(ids),
                                          "elapsed_s": time.monotonic() - started}),
                              flush=True)
                stream.flush(); os.fsync(stream.fileno())
            capture_sha = sha256_file(output / "capture.jsonl")
            with guard.operation("write_report", update=True):
                write_json(output / "capture.complete.json", {
                    "schema": CLOSURE_SCHEMA, "completed": True,
                    "mode": mode, "shard": shard, "rows": len(ids),
                    "capture_sha256": capture_sha, "labels_opened": False,
                    "model_loaded": False, "model_inference": False})
                projections = (projected_shard_seconds(groups, shards, timings,
                                                       initialization)
                               if mode == "probe"
                               else probe_report["projected_shard_seconds"])
                output_projections = (
                    projected_shard_output_bytes(groups, shards, output_bytes)
                    if mode == "probe"
                    else probe_report["projected_shard_output_bytes"])
                projected_total_output = sum(output_projections)
                qualified = all(value <= LIMITS["whole_seconds"]
                                for value in projections) and all(
                                    value <= DISK_LIMITS["per_shard_output_bytes"]
                                    for value in output_projections) and (
                                    projected_total_output
                                    <= DISK_LIMITS["all_shards_output_bytes"])
                require(qualified,
                        "probe exceeded time, per-shard disk, or total disk budget")
                report = {
                    "schema": SCHEMA, "completed": True, "mode": mode,
                    "shard": shard, "rows": len(ids), "ids": ids,
                    "counts": dict(counts),
                    "seed_termination_reasons": dict(sorted(reasons.items())),
                    "timing_seconds": timings, "initialization_seconds": initialization,
                    "projected_shard_seconds": projections, "qualified": qualified,
                    "serialized_output_bytes": output_bytes,
                    "capture_output_bytes": total_output_bytes,
                    "projected_shard_output_bytes": output_projections,
                    "projected_total_output_bytes": projected_total_output,
                    "limits": LIMITS, "disk_limits": DISK_LIMITS,
                    "pins": dict(sorted(pins.items())),
                    "capture_sha256": capture_sha,
                    "closure_sha256": sha256_file(output / "capture.complete.json"),
                    "producer_sha256": sha256_file(Path(__file__)),
                    "generator_sha256": STATIC_HASHES[GENERATOR],
                    "generator_schema": GENERATOR_SCHEMA,
                    "generator_limits": {"max_proposals": MAX_PROPOSALS,
                                         "max_pops_per_seed": MAX_POPS_PER_SEED,
                                         "max_frontier_per_seed": MAX_FRONTIER_PER_SEED,
                                         "max_seeds": MAX_SEEDS},
                    "probe_binding": (None if mode == "probe" else {
                        "directory": str(Path(probe).resolve()),
                        "report_sha256": probe_report_sha256,
                        "capture_sha256": probe_report["capture_sha256"],
                        "closure_sha256": probe_report["closure_sha256"],
                        "completion_sha256": sha256_file(Path(probe) / "completion.json"),
                        "proof_sha256": sha256_file(
                            Path(probe) / "independent-verification.json")}),
                    "source_only": True, "train_only": True,
                    "labels_opened": False, "fit_feature_files_opened": False,
                    "dev_opened": False, "test_opened": False,
                    "model_loaded": False, "model_inference": False,
                    "neural_training": False, "production_changed": False,
                }
                write_json(output / "report.json", report)
            terminal = guard.sample_and_append("after_report")
            with guard.operation("write_completion", update=True):
                require(all(sha256_file(path) == digest for path, digest in pins.items()),
                        "final pinned source changed")
                write_json(output / "completion.json", {
                    "completed": True,
                    "report_sha256": sha256_file(output / "report.json"),
                    "resource_after_report": terminal})
            guard.sample_and_append("after_completion")
        return report
    except BaseException as error:
        failure = output / "failure.json"
        if not failure.exists():
            write_json(failure, {"completed": False,
                                 "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("probe", "capture"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard", type=int, choices=range(6))
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--probe-report-sha256")
    arguments = parser.parse_args()
    result = run(arguments.output, arguments.mode, arguments.shard,
                 arguments.probe, arguments.probe_report_sha256)
    print(json.dumps({key: result[key] for key in
                      ("completed", "mode", "shard", "rows", "qualified")},
                     ensure_ascii=False))
