"""Capture a frozen document shard without opening sentence references.

The baseline is reconstructed from the actual pre-calibration candidate paths.
It is not a second parent execution or a measurement of parent latency.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.preflight_document_phonetic_choice import (
    authorize as authorize_inputs, load_engines, LIMITS, PRODUCTION_VERSION,
    verify_capture,
)
from scripts.evaluate_extended_runtime import (
    require, path_key, sha256_file, verify_pins, write_json, iter_jsonl,
)
from scripts.evaluate_phonetic_choice_dev import render_public
from scripts.evaluate_sentence_only import predict_sentence
from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes

SCHEMA = "document-phonetic-shard-capture-v1"
PLAN = ROOT / ".omx/plans/document-phonetic-choice-sharded-v1.md"
SHARDS = ROOT / ".omx/artifacts/document-sentence-shards-v1"
PREFLIGHT = ROOT / ".omx/artifacts/document-phonetic-choice-preflight-v1"
VERIFIER = ROOT / "scripts/verify_document_phonetic_shard.py"
PREFLIGHT_PROOF_SHA = "46b678cf26b1768f3620b3220f06cb35ab554f8523b3c5940f5af78806e4809f"
# Filled only after source-only partition and independent verification close.
SHARD_HASHES = {
    "manifest.json": "a09e13062f759ec8ac8d13b44344b63724e50f8d0302e4a945bfa13fdf797fc9",
    "shard-0.jsonl": "9c9e7f49f95d90f38d530cb0bc879eefe0be98c2fc9593e4aa24500129ffae48",
    "shard-1.jsonl": "3aeecc85a58186a4bbb9f0ac45edd52038488f31bc01b9749bb92aaddad04df5",
    "independent-verification.json": "f0eb4e1d4585b5c2d4873442186b896ad47f55c49324aa6c599f1cf3143376a3",
}
BASELINE_KIND = "public_replay_of_actual_precalibration_paths"


def baseline_public(source, response, paths):
    if response.get("sentence_ranking_applied") is True:
        require(bool(paths), "ranked response has no pre-calibration paths")
        candidates = render_public(source, paths)
    else:
        require(not paths and not response["phonetic_choice"]["rows"],
                "unranked response has calibration paths")
        candidates = deepcopy(response["candidates"])
    return {"kind": BASELINE_KIND, "version": PRODUCTION_VERSION,
            "candidates": candidates, "result": candidates[0]["text"],
            "sentence_ranking_applied": response["sentence_ranking_applied"]}


def capture_record(identifier, source, response, paths, weights):
    baseline = baseline_public(source, response, paths)
    # A private compatibility view lets the established assertions validate
    # calibration. It is never emitted as an independently executed response.
    proxy = dict(response, version=PRODUCTION_VERSION,
                 candidates=baseline["candidates"], result=baseline["result"])
    verify_capture(source, proxy, response, paths, weights)
    return {"id": identifier, "request": {"input": source, "context": ""},
            "baseline": baseline, "proposed": response, "base_paths": paths}


def authenticate_previous(previous, shard):
    if shard == 0:
        require(previous is None, "first shard must not have a predecessor")
        return {}
    require(shard == 1 and previous is not None,
            "second shard requires verified terminal first shard")
    previous = Path(previous).resolve()
    require(not any((previous / name).exists() for name in
                    ("failure.json", "resource-watchdog-failure.json")),
            "predecessor has failure evidence")
    report = json.loads((previous / "report.json").read_text())
    proof = json.loads((previous / "independent-verification.json").read_text())
    completion = json.loads((previous / "completion.json").read_text())
    launch = json.loads(previous.with_suffix(".launch.json").read_text())
    pid = launch["pid"]
    require(type(pid) is int and pid > 0, "predecessor PID missing")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError("predecessor process is still present")
    require(report.get("completed") is True and report.get("shard") == 0
            and report.get("schema") == SCHEMA
            and proof.get("verified") is True
            and proof.get("schema") == "document-phonetic-shard-independent-verification-v1"
            and completion.get("completed") is True
            and proof.get("report_sha256") == completion.get("report_sha256")
            == sha256_file(previous / "report.json"), "predecessor verification differs")
    for name, field in (("capture.jsonl", "capture_sha256"),
                        ("capture.complete.json", "closure_sha256"),
                        ("completion.json", "completion_sha256")):
        require(proof.get(field) == sha256_file(previous / name),
                "predecessor proof artifact differs")
    require(report.get("partition_hashes") == SHARD_HASHES,
            "predecessor partition differs")
    require(proof.get("verifier_sha256") == sha256_file(VERIFIER)
            == report["pins"].get(path_key(VERIFIER))
            and proof.get("shard") == 0
            and proof.get("launch_sha256") == sha256_file(previous.with_suffix(".launch.json"))
            and launch.get("runner_sha256") == sha256_file(Path(__file__))
            and launch.get("output") == str(previous)
            and launch.get("command") == [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()),
                "--output", str(previous), "--shard", "0"],
            "predecessor verifier or launch identity differs")
    require(report.get("limits") == LIMITS
            and report.get("performance_labels_opened") is False
            and report.get("accuracy_measured") is False
            and report["pins"].get(path_key(Path(__file__))) == sha256_file(Path(__file__)),
            "predecessor execution contract differs")
    verify_pins(report["pins"])
    from scripts.verify_production_phonetic_choice_dev import verify_resources
    previous_requests = list(iter_jsonl(SHARDS / "shard-0.jsonl"))
    require(report["rows"] == proof.get("rows") == len(previous_requests), "predecessor row count differs")
    with (previous / "capture.jsonl").open(encoding="utf-8") as stream:
        for request in previous_requests:
            line = stream.readline()
            require(bool(line), "predecessor capture truncated")
            captured = json.loads(line)
            require(captured.get("id") == request["id"]
                    and captured.get("request") == {"input": request["input"], "context": ""},
                    "predecessor source sequence differs")
        require(not stream.read().strip(), "predecessor capture has extra rows")
    verify_resources(list(iter_jsonl(previous / "resources.jsonl")),
        ["authenticate_sources", "load_models",
         *("predict:" + row["id"] for row in previous_requests),
         "close_capture", "write_report", "write_completion"])
    return {path_key(path): sha256_file(path) for path in
            [*(previous / name for name in ("report.json", "completion.json",
               "independent-verification.json", "capture.complete.json", "capture.jsonl",
               "resources.jsonl")),
             previous.with_suffix(".launch.json")]}


def authorize(shard, previous):
    require(type(shard) is int and shard in (0, 1), "invalid shard index")
    require(set(SHARD_HASHES) == {"manifest.json", "shard-0.jsonl", "shard-1.jsonl",
                                "independent-verification.json"},
            "partition has not been independently frozen")
    for name, digest in SHARD_HASHES.items():
        require(sha256_file(SHARDS / name) == digest, "partition identity differs")
    require(sha256_file(PREFLIGHT / "independent-verification.json") == PREFLIGHT_PROOF_SHA,
            "preflight proof differs")
    proof = json.loads((SHARDS / "independent-verification.json").read_text())
    manifest = json.loads((SHARDS / "manifest.json").read_text())
    require(proof.get("verified") is True, "partition verification incomplete")
    all_requests, version, weights, pins = authorize_inputs()
    previous_pins = authenticate_previous(previous, shard)
    meta = manifest["shards"][shard]
    require(meta["index"] == shard and meta["file"] == f"shard-{shard}.jsonl"
            and meta["qualified"] is True
            and meta["projected_seconds"] <= LIMITS["whole_seconds"],
            "shard resource qualification differs")
    requests = list(iter_jsonl(SHARDS / meta["file"]))
    source_by_id = {r["id"]: r for r in all_requests}
    require(len(requests) == meta["rows"] and requests
            and len({r["id"] for r in requests}) == len(requests)
            and all(r == source_by_id.get(r["id"]) for r in requests),
            "shard source request differs")
    for path in (Path(__file__), PLAN, ROOT / "scripts/preflight_document_phonetic_choice.py",
                 ROOT / "scripts/prepare_document_sentence_shards.py",
                 ROOT / "scripts/verify_document_sentence_shards.py",
                 ROOT / "scripts/verify_production_phonetic_choice_dev.py", VERIFIER):
        pins[path_key(path)] = sha256_file(path)
    for name, digest in SHARD_HASHES.items():
        pins[path_key(SHARDS / name)] = digest
    pins[path_key(PREFLIGHT / "independent-verification.json")] = PREFLIGHT_PROOF_SHA
    pins.update(previous_pins)
    verify_pins(pins)
    return requests, version, weights, pins


def run(output, shard, previous=None):
    started = time.monotonic()
    baseline_swap = system_swap_bytes()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    count = 0
    try:
        with AtomicResourceWatchdog(started, baseline_swap, output / "resources.jsonl",
                                    limits=LIMITS, interval_s=1.) as watchdog:
            with watchdog.operation("authenticate_sources", update=True):
                requests, version, weights, pins = authorize(shard, previous)
                write_json(output / "pins.json", pins)
            with watchdog.operation("load_models", update=True):
                parent, zero, learned = load_engines(version)
                for engine in (parent, zero, learned):
                    for key, digest in engine.model_artifacts.items():
                        path = Path(key) if Path(key).is_absolute() else ROOT / key
                        require(pins.get(path_key(path)) == digest, "unbound model artifact")
                # Shared immutable model loading is reused from the preflight;
                # only the learned adapter is invoked on this shard.
            with (output / "capture.jsonl").open("x", encoding="utf-8") as stream:
                for request in requests:
                    with watchdog.operation("predict:" + request["id"], update=True):
                        response = predict_sentence(learned, request["input"])
                        paths = deepcopy(learned.choice_base_paths)
                        record = capture_record(request["id"], request["input"],
                                                response, paths, weights)
                        stream.write(json.dumps(record, ensure_ascii=False,
                                                allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        count += 1
                    if count % 25 == 0 or count == len(requests):
                        print(json.dumps({"shard": shard, "captured": count,
                            "total": len(requests), "elapsed_s": time.monotonic() - started}),
                            flush=True)
            with watchdog.operation("close_capture", update=True):
                capture_sha = sha256_file(output / "capture.jsonl")
                write_json(output / "capture.complete.json", {
                    "schema": SCHEMA + "-closure", "shard": shard, "rows": count,
                    "capture_sha256": capture_sha, "performance_labels_opened": False})
                verify_pins(pins)
            report = {"schema": SCHEMA, "completed": True, "shard": shard,
                "rows": count, "version": version, "weights": weights,
                "partition_hashes": SHARD_HASHES, "pins": pins, "limits": LIMITS,
                "capture_sha256": capture_sha,
                "closure_sha256": sha256_file(output / "capture.complete.json"),
                "comparison": BASELINE_KIND, "parent_inference_run": False,
                "baseline_latency_measured": False, "performance_labels_opened": False,
                "accuracy_measured": False, "production_changed": False}
            with watchdog.operation("write_report", update=True):
                write_json(output / "report.json", report)
            terminal = watchdog.sample_and_append("after_report")
            with watchdog.operation("write_completion", update=True):
                write_json(output / "completion.json", {"completed": True,
                    "report_sha256": sha256_file(output / "report.json"),
                    "resource_after_report": terminal})
            watchdog.sample_and_append("after_completion")
        return {"completed": True, "shard": shard, "rows": count}
    except BaseException as error:
        write_json(output / "failure.json", {"completed": False, "shard": shard,
            "captured": count, "error": f"{type(error).__name__}: {error}",
            "performance_labels_opened": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shard", required=True, choices=(0, 1), type=int)
    parser.add_argument("--previous", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.shard, args.previous)))
