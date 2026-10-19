"""Source-only resource/parity probe before whole document sentence evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_extended_runtime import (
    require, path_key, sha256_file, verify_pins, write_json, iter_jsonl,
)
from scripts.evaluate_production_phonetic_choice_test import (
    CHECKPOINT, WEIGHTS, LIMITS, PRODUCTION_VERSION, verify_capture,
)
from scripts.evaluate_sentence_only import predict_sentence
from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes

SCHEMA = "document-phonetic-choice-preflight-v1"
PLAN = ROOT / ".omx/plans/document-phonetic-choice-preflight-v1.md"
INPUTS = ROOT / ".omx/artifacts/document-sentence-inputs-v1"
TEST_RUN = ROOT / ".omx/artifacts/production-phonetic-choice-test-v1"
TEST_HASHES = {
    "report.json": "e8592ca1435c8ea1da37e7ec63a6ee3ad71ecc3bf2951541b5b791158de8360b",
    "completion.json": "b8a29e16601b5f533ea1f83dd21b68220ea08d2427272211639d1438f9be31e7",
    "independent-verification.json": "0ce8bc781d55ce2ac159447ad3e17082c331f17527ad3aa819cba7ab0865212c",
}
INPUT_HASHES = {
    "manifest.json": "acea9e1d4d4929253f0d4adfc2aea8ba7c877cf5093b0499961619a35a7632a3",
    "inputs.jsonl": "536707f05503856f4f805bb9775bd9c5dc932c94783e669a5d62fe3023b0b1db",
    "independent-verification.json": "c12e49e6f735da775e4c996ef0bea0ecdc699cb156d31bdf8ceef01b65d3935e",
}
HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")


def byte_probe_groups(requests):
    require(len(requests) >= 16, "too few source requests")
    ordered = sorted(requests, key=lambda row: (len(row["input"].encode("utf-8")), row["id"]))
    small, extra = divmod(len(ordered), 16)
    groups, offset = [], 0
    for index in range(16):
        count = small + int(index < extra)
        group = ordered[offset:offset + count]
        groups.append({"ids": [r["id"] for r in group], "probe_id": group[-1]["id"],
                       "count": count, "max_utf8_bytes": len(group[-1]["input"].encode("utf-8"))})
        offset += count
    return groups


def projection(elapsed, timings):
    paired = 1.3 * (elapsed + sum((row["parent_seconds"] + row["learned_seconds"])
                                  * row["group_size"] for row in timings)) + 60.
    shared = 1.3 * (elapsed + sum(row["learned_seconds"] * row["group_size"]
                                  for row in timings)) + 60.
    return {"projected_paired_seconds": paired, "projected_shared_seconds": shared,
            "paired_qualified": paired <= LIMITS["whole_seconds"],
            "shared_qualified": shared <= LIMITS["whole_seconds"]}


def authorize():
    require(set(INPUT_HASHES) == {"manifest.json", "inputs.jsonl", "independent-verification.json"},
            "document input identities not yet frozen")
    for folder, expected in ((TEST_RUN, TEST_HASHES), (INPUTS, INPUT_HASHES)):
        require(all(sha256_file(folder / name) == digest for name, digest in expected.items()),
                "qualified artifact identity changed")
    test = json.loads((TEST_RUN / "report.json").read_text())
    proof = json.loads((TEST_RUN / "independent-verification.json").read_text())
    require(test["completed"] is True and test["quality_gate"] is True
            and proof["verified"] is True
            and proof["report_sha256"] == TEST_HASHES["report.json"],
            "TEST regression qualification differs")
    manifest = json.loads((INPUTS / "manifest.json").read_text())
    input_proof = json.loads((INPUTS / "independent-verification.json").read_text())
    require(manifest["completed"] is True and input_proof["verified"] is True,
            "document input verification incomplete")
    requests = list(iter_jsonl(INPUTS / "inputs.jsonl"))
    require(len(requests) == 1046 and len({r["id"] for r in requests}) == 1046,
            "document input denominator differs")
    require(all(set(r) == {"id", "input"} and isinstance(r["id"], str)
                and isinstance(r["input"], str) and r["input"]
                and HANGUL.search(r["input"]) is None for r in requests),
            "source-only document input contract differs")
    pins = dict(test["pins"])
    for raw, digest in manifest["source_hashes"].items():
        path = Path(raw) if Path(raw).is_absolute() else ROOT / raw
        key = path_key(path)
        require(key not in pins or pins[key] == digest, "document source pin disagreement")
        pins[key] = digest
    # Hash authentication does not deserialize target-bearing corpora.
    for folder, expected in ((TEST_RUN, TEST_HASHES), (INPUTS, INPUT_HASHES)):
        for name, digest in expected.items():
            pins[path_key(folder / name)] = digest
    for path in (Path(__file__), PLAN, ROOT / "scripts/evaluate_production_phonetic_choice_test.py"):
        pins[path_key(path)] = sha256_file(path)
    verify_pins(pins)
    return requests, test["version"], test["weights"], pins


def load_engines(version):
    import torch
    from hanglish.core.ctc_union_converter import CTCUnionConverter
    from hanglish.core.production_phonetic_choice_converter import ProductionPhoneticChoiceConverter
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    class TracedProduction(CTCUnionConverter):
        def _extend_sentence_paths(self, text, context, paths):
            result = super()._extend_sentence_paths(text, context, paths)
            self.full_paths = deepcopy(result)
            return result

    parent = TracedProduction()
    parent.load_models(); parent.load_word_model(); parent.load_growth_model()
    require(parent.version == PRODUCTION_VERSION, "production version differs")

    def make(override, model=None):
        value = ProductionPhoneticChoiceConverter(phonetic_checkpoint=CHECKPOINT,
            calibration_checkpoint=WEIGHTS, calibration_weights=override,
            external_lexicon=False, word_model=parent.word_model,
            growth_model=parent.growth_model, phonetic_model=model)
        shared = ("generator", "numeric_units", "ranker", "sentence_ranker", "transducer")
        for name in shared:
            setattr(value, name, getattr(parent, name))
        value.load_models(); value.load_phonetic_model()
        require(value.choice_parent_version == parent.version
                and all(getattr(value, name) is getattr(parent, name)
                        for name in (*shared, "word_model", "growth_model")),
                "frozen model sharing differs")
        return value
    learned = make(None); zero = make([0.] * 4, learned.phonetic_model)
    require(learned.version == version and zero.phonetic_model is learned.phonetic_model,
            "qualified calibration identity differs")
    return parent, zero, learned


def run(output):
    started = time.monotonic(); baseline = system_swap_bytes()
    output.mkdir(parents=True, exist_ok=False)
    timings = []
    try:
        with AtomicResourceWatchdog(started, baseline, output / "resources.jsonl",
                                    limits=LIMITS, interval_s=1.) as watchdog:
            with watchdog.operation("authenticate_source_inputs", update=True):
                requests, version, weights, pins = authorize()
                groups = byte_probe_groups(requests)
                write_json(output / "pins.json", pins)
                write_json(output / "schedule.json", {"groups": groups,
                    "selection": "16 balanced UTF-8-length groups, longest source then ID"})
            with watchdog.operation("load_models", update=True):
                parent, zero, learned = load_engines(version)
                for engine in (parent, zero, learned):
                    for key, digest in engine.model_artifacts.items():
                        path = Path(key) if Path(key).is_absolute() else ROOT / key
                        require(pins.get(path_key(path)) == digest, "unbound model artifact")
            sources = {row["id"]: row["input"] for row in requests}
            with (output / "capture.jsonl").open("x") as stream:
                for group in groups:
                    identifier = group["probe_id"]; source = sources[identifier]
                    tick = time.monotonic()
                    with watchdog.operation("parent:" + identifier, update=True):
                        parent.full_paths = []
                        before = predict_sentence(parent, source)
                    parent_seconds = time.monotonic() - tick
                    with watchdog.operation("zero:" + identifier, update=True):
                        unchanged = predict_sentence(zero, source)
                        verify_capture(source, before, unchanged, zero.choice_base_paths, [0.] * 4)
                        require(parent.full_paths == zero.choice_base_paths
                                and before["candidates"] == unchanged["candidates"],
                                "actual zero parity differs")
                    tick = time.monotonic()
                    with watchdog.operation("learned:" + identifier, update=True):
                        after = predict_sentence(learned, source)
                        verify_capture(source, before, after, learned.choice_base_paths, weights)
                        require(parent.full_paths == learned.choice_base_paths,
                                "calibrated base paths differ from actual parent")
                        record = {"id": identifier, "request": {"input": source, "context": ""},
                            "parent": before, "zero": unchanged, "proposed": after,
                            "parent_paths": parent.full_paths, "zero_base_paths": zero.choice_base_paths,
                            "base_paths": learned.choice_base_paths}
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush(); os.fsync(stream.fileno())
                    timings.append({"id": identifier, "group_size": group["count"],
                        "parent_seconds": parent_seconds,
                        "learned_seconds": time.monotonic() - tick,
                        "parent_engine_ms": before["engine_ms"],
                        "zero_engine_ms": unchanged["engine_ms"],
                        "learned_engine_ms": after["engine_ms"]})
                    print(json.dumps({"captured": len(timings), "total": 16,
                                      "elapsed_s": time.monotonic() - started}), flush=True)
            with watchdog.operation("close_capture", update=True):
                capture_sha = sha256_file(output / "capture.jsonl")
                write_json(output / "capture.complete.json", {"schema": SCHEMA + "-closure",
                    "rows": 16, "capture_sha256": capture_sha, "performance_labels_opened": False})
                verify_pins(pins)
            elapsed = time.monotonic() - started
            report = {"schema": SCHEMA, "completed": True, "pins": pins, "limits": LIMITS,
                "input_rows": 1046, "probe_rows": 16, "elapsed_seconds_for_projection": elapsed,
                "timings": timings, **projection(elapsed, timings),
                "capture_sha256": capture_sha,
                "closure_sha256": sha256_file(output / "capture.complete.json"),
                "schedule_sha256": sha256_file(output / "schedule.json"),
                "weights": weights, "version": version, "production_changed": False,
                "performance_labels_opened": False, "accuracy_measured": False,
                "timing_scope": "16 fresh serial parent/zero/learned probes sharing frozen models; length-based projection is not an upper bound"}
            with watchdog.operation("write_report", update=True):
                write_json(output / "report.json", report)
            terminal = watchdog.sample_and_append("after_report")
            with watchdog.operation("write_completion", update=True):
                write_json(output / "completion.json", {"completed": True,
                    "report_sha256": sha256_file(output / "report.json"), "resource_after_report": terminal})
            watchdog.sample_and_append("after_completion")
        return report
    except BaseException as error:
        write_json(output / "failure.json", {"completed": False,
            "error": f"{type(error).__name__}: {error}", "captured": len(timings),
            "performance_labels_opened": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    value = run(parser.parse_args().output)
    print(json.dumps({key: value[key] for key in ("completed", "paired_qualified", "shared_qualified")}))
