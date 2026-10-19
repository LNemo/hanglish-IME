"""Actual source-only DEV evaluation of frozen phonetic choice calibration."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_extended_runtime import (
    load_reference_authorization, validate_source_reference, verify_pins,
    iter_jsonl, write_json, path_key, sha256_file, require,
)
from scripts.evaluate_ctc_word_runtime import latency, verify_runtime_contract
from scripts.evaluate_sentence_only import predict_sentence
from scripts.evaluate_phonetic_word_union_dev import aggregate, CHECKPOINT, CHECKPOINT_SHA
from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes

SCHEMA = "phonetic-choice-runtime-dev-v1"
FROZEN_UNION = ROOT / ".omx/artifacts/phonetic-word-union-dev-v1"
CALIBRATION = ROOT / ".omx/artifacts/phonetic-choice-calibration-training-v1"
PLAN = ROOT / ".omx/plans/phonetic-choice-runtime-dev-v1.md"
LIMITS = {"whole_seconds": 1800., "update_seconds": 60.,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
FROZEN = {
    FROZEN_UNION / "report.json": "a87575bc2080fbe649029e59ce55c38131096a806b0b99203d40b0bfceeec368",
    FROZEN_UNION / "capture.jsonl": "e8d45820b7edb691bf18f011f63fe0b4a0577a387c4e59d0fdf584e36bcb6131",
    FROZEN_UNION / "independent-verification.json": "82e99e5799ca005d17442b9cb90e62c8e98e846d369b27f648fd0184fae6df94",
    CALIBRATION / "weights.json": "930e3ef942061e6d459d7527c0e073fc6ad5626d59f5170e9b11df8499b39f4b",
    CALIBRATION / "report.json": "b692262a2fcd8c525d543ca58aa86218948a5a04f5ab60b2afb4ebcb0c7143a6",
    CALIBRATION / "independent-verification.json": "7181d0ca4e783c0cf40e10ed28f26d4e2401e49829f5aade13f2123d5da06a83",
    CHECKPOINT: CHECKPOINT_SHA,
}


def render_public(source, paths):
    """Independently reproduce NeuralConverter's public truncation/raw slot."""
    require(bool(paths), "public rendering requires at least one path")
    candidates = [{"text": text, "score": round(score, 4), "segments": segments}
                  for text, score, segments in paths[:5]]
    if not any(row["text"] == source for row in candidates):
        candidates = candidates[:4] + [{"text": source, "score": None, "segments": [{
            "input_start": 0, "input_end": len(source), "source": source,
            "text": source, "kind": "raw"}]}]
    return candidates


def probe_groups(requests):
    ordered = sorted(requests, key=lambda row: (len(row[1]), row[0]))
    require(len(ordered) >= 16, "insufficient requests for fixed probe")
    small, extra = divmod(len(ordered), 16)
    groups = []
    offset = 0
    for i in range(16):
        count = small + int(i < extra)
        group = ordered[offset:offset + count]
        groups.append({"ids": [r[0] for r in group], "probe_id": group[-1][0],
                       "count": count, "maximum_source_length": len(group[-1][1])})
        offset += count
    return groups


def without_timing(value):
    return {key: item for key, item in value.items() if key != "generation_ms"}


def verify_capture(source, frozen, response, base_paths, weights):
    before = frozen["proposed"]
    verify_runtime_contract(source, before, response)
    require(response["context"] == "" and response["external_context_used"] is False,
            "external context crossed prediction boundary")
    for name in ("ctc_word", "ctc_union"):
        require(without_timing(before[name]) == without_timing(response[name]),
                f"calibration changed {name} generation")
    old_trace, trace = before["phonetic_word_union"], response["phonetic_word_union"]
    for name in old_trace:
        if name not in {"generation_ms", "public", "selected_extras", "missing_extras"}:
            require(trace[name] == old_trace[name], f"frozen union changed: {name}")
    choice = response["phonetic_choice"]
    require(choice["effective_weights"] == list(weights), "effective weights differ")
    if response["sentence_ranking_applied"]:
        require(bool(base_paths) and len({p[0] for p in base_paths}) == len(base_paths),
                "complete base pool is missing or duplicated")
        require(base_paths == sorted(base_paths, key=lambda p: (-p[1], p[0])),
                "base pool order differs")
        require([p[0] for p in base_paths] == old_trace["union"], "full union text order differs")
        require(render_public(source, base_paths) == before["candidates"],
                "shadow zero public candidates differ from frozen union")
        rows = choice["rows"]
        require(len(rows) == len(base_paths), "calibration candidate denominator differs")
        calibrated = []
        for path, row in zip(base_paths, rows):
            text, score, segments = path
            require(row["text"] == text and row["base_score"] == score
                    and len(row["features"]) == 4
                    and all(math.isfinite(v) for v in row["features"]),
                    "calibration candidate/feature correspondence differs")
            zero = score + sum(0. * v for v in row["features"])
            expected = score + sum(w * v for w, v in zip(weights, row["features"]))
            require(zero == score and row["adjusted_score"] == expected,
                    "calibration or zero-weight score differs")
            calibrated.append((text, expected, segments))
        require(choice["zero_weight_parity"] is True, "zero-weight trace parity failed")
        calibrated.sort(key=lambda p: (-p[1], p[0]))
        require(render_public(source, calibrated) == response["candidates"],
                "actual calibrated public output differs from full path scores")
    else:
        require(not base_paths and not choice["rows"], "unranked input was calibrated")
        require(response["candidates"] == before["candidates"], "unranked candidates changed")
    require(response["result"] == response["candidates"][0]["text"], "selected result differs")


def authorize():
    require(all(sha256_file(path) == value for path, value in FROZEN.items()),
            "frozen experimental artifact changed")
    authorization = load_reference_authorization("dev")
    reference_rows = list(iter_jsonl(authorization["paths"]["capture"]))
    seen = set()
    requests = [validate_source_reference(row, seen) for row in reference_rows]
    frozen = list(iter_jsonl(FROZEN_UNION / "capture.jsonl"))
    require(len(requests) == len(frozen) == 316, "DEV denominator differs")
    for (identifier, source, production), row in zip(requests, frozen):
        require(row["id"] == identifier and row["request"] == {"input": source, "context": ""}
                and row["parent"]["result"] == production["result"]
                and row["parent"]["candidates"] == production["candidates"],
                "frozen source or production identity differs")
    union_report = json.loads((FROZEN_UNION / "report.json").read_text())
    training_report = json.loads((CALIBRATION / "report.json").read_text())
    for directory, report in ((FROZEN_UNION, union_report), (CALIBRATION, training_report)):
        proof = json.loads((directory / "independent-verification.json").read_text())
        require(proof["verified"] is True
                and proof["report_sha256"] == sha256_file(directory / "report.json")
                and report["completed"] is True, "independent parent proof differs")
    weights = json.loads((CALIBRATION / "weights.json").read_text())["weights"]
    pins = {path_key(path): value for path, value in FROZEN.items()}
    for records in (authorization["report"]["pins"], union_report["pins"], training_report["pins"]):
        for path, value in records.items():
            key = path_key(Path(path) if Path(path).is_absolute() else ROOT / path)
            require(key not in pins or pins[key] == value, "parent pin disagreement")
            pins[key] = value
    for path in (Path(__file__), PLAN, ROOT / "hanglish/core/phonetic_choice_converter.py",
                 ROOT / "hanglish/models/phonetic_choice_features.py",
                 ROOT / "hanglish/models/phonetic_likelihood.py",
                 ROOT / "scripts/source_span_resources.py"):
        key = path_key(path)
        require(key not in pins or pins[key] == sha256_file(path), "inherited pin differs")
        pins[key] = sha256_file(path)
    verify_pins(pins, defer=authorization["deferred_label"])
    return authorization, requests, {row["id"]: row for row in frozen}, weights, pins


def compare_all(captures, labels):
    production = aggregate(captures, labels)
    union_records = [{**r, "parent": r["union_reference"]} for r in captures]
    union = aggregate(union_records, labels)
    require(production["parent"]["counts"] == {"rows": 316, "top1": 262, "top5": 282, "errors": 0}
            and union["parent"]["counts"] == {"rows": 316, "top1": 273, "top5": 291, "errors": 0},
            "historical reference metrics differ")
    # Its stage counts describe production origins, so report them only once.
    union.pop("stage_counts")
    return {"versus_production": production, "versus_union": union,
            "quality_gate": production["quality_gate"] and union["quality_gate"]}


def run(output):
    started = time.monotonic(); baseline_swap = system_swap_bytes()
    output.mkdir(parents=True, exist_ok=False)
    captures = []
    try:
        with AtomicResourceWatchdog(started, baseline_swap, output / "resources.jsonl",
                                    limits=LIMITS, interval_s=1.) as watchdog:
            with watchdog.operation("authenticate_inputs", update=True):
                authorization, requests, frozen, weights, pins = authorize()
                groups = probe_groups(requests)
                write_json(output / "pins.json", pins)
                write_json(output / "probe-schedule.json", {"groups": groups,
                    "selection": "16 source-length balanced groups; longest source per group"})
            with watchdog.operation("load_models", update=True):
                import torch
                from hanglish.core.phonetic_word_union_converter import PhoneticWordUnionConverter
                from hanglish.core.phonetic_choice_converter import PhoneticChoiceConverter
                torch.set_num_threads(2); torch.set_num_interop_threads(1)
                torch.use_deterministic_algorithms(True)

                class TracedUnion(PhoneticWordUnionConverter):
                    def _extend_sentence_paths(self, text, context, paths):
                        result = super()._extend_sentence_paths(text, context, paths)
                        self.full_paths = deepcopy(result)
                        return result

                reference = TracedUnion(phonetic_checkpoint=CHECKPOINT)
                reference.load_models(); reference.load_word_model()
                reference.load_growth_model(); reference.load_phonetic_model()

                def make_adapter(override):
                    value = PhoneticChoiceConverter(phonetic_checkpoint=CHECKPOINT,
                        calibration_checkpoint=CALIBRATION / "weights.json", calibration_weights=override,
                        external_lexicon=False, word_model=reference.word_model,
                        growth_model=reference.growth_model, phonetic_model=reference.phonetic_model)
                    for name in ("generator", "numeric_units", "ranker", "sentence_ranker", "transducer"):
                        setattr(value, name, getattr(reference, name))
                    value.load_models()
                    require(value.choice_parent_version == reference.version, "loaded parent version differs")
                    require(all(getattr(value, name) is getattr(reference, name) for name in (
                        "generator", "ranker", "sentence_ranker", "transducer", "word_model",
                        "growth_model", "phonetic_model")), "shared frozen model objects differ")
                    return value

                zero = make_adapter([0., 0., 0., 0.])
                proposed = make_adapter(None)
                for engine in (reference, zero, proposed):
                    for key, value in engine.model_artifacts.items():
                        resolved = path_key(Path(key) if Path(key).is_absolute() else ROOT / key)
                        require(pins.get(resolved) == value, f"unbound runtime artifact: {key}")
            sources = {identifier: source for identifier, source, _ in requests}
            probes = []
            with (output / "probe.jsonl").open("x") as stream:
                for group in groups:
                    identifier = group["probe_id"]; source = sources[identifier]
                    with watchdog.operation("probe_union:" + identifier, update=True):
                        reference.full_paths = []
                        before = predict_sentence(reference, source)
                    tick = time.monotonic()
                    with watchdog.operation("probe_zero:" + identifier, update=True):
                        after = predict_sentence(zero, source)
                        verify_capture(source, frozen[identifier], after, zero.choice_base_paths, [0.] * 4)
                        require(reference.full_paths == zero.choice_base_paths
                                and before["candidates"] == after["candidates"]
                                and before["result"] == after["result"], "actual zero adapter parity failed")
                        record = {"id": identifier, "request": {"input": source, "context": ""},
                            "parent": before, "proposed": after, "base_paths": zero.choice_base_paths,
                            "exact_full_path_parity": True}
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush(); os.fsync(stream.fileno())
                    probes.append({"id": identifier, "group_size": group["count"],
                                   "zero_seconds": time.monotonic() - tick,
                                   "union_engine_ms": before["engine_ms"],
                                   "zero_engine_ms": after["engine_ms"]})
            projected = 1.3 * (time.monotonic() - started
                               + sum(p["zero_seconds"] * p["group_size"] for p in probes)) + 60.
            write_json(output / "qualification.json", {"qualified": projected <= LIMITS["whole_seconds"],
                "projected_seconds": projected, "probes": probes,
                "projection": "1.3*(elapsed+sum(zero_seconds*group_size))+60"})
            require(projected <= LIMITS["whole_seconds"], "actual DEV resource projection failed")
            with (output / "capture.jsonl").open("x") as stream:
                for identifier, source, _ in requests:
                    with watchdog.operation("convert:" + identifier, update=True):
                        response = predict_sentence(proposed, source)
                        verify_capture(source, frozen[identifier], response, proposed.choice_base_paths, weights)
                        record = {"id": identifier, "request": {"input": source, "context": ""},
                            "parent": frozen[identifier]["parent"],
                            "union_reference": frozen[identifier]["proposed"],
                            "parent_pool": frozen[identifier]["parent_pool"],
                            "proposed": response, "base_paths": proposed.choice_base_paths}
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush(); os.fsync(stream.fileno())
                        # The complete evidence is durable above. Aggregate only
                        # needs public candidates and stage diagnostics, not a
                        # second in-memory copy of every candidate's segments.
                        compact_choice = {k: v for k, v in response["phonetic_choice"].items()
                                          if k not in {"rows", "score_records"}}
                        compact_response = {**response, "phonetic_choice": compact_choice}
                        captures.append({k: (compact_response if k == "proposed" else v)
                                         for k, v in record.items() if k != "base_paths"})
                    if len(captures) % 25 == 0:
                        print(json.dumps({"captured": len(captures), "total": 316,
                                          "elapsed_s": time.monotonic() - started}), flush=True)
            capture_sha = sha256_file(output / "capture.jsonl")
            with watchdog.operation("close_capture", update=True):
                write_json(output / "capture.complete.json", {"schema": SCHEMA + "-closure",
                    "rows": len(captures), "capture_sha256": capture_sha, "labels_opened": False})
            with watchdog.operation("late_label_join", update=True):
                verify_pins(pins)
                labels = list(iter_jsonl(authorization["deferred_label"]))
                metrics = compare_all(captures, labels)
                require(sha256_file(output / "capture.jsonl") == capture_sha, "closed capture changed")
            measured = latency(captures)
            union_latency = latency([{**r, "parent": r["union_reference"]} for r in captures])["parent"]
            def component_latency(field):
                records = [{"parent": {"engine_ms": row["proposed"]["phonetic_choice"][field]},
                            "proposed": {"engine_ms": row["proposed"]["phonetic_choice"][field]}}
                           for row in captures]
                return latency(records)["proposed"]
            report = {"schema": SCHEMA, "completed": True, "split": "dev", **metrics,
                "input_contract": "roman_sentence_only", "pins": pins, "limits": LIMITS,
                "capture_sha256": capture_sha,
                "closure_sha256": sha256_file(output / "capture.complete.json"),
                "probe_sha256": sha256_file(output / "probe.jsonl"),
                "qualification_sha256": sha256_file(output / "qualification.json"),
                "latency": {"actual_calibrated": measured["proposed"],
                            "historical_production": measured["parent"], "historical_union": union_latency,
                            "likelihood": component_latency("likelihood_ms"),
                            "calibration_without_likelihood": component_latency("calibration_ms")},
                "timing_scope": "actual calibrated runtime; full historical reference times are not paired",
                "actual_zero_probe_rows": 16, "full_zero_shadow_parity_rows": 316,
                "zero_probe_timing": probes, "production_changed": False, "test_accessed": False,
                "labels_opened_after_capture": True, "gold_not_passed_to_predictor": True,
                "weights": weights, "version": proposed.version}
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
            "error": f"{type(error).__name__}: {error}", "captured": len(captures)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps({"completed": result["completed"], "quality_gate": result["quality_gate"]}))
