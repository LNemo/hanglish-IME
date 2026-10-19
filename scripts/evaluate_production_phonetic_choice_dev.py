"""Run frozen phonetic calibration on actual production candidates only."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_phonetic_choice_dev import (
    authorize as authorize_parent, render_public, probe_groups, without_timing,
    CALIBRATION, CHECKPOINT, LIMITS,
)
from scripts.evaluate_extended_runtime import (
    require, path_key, sha256_file, verify_pins, write_json, iter_jsonl, literal_signature,
)
from scripts.evaluate_phonetic_word_union_dev import ENGLISH
from scripts.evaluate_ctc_word_runtime import compare, latency, verify_runtime_contract
from scripts.evaluate_sentence_only import predict_sentence
from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes

SCHEMA = "production-phonetic-choice-dev-v1"
PLAN = ROOT / ".omx/plans/production-phonetic-choice-dev-v1.md"
PRODUCTION_VERSION = "experimental-ctc-union-1-488fc89732af"


def authorize():
    authorization, requests, frozen, weights, pins = authorize_parent()
    for path in (Path(__file__), PLAN,
                 ROOT / "hanglish/core/production_phonetic_choice_converter.py",
                 ROOT / "scripts/evaluate_phonetic_choice_dev.py",
                 ROOT / "scripts/evaluate_ctc_word_runtime.py",
                 ROOT / "scripts/evaluate_sentence_only.py"):
        key = path_key(path)
        digest = sha256_file(path)
        require(key not in pins or pins[key] == digest, "inherited pin disagreement")
        pins[key] = digest
    verify_pins(pins, defer=authorization["deferred_label"])
    return authorization, requests, frozen, weights, pins


def verify_capture(source, frozen, response, base_paths, weights):
    before = frozen["parent"]
    verify_runtime_contract(source, before, response)
    require(before["version"] == PRODUCTION_VERSION, "production identity differs")
    require(response["input"] == source and response["context"] == ""
            and response["external_context_used"] is False, "source-only contract differs")
    require("phonetic_word_union" not in response, "phonetic proposal path was executed")
    for name in ("ctc_word", "ctc_union"):
        require(without_timing(before[name]) == without_timing(response[name]),
                f"production generation changed: {name}")
    choice = response["phonetic_choice"]
    require(choice["effective_weights"] == list(weights)
            and choice["parent_version"] == PRODUCTION_VERSION,
            "calibration configuration differs")
    if response["sentence_ranking_applied"]:
        require(bool(base_paths) and len({p[0] for p in base_paths}) == len(base_paths),
                "base paths missing or duplicated")
        require(base_paths == sorted(base_paths, key=lambda p: (-p[1], p[0]))
                and [p[0] for p in base_paths] == frozen["parent_pool"],
                "production full pool order differs")
        require(render_public(source, base_paths) == before["candidates"],
                "shadow zero public output differs from production")
        rows = choice["rows"]
        require(len(rows) == len(base_paths), "calibration denominator differs")
        calibrated = []
        for (text, score, segments), row in zip(base_paths, rows):
            features = row["features"]
            require(row["text"] == text and row["base_score"] == score
                    and row["segments"] == segments and len(features) == 4
                    and all(type(v) in (int, float) and math.isfinite(v) for v in features),
                    "base path/feature identity differs")
            adjusted = score + sum(w * v for w, v in zip(weights, features))
            require(score + sum(0. * v for v in features) == score
                    and row["adjusted_score"] == adjusted,
                    "zero or learned calibration score differs")
            calibrated.append((text, adjusted, segments))
        require(choice["zero_weight_parity"] is True, "shadow zero parity failed")
        calibrated.sort(key=lambda p: (-p[1], p[0]))
        require(render_public(source, calibrated) == response["candidates"],
                "actual public output differs from calibrated paths")
    else:
        require(not base_paths and not frozen["parent_pool"] and not choice["rows"]
                and response["candidates"] == before["candidates"],
                "unranked input changed")


def aggregate(captures, labels):
    require([r["id"] for r in captures] == [r["id"] for r in labels],
            "late label identities differ")
    public = compare(captures, labels)
    english, stages = Counter(), Counter()
    for row, case in zip(captures, labels):
        require(row["request"] == {"input": case["input"], "context": ""}
                and case.get("context", "") == "" and case["split"] == "dev",
                "late label input differs")
        gold = set(case["expected"])
        if row["proposed"]["sentence_ranking_applied"]:
            stages["ranked_rows"] += 1
            stages["pool_correct"] += bool(gold & set(row["parent_pool"]))
        source_literal = literal_signature(row["request"]["input"])
        english["literal_signature_regressions"] += (
            literal_signature(row["parent"]["result"]) == source_literal
            and literal_signature(row["proposed"]["result"]) != source_literal)
        signatures = {tuple(ENGLISH.findall(text)) for text in gold}
        if any(signatures):
            english["rows_with_reference_ascii_words"] += 1
            before = tuple(ENGLISH.findall(row["parent"]["result"])) in signatures
            after = tuple(ENGLISH.findall(row["proposed"]["result"])) in signatures
            english["parent_exact_signature"] += before
            english["proposed_exact_signature"] += after
            english["signature_regressions"] += before and not after
    require(public["parent"]["counts"] == dict(rows=316, top1=262, top5=282, errors=0),
            "production reference metrics differ")
    return {
        **{name: {k: v for k, v in public[name].items() if k != "outcomes"}
           for name in ("parent", "proposed")},
        **{name: len(public[name]) for name in ("corrections", "regressions", "top5_losses")},
        "improvement_gate": public["improvement_gate"],
        "quality_gate": bool(public["improvement_gate"]
                             and not english["signature_regressions"]
                             and not english["literal_signature_regressions"]),
        "english_ascii_signature": dict(english), "stage_counts": dict(stages),
    }


def run(output):
    started = time.monotonic(); baseline = system_swap_bytes()
    output.mkdir(parents=True, exist_ok=False)
    captures = []
    try:
        with AtomicResourceWatchdog(started, baseline, output / "resources.jsonl",
                                    limits=LIMITS, interval_s=1.) as watchdog:
            with watchdog.operation("authenticate_inputs", update=True):
                authorization, requests, frozen, weights, pins = authorize()
                groups = probe_groups(requests)
                write_json(output / "pins.json", pins)
                write_json(output / "probe-schedule.json", {"groups": groups,
                    "selection": "16 source-length balanced groups; longest source per group"})
            with watchdog.operation("load_models", update=True):
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

                reference = TracedProduction()
                reference.load_models(); reference.load_word_model(); reference.load_growth_model()
                require(reference.version == PRODUCTION_VERSION, "production version differs")

                def make_adapter(override, phonetic_model=None):
                    value = ProductionPhoneticChoiceConverter(phonetic_checkpoint=CHECKPOINT,
                        calibration_checkpoint=CALIBRATION / "weights.json", calibration_weights=override,
                        external_lexicon=False, word_model=reference.word_model,
                        growth_model=reference.growth_model, phonetic_model=phonetic_model)
                    for name in ("generator", "numeric_units", "ranker", "sentence_ranker", "transducer"):
                        setattr(value, name, getattr(reference, name))
                    value.load_models(); value.load_phonetic_model()
                    require(value.choice_parent_version == reference.version, "loaded parent differs")
                    require(all(getattr(value, name) is getattr(reference, name) for name in (
                        "generator", "ranker", "sentence_ranker", "transducer", "word_model", "growth_model")),
                        "frozen production model sharing differs")
                    return value

                proposed = make_adapter(None)
                zero = make_adapter([0.] * 4, proposed.phonetic_model)
                require(zero.phonetic_model is proposed.phonetic_model, "phonetic model sharing differs")
                for engine in (reference, proposed, zero):
                    for key, value in engine.model_artifacts.items():
                        resolved = path_key(Path(key) if Path(key).is_absolute() else ROOT / key)
                        require(pins.get(resolved) == value, f"unbound runtime artifact: {key}")
            sources = {identifier: source for identifier, source, _ in requests}
            probes = []
            with (output / "probe.jsonl").open("x") as stream:
                for group in groups:
                    identifier = group["probe_id"]; source = sources[identifier]
                    with watchdog.operation("probe_production:" + identifier, update=True):
                        reference.full_paths = []
                        before = predict_sentence(reference, source)
                    tick = time.monotonic()
                    with watchdog.operation("probe_zero:" + identifier, update=True):
                        after = predict_sentence(zero, source)
                        verify_capture(source, frozen[identifier], after, zero.choice_base_paths, [0.] * 4)
                        require(reference.full_paths == zero.choice_base_paths
                                and before["candidates"] == after["candidates"]
                                and before["result"] == after["result"], "actual zero parity failed")
                        record = {"id": identifier, "request": {"input": source, "context": ""},
                            "parent": before, "proposed": after,
                            "parent_paths": reference.full_paths, "base_paths": zero.choice_base_paths,
                            "exact_full_path_parity": True}
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush(); os.fsync(stream.fileno())
                    probes.append({"id": identifier, "group_size": group["count"],
                        "zero_seconds": time.monotonic() - tick,
                        "production_engine_ms": before["engine_ms"], "zero_engine_ms": after["engine_ms"]})
            projection_elapsed = time.monotonic() - started
            projected = 1.3 * (projection_elapsed
                               + sum(p["zero_seconds"] * p["group_size"] for p in probes)) + 60.
            write_json(output / "qualification.json", {"qualified": projected <= LIMITS["whole_seconds"],
                "projected_seconds": projected, "elapsed_seconds": projection_elapsed, "probes": probes,
                "projection": "1.3*(elapsed+sum(zero_seconds*group_size))+60"})
            require(projected <= LIMITS["whole_seconds"], "actual DEV resource projection failed")
            with (output / "capture.jsonl").open("x") as stream:
                for identifier, source, _ in requests:
                    with watchdog.operation("convert:" + identifier, update=True):
                        response = predict_sentence(proposed, source)
                        verify_capture(source, frozen[identifier], response, proposed.choice_base_paths, weights)
                        record = {"id": identifier, "request": {"input": source, "context": ""},
                            "parent": frozen[identifier]["parent"], "parent_pool": frozen[identifier]["parent_pool"],
                            "proposed": response, "base_paths": proposed.choice_base_paths}
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        stream.flush(); os.fsync(stream.fileno())
                        compact = {k: v for k, v in response["phonetic_choice"].items()
                                   if k not in {"rows", "score_records"}}
                        captures.append({k: ({**response, "phonetic_choice": compact} if k == "proposed" else v)
                                         for k, v in record.items() if k != "base_paths"})
                    if len(captures) % 25 == 0:
                        print(json.dumps({"captured": len(captures), "total": 316,
                                          "elapsed_s": time.monotonic() - started}), flush=True)
            with watchdog.operation("close_capture", update=True):
                capture_sha = sha256_file(output / "capture.jsonl")
                write_json(output / "capture.complete.json", {"schema": SCHEMA + "-closure",
                    "rows": len(captures), "capture_sha256": capture_sha, "labels_opened": False})
            with watchdog.operation("late_label_join", update=True):
                verify_pins(pins)
                labels = list(iter_jsonl(authorization["deferred_label"]))
                metrics = aggregate(captures, labels)
                require(sha256_file(output / "capture.jsonl") == capture_sha, "closed capture changed")
            measured = latency(captures)
            def component_latency(field):
                rows = [{name: {"engine_ms": row["proposed"]["phonetic_choice"][field]}
                         for name in ("parent", "proposed")} for row in captures]
                return latency(rows)["proposed"]
            report = {"schema": SCHEMA, "completed": True, "split": "dev", **metrics,
                "input_contract": "roman_sentence_only", "pins": pins, "limits": LIMITS,
                "capture_sha256": capture_sha, "closure_sha256": sha256_file(output / "capture.complete.json"),
                "probe_sha256": sha256_file(output / "probe.jsonl"),
                "qualification_sha256": sha256_file(output / "qualification.json"),
                "latency": {"actual_calibrated": measured["proposed"],
                    "historical_production": measured["parent"],
                    "likelihood": component_latency("likelihood_ms"),
                    "calibration_without_likelihood": component_latency("calibration_ms")},
                "timing_scope": "actual production-pool calibrated runtime; historical production times are not paired",
                "actual_zero_probe_rows": 16, "full_zero_shadow_parity_rows": 316,
                "zero_probe_timing": probes, "production_changed": False, "test_accessed": False,
                "labels_opened_after_capture": True, "gold_not_passed_to_predictor": True,
                "weights": weights, "version": proposed.version, "parent_version": reference.version,
                "selection_scope": "post-hoc DEV-informed architecture; not fresh generalization evidence"}
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
