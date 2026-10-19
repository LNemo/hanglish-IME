"""Run the frozen production-pool phonetic calibration on TEST1000."""
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
from scripts.evaluate_extended_runtime import (
    iter_jsonl, literal_signature, load_reference_authorization, path_key,
    require, sha256_file, validate_source_reference, verify_pins, write_json,
)
from scripts.evaluate_ctc_word_runtime import compare, latency, verify_runtime_contract
from scripts.evaluate_phonetic_choice_dev import (
    CHECKPOINT, CHECKPOINT_SHA, CALIBRATION, render_public, probe_groups,
)
from scripts.evaluate_phonetic_word_union_dev import ENGLISH
from scripts.evaluate_production_phonetic_choice_dev import (
    PRODUCTION_VERSION, verify_capture as verify_dev_capture,
)
from scripts.evaluate_sentence_only import predict_sentence
from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes


SCHEMA = "production-phonetic-choice-test-v1"
PLAN = ROOT / ".omx/plans/production-phonetic-choice-test-v1.md"
DEV_RUN = ROOT / ".omx/artifacts/production-phonetic-choice-dev-v1"
ADAPTER = ROOT / "hanglish/core/production_phonetic_choice_converter.py"
WEIGHTS = CALIBRATION / "weights.json"
EXPECTED_ADAPTER_SHA256 = (
    "961aabb7feaadbd14c1cb1b93c98d132c47173e134484400ce8bccbb83015892"
)
EXPECTED_WEIGHTS_SHA256 = (
    "930e3ef942061e6d459d7527c0e073fc6ad5626d59f5170e9b11df8499b39f4b"
)
UNRESOLVED = "__FILL_AFTER_TERMINAL_DEV_PROOF__"
DEV_EXPECTED = {
    "report.json": "479ad53464e3c2ef978002fa5616ddaf9afadac936941a859ae5e01f3422b5cc",
    "capture.jsonl": "f562b5e813484fa49e2d7bb7d231bde63dd40201120eb0bae2d1043f7f4dc432",
    "capture.complete.json": "4ff2664bc0b79762864cef0bef065a287195381ef7f7ed0d63b60c891982a4d4",
    "completion.json": "2ed7af7624459493d8b527db40501426f247fc715674388bc9e178a238f444c6",
    "independent-verification.json": "a7c64c74f7fd756ef53834be326cb96e41927d472a7131020181bd811462977a",
}
LIMITS = {"whole_seconds": 1800.0, "update_seconds": 60.0,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
EXPECTED_PARENT = {"rows": 1000, "top1": 953, "top5": 972, "errors": 0}
EXPECTED_DEV_PARENT = {"rows": 316, "top1": 262, "top5": 282, "errors": 0}


def _resolved_dev_hashes(expected=DEV_EXPECTED):
    require(set(expected) == set(DEV_EXPECTED)
            and all(isinstance(value, str) and len(value) == 64
                    and value != UNRESOLVED for value in expected.values()),
            "terminal DEV artifact hashes have not been frozen")
    return dict(expected)


def validate_dev_gate(report, completion, closure, proof, *, hashes):
    """Validate the terminal DEV decision without touching TEST authority."""
    require(report.get("schema") == "production-phonetic-choice-dev-v1"
            and report.get("completed") is True and report.get("split") == "dev"
            and report.get("quality_gate") is True
            and report.get("improvement_gate") is True,
            "production-only DEV quality gate failed")
    require(report.get("parent", {}).get("counts") == EXPECTED_DEV_PARENT
            and report.get("proposed", {}).get("counts", {}).get("top1", 0) > 262
            and report.get("proposed", {}).get("counts", {}).get("top5", 0) >= 282
            and report.get("regressions") == 0
            and report.get("top5_losses") == 0,
            "production-only DEV strict comparison differs")
    ascii_metrics = report.get("english_ascii_signature", {})
    require(ascii_metrics.get("signature_regressions", 0) == 0
            and ascii_metrics.get("literal_signature_regressions", 0) == 0,
            "production-only DEV preservation gate failed")
    require(report.get("parent_version") == PRODUCTION_VERSION
            and report.get("weights")
            == json.loads(WEIGHTS.read_text(encoding="utf-8"))["weights"]
            and report.get("production_changed") is False
            and report.get("test_accessed") is False
            and report.get("labels_opened_after_capture") is True
            and report.get("gold_not_passed_to_predictor") is True,
            "production-only DEV runtime contract differs")
    require(closure == {"schema": "production-phonetic-choice-dev-v1-closure",
                        "rows": 316, "capture_sha256": hashes["capture.jsonl"],
                        "labels_opened": False},
            "production-only DEV source closure differs")
    require(completion.get("completed") is True
            and completion.get("report_sha256") == hashes["report.json"],
            "production-only DEV completion differs")
    require(proof.get("verified") is True
            and proof.get("report_sha256") == hashes["report.json"]
            and proof.get("capture_sha256") == hashes["capture.jsonl"]
            and proof.get("closure_sha256") == hashes["capture.complete.json"]
            and proof.get("completion_sha256") == hashes["completion.json"]
            and proof.get("decision", {}).get("quality_gate") is True,
            "production-only DEV independent proof differs")
    pins = report.get("pins")
    require(isinstance(pins, dict)
            and pins.get(path_key(ADAPTER)) == EXPECTED_ADAPTER_SHA256
            and pins.get(path_key(WEIGHTS)) == EXPECTED_WEIGHTS_SHA256
            and sha256_file(ADAPTER) == EXPECTED_ADAPTER_SHA256
            and sha256_file(WEIGHTS) == EXPECTED_WEIGHTS_SHA256,
            "production-only DEV adapter/calibration identity differs")
    return {"report_sha256": hashes["report.json"],
            "capture_sha256": hashes["capture.jsonl"],
            "closure_sha256": hashes["capture.complete.json"],
            "completion_sha256": hashes["completion.json"],
            "proof_sha256": hashes["independent-verification.json"],
            "version": report.get("version"), "weights": report["weights"]}


def authenticate_dev(directory=DEV_RUN, expected=DEV_EXPECTED):
    hashes = _resolved_dev_hashes(expected)
    directory = Path(directory)
    require(all((directory / name).is_file()
                and sha256_file(directory / name) == digest
                for name, digest in hashes.items()),
            "terminal DEV artifact identity differs")
    values = {name: json.loads((directory / name).read_text(encoding="utf-8"))
              for name in hashes if name != "capture.jsonl"}
    identity = validate_dev_gate(
        values["report.json"], values["completion.json"],
        values["capture.complete.json"], values["independent-verification.json"],
        hashes=hashes)
    return identity, values["report.json"], hashes


def authorize(*, dev_loader=authenticate_dev,
              reference_loader=load_reference_authorization):
    # This call and its complete quality/proof validation precede any TEST
    # source authorization.  A sentinel or failed DEV gate stops here.
    dev_identity, dev_report, dev_hashes = dev_loader()
    authorization = reference_loader("test")
    rows = list(iter_jsonl(authorization["paths"]["capture"]))
    seen = set()
    requests = [validate_source_reference(row, seen) for row in rows]
    require(len(requests) == 1000, "TEST source denominator differs")
    frozen = {identifier: parent for identifier, _, parent in requests}

    pins = dict(authorization["report"]["pins"])
    for name, digest in dev_hashes.items():
        pins[path_key(DEV_RUN / name)] = digest
    for raw, digest in dev_report["pins"].items():
        key = path_key(Path(raw) if Path(raw).is_absolute() else ROOT / raw)
        require(key not in pins or pins[key] == digest, "DEV/TEST pin disagreement")
        pins[key] = digest
    tracked = [Path(__file__), PLAN, ADAPTER, WEIGHTS, CHECKPOINT,
               ROOT / "scripts/evaluate_production_phonetic_choice_dev.py",
               ROOT / "scripts/evaluate_phonetic_choice_dev.py",
               ROOT / "scripts/evaluate_extended_runtime.py",
               ROOT / "scripts/evaluate_ctc_word_runtime.py",
               ROOT / "scripts/evaluate_sentence_only.py",
               ROOT / "scripts/source_span_resources.py"]
    for path in tracked:
        digest = sha256_file(path)
        key = path_key(path)
        require(key not in pins or pins[key] == digest, "fixed runtime pin disagreement")
        pins[key] = digest
    require(sha256_file(CHECKPOINT) == CHECKPOINT_SHA,
            "phonetic checkpoint identity differs")
    verify_pins(pins, defer=authorization["deferred_label"])
    return authorization, requests, frozen, dev_identity["weights"], pins, dev_identity


def verify_capture(source, parent, response, base_paths, weights):
    """Verify learned output without inventing a historical TEST full pool."""
    derived = {"parent": parent, "parent_pool": [path[0] for path in base_paths]}
    verify_dev_capture(source, derived, response, base_paths, weights)
    require(response["phonetic_choice"]["parent_version"] == PRODUCTION_VERSION,
            "calibrated TEST parent identity differs")
    require("phonetic_word_union" not in response,
            "phonetic candidate proposals ran during TEST")


def aggregate(captures, labels):
    require(len(captures) == len(labels) == 1000
            and [row["id"] for row in captures] == [row["id"] for row in labels],
            "late TEST label identities differ")
    public = compare(captures, labels)
    english = Counter()
    stages = Counter()
    for row, case in zip(captures, labels):
        require(case.get("split") == "test" and case.get("context", "") == ""
                and row["request"] == {"input": case.get("input"), "context": ""},
                "late TEST label source contract differs")
        gold = set(case["expected"])
        if row["proposed"].get("sentence_ranking_applied"):
            stages["ranked_rows"] += 1
            in_pool = bool(gold & set(row["actual_base_pool"]))
            in_public = bool(gold & {
                candidate["text"] for candidate in row["proposed"]["candidates"]})
            selected = row["proposed"]["result"] in gold
            stages["actual_base_pool_gold"] += in_pool
            stages["public_top5_gold"] += in_public
            stages["selected_gold"] += selected
            stages["actual_pool_to_public_loss"] += in_pool and not in_public
            stages["actual_pool_to_first_loss"] += in_pool and not selected
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
    require(public["parent"]["counts"] == EXPECTED_PARENT,
            "frozen TEST production metrics differ")
    gate = (public["improvement_gate"]
            and english["signature_regressions"] == 0
            and english["literal_signature_regressions"] == 0)
    return {
        "parent": {k: v for k, v in public["parent"].items() if k != "outcomes"},
        "proposed": {k: v for k, v in public["proposed"].items() if k != "outcomes"},
        **{name: len(public[name])
           for name in ("corrections", "regressions", "top5_losses")},
        "improvement_gate": public["improvement_gate"], "quality_gate": gate,
        "english_ascii_signature": dict(english),
        "actual_base_pool_stage_counts": dict(stages),
    }


def run(output, *, final=False):
    require(final, "TEST1000 requires the frozen --final execution flag")
    output = Path(output)
    require(not output.exists(), "output already exists")
    started = time.monotonic()
    baseline = system_swap_bytes()
    output.mkdir(parents=True, exist_ok=False)
    captures = []
    test_labels_opened = False
    try:
        with AtomicResourceWatchdog(started, baseline, output / "resources.jsonl",
                                    limits=LIMITS, interval_s=1.0) as watchdog:
            with watchdog.operation("authenticate_dev_then_test_sources", update=True):
                authorization, requests, frozen, weights, pins, dev_identity = authorize()
                groups = probe_groups(requests)
                write_json(output / "pins.json", pins)
                write_json(output / "probe-schedule.json", {
                    "groups": groups,
                    "selection": "16 source-length balanced groups; longest source per group",
                })
            with watchdog.operation("load_models", update=True):
                import torch
                from hanglish.core.ctc_union_converter import CTCUnionConverter
                from hanglish.core.production_phonetic_choice_converter import (
                    ProductionPhoneticChoiceConverter,
                )
                torch.set_num_threads(2)
                torch.set_num_interop_threads(1)
                torch.use_deterministic_algorithms(True)

                class TracedProduction(CTCUnionConverter):
                    def _extend_sentence_paths(self, text, context, paths):
                        result = super()._extend_sentence_paths(text, context, paths)
                        self.full_paths = deepcopy(result)
                        return result

                reference = TracedProduction()
                reference.load_models()
                reference.load_word_model()
                reference.load_growth_model()
                require(reference.version == PRODUCTION_VERSION,
                        "production TEST parent version differs")

                def make_adapter(override, phonetic_model=None):
                    value = ProductionPhoneticChoiceConverter(
                        phonetic_checkpoint=CHECKPOINT,
                        calibration_checkpoint=WEIGHTS,
                        calibration_weights=override,
                        external_lexicon=False,
                        word_model=reference.word_model,
                        growth_model=reference.growth_model,
                        phonetic_model=phonetic_model,
                    )
                    for name in ("generator", "numeric_units", "ranker",
                                 "sentence_ranker", "transducer"):
                        setattr(value, name, getattr(reference, name))
                    value.load_models()
                    value.load_phonetic_model()
                    require(value.choice_parent_version == reference.version,
                            "loaded TEST production parent differs")
                    return value

                proposed = make_adapter(None)
                zero = make_adapter([0.0] * 4, proposed.phonetic_model)
                require(zero.phonetic_model is proposed.phonetic_model,
                        "TEST phonetic model sharing differs")
                shared = ("generator", "ranker", "sentence_ranker", "transducer",
                          "word_model", "growth_model")
                require(all(getattr(proposed, name) is getattr(reference, name)
                            and getattr(zero, name) is getattr(reference, name)
                            for name in shared),
                        "TEST production model sharing differs")
                require(proposed.version == dev_identity["version"],
                        "TEST adapter version differs from qualified DEV")
                for engine in (reference, proposed, zero):
                    for raw, digest in engine.model_artifacts.items():
                        path = Path(raw) if Path(raw).is_absolute() else ROOT / raw
                        require(pins.get(path_key(path)) == digest,
                                "unbound TEST runtime artifact: " + raw)

            sources = {identifier: source for identifier, source, _ in requests}
            probes = []
            with (output / "probe.jsonl").open("x", encoding="utf-8") as stream:
                for group in groups:
                    identifier = group["probe_id"]
                    source = sources[identifier]
                    with watchdog.operation("probe_production:" + identifier, update=True):
                        reference.full_paths = []
                        before = predict_sentence(reference, source)
                    tick = time.monotonic()
                    with watchdog.operation("probe_zero:" + identifier, update=True):
                        after = predict_sentence(zero, source)
                        verify_runtime_contract(source, frozen[identifier], before)
                        require(before["result"] == frozen[identifier]["result"]
                                and before["candidates"] == frozen[identifier]["candidates"],
                                "fresh TEST production public output differs")
                        verify_capture(source, frozen[identifier], after,
                                       zero.choice_base_paths, [0.0] * 4)
                        require(reference.full_paths == zero.choice_base_paths
                                and before["result"] == after["result"]
                                and before["candidates"] == after["candidates"],
                                "actual TEST zero full-path parity failed")
                        record = {"id": identifier,
                                  "request": {"input": source, "context": ""},
                                  "parent": before, "proposed": after,
                                  "parent_paths": reference.full_paths,
                                  "base_paths": zero.choice_base_paths,
                                  "exact_full_path_parity": True}
                        stream.write(json.dumps(record, ensure_ascii=False,
                                                allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                    probes.append({"id": identifier, "group_size": group["count"],
                                   "zero_seconds": time.monotonic() - tick,
                                   "production_engine_ms": before["engine_ms"],
                                   "zero_engine_ms": after["engine_ms"]})
            elapsed = time.monotonic() - started
            projected = 1.3 * (elapsed + sum(
                row["zero_seconds"] * row["group_size"] for row in probes)) + 60.0
            qualification = {"qualified": projected <= LIMITS["whole_seconds"],
                             "projected_seconds": projected,
                             "elapsed_seconds": elapsed, "probes": probes,
                             "projection":
                             "1.3*(elapsed+sum(zero_seconds*group_size))+60"}
            write_json(output / "qualification.json", qualification)
            require(qualification["qualified"],
                    "actual TEST1000 resource projection failed")

            with (output / "capture.jsonl").open("x", encoding="utf-8") as stream:
                for identifier, source, _ in requests:
                    with watchdog.operation("convert:" + identifier, update=True):
                        response = predict_sentence(proposed, source)
                        base_paths = proposed.choice_base_paths
                        verify_capture(source, frozen[identifier], response,
                                       base_paths, weights)
                        record = {"id": identifier,
                                  "request": {"input": source, "context": ""},
                                  "parent": frozen[identifier],
                                  "proposed": response,
                                  "actual_base_pool": [path[0] for path in base_paths],
                                  "base_paths": base_paths}
                        stream.write(json.dumps(record, ensure_ascii=False,
                                                allow_nan=False) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())
                        compact_choice = {
                            key: value for key, value in response["phonetic_choice"].items()
                            if key not in {"rows", "score_records"}}
                        captures.append({"id": identifier, "request": record["request"],
                                         "parent": record["parent"],
                                         "actual_base_pool": record["actual_base_pool"],
                                         "proposed": {**response,
                                                      "phonetic_choice": compact_choice}})
                    if len(captures) % 25 == 0:
                        print(json.dumps({"captured": len(captures), "total": 1000,
                                          "elapsed_s": time.monotonic() - started}),
                              flush=True)
            with watchdog.operation("close_capture", update=True):
                capture_sha = sha256_file(output / "capture.jsonl")
                write_json(output / "capture.complete.json", {
                    "schema": SCHEMA + "-closure", "rows": len(captures),
                    "capture_sha256": capture_sha, "source_only": True,
                    "labels_opened": False})
            with watchdog.operation("late_test_label_join", update=True):
                test_labels_opened = True
                verify_pins(pins)
                labels = list(iter_jsonl(authorization["deferred_label"]))
                metrics = aggregate(captures, labels)
                require(sha256_file(output / "capture.jsonl") == capture_sha,
                        "closed TEST capture changed")

            measured = latency(captures)
            def component_latency(field):
                values = [{"parent": {"engine_ms": row["proposed"]["phonetic_choice"][field]},
                           "proposed": {"engine_ms": row["proposed"]["phonetic_choice"][field]}}
                          for row in captures]
                return latency(values)["proposed"]

            report = {"schema": SCHEMA, "completed": True, "split": "test",
                      **metrics, "input_contract": "roman_sentence_only",
                      "pins": pins, "limits": LIMITS,
                      "dev_qualification": dev_identity,
                      "capture_sha256": capture_sha,
                      "closure_sha256": sha256_file(output / "capture.complete.json"),
                      "probe_sha256": sha256_file(output / "probe.jsonl"),
                      "qualification_sha256": sha256_file(output / "qualification.json"),
                      "latency": {"actual_calibrated": measured["proposed"],
                                  "historical_production": measured["parent"],
                                  "likelihood": component_latency("likelihood_ms"),
                                  "calibration_without_likelihood":
                                  component_latency("calibration_ms")},
                      "timing_scope": "actual calibrated runtime; historical production timings are not fresh paired measurements",
                      "actual_zero_probe_rows": 16,
                      "full_zero_shadow_parity_rows": 1000,
                      "zero_probe_timing": probes, "weights": weights,
                      "version": proposed.version,
                      "parent_version": reference.version,
                      "production_changed": False, "test_accessed": True,
                      "labels_opened_after_capture": True,
                      "gold_not_passed_to_predictor": True,
                      "repeatedly_seen_regression_set": True,
                      "fully_blind_generalization": False,
                      "performance_tuning_allowed": False,
                      "historical_full_parent_paths_available": False}
            with watchdog.operation("write_report", update=True):
                write_json(output / "report.json", report)
            terminal = watchdog.sample_and_append("after_report")
            with watchdog.operation("write_completion", update=True):
                write_json(output / "completion.json", {
                    "completed": True,
                    "report_sha256": sha256_file(output / "report.json"),
                    "resource_after_report": terminal})
            watchdog.sample_and_append("after_completion")
        return report
    except BaseException as error:
        if not (output / "failure.json").exists():
            write_json(output / "failure.json", {"completed": False,
                "error": f"{type(error).__name__}: {error}",
                "captured": len(captures),
                "test_labels_opened": test_labels_opened})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    result = run(args.output, final=args.final)
    print(json.dumps({"completed": result["completed"],
                      "quality_gate": result["quality_gate"]}))
