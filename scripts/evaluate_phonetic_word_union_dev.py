"""Frozen DEV comparison of production and additional phonetic word candidates."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_extended_runtime import (
    load_reference_authorization, validate_source_reference, verify_pins,
    iter_jsonl, write_json, path_key, sha256_file, require, literal_signature,
)
from scripts.evaluate_ctc_word_runtime import compare, latency, verify_runtime_contract
from scripts.evaluate_sentence_only import predict_sentence
from scripts.source_span_resources import AtomicResourceWatchdog, system_swap_bytes

SCHEMA = "phonetic-word-union-dev-v1"
CHECKPOINT = ROOT / ".omx/artifacts/observed-phonetic-augmented-training-v1/phonetic_transducer.pt"
CHECKPOINT_SHA = "f06b0a7cd6c5ccbdfaa376cc29f59e90b3fcf0ae3a157fd2c4f7efaa73b4180d"
LIMITS = {"whole_seconds": 1800, "update_seconds": 60,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
PLAN = ROOT / ".omx/plans/phonetic-word-union-dev-v1.md"
ENGLISH = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def verify_pair(source, before, after, parent_pool, parent_version):
    """Check source coverage, production identity and every observable union stage."""
    verify_runtime_contract(source, before, after)
    require(before["version"] == parent_version, "parent runtime version differs")
    trace = after["phonetic_word_union"]
    require(trace["experimental"] is True and trace["parent_version"] == parent_version,
            "phonetic parent identity differs")
    for name in ("ctc_word", "ctc_union"):
        old, new = dict(before[name]), dict(after[name])
        old.pop("generation_ms", None); new.pop("generation_ms", None)
        require(old == new, f"phonetic adapter changed {name}")
    require(trace["production_pool_texts"] == parent_pool,
            "phonetic adapter changed the complete production pool")
    generated = [row["text"] for row in trace["generated_extras"]]
    scoreable, union = trace["scoreable_extras"], trace["union"]
    require(len(generated) == len(set(generated)) <= 96
            and not set(generated) & set(parent_pool)
            and set(scoreable) <= set(generated)
            and len(union) == len(set(union))
            and set(union) == set(parent_pool) | set(scoreable),
            "phonetic pool composition differs")
    public = [row["text"] for row in after["candidates"]]
    require(trace["public"] == public
            and trace["selected_extras"] == [s for s in generated if s in public]
            and trace["missing_extras"] == [s for s in generated if s not in public],
            "phonetic public trace differs")
    require(type(trace["generation_requests"]) is int
            and 0 <= trace["generation_requests"] <= 96,
            "phonetic generation request budget differs")
    for row in trace["generated_extras"]:
        start, end = row["source_start"], row["source_end"]
        require(type(start) is int and type(end) is int
                and 0 <= start < end <= len(source)
                and row["source"] == source[start:end], "proposal source offset differs")
    if after.get("sentence_ranking_applied"):
        require(set(public) <= set(union) | {source}, "public path outside union")
    else:
        require(not generated and not parent_pool and not union,
                "unranked input unexpectedly generated paths")


def aggregate(captures, labels):
    require([r["id"] for r in captures] == [r["id"] for r in labels],
            "capture/label identities differ")
    public = compare(captures, labels)
    counts = Counter(); english = Counter()
    for row, case in zip(captures, labels):
        require(row["request"] == {"input": case["input"], "context": ""}
                and case.get("context", "") == "" and case["split"] == "dev",
                "late label source contract differs")
        gold = set(case["expected"]); trace = row["proposed"]["phonetic_word_union"]
        parent = set(row["parent_pool"])
        stages = {"parent_pool": parent,
                  "with_generated": parent | {r["text"] for r in trace["generated_extras"]},
                  "scoreable_union": set(trace["union"]),
                  "public_top5": set(trace["public"][:5]),
                  "selected": {row["proposed"]["result"]}}
        # Pool metrics count only rows that actually enter the sentence ranking stage.
        if row["proposed"].get("sentence_ranking_applied"):
            counts["ranked_rows"] += 1
            for stage, texts in stages.items():
                counts[stage + "_correct"] += bool(gold & texts)
        counts["rows"] += 1
        counts["generated_extras"] += len(trace["generated_extras"])
        counts["scoreable_extras"] += len(trace["scoreable_extras"])
        counts["generation_budget_exhausted_rows"] += trace["generation_budget_exhausted"]
        counts["skipped_known_english_spans"] += len(trace.get("skipped_known_english", []))
        if row["proposed"].get("sentence_ranking_applied"):
            counts["marginal_generated_gold_rows"] += not bool(gold & parent) and bool(gold & stages["with_generated"])
            counts["generated_to_scoreable_gold_losses"] += bool(gold & stages["with_generated"]) and not bool(gold & stages["scoreable_union"])
            counts["union_to_public_gold_losses"] += bool(gold & stages["scoreable_union"]) and not bool(gold & stages["public_top5"])
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
    # Existing strict quality gate is retained; ASCII signatures supplement full-text metrics.
    gate = (public["improvement_gate"] and not english["signature_regressions"]
            and not english["literal_signature_regressions"])
    return {"parent": {k: v for k, v in public["parent"].items() if k != "outcomes"},
            "proposed": {k: v for k, v in public["proposed"].items() if k != "outcomes"},
            **{k: len(public[k]) for k in ("corrections", "regressions", "top5_losses")},
            "improvement_gate": public["improvement_gate"], "quality_gate": gate,
            "stage_counts": dict(counts), "english_ascii_signature": dict(english),
            "english_scope": "Exact case-sensitive ASCII word sequence on references containing ASCII words; not all English-copy behavior."}


def authorize():
    authorization = load_reference_authorization("dev")
    require(sha256_file(CHECKPOINT) == CHECKPOINT_SHA, "phonetic checkpoint differs")
    references = list(iter_jsonl(authorization["paths"]["capture"]))
    seen = set(); requests = [validate_source_reference(row, seen) for row in references]
    require(len(requests) == 316, "DEV request count differs")
    training = CHECKPOINT.parent
    proof = json.loads((training / "independent-verification.json").read_text())
    require(proof["verified"] is True and proof["checkpoint_sha256"] == CHECKPOINT_SHA
            and proof["report_sha256"] == sha256_file(training / "report.json"),
            "phonetic training proof differs")
    tracked = [Path(__file__), PLAN, CHECKPOINT, CHECKPOINT.with_suffix(".json"),
               training / "independent-verification.json", training / "report.json"]
    tracked += [ROOT / p for p in (
        "scripts/evaluate_extended_runtime.py", "scripts/source_span_resources.py",
        "hanglish/core/phonetic_word_union_converter.py",
        "hanglish/models/batched_phonetic_decode.py")]
    pins = dict(authorization["report"]["pins"])
    pins.update({path_key(p): sha256_file(p) for p in tracked})
    return authorization, requests, pins


def run(output):
    started = time.monotonic(); baseline = system_swap_bytes()
    output = Path(output); require(not output.exists(), "output already exists")
    output.mkdir(parents=True, exist_ok=False)
    captures = []
    try:
        with AtomicResourceWatchdog(started, baseline, output / "resources.jsonl",
                                    limits=LIMITS, interval_s=1.) as watchdog:
            with watchdog.operation("authenticate_inputs", update=True):
                authorization, requests, pins = authorize()
                write_json(output / "pins.json", pins)
            with watchdog.operation("load_models", update=True):
                import torch
                from hanglish.core.ctc_union_converter import CTCUnionConverter
                from hanglish.core.phonetic_word_union_converter import PhoneticWordUnionConverter
                torch.set_num_threads(2); torch.set_num_interop_threads(1)

                class TracedProduction(CTCUnionConverter):
                    def _extend_sentence_paths(self, text, context, paths):
                        paths = super()._extend_sentence_paths(text, context, paths)
                        self.production_pool = [p[0] for p in paths]
                        return paths

                parent = TracedProduction(); parent.load_models()
                parent.load_word_model(); parent.load_growth_model()
                proposed = PhoneticWordUnionConverter(
                    phonetic_checkpoint=CHECKPOINT, external_lexicon=False,
                    word_model=parent.word_model, growth_model=parent.growth_model)
                for name in ("generator", "numeric_units", "ranker", "sentence_ranker", "transducer"):
                    setattr(proposed, name, getattr(parent, name))
                proposed.load_models(); proposed.load_phonetic_model()
                require(parent.version == "experimental-ctc-union-1-488fc89732af",
                        "current production version differs")
                require(proposed.phonetic_parent_version == parent.version,
                        "loaded phonetic parent version differs")
                require(parent.ranker_kind == proposed.ranker_kind and all(
                    getattr(parent, name) is getattr(proposed, name) for name in (
                        "generator", "ranker", "sentence_ranker", "transducer",
                        "word_model", "growth_model")), "shared frozen model identity differs")
            for artifacts in (parent.model_artifacts, proposed.model_artifacts,
                              proposed.ctc_word_artifacts, proposed.growth_artifacts,
                              proposed.phonetic_artifacts):
                for key, value in artifacts.items():
                    resolved = path_key(Path(key) if Path(key).is_absolute() else ROOT / key)
                    require(pins.get(resolved) == value, f"unbound runtime artifact: {key}")
            with (output / "capture.jsonl").open("x", encoding="utf-8") as stream:
                for identifier, source, frozen in requests:
                    with watchdog.operation("convert:" + identifier, update=True):
                        parent.production_pool = []
                        before = predict_sentence(parent, source)
                        after = predict_sentence(proposed, source)
                        require(before["result"] == frozen["result"]
                                and before["candidates"] == frozen["candidates"],
                                "production differs from frozen DEV capture")
                        verify_pair(source, before, after, parent.production_pool, parent.version)
                        record = {"id": identifier, "request": {"input": source, "context": ""},
                                  "parent": before, "proposed": after,
                                  "parent_pool": parent.production_pool}
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush(); os.fsync(stream.fileno()); captures.append(record)
                    if len(captures) % 25 == 0:
                        print(json.dumps({"captured": len(captures), "total": 316,
                                          "elapsed_s": time.monotonic() - started}), flush=True)
            capture_sha = sha256_file(output / "capture.jsonl")
            write_json(output / "capture.complete.json", {
                "schema": SCHEMA + "-closure", "rows": len(captures),
                "capture_sha256": capture_sha, "labels_opened": False})
            with watchdog.operation("late_label_join", update=True):
                verify_pins(pins)
                labels = list(iter_jsonl(authorization["deferred_label"]))
                metrics = aggregate(captures, labels)
                require(metrics["parent"]["counts"] == {
                    "rows": 316, "top1": 262, "top5": 282, "errors": 0},
                    "production metric parity differs")
                require(sha256_file(output / "capture.jsonl") == capture_sha,
                        "capture changed during label join")
            report = {"schema": SCHEMA, "completed": True, "split": "dev",
                      "input_contract": "roman_sentence_only", **metrics,
                      "capture_sha256": capture_sha,
                      "closure_sha256": sha256_file(output / "capture.complete.json"),
                      "pins": pins, "limits": LIMITS, "latency": latency(captures),
                      "timing_scope": "paired serial engines sharing frozen models",
                      "parent_version": parent.version, "proposed_version": proposed.version,
                      "production_changed": False, "test_accessed": False,
                      "labels_opened_after_capture": True,
                      "gold_not_passed_to_predictor": True}
            with watchdog.operation("write_report", update=True):
                write_json(output / "report.json", report)
            terminal = watchdog.sample_and_append("after_report")
            with watchdog.operation("write_completion", update=True):
                write_json(output / "completion.json", {
                    "completed": True, "report_sha256": sha256_file(output / "report.json"),
                    "resource_after_report": terminal})
            watchdog.sample_and_append("after_completion")
        return report
    except BaseException as error:
        write_json(output / "failure.json", {"completed": False,
            "error": f"{type(error).__name__}: {error}", "captured": len(captures)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); result = run(args.output)
    print(json.dumps({k: result[k] for k in ("completed", "quality_gate", "corrections",
                                            "regressions", "top5_losses")}), flush=True)
