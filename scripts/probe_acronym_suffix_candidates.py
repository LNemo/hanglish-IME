"""Train-only actual-runtime probe for uppercase acronym plus Korean suffix candidates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hanglish.core.lattice import WordOption
from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.core.phonology import romanize_word
from scripts.evaluate_roberta_context_ranker import SentencePoolRecorder
from scripts.prepare_current_context_data import (
    OUTPUT as PREPARED,
    RUNTIME_VERSION,
    source_hashes as preparation_source_hashes,
    validate_data,
)
from scripts.train_natural_sentence_ranker import rss_bytes


PREPARATION_MANIFEST = "b280bd12081c90fdd6bf53a3356b0a5f0ff0ccf83b905ca26b027c9d09027989"
OUTPUT = ROOT / ".omx/artifacts/acronym-suffix-candidates-v1"
ACRONYM_SUFFIX = re.compile(r"([A-Z]{2,})([a-z]{2,})\Z")
HANGUL = re.compile(r"[가-힣]+\Z")
POLICY = {
    "name": "uppercase-acronym-learned-suffix-v1",
    "rows": 512,
    "additional_candidates": 4,
    "maximum_source_length": 40,
    "word_cache_size": 1024,
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 1800,
    "progress_interval": 32,
    "selection": "First 512 fixed natural TRAIN rows, selected before labels",
}


def validate_extension(source, baseline, result, limit=POLICY["additional_candidates"]):
    if type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError("acronym suffix limit must be an integer from 1 to 8")
    baseline = tuple(baseline)
    result = tuple(result)
    if result[:len(baseline)] != baseline:
        raise ValueError("acronym suffix extension changed the baseline prefix")
    additions = result[len(baseline):]
    baseline_texts = {option.text for option in baseline}
    if (len(additions) > limit
            or len({option.text for option in additions}) != len(additions)
            or any(option.text in baseline_texts for option in additions)):
        raise ValueError("acronym suffix extension is duplicated or exceeds its bound")
    match = ACRONYM_SUFFIX.fullmatch(source)
    if additions and match is None:
        raise ValueError("acronym suffix extension changed an excluded source")
    prefix = "" if match is None else match[1]
    for option in additions:
        if not isinstance(option, WordOption):
            raise ValueError("acronym suffix extension emitted an invalid candidate")
        suffix = option.text[len(prefix):] if option.text.startswith(prefix) else ""
        if (not math.isfinite(option.score)
                or option.kind != "mixed" or not HANGUL.fullmatch(suffix)
                or romanize_word(option.text) != source):
            raise ValueError("acronym suffix extension emitted an invalid candidate")
    return additions


def extend_options(converter, source, learned, baseline,
                   limit=POLICY["additional_candidates"]):
    """Append source-only candidates while preserving every baseline option exactly."""
    baseline = tuple(baseline)
    match = ACRONYM_SUFFIX.fullmatch(source)
    if (not learned or match is None
            or len(source) > POLICY["maximum_source_length"]):
        return baseline
    prefix, suffix = match.groups()
    suffix_options = converter._word_options(suffix, learned)
    seen = {option.text for option in baseline}
    additions = []
    for option in suffix_options:
        if not HANGUL.fullmatch(option.text):
            continue
        text = prefix + option.text
        if text in seen or romanize_word(text) != source:
            continue
        additions.append(WordOption(text, option.score, "mixed"))
        seen.add(text)
        if len(additions) == limit:
            break
    result = baseline + tuple(additions)
    validate_extension(source, baseline, result, limit)
    return result


def install(converter, limit=POLICY["additional_candidates"]):
    audit = {}

    def options(source, learned):
        baseline = converter._word_options(source, learned)
        result = extend_options(converter, source, learned, baseline, limit)
        additions = validate_extension(source, baseline, result, limit)
        audit[(source, learned)] = {
            "source": source,
            "learned": learned,
            "baseline": [{"text": option.text, "score": option.score, "kind": option.kind}
                         for option in baseline],
            "additions": [{"text": option.text, "score": option.score, "kind": option.kind}
                          for option in additions],
            "result_count": len(result),
        }
        return result

    converter.word_options = lru_cache(maxsize=POLICY["word_cache_size"])(options)
    return audit


def metrics(cases):
    result = {"rows": len(cases)}
    for side in ("before", "new"):
        result[f"{side}_top1"] = sum(case[f"{side}_top1"] for case in cases)
        result[f"{side}_top5"] = sum(case[f"{side}_top5"] for case in cases)
        result[f"{side}_pool_covered"] = sum(
            case[f"{side}_pool_covered"] is True for case in cases
        )
        result[f"{side}_sentence_scored"] = sum(
            case[f"{side}_pool_covered"] is not None for case in cases
        )
    result["corrections"] = [case["id"] for case in cases
                             if case["new_top1"] and not case["before_top1"]]
    result["regressions"] = [case["id"] for case in cases
                             if case["before_top1"] and not case["new_top1"]]
    result["top5_gains"] = [case["id"] for case in cases
                            if case["new_top5"] and not case["before_top5"]]
    result["top5_regressions"] = [case["id"] for case in cases
                                  if case["before_top5"] and not case["new_top5"]]
    result["pool_coverage_gains"] = [case["id"] for case in cases
                                     if case["new_pool_covered"] is True
                                     and case["before_pool_covered"] is not True]
    result["pool_coverage_losses"] = [case["id"] for case in cases
                                      if case["before_pool_covered"] is True
                                      and case["new_pool_covered"] is not True]
    return result


def select_train_rows(rows, limit=POLICY["rows"]):
    """Freeze a source-order prefix without inspecting any label field."""
    selected = list(rows[:limit])
    if len(selected) != limit:
        raise ValueError("current natural TRAIN selection is incomplete")
    return selected


def source_hashes():
    hashes = preparation_source_hashes()
    paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/evaluate_roberta_context_ranker.py",
        ROOT / "scripts/prepare_current_context_data.py",
        ROOT / "scripts/train_natural_sentence_ranker.py",
        ROOT / "hanglish/core/neural_converter.py",
        ROOT / "hanglish/core/phonology.py",
        PREPARED / "manifest.json",
        PREPARED / "natural_train.jsonl",
        PREPARED / "natural_dev.jsonl",
        PREPARED / "retention_train.jsonl",
    ]
    for path in paths:
        hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    return dict(sorted(hashes.items()))


def validate_audits(audits):
    for item in audits.values():
        baseline = tuple(WordOption(value["text"], value["score"], value["kind"])
                         for value in item["baseline"])
        additions = tuple(WordOption(value["text"], value["score"], value["kind"])
                          for value in item["additions"])
        validate_extension(item["source"], baseline, baseline + additions)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest, data = validate_data(PREPARED, PREPARATION_MANIFEST)
    if manifest.get("runtime_version") != RUNTIME_VERSION:
        raise ValueError("current prepared runtime differs from the pinned version")
    # Fix the source selection before any expected labels are consulted.
    selected = select_train_rows(data["natural_train"])
    before = source_hashes()
    torch.set_num_threads(POLICY["threads"])
    converter = NeuralConverter(ranker_kind="hybrid")
    converter.load_models()
    if converter.version != RUNTIME_VERSION:
        raise ValueError("actual hybrid runtime differs from current preparation")
    parent_options = converter.word_options
    audits = install(converter)
    child_options = converter.word_options
    recorder = SentencePoolRecorder(converter.sentence_ranker)
    converter.sentence_ranker = recorder
    args.output.mkdir(parents=True)
    selection_path = args.output / "selection.json"
    selection_path.write_text(json.dumps(selected, ensure_ascii=False, sort_keys=True) + "\n")
    cases_path = args.output / "cases.jsonl"
    cases = []
    started = time.monotonic()
    with cases_path.open("x") as stream:
        for index, row in enumerate(selected):
            try:
                converter.word_options = child_options
                recorder.candidates = None
                converted = converter.convert(row["input"], row["context"])
                pool = recorder.candidates
            finally:
                converter.word_options = parent_options
            actual = [candidate["text"] for candidate in converted["candidates"]]
            if (not actual or len(actual) > 5
                    or converted.get("sentence_ranking_applied") != (pool is not None)
                    or any(not isinstance(candidate, str) or not candidate for candidate in actual)
                    or (pool is not None and (not pool or len(set(pool)) != len(pool)
                        or any(not isinstance(candidate, str) or not candidate for candidate in pool)))):
                raise ValueError("actual child conversion returned malformed candidates")
            before_actual = row["actual_candidates"]
            before_pool = row["candidates"] if row["sentence_eligible"] else None
            expected = row["expected"]
            case = {
                "id": row["id"], "input": row["input"], "context": row["context"],
                "expected": expected, "before_candidates": before_actual,
                "new_candidates": actual, "before_pool": before_pool, "new_pool": pool,
                "before_top1": before_actual[0] in expected,
                "new_top1": actual[0] in expected,
                "before_top5": any(candidate in expected for candidate in before_actual),
                "new_top5": any(candidate in expected for candidate in actual),
                "before_pool_covered": (None if before_pool is None else
                                         any(candidate in expected for candidate in before_pool)),
                "new_pool_covered": (None if pool is None else
                                      any(candidate in expected for candidate in pool)),
                "first_choice_changed": before_actual[0] != actual[0],
                "engine_ms": converted["engine_ms"],
            }
            cases.append(case)
            stream.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
            elapsed = time.monotonic() - started
            if (rss_bytes() > POLICY["rss_limit_bytes"]
                    or elapsed > POLICY["time_limit_seconds"]):
                raise RuntimeError("acronym suffix TRAIN probe resource bound exceeded")
            if (index + 1) % POLICY["progress_interval"] == 0:
                print(json.dumps({"event": "acronym_suffix_probe", "rows": index + 1,
                                  "elapsed_seconds": round(elapsed, 2)}), flush=True)
    validate_audits(audits)
    audit_path = args.output / "option-audit.jsonl"
    with audit_path.open("x") as stream:
        for key in sorted(audits):
            stream.write(json.dumps(audits[key], ensure_ascii=False, sort_keys=True) + "\n")
    validate_data(PREPARED, PREPARATION_MANIFEST)
    after = source_hashes()
    if (before != after or converter.word_options is not parent_options
            or json.loads(selection_path.read_text()) != selected):
        raise ValueError("probe source, model, selection, or runtime state changed")
    changed = [item for item in audits.values() if item["additions"]]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Actual child conversion on fixed natural TRAIN first512 only",
        "policy": POLICY,
        "parent_version": RUNTIME_VERSION,
        "preparation_manifest_sha256": PREPARATION_MANIFEST,
        "source_hashes": before,
        "selection_sha256": file_sha256(selection_path),
        "cases_sha256": file_sha256(cases_path),
        "option_audit_sha256": file_sha256(audit_path),
        "metrics": metrics(cases),
        "candidate_audit": {
            "unique_calls": len(audits),
            "changed_calls": len(changed),
            "added_candidates": sum(len(item["additions"]) for item in changed),
            "max_added": max((len(item["additions"]) for item in changed), default=0),
            "baseline_prefix_preserved": True,
            "scores_unmodified_from_suffix_options": True,
            "all_additions_forward_valid": True,
        },
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
        "timing_policy": "One child conversion per row; descriptive latency only.",
        "natural_dev_inference_run": False,
        "development_inference_run": False,
        "test_data_read": False,
        "deployed": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"metrics": report["metrics"],
                      "candidate_audit": report["candidate_audit"],
                      "elapsed_seconds": report["elapsed_seconds"],
                      "peak_rss_bytes": report["peak_rss_bytes"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
