"""Model-free lattice audit from frozen current features and cached word options."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
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
from scripts.prepare_context_training import proposal_paths
from scripts.prepare_current_context_data import (
    OUTPUT as PREPARED,
    RUNTIME_VERSION,
    source_hashes as preparation_source_hashes,
    validate_data,
)
from scripts.prepare_natural_word_ranker import capture_groups
from scripts.probe_acronym_suffix_candidates import (
    OUTPUT as PROBE,
    POLICY as PROBE_POLICY,
    source_hashes as probe_source_hashes,
    validate_audits,
)
from scripts.probe_proposal_search import can_compose
from scripts.train_natural_sentence_ranker import rss_bytes


PREPARATION_MANIFEST = "b280bd12081c90fdd6bf53a3356b0a5f0ff0ccf83b905ca26b027c9d09027989"
OUTPUT = ROOT / ".omx/artifacts/current-cached-lattice-v1"
PROFILE = {
    "name": "current-cached-lattice-train-v1",
    "rows": 512,
    "threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 300,
    "progress_interval": 32,
    "model_inference_calls": 0,
}
TARGET_RUN = re.compile(r"[A-Za-z가-힣]+|.", re.DOTALL)
FEATURE_KEYS = (
    "source", "start", "end", "context", "candidates",
    "base_scores", "kinds", "teacher_scores",
)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def option_from_dict(value):
    if (not isinstance(value, dict) or set(value) != {"text", "score", "kind"}
            or not isinstance(value["text"], str) or not value["text"]
            or type(value["score"]) not in (int, float)
            or not math.isfinite(value["score"])
            or not isinstance(value["kind"], str) or not value["kind"]):
        raise ValueError("cached word option is malformed")
    return WordOption(value["text"], float(value["score"]), value["kind"])


class CachedOptions:
    """Fail closed: the audit may use only baseline tuples recorded by the probe."""

    def __init__(self, rows):
        self.values = {}
        self.calls = 0
        self.cache_misses = 0
        for row in rows:
            if (not isinstance(row, dict) or type(row.get("learned")) is not bool
                    or not isinstance(row.get("source"), str)):
                raise ValueError("option audit row is malformed")
            key = (row["source"], row["learned"])
            if key in self.values:
                raise ValueError("option audit contains a duplicate lookup key")
            baseline = tuple(option_from_dict(value) for value in row.get("baseline", []))
            if not baseline:
                raise ValueError("option audit baseline is empty")
            self.values[key] = baseline

    def __call__(self, source, learned):
        self.calls += 1
        key = (source, learned)
        if key not in self.values:
            self.cache_misses += 1
            raise KeyError(f"uncaptured word option lookup: {source!r}, learned={learned!r}")
        return self.values[key]


class ReplayRanker:
    """Return exact captured word scores and reject every unexpected call."""

    def __init__(self, groups):
        self.expected = [(group["context"], candidate, float(score))
                         for group in groups
                         for candidate, score in zip(
                             group["candidates"], group["teacher_scores"]
                         )]
        self.calls = 0

    def score_pairs(self, contexts, candidates):
        self.calls += 1
        observed = list(zip(contexts, candidates))
        expected = [(context, candidate) for context, candidate, _ in self.expected]
        if len(contexts) != len(candidates) or observed != expected:
            raise ValueError("word score replay differs from captured runtime features")
        return [score for _, _, score in self.expected]


class CaptureConverter(NeuralConverter):
    def _prepare_candidates(self, *args, **kwargs):
        result = super()._prepare_candidates(*args, **kwargs)
        self.last_static_pieces = result[1]
        return result


def target_segments(source, expected):
    """Align whole letter runs and individual literal characters to source offsets."""
    if romanize_word(expected) != source:
        return None
    offset = 0
    segments = []
    for match in TARGET_RUN.finditer(expected):
        text = match.group()
        roman = romanize_word(text)
        segments.append((offset, offset + len(roman), text,
                         "letter_run" if text[0].isalpha() else "literal"))
        offset += len(roman)
    if offset != len(source):
        return None
    return segments


def align_piece(expected_values, source, start, end):
    targets = set()
    boundary_cut = False
    valid = 0
    for expected in expected_values:
        segments = target_segments(source, expected)
        if segments is None:
            continue
        valid += 1
        contained = [segment for segment in segments
                     if start <= segment[0] and segment[1] <= end]
        if (contained and contained[0][0] == start and contained[-1][1] == end
                and all(left[1] == right[0] for left, right in zip(contained, contained[1:]))):
            targets.add(''.join(segment[2] for segment in contained))
            continue
        if any(left < end and start < right for left, right, _, _ in segments):
            boundary_cut = True
    return {"targets": sorted(targets), "boundary_cut": boundary_cut,
            "valid_expected": valid}


def classify(expected, pieces, static_pieces, proposal_texts, sentence_eligible):
    if sentence_eligible and any(value in expected for value in proposal_texts):
        return "pool_covered"
    if not sentence_eligible and any(value in expected for value in proposal_texts):
        return "deterministic_path_present"
    if not any(can_compose(value, pieces) or can_compose(value, static_pieces)
               for value in expected):
        return "missing_fullpiece_candidate"
    if not any(can_compose(value, pieces, 8) or can_compose(value, static_pieces, 8)
               for value in expected):
        return "top8_option_cap"
    return "beam_pruning"


def missing_pieces(row, pieces, static_pieces):
    result = []
    for source, offset, options in pieces:
        aligned = align_piece(row["expected"], row["input"], offset, offset + len(source))
        if not aligned["targets"]:
            if aligned["boundary_cut"]:
                result.append({"source": source, "start": offset,
                               "end": offset + len(source), "kind": "unaligned_boundary",
                               "targets": [], "options": [asdict(option) for option in options]})
            continue
        if any(option.text in aligned["targets"] for option in options):
            continue
        static_options = next((values for static_source, static_offset, values in static_pieces
                               if static_source == source and static_offset == offset), ())
        result.append({
            "source": source, "start": offset, "end": offset + len(source),
            "kind": ("learned_piece_filter" if any(
                        option.text in aligned["targets"] for option in static_options)
                     else "protected_or_unscored_raw" if all(
                        option.kind == "raw" for option in options)
                     else "missing_ranked_piece_candidate"),
            "targets": aligned["targets"],
            "options": [asdict(option) for option in options],
            "static_options": [asdict(option) for option in static_options],
        })
    return result


def feature_records(groups):
    return [(group["context"], candidate, score)
            for group in groups
            for candidate, score in zip(group["candidates"], group["teacher_scores"])]


def replay_case(converter, cache, row):
    replay = ReplayRanker(row["groups"])
    converter.ranker = replay
    converter.word_options = cache
    pieces, static_pieces, _, _, _ = converter._prepare_candidates(
        row["input"], row["context"], learned=True, raw=False
    )
    captured = capture_groups(static_pieces, feature_records(row["groups"]), row["context"])
    actual_features = [{key: group[key] for key in FEATURE_KEYS} for group in captured]
    frozen_features = [{key: group[key] for key in FEATURE_KEYS} for group in row["groups"]]
    if actual_features != frozen_features:
        raise ValueError("cached static pieces differ from frozen word features")
    paths = proposal_paths(pieces, static_pieces)
    texts = [path[0] for path in paths]
    if row["sentence_eligible"] and texts != row["candidates"]:
        raise ValueError("cached learned/static proposal union differs from frozen pool")
    reason = classify(row["expected"], pieces, static_pieces, texts,
                      row["sentence_eligible"])
    return {
        "id": row["id"], "input": row["input"], "context": row["context"],
        "expected": row["expected"], "sentence_eligible": row["sentence_eligible"],
        "pool_covered": (None if not row["sentence_eligible"] else
                         any(value in row["expected"] for value in row["candidates"])),
        "classification": reason,
        "proposal_count": len(texts),
        "proposal_texts": texts,
        "learned_pieces": [[source, offset, [asdict(option) for option in options]]
                           for source, offset, options in pieces],
        "static_pieces": [[source, offset, [asdict(option) for option in options]]
                          for source, offset, options in static_pieces],
        "missing_pieces": missing_pieces(row, pieces, static_pieces),
        "word_score_replay_calls": replay.calls,
    }


def validate_probe(report_sha256, report, option_rows, selection, cases, fixed):
    if file_sha256(PROBE / "report.json") != report_sha256:
        raise ValueError("acronym probe report differs from explicit SHA")
    if (report.get("policy") != PROBE_POLICY
            or report.get("parent_version") != RUNTIME_VERSION
            or report.get("preparation_manifest_sha256") != PREPARATION_MANIFEST
            or report.get("source_hashes") != probe_source_hashes()
            or report.get("natural_dev_inference_run") is not False
            or report.get("development_inference_run") is not False
            or report.get("test_data_read") is not False
            or report.get("deployed") is not False
            or report.get("metrics", {}).get("rows") != PROFILE["rows"]
            or file_sha256(PROBE / "selection.json") != report.get("selection_sha256")
            or file_sha256(PROBE / "cases.jsonl") != report.get("cases_sha256")
            or file_sha256(PROBE / "option-audit.jsonl") != report.get("option_audit_sha256")):
        raise ValueError("acronym probe provenance or policy is incomplete")
    if not (len(selection) == len(cases) == len(fixed) == PROFILE["rows"]):
        raise ValueError("acronym probe fixed selection is incomplete")
    keys = ("id", "input", "context", "expected")
    for selected, case, source in zip(selection, cases, fixed):
        if (any(selected.get(key) != source.get(key) for key in keys)
                or any(case.get(key) != source.get(key) for key in keys)):
            raise ValueError("acronym probe source identity or order differs")
    audits = {(row["source"], row["learned"]): row for row in option_rows}
    if len(audits) != len(option_rows):
        raise ValueError("option audit contains duplicate lookup rows")
    validate_audits(audits)


def source_hashes():
    hashes = probe_source_hashes()
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/probe_acronym_suffix_candidates.py",
        ROOT / "scripts/prepare_context_training.py",
        ROOT / "scripts/prepare_natural_word_ranker.py",
        ROOT / "scripts/probe_proposal_search.py",
        ROOT / "scripts/train_natural_sentence_ranker.py",
        PROBE / "report.json", PROBE / "selection.json",
        PROBE / "cases.jsonl", PROBE / "option-audit.jsonl",
    )
    for path in paths:
        hashes[str(path.relative_to(ROOT))] = file_sha256(path)
    return dict(sorted(hashes.items()))


def run(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    report_path = PROBE / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError("completed acronym probe report is required")
    manifest, data = validate_data(PREPARED, PREPARATION_MANIFEST)
    if manifest.get("runtime_version") != RUNTIME_VERSION:
        raise ValueError("current preparation runtime differs")
    fixed = data["natural_train"][:PROFILE["rows"]]
    report = json.loads(report_path.read_text())
    option_rows = read_jsonl(PROBE / "option-audit.jsonl")
    selection = json.loads((PROBE / "selection.json").read_text())
    cases = read_jsonl(PROBE / "cases.jsonl")
    validate_probe(args.probe_report_sha256, report, option_rows, selection, cases, fixed)
    before = source_hashes()
    torch.set_num_threads(PROFILE["threads"])
    cache = CachedOptions(option_rows)
    converter = CaptureConverter(context_ranking=True, ranker_kind="hybrid")
    output_rows = []
    counts = Counter()
    missing_kinds = Counter()
    args.output.mkdir(parents=True)
    cases_path = args.output / "cases.jsonl"
    with cases_path.open("x") as stream:
        for index, row in enumerate(fixed):
            outcome = replay_case(converter, cache, row)
            output_rows.append(outcome)
            counts[outcome["classification"]] += 1
            missing_kinds.update(item["kind"] for item in outcome["missing_pieces"])
            stream.write(json.dumps(outcome, ensure_ascii=False, sort_keys=True) + "\n")
            elapsed = time.monotonic() - started
            if (rss_bytes() > PROFILE["rss_limit_bytes"]
                    or elapsed > PROFILE["time_limit_seconds"]):
                raise RuntimeError("cached lattice audit resource bound exceeded")
            if (index + 1) % PROFILE["progress_interval"] == 0:
                print(json.dumps({"event": "cached_lattice_audit", "rows": index + 1,
                                  "elapsed_seconds": round(elapsed, 2)}), flush=True)
    validate_data(PREPARED, PREPARATION_MANIFEST)
    validate_probe(args.probe_report_sha256, json.loads(report_path.read_text()),
                   read_jsonl(PROBE / "option-audit.jsonl"),
                   json.loads((PROBE / "selection.json").read_text()),
                   read_jsonl(PROBE / "cases.jsonl"), fixed)
    after = source_hashes()
    if before != after or cache.cache_misses:
        raise ValueError("audit sources changed or cached lookup failed")
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Model-free current natural TRAIN first512 cached-lattice audit",
        "profile": PROFILE,
        "runtime_version": RUNTIME_VERSION,
        "preparation_manifest_sha256": PREPARATION_MANIFEST,
        "probe_report_sha256": args.probe_report_sha256,
        "source_hashes": before,
        "classification_counts": dict(counts),
        "missing_piece_kind_counts": dict(missing_kinds),
        "pool_covered": sum(row["pool_covered"] is True for row in output_rows),
        "sentence_pool_rows": sum(row["pool_covered"] is not None for row in output_rows),
        "unknown_alignment_rows": sum(any(
            item["kind"] == "unaligned_boundary" for item in row["missing_pieces"]
        ) for row in output_rows),
        "cached_option_calls": cache.calls,
        "cache_misses": cache.cache_misses,
        "word_score_replay_calls": sum(row["word_score_replay_calls"] for row in output_rows),
        "actual_model_inference_calls": 0,
        "external_dictionary_reconstruction": "Fresh deterministic CandidateGenerator initialization",
        "cases_sha256": file_sha256(cases_path),
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": rss_bytes(),
        "natural_dev_inference_run": False,
        "development_inference_run": False,
        "test_data_read": False,
        "deployed": False,
    }
    (args.output / "report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({key: output[key] for key in (
        "classification_counts", "missing_piece_kind_counts", "pool_covered",
        "unknown_alignment_rows", "cache_misses", "actual_model_inference_calls",
        "elapsed_seconds", "peak_rss_bytes")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-report-sha256", required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
