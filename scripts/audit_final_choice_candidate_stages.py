"""Audit candidate-stage losses on the first 512 frozen natural FIT rows."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from hanglish.core.neural_converter import NeuralConverter, beam_paths, file_sha256
from scripts.audit_current_cached_lattice import classify, missing_pieces
from scripts.final_choice_capture import _response_flags, validate_response
from scripts.probe_proposal_search import can_compose
from scripts.train_final_choice_selector import load_fit_role


FIT = ROOT / "data/natural_context/final-choice-v1/natural_fit"
FIT_MANIFEST_SHA256 = "fc22b1635cc496ed70aa757750944726d583d640f2d9f3a598e380f3e262a1c9"
RUNTIME_VERSION = "neural-1-867f13596ed5"
OUTPUT = ROOT / ".omx/artifacts/final-choice-candidate-stages-v1"
PROFILE = {
    "name": "production-final-choice-natural-fit-first512-stage-audit-v1",
    "rows": 512, "threads": 2,
    "rss_limit_bytes": 3 * 1024**3, "time_limit_seconds": 1800,
    "progress_interval": 32,
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def require_resources(started):
    if (rss_bytes() > PROFILE["rss_limit_bytes"]
            or time.monotonic() - started > PROFILE["time_limit_seconds"]):
        raise RuntimeError("candidate-stage audit resource bound exceeded")


def unique_paths(paths, limit=48):
    result = {}
    for path in paths:
        result.setdefault(path[0], path)
        if len(result) == limit:
            break
    return list(result.values())


class StageConverter(NeuralConverter):
    def _prepare_candidates(self, *args, **kwargs):
        result = super()._prepare_candidates(*args, **kwargs)
        self.audit_pieces, self.audit_static_pieces = result[:2]
        return result


class SentenceRecorder:
    def __init__(self, delegate):
        self.delegate = delegate
        self.candidates = None
        self.scores = None

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def reset(self):
        self.candidates = None
        self.scores = None

    def score_pairs(self, contexts, candidates):
        if self.candidates is not None or self.scores is not None:
            raise RuntimeError("sentence ranker was invoked more than once for one conversion")
        if (not isinstance(contexts, list) or not isinstance(candidates, list)
                or len(contexts) != len(candidates) or not candidates
                or any(not isinstance(value, str) for value in candidates)):
            raise ValueError("sentence ranker received malformed candidate inputs")
        self.candidates = list(candidates)
        returned = self.delegate.score_pairs(contexts, candidates)
        if not isinstance(returned, (list, tuple)):
            raise RuntimeError("sentence ranker returned a malformed score vector")
        self.scores = list(returned)
        if (len(self.scores) != len(self.candidates)
                or any(type(score) not in (int, float) or not math.isfinite(score)
                       for score in self.scores)):
            raise RuntimeError("sentence ranker returned a malformed score vector")
        return list(self.scores)


def reconstruct_pool(converter, pieces, static_pieces, *, sentence_possible):
    learned = beam_paths(pieces)
    static = beam_paths(static_pieces)
    complete = unique_paths([*learned, *static])
    base_count = len(complete)
    all_fit = all(converter.sentence_ranker.can_score(path[0]) for path in complete)
    if sentence_possible and all_fit:
        extras = converter._numeric_sentence_extras(pieces, static_pieces, complete)
        if extras and all(converter.sentence_ranker.can_score(path[0]) for path in extras):
            complete.extend(extras)
    return learned, static, complete, base_count, all_fit


def actual_pre_display_paths(recorder, learned_paths):
    if recorder.candidates is None:
        return [path[0] for path in learned_paths]
    if recorder.scores is None or len(recorder.scores) != len(recorder.candidates):
        raise ValueError("observed sentence scores are incomplete")
    return [text for text, _ in sorted(
        zip(recorder.candidates, recorder.scores), key=lambda item: (-item[1], item[0])
    )]


def stage_reason(row, response, recorder, pieces, static_pieces, learned_paths,
                 reconstructed, base_count, base_all_fit):
    expected = row["expected"]
    top = [candidate["text"] for candidate in response["candidates"]]
    observed = recorder.candidates
    pool = observed if observed is not None else [path[0] for path in learned_paths]
    ranked = actual_pre_display_paths(recorder, learned_paths)
    raw_appended = response["candidates"][-1].get("score") is None
    displaced = (raw_appended and not any(value in expected for value in top)
                 and len(ranked) >= 5 and ranked[4] in expected)
    sentence_possible = len(beam_paths(pieces)) > 1 and len(row["input"].split()) > 1
    length_bypass = sentence_possible and observed is None and not base_all_fit
    lattice_reason = classify(expected, pieces, static_pieces,
                              [path[0] for path in reconstructed[:base_count]],
                              observed is not None)
    if any(value in expected for value in top):
        reason = "top5_covered"
    elif displaced:
        reason = "raw_slot_displacement"
    elif observed is not None and any(value in expected for value in observed):
        reason = "sentence_ranking_loss"
    elif length_bypass:
        reason = "length_bypass"
    elif lattice_reason == "missing_fullpiece_candidate":
        reason = "missing_piece"
    else:
        reason = lattice_reason
    return reason, pool, ranked, displaced, length_bypass, lattice_reason


def audit_case(converter, recorder, row):
    recorder.reset()
    response = converter.convert(row["input"], row["context"])
    validate_response(response, row["input"], row["context"], RUNTIME_VERSION,
                      row["response_flags"]["model_artifacts"], raw=False)
    if (response["candidates"] != row["candidates"]
            or response["result"] != row["result"]
            or _response_flags(response) != row["response_flags"]
            or response["version"] != row["runtime_version"]):
        raise ValueError("fresh production conversion differs from frozen natural FIT")
    pieces = converter.audit_pieces
    static = converter.audit_static_pieces
    sentence_possible = len(beam_paths(pieces)) > 1 and len(row["input"].split()) > 1
    learned_paths, static_paths, reconstructed, base_count, base_all_fit = reconstruct_pool(
        converter, pieces, static, sentence_possible=sentence_possible,
    )
    reconstructed_texts = [path[0] for path in reconstructed]
    if recorder.candidates is not None:
        if recorder.candidates != reconstructed_texts:
            raise ValueError("observed sentence pool differs from reconstructed runtime pool")
        if recorder.scores is None or len(recorder.scores) != len(recorder.candidates):
            raise ValueError("observed sentence scores are incomplete")

    # Labels are first consulted after conversion and source-only reconstruction.
    reason, pool, ranked, displaced, length_bypass, lattice_reason = stage_reason(
        row, response, recorder, pieces, static, learned_paths,
        reconstructed, base_count, base_all_fit,
    )
    expected = row["expected"]
    top = [candidate["text"] for candidate in response["candidates"]]
    missing = missing_pieces(row, pieces, static)
    return {
        "id": row["id"], "group_id": row["group_id"],
        "input": row["input"], "context": row["context"],
        "expected": list(expected), "result": response["result"],
        "top1": response["result"] in expected,
        "top5": any(value in expected for value in top),
        "pool_covered": (None if recorder.candidates is None
                         else any(value in expected for value in recorder.candidates)),
        "reconstructed_pool_covered": any(value in expected for value in reconstructed_texts),
        "classification": reason, "lattice_classification": lattice_reason,
        "raw_slot_displacement": displaced, "length_bypass": length_bypass,
        "learned_beam_size": len(learned_paths), "static_beam_size": len(static_paths),
        "base_pool_size": base_count, "reconstructed_pool_size": len(reconstructed),
        "observed_sentence_pool": recorder.candidates,
        "observed_sentence_scores": recorder.scores,
        "actual_pre_display_ranked": ranked,
        "learned_pieces": [[source, offset, [asdict(option) for option in options]]
                           for source, offset, options in pieces],
        "static_pieces": [[source, offset, [asdict(option) for option in options]]
                          for source, offset, options in static],
        "missing_pieces": missing,
        "fullpiece_composable": any(can_compose(value, pieces)
                                    or can_compose(value, static) for value in expected),
        "top8_composable": any(can_compose(value, pieces, 8)
                               or can_compose(value, static, 8) for value in expected),
    }


def source_hashes(loaded, runner_sha):
    hashes = dict(loaded["manifest"]["source_hashes"])
    additions = {
        "scripts/audit_final_choice_candidate_stages.py": runner_sha,
        "scripts/train_final_choice_selector.py": file_sha256(
            ROOT / "scripts/train_final_choice_selector.py"),
        "scripts/audit_current_cached_lattice.py": file_sha256(
            ROOT / "scripts/audit_current_cached_lattice.py"),
        "scripts/probe_proposal_search.py": file_sha256(
            ROOT / "scripts/probe_proposal_search.py"),
        str((FIT / "manifest.json").relative_to(ROOT)): FIT_MANIFEST_SHA256,
        str((FIT / "cases.jsonl").relative_to(ROOT)): sha256_bytes(loaded["cases_raw"]),
        str((FIT / "selection.jsonl").relative_to(ROOT)): sha256_bytes(loaded["selection_raw"]),
    }
    for name, digest in additions.items():
        if name in hashes and hashes[name] != digest:
            raise ValueError(f"candidate-stage source hash conflicts: {name}")
        hashes[name] = digest
    return dict(sorted(hashes.items()))


def verify_hashes(hashes):
    for name, digest in hashes.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if file_sha256(path) != digest:
            raise ValueError(f"candidate-stage audit input changed: {name}")


def run(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    runner_source = Path(__file__).read_bytes()
    loaded = load_fit_role(FIT, FIT_MANIFEST_SHA256, "natural_fit")
    rows = loaded["rows"][:PROFILE["rows"]]
    if len(rows) != PROFILE["rows"]:
        raise ValueError("natural FIT first512 is incomplete")
    hashes = source_hashes(loaded, sha256_bytes(runner_source))
    verify_hashes(hashes)
    torch.set_num_threads(PROFILE["threads"])
    os.environ["OMP_NUM_THREADS"] = str(PROFILE["threads"])
    converter = StageConverter(ranker_kind="hybrid", external_lexicon=True)
    converter.load_models()
    if (converter.version != RUNTIME_VERSION
            or converter.model_artifacts != loaded["manifest"]["model_artifact_hashes"]):
        raise ValueError("fresh converter differs from frozen production FIT runtime")
    recorder = SentenceRecorder(converter.sentence_ranker)
    converter.sentence_ranker = recorder
    counts = Counter()
    outcomes = []
    args.output.mkdir(parents=True)
    cases_path = args.output / "cases.jsonl"
    with cases_path.open("x") as stream:
        for index, row in enumerate(rows, 1):
            outcome = audit_case(converter, recorder, row)
            outcomes.append(outcome)
            counts[outcome["classification"]] += 1
            stream.write(json.dumps(outcome, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            require_resources(started)
            if index % PROFILE["progress_interval"] == 0:
                print(json.dumps({"event": "candidate_stage_audit", "rows": index}), flush=True)
    verify_hashes(hashes)
    if (converter.version != RUNTIME_VERSION
            or converter.model_artifacts != loaded["manifest"]["model_artifact_hashes"]
            or Path(__file__).read_bytes() != runner_source):
        raise RuntimeError("candidate-stage runtime or source changed")
    report = {
        "schema": "production-final-choice-natural-fit-stage-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "completed": True,
        "scope": "current production natural FIT first512 candidate-stage audit",
        "profile": PROFILE, "runtime_version": RUNTIME_VERSION,
        "fit_manifest_sha256": FIT_MANIFEST_SHA256,
        "source_hashes": hashes, "source_hashes_after": hashes,
        "cases_sha256": file_sha256(cases_path),
        "classification_counts": dict(sorted(counts.items())),
        "top1": sum(row["top1"] for row in outcomes),
        "top5": sum(row["top5"] for row in outcomes),
        "observed_sentence_pool_rows": sum(row["pool_covered"] is not None for row in outcomes),
        "observed_sentence_pool_covered": sum(row["pool_covered"] is True for row in outcomes),
        "reconstructed_pool_covered": sum(row["reconstructed_pool_covered"] for row in outcomes),
        "raw_slot_displacements": sum(row["raw_slot_displacement"] for row in outcomes),
        "length_bypasses": sum(row["length_bypass"] for row in outcomes),
        "sample_train_errors": [{"id": row["id"], "input": row["input"],
                                 "expected": row["expected"],
                                 "classification": row["classification"]}
                                for row in outcomes if not row["top1"]][:10],
        "labels_consulted_after_inference": True,
        "fresh_conversion_calls": PROFILE["rows"],
        "actual_model_inference_run": True,
        "additional_model_inference_calls": 0,
        "evaluation_data_read": False, "development_data_read": False,
        "final_data_read": False, "deployed": False,
        "elapsed_seconds": time.monotonic() - started, "peak_rss_bytes": rss_bytes(),
    }
    (args.output / "source.py").write_bytes(runner_source)
    require_resources(started)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"classification_counts": report["classification_counts"],
                      "top1": report["top1"], "top5": report["top5"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
