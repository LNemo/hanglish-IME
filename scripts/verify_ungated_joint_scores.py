"""Independently replay frozen SentenceRanker scores for ungated joint FIT pools."""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import score_ungated_joint_fit as producer


ARMS = ("original", "roman_repaired")
LIMITS = {"whole_seconds": 600.0, "row_seconds": 30.0,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
ASCII = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
EXPECTED_TOTAL = {"rows": 13, "unique_supported_scored": 2065,
                  "unique_unsupported_unscored": 1301}
EXPECTED_ARMS = {
    "original": {"rows": 13, "pool": 2548, "supported": 1440,
                 "unsupported": 1108, "fallback_rows": 3},
    "roman_repaired": {"rows": 13, "pool": 2548, "supported": 1573,
                       "unsupported": 975, "fallback_rows": 3},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def iter_jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            require(bool(line.strip()), f"blank row in {path.name}")
            yield json.loads(line)


def swap_bytes() -> int:
    result = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "vm.swapusage"], capture_output=True,
        text=True, check=True)
    match = re.search(r"used\s*=\s*([0-9.]+)M", result.stdout)
    require(match is not None, "swap telemetry unavailable")
    return round(float(match.group(1)) * 1024**2)


class ResourceGuard:
    def __init__(self, swap_reader=swap_bytes):
        self.started = time.monotonic()
        self.baseline_swap = swap_reader()
        self.swap_reader = swap_reader
        self.peak_rss = 0
        self.peak_swap_growth = 0

    def observe(self, row_started=None):
        elapsed = time.monotonic() - self.started
        row_elapsed = None if row_started is None else time.monotonic() - row_started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss *= 1 if sys.platform == "darwin" else 1024
        growth = self.swap_reader() - self.baseline_swap
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_swap_growth = max(self.peak_swap_growth, growth)
        require(elapsed <= LIMITS["whole_seconds"]
                and (row_elapsed is None or row_elapsed <= LIMITS["row_seconds"])
                and rss <= LIMITS["rss_bytes"]
                and growth <= LIMITS["swap_growth_bytes"],
                "independent score replay resource limit exceeded")
        return {"seconds": elapsed, "row_seconds": row_elapsed,
                "rss_bytes": rss, "swap_growth_bytes": growth}


def token_count(ranker, text: str) -> int:
    require(isinstance(text, str) and text, "candidate text differs")
    normalized = " ".join(text.strip().split())
    encoded = ranker.tokenizer(
        normalized, add_special_tokens=False, truncation=False)
    input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    require(isinstance(input_ids, list)
            and all(type(value) is int for value in input_ids),
            "untruncated tokenizer output differs")
    reserved = ranker.tokenizer.num_special_tokens_to_add(pair=True)
    require(type(reserved) is int and reserved >= 0,
            "pair special-token count differs")
    return len(input_ids) + reserved


def _group(source, provenance: str, expected: int, ranker):
    require(isinstance(source, list) and len(source) == expected,
            "source candidate group count differs")
    result, seen = [], set()
    for candidate in source:
        require(isinstance(candidate, dict), "source candidate differs")
        text = candidate.get("text")
        require(isinstance(text, str) and text and text not in seen,
                "source candidate text uniqueness differs")
        tokens = token_count(ranker, text)
        support = "supported" if tokens <= ranker.max_length else "unsupported"
        require(candidate.get("sentence_tokens") == tokens
                and candidate.get("sentence_support") == support,
                "source candidate token support differs")
        result.append({"text": text, "support": support,
                       "provenance": provenance})
        seen.add(text)
    return result


def _pool(groups):
    values, order = {}, []
    for group in groups:
        for candidate in group:
            text = candidate["text"]
            if text not in values:
                values[text] = {"text": text, "support": candidate["support"],
                                "provenance": [candidate["provenance"]]}
                order.append(text)
            else:
                require(values[text]["support"] == candidate["support"],
                        "duplicate support differs")
                if candidate["provenance"] not in values[text]["provenance"]:
                    values[text]["provenance"].append(candidate["provenance"])
    return [values[text] for text in order]


def reconstruct(source_parent, source_joint, ranker):
    require(isinstance(source_parent, dict) and isinstance(source_joint, dict)
            and source_parent.get("id") == source_joint.get("id")
            and source_parent.get("request") == source_joint.get("request")
            and source_parent.get("request", {}).get("context") == ""
            and set(source_joint.get("arms", {})) == set(ARMS),
            "ordered source identity differs")
    fallback = source_parent.get("fallback_public")
    require(isinstance(fallback, list) and len(fallback) == 5,
            "five-slot fallback differs")
    raw_source = source_parent["request"]["input"]
    raw = [candidate for candidate in fallback
           if candidate.get("text") == raw_source
           and isinstance(candidate.get("segments"), list)
           and candidate["segments"]
           and all(segment.get("kind") == "raw"
                   for segment in candidate["segments"])]
    require(len(raw) == 1 and fallback[-1] == raw[0],
            "exact raw reserve differs")
    parents = _group(source_parent.get("parents"), "parent", 4, ranker)
    singles = _group(source_parent.get("proposals"), "single", 96, ranker)
    arms = {}
    for arm in ARMS:
        proposals = _group(
            source_joint["arms"][arm].get("proposals"),
            "joint_" + arm, 96, ranker)
        arms[arm] = _pool((parents, singles, proposals))
    return {"id": source_parent["id"], "request": source_parent["request"],
            "fallback_public": deepcopy(fallback), "raw_reserve": deepcopy(raw[0]),
            "arms": arms}


def recreate_scored_row(source_parent, source_joint, ranker):
    projected = reconstruct(source_parent, source_joint, ranker)
    union, support_by_text = [], {}
    for arm in ARMS:
        for candidate in projected["arms"][arm]:
            text, support = candidate["text"], candidate["support"]
            if text not in support_by_text:
                union.append(text)
                support_by_text[text] = support
            else:
                require(support_by_text[text] == support,
                        "cross-arm support differs")
    supported = [text for text in union if support_by_text[text] == "supported"]
    unsupported = [text for text in union if support_by_text[text] == "unsupported"]
    scores = ranker.score_pairs([""] * len(supported), supported) if supported else []
    require(len(scores) == len(supported)
            and all(not isinstance(score, bool) and isinstance(score, (int, float))
                    and math.isfinite(score) for score in scores),
            "fresh sentence logits differ")
    score_by_text = {text: float(score) for text, score in zip(supported, scores)}
    arms = {}
    for arm in ARMS:
        pool = projected["arms"][arm]
        supported_order = sorted(
            (candidate["text"] for candidate in pool
             if candidate["support"] == "supported"),
            key=lambda text: (-score_by_text[text], text))
        arm_unsupported = [candidate["text"] for candidate in pool
                           if candidate["support"] == "unsupported"]
        if supported_order:
            public = [{"text": text, "sentence_score": score_by_text[text]}
                      for text in supported_order[:4]]
            public.append(deepcopy(projected["raw_reserve"]))
            fallback_used = False
        else:
            public = deepcopy(projected["fallback_public"])
            fallback_used = True
        arms[arm] = {"pool": deepcopy(pool), "supported_order": supported_order,
                     "unsupported_texts": arm_unsupported, "public": public,
                     "fallback_used": fallback_used}
    return {"id": projected["id"], "request": projected["request"],
            "fallback_public": deepcopy(projected["fallback_public"]),
            "supported_scores": [{"text": text, "score": score_by_text[text]}
                                 for text in supported],
            "unsupported_texts": unsupported, "arms": arms}


def _close(left, right):
    return (not isinstance(left, bool) and isinstance(left, (int, float))
            and not isinstance(right, bool) and isinstance(right, (int, float))
            and math.isfinite(left) and math.isfinite(right)
            and math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6))


def compare_scored(saved, fresh):
    require(isinstance(saved, dict) and set(saved) == set(fresh)
            and saved["id"] == fresh["id"]
            and saved["request"] == fresh["request"]
            and saved["fallback_public"] == fresh["fallback_public"]
            and saved["unsupported_texts"] == fresh["unsupported_texts"],
            "captured score row identity differs")
    require([row.get("text") for row in saved.get("supported_scores", [])]
            == [row["text"] for row in fresh["supported_scores"]],
            "captured supported score order differs")
    require(len(saved["supported_scores"]) == len(fresh["supported_scores"])
            and all(_close(actual.get("score"), expected["score"])
                    for actual, expected in zip(saved["supported_scores"],
                                                fresh["supported_scores"])),
            "captured sentence logits differ")
    for arm in ARMS:
        actual, expected = saved["arms"].get(arm), fresh["arms"][arm]
        require(isinstance(actual, dict) and set(actual) == set(expected)
                and actual["pool"] == expected["pool"]
                and actual["supported_order"] == expected["supported_order"]
                and actual["unsupported_texts"] == expected["unsupported_texts"]
                and actual["fallback_used"] is expected["fallback_used"]
                and len(actual["public"]) == len(expected["public"]),
                "captured arm pool or public structure differs")
        for saved_public, fresh_public in zip(actual["public"], expected["public"]):
            if "sentence_score" in fresh_public:
                require(set(saved_public) == {"text", "sentence_score"}
                        and saved_public["text"] == fresh_public["text"]
                        and _close(saved_public["sentence_score"],
                                   fresh_public["sentence_score"]),
                        "captured public score differs")
            else:
                require(saved_public == fresh_public,
                        "captured raw reserve or fallback differs")


def ascii_words(text):
    return tuple(ASCII.findall(text))


def metric_row(scored, label):
    require(scored.get("id") == label.get("id")
            and scored.get("request") == {"input": label.get("input"), "context": ""}
            and isinstance(label.get("expected"), list) and label["expected"],
            "FIT identity differs")
    gold = set(label["expected"])
    ascii_references = {ascii_words(text) for text in gold}
    has_ascii = any(ascii_references)
    fallback = [candidate["text"] for candidate in scored["fallback_public"]]
    parent_supported = {candidate["text"]
                        for candidate in scored["arms"]["original"]["pool"]
                        if "parent" in candidate["provenance"]
                        and candidate["support"] == "supported"}

    def branch(public, complete_pool, supported_pool):
        return {"first": public[0] in gold, "top5": bool(set(public) & gold),
                "all_pool_covered": bool(set(complete_pool) & gold),
                "scorable_pool_covered": bool(set(supported_pool) & gold),
                "ascii_reference": bool(has_ascii),
                "ascii_exact": bool(has_ascii
                                    and ascii_words(public[0]) in ascii_references),
                "public": public}

    branches = {"captured_fallback": branch(fallback, fallback, parent_supported)}
    for arm in ARMS:
        record = scored["arms"][arm]
        branches[arm] = branch(
            [candidate["text"] for candidate in record["public"]],
            [candidate["text"] for candidate in record["pool"]],
            record["supported_order"])
    pairs = {"fallback_to_original": ("captured_fallback", "original"),
             "fallback_to_roman_repaired": ("captured_fallback", "roman_repaired"),
             "original_to_roman_repaired": ("original", "roman_repaired")}
    transitions = {}
    for name, (before, after) in pairs.items():
        left, right = branches[before], branches[after]
        transitions[name] = {
            "corrections": int(not left["first"] and right["first"]),
            "regressions": int(left["first"] and not right["first"]),
            "top5_gains": int(not left["top5"] and right["top5"]),
            "top5_losses": int(left["top5"] and not right["top5"]),
            "all_pool_gains": int(not left["all_pool_covered"]
                                  and right["all_pool_covered"]),
            "all_pool_losses": int(left["all_pool_covered"]
                                   and not right["all_pool_covered"]),
            "scorable_pool_gains": int(not left["scorable_pool_covered"]
                                       and right["scorable_pool_covered"]),
            "scorable_pool_losses": int(left["scorable_pool_covered"]
                                        and not right["scorable_pool_covered"]),
            "ascii_gains": int(not left["ascii_exact"] and right["ascii_exact"]),
            "ascii_regressions": int(left["ascii_exact"] and not right["ascii_exact"]),
        }
    return {"id": scored["id"], "branches": branches,
            "transitions": transitions}


def aggregate(outcomes):
    branch_fields = ("first", "top5", "all_pool_covered", "scorable_pool_covered",
                     "ascii_reference", "ascii_exact")
    branches = {name: Counter(rows=0, **{field: 0 for field in branch_fields})
                for name in ("captured_fallback", *ARMS)}
    transition_names = ("fallback_to_original", "fallback_to_roman_repaired",
                        "original_to_roman_repaired")
    transitions = {name: Counter() for name in transition_names}
    for outcome in outcomes:
        for name, values in outcome["branches"].items():
            branches[name]["rows"] += 1
            for field in branch_fields:
                branches[name][field] += int(values[field])
        for name, values in outcome["transitions"].items():
            transitions[name].update(values)
    return {"branches": {name: dict(values) for name, values in branches.items()},
            "transitions": {name: dict(values)
                            for name, values in transitions.items()}}


def load_labels_after_scores(path: Path, expected_sha: str, *, scores_verified: bool):
    require(scores_verified, "FIT labels cannot open before score replay verification")
    require(digest(path) == expected_sha, "FIT labels changed")
    labels = {}
    for row in iter_jsonl(path):
        require(row.get("id") not in labels, "duplicate FIT label")
        labels[row["id"]] = row
    require(len(labels) == 1642, "FIT label denominator differs")
    return labels


def _load_ranker():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    from hanglish.models.sentence_ranker import SentenceRanker
    ranker = SentenceRanker.load(producer.CHECKPOINT)
    require(ranker.max_length == 96 and ranker.batch_size == 8
            and all(parameter.device.type == "cpu"
                    for parameter in ranker.model.parameters()),
            "frozen CPU sentence-ranker settings differ")
    return ranker


def verify(directory: Path):
    directory = directory.resolve()
    proof_path = directory / "independent-verification.json"
    require(directory.is_dir() and not proof_path.exists(),
            "score directory missing or independent proof already exists")
    report_path = directory / "report.json"
    preliminary = read_json(report_path)
    joint = Path(preliminary.get("joint_directory", ""))
    report, closure = producer.authenticate_scoring(joint, directory)
    frozen = {name: digest(directory / name) for name in
              ("report.json", "capture.complete.json", "capture.jsonl",
               "resources.jsonl")}
    require(report.get("counts", {}).get("total") == EXPECTED_TOTAL
            and report.get("counts", {}).get("arms") == EXPECTED_ARMS,
            "reported score denominators differ")
    guard = ResourceGuard()
    guard.observe()
    ranker = _load_ranker()
    guard.observe()

    totals = Counter(rows=0, unique_supported_scored=0,
                     unique_unsupported_unscored=0)
    arm_counts = {arm: Counter(rows=0, pool=0, supported=0,
                               unsupported=0, fallback_rows=0)
                  for arm in ARMS}
    fresh_rows, ordered_ids = [], []
    with (producer.PARENT / "capture.jsonl").open() as parents, \
            (joint / "capture.jsonl").open() as joints, \
            (directory / "capture.jsonl").open() as saved_rows:
        for row_index in range(13):
            row_started = time.monotonic()
            lines = (next(parents, None), next(joints, None), next(saved_rows, None))
            require(all(line is not None for line in lines), "score replay row missing")
            parent, joint_row, saved = (json.loads(line) for line in lines)
            fresh = recreate_scored_row(parent, joint_row, ranker)
            compare_scored(saved, fresh)
            require(fresh["id"] not in ordered_ids and saved["id"] == fresh["id"]
                    and saved["request"] == parent["request"] == joint_row["request"],
                    "ordered score ID or request differs")
            ordered_ids.append(fresh["id"])
            fresh_rows.append(fresh)
            totals.update(rows=1,
                          unique_supported_scored=len(fresh["supported_scores"]),
                          unique_unsupported_unscored=len(fresh["unsupported_texts"]))
            for arm in ARMS:
                arm_row = fresh["arms"][arm]
                arm_counts[arm].update(
                    rows=1, pool=len(arm_row["pool"]),
                    supported=len(arm_row["supported_order"]),
                    unsupported=len(arm_row["unsupported_texts"]),
                    fallback_rows=int(arm_row["fallback_used"]))
            guard.observe(row_started)
        require(next(parents, None) is None and next(joints, None) is None
                and next(saved_rows, None) is None,
                "score replay has extra rows")
    require(len(ordered_ids) == len(set(ordered_ids)) == 13
            and dict(totals) == EXPECTED_TOTAL == report["counts"]["total"]
            and {arm: dict(arm_counts[arm]) for arm in ARMS}
                == EXPECTED_ARMS == report["counts"]["arms"],
            "fresh score counts differ")
    scoring_report, scoring_closure = producer.authenticate_scoring(joint, directory)
    require(scoring_report == report and scoring_closure == closure
            and all(digest(directory / name) == expected
                    for name, expected in frozen.items()),
            "score artifacts changed during replay")
    scores_verified = True

    labels = load_labels_after_scores(
        producer.LABELS, producer.LABEL_SHA, scores_verified=scores_verified)
    saved_outcomes = list(iter_jsonl(directory / "outcomes.jsonl"))
    outcomes = []
    for fresh, saved in zip(fresh_rows, saved_outcomes):
        require(fresh["id"] in labels, "selected FIT label missing")
        outcome = metric_row(fresh, labels[fresh["id"]])
        require(outcome == saved, "independent FIT outcome differs")
        outcomes.append(outcome)
    require(len(saved_outcomes) == len(outcomes) == 13,
            "FIT outcome denominator differs")
    summary = aggregate(outcomes)
    evaluation_path = directory / "evaluation.json"
    evaluation = read_json(evaluation_path)
    require(evaluation.get("completed") is True and evaluation.get("rows") == 13
            and evaluation.get("summary") == summary
            and evaluation.get("score_report_sha256") == frozen["report.json"]
            and evaluation.get("score_closure_sha256")
                == frozen["capture.complete.json"]
            and evaluation.get("score_capture_sha256") == frozen["capture.jsonl"]
            and evaluation.get("outcomes_sha256") == digest(directory / "outcomes.jsonl")
            and evaluation.get("labels_sha256") == producer.LABEL_SHA
            and evaluation.get("labels_opened_after_source_only_closure") is True
            and evaluation.get("neural_training") is False
            and evaluation.get("generation_inference") is False
            and evaluation.get("dev_opened") is False
            and evaluation.get("test_opened") is False
            and evaluation.get("production_changed") is False,
            "FIT evaluation report differs")
    final = guard.observe()
    closing_report, closing_closure = producer.authenticate_scoring(joint, directory)
    require(closing_report == report and closing_closure == closure
            and digest(producer.LABELS) == producer.LABEL_SHA
            and all(digest(directory / name) == expected
                    for name, expected in frozen.items()),
            "verified evidence changed")
    result = {
        "schema": "ungated-joint-sentence-score-independent-verification-v1",
        "verified": True, "source_closure_verified": True,
        "scores_verified": True, "evaluation_verified": True,
        "rows": 13, "ordered_ids": ordered_ids,
        "counts": {"total": dict(totals),
                   "arms": {arm: dict(arm_counts[arm]) for arm in ARMS}},
        "summary": summary,
        "score_report_sha256": frozen["report.json"],
        "score_closure_sha256": frozen["capture.complete.json"],
        "score_capture_sha256": frozen["capture.jsonl"],
        "score_resources_sha256": frozen["resources.jsonl"],
        "evaluation_sha256": digest(evaluation_path),
        "outcomes_sha256": digest(directory / "outcomes.jsonl"),
        "labels_sha256": producer.LABEL_SHA,
        "verifier_sha256": digest(Path(__file__).resolve()),
        "settings": {"max_length": 96, "batch_size": 8,
                     "threads": 1, "interop_threads": 1,
                     "external_context": "", "score_tolerance": 1e-6},
        "resources": {"seconds": final["seconds"],
                      "max_rss_bytes": guard.peak_rss,
                      "max_swap_growth_bytes": guard.peak_swap_growth,
                      "limits": LIMITS},
        "labels_opened_after_score_replay": True,
        "unsupported_candidates_retained": True,
        "no_truncation": True, "dev_opened": False, "test_opened": False,
        "production_changed": False,
        "scope": "Independent tokenizer support, 2,065-logit replay, public rendering, and FIT metric verification on 13 selected rows.",
    }
    with proof_path.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)
