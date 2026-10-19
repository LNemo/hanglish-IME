"""Score closed ungated joint FIT pools, then evaluate after source-only closure."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PARENT = ROOT / ".omx/artifacts/ungated-phonetic-fit-v1"
CHECKPOINT = ROOT / "checkpoints/sentence_ranker"
PLAN = ROOT / ".omx/plans/ungated-joint-score-fit-v1.md"
LABELS = ROOT / ".omx/artifacts/current-phonetic-train-features-v1/rows.jsonl"
LABEL_SHA = "4c13063656d9a108316b9ce1ceec394a61e2c45c2ba86e7a8c8361f7480d6cb4"
ARMS = ("original", "roman_repaired")
LIMITS = {"whole_seconds": 600.0, "row_seconds": 30.0,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 512 * 1024**2,
          "output_bytes": 64 * 1024**2}
MODEL_FILES = {
    "config.json": "8703d582f1e16300ab44e501e4a131b15f0a133b7b74bbcfb51e7e53009fe05a",
    "metadata.json": "8a6a1f9736fab0173003751e70c5e7ef0fa139a0e773444986ede044e0e55d14",
    "model.safetensors": "a68f3bd26733e9bc897a182f78b1decafefa3582fa82b6b2cfcaa34672412ffa",
    "special_tokens_map.json": "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
    "tokenizer.json": "5ee39bd824ba876371d68c5e958a8169c29b29060b8db19e6269403f3d3c7999",
    "tokenizer_config.json": "5a85ad076d4ae965ccff1888927079a924e3ffe401be44da855ff08a0ad0c45d",
    "vocab.txt": "6e886927dfcecd22029b1ba80c10a1374740259c1067fc3a28d964b7ae2d55a7",
}
ASCII = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def verify_pins(pins):
    need(isinstance(pins, dict) and bool(pins), "source pins missing")
    for path, expected in pins.items():
        need(isinstance(path, str) and isinstance(expected, str) and len(expected) == 64
             and sha(Path(path)) == expected, "source pin changed: " + str(path))


def model_pins():
    actual = {path.name for path in CHECKPOINT.iterdir() if path.is_file()}
    need(actual == set(MODEL_FILES), "sentence-ranker bundle differs")
    pins = {}
    for name, expected in MODEL_FILES.items():
        path = CHECKPOINT / name
        need(sha(path) == expected, "sentence-ranker artifact changed: " + name)
        pins[str(path.resolve())] = expected
    return pins


def authenticate_sources(joint: Path):
    parent_report_path = PARENT / "report.json"
    parent_proof_path = PARENT / "verification.json"
    parent_report, parent_proof = read(parent_report_path), read(parent_proof_path)
    need(parent_report.get("completed") is True and parent_report.get("rows") == 13
         and parent_report.get("labels_opened") is False
         and parent_proof.get("verified") is True and parent_proof.get("rows") == 13
         and parent_proof.get("report_sha256") == sha(parent_report_path)
         and parent_proof.get("capture_sha256") == parent_report.get("capture_sha256")
             == sha(PARENT / "capture.jsonl")
         and parent_proof.get("resources_sha256") == parent_report.get("resources_sha256")
             == sha(PARENT / "resources.jsonl"),
         "closed verified parent capture required")
    verify_pins(parent_report["pins"])

    joint_report_path = joint / "report.json"
    joint_proof_path = joint / "verification.json"
    joint_report, joint_proof = read(joint_report_path), read(joint_proof_path)
    need(joint_report.get("schema") == "ungated-joint-fit-v1"
         and joint_report.get("completed") is True and joint_report.get("rows") == 13
         and joint_report.get("labels_opened") is False
         and joint_report.get("model_inference") is False
         and joint_report.get("production_changed") is False
         and joint_report.get("repair_mixed") is True
         and joint_report.get("prioritize_beams") is True
         and joint_proof.get("verified") is True and joint_proof.get("rows") == 13
         and joint_proof.get("labels_opened") is False
         and joint_proof.get("model_inference") is False
         and joint_proof.get("report_sha256") == sha(joint_report_path)
         and joint_proof.get("capture_sha256") == joint_report.get("capture_sha256")
             == sha(joint / "capture.jsonl"),
         "closed verified ranked-joint capture required")
    verify_pins(joint_report["pins"])
    for arm in ARMS:
        counts = joint_report.get("counts", {}).get(arm, {})
        need(counts.get("rows") == 13 and counts.get("proposals") == 1248
             and counts.get("supported", 0) + counts.get("unsupported", 0) == 1248,
             "joint arm count differs")

    pins = {**parent_report["pins"], **joint_report["pins"], **model_pins()}
    for path in (parent_report_path, parent_proof_path, PARENT / "capture.jsonl",
                 PARENT / "resources.jsonl", joint_report_path, joint_proof_path,
                 joint / "capture.jsonl", Path(__file__).resolve(), PLAN):
        pins[str(path.resolve())] = sha(path)
    verify_pins(pins)
    return parent_report, joint_report, pins


def _candidate(candidate, provenance):
    need(isinstance(candidate, dict) and isinstance(candidate.get("text"), str)
         and candidate["text"] and candidate.get("sentence_support") in
             {"supported", "unsupported"}, "candidate support record differs")
    return {"text": candidate["text"], "support": candidate["sentence_support"],
            "provenance": provenance}


def _group(candidates, provenance, expected):
    need(isinstance(candidates, list) and len(candidates) == expected
         and len({candidate.get("text") for candidate in candidates}) == expected,
         "candidate group count or uniqueness differs")
    return [_candidate(candidate, provenance) for candidate in candidates]


def _pool(groups):
    by_text, order = {}, []
    for group in groups:
        for candidate in group:
            text = candidate["text"]
            if text not in by_text:
                by_text[text] = {"text": text, "support": candidate["support"],
                                 "provenance": [candidate["provenance"]]}
                order.append(text)
            else:
                need(by_text[text]["support"] == candidate["support"],
                     "duplicate candidate has inconsistent support")
                if candidate["provenance"] not in by_text[text]["provenance"]:
                    by_text[text]["provenance"].append(candidate["provenance"])
    return [by_text[text] for text in order]


def source_projection(parent, joint):
    need(parent.get("id") == joint.get("id")
         and parent.get("request") == joint.get("request")
         and parent.get("request", {}).get("context") == ""
         and set(joint.get("arms", {})) == set(ARMS), "source row identity differs")
    fallback = parent.get("fallback_public")
    need(isinstance(fallback, list) and len(fallback) == 5, "captured fallback must contain five slots")
    source = parent["request"]["input"]
    raw = [candidate for candidate in fallback
           if candidate.get("text") == source and candidate.get("segments")
           and all(segment.get("kind") == "raw" for segment in candidate["segments"])]
    need(len(raw) == 1 and fallback[-1] == raw[0], "canonical raw reserve differs")
    parents = _group(parent.get("parents"), "parent", 4)
    singles = _group(parent.get("proposals"), "single", 96)
    arms = {}
    for arm in ARMS:
        proposals = _group(joint["arms"][arm].get("proposals"), "joint_" + arm, 96)
        arms[arm] = _pool((parents, singles, proposals))
    return {"id": parent["id"], "request": parent["request"],
            "fallback_public": deepcopy(fallback), "raw_reserve": deepcopy(raw[0]),
            "arms": arms}


def score_row(projected, ranker):
    union, statuses = [], {}
    for arm in ARMS:
        for candidate in projected["arms"][arm]:
            text, support = candidate["text"], candidate["support"]
            if text in statuses:
                need(statuses[text] == support, "cross-arm support status differs")
            else:
                statuses[text] = support
                union.append(text)
    for text in union:
        actual = "supported" if ranker.can_score(text) else "unsupported"
        need(actual == statuses[text], "frozen tokenizer support differs")
    supported = [text for text in union if statuses[text] == "supported"]
    unsupported = [text for text in union if statuses[text] == "unsupported"]
    scores = ranker.score_pairs([""] * len(supported), supported) if supported else []
    need(len(scores) == len(supported)
         and all(isinstance(score, (int, float)) and not isinstance(score, bool)
                 and math.isfinite(score) for score in scores),
         "sentence-ranker scores differ")
    score_by_text = {text: float(score) for text, score in zip(supported, scores)}
    arms = {}
    for arm in ARMS:
        pool = projected["arms"][arm]
        supported_order = sorted(
            (candidate["text"] for candidate in pool if candidate["support"] == "supported"),
            key=lambda text: (-score_by_text[text], text))
        unsupported_texts = [candidate["text"] for candidate in pool
                             if candidate["support"] == "unsupported"]
        if supported_order:
            public = [{"text": text, "sentence_score": score_by_text[text]}
                      for text in supported_order[:4]]
            public.append(deepcopy(projected["raw_reserve"]))
            fallback_used = False
        else:
            public = deepcopy(projected["fallback_public"])
            fallback_used = True
        arms[arm] = {
            "pool": deepcopy(pool), "supported_order": supported_order,
            "unsupported_texts": unsupported_texts, "public": public,
            "fallback_used": fallback_used,
        }
    return {
        "id": projected["id"], "request": projected["request"],
        "fallback_public": deepcopy(projected["fallback_public"]),
        "supported_scores": [{"text": text, "score": score_by_text[text]} for text in supported],
        "unsupported_texts": unsupported, "arms": arms,
    }


def _output_size(directory):
    return sum(path.stat().st_size for path in directory.iterdir() if path.is_file())


def capture(joint: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    from scripts.train_character_edge_full_fit import ResourceGuard
    guard = ResourceGuard(output)

    def observe(stage):
        value = guard.observe(stage)
        need(value["seconds"] <= LIMITS["whole_seconds"]
             and value["rss_bytes"] <= LIMITS["rss_bytes"]
             and value["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"],
             "scoring resource limit exceeded")
        need(_output_size(output) <= LIMITS["output_bytes"], "scoring output limit exceeded")
        return value

    try:
        parent_report, joint_report, pins = authenticate_sources(joint)
        observe("authenticated")
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        from hanglish.models.sentence_ranker import SentenceRanker
        ranker = SentenceRanker.load(CHECKPOINT)
        need(ranker.max_length == 96 and ranker.batch_size == 8,
             "sentence-ranker settings differ")
        observe("model-loaded")
        counts = {arm: Counter(rows=0, pool=0, supported=0, unsupported=0,
                               fallback_rows=0) for arm in ARMS}
        totals = Counter(rows=0, unique_supported_scored=0, unique_unsupported_unscored=0)
        with (PARENT / "capture.jsonl").open() as parent_stream, \
                (joint / "capture.jsonl").open() as joint_stream, \
                (output / "capture.jsonl").open("x") as target:
            for row_index in range(13):
                started = time.monotonic()
                parent_line, joint_line = next(parent_stream, None), next(joint_stream, None)
                need(parent_line is not None and joint_line is not None, "source capture missing row")
                projected = source_projection(json.loads(parent_line), json.loads(joint_line))
                result = score_row(projected, ranker)
                payload = json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
                need(_output_size(output) + len(payload.encode())
                     <= LIMITS["output_bytes"], "scoring row exceeds output limit")
                target.write(payload)
                target.flush()
                totals["rows"] += 1
                totals["unique_supported_scored"] += len(result["supported_scores"])
                totals["unique_unsupported_unscored"] += len(result["unsupported_texts"])
                for arm in ARMS:
                    record = result["arms"][arm]
                    counts[arm]["rows"] += 1
                    counts[arm]["pool"] += len(record["pool"])
                    counts[arm]["supported"] += len(record["supported_order"])
                    counts[arm]["unsupported"] += len(record["unsupported_texts"])
                    counts[arm]["fallback_rows"] += int(record["fallback_used"])
                observe("row-" + result["id"])
                need(time.monotonic() - started <= LIMITS["row_seconds"],
                     "single scoring row exceeded time limit")
            need(next(parent_stream, None) is None and next(joint_stream, None) is None,
                 "source capture has extra rows")
        verify_pins(pins)
        final = observe("complete")
        report = {
            "schema": "ungated-joint-sentence-scoring-v1", "completed": True, "rows": 13,
            "counts": {"total": dict(totals),
                       "arms": {arm: dict(counts[arm]) for arm in ARMS}},
            "joint_directory": str(joint.resolve()), "parent_directory": str(PARENT.resolve()),
            "source_report_sha256": sha(joint / "report.json"),
            "source_proof_sha256": sha(joint / "verification.json"),
            "parent_report_sha256": sha(PARENT / "report.json"),
            "parent_proof_sha256": sha(PARENT / "verification.json"),
            "pins": pins, "limits": LIMITS, "resources": final,
            "max_rss_bytes": guard.peak_rss, "max_swap_growth_bytes": guard.peak_swap_growth,
            "hashes": {"capture.jsonl": sha(output / "capture.jsonl"),
                       "resources.jsonl": sha(output / "resources.jsonl")},
            "settings": {"max_length": 96, "batch_size": 8, "threads": 1,
                         "interop_threads": 1, "external_context": ""},
            "model": {"class": "SentenceRanker", "format_version": 1,
                      "checkpoint": str(CHECKPOINT.resolve()),
                      "model_sha256": MODEL_FILES["model.safetensors"]},
            "labels_opened": False, "neural_inference": True, "generation_inference": False,
            "neural_training": False, "dev_opened": False, "test_opened": False,
            "production_changed": False,
            "scope": "13 source-selected FIT rows; frozen old SentenceRanker selection only.",
        }
        write(output / "report.json", report)
        need(_output_size(output) <= LIMITS["output_bytes"], "final scoring output limit exceeded")
        write(output / "capture.complete.json", {
            "completed": True, "rows": 13,
            "capture_sha256": report["hashes"]["capture.jsonl"],
            "resources_sha256": report["hashes"]["resources.jsonl"],
            "report_sha256": sha(output / "report.json"), "labels_opened": False,
        })
        need(_output_size(output) <= LIMITS["output_bytes"], "closed scoring output limit exceeded")
        print(json.dumps(report["counts"]), flush=True)
    except Exception as error:
        if not (output / "failure.json").exists():
            write(output / "failure.json", {"type": type(error).__name__, "error": str(error)})
        raise


def authenticate_scoring(joint: Path, output: Path):
    _, _, source_pins = authenticate_sources(joint)
    report, closure = read(output / "report.json"), read(output / "capture.complete.json")
    need(report.get("schema") == "ungated-joint-sentence-scoring-v1"
         and report.get("completed") is True and report.get("rows") == 13
         and report.get("joint_directory") == str(joint.resolve())
         and report.get("parent_directory") == str(PARENT.resolve())
         and report.get("labels_opened") is False
         and report.get("neural_inference") is True
         and report.get("generation_inference") is False
         and report.get("neural_training") is False
         and report.get("production_changed") is False
         and report.get("settings") == {"max_length": 96, "batch_size": 8, "threads": 1,
                                        "interop_threads": 1, "external_context": ""}
         and report.get("pins") == source_pins
         and report.get("hashes", {}).get("capture.jsonl") == sha(output / "capture.jsonl")
         and report.get("hashes", {}).get("resources.jsonl") == sha(output / "resources.jsonl")
         and closure == {"completed": True, "rows": 13,
                         "capture_sha256": report["hashes"]["capture.jsonl"],
                         "resources_sha256": report["hashes"]["resources.jsonl"],
                         "report_sha256": sha(output / "report.json"),
                         "labels_opened": False},
         "closed source-only scoring capture required")
    verify_pins(report["pins"])
    return report, closure


def ascii_words(text):
    return tuple(ASCII.findall(text))


def evaluate_row(scored, label):
    need(scored.get("id") == label.get("id")
         and scored.get("request") == {"input": label.get("input"), "context": ""}
         and isinstance(label.get("expected"), list) and label["expected"],
         "FIT scoring identity differs")
    gold = set(label["expected"])
    ascii_references = {ascii_words(text) for text in gold}
    has_ascii = any(ascii_references)
    fallback_public = [candidate["text"] for candidate in scored["fallback_public"]]
    parent_scorable = {candidate["text"] for candidate in scored["arms"]["original"]["pool"]
                       if "parent" in candidate["provenance"]
                       and candidate["support"] == "supported"}

    def branch(public, full_pool, scorable_pool):
        return {"first": public[0] in gold, "top5": bool(set(public) & gold),
                "all_pool_covered": bool(set(full_pool) & gold),
                "scorable_pool_covered": bool(set(scorable_pool) & gold),
                "ascii_reference": bool(has_ascii),
                "ascii_exact": bool(has_ascii and ascii_words(public[0]) in ascii_references),
                "public": public}

    branches = {"captured_fallback": branch(fallback_public, fallback_public, parent_scorable)}
    for arm in ARMS:
        record = scored["arms"][arm]
        public = [candidate["text"] for candidate in record["public"]]
        full = [candidate["text"] for candidate in record["pool"]]
        branches[arm] = branch(public, full, record["supported_order"])
    transitions = {}
    pairs = {"fallback_to_original": ("captured_fallback", "original"),
             "fallback_to_roman_repaired": ("captured_fallback", "roman_repaired"),
             "original_to_roman_repaired": ("original", "roman_repaired")}
    for name, (before, after) in pairs.items():
        left, right = branches[before], branches[after]
        transitions[name] = {
            "corrections": int(not left["first"] and right["first"]),
            "regressions": int(left["first"] and not right["first"]),
            "top5_gains": int(not left["top5"] and right["top5"]),
            "top5_losses": int(left["top5"] and not right["top5"]),
            "all_pool_gains": int(not left["all_pool_covered"] and right["all_pool_covered"]),
            "all_pool_losses": int(left["all_pool_covered"] and not right["all_pool_covered"]),
            "scorable_pool_gains": int(not left["scorable_pool_covered"]
                                       and right["scorable_pool_covered"]),
            "scorable_pool_losses": int(left["scorable_pool_covered"]
                                        and not right["scorable_pool_covered"]),
            "ascii_gains": int(not left["ascii_exact"] and right["ascii_exact"]),
            "ascii_regressions": int(left["ascii_exact"] and not right["ascii_exact"]),
        }
    return {"id": scored["id"], "branches": branches, "transitions": transitions}


def aggregate(outcomes):
    branch_fields = ("first", "top5", "all_pool_covered", "scorable_pool_covered",
                     "ascii_reference", "ascii_exact")
    branches = {name: Counter(rows=0, **{field: 0 for field in branch_fields})
                for name in ("captured_fallback", *ARMS)}
    transitions = {name: Counter() for name in
                   ("fallback_to_original", "fallback_to_roman_repaired",
                    "original_to_roman_repaired")}
    for row in outcomes:
        for name, values in row["branches"].items():
            branches[name]["rows"] += 1
            for field in branch_fields:
                branches[name][field] += int(values[field])
        for name, values in row["transitions"].items():
            transitions[name].update(values)
    return {"branches": {name: dict(value) for name, value in branches.items()},
            "transitions": {name: dict(value) for name, value in transitions.items()}}


def evaluate(joint: Path, output: Path):
    report, closure = authenticate_scoring(joint, output)
    need(not (output / "evaluation.json").exists() and not (output / "outcomes.jsonl").exists(),
         "FIT scoring evaluation already exists")
    # The first FIT-label access is after source and scoring closure authentication.
    need(sha(LABELS) == LABEL_SHA, "FIT labels changed")
    labels = {}
    with LABELS.open() as stream:
        for line in stream:
            row = json.loads(line)
            need(row["id"] not in labels, "duplicate FIT label")
            labels[row["id"]] = row
    need(len(labels) == 1642, "FIT label denominator differs")
    outcomes = []
    with (output / "capture.jsonl").open() as stream:
        for line in stream:
            scored = json.loads(line)
            need(scored["id"] in labels, "unknown selected FIT ID")
            outcomes.append(evaluate_row(scored, labels[scored["id"]]))
    need(len(outcomes) == len({row["id"] for row in outcomes}) == 13,
         "selected FIT scoring denominator differs")
    summary = aggregate(outcomes)
    with (output / "outcomes.jsonl").open("x") as stream:
        for row in outcomes:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    closing_report, closing_closure = authenticate_scoring(joint, output)
    need(closing_report == report and closing_closure == closure and sha(LABELS) == LABEL_SHA,
         "source, scoring closure, or labels changed during evaluation")
    result = {
        "completed": True, "rows": 13, "summary": summary,
        "score_report_sha256": sha(output / "report.json"),
        "score_closure_sha256": sha(output / "capture.complete.json"),
        "score_capture_sha256": report["hashes"]["capture.jsonl"],
        "outcomes_sha256": sha(output / "outcomes.jsonl"), "labels_sha256": LABEL_SHA,
        "labels_opened_after_source_only_closure": True,
        "scope": "13 selected FIT rows; captured fallback versus two fixed joint-search arms.",
        "limitations": ["Previously exposed FIT labels, not blind generalization.",
                        "Existing SentenceRanker only; no training or new candidate generation.",
                        "Full and scoreable pool coverage are reported separately."],
        "neural_training": False, "generation_inference": False,
        "dev_opened": False, "test_opened": False, "production_changed": False,
    }
    write(output / "evaluation.json", result)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "evaluate"))
    parser.add_argument("--joint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    {"capture": capture, "evaluate": evaluate}[args.stage](args.joint, args.output)


if __name__ == "__main__":
    main()
