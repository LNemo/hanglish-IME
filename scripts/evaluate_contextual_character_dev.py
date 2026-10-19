"""Late-label DEV evaluation for the closed contextual-character capture."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_recombination_tokenization import digest, require
from scripts.evaluate_current_phonetic_choice_dev import ascii_words, literal_signature


DEV_SHA = "593e1a032b54bf4d2c541da6754cfc68daf759cdcee7f8fa1f0a746459910ac9"
PARENT_CAPTURE_SHA = "8985ec4f50b0151987e69cad63a3033203d1407af68cbc9c250fe82c7a7858f1"
PARENT_REPORT_SHA = "ea29e58810b9543205d4ec94b244e5aa9a1138f74a4acb95182e3c4f13a69614"
PARENT_PROOF_SHA = "453bb4a9993d3d083c7f0e0406cde8d5015faa27df36f73f243c2f52ce1998cf"

ARMS = (
    "current_production", "old_calibrated", "expanded_calibrated",
    "old_character", "expanded_character", "old_contextual", "expanded_contextual",
)
OLD_POOL_ARMS = {"old_calibrated", "old_character", "old_contextual"}
TRANSITIONS = {
    "union_only": ("current_production", "old_calibrated"),
    "recombination_only": ("old_calibrated", "expanded_calibrated"),
    "character_old_pool": ("old_calibrated", "old_character"),
    "character_expanded_pool": ("expanded_calibrated", "expanded_character"),
    "contextual_old_pool": ("old_calibrated", "old_contextual"),
    "contextual_expanded_pool": ("expanded_calibrated", "expanded_contextual"),
    "old_contextual_vs_character": ("old_character", "old_contextual"),
    "expanded_contextual_vs_character": ("expanded_character", "expanded_contextual"),
    "old_character_vs_production": ("current_production", "old_character"),
    "character_overall": ("current_production", "expanded_character"),
    "old_contextual_vs_production": ("current_production", "old_contextual"),
    "contextual_overall": ("current_production", "expanded_contextual"),
}
BASELINE = {
    "current_production": (267, 284),
    "old_calibrated": (277, 295),
    "expanded_calibrated": (276, 296),
    "old_character": (275, 296),
    "expanded_character": (275, 297),
}


def _read_json(path: Path):
    return json.loads(path.read_text())


def _length_stratum(source: str) -> str:
    size = len(source.encode("utf-8"))
    if size <= 64:
        return "le_64"
    if size <= 128:
        return "65_128"
    if size <= 256:
        return "129_256"
    return "gt_256"


def _public_texts(value, message="public candidates differ"):
    require(isinstance(value, list) and 1 <= len(value) <= 5
            and all(isinstance(item, dict) and isinstance(item.get("text"), str)
                    for item in value), message)
    texts = [item["text"] for item in value]
    require(len(texts) == len(set(texts)), message)
    return texts


def _validate_contextual_source(parent, capture):
    require(set(capture) == {"id", "request", "dataset_split", "source_row_index",
                             "contextual_scores", "predictions"}
            and capture.get("dataset_split") == "dev"
            and capture.get("id") == parent.get("id")
            and capture.get("request") == parent.get("request")
            and capture.get("source_row_index") == parent.get("source_row_index")
            and set(capture.get("contextual_scores", {})) == {"old_contextual", "expanded_contextual"}
            and set(capture.get("predictions", {})) == {"old_contextual", "expanded_contextual"},
            "contextual capture row differs from parent")
    scoring = parent["scoring"]
    old = scoring["original"]
    novel = [item for item in scoring["novel"] if item["status"] == "scored"]
    expanded = old + novel
    expected = {"old_contextual": old, "expanded_contextual": expanded}
    if scoring["ranked"]:
        for arm, pool in expected.items():
            scores = capture["contextual_scores"][arm]
            require(isinstance(scores, list) and len(scores) == len(pool)
                    and all(type(value) is float and math.isfinite(value)
                            for value in scores), "contextual score pool differs")
            texts = _public_texts(capture["predictions"][arm])
            require(set(texts) <= ({item["text"] for item in pool} | {parent["request"]["input"]}),
                    "contextual prediction is outside its frozen pool")
        require(capture["contextual_scores"]["expanded_contextual"][:len(old)]
                == capture["contextual_scores"]["old_contextual"],
                "old contextual scores are not the expanded-pool prefix")
    else:
        require(not old and not scoring["novel"]
                and capture["contextual_scores"] == {"old_contextual": [], "expanded_contextual": []},
                "no-pool row has contextual scores")
        fallback = [item["text"] for item in scoring["fallback_public"]]
        for arm in ("old_contextual", "expanded_contextual"):
            require(_public_texts(capture["predictions"][arm]) == fallback,
                    "no-pool fallback changed")


def evaluate_row(parent, capture, label):
    _validate_contextual_source(parent, capture)
    require(parent["id"] == label.get("id")
            and parent["request"] == {"input": label.get("input"), "context": ""}
            and label.get("split") == "dev" and label.get("context", "") == ""
            and isinstance(label.get("expected"), list) and bool(label["expected"]),
            "DEV row contract differs")
    predictions = dict(parent["predictions"])
    require(set(predictions) == set(ARMS[:5]), "parent prediction arms differ")
    predictions.update(capture["predictions"])
    gold = set(label["expected"])
    signatures = {ascii_words(text) for text in gold}
    has_ascii = any(signatures)
    source_literal = literal_signature(label["input"])
    scoring = parent["scoring"]
    old = {item["text"] for item in scoring["original"]}
    expanded = old | {item["text"] for item in scoring["novel"] if item["status"] == "scored"}
    arms = {}
    for arm in ARMS:
        texts = _public_texts(predictions[arm])
        pool = old if arm in OLD_POOL_ARMS else expanded
        arms[arm] = {
            "first": texts[0] in gold,
            "top5": bool(set(texts) & gold),
            "ascii_reference": bool(has_ascii),
            "ascii_exact": bool(has_ascii and ascii_words(texts[0]) in signatures),
            "literal_exact": literal_signature(texts[0]) == source_literal,
            "pool_covered": None if arm == "current_production" else bool(pool & gold),
            "public": texts,
        }
    transitions = {}
    for name, (before, after) in TRANSITIONS.items():
        left, right = arms[before], arms[after]
        transitions[name] = {
            "corrections": int(not left["first"] and right["first"]),
            "regressions": int(left["first"] and not right["first"]),
            "top5_gains": int(not left["top5"] and right["top5"]),
            "top5_losses": int(left["top5"] and not right["top5"]),
            "ascii_regressions": int(left["ascii_exact"] and not right["ascii_exact"]),
            "literal_regressions": int(left["literal_exact"] and not right["literal_exact"]),
        }
    return {
        "id": parent["id"], "request": parent["request"], "arms": arms,
        "transitions": transitions, "ranked": scoring["ranked"],
        "strata": {"source_utf8_length": _length_stratum(label["input"]),
                    "gold_ascii": "gold_ascii" if has_ascii else "no_gold_ascii"},
    }


def _safe_gate(branches, transitions, arm, baseline, transition):
    delta = transitions[transition]
    return (branches[arm]["first"] > branches[baseline]["first"]
            and not any(delta[key] for key in
                        ("regressions", "top5_losses", "ascii_regressions", "literal_regressions")))


def aggregate_group(outcomes):
    branches = {arm: Counter(rows=0, first=0, top5=0, ascii_reference=0,
                             ascii_exact=0, literal_exact=0, pool_covered=0) for arm in ARMS}
    transitions = {name: Counter(corrections=0, regressions=0, top5_gains=0,
                                 top5_losses=0, ascii_regressions=0,
                                 literal_regressions=0) for name in TRANSITIONS}
    for row in outcomes:
        for arm, values in row["arms"].items():
            branches[arm]["rows"] += 1
            for key in ("first", "top5", "ascii_reference", "ascii_exact",
                        "literal_exact", "pool_covered"):
                if values[key] is not None:
                    branches[arm][key] += int(values[key])
        for name, values in row["transitions"].items():
            transitions[name].update(values)
    branches["current_production"]["pool_covered"] = None
    branch_dict = {name: dict(value) for name, value in branches.items()}
    transition_dict = {name: dict(value) for name, value in transitions.items()}
    production = {
        "old_contextual": _safe_gate(branch_dict, transition_dict, "old_contextual",
                                     "current_production", "old_contextual_vs_production"),
        "expanded_contextual": _safe_gate(branch_dict, transition_dict, "expanded_contextual",
                                          "current_production", "contextual_overall"),
    }
    matched = {
        "old_contextual": {
            "vs_calibrated": _safe_gate(branch_dict, transition_dict, "old_contextual",
                                        "old_calibrated", "contextual_old_pool"),
            "vs_character": _safe_gate(branch_dict, transition_dict, "old_contextual",
                                       "old_character", "old_contextual_vs_character"),
        },
        "expanded_contextual": {
            "vs_calibrated": _safe_gate(branch_dict, transition_dict, "expanded_contextual",
                                        "expanded_calibrated", "contextual_expanded_pool"),
            "vs_character": _safe_gate(branch_dict, transition_dict, "expanded_contextual",
                                       "expanded_character", "expanded_contextual_vs_character"),
        },
    }
    return {"branches": branch_dict, "transitions": transition_dict,
            "quality_gates": {"production": production, "matched_pool": matched}}


def aggregate(outcomes):
    result = aggregate_group(outcomes)
    def diagnostic(rows):
        value = aggregate_group(rows)
        value.pop("quality_gates")
        return value
    result["strata"] = {
        axis: {value: diagnostic([row for row in outcomes if row["strata"][axis] == value])
               for value in values}
        for axis, values in {
            "source_utf8_length": ("le_64", "65_128", "129_256", "gt_256"),
            "gold_ascii": ("gold_ascii", "no_gold_ascii"),
        }.items()
    }
    return result


def authenticate_capture(directory: Path):
    report = _read_json(directory / "report.json")
    proof = _read_json(directory / "independent-verification.json")
    capture_sha = digest(directory / "capture.jsonl")
    require(report.get("schema") == "contextual-character-dev-capture-v1"
            and report.get("completed") is True and report.get("rows") == 316
            and report.get("dataset_split") == "dev" and report.get("candidate_chunk") == 32
            and report.get("labels_opened") is False and report.get("test_opened") is False
            and report.get("production_changed") is False
            and proof.get("verified") is True and proof.get("rows") == 316
            and proof.get("labels_opened") is False
            and proof.get("report_sha256") == digest(directory / "report.json")
            and proof.get("capture_sha256") == capture_sha
            and report.get("hashes", {}).get("capture.jsonl") == capture_sha
            and report.get("hashes", {}).get("resources.jsonl") == digest(directory / "resources.jsonl"),
            "complete independently verified contextual DEV capture required")
    expected_counts = {"ranked_rows": 312, "no_pool_rows": 4, "old_candidates": 47694,
                       "novel_candidates": 27831, "scored_novel": 27831,
                       "unsupported_novel": 0}
    require(report.get("counts") == expected_counts, "contextual capture counts differ")
    require(Path(report["training_directory"]).is_dir(), "contextual training directory missing")
    for path, expected in report.get("pins", {}).items():
        require(digest(Path(path)) == expected, "closed contextual source pin changed")
    parent = Path(report["parent_directory"])
    parent_report = _read_json(parent / "report.json")
    parent_proof = _read_json(parent / "independent-verification.json")
    require(digest(parent / "capture.jsonl") == PARENT_CAPTURE_SHA == capture_sha_from(parent_report)
            and digest(parent / "report.json") == PARENT_REPORT_SHA
            and digest(parent / "independent-verification.json") == PARENT_PROOF_SHA
            and parent_proof.get("verified") is True and parent_proof.get("rows") == 316,
            "fixed parent DEV capture differs")
    return report, parent, capture_sha


def capture_sha_from(report):
    return report.get("capture_sha256")


def _assert_historical_baselines(summary):
    for arm, expected in BASELINE.items():
        branch = summary["branches"][arm]
        require((branch["rows"], branch["first"], branch["top5"]) == (316, *expected),
                "historical DEV comparator differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report, parent_directory, capture_sha = authenticate_capture(args.capture)

    # First canonical DEV-label access occurs only after the source-only proof and pins close.
    label_path = ROOT / "data/challenge/sentence-v3/dev.jsonl"
    require(digest(label_path) == DEV_SHA, "DEV label file changed")
    labels = [json.loads(line) for line in label_path.open()]
    require(len(labels) == len({row["id"] for row in labels}) == 316, "DEV label count differs")
    args.output.mkdir(parents=True, exist_ok=False)
    outcomes = []
    observed = Counter(ranked_rows=0, no_pool_rows=0, old_candidates=0,
                       novel_candidates=0, scored_novel=0, unsupported_novel=0)
    with (parent_directory / "capture.jsonl").open() as parent_stream, \
            (args.capture / "capture.jsonl").open() as capture_stream:
        for row_index, label in enumerate(labels):
            parent_line, capture_line = next(parent_stream, None), next(capture_stream, None)
            require(parent_line is not None and capture_line is not None, "capture missing row")
            parent, contextual = json.loads(parent_line), json.loads(capture_line)
            require(parent.get("source_row_index") == contextual.get("source_row_index") == row_index,
                    "source row index differs")
            outcome = evaluate_row(parent, contextual, label)
            outcomes.append(outcome)
            scoring = parent["scoring"]
            observed["ranked_rows" if scoring["ranked"] else "no_pool_rows"] += 1
            observed["old_candidates"] += len(scoring["original"])
            observed["novel_candidates"] += len(scoring["novel"])
            observed["scored_novel"] += sum(item["status"] == "scored" for item in scoring["novel"])
            observed["unsupported_novel"] += sum(item["status"] == "unsupported" for item in scoring["novel"])
        require(next(parent_stream, None) is None and next(capture_stream, None) is None,
                "extra capture row")
    require(dict(observed) == report["counts"], "parent pools and contextual report counts differ")
    summary = aggregate(outcomes)
    _assert_historical_baselines(summary)
    with (args.output / "outcomes.jsonl").open("x") as stream:
        for row in outcomes:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    require(digest(args.capture / "capture.jsonl") == capture_sha
            and digest(label_path) == DEV_SHA, "capture or labels changed during evaluation")
    closing_report, closing_parent, closing_capture_sha = authenticate_capture(args.capture)
    require(closing_report == report and closing_parent == parent_directory
            and closing_capture_sha == capture_sha,
            "source proof or pins changed during evaluation")
    result = {
        "completed": True, "rows": 316, "no_pool_rows": 4, "summary": summary,
        "capture_directory": str(args.capture.resolve()), "capture_sha256": capture_sha,
        "capture_report_sha256": digest(args.capture / "report.json"),
        "capture_proof_sha256": digest(args.capture / "independent-verification.json"),
        "parent_directory": str(parent_directory.resolve()),
        "parent_capture_sha256": PARENT_CAPTURE_SHA, "parent_report_sha256": PARENT_REPORT_SHA,
        "parent_proof_sha256": PARENT_PROOF_SHA, "labels_sha256": DEV_SHA,
        "outcomes_sha256": digest(args.output / "outcomes.jsonl"),
        "evaluator_sha256": digest(__file__), "labels_opened_after_verified_capture": True,
        "test_opened": False, "neural_training": False, "production_changed": False,
        "automatic_deployment": False,
        "scope": ("Previously exposed DEV diagnostic over seven frozen arms; not untouched "
                  "generalization, production promotion, 1000-case regression, or web validation."),
    }
    (args.output / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
