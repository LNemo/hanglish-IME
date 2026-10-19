"""Independent recount of contextual-character DEV outcomes, strata, and gates."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_recombination_tokenization import digest, require


DEV_SHA = "593e1a032b54bf4d2c541da6754cfc68daf759cdcee7f8fa1f0a746459910ac9"
PARENT_CAPTURE_SHA = "8985ec4f50b0151987e69cad63a3033203d1407af68cbc9c250fe82c7a7858f1"
PARENT_REPORT_SHA = "ea29e58810b9543205d4ec94b244e5aa9a1138f74a4acb95182e3c4f13a69614"
PARENT_PROOF_SHA = "453bb4a9993d3d083c7f0e0406cde8d5015faa27df36f73f243c2f52ce1998cf"
ARMS = ("current_production", "old_calibrated", "expanded_calibrated",
        "old_character", "expanded_character", "old_contextual", "expanded_contextual")
OLD = {"old_calibrated", "old_character", "old_contextual"}
COMPARISONS = {
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
ASCII = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
LITERAL_DROP = re.compile(r"[a-z가-힣ᄀ-ᇿ㄰-㆏]")


def _ascii_words(text):
    return tuple(ASCII.findall(text))


def _literal(text):
    return LITERAL_DROP.sub("", text)


def _bucket(text):
    size = len(text.encode("utf-8"))
    return "le_64" if size <= 64 else "65_128" if size <= 128 else "129_256" if size <= 256 else "gt_256"


def _texts(public):
    require(isinstance(public, list) and 1 <= len(public) <= 5
            and all(isinstance(item, dict) and isinstance(item.get("text"), str) for item in public),
            "public candidates differ")
    values = [item["text"] for item in public]
    require(len(values) == len(set(values)), "duplicate public candidate")
    return values


def independent_outcome(parent, contextual, label):
    require(set(contextual) == {"id", "request", "dataset_split", "source_row_index",
                                "contextual_scores", "predictions"}
            and parent.get("id") == contextual.get("id") == label.get("id")
            and parent.get("request") == contextual.get("request")
            == {"input": label.get("input"), "context": ""}
            and contextual.get("dataset_split") == label.get("split") == "dev"
            and contextual.get("source_row_index") == parent.get("source_row_index")
            and label.get("context", "") == "" and label.get("expected"),
            "row identity differs")
    require(set(parent["predictions"]) == set(ARMS[:5])
            and set(contextual["predictions"]) == {"old_contextual", "expanded_contextual"}
            and set(contextual["contextual_scores"]) == {"old_contextual", "expanded_contextual"},
            "prediction arms differ")
    scoring = parent["scoring"]
    originals = scoring["original"]
    scored_novel = [item for item in scoring["novel"] if item["status"] == "scored"]
    if scoring["ranked"]:
        for arm, pool in (("old_contextual", originals),
                          ("expanded_contextual", originals + scored_novel)):
            values = contextual["contextual_scores"][arm]
            require(len(values) == len(pool)
                    and all(type(value) is float and math.isfinite(value) for value in values),
                    "contextual score pool differs")
            require(set(_texts(contextual["predictions"][arm]))
                    <= ({item["text"] for item in pool} | {parent["request"]["input"]}),
                    "contextual prediction outside pool")
        require(contextual["contextual_scores"]["expanded_contextual"][:len(originals)]
                == contextual["contextual_scores"]["old_contextual"],
                "old contextual scores are not expanded prefix")
    else:
        require(not originals and not scoring["novel"]
                and contextual["contextual_scores"] == {"old_contextual": [], "expanded_contextual": []},
                "no-pool contextual scores differ")
        fallback = [item["text"] for item in scoring["fallback_public"]]
        require(_texts(contextual["predictions"]["old_contextual"]) == fallback
                and _texts(contextual["predictions"]["expanded_contextual"]) == fallback,
                "no-pool fallback differs")
    predictions = {**parent["predictions"], **contextual["predictions"]}
    expected = set(label["expected"])
    ascii_refs = {_ascii_words(text) for text in expected}
    has_ascii = any(ascii_refs)
    old_pool = {item["text"] for item in originals}
    expanded_pool = old_pool | {item["text"] for item in scored_novel}
    arms = {}
    for arm in ARMS:
        texts = _texts(predictions[arm])
        covered = None if arm == "current_production" else bool(
            expected & (old_pool if arm in OLD else expanded_pool))
        arms[arm] = {
            "first": texts[0] in expected, "top5": bool(expected & set(texts)),
            "ascii_reference": bool(has_ascii),
            "ascii_exact": bool(has_ascii and _ascii_words(texts[0]) in ascii_refs),
            "literal_exact": _literal(texts[0]) == _literal(label["input"]),
            "pool_covered": covered, "public": texts,
        }
    changes = {}
    for name, (before, after) in COMPARISONS.items():
        left, right = arms[before], arms[after]
        changes[name] = {
            "corrections": int(right["first"] and not left["first"]),
            "regressions": int(left["first"] and not right["first"]),
            "top5_gains": int(right["top5"] and not left["top5"]),
            "top5_losses": int(left["top5"] and not right["top5"]),
            "ascii_regressions": int(left["ascii_exact"] and not right["ascii_exact"]),
            "literal_regressions": int(left["literal_exact"] and not right["literal_exact"]),
        }
    return {"id": parent["id"], "request": parent["request"], "arms": arms,
            "transitions": changes, "ranked": scoring["ranked"],
            "strata": {"source_utf8_length": _bucket(label["input"]),
                       "gold_ascii": "gold_ascii" if has_ascii else "no_gold_ascii"}}


def _gate(branches, transitions, arm, baseline, comparison):
    delta = transitions[comparison]
    return (branches[arm]["first"] > branches[baseline]["first"]
            and sum(delta[key] for key in ("regressions", "top5_losses",
                                           "ascii_regressions", "literal_regressions")) == 0)


def independent_group(rows):
    branches = {arm: Counter(rows=0, first=0, top5=0, ascii_reference=0,
                             ascii_exact=0, literal_exact=0, pool_covered=0) for arm in ARMS}
    transitions = {name: Counter(corrections=0, regressions=0, top5_gains=0,
                                 top5_losses=0, ascii_regressions=0,
                                 literal_regressions=0) for name in COMPARISONS}
    for row in rows:
        for arm in ARMS:
            values = row["arms"][arm]
            branches[arm]["rows"] += 1
            for field in ("first", "top5", "ascii_reference", "ascii_exact",
                          "literal_exact", "pool_covered"):
                if values[field] is not None:
                    branches[arm][field] += int(values[field])
        for name in COMPARISONS:
            transitions[name].update(row["transitions"][name])
    branches["current_production"]["pool_covered"] = None
    b = {key: dict(value) for key, value in branches.items()}
    t = {key: dict(value) for key, value in transitions.items()}
    production = {
        "old_contextual": _gate(b, t, "old_contextual", "current_production",
                                "old_contextual_vs_production"),
        "expanded_contextual": _gate(b, t, "expanded_contextual", "current_production",
                                     "contextual_overall"),
    }
    matched = {
        "old_contextual": {
            "vs_calibrated": _gate(b, t, "old_contextual", "old_calibrated", "contextual_old_pool"),
            "vs_character": _gate(b, t, "old_contextual", "old_character", "old_contextual_vs_character"),
        },
        "expanded_contextual": {
            "vs_calibrated": _gate(b, t, "expanded_contextual", "expanded_calibrated", "contextual_expanded_pool"),
            "vs_character": _gate(b, t, "expanded_contextual", "expanded_character", "expanded_contextual_vs_character"),
        },
    }
    return {"branches": b, "transitions": t,
            "quality_gates": {"production": production, "matched_pool": matched}}


def independent_summary(rows):
    result = independent_group(rows)
    def diagnostic(values):
        summary = independent_group(values)
        summary.pop("quality_gates")
        return summary
    result["strata"] = {
        "source_utf8_length": {bucket: diagnostic(
            [row for row in rows if row["strata"]["source_utf8_length"] == bucket])
            for bucket in ("le_64", "65_128", "129_256", "gt_256")},
        "gold_ascii": {bucket: diagnostic(
            [row for row in rows if row["strata"]["gold_ascii"] == bucket])
            for bucket in ("gold_ascii", "no_gold_ascii")},
    }
    return result


def _json(path):
    return json.loads(path.read_text())


def authenticate(evaluation: Path):
    report = _json(evaluation / "report.json")
    capture = Path(report["capture_directory"])
    parent = Path(report["parent_directory"])
    require(report.get("completed") is True and report.get("rows") == 316
            and report.get("no_pool_rows") == 4
            and report.get("labels_opened_after_verified_capture") is True
            and report.get("test_opened") is False and report.get("production_changed") is False
            and report.get("automatic_deployment") is False
            and digest(capture / "capture.jsonl") == report.get("capture_sha256")
            and digest(capture / "report.json") == report.get("capture_report_sha256")
            and digest(capture / "independent-verification.json") == report.get("capture_proof_sha256")
            and digest(parent / "capture.jsonl") == report.get("parent_capture_sha256") == PARENT_CAPTURE_SHA
            and digest(parent / "report.json") == report.get("parent_report_sha256") == PARENT_REPORT_SHA
            and digest(parent / "independent-verification.json") == report.get("parent_proof_sha256") == PARENT_PROOF_SHA
            and digest(evaluation / "outcomes.jsonl") == report.get("outcomes_sha256")
            and digest(ROOT / "scripts/evaluate_contextual_character_dev.py") == report.get("evaluator_sha256"),
            "evaluation evidence changed")
    capture_report = _json(capture / "report.json")
    proof = _json(capture / "independent-verification.json")
    require(capture_report.get("schema") == "contextual-character-dev-capture-v1"
            and capture_report.get("completed") is True and capture_report.get("rows") == 316
            and capture_report.get("dataset_split") == "dev"
            and capture_report.get("candidate_chunk") == 32
            and capture_report.get("labels_opened") is False
            and proof.get("verified") is True and proof.get("rows") == 316
            and proof.get("labels_opened") is False
            and proof.get("report_sha256") == digest(capture / "report.json")
            and proof.get("capture_sha256") == digest(capture / "capture.jsonl")
            and capture_report.get("hashes", {}).get("capture.jsonl") == digest(capture / "capture.jsonl")
            and capture_report.get("hashes", {}).get("resources.jsonl") == digest(capture / "resources.jsonl"),
            "source-only proof is not closed")
    for path, expected in capture_report.get("pins", {}).items():
        require(digest(Path(path)) == expected, "source pin changed")
    return report, capture, parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report, capture, parent = authenticate(args.directory)
    starting_report_sha = digest(args.directory / "report.json")

    # First label access is after independent source-capture and evaluation authentication.
    labels_path = ROOT / "data/challenge/sentence-v3/dev.jsonl"
    require(digest(labels_path) == report.get("labels_sha256") == DEV_SHA, "DEV labels changed")
    labels = [json.loads(line) for line in labels_path.open()]
    require(len(labels) == 316, "label denominator differs")
    outcomes = []
    seen = set()
    observed = Counter(ranked_rows=0, no_pool_rows=0, old_candidates=0,
                       novel_candidates=0, scored_novel=0, unsupported_novel=0)
    with (parent / "capture.jsonl").open() as parent_stream, \
            (capture / "capture.jsonl").open() as contextual_stream, \
            (args.directory / "outcomes.jsonl").open() as outcome_stream:
        for row_index, label in enumerate(labels):
            lines = (next(parent_stream, None), next(contextual_stream, None),
                     next(outcome_stream, None))
            require(all(line is not None for line in lines), "capture or outcome missing row")
            source, contextual, saved = (json.loads(line) for line in lines)
            require(source.get("source_row_index") == contextual.get("source_row_index") == row_index,
                    "source row index differs")
            outcome = independent_outcome(source, contextual, label)
            require(outcome["id"] not in seen and outcome == saved,
                    "recorded per-row outcome differs")
            seen.add(outcome["id"])
            outcomes.append(outcome)
            scoring = source["scoring"]
            observed["ranked_rows" if scoring["ranked"] else "no_pool_rows"] += 1
            observed["old_candidates"] += len(scoring["original"])
            observed["novel_candidates"] += len(scoring["novel"])
            observed["scored_novel"] += sum(item["status"] == "scored" for item in scoring["novel"])
            observed["unsupported_novel"] += sum(item["status"] == "unsupported" for item in scoring["novel"])
        require(next(parent_stream, None) is None and next(contextual_stream, None) is None
                and next(outcome_stream, None) is None, "extra capture or outcome row")
    require(dict(observed) == _json(capture / "report.json")["counts"],
            "parent pool recount differs from contextual report")
    summary = independent_summary(outcomes)
    require(summary == report.get("summary"), "metrics, strata, or gates differ")
    known = {"current_production": (267, 284), "old_calibrated": (277, 295),
             "expanded_calibrated": (276, 296), "old_character": (275, 296),
             "expanded_character": (275, 297)}
    for arm, pair in known.items():
        branch = summary["branches"][arm]
        require((branch["rows"], branch["first"], branch["top5"]) == (316, *pair),
                "historical baseline differs")
    require(digest(labels_path) == DEV_SHA and digest(capture / "capture.jsonl") == report["capture_sha256"]
            and digest(args.directory / "outcomes.jsonl") == report["outcomes_sha256"],
            "evidence changed during recount")
    closing_report, closing_capture, closing_parent = authenticate(args.directory)
    require(closing_report == report and closing_capture == capture and closing_parent == parent
            and digest(args.directory / "report.json") == starting_report_sha,
            "evaluation proof, source proof, resources, or pins changed during recount")
    result = {
        "verified": True, "rows": 316, "no_pool_rows": 4, "summary": summary,
        "report_sha256": starting_report_sha,
        "outcomes_sha256": report["outcomes_sha256"], "labels_sha256": DEV_SHA,
        "verifier_sha256": digest(__file__),
        "scope": ("Independent seven-arm outcome, UTF-8 length/gold-ASCII stratum, transition, "
                  "matched-pool and production-gate recount; no model training or inference."),
    }
    with (args.directory / "independent-verification.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
