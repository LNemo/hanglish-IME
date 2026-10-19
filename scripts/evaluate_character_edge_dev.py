"""Join DEV labels only after the fixed source-only capture is independently closed."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_recombination_tokenization import digest, require
from scripts.evaluate_current_phonetic_choice_dev import ascii_words, literal_signature

ARMS = ("current_production", "old_calibrated", "expanded_calibrated",
        "old_character", "expanded_character")
TRANSITIONS = {
    "union_only": ("current_production", "old_calibrated"),
    "recombination_only": ("old_calibrated", "expanded_calibrated"),
    "character_old_pool": ("old_calibrated", "old_character"),
    "character_expanded_pool": ("expanded_calibrated", "expanded_character"),
    "old_character_vs_production": ("current_production", "old_character"),
    "overall": ("current_production", "expanded_character"),
}
DEV_SHA = "593e1a032b54bf4d2c541da6754cfc68daf759cdcee7f8fa1f0a746459910ac9"


def evaluate_row(capture, label):
    require(capture["dataset_split"] == "dev"
            and capture["id"] == label["id"]
            and capture["request"] == {"input": label["input"], "context": ""}
            and label["split"] == "dev" and label.get("context", "") == ""
            and isinstance(label["expected"], list) and bool(label["expected"])
            and set(capture["predictions"]) == set(ARMS), "DEV row contract differs")
    gold = set(label["expected"])
    signatures = {ascii_words(text) for text in gold}
    has_ascii = any(signatures)
    source_literal = literal_signature(label["input"])
    scoring = capture["scoring"]
    old = {c["text"] for c in scoring["original"]}
    expanded = old | {c["text"] for c in scoring["novel"] if c["status"] == "scored"}
    outcomes = {}
    for arm in ARMS:
        public = capture["predictions"][arm]
        require(isinstance(public, list) and 1 <= len(public) <= 5
                and all(isinstance(c.get("text"), str) for c in public), "public candidates differ")
        texts = [c["text"] for c in public]
        pool = old if arm.startswith("old_") else expanded
        outcomes[arm] = {
            "first": texts[0] in gold, "top5": bool(set(texts) & gold),
            "ascii_reference": bool(has_ascii),
            "ascii_exact": bool(has_ascii and ascii_words(texts[0]) in signatures),
            "literal_exact": literal_signature(texts[0]) == source_literal,
            "pool_covered": None if arm == "current_production" else bool(pool & gold),
            "public": texts,
        }
    changes = {}
    for name, (before, after) in TRANSITIONS.items():
        a, b = outcomes[before], outcomes[after]
        changes[name] = {
            "corrections": int(not a["first"] and b["first"]),
            "regressions": int(a["first"] and not b["first"]),
            "top5_gains": int(not a["top5"] and b["top5"]),
            "top5_losses": int(a["top5"] and not b["top5"]),
            "ascii_regressions": int(a["ascii_exact"] and not b["ascii_exact"]),
            "literal_regressions": int(a["literal_exact"] and not b["literal_exact"]),
        }
    return {"id": capture["id"], "request": capture["request"], "arms": outcomes,
            "transitions": changes, "ranked": scoring["ranked"]}


def aggregate(outcomes):
    branches = {arm: Counter(rows=0, first=0, top5=0, ascii_reference=0,
                             ascii_exact=0, literal_exact=0, pool_covered=0) for arm in ARMS}
    changes = {name: Counter(corrections=0, regressions=0, top5_gains=0, top5_losses=0,
                            ascii_regressions=0, literal_regressions=0) for name in TRANSITIONS}
    for row in outcomes:
        for arm, values in row["arms"].items():
            branches[arm]["rows"] += 1
            for key in ("first", "top5", "ascii_reference", "ascii_exact", "literal_exact", "pool_covered"):
                branches[arm][key] += int(values[key]) if values[key] is not None else 0
        for name, values in row["transitions"].items():
            changes[name].update(values)
    branches["current_production"]["pool_covered"] = None
    gates = {}
    for arm, transition in (("old_character", "old_character_vs_production"),
                            ("expanded_character", "overall")):
        delta = changes[transition]
        gates[arm] = (branches[arm]["first"] > branches["current_production"]["first"]
                      and not any(delta[key] for key in ("regressions", "top5_losses",
                                                        "ascii_regressions", "literal_regressions")))
    return {"branches": {k:dict(v) for k,v in branches.items()},
            "transitions": {k:dict(v) for k,v in changes.items()}, "quality_gates": gates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    capture_report = json.loads((args.capture/"report.json").read_text())
    proof = json.loads((args.capture/"independent-verification.json").read_text())
    closure = json.loads((args.capture/"capture.complete.json").read_text())
    capture_sha = digest(args.capture/"capture.jsonl")
    require(capture_report["completed"] and capture_report["mode"] == "full"
            and capture_report["rows"] == 316 and proof["verified"] and proof["rows"] == 316
            and proof["report_sha256"] == digest(args.capture/"report.json")
            and proof["capture_sha256"] == capture_sha == closure["capture_sha256"]
            and closure["rows"] == 316 and closure["labels_opened"] is False,
            "complete independently verified DEV capture required before labels")
    for path, expected in capture_report["pins"].items():
        require(digest(Path(path)) == expected, "closed source/model pin changed")
    # The first label access occurs after all source-only closure checks above.
    label_path = ROOT/"data/challenge/sentence-v3/dev.jsonl"
    require(digest(label_path) == DEV_SHA, "DEV label file changed")
    labels = [json.loads(line) for line in label_path.open()]
    require(len(labels) == len({r["id"] for r in labels}) == 316, "DEV label count differs")
    args.output.mkdir(parents=True, exist_ok=False)
    outcomes = []
    with (args.capture/"capture.jsonl").open() as stream:
        for label in labels:
            line = next(stream, None)
            require(line is not None, "capture missing row")
            outcomes.append(evaluate_row(json.loads(line), label))
        require(next(stream, None) is None, "extra capture row")
    summary = aggregate(outcomes)
    production = summary["branches"]["current_production"]
    require((production["first"], production["top5"], production["ascii_reference"], production["ascii_exact"])
            == (267,284,59,55), "actual production comparator differs")
    with (args.output/"outcomes.jsonl").open("x") as stream:
        for row in outcomes:
            stream.write(json.dumps(row, ensure_ascii=False)+"\n")
    require(digest(args.capture/"capture.jsonl") == capture_sha and digest(label_path) == DEV_SHA,
            "capture or labels changed during evaluation")
    result = {"completed": True, "rows":316, "summary":summary,
              "capture_directory":str(args.capture.resolve()), "capture_sha256":capture_sha,
              "capture_report_sha256":digest(args.capture/"report.json"),
              "capture_proof_sha256":digest(args.capture/"independent-verification.json"),
              "labels_sha256":DEV_SHA, "outcomes_sha256":digest(args.output/"outcomes.jsonl"),
              "evaluator_sha256":digest(__file__), "labels_opened_after_verified_capture":True,
              "test_opened":False, "neural_training":False, "production_changed":False,
              "scope":"Previously used DEV comparison; not untouched generalization or live runtime/web validation."}
    (args.output/"report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
