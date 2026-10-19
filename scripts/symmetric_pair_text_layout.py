"""Compact semantic pair views, keeping bookkeeping outside encoder text.

This layout intentionally drops rank/score/kind metadata from the neural view.
The complete source-side pair envelope still retains it for action protection,
provenance and auditing. Full roman input and both candidate texts are retained.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.probe_symmetric_pair_selector import (
    MODEL, RECIPE, OUTPUT as PRIOR_OUTPUT, canonical, sha, write_json,
)

OUTPUT = ROOT / ".omx/artifacts/symmetric-pair-text-layout-v1"


def semantic_views(row, current, alternative):
    context = row["context"]
    limit = RECIPE["context_chars"] // 2
    if len(context) > RECIPE["context_chars"]:
        context = context[:limit] + "[…중략…]" + context[-limit:]
    # JSON escaping retains arbitrary punctuation/newlines exactly. Position
    # in each candidate tuple identifies its immutable policy role.
    a, b = ["대안", alternative["text"]], ["현재", current["text"]]
    return tuple(json.dumps([row["input"], context, first, second],
        ensure_ascii=False, separators=(",", ":")) for first, second in ((a, b), (b, a)))


def run(output=OUTPUT):
    output.mkdir(exist_ok=False, parents=True)
    config = json.loads((PRIOR_OUTPUT / "configuration.json").read_bytes())
    for name, expected in config["pins"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"prior experiment source changed: {name}")
    rows = json.loads((PRIOR_OUTPUT / "sources.json").read_bytes())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    eligible_rows, all_pair_rows = set(), set()
    eligible_pairs = old_eligible_pairs = pairs = 0
    lengths = []
    with (PRIOR_OUTPUT / "pairs.jsonl").open() as stream, (output / "views.jsonl").open("xb") as dest:
        for line in stream:
            p = json.loads(line); row = rows[p["row_index"]]
            views = semantic_views(row, p["current"], p["alternative"])
            for text in views:
                decoded = json.loads(text)
                assert decoded[0] == row["input"]
                assert {tuple(decoded[2]), tuple(decoded[3])} == {
                    ("현재", p["current"]["text"]), ("대안", p["alternative"]["text"])}
            lens = [len(v) for v in tokenizer(list(views), truncation=False)["input_ids"]]
            old_lens = [len(v) for v in tokenizer(p["views"], truncation=False)["input_ids"]]
            allowed = max(lens) <= RECIPE["max_tokens"]
            old_eligible_pairs += max(old_lens) <= RECIPE["max_tokens"]
            all_pair_rows.add(p["row_index"])
            if allowed:
                eligible_pairs += 1; eligible_rows.add(p["row_index"])
            pairs += 1; lengths.append(max(lens))
            dest.write(canonical({"index": p["index"], "row_index": p["row_index"],
                "views": views, "token_lengths": lens, "eligible": allowed}) + b"\n")
    lengths.sort()
    report = {"scope": "source-only token coverage; no labels, model weights or training",
        "rows": len(rows), "rows_with_pairs": len(all_pair_rows), "pairs": pairs,
        "previous_eligible_pairs": old_eligible_pairs, "eligible_pairs": eligible_pairs,
        "rows_with_eligible_pairs": len(eligible_rows),
        "max_tokens": max(lengths), "median_tokens": lengths[len(lengths)//2],
        "p95_tokens": lengths[int(.95*len(lengths))],
        "complete_roman_and_candidate_text_roundtrip": True,
        "removed_model_fields": ["origin scores", "origin ranks", "kind arrays"],
        "bookkeeping_preserved_in_source_pairs": True,
        "training_started": False, "production_changed": False,
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in (
            Path(__file__), PRIOR_OUTPUT / "sources.json", PRIOR_OUTPUT / "pairs.jsonl")},
        "views_sha256": sha(output / "views.jsonl")}
    write_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    run()
