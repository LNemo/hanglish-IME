"""Prepare additional TRAIN-root inflections; no model or sentence evaluation.

Surface attestation and synthetic Romanization are separate provenance claims.
All old held pairs are used only to quarantine connected additions, never to
select additions by model accuracy. Existing dataset bytes remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_phonetic_training_families as family
from hanglish.core.lexical_data import invalid_past_duplicate
import csv

BASE = ROOT / ".omx/artifacts/phonetic-supported-data-v1"
OUTPUT = ROOT / ".omx/artifacts/attested-inflection-delta-v2"
PINS = {
    BASE / "pairs.jsonl": "cf3e43f75420c17fbe8e1ec118a29a0f6a71c0e55cedbdbae5d7ef3668c84a84",
    BASE / "manifest.json": "9a7d7a7f2b0fab35ba31c8aae20b28d5af3a5bbfd0d56cb67afb4c0e45a16bba",
    family.MECAB / "Inflect.csv": "ff9a82d6f97aa75b86f03c33e2d615486200d7433f1c86c2e33a8c18d647064c",
}
ENDING_POS = frozenset({"EC", "EF", "ETM", "ETN"})
ROLES = frozenset({"train", "lexical_holdout", "morph_combination_holdout"})


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lexical_roots(analyses):
    return {part for analysis in analyses for part in analysis
            if part[1] in family.LEXICAL_POS}


def inherited_roots(row):
    return {(part["surface"], part["pos"]) for part in row["family_roots"]}


def propose(base, analyses, accepted_surfaces, romanize=family.eligible_pair):
    anchors = {root for row in base if row["split"] == "train"
               for root in inherited_roots(row)}
    existing = {(row["roman"], row["surface"]) for row in base}
    candidates, counts = [], Counter()
    for surface, variants in sorted(analyses.items()):
        if surface not in accepted_surfaces:
            continue
        supported = sorted(analysis for analysis in variants
                           if analysis[-1][1] in ENDING_POS
                           and (lexical := family.first_lexical(analysis)) is not None
                           and (lexical[1], lexical[2]) in anchors)
        if not supported:
            continue
        counts["anchored_attested_surfaces"] += 1
        pair = romanize(surface)
        if pair is None:
            counts["outside_length_or_script_contract"] += 1
            continue
        if pair in existing:
            counts["existing_pair"] += 1
            continue
        candidates.append({
            "roman": pair[0], "surface": pair[1], "split": "train",
            "anchoring_analyses": [[list(part) for part in a] for a in supported],
            "all_lexical_roots": [list(root) for root in sorted(lexical_roots(variants))],
            "input_provenance": "bounded_romanize_word_synthetic_not_independent_pronunciation",
            "surface_provenance": "exact_accepted_mecab_inflect_surface",
        })
    counts["proposed_new_pairs"] = len(candidates)
    return candidates, dict(counts)


def quarantine(base, candidates, analyses):
    """Close bridges transitively before consulting held roles."""
    if any(row.get("split") not in ROLES for row in base):
        raise ValueError("unknown original split")
    pairs = [(row["roman"], row["surface"]) for row in [*base, *candidates]]
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate original or proposed pair")
    union = family.UnionFind()
    owners = {}
    for index, row in enumerate([*base, *candidates]):
        union.add(index)
        roots = lexical_roots(analyses.get(row["surface"], ()))
        if index < len(base):
            roots |= inherited_roots(row)
        keys = [("roman", row["roman"]), ("surface", row["surface"])]
        keys.extend(("root", *root) for root in roots)
        if index < len(base):
            keys.append(("family", row["family_sha256"]))
        for key in keys:
            if key in owners:
                union.union(index, owners[key])
            else:
                owners[key] = index
    members = defaultdict(list)
    for index in range(len(pairs)):
        members[union.find(index)].append(index)
    held = defaultdict(set)
    for index, row in enumerate(base):
        if row["split"] != "train":
            held[union.find(index)].add(row["split"])
    component_ids = {
        root: hashlib.sha256(canonical(sorted(pairs[i] for i in indices))).hexdigest()
        for root, indices in members.items()
    }
    accepted, rejected = [], []
    for offset, candidate in enumerate(candidates, len(base)):
        root = union.find(offset)
        if held[root]:
            rejected.append({
                "pair_sha256": hashlib.sha256(canonical(pairs[offset])).hexdigest(),
                "component_sha256": component_ids[root],
                "held_splits": sorted(held[root]),
            })
        else:
            accepted.append({**candidate, "component_sha256": component_ids[root]})
    return accepted, rejected


def build(output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    for path, expected in PINS.items():
        if digest(path) != expected:
            raise ValueError(f"source hash differs: {path}")
    base = [json.loads(line) for line in (BASE / "pairs.jsonl").read_text().splitlines()]
    manifest = json.loads((BASE / "manifest.json").read_text())
    if len(base) != 54676 or dict(Counter(r["split"] for r in base)) != manifest["split_counts"]:
        raise ValueError("original dataset differs")
    if manifest["pairs_jsonl_sha256"] != PINS[BASE / "pairs.jsonl"]:
        raise ValueError("original manifest does not authenticate pairs")
    source_paths = {Path(__file__), ROOT / "scripts/audit_phonetic_training_families.py",
                    ROOT / "hanglish/core/lexical_data.py", ROOT / "hanglish/core/phonology.py",
                    ROOT / "hanglish/core/predicate_boundaries.py",
                    ROOT / "hanglish/core/data/predicate_boundary_inventory.json",
                    family.MECAB / "SOURCE.json"}
    source_paths.update(ROOT / key for key in manifest["source_hashes"])
    pins = {str(path.relative_to(ROOT)): digest(path) for path in source_paths | set(PINS)}
    analyses = family.read_inflect()
    raw = list(csv.reader((family.MECAB / "Inflect.csv").read_text().splitlines()))
    siblings = {(row[0], row[11]) for row in raw if len(row) >= 12}
    accepted_surfaces = {row[0] for row in raw if len(row) >= 12
                         and not invalid_past_duplicate(row, siblings)}
    proposed, counts = propose(base, analyses, accepted_surfaces)
    added, rejected = quarantine(base, proposed, analyses)
    counts.update(accepted_new_pairs=len(added), quarantined_new_pairs=len(rejected),
                  accepted_components=len({r["component_sha256"] for r in added}),
                  proposed_components=len({r["component_sha256"] for r in [*added, *rejected]}))
    if counts["proposed_new_pairs"] != len(added) + len(rejected):
        raise AssertionError("candidate accounting differs")
    for key, expected in pins.items():
        if digest(ROOT / key) != expected:
            raise ValueError(f"source changed during build: {key}")
    output.mkdir(parents=True)
    outputs = {}
    for name, rows in [("delta.jsonl", added), ("quarantine.jsonl", rejected)]:
        content = b"".join(canonical(row) + b"\n" for row in rows)
        (output / name).write_bytes(content)
        outputs[name] = {"rows": len(rows), "sha256": hashlib.sha256(content).hexdigest()}
    result = {
        "schema": "attested-inflection-delta-v1", "completed": True,
        "counts": counts, "original_splits": manifest["split_counts"],
        "source_hashes": pins, "outputs": outputs,
        "training_ready": False, "model_activity": False, "production_changed": False,
        "held_pair_access": "original held pair spellings, Roman inputs and roots used solely for exclusion closure",
        "sentence_dev_hold_test_read": False,
        "limitations": [
            "Surface attestation is not independent pronunciation verification or grammaticality in context.",
            "Closure covers recorded roots and valid MeCab analyses; unknown lexical relationships may remain.",
            "Prior held splits have already been evaluated; they are preservation checks, not fresh evaluation.",
            "Additional source and pronunciation review and a fixed comparison plan are required before training.",
        ],
    }
    (output / "manifest.json").write_bytes(canonical(result) + b"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    result = build(parser.parse_args().output)
    print(json.dumps(result["counts"], ensure_ascii=False))
