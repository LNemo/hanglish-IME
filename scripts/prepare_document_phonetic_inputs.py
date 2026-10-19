"""Freeze source-only word requests from the evaluation-only document reserve.

References are parsed to synthesize Roman inputs and count their support, never
to score a model or choose a subset by performance. No reference text is emitted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.phonology import romanize_word

RESERVE = ROOT / ".omx/artifacts/document-reserve-v2"
RAW = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
OUTPUT = ROOT / ".omx/artifacts/document-phonetic-inputs-v1"
PINS = {
    RAW: "904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b",
    RESERVE / "selection.jsonl": "7d62027d11dcf1320b2a1bac7bcb83e1a31d31ce521808bbe4d1faadf6d4b810",
    RESERVE / "manifest.json": "19223e3c507432c168205732c26c4a95dc1e2391a0698dea96eaac63d70c3859",
    ROOT / "hanglish/core/phonology.py": "e593f06409646b74d8aa8ca62663dcdd1b62759254ae1d1d23fc948fed54b865",
    ROOT / "hanglish/core/predicate_boundaries.py": "d0d29ca399220e6707e674ebf29a1368162e21741c1f0691d985d284ffc9abb1",
    ROOT / "hanglish/core/data/predicate_boundary_inventory.json": "5da10f07dd9b1a494ad7465c0e215c78982b6d7e252b1f603f7232cf714bb06a",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def project(rows, romanize=romanize_word):
    pairs, inputs, ids = set(), set(), set()
    counts = Counter()
    for row in rows:
        if not isinstance(row.get("id"), str) or row["id"] in ids:
            raise ValueError("invalid or duplicate document ID")
        if not isinstance(row.get("text"), str) or not row["text"]:
            raise ValueError("invalid document reference")
        ids.add(row["id"])
        words = re.findall("[가-힣]+", row["text"])
        counts["documents_without_hangul"] += not words
        for word in words:
            source = romanize(word)
            counts["reference_occurrences"] += 1
            pairs.add((source, word))
            if not re.fullmatch("[a-z]{1,48}", source):
                counts["input_unsupported_occurrences"] += 1
            else:
                inputs.add(source)
            counts["output_bound_exceeded_occurrences"] += len(word) > 24
    requests = [{"id": hashlib.sha256(s.encode()).hexdigest(), "input": s}
                for s in sorted(inputs, key=lambda s: (len(s), s))]
    counts.update(documents=len(ids), reference_pairs=len(pairs), unique_inputs=len(inputs),
                  input_unsupported_pairs=sum(not re.fullmatch("[a-z]{1,48}", s)
                                              for s, _ in pairs),
                  output_bound_exceeded_pairs=sum(len(t) > 24 for _, t in pairs))
    return requests, dict(counts)


def run(output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    for path, expected in PINS.items():
        if sha(path) != expected:
            raise ValueError(f"source changed: {path}")
    pins = {str(p.relative_to(ROOT)): v for p, v in PINS.items()}
    pins[str(Path(__file__).resolve().relative_to(ROOT))] = sha(__file__)
    selection = [json.loads(s) for s in (RESERVE / "selection.jsonl").read_text().splitlines()]
    raw = [json.loads(s) for s in RAW.read_text().splitlines()]
    by_id = {r["id"]: r for r in raw}
    if len(by_id) != len(raw) or len(selection) != 1046:
        raise ValueError("source or reserve cardinality differs")
    chosen = []
    for r in selection:
        match = by_id[r["id"]]
        if match["group_id"] != r["group_id"]:
            raise ValueError("reserve group mapping differs")
        chosen.append(match)
    requests, counts = project(chosen)
    content = b"".join(canonical(r) + b"\n" for r in requests)
    manifest = {
        "schema": "document-phonetic-inputs-v1", "completed": True,
        "counts": counts, "source_hashes": pins,
        "inputs_sha256": hashlib.sha256(content).hexdigest(),
        "input_length_histogram": dict(sorted(Counter(len(r["input"]) for r in requests).items())),
        "model_activity": False, "performance_labels_joined": False,
        "reference_access": "Whole references parsed only for maximal Hangul runs, synthetic inputs and support accounting.",
        "evaluation_contract": {
            "beam_size": 8, "context": "none", "single_syllable_inputs_included": True,
            "unsupported_references_remain_in_evaluation_denominator": True,
            "source_only_requests": True, "candidate_checkpoint_selection": "fixed final update3954",
        },
        "limits": [
            "Direct word candidate diagnostic, not sentence accuracy or English-copy testing.",
            "Same-input variant references stay distinct in the later joined reference set.",
            "Synthetic source spelling does not independently validate actual user pronunciation.",
            "Document holdout does not imply lexical-family or upstream-pretraining isolation.",
        ],
    }
    for path, expected in pins.items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"source changed during build: {path}")
    output.mkdir(parents=True)
    (output / "inputs.jsonl").write_bytes(content)
    (output / "manifest.json").write_bytes(canonical(manifest) + b"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    print(json.dumps(run(parser.parse_args().output)["counts"], ensure_ascii=False))
