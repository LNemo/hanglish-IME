"""Document/family-separated TRAIN-only partition for the pko-T5 pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.sentence_data import read_split

SEED = "pko-inner-validation-v1-20260921"
TRAIN_FILES = {
    "data/challenge/sentence-v3/train.jsonl":
        "e73ac3b3baf41075028c3b5629b4412336926faafdb8353b20c61a200bda4c42",
    "data/challenge/sentence-train-extra-v3/train.jsonl":
        "e0729e728c0c5bf2827c83b666380c4f8c4757ee204cdb248578327667a0539b",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def leakage_keys(row):
    """Transitive links include source documents and duplicate surface forms."""
    if not row.get("group_id") or not row.get("family"):
        raise ValueError("TRAIN row lacks document/family grouping")
    if not isinstance(row.get("source_ids"), list):
        raise ValueError("TRAIN row lacks source identity list")
    return {
        "group:" + row["group_id"], "family:" + row["family"],
        "input:" + normalized(row["input"]),
        *("target:" + normalized(t) for t in row["expected"]),
        *("source:" + s for s in row["source_ids"]),
    }


def partition(rows):
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("TRAIN IDs must be nonempty and unique")
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners = {}
    for i, row in enumerate(rows):
        for key in sorted(leakage_keys(row)):
            if key in owners:
                parents[find(i)] = find(owners[key])
            else:
                owners[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(row["id"])
    assignments = []
    for ids in groups.values():
        ids = sorted(ids)
        digest = hashlib.sha256((SEED + "\n" + "\n".join(ids)).encode()).hexdigest()
        split = "inner_validation" if int(digest[:16], 16) % 10 == 0 else "fit"
        assignments.append({"group_sha256": digest, "split": split, "ids": ids})
    assignments.sort(key=lambda g: g["group_sha256"])
    owner = {ident: g["split"] for g in assignments for ident in g["ids"]}
    split_keys = {s: set() for s in ("fit", "inner_validation")}
    for row in rows:
        split_keys[owner[row["id"]]].update(leakage_keys(row))
    if split_keys["fit"] & split_keys["inner_validation"]:
        raise ValueError("cross-partition document/family/surface leakage")
    return assignments


def run(output):
    for name, expected in TRAIN_FILES.items():
        if sha(ROOT / name) != expected:
            raise ValueError("canonical TRAIN checksum differs")
    files = [ROOT / name for name in TRAIN_FILES]
    files += [p.parent / "manifest.json" for p in files]
    files += [Path(__file__), ROOT / "scripts/sentence_data.py"]
    pins = {str(p.relative_to(ROOT)): sha(p) for p in files}
    rows = []
    for name in TRAIN_FILES:
        rows.extend(read_split((ROOT / name).parent, "train"))
    if len(rows) != 7704:
        raise ValueError("expected fixed TRAIN union of7704 rows")
    groups = partition(rows)
    counts = {s: sum(len(g["ids"]) for g in groups if g["split"] == s)
              for s in ("fit", "inner_validation")}
    if not all(counts.values()):
        raise ValueError("both partitions must contain rows")
    result = {
        "completed": True, "seed": SEED, "rows": len(rows), "counts": counts,
        "group_count": len(groups), "largest_group": max(len(g["ids"]) for g in groups),
        "groups": groups, "pins": pins,
        "policy": "hash-selected10% of transitive document/family/normalized-surface groups",
        "leakage_key_overlap": 0, "model_input_fields": ["input"],
        "targets_are_training_labels_only": True,
        "target_policy": "expected[0] for teacher-forced training; all expected for validation",
        "multiple_target_rows": sum(len(r["expected"]) > 1 for r in rows),
        "dev_accessed": False, "test_accessed": False, "model_inference_performed": False,
    }
    if any(sha(ROOT / name) != digest for name, digest in pins.items()):
        raise ValueError("partition sources changed during preparation")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("rows", "counts", "group_count", "largest_group")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
