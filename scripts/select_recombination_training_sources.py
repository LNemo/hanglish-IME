"""Select fresh TRAIN-only sentence inputs for future recombination capture."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_recombination_ranker_data import assign_components


TRAIN = ROOT / "data/challenge/sentence-v3/train.jsonl"
OLD_SELECTION = ROOT / ".omx/artifacts/roman-sentence-train-capture-v1/selection.json"
HELPER = ROOT / "scripts/prepare_recombination_ranker_data.py"
PLAN = ROOT / ".omx/plans/recombination-fresh-train-v1.md"
TRAIN_SHA256 = "e73ac3b3baf41075028c3b5629b4412336926faafdb8353b20c61a200bda4c42"
OLD_SELECTION_SHA256 = "bb2bc59d7e0b44a47e1661eb20e045ba0531c4ed804a4d683c04b0e8c9ff03f1"
SCHEMA = "recombination-fresh-train-sources-v1"
SELECTION_SALT = "recombination-fresh-train-v1"
EXPECTED_TRAIN_ROWS = 5656
EXPECTED_OLD_ROWS = 512
TOTAL_ROWS = 2048
NATURAL_ROWS = 1536
OTHER_ROWS = 512
NATURAL_CATEGORY = "natural_sentence_context"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def stable_key(row):
    digest = hashlib.sha256(
        (SELECTION_SALT + "\0" + row["id"]).encode()).hexdigest()
    return digest, row["id"]


def _round_robin(queues, limit):
    selected = []
    categories = deque(sorted(key for key, values in queues.items() if values))
    while categories and len(selected) < limit:
        category = categories.popleft()
        selected.append(queues[category].popleft())
        if queues[category]:
            categories.append(category)
    return selected


def select(rows, old_ids, *, total_rows=TOTAL_ROWS,
           natural_rows=NATURAL_ROWS, other_rows=OTHER_ROWS):
    """Exclude old-linked components and select by identity/category only."""
    require(isinstance(rows, list) and rows, "TRAIN rows are missing")
    identifiers = [row.get("id") for row in rows]
    require(all(isinstance(value, str) and value for value in identifiers)
            and len(identifiers) == len(set(identifiers)),
            "TRAIN IDs must be nonempty and unique")
    require(isinstance(old_ids, list)
            and all(isinstance(value, str) and value for value in old_ids)
            and len(old_ids) == len(set(old_ids))
            and set(old_ids) <= set(identifiers),
            "old selection IDs are missing or duplicate")
    require(total_rows == natural_rows + other_rows,
            "selection quotas do not sum to total")
    for row in rows:
        require(row.get("split") == "train"
                and row.get("input_contract") == "roman_sentence_only"
                and isinstance(row.get("category"), str) and row["category"]
                and isinstance(row.get("input"), str),
                "TRAIN selection metadata is malformed")
    assignments = assign_components(rows)
    blocked_components = {assignments[identifier]["component_id"]
                          for identifier in old_ids}
    eligible = [row for row in rows
                if assignments[row["id"]]["component_id"]
                not in blocked_components]
    natural = sorted((row for row in eligible
                      if row["category"] == NATURAL_CATEGORY), key=stable_key)
    other_queues = defaultdict(deque)
    for row in sorted((row for row in eligible
                       if row["category"] != NATURAL_CATEGORY), key=stable_key):
        other_queues[row["category"]].append(row)
    chosen_natural = natural[:natural_rows]
    natural_remaining = natural[natural_rows:]
    chosen_other = _round_robin(other_queues, other_rows)
    if len(chosen_other) < other_rows:
        shortage = other_rows - len(chosen_other)
        chosen_natural.extend(natural_remaining[:shortage])
        natural_remaining = natural_remaining[shortage:]
    if len(chosen_natural) < natural_rows:
        shortage = natural_rows - len(chosen_natural)
        chosen_other.extend(_round_robin(other_queues, shortage))
    selected = chosen_natural + chosen_other
    require(len(selected) == total_rows,
            "insufficient eligible TRAIN rows for fixed selection")
    selected = sorted(selected, key=stable_key)
    selected_ids = {row["id"] for row in selected}
    require(not selected_ids & set(old_ids)
            and not ({assignments[value]["component_id"]
                      for value in selected_ids} & blocked_components),
            "old TRAIN component entered fresh selection")
    all_components = {value["component_id"] for value in assignments.values()}
    diagnostics = {
        "all_train_components": len(all_components),
        "blocked_components": len(blocked_components),
        "blocked_rows": sum(value["component_id"] in blocked_components
                            for value in assignments.values()),
        "eligible_components": len(all_components - blocked_components),
        "eligible_rows": len(eligible),
        "old_selection_rows": len(old_ids),
    }
    return selected, assignments, diagnostics


def run(output, *, train_path=TRAIN, old_selection_path=OLD_SELECTION,
        train_sha=TRAIN_SHA256, old_selection_sha=OLD_SELECTION_SHA256,
        expected_train_rows=EXPECTED_TRAIN_ROWS,
        expected_old_rows=EXPECTED_OLD_ROWS, total_rows=TOTAL_ROWS,
        natural_rows=NATURAL_ROWS, other_rows=OTHER_ROWS):
    """Write a label-free source selection and its global TRAIN partition."""
    require(sha256_file(train_path) == train_sha,
            "canonical TRAIN hash differs")
    require(sha256_file(old_selection_path) == old_selection_sha,
            "old TRAIN selection hash differs")
    rows = read_jsonl(train_path)
    old_ids = json.loads(Path(old_selection_path).read_text(encoding="utf-8"))
    require(len(rows) == expected_train_rows,
            "canonical TRAIN row count differs")
    require(isinstance(old_ids, list) and len(old_ids) == expected_old_rows,
            "old TRAIN selection row count differs")
    selected, assignments, diagnostics = select(
        rows, old_ids, total_rows=total_rows, natural_rows=natural_rows,
        other_rows=other_rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    selection_path = output / "selection.jsonl"
    with selection_path.open("x", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps({"id": row["id"], "input": row["input"]},
                                    ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    selected_ids = [row["id"] for row in selected]
    pins = {}
    for path in (Path(train_path), Path(old_selection_path), HELPER, PLAN,
                 Path(__file__).resolve()):
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        pins[name] = sha256_file(path)
    require(pins[str(Path(train_path).relative_to(ROOT))
                 if Path(train_path).is_relative_to(ROOT)
                 else str(train_path)] == train_sha,
            "canonical TRAIN changed during selection")
    require(pins[str(Path(old_selection_path).relative_to(ROOT))
                 if Path(old_selection_path).is_relative_to(ROOT)
                 else str(old_selection_path)] == old_selection_sha,
            "old TRAIN selection changed during selection")
    partition = [{"id": identifier,
                  "component_id": assignments[identifier]["component_id"],
                  "split": assignments[identifier]["split"]}
                 for identifier in selected_ids]
    manifest = {
        "schema": SCHEMA, "completed": True,
        "input_contract": "roman_sentence_only",
        "rows": len(selected), "selected_ids": selected_ids,
        "counts": {"rows": len(selected),
                   "natural": sum(row["category"] == NATURAL_CATEGORY
                                  for row in selected),
                   "other": sum(row["category"] != NATURAL_CATEGORY
                                for row in selected)},
        "category_counts": dict(sorted(Counter(
            row["category"] for row in selected).items())),
        "component_counts": diagnostics, "partition": partition,
        "selection_policy": {
            "salt": SELECTION_SALT, "total_rows": total_rows,
            "natural_rows": natural_rows, "other_rows": other_rows,
            "other_policy": "stable category round-robin",
            "quota_fallback": "short side filled from remaining opposite side",
        },
        "outputs": {"selection.jsonl": {
            "sha256": sha256_file(selection_path), "rows": len(selected),
            "fields": ["id", "input"]}},
        "pins": dict(sorted(pins.items())),
        "component_metadata_used_for_exclusion_and_partition": True,
        "targets_in_selection_artifact": False,
        "model_loaded": False, "model_inference": False,
        "dev_accessed": False, "test_accessed": False,
        "production_changed": False,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    print(json.dumps({key: result[key] for key in
                      ("completed", "rows", "category_counts",
                       "component_counts")}, ensure_ascii=False), flush=True)
