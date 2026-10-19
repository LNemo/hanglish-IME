"""Freeze fresh evaluation IDs from the v1 reserve, without using target text."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARTITION = ROOT / ".omx/artifacts/final-choice-partition-v1/manifest.json"
PARTITION_SHA = "e997f06558bdb5633446171d35912140fb34b55066c13fa13b668939d6347846"
RAW = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
RAW_SHA = "904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b"
PLAN = ROOT / ".omx/artifacts/balanced-final-choice-plan-v2.md"
PLAN_SHA = "004990ee8f30c049e03b2bbd38e1b0bdd71fcac3ee4a7ca72c7d6821560adbbc"
SALT = "balanced-final-choice-v2-2026-09-15"
TARGET = 1024
OUTPUT = ROOT / ".omx/artifacts/balanced-final-choice-partition-v2/manifest.json"


def sha(value):
    return hashlib.sha256(value).hexdigest()


def select_ids(metadata, natural, target=TARGET):
    """Inspect only id/group_id; retain complete groups and original row order."""
    if type(target) is not int or target <= 0:
        raise ValueError("target must be a positive integer")
    reserve_ids = natural["reserve_ids"]
    reserve_groups = natural["reserve_group_ids"]
    if (not reserve_ids or not reserve_groups
            or len(reserve_ids) != len(set(reserve_ids))
            or len(reserve_groups) != len(set(reserve_groups))):
        raise ValueError("invalid reserve identity")
    protected_ids = set().union(*(set(natural[k]) for k in (
        "fit_ids", "calibration_ids", "holdout_ids", "excluded_historical_group_sibling_ids")))
    protected_groups = set().union(*(set(natural[k]) for k in (
        "excluded_historical_group_ids", "calibration_group_ids", "holdout_group_ids")))
    if protected_ids.intersection(reserve_ids) or protected_groups.intersection(reserve_groups):
        raise ValueError("reserve overlaps exposed partitions")
    by_group = defaultdict(list)
    seen = set()
    reserve_set, group_set = set(reserve_ids), set(reserve_groups)
    for row in metadata:
        identity, group = row["id"], row["group_id"]
        if not isinstance(identity, str) or not identity or not isinstance(group, str) or not group:
            raise ValueError("invalid source identity")
        if identity in seen:
            raise ValueError("duplicate source ID")
        seen.add(identity)
        if (identity in reserve_set) != (group in group_set):
            raise ValueError("reserve does not contain complete source groups")
        if identity in reserve_set:
            by_group[group].append(identity)
    if set(by_group) != group_set or set(x for ids in by_group.values() for x in ids) != reserve_set:
        raise ValueError("source reserve identity mismatch")
    if [identity for group in reserve_groups for identity in by_group[group]] != reserve_ids:
        raise ValueError("v1 reserve source order differs")
    ordered = sorted(group_set, key=lambda group: (sha((SALT + "\0" + group).encode()), group))
    chosen_groups, chosen_ids = [], []
    for group in ordered:
        if len(chosen_ids) >= target:
            break
        chosen_groups.append(group)
        chosen_ids.extend(by_group[group])
    if len(chosen_ids) < target:
        raise ValueError("insufficient untouched reserve")
    return {
        "eval_group_ids": chosen_groups, "eval_ids": chosen_ids,
        "remaining_group_ids": ordered[len(chosen_groups):],
        "remaining_ids": [identity for group in ordered[len(chosen_groups):] for identity in by_group[group]],
    }


def run(output):
    if output.exists():
        raise FileExistsError(output)
    pins = {PARTITION: PARTITION_SHA, RAW: RAW_SHA, PLAN: PLAN_SHA}
    snapshots = {p: p.read_bytes() for p in pins}
    if any(sha(snapshots[p]) != expected for p, expected in pins.items()):
        raise ValueError("pinned v2 partition input changed")
    manifest = json.loads(snapshots[PARTITION])
    # Parsing verifies the complete pinned file; only identity metadata is used.
    metadata = [{"id": row["id"], "group_id": row["group_id"]}
                for line in snapshots[RAW].splitlines() if line
                for row in [json.loads(line)]]
    if len(metadata) != 9002:
        raise ValueError("raw TRAIN row count differs")
    selected = select_ids(metadata, manifest["natural"])
    if any(p.read_bytes() != snapshots[p] for p in pins):
        raise RuntimeError("partition inputs changed during selection")
    report = {
        "schema": "balanced-final-choice-fresh-evaluation-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "salt": SALT, "target_rows": TARGET, **selected,
        "rows": len(selected["eval_ids"]), "groups": len(selected["eval_group_ids"]),
        "source_hashes": {str(p.relative_to(ROOT)): h for p, h in pins.items()},
        "selector_plan_sha256": PLAN_SHA,
        "builder_sha256": sha(Path(__file__).read_bytes()),
        "selection_uses_target_text": False, "model_inference_run": False,
        "training_run": False, "old_splits_modified": False,
        "upstream_pretraining_overlap": "unknown",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"rows": report["rows"], "groups": report["groups"],
                      "manifest_sha256": sha(output.read_bytes())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)
