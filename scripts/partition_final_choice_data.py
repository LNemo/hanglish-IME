"""Build deterministic TRAIN-only partitions for final-choice learning.

This script reads no development or final corpus and performs no model
inference. Partitions depend only on source IDs, natural group IDs, challenge
families, and the explicit augmentation source/target overlap guard.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.neural_converter import file_sha256
from hanglish.core.phonology import romanize_word


OUTPUT = ROOT / ".omx/artifacts/final-choice-partition-v1"
NATURAL_RAW = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
HISTORICAL = ROOT / "data/natural_context/current-context-v1/natural_train.jsonl"
CHALLENGE = ROOT / "data/challenge/train.jsonl"
AUGMENTATION = ROOT / "data/challenge/context-augmentation/train.jsonl"
INPUTS = {
    "natural_raw_train": (NATURAL_RAW, "904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b", 9002),
    "historical_current_train": (HISTORICAL, "807d8040357c2cee05a9c286bb9dc04a37d90b82705900f8b09c23d3c0081ef4", 2048),
    "challenge_train": (CHALLENGE, "c7336caf348f0c4629c9ad65581d264b53df40a04474e67ae3bce18d501dd9b6", 1820),
    "augmentation_train": (AUGMENTATION, "001388eb3408d0cb9ad782f9dc67a47cd3893a4070b5744d4b44cf964cc81a31", 240),
}
SALT = "final-choice-v1-2026-09-15"
CALIBRATION_TARGET = 1024
HOLDOUT_TARGET = 512
HISTORICAL_GROUP_COUNT = 1827


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def read_jsonl_bytes(value):
    return [json.loads(line) for line in value.decode().splitlines() if line]


def read_inputs(inputs=INPUTS):
    rows, snapshots = {}, {}
    for name, (path, expected, count) in inputs.items():
        raw = path.read_bytes()
        if sha256_bytes(raw) != expected:
            raise ValueError(f"pinned TRAIN input hash differs: {name}")
        parsed = read_jsonl_bytes(raw)
        if len(parsed) != count:
            raise ValueError(f"pinned TRAIN input row count differs: {name}")
        rows[name] = parsed
        try:
            display_path = str(path.relative_to(ROOT))
        except ValueError:
            display_path = str(path)
        snapshots[name] = {"path": display_path, "sha256": expected,
                           "bytes": len(raw), "rows": count}
    return rows, snapshots


def unique_rows(rows, role, required):
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key]
                                            for key in required):
            raise ValueError(f"{role} row is missing a required source field")
        if row["id"] in seen:
            raise ValueError(f"{role} contains duplicate IDs")
        seen.add(row["id"])
    return seen


def salted_key(salt, value):
    return hashlib.sha256((salt + "\0" + value).encode("utf-8")).hexdigest(), value


def take_whole_groups(group_rows, ordered_groups, start, target):
    chosen, count, cursor = [], 0, start
    while cursor < len(ordered_groups) and count < target:
        group = ordered_groups[cursor]
        chosen.append(group)
        count += len(group_rows[group])
        cursor += 1
    if count < target:
        raise ValueError("not enough whole natural groups for requested partition")
    return chosen, cursor


def source_target_pairs(rows):
    pairs = set()
    for row in rows:
        expected = row.get("expected")
        if (not isinstance(row.get("input"), str) or not row["input"]
                or not isinstance(expected, list) or not expected
                or any(not isinstance(target, str) or not target for target in expected)):
            raise ValueError("source/target overlap guard received malformed rows")
        pairs.update((row["input"], target) for target in expected)
    return pairs



def augmentation_decision(rows, natural_partition, preservation_ids):
    natural_by_id = {row["id"]: row for row in rows["natural_raw_train"]}
    challenge_by_id = {row["id"]: row for row in rows["challenge_train"]}
    protected = {}
    for role in ("calibration", "holdout"):
        protected["natural_" + role] = {
            (romanize_word(natural_by_id[source_id]["text"]),
             natural_by_id[source_id]["text"])
            for source_id in natural_partition[role + "_ids"]
        }
    protected["challenge_preservation"] = source_target_pairs(
        [challenge_by_id[source_id] for source_id in preservation_ids]
    )
    pairs = source_target_pairs(rows["augmentation_train"])
    overlaps = {role: len(pairs & values) for role, values in protected.items()}
    included = not any(overlaps.values())
    all_ids = [row["id"] for row in rows["augmentation_train"]]
    return {
        "included_in_fit": included,
        "fit_ids": all_ids if included else [],
        "omitted_ids": [] if included else all_ids,
        "decision_reason": ("no_exact_source_target_overlap_with_protected_splits"
                            if included else
                            "omitted_exact_source_target_overlap_with_protected_splits"),
        "protected_exact_source_target_overlap_counts": overlaps,
    }


def build_partition(rows, calibration_target=CALIBRATION_TARGET,
                    holdout_target=HOLDOUT_TARGET):
    natural = rows["natural_raw_train"]
    historical = rows["historical_current_train"]
    challenge = rows["challenge_train"]
    augmentation = rows["augmentation_train"]
    natural_ids = unique_rows(natural, "natural raw TRAIN", ("id", "group_id"))
    historical_ids = unique_rows(historical, "historical current TRAIN", ("id", "group_id"))
    unique_rows(challenge, "challenge TRAIN", ("id", "family"))
    unique_rows(augmentation, "augmentation TRAIN", ("id", "input"))
    if any(row.get("split") != "train" for values in rows.values() for row in values):
        raise ValueError("partition input contains a non-TRAIN row")
    if not historical_ids <= natural_ids:
        raise ValueError("historical FIT IDs are absent from raw natural TRAIN")
    natural_by_id = {row["id"]: row for row in natural}
    if any(natural_by_id[row["id"]]["group_id"] != row["group_id"] for row in historical):
        raise ValueError("historical natural group lineage differs")

    historical_groups = {natural_by_id[source_id]["group_id"] for source_id in historical_ids}
    group_rows = defaultdict(list)
    for row in natural:
        if row["group_id"] not in historical_groups:
            group_rows[row["group_id"]].append(row["id"])
    ordered_groups = sorted(group_rows, key=lambda group: salted_key(SALT, group))
    calibration_groups, cursor = take_whole_groups(
        group_rows, ordered_groups, 0, calibration_target
    )
    holdout_groups, cursor = take_whole_groups(
        group_rows, ordered_groups, cursor, holdout_target
    )
    reserve_groups = ordered_groups[cursor:]

    challenge_by_family = defaultdict(list)
    for row in challenge:
        challenge_by_family[row["family"]].append(row["id"])
    preservation_families = sorted(
        (family for family in challenge_by_family
         if int(salted_key(SALT, family)[0], 16) % 5 == 0),
        key=lambda family: salted_key(SALT, family),
    )
    preservation_family_set = set(preservation_families)
    fit_families = sorted(
        (family for family in challenge_by_family if family not in preservation_family_set),
        key=lambda family: salted_key(SALT, family),
    )
    challenge_fit_ids = [source_id for family in fit_families
                         for source_id in challenge_by_family[family]]
    challenge_preservation_ids = [source_id for family in preservation_families
                                  for source_id in challenge_by_family[family]]

    def ids_for(groups):
        return [source_id for group in groups for source_id in group_rows[group]]

    result = {
        "policy": {
            "salt": SALT,
            "natural_calibration_target_rows": calibration_target,
            "natural_holdout_target_rows": holdout_target,
            "challenge_preservation_rule": "salted family SHA-256 modulo5 equals0",
            "challenge_fit_use": "selector_and_distillation_model_optimization",
            "challenge_preservation_use": "evaluation_only_never_model_optimization",
            "id_group_family_assignment_uses_labels": False,
            "augmentation_guard_uses_labels_only_for_overlap_exclusion": True,
        },
        "natural": {
            "fit_ids": [row["id"] for row in historical],
            "excluded_historical_group_ids": sorted(historical_groups),
            "excluded_historical_group_sibling_ids": [
                row["id"] for row in natural
                if row["group_id"] in historical_groups and row["id"] not in historical_ids
            ],
            "calibration_group_ids": calibration_groups,
            "calibration_ids": ids_for(calibration_groups),
            "holdout_group_ids": holdout_groups,
            "holdout_ids": ids_for(holdout_groups),
            "reserve_group_ids": reserve_groups,
            "reserve_ids": ids_for(reserve_groups),
        },
        "challenge": {
            "fit_family_ids": fit_families,
            "fit_ids": challenge_fit_ids,
            "preservation_family_ids": preservation_families,
            "preservation_ids": challenge_preservation_ids,
        },
    }
    result["augmentation"] = augmentation_decision(
        rows, result["natural"], challenge_preservation_ids
    )
    validate_partition(rows, result, calibration_target, holdout_target)
    return result


def validate_partition(rows, partition, calibration_target=CALIBRATION_TARGET,
                       holdout_target=HOLDOUT_TARGET):
    natural = rows["natural_raw_train"]
    historical = rows["historical_current_train"]
    challenge = rows["challenge_train"]
    raw_by_id = {row["id"]: row for row in natural}
    natural_data = partition["natural"]
    policy = partition.get("policy", {})
    if (policy.get("salt") != SALT
            or policy.get("natural_calibration_target_rows") != calibration_target
            or policy.get("natural_holdout_target_rows") != holdout_target
            or policy.get("challenge_preservation_rule") !=
            "salted family SHA-256 modulo5 equals0"
            or policy.get("challenge_fit_use") !=
            "selector_and_distillation_model_optimization"
            or policy.get("challenge_preservation_use") !=
            "evaluation_only_never_model_optimization"
            or policy.get("id_group_family_assignment_uses_labels") is not False
            or policy.get("augmentation_guard_uses_labels_only_for_overlap_exclusion") is not True):
        raise ValueError("partition policy or salts differ")
    if natural_data.get("fit_ids") != [row["id"] for row in historical]:
        raise ValueError("natural FIT IDs differ from exact historical order")
    roles = (natural_data.get("fit_ids", []), natural_data.get("calibration_ids", []),
             natural_data.get("holdout_ids", []), natural_data.get("reserve_ids", []))
    if any(len(values) != len(set(values)) for values in roles):
        raise ValueError("natural partition contains duplicate IDs")
    if any(set(left) & set(right) for index, left in enumerate(roles)
           for right in roles[index + 1:]):
        raise ValueError("natural partition IDs overlap")
    excluded = set(natural_data.get("excluded_historical_group_ids", []))
    if excluded != {raw_by_id[row["id"]]["group_id"] for row in historical}:
        raise ValueError("historical groups are not exactly excluded")
    historical_ids = set(natural_data["fit_ids"])
    expected_siblings = [row["id"] for row in natural
                         if row["group_id"] in excluded and row["id"] not in historical_ids]
    if natural_data.get("excluded_historical_group_sibling_ids") != expected_siblings:
        raise ValueError("historical group siblings are not explicitly excluded")
    for role, group_key in (("calibration_ids", "calibration_group_ids"),
                            ("holdout_ids", "holdout_group_ids"),
                            ("reserve_ids", "reserve_group_ids")):
        ids = natural_data.get(role, [])
        groups = natural_data.get(group_key, [])
        observed = {raw_by_id[source_id]["group_id"] for source_id in ids}
        if observed != set(groups) or observed & excluded:
            raise ValueError("natural role split a group or reused historical groups")
        all_group_ids = {row["id"] for row in natural if row["group_id"] in observed}
        if set(ids) != all_group_ids:
            raise ValueError("natural role does not contain whole groups")
    if len(natural_data["calibration_ids"]) < calibration_target or len(
            natural_data["holdout_ids"]) < holdout_target:
        raise ValueError("natural calibration or holdout target was not reached")
    remaining_ids = {row["id"] for row in natural if row["group_id"] not in excluded}
    if set().union(*(set(values) for values in roles[1:])) != remaining_ids:
        raise ValueError("remaining natural groups are not completely partitioned")
    remaining_groups = defaultdict(list)
    for row in natural:
        if row["group_id"] not in excluded:
            remaining_groups[row["group_id"]].append(row["id"])
    ordered = sorted(remaining_groups, key=lambda group: salted_key(SALT, group))
    expected_calibration, cursor = take_whole_groups(
        remaining_groups, ordered, 0, calibration_target
    )
    expected_holdout, cursor = take_whole_groups(
        remaining_groups, ordered, cursor, holdout_target
    )
    if (natural_data["calibration_group_ids"] != expected_calibration
            or natural_data["holdout_group_ids"] != expected_holdout
            or natural_data["reserve_group_ids"] != ordered[cursor:]):
        raise ValueError("natural group hash ordering differs")
    for role, expected_groups in (
            ("calibration", expected_calibration), ("holdout", expected_holdout),
            ("reserve", ordered[cursor:])):
        expected_ids = [source_id for group in expected_groups
                        for source_id in remaining_groups[group]]
        if natural_data[role + "_ids"] != expected_ids:
            raise ValueError("natural within-group ID ordering differs")

    challenge_data = partition["challenge"]
    fit, preservation = set(challenge_data.get("fit_ids", [])), set(
        challenge_data.get("preservation_ids", []))
    if fit & preservation or fit | preservation != {row["id"] for row in challenge}:
        raise ValueError("challenge IDs overlap or are incomplete")
    family_by_id = {row["id"]: row["family"] for row in challenge}
    fit_families = {family_by_id[source_id] for source_id in fit}
    preservation_families = {family_by_id[source_id] for source_id in preservation}
    if (fit_families & preservation_families
            or fit_families != set(challenge_data.get("fit_family_ids", []))
            or preservation_families != set(challenge_data.get("preservation_family_ids", []))):
        raise ValueError("challenge family partition leaks")
    challenge_by_family = defaultdict(list)
    for row in challenge:
        challenge_by_family[row["family"]].append(row["id"])
    expected_preservation = sorted(
        (family for family in challenge_by_family
         if int(salted_key(SALT, family)[0], 16) % 5 == 0),
        key=lambda family: salted_key(SALT, family),
    )
    expected_preservation_set = set(expected_preservation)
    expected_fit = sorted(
        (family for family in challenge_by_family
         if family not in expected_preservation_set),
        key=lambda family: salted_key(SALT, family),
    )
    if (challenge_data.get("fit_family_ids") != expected_fit
            or challenge_data.get("preservation_family_ids") != expected_preservation
            or challenge_data.get("fit_ids") != [
                source_id for family in expected_fit
                for source_id in challenge_by_family[family]
            ]
            or challenge_data.get("preservation_ids") != [
                source_id for family in expected_preservation
                for source_id in challenge_by_family[family]
            ]):
        raise ValueError("challenge family hash split differs")

    if partition["augmentation"] != augmentation_decision(
            rows, natural_data, challenge_data["preservation_ids"]):
        raise ValueError("augmentation source/target overlap decision differs")


def manifest_counts(partition):
    return {
        "natural_fit_rows": len(partition["natural"]["fit_ids"]),
        "natural_excluded_historical_groups": len(partition["natural"]["excluded_historical_group_ids"]),
        "natural_excluded_historical_group_siblings": len(partition["natural"]["excluded_historical_group_sibling_ids"]),
        "natural_calibration_rows": len(partition["natural"]["calibration_ids"]),
        "natural_calibration_groups": len(partition["natural"]["calibration_group_ids"]),
        "natural_holdout_rows": len(partition["natural"]["holdout_ids"]),
        "natural_holdout_groups": len(partition["natural"]["holdout_group_ids"]),
        "natural_reserve_rows": len(partition["natural"]["reserve_ids"]),
        "challenge_fit_rows": len(partition["challenge"]["fit_ids"]),
        "challenge_preservation_rows": len(partition["challenge"]["preservation_ids"]),
        "augmentation_fit_rows": len(partition["augmentation"]["fit_ids"]),
    }


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    source = Path(__file__).read_bytes()
    source_sha256 = sha256_bytes(source)
    dependency_hashes = {"hanglish/core/phonology.py": file_sha256(ROOT / "hanglish/core/phonology.py")}
    rows, inputs = read_inputs()
    partition = build_partition(rows)
    if (len(partition["natural"]["fit_ids"]) != 2048
            or len(partition["natural"]["excluded_historical_group_ids"]) != HISTORICAL_GROUP_COUNT):
        raise ValueError("historical natural FIT count or group count differs")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "deterministic TRAIN-only final-choice source partition",
        "inputs": inputs,
        "source_sha256": source_sha256,
        "dependency_hashes": dependency_hashes,
        **partition,
        "counts": manifest_counts(partition),
        "model_inference_run": False,
        "training_run": False,
        "development_data_read": False,
        "test_data_read": False,
        "deployed": False,
    }
    args.output.mkdir(parents=True)
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    source_path = args.output / "source.py"
    source_path.write_bytes(source)
    for name, (path, expected, _) in INPUTS.items():
        if file_sha256(path) != expected:
            raise ValueError(f"TRAIN input changed after partitioning: {name}")
    for relative, expected in dependency_hashes.items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError("partition romanizer changed")
    validate_partition(rows, manifest)
    if (file_sha256(Path(__file__)) != source_sha256
            or file_sha256(source_path) != source_sha256):
        raise ValueError("partition source or snapshot changed")
    print(json.dumps({"counts": manifest["counts"],
                      "augmentation": manifest["augmentation"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
