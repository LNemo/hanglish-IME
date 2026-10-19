"""Freeze a previously unevaluated, source-only boundary evaluation reserve."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".omx/artifacts/boundary-reserve-v1"
V1 = ROOT / ".omx/artifacts/final-choice-partition-v1/manifest.json"
V2 = ROOT / ".omx/artifacts/balanced-final-choice-partition-v2/manifest.json"
V3 = ROOT / ".omx/artifacts/finite-final-choice-partition-v3/manifest.json"
RAW = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
EXPOSED = ROOT / ".omx/artifacts/long-context-lora-data-v1-review2/fresh_selection.jsonl"
TARGET = 512

PINS = {
    V1: "e997f06558bdb5633446171d35912140fb34b55066c13fa13b668939d6347846",
    V2: "57170914d4f14d5cd725ea0fd52a96b68b0859ba83041b93f1ecc09daa7cc8ca",
    V3: "8f699052b3e8d6e07dcaf59dc8696cc13a8ffb15b4e1bb5df62571e906ae3e61",
    RAW: "904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b",
    EXPOSED: "fb1257833c99acc406a831336f535d9ef3a786fc38bb08a84902598e5fe2287d",
}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def path_key(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read_pinned(path: Path, expected_sha256: str) -> bytes:
    raw = path.read_bytes()
    require(digest(raw) == expected_sha256, f"pinned source differs: {path}")
    return raw


def read_metadata(raw: bytes, *, expected_rows: int | None = None) -> list[dict[str, str]]:
    """Project the raw corpus to identity metadata; no target field is accessed."""
    metadata: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        if not line:
            continue
        parsed = json.loads(line)
        identity, group = parsed.get("id"), parsed.get("group_id")
        require(isinstance(identity, str) and identity, "invalid source identity")
        require(isinstance(group, str) and group, "invalid source group")
        require(identity not in seen, "duplicate source identity")
        seen.add(identity)
        metadata.append({"id": identity, "group_id": group})
    if expected_rows is not None:
        require(len(metadata) == expected_rows, "raw source row count differs")
    return metadata


def read_selection(raw: bytes, *, expected_rows: int | None = None) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in raw.splitlines() if line]
    require(all(set(row) == {"id", "group_id"} for row in rows),
            "selection contains non-identity fields")
    require(all(isinstance(row["id"], str) and row["id"]
                and isinstance(row["group_id"], str) and row["group_id"] for row in rows),
            "invalid selection identity")
    require(len({row["id"] for row in rows}) == len(rows), "duplicate selection identity")
    if expected_rows is not None:
        require(len(rows) == expected_rows, "selection row count differs")
    return rows


def select_ids(
    metadata: list[dict[str, str]],
    v3: dict,
    exposed: list[dict[str, str]],
    *,
    target: int = TARGET,
) -> dict:
    require(type(target) is int and target > 0, "invalid target")
    remaining_ids = v3.get("remaining_ids")
    remaining_groups = v3.get("remaining_group_ids")
    require(isinstance(remaining_ids, list) and isinstance(remaining_groups, list),
            "v3 reserve inventory missing")
    require(all(isinstance(value, str) and value for value in remaining_ids + remaining_groups),
            "invalid v3 reserve identity")
    require(len(set(remaining_ids)) == len(remaining_ids), "duplicate v3 reserve identity")
    require(len(set(remaining_groups)) == len(remaining_groups), "duplicate v3 reserve group")
    reserve_ids, reserve_groups = set(remaining_ids), set(remaining_groups)

    by_group: dict[str, list[str]] = defaultdict(list)
    source_ids: set[str] = set()
    for row in metadata:
        identity, group = row["id"], row["group_id"]
        require(identity not in source_ids, "duplicate source identity")
        source_ids.add(identity)
        require((identity in reserve_ids) == (group in reserve_groups),
                "v3 reserve contains a partial document group")
        if identity in reserve_ids:
            by_group[group].append(identity)
    require(set(by_group) == reserve_groups, "v3 reserve group inventory differs")
    require({identity for ids in by_group.values() for identity in ids} == reserve_ids,
            "v3 reserve ID inventory differs")

    exposed_ids = {row["id"] for row in exposed}
    exposed_groups = {row["group_id"] for row in exposed}
    require(len(exposed_ids) == len(exposed), "duplicate exposed identity")
    require(exposed_ids <= reserve_ids and exposed_groups <= reserve_groups,
            "exposed selection is outside v3 reserve")
    require(all((identity in exposed_ids) == (group in exposed_groups)
                for group, identities in by_group.items() for identity in identities),
            "exposed selection contains a partial document group")
    require(all(row["id"] in by_group[row["group_id"]] for row in exposed),
            "exposed identity/group mapping differs")

    available_groups = [group for group in remaining_groups if group not in exposed_groups]
    available_ids = [identity for group in available_groups for identity in by_group[group]]
    require(not set(available_ids) & exposed_ids,
            "previously unevaluated reserve overlaps exposed selection")
    selected_groups: list[str] = []
    selected_ids: list[str] = []
    for group in available_groups:
        if len(selected_ids) >= target:
            break
        selected_groups.append(group)
        selected_ids.extend(by_group[group])
    require(len(selected_ids) >= target, "insufficient previously unevaluated reserve")
    return {
        "ids": selected_ids,
        "group_ids": selected_groups,
        "available_ids": available_ids,
        "available_group_ids": available_groups,
        "by_group": dict(by_group),
    }


def validate_selection(selected: dict, exposed: list[dict[str, str]], *, target: int) -> None:
    ids, groups = selected["ids"], selected["group_ids"]
    exposed_ids = {row["id"] for row in exposed}
    exposed_groups = {row["group_id"] for row in exposed}
    require(len(ids) >= target and len(set(ids)) == len(ids), "invalid selected IDs")
    require(len(set(groups)) == len(groups), "invalid selected groups")
    require(not set(ids) & exposed_ids and not set(groups) & exposed_groups,
            "selected reserve overlaps exposed selection")
    require(ids == [identity for group in groups for identity in selected["by_group"][group]],
            "selected reserve is not whole-group ordered")


def run(output: Path = OUTPUT, *, refresh_manifest: bool = False) -> dict:
    if refresh_manifest:
        require(output.is_dir() and (output / "manifest.json").is_file()
                and (output / "selection.jsonl").is_file(),
                "completed output is required to refresh provenance")
    else:
        require(not output.exists(), f"output already exists: {output}")
    builder_raw = Path(__file__).read_bytes()
    snapshots = {path: read_pinned(path, expected) for path, expected in PINS.items()}
    v1, v2, v3 = (json.loads(snapshots[path]) for path in (V1, V2, V3))
    require(v3.get("source_hashes", {}).get(path_key(V1)) == PINS[V1], "v3 does not pin v1")
    require(v3.get("source_hashes", {}).get(path_key(V2)) == PINS[V2], "v3 does not pin v2")
    require(v3.get("source_hashes", {}).get(path_key(RAW)) == PINS[RAW], "v3 does not pin raw source")
    require(isinstance(v1.get("natural"), dict) and isinstance(v2, dict), "partition lineage invalid")
    require(len(v3.get("remaining_ids", [])) == 2077
            and len(v3.get("remaining_group_ids", [])) == 1607, "v3 remainder size differs")

    metadata = read_metadata(snapshots[RAW], expected_rows=9002)
    exposed = read_selection(snapshots[EXPOSED], expected_rows=512)
    require(len({row["group_id"] for row in exposed}) == 387, "exposed group count differs")
    selected = select_ids(metadata, v3, exposed)
    require(len(selected["available_ids"]) == 1565
            and len(selected["available_group_ids"]) == 1220,
            "previously unevaluated reserve size differs")
    require(len(selected["ids"]) == 512 and len(selected["group_ids"]) == 394,
            "boundary reserve size differs")
    validate_selection(selected, exposed, target=TARGET)

    for path, expected in PINS.items():
        require(digest(path.read_bytes()) == expected, f"source changed during selection: {path}")
    require(Path(__file__).read_bytes() == builder_raw, "builder changed during selection")

    rows = [{"id": identity, "group_id": group}
            for group in selected["group_ids"] for identity in selected["by_group"][group]]
    selection_raw = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    if refresh_manifest:
        require((output / "selection.jsonl").read_bytes() == selection_raw,
                "refusing to refresh manifest because selection bytes differ")
    manifest = {
        "schema": "boundary-reserve-selection-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "groups": len(selected["group_ids"]),
        "available_rows": len(selected["available_ids"]),
        "available_groups": len(selected["available_group_ids"]),
        "target_rows": TARGET,
        "selection_sha256": digest(selection_raw),
        "ordered_ids_sha256": digest(json.dumps(selected["ids"], separators=(",", ":")).encode()),
        "ordered_group_ids_sha256": digest(json.dumps(selected["group_ids"], separators=(",", ":")).encode()),
        "source_hashes": {path_key(path): value for path, value in PINS.items()},
        "builder_sha256": digest(builder_raw),
        "raw_target_bytes_read": True,
        "raw_json_rows_parsed": True,
        "raw_target_fields_parsed": True,
        "selection_fields_used": ["id", "group_id"],
        "target_fields_used_for_selection": False,
        "target_text_copied": False,
        "selection_uses_target_text": False,
        "whole_groups_preserved": True,
        "selection_order": "finite-final-choice-partition-v3.remaining_group_ids",
        "model_inference_run": False,
        "training_run": False,
        "source_rows_modified": False,
    }
    if not refresh_manifest:
        output.mkdir(parents=True, exist_ok=False)
        (output / "selection.jsonl").write_bytes(selection_raw)
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--refresh-manifest", action="store_true")
    args = parser.parse_args()
    result = run(args.output, refresh_manifest=args.refresh_manifest)
    print(json.dumps({key: result[key] for key in ("rows", "groups", "selection_sha256")}))
