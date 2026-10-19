#!/usr/bin/env python3
"""Prepare a held-closed phonetic delta from exact TRAIN reference surfaces."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

from scripts import audit_phonetic_training_families as family
from scripts import prepare_attested_inflection_delta as inflection_delta


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILES = (
    ROOT / "data/challenge/sentence-v3/train.jsonl",
    ROOT / "data/challenge/sentence-train-extra-v3/train.jsonl",
)
BASE = ROOT / ".omx/artifacts/phonetic-supported-data-v1"
RESERVE = ROOT / ".omx/artifacts/document-reserve-v2"
RAW_NATURAL = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
OUTPUT = ROOT / ".omx/artifacts/observed-phonetic-delta-v1"

EXPECTED_HASHES = {
    TRAIN_FILES[0]: "e73ac3b3baf41075028c3b5629b4412336926faafdb8353b20c61a200bda4c42",
    TRAIN_FILES[1]: "e0729e728c0c5bf2827c83b666380c4f8c4757ee204cdb248578327667a0539b",
    BASE / "pairs.jsonl": "cf3e43f75420c17fbe8e1ec118a29a0f6a71c0e55cedbdbae5d7ef3668c84a84",
    BASE / "manifest.json": "9a7d7a7f2b0fab35ba31c8aae20b28d5af3a5bbfd0d56cb67afb4c0e45a16bba",
    family.MECAB / "Inflect.csv": "ff9a82d6f97aa75b86f03c33e2d615486200d7433f1c86c2e33a8c18d647064c",
    RESERVE / "selection.jsonl": "7d62027d11dcf1320b2a1bac7bcb83e1a31d31ce521808bbe4d1faadf6d4b810",
    RESERVE / "manifest.json": "19223e3c507432c168205732c26c4a95dc1e2391a0698dea96eaac63d70c3859",
    RAW_NATURAL: "904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b",
    ROOT / "scripts/audit_phonetic_training_families.py": "5c2faeac18e4580a4067fae8602eb60afe6139d568d8cc7f54f1ae753e7ab1b2",
    ROOT / "scripts/prepare_attested_inflection_delta.py": "bf75fe456dfcf3447f40f9b08c232fb7c3009d4a177d467fbca578ad0d663de0",
    ROOT / "hanglish/core/phonology.py": "e593f06409646b74d8aa8ca62663dcdd1b62759254ae1d1d23fc948fed54b865",
    ROOT / "hanglish/core/predicate_boundaries.py": "d0d29ca399220e6707e674ebf29a1368162e21741c1f0691d985d284ffc9abb1",
    ROOT / "hanglish/core/data/predicate_boundary_inventory.json": "5da10f07dd9b1a494ad7465c0e215c78982b6d7e252b1f603f7232cf714bb06a",
    ROOT / "hanglish/core/lexical_data.py": "0478e010f6626f14350b97c2db02440d86fbfac9e18b233598171a926f16b955",
    ROOT / "hanglish/core/lattice.py": "7a4c0f26eee4f3e694b46c3bc24423000270e493f8344ccca3123a7d9898ce9e",
}


@dataclass(frozen=True)
class ExpectedCounts:
    source_rows: int = 7704
    eligible_pairs: int = 46292
    old_pairs: int = 54676
    existing_observed_pairs: int = 13183
    proposed_pairs: int = 33109
    accepted_pairs: int = 32984
    quarantined_pairs: int = 125
    accepted_components: int = 32924
    quarantined_components: int = 19
    known_train_root_pairs: int = 84
    unknown_train_root_pairs: int = 32900


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"invalid row {number}: {path}")
            rows.append(value)
    return rows


def collect_observed(source_rows: dict[str, list[dict]], eligible=family.eligible_pair):
    provenance: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set))
    source_counts = {}
    all_references = set()
    for source, rows in sorted(source_rows.items()):
        seen_ids, run_count = set(), 0
        for row in rows:
            row_id = row.get("id")
            require(isinstance(row_id, str) and row_id, f"invalid row id: {source}")
            require(row_id not in seen_ids, f"duplicate provenance row id: {source}")
            seen_ids.add(row_id)
            require(row.get("split") == "train", f"non-TRAIN source row: {source}")
            expected = row.get("expected")
            require(isinstance(expected, list) and expected,
                    f"missing expected references: {source}")
            require(all(isinstance(text, str) and text for text in expected),
                    f"invalid expected reference: {source}")
            for text in expected:
                all_references.add(text)
                for surface in family.HANGUL.findall(text):
                    run_count += 1
                    pair = eligible(surface)
                    if pair is not None:
                        provenance[pair][source].add(row_id)
        source_counts[source] = {"rows": len(rows), "hangul_run_occurrences": run_count}
    return provenance, source_counts, all_references


def validate_reserve(source_rows: dict[str, list[dict]], all_references: set[str],
                     reserve_rows: list[dict], raw_rows: list[dict]) -> dict:
    raw_by_id = {}
    for row in raw_rows:
        row_id, group_id = row.get("id"), row.get("group_id")
        require(isinstance(row_id, str) and row_id, "invalid raw id")
        require(isinstance(group_id, str) and group_id, "invalid raw group_id")
        require(row_id not in raw_by_id, "duplicate raw id")
        raw_by_id[row_id] = row
    reserve_ids, reserve_groups = set(), set()
    for row in reserve_rows:
        row_id, group_id = row.get("id"), row.get("group_id")
        require(row_id in raw_by_id, "reserve contains unknown raw id")
        require(raw_by_id[row_id].get("group_id") == group_id,
                "reserve raw id/group mapping changed")
        require(row_id not in reserve_ids, "duplicate reserve id")
        reserve_ids.add(row_id)
        reserve_groups.add(group_id)
    raw_source_ids = {
        source_id for row_id in reserve_ids
        for source_id in raw_by_id[row_id].get("source_ids", [])
        if isinstance(source_id, str)
    }
    raw_provenance_ids = set()
    for row_id in reserve_ids:
        for item in raw_by_id[row_id].get("provenance", []):
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if key.endswith("id") and isinstance(value, str):
                    raw_provenance_ids.add(value)
                elif key.endswith("ids") and isinstance(value, list):
                    raw_provenance_ids.update(x for x in value if isinstance(x, str))
    reserve_texts = {raw_by_id[row_id].get("text") for row_id in reserve_ids}
    reserve_texts.discard(None)
    overlaps = Counter()
    for rows in source_rows.values():
        for row in rows:
            refs = set(row.get("source_ids", [])) | set(row.get("source_row_ids", []))
            overlaps["direct_ids"] += row["id"] in reserve_ids
            overlaps["groups"] += row.get("group_id") in reserve_groups
            overlaps["references_to_reserve_ids"] += bool(refs & reserve_ids)
            overlaps["references_to_raw_source_ids"] += bool(refs & raw_source_ids)
            overlaps["references_to_raw_provenance_ids"] += bool(refs & raw_provenance_ids)
    overlaps["exact_full_references"] = len(all_references & reserve_texts)
    require(not any(overlaps.values()), f"document reserve overlap: {dict(overlaps)}")
    return {
        "reserve_rows": len(reserve_ids), "reserve_groups": len(reserve_groups),
        "raw_source_ids": len(raw_source_ids),
        "raw_provenance_ids": len(raw_provenance_ids),
        "overlaps": dict(overlaps),
    }


def root_stratum(surface: str, analyses, train_roots: set[tuple[str, str]]) -> str:
    variants = analyses.get(surface, ())
    roots = inflection_delta.lexical_roots(variants)
    if roots & train_roots:
        return "connected_existing_train_root"
    if roots:
        return "recorded_root_not_connected_existing_train"
    return "no_recorded_inflect_root"


def prepare(source_rows: dict[str, list[dict]], base: list[dict], analyses,
            reserve_rows: list[dict], raw_rows: list[dict],
            eligible=family.eligible_pair,
            quarantine_fn=inflection_delta.quarantine):
    require(len({(row.get("roman"), row.get("surface")) for row in base}) == len(base),
            "duplicate base pair")
    provenance, source_counts, references = collect_observed(source_rows, eligible)
    reserve = validate_reserve(source_rows, references, reserve_rows, raw_rows)
    old_pairs = {(row["roman"], row["surface"]) for row in base}
    proposed_pairs = sorted(set(provenance) - old_pairs)
    proposals = []
    for roman, surface in proposed_pairs:
        observations = [
            {"source_file": source, "row_ids": sorted(row_ids)}
            for source, row_ids in sorted(provenance[(roman, surface)].items())
        ]
        proposals.append({
            "roman": roman, "surface": surface, "split": "train",
            "observations": observations,
            "surface_provenance": "exact_maximal_hangul_run_in_train_expected_reference",
            "input_provenance": "bounded_romanize_word_synthetic_not_independent_pronunciation",
        })
    accepted, rejected = quarantine_fn(base, proposals, analyses)
    train_roots = {root for row in base if row["split"] == "train"
                   for root in inflection_delta.inherited_roots(row)}
    accepted = [{**row, "recorded_root_stratum": root_stratum(
        row["surface"], analyses, train_roots)} for row in accepted]
    accepted.sort(key=lambda row: (row["roman"], row["surface"]))
    rejected.sort(key=lambda row: row["pair_sha256"])
    strata = Counter(row["recorded_root_stratum"] for row in accepted)
    by_roman = defaultdict(set)
    for row in accepted:
        by_roman[row["roman"]].add(row["surface"])
    collisions = {roman: surfaces for roman, surfaces in by_roman.items()
                  if len(surfaces) > 1}
    counts = {
        "source_rows": sum(len(rows) for rows in source_rows.values()),
        "eligible_pairs": len(provenance), "old_pairs": len(old_pairs),
        "existing_observed_pairs": len(set(provenance) & old_pairs),
        "proposed_pairs": len(proposals), "accepted_pairs": len(accepted),
        "quarantined_pairs": len(rejected),
        "accepted_components": len({row["component_sha256"] for row in accepted}),
        "quarantined_components": len({row["component_sha256"] for row in rejected}),
        "known_train_root_pairs": strata["connected_existing_train_root"],
        "unknown_train_root_pairs": len(accepted) - strata["connected_existing_train_root"],
        "recorded_unconnected_root_pairs": strata[
            "recorded_root_not_connected_existing_train"],
        "no_recorded_inflect_root_pairs": strata["no_recorded_inflect_root"],
        "same_roman_collision_keys": len(collisions),
        "same_roman_collision_pairs": sum(len(x) for x in collisions.values()),
    }
    require(counts["eligible_pairs"] == counts["existing_observed_pairs"]
            + counts["proposed_pairs"], "existing/proposed accounting differs")
    require(counts["proposed_pairs"] == counts["accepted_pairs"]
            + counts["quarantined_pairs"], "accepted/quarantine accounting differs")
    return accepted, rejected, counts, source_counts, reserve


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in rows:
            line = canonical(row) + b"\n"
            handle.write(line)
            digest.update(line)
        handle.flush()
        os.fsync(handle.fileno())
    return {"rows": len(rows), "sha256": digest.hexdigest()}


def run(output: Path = OUTPUT, expected_hashes=EXPECTED_HASHES,
        expected: ExpectedCounts = ExpectedCounts()) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    for path, wanted in expected_hashes.items():
        require(path.is_file(), f"missing frozen source: {path}")
        require(sha256_file(path) == wanted, f"frozen source changed: {path}")
    source_rows = {str(path.relative_to(ROOT)): read_jsonl(path) for path in TRAIN_FILES}
    base = read_jsonl(BASE / "pairs.jsonl")
    base_manifest = read_json(BASE / "manifest.json")
    require(base_manifest.get("pairs_jsonl_sha256") == expected_hashes[BASE / "pairs.jsonl"],
            "base manifest does not authenticate pairs")
    accepted, rejected, counts, source_counts, reserve = prepare(
        source_rows, base, family.read_inflect(),
        read_jsonl(RESERVE / "selection.jsonl"), read_jsonl(RAW_NATURAL))
    for field in ExpectedCounts.__dataclass_fields__:
        require(counts[field] == getattr(expected, field), f"count changed: {field}")

    self_path = Path(__file__).resolve()
    source_hashes = {str(path.relative_to(ROOT)): wanted
                     for path, wanted in expected_hashes.items()}
    source_hashes[str(self_path.relative_to(ROOT))] = sha256_file(self_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        outputs = {
            "delta.jsonl": write_jsonl(temp / "delta.jsonl", accepted),
            "quarantine.jsonl": write_jsonl(temp / "quarantine.jsonl", rejected),
        }
        manifest = {
            "schema": "observed-phonetic-delta-v1", "completed": True,
            "counts": counts, "source_counts": source_counts,
            "reserve_separation": reserve, "source_hashes": source_hashes,
            "outputs": outputs,
            "contracts": {
                "source_split": "train", "source_field": "expected",
                "surface_extraction": "maximal [가-힣]+ runs",
                "existing_base_pairs_excluded": True,
                "held_closure": "all recorded roots plus roman/surface/family transitive closure",
                "old_artifacts_overwritten": False, "model_activity": False,
                "training_performed": False, "production_changed": False,
                "document_reserve_eval_only": True,
            },
            "held_pair_access": (
                "Previously exposed original held pair fields are used only for preservation "
                "exclusion closure; they are not fresh evaluation evidence."),
            "limitations": [
                "Observed corpus surfaces are attestations, not normative pronunciation or grammaticality proof.",
                "Roman inputs are synthetic outputs of the current romanizer, not independent pronunciation validation.",
                "32900 accepted pairs are not connected to recorded existing TRAIN roots; no new lexical-family independence is claimed.",
                "The 1046-row document reserve remains evaluation-only and was used solely for identity and exact-reference exclusion.",
                "This dataset is only a candidate for a predeclared controlled training experiment.",
            ],
        }
        (temp / "manifest.json").write_bytes(canonical(manifest) + b"\n")
        for relative, wanted in source_hashes.items():
            require(sha256_file(ROOT / relative) == wanted,
                    f"source changed during preparation: {relative}")
        os.replace(temp, output)
        return manifest
    except Exception:
        for child in temp.iterdir():
            child.unlink()
        temp.rmdir()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    result = run(parser.parse_args().output)
    print(json.dumps(result["counts"], ensure_ascii=False, sort_keys=True))
