#!/usr/bin/env python3
"""Independently verify the TRAIN-only observed phonetic delta artifact."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / ".omx/artifacts/observed-phonetic-delta-v1"
PROOF = OUTPUT / "independent-verification.json"
TRAIN_FILES = (
    ROOT / "data/challenge/sentence-v3/train.jsonl",
    ROOT / "data/challenge/sentence-train-extra-v3/train.jsonl",
)
BASE_PAIRS = ROOT / ".omx/artifacts/phonetic-supported-data-v1/pairs.jsonl"
BASE_MANIFEST = ROOT / ".omx/artifacts/phonetic-supported-data-v1/manifest.json"
INFLECT = ROOT / "data/external/mecab-ko-dic/Inflect.csv"
RESERVE = ROOT / ".omx/artifacts/document-reserve-v2/selection.jsonl"
RESERVE_MANIFEST = ROOT / ".omx/artifacts/document-reserve-v2/manifest.json"
RAW_NATURAL = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
PHONOLOGY = ROOT / "hanglish/core/phonology.py"
PRODUCER = ROOT / "scripts/prepare_observed_phonetic_delta.py"

EXPECTED_HASHES = {
    TRAIN_FILES[0]: "e73ac3b3baf41075028c3b5629b4412336926faafdb8353b20c61a200bda4c42",
    TRAIN_FILES[1]: "e0729e728c0c5bf2827c83b666380c4f8c4757ee204cdb248578327667a0539b",
    BASE_PAIRS: "cf3e43f75420c17fbe8e1ec118a29a0f6a71c0e55cedbdbae5d7ef3668c84a84",
    BASE_MANIFEST: "9a7d7a7f2b0fab35ba31c8aae20b28d5af3a5bbfd0d56cb67afb4c0e45a16bba",
    INFLECT: "ff9a82d6f97aa75b86f03c33e2d615486200d7433f1c86c2e33a8c18d647064c",
    RESERVE: "7d62027d11dcf1320b2a1bac7bcb83e1a31d31ce521808bbe4d1faadf6d4b810",
    RESERVE_MANIFEST: "19223e3c507432c168205732c26c4a95dc1e2391a0698dea96eaac63d70c3859",
    RAW_NATURAL: "904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b",
    PHONOLOGY: "e593f06409646b74d8aa8ca62663dcdd1b62759254ae1d1d23fc948fed54b865",
    PRODUCER: "a69f0874f30ccf53d8e3d9dbc9bd3661c2a77050151e3c443cb38cca69d70631",
    OUTPUT / "manifest.json": "f90e21a92bb1b3ebed4ff59aeaab69d5ed54f24c59aafbc14cdd6e67df5991ef",
    OUTPUT / "delta.jsonl": "bc299116e618ef37c044d8ac0f3061a55e46efdc1b4f1d0c8d9dc1b8ab655e1a",
    OUTPUT / "quarantine.jsonl": "aebea22744a9e1567fa55648f8c7baa1ac00034eaf6bd72a7602cefa46fdea07",
}
LEXICAL_POS = frozenset({"VV", "VA", "VX", "VCP", "VCN", "NNG", "NNP", "NP", "NR", "XR"})
HANGUL = re.compile(r"[가-힣]+")
HANGUL_FULL = re.compile(r"[가-힣]+\Z")
HOMONYM_DIGITS = re.compile(r"(?<=[가-힣])[0-9]+\Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def rows(path: Path) -> list[dict]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                require(isinstance(value, dict), f"non-object row {number}: {path}")
                result.append(value)
    return result


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.weight = [1] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.weight[left] < self.weight[right]:
            left, right = right, left
        self.parent[right] = left
        self.weight[left] += self.weight[right]


def read_analyses() -> dict[str, set[tuple[tuple[str, str], ...]]]:
    result: dict[str, set[tuple[tuple[str, str], ...]]] = defaultdict(set)
    with INFLECT.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 12 or not HANGUL_FULL.fullmatch(row[0]):
                continue
            analysis = []
            for raw in row[11].split("+"):
                fields = raw.split("/")
                if len(fields) < 2 or not fields[0] or not fields[1]:
                    analysis = []
                    break
                analysis.append((HOMONYM_DIGITS.sub("", fields[0]), fields[1]))
            if analysis and any(pos in LEXICAL_POS for _, pos in analysis):
                result[row[0]].add(tuple(analysis))
    return result


def lexical_roots(surface: str, analyses: dict) -> set[tuple[str, str]]:
    return {part for analysis in analyses.get(surface, ()) for part in analysis
            if part[1] in LEXICAL_POS}


def verify_reserve(source_rows: list[list[dict]], references: set[str]) -> dict:
    raw = rows(RAW_NATURAL)
    raw_by_id = {row.get("id"): row for row in raw}
    require(len(raw_by_id) == len(raw) and None not in raw_by_id, "raw ids invalid")
    reserve_rows = rows(RESERVE)
    reserve_ids: set[str] = set()
    reserve_groups: set[str] = set()
    for row in reserve_rows:
        row_id, group_id = row.get("id"), row.get("group_id")
        require(row_id in raw_by_id and raw_by_id[row_id].get("group_id") == group_id,
                "reserve identity differs")
        require(row_id not in reserve_ids, "duplicate reserve id")
        reserve_ids.add(row_id)
        reserve_groups.add(group_id)
    source_ids = {item for row_id in reserve_ids
                  for item in raw_by_id[row_id].get("source_ids", [])
                  if isinstance(item, str)}
    provenance_ids: set[str] = set()
    for row_id in reserve_ids:
        for item in raw_by_id[row_id].get("provenance", []):
            if not isinstance(item, dict):
                continue
            for key, value in item.items():
                if key.endswith("id") and isinstance(value, str):
                    provenance_ids.add(value)
                elif key.endswith("ids") and isinstance(value, list):
                    provenance_ids.update(x for x in value if isinstance(x, str))
    overlaps = Counter()
    for group in source_rows:
        for row in group:
            linked = set(row.get("source_ids", [])) | set(row.get("source_row_ids", []))
            overlaps["direct_ids"] += row["id"] in reserve_ids
            overlaps["groups"] += row.get("group_id") in reserve_groups
            overlaps["references_to_reserve_ids"] += bool(linked & reserve_ids)
            overlaps["references_to_raw_source_ids"] += bool(linked & source_ids)
            overlaps["references_to_raw_provenance_ids"] += bool(linked & provenance_ids)
    reserve_texts = {raw_by_id[row_id].get("text") for row_id in reserve_ids}
    reserve_texts.discard(None)
    overlaps["exact_full_references"] = len(references & reserve_texts)
    require(len(reserve_ids) == 1046 and not any(overlaps.values()), "reserve overlaps TRAIN")
    return {"reserve_rows": len(reserve_ids), "reserve_groups": len(reserve_groups),
            "raw_source_ids": len(source_ids), "raw_provenance_ids": len(provenance_ids),
            "overlaps": dict(overlaps)}


def verify() -> dict:
    for path, wanted in EXPECTED_HASHES.items():
        require(path.is_file() and sha256(path) == wanted, f"hash mismatch: {path}")

    # Import only the hash-pinned pure romanizer. No producer or quarantine implementation is imported.
    from hanglish.core.phonology import romanize_word

    manifest = json.loads((OUTPUT / "manifest.json").read_text())
    base = rows(BASE_PAIRS)
    accepted_actual = rows(OUTPUT / "delta.jsonl")
    rejected_actual = rows(OUTPUT / "quarantine.jsonl")
    require(len(base) == 54676, "base row count differs")
    require(len({(r["roman"], r["surface"]) for r in base}) == len(base), "base duplicates")

    provenance: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    references: set[str] = set()
    source_groups: list[list[dict]] = []
    source_counts = {}
    for path in TRAIN_FILES:
        group = rows(path)
        source_groups.append(group)
        relative = str(path.relative_to(ROOT))
        seen, occurrences = set(), 0
        for row in group:
            row_id = row.get("id")
            require(isinstance(row_id, str) and row_id and row_id not in seen, "invalid TRAIN id")
            seen.add(row_id)
            require(row.get("split") == "train", "non-TRAIN source row")
            expected = row.get("expected")
            require(isinstance(expected, list) and expected and
                    all(isinstance(x, str) and x for x in expected), "invalid expected references")
            for text in expected:
                references.add(text)
                for surface in HANGUL.findall(text):
                    occurrences += 1
                    if len(surface) > 24:
                        continue
                    roman = romanize_word(surface)
                    if (roman.isascii() and roman.isalpha() and roman.islower()
                            and 2 <= len(roman) <= 40):
                        provenance[(roman, surface)][relative].add(row_id)
        source_counts[relative] = {"rows": len(group), "hangul_run_occurrences": occurrences}

    old_pairs = {(r["roman"], r["surface"]) for r in base}
    proposed_pairs = sorted(set(provenance) - old_pairs)
    proposal_rows = []
    for pair in proposed_pairs:
        proposal_rows.append({
            "roman": pair[0], "surface": pair[1], "split": "train",
            "observations": [{"source_file": source, "row_ids": sorted(ids)}
                             for source, ids in sorted(provenance[pair].items())],
            "surface_provenance": "exact_maximal_hangul_run_in_train_expected_reference",
            "input_provenance": "bounded_romanize_word_synthetic_not_independent_pronunciation",
        })
    require(len(source_groups[0]) == 5656 and len(source_groups[1]) == 2048, "TRAIN sizes differ")
    require(len(provenance) == 46292 and len(set(provenance) & old_pairs) == 13183,
            "observed candidate accounting differs")
    require(len(proposal_rows) == 33109, "raw proposal count differs")

    analyses = read_analyses()
    combined = [*base, *proposal_rows]
    dsu = DisjointSet(len(combined))
    owners: dict[tuple, int] = {}
    for index, row in enumerate(combined):
        roots = lexical_roots(row["surface"], analyses)
        if index < len(base):
            roots |= {(item["surface"], item["pos"]) for item in row["family_roots"]}
        keys = [("roman", row["roman"]), ("surface", row["surface"])]
        keys.extend(("root", root, pos) for root, pos in roots)
        if index < len(base):
            keys.append(("family", row["family_sha256"]))
        for key in keys:
            if key in owners:
                dsu.union(index, owners[key])
            else:
                owners[key] = index
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(combined)):
        components[dsu.find(index)].append(index)
    held: dict[int, set[str]] = defaultdict(set)
    for index, row in enumerate(base):
        if row["split"] != "train":
            held[dsu.find(index)].add(row["split"])
    component_sha = {root: hashlib.sha256(canonical(sorted(
        (combined[index]["roman"], combined[index]["surface"]) for index in members))).hexdigest()
        for root, members in components.items()}
    train_roots = {(item["surface"], item["pos"]) for row in base if row["split"] == "train"
                   for item in row["family_roots"]}
    accepted_expected, rejected_expected = [], []
    for offset, proposal in enumerate(proposal_rows, len(base)):
        root = dsu.find(offset)
        pair = (proposal["roman"], proposal["surface"])
        if held[root]:
            rejected_expected.append({
                "pair_sha256": hashlib.sha256(canonical(pair)).hexdigest(),
                "component_sha256": component_sha[root], "held_splits": sorted(held[root]),
            })
        else:
            roots = lexical_roots(proposal["surface"], analyses)
            if roots & train_roots:
                stratum = "connected_existing_train_root"
            elif roots:
                stratum = "recorded_root_not_connected_existing_train"
            else:
                stratum = "no_recorded_inflect_root"
            accepted_expected.append({**proposal, "component_sha256": component_sha[root],
                                      "recorded_root_stratum": stratum})
    accepted_expected.sort(key=lambda r: (r["roman"], r["surface"]))
    rejected_expected.sort(key=lambda r: r["pair_sha256"])
    require(accepted_actual == accepted_expected, "delta content/provenance differs")
    require(rejected_actual == rejected_expected, "quarantine closure differs")

    strata = Counter(row["recorded_root_stratum"] for row in accepted_actual)
    roman_surfaces: dict[str, set[str]] = defaultdict(set)
    for row in accepted_actual:
        roman_surfaces[row["roman"]].add(row["surface"])
    collisions = {key: value for key, value in roman_surfaces.items() if len(value) > 1}
    reserve = verify_reserve(source_groups, references)
    counts = {
        "source_rows": sum(len(x) for x in source_groups), "eligible_pairs": len(provenance),
        "old_pairs": len(old_pairs), "existing_observed_pairs": len(set(provenance) & old_pairs),
        "proposed_pairs": len(proposal_rows), "accepted_pairs": len(accepted_actual),
        "quarantined_pairs": len(rejected_actual),
        "accepted_components": len({r["component_sha256"] for r in accepted_actual}),
        "quarantined_components": len({r["component_sha256"] for r in rejected_actual}),
        "known_train_root_pairs": strata["connected_existing_train_root"],
        "unknown_train_root_pairs": len(accepted_actual) - strata["connected_existing_train_root"],
        "recorded_unconnected_root_pairs": strata["recorded_root_not_connected_existing_train"],
        "no_recorded_inflect_root_pairs": strata["no_recorded_inflect_root"],
        "same_roman_collision_keys": len(collisions),
        "same_roman_collision_pairs": sum(len(x) for x in collisions.values()),
    }
    require(counts == manifest["counts"], "manifest counts differ")
    require(source_counts == manifest["source_counts"], "source counts differ")
    require(reserve == manifest["reserve_separation"], "reserve evidence differs")
    require(manifest["outputs"]["delta.jsonl"] == {"rows": len(accepted_actual),
            "sha256": sha256(OUTPUT / "delta.jsonl")}, "delta output pin differs")
    require(manifest["outputs"]["quarantine.jsonl"] == {"rows": len(rejected_actual),
            "sha256": sha256(OUTPUT / "quarantine.jsonl")}, "quarantine output pin differs")
    for relative, wanted in manifest["source_hashes"].items():
        require(sha256(ROOT / relative) == wanted, f"manifest source pin differs: {relative}")

    verifier_hash = sha256(Path(__file__))
    proof = {
        "schema": "observed-phonetic-delta-independent-verification-v1",
        "verified": True,
        "manifest_sha256": sha256(OUTPUT / "manifest.json"),
        "delta_sha256": sha256(OUTPUT / "delta.jsonl"),
        "quarantine_sha256": sha256(OUTPUT / "quarantine.jsonl"),
        "producer_sha256": sha256(PRODUCER), "verifier_sha256": verifier_hash,
        "counts": counts, "source_counts": source_counts, "reserve_separation": reserve,
        "contracts_verified": {
            "train_expected_only": True, "maximal_hangul_runs": True,
            "old_pairs_subtracted": True, "full_transitive_held_closure": True,
            "delta_rows_and_provenance_exact": True, "quarantine_rows_exact": True,
            "source_and_output_hashes_exact": True, "document_reserve_disjoint": True,
            "neural_inference_or_training_performed": False,
        },
        "limitations": [
            "This verifies deterministic construction and held-closure separation, not pronunciation quality.",
            "Roman strings are outputs of the hash-pinned current romanizer, not independent phonetic labels.",
            "Most accepted pairs have no recorded connection to an existing TRAIN lexical root.",
            "TRAIN attestation and reserve disjointness do not establish performance on unseen text.",
        ],
    }
    PROOF.write_bytes(canonical(proof) + b"\n")
    return proof


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, sort_keys=True))
