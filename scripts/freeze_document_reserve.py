#!/usr/bin/env python3
"""Freeze the remaining natural TRAIN document reserve using identities only."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
FINAL = ROOT / ".omx/artifacts/final-choice-partition-v1/manifest.json"
BALANCED = ROOT / ".omx/artifacts/balanced-final-choice-partition-v2/manifest.json"
FINITE = ROOT / ".omx/artifacts/finite-final-choice-partition-v3/manifest.json"
LONG_CONTEXT = ROOT / ".omx/artifacts/long-context-lora-data-v1-review2/fresh_selection.jsonl"
BOUNDARY = ROOT / ".omx/artifacts/boundary-reserve-v1/selection.jsonl"
AUDITS = (
    ROOT / ".omx/artifacts/natural-context-runtime-audit-v1.jsonl",
    ROOT / ".omx/artifacts/natural-context-runtime-audit-numeric-v1.jsonl",
    ROOT / ".omx/artifacts/natural-context-runtime-audit-numeric-v2.jsonl",
)
AUDIT_REPORTS = (
    ROOT / ".omx/artifacts/natural-context-runtime-audit-v1.json",
    ROOT / ".omx/artifacts/natural-context-runtime-audit-numeric-v1.json",
    ROOT / ".omx/artifacts/natural-context-runtime-audit-numeric-v2.json",
)

EXPECTED_HASHES = {
    RAW: "904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b",
    FINAL: "e997f06558bdb5633446171d35912140fb34b55066c13fa13b668939d6347846",
    BALANCED: "57170914d4f14d5cd725ea0fd52a96b68b0859ba83041b93f1ecc09daa7cc8ca",
    FINITE: "8f699052b3e8d6e07dcaf59dc8696cc13a8ffb15b4e1bb5df62571e906ae3e61",
    LONG_CONTEXT: "fb1257833c99acc406a831336f535d9ef3a786fc38bb08a84902598e5fe2287d",
    BOUNDARY: "3db429739cea0834d7a6bb284944bf3107060d2cf4f0a2fd252ed48c094771d0",
    AUDITS[0]: "03c3e16ace28a619315b3d11a8153471f5ef60548b0242625ab22889cca33add",
    AUDITS[1]: "e7ee0f39d82920b5e5001ca6055b665627b1262ba584408aebc1d53f41d49f12",
    AUDITS[2]: "9811682bef335fd9aa79acad5475704820a384dbfe24561c0995927a13604ef3",
    AUDIT_REPORTS[0]: "c9c4cbc7532386e95a70eee283c0807fd016e063c044d138af2bb4f923a80f88",
    AUDIT_REPORTS[1]: "c0d98e137a71b8963e9821a5da07e8f161ac2ef480e3c35760454203b60e8d7d",
    AUDIT_REPORTS[2]: "0b6c024a117799e42b3fa192f373051abcc4426ffc6ce2b831f90adec8010d5e",
}


@dataclass(frozen=True)
class Counts:
    raw_rows: int = 9002
    final_reserve_rows: int = 4126
    balanced_eval_rows: int = 1025
    balanced_remaining_rows: int = 3101
    finite_eval_rows: int = 1024
    finite_eval_groups: int = 785
    finite_remaining_rows: int = 2077
    finite_remaining_groups: int = 1607
    long_rows: int = 512
    long_groups: int = 387
    boundary_rows: int = 512
    boundary_groups: int = 394
    nominal_rows: int = 1053
    nominal_groups: int = 826
    audit_overlap_ids: int = 5
    audit_overlap_groups: int = 5
    excluded_whole_group_rows: int = 7
    output_rows: int = 1046
    output_groups: int = 821


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def identities(rows: list[dict], source: str) -> tuple[list[str], list[str]]:
    ids, groups = [], []
    for row in rows:
        row_id, group_id = row.get("id"), row.get("group_id")
        require(isinstance(row_id, str) and row_id, f"invalid id in {source}")
        require(isinstance(group_id, str) and group_id,
                f"invalid group_id in {source}")
        ids.append(row_id)
        groups.append(group_id)
    require(len(ids) == len(set(ids)), f"duplicate id in {source}")
    return ids, groups


def list_of_strings(value: object, name: str) -> list[str]:
    require(isinstance(value, list) and all(isinstance(x, str) and x for x in value),
            f"invalid {name}")
    require(len(value) == len(set(value)), f"duplicate {name}")
    return value


def validate_whole_groups(selected_ids: set[str], universe_ids: set[str],
                          id_to_group: dict[str, str], name: str) -> set[str]:
    require(selected_ids <= universe_ids, f"{name} contains unknown ids")
    selected_groups = {id_to_group[row_id] for row_id in selected_ids}
    expected = {row_id for row_id in universe_ids
                if id_to_group[row_id] in selected_groups}
    require(selected_ids == expected, f"{name} contains a partial group")
    return selected_groups


def validate_row_mapping(ids: list[str], groups: list[str],
                         id_to_group: dict[str, str], name: str) -> None:
    for row_id, group_id in zip(ids, groups):
        require(row_id in id_to_group, f"{name} contains unknown id")
        require(id_to_group[row_id] == group_id,
                f"{name} id/group mapping changed")


def digest_lines(values: list[str]) -> str:
    return hashlib.sha256("".join(value + "\n" for value in values).encode()).hexdigest()


def derive_reserve(paths: dict[str, Path], expected_hashes: dict[Path, str],
                   counts: Counts = Counts()) -> tuple[list[dict], dict]:
    for path, expected in expected_hashes.items():
        require(path.is_file(), f"missing frozen source: {path}")
        require(sha256_file(path) == expected, f"frozen source changed: {path}")

    raw_rows = read_jsonl(paths["raw"])
    raw_ids, raw_groups = identities(raw_rows, "raw")
    require(len(raw_ids) == counts.raw_rows, "raw row count changed")
    id_to_group = dict(zip(raw_ids, raw_groups))
    raw_set = set(raw_ids)

    final = read_json(paths["final"])
    final_reserve = set(list_of_strings(final.get("natural", {}).get("reserve_ids"),
                                        "final reserve ids"))
    final_groups = set(list_of_strings(
        final.get("natural", {}).get("reserve_group_ids"), "final reserve groups"))
    require(len(final_reserve) == counts.final_reserve_rows,
            "final reserve row count changed")
    require(validate_whole_groups(final_reserve, raw_set, id_to_group, "final reserve")
            == final_groups, "final reserve group mapping changed")

    balanced = read_json(paths["balanced"])
    balanced_eval = set(list_of_strings(balanced.get("eval_ids"), "balanced eval ids"))
    balanced_remaining = set(list_of_strings(
        balanced.get("remaining_ids"), "balanced remaining ids"))
    require(len(balanced_eval) == counts.balanced_eval_rows
            and len(balanced_remaining) == counts.balanced_remaining_rows,
            "balanced counts changed")
    require(not balanced_eval & balanced_remaining
            and balanced_eval | balanced_remaining == final_reserve,
            "balanced partition equality failed")
    require(validate_whole_groups(balanced_eval, final_reserve, id_to_group,
                                  "balanced eval") == set(list_of_strings(
                                      balanced.get("eval_group_ids"),
                                      "balanced eval groups")),
            "balanced eval group mapping changed")
    require(validate_whole_groups(balanced_remaining, final_reserve, id_to_group,
                                  "balanced remaining") == set(list_of_strings(
                                      balanced.get("remaining_group_ids"),
                                      "balanced remaining groups")),
            "balanced remaining group mapping changed")

    finite = read_json(paths["finite"])
    finite_eval = set(list_of_strings(finite.get("eval_ids"), "finite eval ids"))
    finite_eval_groups = set(list_of_strings(
        finite.get("eval_group_ids"), "finite eval groups"))
    finite_remaining_list = list_of_strings(
        finite.get("remaining_ids"), "finite remaining ids")
    finite_remaining = set(finite_remaining_list)
    finite_groups = set(list_of_strings(
        finite.get("remaining_group_ids"), "finite remaining groups"))
    require(len(finite_eval) == counts.finite_eval_rows
            and len(finite_eval_groups) == counts.finite_eval_groups
            and len(finite_remaining) == counts.finite_remaining_rows
            and len(finite_groups) == counts.finite_remaining_groups,
            "finite counts changed")
    require(not finite_eval & finite_remaining
            and finite_eval | finite_remaining == balanced_remaining,
            "finite partition equality failed")
    require(validate_whole_groups(finite_eval, balanced_remaining, id_to_group,
                                  "finite eval") == finite_eval_groups,
            "finite eval group mapping changed")
    require(validate_whole_groups(finite_remaining, balanced_remaining, id_to_group,
                                  "finite remaining") == finite_groups,
            "finite remaining group mapping changed")

    long_rows = read_jsonl(paths["long"])
    long_ids_list, long_group_list = identities(long_rows, "long-context selection")
    validate_row_mapping(long_ids_list, long_group_list, id_to_group,
                         "long-context selection")
    long_ids, long_groups = set(long_ids_list), set(long_group_list)
    require(len(long_ids) == counts.long_rows and len(long_groups) == counts.long_groups,
            "long-context selection counts changed")
    require(validate_whole_groups(long_ids, finite_remaining, id_to_group,
                                  "long-context selection") == long_groups,
            "long-context group mapping changed")

    boundary_rows = read_jsonl(paths["boundary"])
    boundary_ids_list, boundary_group_list = identities(
        boundary_rows, "boundary selection")
    validate_row_mapping(boundary_ids_list, boundary_group_list, id_to_group,
                         "boundary selection")
    boundary_ids, boundary_groups = set(boundary_ids_list), set(boundary_group_list)
    require(len(boundary_ids) == counts.boundary_rows
            and len(boundary_groups) == counts.boundary_groups,
            "boundary selection counts changed")
    require(not long_ids & boundary_ids and not long_groups & boundary_groups,
            "long-context and boundary selections overlap")
    require(validate_whole_groups(boundary_ids, finite_remaining, id_to_group,
                                  "boundary selection") == boundary_groups,
            "boundary group mapping changed")

    nominal = finite_remaining - long_ids - boundary_ids
    nominal_groups = finite_groups - long_groups - boundary_groups
    require(len(nominal) == counts.nominal_rows
            and len(nominal_groups) == counts.nominal_groups,
            "nominal remainder counts changed")
    require(validate_whole_groups(nominal, finite_remaining, id_to_group,
                                  "nominal remainder") == nominal_groups,
            "nominal remainder group mapping changed")

    audit_overlap_sets = []
    audit_group_sets = []
    for audit_path in paths["audits"]:
        audit_ids_list, audit_groups_list = identities(
            read_jsonl(audit_path), f"audit {audit_path}")
        for row_id, group_id in zip(audit_ids_list, audit_groups_list):
            if row_id in raw_set:
                require(id_to_group[row_id] == group_id,
                        f"audit {audit_path} id/group mapping changed")
            else:
                require(group_id not in nominal_groups,
                        f"audit {audit_path} has unknown id in nominal group")
        overlap = set(audit_ids_list) & nominal
        audit_overlap_sets.append(overlap)
        audit_group_sets.append({id_to_group[row_id] for row_id in overlap})
    require(all(values == audit_overlap_sets[0] for values in audit_overlap_sets[1:]),
            "audit overlap ids differ")
    require(all(values == audit_group_sets[0] for values in audit_group_sets[1:]),
            "audit overlap groups differ")
    exposed_ids, exposed_groups = audit_overlap_sets[0], audit_group_sets[0]
    require(len(exposed_ids) == counts.audit_overlap_ids
            and len(exposed_groups) == counts.audit_overlap_groups,
            "audit overlap counts changed")

    output_ids = {row_id for row_id in nominal
                  if id_to_group[row_id] not in exposed_groups}
    output_groups = nominal_groups - exposed_groups
    require(len(nominal) - len(output_ids) == counts.excluded_whole_group_rows,
            "whole-group audit exclusion size changed")
    require(len(output_ids) == counts.output_rows
            and len(output_groups) == counts.output_groups,
            "output reserve counts changed")
    require(validate_whole_groups(output_ids, nominal, id_to_group,
                                  "output reserve") == output_groups,
            "output reserve group mapping changed")

    ordered_ids = [row_id for row_id in finite_remaining_list if row_id in output_ids]
    selection = [{"id": row_id, "group_id": id_to_group[row_id]}
                 for row_id in ordered_ids]
    evidence = {
        "nominal_rows": len(nominal), "nominal_groups": len(nominal_groups),
        "audit_overlap_ids": len(exposed_ids),
        "audit_overlap_groups": len(exposed_groups),
        "excluded_whole_group_rows": len(nominal) - len(output_ids),
        "rows": len(selection), "groups": len(output_groups),
        "ordered_ids_sha256": digest_lines(ordered_ids),
        "ordered_group_ids_sha256": digest_lines(
            list(dict.fromkeys(row["group_id"] for row in selection))),
    }
    return selection, evidence


def write_output(output: Path, selection: list[dict], evidence: dict,
                 expected_hashes: dict[Path, str], producer_sha: str) -> None:
    require(not output.exists(), f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output.name + ".tmp-", dir=output.parent))
    try:
        selection_path = temporary / "selection.jsonl"
        with selection_path.open("w", encoding="utf-8") as handle:
            for row in selection:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        manifest = {
            "schema": "document-reserve-v2",
            "completed": True,
            "role": "evaluation_only",
            "counts": evidence,
            "outputs": {
                "selection.jsonl": {
                    "rows": len(selection), "sha256": sha256_file(selection_path),
                    "fields": ["id", "group_id"],
                }
            },
            "pins": {str(path.relative_to(ROOT)): digest
                     for path, digest in expected_hashes.items()},
            "producer": {
                "path": "scripts/freeze_document_reserve.py",
                "sha256": producer_sha,
            },
            "contracts": {
                "metadata_only": True,
                "whole_document_groups": True,
                "selection_uses_target_text": False,
                "training_allowed": False,
                "model_loaded": False,
                "model_inference": False,
                "previously_exposed_audit_groups_excluded": True,
            },
            "caveats": [
                "Raw JSON and target-bearing audit rows were parsed to project identity and group fields; label bytes were previously parsed by upstream selection tooling.",
                "The reserve is disjoint from serialized repository experiment identities and whole document groups only.",
                "Upstream model pretraining overlap is unknown.",
                "Lexical, lemma, and inflection-family closure is unknown; this is not a fresh lexical-family holdout.",
                "Absence from ID-keyed serialized provenance does not prove absence from unrecorded training or external pretraining.",
            ],
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                            sort_keys=True) + "\n", encoding="utf-8")
        require(sha256_file(Path(__file__)) == producer_sha,
                "producer changed during execution")
        for path, expected in expected_hashes.items():
            require(sha256_file(path) == expected, f"source changed during execution: {path}")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run(output: Path, paths: dict[str, object] | None = None,
        expected_hashes: dict[Path, str] | None = None,
        counts: Counts = Counts()) -> None:
    paths = paths or {
        "raw": RAW, "final": FINAL, "balanced": BALANCED, "finite": FINITE,
        "long": LONG_CONTEXT, "boundary": BOUNDARY, "audits": AUDITS,
    }
    expected_hashes = expected_hashes or EXPECTED_HASHES
    producer_sha = sha256_file(Path(__file__))
    selection, evidence = derive_reserve(paths, expected_hashes, counts)
    write_output(output, selection, evidence, expected_hashes, producer_sha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / ".omx/artifacts/document-reserve-v2")
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()
