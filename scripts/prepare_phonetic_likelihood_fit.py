"""Prepare the fixed FIT candidate domain for phonetic-likelihood scoring.

This is a pure data projection.  It authenticates the previously closed
full-FIT audit, preserves every candidate/path/base score/positive index, and
emits only source/target pairs that satisfy the phonetic model's explicit
input shape.  It never opens the mixed prepared labels or any hold/DEV/TEST
artifact.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / ".omx/artifacts/shared-edge-full-fit-audit-v1"
CORRESPONDENCE = ROOT / ".omx/artifacts/source-correspondence-fit-v2"
SCHEMA = "phonetic-likelihood-fit-v1"

FROZEN = {
    AUDIT / "selection.json": "37e12389f7a97dc831b57300a1662d3d01a9ab11ea3c4b01f37388bc3e094498",
    AUDIT / "report.json": "178bb0edc97d1441fd84c7fe3a51139678574b7b0c2de3a7d9da8dd93059b0af",
    AUDIT / "independent-verification.json": "1acdfae9db98c9d632b0a2e707e7cbb128ea8b0c3255a12d9f643cbcbae26b53",
    CORRESPONDENCE / "fit-paths.jsonl": "f1cd66da23cacc6025e8aa2203944f664f0d74d916a0bde135cac65ef56afd8c",
    CORRESPONDENCE / "independent-verification.json": "449a8f286a60cc4a7c542612054c4ca762378caedfb0cbc9ca350801030f2de2",
}

LOWER_ASCII = re.compile(r"[a-z]+\Z")
HANGUL = re.compile(r"[가-힣]+\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def jsonl(path: Path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value) -> None:
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True,
                  indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def pair_id(source: str, target: str) -> str:
    payload = json.dumps([SCHEMA, source, target], ensure_ascii=False,
                         separators=(",", ":")).encode()
    return "phonetic-pair-" + hashlib.sha256(payload).hexdigest()


def classify_pair(source: str, target: str) -> str:
    if not source or not target:
        return "empty"
    if source == target:
        return "copy"
    if LOWER_ASCII.fullmatch(source) is None:
        return "unsupported_ascii"
    if HANGUL.fullmatch(target) is None:
        return "unsupported_shape"
    if len(source) > 48 or len(target) > 24:
        return "length"
    return "supported"


def validate_selection(selection) -> dict[str, dict]:
    require(isinstance(selection, list) and len(selection) == 1483,
            "full-FIT selection denominator differs")
    result = {}
    for row in selection:
        require(isinstance(row, dict) and isinstance(row.get("id"), str)
                and row["id"] not in result, "invalid or duplicate selection ID")
        candidates = row.get("candidates")
        scores = row.get("base_scores")
        positive = row.get("positive_indices")
        selected = row.get("selected_indices")
        require(isinstance(candidates, list) and candidates
                and len(candidates) == len(scores) == len(row.get("candidate_path_counts", []))
                and len(set(candidates)) == len(candidates),
                "selection candidate domain differs")
        require(all(isinstance(text, str) and text for text in candidates)
                and all(type(score) in (int, float) and math.isfinite(score)
                        for score in scores), "selection values are invalid")
        require(isinstance(positive, list) and positive
                and all(type(index) is int and 0 <= index < len(candidates)
                        for index in positive)
                and isinstance(selected, list)
                and set(positive).issubset(selected),
                "selection positives are not preserved")
        result[row["id"]] = row
    require(sum(len(row["candidates"]) for row in selection) == 202603,
            "full-FIT candidate denominator differs")
    return result


def project_row(chosen: dict, record: dict, requests: dict[str, dict],
                romanize_word) -> tuple[dict, Counter]:
    require(record.get("schema") == "source-correspondence-v1"
            and record.get("id") == chosen["id"]
            and record.get("input") == chosen["input"]
            and record.get("component_id") == chosen["component_id"],
            "source-correspondence identity differs")
    source = chosen["input"]
    edges = record.get("edges")
    paths = record.get("paths")
    require(isinstance(edges, list) and isinstance(paths, list),
            "source-correspondence payload is malformed")

    by_text: dict[str, list[dict]] = {text: [] for text in chosen["candidates"]}
    used_edge_indices = set()
    seen_path_ids = set()
    for path in paths:
        text = path.get("text")
        if text not in by_text:
            continue
        indices = path.get("edge_indices")
        require(isinstance(path.get("path_id"), str)
                and path["path_id"] not in seen_path_ids
                and isinstance(indices, list) and indices,
                "candidate path is invalid or duplicated")
        cursor = 0
        pieces = []
        for index in indices:
            require(type(index) is int and 0 <= index < len(edges),
                    "candidate path edge index is invalid")
            edge = edges[index]
            start, end, target = edge.get("input_start"), edge.get("input_end"), edge.get("text")
            require(type(start) is int and type(end) is int and start == cursor
                    and start < end <= len(source) and isinstance(target, str),
                    "candidate path does not cover the source contiguously")
            cursor = end
            pieces.append(target)
            used_edge_indices.add(index)
        require(cursor == len(source) and "".join(pieces) == text,
                "candidate path output differs")
        seen_path_ids.add(path["path_id"])
        by_text[text].append({"path_id": path["path_id"],
                              "original_edge_indices": list(indices)})

    require(all(by_text.values()), "full-FIT candidate path is missing")
    require([len(by_text[text]) for text in chosen["candidates"]]
            == chosen["candidate_path_counts"], "candidate path counts differ")

    old_to_new = {old: new for new, old in enumerate(sorted(used_edge_indices))}
    projected_edges = []
    counts = Counter()
    for old in sorted(used_edge_indices):
        edge = edges[old]
        start, end, target = edge["input_start"], edge["input_end"], edge["text"]
        source_slice = source[start:end]
        status = classify_pair(source_slice, target)
        value = {"input_start": start, "input_end": end,
                 "source": source_slice, "target": target, "status": status}
        counts["edges"] += 1
        counts[status + "_edges"] += 1
        if status == "supported":
            identifier = pair_id(source_slice, target)
            value["pair_id"] = identifier
            forward = romanize_word(target) == source_slice
            value["romanize_match"] = forward
            counts["supported_romanize_match_edges" if forward
                   else "supported_romanize_mismatch_edges"] += 1
            expected = {"pair_id": identifier, "source": source_slice,
                        "target": target, "romanize_match": forward}
            prior = requests.setdefault(identifier, expected)
            require(prior == expected, "phonetic pair ID collision")
        projected_edges.append(value)

    candidate_paths = []
    for text in chosen["candidates"]:
        alternatives = []
        for path in sorted(by_text[text], key=lambda item: item["path_id"]):
            alternatives.append({"path_id": path["path_id"],
                                 "edge_indices": [old_to_new[i]
                                                  for i in path["original_edge_indices"]]})
        candidate_paths.append(alternatives)
        counts["paths"] += len(alternatives)
    counts["candidates"] += len(chosen["candidates"])
    return {
        "schema": SCHEMA + "-row",
        "id": chosen["id"],
        "input": source,
        "component_id": chosen["component_id"],
        "candidates": list(chosen["candidates"]),
        "base_scores": list(chosen["base_scores"]),
        "positive_indices": list(chosen["positive_indices"]),
        "selected_indices": list(chosen["selected_indices"]),
        "edges": projected_edges,
        "candidate_paths": candidate_paths,
    }, counts


def authenticate() -> dict[str, str]:
    require(all(path.is_file() and sha256(path) == digest
                for path, digest in FROZEN.items()), "frozen FIT input changed")
    report = read_json(AUDIT / "report.json")
    proof = read_json(AUDIT / "independent-verification.json")
    require(report.get("schema") == "shared-edge-full-fit-audit-v1"
            and report.get("completed") is True and report.get("rows") == 1483
            and report.get("candidates") == 202603
            and report.get("outputs", {}).get("selection.json")
            == FROZEN[AUDIT / "selection.json"], "full-FIT report differs")
    require(proof.get("schema") == "shared-edge-full-fit-independent-verification-v1"
            and proof.get("verified") is True
            and proof.get("report_sha256") == FROZEN[AUDIT / "report.json"]
            and proof.get("counts", {}).get("candidates") == 202603
            and proof.get("contracts", {}).get("full_nonraw_domain_and_all_selected_positives_retained") is True
            and proof.get("contracts", {}).get("inner_hold_labels_opened") is False
            and proof.get("contracts", {}).get("dev_opened") is False
            and proof.get("contracts", {}).get("test_opened") is False,
            "full-FIT independent proof differs")
    correspondence = read_json(CORRESPONDENCE / "independent-verification.json")
    require(correspondence.get("verified") is True
            and correspondence.get("capture_sha256") == FROZEN[CORRESPONDENCE / "fit-paths.jsonl"]
            and correspondence.get("counts", {}).get("rows") == 1642
            and correspondence.get("contracts", {}).get("labels_opened") is False
            and correspondence.get("contracts", {}).get("hold_candidate_payloads_parsed") is False
            and correspondence.get("contracts", {}).get("dev_opened") is False
            and correspondence.get("contracts", {}).get("test_opened") is False,
            "source-correspondence proof differs")
    return {str(path): digest for path, digest in FROZEN.items()}


def run(output: Path) -> dict:
    from hanglish.core.phonology import romanize_word

    pins = authenticate()
    selection = read_json(AUDIT / "selection.json")
    selected = validate_selection(selection)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    requests: dict[str, dict] = {}
    totals = Counter()
    found = set()
    rows_path = output / "rows.jsonl"
    with rows_path.open("x") as stream:
        for record in jsonl(CORRESPONDENCE / "fit-paths.jsonl"):
            chosen = selected.get(record.get("id"))
            if chosen is None:
                continue
            require(record["id"] not in found, "duplicate source-correspondence ID")
            projected, counts = project_row(chosen, record, requests, romanize_word)
            stream.write(json.dumps(projected, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False) + "\n")
            found.add(record["id"])
            totals.update(counts)
        stream.flush()
        os.fsync(stream.fileno())
    require(found == set(selected), "full-FIT source rows are missing")
    totals["rows"] = len(found)
    totals["unique_supported_pairs"] = len(requests)
    for status in ("copy", "supported", "unsupported_ascii",
                   "unsupported_shape", "length", "empty"):
        totals.setdefault(status + "_edges", 0)
    totals.setdefault("supported_romanize_match_edges", 0)
    totals.setdefault("supported_romanize_mismatch_edges", 0)
    require(totals["candidates"] == 202603 and totals["paths"] == 202607,
            "full-FIT candidate/path denominator differs")

    requests_path = output / "requests.jsonl"
    with requests_path.open("x") as stream:
        for identifier in sorted(requests):
            stream.write(json.dumps(requests[identifier], ensure_ascii=False,
                                    sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    pins[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
    require(all(sha256(Path(path)) == digest for path, digest in pins.items()),
            "pinned input changed during preparation")
    manifest = {
        "schema": SCHEMA,
        "scope": "fixed TRAIN FIT candidate domain only",
        "counts": dict(sorted(totals.items())),
        "outputs": {"rows.jsonl": sha256(rows_path),
                    "requests.jsonl": sha256(requests_path)},
        "pins": dict(sorted(pins.items())),
        "contracts": {
            "model_loaded": False,
            "neural_inference_performed": False,
            "inner_hold_labels_opened": False,
            "dev_opened": False,
            "test_opened": False,
            "reserve_opened": False,
            "all_full_fit_candidates_preserved": True,
            "all_selected_positives_preserved": True,
            "unsupported_edges_excluded_from_requests_not_from_paths": True,
        },
        "pair_domain": {
            "source": "lowercase ASCII letters, 1..48 codepoints",
            "target": "Hangul syllables, 1..24 codepoints",
            "truncation": False,
            "romanize_match": "diagnostic only; not an eligibility condition",
        },
        "limitations": [
            "Candidate paths are alignment hypotheses, not gold edge segmentation.",
            "Sentence-level positive membership is not per-edge phonetic supervision.",
            "The fixed 1,483-row FIT selection excludes 159 FIT rows without the earlier positive/negative training eligibility.",
            "This artifact contains no fresh holdout or generalization evidence.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    value = run(args.output)
    print(json.dumps(value["counts"], sort_keys=True))
