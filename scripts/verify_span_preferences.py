"""Independent verifier for FIT-only conditional span preferences.

This verifier intentionally uses only the Python standard library.  It does
not import the producer, span supervision, source correspondence, model code,
torch, or any evaluation labels.  Compact source paths and preference labels
are reconstructed independently from the frozen FIT artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / ".omx/artifacts/source-correspondence-fit-v2"
SELECTION = ROOT / ".omx/artifacts/recombination-paired-training-v1/selection.json"
OUTPUT = ROOT / ".omx/artifacts/conditional-span-preferences-fit-v1"
PREFERENCES = OUTPUT / "preferences.jsonl"
REPORT = OUTPUT / "report.json"
PROOF = OUTPUT / "independent-verification.json"
SCHEMA = "conditional-span-preference-v1"
PATH_SCHEMA = "source-correspondence-v1"

FIXED = {
    DATA / "independent-verification.json":
        "449a8f286a60cc4a7c542612054c4ca762378caedfb0cbc9ca350801030f2de2",
    DATA / "fit-paths.jsonl":
        "f1cd66da23cacc6025e8aa2203944f664f0d74d916a0bde135cac65ef56afd8c",
    SELECTION:
        "cbac8994a33d2a8b4d0f0c4596a1a07e206d6b75562c3416ff49877864829fd8",
}
PRODUCER_CODE = (
    ROOT / "scripts/prepare_span_preferences.py",
    ROOT / "hanglish/models/span_supervision.py",
    ROOT / "hanglish/core/source_correspondence.py",
    ROOT / "scripts/train_source_span_pair.py",
    ROOT / "scripts/preflight_source_span_residual.py",
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def digest(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def json_rows(path):
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"malformed JSONL at line {number}") from error


def write_exclusive(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _ascii(value):
    return any("a" <= char <= "z" or "A" <= char <= "Z"
               or "0" <= char <= "9" for char in value)


def expand_path(record, path_record):
    """Validate and independently expand one compact path."""
    need(isinstance(record, dict) and record.get("schema") == PATH_SCHEMA,
         "path record schema differs")
    source, edges = record.get("input"), record.get("edges")
    need(isinstance(source, str) and source and isinstance(edges, list) and edges,
         "compact source or edges malformed")
    need(isinstance(path_record, dict)
         and set(path_record) == {"path_id", "text", "edge_indices"}
         and isinstance(path_record["path_id"], str)
         and isinstance(path_record["text"], str)
         and isinstance(path_record["edge_indices"], list)
         and path_record["edge_indices"], "compact path malformed")
    spans = []
    input_cursor = output_cursor = 0
    for index in path_record["edge_indices"]:
        need(type(index) is int and 0 <= index < len(edges), "edge index malformed")
        edge = edges[index]
        need(isinstance(edge, dict)
             and set(edge) == {"input_start", "input_end", "text"},
             "compact edge fields malformed")
        start, end, text = edge["input_start"], edge["input_end"], edge["text"]
        need(type(start) is int and type(end) is int and isinstance(text, str)
             and start == input_cursor and start < end <= len(source),
             "compact edge source coverage differs")
        source_text = source[start:end]
        action = "copy" if source_text == text else "convert"
        span = {"input_start": start, "input_end": end,
                "output_start": output_cursor,
                "output_end": output_cursor + len(text),
                "source": source_text, "text": text, "action": action,
                "unresolved_ascii_output": action == "convert" and _ascii(text)}
        spans.append(span)
        input_cursor = end
        output_cursor += len(text)
    text = "".join(span["text"] for span in spans)
    need(input_cursor == len(source) and text == path_record["text"],
         "complete path source or output differs")
    identity_value = [source, [[span["input_start"], span["input_end"], span["text"]]
                               for span in spans]]
    path_id = "alignment-v1-" + digest(identity_value)
    need(path_id == path_record["path_id"], "path identity differs")
    return {"path_id": path_id, "source": source, "text": text, "spans": spans}


def selected_groups(selection_row, fit_record):
    """Validate the frozen selected domain and recover every selected path."""
    need(isinstance(selection_row, dict), "selection row malformed")
    need(fit_record.get("id") == selection_row.get("id")
         and fit_record.get("input") == selection_row.get("input")
         and fit_record.get("component_id") == selection_row.get("component_id"),
         "FIT row identity differs")
    arm = selection_row.get("arms", {}).get("b_union")
    need(isinstance(arm, dict), "selected union arm missing")
    candidates, positives = arm.get("candidates"), arm.get("positive_indices")
    need(isinstance(candidates, list) and 2 <= len(candidates) <= 8
         and all(isinstance(value, str) for value in candidates)
         and len(set(candidates)) == len(candidates), "selected candidates malformed")
    need(isinstance(positives, list) and positives
         and all(type(index) is int and 0 <= index < len(candidates)
                 for index in positives)
         and len(set(positives)) == len(positives)
         and len(positives) < len(candidates), "accepted candidate indices malformed")
    paths = fit_record.get("paths")
    need(isinstance(paths, list) and paths, "FIT compact paths missing")
    seen_ids = set()
    groups = defaultdict(list)
    for compact in paths:
        path = expand_path(fit_record, compact)
        need(path["path_id"] not in seen_ids, "duplicate structural path")
        seen_ids.add(path["path_id"])
        if path["text"] in candidates:
            groups[path["text"]].append(path)
    need(set(groups) == set(candidates), "selected candidate path omitted")
    for group in groups.values():
        group.sort(key=lambda item: item["path_id"])
    return candidates, positives, [groups[text] for text in candidates]


def contrast(left, right):
    """Compare paths at common source boundaries without edge-internal guesses."""
    need(left["source"] == right["source"], "path sources differ")
    boundaries = sorted(({0} | {s["input_end"] for s in left["spans"]})
                        & ({0} | {s["input_end"] for s in right["spans"]}))
    need(boundaries[0] == 0 and boundaries[-1] == len(left["source"]),
         "common boundaries do not cover source")
    offsets = []
    for path in (left, right):
        offsets.append({0: 0, **{span["input_end"]: span["output_end"]
                                for span in path["spans"]}})
    blocks = []
    for start, end in zip(boundaries, boundaries[1:]):
        sides, signatures = [], []
        for position, path in enumerate((left, right)):
            members = [span for span in path["spans"]
                       if start <= span["input_start"] and span["input_end"] <= end]
            need(members and members[0]["input_start"] == start
                 and members[-1]["input_end"] == end,
                 "common boundary does not contain complete spans")
            out_start, out_end = offsets[position][start], offsets[position][end]
            sides.append({"output_start": out_start, "output_end": out_end,
                          "text": path["text"][out_start:out_end]})
            signatures.append([(span["input_start"], span["input_end"], span["text"])
                               for span in members])
        relation = ("output_changed" if sides[0]["text"] != sides[1]["text"]
                    else "segmentation_only" if signatures[0] != signatures[1]
                    else "same")
        blocks.append({"input_start": start, "input_end": end,
                       "left": sides[0], "right": sides[1], "relation": relation})
    return blocks


def _side(path, block, name):
    start, end = block["input_start"], block["input_end"]
    indices = [index for index, span in enumerate(path["spans"])
               if start <= span["input_start"] and span["input_end"] <= end]
    need(indices and path["spans"][indices[0]]["input_start"] == start
         and path["spans"][indices[-1]]["input_end"] == end,
         "preference block does not cover complete spans")
    return {"path_id": path["path_id"], "span_indices": indices,
            "output_start": block[name]["output_start"],
            "output_end": block[name]["output_end"],
            "text": block[name]["text"],
            "actions": [path["spans"][index]["action"] for index in indices],
            "unresolved_ascii_output": any(
                path["spans"][index]["unresolved_ascii_output"] for index in indices)}


def reconstruct_preferences(source, candidates, positive_indices, groups):
    """Recompute the exhaustive accepted-versus-rejected local pair set."""
    positives = set(positive_indices)
    counts = Counter()
    records = {}
    for preferred_index in sorted(positives):
        for dispreferred_index in range(len(candidates)):
            if dispreferred_index in positives:
                continue
            for preferred_path in groups[preferred_index]:
                for dispreferred_path in groups[dispreferred_index]:
                    counts["path_pairs"] += 1
                    blocks = contrast(preferred_path, dispreferred_path)
                    changed = [block for block in blocks
                               if block["relation"] == "output_changed"]
                    counts[f"changed_blocks_{len(changed)}"] += 1
                    if len(changed) != 1:
                        counts["excluded_multiple_or_zero_changes"] += 1
                        continue
                    if any(block["relation"] == "segmentation_only" for block in blocks):
                        counts["excluded_outside_segmentation_changes"] += 1
                        continue
                    block = changed[0]
                    start, end = block["input_start"], block["input_end"]
                    if start == 0 and end == len(source):
                        counts["excluded_whole_source"] += 1
                        continue
                    preferred = _side(preferred_path, block, "left")
                    dispreferred = _side(dispreferred_path, block, "right")
                    prefix = preferred_path["text"][:preferred["output_start"]]
                    suffix = preferred_path["text"][preferred["output_end"]:]
                    need(prefix == dispreferred_path["text"][:dispreferred["output_start"]]
                         and suffix == dispreferred_path["text"][dispreferred["output_end"]:],
                         "outside output context differs")
                    key = digest([source, start, end, preferred_index,
                                  dispreferred_index, prefix, suffix,
                                  preferred["text"], dispreferred["text"]])
                    if key not in records:
                        records[key] = {
                            "preference_id": key, "input_start": start,
                            "input_end": end, "source": source[start:end],
                            "preferred_index": preferred_index,
                            "dispreferred_index": dispreferred_index,
                            "prefix": prefix, "suffix": suffix,
                            "preferred_text": preferred["text"],
                            "dispreferred_text": dispreferred["text"], "witnesses": []}
                    records[key]["witnesses"].append({
                        "preferred": preferred, "dispreferred": dispreferred})
    intervals = defaultdict(set)
    for pair in records.values():
        intervals[(pair["preferred_index"], pair["dispreferred_index"])].add(
            (pair["input_start"], pair["input_end"]))
    result = []
    for key in sorted(records):
        pair = records[key]
        candidate_pair = (pair["preferred_index"], pair["dispreferred_index"])
        if len(intervals[candidate_pair]) != 1:
            counts["excluded_ambiguous_interval_path_pairs"] += len(pair["witnesses"])
            continue
        result.append(pair)
        counts["eligible_path_pairs"] += len(pair["witnesses"])
    counts["distinct_preferences"] = len(result)
    counts["rows_with_preferences"] = int(bool(result))
    return result, dict(sorted(counts.items()))


def expected_row(selection_row, fit_record):
    candidates, positives, groups = selected_groups(selection_row, fit_record)
    preferences, counts = reconstruct_preferences(
        selection_row["input"], candidates, positives, groups)
    return {"schema": SCHEMA, "id": selection_row["id"],
            "input": selection_row["input"],
            "component_id": selection_row["component_id"],
            "candidates": candidates, "positive_indices": positives,
            "preferences": preferences, "counts": counts}


def aggregate(rows):
    counts = Counter()
    components, supervised = set(), set()
    local, contextual = defaultdict(set), defaultdict(set)
    actions, flags = Counter(), Counter()
    for row in rows:
        counts.update(row["counts"])
        components.add(row["component_id"])
        if row["preferences"]:
            supervised.add(row["component_id"])
        for pair in row["preferences"]:
            ordered = sorted([pair["preferred_text"], pair["dispreferred_text"]])
            direction = int(pair["preferred_text"] == ordered[1])
            local[digest([pair["source"], ordered])].add(direction)
            contextual[digest([row["input"], pair["input_start"], pair["input_end"],
                               pair["prefix"], pair["suffix"], ordered])].add(direction)
            for witness in pair["witnesses"]:
                for side in ("preferred", "dispreferred"):
                    actions.update(f"{side}:{action}"
                                   for action in witness[side]["actions"])
                    if witness[side]["unresolved_ascii_output"]:
                        flags[f"{side}:unresolved_ascii_witnesses"] += 1
    counts["rows"] = len(rows)
    counts["components"] = len(components)
    counts["supervised_components"] = len(supervised)
    return {
        "counts": dict(sorted(counts.items())),
        "witness_actions": dict(sorted(actions.items())),
        "flags": dict(sorted(flags.items())),
        "local_unordered_pairs": len(local),
        "local_pairs_with_both_preferences": sum(len(value) == 2 for value in local.values()),
        "contextual_unordered_pairs": len(contextual),
        "contextual_pairs_with_both_preferences":
            sum(len(value) == 2 for value in contextual.values()),
    }


def validate_rows(actual_rows, expected_rows):
    """Require exact ordered rows, including labels, witnesses, and omissions."""
    need(len(actual_rows) == len(expected_rows), "preference row omission or addition")
    for index, (actual, expected) in enumerate(zip(actual_rows, expected_rows)):
        need(actual == expected, f"preference row differs at selected index {index}")


def verify(output=OUTPUT):
    """Verify the actual immutable artifact and return proof contents."""
    output = Path(output)
    need(output.resolve() == OUTPUT.resolve(), "alternate output is not authorized")
    need(all(sha(path) == expected for path, expected in FIXED.items()),
         "frozen FIT authorization changed")
    fit_proof = read(DATA / "independent-verification.json")
    need(fit_proof.get("verified") is True
         and fit_proof.get("counts", {}).get("rows") == 1642
         and fit_proof.get("capture_sha256") == FIXED[DATA / "fit-paths.jsonl"],
         "source correspondence authorization differs")
    selection = read(SELECTION)
    selected_rows = selection.get("rows")
    need(isinstance(selected_rows, list) and len(selected_rows) == 1483
         and len({row.get("id") for row in selected_rows}) == 1483,
         "selected FIT denominator differs")
    selected = {row["id"]: row for row in selected_rows}
    fit = {}
    for record in json_rows(DATA / "fit-paths.jsonl"):
        row_id = record.get("id")
        if row_id in selected:
            need(row_id not in fit, "duplicate selected FIT path row")
            fit[row_id] = record
    need(set(fit) == set(selected), "selected FIT path rows omitted")

    actual_rows = list(json_rows(output / "preferences.jsonl"))
    expected_rows = []
    for selection_row in selected_rows:
        expected = expected_row(selection_row, fit[selection_row["id"]])
        expected_rows.append(expected)
    validate_rows(actual_rows, expected_rows)
    summary = aggregate(expected_rows)

    report = read(output / "report.json")
    need(isinstance(report.get("elapsed_seconds"), (int, float))
         and not isinstance(report.get("elapsed_seconds"), bool)
         and math.isfinite(report["elapsed_seconds"]) and report["elapsed_seconds"] >= 0,
         "producer elapsed time malformed")
    expected_pins = {str(path): sha(path) for path in (*FIXED, *PRODUCER_CODE)}
    stable_report = {
        "schema": SCHEMA,
        "scope": "authorized_fit_selected_candidates_only",
        **summary,
        "labels": "accepted full candidate preferred in the shared output context",
        "not_claimed": ["phonetic invalidity of dispreferred text",
                        "gold internal segmentation", "unseen sentence accuracy",
                        "candidate generation improvement"],
        "weighting_contract":
            "row mean of distinct contrasts, each mean of distinct path witnesses",
        "pins": expected_pins,
        "preferences_sha256": sha(output / "preferences.jsonl"),
    }
    actual_stable = dict(report)
    actual_stable.pop("elapsed_seconds", None)
    need(actual_stable == stable_report, "producer report or aggregate differs")
    need(report["preferences_sha256"] == sha(output / "preferences.jsonl"),
         "preference stream hash differs")
    return {
        "schema": "conditional-span-preference-independent-verification-v1",
        "verified": True,
        **summary,
        "exclusion_counts": {
            "multiple_or_zero_changes": summary["counts"].get(
                "excluded_multiple_or_zero_changes", 0),
            "outside_segmentation_changes": summary["counts"].get(
                "excluded_outside_segmentation_changes", 0),
            "whole_source": summary["counts"].get("excluded_whole_source", 0),
            "ambiguous_interval_path_pairs": summary["counts"].get(
                "excluded_ambiguous_interval_path_pairs", 0),
        },
        "preference_rows": len(expected_rows),
        "selection_sha256": sha(SELECTION),
        "fit_paths_sha256": sha(DATA / "fit-paths.jsonl"),
        "preferences_sha256": sha(output / "preferences.jsonl"),
        "report_sha256": sha(output / "report.json"),
        "verifier_sha256": sha(Path(__file__)),
        "contracts": {
            "accepted_labels_reconstructed_from_frozen_selection": True,
            "all_selected_candidate_paths_expanded_from_compact_edges": True,
            "common_source_boundaries_recomputed": True,
            "exact_preference_pair_set_and_witnesses_recomputed": True,
            "multi_change_outside_segmentation_whole_source_and_ambiguous_excluded": True,
            "row_and_report_aggregates_recomputed": True,
            "test_hold_and_canonical_labels_accessed": False,
            "models_or_torch_loaded": False,
        },
        "limitations": [
            "The frozen FIT selection supplies accepted full-candidate indices; this verifier does not establish their truth outside that authorized training scope.",
            "Producer path edges are alignment hypotheses, not gold internal segmentation or phonetic-validity labels.",
            "This proof establishes deterministic extraction integrity, not unseen accuracy or candidate-generation coverage.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    proof = verify(args.output)
    write_exclusive(args.output / "independent-verification.json", proof)
    print(json.dumps({"verified": proof["verified"],
                      "preference_rows": proof["preference_rows"],
                      "counts": proof["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
