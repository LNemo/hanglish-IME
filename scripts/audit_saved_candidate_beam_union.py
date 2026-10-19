"""Upper-bound TRAIN reachability from saved candidate edges plus frozen beams."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.neural_converter import LETTERS, PROTECTED, TOKENS
from hanglish.core.phonology import romanize_word


INPUT = ROOT / ".omx/artifacts/semantic-pair-training-input-v2"
CENSUS = ROOT / ".omx/artifacts/phonetic-training-beam-census-v1"
TARGETS = ROOT / "data/natural_context/final-choice-v1/natural_fit/cases.jsonl"
MECHANICAL = ROOT / ".omx/artifacts/multistep-morphology-coverage-v2/source-capture.jsonl"
OUTPUT = ROOT / ".omx/artifacts/saved-candidate-beam-union-v1"
INPUT_MANIFEST_SHA = "3ab561d1a767e7a65b4706c2cd0f668366693536849e17f2f3e620fbcf5550a6"
INPUT_ROWS_SHA = "6a1f7f61a7f7ffeb9209772559412931137d71ef0173d7429264d9cb53aed396"
CENSUS_REPORT_SHA = "c4d517d7ef913dab595f1a11630fa9f771b84b1baa8e7deb24eeba11e9e686fa"
CENSUS_SELECTION_SHA = "6439732e7742238156c247863abdda51783e74a1809c65a67c98ff12f7345251"
CENSUS_CAPTURE_SHA = "6afd08b45a6eb49b566e7d835a40e3965279fd6cc9ea68e84435041171bb4fed"
TARGETS_SHA = "789e45a3b6028e7416ca3b82739c906c558ec414320796c53b0b76ed251b9a18"
MECHANICAL_SHA = "472a5d7ebcb376e89d040583264d5b2d2b39204500499e0cf8d6b6c738b52adb"
ROWS = 2048
INPUTS = 18122


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_jsonl(raw):
    return [json.loads(line) for line in raw.splitlines() if line]


def verify_pins(pins):
    require(isinstance(pins, dict) and pins, "census source pins missing")
    for name, digest in pins.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        require(path.is_file() and sha(path) == digest, f"source pin differs: {name}")


def source_occurrences(text):
    """Return frozen-token-pattern letter spans and their protection state."""
    if len(LETTERS.findall(text)) > 128:
        return []
    found = []
    for match in TOKENS.finditer(text):
        chunk = match.group()
        if chunk.isspace():
            continue
        protected = bool(PROTECTED.fullmatch(chunk))
        for part in LETTERS.finditer(chunk):
            source = part.group()
            start = match.start() + part.start()
            if (source.isascii() and source.isalpha() and source.islower()
                    and 2 <= len(source) <= 40):
                found.append((source, start, start + len(source), protected))
    return found


def validate_candidate(candidate, source_text):
    require(set(candidate) == {"object", "source_index"}
            and type(candidate["source_index"]) is int,
            "saved candidate envelope differs")
    value = candidate["object"]
    segments = value.get("segments")
    require(isinstance(value.get("text"), str) and isinstance(segments, list) and segments,
            "saved candidate object differs")
    cursor = 0
    rendered = []
    for segment in segments:
        start, end = segment.get("input_start"), segment.get("input_end")
        require(type(start) is int and type(end) is int and start == cursor < end
                and end <= len(source_text)
                and segment.get("source") == source_text[start:end]
                and isinstance(segment.get("text"), str) and segment["text"]
                and isinstance(segment.get("kind"), str) and segment["kind"],
                "saved candidate segment path differs")
        cursor = end
        rendered.append(segment["text"])
    require(cursor == len(source_text) and "".join(rendered) == value["text"],
            "saved candidate path does not close")
    return value


def add_edge(edges, *, start, end, text, source, provenance):
    require(type(start) is int and type(end) is int and 0 <= start < end
            and isinstance(text, str) and text and isinstance(source, str),
            "zero or malformed DAG edge")
    key = (start, end, text, source)
    edges.setdefault(key, []).append(provenance)


def beam_edges(row, beam_records):
    edges = {}
    for source, start, end, protected in source_occurrences(row["input"]):
        if protected:
            continue
        captured = beam_records.get(source)
        require(captured is not None and row["id"] in captured["unprotected_row_ids"],
                "unprotected source occurrence is absent from census membership")
        for rank, output in enumerate(captured["outputs"]):
            add_edge(edges, start=start, end=end, text=output["text"], source=source,
                     provenance={"type": "phonetic_beam", "rank": rank + 1})
    return edges


def mechanical_edges(row, mechanical):
    require(mechanical is not None and mechanical.get("id") == row["id"]
            and mechanical.get("input") == row["input"],
            "mechanical row identity differs")
    edges = {}
    cursor = 0
    for piece_index, piece in enumerate(mechanical.get("pieces", [])):
        start, end, source = piece.get("start"), piece.get("end"), piece.get("source")
        require(start == cursor and type(end) is int and start < end <= len(row["input"])
                and row["input"][start:end] == source
                and isinstance(piece.get("baseline"), list) and piece["baseline"],
                "mechanical source piece differs")
        for option_index, option in enumerate(piece["baseline"]):
            require(isinstance(option.get("text"), str) and option["text"]
                    and isinstance(option.get("kind"), str) and option["kind"]
                    and type(option.get("score")) in (int, float)
                    and math.isfinite(option["score"]),
                    "mechanical baseline option differs")
            add_edge(edges, start=start, end=end, text=option["text"], source=source,
                     provenance={"type": "mechanical_baseline", "piece_index": piece_index,
                                 "option_index": option_index, "kind": option["kind"]})
        cursor = end
    require(cursor == len(row["input"]), "mechanical pieces do not close")
    return edges


def merged_edges(*mappings):
    combined = {}
    for mapping in mappings:
        for key, provenance in mapping.items():
            combined.setdefault(key, []).extend(provenance)
    return combined


def edge_records(edges):
    return [
        {"start": start, "end": end, "text": text, "source": source,
         "provenance": sorted(provenance, key=lambda item: canonical(item))}
        for (start, end, text, source), provenance in sorted(edges.items())
    ]


def arm_edges(graph, allowed):
    return [edge for edge in graph["edges"]
            if any(item["type"] in allowed for item in edge["provenance"])]


def build_graph(row, beam_records, mechanical=None):
    edges = {}
    saved_texts = []
    require(row.get("finite_candidate_indices")
            == [candidate.get("source_index") for candidate in row["finite_candidates"]]
            and len(set(row["finite_candidate_indices"]))
            == len(row["finite_candidate_indices"]),
            "finite candidate source indices differ")
    for candidate in row["finite_candidates"]:
        value = validate_candidate(candidate, row["input"])
        saved_texts.append(value["text"])
        for segment_index, segment in enumerate(value["segments"]):
            add_edge(edges, start=segment["input_start"], end=segment["input_end"],
                     text=segment["text"], source=segment["source"], provenance={
                         "type": "saved_candidate", "source_index": candidate["source_index"],
                         "segment_index": segment_index, "kind": segment["kind"]})
    saved, beams = edges, beam_edges(row, beam_records)
    mechanical_values = {} if mechanical is None else mechanical_edges(row, mechanical)
    saved_beam = merged_edges(saved, beams)
    saved_mechanical = merged_edges(saved, mechanical_values)
    full = merged_edges(saved, mechanical_values, beams)
    require(set(saved).issubset(saved_beam)
            and set(saved).issubset(saved_mechanical) and set(saved).issubset(full),
            "adding captured edges removed a saved edge")
    baseline = edge_records(saved)
    require(baseline and all(reachable(text, baseline, len(row["input"]))
                             for text in saved_texts),
            "saved candidate path was lost from its edge graph")
    return {"index": row["index"], "id": row["id"], "input": row["input"],
            "saved_candidate_texts": saved_texts,
            "edges": edge_records(full)}


def reachable(target, edges, source_length):
    outgoing = defaultdict(list)
    for edge in edges:
        require(0 <= edge["start"] < edge["end"] <= source_length,
                "DAG edge boundary differs")
        outgoing[edge["start"]].append(edge)
    states = {(0, 0)}
    agenda = [(0, 0)]
    while agenda:
        source_at, target_at = agenda.pop()
        for edge in outgoing.get(source_at, ()):
            text = edge["text"]
            if target.startswith(text, target_at):
                state = (edge["end"], target_at + len(text))
                if state not in states:
                    states.add(state)
                    agenda.append(state)
    return (source_length, len(target)) in states


def load_source_inputs():
    manifest_raw = (INPUT / "manifest.json").read_bytes()
    rows_raw = (INPUT / "rows.jsonl").read_bytes()
    require(sha_bytes(manifest_raw) == INPUT_MANIFEST_SHA
            and sha_bytes(rows_raw) == INPUT_ROWS_SHA,
            "frozen semantic source input differs")
    manifest = json.loads(manifest_raw)
    require(manifest.get("completed") is True and manifest.get("rows") == 3492
            and manifest.get("role_rows") == {"natural_fit": 2048,
                                                "challenge_fit": 1444}
            and manifest.get("labels_or_expected_saved") is False,
            "semantic source manifest differs")
    rows = [row for row in parse_jsonl(rows_raw)
            if row.get("source_role") == "natural_fit"]
    require(len(rows) == ROWS and [row["index"] for row in rows] == list(range(ROWS))
            and len({row["id"] for row in rows}) == ROWS
            and all("expected" not in row for row in rows),
            "natural source-only rows differ")
    return rows


def load_census():
    report_raw = (CENSUS / "report.json").read_bytes()
    selection_raw = (CENSUS / "selection.jsonl").read_bytes()
    capture_raw = (CENSUS / "capture.jsonl").read_bytes()
    require(sha_bytes(report_raw) == CENSUS_REPORT_SHA
            and sha_bytes(selection_raw) == CENSUS_SELECTION_SHA
            and sha_bytes(capture_raw) == CENSUS_CAPTURE_SHA,
            "frozen beam census differs")
    report = json.loads(report_raw)
    require(report.get("completed") is True and report.get("target_fields_read") is False
            and report.get("unique_inputs") == INPUTS
            and report.get("captured_inputs") == INPUTS
            and report.get("selection_sha256") == CENSUS_SELECTION_SHA
            and report.get("capture_sha256") == CENSUS_CAPTURE_SHA
            and report.get("source_hashes_start") == report.get("source_hashes_end"),
            "closed beam census contract differs")
    verify_pins(report["source_hashes_start"])
    selection, capture = parse_jsonl(selection_raw), parse_jsonl(capture_raw)
    require(len(selection) == len(capture) == INPUTS, "beam census denominator differs")
    result = {}
    for selected, captured in zip(selection, capture):
        source = selected.get("input")
        protected_ids = selected.get("protected_row_ids")
        unprotected_ids = selected.get("unprotected_row_ids")
        row_ids = selected.get("row_ids")
        require(set(selected) == {"index", "input", "row_ids",
                                  "protected_row_ids", "unprotected_row_ids"}
                and selected["index"] == len(result)
                and isinstance(source, str) and source.isascii()
                and source.isalpha() and source.islower() and 2 <= len(source) <= 40
                and all(isinstance(values, list) and values == sorted(set(values))
                        for values in (row_ids, protected_ids, unprotected_ids))
                and row_ids == sorted(set(protected_ids) | set(unprotected_ids)),
                "beam selection membership differs")
        require(captured.get("index") == selected.get("index")
                and captured.get("input") == selected.get("input")
                and set(captured) == {"index", "input", "outputs"}
                and isinstance(captured["outputs"], list)
                and len(captured["outputs"]) <= 8,
                "beam capture and selection differ")
        previous = float("inf")
        seen = set()
        for output in captured["outputs"]:
            require(set(output) == {"text", "score"}
                    and isinstance(output["text"], str)
                    and output["text"] and all("가" <= char <= "힣"
                                               for char in output["text"])
                    and output["text"] not in seen
                    and romanize_word(output["text"]) == source
                    and type(output["score"]) in (int, float)
                    and math.isfinite(output["score"])
                    and output["score"] <= previous,
                    "beam output differs")
            previous = output["score"]
            seen.add(output["text"])
        require(source not in result, "beam input duplicated")
        result[source] = {"outputs": captured["outputs"],
                          "protected_row_ids": frozenset(protected_ids),
                          "unprotected_row_ids": frozenset(unprotected_ids)}
    return result


def load_mechanical():
    raw = MECHANICAL.read_bytes()
    require(sha_bytes(raw) == MECHANICAL_SHA, "frozen mechanical capture differs")
    rows = parse_jsonl(raw)
    require(len(rows) == ROWS and len({row.get("id") for row in rows}) == ROWS,
            "mechanical capture denominator differs")
    return rows


def write_source_graphs(output, rows, beams, mechanical_rows):
    output.mkdir(parents=True)
    graph_path = output / "source-graphs.jsonl"
    saved_edges = union_edges = beam_edges = 0
    with graph_path.open("xb") as stream:
        for row, mechanical in zip(rows, mechanical_rows):
            graph = build_graph(row, beams, mechanical)
            saved = arm_edges(graph, {"saved_candidate"})
            saved_edges += len(saved)
            union_edges += len(graph["edges"])
            beam_edges += sum(any(item["type"] == "phonetic_beam"
                                  for item in edge["provenance"])
                              for edge in graph["edges"])
            stream.write(canonical(graph) + b"\n")
    closure = {
        "schema": "saved-candidate-beam-union-source-closure-v1",
        "completed": True, "rows": len(rows),
        "saved_edges": saved_edges, "union_edges": union_edges,
        "edges_with_beam_provenance": beam_edges,
        "source_graphs_sha256": sha(graph_path),
        "source_hashes": {
            ".omx/artifacts/semantic-pair-training-input-v2/manifest.json": INPUT_MANIFEST_SHA,
            ".omx/artifacts/semantic-pair-training-input-v2/rows.jsonl": INPUT_ROWS_SHA,
            ".omx/artifacts/phonetic-training-beam-census-v1/report.json": CENSUS_REPORT_SHA,
            ".omx/artifacts/phonetic-training-beam-census-v1/selection.jsonl": CENSUS_SELECTION_SHA,
            ".omx/artifacts/phonetic-training-beam-census-v1/capture.jsonl": CENSUS_CAPTURE_SHA,
            ".omx/artifacts/multistep-morphology-coverage-v2/source-capture.jsonl": MECHANICAL_SHA,
            "scripts/audit_saved_candidate_beam_union.py": sha(Path(__file__)),
        },
        "target_fields_read": False, "model_inference_run": False,
    }
    closure_path = output / "source-closure.json"
    closure_path.write_bytes(canonical(closure) + b"\n")
    return graph_path, closure_path, closure


def evaluate_graphs(graphs, targets):
    require(len(graphs) == len(targets) == ROWS
            and [row["id"] for row in graphs] == [row["id"] for row in targets],
            "source graph and target denominator differ")
    counts = Counter({key: 0 for key in (
        "saved_exact", "newly_recombined", "newly_beam", "still_missing",
        "full_partition_saved_exact", "full_partition_newly_recombined",
        "full_partition_newly_mechanical",
        "full_partition_newly_beam_after_mechanical",
        "full_partition_still_missing",
        "nonrecombined_beam_and_mechanical", "nonrecombined_beam_only",
        "nonrecombined_mechanical_only", "nonrecombined_interaction_only",
        "nonrecombined_neither")})
    counts["rows"] = ROWS
    outcomes = []
    for graph, target in zip(graphs, targets):
        expected = target.get("expected")
        require(isinstance(expected, list) and expected,
                "accepted target list differs")
        saved_exact = any(value in graph["saved_candidate_texts"] for value in expected)
        saved = arm_edges(graph, {"saved_candidate"})
        saved_beam_edges = arm_edges(graph, {"saved_candidate", "phonetic_beam"})
        saved_mechanical_edges = arm_edges(
            graph, {"saved_candidate", "mechanical_baseline"})
        full_edges = graph["edges"]
        recombined = any(reachable(value, saved, len(graph["input"]))
                         for value in expected)
        beam_union = any(reachable(value, saved_beam_edges, len(graph["input"]))
                         for value in expected)
        mechanical_union = any(reachable(value, saved_mechanical_edges,
                                         len(graph["input"])) for value in expected)
        full_union = any(reachable(value, full_edges,
                                   len(graph["input"])) for value in expected)
        require(not saved_exact or recombined, "saved exact path is not reachable")
        require(not recombined or (beam_union and mechanical_union and full_union),
                "captured edge arm lost an original path")
        category = ("saved_exact" if saved_exact else
                    "newly_recombined" if recombined else
                    "newly_beam" if beam_union else "still_missing")
        counts[category] += 1
        full_category = ("saved_exact" if saved_exact else
                         "newly_recombined" if recombined else
                         "newly_mechanical" if mechanical_union else
                         "newly_beam_after_mechanical" if full_union else "still_missing")
        counts["saved_beam_reachable"] += beam_union
        counts["saved_mechanical_reachable"] += mechanical_union
        counts["saved_mechanical_beam_reachable"] += full_union
        counts["full_partition_" + full_category] += 1
        if not recombined:
            overlap = ("beam_and_mechanical" if beam_union and mechanical_union else
                       "beam_only" if beam_union else
                       "mechanical_only" if mechanical_union else
                       "interaction_only" if full_union else "neither")
            counts["nonrecombined_" + overlap] += 1
        outcomes.append({"id": graph["id"], "category": category,
                         "saved_exact": saved_exact,
                         "saved_edges_reachable": recombined,
                         "saved_beam_reachable": beam_union,
                         "saved_mechanical_reachable": mechanical_union,
                         "saved_mechanical_beam_reachable": full_union,
                         "full_partition_category": full_category})
    require(sum(counts[key] for key in (
        "saved_exact", "newly_recombined", "newly_beam", "still_missing")) == ROWS,
        "coverage partition does not close")
    require(sum(counts["full_partition_" + key] for key in (
        "saved_exact", "newly_recombined", "newly_mechanical",
        "newly_beam_after_mechanical", "still_missing")) == ROWS,
        "mechanical plus beam coverage partition does not close")
    require(sum(counts["nonrecombined_" + key] for key in (
        "beam_and_mechanical", "beam_only", "mechanical_only",
        "interaction_only", "neither")) == ROWS - counts["saved_exact"]
        - counts["newly_recombined"], "captured-arm overlap does not close")
    return dict(counts), outcomes


def run(output=OUTPUT):
    output = Path(output)
    require(not output.exists(), "audit output already exists")
    rows = load_source_inputs()
    beams = load_census()
    mechanical = load_mechanical()
    require([row["id"] for row in mechanical] == [row["id"] for row in rows]
            and [row["input"] for row in mechanical] == [row["input"] for row in rows],
            "mechanical capture source order differs")
    graph_path, closure_path, closure = write_source_graphs(
        output, rows, beams, mechanical)
    # The target-bearing file is first opened only after source graphs are closed and hashed.
    targets_raw = TARGETS.read_bytes()
    require(sha_bytes(targets_raw) == TARGETS_SHA, "natural TRAIN target hash differs")
    targets = parse_jsonl(targets_raw)
    graphs_raw = graph_path.read_bytes()
    require(sha_bytes(graphs_raw) == closure["source_graphs_sha256"],
            "source graph changed after target access")
    graphs = parse_jsonl(graphs_raw)
    counts, outcomes = evaluate_graphs(graphs, targets)
    report = {
        "schema": "saved-candidate-beam-union-v1", "completed": True,
        "scope": "2048-row TRAIN upper-bound candidate reachability diagnostic",
        "counts": counts, "outcomes": outcomes,
        "source_graphs_sha256": closure["source_graphs_sha256"],
        "source_closure_sha256": sha(closure_path), "targets_sha256": TARGETS_SHA,
        "source_graphs_closed_before_targets": True,
        "all_original_paths_preserved": True,
        "model_inference_run": False, "ranking_or_pruning_measured": False,
        "runtime_accuracy_claimed": False,
        "interpretation": (
            "Upper-bound reachability only: saved edges may recombine into paths that the "
            "runtime never enumerates, ranks, or displays."
        ),
    }
    (output / "report.json").write_bytes(canonical(report) + b"\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
