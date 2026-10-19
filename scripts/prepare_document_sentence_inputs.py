"""Freeze full-sentence source-only requests from document-reserve-v2.

Reference text is read only to synthesize deterministic Roman input and verify
the frozen selection.  Reference text and document groups are never emitted in
the request stream.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.phonology import romanize_word


RESERVE = ROOT / ".omx/artifacts/document-reserve-v2"
RAW = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
OUTPUT = ROOT / ".omx/artifacts/document-sentence-inputs-v1"
PINS = {
    RAW: "904aab04033ac758d5ccda7d8cdd75d9bd14872ad31017a14d884633e5de4d3b",
    RESERVE / "selection.jsonl":
        "7d62027d11dcf1320b2a1bac7bcb83e1a31d31ce521808bbe4d1faadf6d4b810",
    RESERVE / "manifest.json":
        "19223e3c507432c168205732c26c4a95dc1e2391a0698dea96eaac63d70c3859",
    RESERVE / "independent-verification.json":
        "e5a4a4da7ede8b38b8fba2cbc0b909e79e5304b4f9df20466b34c0068fab8a80",
    ROOT / "hanglish/core/phonology.py":
        "e593f06409646b74d8aa8ca62663dcdd1b62759254ae1d1d23fc948fed54b865",
    ROOT / "hanglish/core/predicate_boundaries.py":
        "d0d29ca399220e6707e674ebf29a1368162e21741c1f0691d985d284ffc9abb1",
    ROOT / "hanglish/core/data/predicate_boundary_inventory.json":
        "5da10f07dd9b1a494ad7465c0e215c78982b6d7e252b1f603f7232cf714bb06a",
}
HANGUL_RUN = re.compile(r"[가-힣]+")
JAMO = re.compile(r"[ᄀ-ᇿ㄰-㆏]")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def parse_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row at {path}:{number}")
            rows.append(value)
    return rows


def _literal_projection(text, romanize):
    """Rebuild a conversion from Hangul runs and untouched literal chunks."""
    pieces = []
    cursor = 0
    literal_chunks = 0
    literal_codepoints = 0
    for match in HANGUL_RUN.finditer(text):
        literal = text[cursor:match.start()]
        if literal:
            literal_chunks += 1
            literal_codepoints += len(literal)
            pieces.append(literal)
        pieces.append(romanize(match.group()))
        cursor = match.end()
    literal = text[cursor:]
    if literal:
        literal_chunks += 1
        literal_codepoints += len(literal)
        pieces.append(literal)
    return "".join(pieces), literal_chunks, literal_codepoints


def _summary(values):
    if not values:
        raise ValueError("cannot summarize an empty source collection")
    return {"minimum": min(values), "maximum": max(values),
            "histogram": {str(key): value for key, value
                          in sorted(Counter(values).items())}}


def project(selection, raw_rows, romanize=romanize_word):
    """Project the exact selection order to full-sentence model inputs."""
    if not isinstance(selection, list) or not isinstance(raw_rows, list):
        raise ValueError("selection and references must be lists")
    selected_ids = set()
    groups = []
    for row in selection:
        if (not isinstance(row, dict) or set(row) != {"id", "group_id"}
                or not isinstance(row["id"], str) or not row["id"]
                or not isinstance(row["group_id"], str) or not row["group_id"]
                or row["id"] in selected_ids):
            raise ValueError("invalid or duplicate reserve selection")
        selected_ids.add(row["id"])
        groups.append(row["group_id"])

    by_id = {}
    for row in raw_rows:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str)
                or not row["id"] or row["id"] in by_id):
            raise ValueError("invalid or duplicate raw document ID")
        by_id[row["id"]] = row

    requests = []
    char_lengths = []
    whitespace_tokens = []
    utf8_lengths = []
    ctc_tokens = []
    counts = Counter()
    seen_inputs = Counter()
    for selected in selection:
        reference = by_id.get(selected["id"])
        if reference is None:
            raise ValueError("reserve selection is missing from raw references")
        if (reference.get("group_id") != selected["group_id"]
                or not isinstance(reference.get("text"), str)
                or not reference["text"]):
            raise ValueError("reserve group mapping or document reference differs")
        text = reference["text"]
        source = romanize(text)
        if not isinstance(source, str) or not source:
            raise ValueError("romanized full-sentence source is invalid")
        literal_projection, literal_chunks, literal_codepoints = (
            _literal_projection(text, romanize))
        if source != literal_projection:
            raise ValueError("non-Hangul literal boundaries were not preserved")

        request = {"id": selected["id"], "input": source}
        if set(request) != {"id", "input"}:
            raise AssertionError("source-only request schema differs")
        requests.append(request)
        seen_inputs[source] += 1

        characters = len(source)
        tokens = len(source.split())
        utf8_bytes = len(source.encode("utf-8"))
        char_lengths.append(characters)
        whitespace_tokens.append(tokens)
        utf8_lengths.append(utf8_bytes)
        ctc_tokens.append(utf8_bytes + 2)
        counts["rows"] += 1
        counts["nonempty_sources"] += bool(source)
        counts["multi_token_sources"] += tokens >= 2
        counts["ctc_token_limit_exceeded_rows"] += utf8_bytes + 2 > 512
        counts["source_utf8_over_512_rows"] += utf8_bytes > 512
        counts["ui_character_limit_exceeded_rows"] += characters > 2000
        counts["residual_precomposed_hangul_codepoints"] += sum(
            len(match.group()) for match in HANGUL_RUN.finditer(source))
        counts["rows_with_residual_precomposed_hangul"] += bool(
            HANGUL_RUN.search(source))
        counts["residual_jamo_codepoints"] += len(JAMO.findall(source))
        counts["rows_with_residual_jamo"] += bool(JAMO.search(source))
        counts["non_hangul_literal_chunks"] += literal_chunks
        counts["non_hangul_literal_codepoints"] += literal_codepoints

    counts["groups"] = len(set(groups))
    counts["unique_inputs"] = len(seen_inputs)
    counts["duplicate_input_rows"] = sum(value - 1 for value in seen_inputs.values())
    counts["non_hangul_literal_preservation_failures"] = 0
    summary = {
        "counts": dict(counts),
        "source_character_lengths": _summary(char_lengths),
        "whitespace_token_counts": _summary(whitespace_tokens),
        "source_utf8_bytes": _summary(utf8_lengths),
        "ctc_input_token_counts": _summary(ctc_tokens),
    }
    return requests, summary


def write_artifacts(output, requests, manifest):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if (not isinstance(requests, list) or not requests
            or any(not isinstance(row, dict) or set(row) != {"id", "input"}
                   for row in requests)):
        raise ValueError("source-only request output schema differs")
    content = b"".join(canonical(row) + b"\n" for row in requests)
    if manifest.get("inputs_sha256") != hashlib.sha256(content).hexdigest():
        raise ValueError("manifest input digest differs")
    output.mkdir(parents=True, exist_ok=False)
    with (output / "inputs.jsonl").open("xb") as stream:
        stream.write(content)
    with (output / "manifest.json").open("xb") as stream:
        stream.write(canonical(manifest) + b"\n")


def run(output=OUTPUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    for path, expected in PINS.items():
        if sha(path) != expected:
            raise ValueError(f"source changed: {path}")
    producer = Path(__file__).resolve()
    pins = {str(path.relative_to(ROOT)): digest for path, digest in PINS.items()}
    pins[str(producer.relative_to(ROOT))] = sha(producer)

    selection = parse_jsonl(RESERVE / "selection.jsonl")
    raw_rows = parse_jsonl(RAW)
    reserve_manifest = json.loads((RESERVE / "manifest.json").read_text())
    if (len(selection) != 1046
            or reserve_manifest.get("counts", {}).get("rows") != 1046
            or reserve_manifest.get("counts", {}).get("groups") != 821):
        raise ValueError("frozen document reserve cardinality differs")
    requests, projection = project(selection, raw_rows)
    counts = projection["counts"]
    if counts["rows"] != 1046 or counts["groups"] != 821:
        raise ValueError("projected document reserve cardinality differs")
    content = b"".join(canonical(row) + b"\n" for row in requests)
    manifest = {
        "schema": "document-sentence-inputs-v1",
        "completed": True,
        "counts": counts,
        "source_hashes": pins,
        "inputs_sha256": hashlib.sha256(content).hexdigest(),
        "source_character_lengths": projection["source_character_lengths"],
        "whitespace_token_counts": projection["whitespace_token_counts"],
        "source_utf8_bytes": projection["source_utf8_bytes"],
        "ctc_input_token_counts": projection["ctc_input_token_counts"],
        "selection_contract": {
            "rows": 1046,
            "groups": 821,
            "order": "exact document-reserve-v2 selection order",
            "whole_document_groups": True,
            "ordered_ids_sha256": reserve_manifest["counts"]["ordered_ids_sha256"],
            "ordered_group_ids_sha256":
                reserve_manifest["counts"]["ordered_group_ids_sha256"],
            "duplicate_inputs_keep_distinct_ids": True,
        },
        "evaluation_contract": {
            "source_only_requests": True,
            "request_fields": ["id", "input"],
            "model_input_fields": ["input"],
            "context": "",
            "full_reference_text_romanized_once": True,
            "all_selected_ids_retained": True,
            "unsupported_sources_filtered": False,
            "reference_text_emitted": False,
            "group_id_emitted": False,
        },
        "model_activity": False,
        "performance_labels_joined": False,
        "reference_access": (
            "Pinned full references parsed only to synthesize complete Roman "
            "inputs and verify reserve identity/group mapping."),
        "limits": [
            "Roman inputs are synthetic outputs of the pinned current romanizer, not independent user pronunciation labels.",
            "Document separation does not establish lexical-family separation or absence from external pretraining.",
            "Rows beyond runtime or UI limits remain in the source-only request set and require aggregate support handling.",
            "No sentence accuracy, candidate quality, English-copy performance, or model latency is measured here.",
        ],
    }
    for raw, expected in pins.items():
        if sha(ROOT / raw) != expected:
            raise ValueError(f"source changed during build: {raw}")
    write_artifacts(output, requests, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    print(json.dumps(run(parser.parse_args().output)["counts"], ensure_ascii=False))
