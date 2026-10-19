#!/usr/bin/env python3
"""Independently verify source-only full-sentence document inputs."""
from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import re

from hanglish.core.phonology import romanize_word


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".omx/artifacts/document-sentence-inputs-v1"
RESERVE = ROOT / ".omx/artifacts/document-reserve-v2"
RAW = ROOT / "data/natural_context/klue-v1.1-pilot-v2/train.jsonl"
PRODUCER = ROOT / "scripts/prepare_document_sentence_inputs.py"
PRODUCER_SHA256 = "c692877f475cbc717eedbf21f3692485ec32ea365e582f8c342fb4395a911a98"
INPUTS_SHA256 = "536707f05503856f4f805bb9775bd9c5dc932c94783e669a5d62fe3023b0b1db"
MANIFEST_SHA256 = "acea9e1d4d4929253f0d4adfc2aea8ba7c877cf5093b0499961619a35a7632a3"
SCHEMA = "document-sentence-inputs-independent-verification-v1"
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
    PRODUCER: PRODUCER_SHA256,
}
HANGUL_RUN = re.compile(r"[가-힣]+")
JAMO = re.compile(r"[ᄀ-ᇿ㄰-㆏]")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    need(isinstance(value, dict), "expected JSON object: " + str(path))
    return value


def jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed JSONL at {path}:{number}") from error
            need(isinstance(value, dict), f"non-object JSONL at {path}:{number}")
            rows.append(value)
    return rows


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def digest_lines(values):
    return hashlib.sha256("".join(value + "\n" for value in values).encode()).hexdigest()


def summary(values):
    need(values, "cannot summarize empty values")
    return {"minimum": min(values), "maximum": max(values),
            "histogram": {str(key): value
                          for key, value in sorted(Counter(values).items())}}


def literal_projection(text):
    pieces = []
    cursor = 0
    literal_chunks = literal_codepoints = 0
    for match in HANGUL_RUN.finditer(text):
        literal = text[cursor:match.start()]
        if literal:
            pieces.append(literal)
            literal_chunks += 1
            literal_codepoints += len(literal)
        pieces.append(romanize_word(match.group()))
        cursor = match.end()
    literal = text[cursor:]
    if literal:
        pieces.append(literal)
        literal_chunks += 1
        literal_codepoints += len(literal)
    return "".join(pieces), literal_chunks, literal_codepoints


def replay(selection, raw_rows, *, expected_rows=1046):
    need(len(selection) == expected_rows, "reserve selection denominator differs")
    selected_ids = []
    selected_groups = []
    for row in selection:
        need(set(row) == {"id", "group_id"}
             and isinstance(row["id"], str) and row["id"]
             and isinstance(row["group_id"], str) and row["group_id"],
             "reserve selection schema differs")
        selected_ids.append(row["id"])
        selected_groups.append(row["group_id"])
    need(len(set(selected_ids)) == len(selected_ids), "duplicate selected ID")

    by_id = {}
    group_members = {}
    for row in raw_rows:
        identifier, group = row.get("id"), row.get("group_id")
        need(isinstance(identifier, str) and identifier and identifier not in by_id
             and isinstance(group, str) and group
             and isinstance(row.get("text"), str) and row["text"],
             "raw reference identity/schema differs")
        by_id[identifier] = row
        group_members.setdefault(group, set()).add(identifier)
    need(set(selected_ids) <= set(by_id), "selection contains unknown raw ID")
    chosen_groups = set(selected_groups)
    need(chosen_groups <= set(group_members), "selection contains unknown raw group")
    expected_group_ids = set().union(*(group_members[group] for group in chosen_groups))
    need(set(selected_ids) == expected_group_ids,
         "selection is not a complete whole-document-group projection")

    requests = []
    chars, tokens, utf8, ctc = [], [], [], []
    counts = Counter()
    inputs = Counter()
    for selected in selection:
        reference = by_id[selected["id"]]
        need(reference["group_id"] == selected["group_id"],
             "selected ID/group mapping differs")
        source = romanize_word(reference["text"])
        projected, chunks, codepoints = literal_projection(reference["text"])
        need(source == projected and source, "literal-preserving romanization differs")
        requests.append({"id": selected["id"], "input": source})
        inputs[source] += 1
        char_count = len(source)
        token_count = len(source.split())
        byte_count = len(source.encode("utf-8"))
        chars.append(char_count)
        tokens.append(token_count)
        utf8.append(byte_count)
        ctc.append(byte_count + 2)
        counts["rows"] += 1
        counts["nonempty_sources"] += bool(source)
        counts["multi_token_sources"] += token_count >= 2
        counts["ctc_token_limit_exceeded_rows"] += byte_count + 2 > 512
        counts["source_utf8_over_512_rows"] += byte_count > 512
        counts["ui_character_limit_exceeded_rows"] += char_count > 2000
        residual_hangul = sum(len(match.group()) for match in HANGUL_RUN.finditer(source))
        residual_jamo = len(JAMO.findall(source))
        counts["residual_precomposed_hangul_codepoints"] += residual_hangul
        counts["rows_with_residual_precomposed_hangul"] += bool(residual_hangul)
        counts["residual_jamo_codepoints"] += residual_jamo
        counts["rows_with_residual_jamo"] += bool(residual_jamo)
        counts["non_hangul_literal_chunks"] += chunks
        counts["non_hangul_literal_codepoints"] += codepoints
    counts["groups"] = len(chosen_groups)
    counts["unique_inputs"] = len(inputs)
    counts["duplicate_input_rows"] = sum(value - 1 for value in inputs.values())
    counts["non_hangul_literal_preservation_failures"] = 0
    return requests, {
        "counts": dict(counts),
        "source_character_lengths": summary(chars),
        "whitespace_token_counts": summary(tokens),
        "source_utf8_bytes": summary(utf8),
        "ctc_input_token_counts": summary(ctc),
        "ordered_ids_sha256": digest_lines(selected_ids),
        "ordered_group_ids_sha256": digest_lines(list(dict.fromkeys(selected_groups))),
    }


def verify(directory=RUN):
    directory = Path(directory)
    proof_path = directory / "independent-verification.json"
    need(directory.is_dir() and not proof_path.exists(), "run/proof state invalid")
    inputs_path = directory / "inputs.jsonl"
    manifest_path = directory / "manifest.json"
    need(inputs_path.is_file() and manifest_path.is_file(), "input artifacts missing")
    need(sha(inputs_path) == INPUTS_SHA256 and sha(manifest_path) == MANIFEST_SHA256,
         "frozen input artifact identity differs")
    need(all(path.is_file() and sha(path) == digest for path, digest in PINS.items()),
         "pinned source or producer changed")

    manifest = read(manifest_path)
    reserve_manifest = read(RESERVE / "manifest.json")
    reserve_proof = read(RESERVE / "independent-verification.json")
    need(reserve_manifest.get("completed") is True
         and reserve_manifest.get("schema") == "document-reserve-v2"
         and reserve_manifest.get("counts", {}).get("rows") == 1046
         and reserve_manifest.get("counts", {}).get("groups") == 821
         and reserve_proof.get("schema") == "document-reserve-independent-check-v1"
         and reserve_proof.get("producer_imported") is False
         and reserve_proof.get("rows") == 1046
         and reserve_proof.get("groups") == 821
         and reserve_proof.get("model_activity") is False
         and reserve_proof.get("manifest_sha256") == PINS[RESERVE / "manifest.json"]
         and reserve_proof.get("selection_sha256") == PINS[RESERVE / "selection.jsonl"],
         "reserve selection proof differs")
    selection = jsonl(RESERVE / "selection.jsonl")
    raw_rows = jsonl(RAW)
    expected, projection = replay(selection, raw_rows)
    actual = jsonl(inputs_path)
    need(len(actual) == 1046 and all(set(row) == {"id", "input"}
                                    for row in actual),
         "source-only request schema/denominator differs")
    need(actual == expected, "request order, ID, or full-text projection differs")
    need(len({row["id"] for row in actual}) == 1046,
         "request IDs are not one-to-one with selection")

    expected_sources = {str(path.relative_to(ROOT)): digest
                        for path, digest in PINS.items()}
    need(manifest.get("schema") == "document-sentence-inputs-v1"
         and manifest.get("completed") is True
         and manifest.get("source_hashes") == expected_sources
         and manifest.get("inputs_sha256") == sha(inputs_path)
         and manifest.get("model_activity") is False
         and manifest.get("performance_labels_joined") is False,
         "manifest source/model identity differs")
    for key in ("counts", "source_character_lengths", "whitespace_token_counts",
                "source_utf8_bytes", "ctc_input_token_counts"):
        need(manifest.get(key) == projection[key], "manifest projection differs: " + key)
    selection_contract = manifest.get("selection_contract")
    need(selection_contract == {
        "rows": 1046, "groups": 821,
        "order": "exact document-reserve-v2 selection order",
        "whole_document_groups": True,
        "ordered_ids_sha256": projection["ordered_ids_sha256"],
        "ordered_group_ids_sha256": projection["ordered_group_ids_sha256"],
        "duplicate_inputs_keep_distinct_ids": True},
        "selection contract differs")
    need(selection_contract["ordered_ids_sha256"]
             == reserve_manifest["counts"]["ordered_ids_sha256"]
         and selection_contract["ordered_group_ids_sha256"]
             == reserve_manifest["counts"]["ordered_group_ids_sha256"],
         "reserve selection order proof differs")
    need(manifest.get("evaluation_contract") == {
        "source_only_requests": True, "request_fields": ["id", "input"],
        "model_input_fields": ["input"], "context": "",
        "full_reference_text_romanized_once": True,
        "all_selected_ids_retained": True,
        "unsupported_sources_filtered": False,
        "reference_text_emitted": False, "group_id_emitted": False},
        "source-only evaluation contract differs")
    need(manifest.get("reference_access") == (
        "Pinned full references parsed only to synthesize complete Roman "
        "inputs and verify reserve identity/group mapping.")
         and manifest.get("limits") == [
            "Roman inputs are synthetic outputs of the pinned current romanizer, not independent user pronunciation labels.",
            "Document separation does not establish lexical-family separation or absence from external pretraining.",
            "Rows beyond runtime or UI limits remain in the source-only request set and require aggregate support handling.",
            "No sentence accuracy, candidate quality, English-copy performance, or model latency is measured here."],
         "manifest scope/limitations differ")

    proof = {
        "schema": SCHEMA, "verified": True,
        "inputs_sha256": sha(inputs_path), "manifest_sha256": sha(manifest_path),
        "producer_sha256": PRODUCER_SHA256,
        "verifier_sha256": sha(Path(__file__)),
        "counts": projection["counts"],
        "lengths": {key: projection[key] for key in (
            "source_character_lengths", "whitespace_token_counts",
            "source_utf8_bytes", "ctc_input_token_counts")},
        "selection": {"rows": 1046, "groups": 821,
                      "ordered_ids_sha256": projection["ordered_ids_sha256"],
                      "ordered_group_ids_sha256": projection["ordered_group_ids_sha256"]},
        "contracts_verified": {
            "all_1046_rows_and_821_whole_groups": True,
            "exact_selection_order_and_id_group_mapping": True,
            "exact_full_text_romanize_word_projection": True,
            "non_hangul_literals_case_digits_punctuation_whitespace_preserved": True,
            "same_input_rows_retained_with_distinct_ids": True,
            "only_id_and_input_emitted": True,
            "no_filtering_context_group_reference_or_target_fields": True,
            "source_hashes_and_reserve_selection_proof": True,
            "model_inference_training_or_http_activity": False,
        },
        "limitations": [
            "Word-level forms from these documents were previously exposed to evaluation; this is not an untouched reserve.",
            "The Roman input is a deterministic synthetic rendering and not an independent pronunciation annotation.",
            "Document separation does not establish lexical-family separation or absence from external pretraining.",
            "This proof covers input construction only and measures no model accuracy, candidate quality, or latency.",
        ],
    }
    with proof_path.open("xb") as stream:
        stream.write(canonical(proof) + b"\n")
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    print(json.dumps(verify(args.run), ensure_ascii=False, sort_keys=True))
