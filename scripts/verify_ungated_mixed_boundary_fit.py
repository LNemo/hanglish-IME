"""Verify a closed mixed-boundary FIT capture from saved option callbacks."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AUDIT = ROOT / ".omx/artifacts/opaque-fit-span-audit-v1"
SHARDS = [ROOT / f".omx/artifacts/current-phonetic-train-shard-{index}-v1/capture.jsonl"
          for index in range(6)]
CHECKPOINT = ROOT / "checkpoints/sentence_ranker"
ROWS = 66
LIMITS = {"whole_seconds": 1200.0, "row_seconds": 120.0,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2,
          "output_bytes": 128 * 1024**2}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def iter_jsonl(path: Path):
    with path.open() as stream:
        for line in stream:
            require(bool(line.strip()), f"blank row in {path.name}")
            yield json.loads(line)


def selected_audit_row(row) -> bool:
    return any(
        "digits" in span.get("flags", [])
        and any(parent.get("kind") == "raw"
                and parent.get("source_copied_exactly") is True
                for parent in span.get("parents", []))
        for span in row.get("opaque_spans", []))


def validate_segments(source, text, segments, label):
    require(isinstance(text, str) and isinstance(segments, list),
            label + " path differs")
    cursor = 0
    for segment in segments:
        require(isinstance(segment, dict), label + " segment differs")
        start, end = segment.get("input_start"), segment.get("input_end")
        require(type(start) is int and type(end) is int and start == cursor
                and start < end <= len(source)
                and segment.get("source") == source[start:end]
                and isinstance(segment.get("text"), str)
                and isinstance(segment.get("kind"), str),
                label + " source coverage differs")
        if segment["kind"] == "raw":
            require(segment["text"] == segment["source"],
                    label + " raw COPY differs")
        cursor = end
    require(cursor == len(source)
            and "".join(segment["text"] for segment in segments) == text,
            label + " full render differs")


def public_parents(source_row):
    request = source_row.get("request")
    require(isinstance(request, dict) and request.get("context") == ""
            and isinstance(request.get("input"), str), "FIT request differs")
    source = request["input"]
    response = source_row.get("response")
    require(isinstance(response, dict) and response.get("input") == source,
            "FIT response source differs")
    parents = []
    for index, candidate in enumerate(response.get("candidates", [])):
        segments = candidate.get("segments") if isinstance(candidate, dict) else None
        require(isinstance(segments, list), "public candidate segments differ")
        if not segments or all(segment.get("kind") == "raw" for segment in segments):
            continue
        validate_segments(source, candidate.get("text"), segments,
                          f"public parent {index}")
        parents.append({"text": candidate["text"], "score": 0.0,
                        "segments": deepcopy(segments)})
    require(1 <= len(parents) <= 4, "public nonraw parent count differs")
    return parents


def validate_callback_records(records):
    from hanglish.core.lattice import WordOption

    require(isinstance(records, list), "callback records differ")
    values, ordered = {}, []
    for record in records:
        require(isinstance(record, dict)
                and set(record) == {"source", "learned", "options"}
                and isinstance(record["source"], str) and record["source"]
                and record["learned"] is True
                and record["source"] not in values
                and isinstance(record["options"], list),
                "callback record differs")
        options = []
        for option in record["options"]:
            require(isinstance(option, dict)
                    and set(option) == {"text", "score", "kind"}
                    and isinstance(option["text"], str) and option["text"]
                    and not isinstance(option["score"], bool)
                    and isinstance(option["score"], (int, float))
                    and math.isfinite(option["score"])
                    and isinstance(option["kind"], str),
                    "saved WordOption differs")
            options.append(WordOption(
                option["text"], float(option["score"]), option["kind"]))
        values[record["source"]] = options
        ordered.append(record["source"])
    return values, ordered


class SavedOptions:
    def __init__(self, values):
        self.values = values
        self.calls = []
        self.cache = {}

    def __call__(self, source, learned=True):
        require(learned is True and source in self.values,
                "replay requested an uncaptured word option")
        if source not in self.cache:
            self.cache[source] = list(self.values[source])
            self.calls.append(source)
        return list(self.cache[source])


def token_support(tokenizer, text):
    normalized = " ".join(text.strip().split())
    encoded = tokenizer(normalized, add_special_tokens=False, truncation=False)
    ids = encoded["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    require(isinstance(ids, list) and all(type(value) is int for value in ids),
            "untruncated tokenizer output differs")
    reserved = tokenizer.num_special_tokens_to_add(pair=True)
    require(type(reserved) is int and reserved == 3,
            "sentence tokenizer pair reserve differs")
    count = len(ids) + reserved
    return {"token_count_with_pair_specials": count,
            "max_tokens": 96, "supported": count <= 96}


def _parent_matches(source, proposal_segments, parent):
    digit_regions = []
    for segment in parent["segments"]:
        copied_digit = (segment["kind"] == "raw"
                        and segment["text"] == segment["source"]
                        and any(character.isdigit() for character in segment["source"]))
        if copied_digit:
            digit_regions.append((segment["input_start"], segment["input_end"],
                                  segment["source"]))
            continue
        matches = [value for value in proposal_segments
                   if value["input_start"] == segment["input_start"]
                   and value["input_end"] == segment["input_end"]]
        if len(matches) != 1 or matches[0] != segment:
            return None
    changed = []
    for start, end, chunk in digit_regions:
        pieces = [segment for segment in proposal_segments
                  if segment["input_start"] >= start and segment["input_end"] <= end]
        if (not pieces or pieces[0]["input_start"] != start
                or pieces[-1]["input_end"] != end):
            return None
        cursor, rendered = start, []
        for piece in pieces:
            if piece["input_start"] != cursor:
                return None
            cursor = piece["input_end"]
            rendered.append(piece["text"])
        if "".join(rendered) != chunk:
            changed.append((start, end, chunk, "".join(rendered), pieces))
    return changed


def validate_changed_chunks(source, proposal, parents):
    from hanglish.core.phonology import romanize_word

    matches = []
    for parent in parents:
        changed = _parent_matches(source, proposal["segments"], parent)
        if changed:
            matches.append(changed)
    require(matches, "proposal does not preserve a parent hypothesis")
    for start, end, chunk, rendered, pieces in matches[0]:
        require(romanize_word(rendered) == chunk,
                "changed digit chunk fails forward correspondence")
        for piece in pieces:
            if piece["kind"] == "raw":
                require(piece["text"] == piece["source"],
                        "changed chunk raw COPY differs")
            else:
                require(romanize_word(piece["text"]) == piece["source"],
                        "new converted edge fails forward correspondence")


def replay_row(audit_row, source_row, captured, tokenizer):
    from hanglish.core.mixed_span_proposals import sentence_extras

    require(selected_audit_row(audit_row)
            and captured.get("id") == audit_row.get("id") == source_row.get("id")
            and captured.get("request") == audit_row.get("request")
                == source_row.get("request")
            and captured.get("source_shard") == audit_row.get("source_shard")
            and captured.get("source_row_index") == audit_row.get("source_row_index")
            and captured.get("opaque_spans") == audit_row.get("opaque_spans"),
            "capture source provenance differs")
    source = audit_row["request"]["input"]
    parents = public_parents(source_row)
    require(captured.get("preserved_parents") == parents,
            "score-neutral public parents differ")
    values, recorded_order = validate_callback_records(
        captured.get("callback_records"))
    callback = SavedOptions(values)
    parent_paths = [(parent["text"], parent["score"], deepcopy(parent["segments"]))
                    for parent in parents]
    frozen = deepcopy(parent_paths)
    extras = sentence_extras(source, parent_paths, callback, limit=96)
    require(parent_paths == frozen and callback.calls == recorded_order
            and len(extras) <= 96, "same-helper replay contract differs")
    saved = captured.get("proposals")
    require(isinstance(saved, list) and len(saved) == len(extras)
            and len(saved) <= 96
            and len({row.get("text") for row in saved}) == len(saved),
            "proposal count, uniqueness, or budget differs")
    parent_texts = {parent["text"] for parent in parents}
    supported = unsupported = 0
    for index, (path, actual) in enumerate(zip(extras, saved)):
        text, score, segments = path
        require(isinstance(actual, dict)
                and actual.get("text") == text
                and not isinstance(actual.get("score"), bool)
                and isinstance(actual.get("score"), (int, float))
                and math.isfinite(actual["score"])
                and math.isclose(float(actual["score"]), float(score),
                                 rel_tol=0.0, abs_tol=1e-12)
                and actual.get("segments") == segments
                and text not in parent_texts,
                f"saved proposal {index} differs")
        validate_segments(source, text, segments, f"proposal {index}")
        validate_changed_chunks(source, actual, parents)
        support = token_support(tokenizer, text)
        require(actual.get("token_support") == support,
                "proposal token support differs")
        supported += int(support["supported"])
        unsupported += int(not support["supported"])
    require(captured.get("proposal_count") == len(saved)
            and captured.get("zero_proposal") is (not saved)
            and captured.get("protected_or_no_option") is (not saved)
            and not isinstance(captured.get("elapsed_seconds"), bool)
            and isinstance(captured.get("elapsed_seconds"), (int, float))
            and math.isfinite(captured["elapsed_seconds"])
            and 0 <= captured["elapsed_seconds"] <= LIMITS["row_seconds"],
            "row diagnostics differ")
    return {"proposals": len(saved), "supported": supported,
            "unsupported": unsupported, "zero_proposal_rows": int(not saved),
            "callback_requests": len(recorded_order)}


def authenticate(directory: Path):
    proof_path = directory / "independent-verification.json"
    require(directory.is_dir() and not proof_path.exists(),
            "capture directory missing or proof already exists")
    report_path, closure_path = directory / "report.json", directory / "capture.complete.json"
    report, closure = read_json(report_path), read_json(closure_path)
    require(report.get("schema") == "ungated-mixed-boundary-fit-capture-v1"
            and report.get("completed") is True
            and report.get("counts", {}).get("rows") == ROWS
            and report.get("limits") == LIMITS
            and report.get("labels_opened") is False
            and report.get("dev_opened") is False
            and report.get("test_opened") is False
            and report.get("sentence_scoring_run") is False
            and report.get("whole_parent_convert_run") is False
            and report.get("production_changed") is False
            and report.get("capture_sha256") == digest(directory / "capture.jsonl")
            and report.get("resources_sha256") == digest(directory / "resources.jsonl")
            and closure == {"completed": True, "rows": ROWS,
                            "report_sha256": digest(report_path),
                            "capture_sha256": report["capture_sha256"],
                            "labels_opened": False},
            "closed mixed-boundary capture differs")
    pins = report.get("pins")
    require(isinstance(pins, dict) and pins
            and all(isinstance(path, str) and isinstance(expected, str)
                    and len(expected) == 64 and digest(Path(path)) == expected
                    for path, expected in pins.items()),
            "capture pins differ")
    audit_report = read_json(AUDIT / "report.json")
    audit_proof = read_json(AUDIT / "independent-verification.json")
    require(audit_report.get("completed") is True
            and audit_report.get("counts", {}).get("rows") == 1642
            and audit_proof.get("verified") is True and audit_proof.get("rows") == 1642
            and audit_proof.get("report_sha256") == digest(AUDIT / "report.json")
            and audit_proof.get("capture_sha256") == digest(AUDIT / "capture.jsonl")
            and all(str(path.resolve()) in pins for path in SHARDS),
            "opaque audit source closure differs")
    resources = list(iter_jsonl(directory / "resources.jsonl"))
    require(resources and [row.get("seconds") for row in resources]
            == sorted(row.get("seconds") for row in resources),
            "resource trace order differs")
    for row in resources:
        require(isinstance(row.get("stage"), str)
                and not isinstance(row.get("seconds"), bool)
                and isinstance(row["seconds"], (int, float))
                and math.isfinite(row["seconds"])
                and 0 <= row["seconds"] <= LIMITS["whole_seconds"]
                and type(row.get("rss_bytes")) is int
                and 0 <= row["rss_bytes"] <= LIMITS["rss_bytes"]
                and type(row.get("swap_growth_bytes")) is int
                and row["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"],
                "resource observation differs")
    require(report.get("resources") == resources[-1]
            and report.get("max_rss_bytes") == max(row["rss_bytes"] for row in resources)
            and report.get("max_swap_growth_bytes")
                == max(0, max(row["swap_growth_bytes"] for row in resources)),
            "reported resource maxima differ")
    frozen = {"report_sha256": digest(report_path),
              "closure_sha256": digest(closure_path),
              "capture_sha256": digest(directory / "capture.jsonl"),
              "resources_sha256": digest(directory / "resources.jsonl")}
    return report, closure, pins, frozen


def source_rows(selected):
    wanted = {(row["source_shard"], row["source_row_index"]): row for row in selected}
    found = {}
    for shard, path in enumerate(SHARDS):
        with path.open() as stream:
            for index, line in enumerate(stream):
                key = (shard, index)
                if key in wanted:
                    found[key] = json.loads(line)
    require(set(found) == set(wanted), "selected source rows missing")
    return [found[(row["source_shard"], row["source_row_index"])]
            for row in selected]


def verify(directory: Path):
    started = time.monotonic()
    directory = directory.resolve()
    report, closure, pins, frozen = authenticate(directory)
    audit_rows = list(iter_jsonl(AUDIT / "capture.jsonl"))
    require(len(audit_rows) == 1642
            and len({row.get("id") for row in audit_rows}) == 1642,
            "opaque audit denominator differs")
    selected = [row for row in audit_rows if selected_audit_row(row)]
    require(len(selected) == len({row["id"] for row in selected}) == ROWS,
            "all-raw-digit selected control differs")
    originals = source_rows(selected)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT, local_files_only=True, trust_remote_code=False)
    require(tokenizer.num_special_tokens_to_add(pair=True) == 3,
            "actual sentence tokenizer contract differs")
    captured_rows = list(iter_jsonl(directory / "capture.jsonl"))
    require(len(captured_rows) == ROWS, "capture denominator differs")
    counts = Counter(rows=0, proposals=0, supported=0, unsupported=0,
                     zero_proposal_rows=0, callback_requests=0)
    ordered_ids = []
    for audit_row, original, captured in zip(selected, originals, captured_rows):
        row_started = time.monotonic()
        values = replay_row(audit_row, original, captured, tokenizer)
        counts.update(rows=1, **values)
        ordered_ids.append(captured["id"])
        require(time.monotonic() - row_started <= LIMITS["row_seconds"],
                "independent replay row time exceeded")
        require(time.monotonic() - started <= LIMITS["whole_seconds"],
                "independent replay whole time exceeded")
    require(dict(counts) == report.get("counts")
            and counts["supported"] + counts["unsupported"] == counts["proposals"],
            "fresh capture counters differ")
    closing_report, closing_closure, closing_pins, closing_frozen = authenticate(directory)
    require(closing_report == report and closing_closure == closure
            and closing_pins == pins and closing_frozen == frozen,
            "capture evidence changed during replay")
    elapsed = time.monotonic() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss *= 1 if sys.platform == "darwin" else 1024
    require(elapsed <= LIMITS["whole_seconds"] and rss <= LIMITS["rss_bytes"],
            "independent verifier resources exceeded")
    result = {
        "schema": "ungated-mixed-boundary-fit-independent-verification-v1",
        "verified": True, "rows": ROWS, "ordered_ids": ordered_ids,
        "counts": dict(counts), "report_sha256": frozen["report_sha256"],
        "closure_sha256": frozen["closure_sha256"],
        "capture_sha256": frozen["capture_sha256"],
        "resources_sha256": frozen["resources_sha256"],
        "verifier_sha256": digest(Path(__file__).resolve()),
        "pins_verified": len(pins),
        "same_helper_saved_option_replay": True,
        "word_model_reinference": False, "sentence_model_inference": False,
        "tokenizer_only": True, "no_truncation": True,
        "unsupported_proposals_retained": True,
        "public_parents_score_neutral_and_preserved": True,
        "new_edges_and_changed_chunks_forward_verified": True,
        "raw_copy_verified": True, "labels_opened": False,
        "dev_opened": False, "test_opened": False,
        "production_changed": False,
        "resources": {"seconds": elapsed, "max_rss_bytes": rss,
                      "limits": LIMITS},
        "scope": "Same mixed_span_proposals helper replay from saved finite WordOptions; independent source, edge, token-support, budget, counter, pin, and closure checks. No neural word reinference.",
    }
    proof_path = directory / "independent-verification.json"
    with proof_path.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    verify(parser.parse_args().directory)
