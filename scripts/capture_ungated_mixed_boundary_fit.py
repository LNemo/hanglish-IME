#!/usr/bin/env python3
"""Capture mixed digit-boundary proposals from closed FIT public parents."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

AUDIT = ROOT / ".omx/artifacts/opaque-fit-span-audit-v1"
GATE = ROOT / ".omx/artifacts/fit-generation-gate-audit-v1.json"
PLAN = ROOT / ".omx/plans/ungated-mixed-boundary-fit-v1.md"
MODEL = ROOT / ".omx/artifacts/observed-phonetic-augmented-training-v1/phonetic_transducer.pt"
SHARDS = [ROOT / f".omx/artifacts/current-phonetic-train-shard-{i}-v1/capture.jsonl"
          for i in range(6)]
ROWS = 66
PARENT_VERSION = "experimental-score-scale-1-f6976e1b269d"
LIMITS = {"whole_seconds": 1200.0, "row_seconds": 120.0,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2,
          "output_bytes": 128 * 1024**2}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2,
                  allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())


def verify_pins(pins):
    for name, expected in pins.items():
        path = Path(name)
        need(path.is_file() and sha(path) == expected, "frozen input differs: " + name)


def selected_audit_row(row):
    """Selection uses only copied raw digit spans from public nonraw parents."""
    for span in row.get("opaque_spans", []):
        if ("digits" in span.get("flags", [])
                and any(parent.get("kind") == "raw"
                        and parent.get("source_copied_exactly") is True
                        for parent in span.get("parents", []))):
            return True
    return False


def authenticate():
    report = read(AUDIT / "report.json")
    proof = read(AUDIT / "independent-verification.json")
    gate = read(GATE)
    need(report.get("completed") is True and report.get("counts", {}).get("rows") == 1642
         and report.get("labels_opened") is False and report.get("neural_inference") is False,
         "opaque source audit is not closed")
    need(proof.get("verified") is True and proof.get("rows") == 1642,
         "opaque source audit is not independently verified")
    need(proof.get("report_sha256") == sha(AUDIT / "report.json")
         and proof.get("capture_sha256") == report.get("capture_sha256")
         == sha(AUDIT / "capture.jsonl"),
         "opaque source proof hashes differ")
    pins = dict(report.get("pins", {}))
    need(all(str(path.resolve()) in pins for path in SHARDS), "six FIT source pins missing")
    need(all(pins[str(path.resolve())] == gate["pins"][str(path.resolve())]
             for path in SHARDS), "FIT audit/gate source hashes differ")
    for path in (AUDIT / "report.json", AUDIT / "capture.jsonl",
                 AUDIT / "independent-verification.json", GATE, PLAN,
                 ROOT / "hanglish/core/mixed_span_proposals.py",
                 ROOT / "hanglish/core/phonology.py", ROOT / "hanglish/core/lattice.py",
                 Path(__file__).resolve()):
        pins[str(path.resolve())] = sha(path)
    verify_pins(pins)
    return pins


def public_parents(source_row):
    source = source_row["request"]["input"]
    response = source_row["response"]
    need(source_row["request"] == {"input": source, "context": ""}
         and response.get("input") == source, "FIT source request differs")
    result = []
    for candidate in response.get("candidates", []):
        segments = candidate.get("segments")
        if not segments or all(segment.get("kind") == "raw" for segment in segments):
            continue
        result.append((candidate["text"], 0.0, deepcopy(segments)))
    need(1 <= len(result) <= 4, "public nonraw parent count differs")
    return result


def validate_path(source, path):
    text, score, segments = path
    need(isinstance(text, str) and not isinstance(score, bool)
         and isinstance(score, (int, float)) and math.isfinite(score)
         and isinstance(segments, list), "proposal path differs")
    cursor = 0
    for segment in segments:
        start, end = segment.get("input_start"), segment.get("input_end")
        need(type(start) is int and type(end) is int and start == cursor and end > start
             and source[start:end] == segment.get("source"), "proposal offsets differ")
        if segment.get("kind") == "raw":
            need(segment.get("text") == segment.get("source"), "raw source was changed")
        cursor = end
    need(cursor == len(source) and text == "".join(s["text"] for s in segments),
         "proposal render differs")


def validate_changed_digit_regions(source, path, parents):
    """Check helper-created chunks without rejudging preserved legacy edges."""
    from hanglish.core.phonology import romanize_word
    _, _, proposal_segments = path
    regions = {(s["input_start"], s["input_end"], s["source"])
               for _, _, segments in parents for s in segments
               if s.get("kind") == "raw" and s.get("text") == s.get("source")
               and any(c.isdigit() for c in s.get("source", ""))}
    changed = 0
    for start, end, chunk in regions:
        pieces = [s for s in proposal_segments
                  if s["input_start"] >= start and s["input_end"] <= end]
        if not pieces or pieces[0]["input_start"] != start or pieces[-1]["input_end"] != end:
            continue
        rendered = "".join(s["text"] for s in pieces)
        if rendered != chunk:
            need(romanize_word(rendered) == chunk,
                 "changed mixed chunk is not forward-valid")
            changed += 1
    need(changed > 0, "proposal did not change a copied raw digit region")


class RecordingOptions:
    def __init__(self, callback):
        self.callback = callback
        self.cache = {}
        self.calls = []

    def __call__(self, source, learned=True):
        need(learned is True, "word option mode differs")
        if source not in self.cache:
            values = list(self.callback(source, learned=True))
            serial = []
            for option in values:
                need(isinstance(option.text, str) and option.text
                     and not isinstance(option.score, bool)
                     and isinstance(option.score, (int, float)) and math.isfinite(option.score)
                     and isinstance(option.kind, str), "word option differs")
                serial.append({"text": option.text, "score": float(option.score),
                               "kind": option.kind})
            self.cache[source] = values
            self.calls.append({"source": source, "learned": True, "options": serial})
        return list(self.cache[source])


def token_support(ranker, text):
    normalized = " ".join(text.strip().split())
    ids = ranker.tokenizer(normalized, add_special_tokens=False,
                           truncation=False)["input_ids"]
    if ids and isinstance(ids[0], list): ids = ids[0]
    reserved = ranker.tokenizer.num_special_tokens_to_add(pair=True)
    count = len(ids) + reserved
    supported = count <= ranker.max_length
    need(supported == ranker.can_score(text), "token support annotation differs")
    return {"token_count_with_pair_specials": count,
            "max_tokens": ranker.max_length, "supported": supported}


def capture_row(audit_row, source_row, word_options, ranker):
    from hanglish.core.mixed_span_proposals import sentence_extras
    need(selected_audit_row(audit_row), "unselected audit row")
    need(audit_row["id"] == source_row["id"]
         and audit_row["request"] == source_row["request"], "source identity differs")
    source = source_row["request"]["input"]
    parents = public_parents(source_row)
    snapshot = deepcopy(parents)
    recorder = RecordingOptions(word_options)
    started = time.monotonic()
    extras = sentence_extras(source, parents, recorder, limit=96)
    need(parents == snapshot and len(extras) <= 96, "parent mutation/proposal budget differs")
    proposals = []
    parent_texts = {p[0] for p in parents}
    for path in extras:
        validate_path(source, path)
        validate_changed_digit_regions(source, path, parents)
        need(path[0] not in parent_texts, "proposal duplicates preserved parent")
        proposals.append({"text": path[0], "score": float(path[1]),
                          "segments": deepcopy(path[2]),
                          "token_support": token_support(ranker, path[0])})
    return {"id": audit_row["id"], "request": deepcopy(audit_row["request"]),
            "source_shard": audit_row["source_shard"],
            "source_row_index": audit_row["source_row_index"],
            "opaque_spans": deepcopy(audit_row["opaque_spans"]),
            "preserved_parents": [{"text": p[0], "score": p[1], "segments": p[2]}
                                  for p in parents],
            "callback_records": recorder.calls, "proposals": proposals,
            "proposal_count": len(proposals), "zero_proposal": not proposals,
            "protected_or_no_option": not proposals,
            "elapsed_seconds": time.monotonic() - started}


def _source_row(audit_row):
    path = SHARDS[audit_row["source_shard"]]
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index == audit_row["source_row_index"]:
                return json.loads(line)
    raise ValueError("source row index missing")


def run(output: Path):
    output.mkdir(parents=True, exist_ok=False)
    from scripts.train_character_edge_full_fit import ResourceGuard
    import torch
    from hanglish.core.score_scale_converter import ScoreScaleConverter
    guard = ResourceGuard(output)
    def observe(stage):
        value = guard.observe(stage)
        need(value["seconds"] <= LIMITS["whole_seconds"]
             and value["rss_bytes"] <= LIMITS["rss_bytes"]
             and value["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"],
             "resources exceeded")
        need(sum(p.stat().st_size for p in output.iterdir() if p.is_file())
             <= LIMITS["output_bytes"], "output exceeded")
        return value
    try:
        pins = authenticate()
        selected = [row for row in (json.loads(line) for line in
                    (AUDIT / "capture.jsonl").open()) if selected_audit_row(row)]
        need(len(selected) == ROWS, "selected FIT row count differs")
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        engine = ScoreScaleConverter(phonetic_checkpoint=MODEL)
        engine.load_models(); engine.load_word_model(); engine.load_growth_model()
        need(engine.version == PARENT_VERSION, "current parent version differs")
        for name, expected in engine.model_artifacts.items():
            path = Path(name) if Path(name).is_absolute() else ROOT / name
            need(sha(path) == expected, "model artifact differs: " + str(path))
            pins[str(path.resolve())] = expected
        verify_pins(pins); observe("models-loaded")
        counts = {"rows": 0, "proposals": 0, "supported": 0,
                  "unsupported": 0, "zero_proposal_rows": 0,
                  "callback_requests": 0}
        with (output / "capture.jsonl").open("x", encoding="utf-8") as target:
            for audit_row in selected:
                row = capture_row(audit_row, _source_row(audit_row),
                                  engine.word_options, engine.sentence_ranker)
                need(row["elapsed_seconds"] <= LIMITS["row_seconds"], "row time exceeded")
                target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                target.flush()
                counts["rows"] += 1; counts["proposals"] += len(row["proposals"])
                counts["supported"] += sum(p["token_support"]["supported"] for p in row["proposals"])
                counts["unsupported"] += sum(not p["token_support"]["supported"] for p in row["proposals"])
                counts["zero_proposal_rows"] += row["zero_proposal"]
                counts["callback_requests"] += len(row["callback_records"])
                observe("row-" + row["id"])
        verify_pins(pins); final = observe("complete")
        report = {"schema": "ungated-mixed-boundary-fit-capture-v1",
                  "completed": True, "counts": counts, "pins": pins, "limits": LIMITS,
                  "resources": final, "max_rss_bytes": guard.peak_rss,
                  "max_swap_growth_bytes": guard.peak_swap_growth,
                  "capture_sha256": sha(output / "capture.jsonl"),
                  "resources_sha256": sha(output / "resources.jsonl"),
                  "labels_opened": False, "dev_opened": False, "test_opened": False,
                  "sentence_scoring_run": False, "whole_parent_convert_run": False,
                  "production_changed": False}
        write(output / "report.json", report)
        write(output / "capture.complete.json", {"completed": True, "rows": ROWS,
              "report_sha256": sha(output / "report.json"),
              "capture_sha256": report["capture_sha256"], "labels_opened": False})
    except Exception as error:
        write(output / "failure.json", {"type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    run(parser.parse_args().output)
