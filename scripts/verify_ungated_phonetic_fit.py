"""Verify the closed ungated-phonetic FIT capture without labels or neural rerun."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.ungated_phonetic_proposals import generate_proposals


AUDIT = ROOT / ".omx/artifacts/fit-generation-gate-audit-v1.json"
AUDIT_SHA = "7ce91a2d28e6bb0c0b802a1369924d2fd78d95a20c8e93bd60b61aa77d87f0b0"
MODEL = ROOT / ".omx/artifacts/observed-phonetic-augmented-training-v1/phonetic_transducer.pt"
MODEL_SHA = "f06b0a7cd6c5ccbdfaa376cc29f59e90b3fcf0ae3a157fd2c4f7efaa73b4180d"
HELPER = ROOT / "hanglish/core/ungated_phonetic_proposals.py"
HELPER_SHA = "821110345736ce889adff33acf01e04e11caf944a2ad084f845213e54c689215"
RUNNER = ROOT / "scripts/capture_ungated_phonetic_fit.py"
RUNNER_SHA = "26e4a0a3a0b1b1150140ba47f5d72db9abc5e2e6ed5a2b719e8c745c87f4a774"
PLAN = ROOT / ".omx/plans/ungated-phonetic-fit-v1.md"
PLAN_SHA = "1f8cd2f26b49bbb9cbb43ec8981bee984a4b82482ed4ee100cc3322be11d3777"
CHECKPOINT = ROOT / "checkpoints/sentence_ranker"
WARNING = "문장 모델의 길이 범위를 넘어 단어별 문맥 순위를 사용했습니다."
LIMITS = {"whole_seconds": 900.0, "row_seconds": 120.0,
          "rss_bytes": 2 * 1024**3, "swap_growth_bytes": 256 * 1024**2,
          "output_bytes": 64 * 1024**2}
ROW_KEYS = {
    "id", "request", "source_shard", "source_row_index", "fallback_public",
    "parents", "proposals", "generation_requests", "generated_by_source",
    "skipped_known_english", "skipped_generation_sources",
    "generation_budget_exhausted",
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def authenticate(directory: Path):
    report_path = directory / "report.json"
    capture_path = directory / "capture.jsonl"
    resources_path = directory / "resources.jsonl"
    report = read_json(report_path)
    need(report.get("schema") == "ungated-phonetic-fit-v1"
         and report.get("completed") is True and report.get("rows") == 13
         and report.get("labels_opened") is False
         and report.get("sentence_model_inference") is False
         and report.get("production_changed") is False
         and report.get("limits") == LIMITS
         and report.get("beam_size") == 8 and report.get("max_sources") == 96
         and report.get("max_extras") == 96
         and report.get("max_sentence_tokens") == 96,
         "ungated capture report contract differs")
    need(report.get("capture_sha256") == sha(capture_path)
         and report.get("resources_sha256") == sha(resources_path),
         "capture or resource digest differs")
    pins = report.get("pins")
    need(isinstance(pins, dict) and bool(pins), "capture pins missing")
    for path, expected in pins.items():
        need(isinstance(path, str) and isinstance(expected, str) and len(expected) == 64
             and sha(Path(path)) == expected, "capture pin changed: " + str(path))
    fixed = {AUDIT: AUDIT_SHA, MODEL: MODEL_SHA, HELPER: HELPER_SHA,
             RUNNER: RUNNER_SHA, PLAN: PLAN_SHA}
    for path, expected in fixed.items():
        resolved = str(path.resolve())
        need(sha(path) == expected and pins.get(resolved) == expected,
             "fixed source or checkpoint differs: " + resolved)
    english = report.get("known_english")
    rebuilt_english = set((ROOT / "data/common_english.txt").read_text().lower().split())
    rebuilt_english.update((ROOT / "data/it_terms.txt").read_text().lower().split())
    rebuilt_english.update({"test", "api", "hello", "world", "openai"})
    need(isinstance(english, list) and english == sorted(set(english))
         and all(isinstance(value, str) and value == value.lower() for value in english)
         and english == sorted(rebuilt_english),
         "known-English inventory differs")
    return report, sha(report_path), sha(capture_path), sha(resources_path)


def load_expected_sources(report):
    audit = read_json(AUDIT)
    need(audit.get("schema") == "fit-generation-gate-audit-v1"
         and audit.get("labels_opened") is False
         and audit.get("neural_inference_run") is False
         and audit.get("max_length") == 96 and audit.get("reserved_pair_tokens") == 3
         and audit.get("counts") == {
             "rows": 1642, "ranked": 1629, "ranked_length_warning": 0,
             "ranked_empty_recorded_pool": 0, "ranked_zero_generation_requests": 0,
             "unranked": 13, "unranked_length_warning": 13,
             "unranked_empty_recorded_pool": 13,
             "unranked_zero_generation_requests": 13,
         }, "fixed source audit differs")
    selected = audit.get("unranked_rows")
    need(isinstance(selected, list) and len(selected) == 13
         and len({row.get("id") for row in selected}) == 13,
         "source audit selection differs")
    wanted = {row["id"]: row for row in selected}
    sources = [ROOT / f".omx/artifacts/current-phonetic-train-shard-{shard}-v1/capture.jsonl"
               for shard in range(6)]
    for path in sources:
        need(report["pins"].get(str(path.resolve())) == sha(path),
             "FIT source is not pinned by capture report")
    found, seen = [], set()
    for shard, path in enumerate(sources):
        with path.open() as stream:
            for row_index, line in enumerate(stream):
                source = json.loads(line)
                identifier = source.get("id")
                need(isinstance(identifier, str) and identifier.startswith("s3-train-")
                     and identifier not in seen, "FIT source identity differs")
                seen.add(identifier)
                ranked = source.get("response", {}).get("sentence_ranking_applied")
                need(type(ranked) is bool and ((not ranked) == (identifier in wanted)),
                     "audit selection no longer matches source")
                if identifier not in wanted:
                    continue
                expected = wanted[identifier]
                response = source["response"]
                trace = response["phonetic_word_union"]
                need(expected.get("shard") == shard and expected.get("row_index") == row_index
                     and expected.get("ranked") is False
                     and expected.get("length_warning") is True
                     and expected.get("empty_recorded_pool") is True
                     and expected.get("generation_requests") == 0
                     and expected.get("generated_extras") == 0
                     and WARNING in response.get("warnings", [])
                     and not source.get("base_paths")
                     and trace.get("generation_requests") == 0
                     and trace.get("generated_extras") == []
                     and source.get("request", {}).get("context") == "",
                     "selected FIT source trace differs")
                found.append((expected, source))
    need(len(seen) == 1642 and len(found) == 13
         and [row[0]["id"] for row in found] == [row["id"] for row in selected],
         "FIT source coverage or selected order differs")
    return found


def _nonraw_parents(public):
    result = []
    for candidate in public:
        segments = candidate.get("segments")
        need(isinstance(segments, list) and bool(segments), "fallback candidate segments differ")
        if all(segment.get("kind") == "raw" for segment in segments):
            continue
        result.append({"text": candidate.get("text"), "segments": segments})
    return result


def _strip_support(candidate):
    need(set(candidate) >= {"text", "segments", "sentence_tokens", "sentence_support"},
         "candidate support fields missing")
    return {key: value for key, value in candidate.items()
            if key not in {"sentence_tokens", "sentence_support"}}


def _verify_support(candidate, tokenizer, maximum):
    text = " ".join(candidate["text"].strip().split())
    tokens = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    count = len(tokens) + tokenizer.num_special_tokens_to_add(pair=True)
    status = "supported" if count <= maximum else "unsupported"
    need(type(candidate.get("sentence_tokens")) is int
         and candidate["sentence_tokens"] == count
         and candidate.get("sentence_support") == status,
         "independent tokenizer support differs")
    return status


def verify_row(captured, expected_entry, source, tokenizer, report):
    need(set(captured) == ROW_KEYS
         and captured.get("id") == expected_entry["id"] == source["id"]
         and captured.get("request") == source["request"]
         and captured.get("source_shard") == expected_entry["shard"]
         and captured.get("source_row_index") == expected_entry["row_index"]
         and captured.get("fallback_public") == source["response"]["candidates"],
         "captured source or full fallback identity differs")
    raw_beams = captured.get("generated_by_source")
    need(isinstance(raw_beams, dict), "captured raw beams differ")
    calls = []

    def recorded_generate(source_word, beam_size=8):
        need(beam_size == 8 and source_word in raw_beams and source_word not in calls,
             "shared-helper replay requested an unknown or duplicate beam")
        calls.append(source_word)
        return raw_beams[source_word]

    expected_parents = _nonraw_parents(source["response"]["candidates"])
    replay = generate_proposals(
        source["request"]["input"], expected_parents, recorded_generate,
        known_english=report["known_english"], max_sources=report["max_sources"],
        max_extras=report["max_extras"])
    need(calls == list(raw_beams), "stored beam order or use differs")
    stripped_parents = [_strip_support(row) for row in captured.get("parents", [])]
    stripped_proposals = [_strip_support(row) for row in captured.get("proposals", [])]
    need(stripped_parents == replay["parents"]
         and stripped_proposals == replay["proposals"]
         and captured.get("generation_requests") == replay["generation_requests"]
         and captured.get("generated_by_source") == replay["generated_by_source"]
         and captured.get("skipped_known_english") == replay["skipped_known_english"]
         and captured.get("skipped_generation_sources") == replay["skipped_generation_sources"]
         and captured.get("generation_budget_exhausted")
             is replay["generation_budget_exhausted"],
         "shared-helper proposal replay differs")
    statuses = Counter()
    for row in captured["parents"]:
        statuses["parent_" + _verify_support(row, tokenizer, report["max_sentence_tokens"])] += 1
    for row in captured["proposals"]:
        statuses[_verify_support(row, tokenizer, report["max_sentence_tokens"])
                 + "_proposals"] += 1
    need(len(captured["proposals"]) == len(replay["proposals"]),
         "unsupported proposal was dropped")
    return {
        "parents": len(captured["parents"]), "proposals": len(captured["proposals"]),
        "generation_requests": replay["generation_requests"], **dict(statuses),
    }


def verify_resources(directory: Path, report, rows):
    values = [json.loads(line) for line in (directory / "resources.jsonl").open()]
    need(values, "resource evidence is empty")
    expected_stages = ["authenticated", "model-loaded"]
    for row in rows:
        expected_stages.extend(["decode-" + row["id"]] * row["generation_requests"])
        expected_stages.append("row-" + row["id"])
    expected_stages.append("complete")
    need([row.get("stage") for row in values] == expected_stages,
         "resource stage sequence differs")
    previous_seconds = -1.0
    previous_row_seconds = values[1]["seconds"]
    peak_rss = peak_swap = 0
    for row in values:
        need(set(row) == {"stage", "seconds", "rss_bytes", "swap_growth_bytes"}
             and isinstance(row["seconds"], (int, float)) and not isinstance(row["seconds"], bool)
             and math.isfinite(row["seconds"]) and row["seconds"] >= previous_seconds
             and type(row["rss_bytes"]) is int and row["rss_bytes"] >= 0
             and type(row["swap_growth_bytes"]) is int
             and row["seconds"] <= LIMITS["whole_seconds"]
             and row["rss_bytes"] <= LIMITS["rss_bytes"]
             and row["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"],
             "resource observation differs or exceeds limit")
        previous_seconds = row["seconds"]
        peak_rss = max(peak_rss, row["rss_bytes"])
        peak_swap = max(peak_swap, row["swap_growth_bytes"])
        if row["stage"].startswith("row-"):
            need(row["seconds"] - previous_row_seconds <= LIMITS["row_seconds"],
                 "single-row elapsed evidence exceeds limit")
            previous_row_seconds = row["seconds"]
    need(report.get("resources") == values[-1]
         and report.get("max_rss_bytes") == peak_rss
         and report.get("max_swap_growth_bytes") == peak_swap
         and (directory / "capture.jsonl").stat().st_size
             + (directory / "resources.jsonl").stat().st_size <= LIMITS["output_bytes"],
         "resource summary or output bound differs")
    return {"observations": len(values), "max_rss_bytes": peak_rss,
            "max_swap_growth_bytes": peak_swap}


def verify_counts(counts, report_counts, captured_rows):
    need(dict(counts) == report_counts
         and counts["supported_proposals"] + counts["unsupported_proposals"]
             == counts["proposals"]
         and counts["parents"] == sum(len(row["parents"]) for row in captured_rows),
         "capture counters differ or unsupported proposals were dropped")


def verify(directory: Path, tokenizer):
    report, report_sha, capture_sha, resources_sha = authenticate(directory)
    expected = load_expected_sources(report)
    captured_rows = []
    counts = Counter(rows=0, parents=0, proposals=0, supported_proposals=0,
                     unsupported_proposals=0, generation_requests=0)
    with (directory / "capture.jsonl").open() as stream:
        for expected_entry, source in expected:
            line = next(stream, None)
            need(line is not None, "capture is missing a selected FIT row")
            captured = json.loads(line)
            row_counts = verify_row(captured, expected_entry, source, tokenizer, report)
            captured_rows.append(captured)
            counts["rows"] += 1
            for key in ("parents", "proposals", "supported_proposals",
                        "unsupported_proposals", "generation_requests"):
                counts[key] += row_counts.get(key, 0)
        need(next(stream, None) is None, "capture has extra rows")
    verify_counts(counts, report.get("counts"), captured_rows)
    resources = verify_resources(directory, report, captured_rows)
    closing_report, closing_report_sha, closing_capture_sha, closing_resources_sha = authenticate(directory)
    need(closing_report == report and closing_report_sha == report_sha
         and closing_capture_sha == capture_sha and closing_resources_sha == resources_sha,
         "capture, resources, or pins changed during verification")
    return {
        "verified": True, "rows": 13, "counts": dict(counts),
        "report_sha256": report_sha, "capture_sha256": capture_sha,
        "resources_sha256": resources_sha, "pins": report["pins"],
        "resource_recount": resources, "labels_opened": False,
        "sentence_model_inference": False, "neural_beam_inference_replayed": False,
        "replay_scope": ("Shared frozen helper replay from captured generated_by_source beams; "
                         "independent source selection, fallback identity, tokenizer support, "
                         "counter and resource recount. No neural beam reinference."),
        "limitations": [
            "This reuses the proposal helper and therefore is not an independent helper implementation.",
            "Only 13 source-selected FIT rows and their public fallback parents are covered.",
            "Single-span proposals do not test joint spelling repair or sentence ranking.",
            "No labels, first/top5 accuracy, new-document generalization, or production behavior are assessed.",
        ],
        "verifier_sha256": sha(__file__),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_directory", type=Path)
    args = parser.parse_args()
    output = args.capture_directory / "verification.json"
    need(not output.exists(), "verification output already exists")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
    need(tokenizer.num_special_tokens_to_add(pair=True) == 3,
         "sentence tokenizer reserved-token count differs")
    result = verify(args.capture_directory, tokenizer)
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
