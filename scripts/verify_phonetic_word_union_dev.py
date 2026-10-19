#!/usr/bin/env python3
"""Pure independent verifier for the closed phonetic-word-union DEV run."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hanglish.core.phonology import romanize_word


SCHEMA = "phonetic-word-union-dev-independent-v1"
RUN_SCHEMA = "phonetic-word-union-dev-v1"
LIMITS = {"whole_seconds": 1800, "update_seconds": 60,
          "rss_bytes": 3 * 1024**3, "swap_growth_bytes": 512 * 1024**2}
CHECKPOINT_SHA = "f06b0a7cd6c5ccbdfaa376cc29f59e90b3fcf0ae3a157fd2c4f7efaa73b4180d"
CHECKPOINT = ROOT / ".omx/artifacts/observed-phonetic-augmented-training-v1/phonetic_transducer.pt"
PARENT_VERSION = "experimental-ctc-union-1-488fc89732af"
REFERENCE_DIR = ROOT / ".omx/artifacts/ctc-union-runtime-dev-v1"
REFERENCE_REPORT_SHA = "05c15eb14ebfd5cad2381d38898fd10f4ac07578789509bf4e14ea528ab6e9f0"
REFERENCE_CAPTURE_SHA = "ff8b9e2a8708654637d2129a348cd6cb25745bd47871c4939121664a2db4cada"
DEV_LABEL = ROOT / "data/challenge/sentence-v3/dev.jsonl"
ENGLISH = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
HANGUL = re.compile(r"[가-힣]+\Z")
LITERAL_HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]\Z")
SOURCE_HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    values = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                require(isinstance(value, dict), f"invalid row {number}: {path}")
                values.append(value)
    return values


def path_key(path: Path) -> str:
    return str(Path(path).resolve())


def load_reference_authorization() -> dict:
    report_path, capture_path = REFERENCE_DIR / "report.json", REFERENCE_DIR / "capture.jsonl"
    require(sha256_file(report_path) == REFERENCE_REPORT_SHA
            and sha256_file(capture_path) == REFERENCE_CAPTURE_SHA,
            "frozen production reference identity differs")
    report = read_json(report_path)
    require(report.get("completed") is True and report.get("split") == "dev"
            and report.get("input_contract") == "roman_sentence_only"
            and report.get("gold_not_passed_to_predictor") is True
            and report.get("capture_sha256") == REFERENCE_CAPTURE_SHA
            and report.get("proposed", {}).get("counts")
                == {"rows": 316, "top1": 262, "top5": 282, "errors": 0}
            and isinstance(report.get("pins"), dict) and report["pins"]
            and path_key(DEV_LABEL) in report["pins"],
            "frozen production reference contract differs")
    for raw, digest in report["pins"].items():
        path = Path(raw) if Path(raw).is_absolute() else ROOT / raw
        if path.resolve() == DEV_LABEL.resolve():
            continue
        require(path.is_file() and sha256_file(path) == digest,
                f"frozen production pin differs: {raw}")
    return {"report": report, "capture": capture_path, "deferred_label": DEV_LABEL}


def validate_source_reference(row: dict, seen: set[str]) -> tuple[str, str, dict]:
    identifier, request = row.get("id"), row.get("request")
    require(isinstance(identifier, str) and identifier not in seen,
            "reference identity absent or duplicated")
    require(isinstance(request, dict) and set(request) == {"input", "context"}
            and request.get("context") == "", "reference request contract differs")
    source = request.get("input")
    require(isinstance(source, str) and len(source.split()) >= 2
            and SOURCE_HANGUL.search(source) is None,
            "reference source is not a complete Roman sentence")
    parent = row.get("proposed")
    require(isinstance(parent, dict) and parent.get("input") == source
            and isinstance(parent.get("candidates"), list) and parent["candidates"],
            "reference parent response differs")
    seen.add(identifier)
    return identifier, source, parent


def verify_candidate(source: str, candidate: object) -> None:
    require(isinstance(candidate, dict) and isinstance(candidate.get("text"), str),
            "runtime candidate text differs")
    cursor, pieces = 0, []
    for segment in candidate.get("segments", []):
        end = segment.get("input_end")
        require(segment.get("input_start") == cursor and type(end) is int
                and cursor < end <= len(source) and segment.get("source") == source[cursor:end]
                and isinstance(segment.get("text"), str),
                "runtime candidate source coverage differs")
        cursor = end; pieces.append(segment["text"])
    require(cursor == len(source) and "".join(pieces) == candidate["text"],
            "runtime candidate text/source coverage differs")


def verify_runtime_contract(source: str, before: dict, after: dict) -> None:
    candidates = after.get("candidates")
    require(isinstance(candidates, list) and 1 <= len(candidates) <= 5
            and after.get("result") == candidates[0].get("text")
            and source in [candidate.get("text") for candidate in candidates],
            "runtime candidates omit source or selected result")
    for candidate in candidates:
        verify_candidate(source, candidate)
        if candidate.get("score") is None:
            expected = {"input_start": 0, "input_end": len(source), "source": source,
                        "text": source, "kind": "raw"}
            require(candidate.get("text") == source and candidate.get("segments") == [expected],
                    "runtime canonical raw candidate differs")
    if source not in [candidate["text"] for candidate in candidates[:4]]:
        require(candidates[-1]["text"] == source, "runtime raw candidate slot differs")
    info = after.get("ctc_word", {})
    names = ("generation_requests", "completed_generations",
             "generated_proposals", "accepted_proposals")
    values = [info.get(name) for name in names]
    require(all(type(value) is int and value >= 0 for value in values),
            "invalid CTC runtime counters")
    requested, complete, generated, accepted = values
    ranked = before.get("sentence_ranking_applied") is True
    require(after.get("sentence_ranking_applied") == ranked,
            "CTC runtime parent ranking gate differs")
    reason = ("unsupported_source" if len(source.split()) < 2 or SOURCE_HANGUL.search(source)
              else "source_token_limit" if len(source.encode("utf-8")) + 2 > 512 else None)
    eligible = ranked and reason is None
    prediction = info.get("prediction")
    require(requested == int(eligible) and 0 <= accepted <= generated <= 96,
            "CTC generation bounds differ")
    if eligible:
        require(isinstance(prediction, dict) and prediction.get("reason") in {
            "complete", "empty_output", "invalid_utf8", "invalid_jamo", "no_valid_path"}
            and complete == int(prediction["reason"] == "complete"),
            "CTC eligible prediction differs")
    else:
        require(not complete and not generated and not accepted
                and ((ranked and isinstance(prediction, dict)
                      and prediction.get("reason") == reason)
                     or (not ranked and prediction is None)),
                "CTC ineligible prediction differs")
    require(complete or (not generated and not accepted),
            "incomplete CTC output supplied proposals")
    elapsed = info.get("generation_ms")
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
            "CTC generation timing differs")


def validate_offset_row(source: str, row: object, *, generated: bool) -> None:
    required = {"source_start", "source_end", "source"}
    if generated:
        required |= {"text", "replacement", "parent_index"}
    require(isinstance(row, dict) and set(row) == required,
            "phonetic proposal audit schema differs")
    start, end = row["source_start"], row["source_end"]
    require(type(start) is int and type(end) is int and 0 <= start < end <= len(source)
            and row["source"] == source[start:end],
            "phonetic proposal source offset differs")
    if generated:
        require(type(row["parent_index"]) is int and row["parent_index"] >= 0
                and isinstance(row["text"], str) and row["text"]
                and isinstance(row["replacement"], str)
                and HANGUL.fullmatch(row["replacement"]) is not None
                and romanize_word(row["replacement"]) == row["source"],
                "phonetic generated proposal differs")


def validate_union_trace(source: str, parent: dict, proposed: dict,
                         parent_pool: list[str], pins: dict[str, str]) -> None:
    verify_runtime_contract(source, parent, proposed)
    for response in (parent, proposed):
        candidates = response.get("candidates")
        require(response.get("input") == source and response.get("context") == ""
                and response.get("external_context_used") is False
                and isinstance(candidates, list) and candidates
                and response.get("result") == candidates[0].get("text"),
                "runtime source-only response differs")
        for candidate in candidates:
            verify_candidate(source, candidate)
    require(parent.get("version") == PARENT_VERSION, "parent runtime version differs")
    trace = proposed.get("phonetic_word_union")
    require(isinstance(trace, dict) and trace.get("experimental") is True
            and trace.get("parent_version") == PARENT_VERSION,
            "phonetic union identity differs")
    for name in ("ctc_word", "ctc_union"):
        before, after = dict(parent.get(name, {})), dict(proposed.get(name, {}))
        before.pop("generation_ms", None); after.pop("generation_ms", None)
        require(before == after, f"phonetic union changed {name}")
    require(isinstance(parent_pool, list) and len(parent_pool) == len(set(parent_pool))
            and trace.get("production_pool_texts") == parent_pool,
            "production pool trace differs")
    generated_rows = trace.get("generated_extras")
    require(isinstance(generated_rows, list) and len(generated_rows) <= 96,
            "generated extras differ")
    for row in generated_rows:
        validate_offset_row(source, row, generated=True)
        require(row["parent_index"] < len(parent_pool), "proposal parent index differs")
    generated = [row["text"] for row in generated_rows]
    require(len(generated) == len(set(generated)) and not set(generated) & set(parent_pool),
            "generated extras duplicate production paths")
    for name in ("skipped_known_english", "skipped_generation_sources"):
        rows = trace.get(name)
        require(isinstance(rows, list), f"{name} trace differs")
        for row in rows:
            validate_offset_row(source, row, generated=False)
    scoreable, union = trace.get("scoreable_extras"), trace.get("union")
    require(isinstance(scoreable, list) and isinstance(union, list)
            and len(scoreable) == len(set(scoreable)) and set(scoreable) <= set(generated)
            and len(union) == len(set(union))
            and set(union) == set(parent_pool) | set(scoreable),
            "scoreable union differs")
    public = [candidate["text"] for candidate in proposed["candidates"]]
    require(trace.get("public") == public
            and trace.get("selected_extras") == [text for text in generated if text in public]
            and trace.get("missing_extras") == [text for text in generated if text not in public],
            "public phonetic trace differs")
    requests = trace.get("generation_requests")
    require(type(requests) is int and 0 <= requests <= 96
            and type(trace.get("generation_budget_exhausted")) is bool,
            "phonetic generation budget differs")
    artifacts = trace.get("artifacts")
    require(isinstance(artifacts, dict) and artifacts
            and all(pins.get(str((ROOT / path).resolve())) == digest
                    for path, digest in artifacts.items()),
            "phonetic response artifacts are not pinned")
    if proposed.get("sentence_ranking_applied"):
        require(set(public) <= set(union) | {source}, "public candidate outside union")
    else:
        require(not generated and not parent_pool and not union,
                "unranked source unexpectedly generated candidates")


def literal_signature(text: str) -> str:
    return "".join(character for character in text
                   if not ("a" <= character <= "z")
                   and LITERAL_HANGUL.fullmatch(character) is None)


def response_metrics(rows: list[dict], labels: list[dict], branch: str) -> dict:
    totals, categories = Counter(), defaultdict(Counter)
    outcomes = []
    for row, label in zip(rows, labels):
        candidates = row[branch].get("candidates")
        texts = [candidate["text"] for candidate in candidates[:5]]
        expected = set(label["expected"])
        top1, top5 = bool(texts) and texts[0] in expected, bool(expected & set(texts))
        counts = {"rows": 1, "top1": int(top1), "top5": int(top5), "errors": 0}
        totals.update(counts); categories[label.get("category", "uncategorized")].update(counts)
        outcomes.append({"id": row["id"], "top1": top1, "top5": top5})
    return {"counts": dict(totals),
            "categories": {key: dict(value) for key, value in categories.items()},
            "passed": bool(rows) and totals["top1"] == len(rows), "outcomes": outcomes}


def aggregate(rows: list[dict], labels: list[dict]) -> dict:
    require(len(rows) == len(labels)
            and [row["id"] for row in rows] == [label.get("id") for label in labels],
            "capture/label identities differ")
    parent = response_metrics(rows, labels, "parent")
    proposed = response_metrics(rows, labels, "proposed")
    changes = Counter(); stages = Counter(); english = Counter()
    for row, label, before, after in zip(
            rows, labels, parent["outcomes"], proposed["outcomes"]):
        require(row["request"] == {"input": label.get("input"), "context": ""}
                and label.get("context", "") == "" and label.get("split") == "dev",
                "late DEV label source differs")
        changes["corrections"] += not before["top1"] and after["top1"]
        changes["regressions"] += before["top1"] and not after["top1"]
        changes["top5_losses"] += before["top5"] and not after["top5"]
        gold = set(label["expected"]); trace = row["proposed"]["phonetic_word_union"]
        parent_pool = set(row["parent_pool"])
        pools = {"parent_pool": parent_pool,
                 "with_generated": parent_pool | {item["text"] for item in trace["generated_extras"]},
                 "scoreable_union": set(trace["union"]),
                 "public_top5": set(trace["public"][:5]),
                 "selected": {row["proposed"]["result"]}}
        ranked = row["proposed"].get("sentence_ranking_applied") is True
        if ranked:
            stages["ranked_rows"] += 1
            for name, values in pools.items():
                stages[name + "_correct"] += bool(gold & values)
            stages["marginal_generated_gold_rows"] += (
                not bool(gold & parent_pool) and bool(gold & pools["with_generated"]))
            stages["generated_to_scoreable_gold_losses"] += (
                bool(gold & pools["with_generated"]) and not bool(gold & pools["scoreable_union"]))
            stages["union_to_public_gold_losses"] += (
                bool(gold & pools["scoreable_union"]) and not bool(gold & pools["public_top5"]))
        stages["rows"] += 1
        stages["generated_extras"] += len(trace["generated_extras"])
        stages["scoreable_extras"] += len(trace["scoreable_extras"])
        stages["generation_budget_exhausted_rows"] += trace["generation_budget_exhausted"]
        stages["skipped_known_english_spans"] += len(trace["skipped_known_english"])
        source_signature = literal_signature(row["request"]["input"])
        english["literal_signature_regressions"] += (
            literal_signature(row["parent"]["result"]) == source_signature
            and literal_signature(row["proposed"]["result"]) != source_signature)
        signatures = {tuple(ENGLISH.findall(text)) for text in gold}
        if any(signatures):
            english["rows_with_reference_ascii_words"] += 1
            parent_match = tuple(ENGLISH.findall(row["parent"]["result"])) in signatures
            proposed_match = tuple(ENGLISH.findall(row["proposed"]["result"])) in signatures
            english["parent_exact_signature"] += parent_match
            english["proposed_exact_signature"] += proposed_match
            english["signature_regressions"] += parent_match and not proposed_match
    improvement = bool(changes["corrections"] and not changes["regressions"]
                       and not changes["top5_losses"])
    quality = bool(improvement and not english["signature_regressions"]
                   and not english["literal_signature_regressions"])
    return {"parent": {key: value for key, value in parent.items() if key != "outcomes"},
            "proposed": {key: value for key, value in proposed.items() if key != "outcomes"},
            "corrections": changes["corrections"], "regressions": changes["regressions"],
            "top5_losses": changes["top5_losses"], "improvement_gate": improvement,
            "quality_gate": quality, "stage_counts": dict(stages),
            "english_ascii_signature": dict(english),
            "english_scope": "Exact case-sensitive ASCII word sequence on references containing ASCII words; not all English-copy behavior."}


def latency(rows: list[dict]) -> dict:
    result = {}
    for branch in ("parent", "proposed"):
        values = sorted(row[branch].get("engine_ms") for row in rows)
        require(values and all(type(value) in (int, float) and math.isfinite(value)
                               and value >= 0 for value in values), "latency differs")
        result[branch] = {
            name: values[max(0, math.ceil(quantile * len(values)) - 1)]
            for name, quantile in (("p50_ms", .5), ("p95_ms", .95), ("max_ms", 1.0))}
    return result


def resource_row_valid(row: object) -> bool:
    return (isinstance(row, dict) and set(row) == {
        "at", "elapsed_s", "operation", "operation_elapsed_s", "rss_bytes", "stage",
        "swap_growth_bytes", "swap_used_bytes"}
        and type(row["elapsed_s"]) in (int, float) and math.isfinite(row["elapsed_s"])
        and 0 <= row["elapsed_s"] <= LIMITS["whole_seconds"]
        and type(row["rss_bytes"]) is int and 0 <= row["rss_bytes"] <= LIMITS["rss_bytes"]
        and type(row["swap_growth_bytes"]) is int
        and row["swap_growth_bytes"] <= LIMITS["swap_growth_bytes"]
        and type(row["swap_used_bytes"]) is int and row["swap_used_bytes"] >= 0)


def verify_resources(rows: list[dict], operations: list[str]) -> dict:
    require(rows, "resource trace empty")
    current, starts, ends, terminal, after_completion = None, [], [], None, None
    previous, baseline = -1.0, None
    for row in rows:
        require(resource_row_valid(row), "resource row invalid")
        datetime.fromisoformat(row["at"])
        require(row["elapsed_s"] >= previous, "resource elapsed regressed")
        previous = row["elapsed_s"]
        row_baseline = row["swap_used_bytes"] - row["swap_growth_bytes"]
        require(row_baseline >= 0, "resource swap baseline invalid")
        baseline = row_baseline if baseline is None else baseline
        require(row_baseline == baseline, "resource swap baseline changed")
        if row["stage"] == "operation_start":
            require(current is None and isinstance(row["operation"], str),
                    "resource operation start invalid")
            current = row["operation"]; starts.append(current)
        elif row["stage"] == "operation_end":
            require(current == row["operation"]
                    and type(row["operation_elapsed_s"]) in (int, float)
                    and math.isfinite(row["operation_elapsed_s"])
                    and 0 <= row["operation_elapsed_s"] <= LIMITS["update_seconds"],
                    "resource operation end invalid")
            ends.append(current); current = None
        elif row["stage"] == "watchdog":
            require(row["operation"] in (None, current), "watchdog operation differs")
        elif row["stage"] == "after_report":
            require(terminal is None and current is None and row["operation"] is None
                    and row["operation_elapsed_s"] is None and starts == operations[:-1],
                    "terminal resource differs")
            terminal = row
        elif row["stage"] == "after_completion":
            require(terminal is not None and after_completion is None and current is None
                    and row["operation"] is None and row["operation_elapsed_s"] is None
                    and starts == operations,
                    "after-completion resource differs")
            after_completion = row
        else:
            raise ValueError(f"unexpected resource stage: {row['stage']}")
    require(starts == operations == ends and terminal is not None and after_completion is not None,
            "resource operation sequence differs")
    return {"rows": len(rows), "operations": len(starts), "final": terminal,
            "after_completion": after_completion,
            "baseline_swap_bytes": baseline,
            "peak_rss_bytes": max(row["rss_bytes"] for row in rows),
            "peak_swap_growth_bytes": max(row["swap_growth_bytes"] for row in rows)}


def require_no_failures(directory: Path) -> None:
    for name in ("failure.json", "failure-report.json", "resource-watchdog-failure.json"):
        require(not (Path(directory) / name).exists(), f"failure artifact exists: {name}")


def verify(directory: Path) -> dict:
    directory = Path(directory); proof_path = directory / "independent-verification.json"
    require(directory.is_dir() and not proof_path.exists(), "run/proof state invalid")
    require_no_failures(directory)
    authorization = load_reference_authorization()
    references = read_jsonl(authorization["capture"])
    seen = set(); requests = [validate_source_reference(row, seen) for row in references]
    require(len(requests) == 316, "DEV source count differs")
    captures = read_jsonl(directory / "capture.jsonl")
    require(len(captures) == len(requests), "capture count differs")
    report, completion = read_json(directory / "report.json"), read_json(directory / "completion.json")
    pins = read_json(directory / "pins.json")
    require(report.get("pins") == pins and isinstance(pins, dict) and pins,
            "report pins differ")
    forbidden = ("/test.jsonl", "document-reserve")
    for raw, digest in pins.items():
        require(not any(token in raw for token in forbidden), "forbidden TEST/reserve pin")
        path = Path(raw) if Path(raw).is_absolute() else ROOT / raw
        if path.resolve() == DEV_LABEL.resolve():
            continue
        require(path.is_file() and sha256_file(path) == digest, f"source pin differs: {raw}")
    training = CHECKPOINT.parent
    training_proof = read_json(training / "independent-verification.json")
    require(sha256_file(CHECKPOINT) == CHECKPOINT_SHA
            and pins.get(str(CHECKPOINT.resolve())) == CHECKPOINT_SHA
            and training_proof.get("verified") is True
            and training_proof.get("checkpoint_sha256") == CHECKPOINT_SHA
            and training_proof.get("report_sha256") == sha256_file(training / "report.json"),
            "phonetic checkpoint/training proof differs")
    for (identifier, source, frozen_parent), row in zip(requests, captures):
        require(set(row) == {"id", "request", "parent", "proposed", "parent_pool"}
                and row["id"] == identifier
                and row["request"] == {"input": source, "context": ""}
                and row["parent"].get("result") == frozen_parent["result"]
                and row["parent"].get("candidates") == frozen_parent["candidates"],
                "capture identity or production parity differs")
        validate_union_trace(source, row["parent"], row["proposed"], row["parent_pool"], pins)
    capture_sha = sha256_file(directory / "capture.jsonl")
    closure = read_json(directory / "capture.complete.json")
    require(closure == {"schema": RUN_SCHEMA + "-closure", "rows": len(captures),
                        "capture_sha256": capture_sha, "labels_opened": False},
            "source-only closure differs")
    require(pins.get(path_key(DEV_LABEL)) == sha256_file(DEV_LABEL),
            "late DEV label pin differs")
    labels = read_jsonl(authorization["deferred_label"])
    metrics = aggregate(captures, labels)
    require(report.get("schema") == RUN_SCHEMA and report.get("completed") is True
            and report.get("split") == "dev" and report.get("input_contract") == "roman_sentence_only"
            and report.get("capture_sha256") == capture_sha
            and report.get("closure_sha256") == sha256_file(directory / "capture.complete.json")
            and report.get("limits") == LIMITS and report.get("parent_version") == PARENT_VERSION
            and report.get("proposed_version")
                == captures[0]["proposed"].get("version")
            and all(row["proposed"].get("version") == report.get("proposed_version")
                    for row in captures)
            and report.get("timing_scope") == "paired serial engines sharing frozen models"
            and report.get("production_changed") is False and report.get("test_accessed") is False
            and report.get("labels_opened_after_capture") is True
            and report.get("gold_not_passed_to_predictor") is True,
            "DEV report contract differs")
    require(metrics["parent"]["counts"] == {"rows": 316, "top1": 262, "top5": 282, "errors": 0},
            "frozen production metrics differ")
    for key, expected in metrics.items():
        require(report.get(key) == expected, f"reported metric differs: {key}")
    require(report.get("latency") == latency(captures), "reported latency differs")
    operations = ["authenticate_inputs", "load_models"]
    operations += [f"convert:{identifier}" for identifier, _, _ in requests]
    operations += ["late_label_join", "write_report", "write_completion"]
    resources = verify_resources(read_jsonl(directory / "resources.jsonl"), operations)
    require(completion == {"completed": True,
                           "report_sha256": sha256_file(directory / "report.json"),
                           "resource_after_report": resources["final"]},
            "completion/resource closure differs")
    proof = {
        "schema": SCHEMA, "verified": True,
        "report_sha256": sha256_file(directory / "report.json"),
        "completion_sha256": sha256_file(directory / "completion.json"),
        "capture_sha256": capture_sha,
        "closure_sha256": sha256_file(directory / "capture.complete.json"),
        "pins_sha256": sha256_file(directory / "pins.json"),
        "checkpoint_sha256": CHECKPOINT_SHA,
        "verifier_sha256": sha256_file(Path(__file__)),
        "counts": {"rows": len(captures), "operations": resources["operations"]},
        "metrics": metrics, "resources": resources, "limits": LIMITS,
        "contracts_verified": {
            "frozen_production_identity_order_and_candidate_parity": True,
            "source_only_context_and_candidate_coverage": True,
            "production_pool_extra_union_and_public_traces": True,
            "independent_quality_stage_ascii_and_literal_metrics": True,
            "capture_closed_before_dev_labels": True,
            "pins_completion_and_atomic_resources": True,
            "failure_artifacts_absent": True,
            "test_or_reserve_accessed": False,
            "neural_model_loaded_or_run_by_verifier": False,
        },
        "limitations": [
            "The verifier authenticates recorded inference and does not rerun either model.",
            "ASCII signatures are a bounded diagnostic, not proof of every English-copy behavior.",
            "DEV evidence is not production approval.",
        ],
    }
    proof_path.write_bytes(canonical(proof) + b"\n")
    return proof


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(); value.add_argument("--run", type=Path, required=True)
    return value


if __name__ == "__main__":
    args = parser().parse_args(); print(json.dumps(verify(args.run), sort_keys=True))
