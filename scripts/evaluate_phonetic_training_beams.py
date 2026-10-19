"""Evaluate a closed source-only phonetic beam census against natural TRAIN labels."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.neural_converter import LETTERS, PROTECTED, TOKENS
from hanglish.core.phonology import romanize_word


CENSUS = ROOT / ".omx/artifacts/phonetic-training-beam-census-v1"
MECHANICAL = ROOT / ".omx/artifacts/multistep-morphology-coverage-v2/source-capture.jsonl"
MECHANICAL_REPORT = ROOT / ".omx/artifacts/multistep-morphology-coverage-v2/report.json"
TARGETS = ROOT / "data/natural_context/final-choice-v1/natural_fit/cases.jsonl"
OUTPUT = ROOT / ".omx/artifacts/phonetic-training-beam-evaluation-v1.json"

CENSUS_REPORT_SHA = "c4d517d7ef913dab595f1a11630fa9f771b84b1baa8e7deb24eeba11e9e686fa"
MECHANICAL_SHA = "472a5d7ebcb376e89d040583264d5b2d2b39204500499e0cf8d6b6c738b52adb"
MECHANICAL_REPORT_SHA = "644fd473bf5cb3ff27c745dfc2d9da4389f6a0f55592dca8fe760e074140d4d4"
TARGETS_SHA = "789e45a3b6028e7416ca3b82739c906c558ec414320796c53b0b76ed251b9a18"
EXPECTED_ROWS = 2048
EXPECTED_INPUTS = 18122
ORDERED_INPUTS_SHA = "d13845be042709a516f8f8540a5597be613988c8f8c9424f797dbe1138a3ea00"
PILOT_ROWS = 256
BEAM_CUTOFFS = (1, 4, 8)
HANGUL_RUN = re.compile(r"[가-힣]+")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()


def digest_json(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_jsonl(path):
    return parse_jsonl(Path(path).read_bytes())


def parse_jsonl(raw):
    return [json.loads(line) for line in raw.splitlines() if line]


def verify_pins(pins):
    require(isinstance(pins, dict) and pins, "source pins missing")
    for name, expected in pins.items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        require(path.is_file() and sha(path) == expected,
                f"source pin differs: {name}")


def eligible_occurrences(text):
    """Reapply frozen production token boundaries without invoking a generator."""
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


def quantile_indices(size, count):
    count = min(size, count)
    if count <= 0:
        return ()
    if count == 1:
        return (0,)
    values = tuple(index * (size - 1) // (count - 1) for index in range(count))
    require(len(set(values)) == count, "selection quantiles are not unique")
    return values


def independent_selection(source_rows, *, pilot_rows=PILOT_ROWS):
    membership = {}
    protected_membership = {}
    unprotected_membership = {}
    positions = Counter()
    for row in source_rows:
        for source, start, end, protected in eligible_occurrences(row["input"]):
            membership.setdefault(source, set()).add(row["id"])
            selected = protected_membership if protected else unprotected_membership
            selected.setdefault(source, set()).add(row["id"])
            positions[(row["id"], source, start, end, protected)] += 1
    base = sorted(membership, key=lambda source: (len(source), source))
    pilots = [base[index] for index in quantile_indices(len(base), pilot_rows)]
    pilot_set = set(pilots)
    ordered = pilots + [source for source in base if source not in pilot_set]
    selection = [{"index": index, "input": source,
                  "row_ids": sorted(membership[source]),
                  "protected_row_ids": sorted(protected_membership.get(source, ())),
                  "unprotected_row_ids": sorted(unprotected_membership.get(source, ()))}
                 for index, source in enumerate(ordered)]
    return selection, positions


def independent_selection_summary(source_rows, selection):
    fragmented = sum(len(LETTERS.findall(row["input"])) > 128 for row in source_rows)
    eligible_rows = sum(bool(eligible_occurrences(row["input"])) for row in source_rows)
    return {
        "source_rows": len(source_rows),
        "rows_with_eligible_inputs": eligible_rows,
        "rows_without_eligible_inputs": len(source_rows) - eligible_rows,
        "fragmented_rows_bypassed": fragmented,
        "unique_inputs": len(selection),
        "min_input_length": min((len(row["input"]) for row in selection), default=None),
        "max_input_length": max((len(row["input"]) for row in selection), default=None),
        "protected_only_inputs": sum(bool(row["protected_row_ids"])
                                     and not row["unprotected_row_ids"]
                                     for row in selection),
        "mixed_protection_inputs": sum(bool(row["protected_row_ids"])
                                       and bool(row["unprotected_row_ids"])
                                       for row in selection),
        "unprotected_only_inputs": sum(bool(row["unprotected_row_ids"])
                                       and not row["protected_row_ids"]
                                       for row in selection),
        "protected_row_memberships": sum(len(row["protected_row_ids"])
                                          for row in selection),
        "unprotected_row_memberships": sum(len(row["unprotected_row_ids"])
                                            for row in selection),
    }


def validate_mechanical_rows(rows):
    require(rows and len({row.get("id") for row in rows}) == len(rows),
            "mechanical source rows are incomplete or duplicated")
    for row in rows:
        require(set(row) == {"id", "input", "pieces"}
                and isinstance(row["id"], str) and isinstance(row["input"], str)
                and isinstance(row["pieces"], list), "mechanical row schema differs")
        cursor = 0
        for piece in row["pieces"]:
            require(piece.get("start") == cursor
                    and piece.get("end") == cursor + len(piece.get("source", ""))
                    and row["input"][piece["start"]:piece["end"]] == piece["source"]
                    and isinstance(piece.get("baseline"), list) and piece["baseline"],
                    "mechanical source piece boundary differs")
            require(all(isinstance(option, dict) and isinstance(option.get("text"), str)
                        for option in piece["baseline"]),
                    "mechanical baseline option differs")
            cursor = piece["end"]
        require(cursor == len(row["input"]), "mechanical pieces do not reproduce input")


def validate_beam_outputs(source, outputs, limit=8):
    require(isinstance(outputs, list) and len(outputs) <= limit,
            "beam output bound differs")
    seen = set()
    previous_score = float("inf")
    for output in outputs:
        text, score = output.get("text"), output.get("score")
        require(set(output) == {"text", "score"}
                and isinstance(text, str) and text
                and all("가" <= char <= "힣" for char in text)
                and text not in seen and romanize_word(text) == source,
                "beam output is not unique forward-valid Hangul")
        require(type(score) in (int, float) and math.isfinite(score),
                "beam score is not finite")
        require(score <= previous_score, "beam scores are not non-increasing")
        seen.add(text)
        previous_score = score


def validate_closed_capture(census=CENSUS, *, report_sha=CENSUS_REPORT_SHA,
                            mechanical=MECHANICAL, mechanical_sha=MECHANICAL_SHA,
                            mechanical_report=MECHANICAL_REPORT,
                            mechanical_report_sha=MECHANICAL_REPORT_SHA,
                            expected_rows=EXPECTED_ROWS, expected_inputs=EXPECTED_INPUTS,
                            pilot_rows=PILOT_ROWS):
    """Close all source-only evidence before any TRAIN target file is opened."""
    census, mechanical = Path(census), Path(mechanical)
    report_path = census / "report.json"
    report_raw = report_path.read_bytes() if report_path.is_file() else b""
    require(report_sha != "PENDING_AFTER_CLOSED_CAPTURE"
            and hashlib.sha256(report_raw).hexdigest() == report_sha,
            "closed census report hash differs")
    require(not (census / "failure.json").exists(), "failed census cannot be evaluated")
    mechanical_raw = mechanical.read_bytes()
    mechanical_report_raw = Path(mechanical_report).read_bytes()
    require(hashlib.sha256(mechanical_raw).hexdigest() == mechanical_sha
            and hashlib.sha256(mechanical_report_raw).hexdigest() == mechanical_report_sha,
            "frozen mechanical capture differs")
    report = json.loads(report_raw)
    require(report.get("schema") == "phonetic-training-beam-census-v1"
            and report.get("completed") is True
            and report.get("source_role") == "natural_fit"
            and report.get("source_rows") == expected_rows
            and report.get("unique_inputs") == expected_inputs
            and report.get("captured_inputs") == expected_inputs
            and report.get("target_fields_read") is False
            and report.get("model_inference_run") is True
            and report.get("training_run") is False
            and report.get("source_hashes_unchanged") is True
            and report.get("profile", {}).get("beam_size") == 8
            and report.get("selection_order") == (
                "256 evenly spaced quantiles of (length,input), including endpoints; "
                "then remaining (length,input) order")
            and report.get("validation") == {
                "all_selected_inputs_captured_once": True,
                "outputs_are_unique_forward_valid_hangul": True,
                "scores_are_finite_non_increasing": True,
                "maximum_outputs_per_input": 8,
            },
            "census is incomplete or outside scope")
    pins = report.get("source_hashes_start")
    require(pins == report.get("source_hashes_end"), "census source pins changed")
    verify_pins(pins)

    selection_path, capture_path = census / "selection.jsonl", census / "capture.jsonl"
    selection_raw, capture_raw = selection_path.read_bytes(), capture_path.read_bytes()
    require(hashlib.sha256(selection_raw).hexdigest() == report.get("selection_sha256")
            and hashlib.sha256(capture_raw).hexdigest() == report.get("capture_sha256"),
            "census payload hash differs")
    selection, captures, source_rows = (parse_jsonl(selection_raw),
                                        parse_jsonl(capture_raw),
                                        parse_jsonl(mechanical_raw))
    require(len(source_rows) == expected_rows, "mechanical row denominator differs")
    validate_mechanical_rows(source_rows)
    expected_selection, positions = independent_selection(source_rows, pilot_rows=pilot_rows)
    require(len(expected_selection) == expected_inputs and selection == expected_selection,
            "selection membership or order differs")
    selection_summary = independent_selection_summary(source_rows, selection)
    require(report.get("selection") == selection_summary,
            "independent source eligibility counts differ")
    require(report.get("ordered_inputs_sha256")
            == digest_json([row["input"] for row in expected_selection]),
            "ordered source-input hash differs")
    if expected_rows == EXPECTED_ROWS and expected_inputs == EXPECTED_INPUTS:
        require(report.get("ordered_inputs_sha256") == ORDERED_INPUTS_SHA,
                "fixed source-input census differs")
    require(len(captures) == expected_inputs, "beam capture denominator differs")
    output_count = empty_count = 0
    by_source = {}
    for selected, captured in zip(selection, captures):
        require(set(captured) == {"index", "input", "outputs"}
                and captured["index"] == selected["index"]
                and captured["input"] == selected["input"],
                "beam capture does not match selection row")
        validate_beam_outputs(captured["input"], captured["outputs"])
        require(captured["input"] not in by_source, "beam source is duplicated")
        by_source[captured["input"]] = {
            "outputs": captured["outputs"],
            "protected_row_ids": frozenset(selected["protected_row_ids"]),
            "unprotected_row_ids": frozenset(selected["unprotected_row_ids"]),
        }
        output_count += len(captured["outputs"])
        empty_count += not captured["outputs"]
    require(report.get("outputs") == output_count
            and report.get("empty_outputs") == empty_count,
            "census output counts differ")

    require(all(count == 1 for count in positions.values()),
            "eligible input position is duplicated")
    return {"report": report, "source_rows": source_rows,
            "selection": selection, "beams": by_source,
            "eligible_occurrences": len(positions),
            "selection_summary": selection_summary}


def composable(target, groups):
    positions = {0}
    for group in groups:
        positions = {offset + len(text) for offset in positions for text in group
                     if target.startswith(text, offset)}
        if not positions:
            return False
    return len(target) in positions


def forward_alignment(source, target):
    """Map complete Hangul runs to source offsets; preserve all other text literally."""
    cursor = 0
    runs = []
    previous = 0
    for match in HANGUL_RUN.finditer(target):
        literal = target[previous:match.start()]
        if not source.startswith(literal, cursor):
            return None
        cursor += len(literal)
        roman = romanize_word(match.group())
        if not source.startswith(roman, cursor):
            return None
        runs.append({"text": match.group(), "start": cursor,
                     "end": cursor + len(roman)})
        cursor += len(roman)
        previous = match.end()
    literal = target[previous:]
    if source[cursor:] != literal:
        return None
    return runs


def unprotected_piece_eligible(row, piece):
    return any(source == piece["source"] and start == piece["start"]
               and end == piece["end"] and not protected
               for source, start, end, protected in eligible_occurrences(row["input"]))


def chunk_piece_indices(run, pieces, beams, row):
    overlapping = [index for index, piece in enumerate(pieces)
                   if piece["end"] > run["start"] and piece["start"] < run["end"]]
    if (not overlapping or pieces[overlapping[0]]["start"] != run["start"]
            or pieces[overlapping[-1]]["end"] != run["end"]):
        return None, "partial_source_piece"
    if any(pieces[index]["source"] not in beams
           or not unprotected_piece_eligible(row, pieces[index])
           for index in overlapping):
        return None, "ineligible_source_piece"
    return overlapping, None


def option_groups(pieces, beams, cutoff, row=None):
    groups = []
    for piece in pieces:
        baseline = [option["text"] for option in piece["baseline"]]
        captured = beams.get(piece["source"])
        added = [] if (not captured or row is None
                       or not unprotected_piece_eligible(row, piece)) else [
            output["text"] for output in captured["outputs"][:cutoff]]
        groups.append(tuple(dict.fromkeys((*baseline, *added))))
    return groups


def load_targets(path=TARGETS, expected_sha=TARGETS_SHA, expected_rows=EXPECTED_ROWS):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha,
            "natural TRAIN target hash differs")
    rows = parse_jsonl(raw)
    require(len(rows) == expected_rows and len({row.get("id") for row in rows}) == expected_rows,
            "natural TRAIN targets are incomplete")
    require(all(isinstance(row.get("expected"), list) and row["expected"]
                and all(isinstance(value, str) for value in row["expected"])
                for row in rows), "accepted target lists differ")
    return rows


def evaluate_metrics(source_rows, beams, target_rows):
    require([row["id"] for row in source_rows] == [row["id"] for row in target_rows],
            "source and target row order differs")
    counts = Counter({
        "rows": len(source_rows),
        "top8_gain_yes__saved_finite_yes": 0,
        "top8_gain_yes__saved_finite_no": 0,
        "top8_gain_no__saved_finite_yes": 0,
        "top8_gain_no__saved_finite_no": 0,
    })
    exclusions = Counter({key: 0 for key in (
        "accepted_target_forward_mismatch", "accepted_target_without_hangul",
        "partial_source_piece", "ineligible_source_piece",
        "no_exact_eligible_source_span")})
    for source, target_row in zip(source_rows, target_rows):
        require(source["input"] == target_row.get("input"), "source input differs from target row")
        expected = target_row["expected"]
        pieces = source["pieces"]
        occurrences = eligible_occurrences(source["input"])
        counts["fragmented_rows_bypassed"] += len(LETTERS.findall(source["input"])) > 128
        counts["rows_without_captured_inputs"] += not occurrences
        baseline_groups = option_groups(pieces, {}, 0)
        baseline_full = any(composable(target, baseline_groups) for target in expected)
        counts["mechanical_full_rows"] += baseline_full
        unions = {}
        for cutoff in BEAM_CUTOFFS:
            union = any(composable(target, option_groups(
                            pieces, beams, cutoff, source))
                        for target in expected)
            unions[cutoff] = union
            counts[f"union_top{cutoff}_full_rows"] += union
            counts[f"union_top{cutoff}_full_gain_rows"] += union and not baseline_full

        finite = target_row.get("finite_indices", [])
        candidates = target_row.get("candidates", [])
        require(isinstance(finite, list) and len(set(finite)) == len(finite)
                and all(type(index) is int and 0 <= index < len(candidates) for index in finite),
                "saved finite candidate indices differ")
        saved = any(candidates[index].get("text") in expected for index in finite)
        counts["saved_finite_full_rows"] += saved
        top8_gain = unions[8] and not baseline_full
        counts["union_top8_gain_already_saved_finite"] += top8_gain and saved
        counts["union_top8_gain_absent_from_saved_finite"] += top8_gain and not saved
        counts[("top8_gain_yes" if top8_gain else "top8_gain_no")
               + ("__saved_finite_yes" if saved else "__saved_finite_no")] += 1

        chunks = {}
        for target in expected:
            aligned = forward_alignment(source["input"], target)
            if aligned is None:
                exclusions["accepted_target_forward_mismatch"] += 1
                continue
            counts["forward_aligned_accepted_targets"] += 1
            if not aligned:
                exclusions["accepted_target_without_hangul"] += 1
            for run in aligned:
                chunks.setdefault((run["start"], run["end"]), set()).add(run["text"])
        for (start, end), accepted_texts in chunks.items():
            run = {"start": start, "end": end}
            counts["accepted_hangul_forms"] += len(accepted_texts)
            occurrence = next((item for item in occurrences
                               if item[1] == run["start"] and item[2] == run["end"]), None)
            if occurrence is None:
                exclusions["no_exact_eligible_source_span"] += 1
            else:
                captured = beams[occurrence[0]]
                direct = [output["text"] for output in captured["outputs"]]
                protection = "protected" if occurrence[3] else "unprotected"
                counts[f"direct_{protection}_pure_hangul_chunks"] += 1
                for cutoff in BEAM_CUTOFFS:
                    counts[f"direct_{protection}_beam_top{cutoff}_covered"] += (
                        bool(accepted_texts.intersection(direct[:cutoff])))

            selected, reason = chunk_piece_indices(run, pieces, beams, source)
            if reason:
                exclusions[reason] += 1
                continue
            counts["pure_hangul_chunks"] += 1
            baseline = [[option["text"] for option in pieces[index]["baseline"]]
                        for index in selected]
            mechanical = any(composable(text, baseline) for text in accepted_texts)
            counts["mechanical_chunk_covered"] += mechanical
            for cutoff in BEAM_CUTOFFS:
                beam_only = [[output["text"] for output in
                              beams[pieces[index]["source"]]["outputs"][:cutoff]]
                             for index in selected]
                union = [tuple(dict.fromkeys((*base, *extra)))
                         for base, extra in zip(baseline, beam_only)]
                beam_covered = any(composable(text, beam_only) for text in accepted_texts)
                union_covered = any(composable(text, union) for text in accepted_texts)
                counts[f"beam_top{cutoff}_chunk_covered"] += beam_covered
                counts[f"union_top{cutoff}_chunk_covered"] += union_covered
                counts[f"union_top{cutoff}_chunk_gain"] += union_covered and not mechanical
    require(sum(counts[key] for key in (
        "top8_gain_yes__saved_finite_yes", "top8_gain_yes__saved_finite_no",
        "top8_gain_no__saved_finite_yes", "top8_gain_no__saved_finite_no"))
        == len(source_rows), "top8 gain and saved-finite cross-tab is not closed")
    return {"counts": dict(counts), "alignment_exclusions": dict(exclusions)}


def evaluate(census=CENSUS, output=OUTPUT, *, report_sha=CENSUS_REPORT_SHA):
    output = Path(output)
    require(not output.exists(), "evaluation output already exists")
    closed = validate_closed_capture(census, report_sha=report_sha)
    # This is the first target-bearing read in the evaluator.
    targets = load_targets()
    metrics = evaluate_metrics(closed["source_rows"], closed["beams"], targets)
    result = {
        "schema": "phonetic-training-beam-evaluation-v1", "completed": True,
        "rows": EXPECTED_ROWS, "unique_source_inputs": EXPECTED_INPUTS,
        "eligible_occurrences": closed["eligible_occurrences"], **metrics,
        "source_eligibility": closed["selection_summary"],
        "beam_cutoffs": list(BEAM_CUTOFFS),
        "census_report_sha256": report_sha,
        "selection_sha256": closed["report"]["selection_sha256"],
        "capture_sha256": closed["report"]["capture_sha256"],
        "mechanical_capture_sha256": MECHANICAL_SHA,
        "targets_sha256": TARGETS_SHA,
        "targets_read_after_closed_capture": True,
        "gold_entered_generator": False,
        "runtime_or_top1_claimed": False,
        "candidate_pruning_cause_claimed": False,
        "chunk_denominator": (
            "unique (source_start, source_end) spans across every accepted forward-aligned "
            "target alternative; each span carries a set of accepted Hangul forms, while "
            "alternatives that induce different source boundaries remain separate spans"
        ),
        "scope": "TRAIN diagnostic candidate coverage; no runtime ranking claim",
    }
    output.write_bytes(canonical(result) + b"\n")
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False))
