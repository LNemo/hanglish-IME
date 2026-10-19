"""Model-free TRAIN probe for bounded structured numeric candidate supplements.

This replays frozen historical ``addbe`` candidate pools.  It does not load a
ranker or transducer and does not claim parity with the current production
runtime.  Labels are consulted only after every source-derived supplement has
been constructed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hanglish.core.lexical_data import read_counter_units
from hanglish.core.neural_converter import NUMERIC_UNITS, file_sha256
from scripts.numeric_structure_candidates import COPULAR_TAILS, NumericSpanGenerator, MAX_PATHS


OUTPUT = ROOT / ".omx/artifacts/numeric-structure-candidates-v1"
CACHED = ROOT / ".omx/artifacts/current-cached-lattice-v1"
ACRONYM = ROOT / ".omx/artifacts/acronym-suffix-candidates-v1"
HISTORICAL_RUNTIME = "neural-1-addbecef42dc"
PREPARATION_MANIFEST = "b280bd12081c90fdd6bf53a3356b0a5f0ff0ccf83b905ca26b027c9d09027989"
PINNED = {
    "cached_report": (CACHED / "report.json", "67dd57c2bcae695951a0b2bffee8c0dce18924616f5f7c6892be739d9c352cfb"),
    "cached_cases": (CACHED / "cases.jsonl", "bf4f684aac134283ce5ba22bcde11ff22adeddbc23715b0ae85c21bb639325f1"),
    "acronym_report": (ACRONYM / "report.json", "7869047ba2f7cbaf54756d8c30067554c0bfb46d8459590e955cc28e4da740a1"),
    "acronym_cases": (ACRONYM / "cases.jsonl", "179bf2d0e1dfb5d0f8f587d5fcdf582ff4e6e775be15d587eba3ac2e4b062452"),
    "option_audit": (ACRONYM / "option-audit.jsonl", "a104ef2cdc7a70e3b0d4ad3bf4d087678438a14afd6847494213a8f85acfad1b"),
    "selection": (ACRONYM / "selection.json", "0b81181065c33a06f31dfab9e3aef4f99aa3d8d54c8655ab6bc6dc38a918d8ec"),
}
PROFILE = {
    "name": "numeric-structure-cached-train-v1",
    "rows": 512,
    "cpu_threads": 2,
    "rss_limit_bytes": 3 * 1024**3,
    "time_limit_seconds": 300,
    "progress_interval": 32,
    "numeric_assignments_per_row": 8,
    "diagnostic_supplements_per_row": 384,
    "primary_overlay_additions_per_row": 48,
    "primary_overlay_total_paths": 96,
    "compact_overlay_additions_per_row": 8,
    "model_inference_calls": 0,
}
EXPECTED_UNIT_COUNT = 679
NUMBER = re.compile(r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def parse_jsonl_bytes(value):
    return [json.loads(line) for line in value.decode().splitlines() if line]


def stat_identity(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def read_pinned_inputs(pinned=PINNED):
    """Read each immutable input exactly once and parse those verified bytes."""
    result = {}
    stats = {}
    for name, (path, expected) in pinned.items():
        stats[name] = stat_identity(path)
        raw = path.read_bytes()
        if sha256_bytes(raw) != expected or stat_identity(path) != stats[name]:
            raise ValueError(f"pinned input differs while reading: {name}")
        result[name] = (parse_jsonl_bytes(raw) if name in {
            "cached_cases", "acronym_cases", "option_audit"
        } else json.loads(raw))
    return result, stats


def validate_input_stats(stats, pinned=PINNED):
    if any(stat_identity(pinned[name][0]) != identity for name, identity in stats.items()):
        raise ValueError("a pinned input changed during the probe")
    for name, (path, expected) in pinned.items():
        if file_sha256(path) != expected:
            raise ValueError(f"pinned input bytes changed during the probe: {name}")


def source_hashes():
    paths = (
        Path(__file__).resolve(),
        ROOT / "scripts/numeric_structure_candidates.py",
        ROOT / "hanglish/core/neural_converter.py",
        ROOT / "hanglish/core/lexical_data.py",
        ROOT / "hanglish/core/lattice.py",
        ROOT / "hanglish/core/phonology.py",
        ROOT / "data/external/mecab-ko-dic/NNBC.csv",
        ROOT / "data/external/mecab-ko-dic/SOURCE.json",
        ROOT / ".omx/artifacts/numeric-structure-next-plan.md",
    )
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def option_texts(piece):
    if (not isinstance(piece, list) or len(piece) != 3
            or not isinstance(piece[0], str) or type(piece[1]) is not int
            or not isinstance(piece[2], list) or not piece[2]):
        raise ValueError("cached piece is malformed")
    result = []
    for option in piece[2]:
        if (not isinstance(option, dict) or not isinstance(option.get("text"), str)
                or not option["text"] or not isinstance(option.get("kind"), str)
                or type(option.get("score")) not in (int, float)
                or not math.isfinite(option["score"])):
            raise ValueError("cached option is malformed")
        if option["text"] not in result:
            result.append(option["text"])
    return result


def validate_piece_layout(text, learned, static):
    if len(learned) != len(static):
        raise ValueError("cached learned/static piece counts differ")
    cursor = 0
    for left, right in zip(learned, static):
        option_texts(left)
        option_texts(right)
        if left[:2] != right[:2] or left[1] != cursor:
            raise ValueError("cached piece boundaries differ")
        source, cursor = left[0], left[1] + len(left[0])
        if text[left[1]:cursor] != source:
            raise ValueError("cached pieces do not cover the input")
    if cursor != len(text):
        raise ValueError("cached pieces do not cover the complete input")


def audit_index(rows):
    result = {}
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("source"), str)
                or row.get("learned") is not True
                or not isinstance(row.get("baseline"), list)
                or not isinstance(row.get("additions"), list)):
            raise ValueError("option audit row is malformed")
        key = (row["source"], True)
        if key in result:
            raise ValueError("option audit contains duplicate source keys")
        values = row["baseline"] + row["additions"]
        texts = []
        for value in values:
            if not isinstance(value.get("text"), str) or not value["text"]:
                raise ValueError("option audit candidate is malformed")
            if value["text"] not in texts:
                texts.append(value["text"])
        if row.get("result_count") != len({value["text"] for value in values}):
            raise ValueError("option audit result count differs")
        result[key] = texts
    return result


def candidate_vocabulary(row, audits):
    learned, static = row["learned_pieces"], row["static_pieces"]
    validate_piece_layout(row["input"], learned, static)
    vocabulary = []
    for learned_piece, static_piece in zip(learned, static):
        values = []
        for value in (*option_texts(learned_piece), *option_texts(static_piece),
                      *audits.get((learned_piece[0], True), ())):
            if value not in values:
                values.append(value)
        vocabulary.append(tuple(values))
    return tuple(vocabulary)


def replay_beam(pieces):
    """Replay the historical option8/beam24 algorithm from cached scores."""
    paths = [("", 0.0)]
    for piece in pieces:
        source, _, options = piece
        option_texts(piece)
        expanded = []
        for prefix, score in paths:
            for option in options[:8]:
                expanded.append((prefix + option["text"], score + float(option["score"])))
        unique = {}
        for item in sorted(expanded, key=lambda item: (-item[1], item[0])):
            unique.setdefault(item[0], item)
        paths = list(unique.values())[:24]
        if not paths:
            raise ValueError(f"cached beam became empty at piece {source!r}")
    return [text for text, _ in paths]


def validate_historical_replay(row):
    unique = {}
    for text in replay_beam(row["learned_pieces"]):
        unique.setdefault(text, None)
    for text in replay_beam(row["static_pieces"]):
        unique.setdefault(text, None)
    replayed = list(unique)[:48]
    if replayed != row.get("proposal_texts") or len(replayed) != row.get("proposal_count"):
        raise ValueError("historical cached beam replay differs")
    return replayed


def unique_assignment(output, vocabulary):
    """Return the sole piece-wise segmentation, rejecting missing/ambiguous paths."""
    @lru_cache(maxsize=None)
    def visit(index, position):
        if index == len(vocabulary):
            return [()] if position == len(output) else []
        complete = []
        for value in vocabulary[index]:
            if not output.startswith(value, position):
                continue
            for suffix in visit(index + 1, position + len(value)):
                complete.append((value, *suffix))
                if len(complete) == 2:
                    return complete
        return complete

    complete = visit(0, 0)
    if len(complete) != 1:
        raise ValueError("preserved output lacks a unique cached-piece segmentation")
    return complete[0]


def numeric_families(source, segments=()):
    families = []
    if source[:1] in "([{\"'“‘":
        families.append("wrapper")
    if any(marker in source for marker in ("~", "-", "–", "—")):
        families.append("range")
    if any(marker in source for marker in ("%", "％", "‰", "℃", "℉", "㎡", "㎾", "㏊", "°C", "°F")):
        families.append("percent_or_unicode_unit")
    if len(NUMBER.findall(source)) >= 2:
        families.append("repeated_numeric_or_date")
    converted = [item["text"] for item in segments if item.get("kind") == "numeric"]
    final = converted[-1] if converted else ""
    if any(final.endswith(tail) and len(final) > len(tail) for tail in COPULAR_TAILS):
        families.append("extended_unit_tail")
    if not families:
        families.append("single_numeric_unit")
    return tuple(families)


def raw_only_piece(learned_piece, static_piece):
    options = [*learned_piece[2], *static_piece[2]]
    return bool(options) and all(
        option.get("kind") == "raw" and option.get("text") == learned_piece[0]
        for option in options
    )


def numeric_assignments(row, generator, limit=PROFILE["numeric_assignments_per_row"]):
    """Build bounded row-wide assignments using source structure alone."""
    piece_choices = []
    timings = []
    for index, (learned_piece, static_piece) in enumerate(zip(
            row["learned_pieces"], row["static_pieces"])):
        source, offset = learned_piece[:2]
        if not any(character.isdigit() for character in source):
            continue
        if not raw_only_piece(learned_piece, static_piece):
            continue
        started = time.perf_counter()
        paths = generator.paths(source, offset)
        timings.append((time.perf_counter() - started) * 1000)
        if len(paths) > MAX_PATHS:
            raise ValueError("numeric helper exceeded its path bound")
        if paths:
            choices = [{
                "piece_index": index, "source": source, "offset": offset,
                "text": text, "grammar_score": score, "segments": segments,
                "converted": True, "families": numeric_families(source, segments),
            } for text, score, segments in paths]
            choices.append({
                "piece_index": index, "source": source, "offset": offset,
                "text": source, "grammar_score": None, "segments": [],
                "converted": False, "families": (),
            })
            piece_choices.append(choices)
    ordered_choices = []
    omitted_fallbacks = 0
    if piece_choices:
        primary = tuple(choices[0] for choices in piece_choices)
        ordered_choices.append(primary)
        # Preserve each piece's raw fallback near the front, independent of
        # how many alternatives a later numeric piece happens to have.
        for index, choices in enumerate(piece_choices):
            fallback = list(primary)
            fallback[index] = choices[-1]
            ordered_choices.append(tuple(fallback))
        omitted_fallbacks = max(0, len(piece_choices) - (limit - 1))
        # Then vary one converted spelling at a time, round-robin by rank.
        maximum_alternatives = max(len(choices) - 2 for choices in piece_choices)
        for alternative_index in range(1, maximum_alternatives + 1):
            for index, choices in enumerate(piece_choices):
                if alternative_index >= len(choices) - 1:
                    continue
                alternative = choices[alternative_index]
                varied = list(primary)
                varied[index] = alternative
                ordered_choices.append(tuple(varied))
        # Fill any remaining bounded slots with deterministic combinations;
        # keep this product lazy because only eight assignments can survive.
        ordered_choices = itertools.chain(ordered_choices, itertools.product(*piece_choices))
    assignments = []
    seen = set()
    for choices in ordered_choices:
        if not any(choice["converted"] for choice in choices):
            continue
        identity = tuple((choice["piece_index"], choice["text"], choice["converted"])
                         for choice in choices)
        if identity in seen:
            continue
        seen.add(identity)
        families = []
        for choice in choices:
            for family in choice["families"]:
                if family not in families:
                    families.append(family)
        assignments.append({
            "replacements": list(choices), "families": families,
            "conversion_count": sum(choice["converted"] for choice in choices),
        })
        if len(assignments) == limit:
            break
    return assignments, timings, omitted_fallbacks


def supplement_candidates(base_outputs, parsed_outputs, assignments,
                          diagnostic_limit=PROFILE["diagnostic_supplements_per_row"]):
    """Apply source-selected replacements without consulting labels or scores."""
    if len(base_outputs) != len(parsed_outputs):
        raise ValueError("candidate segmentation count differs")
    seen = set(base_outputs)
    supplements = []
    for assignment in assignments:
        for output, parsed in zip(base_outputs, parsed_outputs):
            replacements = assignment["replacements"]
            if any(parsed[item["piece_index"]] != item["source"] for item in replacements):
                continue
            values = list(parsed)
            for item in replacements:
                values[item["piece_index"]] = item["text"]
            candidate = "".join(values)
            if candidate == output or candidate in seen:
                continue
            seen.add(candidate)
            supplements.append({
                "text": candidate, "replacements": replacements,
                "families": assignment["families"],
            })
            if len(supplements) == diagnostic_limit:
                return supplements
    return supplements


def propose_case(cached, acronym, audits, generator):
    """Construct all proposals without accepting or reading expected labels."""
    validate_historical_replay(cached)
    vocabulary = candidate_vocabulary(cached, audits)
    eligible = cached["sentence_eligible"]
    if eligible != (acronym["new_pool"] is not None):
        raise ValueError("acronym sentence eligibility differs from cached replay")
    base_outputs = acronym["new_pool"] if eligible else acronym["new_candidates"]
    if not isinstance(base_outputs, list) or not base_outputs or len(set(base_outputs)) != len(base_outputs):
        raise ValueError("frozen acronym outputs are malformed")
    visible = acronym.get("new_candidates")
    if (not isinstance(visible, list) or not 1 <= len(visible) <= 5
            or visible[-1] != cached["input"]
            or (eligible and len(base_outputs) > 48)
            or (not eligible and base_outputs != visible)):
        raise ValueError("frozen acronym visible/raw candidate contract differs")
    parsed = [unique_assignment(output, vocabulary) for output in base_outputs]
    assignments, timings, omitted_fallbacks = numeric_assignments(cached, generator)
    supplements = supplement_candidates(base_outputs, parsed, assignments)
    # Apply the first source-ordered grammar assignment to every preserved
    # base path. This keeps nonnumeric alternatives represented. The compact
    # first-eight comparison is retained as a separate diagnostic.
    primary_additions = (supplement_candidates(
        base_outputs, parsed, assignments[:1],
        PROFILE["primary_overlay_additions_per_row"],
    ) if eligible else [])
    compact_additions = supplements[:PROFILE["compact_overlay_additions_per_row"]] if eligible else []
    overlay = base_outputs + [item["text"] for item in primary_additions]
    compact_overlay = base_outputs + [item["text"] for item in compact_additions]
    if (overlay[:len(base_outputs)] != base_outputs
            or compact_overlay[:len(base_outputs)] != base_outputs
            or len(primary_additions) > PROFILE["primary_overlay_additions_per_row"]
            or len(overlay) > PROFILE["primary_overlay_total_paths"]
            or len(compact_overlay) > 56):
        raise ValueError("numeric overlay changed the frozen baseline prefix")
    return {
        "sentence_eligible": eligible,
        "raw_original": cached["input"],
        "baseline_paths": base_outputs,
        "baseline_visible_candidates": acronym["new_candidates"],
        "numeric_assignments": assignments,
        "diagnostic_supplements": supplements,
        "overlay_additions": primary_additions,
        "overlay_paths": overlay,
        "compact_overlay_additions": compact_additions,
        "compact_overlay_paths": compact_overlay,
        "parser_timings_ms": timings,
        "omitted_fallback_variants": omitted_fallbacks,
    }


def validate_inputs(data):
    cached_report, acronym_report = data["cached_report"], data["acronym_report"]
    cached, acronym, selection = (data["cached_cases"], data["acronym_cases"], data["selection"])
    if (cached_report.get("runtime_version") != HISTORICAL_RUNTIME
            or acronym_report.get("parent_version") != HISTORICAL_RUNTIME
            or cached_report.get("preparation_manifest_sha256") != PREPARATION_MANIFEST
            or acronym_report.get("preparation_manifest_sha256") != PREPARATION_MANIFEST
            or cached_report.get("probe_report_sha256") != PINNED["acronym_report"][1]
            or cached_report.get("cases_sha256") != PINNED["cached_cases"][1]
            or acronym_report.get("cases_sha256") != PINNED["acronym_cases"][1]
            or acronym_report.get("option_audit_sha256") != PINNED["option_audit"][1]
            or acronym_report.get("selection_sha256") != PINNED["selection"][1]
            or cached_report.get("actual_model_inference_calls") != 0
            or acronym_report.get("metrics", {}).get("rows") != PROFILE["rows"]
            or any(report.get(flag) is not False for report in (cached_report, acronym_report)
                   for flag in ("natural_dev_inference_run", "development_inference_run",
                                "test_data_read", "deployed"))):
        raise ValueError("frozen report lineage is incomplete")
    source_chain = cached_report.get("source_hashes", {})
    for name in ("acronym_report", "acronym_cases", "option_audit", "selection"):
        relative = str(PINNED[name][0].relative_to(ROOT))
        if source_chain.get(relative) != PINNED[name][1]:
            raise ValueError("cached report does not pin acronym input lineage")
    if (len(cached) != PROFILE["rows"] or len(acronym) != PROFILE["rows"]
            or len(selection) != PROFILE["rows"]):
        raise ValueError("frozen TRAIN selection is incomplete")
    identities = set()
    for frozen, child, selected in zip(cached, acronym, selection):
        identity = tuple(frozen.get(key) for key in ("id", "input", "context"))
        if identity in identities or identity != tuple(child.get(key) for key in (
                "id", "input", "context")) or identity != tuple(
                selected.get(key) for key in ("id", "input", "context")):
            raise ValueError("frozen TRAIN identities or order differ")
        identities.add(identity)
        if selected.get("origin") != "natural" or selected.get("split") != "train":
            raise ValueError("selection contains a non-TRAIN source")
        if frozen["sentence_eligible"]:
            if frozen["proposal_texts"] != child["before_pool"] or child["before_pool"] != selected["candidates"]:
                raise ValueError("historical eligible pools differ")
        elif child["before_pool"] is not None or selected["candidates"] != selected["actual_candidates"]:
            raise ValueError("historical fallback route differs")


def validate_labels(data):
    """Validate frozen TRAIN labels only after all proposals have been built."""
    if (data["cached_report"].get("pool_covered") != 309
            or data["acronym_report"].get("metrics", {}).get("before_pool_covered") != 309
            or data["acronym_report"].get("metrics", {}).get("new_pool_covered") != 319):
        raise ValueError("frozen TRAIN aggregate coverage differs")
    for frozen, child, selected in zip(
            data["cached_cases"], data["acronym_cases"], data["selection"]):
        labels = frozen.get("expected")
        if (not isinstance(labels, list) or not labels
                or labels != child.get("expected") or labels != selected.get("expected")):
            raise ValueError("frozen TRAIN labels or order differ")


def percentile(values, proportion):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(proportion * len(ordered)) - 1)]


def matched_recovery_families(supplements, expected):
    return {family for item in supplements if item["text"] in expected
            for family in item["families"]}


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def run(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    os.environ["OMP_NUM_THREADS"] = str(PROFILE["cpu_threads"])
    probe_source = Path(__file__).read_bytes()
    probe_source_sha256 = sha256_bytes(probe_source)
    data, input_stats = read_pinned_inputs()
    validate_inputs(data)
    before = source_hashes()
    audits = audit_index(data["option_audit"])
    counter_directory = ROOT / "data/external/mecab-ko-dic"
    units = tuple(sorted(set(NUMERIC_UNITS) | set(read_counter_units(counter_directory)),
                         key=lambda unit: (-len(unit), unit)))
    if len(units) != EXPECTED_UNIT_COUNT:
        raise ValueError("pinned numeric unit inventory count differs")
    generator = NumericSpanGenerator(units)
    proposals = []
    # Candidate production is complete before expected labels are read below.
    for index, (cached, acronym) in enumerate(zip(data["cached_cases"], data["acronym_cases"])):
        proposed = propose_case(cached, acronym, audits, generator)
        proposals.append(proposed)
        elapsed = time.monotonic() - started
        if rss_bytes() > PROFILE["rss_limit_bytes"] or elapsed > PROFILE["time_limit_seconds"]:
            raise RuntimeError("numeric structure TRAIN probe resource bound exceeded")
        if (index + 1) % PROFILE["progress_interval"] == 0:
            print(json.dumps({"event": "numeric_structure_probe", "rows": index + 1,
                              "elapsed_seconds": round(elapsed, 2)}), flush=True)

    validate_labels(data)
    family_attempts, family_recoveries = Counter(), Counter()
    output_rows = []
    timings = []
    for cached, acronym, proposed in zip(data["cached_cases"], data["acronym_cases"], proposals):
        expected = cached["expected"]
        if not isinstance(expected, list) or not expected:
            raise ValueError("TRAIN label list is malformed")
        baseline = proposed["baseline_paths"]
        diagnostic = baseline + [item["text"] for item in proposed["diagnostic_supplements"]]
        overlay = proposed["overlay_paths"]
        compact_overlay = proposed["compact_overlay_paths"]
        before_covered = acronym["new_top5"] if not proposed["sentence_eligible"] else any(
            item in expected for item in baseline)
        actual_top5 = any(item in expected for item in proposed["baseline_visible_candidates"])
        actual_pool = (None if not proposed["sentence_eligible"] else
                       any(item in expected for item in baseline))
        if (acronym.get("new_top5") != actual_top5
                or acronym.get("new_pool_covered") != actual_pool):
            raise ValueError("frozen acronym label metrics differ from its candidates")
        diagnostic_covered = any(item in expected for item in diagnostic)
        overlay_covered = before_covered if not proposed["sentence_eligible"] else any(
            item in expected for item in overlay)
        compact_covered = before_covered if not proposed["sentence_eligible"] else any(
            item in expected for item in compact_overlay)
        families = sorted({family for item in proposed["numeric_assignments"]
                           for family in item["families"]})
        family_attempts.update(families)
        if diagnostic_covered and not before_covered:
            family_recoveries.update(matched_recovery_families(
                proposed["diagnostic_supplements"], expected
            ))
        missing_recovered = []
        for missing in cached["missing_pieces"]:
            matching = [replacement
                        for item in proposed["numeric_assignments"]
                        for replacement in item["replacements"]
                        if replacement["converted"]
                        and replacement["source"] == missing["source"]
                        and replacement["offset"] == missing["start"]]
            if any(item["text"] in missing.get("targets", ()) for item in matching):
                missing_recovered.append({"source": missing["source"], "start": missing["start"]})
        timings.extend(proposed.pop("parser_timings_ms"))
        output_rows.append({
            "id": cached["id"], "input": cached["input"], "context": cached["context"],
            "expected": expected, **proposed,
            "frozen_child_covered": bool(before_covered),
            "diagnostic_covered": diagnostic_covered,
            "overlay_covered": overlay_covered,
            "compact_overlay_covered": compact_covered,
            "missing_pieces_recovered": missing_recovered,
        })

    args.output.mkdir(parents=True)
    cases_path = args.output / "cases.jsonl"
    with cases_path.open("x") as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    validate_input_stats(input_stats)
    after = source_hashes()
    elapsed = time.monotonic() - started
    if before != after or rss_bytes() > PROFILE["rss_limit_bytes"] or elapsed > PROFILE["time_limit_seconds"]:
        raise ValueError("probe sources or resource bounds changed")
    historical = sum(row.get("pool_covered") is True for row in data["cached_cases"])
    frozen = sum(row["frozen_child_covered"] for row in output_rows)
    diagnostic = sum(row["diagnostic_covered"] for row in output_rows)
    overlay = sum(row["overlay_covered"] for row in output_rows)
    compact = sum(row["compact_overlay_covered"] for row in output_rows)
    recovered_output_count = sum(len(row["baseline_paths"]) for row in output_rows)
    if historical != 309 or frozen != 319 or recovered_output_count != 21716:
        raise ValueError("frozen historical coverage or recovered path count differs")
    unit_inventory = json.dumps(units, ensure_ascii=False, separators=(",", ":")).encode()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Frozen natural TRAIN first512 cached candidate-structure evaluation",
        "profile": PROFILE,
        "historical_runtime_version": HISTORICAL_RUNTIME,
        "current_production_baseline_claim": False,
        "lineage_note": "Cached replay is model-free; the frozen acronym inputs contain historically captured inference.",
        "preparation_manifest_sha256": PREPARATION_MANIFEST,
        "numeric_unit_count": len(units),
        "numeric_unit_inventory": units,
        "numeric_unit_inventory_sha256": sha256_bytes(unit_inventory),
        "pinned_inputs": {name: digest for name, (_, digest) in PINNED.items()},
        "source_hashes": before,
        "source_hashes_after": after,
        "probe_source_sha256": probe_source_sha256,
        "effective_omp_num_threads": os.environ["OMP_NUM_THREADS"],
        "rows": len(output_rows),
        "uniquely_recovered_frozen_outputs": recovered_output_count,
        "historical_cached_coverage": historical,
        "frozen_acronym_child_coverage": frozen,
        "diagnostic_coverage": diagnostic,
        "overlay_coverage": overlay,
        "compact_overlay_coverage": compact,
        "diagnostic_overlay_gap": diagnostic - overlay,
        "diagnostic_compact_overlay_gap": diagnostic - compact,
        "missing_piece_recoveries": sum(len(row["missing_pieces_recovered"]) for row in output_rows),
        "family_attempt_rows": dict(family_attempts),
        "family_recovery_rows": dict(family_recoveries),
        "parser_calls": len(timings),
        "parser_latency_ms": {"p50": statistics.median(timings) if timings else 0.0,
                              "p95": percentile(timings, .95)},
        "maximum_numeric_assignments": max((len(row["numeric_assignments"]) for row in output_rows), default=0),
        "maximum_converted_pieces_per_assignment": max((
            item["conversion_count"] for row in output_rows
            for item in row["numeric_assignments"]
        ), default=0),
        "omitted_single_piece_fallback_variants": sum(
            row["omitted_fallback_variants"] for row in output_rows
        ),
        "maximum_diagnostic_supplements": max((len(row["diagnostic_supplements"]) for row in output_rows), default=0),
        "maximum_overlay_additions": max((len(row["overlay_additions"]) for row in output_rows), default=0),
        "maximum_compact_overlay_additions": max((len(row["compact_overlay_additions"]) for row in output_rows), default=0),
        "baseline_prefix_preserved": all(row["overlay_paths"][:len(row["baseline_paths"])] == row["baseline_paths"] for row in output_rows),
        "raw_original_kept_separately": True,
        "all_preserved_outputs_uniquely_segmented": True,
        "all_numeric_paths_validated_by_helper": True,
        "only_selected_raw_pieces_replaced": all(
            raw_only_piece(row["learned_pieces"][replacement["piece_index"]],
                           row["static_pieces"][replacement["piece_index"]])
            and replacement["source"] == row["learned_pieces"][replacement["piece_index"]][0]
            for row, output in zip(data["cached_cases"], output_rows)
            for assignment in output["numeric_assignments"]
            for replacement in assignment["replacements"]
        ),
        "labels_used_for_candidate_production": False,
        "numeric_scores_are_grammar_bookkeeping_only": True,
        "first_choice_accuracy_claim": False,
        "new_model_inference_calls": 0,
        "natural_dev_read": False,
        "test_data_read": False,
        "deployed": False,
        "cases_sha256": file_sha256(cases_path),
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": rss_bytes(),
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if file_sha256(Path(__file__)) != probe_source_sha256:
        raise ValueError("probe source changed before snapshot")
    source_path = args.output / "source.py"
    source_path.write_bytes(probe_source)
    if file_sha256(source_path) != probe_source_sha256:
        raise ValueError("probe source snapshot differs")
    print(json.dumps({key: report[key] for key in (
        "historical_cached_coverage", "frozen_acronym_child_coverage",
        "diagnostic_coverage", "overlay_coverage", "missing_piece_recoveries",
        "parser_latency_ms", "elapsed_seconds", "peak_rss_bytes")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args())
