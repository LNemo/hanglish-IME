import os
from types import SimpleNamespace

import pytest

from scripts.numeric_structure_candidates import NumericSpanGenerator
from scripts.probe_numeric_structure_candidates import (
    audit_index,
    matched_recovery_families,
    numeric_assignments,
    numeric_families,
    propose_case,
    read_pinned_inputs,
    supplement_candidates,
    unique_assignment,
    validate_historical_replay,
    validate_input_stats,
)


def option(text, kind="raw", score=0.0):
    return {"text": text, "kind": kind, "score": score}


def cached_row(source="11wol29ilbuteo"):
    piece = [source, 0, [option(source)]]
    return {
        "id": "train-1", "input": source, "context": "source context",
        "sentence_eligible": True,
        "learned_pieces": [piece], "static_pieces": [piece],
        "proposal_texts": [source], "proposal_count": 1,
    }


def child_row(source="11wol29ilbuteo", eligible=True):
    return {
        "new_pool": [source] if eligible else None,
        "new_candidates": [source],
    }


def test_unique_assignment_rejects_missing_and_ambiguous_segmentations():
    assert unique_assignment("abc", (("a",), ("bc",))) == ("a", "bc")
    with pytest.raises(ValueError):
        unique_assignment("other", (("a",), ("bc",)))
    with pytest.raises(ValueError):
        unique_assignment("abc", (("a", "ab"), ("bc", "c")))


def test_source_only_numeric_proposals_preserve_baseline_and_bound_overlay():
    row = cached_row()
    proposed = propose_case(row, child_row(), {}, NumericSpanGenerator())
    assert proposed["baseline_paths"] == ["11wol29ilbuteo"]
    assert proposed["overlay_paths"][0] == "11wol29ilbuteo"
    assert "11월29일부터" in proposed["overlay_paths"]
    assert len(proposed["numeric_assignments"]) <= 8
    assert len(proposed["overlay_additions"]) <= 48
    assert len(proposed["compact_overlay_additions"]) <= 8
    assert proposed["raw_original"] == "11wol29ilbuteo"


def test_expected_label_cannot_affect_candidate_production():
    row = cached_row()
    first = propose_case({**row, "expected": ["11월29일부터"]}, child_row(), {}, NumericSpanGenerator())
    second = propose_case({**row, "expected": ["unrelated"]}, child_row(), {}, NumericSpanGenerator())
    first.pop("parser_timings_ms")
    second.pop("parser_timings_ms")
    assert first == second


def test_sentence_ineligible_route_retains_actual_candidates_without_overlay():
    row = cached_row()
    row["sentence_eligible"] = False
    proposed = propose_case(row, child_row(eligible=False), {}, NumericSpanGenerator())
    assert proposed["diagnostic_supplements"]
    assert proposed["overlay_additions"] == []
    assert proposed["overlay_paths"] == [row["input"]]


def test_only_unchanged_digit_bearing_raw_only_whole_pieces_are_selected():
    row = cached_row("11wol29ilbuteo")
    row["learned_pieces"][0][2].append(option("11월29일부터", "numeric", 6.0))
    assert numeric_assignments(row, NumericSpanGenerator())[0] == []
    non_digit = cached_row("wol")
    assert numeric_assignments(non_digit, NumericSpanGenerator())[0] == []


def test_diagnostic_supplements_deduplicate_and_obey_cap():
    assignments = [
        {"replacements": [{"piece_index": 0, "source": "1x", "text": f"1값{i}"}],
         "families": ("single_numeric_unit",)}
        for i in range(8)
    ]
    base = ["1x" + chr(65 + i) for i in range(48)]
    parsed = [("1x", chr(65 + i)) for i in range(48)]
    supplements = supplement_candidates(base, parsed, assignments)
    assert len(supplements) == 384
    assert len({item["text"] for item in supplements}) == 384


def test_primary_overlay_applies_first_assignment_to_every_base_path():
    source = "11wol29ilbuteo"
    row = cached_row(source + " A")
    first = [source, 0, [option(source)]]
    second = [" A", len(source), [option(" A"), option(" B", "raw", -1.0)]]
    row["learned_pieces"] = [first, second]
    row["static_pieces"] = [first, second]
    row["proposal_texts"] = [source + " A", source + " B"]
    row["proposal_count"] = 2
    child = {"new_pool": [source + " A", source + " B"],
             "new_candidates": [source + " A"]}
    proposed = propose_case(row, child, {}, NumericSpanGenerator())
    assert [item["text"] for item in proposed["overlay_additions"]] == [
        "11월29일부터 A", "11월29일부터 B",
    ]


def test_combined_assignments_convert_all_numeric_pieces_then_keep_partial_fallbacks():
    first_source, second_source = "11wol", "29ilbuteo"
    pieces = [
        [first_source, 0, [option(first_source)]],
        [" ", len(first_source), [option(" ")]],
        [second_source, len(first_source) + 1, [option(second_source)]],
    ]
    raw = first_source + " " + second_source
    row = {**cached_row(raw), "learned_pieces": pieces, "static_pieces": pieces,
           "proposal_texts": [raw], "proposal_count": 1}
    proposed = propose_case(row, child_row(raw), {}, NumericSpanGenerator())
    supplements = [item["text"] for item in proposed["diagnostic_supplements"]]
    assert proposed["overlay_additions"][0]["text"] == "11월 29일부터"
    assert "11월 29ilbuteo" in supplements
    assert "11wol 29일부터" in supplements
    assert proposed["numeric_assignments"][0]["conversion_count"] == 2


def test_each_piece_raw_fallback_precedes_many_later_alternatives():
    class ManyPaths:
        def paths(self, source, offset):
            count = 1 if source == "1x" else 8
            return tuple((f"{source}-값{index}", 6.0, []) for index in range(count))

    pieces = [["1x", 0, [option("1x")]], [" ", 2, [option(" ")]],
              ["2y", 3, [option("2y")]]]
    row = {**cached_row("1x 2y"), "learned_pieces": pieces, "static_pieces": pieces}
    assignments, _, omitted = numeric_assignments(row, ManyPaths())
    rendered = [tuple(item["text"] for item in assignment["replacements"])
                for assignment in assignments]
    assert ("1x-값0", "2y") in rendered
    assert ("1x", "2y-값0") in rendered
    assert len(assignments) == 8
    assert omitted == 0


def test_historical_beam_replay_uses_option8_beam24_and_rejects_drift():
    piece = ["x", 0, [option(str(index), "raw", float(index)) for index in range(10)]]
    row = {"learned_pieces": [piece], "static_pieces": [piece],
           "proposal_texts": [str(index) for index in range(7, -1, -1)],
           "proposal_count": 8}
    assert validate_historical_replay(row) == row["proposal_texts"]
    with pytest.raises(ValueError):
        validate_historical_replay({**row, "proposal_texts": ["drift"]})


def test_family_labels_are_source_structural_and_not_overcredited():
    numeric = lambda text: [{"kind": "numeric", "text": text}]
    assert numeric_families("12il", numeric("일")) == ("single_numeric_unit",)
    assert numeric_families("1il", numeric("일")) == ("single_numeric_unit",)
    assert "extended_unit_tail" in numeric_families("1nyeonida", numeric("년이다"))
    assert "extended_unit_tail" in numeric_families("3gaeworiraneun", numeric("개월이라는"))
    assert "extended_unit_tail" not in numeric_families("3gaewol", numeric("개월"))
    assert "range" in numeric_families("5~6gotgwa")
    supplements = [
        {"text": "wrong", "families": ["range"]},
        {"text": "right", "families": ["single_numeric_unit"]},
    ]
    assert matched_recovery_families(supplements, ["right"]) == {"single_numeric_unit"}


def test_audit_index_rejects_duplicate_and_inconsistent_rows():
    row = {"source": "suffix", "learned": True,
           "baseline": [option("접미", "neural")], "additions": [], "result_count": 1}
    assert audit_index([row])[("suffix", True)] == ["접미"]
    with pytest.raises(ValueError):
        audit_index([row, row])
    with pytest.raises(ValueError):
        audit_index([{**row, "result_count": 2}])


class CountingPath:
    def __init__(self, payload):
        self.payload = payload
        self.reads = 0
        self.identity = (1, 2, len(payload), 3)

    def stat(self):
        return SimpleNamespace(st_dev=self.identity[0], st_ino=self.identity[1],
                               st_size=self.identity[2], st_mtime_ns=self.identity[3])

    def read_bytes(self):
        self.reads += 1
        return self.payload


def test_pinned_inputs_are_read_once_and_later_stat_changes_fail():
    import hashlib
    report = CountingPath(b"{}")
    cases = CountingPath(b'{"id":"one"}\n')
    pinned = {
        "cached_report": (report, hashlib.sha256(report.payload).hexdigest()),
        "cached_cases": (cases, hashlib.sha256(cases.payload).hexdigest()),
    }
    data, stats = read_pinned_inputs(pinned)
    assert data == {"cached_report": {}, "cached_cases": [{"id": "one"}]}
    assert report.reads == cases.reads == 1
    cases.identity = (1, 2, len(cases.payload), 4)
    with pytest.raises(ValueError):
        validate_input_stats(stats, pinned)


def test_pinned_input_hashes_exact_bytes_before_parsing():
    bad = CountingPath(b"{}")
    with pytest.raises(ValueError):
        read_pinned_inputs({"cached_report": (bad, "0" * 64)})
    assert bad.reads == 1


def test_post_run_hash_rejects_changed_bytes_even_with_restored_stat(tmp_path):
    import hashlib
    path = tmp_path / "report.json"
    path.write_bytes(b"{}")
    pinned = {"cached_report": (path, hashlib.sha256(b"{}").hexdigest())}
    _, stats = read_pinned_inputs(pinned)
    identity = path.stat()
    path.write_bytes(b"[]")
    os.utime(path, ns=(identity.st_atime_ns, identity.st_mtime_ns))
    assert (path.stat().st_dev, path.stat().st_ino, path.stat().st_size,
            path.stat().st_mtime_ns) == stats["cached_report"]
    with pytest.raises(ValueError, match="bytes changed"):
        validate_input_stats(stats, pinned)
