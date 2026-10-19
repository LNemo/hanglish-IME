from copy import deepcopy
from dataclasses import dataclass

import pytest

from hanglish.core.copy_boundaries import refine_path
from hanglish.core.lattice import CandidateGenerator


@dataclass(frozen=True)
class Option:
    text: str
    kind: str


class FakeGenerator:
    def __init__(self, suffixes, options):
        self.english_suffixes = list(suffixes)
        self.values = {key: list(value) for key, value in options.items()}
        self.calls = []

    def options(self, source):
        self.calls.append(source)
        return self.values.get(source, ())


def segment(source, text, kind="mixed", start=0):
    return {"input_start": start, "input_end": start + len(source),
            "source": source, "text": text, "kind": kind}


def test_admitted_mixed_option_is_refined_with_exact_provenance():
    generator = FakeGenerator(
        [("eul", "을"), ("reul", "를")],
        {"testreul": [Option("test를", "mixed")]},
    )
    original = [segment("testreul", "test를")]

    result = refine_path("testreul", "test를", original, generator=generator)

    assert result["segments"] == [
        segment("test", "test", "raw"),
        segment("reul", "를", "mixed", 4),
    ]
    assert all(set(row) == {"input_start", "input_end", "source", "text", "kind"}
               for row in result["segments"])
    assert result["coarse_segments"] == original
    assert result["refined"] is True
    derivation = result["derivations"][0]
    assert derivation["status"] == "refined"
    assert derivation["suffix_inventory"]["roman"] == "reul"
    assert derivation["source_boundary"] == [0, 8]
    assert derivation["output_boundary"] == [0, 5]
    assert [child["action"] for child in derivation["children"]] == ["copy", "convert"]
    assert result["copy_spans"] == [{
        "parent_index": 0,
        "rule_id": derivation["rule_id"],
        "source_boundary": [0, 4],
        "output_boundary": [0, 4],
        "source": "test",
        "text": "test",
        "origin": "generator_admitted_suffix_derivation",
        "automatic_hypothesis": True,
        "user_committed": False,
    }]


def test_inventory_match_without_generator_admission_stays_coarse():
    generator = FakeGenerator([("reul", "를")], {"novelreul": []})
    coarse = segment("novelreul", "novel를")
    result = refine_path("novelreul", "novel를", [coarse], generator=generator)
    assert result["segments"] == [coarse]
    assert result["derivations"][0]["reason"] == "generator_option_not_admitted"
    assert result["copy_spans"] == []


def test_distinct_matching_splits_are_ambiguous_but_duplicate_rows_are_one_rule():
    ambiguous = FakeGenerator(
        [("de", "를"), ("cde", "c를")],
        {"abcde": [Option("abc를", "mixed")]},
    )
    coarse = segment("abcde", "abc를")
    result = refine_path("abcde", "abc를", [coarse], generator=ambiguous)
    assert result["segments"] == [coarse]
    assert result["derivations"][0]["reason"] == "ambiguous_inventory_derivation"
    assert len(result["derivations"][0]["matching_derivations"]) == 2

    duplicate = FakeGenerator(
        [("reul", "를"), ("reul", "를")],
        {"testreul": [Option("test를", "mixed")]},
    )
    result = refine_path("testreul", "test를",
                         [segment("testreul", "test를")], generator=duplicate)
    assert result["refined"] is True


def test_longest_particle_derivation_is_proven_by_target_not_guessed():
    generator = FakeGenerator(
        [("eul", "을"), ("reul", "를")],
        {"testreul": [Option("test를", "mixed")]},
    )
    result = refine_path("testreul", "test를",
                         [segment("testreul", "test를")], generator=generator)
    assert result["segments"][1] == segment("reul", "를", "mixed", 4)

    generator = FakeGenerator(
        [("eul", "을"), ("reul", "를")],
        {"boxeul": [Option("box을", "mixed")]},
    )
    result = refine_path("boxeul", "box을",
                         [segment("boxeul", "box을")], generator=generator)
    assert result["segments"][1] == segment("eul", "을", "mixed", 3)


def test_uppercase_suffix_replays_lowercase_inventory_but_preserves_source():
    generator = FakeGenerator(
        [("reul", "를")], {"testREUL": [Option("test를", "mixed")]})
    result = refine_path("testREUL", "test를",
                         [segment("testREUL", "test를")], generator=generator)
    assert result["segments"] == [
        segment("test", "test", "raw"),
        segment("REUL", "를", "mixed", 4),
    ]
    assert result["derivations"][0]["suffix_inventory"]["roman"] == "reul"


@pytest.mark.parametrize("source,target", [
    ("Testreul", "test를"),       # changed case is not an identity COPY
    ("testreul", "te스트를"),    # interior ASCII/Hangul alignment is unknown
    ("testreul", "테스트를"),    # output has no exact inventory derivation
])
def test_changed_or_interior_output_is_never_guessed(source, target):
    generator = FakeGenerator(
        [("reul", "를")], {source: [Option(target, "mixed")]})
    coarse = segment(source, target)
    result = refine_path(source, target, [coarse], generator=generator)
    assert result["segments"] == [coarse]
    assert result["derivations"][0]["reason"] == "no_exact_inventory_derivation"
    assert result["copy_spans"] == []


def test_unicode_prefix_offsets_use_codepoints_and_preserve_text():
    generator = FakeGenerator(
        [("reul", "를")], {"éAPIreul": [Option("éAPI를", "mixed")]})
    result = refine_path("éAPIreul", "éAPI를",
                         [segment("éAPIreul", "éAPI를")], generator=generator)
    assert result["segments"] == [
        segment("éAPI", "éAPI", "raw"),
        segment("reul", "를", "mixed", 4),
    ]
    assert result["copy_spans"][0]["source_boundary"] == [0, 4]
    assert result["copy_spans"][0]["output_boundary"] == [0, 4]


def test_full_path_uses_global_input_and_output_codepoint_offsets():
    generator = FakeGenerator(
        [("reul", "를")], {"testreul": [Option("test를", "mixed")]})
    rows = [segment("go ", "가 ", "lexicon"),
            segment("testreul", "test를", "mixed", 3)]
    result = refine_path("go testreul", "가 test를", rows, generator=generator)
    assert result["segments"] == [
        segment("go ", "가 ", "lexicon"),
        segment("test", "test", "raw", 3),
        segment("reul", "를", "mixed", 7),
    ]
    derivation = result["derivations"][1]
    assert derivation["source_boundary"] == [3, 11]
    assert derivation["output_boundary"] == [2, 7]
    assert result["copy_spans"][0]["source_boundary"] == [3, 7]
    assert result["copy_spans"][0]["output_boundary"] == [2, 6]


def test_existing_identity_copy_and_protected_raw_punctuation_are_preserved():
    generator = FakeGenerator([("reul", "를")], {})
    rows = [segment("API", "API", "lexicon"), segment("!", "!", "raw", 3)]
    result = refine_path("API!", "API!", rows, generator=generator)
    assert result["segments"] == rows
    assert [row["status"] for row in result["derivations"]] == [
        "already_copy", "already_copy"]
    assert [row["origin"] for row in result["copy_spans"]] == [
        "existing_identity", "existing_identity"]
    assert generator.calls == []


def test_nonmixed_changed_segment_is_preserved_without_substring_alignment():
    generator = FakeGenerator([("reul", "를")], {})
    coarse = segment("abc", "ABC", "lexicon")
    result = refine_path("abc", "ABC", [coarse], generator=generator)
    assert result["segments"] == [coarse]
    assert result["derivations"][0]["status"] == "not_applicable"


@pytest.mark.parametrize("rows", [
    [segment("abc", "abc", "raw", 1)],
    [{**segment("abc", "abc", "raw"), "input_end": 2}],
    [{**segment("abc", "abc", "raw"), "source": "abd"}],
])
def test_malformed_parent_offsets_are_rejected(rows):
    generator = FakeGenerator([], {})
    with pytest.raises(ValueError, match="source offsets"):
        refine_path("abc", "abc", rows, generator=generator)


def test_inputs_and_generator_resources_are_not_mutated():
    generator = FakeGenerator(
        [("reul", "를")], {"testreul": [Option("test를", "mixed")]})
    rows = [segment("testreul", "test를")]
    before_rows = deepcopy(rows)
    before_suffixes = deepcopy(generator.english_suffixes)
    before_values = deepcopy(generator.values)
    result = refine_path("testreul", "test를", rows, generator=generator)
    result["coarse_segments"][0]["text"] = "mutated result"
    assert rows == before_rows
    assert generator.english_suffixes == before_suffixes
    assert generator.values == before_values


@pytest.mark.parametrize(("source", "target", "split", "suffix"), [
    ("fluxwidgetreul", "fluxwidget를", 10, "reul"),
    ("quasarboxeul", "quasarbox을", 9, "eul"),
    ("fluxwidgetREUL", "fluxwidget를", 10, "REUL"),
])
def test_real_generator_refines_unseen_english_prefix_plus_particle(
        source, target, split, suffix):
    generator = CandidateGenerator(external_lexicon=False)
    assert any(option.text == target and option.kind == "mixed"
               for option in generator.options(source))
    result = refine_path(source, target, [segment(source, target)],
                         generator=generator)
    assert result["refined"] is True
    assert result["segments"][0]["source"] == source[:split]
    assert result["segments"][0]["kind"] == "raw"
    assert result["segments"][1]["source"] == suffix
    assert "fluxwidget" not in generator.english
    assert "quasarbox" not in generator.english
