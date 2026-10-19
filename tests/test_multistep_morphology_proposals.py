from copy import deepcopy
import itertools

import pytest

from hanglish.core.phonology import romanize_word
from scripts.multistep_morphology_proposals import (
    MAX_PROPOSALS, MultiStepMorphology, read_attested_plural_bases,
)
from scripts.probe_multistep_morphology_coverage import composable, source_pieces


ENTRIES = [
    ("먹", "VV", 10), ("좋", "VA", 11), ("두", "VX", 12),
    ("학생", "NNG", 20), ("학교", "NNG", 21), ("길", "NNG", 22),
    ("그", "NP", 23), ("사람들", "NNG", 24), ("금지", "MAG", 1),
]


def grammar(entries=ENTRIES, english=("test",)):
    plurals = [("학생", "NNG", 20, "학생/NNG/*+들/XSN/*+을/JKO/*")]
    return MultiStepMorphology(entries, "f" * 64, english, plurals)


@pytest.mark.parametrize("text,family", [
    ("먹기를", "predicate_nominalizer_case"),
    ("좋기만", "predicate_nominalizer_case"),
    ("학생들을", "nominal_plural_case"),
    ("학교에서도", "nominal_postposition_auxiliary"),
    ("길로도", "nominal_postposition_auxiliary"),
    ("학생만을", "nominal_man_case"),
])
def test_independently_specified_grammar_families(text, family):
    proposals = grammar().propose(romanize_word(text))
    found = next(item for item in proposals if item.text == text)
    assert found.family == family
    assert "".join(found.constituents) == text
    assert found.stem_pos in {"VV", "VA", "VX", "NNG", "NNP", "NP", "NR"}
    assert found.lexicon_fingerprint == "f" * 64


def test_pos_and_duplicate_plural_restrictions_are_enforced():
    value = grammar()
    assert value.propose(romanize_word("학생기를")) == ()
    assert value.propose(romanize_word("먹들을")) == ()
    assert value.propose(romanize_word("사람들들을")) == ()
    assert value.propose(romanize_word("금지들을")) == ()


def test_plural_requires_independently_attested_noun_plus_deul_analysis():
    unattested = MultiStepMorphology(ENTRIES, "f" * 64, (), ())
    assert unattested.propose(romanize_word("학생들을")) == ()
    attested = grammar().propose(romanize_word("학생들을"))
    proposal = next(item for item in attested if item.text == "학생들을")
    assert proposal.provenance.startswith("attested Inflect noun+들/XSN")


def test_plural_parser_normalizes_sense_digit_and_rejects_surface_mismatch(tmp_path):
    path = tmp_path / "Inflect.csv"
    rows = [
        ["단편들을", "", "", "10", "NNG+XSN+JKO", "", "", "", "", "", "",
         "단편3/NNG/*+들/XSN/*+을/JKO/*"],
        ["이웃들이", "", "", "11", "NNG+XSN+JKS", "", "", "", "", "", "",
         "이웃사람/NNG/*+들/XSN/*+이/JKS/*"],
        ["학생들다", "", "", "12", "NNG+XSN+EF", "", "", "", "", "", "",
         "학생/NNG/*+들/XSN/*+다/EF/*"],
    ]
    import csv
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    assert read_attested_plural_bases(path) == (
        ("단편", "NNG", 10, "단편3/NNG/*+들/XSN/*+을/JKO/*"),
    )


@pytest.mark.parametrize("source", ["test", "Test", "api", "a1", "한글", "a", "a" * 41, ""])
def test_english_raw_and_out_of_scope_inputs_are_ignored(source):
    assert grammar().propose(source) == ()


def test_coda_allomorph_and_boundary_are_forward_verified():
    value = grammar()
    accepted = value.propose(romanize_word("길로도"))
    assert "길로도" in {item.text for item in accepted}
    assert "길으로도" not in {item.text for item in value.propose(
        romanize_word("길으로도"))}
    assert all(romanize_word(item.text) == romanize_word("길로도")
               for item in accepted)


def test_proposals_are_deterministic_bounded_and_do_not_mutate_inputs():
    homophones = ["".join(parts) for parts in itertools.product(
        ("낫", "낮", "낯", "낱"), repeat=4)]
    entries = [(stem, "NNG", cost) for cost, stem in enumerate(homophones, 1)]
    before = deepcopy(entries)
    value = grammar(entries, english=())
    source = romanize_word(homophones[0] + "에서도")
    matching_costs = [cost for stem, _, cost in entries
                      if romanize_word(stem + "에서도") == source]
    assert len(matching_costs) > MAX_PROPOSALS
    first = value.propose(source)
    assert entries == before and first == value.propose(source)
    assert len(first) == MAX_PROPOSALS
    assert [item.stem_cost for item in first] == sorted(matching_costs)[:MAX_PROPOSALS]
    assert all(romanize_word(item.text) == source for item in first)


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def options(self, source):
        self.calls.append(source)
        from hanglish.core.lattice import WordOption
        return (WordOption(source, 0.0, "raw"),)


def test_source_capture_is_additive_and_preserves_input_and_baseline():
    generator = FakeGenerator()
    morphology = grammar()
    source = "test meokgireul `code`"
    pieces = source_pieces(source, generator, morphology)
    assert "".join(piece["source"] for piece in pieces) == source
    assert generator.calls == ["test", "meokgireul"]
    target = "test 먹기를 `code`"
    before = [[option["text"] for option in piece["baseline"]] for piece in pieces]
    after = [
        [*old, *(option["text"] for option in piece["proposals"])]
        for old, piece in zip(before, pieces)
    ]
    assert not composable(target, before)
    assert composable(target, after)
    assert all(piece["proposals"] == [] for piece in pieces
               if piece["source"] in {"test", "`code`"})
