from copy import deepcopy
from pathlib import Path

import pytest

from hanglish.core.lattice import CandidateGenerator, WordOption
from hanglish.core.lexical_data import read_counter_units
from hanglish.core.neural_converter import NUMERIC_UNITS
from hanglish.core.numeric_composition import (
    composition_extras,
    numeric_suffix_proposals,
)
from hanglish.core.phonology import romanize_word


def option(text, score=6.0, kind="lexicon"):
    return WordOption(text, score, kind)


class StubGenerator:
    def __init__(self, values):
        self.values = values

    def options(self, source):
        return tuple(self.values.get(source, (option(source, 0.0, "raw"),)))


def piece(source, offset):
    return (source, offset, (option(source, 0.0, "raw"),))


def parent(pieces, tag=""):
    segments = [{
        "input_start": offset,
        "input_end": offset + len(source),
        "source": source,
        "text": source,
        "kind": "raw",
        **({"tag": tag} if tag else {}),
    } for source, offset, _ in pieces]
    return ("".join(item["text"] for item in segments), 3.25, segments)


def test_real_resources_license_counter_lexical_tail_and_decimal():
    generator = CandidateGenerator(external_lexicon=True)
    counters = read_counter_units(Path("data/external/mecab-ko-dic"))
    units = tuple(sorted(set(NUMERIC_UNITS) | set(counters), key=lambda x: (-len(x), x)))

    assert numeric_suffix_proposals("10nyeongan", generator, units) == ("10년간",)
    assert numeric_suffix_proposals("48.2bungi", generator, units) == ("48.2분기",)
    assert numeric_suffix_proposals("48.2reul", generator, units) == ()
    assert numeric_suffix_proposals("9inseung", generator, units) == ("9인세웅", "9인승")
    assert numeric_suffix_proposals("1480gaejeomkkaji", generator, units) == (
        "1480개젊까지", "1480개점까지", "1480개제옮까지", "1480개제옴까지",
    )


def test_proposals_require_full_rr_lexical_score_and_licensed_counter():
    generator = StubGenerator({
        "bun": (option("분"),),
        "gi": (option("기", 5.5), option("기타", 9.0)),
        "foo": (option("푸"),),
    })
    assert numeric_suffix_proposals("2bungi", generator) == ("2분기",)
    assert numeric_suffix_proposals("2foogi", generator) == ()
    generator.values["gi"] = (option("기", 5.49),)
    assert numeric_suffix_proposals("2bungi", generator) == ()


@pytest.mark.parametrize("source", [
    "1", "1" + "a" * 41, "1Bungi", "1e3", "0xff", "192.168.0.1",
    "v2code", "test1bungi", "1_bungi", "1/bungi", "1@bungi",
])
def test_protected_or_non_grammar_shapes_are_rejected(source):
    assert numeric_suffix_proposals(source, StubGenerator({})) == ()


def test_two_piece_partial_cartesian_paths_offsets_order_and_preservation():
    generator = StubGenerator({
        "bun": (option("분"),), "gi": (option("기"),),
        "nyeon": (option("년"),), "gan": (option("간"),),
    })
    pieces = [piece("2bungi", 0), piece(" ", 6), piece("10nyeongan", 7)]
    old_parent = parent(pieces, "parent")
    old_extra = ("old", 1.0, [{"sentinel": True}])
    before = deepcopy((pieces, old_parent, old_extra))
    result = composition_extras(
        pieces, deepcopy(pieces), [old_parent], [old_parent, old_extra], generator
    )

    assert result.bypassed is False
    assert result.eligible_piece_count == 2
    assert result.retained_combination_count == 3
    assert [path[0] for path in result.paths] == [
        "2bungi 10년간", "2분기 10nyeongan", "2분기 10년간",
    ]
    assert (pieces, old_parent, old_extra) == before
    for _, score, segments in result.paths:
        assert score == 3.25
        assert segments[0]["input_start"] == 0
        assert segments[-1]["input_end"] == len("2bungi 10nyeongan")
        cursor = 0
        for segment in segments:
            assert segment["input_start"] == cursor
            cursor = segment["input_end"]
            if segment["kind"] == "raw":
                assert segment["text"] == segment["source"]
            elif segment["kind"] == "numeric":
                assert romanize_word(segment["text"]) == segment["source"]
        for index, segment in enumerate(segments):
            if segment["kind"] == "numeric":
                prefix = segments[index - 1]
                assert prefix["kind"] == "raw"
                assert prefix["source"] == prefix["text"]
                assert prefix["source"].replace(".", "").isdigit()


def test_unicode_tuple_order_combination_cap_parent_first_and_path_cap():
    pieces = []
    cursor = 0
    sources = []
    for index in range(1, 5):
        source = str(index) + "inseung"
        sources.append(source)
        pieces.append(piece(source, cursor))
        cursor += len(source)
    pieces.append(piece("x", cursor))
    # A deterministic fake is simpler and isolates full-path enumeration.
    proposals = {
        source: (f"{index}인세웅", f"{index}인승")
        for index, source in enumerate(sources, 1)
    }
    generator = StubGenerator({})
    import hanglish.core.numeric_composition as module
    original = module.numeric_suffix_proposals
    module.numeric_suffix_proposals = lambda source, *_: proposals[source]
    try:
        parents = [parent(pieces, str(index)) for index in range(4)]
        for index, path in enumerate(parents):
            path[2][-1]["text"] = "가나다라"[index]
            path[2][-1]["kind"] = "lexicon"
            path = ("".join(segment["text"] for segment in path[2]), path[1], path[2])
            parents[index] = path
        result = composition_extras(pieces, deepcopy(pieces), parents, parents, generator)
    finally:
        module.numeric_suffix_proposals = original

    assert result.retained_combination_count == 16
    assert len(result.paths) == 48
    assert all(path[2][-1]["tag"] == "0" for path in result.paths[:16])
    assert all(path[2][-1]["tag"] == "1" for path in result.paths[16:32])
    assert all(path[2][-1]["tag"] == "2" for path in result.paths[32:])


def test_fifth_eligible_piece_bypasses_and_alignment_is_strict():
    generator = StubGenerator({})
    pieces = []
    cursor = 0
    for index in range(5):
        source = f"{index + 1}bungi"
        pieces.append(piece(source, cursor))
        cursor += len(source)
    preserved = [parent(pieces)]
    result = composition_extras(pieces, deepcopy(pieces), preserved, preserved, generator)
    assert result.bypassed is True
    assert result.eligible_piece_count == 5
    assert result.paths == ()

    bad_static = deepcopy(pieces)
    bad_static[0] = (bad_static[0][0], 1, bad_static[0][2])
    with pytest.raises(ValueError, match="alignment"):
        composition_extras(pieces, bad_static, preserved, preserved, generator)


def test_non_raw_eligibility_and_existing_output_dedup():
    generator = StubGenerator({"bun": (option("분"),), "gi": (option("기"),)})
    pieces = [piece("2bungi", 0)]
    non_raw = [("2bungi", 0, (option("2분기"),))]
    preserved = [parent(pieces)]
    assert composition_extras(pieces, non_raw, preserved, preserved, generator).paths == ()

    existing = ("2분기", 7.0, [{"old": True}])
    result = composition_extras(pieces, deepcopy(pieces), preserved, [*preserved, existing], generator)
    assert result.paths == ()
