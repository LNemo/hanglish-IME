from copy import deepcopy
import math

import pytest

from hanglish.core.ungated_phonetic_proposals import generate_proposals


def segment(source, text=None, *, start=0, kind="raw"):
    return {
        "input_start": start,
        "input_end": start + len(source),
        "source": source,
        "text": source if text is None else text,
        "kind": kind,
    }


class Generator:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def __call__(self, source, beam_size=8):
        self.calls.append((source, beam_size))
        return deepcopy(self.values.get(source, []))


def test_single_parent_generates_without_any_ranker_or_score():
    parents = [{"text": "gim", "segments": [segment("gim")]}]
    generator = Generator({"gim": [("김", -0.1)]})

    result = generate_proposals("gim", parents, generator)

    assert generator.calls == [("gim", 8)]
    assert result["generation_requests"] == 1
    assert result["generated_by_source"] == {"gim": [("김", -0.1)]}
    assert result["proposals"] == [{
        "text": "김",
        "segments": [segment("gim", "김", kind="ungated_phonetic_word")],
        "parent_index": 0,
        "source_start": 0,
        "source_end": 3,
        "source": "gim",
        "replacement": "김",
    }]
    assert "score" not in result["proposals"][0]


def test_long_candidate_is_emitted_without_tokenizer_length_gate():
    prefix = "한" * 5000
    source = "x " + "gim"
    parents = [{
        "text": prefix + " gim",
        "segments": [segment("x ", prefix + " "), segment("gim", start=2)],
    }]

    result = generate_proposals(
        source, parents, Generator({"gim": [("김", -0.1)]}))

    assert result["proposals"][0]["text"] == prefix + " 김"
    assert len(result["proposals"][0]["text"]) > 4096


def test_repeated_source_is_generated_once_and_reused_per_occurrence():
    source = "gim gim"
    parents = [{
        "text": source,
        "segments": [segment("gim"), segment(" ", start=3),
                     segment("gim", start=4)],
    }]
    generator = Generator({"gim": [("김", -0.1)]})

    result = generate_proposals(source, parents, generator)

    assert generator.calls == [("gim", 8)]
    assert [row["text"] for row in result["proposals"]] == ["김 gim", "gim 김"]


def test_source_and_extra_caps_are_bounded_and_audited():
    source = "ga na da"
    parents = [{
        "text": source,
        "segments": [segment("ga"), segment(" ", start=2),
                     segment("na", start=3), segment(" ", start=5),
                     segment("da", start=6)],
    }]
    generator = Generator({
        "ga": [("가", -0.1)],
        "na": [("나", -0.1)],
        "da": [("다", -0.1)],
    })

    result = generate_proposals(
        source, parents, generator, max_sources=2, max_extras=2)

    assert generator.calls == [("ga", 8), ("na", 8)]
    assert [row["text"] for row in result["proposals"]] == ["가 na da", "ga 나 da"]
    assert result["skipped_generation_sources"] == [{
        "source_start": 6, "source_end": 8, "source": "da"}]
    assert result["generation_budget_exhausted"] is True


@pytest.mark.parametrize("parents", [
    [{"text": "gim", "segments": [segment("gim", start=1)]}],
    [{"text": "wrong", "segments": [segment("gim")]}],
    [{"text": "gim", "segments": [
        {"input_start": 0, "input_end": 3, "source": "bad",
         "text": "gim", "kind": "raw"}]}],
])
def test_invalid_parent_offsets_or_rendering_fail_closed(parents):
    with pytest.raises(ValueError, match="source offsets|cover source|render"):
        generate_proposals("gim", parents, Generator({}))


@pytest.mark.parametrize("beam,match", [
    ([("김", -0.1)] * 9, "at most"),
    ([("김", -0.1), ("김", -0.2)], "duplicates"),
    ([("김", -0.2), ("깅", -0.1)], "sorted"),
    ([("김", math.nan)], "malformed"),
    ([("김", True)], "malformed"),
    ([("한", -0.1)], "forward romanization"),
    ([("gim", -0.1)], "malformed"),
])
def test_invalid_beam_fails_closed(beam, match):
    with pytest.raises(ValueError, match=match):
        generate_proposals(
            "gim", [{"text": "gim", "segments": [segment("gim")]}],
            Generator({"gim": beam}))


def test_inputs_and_mixed_ascii_are_preserved():
    source = "test gim"
    parents = [{
        "text": "test gim",
        "segments": [segment("test"), segment(" ", start=4),
                     segment("gim", start=5)],
        "metadata": {"kept": True},
    }]
    before = deepcopy(parents)
    generator = Generator({"test": [("테스트", -0.1)], "gim": [("김", -0.1)]})

    result = generate_proposals(
        source, parents, generator, known_english={"TEST"})

    assert parents == before
    assert result["parents"] == before and result["parents"] is not parents
    assert generator.calls == [("gim", 8)]
    assert result["skipped_known_english"] == [{
        "source_start": 0, "source_end": 4, "source": "test"}]
    assert result["proposals"][0]["text"] == "test 김"
    assert result["proposals"][0]["segments"][0] == parents[0]["segments"][0]


def test_parent_segment_and_beam_order_are_stable_and_outputs_are_unique():
    source = "gim gat"
    first = [segment("gim"), segment(" ", start=3), segment("gat", start=4)]
    second = deepcopy(first)
    second[0]["text"] = "김"
    parents = [{"text": "gim gat", "segments": first},
               {"text": "김 gat", "segments": second}]
    generator = Generator({
        "gim": [("김", -0.1)],
        "gat": [("갇", -0.1), ("갓", -0.2)],
    })

    result = generate_proposals(source, parents, generator)

    assert [row["text"] for row in result["proposals"]] == [
        "gim 갇", "gim 갓", "김 갇", "김 갓"]
    assert [row["parent_index"] for row in result["proposals"]] == [0, 0, 1, 1]


@pytest.mark.parametrize("name,value", [
    ("max_sources", 0), ("max_sources", 97), ("max_sources", True),
    ("max_extras", 0), ("max_extras", 97), ("max_extras", 1.5),
])
def test_limits_are_validated(name, value):
    kwargs = {name: value}
    with pytest.raises(ValueError, match=name):
        generate_proposals("", [{"text": "", "segments": []}],
                           Generator({}), **kwargs)
