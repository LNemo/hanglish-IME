from copy import deepcopy
from dataclasses import dataclass
import math

import pytest

from hanglish.core.copy_child_proposals import child_requests, propose_children


@dataclass(frozen=True)
class Option:
    text: str
    kind: str


class FakeGenerator:
    def __init__(self, suffixes, options):
        self.english_suffixes = list(suffixes)
        self.values = {key: list(value) for key, value in options.items()}

    def options(self, source):
        return self.values.get(source, ())


def segment(source, text, kind="mixed", start=0):
    return {"input_start": start, "input_end": start + len(source),
            "source": source, "text": text, "kind": kind}


def widget_generator(source="widgetreul", target="widget를"):
    return FakeGenerator([("reul", "를")],
                         {source: [Option(source, "raw"),
                                   Option(target, "mixed")]})


def test_discovers_convert_child_before_model_and_records_normalization():
    result = child_requests(
        "widgetreul", "widget를", [segment("widgetreul", "widget를")],
        generator=widget_generator())

    assert result["requests"] == [{
        "request_index": 0, "parent_index": 0, "segment_index": 1,
        "input_start": 6, "input_end": 10, "source": "reul",
        "model_source": "reul",
        "normalization": {"operation": "ascii_lower",
                          "scope": "convert_child_only",
                          "input": "reul", "output": "reul"},
    }]
    assert result["refinement"]["segments"][0]["text"] == "widget"


def test_alternative_forward_valid_suffix_is_one_child_proposal():
    result = propose_children(
        "widgetreul", "widget를", [segment("widgetreul", "widget를")],
        generator=widget_generator(), generated={"reul": [("를", -0.1),
                                                           ("륿", -0.2)]})

    assert [row["text"] for row in result["proposals"]] == ["widget륿"]
    proposal = result["proposals"][0]
    assert proposal["segments"] == [
        segment("widget", "widget", "raw"),
        segment("reul", "륿", "copy_child_phonetic", 6),
    ]
    assert proposal["source_span"] == [6, 10]
    assert proposal["replacement"] == {"from": "를", "to": "륿"}
    assert proposal["phonetic_log_probability"] == -0.2
    assert "score" not in proposal and "coarse_score" not in proposal


def test_equal_probability_beam_preserves_decoder_order_without_lexical_tiebreak():
    result = propose_children(
        "widgetreul", "widget를", [segment("widgetreul", "widget를")],
        generator=widget_generator(), generated={"reul": [("릀", -0.2),
                                                           ("륿", -0.2)]})
    assert [row["replacement"]["to"] for row in result["proposals"]] == [
        "릀", "륿"]


def test_generated_output_cannot_rewrite_copied_prefix():
    with pytest.raises(ValueError, match="Hangul|malformed"):
        propose_children(
            "widgetreul", "widget를", [segment("widgetreul", "widget를")],
            generator=widget_generator(), generated={"reul": [("widget를", 0.0)]})


def test_parent_raw_and_convert_interpretations_remain_available():
    generator = widget_generator()
    rows = [segment("widgetreul", "widget를")]
    result = propose_children(
        "widgetreul", "widget를", rows, generator=generator,
        generated={"reul": [("륿", -1.0)]},
        existing_texts=["widgetreul"])
    assert [option.text for option in generator.values["widgetreul"]] == [
        "widgetreul", "widget를"]
    assert result["refinement"]["coarse_segments"] == rows
    assert result["proposals"][0]["text"] == "widget륿"


def test_uppercase_normalizes_only_convert_child():
    generator = widget_generator("WIDGETREUL", "WIDGET를")
    result = propose_children(
        "WIDGETREUL", "WIDGET를", [segment("WIDGETREUL", "WIDGET를")],
        generator=generator, generated={"reul": [("륿", -1.0)]})
    request = result["requests"][0]
    assert request["source"] == "REUL" and request["model_source"] == "reul"
    assert request["normalization"]["scope"] == "convert_child_only"
    assert result["proposals"][0]["text"] == "WIDGET륿"


def test_one_character_suffix_is_supported():
    generator = FakeGenerator([("i", "이")],
                              {"widgeti": [Option("widget이", "mixed")]})
    result = child_requests(
        "widgeti", "widget이", [segment("widgeti", "widget이")],
        generator=generator)
    assert result["requests"][0]["source"] == "i"
    assert result["requests"][0]["input_start"] == 6


def test_forty_letter_child_is_supported_and_longer_child_is_skipped():
    forty = "a" * 40
    source = "xx" + forty
    generator = FakeGenerator([(forty, "가")],
                              {source: [Option("xx가", "mixed")]})
    result = child_requests(source, "xx가", [segment(source, "xx가")],
                            generator=generator)
    assert result["requests"][0]["model_source"] == forty

    forty_one = "a" * 41
    source = "xx" + forty_one
    generator = FakeGenerator([(forty_one, "가")],
                              {source: [Option("xx가", "mixed")]})
    result = child_requests(source, "xx가", [segment(source, "xx가")],
                            generator=generator)
    assert result["requests"] == []
    assert result["diagnostics"]["skipped_children"][0]["reason"].startswith(
        "convert_child_is_not")


def test_empty_beam_is_valid_and_explicit():
    result = propose_children(
        "widgetreul", "widget를", [segment("widgetreul", "widget를")],
        generator=widget_generator(), generated={"reul": []})
    assert result["proposals"] == []
    assert result["diagnostics"]["generation"][0]["status"] == "empty_beam"


def test_unicode_source_offsets_are_codepoints():
    generator = FakeGenerator([("reul", "를")],
                              {"éAPIreul": [Option("éAPI를", "mixed")]})
    result = propose_children(
        "éAPIreul", "éAPI를", [segment("éAPIreul", "éAPI를")],
        generator=generator, generated={"reul": [("륿", -1.0)]})
    assert result["requests"][0]["input_start"] == 4
    assert result["proposals"][0]["source_span"] == [4, 8]
    assert result["proposals"][0]["text"] == "éAPI륿"


def test_multiple_children_replace_one_at_a_time_and_preserve_other_copy():
    generator = FakeGenerator(
        [("reul", "를"), ("eul", "을")],
        {"widgetreul": [Option("widget를", "mixed")],
         "boxeul": [Option("box을", "mixed")]})
    rows = [segment("widgetreul", "widget를"),
            segment(" ", " ", "raw", 10),
            segment("boxeul", "box을", "mixed", 11)]
    result = propose_children(
        "widgetreul boxeul", "widget를 box을", rows, generator=generator,
        generated={"reul": [("륿", -1.0)], "eul": [("읇", -1.0)]})
    assert [row["text"] for row in result["proposals"]] == [
        "widget륿 box을", "widget를 box읇"]
    first, second = result["proposals"]
    assert first["segments"][0] == segment("widget", "widget", "raw")
    assert first["segments"][3] == segment("box", "box", "raw", 11)
    assert second["segments"][0] == segment("widget", "widget", "raw")
    assert second["segments"][3] == segment("box", "box", "raw", 11)
    assert "".join(row["text"] for row in first["segments"]) == first["text"]


@pytest.mark.parametrize("beam,match", [
    ([("를", -0.1), ("를", -0.2)], "duplicates"),
    ([("를", -0.2), ("륿", -0.1)], "sorted"),
    ([("를", math.nan)], "malformed"),
    ([("를", True)], "malformed"),
    ([("를", 0.0, "extra")], "malformed"),
    ([("가", 0.0)], "forward romanization"),
])
def test_malformed_duplicate_nonfinite_unsorted_or_wrong_forward_beams_fail(beam, match):
    with pytest.raises(ValueError, match=match):
        propose_children(
            "widgetreul", "widget를", [segment("widgetreul", "widget를")],
            generator=widget_generator(), generated={"reul": beam})


def test_missing_generation_is_an_error_not_silent_empty():
    with pytest.raises(ValueError, match="missing generated beam"):
        propose_children(
            "widgetreul", "widget를", [segment("widgetreul", "widget를")],
            generator=widget_generator(), generated={})


def test_unresolved_parent_has_no_request_and_remains_in_metadata():
    generator = FakeGenerator([("reul", "를")], {"widgetreul": []})
    result = propose_children(
        "widgetreul", "widget를", [segment("widgetreul", "widget를")],
        generator=generator, generated={})
    assert result["requests"] == [] and result["proposals"] == []
    assert result["refinement"]["derivations"][0]["status"] == "unresolved"
    assert result["diagnostics"]["unresolved_parents"][0]["parent_index"] == 0


def test_inputs_are_not_mutated_and_existing_text_is_respected():
    generator = widget_generator()
    rows = [segment("widgetreul", "widget를")]
    generated = {"reul": [("륿", -1.0)]}
    before = deepcopy((rows, generated, generator.values))
    result = propose_children(
        "widgetreul", "widget를", rows, generator=generator,
        generated=generated, existing_texts=("widget륿",))
    assert result["proposals"] == []
    assert (rows, generated, generator.values) == before


def test_existing_texts_rejects_a_bare_string():
    with pytest.raises(TypeError, match="iterable of strings"):
        propose_children(
            "widgetreul", "widget를", [segment("widgetreul", "widget를")],
            generator=widget_generator(), generated={"reul": []},
            existing_texts="widget륿")


@pytest.mark.parametrize("limit", [0, -1, 97, True, 1.5, "1"])
def test_limit_validation(limit):
    with pytest.raises(ValueError, match="limit"):
        propose_children(
            "widgetreul", "widget를", [segment("widgetreul", "widget를")],
            generator=widget_generator(), generated={"reul": []}, limit=limit)


def test_limit_is_global_and_deterministic():
    result = propose_children(
        "widgetreul", "widget를", [segment("widgetreul", "widget를")],
        generator=widget_generator(),
        generated={"reul": [("륿", -0.1), ("릀", -0.2)]}, limit=1)
    assert [row["text"] for row in result["proposals"]] == ["widget륿"]
    assert result["diagnostics"]["limit_reached"] is True
