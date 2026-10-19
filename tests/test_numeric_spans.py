from types import SimpleNamespace

import pytest

from hanglish.core.lattice import WordOption
from hanglish.core.neural_converter import NeuralConverter


@pytest.fixture(scope="module")
def converter():
    return NeuralConverter(context_ranking=False, external_lexicon=False)


@pytest.mark.parametrize("source,expected", [
    ("12il", "12일"),
    ("5nyeon", "5년"),
    ("20segi", "20세기"),
    ("30yeojeomeul", "30여점을"),
    ("3.5won", "3.5원"),
    ("12manmyeong", "12만명"),
    ("5nyeonman", "5년만"),
    ("3myeongui", "3명의"),
])
def test_number_followed_by_supported_korean_unit_is_a_candidate(converter, source, expected):
    result = converter.convert(source, engine="morphology")

    assert expected in [candidate["text"] for candidate in result["candidates"]]
    assert source in [candidate["text"] for candidate in result["candidates"]]
    expected_candidate = next(candidate for candidate in result["candidates"] if candidate["text"] == expected)
    assert "".join(segment["source"] for segment in expected_candidate["segments"]) == source
    assert "".join(segment["text"] for segment in expected_candidate["segments"]) == expected


@pytest.mark.parametrize("source", [
    "10kg", "60fps", "3d", "v1.2.3", "192.168.0.1", "0xff", "1e10",
    "https://example.com/v2", "user2@example.com", "main.py:42", "`12il`",
    "12abc", "3nyeon2gaewol",
])
def test_numeric_code_measurement_and_repeated_spans_stay_raw(converter, source):
    result = converter.convert(source, engine="morphology")

    assert result["result"] == source
    assert [candidate["text"] for candidate in result["candidates"]] == [source]


def test_numeric_unit_split_preserves_punctuation_and_exact_offsets(converter):
    source = "약 12il, 5nyeon!"
    result = converter.convert(source, engine="morphology")
    candidate = next(candidate for candidate in result["candidates"] if candidate["text"] == "약 12일, 5년!")

    assert "".join(segment["source"] for segment in candidate["segments"]) == source
    assert "".join(segment["text"] for segment in candidate["segments"]) == candidate["text"]
    for segment in candidate["segments"]:
        assert source[segment["input_start"]:segment["input_end"]] == segment["source"]


@pytest.mark.parametrize("source", [
    "3gaeeul", "3gaeeun", "3myeongneun", "3myeongga", "3gaeeuro", "30yakjeom",
])
def test_invalid_unit_particle_and_postnumber_yak_stay_raw(converter, source):
    assert converter.convert(source, engine="morphology")["result"] == source


@pytest.mark.parametrize("surface", ["개를", "개는", "명이", "명을", "개로", "개월로", "명으로"])
def test_valid_unit_particle_allomorphs(surface):
    from hanglish.core.neural_converter import numeric_unit_surface
    assert numeric_unit_surface(surface)


def test_strong_raw_english_evidence_keeps_numeric_chunk_protected():
    converter = object.__new__(NeuralConverter)
    options = (
        WordOption("일", 7., "lexicon"),
        WordOption("il", 7.5, "raw"),
    )
    converter.generator = SimpleNamespace(options=lambda source: options)
    converter.word_options = lambda source, learned: options

    assert converter._numeric_unit_pieces("12il", 0, learned=False) is None


@pytest.fixture(scope='module')
def counter_converter():
    return NeuralConverter(context_ranking=False, external_lexicon=True)


@pytest.mark.parametrize('source,expected', [('1juire','1주일에'),('1wi','1위'),
    ('5deunggeubeuro','5등급으로'),('3gagu','3가구'),('2hangnyeon','2학년')])
def test_official_counter_grammar_generates_numeric_candidates(counter_converter, source, expected):
    result = counter_converter.convert(source, engine='morphology')
    assert expected in [c['text'] for c in result['candidates']]
    assert source in [c['text'] for c in result['candidates']]
    selected = next(c for c in result['candidates'] if c['text']==expected)
    assert ''.join(s['source'] for s in selected['segments']) == source
    for segment in selected['segments']:
        assert source[segment['input_start']:segment['input_end']] == segment['source']


def test_disabled_external_lexicon_keeps_builtin_unit_grammar(converter, counter_converter):
    assert '주일' not in converter.numeric_units
    assert '주일' in counter_converter.numeric_units
    assert '주일' in counter_converter.numeric_units[:counter_converter.numeric_units.index('주')]


@pytest.mark.parametrize('source', ['12hours','5test','4API','3gagueul','3hangnyeonneun',
                                  '10kg','3nyeon2gaewol','https://example.com/v2'])
def test_extended_counter_grammar_preserves_english_codes_and_invalid_particles(counter_converter, source):
    assert counter_converter.convert(source, engine='morphology')['result'] == source
