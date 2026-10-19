import copy

from hanglish.core.lattice import WordOption
from hanglish.core.mixed_span_proposals import chunk_paths, sentence_extras


MAPPING = {
    "je": [("제", 8.0)],
    "hoe": [("회", 7.0)],
    "bunkke": [("분께", 9.0)],
    "nyeoneul": [("년을", 9.0)],
    "ga": [("가", 8.0)],
    "ssireul": [("씨를", 9.0)],
    "person": [("퍼손", 4.0)],
}


def fake_options(source, learned=True):
    assert learned is True
    result = [WordOption(text, score, "fake") for text, score in MAPPING.get(source, [])]
    result.append(WordOption("무효", 100.0, "fake"))
    result.append(WordOption(source, 0.0, "raw"))
    return tuple(result)


def texts(source):
    return [path[0] for path in chunk_paths(source, 0, fake_options)]


def test_required_mixed_digit_boundaries_convert_forward_valid_letters():
    assert texts("je23hoe")[0] == "제23회"
    assert texts("45bunkke")[0] == "45분께"
    assert texts("‘22nyeoneul")[0] == "‘22년을"
    assert texts("12㎞ga")[0] == "12㎞가"
    assert texts("person(42)ssireul")[0] == "person(42)씨를"
    assert all(len(chunk_paths(value, 0, fake_options)) <= 4
               for value in ("je23hoe", "person(42)ssireul"))


def test_literals_case_units_and_offsets_are_preserved_exactly():
    path = chunk_paths("12㎞ga", 7, fake_options)[0]
    assert path[0] == "12㎞가"
    assert path[2] == [
        {"input_start": 7, "input_end": 10, "source": "12㎞",
         "text": "12㎞", "kind": "raw"},
        {"input_start": 10, "input_end": 12, "source": "ga",
         "text": "가", "kind": "fake"},
    ]
    assert texts("12kg+ga")[0] == "12kg+가"
    assert chunk_paths("12kg", 0, fake_options) == []
    assert chunk_paths("12ABC", 0, fake_options) == []


def test_protected_url_email_code_filename_version_and_separators_skip():
    protected = (
        "https://x2.test", "www.x2.test", "me2@example.com", "`x2`",
        "file_2ga", "dir/2ga", "dir\\2ga", "report2.txt", "v1.2.3",
        "HTTP2", "ABC-42", "abc123",
    )
    assert all(chunk_paths(value, 0, fake_options) == [] for value in protected)


def test_limits_long_or_over_fragmented_chunks_and_preserves_tonne_unit():
    assert chunk_paths("1" + "a" * 97, 0, fake_options) == []
    fragmented = "1" + "-a" * 25
    assert chunk_paths(fragmented, 0, fake_options) == []
    assert chunk_paths("12t", 0, fake_options) == []


def test_sentence_extras_replaces_repeated_raw_segments_without_mutating_parent():
    source = "je23hoe wa je23hoe"
    segments = [
        {"input_start": 0, "input_end": 7, "source": "je23hoe",
         "text": "je23hoe", "kind": "raw"},
        {"input_start": 7, "input_end": 11, "source": " wa ",
         "text": " wa ", "kind": "raw"},
        {"input_start": 11, "input_end": 18, "source": "je23hoe",
         "text": "je23hoe", "kind": "raw"},
    ]
    parents = [(source, 3.5, segments)]
    frozen = copy.deepcopy(parents)
    extras = sentence_extras(source, parents, fake_options)
    assert parents == frozen
    assert extras[0][0] == "제23회 wa 제23회"
    assert extras[0][1] == 3.5
    assert [(item["input_start"], item["input_end"])
            for item in extras[0][2]] == [
                (0, 2), (2, 4), (4, 7), (7, 11),
                (11, 13), (13, 15), (15, 18),
            ]
    assert all(item[0] != source for item in extras)


def test_first_best_variant_from_each_parent_precedes_alternates_and_deduplicates():
    source = "x je23hoe"
    base = [
        {"input_start": 0, "input_end": 1, "source": "x",
         "text": "가", "kind": "fake"},
        {"input_start": 1, "input_end": 2, "source": " ",
         "text": " ", "kind": "raw"},
        {"input_start": 2, "input_end": 9, "source": "je23hoe",
         "text": "je23hoe", "kind": "raw"},
    ]
    parent_a = ("가 je23hoe", 10.0, copy.deepcopy(base))
    other = copy.deepcopy(base)
    other[0]["text"] = "나"
    parent_b = ("나 je23hoe", 9.0, other)
    extras = sentence_extras(source, [parent_a, parent_b], fake_options, limit=3)
    assert [path[0] for path in extras[:2]] == ["가 제23회", "나 제23회"]
    assert len(extras) == 3
    assert len({path[0] for path in extras}) == len(extras)
    assert all(path[0] not in {parent_a[0], parent_b[0]} for path in extras)


def test_nonraw_or_nonmatching_raw_parent_segments_are_never_changed_and_bounds_hold():
    source = "je23hoe"
    ineligible = [
        ("제23회", 1.0, [{"input_start": 0, "input_end": 7,
                           "source": source, "text": "제23회", "kind": "mixed"}]),
    ]
    assert sentence_extras(source, ineligible, fake_options) == []
    parents = [
        (source, float(index), [
            {"input_start": 0, "input_end": 7, "source": source,
             "text": source, "kind": "raw"}
        ]) for index in range(12)
    ]
    extras = sentence_extras(source, parents, fake_options, limit=5)
    assert len(extras) <= 5


def test_duplicate_parent_spans_share_one_chunk_lookup_and_outputs_are_copied():
    source = "x je23hoe"
    calls = []
    def counted(source, learned=True):
        calls.append(source)
        return fake_options(source, learned)
    def parent(prefix):
        return (prefix + " je23hoe", 1.0, [
            {"input_start": 0, "input_end": 1, "source": "x",
             "text": prefix, "kind": "fake"},
            {"input_start": 1, "input_end": 2, "source": " ",
             "text": " ", "kind": "raw"},
            {"input_start": 2, "input_end": 9, "source": "je23hoe",
             "text": "je23hoe", "kind": "raw"},
        ])
    parents = [parent("가"), parent("나"), parent("다")]
    frozen = copy.deepcopy(parents)
    extras = sentence_extras(source, parents, counted, limit=4)
    assert calls == ["je", "hoe"]
    assert parents == frozen and len(extras) == 4
    assert [path[0] for path in extras[:3]] == [
        "가 제23회", "나 제23회", "다 제23회",
    ]
    assert extras[0][2] is not extras[1][2]
    assert extras[0][2][-1] is not extras[1][2][-1]
