import threading
from types import SimpleNamespace

import pytest

from hanglish.core.ctc_union_converter import CTCUnionConverter
from hanglish.core.phonetic_word_union_converter import PhoneticWordUnionConverter


class Generator:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def generate(self, source, beam_size=8):
        self.calls.append((source, beam_size))
        return list(self.values.get(source, []))


class Ranker:
    def __init__(self, scores=None, rejected=()):
        self.scores = scores
        self.rejected = set(rejected)
        self.can_score_calls = []
        self.score_calls = []

    def can_score(self, text):
        self.can_score_calls.append(text)
        return text not in self.rejected

    def score_pairs(self, contexts, texts):
        self.score_calls.append((list(contexts), list(texts)))
        if self.scores is not None:
            return list(self.scores)
        return [20.0 - index for index in range(len(texts))]


def raw_segments(source):
    return [{"input_start": 0, "input_end": len(source), "source": source,
             "text": source, "kind": "raw"}]


def token_segments(source):
    result = []
    cursor = 0
    for index, token in enumerate(source.split(" ")):
        if index:
            result.append({"input_start": cursor, "input_end": cursor + 1,
                           "source": " ", "text": " ", "kind": "protected"})
            cursor += 1
        result.append({"input_start": cursor, "input_end": cursor + len(token),
                       "source": token, "text": token, "kind": "raw"})
        cursor += len(token)
    return result


def converter(generator, ranker=None):
    value = object.__new__(PhoneticWordUnionConverter)
    value.lock = threading.RLock()
    value.phonetic_model = generator
    value._phonetic_model_loaded = True
    value.sentence_ranker = ranker or Ranker()
    value.phonetic_word_trace = value._empty_trace()
    value.phonetic_word_ms = 0.0
    return value


def test_parent_runs_once_and_single_span_replacements_preserve_bytes_offsets(monkeypatch):
    source = "보호 gim / han!"
    segments = [
        {"input_start": 0, "input_end": 3, "source": "보호 ", "text": "보호 ",
         "kind": "protected"},
        {"input_start": 3, "input_end": 6, "source": "gim", "text": "gim", "kind": "raw"},
        {"input_start": 6, "input_end": 9, "source": " / ", "text": " / ",
         "kind": "protected"},
        {"input_start": 9, "input_end": 13, "source": "han!", "text": "han!", "kind": "raw"},
    ]
    parent = [(source, 7.0, segments)]
    calls = []
    monkeypatch.setattr(
        CTCUnionConverter, "_extend_sentence_paths",
        lambda self, text, context, paths: calls.append((text, context, paths)) or parent)
    model = Generator({"gim": [("김", -0.1)]})
    ranker = Ranker()
    value = converter(model, ranker)

    result = value._extend_sentence_paths(source, "문맥", [("ignored", 0.0, [])])

    assert len(calls) == 1
    assert model.calls == [("gim", 8)]
    assert parent[0] in result
    extra = next(row for row in result if row[0] == "보호 김 / han!")
    assert extra[1] == 20.0
    assert extra[2][0] == segments[0] and extra[2][2:] == segments[2:]
    assert [(row["input_start"], row["input_end"], row["source"])
            for row in extra[2]] == [(row["input_start"], row["input_end"], row["source"])
                                   for row in segments]
    assert extra[2][1]["kind"] == "phonetic_word_union"
    assert ranker.score_calls == [(["문맥"], ["보호 김 / han!"])]
    assert value.phonetic_word_trace["production_pool_texts"] == [source]


def test_uppercase_punctuation_and_non_ascii_segments_never_generate(monkeypatch):
    source = "Test 한글 https://x"
    parent = [(source, 3.0, token_segments(source))]
    monkeypatch.setattr(CTCUnionConverter, "_extend_sentence_paths",
                        lambda self, *args: parent)
    model = Generator({})
    value = converter(model)
    assert value._extend_sentence_paths(source, "", []) is parent
    assert model.calls == []
    assert value.sentence_ranker.score_calls == []


def test_artifact_identity_pins_batched_decoder(tmp_path):
    value = object.__new__(PhoneticWordUnionConverter)
    value.phonetic_checkpoint = tmp_path / "not-loaded.pt"
    artifacts = value._phonetic_artifact_identity()
    assert "hanglish/models/batched_phonetic_decode.py" in artifacts
    assert len(artifacts["hanglish/models/batched_phonetic_decode.py"]) == 64


def test_known_english_and_hangul_are_preserved_without_model_call(monkeypatch):
    source = "test 한글"
    parent = [(source, 3.0, token_segments(source))]
    monkeypatch.setattr(CTCUnionConverter, "_extend_sentence_paths",
                        lambda self, *args: parent)
    model = Generator({"test": [("가", -0.1)]})
    value = converter(model)
    value.generator = SimpleNamespace(english={"test"})
    assert value._extend_sentence_paths(source, "", []) is parent
    assert model.calls == []
    assert value.phonetic_word_trace["generated_extras"] == []
    assert value.phonetic_word_trace["skipped_known_english"] == [
        {"source_start": 0, "source_end": 4, "source": "test"}]


def test_existing_english_particle_path_survives_whole_word_hangul_alternative(monkeypatch):
    source = "apireul"
    segments = [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": "api를", "kind": "mixed",
    }]
    parent_path = ("api를", 7.25, segments)
    parent = [parent_path]
    monkeypatch.setattr(CTCUnionConverter, "_extend_sentence_paths",
                        lambda self, *args: parent)
    model = Generator({source: [("아피를", -0.1)]})
    value = converter(model, Ranker([8.5]))
    value.generator = SimpleNamespace(english={"api"})

    result = value._extend_sentence_paths(source, "", [])

    assert parent_path in result
    retained = next(row for row in result if row[0] == "api를")
    assert retained[1] == 7.25 and retained[2] is segments
    assert retained[2] == [{
        "input_start": 0, "input_end": len(source), "source": source,
        "text": "api를", "kind": "mixed",
    }]
    alternative = next(row for row in result if row[0] == "아피를")
    assert alternative[1] == 8.5
    assert alternative[2][0] == {
        "input_start": 0, "input_end": len(source), "source": source,
        "text": "아피를", "kind": "phonetic_word_union",
    }
    assert model.calls == [(source, 8)]


def test_duplicate_outputs_and_existing_parent_text_are_retained_once(monkeypatch):
    source = "gim han"
    first = token_segments(source)
    existing_segments = [dict(row) for row in first]
    existing_segments[0]["text"] = "김"
    parent = [(source, 5.0, first), ("김 han", 4.0, existing_segments)]
    monkeypatch.setattr(CTCUnionConverter, "_extend_sentence_paths",
                        lambda self, *args: parent)
    model = Generator({"gim": [("김", -0.1)], "han": [("한", -0.2)]})
    value = converter(model)
    result = value._extend_sentence_paths(source, "", [])
    texts = [row[0] for row in result]
    assert texts.count("김 han") == 1
    assert set(texts) == {"gim han", "김 han", "gim 한", "김 한"}
    assert all(row in result for row in parent)
    assert len(value.sentence_ranker.score_calls) == 1
    assert value.sentence_ranker.score_calls[0][1] == ["gim 한", "김 한"]


def test_generated_pool_is_capped_at_96_and_scored_once(monkeypatch):
    source = " ".join(["ga"] * 97)
    parent = [(source, 1.0, token_segments(source))]
    monkeypatch.setattr(CTCUnionConverter, "_extend_sentence_paths",
                        lambda self, *args: parent)
    model = Generator({"ga": [("가", -0.1)]})
    value = converter(model)
    result = value._extend_sentence_paths(source, "", [])
    assert model.calls == [("ga", 8)]
    assert len(value.phonetic_word_trace["generated_extras"]) == 96
    assert len(value.phonetic_word_trace["scoreable_extras"]) == 96
    assert len(value.sentence_ranker.score_calls) == 1
    assert len(value.sentence_ranker.score_calls[0][1]) == 96
    assert len(result) == 97


def test_distinct_source_generation_requests_are_capped_at_96(monkeypatch):
    words = [chr(97 + first) + chr(97 + second)
             for first in range(4) for second in range(26)][:97]
    source = " ".join(words)
    parent = [(source, 1.0, token_segments(source))]
    monkeypatch.setattr(CTCUnionConverter, "_extend_sentence_paths",
                        lambda self, *args: parent)
    model = Generator({word: [] for word in words})
    value = converter(model)
    assert value._extend_sentence_paths(source, "", []) is parent
    assert len(model.calls) == 96
    assert [row[0] for row in model.calls] == words[:96]
    assert value.phonetic_word_trace["generation_requests"] == 96
    assert value.phonetic_word_trace["generation_budget_exhausted"] is True
    assert value.phonetic_word_trace["skipped_generation_sources"] == [{
        "source_start": source.rfind(words[-1]),
        "source_end": len(source), "source": words[-1]}]


def test_unscoreable_extras_are_traced_but_do_not_change_parent(monkeypatch):
    source = "gim"
    parent = [(source, 1.0, raw_segments(source))]
    monkeypatch.setattr(CTCUnionConverter, "_extend_sentence_paths",
                        lambda self, *args: parent)
    ranker = Ranker(rejected={"김"})
    value = converter(Generator({"gim": [("김", -0.1)]}), ranker)
    assert value._extend_sentence_paths(source, "", []) is parent
    assert value.phonetic_word_trace["generated_extras"][0]["text"] == "김"
    assert value.phonetic_word_trace["scoreable_extras"] == []
    assert ranker.score_calls == []


def test_invalid_generator_and_ranker_values_fail_closed(monkeypatch):
    source = "gim"
    parent = [(source, 1.0, raw_segments(source))]
    monkeypatch.setattr(CTCUnionConverter, "_extend_sentence_paths",
                        lambda self, *args: parent)
    with pytest.raises(RuntimeError, match="invalid candidate"):
        converter(Generator({"gim": [("한", -0.1)]}))._extend_sentence_paths(source, "", [])
    value = converter(Generator({"gim": [("김", -0.1)]}), Ranker([float("nan")]))
    with pytest.raises(RuntimeError, match="ranker returned invalid"):
        value._extend_sentence_paths(source, "", [])


def test_convert_trace_distinguishes_selected_and_missing(monkeypatch):
    value = converter(Generator({}))
    value.phonetic_parent_version = "parent"
    value.phonetic_artifacts = {"checkpoint": "hash"}

    def parent_convert(self, text, context, engine, mode, raw):
        self.phonetic_word_trace["generated_extras"] = [
            {"text": "선택"}, {"text": "누락"}]
        return {"candidates": [{"text": "선택"}], "result": "선택"}

    monkeypatch.setattr(CTCUnionConverter, "convert", parent_convert)
    response = value.convert("gim")
    trace = response["phonetic_word_union"]
    assert trace["public"] == ["선택"]
    assert trace["selected_extras"] == ["선택"]
    assert trace["missing_extras"] == ["누락"]
    assert trace["experimental"] is True
