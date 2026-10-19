from functools import lru_cache

from hanglish.core.lattice import WordOption
import scripts.trace_conversion_candidates as module


class FakeGenerator:
    english_suffixes = {}

    def __init__(self):
        self.calls = []

    @lru_cache(maxsize=32)
    def options(self, source):
        self.calls.append(source)
        values = {
            "ga": (WordOption("가", 6.0, "lexicon"), WordOption("ga", 0.0, "raw")),
            "na": (WordOption("나", 6.0, "lexicon"), WordOption("너", 5.8, "lexicon"),
                   WordOption("na", 0.0, "raw")),
        }
        return values.get(source, (WordOption(source, 0.0, "raw"),))


class FakeWordRanker:
    def score_pairs(self, contexts, candidates):
        return [{"가": 0.2, "나": 0.3, "너": 0.4}.get(value, 0.0)
                for value in candidates]


class FakeSentenceRanker:
    def can_score(self, text):
        return True

    def score_pairs(self, contexts, candidates):
        return [1.0] * len(candidates)


class FakeTransducer:
    def generate(self, source, beam_size=8):
        raise AssertionError("known fake words must not invoke neural generation")

    def generate_constrained(self, source, beam_size=8):
        raise AssertionError("known fake words must not invoke constrained generation")


def fake_converter():
    converter = module.TraceConverter(
        external_lexicon=False, ranker_kind="hybrid", context_ranking=True)
    converter.generator = FakeGenerator()
    converter._install_source_wrappers()
    converter.ranker = FakeWordRanker()
    converter.sentence_ranker = FakeSentenceRanker()
    converter.transducer = FakeTransducer()
    converter.models_loaded = True
    converter.model_artifacts = {"fake": "a" * 64}
    converter._install_model_recorders()
    return converter


def test_fake_rankers_trace_exact_runtime_parity_and_ties():
    trace = module.trace_conversion(fake_converter(), "ga na", "앞 문맥")
    assert trace["parity_verified"] is True
    assert trace["result"] == trace["displayed_candidates"][0]["text"]
    assert len(trace["word_ranker_score_pairs"]) == 1
    assert len(trace["sentence_score_pairs"]) == 1
    sentence = trace["sentence_score_pairs"][0]
    assert sentence["contexts"] == ["앞 문맥"] * len(sentence["candidates"])
    assert sentence["scores"] == [1.0] * len(sentence["candidates"])
    # Equal sentence scores use the runtime's text tie-break.
    ranked = [item["text"] for item in trace["final_pre_display_paths"]]
    assert ranked == sorted(ranked)
    assert "expected" not in trace and "labels" not in trace


def test_cached_word_options_calls_remain_visible():
    converter = fake_converter()
    first = module.trace_conversion(converter, "ga na")
    second = module.trace_conversion(converter, "ga na")
    assert first["word_option_computations"]
    assert second["word_option_computations"] == []
    assert len(second["word_option_calls"]) == 2
    assert all(call["cache_hit"] for call in second["word_option_calls"])
    assert second["mechanical_generator_calls"] == []


def test_explicit_protected_chunk_is_raw_and_never_converted():
    trace = module.trace_conversion(fake_converter(), "ga https://x.test na")
    assert trace["protected_chunks"] == [
        {"source": "https://x.test", "offset": 3, "preserved_raw": True}
    ]
    called = {item["source"] for item in trace["mechanical_generator_calls"]}
    called.update(item["source"] for item in trace["word_option_calls"])
    assert "https://x.test" not in called
    protected_piece = next(
        item for item in trace["pieces_after_word_ranking"]
        if item["source"] == "https://x.test"
    )
    assert protected_piece["options"] == [
        {"text": "https://x.test", "score": 0.0, "kind": "raw"}
    ]


def test_bounded_beam_uses_eight_options_and_twenty_four_paths():
    pieces = []
    for index, source in enumerate(("a", "b", "c")):
        options = tuple(WordOption(f"{source}{number}", 10.0 - number, "fake")
                        for number in range(10))
        pieces.append((source, index, options))
    paths, steps = module.bounded_beam_trace(pieces)
    assert [step["used_options"] for step in steps] == [8, 8, 8]
    assert all(step["retained_count"] <= 24 for step in steps)
    assert len(paths) == 24


def test_display_keeps_original_in_fifth_slot_without_changing_top_four():
    paths = [
        (f"후보{index}", 10.0 - index, [{"rank": index}])
        for index in range(7)
    ]
    displayed = module.displayed_candidates(paths, "raw input")
    assert [item["text"] for item in displayed[:4]] == [f"후보{i}" for i in range(4)]
    assert displayed[4] == {
        "text": "raw input", "score": None,
        "segments": [{"input_start": 0, "input_end": 9, "source": "raw input",
                      "text": "raw input", "kind": "raw"}],
    }


def test_mechanical_trace_never_loads_models():
    converter = fake_converter()
    converter.models_loaded = False

    def forbidden():
        raise AssertionError("mechanical conversion attempted to load models")

    converter.load_models = forbidden
    trace = module.trace_conversion(converter, "ga", engine="mechanical")
    assert trace["sentence_score_pairs"] == []
    assert trace["word_ranker_score_pairs"] == []
    assert trace["parity_verified"] is True


def test_real_mechanical_generator_wrapper_dispatches_to_generator_cache():
    converter = module.TraceConverter(external_lexicon=False, context_ranking=False)
    trace = module.trace_conversion(converter, "ga", engine="mechanical")
    assert trace["mechanical_generator_calls"]
    assert trace["mechanical_generator_calls"][0]["source"] == "ga"
    assert trace["parity_verified"] is True
