from functools import lru_cache
import threading

import pytest

from hanglish.core.lattice import WordOption
from hanglish.core.neural_converter import NeuralConverter
from hanglish.core.phonology import romanize_word
import scripts.lexical_boundary_runtime as module


class Generator:
    english = {"test"}
    english_suffixes = ()
    korean_boundaries = (("i", "이", ""), ("gi", "이", "k"))

    def __init__(self, *, protected=False):
        self.protected = protected
        self.lexicon = {}
        for text in ("국", "립", "반북", "의식"):
            self.lexicon.setdefault(romanize_word(text), []).append(
                WordOption(text, 7.0, "lexicon")
            )

    def _lexical_options(self, key):
        return tuple(self.lexicon.get(key, ()))

    def options(self, source):
        if source == "gungnip":
            if self.protected:
                return (
                    WordOption("gungnip", 6.0, "mixed"),
                    WordOption("궁닙", 5.8, "lexicon"),
                )
            return (
                WordOption("궁닙", 6.0, "lexicon"),
                WordOption(source, 0.0, "raw"),
            )
        if source == "test":
            return (WordOption(source, 7.0, "english"),)
        return (WordOption(source, 0.0, "raw"),)


class WordRanker:
    def __init__(self, malformed=None):
        self.calls = []
        self.malformed = malformed

    def score_pairs(self, contexts, candidates):
        self.calls.append((list(contexts), list(candidates)))
        if self.malformed == "short":
            return [1.0] * max(0, len(candidates) - 1)
        if self.malformed == "nan":
            return [float("nan")] * len(candidates)
        if self.malformed == "bool":
            return [True] * len(candidates)
        return [8.0 if candidate == "국립" else 2.0 for candidate in candidates]


class SentenceRanker:
    def __init__(self, *, unscoreable=(), malformed=None):
        self.unscoreable = set(unscoreable)
        self.malformed = malformed
        self.calls = []

    def can_score(self, candidate):
        return candidate not in self.unscoreable

    def score_pairs(self, contexts, candidates):
        self.calls.append((list(contexts), list(candidates)))
        if self.malformed == "short":
            return [1.0] * max(0, len(candidates) - 1)
        if self.malformed == "inf":
            return [float("inf")] * len(candidates)
        if self.malformed == "text":
            return ["1.0"] * len(candidates)
        return [20.0 if candidate == "국립" else 1.0 for candidate in candidates]


def converter(*, protected=False, word_ranker=None, sentence_ranker=None):
    value = module.BoundaryCompositionRuntime.__new__(
        module.BoundaryCompositionRuntime
    )
    value.generator = Generator(protected=protected)
    value.lock = threading.RLock()
    value.context_ranking = True
    value.ranker_kind = "sentence"
    value.ranker = word_ranker or WordRanker()
    value.sentence_ranker = sentence_ranker or SentenceRanker()
    value.transducer = None
    value.numeric_units = ()
    value.numeric_structure = None
    value.models_loaded = True
    value.version = "test"
    value.model_artifacts = {"ranker": "sha"}
    value.load_models = lambda: None
    value.word_options = lru_cache(maxsize=128)(value._word_options)
    value.initialize_boundary_overlay()
    return value


def test_boundary_path_is_word_and_sentence_ranked_without_pruning_parent_pool():
    value = converter()
    original_word_options = value.word_options
    original_ranker = value.ranker
    original_sentence_ranker = value.sentence_ranker
    response = value.convert("gungnip", context="gonggong gigwan")

    assert response["result"] == "국립"
    assert response["sentence_ranking_applied"] is True
    trace = value.boundary_trace
    assert trace["activated"] is True
    assert trace["reason"] == "accepted_new_extras"
    assert [item["text"] for item in trace["preserved_pool"]] == [
        "궁닙", "gungnip"
    ]
    assert [item["text"] for item in trace["accepted_new_extras"]] == ["국립"]
    assert value.sentence_ranker.calls[0][1] == ["궁닙", "gungnip", "국립"]
    assert value.ranker.calls[-1][1] == ["국립", "궁닙"]
    assert value.word_options is original_word_options
    assert value.ranker is original_ranker
    assert value.sentence_ranker is original_sentence_ranker


def test_new_path_keeps_original_source_offsets_and_complete_candidate_shape():
    value = converter()
    value.convert("gungnip")
    candidate = value.boundary_trace["accepted_new_extras"][0]
    assert candidate["segments"] == [{
        "input_start": 0,
        "input_end": 7,
        "source": "gungnip",
        "text": "국립",
        "kind": "morphology",
    }]
    assert isinstance(candidate["score"], float)


def test_trace_contains_source_only_changed_boundary_derivation():
    value = converter()
    value.convert("gungnip")
    derivations = value.boundary_trace["derived_options"]
    assert any(
        item["text"] == "국립"
        and item["left_lexeme"] == "국"
        and item["right_lexeme"] == "립"
        and item["boundary"] == "changed"
        and item["left_key"] == "guk"
        and item["right_key"] == "rip"
        and item["suffix"] is None
        for item in derivations
    )


def test_suffix_derivation_records_boundary_keys_and_suffix_source():
    value = converter()
    proposed = value._boundary_composer.propose("banbuguisigi")
    assert [option.text for option in proposed] == ["반북의식이"]
    records = value._explain_options(
        "banbuguisigi", tuple(option.text for option in proposed)
    )
    assert any(
        item["text"] == "반북의식이"
        and item["left_lexeme"] == "반북"
        and item["right_lexeme"] == "의식"
        and item["suffix"] == "이"
        and item["suffix_surface"] == "gi"
        and item["suffix_restored"] == "k"
        for item in records
    )


def test_unscoreable_boundary_path_is_traced_but_not_added():
    ranker = SentenceRanker(unscoreable={"국립"})
    value = converter(sentence_ranker=ranker)
    response = value.convert("gungnip")
    assert response["result"] == "gungnip"
    assert value.boundary_trace["reason"] == "unscoreable_new_extras"
    assert [item["text"] for item in value.boundary_trace["generated_new_extras"]] == ["국립"]
    assert value.boundary_trace["accepted_new_extras"] == []
    assert ranker.calls[0][1] == ["궁닙", "gungnip"]


def test_protected_mixed_candidate_prevents_boundary_rewrite():
    value = converter(protected=True)
    response = value.convert("gungnip")
    assert response["result"] in {"gungnip", "궁닙"}
    assert value.boundary_trace["activated"] is False
    assert value.boundary_trace["generated_new_extras"] == []
    assert value.boundary_trace["augmented_piece_count"] == 0


@pytest.mark.parametrize("kwargs", [
    {"raw": True},
    {"engine": "rules"},
])
def test_raw_and_nonneural_branches_are_parent_identical_and_clear_trace(kwargs):
    value = converter()
    value.convert("gungnip")
    actual = value.convert("gungnip", **kwargs)
    expected = NeuralConverter.convert(value, "gungnip", **kwargs)
    actual.pop("engine_ms")
    expected.pop("engine_ms")
    assert actual == expected
    assert value.boundary_trace == value._empty_trace("production_branch_bypass")


def test_disabled_overlay_is_parent_identical():
    value = converter()
    value.boundary_overlay_enabled = False
    actual = value.convert("gungnip", context="gonggong gigwan")
    expected = NeuralConverter.convert(
        value, "gungnip", context="gonggong gigwan"
    )
    actual.pop("engine_ms")
    expected.pop("engine_ms")
    assert actual == expected
    assert value.boundary_trace == value._empty_trace("production_branch_bypass")


def test_fresh_instance_lazy_load_keeps_installed_model_objects():
    value = converter()
    installed_word = WordRanker()
    installed_sentence = SentenceRanker()
    value.ranker = None
    value.sentence_ranker = None
    value.models_loaded = False

    def load():
        value.ranker = installed_word
        value.sentence_ranker = installed_sentence
        value.models_loaded = True

    value.load_models = load
    assert value.convert("gungnip")["result"] == "국립"
    assert value.ranker is installed_word
    assert value.sentence_ranker is installed_sentence
    assert value.models_loaded is True


@pytest.mark.parametrize("source", ["test", "Test", "guk-립", "A1"])
def test_english_nonlowercase_and_nonascii_inputs_are_untouched(source):
    value = converter()
    response = value.convert(source)
    assert response["result"] == source
    assert value.boundary_trace["activated"] is False
    assert value.boundary_trace["generated_new_extras"] == []


@pytest.mark.parametrize("kind", ["short", "nan", "bool"])
def test_malformed_or_nonfinite_word_score_vectors_fail(kind):
    value = converter(word_ranker=WordRanker(malformed=kind))
    with pytest.raises(ValueError, match="word ranker"):
        value.convert("gungnip")
    assert value.word_options is value._boundary_base_word_options


@pytest.mark.parametrize("kind", ["short", "inf", "text"])
def test_malformed_or_nonfinite_sentence_score_vectors_fail(kind):
    value = converter(sentence_ranker=SentenceRanker(malformed=kind))
    with pytest.raises(ValueError, match="sentence ranker"):
        value.convert("gungnip")
    assert value.word_options is value._boundary_base_word_options


def test_old_numeric_extras_keep_production_all_or_none_rule(monkeypatch):
    value = converter(sentence_ranker=SentenceRanker(unscoreable={"bad"}))
    old = [
        ("old", 0.0, []),
        ("bad", 0.0, []),
    ]
    monkeypatch.setattr(
        NeuralConverter, "_numeric_sentence_extras", lambda *args: old
    )
    response = value.convert("gungnip")
    sentence_pool = value.sentence_ranker.calls[0][1]
    assert "old" not in sentence_pool and "bad" not in sentence_pool
    assert response["result"] == "국립"


def test_new_sentence_paths_are_deduplicated_and_bounded(monkeypatch):
    value = converter()
    paths = [
        (f"새경로{i}", float(i), [{"input_start": 0, "input_end": 7,
          "source": "gungnip", "text": f"새경로{i}", "kind": "morphology"}])
        for i in range(80)
    ]
    calls = 0

    def fake_beam(_pieces):
        nonlocal calls
        calls += 1
        return paths[:60] if calls % 2 else paths[20:]

    monkeypatch.setattr(module, "beam_paths", fake_beam)
    value._boundary_current_text = "gungnip"
    value._boundary_current_context = ""
    extras = value._numeric_sentence_extras([], [], [])
    assert len(extras) == module.MAX_NEW_PATHS
    assert len({path[0] for path in extras}) == module.MAX_NEW_PATHS


def test_option_union_is_strongest_deterministic_capped_and_retains_raw():
    original = tuple(
        [WordOption("same", 1.0, "lexicon"), WordOption("raw", -1.0, "raw")]
        + [WordOption(f"o{i}", float(i), "lexicon") for i in range(20)]
    )
    merged = module._merge_options(
        original, (WordOption("same", 9.0, "morphology"),)
    )
    assert len(merged) == module.MAX_WORD_OPTIONS
    assert next(option for option in merged if option.text == "same").score == 9.0
    assert any(option.kind == "raw" for option in merged)
    assert merged == module._merge_options(original, tuple(reversed((WordOption("same", 9.0, "morphology"),))))


def test_runtime_builder_has_no_label_or_expected_parameter():
    for function in (
        module.BoundaryCompositionRuntime._numeric_sentence_extras,
        module.BoundaryCompositionRuntime._build_boundary_options,
    ):
        names = function.__code__.co_varnames
        assert "expected" not in names
        assert "label" not in names
        assert "target" not in names
