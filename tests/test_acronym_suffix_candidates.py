from types import SimpleNamespace

import pytest

from hanglish.core.lattice import WordOption
from scripts.probe_acronym_suffix_candidates import (
    extend_options,
    install,
    select_train_rows,
    validate_extension,
)


def option(text, score=1., kind="neural"):
    return WordOption(text, score, kind)


class Converter:
    def __init__(self, suffix_options):
        self.suffix_options = tuple(suffix_options)
        self.transducer = object()

    def _word_options(self, source, learned):
        if source == "neun":
            return self.suffix_options
        return (option(source, 6., "raw"), option("기존", 5., "lexicon"))


def test_uppercase_acronym_suffix_preserves_prefix_scores_and_adds_mixed():
    converter = Converter([
        option("는", 3.25), option("는", 2.), option("닌", 1.), option("neun", 0., "raw")
    ])
    baseline = converter._word_options("AIneun", True)

    result = extend_options(converter, "AIneun", True, baseline)

    assert result[:len(baseline)] == baseline
    assert result[len(baseline):] == (option("AI는", 3.25, "mixed"),)


def test_extension_deduplicates_and_caps_four_in_suffix_order():
    suffixes = [option(text, float(index)) for index, text in enumerate(
        ["가", "각", "간", "갇", "갈", "감"]
    )]
    # All test alternatives need a matching source, so use a controlled converter
    # and monkeypatch forward validation to isolate ordering and the cap.
    converter = Converter(suffixes)
    baseline = (option("AA가", 9., "mixed"), option("raw", 0., "raw"))
    import scripts.probe_acronym_suffix_candidates as module
    original = module.romanize_word
    module.romanize_word = lambda text: "AAneun"
    try:
        result = extend_options(converter, "AAneun", True, baseline)
    finally:
        module.romanize_word = original

    assert result[:2] == baseline
    assert [value.text for value in result[2:]] == ["AA각", "AA간", "AA갇", "AA갈"]
    assert [value.score for value in result[2:]] == [1., 2., 3., 4.]


@pytest.mark.parametrize("source,learned", [
    ("Aneun", True), ("AiNeun", True), ("AI", True), ("AIneun", False),
    ("AINEUN", True), ("A1neun", True), ("A" * 39 + "ne", True),
])
def test_excluded_sources_are_returned_unchanged(source, learned):
    converter = Converter([option("는", 3.)])
    baseline = (option(source, 0., "raw"),)

    assert extend_options(converter, source, learned, baseline) == baseline


def test_install_is_source_only_cached_and_leaves_nonlearned_unchanged():
    class TraceConverter(Converter):
        def __init__(self):
            super().__init__([option("는", 3.)])
            self.calls = []
            self.word_options = self._word_options

        def _word_options(self, source, learned):
            self.calls.append((source, learned))
            return super()._word_options(source, learned)

    converter = TraceConverter()
    audit = install(converter)

    first = converter.word_options("AIneun", True)
    second = converter.word_options("AIneun", True)
    raw = converter.word_options("AIneun", False)

    assert first == second
    assert [value.text for value in first][-1] == "AI는"
    assert raw == converter._word_options("AIneun", False)
    assert ("AIneun", True) in audit and ("AIneun", False) in audit
    assert converter.calls.count(("AIneun", True)) == 1


def test_train_prefix_selection_does_not_read_labels():
    class NoLabels(dict):
        def __getitem__(self, key):
            if key == "expected":
                raise AssertionError("selection read a target label")
            return super().__getitem__(key)

    rows = [NoLabels(id=index, expected=["hidden"]) for index in range(3)]

    assert select_train_rows(rows, 2) == rows[:2]


@pytest.mark.parametrize("mutation", ["prefix", "duplicate", "too_many", "kind", "hangul", "forward"])
def test_malformed_extensions_are_rejected(monkeypatch, mutation):
    import scripts.probe_acronym_suffix_candidates as module
    baseline = (option("AIneun", 0., "raw"),)
    additions = [option("AI는", 1., "mixed")]
    if mutation == "prefix":
        result = tuple(additions) + baseline
    elif mutation == "duplicate":
        result = baseline + (baseline[0],)
    elif mutation == "too_many":
        result = baseline + tuple(option(f"AI값{i}", 1., "mixed") for i in range(5))
    elif mutation == "kind":
        result = baseline + (option("AI는", 1., "neural"),)
    elif mutation == "hangul":
        result = baseline + (option("AIneun", 1., "mixed"),)
    else:
        monkeypatch.setattr(module, "romanize_word", lambda text: "changed")
        result = baseline + tuple(additions)
    with pytest.raises(ValueError):
        validate_extension("AIneun", baseline, result)
