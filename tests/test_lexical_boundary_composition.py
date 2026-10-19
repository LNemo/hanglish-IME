from collections import defaultdict

import pytest

from hanglish.core.lattice import WordOption
from hanglish.core.phonology import romanize_word
from scripts.lexical_boundary_composition import BoundaryComposer, boundary_rules


class Lexicon:
    def __init__(self, words):
        self.words = defaultdict(list)
        for text in words:
            self.words[romanize_word(text)].append(WordOption(text, 6.0, "lexicon"))
        self.english = {"test"}
        self.korean_boundaries = [("i", "이", ""), ("gi", "이", "k")]

    def _lexical_options(self, key):
        return self.words.get(key, ())


@pytest.mark.parametrize("left,right", [
    ("국", "립"), ("독", "립"), ("신", "라"), ("설", "날"),
    ("침입", "해"), ("반북", "의식"), ("한국", "어"),
    ("꽃", "잎"), ("닫", "이"), ("입", "문"), ("서울", "역"),
    ("연구", "소"),
])
def test_unregistered_whole_words_join_known_components(left, right):
    composer = BoundaryComposer(Lexicon([left, right]))
    expected = left + right
    source = romanize_word(expected)
    assert expected in {o.text for o in composer.propose(source)}
    assert all(romanize_word(o.text) == source for o in composer.propose(source))


def test_two_boundaries_are_restored_without_registering_compound():
    composer = BoundaryComposer(Lexicon(["반북", "의식"]))
    assert "반북의식이" in {o.text for o in composer.propose("banbuguisigi")}


@pytest.mark.parametrize("source", ["test", "Test", "test_reul", "http://a", "a1", "한글", "a" * 41, ""])
def test_english_identifiers_and_out_of_scope_inputs_are_not_rewritten(source):
    assert BoundaryComposer(Lexicon(["테", "스트"])).propose(source) == ()


def test_candidates_cannot_be_invented_without_both_dictionary_components():
    assert BoundaryComposer(Lexicon(["국"])).propose("gungnip") == ()


def test_rule_derivation_is_deterministic_and_keeps_vowel_context():
    assert boundary_rules() == tuple(sorted(set(boundary_rules())))
    assert ("chi", "t", "i") in boundary_rules()
    assert ("ngna", "k", "ra") in boundary_rules()


def test_homophones_are_preserved_with_deterministic_cap():
    words = ["낫", "낮", "낯", "낱", "나", "이", "마", "무"]
    composer = BoundaryComposer(Lexicon(words))
    source = romanize_word("낫마")
    found = composer.propose(source)
    assert {"낫마", "낮마", "낯마", "낱마"} <= {o.text for o in found}
    assert len(found) <= 16 and composer.propose(source) == found
