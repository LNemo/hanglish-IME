import pytest

from hanglish.core.phonology import (
    _Syllable, _apply_pair, _CODAS, _ONSETS, _VOWELS, _INSERTION_EXCEPTIONS, romanize_word,
)
from scripts.constrained_phonetic_search import stable_roman_prefix, compatible_prefix


def test_pair_rules_only_mutate_fields_needed_by_prefix_proof():
    for coda in _CODAS:
        for onset in _ONSETS:
            for vowel in _VOWELS:
                left, right = _Syllable('ㄱ','ㅏ',coda), _Syllable(onset,vowel,'ㄹ')
                _apply_pair(left,right)
                assert (left.onset,left.vowel,right.vowel,right.coda)==('ㄱ','ㅏ',vowel,'ㄹ')


# 입 is now a deferred lexical prefix of 입원료; it is covered by the
# exception-prefix test below rather than the local-only stability proof.
@pytest.mark.parametrize('prefix',['법','입니다','국립','같이','굳히','좋고','신라','먹어','읽어','침입해','앞장'])
def test_stable_prefix_survives_every_appended_onset_and_vowel(prefix):
    stable=stable_roman_prefix(prefix)
    assert stable is not None
    for onset in range(19):
        for vowel in range(21):
            suffix=chr(0xAC00+(onset*21+vowel)*28)
            assert romanize_word(prefix+suffix).startswith(stable)


def test_insertion_exception_prefixes_defer_pruning():
    for key in _INSERTION_EXCEPTIONS:
        for end in range(1,len(key)+1):
            prefix=key[:end]
            assert stable_roman_prefix(prefix) is None
            assert compatible_prefix(prefix,romanize_word(key))
    assert not romanize_word('학여').startswith('hang')
    assert compatible_prefix('학여','hangnyeoul')


def test_complete_paths_still_require_exact_forward_equality():
    assert compatible_prefix('입니다','imnida',complete=True)
    assert not compatible_prefix('입','imnida',complete=True)
    assert not compatible_prefix('학여','hangnyeoul',complete=True)
    assert compatible_prefix('학여울','hangnyeoul',complete=True)


@pytest.mark.parametrize("word", ["신문로에서", "집현전에", "솜이불을", "맛없어요"])
def test_lexical_suffixes_do_not_prune_a_valid_complete_word(word):
    target = romanize_word(word)
    for end in range(1, len(word) + 1):
        assert compatible_prefix(word[:end], target)
    assert compatible_prefix(word, target, complete=True)


def test_incompatible_stable_prefix_is_rejected():
    assert not compatible_prefix('가','imnida')
    assert compatible_prefix('입','imnida')
