import pytest

from hanglish.core.phonology import romanize_word


@pytest.mark.parametrize(
    ("hangul", "expected"),
    [
        ("입니다", "imnida"),
        ("합니다", "hamnida"),
        ("없는", "eomneun"),
        ("같이", "gachi"),
        ("국물", "gungmul"),
        ("한국어", "hangugeo"),
        ("학생입니다", "haksaengimnida"),
    ],
)
def test_requested_candidate_validation_words(hangul, expected):
    assert romanize_word(hangul) == expected


@pytest.mark.parametrize(
    ("hangul", "expected"),
    [
        ("밥이", "babi"),
        ("꽃이", "kkochi"),
        ("옷을", "oseul"),
        ("읽어요", "ilgeoyo"),
    ],
)
def test_liaison_before_a_vowel(hangul, expected):
    assert romanize_word(hangul) == expected


@pytest.mark.parametrize(
    ("hangul", "expected"),
    [
        ("국립", "gungnip"),
        ("신라", "silla"),
        ("설날", "seollal"),
        ("심리", "simni"),
        ("정리", "jeongni"),
    ],
)
def test_nasal_and_liquid_assimilation(hangul, expected):
    assert romanize_word(hangul) == expected


@pytest.mark.parametrize(
    ("hangul", "expected"),
    [
        ("좋아", "joa"),
        ("좋아요", "joayo"),
        ("싫어요", "sireoyo"),
        ("좋고", "joko"),
        ("놓다", "nota"),
        ("잡혀", "japyeo"),
    ],
)
def test_h_loss_and_aspiration(hangul, expected):
    assert romanize_word(hangul) == expected


@pytest.mark.parametrize(
    ("hangul", "expected"),
    [
        ("같이", "gachi"),
        ("해돋이", "haedoji"),
        ("맞히다", "machida"),
        ("굳히다", "guchida"),
        ("닫힌", "dachin"),
        ("묻힌", "muchin"),
        ("걷힌", "geochin"),
        ("굳혀", "guchyeo"),
        ("벼훑이", "byeohulchi"),
    ],
)
def test_palatalization(hangul, expected):
    assert romanize_word(hangul) == expected


def test_written_tense_onsets_are_kept_but_automatic_tensing_is_not_added():
    assert romanize_word("꼬리") == "kkori"
    assert romanize_word("국밥") == "gukbap"


def test_documented_n_insertion_exceptions_are_explicit():
    assert romanize_word("학여울") == "hangnyeoul"
    assert romanize_word("알약") == "allyak"


def test_non_hangul_segments_and_case_are_preserved_as_boundaries():
    assert romanize_word("API한국어-Test를 2회") == "APIhangugeo-Testreul 2hoe"


def test_empty_text_and_wrong_type():
    assert romanize_word("") == ""
    with pytest.raises(TypeError, match="text must be a string"):
        romanize_word(None)
