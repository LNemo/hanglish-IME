"""Normal RR examples, independently specified from NIKL's official rules.

Authority: NIKL Korean Language Norms, RR chapters 1–3 and Standard
Pronunciation articles 12, 15, 20 and 29. Names are lowercased here because
romanize_word is the IME's phonetic key, not a proper-name formatter.
https://www.korean.go.kr/kornorms/regltn/regltnView.do?regltn_code=0004
"""
import json
from pathlib import Path

import pytest

from hanglish.core.phonology import romanize_word

_FIXTURE_DOCUMENT = json.loads(
    (Path(__file__).parent / "fixtures/rr_official_examples.json").read_text()
)
_OFFICIAL_FIXTURES = _FIXTURE_DOCUMENT["fixtures"]


def test_fixture_scope_is_explicit():
    assert _FIXTURE_DOCUMENT["schema"] == "hanglish-official-rr-fixtures-v1"
    assert _FIXTURE_DOCUMENT["fixture_count"] == len(_OFFICIAL_FIXTURES)
    assert {row["mode"] for row in _OFFICIAL_FIXTURES} == {
        "rr_normal", "rr_normal_lexical_compound", "rr_normal_morphology",
        "rr_person_name", "rr_proper_name", "rr_administrative_name",
        "rr_heritage_or_feature_name", "rr_hyphen_optional", "academic_reversible",
    }
    assert len({(row["hangul"], row["mode"]) for row in _OFFICIAL_FIXTURES}) == len(
        _OFFICIAL_FIXTURES)


@pytest.mark.parametrize("case", [
    row for row in _OFFICIAL_FIXTURES if row["mode"] == "rr_normal"
], ids=lambda case: case["hangul"])
def test_source_cited_normal_rr_corpus(case):
    assert romanize_word(case["hangul"]) == case["rr"]


@pytest.mark.parametrize(("text", "expected"), list(zip(
    "아애야얘어에여예오와왜외요우워웨위유으의이",
    "a ae ya yae eo e yeo ye o wa wae oe yo u wo we wi yu eu ui i".split(),
)))
def test_complete_official_vowel_table(text, expected):
    assert romanize_word(text) == expected


@pytest.mark.parametrize(("text", "expected"), list(zip(
    "가까나다따라마바빠사싸아자짜차카타파하",
    "ga kka na da tta ra ma ba ppa sa ssa a ja jja cha ka ta pa ha".split(),
)))
def test_complete_official_initial_consonant_table(text, expected):
    assert romanize_word(text) == expected


@pytest.mark.parametrize(("text", "expected"), [
    ("구미", "gumi"), ("영동", "yeongdong"), ("백암", "baegam"),
    ("옥천", "okcheon"), ("합덕", "hapdeok"), ("호법", "hobeop"),
    ("월곶", "wolgot"), ("벚꽃", "beotkkot"), ("한밭", "hanbat"),
    ("구리", "guri"), ("설악", "seorak"), ("칠곡", "chilgok"),
    ("임실", "imsil"), ("울릉", "ulleung"), ("대관령", "daegwallyeong"),
    ("광희문", "gwanghuimun"), ("백마", "baengma"),
    ("신문로", "sinmunno"), ("종로", "jongno"), ("왕십리", "wangsimni"),
    ("별내", "byeollae"), ("신라", "silla"), ("학여울", "hangnyeoul"),
    ("알약", "allyak"), ("해돋이", "haedoji"), ("같이", "gachi"),
    ("굳히다", "guchida"), ("좋고", "joko"), ("놓다", "nota"),
    ("잡혀", "japyeo"), ("낳지", "nachi"), ("묵호", "mukho"),
    ("집현전", "jiphyeonjeon"), ("압구정", "apgujeong"),
    ("낙동강", "nakdonggang"), ("샛별", "saetbyeol"),
])
def test_official_rr_sound_examples(text, expected):
    assert romanize_word(text) == expected


@pytest.mark.parametrize(("text", "expected"), [
    ("밝히다", "balkida"), ("밝혀", "balkyeo"), ("얽히다", "eolkida"),
    ("의견란", "uigyeonnan"), ("생산량", "saengsannyang"),
    ("상견례", "sanggyeonnye"), ("횡단로", "hoengdanno"),
    ("이원론", "iwonnon"), ("입원료", "ibwonnyo"),
    ("구근류", "gugeunnyu"), ("솜이불", "somnibul"),
    ("꽃잎", "kkonnip"), ("막일", "mangnil"),
    ("맛없다", "madeopda"), ("값어치", "gabeochi"),
])
def test_standard_pronunciation_supporting_rr(text, expected):
    assert romanize_word(text) == expected


@pytest.mark.parametrize(("text", "expected"), [
    ("신문로에서", "sinmunnoeseo"), ("집현전에", "jiphyeonjeone"),
    ("묵호는", "mukhoneun"), ("꽃잎이", "kkonnipi"),
    ("솜이불을", "somnibureul"), ("값어치가", "gabeochiga"),
    ("의견란은", "uigyeonnaneun"), ("맛없어요", "madeopseoyo"),
    # Do not infer compound boundaries from every consonant+vowel sequence.
    ("옷이", "osi"), ("밥이", "babi"), ("읽어요", "ilgeoyo"),
    ("신라", "silla"), ("설날", "seollal"),
    ("신문로봇", "sinmullobot"),
    ("학여울-알약", "hangnyeoul-allyak"),
    ("API집현전-test", "APIjiphyeonjeon-test"),
])
def test_known_boundaries_and_negative_controls(text, expected):
    assert romanize_word(text) == expected


def test_normal_rr_is_not_academic_reversible_transliteration():
    assert romanize_word("먹는") == "meongneun"
    assert romanize_word("값") == "gap"
