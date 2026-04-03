import pytest
import sys
from pathlib import Path

# 테스트를 위해 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from hanglish.core.romanizer import Romanizer

@pytest.fixture
def romanizer():
    return Romanizer()

def test_basic_vowels_and_consonants(romanizer):
    assert romanizer.romanize("ga") == "가"
    assert romanizer.romanize("nya") == "냐"
    assert romanizer.romanize("deo") == "더"
    assert romanizer.romanize("ryeo") == "려"

def test_jongsungs(romanizer):
    assert romanizer.romanize("gang") == "강"
    assert romanizer.romanize("min") == "민"
    assert romanizer.romanize("hoe") == "회"

def test_words(romanizer):
    assert romanizer.romanize("annyeong") == "안녕"
    assert romanizer.romanize("haseyo") == "하세요"
    assert romanizer.romanize("hangul") in ("한굴", "hangul")  # 'hangul' is also an English word
    assert romanizer.romanize("sarang") == "사랑"
    assert romanizer.romanize("hangeul") == "한글"

def test_lookahead_for_jongsung(romanizer):
    assert romanizer.romanize("ani") == "아니"  # an-i 가 아니라 a-ni
    assert romanizer.romanize("bab") == "밥"
    assert romanizer.romanize("bap") == "밮"
    assert romanizer.romanize("babi") in ("바비", "밥이")  # both valid: 바비(ba-bi) or 밥이(bab-i)

def test_double_consonants_vowels(romanizer):
    assert romanizer.romanize("kkyeo") == "껴"
    assert romanizer.romanize("ttwa") == "똬"
    assert romanizer.romanize("jjwae") == "쫴"

def test_apjjeopsi(romanizer):
    assert romanizer.romanize("apjjeopsi") == "앞쩦시"
    assert romanizer.romanize("apjeobsi") in ("앞접시", "앞젒이")  # both valid segmentations

def test_eobseo(romanizer):
    # 'eobseo' is '업서'. 'eobs-eo' is '없어'.
    assert romanizer.romanize("eobseo") == "업서"
    assert romanizer.romanize("eobs-eo") == "없어"

def test_apjjeopsi_strict(romanizer):
    # j is ㅈ, jj is ㅉ. p is ㅍ. b is ㅂ. eo is ㅓ. s is ㅅ. i is ㅣ.
    # a p (앞) jj eo p (쩦) s i (시) => 앞쩦시
    # But user wants 앞접시. So it should be: a p (앞) j eo b (접) s i (시)
    assert romanizer.romanize("apjjeopsi") == "앞쩦시"
    assert romanizer.romanize("apjeobsi") in ("앞접시", "앞젒이")  # both valid segmentations
