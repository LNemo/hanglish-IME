
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hanglish.core.romanizer import Romanizer
from hanglish.core.candidate import CandidateRanker, RankedCandidate


@pytest.fixture
def romanizer():
    return Romanizer()


@pytest.fixture
def ranker():
    return CandidateRanker()


def test_multiple_candidates(romanizer):
    candidates = romanizer.get_candidates("eobseo")
    assert "없어" in candidates
    assert "업서" in candidates

    candidates_jeong = romanizer.get_candidates("jeongbo")
    assert "정보" in candidates_jeong

    candidates_gangan = romanizer.get_candidates("gangan")
    assert "간간" in candidates_gangan
    assert "강안" in candidates_gangan


def test_rank_korean(ranker, romanizer):
    candidates = romanizer.get_candidates("annyeong")
    ranked = ranker.rank("annyeong", raw_candidates=candidates)
    assert len(ranked) > 0
    assert isinstance(ranked[0], RankedCandidate)
    assert ranked[0].text == "안녕"


def test_rank_english_passthrough(ranker):
    ranked = ranker.rank("hello")
    assert len(ranked) == 1
    assert ranked[0].text == "hello"


def test_rank_mixed_appleun(ranker):
    ranked = ranker.rank("Appleun")
    assert len(ranked) >= 1
    assert "Apple" in ranked[0].text
    assert "은" in ranked[0].text


def test_romanize_nlp_annyeong(romanizer):
    result = romanizer.romanize("annyeong")
    assert result == "안녕"


def test_romanize_nlp_haseyo(romanizer):
    result = romanizer.romanize("haseyo")
    assert result == "하세요"


def test_english_passthrough_hello(ranker):
    ranked = ranker.rank("hello", raw_candidates=["헬로"])
    assert ranked[0].text == "hello"


def test_english_passthrough_world(ranker):
    ranked = ranker.rank("world", raw_candidates=["워ㄹㄹㄷ"])
    assert ranked[0].text == "world"


def test_short_korean_priority(ranker):
    ranked = ranker.rank("ga", raw_candidates=["가"])
    assert ranked[0].text == "가"


def test_hangugeo_ranking():
    r = Romanizer()
    assert r.romanize("hangugeo") == "한국어"


def test_helloyo_mixed():
    r = Romanizer()
    assert r.romanize("Helloyo") == "Hello요"
