import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hanglish.core.ngram import NgramModel


@pytest.fixture
def ngram():
    return NgramModel()


def test_prob_unigram_returns_negative(ngram):
    prob = ngram.prob_unigram("안")
    assert isinstance(prob, float)
    assert prob < 0  # log probability is always negative


def test_prob_unigram_unknown_syllable(ngram):
    prob = ngram.prob_unigram("뷁")
    assert isinstance(prob, float)
    assert prob < 0  # smoothed, still returns a value


def test_prob_bigram(ngram):
    prob = ngram.prob_bigram("안", "녕")
    assert isinstance(prob, float)
    assert prob < 0


def test_score_sequence(ngram):
    score = ngram.score_sequence(["안", "녕"])
    assert isinstance(score, float)
    assert score < 0


def test_score_empty_sequence(ngram):
    assert ngram.score_sequence([]) == 0.0


def test_common_syllable_higher_prob(ngram):
    # 자주 나오는 음절이 희귀 음절보다 확률이 높아야 함
    prob_common = ngram.prob_unigram("니")  # common in corpus
    prob_rare = ngram.prob_unigram("뷁")    # not in corpus
    assert prob_common > prob_rare
