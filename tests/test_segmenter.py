import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hanglish.core.segmenter import ViterbiSegmenter, SegmentResult


@pytest.fixture
def segmenter():
    return ViterbiSegmenter()


def test_segment_basic(segmenter):
    result = segmenter.segment("안녕하세요")
    assert isinstance(result, SegmentResult)
    assert len(result.segments) > 0
    assert isinstance(result.score, float)
    # Joined segments should equal original syllables
    assert "".join(result.segments) == "안녕하세요"


def test_segment_empty(segmenter):
    result = segmenter.segment("")
    assert result.segments == []
    assert result.score == 0.0


def test_segment_single_syllable(segmenter):
    result = segmenter.segment("안")
    assert result.segments == ["안"]
    assert result.score < 0  # log prob is negative


def test_score_candidate(segmenter):
    score = segmenter.score_candidate("안녕하세요")
    assert isinstance(score, float)


def test_different_segmentations_have_different_scores(segmenter):
    # 순은 vs 수는 should have different scores
    score1 = segmenter.score_candidate("순은")
    score2 = segmenter.score_candidate("수는")
    assert score1 != score2


def test_particle_bonus(segmenter):
    # "순은" where 은 is a known particle — should get bonus
    result = segmenter.segment("순은")
    # The particle bonus depends on whether 은 maps correctly
    assert isinstance(result.particle_bonus, float)


def test_segments_reconstruct_input(segmenter):
    inputs = ["안녕", "사랑", "한국어"]
    for text in inputs:
        result = segmenter.segment(text)
        reconstructed = "".join(result.segments)
        assert reconstructed == text
