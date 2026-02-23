
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hanglish.core.romanizer import Romanizer

@pytest.fixture
def romanizer():
    return Romanizer()

def test_multiple_candidates(romanizer):
    candidates = romanizer.get_candidates("eobseo")
    assert "없어" in candidates
    assert "업서" in candidates
    
    candidates_jeong = romanizer.get_candidates("jeongbo")
    assert "정보" in candidates_jeong

    candidates_gangan = romanizer.get_candidates("gangan")
    assert "간간" in candidates_gangan
    assert "강안" in candidates_gangan
