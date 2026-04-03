import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hanglish.core.discriminator import LanguageDiscriminator, Segment


@pytest.fixture
def disc():
    return LanguageDiscriminator()


def test_uppercase_english(disc):
    result = disc.classify("Hello")
    assert len(result) >= 1
    assert result[0].lang == "en"


def test_uppercase_with_particle(disc):
    result = disc.classify("Appleun")
    assert len(result) == 2
    assert result[0].lang == "en"
    assert result[0].text == "Apple"
    assert result[1].lang == "ko"
    assert result[1].text == "un"  # apple(longest prefix) + un(은)


def test_lowercase_english(disc):
    result = disc.classify("hello")
    assert len(result) == 1
    assert result[0].lang == "en"


def test_korean_romanization(disc):
    result = disc.classify("annyeong")
    assert len(result) == 1
    assert result[0].lang == "ko"


def test_delimiter_split(disc):
    result = disc.classify("Apple-eun")
    assert len(result) == 2
    assert result[0].lang == "en"
    assert result[0].text == "Apple"
    assert result[1].lang == "ko"
    assert result[1].text == "eun"


def test_english_with_suffix(disc):
    result = disc.classify("appleun")
    assert len(result) == 2
    assert result[0].text == "apple"
    assert result[0].lang == "en"
    assert result[1].text == "un"  # apple(longest prefix) + un(은)
    assert result[1].lang == "ko"


def test_ending_split(disc):
    result = disc.classify("Helloyo")
    assert len(result) == 2
    assert result[0].text == "Hello"
    assert result[0].lang == "en"
    assert result[1].text == "yo"
    assert result[1].lang == "ko"


def test_empty_input(disc):
    assert disc.classify("") == []
