import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hanglish.core.dictionary import EnglishDictionary


@pytest.fixture
def dictionary():
    return EnglishDictionary()


def test_is_english_true(dictionary):
    assert dictionary.is_english("hello") is True
    assert dictionary.is_english("world") is True
    assert dictionary.is_english("apple") is True


def test_is_english_false(dictionary):
    assert dictionary.is_english("annyeong") is False
    assert dictionary.is_english("haseyo") is False


def test_is_english_case_insensitive(dictionary):
    assert dictionary.is_english("Hello") is True
    assert dictionary.is_english("APPLE") is True


def test_prefix_match(dictionary):
    matches = dictionary.prefix_match("helloworld")
    assert len(matches) > 0
    assert matches[0] == "helloworld" or matches[0] == "hello"
    assert "hello" in matches


def test_prefix_match_apple(dictionary):
    matches = dictionary.prefix_match("appleun")
    assert "apple" in matches


def test_prefix_match_no_match(dictionary):
    matches = dictionary.prefix_match("zzqxwv")
    assert len(matches) == 0
