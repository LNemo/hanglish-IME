import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hanglish.core.trie import Trie, ParticleTrie


@pytest.fixture
def trie():
    t = Trie()
    t.insert("eun", "은")
    t.insert("neun", "는")
    t.insert("eul", "을")
    t.insert("reul", "를")
    t.insert("euro", "으로")
    return t


def test_search_found(trie):
    assert trie.search("eun") == "은"
    assert trie.search("neun") == "는"


def test_search_not_found(trie):
    assert trie.search("nonexistent") is None
    assert trie.search("eu") is None  # partial match, not a complete entry


def test_starts_with(trie):
    results = trie.starts_with("eu")
    keys = [r[0] for r in results]
    assert "eun" in keys
    assert "eul" in keys
    assert "euro" in keys


def test_starts_with_no_match(trie):
    assert trie.starts_with("xyz") == []


def test_find_suffixes(trie):
    results = trie.find_suffixes("appleeun")
    assert len(results) >= 1
    positions = [(pos, rom, han) for pos, rom, han in results]
    assert any(pos == 5 and rom == "eun" and han == "은" for pos, rom, han in positions)


def test_find_suffixes_no_match(trie):
    results = trie.find_suffixes("xyz")
    assert results == []


def test_particle_trie_loads():
    pt = ParticleTrie()
    assert pt.is_particle("eun") == "은"
    assert pt.is_particle("neun") == "는"
    assert pt.is_particle("nonexistent") is None


def test_particle_trie_find_suffixes():
    pt = ParticleTrie()
    results = pt.find_particle_suffixes("appleeun")
    assert any(han == "은" for _, _, han in results)
