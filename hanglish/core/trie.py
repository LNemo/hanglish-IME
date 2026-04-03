"""Trie data structure for Korean particle/ending romanization lookup."""

from __future__ import annotations

import json
from pathlib import Path


class TrieNode:
    __slots__ = ("children", "is_end", "value")

    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.is_end: bool = False
        self.value: str = ""


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, romanized: str, hangul: str) -> None:
        node = self.root
        for ch in romanized:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True
        node.value = hangul

    def search(self, text: str) -> str | None:
        node = self.root
        for ch in text:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node.value if node.is_end else None

    def starts_with(self, prefix: str) -> list[tuple[str, str]]:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        results: list[tuple[str, str]] = []
        self._collect(node, prefix, results)
        return results

    def _collect(self, node: TrieNode, current: str, results: list[tuple[str, str]]) -> None:
        if node.is_end:
            results.append((current, node.value))
        for ch, child in node.children.items():
            self._collect(child, current + ch, results)

    def find_suffixes(self, text: str) -> list[tuple[int, str, str]]:
        """Find all particle/ending suffixes at the end of text.

        Returns list of (start_position, romanized_suffix, hangul_value).
        Searches backwards from the end, trying all possible suffix lengths.
        """
        results: list[tuple[int, str, str]] = []
        text_len = len(text)
        for start in range(text_len - 1, -1, -1):
            suffix = text[start:]
            node = self.root
            matched = True
            for ch in suffix:
                if ch not in node.children:
                    matched = False
                    break
                node = node.children[ch]
            if matched and node.is_end:
                results.append((start, suffix, node.value))
        return results


class ParticleTrie:
    """Pre-built Trie loaded from particles.json."""

    def __init__(self, data_path: str | Path | None = None):
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent / "data" / "particles.json"
        else:
            data_path = Path(data_path)

        self.particle_trie = Trie()
        self.ending_trie = Trie()

        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for romanized, hangul in data.get("particles", {}).items():
            self.particle_trie.insert(romanized, hangul)

        for romanized, hangul in data.get("endings", {}).items():
            self.ending_trie.insert(romanized, hangul)

    def find_particle_suffixes(self, text: str) -> list[tuple[int, str, str]]:
        return self.particle_trie.find_suffixes(text)

    def is_particle(self, text: str) -> str | None:
        return self.particle_trie.search(text)

    def is_ending(self, text: str) -> str | None:
        return self.ending_trie.search(text)
