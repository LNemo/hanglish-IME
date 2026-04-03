"""Language discriminator for Korean/English mixed input."""

from __future__ import annotations

from dataclasses import dataclass

from hanglish.core.dictionary import EnglishDictionary
from hanglish.core.trie import ParticleTrie


@dataclass
class Segment:
    text: str
    lang: str  # "en" or "ko"
    confidence: float = 1.0


class LanguageDiscriminator:
    def __init__(
        self,
        dictionary: EnglishDictionary | None = None,
        particle_trie: ParticleTrie | None = None,
    ):
        self._dict = dictionary or EnglishDictionary()
        self._particle_trie = particle_trie or ParticleTrie()

    def classify(self, text: str) -> list[Segment]:
        if not text:
            return []

        # Rule 4: Explicit delimiter splits
        if "-" in text:
            return self._split_on_delimiter(text)

        # Rule 1: Uppercase triggers English priority
        if any(c.isupper() for c in text):
            return self._classify_with_uppercase(text)

        # Rule 2: Full English dictionary match
        if self._dict.is_english(text):
            return [Segment(text=text, lang="en", confidence=0.9)]

        # Rule 3: Suffix speller — try to split english_word + particle/ending
        result = self._try_suffix_split(text)
        if result:
            en_part = result[0].text
            # Only split for long prefixes (5+ chars) to avoid
            # false splits like "hang"+"eul", "has"+"eyo"
            if len(en_part) >= 5:
                return result

        # Rule 5: Default to Korean
        return [Segment(text=text, lang="ko", confidence=0.7)]

    def _split_on_delimiter(self, text: str) -> list[Segment]:
        parts = text.split("-")
        segments: list[Segment] = []
        for part in parts:
            if not part:
                continue
            if self._dict.is_english(part.lower()) or any(c.isupper() for c in part):
                segments.append(Segment(text=part, lang="en", confidence=0.9))
            else:
                segments.append(Segment(text=part, lang="ko", confidence=0.8))
        return segments

    def _classify_with_uppercase(self, text: str) -> list[Segment]:
        result = self._try_prefix_then_particle(text)
        if result:
            return result
        # No suffix split found — treat whole thing as English
        return [Segment(text=text, lang="en", confidence=0.85)]

    def _try_suffix_split(self, text: str) -> list[Segment] | None:
        return self._try_prefix_then_particle(text)

    def _try_prefix_then_particle(self, text: str) -> list[Segment] | None:
        """Split text into longest english_prefix + particle_suffix.

        Uses English dictionary prefix match first, then checks if the
        remainder is a known particle. This resolves ambiguity like
        'appleun' -> 'apple' + 'eun' (not 'appl' + 'eun').
        """
        text_lower = text.lower()
        prefixes = self._dict.prefix_match(text_lower)
        for prefix in prefixes:  # longest first
            remainder = text_lower[len(prefix):]
            if remainder:
                # Check both particles AND endings
                is_particle = self._particle_trie.is_particle(remainder)
                is_ending = self._particle_trie.is_ending(remainder)
                if is_particle or is_ending:
                    english_part = text[:len(prefix)]
                    return [
                        Segment(text=english_part, lang="en", confidence=0.9),
                        Segment(text=remainder, lang="ko", confidence=0.9),
                    ]
        return None
