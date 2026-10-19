"""Experimental lexical joins across liaison/assimilation boundaries.

This proposal source leaves production candidates and scores untouched. Rules
are derived from the existing forward phonology, and every complete proposal
must reproduce the input. No evaluation spelling is added to the lexicon.
"""
from functools import lru_cache
import math
import re

from hanglish.core.lattice import WordOption, keep_strongest, suffix_allowed
from hanglish.core.phonology import romanize_word

ROMAN = re.compile(r"[a-z]{2,40}\Z")
HANGUL = re.compile(r"[가-힣]{1,12}\Z")
MAX_LEXEMES_PER_SIDE = 4
MAX_PROPOSALS = 16


@lru_cache(maxsize=1)
def boundary_rules():
    """Map observed boundary sound to the two isolated dictionary keys.

    Keep the following vowel in each pattern: palatalization depends on it.
    Do not replace substrings in a spelling; use matches only to query lexemes.
    """
    rules = set()
    for final in range(1, 28):
        left = chr(ord("아") + final)
        isolated_tail = romanize_word(left)[1:]
        for onset in range(19):
            for vowel in range(21):
                right = chr(0xAC00 + onset * 588 + vowel * 28)
                right_key = romanize_word(right)
                surface = romanize_word(left + right)[1:]
                if surface != isolated_tail + right_key:
                    rules.add((surface, isolated_tail, right_key))
    return tuple(sorted(rules))


class BoundaryComposer:
    def __init__(self, generator):
        self.generator = generator
        grouped = {}
        for rule in boundary_rules():
            grouped.setdefault(rule[0][0], []).append(rule)
        self.rules_by_initial = grouped
        self.compounds = lru_cache(maxsize=4096)(self._compounds)
        self.propose = lru_cache(maxsize=4096)(self._propose)

    def _lexemes(self, key):
        values = self.generator._lexical_options(key)
        return sorted((o for o in values if HANGUL.fullmatch(o.text)
                       and type(o.score) in (int, float) and math.isfinite(o.score)),
                      key=lambda o: (-o.score, o.text))[:MAX_LEXEMES_PER_SIDE]

    def _compounds(self, roman):
        if not isinstance(roman, str) or ROMAN.fullmatch(roman) is None:
            return ()
        keys = set()
        # Also include unchanged joins. They can be whole words with no suffix,
        # which the existing stem-only composition path does not always query.
        for split in range(1, len(roman)):
            keys.add((roman[:split], roman[split:]))
            for surface, left_tail, right_head in self.rules_by_initial.get(roman[split], ()):
                if roman.startswith(surface, split):
                    keys.add((roman[:split] + left_tail,
                              right_head + roman[split + len(surface):]))
        found = {}
        for left_key, right_key in sorted(keys):
            left = self._lexemes(left_key)
            if not left:
                continue
            right = self._lexemes(right_key)
            for first in left:
                for second in right:
                    text = first.text + second.text
                    if len(text) <= 12 and romanize_word(text) == roman:
                        keep_strongest(found, WordOption(
                            text, min(first.score, second.score) - .4, "morphology"))
        return tuple(sorted(found.values(), key=lambda o: (-o.score, o.text))[:MAX_PROPOSALS])

    def _propose(self, source):
        if (not isinstance(source, str) or ROMAN.fullmatch(source) is None
                or source in self.generator.english):
            return ()
        found = {o.text: o for o in self.compounds(source)}
        for surface, suffix, restored in self.generator.korean_boundaries:
            if not source.endswith(surface) or len(source) <= len(surface):
                continue
            stem_key = source[:-len(surface)] + restored
            for stem in self.compounds(stem_key):
                text = stem.text + suffix
                if (len(text) <= 16 and suffix_allowed(stem.text, suffix)
                        and romanize_word(text) == source):
                    keep_strongest(found, WordOption(text, stem.score - .2, "morphology"))
        return tuple(sorted(found.values(), key=lambda o: (-o.score, o.text))[:MAX_PROPOSALS])
