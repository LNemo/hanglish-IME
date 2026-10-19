"""Bounded, on-demand predicate endings with shared phonological prefixes.

Four boundary indexes replace a stem-by-ending cross product. Irregular
connective forms require attested dictionary surfaces. This is not a complete
Korean morphology analyser; see docs/productive-endings.md for its scope.
"""
from collections import defaultdict

from hanglish.core.phonology import romanize_word


# Each representative has an open first syllable, so the remainder can be
# shared without hiding a second consonant boundary. Use a *complete* ending
# in the forward probe: bare 거 would not license the predicate ㄺ+ㄱ rule.
ENDING_GROUPS = (
    ('거나', ('거나', '거든', '거든요', '거니와', '거늘')),
    ('고서', ('고서', '고도', '고요')),
    ('기로', ('기로', '기에')),
)
AMBIGUOUS_IRREGULAR_CODAS = {7, 17, 19, 27}  # ㄷ, ㅂ, ㅅ, ㅎ


def regular_ni(stem):
    """Connective -(으)니; uncertain irregular classes remain dictionary-only."""
    final = (ord(stem[-1]) - 0xAC00) % 28
    if final in AMBIGUOUS_IRREGULAR_CODAS:
        return None
    if final == 8:  # ㄹ drops before the connective 니.
        return stem[:-1] + chr(ord(stem[-1]) - final) + '니'
    return stem + ('으니' if final else '니')


class ProductiveEndings:
    def __init__(self, stems, attested_ni):
        self.groups = []
        self.indexed_forms = 0
        stems = sorted(set(stems))
        for representative, endings in ENDING_GROUPS:
            index = defaultdict(set)
            tail = romanize_word(representative[1:])
            for stem in stems:
                joined = romanize_word(stem + representative)
                if joined.endswith(tail):
                    index[joined[:-len(tail)]].add(stem)
            self.groups.append((index, tuple((ending, romanize_word(ending[1:]))
                                             for ending in endings)))
            self.indexed_forms += sum(map(len, index.values()))
        self.ni = defaultdict(set)
        for stem in stems:
            forms = attested_ni.get(stem) or {regular_ni(stem)}
            for form in forms:
                if form:
                    self.ni[romanize_word(form)].add(form)
        self.indexed_forms += sum(map(len, self.ni.values()))

    def options(self, source):
        found = set()
        for index, endings in self.groups:
            for ending, tail in endings:
                if source.endswith(tail) and len(source) > len(tail):
                    for stem in index.get(source[:-len(tail)], ()):
                        text = stem + ending
                        if romanize_word(text) == source:
                            found.add(text)
        for tail in ('', '까'):
            roman_tail = romanize_word(tail)
            if roman_tail and not source.endswith(roman_tail):
                continue
            prefix = source[:-len(roman_tail)] if roman_tail else source
            for form in self.ni.get(prefix, ()):
                text = form + tail
                if romanize_word(text) == source:
                    found.add(text)
        return tuple(sorted(found))
