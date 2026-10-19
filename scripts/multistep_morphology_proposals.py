"""Source-only proposals for three conservative multi-step morphology families."""
from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

from hanglish.core.lattice import DATA, suffix_allowed
from hanglish.core.lexical_data import read_lexicon
from hanglish.core.phonology import romanize_word


ROMAN = re.compile(r"[a-z]{2,40}\Z")
PREDICATE_POS = frozenset({"VV", "VA", "VX"})
NOMINAL_POS = frozenset({"NNG", "NNP", "NP", "NR"})
NOMINALIZER_SUFFIXES = ("가", "를", "는", "에", "에서", "로", "부터", "까지", "도", "만")
PLURAL_CASE_SUFFIXES = ("은", "는", "이", "가", "을", "를", "에", "에서", "에게", "한테", "께", "로", "으로", "부터", "까지", "도", "만")
PLURAL_FOLLOW_POS = frozenset({"JKS", "JKO", "JKB", "JKG", "JX", "JC"})
STACKED_POSTPOSITIONS = (
    ("에", "도"), ("에", "만"),
    ("에서", "도"), ("에서", "만"), ("에서", "는"),
    ("에게", "도"), ("에게", "만"), ("에게", "는"),
    ("으로", "도"), ("으로", "는"), ("로", "도"), ("로", "는"),
    ("까지", "는"), ("부터", "는"),
)
MAN_FOLLOWERS = ("이", "을", "은")
FAMILY_ORDER = {
    "predicate_nominalizer_case": 0,
    "nominal_plural_case": 1,
    "nominal_postposition_auxiliary": 2,
    "nominal_man_case": 3,
}
MAX_PROPOSALS = 16


@dataclass(frozen=True)
class MorphProposal:
    text: str
    family: str
    stem: str
    stem_pos: str
    stem_cost: int
    suffixes: tuple[str, ...]
    constituents: tuple[str, ...]
    lexicon_fingerprint: str
    provenance: str = "mecab-ko-dic POS + fixed source-only morphology grammar"


@lru_cache(maxsize=None)
def boundary_variants(suffix: str):
    """Return observed suffix surface and the isolated stem tail to restore."""
    if not suffix or any(not ("가" <= char <= "힣") for char in suffix):
        raise ValueError("suffix must be nonempty Hangul")
    variants = {(romanize_word(suffix), "")}
    for final in range(1, 28):
        sample = chr(0xAC00 + final)
        isolated = romanize_word(sample)
        joined = romanize_word(sample + suffix)
        common = 0
        while (common < min(len(isolated), len(joined))
               and isolated[common] == joined[common]):
            common += 1
        surface = joined[common:]
        if surface:
            variants.add((surface, isolated[common:]))
    return tuple(sorted(variants, key=lambda item: (-len(item[0]), item)))


def read_attested_plural_bases(path):
    """Read only explicit noun+들/XSN analyses from the pinned POS lexicon."""
    found = {}
    with Path(path).open(encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) < 12:
                continue
            parts = [part.split("/") for part in row[11].split("+")]
            if (len(parts) != 3 or len(parts[0]) < 2 or len(parts[1]) < 2
                    or len(parts[2]) < 2 or parts[0][1] not in NOMINAL_POS
                    or parts[1][:2] != ["들", "XSN"]
                    or parts[2][1] not in PLURAL_FOLLOW_POS):
                continue
            final_surface = re.sub(r"\d+$", "", parts[2][0])
            if (final_surface not in PLURAL_CASE_SUFFIXES
                    or not row[0].endswith(final_surface)):
                continue
            plural = row[0][:-len(final_surface)]
            if not plural.endswith("들") or len(plural) < 2:
                continue
            stem = plural[:-1]
            analyzed_stem = re.sub(r"\d+$", "", parts[0][0])
            if analyzed_stem != stem:
                continue
            record = (stem, parts[0][1], int(row[3]), row[11])
            previous = found.get((stem, parts[0][1]))
            if previous is None or record[2:] < previous[2:]:
                found[(stem, parts[0][1])] = record
    return tuple(sorted(found.values(), key=lambda item: (item[2], item[0], item[1], item[3])))


class MultiStepMorphology:
    def __init__(self, entries, lexicon_fingerprint, english=(), attested_plurals=()):
        if not isinstance(lexicon_fingerprint, str) or not lexicon_fingerprint:
            raise ValueError("lexicon fingerprint is required")
        self.lexicon_fingerprint = lexicon_fingerprint
        self.english = frozenset(english)
        indexed = defaultdict(dict)
        for word, pos, cost in entries:
            if (pos not in PREDICATE_POS | NOMINAL_POS
                    or not isinstance(word, str) or not word
                    or any(not ("가" <= char <= "힣") for char in word)
                    or type(cost) is not int):
                continue
            key = romanize_word(word)
            identity = (word, pos)
            previous = indexed[key].get(identity)
            if previous is None or cost < previous[2]:
                indexed[key][identity] = (word, pos, cost)
        self.index = {
            key: tuple(sorted(values.values(), key=lambda item: (item[2], item[0], item[1])))
            for key, values in indexed.items()
        }
        plural_index = defaultdict(list)
        for stem, pos, cost, evidence in attested_plurals:
            if (pos in NOMINAL_POS and isinstance(stem, str) and stem
                    and not stem.endswith("들") and type(cost) is int
                    and isinstance(evidence, str) and evidence):
                plural_index[romanize_word(stem + "들")].append(
                    (stem, pos, cost, evidence)
                )
        self.plural_index = {
            key: tuple(sorted(values, key=lambda item: (item[2], item[0], item[1], item[3])))
            for key, values in plural_index.items()
        }
        self.propose = lru_cache(maxsize=4096)(self._propose)

    @classmethod
    def from_data(cls, data=DATA):
        data = Path(data)
        entries, fingerprint, _ = read_lexicon(data / "external/mecab-ko-dic")
        plurals = read_attested_plural_bases(data / "external/mecab-ko-dic/Inflect.csv")
        english = set((data / "common_english.txt").read_text().lower().split())
        english.update((data / "it_terms.txt").read_text().lower().split())
        english.update({"test", "api", "hello", "world", "openai"})
        return cls(entries, fingerprint, english, plurals)

    def _stems(self, source, suffix_text, allowed_pos):
        found = {}
        for surface, restored in boundary_variants(suffix_text):
            if not source.endswith(surface) or len(source) <= len(surface):
                continue
            stem_key = source[:-len(surface)] + restored
            for stem in self.index.get(stem_key, ()):
                if stem[1] in allowed_pos:
                    found[(stem[0], stem[1])] = stem
        return tuple(sorted(found.values(), key=lambda item: (item[2], item[0], item[1])))

    def _candidate(self, source, family, stem, suffixes):
        word, pos, cost = stem
        text = word + "".join(suffixes)
        if romanize_word(text) != source:
            return None
        return MorphProposal(
            text=text, family=family, stem=word, stem_pos=pos, stem_cost=cost,
            suffixes=tuple(suffixes), constituents=(word, *suffixes),
            lexicon_fingerprint=self.lexicon_fingerprint,
        )

    def _propose(self, source):
        if (not isinstance(source, str) or ROMAN.fullmatch(source) is None
                or source in self.english):
            return ()
        found = {}

        for case in NOMINALIZER_SUFFIXES:
            chain = "기" + case
            for stem in self._stems(source, chain, PREDICATE_POS):
                base = stem[0] + "기"
                if suffix_allowed(base, case):
                    candidate = self._candidate(
                        source, "predicate_nominalizer_case", stem, ("기", case)
                    )
                    if candidate is not None:
                        found.setdefault(candidate.text, candidate)

        for case in PLURAL_CASE_SUFFIXES:
            for surface, restored in boundary_variants(case):
                if not source.endswith(surface) or len(source) <= len(surface):
                    continue
                plural_key = source[:-len(surface)] + restored
                for stem_text, pos, cost, evidence in self.plural_index.get(plural_key, ()):
                    base = stem_text + "들"
                    stem = (stem_text, pos, cost)
                    if not suffix_allowed(base, case):
                        continue
                    candidate = self._candidate(
                        source, "nominal_plural_case", stem, ("들", case)
                    )
                    if candidate is not None:
                        candidate = MorphProposal(
                            **{**candidate.__dict__,
                               "provenance": "attested Inflect noun+들/XSN: " + evidence}
                        )
                        found.setdefault(candidate.text, candidate)

        for postposition, auxiliary in STACKED_POSTPOSITIONS:
            chain = postposition + auxiliary
            for stem in self._stems(source, chain, NOMINAL_POS):
                if (not suffix_allowed(stem[0], postposition)
                        or not suffix_allowed(stem[0] + postposition, auxiliary)):
                    continue
                candidate = self._candidate(
                    source, "nominal_postposition_auxiliary", stem,
                    (postposition, auxiliary),
                )
                if candidate is not None:
                    found.setdefault(candidate.text, candidate)

        for follower in MAN_FOLLOWERS:
            chain = "만" + follower
            for stem in self._stems(source, chain, NOMINAL_POS):
                if not suffix_allowed(stem[0] + "만", follower):
                    continue
                candidate = self._candidate(
                    source, "nominal_man_case", stem, ("만", follower)
                )
                if candidate is not None:
                    found.setdefault(candidate.text, candidate)

        ordered = sorted(found.values(), key=lambda item: (
            FAMILY_ORDER[item.family], item.stem_cost, item.text,
            item.stem_pos, item.constituents,
        ))
        return tuple(ordered[:MAX_PROPOSALS])


__all__ = [
    "MAN_FOLLOWERS", "MAX_PROPOSALS", "MorphProposal", "MultiStepMorphology",
    "NOMINALIZER_SUFFIXES", "NOMINAL_POS", "PLURAL_CASE_SUFFIXES",
    "PLURAL_FOLLOW_POS",
    "PREDICATE_POS", "STACKED_POSTPOSITIONS", "boundary_variants",
    "read_attested_plural_bases",
]
