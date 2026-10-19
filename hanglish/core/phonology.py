"""Bounded Korean-to-Revised-Romanization conversion.

This module supplies a deterministic forward check for candidate generation.  It
implements the common, local sound changes documented in ``docs/phonology.md``;
it includes a bounded POS-backed predicate rule, not a complete pronunciation
or morphology analyser.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from hanglish.core.predicate_boundaries import lateral_boundaries


_HANGUL_RUN = re.compile(r"[\uac00-\ud7a3]+")

_ONSETS = (
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)
_VOWELS = (
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
)
_CODAS = (
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)

_ONSET_RR = dict(zip(_ONSETS, (
    "g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s",
    "ss", "", "j", "jj", "ch", "k", "t", "p", "h",
)))
_VOWEL_RR = dict(zip(_VOWELS, (
    "a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa",
    "wae", "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i",
)))
_CODA_RR = {
    "": "", "ㄱ": "k", "ㄲ": "k", "ㄳ": "k", "ㄴ": "n",
    "ㄵ": "n", "ㄶ": "n", "ㄷ": "t", "ㄹ": "l", "ㄺ": "k",
    "ㄻ": "m", "ㄼ": "l", "ㄽ": "l", "ㄾ": "l", "ㄿ": "p",
    "ㅀ": "l", "ㅁ": "m", "ㅂ": "p", "ㅄ": "p", "ㅅ": "t",
    "ㅆ": "t", "ㅇ": "ng", "ㅈ": "t", "ㅊ": "t", "ㅋ": "k",
    "ㅌ": "t", "ㅍ": "p", "ㅎ": "t",
}

# These are official RR examples whose inserted ㄴ/ㄹ cannot be inferred from
# adjacent written jamo alone.  Keeping this map small makes the boundary clear.
_INSERTION_EXCEPTIONS = {
    "학여울": "항녀울",
    "알약": "알략",
}

# ㄼ-final stem syllables with the passive/causative suffix -히-. Match
# prefixed stems too (짓밟-, 드넓-), but not nouns such as 여덟 in 여덟하고.
_LABIAL_HI_STEM_FINALS = {"밟", "넓"}

_LIAISON_SPLIT = {
    "ㄱ": ("", "ㄱ"), "ㄲ": ("", "ㄲ"), "ㄳ": ("ㄱ", "ㅅ"),
    "ㄴ": ("", "ㄴ"), "ㄵ": ("ㄴ", "ㅈ"), "ㄶ": ("ㄴ", "ㅎ"),
    "ㄷ": ("", "ㄷ"), "ㄹ": ("", "ㄹ"), "ㄺ": ("ㄹ", "ㄱ"),
    "ㄻ": ("ㄹ", "ㅁ"), "ㄼ": ("ㄹ", "ㅂ"), "ㄽ": ("ㄹ", "ㅅ"),
    "ㄾ": ("ㄹ", "ㅌ"), "ㄿ": ("ㄹ", "ㅍ"), "ㅀ": ("ㄹ", "ㅎ"),
    "ㅁ": ("", "ㅁ"), "ㅂ": ("", "ㅂ"), "ㅄ": ("ㅂ", "ㅅ"),
    "ㅅ": ("", "ㅅ"), "ㅆ": ("", "ㅆ"), "ㅈ": ("", "ㅈ"),
    "ㅊ": ("", "ㅊ"), "ㅋ": ("", "ㅋ"), "ㅌ": ("", "ㅌ"),
    "ㅍ": ("", "ㅍ"), "ㅎ": ("", "ㅎ"),
}

_VELAR_CODAS = {"ㄱ", "ㄲ", "ㄳ", "ㄺ", "ㅋ"}
_CORONAL_CODAS = {"ㄷ", "ㅅ", "ㅆ", "ㅈ", "ㅊ", "ㅌ", "ㅎ"}
_LABIAL_CODAS = {"ㅂ", "ㅄ", "ㅍ", "ㄿ"}
_ASPIRATE = {
    ("ㄱ", "ㅎ"): "ㅋ", ("ㄲ", "ㅎ"): "ㅋ", ("ㅋ", "ㅎ"): "ㅋ",
    ("ㄷ", "ㅎ"): "ㅌ", ("ㅅ", "ㅎ"): "ㅌ", ("ㅆ", "ㅎ"): "ㅌ",
    ("ㅈ", "ㅎ"): "ㅊ", ("ㅊ", "ㅎ"): "ㅊ", ("ㅂ", "ㅎ"): "ㅍ",
    ("ㅍ", "ㅎ"): "ㅍ",
}


@dataclass
class _Syllable:
    onset: str
    vowel: str
    coda: str


def _decompose(char: str) -> _Syllable:
    value = ord(char) - 0xAC00
    onset, rest = divmod(value, 21 * 28)
    vowel, coda = divmod(rest, 28)
    return _Syllable(_ONSETS[onset], _VOWELS[vowel], _CODAS[coda])


def _apply_pair(left: _Syllable, right: _Syllable) -> None:
    """Apply supported sound changes at one written syllable boundary."""
    if not left.coda:
        return

    # ㅎ is deleted before a vowel; ㄶ/ㅀ retain their first component.
    if right.onset == "ㅇ" and left.coda in {"ㅎ", "ㄶ", "ㅀ"}:
        carried = {"ㅎ": "ㅇ", "ㄶ": "ㄴ", "ㅀ": "ㄹ"}[left.coda]
        left.coda = ""
        right.onset = carried
        return

    # Palatalization before 이 is applied before ordinary liaison.  In ㄾ, the
    # ㄹ remains as the coda while its second component ㅌ carries over as ㅊ.
    if right.onset == "ㅇ" and right.vowel == "ㅣ" and left.coda in {"ㄷ", "ㅌ", "ㄾ"}:
        right.onset = "ㅈ" if left.coda == "ㄷ" else "ㅊ"
        left.coda = "ㄹ" if left.coda == "ㄾ" else ""
        return

    # A final consonant (or the second consonant of a cluster) carries over.
    # Final ㅇ has no corresponding initial sound and therefore stays put.
    if right.onset == "ㅇ" and left.coda in _LIAISON_SPLIT:
        left.coda, right.onset = _LIAISON_SPLIT[left.coda]
        if right.onset == "ㅎ":
            right.onset = "ㅇ"
        return

    # ㄷ+히/혀 first aspirates and then palatalizes: 굳히다 -> 구치다.
    if left.coda == "ㄷ" and right.onset == "ㅎ" and right.vowel in {"ㅣ", "ㅕ"}:
        left.coda = ""
        right.onset = "ㅊ"
        return

    # A following ㅎ combines with an obstruent final into an aspirated onset.
    aspirated = _ASPIRATE.get((left.coda, right.onset))
    if aspirated is not None:
        left.coda = ""
        right.onset = aspirated
        return

    # Final ㅎ combines with a following plain obstruent: 좋고 -> 조코.
    if left.coda in {"ㅎ", "ㄶ", "ㅀ"} and right.onset in {"ㄱ", "ㄷ", "ㅈ"}:
        left.coda = {"ㅎ": "", "ㄶ": "ㄴ", "ㅀ": "ㄹ"}[left.coda]
        right.onset = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅈ": "ㅊ"}[right.onset]
        return

    # Obstruents assimilate to the place of a following nasal.  Before ㄹ,
    # the obstruent nasalizes and ㄹ is pronounced ㄴ: 국립 -> 궁닙.
    if right.onset in {"ㄴ", "ㅁ", "ㄹ"}:
        nasalized = True
        if left.coda in _VELAR_CODAS:
            left.coda = "ㅇ"
        elif left.coda in _CORONAL_CODAS:
            left.coda = "ㄴ"
        elif left.coda in _LABIAL_CODAS:
            left.coda = "ㅁ"
        else:
            nasalized = False
        if nasalized and right.onset == "ㄹ":
            right.onset = "ㄴ"

    # ㄹ is pronounced ㄴ after the written nasal codas ㅁ and ㅇ.
    if right.onset == "ㄹ" and left.coda in {"ㅁ", "ㅇ"}:
        right.onset = "ㄴ"

    # Liquid assimilation applies in both orders at a local boundary.
    if left.coda == "ㄴ" and right.onset == "ㄹ":
        left.coda = "ㄹ"
        right.onset = "ㄹ"
    elif left.coda == "ㄹ" and right.onset == "ㄴ":
        right.onset = "ㄹ"


def _romanize_run(run: str) -> str:
    pronunciation = _INSERTION_EXCEPTIONS.get(run, run)
    syllables = [_decompose(char) for char in pronunciation]
    # Predicate-final ㄺ is lateral before ㄱ; noun clusters (흙과, 닭고기)
    # retain the ordinary coda rule. Automatic tensing is not written in RR.
    for index in lateral_boundaries(pronunciation):
        syllables[index].coda = "ㄹ"
    # 밟- is the lexical ㄼ exception: use ㅂ before a consonantal ending
    # (then apply nasal assimilation). Keep the cluster for vowel liaison and
    # ㅎ aspiration, which retain ㄹ. This also covers prefixed 짓밟-.
    for index, char in enumerate(pronunciation):
        right = syllables[index + 1] if index + 1 < len(syllables) else None
        if (right is not None and char in _LABIAL_HI_STEM_FINALS
                and right.onset == "ㅎ" and right.vowel in {"ㅣ", "ㅕ"}):
            syllables[index].coda = "ㄹ"
            right.onset = "ㅍ"
            continue
        if char == "밟" and (
            right is None or right.onset not in {"ㅇ", "ㅎ"}
        ):
            syllables[index].coda = "ㅂ"
    for left, right in zip(syllables, syllables[1:]):
        _apply_pair(left, right)
    pieces = []
    for index, item in enumerate(syllables):
        onset = _ONSET_RR[item.onset]
        if item.onset == "ㄹ" and index and syllables[index - 1].coda == "ㄹ":
            onset = "l"
        pieces.append(onset + _VOWEL_RR[item.vowel] + _CODA_RR[item.coda])
    return "".join(pieces)


def romanize_word(text: str) -> str:
    """Romanize contiguous precomposed Hangul runs and preserve other text.

    Phonological changes never cross a non-Hangul boundary.  Latin text,
    punctuation, whitespace, digits, and their original case are copied exactly.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return _HANGUL_RUN.sub(lambda match: _romanize_run(match.group()), text)


__all__ = ["romanize_word"]
