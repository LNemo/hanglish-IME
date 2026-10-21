"""Bounded Korean-to-Revised-Romanization conversion.

This module supplies a deterministic forward check for candidate generation.  It
implements local sound changes plus source-backed lexical exceptions and a
bounded POS-backed predicate rule. It is not a complete pronunciation or
morphology analyser, a proper-name formatter, or academic transliteration.
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
_LEXICAL_PRONUNCIATIONS = {
    "학여울": "항녀울",
    "알약": "알략",
    # Standard Pronunciation art.20: ㄴ+ㄹ -> ㄴㄴ in these lexical items.
    # 신문로 is also an explicit RR ch.3 art.1 example.
    "신문로": "신문노",
    "의견란": "의견난",
    "생산량": "생산냥",
    "상견례": "상견녜",
    "횡단로": "횡단노",
    "이원론": "이원논",
    "입원료": "입원뇨",
    "구근류": "구근뉴",
    "임진란": "임진난",
    "결단력": "결단녁",
    "공권력": "공권녁",
    "동원령": "동원녕",
    # Standard Pronunciation art.29: known compound boundaries, not every
    # consonant+이/여 boundary. Retain 잎's ㅍ for subsequent vowel liaison.
    "솜이불": "솜니불",
    "꽃잎": "꽃닢",
    "막일": "막닐",
    "홑이불": "홑니불",
    "삯일": "삯닐",
    "맨입": "맨닙",
    "내복약": "내복냑",
    "한여름": "한녀름",
    "색연필": "색년필",
    "늑막염": "늑막념",
    "콩엿": "콩녓",
    "담요": "담뇨",
    "눈요기": "눈뇨기",
    "영업용": "영업뇽",
    "식용유": "식용뉴",
    "백분율": "백분뉼",
    "밤윷": "밤뉻",
    "남존여비": "남존녀비",
    "신여성": "신녀성",
    # Standard Pronunciation art.15: neutralize before an independent
    # vowel-initial morpheme, unlike ordinary particle/ending liaison.
    "값어치": "갑어치",
    "닭앞": "닥앞",
}

# RR ch.3 art.1(4): noun ㄱ/ㄷ/ㅂ + ㅎ keeps the written ㅎ. A local
# aspiration rule cannot distinguish these nouns from 잡혀 -> japyeo.
_NOUN_H_BOUNDARIES = {
    "묵호": 0, "집현전": 0, "맏형": 0, "직행열차": 0, "각하": 0, "오죽헌": 1,
}
_PARTICLES = (
    "으로부터", "에서부터", "에게서", "한테서", "으로", "에서", "에게", "한테",
    "까지", "부터", "처럼", "보다", "하고", "이라도", "라도", "이나", "나",
    "이", "가", "을", "를", "은", "는", "에", "의", "와", "과", "로", "도", "만",
)
_LEXICAL_PREDICATE_ENDINGS = frozenset({
    "", "다", "고", "게", "기", "는", "던", "지", "지만", "으면", "어서",
    "어", "어요", "었다", "었어요", "습니다", "습니까", "음",
})
_LEXICAL_PREDICATES = {"맛없": "맏없", "넋없": "넉없", "값있": "갑있"}

# Compatibility contract for the model's conservative prefix pruning. Every
# lexically conditioned rewrite must be represented, including noun ㅎ rules.
_INSERTION_EXCEPTIONS = {
    **_LEXICAL_PRONUNCIATIONS,
    **{noun: noun for noun in _NOUN_H_BOUNDARIES},
    **_LEXICAL_PREDICATES,
}


def _known_noun_tail(tail: str, depth: int = 0) -> bool:
    """Recognize at most two particles; do not rewrite an arbitrary prefix."""
    if not tail:
        return True
    return depth < 2 and any(
        tail.startswith(particle) and _known_noun_tail(tail[len(particle):], depth + 1)
        for particle in _PARTICLES
    )


def _lexical_pronunciation(run: str) -> tuple[str, frozenset[int]]:
    for noun, boundary in _NOUN_H_BOUNDARIES.items():
        if run.startswith(noun) and _known_noun_tail(run[len(noun):]):
            pronunciation = run
            if noun == "직행열차":
                pronunciation = "직행녈차" + run[len(noun):]
            return pronunciation, frozenset({boundary})
    for noun, pronunciation in _LEXICAL_PRONUNCIATIONS.items():
        if run.startswith(noun) and _known_noun_tail(run[len(noun):]):
            return pronunciation + run[len(noun):], frozenset()
    for stem, pronunciation in _LEXICAL_PREDICATES.items():
        if run.startswith(stem) and run[len(stem):] in _LEXICAL_PREDICATE_ENDINGS:
            return pronunciation + run[len(stem):], frozenset()
    return run, frozenset()

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
        # Simplify clusters before nasal/liquid assimilation (arts.10–12,18,20).
        left.coda = {
            "ㄵ": "ㄴ", "ㄶ": "ㄴ", "ㄻ": "ㅁ", "ㄼ": "ㄹ",
            "ㄽ": "ㄹ", "ㄾ": "ㄹ", "ㅀ": "ㄹ",
        }.get(left.coda, left.coda)
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
    pronunciation, noun_h_boundaries = _lexical_pronunciation(run)
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
        # Standard Pronunciation art.12: retain ㄹ in 밝히다/얽히다.
        # Do not apply this to a noun followed by unrelated 하고.
        if (right is not None and syllables[index].coda == "ㄺ"
                and right.onset == "ㅎ" and right.vowel in {"ㅣ", "ㅕ"}):
            syllables[index].coda = "ㄹ"
            right.onset = "ㅋ"
            continue
        if (right is not None and syllables[index].coda == "ㄵ"
                and right.onset == "ㅎ" and right.vowel in {"ㅣ", "ㅕ"}):
            syllables[index].coda = "ㄴ"
            right.onset = "ㅊ"
            continue
        if (right is not None and char in _LABIAL_HI_STEM_FINALS
                and right.onset == "ㅎ" and right.vowel in {"ㅣ", "ㅕ"}):
            syllables[index].coda = "ㄹ"
            right.onset = "ㅍ"
            continue
        if char == "밟" and (
            right is None or right.onset not in {"ㅇ", "ㅎ"}
        ):
            syllables[index].coda = "ㅂ"
    for index, (left, right) in enumerate(zip(syllables, syllables[1:])):
        if index in noun_h_boundaries:
            continue
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
