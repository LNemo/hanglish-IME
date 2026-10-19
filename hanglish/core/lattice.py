"""Word candidates from a reusable lexicon and productive morphology.

Evaluation data is intentionally not a runtime dependency. Known spellings are
indexed by a forward phonology implementation, including new stem/suffix pairs.
"""

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re

from hanglish.core.phonology import romanize_word
from hanglish.core.lexical_data import read_attested_ni_forms, read_lexicon
from hanglish.core.productive_endings import ProductiveEndings


DATA = Path(__file__).resolve().parents[2] / "data"
HANGUL = re.compile(r"[가-힣]+")
ROMAN = re.compile(r"[a-z]+\Z")


@dataclass(frozen=True)
class WordOption:
    text: str
    score: float
    kind: str


def keep_strongest(options, option):
    """Merge independent evidence without letting a later source lower it."""
    previous = options.get(option.text)
    if previous is None or option.score > previous.score:
        options[option.text] = option


def suffix_allowed(stem, suffix):
    """Productive noun allomorphs; ㄹ uses 로 rather than 으로."""
    final = (ord(stem[-1]) - 0xAC00) % 28
    pairs = {"은": True, "는": False, "이": True, "가": False,
             "을": True, "를": False, "과": True, "와": False,
             "이에요": True, "예요": False, "이랑": True, "랑": False,
             "이었다": True, "였다": False, "이었어요": True, "였어요": False}
    if suffix in pairs:
        return bool(final) == pairs[suffix]
    if suffix.startswith("으로"):
        return final not in (0, 8)
    if suffix.startswith("로"):
        return final in (0, 8)
    return True


def conjugated_base(stem):
    """Regular 아/어 contraction; irregular verbs remain lexical/model cases."""
    value = ord(stem[-1]) - 0xAC00
    onset, rest = divmod(value, 588)
    vowel, final = divmod(rest, 28)
    if final:
        return stem + ("아" if vowel in (0, 8) else "어")
    if stem.endswith("하"):
        return stem[:-1] + "해"
    contracted = {0: 0, 1: 1, 4: 4, 5: 5, 8: 9, 13: 14, 20: 6}
    if vowel == 18:  # ㅡ deletion depends on the preceding syllable vowel.
        previous = (ord(stem[-2]) - 0xAC00) // 28 % 21 if len(stem) > 1 else -1
        contracted[18] = 0 if previous in (0, 8) else 4
    if vowel in contracted:
        return stem[:-1] + chr(0xAC00 + onset * 588 + contracted[vowel] * 28)
    return stem + "어"


def replace_final(syllable, final):
    """Return a Hangul syllable with a different jongseong index."""
    value = ord(syllable) - 0xAC00
    return chr(0xAC00 + value // 28 * 28 + final)


def predicate_present_forms(stem):
    """Productive declarative and adnominal present forms."""
    final = (ord(stem[-1]) - 0xAC00) % 28
    if final == 0:
        declarative = stem[:-1] + replace_final(stem[-1], 4) + "다"
        adnominal = stem + "는"
    elif final == 8:  # ㄹ drops before ㄴ.
        base = stem[:-1] + replace_final(stem[-1], 0)
        declarative = base[:-1] + replace_final(base[-1], 4) + "다"
        adnominal = base + "는"
    else:
        declarative = stem + "는다"
        adnominal = stem + "는"
    return declarative, adnominal


class CandidateGenerator:
    def __init__(self, data=DATA, external_lexicon=True):
        self.index = defaultdict(dict)
        self.inflections = defaultdict(dict)
        self.dictionary_index = defaultdict(dict)
        self.data = Path(data)
        seed = json.loads((self.data / "morphology_lexicon.json").read_text())
        self.nouns = set(seed["nouns"].split())
        forms = set(seed["forms"].split())
        self.suffixes = seed["suffixes"]
        source = self.data / "challenge" / "train_lexicon.json"
        self.training_words = set()
        if source.exists():
            self.training_words.update(item["text"] for item in json.loads(source.read_text())["entries"])
        for word in sorted(self.nouns | forms | self.training_words):
            if HANGUL.fullmatch(word):
                self.add(word, 7., "lexicon")
        # Independent pre-existing language frequencies, never extracted from
        # evaluation answers. Keep the upstream package/data licensing intact.
        self.external_word_count = 0
        self.external_version = None
        self.external_fingerprint = "absent"
        try:
            from importlib.metadata import version
            from wordfreq import top_n_list, get_frequency_dict
        except ImportError:
            pass
        else:
            self.external_version = version("wordfreq")
            frequencies = get_frequency_dict("ko")
            external = {}
            for word in (top_n_list("ko", 50000) if external_lexicon else ()):
                if HANGUL.fullmatch(word) and len(word) <= 12:
                    external[word] = frequencies[word]
                    zipf = math.log10(frequencies[word]) + 9
                    self.add(word, 6. + .1 * zipf, "lexicon")
                    self.external_word_count += 1
            self.external_fingerprint = hashlib.sha256(
                (self.external_version + json.dumps(external, ensure_ascii=False, sort_keys=True)).encode()).hexdigest()
        # POS-aware lexical data is a separate runtime resource. It does not
        # silently alter the transducer's existing independent word split.
        predicates = set()
        roots = set()
        dictionary_entries, dictionary_hash, self.dictionary_rejected_rows = read_lexicon(self.data / "external/mecab-ko-dic") if external_lexicon else ([], "absent", 0)
        self.dictionary_fingerprint = dictionary_hash
        self.external_fingerprint = hashlib.sha256((self.external_fingerprint + dictionary_hash).encode()).hexdigest()
        for word, pos, cost in dictionary_entries:
            roman = romanize_word(word)
            option = WordOption(word, 6.1 + max(0., min(.4, (6000 - cost) / 10000)), "lexicon")
            keep_strongest(self.dictionary_index[roman], option)
            if pos in ("VV", "VA", "VX", "VCP", "VCN"):
                predicates.add(word)
            elif pos == "XR":
                roots.add(word)
        predicates.update(root + "하" for root in roots)
        for stem in sorted(self.nouns):
            for suffix in self.suffixes:
                if suffix_allowed(stem, suffix):
                    self.add(stem + suffix, 6.8, "morphology")
            # Productive 하다 predicates over nouns, without a sentence lookup.
            for ending in ("하다", "하기", "합니다", "해요", "하고", "하는", "하면", "했다", "했어요", "했습니다", "하세요"):
                self.add(stem + ending, 6.4, "morphology")
        # Existing authored spellings also capture explicitly supported lexical
        # exceptions to productive forward phonology.
        for roman, words in json.loads((self.data / "workbench_lexicon.json").read_text()).items():
            for i, word in enumerate(words):
                keep_strongest(self.index[roman], WordOption(word, 7.1 - i * .05, "lexicon"))
        self.english = set((self.data / "common_english.txt").read_text().lower().split())
        self.english.update((self.data / "it_terms.txt").read_text().lower().split())
        self.english.update({"test", "api", "hello", "world", "openai"})
        self.english_suffixes = sorted({romanize_word(s): s for s in self.suffixes}.items(), key=lambda pair: -len(pair[0]))
        self.derivations = [(romanize_word(s), s) for s in ("자", "원", "장", "실", "부", "법", "력", "인", "감", "성")]
        endings = ("하다", "하기", "한다", "합니다", "해요", "하고", "하는", "하면", "했다", "했어요", "했습니다", "하세요")
        self.korean_suffixes = self.english_suffixes + [(romanize_word(ending), ending) for ending in endings]
        for ending in ("었다", "었어요", "었습니다", "았어요", "았다", "았습니다", "는다", "습니다",
                       "어서", "아서", "어", "아", "어요", "아요", "고", "지", "게", "겠다", "겠어요", "겠습니다", "도록", "지만", "는데", "다", "면", "으면"):
            self.korean_suffixes.append((romanize_word(ending), ending))
        # A suffix's surface romanization changes at a coda boundary (for
        # example, ㄱ+ㅎ or a complex coda followed by a vowel). Derive every
        # jongseong class from the forward implementation so runtime lookup can
        # restore the isolated stem before applying the suffix.
        boundaries = set()
        for isolated, hangul in self.korean_suffixes:
            boundaries.add((isolated, hangul, ""))
            for final in range(1, 28):
                sample = chr(0xAC00 + final)
                stem_roman = romanize_word(sample)
                joined_roman = romanize_word(sample + hangul)
                common = 0
                while common < min(len(stem_roman), len(joined_roman)) and stem_roman[common] == joined_roman[common]:
                    common += 1
                surface = joined_roman[common:]
                if surface:
                    boundaries.add((surface, hangul, stem_roman[common:]))
        self.korean_boundaries = sorted(boundaries, key=lambda item: (-len(item[0]), item))
        # A separate inflection index preserves the independent seq2seq word
        # split: synthetic runtime inflections are not silently added to it.
        predicates.update(word[:-1] for word in forms if len(word) > 1 and word.endswith("다"))
        for stem in sorted(predicates):
            if not HANGUL.fullmatch(stem) or len(stem) > 8:
                continue
            base = conjugated_base(stem)
            past = base[:-1] + chr(ord(base[-1]) + 20)
            forms_to_add = [base, base + "요", past + "다", past + "어요", past + "습니다"]
            forms_to_add.extend(stem + ending for ending in ("다", "기", "고", "게", "면", "지만", "도록"))
            forms_to_add.extend(predicate_present_forms(stem))
            for form in forms_to_add:
                roman = romanize_word(form)
                self.inflections[roman][form] = WordOption(form, 5.8, "morphology")
        attested_ni = read_attested_ni_forms(self.data / 'external/mecab-ko-dic') if external_lexicon else {}
        self.productive_endings = ProductiveEndings(
            {stem for stem in predicates if HANGUL.fullmatch(stem) and len(stem) <= 8},
            attested_ni,
        )
        # Caches belong to this instance; a global cached method would retain
        # discarded generators and their sizeable dictionaries during retrains.
        self.stem_options = lru_cache(maxsize=4096)(self._stem_options)
        self.options = lru_cache(maxsize=4096)(self._options)

    def add(self, word, score, kind):
        roman = romanize_word(word)
        if ROMAN.fullmatch(roman):
            keep_strongest(self.index[roman], WordOption(word, score, kind))

    def _lexical_options(self, roman):
        combined = {}
        for source in (self.dictionary_index.get(roman, {}), self.index.get(roman, {})):
            for option in source.values():
                keep_strongest(combined, option)
        return tuple(combined.values())

    def _stem_options(self, roman):
        direct = self._lexical_options(roman)
        derived = {}
        for item in direct:
            keep_strongest(derived, item)
        for ending, hangul in self.derivations:
            if roman.endswith(ending) and len(roman) > len(ending):
                prefix = roman[:-len(ending)]
                for stem in self.index.get(prefix, {}).values():
                    text = stem.text + hangul
                    if romanize_word(text) == roman:
                        keep_strongest(derived, WordOption(text, stem.score - .3, "morphology"))
        # A bounded two-lexeme composition recovers productive names and
        # compounds whose boundary pronunciation differs from either word in
        # isolation (for example, a noun followed by 로). Forward verification
        # prevents the split heuristic from accepting a different reading.
        for split in range(2, len(roman) - 1):
            left = self._lexical_options(roman[:split])
            right = self._lexical_options(roman[split:])
            if not left or not right:
                continue
            for first in sorted(left, key=lambda option: -option.score)[:4]:
                for second in sorted(right, key=lambda option: -option.score)[:4]:
                    text = first.text + second.text
                    if len(text) <= 12 and romanize_word(text) == roman:
                        keep_strongest(derived, WordOption(text, min(first.score, second.score) - .4, "morphology"))
        return tuple(sorted(derived.values(), key=lambda option: (-option.score, option.text))[:16])

    def _options(self, original):
        from hanglish.core.workbench import syllables

        if not original or len(original) > 40:
            return (WordOption(original, 0., "raw"),)
        token = original.lower()
        found = {}
        for source in (self.dictionary_index.get(token, {}), self.index.get(token, {}), self.inflections.get(token, {})):
            for option in source.values():
                keep_strongest(found, option)
        for text in self.productive_endings.options(token):
            keep_strongest(found, WordOption(text, 5.8, "morphology"))
        whole_word = bool(found)
        english = token in self.english or not original.islower()
        keep_strongest(found, WordOption(original, 7.5 if english else 0., "raw"))
        mixed_matches = []
        for ending, hangul in self.english_suffixes:
            if token.endswith(ending) and len(token) > len(ending) + 1:
                prefix = original[:-len(ending)]
                # Unknown Latin words stay available for code-switching. Known
                # Korean stems take precedence; users can always select raw.
                known_english = prefix.lower() in self.english
                case_cued = not prefix.islower()
                if whole_word and original.islower():
                    continue
                if known_english or case_cued or (prefix.lower() not in self.index and prefix.lower() not in self.dictionary_index):
                    mixed_matches.append((ending, hangul, prefix, known_english, case_cued))
                    if original.islower():
                        # Unknown lowercase words have no case cue. Preserve the
                        # established longest-particle boundary (reul over eul).
                        break
        # For an identifier ending in e, both ``Name+do`` and ``Nam+edo`` are
        # mechanically possible. Preserve a competing split when its prefix is
        # independently known English; otherwise keep the full identifier and
        # additive 도. Other particles retain the longest-match policy, which
        # is needed for names followed by 에서, 으로, 등을.
        if not original.islower() and any(match[0] == "do" for match in mixed_matches):
            mixed_matches = [
                match for match in mixed_matches
                if match[0] == "do" or match[3]
            ]
        for ending, hangul, prefix, known_english, case_cued in mixed_matches:
            text = prefix + hangul
            if known_english:
                score = 8.5 + min(len(prefix), 40) * .000001
            elif case_cued:
                # Prefer the established longest-particle parse. The special
                # 도 gate above has already protected an unknown name's final
                # e from being consumed by 에도.
                score = 8. + min(len(ending), 40) * .001 + min(len(prefix), 40) * .000001
            else:
                score = 1.5
            keep_strongest(found, WordOption(text, score, "mixed"))
        # Productive compound nouns and endings are searched on romanized
        # boundaries only; cross-boundary phonology is indexed above.
        for ending, hangul, restore in self.korean_boundaries:
            if token.endswith(ending) and len(token) > len(ending):
                prefix = token[:-len(ending)] + restore
                prefixes = {prefix}
                # Undo voicing and liaison, including every Korean complex
                # coda. These are pronunciation classes, not word exceptions;
                # the full candidate still has to pass forward romanization.
                boundary_reversals = (
                    ("ks", "k"), ("nj", "n"), ("lg", "k"), ("lm", "m"),
                    ("lb", "p"), ("ls", "l"), ("lt", "l"), ("lp", "p"),
                    ("ps", "p"), ("ss", "t"), ("ch", "t"),
                    ("g", "k"), ("d", "t"), ("b", "p"), ("r", "l"),
                    ("s", "t"), ("j", "t"), ("h", "t"),
                )
                for surface, isolated in boundary_reversals:
                    if prefix.endswith(surface):
                        prefixes.add(prefix[:-len(surface)] + isolated)
                for key in sorted(prefixes):
                    for stem in self.stem_options(key):
                        if HANGUL.fullmatch(stem.text) and suffix_allowed(stem.text, hangul):
                            text = stem.text + hangul
                            if romanize_word(text) == token:
                                keep_strongest(found, WordOption(text, stem.score - .2, "morphology"))
        for item in syllables(token):
            found.setdefault(item.text, WordOption(item.text, 1 + item.score * .1, "syllable"))
        values = sorted(found.values(), key=lambda x: (-x.score, x.text))[:16]
        if not any(x.text == original for x in values):
            values[-1] = found[original]
        return tuple(values)
