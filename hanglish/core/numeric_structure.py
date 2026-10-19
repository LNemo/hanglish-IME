"""Bounded numeric structures supplementing preserved sentence candidates."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import itertools
import math
import re

from hanglish.core.lattice import suffix_allowed
from hanglish.core.neural_converter import (
    ASCII_NUMERIC_UNITS, NUMERIC_MAGNITUDES, NUMERIC_PARTICLES, NUMERIC_UNITS,
)
from hanglish.core.phonology import romanize_word

MAX_LENGTH = 96
MAX_ATOMS = 16
MAX_CONVERSIONS = 4
MAX_PATHS = 8
MAX_OPTIONS = 4
GRAMMAR_SCORE = 6.
HANGUL = re.compile(r'[가-힣]+\Z')
NUMBER = r'(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?'
NUMBER_FULL = re.compile(NUMBER + r'\Z')
ATOM = re.compile(NUMBER + r'|[a-z]+|°[CF]|[%％‰℃℉㎡㎾㏊~–—-]')
SYMBOL_UNITS = frozenset(('%', '％', '‰', '℃', '℉', '㎡', '㎾', '㏊', '°C', '°F'))
SYMBOL_READINGS = {'%': '퍼센트', '％': '퍼센트', '‰': '퍼밀',
                   '℃': '도', '℉': '도', '°C': '도', '°F': '도',
                   '㎡': '제곱미터', '㎾': '킬로와트', '㏊': '헥타르'}
RANGES = frozenset(('~', '-', '–', '—'))
WRAPPERS = {'(': ')', '[': ']', '{': '}', '“': '”', '‘': '’', '"': '"', "'": "'"}
# These are productive noun endings. Full forward RR includes liaison across
# the noun boundary; the source is never split at guessed roman syllables.
COPULAR_TAILS = ('이다', '입니다', '이었다', '였다', '이었어요', '였어요',
                 '이라는', '라는', '이란', '란', '이라고', '라고',
                 '이라면', '라면', '이고', '이며', '인', '일', '인데')
QUANTITATIVE_TAILS = ('가량', '정도', '당', '이상', '이하', '미만', '초과')
COPULAR_PAIRS = {'이라는': True, '라는': False, '이란': True, '란': False,
                 '이라고': True, '라고': False, '이라면': True, '라면': False}
POSTFIX_INDEX = defaultdict(list)
for _tail in (*NUMERIC_PARTICLES, *QUANTITATIVE_TAILS, *COPULAR_TAILS):
    POSTFIX_INDEX[romanize_word(_tail)].append(_tail)
ASCII_INVARIANT_TAILS = frozenset(QUANTITATIVE_TAILS + (
    '에게서', '까지', '부터', '보다', '처럼', '만큼', '에게', '에서', '마다',
    '쯤', '에', '의', '도', '만', '이다', '입니다', '이고', '이며', '인', '일', '인데',
))


def valid_tail(stem, tail):
    if tail in COPULAR_PAIRS:
        return bool((ord(stem[-1]) - 0xAC00) % 28) == COPULAR_PAIRS[tail]
    return suffix_allowed(stem, tail)


def segment(source, start, text, kind='raw'):
    return {'input_start': start, 'input_end': start + len(source),
            'source': source, 'text': text, 'kind': kind}


def validate_path(source, offset, path):
    text, score, segments = path
    cursor = offset
    if (type(score) not in (int, float) or not math.isfinite(score)
            or len(source) > MAX_LENGTH or not segments):
        raise ValueError('invalid numeric supplemental path')
    conversions = 0
    for item in segments:
        a, b = item['input_start'], item['input_end']
        if (a != cursor or b <= a or b > offset + len(source)
                or source[a-offset:b-offset] != item['source']):
            raise ValueError('numeric path does not cover contiguous source offsets')
        if item['kind'] == 'raw':
            if item['source'] != item['text']:
                raise ValueError('numeric path changed a protected literal')
        elif item['kind'] == 'numeric':
            conversions += 1
            if not HANGUL.fullmatch(item['text']) or romanize_word(item['text']) != item['source']:
                raise ValueError('numeric path is not forward valid')
        else:
            raise ValueError('numeric path has unknown segment kind')
        cursor = b
    if (cursor != offset + len(source) or ''.join(s['text'] for s in segments) != text
            or not 1 <= conversions <= MAX_CONVERSIONS
            or score != GRAMMAR_SCORE * conversions or romanize_word(text) != source):
        raise ValueError('numeric path output or conversion bounds differ')


class NumericSpanGenerator:
    """A forward index of licensed unit forms plus a bounded source lexer.

    The unit vocabulary comes from the existing counter lexicon, never labels.
    Initialization cost is separate from per-token parsing latency.
    """
    def __init__(self, units=NUMERIC_UNITS):
        self.units = tuple(sorted(set(u for u in units if HANGUL.fullmatch(u))))
        if len(self.units) > 1024:
            raise ValueError('numeric unit vocabulary exceeds its bound')
        self.index = defaultdict(list)
        tails = ('', *NUMERIC_PARTICLES, *QUANTITATIVE_TAILS, *COPULAR_TAILS)
        stems = {magnitude + approx + unit for magnitude, approx, unit in itertools.product(
            ('', *NUMERIC_MAGNITUDES), ('', '여'), self.units)}
        stems.update(NUMERIC_MAGNITUDES)
        for stem in sorted(stems):
            for tail in tails:
                if tail and not valid_tail(stem, tail):
                    continue
                target = stem + tail
                source = romanize_word(target)
                if 1 <= len(source) <= 40:
                    self.index[source].append(target)
        for source, values in self.index.items():
            self.index[source] = tuple(sorted(set(values), key=lambda x: (len(x), x))[:MAX_OPTIONS])
        self.paths = lru_cache(maxsize=1024)(self._paths)

    def _latin_options(self, source, offset, after_symbol=None):
        values = []
        if after_symbol:
            targets = POSTFIX_INDEX.get(source, ())
            values += [[segment(source, offset, target, 'numeric')] for target in targets
                       if valid_tail(SYMBOL_READINGS[after_symbol], target)]
        elif source not in ASCII_NUMERIC_UNITS:
            values += [[segment(source, offset, target, 'numeric')]
                       for target in self.index.get(source, ())]
        # A copied ASCII unit can take a separate Korean postfix. Bare ASCII
        # units never gain an interpretation. Ambiguous splits remain bounded.
        if not after_symbol:
            for unit in sorted(ASCII_NUMERIC_UNITS, key=lambda x: (-len(x), x)):
                if source.startswith(unit) and len(source) > len(unit):
                    suffix = source[len(unit):]
                    for target in POSTFIX_INDEX.get(suffix, ()):
                        # An ASCII abbreviation has multiple possible spoken
                        # readings. Only reading-independent tails are licensed
                        # here; particle agreement needs a later reading model.
                        if target not in ASCII_INVARIANT_TAILS:
                            continue
                        values.append([segment(unit, offset, unit),
                                       segment(suffix, offset + len(unit), target, 'numeric')])
        unique = {}
        for value in values:
            unique.setdefault(''.join(s['text'] for s in value), value)
        return list(unique.values())[:MAX_OPTIONS]

    def _paths(self, chunk, offset=0):
        if (not isinstance(offset, int) or offset < 0 or len(chunk) > MAX_LENGTH
                or not re.search(r'[0-9]', chunk) or re.search(r'[`@_/\\:]', chunk)):
            return ()
        # Scientific, hex, address and version syntax must not become counters.
        if (re.search(r'[0-9]+(?:\.[0-9]+){2,}', chunk)
                or re.search(r'0[xX][0-9a-fA-F]+', chunk)
                or re.search(r'[0-9]+(?:\.[0-9]+)?[eE][+-]?[0-9]+', chunk)):
            return ()
        left, right = 0, len(chunk)
        while right > left and chunk[right-1] in '.,!?':
            right -= 1
        wrappers = 0
        while left < right and chunk[left] in WRAPPERS:
            if chunk[right-1] != WRAPPERS[chunk[left]]:
                return ()
            left += 1
            right -= 1
            wrappers += 1
            if wrappers > 2:
                return ()
        body = chunk[left:right]
        matches = list(ATOM.finditer(body))
        if (not matches or len(matches) + 2 * wrappers > MAX_ATOMS
                or ''.join(m.group() for m in matches) != body
                or not NUMBER_FULL.fullmatch(matches[0].group())):
            return ()
        atoms = [(m.group(), offset + left + m.start()) for m in matches]
        pieces = []
        if left:
            pieces.append([[segment(chunk[:left], offset, chunk[:left])]])
        cursor = 0
        conversions = 0
        while cursor < len(atoms):
            number, start = atoms[cursor]
            if not NUMBER_FULL.fullmatch(number):
                return ()
            pieces.append([[segment(number, start, number)]])
            cursor += 1
            after_symbol = None
            if cursor < len(atoms) and atoms[cursor][0] in SYMBOL_UNITS:
                symbol, start = atoms[cursor]
                pieces.append([[segment(symbol, start, symbol)]])
                cursor += 1
                after_symbol = symbol
            if cursor < len(atoms) and re.fullmatch('[a-z]+', atoms[cursor][0]):
                latin, start = atoms[cursor]
                options = self._latin_options(latin, start, after_symbol)
                if not after_symbol and latin in ASCII_NUMERIC_UNITS:
                    options = [[segment(latin, start, latin)]]
                if not options:
                    return ()
                pieces.append(options)
                conversions += int(any(s['kind'] == 'numeric' for s in options[0]))
                cursor += 1
            if cursor < len(atoms) and atoms[cursor][0] in RANGES:
                marker, start = atoms[cursor]
                pieces.append([[segment(marker, start, marker)]])
                cursor += 1
                if cursor == len(atoms):
                    return ()
            if conversions > MAX_CONVERSIONS:
                return ()
        if not conversions:
            return ()
        if right < len(chunk):
            pieces.append([[segment(chunk[right:], offset + right, chunk[right:])]])
        paths = [[]]
        for options in pieces:
            paths = [prefix + suffix for prefix in paths for suffix in options][:MAX_PATHS]
        result = []
        for spans in paths:
            text = ''.join(s['text'] for s in spans)
            path = (text, GRAMMAR_SCORE * conversions, spans)
            validate_path(chunk, offset, path)
            result.append(path)
        return tuple(result)


def validate_complete_segments(pieces, path):
    text, score, segments = path
    if not math.isfinite(score) or not segments:
        raise ValueError("numeric augmented path shape differs")
    source_text = "".join(source for source, _, _ in pieces)
    cursor = 0
    for source, offset, _ in pieces:
        if offset != cursor:
            raise ValueError("runtime pieces do not cover contiguous input")
        cursor += len(source)
    cursor = 0
    for segment in segments:
        if (segment["input_start"] != cursor or segment["input_end"] <= cursor
                or segment["source"] != source_text[
                    segment["input_start"]:segment["input_end"]]):
            raise ValueError("numeric augmented segments do not preserve source offsets")
        if segment["kind"] == "raw" and segment["text"] != segment["source"]:
            raise ValueError("numeric augmented path changed a raw literal")
        if (segment["kind"] == "numeric"
                and romanize_word(segment["text"]) != segment["source"]):
            raise ValueError("numeric augmented segment is not forward valid")
        cursor = segment["input_end"]
    if cursor != len(source_text) or text != "".join(
            segment["text"] for segment in segments):
        raise ValueError("numeric augmented path output differs from its segments")


def sentence_extras(pieces, static_pieces, original_pool, generator):
    """Apply the first numeric interpretation to every preserved parent path."""
    if len(pieces) != len(static_pieces) or len(original_pool) > 48:
        raise ValueError("numeric sentence input exceeds the accepted shape")
    replacements = {}
    for index, (learned, static) in enumerate(zip(pieces, static_pieces)):
        source, offset, options = learned
        if static[:2] != learned[:2]:
            raise ValueError("numeric learned/static source alignment differs")
        combined = [*options, *static[2]]
        if (not any(character.isdigit() for character in source) or not combined
                or not all(option.kind == "raw" and option.text == source
                           for option in combined)):
            continue
        candidates = generator.paths(source, offset)
        if candidates:
            replacements[index] = candidates[0]
    if not replacements:
        return []
    seen = {path[0] for path in original_pool}
    extras = []
    for _, score, segments in original_pool:
        if len(segments) != len(pieces):
            raise ValueError("numeric sentence path no longer matches its pieces")
        output_segments = []
        for index, segment in enumerate(segments):
            replacement = replacements.get(index)
            if replacement is None:
                output_segments.append(dict(segment))
            else:
                if (segment["source"] != pieces[index][0]
                        or segment["text"] != segment["source"]
                        or segment["kind"] != "raw"):
                    raise ValueError("numeric supplement changed a non-raw segment")
                output_segments.extend(dict(item) for item in replacement[2])
        text = "".join(segment["text"] for segment in output_segments)
        path = (text, score, output_segments)
        validate_complete_segments(pieces, path)
        if text not in seen:
            seen.add(text)
            extras.append(path)
        if len(extras) == 48:
            break
    return extras
