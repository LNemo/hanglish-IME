"""Morphological lattice with optional learned context and phonetic models."""

from functools import lru_cache
import hashlib
from importlib.util import find_spec
import math
from pathlib import Path
import re
import threading
import time

from hanglish.core.lattice import CandidateGenerator, WordOption, suffix_allowed
from hanglish.core.lexical_data import read_counter_units
from hanglish.core.phonology import romanize_word


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "checkpoints" / "context_ranker.pt"
TRANSDUCER = ROOT / "checkpoints" / "phonetic_transducer.pt"
SEMANTIC_CHECKPOINT = ROOT / "checkpoints" / "semantic_ranker"
SENTENCE_CHECKPOINT = ROOT / "checkpoints" / "sentence_ranker"
TOKENS = re.compile(r"`[^`\n]+`[A-Za-z]*[.,!?]?|\s+|[^\s]+")
LETTERS = re.compile(r"[A-Za-z]+|[^A-Za-z]+")
PROTECTED = re.compile(r"`[^`\n]+`[A-Za-z]*[.,!?]?|(?:https?://|www\.)\S+|\S+@\S+|\S*[_/\\\d]\S*", re.I)
QUOTED_SUFFIX = re.compile(r"(`[^`\n]+`)([A-Za-z]+)([.,!?]?)\Z")
NUMERIC_SUFFIX = re.compile(r"(\d+(?:\.\d+)?)([a-z]{1,40})([.,!?]?)\Z")
ACRONYM_SUFFIX = re.compile(r"([A-Z]{2,})([a-z]{2,})\Z")
HANGUL_SUFFIX = re.compile(r"[가-힣]+\Z")
# This is deliberately a language grammar rather than a general digit-token
# escape hatch.  It admits one integer/decimal followed by a Korean counter,
# time, currency, or quantity unit, with a bounded magnitude/approximation and
# one particle.  Repeated number-unit spans and ASCII measurement/code units
# remain protected as a single raw chunk.
NUMERIC_MAGNITUDES = ("십", "백", "천", "만", "억", "조")
NUMERIC_APPROXIMATIONS = ("여",)
NUMERIC_UNITS = (
    "퍼센트", "킬로미터", "센티미터", "밀리미터", "킬로그램", "밀리그램",
    "리터", "밀리리터", "세기", "개월", "시간", "마리",
    "년", "월", "일", "시", "분", "초", "주", "달", "원", "명", "개",
    "번", "회", "차", "건", "점", "살", "층", "호", "권", "대", "잔",
    "장", "곳", "배", "도",
)
NUMERIC_PARTICLES = (
    "으로부터", "에서부터", "에게서", "까지", "부터", "보다", "처럼", "만큼",
    "으로", "에게", "에서", "마다", "쯤", "은", "는", "이", "가", "을", "를",
    "과", "와", "에", "로", "의", "도", "만",
)
ASCII_NUMERIC_UNITS = frozenset({
    "a", "b", "c", "d", "db", "dpi", "fps", "g", "gb", "ghz", "hz", "kb",
    "kg", "khz", "km", "kw", "l", "m", "mah", "mb", "mg", "mhz", "ml",
    "mm", "ms", "px", "rpm", "tb", "v", "w", "wh",
})
BUNDLE_FILES = {"model.safetensors", "config.json", "tokenizer.json", "vocab.txt",
                "tokenizer_config.json", "special_tokens_map.json", "metadata.json"}


def numeric_unit_surface(text, units=NUMERIC_UNITS):
    """Whether Hangul *text* is a supported quantity suffix.

    Accepted form: one optional magnitude syllable, one optional approximation,
    one supported unit, and one optional particle.  The candidate still has to
    be produced by the existing generator and round-trip through forward RR.
    """
    cursor = 0
    for magnitude in NUMERIC_MAGNITUDES:
        if text.startswith(magnitude):
            cursor = len(magnitude)
            break
    for approximation in NUMERIC_APPROXIMATIONS:
        if text.startswith(approximation, cursor):
            cursor += len(approximation)
            break
    for unit in units:
        if not text.startswith(unit, cursor):
            continue
        tail = text[cursor + len(unit):]
        return not tail or (tail in NUMERIC_PARTICLES and suffix_allowed(unit, tail))
    return False


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_ranker_kind():
    if SEMANTIC_CHECKPOINT.exists():
        return "hybrid" if SENTENCE_CHECKPOINT.exists() else "semantic"
    return "mlp"


def neural_readiness():
    from hanglish.core import predicate_boundaries
    try:
        if not predicate_boundaries.INVENTORY.is_file():
            raise FileNotFoundError('predicate boundary inventory is missing')
        predicate_boundaries._inventory()
    except (OSError, ValueError, KeyError, TypeError):
        return False, "용언 경계 사전 자료 확인 필요"
    if find_spec("torch") is None:
        return False, "학습 엔진 의존성 설치 필요"
    ranker_path = SEMANTIC_CHECKPOINT / "metadata.json" if SEMANTIC_CHECKPOINT.exists() else CHECKPOINT
    if ranker_path.parent == SEMANTIC_CHECKPOINT and find_spec("transformers") is None:
        return False, "의미 모델 의존성 설치 필요"
    required = [ranker_path, TRANSDUCER, TRANSDUCER.with_suffix(".json")]
    if default_ranker_kind() == "hybrid":
        required.append(SENTENCE_CHECKPOINT / "metadata.json")
    if ranker_path == CHECKPOINT:
        required.append(CHECKPOINT.with_suffix(".json"))
    if not all(path.is_file() for path in required):
        return False, "문맥·음운 모델 체크포인트 필요"
    return True, "로컬 학습 모델"


def artifact_version(external_fingerprint="absent"):
    digest = hashlib.sha256()
    digest.update(external_fingerprint.encode())
    paths = [Path(__file__), ROOT / "hanglish/core/lattice.py", ROOT / "hanglish/core/phonology.py",
             ROOT / "hanglish/core/predicate_boundaries.py",
             ROOT / "hanglish/core/productive_endings.py",
             ROOT / "hanglish/core/data/predicate_boundary_inventory.json",
             ROOT / "hanglish/core/lexical_data.py", ROOT / "hanglish/models/semantic_ranker.py",
             ROOT / "hanglish/models/context_ranker.py", ROOT / "hanglish/models/phonetic_transducer.py",
             ROOT / "hanglish/models/sentence_ranker.py", ROOT / "hanglish/core/numeric_structure.py",
             ROOT / "data/morphology_lexicon.json", ROOT / "data/workbench_lexicon.json",
             ROOT / "data/common_english.txt", ROOT / "data/it_terms.txt",
             ROOT / "data/challenge/train_lexicon.json", CHECKPOINT, TRANSDUCER,
             ROOT / "data/external/mecab-ko-dic/NNBC.csv",
             CHECKPOINT.with_suffix(".json"), TRANSDUCER.with_suffix(".json")]
    for directory in (SEMANTIC_CHECKPOINT, SENTENCE_CHECKPOINT):
        for path in sorted(directory.glob("*")):
            if path.name in BUNDLE_FILES:
                paths.append(path)
            else:
                # Invalid bundle entries are rejected by the safe loader.
                # Record their names without reading arbitrary-sized contents.
                digest.update(("unexpected:" + str(path.relative_to(ROOT))).encode())
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        if path.exists():
            digest.update(file_sha256(path).encode())
    return "neural-1-" + digest.hexdigest()[:12]


def beam_paths(pieces):
    paths = [("", 0., [])]
    for source, offset, options in pieces:
        expanded = []
        for prefix, score, segments in paths:
            for option in options[:8]:
                segment = {"input_start": offset, "input_end": offset + len(source),
                           "source": source, "text": option.text, "kind": option.kind}
                expanded.append((prefix + option.text, score + option.score, segments + [segment]))
        unique = {}
        for item in sorted(expanded, key=lambda item: (-item[1], item[0])):
            unique.setdefault(item[0], item)
        paths = list(unique.values())[:24]
    return paths


class NeuralConverter:
    def __init__(self, *, context_ranking=True, external_lexicon=True, ranker_kind="auto"):
        if ranker_kind not in ("auto", "mlp", "semantic", "sentence", "hybrid"):
            raise ValueError("지원하지 않는 문맥 모델입니다.")
        from hanglish.core import predicate_boundaries
        if not predicate_boundaries.INVENTORY.is_file():
            raise FileNotFoundError('predicate boundary inventory is missing')
        predicate_boundaries._inventory()
        self.generator = CandidateGenerator(external_lexicon=external_lexicon)
        counter_units = read_counter_units(ROOT / "data/external/mecab-ko-dic") if external_lexicon else ()
        self.numeric_units = tuple(sorted(set(NUMERIC_UNITS) | set(counter_units), key=lambda unit: (-len(unit), unit)))
        self.numeric_structure = None
        self.context_ranking = context_ranking
        self.ranker_kind = ranker_kind
        self.ranker = None
        self.sentence_ranker = None
        self.transducer = None
        self.models_loaded = False
        self.version = artifact_version(self.generator.external_fingerprint)
        self.lock = threading.RLock()
        self.word_options = lru_cache(maxsize=1024)(self._word_options)

    def load_models(self):
        with self.lock:
            if self.models_loaded:
                return
            import torch
            torch.set_num_threads(2)
            if self.ranker_kind == "auto":
                self.ranker_kind = default_ranker_kind()
            if self.ranker is None:
                semantic = self.ranker_kind in ("semantic", "sentence", "hybrid")
                if semantic:
                    from hanglish.models.semantic_ranker import SemanticRanker
                    self.ranker = SemanticRanker.load(SEMANTIC_CHECKPOINT)
                    if self.ranker_kind not in ("sentence", "hybrid"):
                        self.ranker_kind = "semantic"
                else:
                    if not CHECKPOINT.exists():
                        raise ValueError("문맥 모델 학습이 아직 완료되지 않았습니다.")
                    from hanglish.models.context_ranker import ContextRanker
                    self.ranker = ContextRanker.load(CHECKPOINT)
                    self.ranker_kind = "mlp"
            if self.ranker_kind in ("sentence", "hybrid") and self.sentence_ranker is None:
                from hanglish.models.sentence_ranker import SentenceRanker
                self.sentence_ranker = SentenceRanker.load(SENTENCE_CHECKPOINT)
            if self.transducer is None:
                if not TRANSDUCER.is_file():
                    raise ValueError("음운 모델 학습이 아직 완료되지 않았습니다.")
                from hanglish.models.phonetic_transducer import PhoneticTransducer
                self.transducer = PhoneticTransducer.load(TRANSDUCER)
            self.version = artifact_version(self.generator.external_fingerprint)
            ranker_files = (sorted(path for path in SEMANTIC_CHECKPOINT.glob("*") if path.is_file())
                            if self.ranker_kind in ("semantic", "sentence", "hybrid") else [CHECKPOINT, CHECKPOINT.with_suffix(".json")])
            if self.ranker_kind in ("sentence", "hybrid"):
                ranker_files.extend(sorted(path for path in SENTENCE_CHECKPOINT.glob("*") if path.is_file()))
            self.model_artifacts = {str(path.relative_to(ROOT)): file_sha256(path)
                                    for path in [*ranker_files, TRANSDUCER, TRANSDUCER.with_suffix(".json")]}
            self.models_loaded = True

    def _word_options(self, source, learned):
        options = list(self.generator.options(source))
        generated = None
        if learned and self.transducer is not None and source.islower() and 1 < len(source) <= 40:
            known = any(o.score >= 5.5 and o.kind != "raw" for o in options)
            if not known:
                with self.lock:
                    generated = self.transducer.generate(source, beam_size=8)
                # A generated spelling can already exist in the mechanical
                # syllable beam. Its learned likelihood must still replace
                # that fallback score, rather than being silently discarded.
                generated_options = {}
                for text, logprob in generated:
                    generated_options[text] = WordOption(text, 4. + logprob / len(text), "neural")
                if generated_options:
                    # Productive English+particle analyses are independent
                    # evidence and must survive a Korean neural hypothesis.
                    preserved = [o for o in options if o.kind != "syllable"]
                    preserved_texts = {o.text for o in preserved}
                    options = [o for o in generated_options.values() if o.text not in preserved_texts] + preserved
        # A language score cannot turn an arbitrary phonetic spelling into a
        # credible lexical analysis. Rank lexical/morphological analyses first;
        # use free spelling candidates only when no such analysis exists.
        credible = [option for option in options if option.score >= 5.5]
        if credible:
            original = next(option for option in options if option.text == source)
            options = credible
            if not any(option.text == source for option in options):
                options.append(original)
        supplement = (
            learned
            and self.transducer is not None
            and source.isascii()
            and source.isalpha()
            and source.islower()
            and 2 <= len(source) <= 40
            and any(option.score >= 5.5 and option.kind == "morphology" for option in options)
            and not any(option.score >= 5.5 and option.kind == "lexicon" for option in options)
            and not any(option.score >= 5.5 and option.kind in ("raw", "mixed") for option in options)
        )
        if supplement:
            if generated is None:
                with self.lock:
                    generated = self.transducer.generate(source, beam_size=8)
            positions = {option.text: index for index, option in enumerate(options)}
            for text, logprob in generated[:2]:
                candidate = WordOption(text, 4. + logprob / len(text), "neural")
                position = positions.get(text)
                if position is None:
                    positions[text] = len(options)
                    options.append(candidate)
                elif candidate.score > options[position].score:
                    options[position] = candidate
        # Unknown-word neural output can miss a spelling already present in
        # the mechanical beam. Preserve a few forward-valid alternatives for
        # the context ranker while leaving strong lexical/English paths alone.
        if (learned and any(option.kind == "neural" for option in options)
                and not any(option.score >= 5.5 for option in options)):
            seen = {option.text for option in options}
            added = 0
            for option in self.generator.options(source):
                if (option.kind != "syllable" or option.text in seen
                        or romanize_word(option.text) != source):
                    continue
                options.append(option)
                seen.add(option.text)
                added += 1
                if added == 4:
                    break
        # Preserve the existing option prefix; constrained search only proposes
        # a few additional spellings for weak, unprotected lowercase words.
        if (learned and self.transducer is not None and source.isascii()
                and source.isalpha() and source.islower() and 2 <= len(source) <= 40
                and not any(option.score >= 5.5 for option in options)):
            with self.lock:
                constrained = self.transducer.generate_constrained(source, beam_size=8)
            seen = {option.text for option in options}
            added = 0
            for text, score in constrained:
                if text in seen:
                    continue
                if not text or romanize_word(text) != source:
                    raise ValueError("constrained generator emitted an invalid spelling")
                options.append(WordOption(text, 4. + score / len(text), "neural"))
                seen.add(text)
                added += 1
                if added == 4:
                    break
        # Keep the acronym verbatim and offer the learned Korean suffix beside
        # every existing interpretation. The lowercase recursive call cannot
        # match ACRONYM_SUFFIX, so this adds only one level of lookup.
        acronym = ACRONYM_SUFFIX.fullmatch(source)
        if learned and acronym is not None and len(source) <= 40:
            prefix, suffix = acronym.groups()
            seen = {option.text for option in options}
            added = 0
            for option in self._word_options(suffix, learned):
                if not HANGUL_SUFFIX.fullmatch(option.text):
                    continue
                text = prefix + option.text
                if text in seen or romanize_word(text) != source:
                    continue
                if not math.isfinite(option.score):
                    raise ValueError("acronym suffix candidate has a non-finite score")
                options.append(WordOption(text, option.score, "mixed"))
                seen.add(text)
                added += 1
                if added == 4:
                    break
        return tuple(options)

    def _numeric_unit_pieces(self, chunk, offset, learned):
        """Split one conservative Arabic-number + Korean-unit token."""
        match = NUMERIC_SUFFIX.fullmatch(chunk)
        if not match:
            return None
        number, suffix, punctuation = match.groups()
        if suffix in ASCII_NUMERIC_UNITS:
            return None
        base_options = self.generator.options(suffix)
        if any(option.text == suffix and option.score >= 6 for option in base_options):
            return None
        accepted = [
            option for option in self.word_options(suffix, learned)
            if option.text != suffix
            and numeric_unit_surface(option.text, getattr(self, "numeric_units", NUMERIC_UNITS))
            and romanize_word(option.text) == suffix
        ]
        if not accepted:
            return None
        raw_option = next(
            (option for option in base_options if option.text == suffix),
            WordOption(suffix, 0., "raw"),
        )
        accepted.append(raw_option)
        pieces = [
            (number, offset, (WordOption(number, 0., "raw"),)),
            (suffix, offset + len(number), tuple(accepted)),
        ]
        if punctuation:
            pieces.append((
                punctuation,
                offset + len(number) + len(suffix),
                (WordOption(punctuation, 0., "raw"),),
            ))
        return pieces

    def _prepare_candidates(self, text, context, *, learned, raw):
        """Build static and word-ranked paths before optional sentence scoring."""
        pieces = []
        warnings = []
        context_used = False
        ranking_applied = False
        fragmented = len(LETTERS.findall(text)) > 128
        if raw or fragmented:
            pieces = [(text, 0, (WordOption(text, 0., "raw"),))]
            if fragmented:
                warnings.append("128개를 넘는 문자 구간은 전체 원문으로 보존했습니다.")
        else:
            for match in TOKENS.finditer(text):
                chunk = match.group()
                quoted = QUOTED_SUFFIX.fullmatch(chunk)
                suffix_map = dict(self.generator.english_suffixes)
                if quoted and quoted[2].lower() in suffix_map:
                    literal, suffix, punctuation = quoted.groups()
                    pieces.append((literal, match.start(), (WordOption(literal, 0., "raw"),)))
                    pieces.append((suffix, match.start() + len(literal),
                                   (WordOption(suffix_map[suffix.lower()], 0., "mixed"),)))
                    if punctuation:
                        pieces.append((punctuation, match.end() - len(punctuation),
                                       (WordOption(punctuation, 0., "raw"),)))
                    continue
                numeric_pieces = self._numeric_unit_pieces(chunk, match.start(), learned)
                if numeric_pieces is not None:
                    pieces.extend(numeric_pieces)
                    continue
                if chunk.isspace() or PROTECTED.fullmatch(chunk):
                    pieces.append((chunk, match.start(), (WordOption(chunk, 0., "raw"),)))
                    continue
                for part in LETTERS.finditer(chunk):
                    source = part.group()
                    if source.isascii() and source.isalpha():
                        options = self.word_options(source, learned)
                        if len(source) > 40:
                            warnings.append("40자를 넘는 영문 토큰은 원문으로 보존했습니다.")
                    else:
                        options = (WordOption(source, 0., "raw"),)
                    pieces.append((source, match.start() + part.start(), options))
        preliminary = [options[0].text for _, _, options in pieces]
        static_pieces = pieces
        if learned and not raw and self.context_ranking:
            contexts, candidates, positions = [], [], []
            for i, (_, _, options) in enumerate(pieces):
                if len(options) <= 1:
                    continue
                lexical = any(o.score >= 5.5 and o.kind != "raw" for o in options)
                generated = self.ranker_kind in ("semantic", "sentence", "hybrid") and any(
                    o.kind == "neural" for o in options)
                if not lexical and not generated:
                    continue
                surrounding = context + " " + "".join(preliminary[:i]) + " [MASK] " + "".join(preliminary[i + 1:])
                for j, option in enumerate(options):
                    if option.kind == "raw" and option.score < 6:
                        continue
                    contexts.append(surrounding)
                    candidates.append(option.text)
                    positions.append((i, j))
            if candidates:
                context_used = bool(context.strip())
                ranking_applied = True
                with self.lock:
                    neural_scores = self.ranker.score_pairs(contexts, candidates)
                adjusted = [list(options) for _, _, options in pieces]
                for options in adjusted:
                    if len(options) > 1:
                        for j, option in enumerate(options):
                            if option.kind == "raw" and option.score < 6:
                                options[j] = WordOption(option.text, -10000., "raw")
                for (i, j), score in zip(positions, neural_scores):
                    option = adjusted[i][j]
                    adjusted[i][j] = WordOption(
                        option.text, float(score) + .15 * option.score, option.kind)
                pieces = [(source, offset, tuple(sorted(options, key=lambda o: (-o.score, o.text))))
                          for (source, offset, _), options in zip(pieces, adjusted)]
        return pieces, static_pieces, warnings, context_used, ranking_applied

    def sentence_proposals(self, text, context=""):
        """Return the exact bounded runtime pool before sentence-model scoring."""
        self.load_models()
        if self.ranker_kind != "semantic":
            raise ValueError("sentence proposals require an explicit semantic word ranker")
        pieces, static_pieces, warnings, context_used, ranking_applied = self._prepare_candidates(
            text, context, learned=True, raw=False)
        proposals = {path[0]: path for path in beam_paths(pieces)}
        for path in beam_paths(static_pieces):
            proposals.setdefault(path[0], path)
        return {
            "paths": list(proposals.values())[:48], "warnings": warnings,
            "context_used": context_used, "ranking_applied": ranking_applied,
        }

    def _numeric_sentence_extras(self, pieces, static_pieces, complete):
        if not any(any(character.isdigit() for character in source)
                   for source, _, _ in pieces):
            return []
        # Import after this module has initialized its shared numeric constants.
        from hanglish.core.numeric_structure import NumericSpanGenerator, sentence_extras
        with self.lock:
            if self.numeric_structure is None:
                self.numeric_structure = NumericSpanGenerator(self.numeric_units)
        return sentence_extras(pieces, static_pieces, complete, self.numeric_structure)

    def _extend_sentence_paths(self, text, context, paths):
        """Optional source-only proposals after the existing sentence scoring."""
        return paths

    def convert(self, text, context="", engine="neural", mode="batch", raw=False):
        start = time.perf_counter()
        learned = engine == "neural"
        if learned and not raw:
            self.load_models()
        sentence_ranking_applied = False
        pieces, static_pieces, warnings, context_used, ranking_applied = self._prepare_candidates(
            text, context, learned=learned, raw=raw)
        paths = beam_paths(pieces)
        sentence_ranker = getattr(self, "sentence_ranker", None)
        use_sentence = self.ranker_kind != "hybrid" or len(text.split()) > 1
        if learned and not raw and self.context_ranking and sentence_ranker is not None and len(paths) > 1 and use_sentence:
            # Union lexical and word-ranked complete paths so an overconfident
            # local decision cannot be the sole sentence-level proposal gate.
            proposals = {path[0]: path for path in paths}
            for path in beam_paths(static_pieces):
                proposals.setdefault(path[0], path)
            complete = list(proposals.values())
            if all(sentence_ranker.can_score(path[0]) for path in complete):
                extras = self._numeric_sentence_extras(pieces, static_pieces, complete)
                if extras and all(sentence_ranker.can_score(path[0]) for path in extras):
                    complete.extend(extras)
                with self.lock:
                    scores = sentence_ranker.score_pairs([context] * len(complete), [path[0] for path in complete])
                paths = sorted([(path[0], float(score), path[2]) for path, score in zip(complete, scores)],
                               key=lambda path: (-path[1], path[0]))
                sentence_ranking_applied = True
                ranking_applied = True
                context_used = bool(context.strip())
            else:
                warnings.append("문장 모델의 길이 범위를 넘어 단어별 문맥 순위를 사용했습니다.")
        if sentence_ranking_applied:
            paths = self._extend_sentence_paths(text, context, paths)
        candidates = [{"text": output, "score": round(score, 4), "segments": segments}
                      for output, score, segments in paths[:5]]
        if not any(candidate["text"] == text for candidate in candidates):
            original = {"text": text, "score": None, "segments": [
                {"input_start": 0, "input_end": len(text), "source": text, "text": text, "kind": "raw"}]}
            candidates = candidates[:4] + [original]
        return {"input": text, "context": context, "engine": engine, "mode": mode, "raw": raw,
                "version": self.version, "candidates": candidates, "result": candidates[0]["text"],
                "ranker": self.ranker_kind if learned and not raw and self.context_ranking else None,
                "model_artifacts": getattr(self, "model_artifacts", {}) if learned and not raw else {},
                "context_used": context_used,
                "external_context_used": context_used, "ranking_applied": ranking_applied,
                "sentence_ranking_applied": sentence_ranking_applied,
                "warnings": list(dict.fromkeys(warnings)),
                "engine_ms": round((time.perf_counter() - start) * 1000, 3)}
