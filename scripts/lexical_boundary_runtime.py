"""Isolated sentence-pool overlay for lexical boundary compositions.

The production converter is left untouched.  This subclass preserves the
complete production sentence pool, then offers a bounded set of additional
paths built from source-derived :class:`BoundaryComposer` word options to the
already loaded word and sentence rankers.
"""
from __future__ import annotations

from functools import lru_cache
import math

from hanglish.core.lattice import WordOption
from hanglish.core.neural_converter import NeuralConverter, beam_paths
from hanglish.core.phonology import romanize_word
from scripts.lexical_boundary_composition import BoundaryComposer

MAX_WORD_OPTIONS = 16
MAX_NEW_PATHS = 48


def _path_record(path):
    text, score, segments = path
    return {
        "text": text,
        "score": score,
        "segments": [dict(segment) for segment in segments],
    }


def _stable_paths(*groups):
    found = {}
    for group in groups:
        for path in group:
            found.setdefault(path[0], path)
    return tuple(found.values())


def _finite_score(value, message):
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(message)
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if not math.isfinite(score):
        raise ValueError(message)
    return score


class _CheckedRanker:
    """Validate model output without changing any accepted score value."""

    def __init__(self, delegate, name):
        self._delegate = delegate
        self._name = name

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def score_pairs(self, contexts, candidates):
        contexts = list(contexts)
        candidates = list(candidates)
        scores = list(self._delegate.score_pairs(contexts, candidates))
        if len(scores) != len(candidates):
            raise ValueError(f"{self._name} returned a malformed score vector")
        for score in scores:
            _finite_score(score, f"{self._name} returned a non-finite score")
        return scores


def _merge_options(original, proposed):
    """Keep the strongest deterministic option per spelling, including raw."""
    strongest = {}
    for option in (*tuple(original), *tuple(proposed)):
        if not isinstance(option, WordOption):
            raise ValueError("boundary option must be a WordOption")
        _finite_score(option.score, "boundary option has a non-finite score")
        previous = strongest.get(option.text)
        if (previous is None or option.score > previous.score
                or (option.score == previous.score and option.kind < previous.kind)):
            strongest[option.text] = option
    ordered = sorted(strongest.values(), key=lambda option: (-option.score, option.text, option.kind))
    raw = next((option for option in ordered if option.kind == "raw"), None)
    kept = ordered[:MAX_WORD_OPTIONS]
    if raw is not None and raw not in kept:
        kept[-1:] = [raw]
        kept.sort(key=lambda option: (-option.score, option.text, option.kind))
    return tuple(kept)


class BoundaryCompositionRuntime(NeuralConverter):
    """Non-production overlay used by the fixed natural FIT512 evaluator."""

    boundary_overlay_enabled = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initialize_boundary_overlay()

    def initialize_boundary_overlay(self):
        """Initialize only overlay-owned state after cloning a loaded parent."""
        self._boundary_composer = BoundaryComposer(self.generator)
        self._boundary_base_word_options = self.word_options
        self._boundary_current_text = None
        self._boundary_current_context = None
        self._boundary_derived = []
        self._boundary_augmented_piece_count = 0
        self._boundary_options = lru_cache(maxsize=4096)(self._build_boundary_options)
        self._boundary_explanations = lru_cache(maxsize=4096)(self._explain_options)
        self.boundary_trace = self._empty_trace("not_run")
        return self

    @staticmethod
    def _empty_trace(reason):
        return {
            "activated": False,
            "reason": reason,
            "preserved_pool": [],
            "generated_new_extras": [],
            "accepted_new_extras": [],
            "augmented_piece_count": 0,
            "derived_options": [],
        }

    @staticmethod
    def _protected(options):
        return any(
            option.score >= 6 and option.kind in ("raw", "english", "mixed")
            for option in options
        )

    def _build_boundary_options(self, source, learned):
        original = tuple(self._boundary_base_word_options(source, learned))
        if (not learned or not isinstance(source, str) or not source.isascii()
                or not source.isalpha() or not source.islower()
                or self._protected(original)):
            return original
        proposed = tuple(self._boundary_composer.propose(source))
        if not proposed:
            return original
        return _merge_options(original, proposed)

    def _direct_derivations(self, roman, wanted):
        if not wanted:
            return []
        records = []
        keys = set()
        for split in range(1, len(roman)):
            keys.add((roman[:split], roman[split:], "unchanged", roman[split:split + 1]))
            for surface, left_tail, right_head in self._boundary_composer.rules_by_initial.get(roman[split], ()):
                if roman.startswith(surface, split):
                    keys.add((roman[:split] + left_tail,
                              right_head + roman[split + len(surface):],
                              "changed", surface))
        for left_key, right_key, boundary, surface in sorted(keys):
            for left in self._boundary_composer._lexemes(left_key):
                for right in self._boundary_composer._lexemes(right_key):
                    text = left.text + right.text
                    if text not in wanted or romanize_word(text) != roman:
                        continue
                    records.append({
                        "source": roman,
                        "text": text,
                        "left_lexeme": left.text,
                        "right_lexeme": right.text,
                        "left_key": left_key,
                        "right_key": right_key,
                        "boundary": boundary,
                        "surface": surface,
                        "suffix": None,
                    })
        return records

    def _explain_options(self, source, texts):
        wanted = set(texts)
        records = self._direct_derivations(source, wanted)
        explained = {record["text"] for record in records}
        for surface, suffix, restored in self.generator.korean_boundaries:
            if not source.endswith(surface) or len(source) <= len(surface):
                continue
            stem_key = source[:-len(surface)] + restored
            stem_wanted = {
                text[:-len(suffix)] for text in wanted
                if suffix and text.endswith(suffix)
            }
            for record in self._direct_derivations(stem_key, stem_wanted):
                text = record["text"] + suffix
                if text not in wanted or romanize_word(text) != source:
                    continue
                item = dict(record)
                item.update({
                    "source": source,
                    "text": text,
                    "suffix": suffix,
                    "suffix_surface": surface,
                    "suffix_restored": restored,
                })
                records.append(item)
                explained.add(text)
        # BoundaryComposer and this explainer share the same source rules.  A
        # missing explanation indicates drift and must not silently pass.
        if explained != wanted:
            raise ValueError("boundary derivation provenance is incomplete")
        unique = {}
        for record in sorted(records, key=lambda item: tuple(str(item.get(key, "")) for key in (
                "text", "left_key", "right_key", "left_lexeme", "right_lexeme", "suffix"))):
            key = tuple((name, record.get(name)) for name in sorted(record))
            unique.setdefault(key, record)
        return tuple(unique.values())

    def _overlay_word_options(self, source, learned):
        original = tuple(self._boundary_base_word_options(source, learned))
        merged = tuple(self._boundary_options(source, learned))
        additions = tuple(option for option in merged if option.text not in {old.text for old in original})
        if additions:
            self._boundary_augmented_piece_count += 1
            self._boundary_derived.extend(
                self._boundary_explanations(source, tuple(option.text for option in additions))
            )
        return merged

    def _numeric_sentence_extras(self, pieces, static_pieces, complete):
        """Return production numeric extras followed by bounded boundary paths."""
        if not getattr(self, "boundary_overlay_enabled", True):
            return super()._numeric_sentence_extras(
                pieces, static_pieces, complete
            )
        sentence_ranker = self.sentence_ranker
        generated_old = tuple(super()._numeric_sentence_extras(
            pieces, static_pieces, complete
        ))
        accepted_old = generated_old if all(
            sentence_ranker.can_score(path[0]) for path in generated_old
        ) else ()
        preserved = _stable_paths(complete, accepted_old)

        self._boundary_derived = []
        self._boundary_augmented_piece_count = 0
        original_word_options = self.word_options
        try:
            self.word_options = self._overlay_word_options
            augmented, augmented_static, *_ = NeuralConverter._prepare_candidates(
                self,
                self._boundary_current_text,
                self._boundary_current_context,
                learned=True,
                raw=False,
            )
        finally:
            self.word_options = original_word_options

        preserved_texts = {path[0] for path in preserved}
        generated = tuple(
            path for path in _stable_paths(
                beam_paths(augmented), beam_paths(augmented_static)
            )
            if path[0] not in preserved_texts
        )[:MAX_NEW_PATHS]
        accepted = tuple(
            path for path in generated if sentence_ranker.can_score(path[0])
        )
        self.boundary_trace = {
            "activated": bool(accepted),
            "reason": (
                "accepted_new_extras" if accepted else
                "unscoreable_new_extras" if generated else
                "no_boundary_candidates"
            ),
            "preserved_pool": [_path_record(path) for path in preserved],
            "generated_new_extras": [_path_record(path) for path in generated],
            "accepted_new_extras": [_path_record(path) for path in accepted],
            "augmented_piece_count": self._boundary_augmented_piece_count,
            "derived_options": [dict(record) for record in self._boundary_derived],
        }
        return [*accepted_old, *accepted]

    def convert(self, text, context="", engine="neural", mode="batch", raw=False):
        # The parent converter and the temporary word-option hook share state;
        # keep one conversion atomic and restore every production object.
        with self.lock:
            self.boundary_trace = self._empty_trace("production_branch_bypass")
            if not getattr(self, "boundary_overlay_enabled", True):
                return super().convert(text, context, engine, mode, raw)
            self._boundary_current_text = text
            self._boundary_current_context = context
            if engine == "neural" and not raw:
                # A fresh instance may install its model objects lazily.  Load
                # them before capturing identities so final restoration cannot
                # discard successfully loaded rankers while leaving
                # ``models_loaded`` true.
                self.load_models()
            original_ranker = getattr(self, "ranker", None)
            original_sentence_ranker = getattr(self, "sentence_ranker", None)
            if original_ranker is not None:
                self.ranker = _CheckedRanker(original_ranker, "word ranker")
            if original_sentence_ranker is not None:
                self.sentence_ranker = _CheckedRanker(
                    original_sentence_ranker, "sentence ranker"
                )
            try:
                return super().convert(text, context, engine, mode, raw)
            finally:
                self.ranker = original_ranker
                self.sentence_ranker = original_sentence_ranker
                self.word_options = self._boundary_base_word_options
                self._boundary_current_text = None
                self._boundary_current_context = None
