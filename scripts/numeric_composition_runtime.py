"""Isolated runtime for the approved numeric-composition experiment.

This module is not imported by the production converter.  It preserves the
production sentence pool and old numeric-extra acceptance rule, appends only
individually scoreable composition extras, and exposes source-only pool facts
for the later FIT evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

from hanglish.core.neural_converter import NeuralConverter, beam_paths
from hanglish.core.numeric_composition import composition_extras


@dataclass(frozen=True)
class RuntimePool:
    """One source-derived sentence pool and its bounded composition facts."""

    base: tuple
    old_extras: tuple
    generated_new_extras: tuple
    new_extras: tuple
    complete: tuple
    eligible_piece_count: int
    retained_combination_count: int
    composition_bypassed: bool
    base_scoreable: bool


def _stable_union(*groups):
    unique = {}
    for group in groups:
        for path in group:
            unique.setdefault(path[0], path)
    return tuple(unique.values())


def _path_record(path):
    text, score, segments = path
    return {
        "text": text,
        "score": score,
        "segments": [dict(segment) for segment in segments],
    }


def source_only_trace(pool, *, reason, activated, production_activated):
    """Serialize candidate construction without accepting labels or targets."""
    return {
        "reason": reason,
        "activated": activated,
        "production_activated": production_activated,
        "base_scoreable": pool.base_scoreable,
        "eligible_piece_count": pool.eligible_piece_count,
        "retained_combination_count": pool.retained_combination_count,
        "composition_bypassed": pool.composition_bypassed,
        "base": [_path_record(path) for path in pool.base],
        "accepted_old_extras": [_path_record(path) for path in pool.old_extras],
        "preserved_pool": [
            _path_record(path) for path in (*pool.base, *pool.old_extras)
        ],
        "production_scoring_pool": [
            _path_record(path) for path in (*pool.base, *pool.old_extras)
        ] if production_activated else [],
        "generated_new_extras": [
            _path_record(path) for path in pool.generated_new_extras
        ],
        "accepted_new_extras": [_path_record(path) for path in pool.new_extras],
        "scoring_pool": [_path_record(path) for path in pool.complete],
    }


def build_runtime_pool(converter, pieces, static_pieces, sentence_ranker):
    """Build the exact preserved-base + old-extra + new-extra experiment pool."""
    base = _stable_union(beam_paths(pieces), beam_paths(static_pieces))
    base_scoreable = all(sentence_ranker.can_score(path[0]) for path in base)
    if not base_scoreable:
        return RuntimePool(
            base, (), (), (), base, 0, 0, False, False
        )

    generated_old = tuple(converter._numeric_sentence_extras(
        pieces, static_pieces, list(base)
    ))
    # Preserve the production rule: old numeric extras are accepted together
    # only when every one of them is scoreable.
    old_extras = generated_old if all(
        sentence_ranker.can_score(path[0]) for path in generated_old
    ) else ()
    preserved = (*base, *old_extras)
    supplement = composition_extras(
        pieces,
        static_pieces,
        base,
        preserved,
        converter.generator,
        getattr(converter, "numeric_units", None),
    )
    generated_new = tuple(supplement.paths)
    # A single oversized composition must not discard accepted old extras.
    new_extras = tuple(
        path for path in generated_new if sentence_ranker.can_score(path[0])
    )
    return RuntimePool(
        base=base,
        old_extras=old_extras,
        generated_new_extras=generated_new,
        new_extras=new_extras,
        complete=(*preserved, *new_extras),
        eligible_piece_count=supplement.eligible_piece_count,
        retained_combination_count=supplement.retained_combination_count,
        composition_bypassed=supplement.bypassed,
        base_scoreable=True,
    )


class NumericCompositionRuntime(NeuralConverter):
    """Non-production converter overlay used by the bounded FIT experiment."""

    composition_enabled = True

    def convert(self, text, context="", engine="neural", mode="batch", raw=False):
        if not getattr(self, "composition_enabled", True):
            if hasattr(self, "numeric_composition_trace"):
                del self.numeric_composition_trace
            return super().convert(text, context, engine, mode, raw)

        start = time.perf_counter()
        learned = engine == "neural"
        if learned and not raw:
            self.load_models()
        sentence_ranking_applied = False
        pieces, static_pieces, warnings, context_used, ranking_applied = (
            self._prepare_candidates(
                text, context, learned=learned, raw=raw
            )
        )
        paths = beam_paths(pieces)
        sentence_ranker = getattr(self, "sentence_ranker", None)
        prerequisites = (
            learned
            and not raw
            and self.context_ranking
            and sentence_ranker is not None
        )
        use_sentence = self.ranker_kind != "hybrid" or len(text.split()) > 1
        conventional = prerequisites and len(paths) > 1 and use_sentence
        single_path_probe = prerequisites and len(paths) == 1
        pool = None

        if conventional or single_path_probe:
            pool = build_runtime_pool(
                self, pieces, static_pieces, sentence_ranker
            )
            has_choice = len(pool.complete) > len(pool.base)
            activate = pool.base_scoreable and (conventional or has_choice)
            # A hybrid single token is activated only by a real accepted
            # numeric choice. Ordinary one-token behavior remains unchanged.
            if activate:
                with self.lock:
                    scores = sentence_ranker.score_pairs(
                        [context] * len(pool.complete),
                        [path[0] for path in pool.complete],
                    )
                paths = sorted(
                    [
                        (path[0], float(score), path[2])
                        for path, score in zip(pool.complete, scores)
                    ],
                    key=lambda path: (-path[1], path[0]),
                )
                sentence_ranking_applied = True
                ranking_applied = True
                context_used = bool(context.strip())
                reason = "sentence_scored"
            elif not pool.base_scoreable and conventional:
                warnings.append(
                    "문장 모델의 길이 범위를 넘어 단어별 문맥 순위를 사용했습니다."
                )
                reason = "base_unscoreable"
            elif not pool.base_scoreable:
                reason = "base_unscoreable"
            else:
                reason = "no_accepted_choice"
            self.numeric_composition_trace = source_only_trace(
                pool,
                reason=reason,
                activated=activate,
                production_activated=conventional and pool.base_scoreable,
            )
        else:
            empty = RuntimePool(
                tuple(paths), (), (), (), tuple(paths), 0, 0, False, False
            )
            self.numeric_composition_trace = source_only_trace(
                empty,
                reason="production_branch_bypass",
                activated=False,
                production_activated=False,
            )

        candidates = [
            {"text": output, "score": round(score, 4), "segments": segments}
            for output, score, segments in paths[:5]
        ]
        if not any(candidate["text"] == text for candidate in candidates):
            original = {
                "text": text,
                "score": None,
                "segments": [{
                    "input_start": 0,
                    "input_end": len(text),
                    "source": text,
                    "text": text,
                    "kind": "raw",
                }],
            }
            candidates = candidates[:4] + [original]
        return {
            "input": text,
            "context": context,
            "engine": engine,
            "mode": mode,
            "raw": raw,
            "version": self.version,
            "candidates": candidates,
            "result": candidates[0]["text"],
            "ranker": (
                self.ranker_kind
                if learned and not raw and self.context_ranking
                else None
            ),
            "model_artifacts": (
                getattr(self, "model_artifacts", {}) if learned and not raw else {}
            ),
            "context_used": context_used,
            "external_context_used": context_used,
            "ranking_applied": ranking_applied,
            "sentence_ranking_applied": sentence_ranking_applied,
            "warnings": list(dict.fromkeys(warnings)),
            "engine_ms": round((time.perf_counter() - start) * 1000, 3),
        }
