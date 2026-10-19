"""Experimental word-proposal union before the existing bounded search."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from hanglish.core.neural_converter import NeuralConverter, beam_paths, file_sha256


def merge_options(baseline, supported):
    """Keep baseline scores, kinds and order; append only new spellings."""
    merged = list(baseline)
    seen = {option.text for option in merged}
    for option in supported:
        if option.text not in seen:
            merged.append(option)
            seen.add(option.text)
    return tuple(merged)


class ProposalUnionConverter(NeuralConverter):
    """Use a verified EnsembleConverter as a model bundle, not a decoder."""

    def __init__(self, models, *, experimental=False):
        if experimental is not True:
            raise ValueError("ProposalUnionConverter requires experimental=True")
        super().__init__(external_lexicon=False)
        self.models = models
        self.last_trace = None

    def load_models(self):
        with self.lock:
            if self.models_loaded:
                return
            self.models.load_models()
            baseline = self.models.baseline
            for name in ("generator", "numeric_units", "ranker", "sentence_ranker",
                         "ranker_kind", "context_ranking"):
                setattr(self, name, getattr(baseline, name))
            self.model_artifacts = dict(self.models.model_artifacts)
            source_sha = file_sha256(Path(__file__))
            self.model_artifacts["experimental/proposal_union_converter.py"] = source_sha
            identity = {"models_version": self.models.version, "source_sha256": source_sha}
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            self.version = "experimental-proposal-union-1-" + digest[:12]
            self.models_loaded = True

    def _word_options(self, source, learned):
        baseline = self.models.baseline.word_options(source, learned)
        if not learned:
            return baseline
        supported = self.models.supported.word_options(source, learned)
        return merge_options(baseline, supported)

    def _prepare_candidates(self, text, context, *, learned, raw):
        result = super()._prepare_candidates(text, context, learned=learned, raw=raw)
        pieces, static, _, _, _ = result
        self.last_trace = {
            "words": [{"source": source, "offset": offset,
                       "before_ranking": [option.text for option in original],
                       "after_ranking": [option.text for option in ranked]}
                      for (source, offset, original), (_, _, ranked) in zip(static, pieces)],
            "static_paths": [path[0] for path in beam_paths(static)],
            "word_ranked_paths": [path[0] for path in beam_paths(pieces)],
            "sentence_pool": [],
        }
        return result

    def _numeric_sentence_extras(self, pieces, static_pieces, complete):
        extras = super()._numeric_sentence_extras(pieces, static_pieces, complete)
        # Mirror the parent's admission condition without changing the pool.
        accepted = extras if all(self.sentence_ranker.can_score(p[0]) for p in extras) else []
        self.last_trace["sentence_pool"] = [path[0] for path in [*complete, *accepted]]
        return extras

    def convert(self, text, context="", engine="neural", mode="batch", raw=False):
        started = time.perf_counter()
        with self.lock, self.models.lock:
            self.last_trace = None
            if raw or engine != "neural":
                return self.models.baseline.convert(text, context, engine=engine,
                                                    mode=mode, raw=raw)
            response = super().convert(text, context, engine=engine, mode=mode, raw=raw)
            response["proposal_union"] = {"experimental": True,
                "merge_stage": "before_word_ranking", "duplicate_policy": "baseline",
                "final_candidate_preservation_guaranteed": False}
            response["engine_ms"] = round((time.perf_counter() - started) * 1000, 3)
            return response
