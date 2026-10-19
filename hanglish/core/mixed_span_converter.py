"""Sentence conversion with bounded, parent-preserving mixed-span proposals."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from hanglish.core.neural_converter import NeuralConverter, file_sha256
from hanglish.core import mixed_span_proposals


class MixedSpanConverter(NeuralConverter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.implementation_artifacts = {
            'hanglish/core/mixed_span_converter.py': file_sha256(Path(__file__)),
            'hanglish/core/mixed_span_proposals.py': file_sha256(Path(mixed_span_proposals.__file__)),
        }
        self._stamp_version()
        self.mixed_span_counts = dict(generated=0, accepted=0)

    def _stamp_version(self):
        self.parent_version = self.version
        payload = json.dumps(dict(parent=self.parent_version,
                                  sources=self.implementation_artifacts), sort_keys=True)
        self.version = 'neural-mixed-1-' + hashlib.sha256(payload.encode()).hexdigest()[:12]

    def load_models(self):
        with self.lock:
            if not self.models_loaded:
                super().load_models()
                self._stamp_version()

    def _extend_sentence_paths(self, text, context, paths):
        extras = mixed_span_proposals.sentence_extras(text, paths, self.word_options, limit=96)
        accepted = [path for path in extras if self.sentence_ranker.can_score(path[0])]
        self.mixed_span_counts = dict(generated=len(extras), accepted=len(accepted))
        if not accepted:
            return paths
        scores = self.sentence_ranker.score_pairs([context] * len(accepted),
                                                  [path[0] for path in accepted])
        if len(scores) != len(accepted) or any(not math.isfinite(score) for score in scores):
            raise RuntimeError('mixed-span ranker returned invalid scores')
        seen = {path[0] for path in paths}
        if len(seen) != len(paths) or any(path[0] in seen for path in accepted):
            raise RuntimeError('mixed-span pool has duplicate paths')
        additional = [(path[0], float(score), path[2]) for path, score in zip(accepted, scores)]
        return sorted([*paths, *additional], key=lambda path: (-path[1], path[0]))

    def convert(self, text, context='', engine='neural', mode='batch', raw=False):
        # Counts belong to this request; serialize the shared converter state.
        with self.lock:
            self.mixed_span_counts = dict(generated=0, accepted=0)
            result = super().convert(text, context, engine, mode, raw)
            result['mixed_span'] = dict(self.mixed_span_counts)
            result['parent_version'] = self.parent_version
            result['implementation_artifacts'] = dict(self.implementation_artifacts)
            return result
