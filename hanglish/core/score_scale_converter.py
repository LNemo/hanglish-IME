"""Frozen scale-aware calibration of the existing production candidate pool."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

from hanglish.core.neural_converter import ROOT, file_sha256
from hanglish.core.production_phonetic_choice_converter import ProductionPhoneticChoiceConverter

DEFAULT_MODEL = ROOT / '.omx/artifacts/phonetic-score-scale-training-v1/model.json'


class ScoreScaleConverter(ProductionPhoneticChoiceConverter):
    def __init__(self, *, score_scale_checkpoint=DEFAULT_MODEL, identity_control=False, **kwargs):
        self.score_scale_checkpoint = Path(score_scale_checkpoint)
        model = json.loads(self.score_scale_checkpoint.read_text())
        parameters, scale = model.get('parameters'), model.get('scale')
        if (model.get('schema') != 'phonetic-score-scale-model-v1'
                or model.get('arm') != 'learned_scaled'
                or not isinstance(parameters, list) or len(parameters) != 5
                or not isinstance(scale, list) or len(scale) != 4
                or not all(type(x) in (int, float) and math.isfinite(x) for x in parameters + scale)
                or parameters[0] < 0 or not all(x > 0 for x in scale)):
            raise ValueError('invalid score-scale model')
        if type(identity_control) is not bool:
            raise ValueError('identity control must be boolean')
        if 'calibration_weights' in kwargs or 'calibration_checkpoint' in kwargs:
            raise ValueError('score-scale checkpoint owns calibration coefficients')
        self.score_scale_parameters = tuple([1., 0., 0., 0., 0.] if identity_control else parameters)
        self.score_scale = tuple(scale)
        self.score_scale_identity = identity_control
        self.score_scale_sha256 = file_sha256(self.score_scale_checkpoint)
        super().__init__(calibration_weights=[w / s for w, s in zip(self.score_scale_parameters[1:], scale)], **kwargs)

    def _calibration_source_artifacts(self, *, include_checkpoint):
        artifacts = super()._calibration_source_artifacts(include_checkpoint=include_checkpoint)
        for path in (Path(__file__), self.score_scale_checkpoint):
            try:
                name = str(path.relative_to(ROOT))
            except ValueError:
                name = 'external/' + path.name
            artifacts[name] = file_sha256(path)
        return artifacts

    def _choice_version(self):
        payload = dict(parent=super()._choice_version(), parameters=self.score_scale_parameters,
                       scale=self.score_scale, checkpoint=self.score_scale_sha256,
                       identity_control=self.score_scale_identity)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
        return 'experimental-score-scale-1-' + digest

    def calibrate_paths(self, source, paths):
        # Reuse the frozen feature/likelihood implementation once per request.
        super().calibrate_paths(source, paths)
        started = time.perf_counter()
        alpha, *weights = self.score_scale_parameters
        result = []
        for path, row in zip(paths, self.choice_trace['rows']):
            residual = sum(w * (v / s) for w, v, s in zip(weights, row['features'], self.score_scale))
            score = alpha * path[1] + residual
            if not math.isfinite(score):
                raise ValueError('nonfinite scale-aware score')
            row['residual'], row['adjusted_score'] = residual, score
            result.append((path[0], score, path[2]))
        result.sort(key=lambda path: (-path[1], path[0]))
        identity = alpha == 1. and all(w == 0. for w in weights)
        parity = result == paths
        if identity and not parity:
            raise ValueError('identity calibration changed production paths')
        # A zero residual alone preserves the base score only when alpha is one.
        self.choice_trace['zero_weight_parity'] = alpha == 1.
        self.choice_trace['zero_weight_score_parity'] = identity and parity
        self.choice_trace['zero_weight_order_parity'] = identity and parity
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.choice_trace['calibration_ms'] += elapsed_ms
        self.choice_ms += elapsed_ms
        return paths if identity else result

    def convert(self, text, context='', engine='neural', mode='batch', raw=False):
        with self.lock:
            response = super().convert(text, context, engine, mode, raw)
            response['phonetic_score_scale'] = dict(
                parameters=list(self.score_scale_parameters), scale=list(self.score_scale),
                checkpoint_sha256=self.score_scale_sha256,
                identity_control=self.score_scale_identity, version=self.version,
                score_formula='alpha*base + sum(weight*feature/scale)')
            return response
