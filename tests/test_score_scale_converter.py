import json
import threading
import pytest
from hanglish.core.score_scale_converter import ScoreScaleConverter
from hanglish.core.production_phonetic_choice_converter import ProductionPhoneticChoiceConverter


def bare(parameters, scale=(1., 1., 1., 1.)):
    value = object.__new__(ScoreScaleConverter)
    value.score_scale_parameters, value.score_scale = tuple(parameters), tuple(scale)
    value.calibration_weights = tuple(w / s for w, s in zip(parameters[1:], scale))
    value.calibration_source = 'override'
    value.lock = threading.RLock()
    value.phonetic_model = object()
    value._phonetic_model_loaded = True
    value.choice_ms = 0.
    calls = []
    def scorer(model, pairs, batch_size):
        calls.append(list(pairs))
        return [dict(source=s, target=t, log_probability=(-.1 if t == '김' else -2.) * (3*len(t)+1),
                     token_count=3*len(t)+1, mean_log_probability=-.1 if t == '김' else -2.) for s, t in pairs]
    value._likelihood_scorer = scorer
    return value, calls


def paths():
    return [(text, score, [dict(input_start=0, input_end=3, source='gim', text=text, kind='fixture')])
            for text, score in [('한', 1.1), ('김', 1.)]]


def test_actual_feature_path_applies_alpha_scale_without_new_candidates():
    value, calls = bare([.5, 1., 0., 0., 0.], (2., 1., 1., 1.))
    original = paths()
    result = value.calibrate_paths('gim', original)
    assert len(calls) == 1
    assert [p[0] for p in result] == ['김', '한']
    for row, path in zip(value.choice_trace['rows'], original):
        assert row['base_score'] == path[1]
        assert row['adjusted_score'] == .5 * path[1] + row['features'][0] / 2.
        assert row['segments'] == path[2]
    assert not value.choice_trace['zero_weight_parity']


def test_identity_returns_same_paths_and_scores():
    value, _ = bare([1., 0., 0., 0., 0.])
    original = paths()
    assert value.calibrate_paths('gim', original) is original
    assert value.choice_trace['zero_weight_score_parity']
    assert value.choice_trace['zero_weight_order_parity']


def test_alpha_only_does_not_claim_identity():
    value, _ = bare([.25, 0., 0., 0., 0.])
    result = value.calibrate_paths('gim', paths())
    assert [p[1] for p in result] == [.275, .25]
    assert not value.choice_trace['zero_weight_score_parity']


def test_constructor_pins_model_and_identity_has_distinct_version(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'scale.json'
    checkpoint.write_text(json.dumps(dict(schema='phonetic-score-scale-model-v1', arm='learned_scaled',
                                          parameters=[.8, .1, .2, .3, .4], scale=[1., 2., 3., 4.])))
    def minimal_parent(self, **kwargs):
        self.calibration_weights = kwargs['calibration_weights']
        self.choice_parent_version = 'fixture-production'
        self.calibration_source = 'override'
        self.phonetic_likelihood_artifacts = {}
        self.calibration_artifacts = self._calibration_source_artifacts(include_checkpoint=False)
        self.version = self._choice_version()
    monkeypatch.setattr(ProductionPhoneticChoiceConverter, '__init__', minimal_parent)
    learned = ScoreScaleConverter(score_scale_checkpoint=checkpoint)
    identity = ScoreScaleConverter(score_scale_checkpoint=checkpoint, identity_control=True)
    assert learned.version != identity.version
    assert learned.calibration_artifacts['external/scale.json'] == learned.score_scale_sha256
    assert identity.score_scale_parameters == (1., 0., 0., 0., 0.)
    checkpoint.write_text(checkpoint.read_text().replace('0.8', '-0.8'))
    with pytest.raises(ValueError, match='invalid'):
        ScoreScaleConverter(score_scale_checkpoint=checkpoint)


def test_source_path_validation_is_still_enforced():
    value, _ = bare([.8, .1, .2, .3, .4])
    original = paths()
    original[0][2][0]['input_start'] = 1
    with pytest.raises(RuntimeError, match='offsets'):
        value.calibrate_paths('gim', original)
