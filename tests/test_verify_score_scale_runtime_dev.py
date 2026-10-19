from copy import deepcopy
import pytest
from scripts import verify_score_scale_runtime_dev as verifier


def fixture(monkeypatch, ranked=True):
    monkeypatch.setattr(verifier, 'sha', lambda path: 'fixture-model')
    model = dict(parameters=[.8, .1, .2, .3, .4], scale=[1., 1., 1., 1.])
    artifacts = {'fixture.py': 'fixture-hash'}
    version = verifier.expected_version(model['parameters'], model['scale'], 'fixture-model', artifacts, False)
    segment = dict(input_start=0, input_end=4, source='test', text='test', kind='raw')
    paths = [['test', 1., [segment]]] if ranked else []
    candidate = dict(text='test', score=.8 if ranked else None, segments=[segment])
    feature = dict(text='test', base_score=1., features=[0., 0., 0., 0.],
                   residual=0., adjusted_score=.8, segments=[segment],
                   edges=[dict(input_start=0, input_end=4, source='test', target='test', status='copy')])
    response = dict(input='test', context='', external_context_used=False,
        version=version, candidates=[candidate], result='test', model_artifacts=artifacts,
        sentence_ranking_applied=ranked,
        phonetic_score_scale=dict(parameters=model['parameters'], scale=model['scale'],
            checkpoint_sha256='fixture-model', identity_control=False, version=version),
        phonetic_choice=dict(rows=[feature] if ranked else [], score_records=[], public=['test'],
            parent_version=verifier.PRODUCTION, version=version, artifacts=artifacts,
            effective_weights=model['parameters'][1:], calibration_source='override'))
    return model, paths, response


@pytest.mark.parametrize('ranked', [True, False])
def test_audits_ranked_copy_and_unranked_passthrough(monkeypatch, ranked):
    model, paths, response = fixture(monkeypatch, ranked)
    artifacts = {'fixture.py': 'fixture-hash'}
    verifier.audit_response('test', response, paths, model, artifacts, artifacts)


@pytest.mark.parametrize('field', ['score', 'features', 'offset', 'source', 'version', 'identity',
                                    'coherent_version', 'missing_model', 'missing_choice', 'effective_weights'])
def test_rejects_corrupted_runtime_contract(monkeypatch, field):
    model, paths, response = fixture(monkeypatch)
    response = deepcopy(response)
    if field == 'score':
        response['phonetic_choice']['rows'][0]['adjusted_score'] += .1
    elif field == 'features':
        response['phonetic_choice']['rows'][0]['features'][0] = 1.
    elif field == 'offset':
        response['candidates'][0]['segments'][0]['input_start'] = 1
    elif field == 'source':
        response['context'] = '정답 힌트'
    elif field == 'version':
        response['version'] = 'wrong-version'
    elif field == 'identity':
        response['phonetic_score_scale']['identity_control'] = True
    elif field == 'coherent_version':
        for value in (response, response['phonetic_score_scale'], response['phonetic_choice']):
            value['version'] = 'experimental-score-scale-1-forged'
    elif field == 'missing_model':
        response['model_artifacts'] = {}
    elif field == 'missing_choice':
        response['phonetic_choice']['artifacts'] = {}
    else:
        response['phonetic_choice']['effective_weights'] = [0., 0., 0., 0.]
    with pytest.raises(ValueError):
        artifacts = {'fixture.py': 'fixture-hash'}
        verifier.audit_response('test', response, paths, model, artifacts, artifacts)


def test_unranked_response_cannot_hide_candidate_paths(monkeypatch):
    model, _, response = fixture(monkeypatch, False)
    with pytest.raises(ValueError, match='unranked'):
        artifacts = {'fixture.py': 'fixture-hash'}
        verifier.audit_response('test', response, [['unexpected', 1., []]], model, artifacts, artifacts)
