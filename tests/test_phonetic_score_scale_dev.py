from copy import deepcopy
import pytest
from scripts.evaluate_phonetic_score_scale_dev import compare, project, render, validate_model


def model():
    return dict(schema='phonetic-score-scale-model-v1', arm='learned_scaled',
                parameters=[1., 2., 0., 0., 0.], scale=[1., 1., 1., 1.])


def record():
    source = 'a b'
    paths = [[text, score, [dict(input_start=0, input_end=3, source=source, text=text, kind='lexicon')]]
             for text, score in [('가 나', 1.), ('다 라', 0.)]]
    features = [dict(text=p[0], base_score=p[1], segments=p[2], features=[value, 0., 0., 0.])
                for p, value in zip(paths, [-1., 0.])]
    public = render(source, paths)
    parent = dict(result=public[0]['text'], candidates=public, sentence_ranking_applied=True)
    return dict(id='fixture', request=dict(input=source, context=''), parent=parent,
                parent_pool=[p[0] for p in paths], base_paths=paths,
                proposed=parent | dict(phonetic_choice=dict(rows=features)))


def test_scaling_reorders_only_existing_candidates_and_preserves_source():
    before = record()
    original = deepcopy(before)
    actual = project(before, model())
    assert actual['proposed']['result'] == '다 라'
    assert {c['text'] for c in actual['proposed']['candidates']} == {'가 나', '다 라', 'a b'}
    assert actual['score_rows'][0]['score'] == -1.
    assert before == original


def test_identity_recovers_original_and_scale_changes_effect():
    weights = model()
    weights['parameters'] = [1., 0., 0., 0., 0.]
    before = record()
    assert project(before, weights)['proposed'] == before['parent']
    weights['parameters'][1] = 2.
    weights['scale'][0] = 10.
    assert project(before, weights)['proposed']['result'] == '가 나'


def test_unranked_branch_stays_unchanged():
    before = record()
    before['base_paths'], before['parent_pool'] = [], []
    before['proposed']['phonetic_choice']['rows'] = []
    before['proposed']['sentence_ranking_applied'] = False
    before['parent']['sentence_ranking_applied'] = False
    assert project(before, model())['proposed'] == before['parent']


@pytest.mark.parametrize('change', ['scale', 'alpha', 'nonfinite'])
def test_rejects_invalid_model(change):
    weights = model()
    if change == 'scale':
        weights['scale'][0] = 0
    elif change == 'alpha':
        weights['parameters'][0] = -1
    else:
        weights['parameters'][2] = float('nan')
    with pytest.raises(ValueError, match='coefficients'):
        validate_model(weights)


def test_feature_misalignment_is_rejected():
    before = record()
    before['proposed']['phonetic_choice']['rows'].reverse()
    with pytest.raises(ValueError, match='alignment'):
        project(before, model())


def test_regressions_reject_even_if_candidate_remains_in_top5():
    actual = project(record(), model())
    labels = [dict(id='fixture', input='a b', context='', split='dev', expected=['가 나'])]
    result = compare([actual], labels, 'parent')
    assert result['regressions'] == 1 and result['top5_losses'] == 0
    assert result['preservation_gate'] is False
    labels[0]['expected'] = ['다 라']
    result = compare([actual], labels, 'parent')
    assert result['corrections'] == 1 and result['preservation_gate'] is True


def test_public_raw_slot_and_ties_match_contract():
    paths = [[str(i), 1., []] for i in reversed(range(6))]
    public = render('raw', paths)
    assert [c['text'] for c in public] == ['0', '1', '2', '3', 'raw']


def test_late_label_identity_mismatch_is_rejected():
    label = dict(id='wrong', input='a b', context='', split='dev', expected=['다 라'])
    with pytest.raises(ValueError, match='identity'):
        compare([project(record(), model())], [label], 'parent')
