from copy import deepcopy
import pytest
from scripts.evaluate_phonetic_score_scale_test import decision, project_test


def comparisons():
    baseline = dict(corrections=1, regressions=0, top5_losses=0,
                    english_ascii_signature=dict(signature_regressions=0, literal_signature_regressions=0))
    return dict(parent=deepcopy(baseline), old_calibration=deepcopy(baseline))


def test_requires_improvement_over_production():
    values = comparisons()
    assert decision(values)
    values['parent']['corrections'] = 0
    assert not decision(values)


@pytest.mark.parametrize('branch', ['parent', 'old_calibration'])
@pytest.mark.parametrize('field', ['regressions', 'top5_losses', 'signature_regressions', 'literal_signature_regressions'])
def test_any_baseline_preservation_loss_rejects(branch, field):
    values = comparisons()
    target = values[branch] if field in ('regressions', 'top5_losses') else values[branch]['english_ascii_signature']
    target[field] = 1
    assert not decision(values)


def test_old_model_need_not_improve_if_preserved():
    values = comparisons()
    values['old_calibration']['corrections'] = 0
    assert decision(values)


def fixture():
    candidate = dict(text='raw', score=None, segments=[dict(input_start=0, input_end=3, source='raw', text='raw', kind='raw')])
    response = dict(result='raw', candidates=[candidate], sentence_ranking_applied=False)
    record = dict(id='fixture', request=dict(input='raw', context=''), parent=response,
                  proposed=response | dict(phonetic_choice=dict(rows=[])), base_paths=[], actual_base_pool=[])
    model = dict(schema='phonetic-score-scale-model-v1', arm='learned_scaled',
                 parameters=[.8, .1, .2, .3, .4], scale=[1., 1., 1., 1.])
    return record, model


def test_test_pool_adapter_preserves_unranked_rows():
    record, model = fixture()
    original = deepcopy(record)
    value = project_test(record, model)
    assert value['actual_base_pool'] == []
    assert 'parent_pool' not in value
    assert value['proposed'] == record['parent']
    assert record == original


def test_test_pool_adapter_rejects_changed_domain():
    record, model = fixture()
    record['actual_base_pool'] = ['invented']
    with pytest.raises(ValueError, match='pool'):
        project_test(record, model)
