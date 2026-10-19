from copy import deepcopy
import pytest
from scripts.verify_phonetic_score_scale_test import metrics


def examples():
    records, labels = [], []
    for index in range(1000):
        before = dict(result='test를 시행', candidates=[dict(text='test를 시행')], sentence_ranking_applied=True)
        after = deepcopy(before)
        if index == 0:
            after = dict(result='테스트를 시행', candidates=[dict(text='테스트를 시행'), dict(text='test를 시행')], sentence_ranking_applied=True)
        records.append(dict(id=str(index), request=dict(input='testreul sihaeng', context=''),
                            parent=before, old_calibration=before, proposed=after,
                            actual_base_pool=['test를 시행', '테스트를 시행']))
        labels.append(dict(id=str(index), input='testreul sihaeng', context='', split='test',
                           expected=['test를 시행'], category='fixture'))
    return records, labels


def test_exact_totals_categories_candidate_and_english_regression():
    records, labels = examples()
    result = metrics(records, labels, 'parent')
    assert result['parent']['counts']['top1'] == 1000
    assert result['proposed']['counts'] == dict(rows=1000, top1=999, top5=1000, errors=0)
    assert result['proposed']['categories']['fixture'] == result['proposed']['counts']
    assert result['regressions'] == result['english_ascii_signature']['signature_regressions'] == 1
    assert result['actual_base_pool_stage_counts']['actual_pool_to_first_loss'] == 1
    assert result['quality_gate'] is False


def test_wrong_split_and_duplicate_rows_fail():
    records, labels = examples()
    labels[0]['split'] = 'dev'
    with pytest.raises(ValueError, match='identity'):
        metrics(records, labels, 'parent')
    labels[0]['split'] = 'test'
    records[1]['id'] = records[0]['id']
    with pytest.raises(ValueError, match='coverage'):
        metrics(records, labels, 'parent')
