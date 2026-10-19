import pytest

from scripts.analyze_sentence_transducer import diagnose, edit_distance


def fixture(text='test를 시행합니다', reason='complete'):
    rows = [dict(id='dev-1', input='testreul sihaenghamnida', expected=['test를 시행합니다'])]
    tokens = (sum(3 if '가' <= char <= '힣' else len(char.encode('utf-8')) for char in text)
              if reason == 'complete' else (512 if reason == 'token_limit' else 2))
    predictions = [dict(id='dev-1', input=rows[0]['input'], response=dict(text=text, reason=reason, generated_tokens=tokens))]
    baseline = [dict(id='dev-1', request=dict(input=rows[0]['input'], context=''),
                     mixed=dict(result='test를 시행함니다', candidates=[dict(text='test를 시행함니다')]))]
    return rows, predictions, baseline


def test_new_candidate_is_coverage_not_claimed_selection():
    result = diagnose(*fixture())
    assert result['metrics']['new_correct_candidate'] == 1
    assert result['metrics']['baseline_top1'] == 0
    assert result['metrics']['displayed_union'] == 1
    assert result['union_is_oracle_coverage_not_top1']
    assert result['metrics']['character_error_rate'] == 0


def test_incomplete_generation_is_wrong_even_with_matching_partial_text():
    with pytest.raises(ValueError, match='incomplete generation'):
        diagnose(*fixture(reason='token_limit'))
    result = diagnose(*fixture(None, reason='token_limit'))
    assert result['metrics']['correct'] == 0
    assert result['metrics']['latin'] == dict(rows=1, exact=0)
    assert result['metrics']['character_error_rate'] == 1


def test_failed_generation_and_literal_corruption():
    assert diagnose(*fixture(None, 'invalid_encoding'))['metrics']['correct'] == 0
    result = diagnose(*fixture('TEST를 시행합니다'))
    assert result['metrics']['latin'] == dict(rows=1, exact=0)
    assert result['metrics']['edit_distance'] == 4


def test_unicode_edit_distance_and_multiple_references():
    assert edit_distance('입니다', '임니다') == 1
    assert edit_distance('', 'abc') == 3
    rows, predictions, baseline = fixture('test를 실행합니다')
    rows[0]['expected'].append('test를 실행합니다')
    assert diagnose(rows, predictions, baseline)['metrics']['character_error_rate'] == 0


@pytest.mark.parametrize('mutation', ['id', 'source', 'context'])
def test_mismatched_captures_are_rejected(mutation):
    rows, predictions, baseline = fixture()
    if mutation == 'id':
        predictions[0]['id'] = 'other'
    elif mutation == 'source':
        predictions[0]['input'] = 'gold answer'
    else:
        baseline[0]['request']['context'] = '한국어 힌트'
    with pytest.raises(ValueError):
        diagnose(rows, predictions, baseline)


@pytest.mark.parametrize('mutation', ['gold', 'response_context', 'reason', 'tokens'])
def test_invalid_generation_contract_is_rejected(mutation):
    rows, predictions, baseline = fixture()
    if mutation == 'gold':
        predictions[0]['expected'] = rows[0]['expected']
    elif mutation == 'response_context':
        predictions[0]['response']['context'] = 'hint'
    elif mutation == 'reason':
        predictions[0]['response']['reason'] = 'other'
    else:
        predictions[0]['response']['generated_tokens'] = 999
    with pytest.raises(ValueError):
        diagnose(rows, predictions, baseline)
