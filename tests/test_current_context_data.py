from copy import deepcopy
import pytest
from scripts.prepare_current_context_data import validate_case, validate_groups, align_groups, visible_candidates


def fixture():
    source = {'id': 'case', 'input': 'gak', 'context': '', 'expected': ['각']}
    row = {**source, 'candidates': ['갂','각'], 'parent_candidates': ['갂','각'],
           'teacher_scores': [1.,2.], 'positive_indices': [1], 'sentence_eligible': True,
           'actual_candidates': ['각','갂','gak'], 'result': '각', 'groups': []}
    return row, source


def test_actual_candidate_order_must_reproduce_teacher_scores():
    row, source = fixture()
    validate_case(row, source)
    row['teacher_scores'] = [3.,2.]
    with pytest.raises(ValueError, match='reproduce'):
        validate_case(row, source)


def test_gold_is_not_injected_when_absent_from_source_only_candidates():
    row, source = fixture()
    row.update(candidates=['갂'], parent_candidates=['갂'], positive_indices=[],
               teacher_scores=[1.], actual_candidates=['갂','gak'], result='갂')
    validate_case(row, source)
    row['positive_indices'] = [0]
    with pytest.raises(ValueError, match='labels'):
        validate_case(row, source)


def test_unaligned_word_group_is_retained_without_invented_labels():
    _, source = fixture()
    group = {'start': 0, 'end': 1, 'source': 'g', 'candidates': ['ㄱ']}
    aligned = align_groups(source, [group])
    assert aligned[0]['label_aligned'] is False
    assert aligned[0]['positive_indices'] == []
    assert aligned[0]['candidates'] == ['ㄱ']


def test_raw_candidate_and_fallback_order_preserved():
    assert visible_candidates(['가','나','다','라','마'], 'abc') == ['가','나','다','라','abc']
    row, source = fixture()
    row.update(sentence_eligible=False, teacher_scores=[], candidates=['각','gak'],
               parent_candidates=['각','gak'],positive_indices=[0],actual_candidates=['각','gak'])
    validate_case(row, source)
    row['teacher_scores'] = [1.,0.]
    with pytest.raises(ValueError, match='fallback'):
        validate_case(row, source)


def valid_word_group(source):
    return align_groups(source, [{'start':0,'end':3,'source':'gak','context':'[MASK]',
        'candidates':['각','갂'],'base_scores':[1.,0.],'teacher_scores':[2.,1.],
        'kinds':['neural','syllable']}])[0]


@pytest.mark.parametrize('key,value', [('start',-1), ('end',4), ('source','other'),
    ('candidates',[]), ('candidates',['각','각']), ('candidates',[1,'각']),
    ('base_scores',[1.]), ('teacher_scores',[float('nan'),1.]),
    ('base_scores',[float('inf'),1.]), ('kinds',['neural']), ('context','no mask')])
def test_rejects_malformed_word_groups(key,value):
    _,source=fixture()
    group=valid_word_group(source)
    validate_groups(source,[group])
    group[key]=value
    with pytest.raises(ValueError):
        validate_groups(source,[group])


def test_duplicate_word_spans_rejected_and_unaligned_valid_span_retained():
    _,source=fixture()
    group=valid_word_group(source)
    with pytest.raises(ValueError,match='duplicated'):
        validate_groups(source,[group,deepcopy(group)])
    unaligned={**group,'start':0,'end':1,'source':'g'}
    unaligned=align_groups(source,[unaligned])
    validate_groups(source,unaligned)
    assert unaligned[0]['label_aligned'] is False
