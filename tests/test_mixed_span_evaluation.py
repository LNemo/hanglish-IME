import pytest
from scripts.evaluate_mixed_span_sentences import merge_scored,assert_segments,summarize


def test_merge_keeps_every_parent_score_and_path():
    parent=[('parent',3.,[{'a':1}]),('other',-2.,[{'a':2}])]
    extra=[('new',10.,[{'a':3}])]
    result=merge_scored(parent,extra,[4.])
    assert result[0]==('new',4.,[{'a':3}])
    assert all(path in result for path in parent)
    with pytest.raises(ValueError):merge_scored(parent,[parent[0]],[10.])


def test_path_offsets_and_literal_preservation_enforced():
    segments=[dict(input_start=0,input_end=2,source='12',text='12',kind='raw'),
              dict(input_start=2,input_end=4,source='ga',text='가',kind='mixed-span')]
    assert_segments('12ga',('12가',2.,segments))
    segments[0]['text']='13'
    with pytest.raises(ValueError):assert_segments('12ga',('13가',2.,segments))


def test_uncovered_rows_remain_in_accuracy_denominator():
    def row(id):
        return dict(id=id,parent_result='wrong',result='wrong',parent_pool=[dict(text='wrong')],
                    scored_pool=[dict(text='wrong')],generated_extras=[],
                    parent_displayed=[dict(text='wrong')],displayed=[dict(text='wrong')])
    captures=[row('a'),row('b')]
    labels=[dict(id=id,input='roman sentence',expected=['gold sentence'],category='test') for id in ('a','b')]
    result=summarize(captures,labels)
    assert result['counts']['rows']==2 and result['counts']['top1']==0
    assert not result['gate_passed']
