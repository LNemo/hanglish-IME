from types import SimpleNamespace
import pytest
import torch
from scripts.prepare_sentence_hard_negatives import make_example,balanced_select,retention_split,aligned_row
from scripts.train_sentence_hard_negatives import objective,evaluate
from scripts.capture_current_sentence_pools import Recorder


def pool(candidates=None,scores=None,expected=None):
    return dict(id='x',group_id='g',category='natural',context='context',sentence_eligible=True,
                candidates=candidates or ['오답','정답'],teacher_scores=scores or [2.,0.],expected=expected or ['정답'])


def test_full_pool_positive_beyond_displayed_top5_is_retained():
    row=pool([str(i) for i in range(10)],list(range(10,0,-1)),['8'])
    example=make_example(row)
    assert example['candidates']==['8','0','1','2','3','4','5','6']
    assert example['positives']==[0]
    assert example['source_pool_size']==10
    assert example['loss_weight']==4 and not example['distill']


def test_correct_teacher_gets_distillation_but_wrong_teacher_does_not():
    wrong=make_example(pool())
    correct=make_example(pool(scores=[0.,2.]))
    assert (wrong['teacher_correct'],wrong['distill'],wrong['loss_weight'])==(False,False,4.)
    assert (correct['teacher_correct'],correct['distill'],correct['loss_weight'])==(True,True,1.)


def test_missing_gold_is_not_inserted_and_fallback_is_not_trained():
    assert make_example(pool(expected=['missing'])) is None
    row=pool();row['sentence_eligible']=False
    assert make_example(row) is None


@pytest.mark.parametrize('bad',['duplicate','nan','length','too_many_gold'])
def test_malformed_pool_is_rejected(bad):
    row=pool()
    if bad=='duplicate':row['candidates']=['a','a']
    if bad=='nan':row['teacher_scores']=[float('nan'),0.]
    if bad=='length':row['teacher_scores']=[1.]
    if bad=='too_many_gold':row=pool([str(i) for i in range(10)],list(range(10)),[str(i) for i in range(9)])
    with pytest.raises(ValueError):make_example(row)


def test_hard_loss_has_fourfold_supervised_gradient():
    example=make_example(pool())
    # prepare places the positive first, so reverse the original teacher order.
    logits=torch.tensor([0.,2.],requires_grad=True)
    loss=objective(logits,[example]);loss.backward()
    reference=torch.tensor([0.,2.],requires_grad=True)
    base=torch.nn.functional.cross_entropy(reference.reshape(1,2),torch.tensor([0]));base.backward()
    assert loss.item()==pytest.approx(4*base.item())
    assert logits.grad.tolist()==pytest.approx((4*reference.grad).tolist())


def test_invalid_hard_weight_and_unused_logits_rejected():
    example=make_example(pool());example['loss_weight']=1.
    with pytest.raises(ValueError,match='policy'):objective(torch.zeros(2),[example])
    with pytest.raises(ValueError,match='unconsumed'):objective(torch.zeros(3),[make_example(pool())])


def test_balancing_is_deterministic_and_does_not_mix_group_partitions():
    rows=[dict(id=f'{category}-{i}',group_id=f'{category}:{i}') for category in ('a','b') for i in range(20)]
    result=balanced_select(rows,8)
    assert result==balanced_select(list(reversed(rows)),8)
    assert sum(r['id'].startswith('a-') for r in result)==4
    fit={r['group_id'] for r in rows if retention_split(r['group_id'])=='fit'}
    held={r['group_id'] for r in rows if retention_split(r['group_id'])=='hold'}
    assert fit and held and not fit&held
    with pytest.raises(ValueError,match='not enough'):balanced_select(rows,41)


def test_fallback_and_missing_gold_stay_in_evaluation_denominator():
    class Ranker:
        model=SimpleNamespace(eval=lambda:None)
        def score_pairs(self,contexts,candidates):return list(reversed(range(len(candidates))))
    rows=[dict(id='a',context='',candidates=['yes','no'],expected=['yes'],sentence_eligible=True),
          dict(id='b',context='',candidates=['no','also no'],expected=['yes'],sentence_eligible=True),
          dict(id='c',context='',candidates=['yes'],expected=['yes'],sentence_eligible=False,fallback_result='yes')]
    result=evaluate(Ranker(),{'held':rows},lambda:None)['held']
    assert (result['rows'],result['correct'],result['covered'])==(3,2,2)
    assert result['predictions'][2]['scores'] is None


def test_source_join_refuses_changed_context_or_split():
    capture=dict(id='x',input='abc',context='ctx',suite='natural_train',calls=[],response={'result':'abc','candidates':[{'text':'abc'}]})
    target=dict(id='x',input='abc',context='ctx',split='train',group_id='g',expected=['에이비씨'])
    assert not aligned_row(capture,target)['sentence_eligible']
    with pytest.raises(ValueError):aligned_row(capture,{**target,'context':'wrong'})
    with pytest.raises(ValueError):aligned_row(capture,{**target,'split':'dev'})


def test_recorder_preserves_delegated_values():
    class Delegate:
        max_length=96
        def score_pairs(self,contexts,candidates):return [.1,.2]
    recorder=Recorder(Delegate())
    assert recorder.score_pairs(['a','a'],['b','c'])==[.1,.2]
    assert recorder.max_length==96
    assert recorder.calls==[dict(contexts=['a','a'],candidates=['b','c'],scores=[.1,.2])]
