import copy
import torch
import pytest

from scripts.capture_roman_sentence_training import select_train
from scripts.train_roman_sentence_pilot import training_group, loss_from_scores, validate_preflight, RECIPE


def row():
    return dict(id='train',input='roman sentence',expected=['정답 문장'],
                candidates=['오답 문장','정답 문장','다른 문장'],
                teacher_scores=[3.,1.,0.],result='오답 문장',sentence_ranked=True)


def test_absent_gold_is_never_injected_into_training_candidates():
    example=row();example['expected']=['없는 정답']
    assert training_group(example) is None


def test_training_uses_original_source_and_only_generated_candidates():
    example=row();group=training_group(example)
    assert group['input']==example['input']
    assert set(group['candidates'])==set(example['candidates'])
    assert 'expected' not in group and 'context' not in group
    assert group['candidates'][group['positive_indices'][0]]=='정답 문장'
    assert not group['distill']


def test_multipositive_loss_rewards_any_allowed_answer_and_has_finite_gradient():
    group=dict(positive_indices=[0,2],distill=False)
    scores=torch.tensor([2.,0.,1.],requires_grad=True)
    loss=loss_from_scores(scores,group)
    expected=-torch.log(torch.softmax(scores,0)[[0,2]].sum())
    assert torch.allclose(loss,expected)
    loss.backward()
    assert torch.isfinite(scores.grad).all() and scores.grad[1]>0


def test_pilot_selection_does_not_consult_expected_or_runtime_performance():
    rows=[dict(id=str(i),category='natural_sentence_context' if i<400 else 'liaison',expected=['a']) for i in range(550)]
    changed=copy.deepcopy(rows)
    for item in changed:item['expected']=['b'];item['result']='wrong'
    assert [r['id'] for r in select_train(rows)]==[r['id'] for r in select_train(changed)]
    assert len(select_train(rows))==512


def proof():
    return dict(completed=True,phase='preflight',pins={},recipe=RECIPE,eligible=5,
                updates=[dict(step=i,loss=.1,gradient_norm=.2,elapsed_s=.5) for i in (1,2,3)],
                peak_rss_bytes=1000000,elapsed_s=5.,max_tokens=200)


def test_preflight_requires_actual_finite_updates_and_resource_bounds():
    validate_preflight(proof(),{},5)
    for field,value in [('updates',[]),('peak_rss_bytes',RECIPE['rss_limit']+1),
                        ('elapsed_s',float('nan')),('max_tokens',385),('eligible',4)]:
        broken=proof();broken[field]=value
        with pytest.raises(ValueError):validate_preflight(broken,{},5)
    broken=proof();broken['updates'][1]['loss']=float('inf')
    with pytest.raises(ValueError):validate_preflight(broken,{},5)
