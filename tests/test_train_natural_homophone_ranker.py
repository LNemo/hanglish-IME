import pytest
import torch
from scripts.train_natural_homophone_ranker import objective


def source(**kwargs):
    return dict(kind='source',candidates=['박','밖'],expected=['밖'],base_scores=[7.,7.],**kwargs)


def test_source_gradient_increases_correct_runtime_candidate():
    logits=torch.tensor([0.,0.],requires_grad=True)
    loss=objective(logits,[source()])
    loss.backward()
    assert loss.item()==pytest.approx(0.69314718)
    assert logits.grad.tolist()==pytest.approx([.5,-.5])


def test_retention_teacher_and_source_losses_have_separate_targets():
    retention=dict(kind='retention',candidates=['a','b'],expected=['missing'],
                   base_scores=[0.,0.],teacher_scores=[3.,0.])
    assert objective(torch.tensor([3.,0.]),[retention]).item()==pytest.approx(0.,abs=1e-6)
    correct=torch.tensor([0.,3.,3.,0.])
    wrong=torch.tensor([3.,0.,0.,3.])
    assert objective(correct,[source(),retention])<objective(wrong,[source(),retention])


@pytest.mark.parametrize('expected',[['missing'],['박','밖']])
def test_noninformative_source_group_cannot_be_trained(expected):
    unit=source();unit['expected']=expected
    with pytest.raises(ValueError,match='informative'):
        objective(torch.zeros(2),[unit])


@pytest.mark.parametrize('size',[1,3])
def test_group_logits_must_match_exactly(size):
    with pytest.raises(ValueError):objective(torch.zeros(size),[source()])
