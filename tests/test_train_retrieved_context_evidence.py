import torch

from scripts.train_retrieved_context_evidence import pack,predict
from hanglish.models.retrieved_context_evidence import ContextBank,RetrievedContextHead
from hanglish.models.candidate_evidence import objective


def test_no_support_preserves_parent_and_zero_loss_gradient():
    row=dict(id='q',group_id='q',kind='source',context='풀 [MASK]',expected=['낫'],candidates=['낫','낯'],teacher_scores=[0.,1.],base_scores=[7.,7.])
    bank=ContextBank([row])
    values,evidence=bank.features(row)
    features={'q':dict(values=values,evidence=evidence)}
    model=RetrievedContextHead()
    with torch.no_grad(): model.output.bias.fill_(1.)
    assert predict(model,[row],features)[0]['result']=='낯'
    loss=objective(model,pack([row],features));loss.backward()
    assert all(torch.count_nonzero(p.grad)==0 for p in model.parameters())


def test_fixed_missing_gold_stays_in_denominator():
    row=dict(id='missing',kind='fixed',fixed_result=None,expected=['낫'])
    prediction=predict(RetrievedContextHead(),[row],{})
    assert prediction[0]['correct'] is False
    assert prediction[0]['result'] is None
