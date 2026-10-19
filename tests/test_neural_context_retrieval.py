import copy

import numpy as np
import torch

from hanglish.models.neural_context_retrieval import NeuralContextBank, mean_content


def row(id, label='낫', group=None):
    return dict(id=id, group_id=group or id, context=id+' [MASK] 문맥',
                expected=[label], candidates=['낫', '낯'], teacher_scores=[1., 2.], base_scores=[7., 7.])


def test_content_mean_excludes_special_and_padding():
    hidden=torch.tensor([[[100.,100.],[2.,4.],[4.,8.],[200.,200.]]])
    result=mean_content(hidden,torch.tensor([[1,1,1,0]]),torch.tensor([[1,0,0,0]]))
    assert result.tolist()==[[3.,6.]]
    assert mean_content(hidden,torch.zeros((1,4)),torch.zeros((1,4))).tolist()==[[0.,0.]]


def test_center_uses_only_bank_and_query_label_is_unused():
    bankrows=[row('a'),row('b','낯')]
    emb={'a':[2.,0.],'b':[0.,2.],'q':[10.,0.]}
    bank=NeuralContextBank(bankrows,emb)
    assert bank.center.tolist()==[1.,1.]
    query=row('q');changed=copy.deepcopy(query);changed['expected']=['낯']
    assert bank.evidence(query)==bank.evidence(changed)
    assert bank.evidence(query)[0][0]['id']=='a'
    assert bank.evidence(query)[1]==[]


def test_exclusions_apply_after_dense_similarity():
    bank=NeuralContextBank([row('a',group='doc'),row('b','낯')],{'a':[2.,0.],'b':[0.,2.],'q':[2.,0.]})
    assert bank.evidence(row('q',group='doc'))==[[],[]]


def test_empty_context_vector_not_shifted_into_false_support():
    bank=NeuralContextBank([row('a'),row('b','낯')],{'a':[2.,0.],'b':[0.,2.],'q':[0.,0.]})
    assert np.array_equal(bank.normalize([0.,0.]),np.zeros(2))
    assert bank.evidence(row('q'))==[[],[]]


def test_query_without_document_metadata_can_retrieve():
    bank=NeuralContextBank([row('a'),row('b','낯')],{'a':[2.,0.],'b':[0.,2.],'q':[2.,0.]})
    query=row('q');del query['group_id']
    assert bank.evidence(query)[0][0]['id']=='a'
