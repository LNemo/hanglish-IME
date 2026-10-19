import copy

import torch

from hanglish.models.learned_context_metric import LearnedContextMetric, prototype_inputs


def row(id, word='낫', group=None):
    return dict(id=id, group_id=group or id, context=id+' [MASK] 문맥',
                candidates=['낫', '낯'], expected=[word])


def test_prototypes_exclude_document_and_do_not_read_query_labels():
    bank=[row('a', group='same'), row('b'), row('c', '낯')]
    query=row('q', group='same')
    vectors={'a':[100.,0.], 'b':[2.,0.], 'c':[0.,2.], 'q':[1.,0.]}
    values=prototype_inputs(bank,[query],vectors)
    changed=copy.deepcopy(query);changed['expected']=['낯']
    other=prototype_inputs(bank,[changed],vectors)
    for key in ('query','prototypes','query_indices','slots','center'):
        assert torch.equal(values[key],other[key])
    assert torch.allclose(values['prototypes'][0].double()+values['center'],torch.tensor([2.,0.],dtype=torch.float64),atol=1e-6)


def test_initial_parent_preservation_and_projection_gradients():
    bank=[row('a'),row('b','낯')];vectors={'a':[2.,0.],'b':[0.,2.],'q':[3.,1.]}
    values=prototype_inputs(bank,[row('q')],vectors)
    model=LearnedContextMetric(2)
    assert model(values).tolist()==[[0.,0.]]
    with torch.no_grad():model.gain.fill_(.5)
    loss=model(values)[0,0];loss.backward()
    assert model.projection.weight.grad.abs().sum()>0
    assert torch.equal(model(values),LearnedContextMetric.restore(model.export())(values))


def test_no_eligible_prototypes_produces_zero_scores():
    query=row('a');values=prototype_inputs([query],[query],{'a':[2.,0.]})
    model=LearnedContextMetric(2)
    with torch.no_grad():model.gain.fill_(1.)
    assert model(values).tolist()==[[0.,0.]]


def test_empty_query_and_shared_source_do_not_create_support():
    a,b,q=row('a'),row('b','낯'),row('q')
    a['source_ids']=['doc'];q['source_ids']=['doc']
    b['context']=q['context']
    vectors={'a':[2.,0.],'b':[0.,2.],'q':[1.,0.]}
    values=prototype_inputs([a,b],[q],vectors)
    assert values['prototypes'].shape==(0,2)
    q['source_ids']=[];vectors['q']=[0.,0.]
    values=prototype_inputs([a,b],[q],vectors)
    assert values['prototypes'].shape==(0,2)
    assert values['query'].tolist()==[[0.,0.]]
