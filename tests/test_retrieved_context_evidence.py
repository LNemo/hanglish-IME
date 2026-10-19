import copy

import torch

from hanglish.models.retrieved_context_evidence import ContextBank, RetrievedContextHead, context_text


def row(id, context, expected='낫', group=None, sources=None):
    return dict(id=id, group_id=group or id, source_ids=sources or [],
                context=context, expected=[expected], candidates=['낫', '낯'],
                teacher_scores=[1., 2.], base_scores=[7., 7.])


def test_query_labels_do_not_affect_retrieval():
    bank = ContextBank([row('a', '풀을 베는 [MASK] 도구'), row('b', '사람의 [MASK] 얼굴', '낯')])
    query = row('q', '풀을 자르는 [MASK] 도구')
    changed = copy.deepcopy(query); changed['expected'] = ['낯']
    assert bank.features(query) == bank.features(changed)
    assert bank.evidence(query)[0][0]['id'] == 'a'


def test_excludes_self_group_source_and_duplicate_context():
    bank = ContextBank([row('a', '풀을 베는 [MASK]', group='g'),
                        row('b', '풀을 베는 [MASK] 도구', sources=['doc']),
                        row('c', '풀을 자르는 [MASK] 도구')])
    query = row('q', '풀을 자르는 [MASK] 도구', group='g', sources=['doc'])
    assert bank.evidence(query) == [[], []]


def test_unknown_query_grams_do_not_extend_vocabulary():
    bank = ContextBank([row('a', '풀을 베는 [MASK] 도구')])
    before = dict(bank.idf)
    assert bank.vector('XYZ') == {}
    assert bank.idf == before


def test_residual_initially_zero_and_unsupported_stays_zero():
    model = RetrievedContextHead()
    values = torch.ones((2, 10), dtype=torch.float64)
    values[0, -1] = 0
    assert model(values).tolist() == [0., 0.]
    with torch.no_grad(): model.output.bias.fill_(1.)
    assert model(values)[0].item() == 0
    assert 0 < model(values)[1].item() < 2
    assert torch.equal(model(values), RetrievedContextHead.restore(model.export())(values))


def test_active_mask_context_window():
    assert context_text('앞 [MASK] 중간 [MASK] 뒤') == '앞 [MASK] 중간 뒤'
    assert context_text('a'*100) == 'a'*88
