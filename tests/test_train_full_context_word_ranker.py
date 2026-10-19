import json

import pytest
import torch

from scripts import train_full_context_word_ranker as training


def test_multi_positive_loss_and_gradients_include_all_accepted_candidates():
    row = dict(candidates=['a','b','c'], expected=['a','c'], base_scores=[0.,0.,0.])
    logits = torch.zeros(3, requires_grad=True)
    loss = training.group_loss(logits, [row])
    assert loss.item() == pytest.approx(-__import__('math').log(2/3))
    loss.backward()
    assert logits.grad[0] < 0 and logits.grad[2] < 0 and logits.grad[1] > 0


def test_uninformative_groups_are_not_training_targets():
    for accepted in ([], ['a','b']):
        with pytest.raises(ValueError, match='uninformative'):
            training.group_loss(torch.zeros(2), [dict(candidates=['a','b'], expected=accepted,
                                                     base_scores=[0.,0.])])


def test_selection_is_document_disjoint_and_independent_of_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(training, 'ROOT', tmp_path)
    monkeypatch.setattr(training, 'DATA', tmp_path)
    def write_data(expected):
        manifest = {'outputs': {}}
        for role in ('natural_train','retention_train'):
            rows = []
            for i in range(60):
                groups = [dict(id=f'{role}-{i}-{j}', source='source', context='[MASK]',
                               candidates=['a','b'], expected=expected, base_scores=[1.,0.],
                               teacher_scores=[1.,0.]) for j in range(4)]
                rows.append(dict(id=f'{role}-{i}', group_id=f'document-{i//2}', groups=groups))
            path=tmp_path/f'{role}.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            manifest['outputs'][role]={'sha256':training.digest(path)}
        path=tmp_path/'manifest.json'
        path.write_text(json.dumps(manifest))
        monkeypatch.setattr(training,'MANIFEST_SHA',training.digest(path))
    write_data(['a'])
    first,_=training.prepare()
    write_data(['b'])
    second,_=training.prepare()
    assert {k:[r['id'] for r in v] for k,v in first.items()} == {k:[r['id'] for r in v] for k,v in second.items()}
    fit={r['group_id'] for k,v in first.items() if k.endswith(':fit') for r in v}
    hold={r['group_id'] for k,v in first.items() if k.endswith(':hold') for r in v}
    assert fit and hold and not fit.intersection(hold)
    for rows in first.values():
        assert all(sum(r['case_id']==row['case_id'] for r in rows)<=2 for row in rows)


def test_missing_target_stays_in_evaluation_denominator():
    class FakeRanker:
        def can_score(self, context, candidate): return True
        def score_pairs(self, contexts, candidates): return [0.,1.]
    row=dict(id='row',context='context',candidates=['a','b'],expected=['missing'],
             teacher_scores=[1.,0.],base_scores=[0.,0.])
    result=training.evaluate(FakeRanker(),{'natural_train:hold':[row]},lambda:None)
    assert result['metrics']['natural_train:hold']['rows']==1
    assert result['metrics']['natural_train:hold']['missing']==1
    assert result['metrics']['natural_train:hold']['correct']==0
