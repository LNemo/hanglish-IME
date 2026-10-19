import pytest
import torch
from scripts.train_runtime_collision_ranker import objective, fixed_accuracy, evaluate


def source(**kwargs):
    value = dict(kind='source', candidates=['박', '밖'], expected=['밖'], base_scores=[7., 7.])
    value.update(kwargs)
    return value


def test_source_objective_prefers_expected_candidate():
    logits = torch.tensor([0., 0.], requires_grad=True)
    loss = objective(logits, [source()]); loss.backward()
    assert loss.item() == pytest.approx(.69314718)
    assert logits.grad.tolist() == pytest.approx([.5, -.5])


def test_noninformative_source_group_rejected():
    with pytest.raises(ValueError, match='informative'):
        objective(torch.zeros(2), [source(expected=['박', '밖'])])


def test_fixed_rows_include_none_as_unaligned_failure():
    rows = [
        {'kind': 'fixed', 'fixed_result': '밖', 'expected': ['밖']},
        {'kind': 'fixed', 'fixed_result': '박', 'expected': ['밖']},
        {'kind': 'fixed', 'fixed_result': None, 'expected': ['밖']},
    ]
    assert fixed_accuracy(rows) == {'rows': 3, 'correct': 1}


def test_evaluation_includes_unaligned_and_uses_runtime_tie_break():
    class Model:
        def eval(self):
            pass

    class Ranker:
        model = Model()

        def score_pairs(self, contexts, candidates):
            return [0.] * len(candidates)

    rows = [dict(id='unaligned', kind='fixed', fixed_result=None, expected=['밖']),
            dict(id='scored', kind='source', context='[MASK]', candidates=['밖', '박'],
                 base_scores=[6., 6.], expected=['박']),
            dict(id='fixed', kind='fixed', fixed_result='밖', expected=['밖'])]
    result = evaluate(Ranker(), {'source_hold': rows}, lambda: None)['source_hold']
    assert result['rows'] == 3 and result['correct'] == 2
    assert [r['id'] for r in result['predictions']] == ['unaligned', 'scored', 'fixed']
    assert result['predictions'][0]['correct'] is False
    assert result['predictions'][1]['result'] == '박'
