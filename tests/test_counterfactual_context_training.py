import torch
import pytest

from scripts.train_counterfactual_context_ranker import flatten, objective, semantic_split


def test_context_split_keeps_eight_fit_four_hold_per_sense_without_overlap():
    rows = [{'id': f'{sense}-{i}', 'input': str(sense//2), 'expected': [str(sense)],
             'context': f'context {sense} {i}'} for sense in range(20) for i in range(12)]
    fit, hold = semantic_split(rows)
    assert len(fit) == 160 and len(hold) == 80
    assert not {r['id'] for r in fit} & {r['id'] for r in hold}
    assert semantic_split(list(reversed(rows)))[0] != []
    reverse_fit, reverse_hold = semantic_split(list(reversed(rows)))
    assert {r['id'] for r in fit} == {r['id'] for r in reverse_fit}
    assert {r['id'] for r in hold} == {r['id'] for r in reverse_hold}


def pair():
    return {'kind': 'pair', 'contexts': ['outside', 'surname'],
            'candidates': ['밖', '박'], 'base_scores': [0., 0.]}


def test_column_loss_adds_context_comparison_beyond_row_preferences():
    good = torch.tensor([4., 0., 0., 4.], requires_grad=True)
    row_offset = torch.tensor([14., 10., 0., 4.], requires_grad=True)
    assert torch.allclose(torch.nn.functional.cross_entropy(good.reshape(2,2), torch.tensor([0,1])),
                          torch.nn.functional.cross_entropy(row_offset.reshape(2,2), torch.tensor([0,1])))
    assert objective(row_offset, [pair()]) > objective(good, [pair()])
    loss = objective(row_offset, [pair()])
    loss.backward()
    assert torch.isfinite(row_offset.grad).all() and row_offset.grad.abs().sum() > 0


def test_pair_loss_is_invariant_to_simultaneous_context_and_candidate_permutation():
    scores = torch.tensor([3., 1., 2., 5.])
    reordered = scores.reshape(2,2).flip((0,1)).reshape(-1)
    assert objective(scores, [pair()]) == objective(reordered, [pair()])
    assert flatten([pair()]) == (['outside','outside','surname','surname'], ['밖','박','밖','박'])


def test_retention_without_gold_still_preserves_teacher_and_validates_shapes():
    unit = {'kind': 'retention', 'context': 'ctx', 'candidates': ['a','b'], 'expected': ['missing'],
            'teacher_scores': [2., 0.], 'base_scores': [0.,0.]}
    assert objective(torch.tensor([2.,0.]), [unit]).item() == pytest.approx(0., abs=1e-6)
    assert objective(torch.tensor([0.,2.]), [unit]).item() > 0
    with pytest.raises(ValueError, match='consume'):
        objective(torch.tensor([2.,0.,1.]), [unit])
