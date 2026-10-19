from io import BytesIO

import pytest
import torch
from torch.nn import functional as F

from scripts.symmetric_pair_head import SymmetricPairHead, relation_loss


def test_zero_initialization_produces_exact_ties():
    head = SymmetricPairHead(4)
    ab = torch.tensor([[1.0, -2.0, 3.0, 4.0]])
    ba = torch.tensor([[5.0, 6.0, -7.0, 8.0]])
    assert torch.count_nonzero(head.projection.weight).item() == 0
    assert torch.equal(head(ab, ba), torch.zeros(1))


def test_order_swap_is_exactly_antisymmetric():
    head = SymmetricPairHead(3)
    with torch.no_grad():
        head.projection.weight.copy_(torch.tensor([[0.25, -0.5, 2.0]]))
    ab = torch.tensor([[1.0, 2.0, 4.0], [-2.0, 1.0, 3.0]])
    ba = torch.tensor([[3.0, -1.0, 2.0], [5.0, 4.0, -3.0]])
    assert torch.equal(head(ab, ba), -head(ba, ab))
    assert torch.equal(head(ab, ba), head.score_features(0.5 * (ab - ba)))


@pytest.mark.parametrize("relation,gradient_sign", [(1.0, -1), (-1.0, 1)])
def test_informative_relation_gradient_moves_margin_in_correct_direction(
        relation, gradient_sign):
    margin = torch.tensor([0.0], requires_grad=True)
    loss = relation_loss(
        margin, torch.tensor([relation]), torch.tensor([7])
    )
    loss.backward()
    assert torch.sign(margin.grad).item() == gradient_sign


def test_zero_head_receives_informative_feature_gradient():
    head = SymmetricPairHead(2)
    delta = torch.tensor([[2.0, -1.0]])
    loss = relation_loss(
        head.score_features(delta), torch.tensor([1.0]), torch.tensor([0])
    )
    loss.backward()
    assert torch.equal(
        torch.sign(head.projection.weight.grad), torch.tensor([[-1.0, 1.0]])
    )


def test_neutral_nonpositive_margin_has_zero_loss_and_gradient():
    margins = torch.tensor([-2.0, 0.0], requires_grad=True)
    loss = relation_loss(margins, torch.zeros(2), torch.tensor([1, 1]))
    loss.backward()
    assert loss.item() == 0.0
    assert torch.equal(margins.grad, torch.zeros_like(margins))


def test_loss_weights_rows_equally_despite_different_alternative_counts():
    margins = torch.tensor([0.0, 0.0, 0.0, 2.0])
    relations = torch.tensor([1.0, 1.0, 1.0, 0.0])
    rows = torch.tensor([10, 10, 10, 20])
    expected = 0.5 * (F.softplus(torch.tensor(0.0)) + torch.tensor(2.0))
    assert torch.allclose(relation_loss(margins, relations, rows), expected)


@pytest.mark.parametrize("features", [
    torch.tensor([[1.0, float("nan")]]),
    torch.tensor([[1.0, float("inf")]]),
])
def test_head_rejects_nonfinite_features(features):
    with pytest.raises(ValueError, match="non-finite"):
        SymmetricPairHead(2)(features, torch.zeros_like(features))


def test_head_rejects_mismatched_or_wrong_shapes():
    head = SymmetricPairHead(2)
    with pytest.raises(ValueError, match="shapes must match"):
        head(torch.zeros(2, 2), torch.zeros(1, 2))
    with pytest.raises(ValueError, match="final dimension"):
        head(torch.zeros(3), torch.zeros(3))
    with pytest.raises(ValueError, match="floating point"):
        head(torch.ones(2, dtype=torch.long), torch.ones(2, dtype=torch.long))


@pytest.mark.parametrize("margins,relations,rows,message", [
    (torch.tensor([float("nan")]), torch.tensor([1.0]), torch.tensor([0]), "non-finite"),
    (torch.tensor([0.0]), torch.tensor([2.0]), torch.tensor([0]), "only -1"),
    (torch.tensor([0.0]), torch.tensor([1.0, -1.0]), torch.tensor([0]), "equal shapes"),
    (torch.tensor([]), torch.tensor([]), torch.tensor([], dtype=torch.long), "at least one"),
    (torch.tensor([0.0]), torch.tensor([1.0]), torch.tensor([0.0]), "integer dtype"),
])
def test_relation_loss_validates_inputs(margins, relations, rows, message):
    with pytest.raises(ValueError, match=message):
        relation_loss(margins, relations, rows)


def test_state_dict_save_load_has_exact_output_parity():
    original = SymmetricPairHead(3)
    with torch.no_grad():
        original.projection.weight.copy_(torch.tensor([[0.5, -1.25, 2.0]]))
    ab = torch.tensor([[1.0, 2.0, 3.0]])
    ba = torch.tensor([[3.0, 1.0, -2.0]])
    buffer = BytesIO()
    torch.save(original.state_dict(), buffer)
    buffer.seek(0)
    restored = SymmetricPairHead(3)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(original(ab, ba), restored(ab, ba))
    assert torch.equal(original.projection.weight, restored.projection.weight)
