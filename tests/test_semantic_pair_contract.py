import json

import pytest
import torch
from torch.nn import functional as F

from scripts.semantic_pair_contract import (
    CONTEXT_OMISSION_MARKER, LabelState, SymmetricPairHead, bound_context,
    canonical_candidates, canonical_label, ordered_views, semantic_pair_loss,
    serialize_pair, swap_label, orient_margin,
)


def test_serializer_has_no_role_metadata_surface():
    payload = serialize_pair("roman", "context", "가", "나")
    assert set(json.loads(payload)) == {
        "roman", "context", "candidate_a_text", "candidate_b_text"
    }
    with pytest.raises(TypeError):
        serialize_pair("roman", "context", "가", "나", incumbent="a")


def test_context_keeps_first_and_last_64_only():
    context = "앞" * 64 + "중간" * 50 + "뒤" * 64
    bounded = bound_context(context)
    assert bounded == "앞" * 64 + CONTEXT_OMISSION_MARKER + "뒤" * 64
    assert bounded.count(CONTEXT_OMISSION_MARKER) == 1


def test_candidate_swap_changes_order_and_preserves_full_text():
    roman = "r" * 1000
    first = "A" + "가" * 300
    second = "B" + "나" * 400
    left, right = ordered_views(roman, "ctx", first, second)
    assert left != right
    decoded = json.loads(left)
    assert decoded["roman"] == roman
    assert (decoded["candidate_a_text"], decoded["candidate_b_text"]) == (
        first, second)
    reverse = json.loads(right)
    assert (reverse["candidate_a_text"], reverse["candidate_b_text"]) == (
        second, first)
    assert ordered_views(roman, "ctx", second, first) == (right, left)


def test_nfc_sort_uses_raw_text_as_deterministic_tiebreak():
    composed = "é"
    decomposed = "e\u0301"
    a, b, swapped = canonical_candidates(composed, decomposed)
    assert (a, b) == tuple(sorted((composed, decomposed)))
    assert swapped == (a == decomposed)
    assert (a, b) == canonical_candidates(decomposed, composed)[:2]


def test_ordered_serialization_pipeline_produces_nonzero_negating_margin():
    head = SymmetricPairHead(1)
    with torch.no_grad():
        head.projection.weight.fill_(1.0)

    def fake_encoder(blob):
        return torch.tensor([[sum((index + 1) * byte
                                  for index, byte in enumerate(blob))]],
                            dtype=torch.float32)

    ab_bytes, ba_bytes = ordered_views("roman", "context", "가", "나다")
    ab, ba = fake_encoder(ab_bytes), fake_encoder(ba_bytes)
    margin = head(ab, ba)
    assert margin.item() != 0.0
    assert torch.equal(margin, -head(ba, ab))


def test_symmetric_head_margin_negates_when_pair_features_swap():
    head = SymmetricPairHead(2)
    with torch.no_grad():
        head.projection.weight.copy_(torch.tensor([[2.0, -1.0]]))
    ab = torch.tensor([[3.0, 5.0]])
    ba = torch.tensor([[1.0, -2.0]])
    assert torch.equal(head(ab, ba), -head(ba, ab))


def test_canonical_margin_restores_logical_order_and_same_winner():
    head = SymmetricPairHead(1)
    with torch.no_grad():
        head.projection.weight.fill_(1)

    def margin(first, second):
        a, b, swapped = canonical_candidates(first, second)
        ab, ba = ordered_views("roman", "ctx", a, b)

        def encode(blob):
            return torch.tensor([[sum((i + 1) * byte for i, byte in enumerate(blob))]], dtype=torch.float32)

        return orient_margin(head(encode(ab), encode(ba)), swapped)

    forward, reverse = margin("가", "나다"), margin("나다", "가")
    assert forward.item() != 0 and torch.equal(forward, -reverse)
    assert ("가" if forward > 0 else "나다") == ("나다" if reverse > 0 else "가")


def test_context_cutoff_and_literal_marker_are_plain_lossy_text():
    assert bound_context("가" * 128) == "가" * 128
    assert bound_context("가" * 129) == "가" * 64 + CONTEXT_OMISSION_MARKER + "가" * 64
    source = CONTEXT_OMISSION_MARKER + "가" * 100 + CONTEXT_OMISSION_MARKER
    assert bound_context(source) == source[:64] + CONTEXT_OMISSION_MARKER + source[-64:]


@pytest.mark.parametrize("label", list(LabelState))
def test_label_swap_and_margin_negation_leave_pair_loss_invariant(label):
    margin = torch.tensor([1.75])
    row = torch.tensor([0])
    original = semantic_pair_loss(margin, [label], row)
    swapped = semantic_pair_loss(-margin, [swap_label(label)], row)
    assert torch.equal(original, swapped)
    assert canonical_label(label, True) is swap_label(label)


def test_tie_and_unknown_are_masked_with_differentiable_zero():
    margins = torch.tensor([4.0, -3.0], requires_grad=True)
    loss = semantic_pair_loss(
        margins, [LabelState.TIE, LabelState.UNKNOWN], torch.tensor([1, 2])
    )
    assert loss.item() == 0.0 and loss.requires_grad
    loss.backward()
    assert torch.equal(margins.grad, torch.zeros_like(margins))


def test_all_masked_large_finite_margins_do_not_overflow_zero_loss():
    margins = torch.tensor([3.0e38, 3.0e38], requires_grad=True)
    loss = semantic_pair_loss(
        margins, [LabelState.TIE, LabelState.UNKNOWN], torch.tensor([1, 2])
    )
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss.backward()
    assert torch.equal(margins.grad, torch.zeros_like(margins))


def test_only_informative_pairs_receive_gradients():
    margins = torch.zeros(4, requires_grad=True)
    loss = semantic_pair_loss(margins, [
        LabelState.A_BETTER, LabelState.TIE,
        LabelState.B_BETTER, LabelState.UNKNOWN,
    ], torch.tensor([1, 1, 2, 2]))
    loss.backward()
    assert margins.grad[0] < 0
    assert margins.grad[2] > 0
    assert margins.grad[1].item() == margins.grad[3].item() == 0.0


def test_rows_have_equal_weight_despite_different_informative_pair_counts():
    margins = torch.tensor([0.0, 0.0, 0.0, 2.0])
    labels = [LabelState.A_BETTER] * 3 + [LabelState.B_BETTER]
    rows = torch.tensor([10, 10, 10, 20])
    expected = 0.5 * (F.softplus(torch.tensor(0.0))
                      + F.softplus(torch.tensor(2.0)))
    assert torch.allclose(semantic_pair_loss(margins, labels, rows), expected)


def test_loss_rejects_nonfinite_or_misaligned_inputs():
    with pytest.raises(ValueError, match="finite"):
        semantic_pair_loss(torch.tensor([float("nan")]),
                           [LabelState.A_BETTER], torch.tensor([0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        semantic_pair_loss(torch.zeros(2), [LabelState.A_BETTER] * 2,
                           torch.tensor([0]))
    with pytest.raises(ValueError, match="LabelState"):
        semantic_pair_loss(torch.zeros(1), [1], torch.tensor([0]))


def test_canonical_label_always_validates_label_type():
    with pytest.raises(TypeError, match="LabelState"):
        canonical_label(1, False)
