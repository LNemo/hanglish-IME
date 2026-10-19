import json

import pytest
import torch

from scripts.listwise_semantic_contract import (
    decision, listwise_none_loss, serialize_candidate,
)
from scripts.semantic_pair_contract import CONTEXT_OMISSION_MARKER


def test_payload_excludes_provenance_and_preserves_literal_source_and_candidate():
    roman = "r" * 900
    candidate = "후보" * 500
    context = "앞" * 64 + "중간" * 100 + "뒤" * 64
    payload = json.loads(serialize_candidate(roman, context, candidate))
    assert set(payload) == {"roman", "context", "candidate_text"}
    assert payload["roman"] == roman and payload["candidate_text"] == candidate
    assert payload["context"] == (
        "앞" * 64 + CONTEXT_OMISSION_MARKER + "뒤" * 64)
    assert all(key not in serialize_candidate(roman, context, candidate)
               for key in ("role", "rank", "score", "origin", "current"))


def test_no_positive_row_has_meaningful_gradients_instead_of_being_excluded():
    scores = torch.tensor([0.0, 0.0], requires_grad=True)
    loss = listwise_none_loss(scores, [2], [[]])
    assert torch.allclose(loss, torch.log(torch.tensor(3.0)))
    loss.backward()
    assert torch.all(scores.grad > 0)


def test_multiaccepted_row_uses_combined_accepted_likelihood():
    scores = torch.tensor([1.0, -2.0, 0.5])
    loss = listwise_none_loss(scores, [3], [[0, 2]])
    expected = (torch.logsumexp(torch.tensor([1.0, -2.0, 0.5, 0.0]), dim=0)
                - torch.logsumexp(torch.tensor([1.0, 0.5]), dim=0))
    assert torch.allclose(loss, expected)


def test_rows_receive_equal_weight_regardless_of_candidate_count():
    scores = torch.zeros(4)
    loss = listwise_none_loss(scores, [1, 3], [[0], []])
    expected = 0.5 * (torch.log(torch.tensor(2.0)) + torch.log(torch.tensor(4.0)))
    assert torch.allclose(loss, expected)


def test_loss_is_invariant_to_candidate_and_positive_permutation():
    original = listwise_none_loss(torch.tensor([2.0, -1.0, 0.5]), [3], [[0, 2]])
    permuted = listwise_none_loss(torch.tensor([0.5, 2.0, -1.0]), [3], [[1, 0]])
    assert torch.equal(original, permuted)


def test_decision_is_identity_invariant_under_permutation():
    identities = ["current", "a", "b"]
    result = decision([0.2, 1.5, -0.5], current_index=0)
    assert identities[result["selected_index"]] == "a"
    permutation = [2, 0, 1]
    permuted_ids = [identities[index] for index in permutation]
    permuted_scores = [[0.2, 1.5, -0.5][index] for index in permutation]
    permuted = decision(permuted_scores, current_index=permuted_ids.index("current"))
    assert permuted_ids[permuted["selected_index"]] == "a"


def test_current_can_win_as_candidate_without_being_an_abstention():
    assert decision([1.0, 0.2], 0) == {
        "selected_index": 0, "abstained": False, "reason": "candidate"}


def test_none_and_ties_retain_current_as_an_explicit_keep_action():
    assert decision([-0.0, 0.0], 1) == {
        "selected_index": 1, "abstained": True, "reason": "none"}
    assert decision([1.0, 1.0, -2.0], 2) == {
        "selected_index": 2, "abstained": True, "reason": "tie"}


@pytest.mark.parametrize("scores,sizes,positives", [
    (torch.tensor([]), [], []),
    (torch.tensor([0.0]), [0], [[]]),
    (torch.tensor([0.0]), [2], [[]]),
    (torch.tensor([0.0]), [1], []),
    (torch.tensor([0.0]), [1], [[1]]),
    (torch.tensor([0.0]), [1], [[0, 0]]),
    (torch.tensor([float("nan")]), [1], [[0]]),
])
def test_loss_rejects_empty_nonfinite_or_mismatched_contract(scores, sizes, positives):
    with pytest.raises(ValueError):
        listwise_none_loss(scores, sizes, positives)


@pytest.mark.parametrize("scores,current", [
    ([], 0), ([float("inf")], 0), ([float("nan")], 0), ([1.0], -1), ([1.0], 1),
])
def test_decision_rejects_empty_nonfinite_or_invalid_current(scores, current):
    with pytest.raises(ValueError):
        decision(scores, current)


def test_all_wrong_row_does_not_fabricate_current_as_positive():
    scores = torch.tensor([3.0, -1.0], requires_grad=True)
    loss = listwise_none_loss(scores, [2], [[]])
    loss.backward()
    assert scores.grad[0] > 0 and scores.grad[1] > 0
    # KEEP is a downstream action; NONE remains the sole accepted training class.
    assert decision([-1.0, -2.0], current_index=0)["abstained"] is True
