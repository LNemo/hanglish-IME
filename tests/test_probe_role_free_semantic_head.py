from copy import deepcopy

import pytest
import torch

from scripts import probe_role_free_semantic_head as module
from scripts.semantic_pair_contract import LabelState, semantic_pair_loss


def pair(current="가", alternative="나", current_is_a=True):
    return {
        "canonical_orientation": {
            "candidate_a_text": "가", "candidate_b_text": "나",
            "current_is_a": current_is_a,
        },
        "current": {"text": current}, "alternative": {"text": alternative},
    }


def test_label_orientation_is_canonical_and_role_swap_changes_action_sign_only():
    original = pair(current_is_a=True)
    swapped = pair(current="나", alternative="가", current_is_a=False)
    assert module.semantic_label(original, ["가"]) is LabelState.A_BETTER
    assert module.semantic_label(swapped, ["가"]) is LabelState.A_BETTER
    assert module.alternative_margin(2.0, True) == -2.0
    assert module.alternative_margin(2.0, False) == 2.0


@pytest.mark.parametrize(("expected", "label"), [
    (["가"], LabelState.A_BETTER), (["나"], LabelState.B_BETTER),
    (["가", "나"], LabelState.TIE), (["다"], LabelState.UNKNOWN),
])
def test_four_state_labels(expected, label):
    assert module.semantic_label(pair(), expected) is label


def test_tie_and_unknown_pairs_are_masked_from_optimizer_loss():
    margins = torch.tensor([1.0, -2.0, 0.0], requires_grad=True)
    loss = semantic_pair_loss(margins,
        [LabelState.TIE, LabelState.UNKNOWN, LabelState.A_BETTER],
        torch.tensor([0, 0, 1]))
    loss.backward()
    assert margins.grad[0].item() == margins.grad[1].item() == 0.0
    assert margins.grad[2].item() < 0.0


def test_group_and_component_cannot_cross_folds():
    rows = [
        {"fold": 0, "component": "c0", "group_id": "g0"},
        {"fold": 0, "component": "c0", "group_id": "g0"},
        {"fold": 1, "component": "c1", "group_id": "g1"},
    ]
    assert module.validate_fold_closure(rows)
    broken = deepcopy(rows)
    broken[1]["fold"] = 1
    with pytest.raises(ValueError, match="crosses folds"):
        module.validate_fold_closure(broken)


def test_decision_requires_unique_positive_alternative_margin_and_keeps_ties():
    pairs = [pair(current_is_a=False), pair(current_is_a=True)]
    assert module.choose_alternative([0, 1], [0.5, -0.25], pairs)[0] == 0
    assert module.choose_alternative([0, 1], [0.5, -0.5], pairs)[0] is None
    assert module.choose_alternative([0], [0.0], pairs)[0] is None
    assert module.choose_alternative([], [], pairs) == (None, [])


def test_metrics_require_all_rows_and_gate_on_natural_fix_without_any_regression():
    rows = [
        {"id": "n", "source_role": "natural_fit", "current_result": "오답"},
        {"id": "c", "source_role": "challenge_fit", "current_result": "정답"},
    ]
    predictions = [
        {"row_index": 0, "id": "n", "chosen_text": "정답"},
        {"row_index": 1, "id": "c", "chosen_text": "정답"},
    ]
    metrics = module.score_predictions(rows, predictions, {0: ["정답"], 1: ["정답"]})
    assert metrics["mechanism_gate_passed"] is True
    assert metrics["natural_correction_ids"] == ["n"]
    with pytest.raises(ValueError, match="all rows"):
        module.score_predictions(rows, predictions[:1], {0: ["정답"]})


def test_pin_verifier_rejects_changed_source(tmp_path):
    source = tmp_path / "source"
    source.write_text("fixed")
    pins = {str(source): module.sha(source)}
    module.verify_pins(pins)
    source.write_text("changed")
    with pytest.raises(ValueError, match="pinned source differs"):
        module.verify_pins(pins)


def test_resource_probe_is_length_stratified_and_includes_longest():
    pairs = [{"token_lengths": [index + 1, index + 2]} for index in range(40)]
    selected = module._probe_positions(pairs, list(range(40)))
    assert len(selected) == 16
    assert selected[0] == 0 and selected[-1] == 39


def test_runtime_attention_lengths_must_match_qualified_pair_lengths():
    pairs = [{"token_lengths": [2, 3]}]
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    module.validate_runtime_lengths(mask, pairs, [0])
    with pytest.raises(ValueError, match="runtime token lengths"):
        module.validate_runtime_lengths(torch.ones(2, 3), pairs, [0])


def test_revised_time_cap_keeps_original_memory_limit(monkeypatch):
    monkeypatch.setattr(module, "rss_bytes", lambda: 1024**3)
    monkeypatch.setattr(module.time, "monotonic", lambda: 2400.0)
    module.require_resources(0.0)
    monkeypatch.setattr(module.time, "monotonic", lambda: 2700.1)
    with pytest.raises(RuntimeError, match="resource cap"):
        module.require_resources(0.0)
    monkeypatch.setattr(module.time, "monotonic", lambda: 1.0)
    monkeypatch.setattr(module, "rss_bytes", lambda: 3 * 1024**3 + 1)
    with pytest.raises(RuntimeError, match="resource cap"):
        module.require_resources(0.0)
