from copy import deepcopy

import pytest
import torch

from scripts import verify_role_free_semantic_head as module


def test_alternative_margin_and_action_sign_are_independent_of_candidate_order():
    pairs = [
        {"canonical_orientation": {"current_is_a": True}},
        {"canonical_orientation": {"current_is_a": False}},
    ]
    chosen, scores = module.decide([0, 1], [-0.25, 0.5], pairs)
    assert scores == [0.25, 0.5] and chosen == 1
    assert module.alternative_margin(2.0, True) == -2.0
    assert module.alternative_margin(2.0, False) == 2.0


def test_zero_negative_and_equal_positive_ties_keep_current():
    pairs = [{"canonical_orientation": {"current_is_a": False}} for _ in range(2)]
    assert module.decide([0], [0.0], pairs)[0] is None
    assert module.decide([0], [-1.0], pairs)[0] is None
    assert module.decide([0, 1], [1.0, 1.0], pairs)[0] is None


def test_replay_row_can_only_keep_current_or_choose_original_alternative():
    row = {"index": 0, "id": "r", "source_role": "natural_fit", "fold": 0,
           "component": "c", "current_result": "현재", "pair_indices": [0]}
    pairs = [{"alternative": {"text": "대안"},
              "canonical_orientation": {"current_is_a": False}}]
    assert module.replay_row(row, pairs, [True], [0.2])["chosen_text"] == "대안"
    assert module.replay_row(row, pairs, [True], [-0.2])["chosen_text"] == "현재"


def test_independent_fold_replay_detects_group_or_normalized_source_leak():
    rows = [
        {"id": "a", "group_id": "g", "input": "same  source", "context": "ctx"},
        {"id": "b", "group_id": "g", "input": "same source", "context": "ctx"},
    ]
    # First derive the correct shared assignment, then corrupt only one row.
    component = module.hashlib.sha256(module.canonical(["a", "b"])).hexdigest()
    fold = int(module.hashlib.sha256((module.FOLD_SALT + component).encode()).hexdigest(), 16) % 4
    for row in rows:
        row.update({"component": component, "fold": fold})
    assert len(module.independent_assign_folds(rows)) == 2
    broken = deepcopy(rows)
    broken[1]["fold"] = (fold + 1) % 4
    with pytest.raises(ValueError, match="fold/component"):
        module.independent_assign_folds(broken)


def test_fold_membership_rejects_train_held_leak():
    rows = [{"component": "shared"}, {"component": "shared"}]
    pairs = [{"fold": 0, "row_index": 0}, {"fold": 1, "row_index": 1}]
    record = {"fold": 0, "train_source_indices": [1], "held_source_indices": [0],
              "train_pairs": 1, "held_pairs": 1, "trainable_parameters": 768,
              "optimizer_updates": 100}
    checkpoint = {"fold": 0, "train_source_indices": [1], "held_source_indices": [0]}
    with pytest.raises(ValueError, match="component leak"):
        module.validate_fold_record(0, record, checkpoint, rows, pairs, [True, True])


def test_metrics_require_full_denominator_and_apply_fixed_gate():
    rows = [{"id": f"r{i}", "source_role": "natural_fit",
             "current_result": "bad" if i == 0 else "ok"}
            for i in range(module.ROWS)]
    predictions = [{"row_index": i, "id": f"r{i}", "chosen_text": "ok"}
                   for i in range(module.ROWS)]
    targets = {i: ["ok"] for i in range(module.ROWS)}
    metrics = module.compute_metrics(rows, predictions, targets)
    assert metrics["mechanism_gate_passed"] is True
    assert metrics["natural_correction_ids"] == ["r0"]
    with pytest.raises(ValueError, match="denominator"):
        module.compute_metrics(rows[:-1], predictions, targets)


def test_pin_verification_fails_after_source_corruption(tmp_path):
    source = tmp_path / "source"
    source.write_text("fixed")
    pins = {str(source): module.sha(source)}
    module.verify_pins(pins)
    source.write_text("changed")
    with pytest.raises(ValueError, match="source pin differs"):
        module.verify_pins(pins)


def test_feature_validation_accepts_exact_zero_and_unit_vectors():
    features = torch.tensor([[0.0, 0.0], [0.6, 0.8], [9.0, 9.0]])
    eligible = torch.tensor([True, True, False])
    assert module.validate_feature_vectors(features, eligible) == 1


def test_feature_validation_rejects_nonzero_nonunit_vector():
    features = torch.tensor([[0.3, 0.4]])
    with pytest.raises(ValueError, match="zero or unit"):
        module.validate_feature_vectors(features, torch.tensor([True]))
