from copy import deepcopy

import pytest
import torch

from scripts.train_listwise_semantic_selector import (
    accepted_indices, fold_indices, metrics, parameter_digest, prediction, target_state,
    restore_fresh_checkpoint, validate_resource_report, RESOURCE,
)


def row(index=0, fold=0):
    return {"index": index, "id": f"r{index}", "group_id": f"g{index}",
            "component": f"c{index}", "fold": fold, "eligible": True,
            "source_role": "natural_fit", "current_result": "현재",
            "current_index": 0, "source_indices": [0, 3],
            "candidates": [{"text": "현재"}, {"text": "대안"}]}


def test_none_target_is_distinct_from_current_only_or_multiple_accepted():
    source = row()
    assert accepted_indices(source, ["없는 정답"]) == []
    assert target_state(source, []) == "NONE"
    assert target_state(source, [0]) == "KEEP_ONLY"
    assert target_state(source, [1]) == "ALT_ONLY"
    assert target_state(source, [0, 1]) == "MULTI_WITH_KEEP"


def test_fold_indices_enforce_both_group_and_component_closure():
    rows = [row(0, 0), row(1, 1), row(2, 1)]
    assert fold_indices(rows, 0) == ([1, 2], [0])
    rows[2]["eligible"] = False
    assert fold_indices(rows, 0) == ([1], [0])
    for key in ("component", "group_id"):
        broken = deepcopy(rows)
        broken[1][key] = broken[0][key]
        with pytest.raises(ValueError, match="overlap"):
            fold_indices(broken, 0)


def test_predictions_preserve_source_mapping_and_distinguish_none_from_keep():
    source = row()
    none = prediction(source, [-2.0, -1.0])
    keep = prediction(source, [1.0, -1.0])
    change = prediction(source, [-1.0, 2.0])
    assert none["chosen_text"] == keep["chosen_text"] == "현재"
    assert none["reason"] == "none" and none["abstained"]
    assert keep["reason"] == "candidate" and not keep["abstained"]
    assert change["chosen_text"] == "대안" and change["source_indices"] == [0, 3]


def test_metrics_keep_missing_gold_in_denominator_and_do_not_reward_none_as_correct_text():
    rows = [row(i, i % 4) for i in range(4)]
    targets = {0: ["대안"], 1: ["현재"], 2: ["없는 정답"], 3: ["현재", "대안"]}
    predictions = [prediction(rows[0], [0.0, 1.0]), prediction(rows[1], [1.0, -1.0]),
                   prediction(rows[2], [-1.0, -2.0]), prediction(rows[3], [0.0, 1.0])]
    result = metrics(rows, predictions, targets)
    assert result["counts"]["rows"] == 4
    assert result["counts"]["before_top1"] == 2
    assert result["counts"]["top1"] == 3
    assert result["counts"]["none_correct"] == 1
    assert result["counts"]["candidate_set_covered"] == 3
    assert result["switch_precision"] == 1.0
    assert result["mechanism_gate_passed"]
    assert result["states"]["NONE"]["top1"] == 0
    predictions[1] = prediction(rows[1], [0.0, 1.0])
    assert not metrics(rows, predictions, targets)["mechanism_gate_passed"]
    with pytest.raises(ValueError, match="denominator"):
        metrics(rows[:-1], predictions, targets)


def test_parameter_digest_detects_trainable_and_frozen_mutations_separately():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    for parameter in model[0].parameters():
        parameter.requires_grad = False
    frozen = parameter_digest(model, trainable=False)
    trained = parameter_digest(model, trainable=True)
    with torch.no_grad():
        model[1].weight.add_(1.0)
    assert parameter_digest(model, trainable=False) == frozen
    assert parameter_digest(model, trainable=True) != trained
    with torch.no_grad():
        model[0].weight.add_(1.0)
    assert parameter_digest(model, trainable=False) != frozen


def test_checkpoint_reconstruction_requires_a_matching_fresh_frozen_base():
    def base():
        torch.manual_seed(4)
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        for parameter in model[0].parameters():
            parameter.requires_grad = False
        return model
    trained = base()
    frozen_sha = parameter_digest(trained, trainable=False)
    with torch.no_grad():
        trained[1].weight.add_(2.0)
    checkpoint = {"frozen_parameters_sha256": frozen_sha,
                  "state_dict": {name: parameter.detach().clone()
                                 for name, parameter in trained.named_parameters()
                                 if parameter.requires_grad}}
    restored, _ = restore_fresh_checkpoint(checkpoint, lambda: (base(), None, None, {}, {}))
    assert restored is not trained
    torch.testing.assert_close(restored(torch.ones(1, 2)), trained(torch.ones(1, 2)), rtol=0, atol=0)
    wrong = base()
    with torch.no_grad():
        wrong[0].weight.add_(1.0)
    with pytest.raises(ValueError, match="fresh upstream"):
        restore_fresh_checkpoint(checkpoint, lambda: (wrong, None, None, {}, {}))
    broken = deepcopy(checkpoint)
    broken["state_dict"].pop("1.bias")
    with pytest.raises(ValueError, match="closure"):
        restore_fresh_checkpoint(broken, lambda: (base(), None, None, {}, {}))


def test_resource_receipt_is_pinned_before_model_or_output_creation(tmp_path):
    copied = tmp_path / "report.json"
    copied.write_bytes(RESOURCE.read_bytes())
    assert validate_resource_report(copied)["passed"]
    copied.write_bytes(copied.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash"):
        validate_resource_report(copied)
