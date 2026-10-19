from copy import deepcopy
import math
import random
from types import SimpleNamespace

import pytest
import torch

from scripts import verify_listwise_semantic_selector as module


def sample_row(index=0, fold=0, component="c", group="g"):
    return {"index": index, "id": f"r{index}", "source_role": "natural_fit",
            "fold": fold, "component": component, "group_id": group,
            "input": "roman", "context": "ctx", "eligible": True,
            "current_index": 0, "current_result": "현재", "source_indices": [0, 1],
            "candidates": [{"text": "현재"}, {"text": "대안"}],
            "neural_views": ["a", "b"], "token_lengths": [2, 2]}


def test_independent_choice_handles_none_tie_current_and_alternative():
    assert module.choose([-0.0, 0.0], 0) == {
        "selected_index": 0, "abstained": True, "reason": "none"}
    assert module.choose([1.0, 1.0], 0)["reason"] == "tie"
    assert module.choose([2.0, 1.0], 0) == {
        "selected_index": 0, "abstained": False, "reason": "candidate"}
    assert module.choose([1.0, 2.0], 0)["selected_index"] == 1


def test_replay_prediction_selects_only_original_candidate_objects():
    row = sample_row()
    prediction = module.replay_prediction(row, [0.1, 0.2])
    assert prediction["chosen_text"] == "대안"
    assert prediction["source_indices"] == [0, 1]
    assert prediction["none_score"] == 0.0


def test_fold_record_requires_exact_membership_and_each_train_row_once():
    rows = [sample_row(0, 0, "c0", "g0"),
            sample_row(1, 1, "c1", "g1"),
            sample_row(2, 1, "c2", "g2")]
    record = {"fold": 0, "train_source_indices": [1, 2],
              "held_source_indices": [0], "training_order": [],
              "optimizer_updates": 2}
    record["training_order"] = list(record["train_source_indices"])
    random.Random(module.RECIPE["seed"]).shuffle(record["training_order"])
    checkpoint = deepcopy(record)
    assert module.validate_fold_record(0, record, checkpoint, rows) == ([1, 2], [0])
    altered_record = deepcopy(checkpoint)
    altered_checkpoint = deepcopy(checkpoint)
    altered_record["training_order"] = list(reversed(checkpoint["training_order"]))
    altered_checkpoint["training_order"] = altered_record["training_order"]
    with pytest.raises(ValueError, match="exactly once"):
        module.validate_fold_record(0, altered_record, altered_checkpoint, rows)

    leaked = deepcopy(rows)
    leaked[1]["group_id"] = "g0"
    with pytest.raises(ValueError, match="group_id overlap"):
        module.validate_fold_record(0, checkpoint, checkpoint, leaked)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)
        self.mutable = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)


def tiny_factory():
    model = TinyModel()
    optimizer = torch.optim.SGD([model.mutable], lr=1.0)
    return model, object(), optimizer, {"layer": 1}, {"loading": "ok"}


def checkpoint_for(value=2.0):
    fresh = TinyModel()
    before_frozen = module.parameter_digest(fresh, trainable=False)
    before_mutable = module.parameter_digest(fresh, trainable=True)
    with torch.no_grad():
        fresh.mutable.fill_(value)
    after_mutable = module.parameter_digest(fresh, trainable=True)
    checkpoint = {"frozen_parameters_sha256": before_frozen,
                  "state_dict": {"mutable": torch.tensor([value])}}
    record = {"frozen_parameters_sha256": before_frozen,
              "trainable_before_sha256": before_mutable,
              "trainable_after_sha256": after_mutable,
              "layer_info": {"layer": 1}, "loading_info": {"loading": "ok"}}
    return checkpoint, record


def test_restore_checkpoint_validates_frozen_base_and_exact_delta_keys():
    checkpoint, record = checkpoint_for()
    model, _ = module.restore_checkpoint(checkpoint, record, tiny_factory)
    assert model.frozen.item() == 3.0 and model.mutable.item() == 2.0
    broken = deepcopy(checkpoint)
    broken["state_dict"]["extra"] = torch.tensor([1.0])
    with pytest.raises(ValueError, match="key closure"):
        module.restore_checkpoint(broken, record, tiny_factory)


def test_parameter_digest_separates_frozen_and_trainable_changes():
    model = TinyModel()
    frozen = module.parameter_digest(model, trainable=False)
    mutable = module.parameter_digest(model, trainable=True)
    with torch.no_grad():
        model.mutable.add_(1)
    assert module.parameter_digest(model, trainable=False) == frozen
    assert module.parameter_digest(model, trainable=True) != mutable


def test_metrics_report_states_repairs_switch_precision_roles_and_folds():
    rows = [sample_row(0), sample_row(1, fold=1, component="c1", group="g1")]
    rows[1]["source_role"] = "challenge_fit"
    predictions = [module.replay_prediction(rows[0], [-1.0, -2.0]),
                   module.replay_prediction(rows[1], [0.0, 2.0])]
    metrics = module.compute_metrics(rows, predictions, {0: [], 1: ["대안"]})
    assert metrics["target_state_rows"] == {"NONE": 1, "ALT_ONLY": 1}
    assert metrics["repair_recall"] == 1.0
    assert metrics["switch_precision"] == 1.0
    assert metrics["roles"]["challenge_fit"]["corrections"] == 1
    assert metrics["folds"]["0"]["none_correct"] == 1


def test_training_evidence_is_recomputed_only_from_train_fold_targets():
    rows = [sample_row(0, 0, "c0", "g0"),
            sample_row(1, 1, "c1", "g1")]
    targets = {0: [], 1: ["현재", "대안"]}
    records = []
    for fold in range(2):
        train, _ = module.fold_indices(rows, fold)
        positives = [module.accepted_indices(rows[index], targets[index]) for index in train]
        states = {}
        for index, accepted in zip(train, positives):
            state = module.target_state(rows[index], accepted)
            states[state] = states.get(state, 0) + 1
        initial = sum(
            math.log(len(rows[index]["candidates"]) + 1)
            - (math.log(len(accepted)) if accepted else 0.0)
            for index, accepted in zip(train, positives)) / len(train)
        records.append({"training_states": states,
                        "initial_uniform_mean_loss": initial,
                        "mean_training_loss": 0.5})
    module.validate_training_evidence(rows, records, targets)
    records[0]["training_states"] = {"NONE": 999}
    with pytest.raises(ValueError, match="training-state"):
        module.validate_training_evidence(rows, records, targets)


def test_pin_verification_rejects_corruption(tmp_path):
    source = tmp_path / "source"
    source.write_text("fixed")
    pins = {str(source): module.sha(source)}
    module.verify_pins(pins)
    source.write_text("changed")
    with pytest.raises(ValueError, match="source pin differs"):
        module.verify_pins(pins)


def test_verifier_resource_guard_rejects_time_and_memory(monkeypatch):
    monkeypatch.setattr(module.time, "monotonic", lambda: module.MAX_VERIFY_SECONDS + 1)
    monkeypatch.setattr(module.resource, "getrusage",
                        lambda _: SimpleNamespace(ru_maxrss=1))
    with pytest.raises(ValueError, match="resource cap"):
        module.verifier_resources(0)

    monkeypatch.setattr(module.time, "monotonic", lambda: 1)
    excessive = module.MAX_RSS + 1 if module.sys.platform == "darwin" else (
        module.MAX_RSS // 1024 + 1)
    monkeypatch.setattr(module.resource, "getrusage",
                        lambda _: SimpleNamespace(ru_maxrss=excessive))
    with pytest.raises(ValueError, match="resource cap"):
        module.verifier_resources(0)
